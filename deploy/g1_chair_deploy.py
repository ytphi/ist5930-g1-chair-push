#!/usr/bin/env python3
"""Run the chair-push policy on the Unitree G1 (or in MuJoCo first, with the same code).

  # 1. sim2sim on the laptop (CPU MuJoCo, simulated camera + AprilTags):
  python deploy/g1_chair_deploy.py --sim --policy runs/chair-push-v1/policy.onnx --video out.mp4
  # 2. real robot on the gantry, standing only:
  python deploy/g1_chair_deploy.py --net enp3s0 --policy ... --command stand
  # 3. real robot, walking from the remote's sticks (no chair needed):
  python deploy/g1_chair_deploy.py --net enp3s0 --policy ... --command joystick
  # 4. real robot, full task (perception running on the robot's PC2):
  python deploy/g1_chair_deploy.py --net enp3s0 --policy ... --command planner
  # Check the camera mounting before step 4 (prints tag poses, robot motors stay off):
  python deploy/g1_chair_deploy.py --net enp3s0 --calib-check

Remote (real robot): START = run the policy, SELECT = damping + exit. Ctrl-C = damping + exit.
Needs: numpy, mujoco, onnxruntime, pyyaml; on the real robot also unitree_sdk2py.
Everything about the policy (joint order, PD gains, default pose, action scale, observation
layout) is read from the ONNX metadata that mjlab writes on export.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import struct
import sys
import time
from dataclasses import dataclass

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chair_planner as P  # noqa: E402

SCENE_XML = os.path.join(HERE, "assets", "chair_scene", "scene.xml")
POLICY_DT = 0.02          # 50 Hz, as in training (decimation 4 x 0.005 s)
SIM_DT = 0.005

# ---------------------------------------------------------------------------- math helpers


def rpy_to_R(r, p, y):
  cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
  return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                   [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                   [-sp, cp * sr, cp * cr]])


def T_from(R=None, p=None):
  T = np.eye(4)
  if R is not None:
    T[:3, :3] = R
  if p is not None:
    T[:3, 3] = p
  return T


def T_inv(T):
  R, p = T[:3, :3], T[:3, 3]
  return T_from(R.T, -R.T @ p)


def quat_to_R(q_wxyz):
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, np.asarray(q_wxyz, dtype=float))
  return R.reshape(3, 3)


def yaw_of_R(R):
  return math.atan2(R[1, 0], R[0, 0])


def pose2_of_T(T) -> P.Pose2:
  return P.Pose2(float(T[0, 3]), float(T[1, 3]), yaw_of_R(T[:3, :3]))


def T_of_pose2(p: P.Pose2):
  return T_from(rpy_to_R(0, 0, p.yaw), [p.x, p.y, 0.0])


# Camera link (x fwd, y left, z up) -> optical (x right, y down, z fwd).
R_LINK_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
# Floor tag (face up, print top toward +x): tag x = -y, tag y = +x, tag z = +z.
R_TABLE_FLOORTAG = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
# Backrest tag (face toward -x of the chair, print top up): x = -y, y = +z, z = -x.
R_CHAIR_BACKTAG = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)


# ---------------------------------------------------------------------------- policy


class Policy:
  TERM_DIMS = {"base_ang_vel": 3, "projected_gravity": 3, "command": 3, "phase": 4,
               "chair_rel": 4, "target_rel": 4}

  def __init__(self, path: str):
    import onnxruntime as ort
    self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    meta = self.sess.get_modelmeta().custom_metadata_map
    f = lambda k: np.array([float(x) for x in meta[k].split(",")])  # noqa: E731
    self.joint_names = meta["joint_names"].split(",")
    self.kp, self.kd = f("joint_stiffness"), f("joint_damping")
    self.default_q, self.action_scale = f("default_joint_pos"), f("action_scale")
    self.obs_names = meta["observation_names"].split(",")
    hist = meta.get("observation_terms_history_length", "")
    assert all(float(h) == 0 for h in hist.split(",") if h), "history obs not supported"
    self.n = len(self.joint_names)
    dims = {**self.TERM_DIMS, "joint_pos": self.n, "joint_vel": self.n, "actions": self.n}
    self.obs_dims = [dims[k] for k in self.obs_names]
    self.obs_dim = self.sess.get_inputs()[0].shape[1]
    assert sum(self.obs_dims) == self.obs_dim, (self.obs_names, self.obs_dims, self.obs_dim)
    self.in_name = self.sess.get_inputs()[0].name

  def __call__(self, terms: dict[str, np.ndarray]) -> np.ndarray:
    obs = np.concatenate([np.asarray(terms[k], dtype=np.float32).reshape(-1) for k in self.obs_names])
    return self.sess.run(None, {self.in_name: obs[None]})[0][0].astype(float)


# ---------------------------------------------------------------------------- kinematics


class Kinematics:
  """FK on the exported scene model: palms, feet and camera in the robot yaw frame."""

  def __init__(self, joint_names, cam_cfg):
    self.m = mujoco.MjModel.from_xml_path(SCENE_XML)
    self.d = mujoco.MjData(self.m)
    self.qadr = np.array([self.m.joint(f"robot/{j}").qposadr[0] for j in joint_names])
    self.palms = [self.m.site(f"robot/{s}_palm").id for s in ("left", "right")]
    self.feet = [self.m.site(f"robot/{s}_foot").id for s in ("left", "right")]
    self.torso = self.m.body("robot/torso_link").id
    rng = self.m.jnt_range[[self.m.joint(f"robot/{j}").id for j in joint_names]]
    mid, half = rng.mean(1), (rng[:, 1] - rng[:, 0]) / 2
    self.q_lo, self.q_hi = mid - 0.95 * half, mid + 0.95 * half   # soft limits for commands
    tc = cam_cfg["torso_to_camera"]
    self.T_torso_cam = T_from(rpy_to_R(*tc[3:]) @ R_LINK_OPTICAL, tc[:3])
    # Foot site height above the floor when standing (from the model keyframe).
    mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
    mujoco.mj_kinematics(self.m, self.d)
    self.foot_site_z = float(min(self.d.site_xpos[i][2] for i in self.feet))

  def update(self, q, R_world_pelvis):
    """Pelvis at the origin, oriented by roll/pitch only (yaw removed) = robot yaw frame."""
    yaw = yaw_of_R(R_world_pelvis)
    R_rp = rpy_to_R(0, 0, -yaw) @ R_world_pelvis
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_rp.reshape(-1))
    self.d.qpos[:] = self.m.qpos0
    self.d.qpos[0:3] = 0.0
    self.d.qpos[3:7] = quat
    self.d.qpos[self.qadr] = q
    mujoco.mj_kinematics(self.m, self.d)
    self.floor_z = float(min(self.d.site_xpos[i][2] for i in self.feet)) - self.foot_site_z

  def palms_from_floor(self):
    out = np.array([self.d.site_xpos[i].copy() for i in self.palms])
    out[:, 2] -= self.floor_z
    return out

  def T_yaw_cam(self):
    T_torso = T_from(self.d.xmat[self.torso].reshape(3, 3), self.d.xpos[self.torso])
    T = T_torso @ self.T_torso_cam
    T[2, 3] -= self.floor_z  # heights measured from the floor
    return T


# ---------------------------------------------------------------------------- world model


class WorldModel:
  """Robot pose in the table frame (from floor tags) and chair pose in the table frame."""

  def __init__(self, tags_cfg):
    s = float(tags_cfg["tag_size"])
    self.floor = {int(k): T_from(R_TABLE_FLOORTAG, [v[0], v[1], 0.0])
                  for k, v in tags_cfg["floor_tags"].items()}
    self.chair_tags = {int(k): T_from(R_CHAIR_BACKTAG, v) for k, v in tags_cfg["chair_tags"].items()}
    self.tag_size = s
    self.robot_in_table: P.Pose2 | None = None
    self.chair_in_table: P.Pose2 | None = None
    self.chair_in_robot_hold: P.Pose2 | None = None
    self.t_floor = self.t_chair = -1e9

  def predict(self, v_body, wz, dt):
    """Dead reckoning between tag sightings (commanded velocity + gyro yaw rate)."""
    if self.robot_in_table is None:
      return
    r = self.robot_in_table
    d = P.rot(r.yaw, (v_body[0] * dt, v_body[1] * dt))
    self.robot_in_table = P.Pose2(r.x + d[0], r.y + d[1], P.wrap(r.yaw + wz * dt))

  def observe(self, detections, T_yaw_cam, now, phase):
    """detections: list of (id, T_cam_tag). T_yaw_cam: camera pose in the robot yaw frame."""
    robot_meas, chair_meas = [], []
    for tid, T_cam_tag in detections:
      T_robot_tag = T_yaw_cam @ T_cam_tag
      if tid in self.floor:
        robot_meas.append(pose2_of_T(self.floor[tid] @ T_inv(T_robot_tag)))
      elif tid in self.chair_tags:
        chair_meas.append(pose2_of_T(T_robot_tag @ T_inv(self.chair_tags[tid])))
    if robot_meas:
      x = np.mean([p.x for p in robot_meas]); y = np.mean([p.y for p in robot_meas])
      yaw = math.atan2(np.mean([math.sin(p.yaw) for p in robot_meas]),
                       np.mean([math.cos(p.yaw) for p in robot_meas]))
      if self.robot_in_table is None or now - self.t_floor > 1.0:
        self.robot_in_table = P.Pose2(x, y, yaw)
      else:  # light smoothing against tag jitter
        r, a = self.robot_in_table, 0.5
        self.robot_in_table = P.Pose2(r.x + a * (x - r.x), r.y + a * (y - r.y),
                                      P.wrap(r.yaw + a * P.wrap(yaw - r.yaw)))
      self.t_floor = now
    if chair_meas:
      self.chair_in_robot_hold = chair_meas[0]
      if self.robot_in_table is not None:
        self.chair_in_table = pose2_of_T(T_of_pose2(self.robot_in_table) @ T_of_pose2(chair_meas[0]))
      self.t_chair = now
    elif phase in (P.GRIP, P.PUSH, P.DONE) and self.chair_in_robot_hold is not None \
        and self.robot_in_table is not None:
      # Holding the chair: it moves with the robot; keep its last relative pose.
      self.chair_in_table = pose2_of_T(T_of_pose2(self.robot_in_table) @ T_of_pose2(self.chair_in_robot_hold))

  def in_robot_frame(self):
    """(chair, table) poses in the robot yaw frame, or None if unknown."""
    if self.robot_in_table is None or self.chair_in_table is None:
      return None
    T_robot_table = T_inv(T_of_pose2(self.robot_in_table))
    return (pose2_of_T(T_robot_table @ T_of_pose2(self.chair_in_table)), pose2_of_T(T_robot_table))


# ---------------------------------------------------------------------------- backends


@dataclass
class RobotState:
  q: np.ndarray
  dq: np.ndarray
  gyro: np.ndarray
  quat: np.ndarray          # wxyz, pelvis in world
  stamp: float
  buttons: dict
  sticks: np.ndarray        # lx, ly, rx, ry


class SimBackend:
  """CPU MuJoCo with the exported training scene. Ground truth feeds a simulated camera."""

  def __init__(self, policy: Policy, tags_cfg, layout_seed=0, level=0.0, video=None):
    self.m = mujoco.MjModel.from_xml_path(SCENE_XML)
    self.m.opt.timestep = SIM_DT
    self.d = mujoco.MjData(self.m)
    mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
    self.qadr = np.array([self.m.joint(f"robot/{j}").qposadr[0] for j in policy.joint_names])
    self.dadr = np.array([self.m.joint(f"robot/{j}").dofadr[0] for j in policy.joint_names])
    act_joint = [self.m.joint(self.m.actuator_trnid[i, 0]).name.removeprefix("robot/") for i in range(self.m.nu)]
    self.act_of_joint = np.array([act_joint.index(j) for j in policy.joint_names])
    self.gyro_adr = self.m.sensor_adr[self.m.sensor("robot/imu_ang_vel").id]
    self.pelvis = self.m.body("robot/pelvis").id
    self.chair_q = self.m.joint("chair/chair_joint").qposadr[0]
    self.table_mocap = self.m.body("table/table").mocapid[0]
    self.d.qpos[self.qadr] = policy.default_q
    self._layout(np.random.default_rng(layout_seed), level)
    mujoco.mj_forward(self.m, self.d)
    self.t = 0.0
    self.gantry = True            # hold the pelvis (like the real gantry) until release()
    self.base0 = self.d.qpos[0:7].copy()
    self.cam = tags_cfg["camera"]
    self.tags_cfg = tags_cfg
    self.renderer = None
    self.frames = []
    if video:
      self.renderer = mujoco.Renderer(self.m, 480, 640)

  def _layout(self, rng, lv):
    """Same distributions as reset_layout() in training."""
    u = lambda a, b: rng.uniform(a, b)  # noqa: E731
    lerp = lambda a, b: a + (b - a) * lv  # noqa: E731
    t_yaw = u(-0.4, 0.4); t_pos = np.array([u(2.7, 3.3), u(-0.5, 0.5)])
    ct = np.array([P.GOAL_X - u(0.6, lerp(1.2, 2.6)), u(-lerp(0.3, 1.3), lerp(0.3, 1.3))])
    c_pos = t_pos + P.rot(t_yaw, ct); c_yaw = t_yaw + u(-lerp(0.3, 1.4), lerp(0.3, 1.4))
    r_dist = u(0.85, lerp(1.3, 2.5)); ang = math.pi + u(-lerp(0.4, 2.2), lerp(0.4, 2.2))
    r_pos = c_pos + P.rot(c_yaw, (r_dist * math.cos(ang), r_dist * math.sin(ang)))
    if P.rot(-t_yaw, r_pos - t_pos)[0] > P.TABLE_EDGE_X - 0.45:
      r_pos = c_pos + P.rot(c_yaw, (-r_dist, 0.0))
    r_yaw = math.atan2(*(c_pos - r_pos)[::-1]) + u(-lerp(0.5, math.pi), lerp(0.5, math.pi))
    q = lambda yaw: [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]  # noqa: E731
    self.d.mocap_pos[self.table_mocap] = [*t_pos, 0.0]; self.d.mocap_quat[self.table_mocap] = q(t_yaw)
    self.d.qpos[self.chair_q:self.chair_q + 7] = [*c_pos, 0.002, *q(c_yaw)]
    self.d.qpos[0:2] = r_pos; self.d.qpos[3:7] = q(r_yaw)

  def read(self) -> RobotState:
    return RobotState(q=self.d.qpos[self.qadr].copy(), dq=self.d.qvel[self.dadr].copy(),
                      gyro=self.d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy(),
                      quat=self.d.xquat[self.pelvis].copy(), stamp=time.time(),
                      buttons={"start": self.t > 1.0}, sticks=np.zeros(4))

  def write(self, q_target, kp, kd):
    del kp, kd  # the model's position actuators carry the trained gains
    self.d.ctrl[self.act_of_joint] = q_target
    for _ in range(int(round(POLICY_DT / SIM_DT))):
      if self.gantry:
        self.d.qpos[0:7] = self.base0; self.d.qvel[0:6] = 0.0
      mujoco.mj_step(self.m, self.d)
    self.t += POLICY_DT
    if self.renderer is not None and int(self.t / POLICY_DT) % 2 == 0:
      cam = mujoco.MjvCamera(); cam.lookat[:] = self.d.xpos[self.pelvis]; cam.lookat[2] = 0.6
      cam.distance, cam.azimuth, cam.elevation = 3.5, 135, -25
      self.renderer.update_scene(self.d, cam); self.frames.append(self.renderer.render())

  def damp(self):
    pass

  def release(self):
    self.gantry = False

  def truth(self):
    """Ground-truth chair and table poses in the robot yaw frame (for logging)."""
    R = quat_to_R(self.d.xquat[self.pelvis]); yaw = yaw_of_R(R)
    T_w_r = T_from(rpy_to_R(0, 0, yaw), [*self.d.xpos[self.pelvis][:2], 0])
    c = self.d.qpos[self.chair_q:self.chair_q + 7]
    T_w_c = T_from(quat_to_R(c[3:]), [c[0], c[1], 0])
    T_w_t = T_from(quat_to_R(self.d.mocap_quat[self.table_mocap]), self.d.mocap_pos[self.table_mocap])
    return pose2_of_T(T_inv(T_w_r) @ T_w_c), pose2_of_T(T_inv(T_w_r) @ T_w_t), T_w_r

  def detections(self, kin: Kinematics, world: WorldModel, rng):
    """Simulated AprilTag detections: in FOV, in range, facing the camera, with noise."""
    chair, table, _ = self.truth()
    T_r_cam = kin.T_yaw_cam()
    T_r_table, T_r_chair = T_of_pose2(table), T_of_pose2(chair)
    out = []
    h, v = math.radians(self.cam["hfov_deg"]) / 2, math.radians(self.cam["vfov_deg"]) / 2
    tags = [(k, T_r_table @ T) for k, T in world.floor.items()] + \
           [(k, T_r_chair @ T) for k, T in world.chair_tags.items()]
    for tid, T_r_tag in tags:
      T_cam_tag = T_inv(T_r_cam) @ T_r_tag
      p = T_cam_tag[:3, 3]
      if p[2] <= 0.1 or np.linalg.norm(p) > self.cam["max_range"]:
        continue
      if abs(math.atan2(p[0], p[2])) > h or abs(math.atan2(p[1], p[2])) > v:
        continue
      if T_cam_tag[:3, 2] @ (-p / np.linalg.norm(p)) < 0.25:   # seen too edge-on or from behind
        continue
      noisy = T_cam_tag.copy()
      noisy[:3, 3] += rng.normal(0, 0.01 + 0.01 * np.linalg.norm(p), 3)
      noisy[:3, :3] = noisy[:3, :3] @ rpy_to_R(*rng.normal(0, 0.02, 3))
      out.append((tid, noisy))
    return out


class RemoteController:
  KEYS = {"R1": 0, "L1": 1, "start": 2, "select": 3, "R2": 4, "L2": 5, "F1": 6, "F3": 7,
          "A": 8, "B": 9, "X": 10, "Y": 11, "up": 12, "right": 13, "down": 14, "left": 15}

  @classmethod
  def parse(cls, data: bytes):
    keys = struct.unpack("H", data[2:4])[0]
    buttons = {k: bool((keys >> i) & 1) for k, i in cls.KEYS.items()}
    lx, rx, ry = struct.unpack("f", data[4:8])[0], struct.unpack("f", data[8:12])[0], struct.unpack("f", data[12:16])[0]
    ly = struct.unpack("f", data[20:24])[0]
    return buttons, np.array([lx, ly, rx, ry])


class RealBackend:
  def __init__(self, net: str, policy: Policy):
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    self.n = policy.n
    ChannelFactoryInitialize(0, net)
    msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()
    _, result = msc.CheckMode()
    while result["name"]:
      print(f"releasing high-level mode {result['name']!r}"); msc.ReleaseMode(); time.sleep(1.0)
      _, result = msc.CheckMode()
    self.cmd = unitree_hg_msg_dds__LowCmd_(); self.crc = CRC()
    self.pub = ChannelPublisher("rt/lowcmd", LowCmd_); self.pub.Init()
    self.state = None; self.stamp = 0.0
    self.sub = ChannelSubscriber("rt/lowstate", LowState_); self.sub.Init(self._on_state, 10)
    t0 = time.time()
    while self.state is None:
      if time.time() - t0 > 5:
        raise RuntimeError("no rt/lowstate after 5 s: check --net and the cable")
      time.sleep(0.05)
    self.mode_machine = self.state.mode_machine

  def _on_state(self, msg):
    self.state, self.stamp = msg, time.time()

  def read(self) -> RobotState:
    s = self.state
    buttons, sticks = RemoteController.parse(bytes(s.wireless_remote))
    return RobotState(q=np.array([s.motor_state[i].q for i in range(self.n)]),
                      dq=np.array([s.motor_state[i].dq for i in range(self.n)]),
                      gyro=np.array(s.imu_state.gyroscope, dtype=float),
                      quat=np.array(s.imu_state.quaternion, dtype=float),
                      stamp=self.stamp, buttons=buttons, sticks=sticks)

  def _send(self, q, kp, kd):
    self.cmd.mode_pr = 0
    self.cmd.mode_machine = self.mode_machine
    for i in range(self.n):
      c = self.cmd.motor_cmd[i]
      c.mode, c.q, c.dq, c.tau, c.kp, c.kd = 1, float(q[i]), 0.0, 0.0, float(kp[i]), float(kd[i])
    self.cmd.crc = self.crc.Crc(self.cmd)
    self.pub.Write(self.cmd)

  def write(self, q_target, kp, kd):
    self._send(q_target, kp, kd)

  def damp(self, kd=3.0):
    self._send(np.zeros(self.n), np.zeros(self.n), np.full(self.n, kd))


class UdpDetections:
  """Receives tag detections from apriltag_perception.py: {"stamp", "tags": [{"id", "T"}]}."""

  def __init__(self, port: int):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.bind(("0.0.0.0", port)); self.sock.setblocking(False)

  def poll(self):
    latest = None
    while True:
      try:
        latest = json.loads(self.sock.recv(65536)); t_recv = time.time()
      except BlockingIOError:
        break
    if latest is None:
      return []
    # Age since the frame was captured. With "lat" (frame-to-send delay measured on PC2's own
    # clock) the two clocks need not be in sync; old senders without it fall back to "stamp".
    if "lat" in latest:
      age = (time.time() - t_recv) + float(latest["lat"])
    else:
      age = time.time() - latest["stamp"]
    if age > 0.3:
      return []
    return [(int(t["id"]), np.array(t["T"]).reshape(4, 4)) for t in latest["tags"]]


# ---------------------------------------------------------------------------- main loop


def run(args):
  policy = Policy(args.policy)
  tags_cfg = yaml.safe_load(open(args.tags))
  kin = Kinematics(policy.joint_names, tags_cfg["camera"])
  world = WorldModel(tags_cfg)
  planner = P.ChairPlanner(dt=POLICY_DT)
  rng = np.random.default_rng(args.seed)
  if args.sim:
    be = SimBackend(policy, tags_cfg, layout_seed=args.seed, level=args.level, video=args.video)
  else:
    be = RealBackend(args.net, policy)
  udp = None if args.sim or args.command != "planner" else UdpDetections(args.udp_port)

  stop = {"flag": False}
  signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
  log = []

  # 1. Ramp to the default pose (3 s), then hold until START.
  s = be.read()
  q0 = s.q.copy()
  ramp_steps = int(3.0 / POLICY_DT)
  for k in range(ramp_steps):
    if stop["flag"]:
      break
    a = (k + 1) / ramp_steps
    be.write((1 - a) * q0 + a * policy.default_q, policy.kp, policy.kd)
    if not args.sim:
      time.sleep(POLICY_DT)
  print("at default pose; press START on the remote to run the policy (SELECT = stop)")
  while not stop["flag"]:
    s = be.read()
    if s.buttons.get("start"):
      break
    if s.buttons.get("select"):
      stop["flag"] = True
    be.write(policy.default_q, policy.kp, policy.kd)
    if not args.sim:
      time.sleep(POLICY_DT)

  # 2. Policy loop. (In sim, the gantry lets go here; on the robot, lower the gantry now.)
  if hasattr(be, "release"):
    be.release()
  last_action = np.zeros(policy.n)
  stall_hist = []
  t_start, step = time.time(), 0
  reason = "ctrl-c"
  # Real planner runs start in a standing hold (zero command) so the robot can be lowered onto
  # the floor and settle; the task begins on A. Added after the 2026-11-16 test, where START was
  # pressed with the feet in the air and the robot "walked" while hanging.
  task_on = args.sim or args.command != "planner"
  hold_hist = []
  if not task_on:
    print("policy running with a zero command (standing). Lower the robot, let it settle, "
          "then press A on the remote to start the chair task (SELECT = stop)")
  while not stop["flag"]:
    t_loop = time.time()
    s = be.read()
    R = quat_to_R(s.quat)
    grav_b = R.T @ np.array([0.0, 0.0, -1.0])
    sim_t = step * POLICY_DT

    # Safety.
    if s.buttons.get("select"):
      reason = "SELECT"; break
    if not task_on and s.buttons.get("A"):
      task_on = True; print("A pressed: chair task started")
    if grav_b[2] > -math.cos(math.radians(45)):
      reason = f"tilt {math.degrees(math.acos(-grav_b[2])):.0f} deg"; break
    if not args.sim and time.time() - s.stamp > 0.1:
      reason = "lowstate stale"; break
    if args.duration and sim_t > args.duration:
      reason = "duration"; break

    kin.update(s.q, R)
    wz_meas = float((R @ s.gyro)[2])

    # Command source.
    stale = False
    if args.command == "planner":
      world.predict(last_cmd if step else np.zeros(3), wz_meas, POLICY_DT)
      dets = be.detections(kin, world, rng) if args.sim else udp.poll()
      seen_ids = sorted(int(t) for t, _ in dets)
      world.observe(dets, kin.T_yaw_cam(), sim_t if args.sim else time.time(), planner.phase)
      poses = world.in_robot_frame()
      now = sim_t if args.sim else time.time()
      if poses is None or (planner.phase == P.APPROACH and now - world.t_floor > 2.0) or \
          (planner.phase in (P.GRIP, P.PUSH) and now - world.t_chair > 1.0):
        stale = True
      if poses is None:
        chair, table = P.Pose2(2.5, 0, 0), P.Pose2(4.0, 0, 0)
      else:
        chair, table = poses
      palms = kin.palms_from_floor()
      palm_dist = np.linalg.norm(palms - P.grip_points_robot(chair), axis=1)
      upright = (-grav_b[2] > P.UPRIGHT_COS) and (-kin.floor_z > P.UPRIGHT_MIN_Z)   # pelvis height above the floor
      out = planner.step(chair, table, palm_dist, chair_table_touch=False, upright=upright, hold=stale)
      # No contact sensor on the robot: the backrest is at the table edge when the chair stops
      # moving close to the goal during the final straight push (chair seen by its tag).
      ct = out.info["chair_in_table"]
      stall_hist.append((sim_t if args.sim else time.time(), ct.copy()))
      stall_hist[:] = [h for h in stall_hist if stall_hist[-1][0] - h[0] <= 0.5]
      if (planner.phase == P.PUSH and planner.wp_reached and ct[0] > P.GOAL_X - 0.15
          and abs(out.info["chair_yaw_err"]) < 0.4 and not stale
          and stall_hist[-1][0] - stall_hist[0][0] >= 0.45
          and np.linalg.norm(stall_hist[-1][1] - stall_hist[0][1]) < 0.02):
        planner.success, planner.phase = True, P.DONE
        out.cmd[:] = 0.0; out.phase = P.DONE
        print("backrest at the table (chair stalled near the goal): DONE")
      cmd, phase = out.cmd, out.phase
      if not task_on:
        # Standing hold before the task: keep observing (localization stays fresh) but do not
        # let the planner advance or the robot move. Report the detection rates so the user
        # only presses A when the chair tag is seen steadily (2026-11-16 test 3: it was seen in
        # 30% of the frames before the start and the task never worked).
        planner.phase = P.APPROACH
        cmd, phase = np.zeros(3), P.APPROACH
        hold_hist.append((any(t in world.chair_tags for t in seen_ids), any(t in world.floor for t in seen_ids)))
        if len(hold_hist) >= 100:
          c = 100 * sum(h[0] for h in hold_hist) / len(hold_hist); f = 100 * sum(h[1] for h in hold_hist) / len(hold_hist)
          ok = c >= 90 and f >= 50
          print(f"hold: chair tag seen {c:3.0f}%  floor tags seen {f:3.0f}%  "
                f"{'OK, press A when the robot stands' if ok else 'NOT ready (chair tag must be >= 90%)'}")
          hold_hist.clear()
      if args.max_speed > 0 and cmd[0] > args.max_speed:
        # Gantry runs: slow the walk so the person pushing the gantry can keep up. The whole
        # command is scaled by the same factor, so the planner's turning curvature is kept.
        cmd = cmd * (args.max_speed / cmd[0])
      if stale:
        cmd = np.zeros(3)   # stand still until perception recovers
      chair_rel, target_rel = out.chair_rel, out.target_rel
    else:
      if args.command == "joystick":
        cmd = np.array([0.5 * s.sticks[1], -0.3 * s.sticks[0], -0.5 * s.sticks[2]])
        phase = P.APPROACH
        chair_rel = np.array([2.5, 0.0, 0.0, 1.0]); target_rel = np.array([4.0, 0.0, 0.0, 1.0])
      else:  # stand
        cmd, phase = np.zeros(3), P.APPROACH
        chair_rel = np.array([2.5, 0.0, 0.0, 1.0]); target_rel = np.array([4.0, 0.0, 0.0, 1.0])
      out = None
    last_cmd = cmd

    terms = {
      "base_ang_vel": s.gyro, "projected_gravity": grav_b,
      "joint_pos": s.q - policy.default_q, "joint_vel": s.dq, "actions": last_action,
      "command": cmd, "phase": np.eye(4)[phase], "chair_rel": chair_rel, "target_rel": target_rel,
    }
    action = policy(terms)
    if not np.all(np.isfinite(action)):
      reason = "non-finite action"; break
    last_action = action
    q_target = np.clip(policy.default_q + policy.action_scale * action, kin.q_lo, kin.q_hi)
    be.write(q_target, policy.kp, policy.kd)

    row = {"t": sim_t, "phase": phase, "cmd": cmd.round(3).tolist(), "stale": stale, "task_on": task_on,
           "ids": seen_ids if args.command == "planner" else [],
           "tilt_deg": round(math.degrees(math.acos(min(1.0, -grav_b[2]))), 1),
           # raw signals for offline analysis (added after the first real stand test, 2026-11-16)
           "gyro": np.round(s.gyro, 3).tolist(), "grav": np.round(grav_b, 3).tolist(),
           "q": np.round(s.q, 4).tolist(), "dq": np.round(s.dq, 3).tolist(),
           "action": np.round(action, 3).tolist(), "q_target": np.round(q_target, 4).tolist()}
    if out is not None:
      row.update(chair_in_table=out.info["chair_in_table"].round(3).tolist(),
                 yaw_err=round(out.info["chair_yaw_err"], 3), wp=out.info["wp_reached"])
    if args.sim:
      tc, tt, _ = be.truth()
      row["true_chair"] = [round(tc.x, 3), round(tc.y, 3), round(tc.yaw, 3)]
      row["true_table"] = [round(tt.x, 3), round(tt.y, 3), round(tt.yaw, 3)]
      if args.command == "planner":
        row["est_chair"] = [round(chair.x, 3), round(chair.y, 3), round(chair.yaw, 3)]
        row["est_table"] = [round(table.x, 3), round(table.y, 3), round(table.yaw, 3)]
        row["n_det"] = len(dets)
    log.append(row)
    if step % 50 == 0:
      print(f"t={sim_t:5.1f}s phase={P.PHASE_NAMES[phase]:8s} cmd={np.round(cmd, 2)} "
            f"tilt={row['tilt_deg']:4.1f} stale={stale}"
            + (f" chair_in_table={row.get('chair_in_table')}" if out is not None else ""))
    step += 1
    if not args.sim:
      time.sleep(max(0.0, POLICY_DT - (time.time() - t_loop)))

  print(f"stopping: {reason} after {step * POLICY_DT:.1f} s; damping")
  for _ in range(int(2.0 / POLICY_DT)):
    be.damp()
    if args.sim:
      break
    time.sleep(POLICY_DT)
  if args.log:
    with open(args.log, "w") as f:
      for r in log:
        f.write(json.dumps(r) + "\n")
    print("log:", args.log)
  if args.sim and args.video and be.frames:
    import mediapy
    mediapy.write_video(args.video, be.frames, fps=int(1 / (2 * POLICY_DT)))
    print("video:", args.video)
  return planner.success, reason


_G1_JOINT_ORDER = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
           "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
           "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
           "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
           "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
           "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
           "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
           "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]


def calib_check(args):
  """Motors stay off. Prints every tag's pose in the robot yaw frame from live detections."""
  tags_cfg = yaml.safe_load(open(args.tags))
  kin = Kinematics(_G1_JOINT_ORDER, tags_cfg["camera"])
  from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
  from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
  ChannelFactoryInitialize(0, args.net)
  st = {}
  sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(lambda msg: st.update(s=msg), 10)
  udp = UdpDetections(args.udp_port)
  print("listening; Ctrl-C to quit. Compare with a tape measure (robot yaw frame: x fwd, y left).")
  while True:
    time.sleep(0.5)
    if "s" not in st:
      print("waiting for rt/lowstate ..."); continue
    s = st["s"]
    q = np.array([s.motor_state[i].q for i in range(29)])
    kin.update(q, quat_to_R(np.array(s.imu_state.quaternion, dtype=float)))
    T_cam = kin.T_yaw_cam()
    print(f"camera height above floor: {T_cam[2, 3]:.3f} m")
    for tid, T in udp.poll():
      p = (T_cam @ T)[:3, 3]
      print(f"  tag {tid:3d}: x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}  dist={np.linalg.norm(T[:3, 3]):.2f} m")


def localize_check(args):
  """Motors stay off. Runs the world model on live detections and prints the robot pose in the
  table frame and the chair pose (table frame and robot frame). Compare with a tape measure."""
  tags_cfg = yaml.safe_load(open(args.tags))
  policy = Policy(args.policy) if args.policy else None
  order = policy.joint_names if policy else _G1_JOINT_ORDER
  kin = Kinematics(order, tags_cfg["camera"])
  from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
  from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
  ChannelFactoryInitialize(0, args.net)
  st = {}
  sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init(lambda msg: st.update(s=msg), 10)
  udp = UdpDetections(args.udp_port)
  world = WorldModel(tags_cfg)
  print("listening; Ctrl-C to quit. Table frame: origin 0.26 m behind the front-edge midpoint, "
        "x into the table, y left. Robot frame: x fwd, y left.")
  last = 0.0
  while True:
    time.sleep(0.02)
    if "s" not in st:
      if time.time() - last > 1: print("waiting for rt/lowstate ..."); last = time.time()
      continue
    s = st["s"]
    q = np.array([s.motor_state[i].q for i in range(29)])
    kin.update(q, quat_to_R(np.array(s.imu_state.quaternion, dtype=float)))
    dets = udp.poll()
    if not dets:
      continue
    T_cam = kin.T_yaw_cam()
    world.observe(dets, T_cam, time.time(), P.APPROACH)
    if time.time() - last < 0.5:
      continue
    last = time.time()
    ids = sorted(t for t, _ in dets)
    r, c = world.robot_in_table, world.chair_in_table
    line = f"tags {ids}"
    if r is not None:
      line += (f" | robot in table: {r.x:+.2f} {r.y:+.2f} yaw {np.degrees(r.yaw):+.0f} deg"
               f" (front edge {-(r.x + P.TABLE_HALF_D):.2f} m ahead)")
    if c is not None:
      line += f" | chair in table: {c.x:+.2f} {c.y:+.2f} yaw {np.degrees(c.yaw):+.0f} deg"
    for tid, T in dets:
      if tid in world.chair_tags:
        ch = pose2_of_T(T_cam @ T @ T_inv(world.chair_tags[tid]))
        line += f" | chair in robot: {ch.x:+.2f} {ch.y:+.2f} yaw {np.degrees(ch.yaw):+.0f} deg"
    print(line)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--policy", help="exported ONNX (from the mjlab run's log dir)")
  ap.add_argument("--sim", action="store_true", help="run in CPU MuJoCo instead of the robot")
  ap.add_argument("--net", default="eth0", help="network interface wired to the G1")
  ap.add_argument("--command", choices=["planner", "joystick", "stand"], default="planner")
  ap.add_argument("--tags", default=os.path.join(HERE, "tags.yaml"))
  ap.add_argument("--udp-port", type=int, default=9870)
  ap.add_argument("--seed", type=int, default=0, help="sim: layout seed")
  ap.add_argument("--level", type=float, default=0.0, help="sim: layout difficulty 0..1")
  ap.add_argument("--duration", type=float, default=0.0, help="stop after N s of policy (0 = no limit)")
  ap.add_argument("--max-speed", type=float, default=0.0,
                  help="planner: cap the forward command (m/s), e.g. 0.3 on gantry runs (0 = off)")
  ap.add_argument("--video", help="sim: write an mp4")
  ap.add_argument("--log", help="write a JSONL log of every step")
  ap.add_argument("--calib-check", action="store_true", help="motors off: print tag poses in the robot frame")
  ap.add_argument("--localize-check", action="store_true",
                  help="motors off: print the world model's robot and chair poses in the table frame")
  args = ap.parse_args()
  if args.calib_check:
    return calib_check(args)
  if args.localize_check:
    return localize_check(args)
  if not args.policy:
    ap.error("--policy is required")
  if args.sim and not args.duration:
    args.duration = 25.0
  ok, reason = run(args)
  if args.sim:
    print("SUCCESS" if ok else f"no success ({reason})")


if __name__ == "__main__":
  main()
