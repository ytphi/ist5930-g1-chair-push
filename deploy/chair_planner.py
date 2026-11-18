"""Numpy port of the training planner (ChairPushCommand in scripts/chair_push_env.py).

Everything is in the robot's yaw frame: the robot is at the origin facing +x, and chair and
table poses come from perception. Keep this file in sync with the training code:
`python deploy/test_planner_match.py` compares the two on random states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- Must match scripts/chair_push_env.py (checked by test_planner_match.py).
BACK_X, BACK_HALF_T = -0.2175, 0.0075   # v8: measured backrest (5 mm plate, 16 in wide), front face still -0.21
BACK_HALF_W = 0.203
BACK_FRONT_X = BACK_X + BACK_HALF_T
GRIP_X = BACK_X - BACK_HALF_T - 0.02     # v8 corner grip
GRIP_Y, GRIP_Z = BACK_HALF_W + 0.025, 0.70
TABLE_HALF_D = 0.26
TABLE_EDGE_X = -TABLE_HALF_D
GOAL_X = TABLE_EDGE_X - BACK_FRONT_X
WAYPOINT_BACK = 0.70
PUSH_DIST = 0.58
APPROACH_VMAX, PUSH_VMAX, FINAL_VMAX = 0.6, 0.40, 0.25
APPROACH_VMIN = 0.15
GRIP_DIST, GRIP_YAW = 0.18, 0.35
UPRIGHT_COS, UPRIGHT_MIN_Z = 0.85, 0.60
YAW_RATE_MAX = 0.6
APPROACH, GRIP, PUSH, DONE = 0, 1, 2, 3
PHASE_NAMES = ("APPROACH", "GRIP", "PUSH", "DONE")


def wrap(a: float) -> float:
  return (a + math.pi) % (2 * math.pi) - math.pi


def rot(yaw: float, v) -> np.ndarray:
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def clip_norm(v: np.ndarray, vmax: float) -> np.ndarray:
  n = max(float(np.linalg.norm(v)), 1e-6)
  return v * min(1.0, vmax / n)


@dataclass
class Pose2:
  x: float
  y: float
  yaw: float

  @property
  def p(self) -> np.ndarray:
    return np.array([self.x, self.y])


@dataclass
class PlannerOut:
  cmd: np.ndarray            # (vx, vy, wz) in the robot frame
  phase: int
  chair_rel: np.ndarray      # (x, y, sin, cos)
  target_rel: np.ndarray     # (x, y, sin, cos)
  info: dict = field(default_factory=dict)


class ChairPlanner:
  def __init__(self, dt: float = 0.02, grip_timeout: float = 3.0):
    self.dt = dt
    self.grip_timeout = grip_timeout
    self.reset()

  def reset(self):
    self.phase = APPROACH
    self.wp_reached = False
    self.grip_timer = 0.0
    self.success = False

  def step(self, chair: Pose2, table: Pose2, palm_dist: np.ndarray | None,
           chair_table_touch: bool = False, upright: bool = True,
           robot: Pose2 = Pose2(0.0, 0.0, 0.0), hold: bool = False) -> PlannerOut:
    """palm_dist: (2,) palm-to-grip distances in meters, or None if unknown (timer only).
    chair_table_touch: contact signal; on the real robot pass False (geometry decides).
    hold: the caller overrides the command with zero / a scan turn (perception stale); as in v7 training,
    APPROACH does not switch to GRIP during a hold."""
    r_pos, r_yaw = robot.p, robot.yaw
    c_pos, c_yaw = chair.p, chair.yaw
    t_pos, t_yaw = table.p, table.yaw

    t_n = np.array([math.cos(t_yaw), math.sin(t_yaw)])
    goal = t_pos + rot(t_yaw, (GOAL_X, 0.0))
    wp = goal - t_n * WAYPOINT_BACK
    c_fwd = np.array([math.cos(c_yaw), math.sin(c_yaw)])
    push_pose = c_pos - c_fwd * PUSH_DIST

    ct = rot(-t_yaw, c_pos - t_pos)
    rc = rot(-c_yaw, r_pos - c_pos)
    chair_yaw_err = wrap(c_yaw - t_yaw)
    push_pose_dist = float(np.linalg.norm(r_pos - push_pose))
    head_err_chair = wrap(c_yaw - r_yaw)

    near = ct[0] > GOAL_X - 0.08
    aligned = abs(ct[1]) < 0.40 and abs(chair_yaw_err) < 0.40
    reached = float(np.linalg.norm(goal - c_pos)) < 0.05 and aligned
    if self.phase == PUSH and ((chair_table_touch and near and aligned) or reached):
      self.success = True

    ph = self.phase
    to_grip = ph == APPROACH and push_pose_dist < GRIP_DIST and abs(head_err_chair) < GRIP_YAW and upright and not hold
    self.grip_timer = self.grip_timer + self.dt if ph == GRIP else 0.0
    palms_ok = palm_dist is not None and float(np.max(palm_dist)) < 0.10
    to_push = ph == GRIP and (palms_ok or self.grip_timer > self.grip_timeout)
    lost = (ph == GRIP and push_pose_dist > 0.30) or (ph == PUSH and push_pose_dist > 0.40)
    if to_grip:
      ph = GRIP
    if to_push:
      ph = PUSH
    if lost and not self.success:
      ph = APPROACH
    if self.success:
      ph = DONE
    self.phase = ph

    if float(np.linalg.norm(c_pos - wp)) < 0.15 or (
        abs(ct[1]) < 0.12 and ct[0] > GOAL_X - WAYPOINT_BACK - 0.05):
      self.wp_reached = True

    # APPROACH
    side = 1.0 if rc[1] >= 0 else -1.0
    if rc[0] > -0.40:
      tgt_c = (rc[0], side * 0.9) if abs(rc[1]) < 0.75 else (-PUSH_DIST, rc[1])
    else:
      tgt_c = (-PUSH_DIST, 0.0)
    tgt_w = c_pos + rot(c_yaw, tgt_c)
    d_app = float(np.linalg.norm(tgt_w - r_pos))
    v_app = (tgt_w - r_pos) / max(d_app, 1e-6) * min(max(1.2 * d_app, APPROACH_VMIN), APPROACH_VMAX)
    to_tgt = tgt_w - r_pos
    head_app = math.atan2(to_tgt[1], to_tgt[0]) if np.linalg.norm(to_tgt) > 1.0 else c_yaw

    # PUSH
    target = goal if self.wp_reached else wp
    to_t = target - c_pos
    dist_t = float(np.linalg.norm(to_t))
    vmax = FINAL_VMAX if self.wp_reached else PUSH_VMAX
    speed = min(max(0.8 * dist_t, 0.10), vmax)
    v_push = to_t / max(dist_t, 1e-6) * speed + 0.5 * (push_pose - r_pos)
    head_push = t_yaw if self.wp_reached else math.atan2(to_t[1], to_t[0])

    if ph == APPROACH:
      v_w, head = v_app, head_app
    elif ph == PUSH:
      v_w, head = v_push, head_push
    else:
      v_w, head = np.zeros(2), r_yaw
    v_b = rot(-r_yaw, v_w)
    wz = float(np.clip(wrap(head - r_yaw), -YAW_RATE_MAX, YAW_RATE_MAX))
    cmd = np.array([v_b[0], v_b[1], wz]) if ph in (APPROACH, PUSH) else np.zeros(3)

    c_rel = rot(-r_yaw, c_pos - r_pos)
    dyaw = wrap(c_yaw - r_yaw)
    t_rel = rot(-r_yaw, target - r_pos)
    tyaw = wrap(t_yaw - r_yaw)
    return PlannerOut(
      cmd=cmd, phase=ph,
      chair_rel=np.array([c_rel[0], c_rel[1], math.sin(dyaw), math.cos(dyaw)]),
      target_rel=np.array([t_rel[0], t_rel[1], math.sin(tyaw), math.cos(tyaw)]),
      info={"push_pose_dist": push_pose_dist, "chair_yaw_err": chair_yaw_err,
            "chair_in_table": ct, "wp_reached": self.wp_reached, "grip_timer": self.grip_timer})


def grip_points_robot(chair: Pose2) -> np.ndarray:
  """Grip points (2,3) in the robot yaw frame, z measured from the floor."""
  out = []
  for gy in (GRIP_Y, -GRIP_Y):
    xy = chair.p + rot(chair.yaw, (GRIP_X, gy))
    out.append([xy[0], xy[1], GRIP_Z])
  return np.array(out)
