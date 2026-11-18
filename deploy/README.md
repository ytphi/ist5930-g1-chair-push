# Chair-push deploy (G1, CGI-112)

Runs the mjlab chair-push policy (`Pai-ChairPush-Flat-Unitree-G1`) on the real G1, or in CPU
MuJoCo with the same code.

| File | What it does |
|---|---|
| `g1_chair_deploy.py` | Main loop at 50 Hz: state → observations → ONNX policy → joint targets. Sim and real backends, safety stops |
| `chair_planner.py` | Numpy copy of the training planner (APPROACH → GRIP → PUSH → DONE) |
| `apriltag_perception.py` | RealSense + AprilTag on the robot's PC2 → tag poses over UDP |
| `tags.yaml` | Tag size, camera mounting, floor-tag and chair-tag positions |
| `assets/chair_scene/` | Exported training scene (G1 + chair + table) for sim2sim and FK; plain `mujoco` loads it |

Everything about the policy (joint order, PD gains, default pose, action scale, observation
order) comes from the ONNX metadata, so a new checkpoint needs no code change.

## How the robot knows where things are
The head camera looks steeply down, so with the URDF mounting (~47° pitch, ~1.25 m high) it
sees the floor only from about 0.5 to 2.5 m ahead. Because of that:
- **Floor tags** (ids 10–14), taped flat in front of the table at measured spots, give the
  robot's pose relative to the table. Between sightings it dead-reckons (command + gyro).
- **Chair tag** (id 0) on the back of the backrest is seen up close. During APPROACH the chair
  doesn't move, so its last seen pose is kept in the table frame. While pushing, the chair
  moves with the robot, so its last pose relative to the robot is kept.
- There is no contact sensor: **DONE** = the chair stops moving for 0.5 s within 15 cm of the
  goal during the final straight push (or the planner's 5 cm goal check).
- If perception is lost (no floor tag for 2 s in APPROACH, no chair tag for 1 s while
  holding it), the command goes to zero and the robot stands until it sees tags again.

## Setup
- Laptop: `pip install mujoco onnxruntime pyyaml numpy mediapy` and
  [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python).
- Robot PC2: `pip install pyrealsense2 opencv-contrib-python numpy`; copy `apriltag_perception.py`.
  Packets older than 0.3 s are dropped; the age is measured without comparing the two
  clocks, so no NTP setup is needed (PC2's clock follows the robot's main board).
  PC2 has no internet: download aarch64/cp38 wheels on the laptop
  (`pip download --platform manylinux2014_aarch64 --python-version 3.8 ...`), scp them and
  `pip install --no-index --find-links . --target ~/apriltag_deps`; run with
  `PYTHONPATH=~/apriltag_deps` (done 2026-11-16: numpy 1.24.4, opencv-contrib 4.10, pyrealsense2 2.55).
- Print tag36h11 ids 0 and 10–14, measure the black square and set `tag_size`.
- Unitree's video hub on PC2 (`/unitree/module/video_hub_pc4/videohub_pc4 /dev/video4`, started
  by `master_service`) holds the D435 color stream, so `pipe.start` fails with "Device or
  resource busy". Before running perception (and after every robot reboot):
  `sudo systemctl stop master_service; sudo pkill -f videohub_pc4` on PC2. Only the app's
  camera feed is lost; motion control lives on the other board (.161).
- On PC2 use `~/run_apriltag.sh --tag-size <m> [--record x.mp4]` (sets PYTHONPATH and --dest).

## Test order (do not skip steps)
1. **Planner check** (instructor only, needs the training repo): `test_planner_match.py` → `MATCH`.
   The planner shipped here already passed it for both policies.
2. **Sim2sim**: `python deploy/g1_chair_deploy.py --sim --policy policies/chair-push-v10.onnx --tags deploy/tags_sim_planned.yaml --video sim.mp4`
   for several `--seed` and `--level` values. It should print `SUCCESS` for most of them.
3. **Camera view**: on PC2, `apriltag_perception.py --record view.mp4` while the robot stands
   on the gantry. Check where the floor tags and the chair tag show up in the image.
4. **Camera calibration**: `g1_chair_deploy.py --net <nic> --calib-check` (motors stay off).
   Put a floor tag at a taped spot (e.g. 1.0 m ahead, 0.3 m left) and compare. Fix
   `torso_to_camera` in `tags.yaml` until it matches within ~3 cm.
4b. **Localization check**: `g1_chair_deploy.py --net <nic> --localize-check` (motors off) runs the
   world model on live detections and prints the robot pose in the table frame (plus the chair
   pose when tag 0 is visible). Compare "front edge N m ahead" with a tape measure.
   Note: `--calib-check` and `--localize-check` both bind UDP 9870, so run one at a time.
5. **Gantry, standing**: `--command stand`. Press START, lower the gantry slowly, keep the
   remote in hand. SELECT = damping.
6. **Floor, walking**: `--command joystick` (left stick = walk, right stick = turn), spotter on
   the safety rope.
7. **Full task**: `--command planner`, perception running, chair 1 m ahead, then farther.
   On gantry runs add `--max-speed 0.3` so the person pushing the gantry can keep up (the
   planner's 0.6 m/s approach outran them on 2026-11-16; the whole command is scaled, so the
   turning curvature is kept). Note: `--sim` with the real CGI-112 tag layout in `tags.yaml`
   sees no floor tag (the sim spawns the robot ~2 m from the table, where those tags are behind
   it) and holds forever; for sim2sim pass `--tags` with the planned layout (10/11 at -1.0,
   12 at -1.8, 13/14 at -2.4).

## Remote buttons
START = policy takes over (planner runs: standing hold with a zero command first),
A = start the chair task (planner runs only; press it after the robot stands on the floor and
the hold line prints `chair tag seen >= 90%`; a flickering chair tag means the mount is bad),
SELECT = damping + exit. Lower the robot onto the floor between START and A, never press
A with the feet in the air: the planner walks immediately.

## Safety stops (all go to damping: kp = 0, kd = 3)
SELECT on the remote, Ctrl-C, tilt > 45°, `rt/lowstate` older than 0.1 s, non-finite action.
Joint targets are clamped to 95% of the joint range. Use a gantry or safety rope until step 7
works reliably.

## Things not verified yet
- Camera mounting (`torso_to_camera`) is taken from the URDF; verify it in step 4.
- The real G1's `imu_state` is assumed to be the pelvis IMU, as in training
  (unitree_rl_gym also uses `imu_type: pelvis` for the G1).
- The sim camera has no occlusion (the chair can hide floor tags on the real robot).
