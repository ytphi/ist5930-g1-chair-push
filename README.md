# IST5930 — G1 chair push

A trained policy for the Unitree G1 humanoid: walk to a rolling chair, take the backrest with
both hands, and push the chair under a table (success = the backrest touches the table edge).
Everything here runs on a laptop CPU. You can test it in MuJoCo (sim2sim) first, then run the
same code on the real robot.

```
policies/   trained ONNX policies (the joint order, gains and observation layout are stored in each file)
deploy/     deploy code, AprilTag perception, tag layouts, MuJoCo scene (see deploy/README.md)
```

## Policies

| File | What it is | Training sim, level 1 | Sim2sim (20 seeds, levels 0 / 0.2 / 0.3 / 0.5) |
|---|---|---|---|
| `policies/chair-push-v10.onnx` (**default**) | Side grip: both palms face inward on the backrest corners | 98.7% success, 0.1% falls | 20 / 13 / 10 / 1 |
| `policies/chair-push-v9.onnx` | Same task, but the right hand lies palm-down on the backrest | 99.6% success | 20 / 12 / 12 / 9 |

How to read the sim2sim column: almost every failure is a run where the chair's AprilTag is
**never seen** from the start pose, so the robot stands and waits. v9 often finds the tag by
luck, because it spins a little when told to stand still. v10 stands still as trained. Counting
only runs where the chair tag was seen: v9 53/53, v10 44/45 (one fall, level 0.2 seed 3).
On the real robot you start the task only after the chair tag is seen (button A, below).

## 1. Install (laptop)

```bash
git clone https://github.com/ytphi/ist5930-g1-chair-push.git
cd ist5930-g1-chair-push
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the real robot, also install [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python).
Use **Python 3.10** for it. With Python 3.11, every DDS topic failed with `PRECONDITION_NOT_MET` on our machine.

## 2. Sim2sim (no robot needed)

```bash
python deploy/g1_chair_deploy.py --sim --policy policies/chair-push-v10.onnx \
    --tags deploy/tags_sim_planned.yaml --seed 0 --level 0 --duration 30 --video sim.mp4
```

It prints the phase (APPROACH → GRIP → PUSH → DONE) once per second and ends with `SUCCESS` or
`no success (...)`. Try other `--seed` values and `--level 0.2`, `0.3`, `0.5` (harder starting layouts).
Use `--tags deploy/tags_sim_planned.yaml` in sim. With the classroom layout in `deploy/tags.yaml`,
the simulated robot starts where it can't see any floor tag, so it waits forever.
Add `--log run.jsonl` to save every step (commands, joint angles, tilt, tags seen).

## 3. Real robot

Follow the **test order in [`deploy/README.md`](deploy/README.md)** and don't skip steps:
camera check, calibration check, standing in the gantry, joystick walking, then the full task.
The short version:

- Print the tags in `deploy/apriltags/apriltags_tag36h11.pdf` (ids 0 and 10–14). Measure the black square and
  pass it as `--tag-size`. Tape the floor tags where `deploy/tags.yaml` says, or measure your own
  layout and edit that file.
- On the robot's PC2: `apriltag_perception.py` streams tag poses to the laptop over UDP.
- On the laptop: `python deploy/g1_chair_deploy.py --net <your NIC> --policy policies/chair-push-v10.onnx --command planner`
- Remote: **START** = the policy takes over and stands; **A** = start the chair task (press it only
  with the feet on the floor and after the log shows `chair tag seen >= 90%`); **SELECT** = damping + exit.
- Always have a spotter and a gantry or safety rope. Leave 5–10 cm of slack in the rope, because a tight rope
  pulls the robot and makes it shuffle and spin.

## Credits

The G1 meshes in `deploy/assets/chair_scene/assets/` come from Unitree's G1 model as packaged in
[mjlab](https://github.com/mujocolab/mjlab). The policies were trained with mjlab 1.6.0 (PPO, 4096 environments).
