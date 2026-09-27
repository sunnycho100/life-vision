# 00. Summary

updated: 2026-09-26 · author: sunnycho100 (agent research)

## Recommended stack

| Layer | Pick | Why | Details |
|---|---|---|---|
| Detector | YOLO11n or YOLO26n, fine-tuned, classes `person` + `head` (above water) | Best tooling, fast on edge | [01](01-detection-model.md) |
| Commercial fallback | RF-DETR (Apache 2.0) | Ultralytics is AGPL-3.0 | [01](01-detection-model.md) |
| Auto-labeling | Grounding DINO via autodistill, then human review | Apache 2.0, text prompts | [02](02-yolo-training-pipeline.md) |
| Adult vs baby | v1: one `person` class. v2: crop classifier | Kids' pool footage needs consent | [01](01-detection-model.md) |
| Tracker | ByteTrack, `track_buffer` 450 frames | Built in, fast | [03](03-tracking.md) |
| Alert logic | Pool zone mask + head-above-water timer + distance gate + head count | Handles resurfacing elsewhere and clear water | [03](03-tracking.md) |
| Motion sim | MuJoCo, ellipsoid drag + our own buoyancy with lung air | Runs on a Mac, 2 s for 4 motions | [04](04-mujoco-sim.md) |
| Image sim | Isaac Sim Replicator, static characters at random depths | Perfect boxes for half-submerged people | [05](05-isaac-sim.md) |
| Edge hardware | Jetson Orin Nano Super, or Pi 5 + Hailo-8L | 30 to 60+ FPS for nano models | [01](01-detection-model.md) |

## Biggest findings
1. **"Disappeared" is not "underwater".** In clear water an overhead camera still sees a submerged person. Detect **head above water** as its own class and time how long each tracked person goes without it.
2. **License.** Anything trained with Ultralytics is AGPL by default. Fine for a portfolio, a blocker for selling. LocateAnything weights are research-only.
3. **The MuJoCo sim works** and shows the difference that matters: the instinctive drowning response stays at the surface for about 18 s, while a silent sink is under in about 1 s. The silent case gives almost no motion to detect, so the timer is the only defense there.
4. **Isaac Sim needs an RTX 4080+ on Linux or Windows.** None of us can run it on a Mac.
5. **Split data by session, not by frame**, or the test score will be fake.

## Open questions
- Do we label `head` (above water) as a separate class? (Recommended: yes.)
- Alarm threshold: 5 s? Tune against false alarms per hour on real footage.
- Does anyone have an RTX 4080+ machine for Isaac Sim?
- Can we get consent to film children in a pool, or do we stay adults-only for v1?
- Portfolio only, or product later? That decides YOLO vs RF-DETR now.
- Sim data for what: detector images (Isaac) or pose sequences (MuJoCo), or both?

## Next step for each teammate

| Person | Role | Next step |
|---|---|---|
| 1 | Simulation | Run `sim/mujoco/drown_sim.py`. Add random parameters and 2D camera projection. Find an RTX machine for Isaac. |
| 2 | Real data | Write the label guide (`person`, `head` above water, rules for half-submerged). Plan the first safe recording session. |
| 3 | Model | Try Grounding DINO auto-labels and a YOLO11n baseline on a public pool dataset while our footage is being recorded. |
| 4 | Backend | Stream ingest + ByteTrack + the alert state machine from 03 with a dummy detector. |
| 5 | Frontend | One page: live view, alert with sound, clip playback, confirm or false alarm button. Agree the API with Person 4. |

## Files in this folder
- [01-detection-model.md](01-detection-model.md)
- [02-yolo-training-pipeline.md](02-yolo-training-pipeline.md)
- [03-tracking.md](03-tracking.md)
- [04-mujoco-sim.md](04-mujoco-sim.md)
- [05-isaac-sim.md](05-isaac-sim.md)
