# Simulation Test Bench

updated: 2026-09-26 · owner: Sunny

MuJoCo scenarios that run the whole pipeline end to end with exact answers. The simulator knows every person's box and head height, so detectors, trackers, and the event engine can be scored without labeling anything.

## Scenarios

`sim/mujoco/scenarios.py` writes to `data/sim_scenarios/`:
- `<name>.tracks_gt.json`: contract A with exact visible box, head box, and head `above` or `below` (below = whole head under the surface)
- `<name>.json`: per-person summary
- `videos/<name>.mp4` and `yolo/` training frames (not in git)

| Scenario | Length | People | Expected engine result |
|---|---|---|---|
| `baseline` | 6 s | stand, swim, float, drowning response, collapse | collapse → red, everyone else green |
| `resurface` | 10 s | diver under about 5 s and up about 2 m away, swimmer, stander, floater | diver yellow, then back to green, never red |
| `crossing` | 8 s | two swimmers passing, two overlapping standers | all green, IDs must not swap |
| `silent_sink_busy` | 10 s | silent sink in the deep end, swimmer, stander, floater, duck-under (under about 2.6 s) | sink → red, duck-under yellow at most |
| `entry` | 6 s | person falls in from the deck, swimmer, floater | entry alert, short yellow at most |

"Collapse" stands for 1 s, then goes limp and folds to the floor with the head under, standing in for fainting or a seizure. "Silent sink" is a toddler-style quiet sink with no struggle.

```bash
.venv/bin/python sim/mujoco/scenarios.py                    # all scenarios, about 2.5 min
.venv/bin/python sim/mujoco/scenarios.py --only resurface
```

## Detector results on sim video

Scored against the simulator's boxes (IoU 0.5). Test scenario `resurface` was never used for training.

| Detector | Scenario | Precision | Recall | Recall, head above | Recall, head fully under |
|---|---|---|---|---|---|
| Pretrained YOLO11n (COCO person) | baseline | 1.00 | 0.30 | 0.33 | 0.00 |
| Pretrained YOLO11n (COCO person) | resurface | 1.00 | 0.25 | 0.30 | 0.02 |
| Pretrained YOLOv8n (COCO person) | baseline | 0.99 | 0.38 | 0.41 | 0.02 |
| Pretrained YOLOv8n (COCO person) | resurface | 0.94 | 0.12 | 0.09 | 0.31 |
| Grounding DINO tiny (prompt: person, mannequin, humanoid robot), every 5th frame | resurface | 0.52 | 0.72 | 0.69 | 0.89 |
| **YOLO11n fine-tuned on sim frames** (180 frames, 30 epochs, about 7 min on a MacBook) | resurface (held out) | **1.00** | **1.00** | 1.00 | 1.00 |
| YOLO11n fine-tuned on sim frames | baseline (seen in training) | 1.00 | 1.00 | 1.00 | 1.00 |

What this says:
- **Pretrained YOLO (v8n or 11n) mostly fails on the sim.** It sees the standing capsule people but not the swimmers and floaters, and almost never someone fully under. Capsule bodies in tinted water are too far from COCO photos.
- **Grounding DINO** finds more, but half its boxes are wrong. Offline labeling at most.
- **A detector fine-tuned on sim frames is perfect on sim**, including a scenario it never saw. Labels come free from the simulator. Perfect on sim only means it learned our capsule bodies, not that it will work on anything else. Use it for sim demo feeds and engine tests.
- Corrections: earlier numbers in this repo's history (96% and 83% for pretrained YOLO, 76% and 92% for the fine-tuned one) were wrong. The first came from renders where the water was accidentally not drawn, the second from a script bug that fed the models color-swapped frames. Both are fixed, and the table above is re-measured.

## Tracking results

ByteTrack (`supervision`) on the held-out `resurface` scenario:

| Detector | Person | Frames tracked | Distinct IDs | ID switches |
|---|---|---|---|---|
| Fine-tuned YOLO11n | diver (goes under and resurfaces) | 299 / 300 | 1 | 0 |
| Fine-tuned YOLO11n | swimmer | 300 / 300 | 1 | 0 |
| Fine-tuned YOLO11n | stander | 300 / 300 | 1 | 0 |
| Fine-tuned YOLO11n | floater | 300 / 300 | 1 | 0 |
| Pretrained YOLOv8n | diver | 49 / 300 | 4 | 3 |
| Pretrained YOLOv8n | stander | 70 / 300 | 2 | 1 |
| Pretrained YOLO11n | stander | 298 / 300 | 1 | 0 |

The sim-trained detector keeps the diver's box through the whole dive, because in clear water the body stays visible. So the box never disappears and a "person missing" timer would never start. This is the clear-water problem from the decisions, shown on real numbers: **the alarm must use head state, not disappearance.**

```bash
.venv/bin/python sim/train_sim_detector.py --test resurface            # writes runs/sim_det/weights/best.pt
.venv/bin/python sim/eval_detectors.py --scenario resurface --detector yolo
.venv/bin/python sim/eval_detectors.py --scenario resurface --detector gdino --every 5
.venv/bin/python sim/eval_detectors.py --scenario resurface --detector runs/sim_det/weights/best.pt --track --video
```
Tracking also writes `data/sim_scenarios/eval/<scenario>.<detector>.tracks.json` (contract A, head `unknown`), which the event engine can read directly.

## RF-DETR-N and tracking edge logic

Stock RF-DETR-N (Apache 2.0, runs locally) sees the capsule people far better than stock YOLO, but with the default tracker it showed 18 IDs for 4 people: boxes flicker, head-only duplicate boxes get their own IDs, and every re-detection starts a new ID.

Edge logic (`--edge` in `sim/eval_detectors.py`):
1. Clean the raw boxes: NMS at IoU 0.6 (RF-DETR returns some people twice), drop a box that contains 2+ other boxes (one box around two overlapping people), and drop a small box mostly inside a much bigger one (head-only or arm-only box).
2. ByteTrack needs 3 consecutive frames to start an ID and keeps lost people 5 s.
3. ID stitching: a new track within 3 box-diagonals of where someone was lost, within 5 s, gets their old ID.
4. A lost person's last box is held for 1 s (the "missing" state).

Held-out `resurface`, 4 real people:

| Setup | Precision | Recall | Recall, fully under | IDs shown | ID switches |
|---|---|---|---|---|---|
| Stock RF-DETR-N + default ByteTrack | 0.81 | 0.79 | 0.60 | 18 | 3 |
| Stock RF-DETR-N + edge logic | 0.86 | 0.79 | 0.60 | 5 | 1 |
| RF-DETR-N fine-tuned on sim (1 epoch) + default ByteTrack | 0.99 | 1.00 | 1.00 | 5 | 4 |
| **RF-DETR-N fine-tuned on sim (1 epoch) + edge logic** | **1.00** | **1.00** | **1.00** | **4** | **0** |

```bash
.venv/bin/pip install "rfdetr[train]"
.venv/bin/python sim/train_sim_rfdetr.py --epochs 1          # held-out val hit P 1.0 / R 1.0 after 1 epoch (about 2 min)
.venv/bin/python sim/eval_detectors.py --scenario resurface --detector runs/sim_rfdetr/checkpoint_best_ema.pth --track --edge --video
```
The fine-tuned checkpoint is 121 MB, too big for git; regenerate it with the script.

Fine-tuned RF-DETR-N + edge logic on every scenario (IDs shown / real people, ID switches): baseline 5/5, 0 · resurface 4/4, 0 · crossing 4/4, 0 · silent_sink_busy 5/5, 0 · entry 3/3, 0. Precision and recall 0.997 to 1.00 everywhere. An earlier containment-only rule dropped one of two overlapping people in `crossing` (recall 0.74); the three-pass cleanup above fixed that.

## How the sim helps with real footage

A detector trained on capsule people will not work on real people, and the reverse is also true (pretrained YOLO above). The sim is not training data for the real detector. What it gives us:

1. **A test bench with exact answers** for the tracker and the event engine. Rohan can run the engine on `*.tracks_gt.json` today and check that each scenario produces the expected colors.
2. **Hard cases we cannot film safely:** silent sink, collapse, a person under for a long time.
3. **Demo feeds** for the 4-screen UI, with the sim-trained detector drawing the boxes.
4. **Motion data** (keypoints in the `.npz` files) for a later pose-based distress layer, with the sim-to-real limits in [archive notes](../archive/sunnycho100/04-mujoco-sim.md).

For real footage tomorrow: start from pretrained YOLO, fine-tune on our own labeled frames, and test only on real clips. Split by recording session.

## Known limits
- One body model (the DeepMind humanoid, about 40 kg). No children yet.
- Water is a flat tinted box. No waves, splash, glare, or refraction.
- Scripted motions: sine-wave strokes, a scripted upright torque for standing, scripted pushes for diving and falling.
- Same 5 skin tones, 4 hair colors, 5 swimsuits.
