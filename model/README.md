# model

- `detector/` (Joanne, joannemiki57): pretrained YOLO person detection and head evidence. Optional fine-tune on the Mibugi dataset.
- `pose/` (Sam, samkwak188): optional YOLO-Pose features for the yellow "distress" state.
- `eval/` (Sam): metrics script, time to alarm, false alarms per hour, missed events.

See [docs/global/architecture.md](../docs/global/architecture.md).

## Isaac Sim baseline (Rohan): YOLO11n + ByteTrack + underwater timer

| File | What |
|---|---|
| `train_yolo.py` | Fine-tunes COCO-pretrained `yolo11n.pt` on Isaac Sim frames with 2 classes: `swimming` (head above or partly above water) and `underwater` (head fully under). |
| `detect_drowning.py` | Runs the model with ByteTrack on a video. Keeps an underwater timer per person, including people the tracker loses, with hysteresis so one-frame class flickers don't reset it. Raises WARNING at 5 s under, and ALARM at 12 s (or 8 s if the person also stops moving). With `--gt` it scores detection and alarm timing against the sim's ground truth. |
| `benchmark_isaac.py` | Scores any detector (Ultralytics weights or zero-shot RF-DETR) on an Isaac clip, split by head above vs fully under water. |
| `run_pipeline.ps1` | Renders the test clip, trains, and runs detection in one go. |

Training data (261 frames) comes from `sim/isaac/make_training_set.ps1`: 81 random stills plus 180 frames from three scripted clips (seeds 1-3). The seed-0 demo clip is held out as the test set.

```
sim\isaac\make_training_set.ps1
sim\isaac\.venv\Scripts\python.exe sim\isaac\pool_video.py --seed 0 --seconds 20 --pathtrace 32 --subframes 1 --yolo-every 1 --out sim\isaac\_out_test0
model\.venv\Scripts\python.exe model\train_yolo.py
model\.venv\Scripts\python.exe model\detect_drowning.py --video sim\isaac\_out_test0\pool.mp4 --gt sim\isaac\_out_test0\labels.jsonl
```

Setup: a Python 3.12 venv in `model/.venv` with `torch` (CUDA 12.8 build), `ultralytics`, `lap`, `imageio-ffmpeg`, and `rfdetr` for the RF-DETR comparison. Training uses `workers=0`, because Windows data-loader workers crashed mid-run.

### Results on the held-out seed-0 clip (300 frames, 1,500 person boxes, 526 with the head fully under)

Found = matched to the true full-body box at IoU 0.5. Measured 2026-09-26.

| Detector | Fine-tuned on | Precision | Found, head above | Found, head under | Sinker found while under |
|---|---|---|---|---|---|
| YOLO11n (COCO) | nothing | 0.95 | 61% | **16%** | 10/235 |
| RF-DETR-S (COCO) | nothing | 0.92 | 99% | 75% | 200/235 |
| Sunny's `sim_yolo11n.pt` | MuJoCo capsule bodies | 0.67 | 81% | 83% | 182/235 |
| **Ours `pool_yolo11n`** | 261 Isaac frames, 40 epochs, about 12 min on an RTX 4070 Laptop | 0.95 | **100%** | **95%** | 221/235 |

Our model also labels the state: YOLO test mAP50 is 0.98 (swimming P 0.97 / R 0.99, underwater P 0.95 / R 0.93). The COCO models only say "person", so they can't tell an underwater person from a swimmer, and in clear water a missing-person timer never starts because they keep seeing people on the bottom. RF-DETR-S is the strongest model with no fine-tuning, so it's the better starting point for real footage.

Drowning alarms with our model (warn at 5 s under, alarm at 12 s or 8 s if still):

| Person | True warning | Model warning | True alarm | Model alarm |
|---|---|---|---|---|
| swimmer, treader | none | none | none | none |
| diver (two dives of about 4 s) | none | 9.3 s (false warning) | none | none |
| sinker | 9.3 s | 8.8 s | 16.3 s | **11.8 s** (still rule) |
| struggler (sinks at 12 s) | 17.6 s | 17.6 s | after the clip | none |

The diver's false warning comes from the timer starting at the first underwater frame and taking about 1 s to clear. That's a trade-off against missing flickering drowners, so tune it on real footage, not on this clip.

**Caveat:** train and test are both synthetic, from the same scene and characters. A good score here shows the pipeline works end to end. It says nothing yet about real pools. Test on real footage next.
