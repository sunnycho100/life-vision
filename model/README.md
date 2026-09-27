# model

Detector training and the alarm rules. Two tracks of work ended up here:

1. **Trained on Isaac Sim** (Rohan): YOLO11n and RF-DETR Small fine-tuned on path-traced pool frames, with two classes (`swimming`, `underwater`) and a GREEN / ORANGE / RED timer. This proved the full pipeline, from detection to alarm, against exact simulator answers.
2. **Fine-tuned on a real wave pool** (Joanne): RF-DETR Small fine-tuned on 80 hand-labeled YouTube frames. This is the model the review app runs, because sim-only models don't carry over to real people.

| File | What |
|---|---|
| `train_yolo.py`, `train_rfdetr.py` | Fine-tune YOLO11n or RF-DETR Small on Isaac Sim frames (same 261 frames and split) |
| `alert_rules.py` | The shared GREEN / ORANGE / RED rules (below) |
| `detect_drowning.py`, `detect_drowning_rfdetr.py` | Detector + ByteTrack + underwater timer on a video, scored against sim ground truth with `--gt` |
| `benchmark_isaac.py` | Scores any detector on an Isaac clip, split by head above vs fully under water |
| `detect_people_rfdetr_s.py` | RF-DETR Small person detection with overlapping tiles, for small far-away swimmers (the app's tiling comes from this) |
| `finetune_rfdetr_s_colab.zip` | The real wave-pool fine-tuning notebook, with its outputs |
| `run_pipeline.ps1` | Renders the Isaac test clip, trains, and runs detection in one go |

## Isaac Sim baseline (Rohan): YOLO11n + ByteTrack + underwater timer

| File | What |
|---|---|
| `train_yolo.py` | Fine-tunes COCO-pretrained `yolo11n.pt` on Isaac Sim frames with 2 classes: `swimming` (head above or partly above water) and `underwater` (head fully under). |
| `alert_rules.py` | The GREEN / ORANGE / RED rules for coloring a person's box, shared by the sim ground truth and the detector (see below). No dependencies. |
| `detect_drowning.py` | Runs the model with ByteTrack on a video and colors each person with `alert_rules.py`. Keeps an underwater timer per person, including people the tracker loses, and smooths the class so one-frame flickers don't reset it. With `--gt` it scores detection and the first ORANGE and RED times against the sim's ground truth. |
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

### Box colors: GREEN / ORANGE / RED

One rule set (`alert_rules.py`) colors every box, in the sim ground truth and in the detector output:

| Color | Rule | Norm it follows |
|---|---|---|
| **RED** alarm | Head fully under water for **10 s** | Ellis & Associates 10/20 rule: 10 s to recognize an aquatic emergency, 20 s to reach the person. Stricter than ASTM F3698-24, which tests that a system alarms for a motionless submerged toddler dummy by 20 s. Someone who goes under holds their breath for at most about a minute and most lose consciousness within about 2 minutes (NEJM 2012), so 10 s leaves time to reach them conscious. |
| **RED** alarm | Head fully under for **5 s and not moving** for 2 s | Motionless under water is the strongest sign: Coral MYLO alarms on "motionless, with the head under the surface"; the ASTM test case is a motionless dummy. |
| **ORANGE** watch | Head fully under for **5 s** | Early warning at half the red time. Our choice, not a published number: tune it on real footage, because kids dive and hold their breath on purpose. |
| **ORANGE** watch | **Drowning signs at the surface for 3 s**: upright, mouth at the waterline, no headway | Instinctive drowning response (Pia 1974; Vittone, U.S. Coast Guard): head low with the mouth at water level, vertical, not making headway. It lasts only 20 to 60 s before the person goes under. |
| **ORANGE** watch | Went under after showing drowning signs | A struggling person who slips under isn't fine for the first 5 s. |
| GREEN | Everything else | |

Red and the "went under after distress" orange clear only after the head has been clearly above water for 2 s.

What each input is, and how we get it:

| Input | Sim ground truth (exact) | Detector (approximated from video) |
|---|---|---|
| Head fully under | Head center plus head radius (0.11 m, scaled for children) below the local, wavy water surface | YOLO `underwater` class, majority over the last second, with the timer backdated to the first underwater frame |
| Not moving | Hip, head, hands and feet moved less than 15% of body height (RMS) in 2 s | Box corners moved less than 15% of the box diagonal (RMS) in 2 s |
| Mouth at the waterline | Head center less than 0.75 head radius above the water | Not observable with the 2-class model yet (needs pose or a head model), so this orange rule only shows in the ground truth |
| Upright | Hip-to-neck axis within 30 degrees of vertical | Same as above |
| No headway | Hip moved less than 30% of body height in 3 s | Same as above |

Sources: [Ellis 10/20 rule](https://jeffellismanagement.com/glossary/10-20-Second-Protection-Rule), [CPSC staff letter on ASTM F3698-24](https://www.cpsc.gov/s3fs-public/June-5-2025-CPSC-Letter-to-ASTM-F15-49-Computer-Vision-Pool-Alarms.pdf), [Coral MYLO FAQ](https://coralmylo.com/faq/), [Szpilman et al., NEJM 2012](https://www.nejm.org/doi/abs/10.1056/NEJMra1013317), [Vittone, "Drowning Doesn't Look Like Drowning"](https://www.army.mil/article/109852/drowning_doesnt_look_like_drowning).

### Results on the held-out seed-0 clip (300 frames, 1,500 person boxes, 526 with the head fully under)

Found = matched to the true full-body box at IoU 0.5. Measured 2026-09-26.

| Detector | Fine-tuned on | Precision | Found, head above | Found, head under | Sinker found while under |
|---|---|---|---|---|---|
| YOLO11n (COCO) | nothing | 0.95 | 61% | **16%** | 10/235 |
| RF-DETR-S (COCO) | nothing | 0.92 | 99% | 75% | 200/235 |
| Sunny's `sim_yolo11n.pt` | MuJoCo capsule bodies | 0.67 | 81% | 83% | 182/235 |
| **Ours `pool_yolo11n`** | 261 Isaac frames, 40 epochs, about 12 min on an RTX 4070 Laptop | 0.95 | **100%** | **95%** | 221/235 |

RF-DETR Small fine-tuned on the same 261 frames (`train_rfdetr.py`) found 99% of swimmers (class right 98%) and 98% of underwater people (class right 94%) on this clip.

Our YOLO11n also labels the state: YOLO test mAP50 is 0.98 (swimming P 0.97 / R 0.99, underwater P 0.95 / R 0.93). The COCO models only say "person", so they can't tell an underwater person from a swimmer, and in clear water a missing-person timer never starts because they keep seeing people on the bottom. RF-DETR-S is the strongest model with no fine-tuning, so it's the better starting point for real footage.

Drowning alarms with our model (warn at 5 s under, alarm at 12 s or 8 s if still):

| Person | True warning | Model warning | True alarm | Model alarm |
|---|---|---|---|---|
| swimmer, treader | none | none | none | none |
| diver (two dives of about 4 s) | none | 9.3 s (false warning) | none | none |
| sinker | 9.3 s | 8.8 s | 16.3 s | **11.8 s** (still rule) |
| struggler (sinks at 12 s) | 17.6 s | 17.6 s | after the clip | none |

The diver's false warning comes from the timer starting at the first underwater frame and taking about 1 s to clear. That's a trade-off against missing flickering drowners, so tune it on real footage, not on this clip.

### Final clip (2026-09-26 21:51): six people, moving water, 30 fps

`sim/isaac/_out_final_20260926-2151` (local), 720p copies in `presentation/videos/`. Same YOLO11n weights, which never saw the new water, poses or child-sized person.

| Detector | Found (IoU 0.5) | Found, head under | State correct |
|---|---|---|---|
| RF-DETR-S (COCO, no fine-tuning) + ByteTrack | 78% | 57% | no state |
| Ours `pool_yolo11n` (IoU 0.4) | 97% swimming, 93% underwater | 93% | 96% / 98% |

| Person | True warning | Model warning | True alarm (still rule) | Model alarm |
|---|---|---|---|---|
| swimmer, floater, treader, diver | none | none | none | none |
| child (silent sink) | 10.5 s | 10.5 s | about 13.4 s | 13.5 s |
| struggler (drowning response, then sinks) | 14.5 s | 13.7 s | about 17.5 s | 16.9 s |

Two fixes in `detect_drowning.py` got the false alerts to zero on this clip: class-agnostic NMS (one box per person), and a stricter merge rule. A new track now takes over a lost person's timer only if it is also underwater and within half a body of where they were lost. Before that, a swimmer passing a sinking person took over their identity and timer.

**Caveat:** train and test are both synthetic, from the same scene and characters. A good score here shows the pipeline works end to end. It says nothing yet about real pools. Test on real footage next.

## RF-DETR-S fine-tuned on real wave-pool footage (Joanne)

`finetune_rfdetr_s_colab.zip` holds `finetune_rfdetr_s_colab.ipynb` (unzip it to open in Jupyter or Colab). It fine-tunes COCO-pretrained RF-DETR-S on `data/aquaperson_wavepool_v1` (80 YouTube wave-pool frames, one box per visible person, submerged parts not boxed). The notebook is saved with its outputs from the Colab run.

- Split: 62 train / 16 valid frames (valid = `NycwxaU4GPw_065` to `_080`, 680 person boxes)
- rfdetr 1.11.0, single class `person`, training scale 672, early stopped with best EMA mAP at epoch 25

| Model (valid set) | AP50 | AP50:95 | Precision @0.25 | Recall @0.25 | F1 |
|---|---|---|---|---|---|
| RF-DETR-S (COCO) | 0.569 | 0.321 | 0.754 | 0.266 | 0.393 |
| **RF-DETR-S fine-tuned** | **0.809** | **0.504** | 0.675 | **0.838** | **0.748** |

This is the detector the review app runs, on the Mac GPU with full frame + 3x2 tiles. Setup: [backend/README.md](../backend/README.md#the-model).

Weights (127 MB, too large for git): [release `rfdetr-s-person-v1`](https://github.com/sunnycho100/life-vision/releases/tag/rfdetr-s-person-v1)

```python
# pip install rfdetr==1.11.0
from rfdetr import RFDETRSmall
model = RFDETRSmall.from_checkpoint("rfdetr_s_person_best.pth")
detections = model.predict("frame.jpg", threshold=0.25)
```

Caveat: 16 validation frames all come from the same video as part of the training set, so these numbers are optimistic for other pools.
