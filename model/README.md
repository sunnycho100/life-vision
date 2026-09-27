# model

- `detector/` (Joanne, joannemiki57): pretrained YOLO person detection and head evidence. Optional fine-tune on the Mibugi dataset.
- `pose/` (Sam, samkwak188): optional YOLO-Pose features for the yellow "distress" state.
- `eval/` (Sam): metrics script, time to alarm, false alarms per hour, missed events.

See [docs/global/architecture.md](../docs/global/architecture.md).

## Isaac Sim baseline (Rohan): YOLO11n + ByteTrack + underwater timer

| File | What |
|---|---|
| `train_yolo.py` | Fine-tunes COCO-pretrained `yolo11n.pt` on Isaac Sim frames with 2 classes: `swimming` (head above or partly above water) and `underwater` (head fully under). |
| `detect_drowning.py` | Runs the model with ByteTrack on a video. Keeps an underwater timer per person, including people the tracker loses. Raises WARNING at 5 s under, and ALARM at 12 s (or 8 s if the person also stops moving). With `--gt` it scores detection and alarm timing against the sim's ground truth. |

Training data (261 frames) comes from `sim/isaac/make_training_set.ps1`: 81 random stills plus 180 frames from three scripted clips (seeds 1-3). The seed-0 demo clip is held out as the test set.

```
sim\isaac\make_training_set.ps1
sim\isaac\.venv\Scripts\python.exe sim\isaac\pool_video.py --seed 0 --seconds 20 --pathtrace 32 --subframes 1 --yolo-every 1 --out sim\isaac\_out_test0
model\.venv\Scripts\python.exe model\train_yolo.py
model\.venv\Scripts\python.exe model\detect_drowning.py --video sim\isaac\_out_test0\pool.mp4 --gt sim\isaac\_out_test0\labels.jsonl
```

Setup: a Python 3.12 venv in `model/.venv` with `torch` (CUDA 12.8 build), `ultralytics`, `lap` and `imageio-ffmpeg`.

**Caveat:** train and test are both synthetic, from the same scene and characters. A good score here shows the pipeline works end to end. It says nothing yet about real pools. Test on real footage next.
