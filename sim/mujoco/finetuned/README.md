# Fine-tuned sim detector

`sim_yolo11n.pt`: YOLO11n fine-tuned on MuJoCo pool frames to detect `person` (1 class). 5.4 MB.

It knows the MuJoCo capsule bodies only. It is a test tool for the sim pipeline, not a model for real footage.

## How it was trained

1. `sim/mujoco/scenarios.py` renders 5 scenarios (baseline, resurface, crossing, silent_sink_busy, entry) at 960x540, 30 fps, and saves every 5th frame.
2. Labels come from the simulator: a segmentation render tags each pixel with its body part, with the water hidden, and each person's box is the tightest box around their pixels. **Boxes cover the full visible body, including the part under water.** No hand labeling.
3. `sim/train_sim_detector.py` fine-tunes the stock `yolo11n.pt` (COCO-pretrained). Split by scenario, not by frame:
   - train: baseline, crossing, silent_sink_busy, entry (180 frames)
   - validation: resurface (60 frames, never trained on)

| Setting | Value |
|---|---|
| Base model | `yolo11n.pt` (Ultralytics, COCO) |
| Epochs | 30 (patience 10) |
| Image size | 960 |
| Batch | 8 |
| Optimizer, lr | auto (SGD / AdamW picked by Ultralytics), lr0 0.01, lrf 0.01 |
| Augmentation | Ultralytics defaults: mosaic 1.0 (off for the last 10 epochs), HSV h 0.015 s 0.7 v 0.4, translate 0.1, scale 0.5, fliplr 0.5 |
| Device, time | Apple MPS (MacBook), about 7 min |
| Seed | 0 |

Full settings: [`train_args.yaml`](train_args.yaml). Per-epoch metrics: [`train_results.csv`](train_results.csv). Final validation on the held-out scenario: precision 1.00, recall 1.00, mAP50 0.995, mAP50-95 0.956.

Reproduce from scratch (about 10 min on a MacBook):
```bash
python3 -m venv .venv && .venv/bin/pip install mujoco numpy pillow ultralytics
.venv/bin/python sim/mujoco/scenarios.py            # renders scenarios + training frames
.venv/bin/python sim/train_sim_detector.py --test resurface --epochs 30
```

## Benchmark it on Isaac Sim (or any video with MOT ground truth)

```bash
pip install ultralytics opencv-python
python sim/mujoco/finetuned/benchmark_mot.py --video <isaac>/pool.mp4 --gt <isaac>/gt.txt --out annotated.mp4
```
Prints precision and recall at IoU 0.5 and 0.3. On the MuJoCo `resurface` video it scores 1.00 / 1.00.

Things to know before reading the Isaac numbers:
- **Different box definitions.** Isaac `gt.txt` uses tight boxes that treat the water as an occluder, so they cover only the part above the surface. This model draws full-body boxes including the part under water. For partly submerged people the boxes will overlap poorly even when the detection is right, so look at IoU 0.3 as well as 0.5, and watch the annotated video.
- **Different bodies.** Isaac uses human characters with skin and clothes. This model has only seen capsule bodies. A low score on Isaac is the expected sim-to-sim gap, and it is a useful number to report.
- For a fair comparison, also run the stock model: `--model yolo11n.pt` (it keeps class "person" only).

To test on Isaac Replicator stills with YOLO labels instead:
```bash
yolo detect val model=sim/mujoco/finetuned/sim_yolo11n.pt data=<replicator>.yaml imgsz=960
```
The class order must be `0: person`.

## License
Fine-tuned from Ultralytics YOLO11n, so it is AGPL-3.0 like the base model.
