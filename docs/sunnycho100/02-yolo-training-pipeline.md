# 02. YOLO Training Pipeline

updated: 2026-09-26 · author: sunnycho100 (agent research)

## Steps

### 1. Record
- Camera where a home camera would go: high corner, looking down at about 30 to 60 degrees. Add one underwater camera if we can.
- Record sessions of: normal swimming, floating, diving, holding breath on purpose, playing, acted distress, going under and coming up somewhere else.
- Vary time of day (glare), water color, number of people.
- Safety: shallow water, a trained person outside the water watching, no pushing breath holds.
- Log each session (date, camera, people, what was acted) in `data/`.

### 2. Extract frames
```bash
ffmpeg -i session01.mp4 -vf fps=2 frames/session01_%05d.jpg
```
2 FPS is enough for detection training. Neighboring frames at 30 FPS are near duplicates.

### 3. Auto-label
Use Grounding DINO through `autodistill` (Apache 2.0) to draft boxes:
```python
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

ontology = CaptionOntology({
    "person in a swimming pool": "person",
    "human head": "head",
})
GroundingDINO(ontology=ontology).label(input_folder="frames", output_folder="dataset_draft")
```
LocateAnything-3B can also draft labels but its weights are research-only.

### 4. Human review
Load drafts into a labeling tool (CVAT or Label Studio), fix every box. Auto-labels will miss people under the surface and in glare, which are exactly the cases we care about.

### 5. YOLO dataset format
```
dataset/
  images/train/*.jpg   images/val/*.jpg   images/test/*.jpg
  labels/train/*.txt   labels/val/*.txt   labels/test/*.txt
pool.yaml
```
Each label line: `class x_center y_center width height` (all 0 to 1).
```yaml
# pool.yaml
path: dataset
train: images/train
val: images/val
test: images/test
names: {0: person, 1: head}
```
**Split by session, not by frame.** Frames from one video in both train and test make the score look much better than it is.

### 6. Fine-tune
```bash
yolo detect train model=yolo11n.pt data=pool.yaml \
  imgsz=960 epochs=100 batch=16 patience=20 close_mosaic=10 \
  hsv_h=0.03 hsv_s=0.7 hsv_v=0.5 degrees=5 translate=0.1 scale=0.5 \
  fliplr=0.5 mosaic=1.0 mixup=0.1
```
| Setting | Why |
|---|---|
| `imgsz=960` | Heads far from the camera are small. Default 640 loses them. |
| `hsv_h=0.03` | Water color varies (blue, green, tiles). Slightly more hue shift than default. |
| `hsv_v=0.5` | Bright glare and shade. |
| `degrees=5`, `scale=0.5` | Different camera mounts and distances. |
| `mixup=0.1` | Helps with overlapping swimmers. |
| `patience=20` | Stop early if val stops improving. |

Glare and splash: install `albumentations`. Ultralytics applies a small set of extra augmentations (blur, CLAHE, and others) when it is installed (unverified which set in the current version). For stronger glare, add synthetic bright spots in a custom augmentation later.

### 7. Evaluate
```bash
yolo detect val model=runs/detect/train/weights/best.pt data=pool.yaml split=test
```

## Partially submerged people
- **`person` box:** everything of the person we can see, including the part under water if it is visible. Do not guess hidden parts.
- **`head` box:** only the head, and only when it is **above** the surface. This is the class the alarm can use directly (see 03-tracking.md).
- A person fully under the surface and still visible from above: label `person`, no `head`.
- Not visible at all: no label.
- Write these rules in the label guide so all five of us label the same way.

## Metrics to report
| Metric | What it tells us |
|---|---|
| mAP50 and mAP50-95 (per class) | Standard detection quality |
| Recall on the submerged subset | Are we missing people who are under or half under the surface? This matters more than overall mAP. |
| Head-above-water precision and recall | How reliable the alarm input is |
| Alert delay (s) | Time from going under to alarm, on acted sessions |
| False alarms per hour | On normal play footage. The number parents will actually feel. |
| FPS on target hardware | Must stay at 10+ FPS with tracking |

## Sources
- [Ultralytics train settings](https://docs.ultralytics.com/), [autodistill](https://github.com/autodistill/autodistill), [autodistill-grounding-dino](https://github.com/autodistill/autodistill-grounding-dino)
- [Automatic labeling with Grounding DINO (TDS)](https://medium.com/data-science/automatic-labeling-of-object-detection-datasets-using-groundingdino-b66c486656fe)
- [Pseudo-labeling with Grounding DINO + YOLOv8 (arXiv)](https://arxiv.org/pdf/2510.25032)
- [Underwater Drowning Detection Dataset (figshare)](https://figshare.com/articles/dataset/Underwater_Drowning_Detection_Dataset/29497235), [Multi-swimmer drowning dataset, ICIAP 2025](https://dl.acm.org/doi/10.1007/978-3-032-10192-1_48)
