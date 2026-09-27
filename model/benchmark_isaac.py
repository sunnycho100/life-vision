"""Score person detectors on an Isaac Sim clip, split by head above vs fully under water.

Detectors (--detector):
  rfdetr-s          RF-DETR Small, COCO-pretrained, no fine-tuning (class "person")
  PATH.pt           any Ultralytics YOLO weights: yolo11n.pt (COCO), Sunny's MuJoCo-trained
                    sim_yolo11n.pt, or our Isaac-trained model/runs/pool_yolo11n/weights/best.pt

Ground truth is labels.jsonl from sim/isaac/pool_video.py: full-body boxes (refraction-corrected)
and each person's head state. Class labels are ignored here; this only asks "was the person found".

Run from the repo root:
    model\\.venv\\Scripts\\python.exe model\\benchmark_isaac.py --detector rfdetr-s --video sim\\isaac\\_out_test0\\pool.mp4
Writes <video>_<detector>.mp4 (white = truth, green = found, red = missed) unless --no-video.
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--detector", required=True)
parser.add_argument("--video", required=True)
parser.add_argument("--gt", help="labels.jsonl (default: next to the video)")
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--no-video", action="store_true")
args = parser.parse_args()


def make_detector(name):
    if name.startswith("rfdetr"):
        from PIL import Image
        from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall
        model = {"rfdetr-n": RFDETRNano, "rfdetr-s": RFDETRSmall, "rfdetr-m": RFDETRMedium}[name]()
        from rfdetr.assets.coco_classes import COCO_CLASSES
        names = COCO_CLASSES if isinstance(COCO_CLASSES, dict) else dict(enumerate(COCO_CLASSES))
        person = [k for k, v in names.items() if v == "person"][0]

        def det(bgr):
            d = model.predict(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), threshold=args.conf)
            keep = d.class_id == person
            return d.xyxy[keep], d.confidence[keep]
        return det

    from ultralytics import YOLO
    model = YOLO(name)
    coco = len(model.names) == 80  # COCO weights: keep class 0 "person" only

    def det(bgr):  # Ultralytics expects OpenCV BGR as-is
        r = model.predict(bgr, imgsz=960, conf=args.conf, classes=[0] if coco else None, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    return det


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def match(gt, pred, thr):
    """Greedy one-to-one IoU matching. Returns {gt index: pred index}."""
    pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)), reverse=True)
    used, out = set(), {}
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


video = Path(args.video)
gt = [json.loads(line) for line in open(args.gt or video.with_name("labels.jsonl"))]
det = make_detector(args.detector)
tag = Path(args.detector).stem if args.detector.endswith(".pt") else args.detector
cap = cv2.VideoCapture(str(video))
W, H, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS) or 15
writer = None
if not args.no_video:
    writer = imageio_ffmpeg.write_frames(str(video.with_name(f"{video.stem}_{tag}.mp4")), (W, H), fps=fps,
                                         codec="libx264", pix_fmt_out="yuv420p", quality=8)
    writer.send(None)

counts = {thr: {"above": [0, 0], "below": [0, 0], "fp": 0, "pred": 0} for thr in (0.5, 0.3)}
per_person = {}  # scenario -> [found while head below, frames head below, found while above, frames above]
for g in gt:
    ok, img = cap.read()
    if not ok:
        break
    boxes, scores = det(img)
    boxes = [list(map(float, b)) for b in boxes]
    people = [(v["scenario"], v["head_state"], v["body_xyxy"]) for v in g["people"].values() if v.get("body_xyxy")]
    truth = [p[2] for p in people]
    for thr, c in counts.items():
        m = match(truth, boxes, thr)
        c["pred"] += len(boxes)
        c["fp"] += len(boxes) - len(m)
        for i, (scn, state, _) in enumerate(people):
            key = "below" if state == "below" else "above"
            c[key][0] += i in m
            c[key][1] += 1
            if thr == 0.5:
                pp = per_person.setdefault(scn, [0, 0, 0, 0])
                k = 0 if key == "below" else 2
                pp[k] += i in m
                pp[k + 1] += 1
    if writer:
        m = match(truth, boxes, 0.5)
        for i, b in enumerate(truth):
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (255, 255, 255) if i in m else (0, 0, 255), 1 if i in m else 3)
        for b, s in zip(boxes, scores):
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 220, 60), 2)
            cv2.putText(img, f"person {s:.2f}", (int(b[0]), max(20, int(b[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (60, 220, 60), 2)
        cv2.rectangle(img, (0, 0), (W, 50), (40, 40, 40), -1)
        cv2.putText(img, f"{tag}   t = {g['t']:5.1f}s   green = detection, red = missed person", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.send(np.ascontiguousarray(img[:, :, ::-1]).tobytes())
if writer:
    writer.close()

res = {"detector": tag}
for thr, c in counts.items():
    tp = c["above"][0] + c["below"][0]
    res[f"iou{thr}"] = {"precision": round(tp / max(c["pred"], 1), 3),
                        "recall": round(tp / max(c["above"][1] + c["below"][1], 1), 3),
                        "recall_head_above": round(c["above"][0] / max(c["above"][1], 1), 3),
                        "recall_head_under": round(c["below"][0] / max(c["below"][1], 1), 3)}
res["per_person_iou0.5"] = {s: {"found_while_under": f"{v[0]}/{v[1]}", "found_while_above": f"{v[2]}/{v[3]}"}
                            for s, v in per_person.items()}
print(json.dumps(res, indent=2))
out = video.with_name(f"benchmark_{tag}.json")
out.write_text(json.dumps(res, indent=2))
