"""Benchmark a YOLO person detector on any video with MOT-format ground truth (e.g. Isaac Sim gt.txt).

gt.txt lines: frame (1-based), id, left, top, width, height, ...  (extra columns ignored)

Usage:
  python benchmark_mot.py --video pool.mp4 --gt gt.txt [--model sim_yolo11n.pt] [--out annotated.mp4]
Prints precision and recall at IoU 0.5 and 0.3. Needs: pip install ultralytics opencv-python
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def matched(gt, pred, thr):
    pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)), reverse=True)
    ug, up, n = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in ug and j not in up:
            ug.add(i); up.add(j); n += 1
    return n


def load_gt(path):
    gt = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        v = line.replace(" ", "").split(",")
        if len(v) >= 6:
            f, x, y, w, h = int(float(v[0])), *map(float, v[2:6])
            gt[f].append([x, y, x + w, y + h])
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--model", default=str(Path(__file__).with_name("sim_yolo11n.pt")))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", help="optional annotated MP4 (white = ground truth, green = detections)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    person_only = [0] if len(model.names) == 80 else None  # COCO weights: class 0 "person"
    gt = load_gt(args.gt)
    cap = cv2.VideoCapture(args.video)
    writer = None
    stats = {0.5: [0, 0, 0], 0.3: [0, 0, 0]}  # tp, n_pred, n_gt
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, classes=person_only, verbose=False)[0]
        pred = r.boxes.xyxy.cpu().tolist()
        g = gt.get(f, [])
        for thr, s in stats.items():
            s[0] += matched(g, pred, thr); s[1] += len(pred); s[2] += len(g)
        if args.out:
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 15, (w, h))
            for b in g:
                cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (255, 255, 255), 1)
            for b in pred:
                cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (60, 220, 60), 2)
            writer.write(img)
    if writer:
        writer.release()
    res = {"model": Path(args.model).name, "frames": f, **{
        f"iou{thr}": {"precision": round(s[0] / max(s[1], 1), 3), "recall": round(s[0] / max(s[2], 1), 3),
                      "tp": s[0], "predicted": s[1], "ground_truth": s[2]} for thr, s in stats.items()}}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
