"""Run a person detector (and optionally a tracker) on a sim scenario video and score it
against the simulator's ground truth.

Detectors:
  yolo      pretrained COCO YOLO11n, class "person"
  rfdetr    RF-DETR Nano (COCO-pretrained, Apache 2.0), class "person"
  gdino     Grounding DINO tiny (open-vocabulary), prompt "a person. a mannequin. a humanoid robot."
  PATH.pt   any YOLO weights, e.g. the sim-trained detector from train_sim_detector.py

Usage:
  python sim/eval_detectors.py --scenario resurface --detector yolo
  python sim/eval_detectors.py --scenario resurface --detector gdino --every 5
  python sim/eval_detectors.py --scenario resurface --detector runs/sim_det/weights/best.pt --track --video
Writes data/sim_scenarios/eval/<scenario>.<detector>.json (+ .tracks.json, .mp4 when tracking).
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCN = ROOT / "data" / "sim_scenarios"
GDINO_PROMPT = "a person. a mannequin. a humanoid robot."


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def match(gt, pred, thr=0.5):
    """Greedy IoU matching. Returns list of (gt_index, pred_index)."""
    pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)), reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i); used_p.add(j); out.append((i, j))
    return out


def make_detector(name, conf):
    if name == "gdino":
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        mid = "IDEA-Research/grounding-dino-tiny"
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        proc = AutoProcessor.from_pretrained(mid)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev).eval()

        def det(img):
            inp = proc(images=img[..., ::-1].copy(), text=GDINO_PROMPT, return_tensors="pt").to(dev)  # cv2 BGR -> RGB
            with torch.no_grad():
                out = model(**inp)
            r = proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=conf, text_threshold=0.25, target_sizes=[img.shape[:2]])[0]
            boxes = r["boxes"].cpu().numpy()
            keep = (boxes[:, 2] - boxes[:, 0]) < img.shape[1] * 0.5  # drop "whole pool" boxes
            return boxes[keep], r["scores"].cpu().numpy()[keep]
        return det

    if name == "rfdetr" or name.endswith(".pth"):  # RF-DETR Nano, Apache 2.0, runs locally
        from rfdetr import RFDETRNano
        model = RFDETRNano() if name == "rfdetr" else RFDETRNano(pretrain_weights=name)
        coco = name == "rfdetr"

        def det(img):
            d = model.predict(img[..., ::-1].copy(), threshold=conf)  # cv2 BGR -> RGB
            keep = d.class_id == 1 if coco else np.ones(len(d), bool)  # COCO category id 1 = person
            return d.xyxy[keep], d.confidence[keep]
        return det

    from ultralytics import YOLO
    model = YOLO("yolo11n.pt" if name == "yolo" else name)
    person_only = len(model.names) == 80  # COCO-pretrained weights: keep class 0 "person" only

    def det(img):
        r = model.predict(img, imgsz=960,  # ultralytics expects cv2 BGR as-is
                           conf=conf, classes=[0] if person_only else None, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    return det


def suppress_contained(boxes, scores, thr=0.7):
    """Drop a box when most of it (thr of its area) sits inside a bigger box: head-only or
    body-part duplicates of one person."""
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else []
    for i, b in enumerate(boxes):
        inside = False
        for j, o in enumerate(boxes):
            if j != i and area[j] > area[i]:
                ix = max(0, min(b[2], o[2]) - max(b[0], o[0])); iy = max(0, min(b[3], o[3]) - max(b[1], o[1]))
                if ix * iy >= thr * area[i]:
                    inside = True
                    break
        if not inside:
            keep.append(i)
    return boxes[keep], scores[keep]


class IdStitcher:
    """Stable person IDs on top of tracker IDs. A new tracker ID that appears near where a
    person was last seen, within gate_s seconds, gets that person's ID back.
    ponytail: image-space distance gate; the event engine adds the pool-zone and entry rules."""

    def __init__(self, gate_s=5.0, gate_scale=3.0):
        self.gate_s, self.gate_scale = gate_s, gate_scale
        self.person_of, self.last, self.next_id = {}, {}, 1  # tracker id -> person, person -> (t, box)

    def update(self, t, tids, boxes):
        visible = {self.person_of[tid] for tid in tids if tid in self.person_of}
        out = []
        for tid, b in zip(tids, boxes):
            if tid not in self.person_of:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                best, best_d = None, None
                for pid, (lt, lb) in self.last.items():
                    if pid in visible or t - lt > self.gate_s:
                        continue
                    d = np.hypot(cx - (lb[0] + lb[2]) / 2, cy - (lb[1] + lb[3]) / 2)
                    if d < self.gate_scale * np.hypot(lb[2] - lb[0], lb[3] - lb[1]) and (best_d is None or d < best_d):
                        best, best_d = pid, d
                if best is None:
                    best, self.next_id = self.next_id, self.next_id + 1
                self.person_of[tid] = best
                visible.add(best)
            pid = self.person_of[tid]
            self.last[pid] = (t, list(b))
            out.append(pid)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--detector", default="yolo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every", type=int, default=1, help="score every Nth frame (tracking forces 1)")
    ap.add_argument("--track", action="store_true", help="run ByteTrack and count ID switches")
    ap.add_argument("--video", action="store_true", help="write an annotated MP4 (needs --track)")
    ap.add_argument("--no-gt", action="store_true", help="leave the white ground-truth boxes out of the MP4")
    ap.add_argument("--edge", action="store_true",
                    help="edge logic: drop contained boxes, tracker needs 3 frames to start an ID and keeps lost "
                         "people 5 s, stable IDs stitched across tracker restarts, last box held 1 s")
    args = ap.parse_args()
    every = 1 if args.track else args.every

    gt = json.loads((SCN / f"{args.scenario}.tracks_gt.json").read_text())
    cap = cv2.VideoCapture(str(SCN / gt["video"]))
    det = make_detector(args.detector, args.conf)
    tag = Path(args.detector).stem if args.detector.endswith((".pt", ".pth")) else args.detector
    tag += ".edge" if args.edge else ""
    out_dir = SCN / "eval"
    out_dir.mkdir(exist_ok=True)

    tracker = None
    if args.track:
        import supervision as sv
        tracker = (sv.ByteTrack(frame_rate=gt["fps"], track_activation_threshold=0.2, lost_track_buffer=150,
                              minimum_consecutive_frames=3) if args.edge else sv.ByteTrack(frame_rate=gt["fps"]))
        stitch = IdStitcher() if args.edge else None
        held = {}  # person id -> (t, box) for drawing a missing person's last box
    writer = None
    if args.video:
        w, h = gt["width"], gt["height"]
        writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{w}x{h}", "-r", str(gt["fps"]), "-i", "-", "-pix_fmt", "yuv420p",
                                   "-vcodec", "libx264", str(out_dir / f"{args.scenario}.{tag}{'.nogt' if args.no_gt else ''}.mp4")],
                                  stdin=subprocess.PIPE)

    tp = fp = fn = 0
    by_head = {"above": [0, 0], "below": [0, 0]}  # [found, total]
    id_hist = {}  # gt person -> list of matched tracker ids over time
    seen = {}  # gt person -> per-frame matched flag, for counting gaps (flicker)
    all_ids = set()  # every ID shown on screen, true people and false boxes
    pred_frames = []
    for f, fr in enumerate(gt["frames"]):
        ok, img = cap.read()
        if not ok:
            break
        if f % every:
            continue
        boxes, scores = det(img)
        if args.edge:
            boxes, scores = suppress_contained(np.asarray(boxes).reshape(-1, 4), np.asarray(scores))
        people = fr["people"]
        g_boxes = [p["bbox"] for p in people]
        pairs = match(g_boxes, boxes)
        tp += len(pairs); fp += len(boxes) - len(pairs); fn += len(g_boxes) - len(pairs)
        found = {i for i, _ in pairs}
        for i, p in enumerate(people):
            by_head[p["head"]][0] += i in found
            by_head[p["head"]][1] += 1
        if tracker:
            import supervision as sv
            dets = sv.Detections(xyxy=boxes.reshape(-1, 4).astype(float), confidence=scores.astype(float),
                                 class_id=np.zeros(len(boxes), int))
            tr = tracker.update_with_detections(dets)
            tr = tr[tr.tracker_id >= 0] if len(tr) else tr  # unconfirmed tracks come back as -1
            t_boxes = tr.xyxy.tolist()
            ids = stitch.update(fr["t"], tr.tracker_id.tolist(), t_boxes) if stitch else tr.tracker_id.tolist()
            tr.tracker_id = np.array(ids, int)
            all_ids.update(int(i) for i in ids)
            hit = set()
            for i, j in match(g_boxes, t_boxes):
                id_hist.setdefault(people[i]["track_id"], []).append(int(tr.tracker_id[j]))
                hit.add(people[i]["track_id"])
            for p in people:
                seen.setdefault(p["track_id"], []).append(p["track_id"] in hit)
            pred_frames.append({"t": fr["t"], "brightness": fr["brightness"], "people": [
                {"track_id": int(tid), "bbox": [round(v) for v in b], "conf": round(float(c), 3),
                 "head": "unknown", "head_conf": 0.0, "head_bbox": None, "keypoints": None}
                for b, tid, c in zip(t_boxes, tr.tracker_id, tr.confidence)]})
            if writer:
                for b in ([] if args.no_gt else g_boxes):
                    cv2.rectangle(img, b[:2], b[2:], (255, 255, 255), 1)
                if args.edge:  # hold a lost person's last box for 1 s, drawn thin and gray
                    for pid, (lt, lb) in stitch.last.items():
                        if pid not in set(ids) and fr["t"] - lt <= 1.0:
                            lb = [int(v) for v in lb]
                            cv2.rectangle(img, lb[:2], lb[2:], (170, 170, 170), 1)
                            cv2.putText(img, f"#{pid}?", (lb[0], lb[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)
                for b, tid in zip(t_boxes, tr.tracker_id):
                    b = [int(v) for v in b]
                    cv2.rectangle(img, b[:2], b[2:], (60, 220, 60), 2)
                    cv2.putText(img, f"#{tid}", (b[0], b[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 2)
                writer.stdin.write(img.tobytes())

    res = {"scenario": args.scenario, "detector": args.detector, "frames_scored": sum(1 for f in range(len(gt["frames"])) if f % every == 0),
           "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
           "tp": tp, "fp": fp, "fn": fn,
           "recall_head_above": round(by_head["above"][0] / max(by_head["above"][1], 1), 3),
           "recall_head_below": round(by_head["below"][0] / max(by_head["below"][1], 1), 3),
           "boxes_head_below": by_head["below"][1]}
    if tracker:
        res["tracking"] = {str(k): {"distinct_ids": len(set(id_hist.get(k, []))),
                                    "id_switches": int(sum(a != b for a, b in zip(id_hist.get(k, []), id_hist.get(k, [])[1:]))),
                                    "frames_matched": len(id_hist.get(k, [])),
                                    "gaps": int(sum(a and not b for a, b in zip(v, v[1:])))} for k, v in sorted(seen.items())}
        res["ids_shown"] = len(all_ids)
        res["people_in_scene"] = len(seen)
        (out_dir / f"{args.scenario}.{tag}.tracks.json").write_text(json.dumps(
            {**{k: gt[k] for k in ("video", "camera_id", "fps", "width", "height")},
             "source": f"{tag}+bytetrack", "frames": pred_frames}))
    if writer:
        writer.stdin.close(); writer.wait()
    (out_dir / f"{args.scenario}.{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
