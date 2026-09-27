"""Prepare human labels, compare detectors, freeze dev policy, evaluate locked frames.

Never treats predictions or unreviewed annotation templates as ground truth.
All coordinates are normalized xyxy. Run python -m tools.evaluate_detection --help.
"""
import argparse
import json
import hashlib
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
from backend.common import ROOT, digest, read_json, write_json
from backend.detector import OnnxDetector
from tools.prepare_rfdetr import frame_samples

YOLO_SHA = "190ba5f1e61411a001683e349d6b2cdb0804c0dc67a5e34cd8ff6fd00ee54b4d"
DEV_TIMES = list(range(8, 54, 3))
LOCKED_TIMES = [60, 65, 70, 75, 80, 85, 90, 94]


def area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def intersection(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / max(1e-12, area(a) + area(b) - overlap)


def inside(point, quad):
    return all((quad[(i + 1) % 4][0] - a[0]) * (point[1] - a[1]) -
               (quad[(i + 1) % 4][1] - a[1]) * (point[0] - a[0]) >= 0 for i, a in enumerate(quad))


def anchor(box, policy="bottom-center"):
    return [(box[0] + box[2]) / 2, box[3] if policy == "bottom-center" else box[1] + .25 * (box[3] - box[1])]


def nms(detections, cutoff):
    retained = []
    for d in sorted(detections, key=lambda v: -v["confidence"]):
        if all(iou(d["bbox_xyxy_normalized"], kept["bbox_xyxy_normalized"]) <= cutoff for kept in retained):
            retained.append(d)
    return retained


class YoloBaseline:
    """Frozen YOLOv8n full-frame comparison; 640 square letterbox, class zero."""
    def __init__(self, path):
        import onnxruntime as ort
        if digest(path) != YOLO_SHA:
            raise ValueError("YOLO baseline checksum mismatch")
        options = ort.SessionOptions(); options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def predict(self, rgb, threshold=.05):
        height, width = rgb.shape[:2]
        ratio = min(640 / width, 640 / height)
        rw, rh = round(width * ratio), round(height * ratio)
        left, top = (640 - rw) // 2, (640 - rh) // 2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top:top + rh, left:left + rw] = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BILINEAR))
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.)
        raw = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0][0]
        if raw.shape[0] != 84:
            raise ValueError("Expected YOLOv8n COCO [1,84,N] output")
        detections = []
        for row in raw[:, raw[4] >= threshold].T:
            cx, cy, w, h = row[:4]
            box = np.clip([(cx - w / 2 - left) / (ratio * width), (cy - h / 2 - top) / (ratio * height),
                           (cx + w / 2 - left) / (ratio * width), (cy + h / 2 - top) / (ratio * height)], 0, 1)
            if area(box) > 0:
                detections.append({"bbox_xyxy_normalized": box.tolist(), "confidence": float(row[4]), "class_name": "person"})
        return nms(detections, .5)


def tiled_predict(detector, rgb, merge_iou=.5):
    """Full frame + six overlapping square crops. Comparison-only, not production default."""
    height, width = rgb.shape[:2]
    side = min(width, height, math.ceil(max(width / 2.6, height / 1.8)))
    boxes = detector.predict(rgb, .05)
    for y in sorted(set(np.linspace(0, height - side, 2).round().astype(int))):
        for x in sorted(set(np.linspace(0, width - side, 3).round().astype(int))):
            for d in detector.predict(rgb[y:y + side, x:x + side], .05):
                b = d["bbox_xyxy_normalized"]
                # The full-frame pass covers discarded inner-edge fragments.
                if (x > 0 and b[0] < .005) or (y > 0 and b[1] < .005) or (x + side < width and b[2] > .995) or (y + side < height and b[3] > .995):
                    continue
                converted = [(x + b[0] * side) / width, (y + b[1] * side) / height,
                             (x + b[2] * side) / width, (y + b[3] * side) / height]
                boxes.append({**d, "bbox_xyxy_normalized": converted})
    return nms(boxes, merge_iou)


def prepare(video, output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "annotations.json"
    if path.exists():
        raise ValueError("Annotation bundle already exists; choose a new output directory.")
    frames = []
    for index, (t, rgb) in enumerate(frame_samples(video, DEV_TIMES + LOCKED_TIMES)):
        image = output / f"frame-{index:02}.png"
        Image.fromarray(rgb).save(image)
        frames.append({"id": index, "split": "dev" if index < 16 else "locked", "time": t, "image": image.name,
                       "image_sha256": digest(image), "width": rgb.shape[1], "height": rgb.shape[0],
                       "reviewed": False, "people": [], "ignore_regions": [],
                       "pool_quad": [[.085, .08], [.95, .14], [.96, .91], [.22, .91]]})
    write_json(path, {"schema": "pool-person-labels/1", "source_sha256": digest(video), "frames": frames,
                      "label_policy": "Visible attributable extent, including clearly visible submerged parts. Tag head_only/partial/deck/occluded; in_pool boolean. Ignore only inseparable crowds at 200% zoom.",
                      "independence": "One recording; dev and locked sets share a camera/session. No generalization claim."})
    print(f"Prepared {len(frames)} UNREVIEWED frames: {path}")


def predict(labels, output, model, manifest, yolo_path, tiled=False, merge_iou=.5):
    data = read_json(labels)
    detector = OnnxDetector(manifest) if model == "rfdetr" else YoloBaseline(yolo_path)
    results = []
    for frame in data["frames"]:
        image = labels.parent / frame["image"]
        if digest(image) != frame["image_sha256"]:
            raise ValueError("Evaluation image changed")
        rgb = np.asarray(Image.open(image).convert("RGB"))
        before = time.perf_counter()
        detections = tiled_predict(detector, rgb, merge_iou) if tiled else detector.predict(rgb, .05)
        results.append({"id": frame["id"], "detections": detections, "elapsed_ms": (time.perf_counter() - before) * 1000})
        print(f"{model} frame {frame['id']}: {len(detections)} candidates >= .05", flush=True)
    artifact_sha = read_json(manifest)["onnx"]["sha256"] if model == "rfdetr" else YOLO_SHA
    write_json(output, {"schema": "pool-person-predictions/1", "model": model, "model_sha256": artifact_sha,
                        "source_sha256": data["source_sha256"], "frames": results, "tiling": tiled, "merge_iou": merge_iou if tiled else None,
                        "candidate_threshold": .05, "preprocessing": "RF official NumPy parity" if model == "rfdetr" else "640 letterbox; Pillow bilinear; baseline runtime differs from prior browser canvas",
                        "image_hashes": {str(f["id"]): f["image_sha256"] for f in data["frames"]}})


def validate_labels(data, predictions, split):
    if data["source_sha256"] != predictions["source_sha256"]:
        raise ValueError("Video fingerprints differ")
    frames = [f for f in data["frames"] if f["split"] == split]
    expected = 16 if split == "dev" else 8
    if len(frames) != expected or any(f.get("reviewed") is not True for f in frames):
        raise ValueError(f"Need all {expected} {split} frames manually reviewed. Unreviewed templates are not ground truth.")
    if len({f["id"] for f in frames}) != len(frames):
        raise ValueError("Duplicate annotation frame IDs")
    predicted = {f["id"]: f for f in predictions["frames"]}
    if len(predicted) != len(predictions["frames"]):
        raise ValueError("Duplicate prediction frame IDs")
    for f in frames:
        if f["id"] not in predicted or predictions["image_hashes"].get(str(f["id"])) != f["image_sha256"]:
            raise ValueError("Missing predictions or mismatched source image")
        for obj in f["people"]:
            box = obj["bbox_xyxy_normalized"]
            if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in box) or area(box) <= 0:
                raise ValueError("Invalid annotation box")
            if not isinstance(obj.get("in_pool"), bool):
                raise ValueError("Each person needs an in_pool boolean")
    return frames, predicted


def development_digest(data):
    return hashlib.sha256(json.dumps([f for f in data["frames"] if f["split"] == "dev"], sort_keys=True).encode()).hexdigest()


def measure(data, predictions, split, threshold, membership="bottom-center"):
    frames, predicted = validate_labels(data, predictions, split)
    tp = fp = fn = duplicates = ignored = membership_errors = 0
    bands = {"under_24px": [0, 0], "24_to_64px": [0, 0], "over_64px": [0, 0], "head_only": [0, 0], "partial": [0, 0]}
    ranked = []
    for frame in frames:
        truth, matched = frame["people"], set()
        for d in sorted(predicted[frame["id"]]["detections"], key=lambda v: -v["confidence"]):
            if d["confidence"] < threshold: continue
            box = d["bbox_xyxy_normalized"]
            candidates = [(iou(box, obj["bbox_xyxy_normalized"]), j) for j, obj in enumerate(truth) if j not in matched]
            overlap, j = max(candidates, default=(0, -1))
            if overlap >= .5:
                tp += 1; matched.add(j); ranked.append((d["confidence"], 1))
                membership_errors += inside(anchor(box, membership), frame["pool_quad"]) != truth[j]["in_pool"]
            elif any(intersection(box, region) / max(area(box), 1e-12) >= .5 for region in frame["ignore_regions"]):
                ignored += 1
            else:
                fp += 1; ranked.append((d["confidence"], 0))
                duplicates += any(iou(box, truth[k]["bbox_xyxy_normalized"]) >= .5 for k in matched)
        fn += len(truth) - len(matched)
        for j, obj in enumerate(truth):
            height = (obj["bbox_xyxy_normalized"][3] - obj["bbox_xyxy_normalized"][1]) * frame["height"]
            group = "under_24px" if height < 24 else "24_to_64px" if height <= 64 else "over_64px"
            for key in [group] + [v for v in obj.get("tags", []) if v in {"head_only", "partial"}]:
                bands[key][0] += int(j in matched); bands[key][1] += 1
    ranks = np.array([r[1] for r in sorted(ranked, key=lambda r: -r[0])], dtype=float)
    precision = np.cumsum(ranks) / np.arange(1, len(ranks) + 1) if len(ranks) else np.array([])
    recall = np.cumsum(ranks) / max(1, tp + fn)
    ap = float(np.mean([precision[recall >= x].max(initial=0) for x in np.linspace(0, 1, 101)]))
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "duplicates": duplicates, "ignored_predictions": ignored,
            "membership_errors_among_matches": membership_errors, "recall_groups": {k: {"matched": v[0], "total": v[1]} for k, v in bands.items()},
            "ap50_above_threshold": ap, "frames": len(frames)}


def choose_anchor(data):
    dev = [f for f in data["frames"] if f["split"] == "dev"]
    errors = {name: sum(inside(anchor(p["bbox_xyxy_normalized"], name), f["pool_quad"]) != p["in_pool"] for f in dev for p in f["people"])
              for name in ["bottom-center", "upper-quarter"]}
    return min(errors, key=errors.get), errors


def select(labels, candidates, output):
    if output.exists(): raise ValueError("Policy already frozen; choose a new experiment directory instead of overwriting it.")
    data, pred = read_json(labels), read_json(candidates)
    validate_labels(data, pred, "dev")
    membership, errors = choose_anchor(data)
    scores = sorted({.2, .5} | {d["confidence"] for f in pred["frames"] if f["id"] in {x["id"] for x in data["frames"] if x["split"] == "dev"} for d in f["detections"] if d["confidence"] >= .2})
    results = [measure(data, pred, "dev", score, membership) for score in scores]
    eligible = [r for r in results if r["tp"] > 0 and r["precision"] >= .9]
    best = max(eligible, key=lambda r: (r["recall"], r["precision"], r["threshold"])) if eligible else None
    write_json(output, {"passed": best is not None, "model": pred["model"], "predictions_sha256": digest(candidates),
                        "dev_labels_sha256": development_digest(data),
                        "labels_sha256_at_selection": digest(labels), "selected": best, "membership_anchor": membership,
                        "anchor_errors_on_ground_truth": errors, "precision_target": .9,
                        "dev_ap50_candidates_above_005": measure(data, pred, "dev", .05, membership)["ap50_above_threshold"],
                        "scope": "Engineering screen on one recording; no generalization claim."})
    print(json.dumps({"passed": bool(best), "selected": best}, indent=2))


def evaluate(labels, candidates, policy, output):
    if output.exists(): raise ValueError("Locked evaluation already exists; do not overwrite or retune against it.")
    frozen = read_json(policy)
    if not frozen["passed"]: raise ValueError("No development configuration passed the precision screen.")
    if frozen["predictions_sha256"] != digest(candidates): raise ValueError("Predictions changed after policy selection")
    data, pred = read_json(labels), read_json(candidates)
    if development_digest(data) != frozen["dev_labels_sha256"]:
        raise ValueError("Development labels changed after policy selection")
    report = measure(data, pred, "locked", frozen["selected"]["threshold"], frozen["membership_anchor"])
    report.update(policy_sha256=digest(policy), labels_sha256=digest(labels), scope="Locked temporal segment from the same camera; not independent generalization.")
    write_json(output, report); print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--video", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("predict"); p.add_argument("--labels", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=["rfdetr", "yolo"], required=True); p.add_argument("--manifest", type=Path, default=ROOT / "artifacts/models/rfdetr-nano/manifest.json")
    p.add_argument("--yolo-path", type=Path, default=ROOT / "frontend/vendor/yolov8n.onnx"); p.add_argument("--tiled", action="store_true"); p.add_argument("--merge-iou", type=float, choices=[.5, .7], default=.5)
    for command in ["select", "evaluate"]:
        p = sub.add_parser(command); p.add_argument("--labels", type=Path, required=True); p.add_argument("--candidates", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
        if command == "evaluate": p.add_argument("--policy", type=Path, required=True)
    args = vars(parser.parse_args()); command = args.pop("command")
    {"prepare": prepare, "predict": predict, "select": select, "evaluate": evaluate}[command](**args)


if __name__ == "__main__":
    main()
