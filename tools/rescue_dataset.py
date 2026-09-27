"""Local rescue-video annotation bundles; predictions are never ground truth.

python -m tools.rescue_dataset --help
"""
import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from backend.common import digest, probe, read_json, write_json

SCHEMA = "pool-rescue-dataset/1"
LABELS = {"visible_distress", "ordinary_swimming", "submerged_visible", "not_visible", "uncertain", "rescue"}


def contained(root, filename):
    path = (Path(root) / filename).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Path escapes dataset directory")
    return path


def box_valid(box):
    return (isinstance(box, list) and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in box)
            and box[2] > box[0] and box[3] > box[1])


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def sample_at(path, times):
    """Decode once, retain actual PTS and frame indices; never derive time from FPS."""
    import av
    pending = iter(times)
    target = next(pending, None)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = stream.start_time
        previous = -math.inf
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f"Frame without timestamp: {path}")
            if origin is None:
                origin = frame.pts
            t = float((frame.pts - origin) * stream.time_base)
            if t < previous:
                raise ValueError(f"Nonmonotonic timestamps: {path}")
            previous = t
            if target is not None and t + 1e-7 >= target:
                yield index, frame.pts, str(stream.time_base), t, frame.to_image()
                target = next(pending, None)
            # Continue decoding to catch truncated/corrupt tails.
        if target is not None:
            raise ValueError(f"Video does not cover requested sample: {path}")


def prepare(raw, output, samples=4):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    videos = sorted(raw.glob("*.mp4"), key=lambda p: p.name.lower())
    if not videos or not 1 <= samples <= 30:
        raise ValueError("Need MP4 recordings and 1..30 samples per source")
    if output.exists():
        raise ValueError("Use a new output directory; existing annotations are never overwritten")
    output.mkdir(parents=True)
    sources, frames, errors, seen = [], [], [], {}
    for video in videos:
        try:
            sha, metadata = digest(video), probe(video)
            if sha in seen:
                raise ValueError(f"Exact duplicate of {seen[sha]}; keep one copy")
            seen[sha] = video.name
            info_path = video.with_suffix(".info.json")
            info = read_json(info_path) if info_path.exists() else {}
            source = {"id": video.stem, "filename": video.name, "sha256": sha,
                      "title": info.get("title", video.stem), "source_url": info.get("webpage_url", ""),
                      "uploader": info.get("uploader", ""), "upload_date": info.get("upload_date"),
                      "metadata_sha256": digest(info_path) if info_path.exists() else None,
                      "group_id": video.stem, "split": None, **metadata,
                      "review": {"reviewer": "", "reviewed": False, "notes": "", "camera_group": "unknown",
                                 "same_event_group": video.stem, "rights_status": "unverified",
                                 "rescuer_entry": None, "rescue_contact": None,
                                 "distress_onset_min": None, "distress_onset_max": None,
                                 "intervals": [], "target_keyframes": []}}
            source_frames = []
            times = [metadata["duration"] * (i + .5) / samples for i in range(samples)]
            sheet = Image.new("RGB", (480 * 2, 296 * math.ceil(samples / 2)), "#142529")
            draw = ImageDraw.Draw(sheet)
            for local, (index, pts, time_base, t, image) in enumerate(sample_at(video, times)):
                name = f"{video.stem}-{local:02}.png"
                image.save(output / name)
                source_frames.append({"id": f"{video.stem}-{local:02}", "source_id": video.stem,
                    "source_sha256": sha, "split": None, "time": t, "frame_index": index, "pts": pts,
                    "time_base": time_base, "image": name, "image_sha256": digest(output / name),
                    "width": image.width, "height": image.height, "reviewed": False, "reviewer": "",
                    "people": [], "ignore_regions": [], "pool_quad": [], "pool_calibrated": False})
                thumb = image.copy(); thumb.thumbnail((480, 270))
                x, y = local % 2 * 480, local // 2 * 296
                sheet.paste(thumb, (x, y + 24))
                draw.text((x + 5, y + 5), f"{video.stem} | {t:.3f}s", fill="white")
            sheet.save(output / f"contact-{video.stem}.jpg")
            frames.extend(source_frames); sources.append(source)
            print(f"Verified {video.name}: {metadata['duration']:.3f}s, {len(source_frames)} frames", flush=True)
        except Exception as exc:
            errors.append({"file": video.name, "error": str(exc)})
    # Split whole sources BEFORE suggestions/annotation. Semantic re-uploads still need human grouping.
    order = sorted(s["id"] for s in sources)
    random.Random(42).shuffle(order)
    n_hold = max(1, round(len(order) * .16)) if len(order) >= 3 else 0
    assignments = {v: "test" if i < n_hold else "valid" if i < 2*n_hold else "train" for i, v in enumerate(order)}
    for s in sources:
        s["split"] = assignments[s["id"]]
    for f in frames:
        f["split"] = assignments[f["source_id"]]
    manifest_id = hashlib.sha256(json.dumps(sorted(s["sha256"] for s in sources)).encode()).hexdigest()
    manifest = {"schema": SCHEMA, "manifest_id": manifest_id, "raw_directory": str(raw),
        "split_seed": 42, "split_reviewed": False, "split_reviewer": "", "sources": sources, "errors": errors,
        "limitations": ["Source-disjoint provisional split; same event, re-uploads and camera grouping require human review.",
                        "Rescue-selected collection lacks independently sampled normal operating footage.",
                        "No source/video title is used as a frame or event label."]}
    write_json(output / "rescue-manifest.json", manifest)
    write_json(output / "annotations.json", {"schema": "pool-person-labels/1", "dataset_id": manifest_id,
        "frames": frames, "label_policy": "Every distinguishable person's visible extent, including visible underwater parts. Never invent a box for an invisible person. Unknown water membership is null.",
        "independence": "Whole-source provisional splits; all labels unreviewed. No model suggestions on valid/test."})
    write_json(output / "preparation-report.json", {"sources": len(sources), "frames": len(frames),
        "seconds": sum(s["duration"] for s in sources), "errors": errors,
        "splits": {split: sum(s["split"] == split for s in sources) for split in ("train", "valid", "test")},
        "reviewed_frames": 0, "reviewed_sources": 0, "training_ready": False})
    return manifest


def validate_manifest(manifest, check_files=False, require_split=False):
    if manifest.get("schema") != SCHEMA or not manifest.get("sources"):
        raise ValueError("Invalid or empty rescue manifest")
    ids, shas, groups = set(), set(), {}
    for s in manifest["sources"]:
        if s["id"] in ids or s["sha256"] in shas:
            raise ValueError("Duplicate source ID or source hash")
        ids.add(s["id"]); shas.add(s["sha256"])
        if s["split"] not in {"train", "valid", "test"} or not number(s["duration"]) or s["duration"] <= 0:
            raise ValueError("Invalid source split or duration")
        review = s["review"]
        group = review.get("same_event_group", "").strip()
        if not group:
            raise ValueError("Each source needs a same-event group, using source ID when distinct")
        if group in groups and groups[group] != s["split"]:
            raise ValueError("Same event crosses splits; regroup before approval")
        groups[group] = s["split"]
        if check_files and digest(contained(manifest["raw_directory"], s["filename"])) != s["sha256"]:
            raise ValueError("Source fingerprint changed")
        if review.get("reviewed") and not review.get("reviewer", "").strip():
            raise ValueError("Reviewed source requires reviewer name")
        intervals = review.get("intervals", [])
        for field in ("rescuer_entry", "rescue_contact", "distress_onset_min", "distress_onset_max"):
            v = review.get(field)
            if v is not None and (not number(v) or not 0 <= v <= s["duration"]):
                raise ValueError("Invalid event boundary: " + field)
        for first, last in (("rescuer_entry", "rescue_contact"), ("distress_onset_min", "distress_onset_max")):
            if review.get(first) is not None and review.get(last) is not None and review[first] > review[last]:
                raise ValueError("Reversed event boundary range")
        if review.get("reviewed") and not intervals and not review.get("notes", "").strip():
            raise ValueError("Reviewed source without intervals needs an exclusion explanation")
        for a in intervals:
            if (a.get("label") not in LABELS or not a.get("target_id", "").strip()
                or not number(a.get("start")) or not number(a.get("end"))
                or not 0 <= a["start"] < a["end"] <= s["duration"] + .001):
                raise ValueError("Invalid temporal label, target or bounds")
        # One observable state per target per time; uncertain is not silently resolved.
        for i, a in enumerate(intervals):
            if any(a["target_id"] == b["target_id"] and min(a["end"], b["end"]) > max(a["start"], b["start"]) + 1e-6
                   for b in intervals[i+1:]):
                raise ValueError("Overlapping intervals for one target; use nonoverlapping observable states")
        for k in review.get("target_keyframes", []):
            if (not k.get("target_id", "").strip() or not number(k.get("media_time"))
                or not 0 <= k["media_time"] <= s["duration"] or not box_valid(k.get("bbox_xyxy_normalized"))):
                raise ValueError("Invalid target keyframe")
    expected = hashlib.sha256(json.dumps(sorted(shas)).encode()).hexdigest()
    if manifest.get("manifest_id") != expected:
        raise ValueError("Manifest source fingerprints changed")
    if require_split and (not manifest.get("split_reviewed") or not manifest.get("split_reviewer", "").strip()
                          or manifest.get("split_fingerprint") != split_fingerprint(manifest)):
        raise ValueError("Human split approval required (and must match current groups/splits)")


def split_fingerprint(manifest):
    return hashlib.sha256(json.dumps(sorted((s["id"], s["sha256"], s["split"], s["review"]["same_event_group"])
                                           for s in manifest["sources"])).encode()).hexdigest()


def suggest(labels, manifest_path, model_manifest, limit=12, threshold=.2):
    import numpy as np
    from backend.detector import OnnxDetector
    from tools.evaluate_detection import tiled_predict
    labels, manifest_path = Path(labels), Path(manifest_path)
    data, manifest = read_json(labels), read_json(manifest_path)
    validate_manifest(manifest)
    if data.get("dataset_id") != manifest["manifest_id"]:
        raise ValueError("Mismatched dataset labels")
    if not 0 < threshold < 1 or limit < 1:
        raise ValueError("Invalid suggestion limit or threshold")
    detector = OnnxDetector(model_manifest)
    count = 0
    source_splits = {s["id"]: s["split"] for s in manifest["sources"]}
    for f in data["frames"]:
        if f["split"] != source_splits[f["source_id"]]:
            raise ValueError("Frame/source split mismatch; synchronize assignments before suggestions")
        if source_splits[f["source_id"]] != "train" or f["reviewed"] or f["people"] or f["ignore_regions"] or f.get("suggestions"):
            continue
        path = contained(labels.parent, f["image"])
        if digest(path) != f["image_sha256"]:
            raise ValueError("Frame fingerprint changed")
        result = tiled_predict(detector, np.asarray(Image.open(path).convert("RGB")))
        # Keep proposals separately as well as editable prefilled boxes; export gated by entire-frame human review.
        f["suggestions"] = {"model_manifest_sha256": digest(model_manifest), "threshold": threshold,
                            "method": "full-frame + six square tiles; cross-pass NMS .5", "human_verified": False}
        f["people"] = [{"bbox_xyxy_normalized": d["bbox_xyxy_normalized"], "in_pool": None, "tags": [],
                        "proposal_confidence": d["confidence"]} for d in result if d["confidence"] >= threshold]
        print(f"Draft {f['id']}: {len(f['people'])} proposals, requires all-person review", flush=True)
        count += 1
        if count == limit:
            break
    target = labels.parent / "annotations-suggested.json"
    if target.exists():
        raise ValueError("Suggestion file already exists; refusing to overwrite")
    write_json(target, data)
    return target


def export_coco(labels, manifest_path, output):
    labels, output = Path(labels), Path(output)
    data, manifest = read_json(labels), read_json(manifest_path)
    validate_manifest(manifest, check_files=True, require_split=True)
    if data.get("dataset_id") != manifest["manifest_id"]:
        raise ValueError("Labels belong to a different dataset")
    if output.exists():
        raise ValueError("Export directory exists; choose a new version")
    sources = {s["id"]: s for s in manifest["sources"]}
    accepted, excluded, seen, hashes = [], [], set(), set()
    for f in data["frames"]:
        if f["id"] in seen or f["image_sha256"] in hashes:
            raise ValueError("Duplicate frame ID/content; remove duplicates before export")
        seen.add(f["id"]); hashes.add(f["image_sha256"])
        source = sources[f["source_id"]]
        if f["source_sha256"] != source["sha256"] or f["split"] != source["split"]:
            raise ValueError("Frame source/split mismatch (regenerate labels after regrouping)")
        if f["split"] != "train" and f.get("suggestions"):
            raise ValueError("Validation/test frames must be independently labeled without model suggestions")
        if not number(f["time"]) or not 0 <= f["time"] <= source["duration"]:
            raise ValueError("Invalid frame timestamp")
        if f.get("reviewed") is not True or not f.get("reviewer", "").strip():
            excluded.append({"id": f["id"], "reason": "unreviewed"}); continue
        if f["ignore_regions"]:
            excluded.append({"id": f["id"], "reason": "crowd ignore not supported safely by current trainer"}); continue
        image = contained(labels.parent, f["image"])
        if digest(image) != f["image_sha256"]:
            raise ValueError("Frame image changed")
        with Image.open(image) as im:
            if im.size != (f["width"], f["height"]):
                raise ValueError("Frame dimensions mismatch")
        if any(not box_valid(p.get("bbox_xyxy_normalized")) for p in f["people"]):
            raise ValueError("Invalid person box")
        accepted.append(f)
    for split in ("train", "valid", "test"):
        if not any(f["split"] == split for f in accepted):
            raise ValueError(f"Need reviewed frames in {split}; unreviewed frames cannot become negative examples")
    output.mkdir(parents=True)
    annotation_id = 1
    for split in ("train", "valid", "test"):
        folder = output / split; folder.mkdir()
        coco = {"info": {"description": "Human-reviewed pool people; single class, no drowning labels"},
                "licenses": [], "images": [], "annotations": [],
                "categories": [{"id": 0, "name": "person", "supercategory": "none"}]}
        for image_id, f in enumerate((f for f in accepted if f["split"] == split), 1):
            name = Path(f["image"]).name
            shutil.copyfile(contained(labels.parent, f["image"]), folder / name)
            coco["images"].append({"id": image_id, "file_name": name, "width": f["width"], "height": f["height"],
                                   "source_id": f["source_id"], "media_time": f["time"]})
            for p in f["people"]:
                x1, y1, x2, y2 = p["bbox_xyxy_normalized"]
                b = [x1*f["width"], y1*f["height"], (x2-x1)*f["width"], (y2-y1)*f["height"]]
                coco["annotations"].append({"id": annotation_id, "image_id": image_id, "category_id": 0,
                                            "bbox": b, "area": b[2]*b[3], "iscrowd": 0})
                annotation_id += 1
        write_json(folder / "_annotations.coco.json", coco)
    write_json(output / "export-report.json", {"labels_sha256": digest(labels), "manifest_sha256": digest(manifest_path),
        "reviewed_frames": len(accepted), "excluded": excluded, "class_mapping": {"0": "person"},
        "rights_status": {s["id"]: s["review"]["rights_status"] for s in sources.values()},
        "warning": "Fine-tuned contiguous person ID 0 differs from pretrained COCO person ID 1. Deployment adapter must be updated and reverified."})
    return len(accepted)


def export_temporal(manifest_path, output):
    manifest, output = read_json(manifest_path), Path(output)
    validate_manifest(manifest, check_files=True, require_split=True)
    if output.exists():
        raise ValueError("Temporal export exists; choose a new version")
    rows = []
    for s in manifest["sources"]:
        review = s["review"]
        if not review.get("reviewed"):
            continue
        for a in review["intervals"]:
            keys = [k for k in review["target_keyframes"] if k["target_id"] == a["target_id"]]
            visible = [k for k in keys if a["start"] <= k["media_time"] <= a["end"]]
            # Preserve uncertainty/rescue for auditing; do not turn into binary training labels.
            entry = review.get("rescuer_entry")
            reasons = []
            if a["label"] not in {"visible_distress", "ordinary_swimming"}:
                reasons.append("uncertain_or_nonbehavior_label")
            if not visible:
                reasons.append("missing_visible_target_keyframe")
            if entry is None:
                reasons.append("rescuer_entry_unknown")
            elif a["end"] > entry:
                reasons.append("overlaps_rescue_response")
            eligible = not reasons
            rows.append({**a, "source_id": s["id"], "source_sha256": s["sha256"], "filename": s["filename"],
                "split": s["split"], "reviewer": review["reviewer"], "target_keyframes": keys,
                "eligible_for_distress_research": eligible,
                "exclusion_reasons": reasons, "rescuer_entry": entry,
                "onset_uncertainty": [review.get("distress_onset_min"), review.get("distress_onset_max")],
                "clinical_drowning_confirmed": None})
    if not rows:
        raise ValueError("No human-reviewed intervals; titles and model outputs are not labels")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare"); p.add_argument("--raw", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--samples", type=int, default=4)
    p = commands.add_parser("suggest"); p.add_argument("--labels", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--model", type=Path, required=True); p.add_argument("--limit", type=int, default=12); p.add_argument("--threshold", type=float, default=.2)
    p = commands.add_parser("validate"); p.add_argument("--manifest", type=Path, required=True)
    p = commands.add_parser("approve-splits"); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--reviewer", required=True); p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("export-coco"); p.add_argument("--labels", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("export-temporal"); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.raw, args.output, args.samples)
    elif args.command == "suggest":
        print(suggest(args.labels, args.manifest, args.model, args.limit, args.threshold))
    elif args.command == "validate":
        m = read_json(args.manifest); validate_manifest(m, check_files=True)
        print(json.dumps({"valid": True, "sources": len(m["sources"]), "reviewed_sources": sum(s["review"]["reviewed"] for s in m["sources"]), "split_reviewed": m["split_reviewed"]}))
    elif args.command == "approve-splits":
        m = read_json(args.manifest); validate_manifest(m, check_files=True)
        if not args.reviewer.strip() or args.output.exists():
            raise ValueError("Need reviewer name and new output filename")
        m.update(split_reviewed=True, split_reviewer=args.reviewer.strip(), split_fingerprint=split_fingerprint(m))
        write_json(args.output, m)
    elif args.command == "export-coco":
        print(f"Exported {export_coco(args.labels, args.manifest, args.output)} reviewed frames")
    elif args.command == "export-temporal":
        print(f"Exported {export_temporal(args.manifest, args.output)} reviewed intervals")


if __name__ == "__main__":
    main()
