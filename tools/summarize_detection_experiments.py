"""Summarize frozen detector candidates without treating counts as accuracy."""
import argparse
import hashlib
import json
from pathlib import Path


THRESHOLDS = (0.2, 0.35, 0.5)
ARTIFACTS = ("rfdetr-full", "yolo-full", "rfdetr-tiled-05", "yolo-tiled-05")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_summary(directory):
    labels_path = directory / "annotations.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    dev_ids = [f["id"] for f in labels["frames"] if f["split"] == "dev"]
    if not dev_ids:
        raise ValueError("No dev frames in annotations")
    reports = []
    for name in ARTIFACTS:
        path = directory / f"{name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        by_id = {f["id"]: f for f in data["frames"]}
        if len(by_id) != len(data["frames"]) or any(i not in by_id for i in dev_ids):
            raise ValueError(f"{name}: duplicate or missing frame IDs")
        if data.get("source_sha256") != labels.get("source_sha256"):
            raise ValueError(f"{name}: source fingerprint differs from annotations")
        if any(data.get("image_hashes", {}).get(str(f["id"])) != f["image_sha256"]
               for f in labels["frames"] if f["id"] in dev_ids):
            raise ValueError(f"{name}: dev image fingerprint mismatch")
        frames = [by_id[i] for i in dev_ids]
        rows = []
        for threshold in THRESHOLDS:
            counts = [sum(d["confidence"] >= threshold for d in f["detections"]) for f in frames]
            rows.append({"threshold": threshold, "first_dev_frame_id": dev_ids[0],
                         "first_dev_frame_count": counts[0], "dev_count": sum(counts),
                         "dev_frames": len(frames)})
        timed = [f.get("elapsed_ms") for f in frames]
        rowspeed = {"dev_mean_ms": sum(timed) / len(timed) if all(v is not None for v in timed) else None,
                    "first_dev_frame_ms": by_id[dev_ids[0]].get("elapsed_ms")}
        reports.append({"name": name, "model": data.get("model"), "model_sha256": data.get("model_sha256"),
                        "source_sha256": data.get("source_sha256"), "predictions_sha256": sha256(path),
                        "dev_frame_ids": dev_ids, "counts": rows, "timing": rowspeed})
    return {"schema": "detection-experiment-summary/1", "interpretation":
            "Candidate detection counts only; not precision, recall, or verified people. Summaries use dev frames only; locked frames are excluded.",
            "thresholds": list(THRESHOLDS), "annotations_sha256": sha256(labels_path), "experiments": reports}


def markdown(summary):
    lines = ["# Detection experiment counts", "", summary["interpretation"], "",
             "Counts include detections with confidence at least the threshold. Timing is the stored prediction elapsed time.", "",
             "| Experiment | Model SHA-256 | First dev frame (.2 / .35 / .5) | Dev totals (.2 / .35 / .5) | First frame ms | Mean dev ms |",
             "|---|---|---:|---:|---:|---:|"]
    for item in summary["experiments"]:
        counts = item["counts"]
        first = " / ".join(str(x["first_dev_frame_count"]) for x in counts)
        total = " / ".join(str(x["dev_count"]) for x in counts)
        first_ms, mean_ms = item["timing"]["first_dev_frame_ms"], item["timing"]["dev_mean_ms"]
        fmt = lambda x: "n/a" if x is None else f"{x:.1f}"
        lines.append(f"| {item['name']} | `{item['model_sha256']}` | {first} | {total} | {fmt(first_ms)} | {fmt(mean_ms)} |")
    lines += ["", f"Annotations SHA-256: `{summary['annotations_sha256']}`.", ""]
    for item in summary["experiments"]:
        lines.append(f"{item['name']} source SHA-256: `{item['source_sha256']}`; predictions SHA-256: `{item['predictions_sha256']}`.")
    return "\n".join(lines) + "\n"


def contact_sheet(directory, output):
    """Draw only prediction boxes on the exact first dev image; labels are absent."""
    from PIL import Image, ImageDraw

    labels = json.loads((directory / "annotations.json").read_text(encoding="utf-8"))
    first = next((f for f in labels["frames"] if f["split"] == "dev"), None)
    if first is None:
        raise ValueError("No dev frames in annotations")
    image_path = directory / first["image"]
    if sha256(image_path) != first["image_sha256"]:
        raise ValueError("First dev image fingerprint mismatch")
    panels = []
    for artifact, threshold in (("rfdetr-full", .5), ("rfdetr-full", .2),
                                ("rfdetr-tiled-05", .5), ("rfdetr-tiled-05", .2)):
        predictions = json.loads((directory / f"{artifact}.json").read_text(encoding="utf-8"))
        frame = next((f for f in predictions["frames"] if f["id"] == first["id"]), None)
        if frame is None:
            raise ValueError(f"{artifact}: missing first dev frame")
        base = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(base)
        w, h = base.size
        chosen = [d for d in frame["detections"] if d["confidence"] >= threshold]
        for detection in chosen:
            x1, y1, x2, y2 = detection["bbox_xyxy_normalized"]
            draw.rectangle((x1 * w, y1 * h, x2 * w, y2 * h), outline=(255, 50, 50), width=2)
        panel = Image.new("RGB", (w, h + 36), "white")
        panel.paste(base, (0, 36))
        ImageDraw.Draw(panel).text((8, 10), f"{artifact}  >= {threshold:.2f}: {len(chosen)} candidates; unverified", fill="black")
        panels.append(panel)
    w, h = panels[0].size
    sheet = Image.new("RGB", (2 * w, 2 * h), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * w, (index // 2) * h))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/evaluation/reference"))
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/evaluation/reference/summary.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("artifacts/evaluation/reference/summary.md"))
    parser.add_argument("--contact-sheet", type=Path, help="Optional PNG of RF full/tiled predictions on the first dev frame")
    args = parser.parse_args()
    result = build_summary(args.directory)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(result), encoding="utf-8")
    if args.contact_sheet:
        contact_sheet(args.directory, args.contact_sheet)


if __name__ == "__main__":
    main()
