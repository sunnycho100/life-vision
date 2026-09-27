"""Audit a YOLO dataset ZIP without extracting files or executing repository code.

Usage: python tools/audit_paper_dataset.py path/to/dataset.zip --output report.json
The audit distinguishes exact image-byte duplicates from filename-stem overlap.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile


def audit(path):
    groups, stems = defaultdict(list), defaultdict(list)
    counts, empty, labels = Counter(), Counter(), defaultdict(Counter)
    annotation_counts = Counter()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if "/images/" not in name or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            split = name.split("/")[0]
            data = archive.read(name)
            counts[split] += 1
            groups[hashlib.sha256(data).hexdigest()].append(name)
            stems[Path(name).name.split(".rf.")[0]].append(name)
            label_path = name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            rows = archive.read(label_path).decode("utf-8").strip().splitlines()
            annotation_counts[len(rows)] += 1
            if not rows:
                empty[split] += 1
            for line in rows:
                labels[split][line.split()[0]] += 1
        metadata = {name: archive.read(name).decode("utf-8", errors="replace")
                    for name in ("data.yaml", "README.dataset.txt", "README.roboflow.txt")
                    if name in archive.namelist()}
    cross = lambda collection: [v for v in collection.values() if len({x.split("/")[0] for x in v}) > 1]
    exact, named = cross(groups), cross(stems)
    overlap = lambda collection, split: len({n for group in collection if any(x.startswith("train/") for x in group) for n in group if n.startswith(split + "/")})
    with open(path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"archive_sha256": digest, "images_per_split": dict(counts),
            "boxes_by_class": {k: dict(v) for k, v in labels.items()},
            "empty_label_images": dict(empty), "boxes_per_image_histogram": dict(annotation_counts),
            "exact_cross_split_groups": exact, "same_stem_cross_split_groups": named,
            "summary": {"exact_cross_split_groups": len(exact), "exact_test_images_also_in_train": overlap(exact, "test"),
                        "exact_valid_images_also_in_train": overlap(exact, "valid"),
                        "same_stem_cross_split_groups": len(named), "same_stem_test_images_also_in_train": overlap(named, "test"),
                        "same_stem_valid_images_also_in_train": overlap(named, "valid")},
            "metadata": metadata}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.archive)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
