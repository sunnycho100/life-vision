"""Controlled Nano inference-resolution experiment on DEVELOPMENT frames only.

Uses the pinned official Python SDK, without changing production ONNX or labels.
Counts are candidates, not accuracy. Run with the reference Python environment.
"""
import argparse
from importlib.metadata import version
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image

from backend.common import ROOT, digest, read_json, write_json
from backend.detector import PythonDetector


def run(labels, manifest, output, resolutions, limit):
    if output.exists():
        raise ValueError("Use a new experiment directory; existing results must be preserved.")
    if any(size <= 0 or size % 32 for size in resolutions):
        raise ValueError("Pinned Nano requires dimensions divisible by 16 * 2 = 32.")
    data = read_json(labels)
    frames = [f for f in data["frames"] if f["split"] == "dev"]
    if limit:
        frames = frames[:limit]
    if not frames:
        raise ValueError("No development frames")
    adapter = PythonDetector(manifest, "cpu")
    output.mkdir(parents=True)
    for size in resolutions:
        results = []
        for frame in frames:
            path = labels.parent / frame["image"]
            if digest(path) != frame["image_sha256"]:
                raise ValueError("Development image hash changed")
            rgb = np.asarray(Image.open(path).convert("RGB"))
            if not results:
                # One unmeasured warmup per shape; no download or model construction in timings.
                adapter.model.predict(rgb, threshold=.05, shape=(size, size))
            before = time.perf_counter()
            prediction = adapter.model.predict(rgb, threshold=.05, shape=(size, size))
            elapsed = (time.perf_counter() - before) * 1000
            scale = np.array([frame["width"], frame["height"]] * 2)
            detections = [{"bbox_xyxy_normalized": np.clip(box / scale, 0, 1).tolist(),
                           "confidence": float(score), "class_name": "person"}
                          for box, score, category in zip(prediction.xyxy, prediction.confidence, prediction.class_id)
                          if int(category) == adapter.manifest["person_id"]]
            results.append({"id": frame["id"], "elapsed_ms": elapsed, "detections": detections})
            counts = {str(t): sum(d["confidence"] >= t for d in detections) for t in [.2, .35, .5]}
            print(f"shape={size}, dev={frame['id']}, candidates={counts}, ms={elapsed:.1f}", flush=True)
        write_json(output / f"nano-{size}.json", {
            "schema": "pool-person-predictions/1", "model": "rfdetr-nano-sdk",
            "model_sha256": adapter.manifest["checkpoint"]["sha256"],
            "source_sha256": data["source_sha256"], "frames": results,
            "image_hashes": {str(f["id"]): f["image_sha256"] for f in frames},
            "resolution": size, "candidate_threshold": .05, "tiling": False,
            "runtime": {**adapter.runtime, "python": platform.python_version(),
                        "machine": platform.machine(), "threads": 4, "numpy": version("numpy")},
            "preprocessing": "Official SDK predict(shape=(size,size)); RGB; FP32",
            "scope": "Development only. Same pretrained weights. No accuracy labels or production promotion."
        })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=ROOT / "artifacts/evaluation/reference/annotations.json")
    p.add_argument("--manifest", type=Path, default=ROOT / "artifacts/models/rfdetr-nano/manifest.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolutions", nargs="+", type=int, default=[384, 512, 640])
    p.add_argument("--limit", type=int, default=0, help="0 runs all development frames")
    args = p.parse_args()
    if args.limit < 0:
        p.error("limit must be nonnegative")
    run(**vars(args))
