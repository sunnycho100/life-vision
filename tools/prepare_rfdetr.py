"""Download the pinned official checkpoint, run SDK reference, export and gate ONNX.

Run from repository root: python -m tools.prepare_rfdetr --video PATH
Requires backend/requirements-reference.txt. Artifacts stay outside git.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import time
from urllib.request import urlopen
import numpy as np
from backend.common import ROOT, digest, read_json, write_json
from backend.detector import PythonDetector, preprocess, decode, onnx_options

URL = "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth"
SHA256 = "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d"


def frame_samples(video, times):
    import av
    next_time = 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        origin = stream.start_time
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if origin is None:
                origin = frame.pts
            t = float((frame.pts - origin) * stream.time_base)
            if t + 1e-7 >= times[next_time]:
                yield t, frame.to_ndarray(format="rgb24")
                next_time += 1
                if next_time == len(times):
                    return
    raise ValueError("Video does not cover all requested verification frames.")


def compare_detections(reference, actual, width, height):
    from scipy.optimize import linear_sum_assignment
    if len(reference) != len(actual):
        raise AssertionError(f"Detection count differs: SDK={len(reference)}, ONNX={len(actual)}")
    if not reference:
        return {"boxes": 0, "max_pixel_error": 0., "max_score_error": 0.}
    scale = np.array([width, height, width, height])
    a = np.array([d["bbox_xyxy_normalized"] for d in reference]) * scale
    b = np.array([d["bbox_xyxy_normalized"] for d in actual]) * scale
    rows, cols = linear_sum_assignment(np.abs(a[:, None] - b[None]).sum(axis=2))
    box_error = float(np.abs(a[rows] - b[cols]).max())
    score_error = max(abs(reference[i]["confidence"] - actual[j]["confidence"]) for i, j in zip(rows, cols))
    if box_error > .5 or score_error > .001:
        raise AssertionError(f"Decoded parity failed: {box_error=}, {score_error=}")
    return {"boxes": len(reference), "max_pixel_error": box_error, "max_score_error": score_error}


def verify_native(manifest_path, provider="CPUExecutionProvider"):
    """Also runnable in the torch-free target environment using saved SDK tensors."""
    import onnxruntime as ort
    manifest_path = Path(manifest_path)
    m = read_json(manifest_path)
    model = manifest_path.parent / m["onnx"]["file"]
    if digest(model) != m["onnx"]["sha256"]:
        raise ValueError("ONNX checksum mismatch")
    options, optimization = onnx_options()
    if provider not in ort.get_available_providers():
        raise ValueError(f"Execution provider unavailable: {provider}")
    session = ort.InferenceSession(str(model), options, providers=[provider])
    reports = []
    for sample in m["verification_frames"]:
        saved = np.load(manifest_path.parent / sample["tensor_file"])
        tensor = preprocess(saved["rgb"])
        pre_error = float(np.abs(tensor - saved["input"]).max())
        if pre_error > 1e-4:
            raise AssertionError(f"Preprocessing parity failed: {pre_error}")
        before = time.perf_counter()
        boxes, logits = session.run(["dets", "labels"], {session.get_inputs()[0].name: saved["input"]})
        raw_error = {"boxes": float(np.abs(boxes - saved["boxes"]).max()), "logits": float(np.abs(logits - saved["logits"]).max())}
        if max(raw_error.values()) > .001:
            raise AssertionError(f"Raw tensor parity failed: {raw_error}")
        boxes, logits = session.run(["dets", "labels"], {session.get_inputs()[0].name: tensor})
        actual = decode(boxes, logits, .2)
        report = compare_detections(sample["reference_detections"], actual, sample["width"], sample["height"])
        reports.append({"time": sample["time"], "preprocessing_max_error": pre_error, "raw_max_error": raw_error,
                        **report, "two_forward_passes_ms": (time.perf_counter() - before) * 1000})
    return {"passed": True, "onnxruntime": ort.__version__, "provider": session.get_providers()[0], "graph_optimization": optimization,
            "platform": sysconfig.get_platform(), "model_sha256": m["onnx"]["sha256"], "frames": reports,
            "adapter_sha256": digest(ROOT / "backend/detector.py"),
            "scope": "Numerical equivalence only; not detection accuracy or generalization."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/models/rfdetr-nano")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--provider", default="CPUExecutionProvider")
    args = p.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if args.verify_only:
        report = verify_native(manifest_path, args.provider)
        write_json(output / ("deployment-parity-" + sysconfig.get_platform() + ".json"), report)
        print(json.dumps(report, indent=2)); return
    if args.video is None:
        p.error("--video is required to verify on real video frames")
    weights = output / "rf-detr-nano.pth"
    if not weights.exists():
        print("Downloading official RF-DETR Nano checkpoint", flush=True)
        part = weights.with_suffix(".part")
        with urlopen(URL, timeout=90) as response, part.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if digest(part) != SHA256:
            raise ValueError("Downloaded checkpoint does not match the pinned SHA-256.")
        part.replace(weights)
    if digest(weights) != SHA256:
        raise ValueError("Checkpoint SHA-256 mismatch.")
    manifest = {"model": "RFDETRNano", "rfdetr_version": "1.11.0", "resolution": 384, "precision": "fp32",
                "person_id": 1, "class_names": {"1": "person"}, "background_id": None, "class_layout": "sparse COCO",
                "num_select": 300, "preprocessing": "RGB float32/255, bilinear half-pixel square resize, antialias=False, ImageNet mean/std",
                "checkpoint": {"file": weights.name, "sha256": SHA256, "url": URL}, "parity": {"passed": False},
                "source_sha256": digest(args.video), "verification_frames": []}
    write_json(manifest_path, manifest)
    print("Loading official Python RF-DETR Nano", flush=True)
    reference = PythonDetector(manifest_path)
    import torch
    import torchvision.transforms.functional as F
    from PIL import Image
    torch.set_num_threads(4)
    for index, (t, rgb) in enumerate(frame_samples(args.video, [8., 22., 45., 70., 90.])):
        print(f"SDK reference at {t:.3f}s", flush=True)
        predictions = reference.predict(rgb, .2)
        tensor = F.normalize(F.resize(F.to_tensor(Image.fromarray(rgb)), [384, 384], antialias=False),
                             [.485, .456, .406], [.229, .224, .225]).unsqueeze(0)
        with torch.no_grad():
            raw = reference.model.model.model(tensor)
        file = f"reference-{index}.npz"
        np.savez_compressed(output / file, rgb=rgb, input=tensor.numpy(), boxes=raw["pred_boxes"].numpy(), logits=raw["pred_logits"].numpy())
        manifest["verification_frames"].append({"time": t, "width": rgb.shape[1], "height": rgb.shape[0],
                                                "tensor_file": file, "reference_detections": predictions})
    print("Exporting official FP32 ONNX graph", flush=True)
    exported = reference.model.export(output_dir=str(output), format="onnx", batch_size=1, dynamic_batch=False,
                                      opset_version=17, fp16=False, verbose=False, output_name="rfdetr-nano-fp32")
    import onnx
    onnx.checker.check_model(str(exported))
    manifest["onnx"] = {"file": Path(exported).name, "sha256": digest(exported), "opset": 17,
                        "optimization_policy": {"win-arm64": "BASIC", "default": "ALL"}}
    manifest["versions"] = {name: version(name) for name in ["rfdetr", "torch", "torchvision", "onnx", "onnxruntime", "numpy", "av"]}
    write_json(manifest_path, manifest)
    (output / "reference-environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    manifest["parity"] = verify_native(manifest_path, args.provider)
    write_json(output / ("deployment-parity-" + sysconfig.get_platform() + ".json"), manifest["parity"])
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["parity"], indent=2), flush=True)


if __name__ == "__main__":
    main()
