"""RF-DETR Nano adapters. Full-frame detection, no tracker and no NMS.

Contract checked against rfdetr 1.11.0 official predict/export implementation.
The NumPy resize implements bilinear half-pixel centers, antialias=False.
"""
from importlib.metadata import version
from pathlib import Path
import sysconfig
import numpy as np
from backend.common import digest, read_json

RF_VERSION = "1.11.0"
MEAN = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
STD = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]


def onnx_options():
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    optimization = "BASIC" if sysconfig.get_platform() == "win-arm64" else "ALL"
    options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, "ORT_ENABLE_" + optimization)
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options, optimization


def preprocess(rgb, size=384):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected native-resolution RGB uint8 HWC pixels.")
    h, w = rgb.shape[:2]
    image = rgb.astype(np.float32) / np.float32(255)
    x = np.maximum((np.arange(size, dtype=np.float32) + .5) * np.float32(w / size) - .5, 0)
    y = np.maximum((np.arange(size, dtype=np.float32) + .5) * np.float32(h / size) - .5, 0)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = (x - x0).astype(np.float32)[None, :, None], (y - y0).astype(np.float32)[:, None, None]
    top = image[y0[:, None], x0] * (1 - wx) + image[y0[:, None], x1] * wx
    bottom = image[y1[:, None], x0] * (1 - wx) + image[y1[:, None], x1] * wx
    resized = (top * (1 - wy) + bottom * wy).transpose(2, 0, 1)
    return np.ascontiguousarray(((resized - MEAN) / STD)[None], dtype=np.float32)


def decode(boxes, logits, threshold=.2, person_id=1, num_select=300, background_id=None):
    """Global top-k query/class pairs, then person filter. Sparse COCO: no final background slot."""
    if boxes.ndim == 3:
        boxes, logits = boxes[0], logits[0]
    if boxes.shape != (logits.shape[0], 4) or not np.isfinite(boxes).all() or not np.isfinite(logits).all():
        raise ValueError("Invalid RF-DETR output tensors.")
    scores = 1 / (1 + np.exp(-np.clip(logits, -88, 88)))
    if background_id is not None:
        scores[:, background_id] = -1
    flat = scores.ravel()
    count = min(num_select, flat.size)
    selected = np.argpartition(-flat, count - 1)[:count]
    selected = selected[np.argsort(-flat[selected], kind="stable")]
    output = []
    for slot in selected:
        query, category = divmod(int(slot), logits.shape[1])
        score = float(flat[slot])
        if score <= threshold or category != person_id:
            continue
        cx, cy, w, h = boxes[query]
        box = np.clip([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0, 1)
        if box[2] > box[0] and box[3] > box[1]:
            output.append({"bbox_xyxy_normalized": [float(v) for v in box], "confidence": score, "class_name": "person"})
    return output


def load_manifest(path):
    path = Path(path).resolve()
    m = read_json(path)
    if m.get("rfdetr_version") != RF_VERSION or m.get("model") != "RFDETRNano" or m.get("resolution") != 384:
        raise ValueError("Expected pinned RF-DETR Nano 1.11.0, 384 × 384 manifest.")
    if m.get("precision") != "fp32" or m.get("person_id") != 1 or m.get("background_id") is not None:
        raise ValueError("Unsupported checkpoint precision or class layout.")
    if m.get("num_select") != 300:
        raise ValueError("Unexpected query selection configuration.")
    return m


class OnnxDetector:
    def __init__(self, manifest_path, provider="CPUExecutionProvider"):
        import onnxruntime as ort
        self.manifest = load_manifest(manifest_path)
        artifact = self.manifest["onnx"]
        model = Path(manifest_path).parent / artifact["file"]
        if digest(model) != artifact["sha256"]:
            raise ValueError("ONNX checksum mismatch.")
        if not self.manifest.get("parity", {}).get("passed"):
            raise ValueError("ONNX export has not passed reference parity. Use the Python backend.")
        if provider not in ort.get_available_providers():
            raise ValueError(f"Requested execution provider unavailable: {provider}")
        # The validated optimization differs by architecture because of tied encoder proposals.
        opts, optimization = onnx_options()
        report_path = Path(manifest_path).parent / ("deployment-parity-" + sysconfig.get_platform() + ".json")
        report = read_json(report_path) if report_path.is_file() else {}
        if not (report.get("passed") and report.get("model_sha256") == artifact["sha256"]
                and report.get("provider") == provider and report.get("onnxruntime") == ort.__version__
                and report.get("graph_optimization") == optimization and report.get("adapter_sha256") == digest(__file__)):
            raise ValueError("Run python -m tools.prepare_rfdetr --verify-only on this runtime before ONNX deployment.")
        self.session = ort.InferenceSession(str(model), opts, providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].shape != [1, 3, 384, 384]:
            raise ValueError("Unexpected ONNX input shape.")
        names = {o.name for o in self.session.get_outputs()}
        if not {"dets", "labels"}.issubset(names):
            raise ValueError("Expected named RF-DETR dets / labels outputs.")
        self.runtime = {"backend": "onnx", "onnxruntime": ort.__version__, "provider": self.session.get_providers()[0],
                        "graph_optimization": optimization, "platform": sysconfig.get_platform()}

    def predict(self, rgb, threshold=.2):
        boxes, logits = self.session.run(["dets", "labels"], {self.input_name: preprocess(rgb)})
        return decode(boxes, logits, threshold, self.manifest["person_id"], self.manifest["num_select"], self.manifest["background_id"])


class PythonDetector:
    def __init__(self, manifest_path, provider="cpu"):
        import torch
        from rfdetr import RFDETRNano
        from rfdetr.assets.coco_classes import COCO_CLASSES
        if version("rfdetr") != RF_VERSION:
            raise ValueError("Install rfdetr==1.11.0 in the reference environment.")
        self.manifest = load_manifest(manifest_path)
        artifact = self.manifest["checkpoint"]
        weights = Path(manifest_path).parent / artifact["file"]
        if digest(weights) != artifact["sha256"] or COCO_CLASSES[self.manifest["person_id"]] != "person":
            raise ValueError("Checkpoint checksum or person class mapping mismatch.")
        torch.set_num_threads(4)
        self.model = RFDETRNano(pretrain_weights=str(weights.resolve()), device=provider)
        self.runtime = {"backend": "python", "torch": torch.__version__, "rfdetr": RF_VERSION, "provider": provider}

    def predict(self, rgb, threshold=.2):
        result = self.model.predict(rgb, threshold=threshold)
        height, width = rgb.shape[:2]
        scale = np.array([width, height, width, height])
        output = []
        for box, score, cls in zip(result.xyxy, result.confidence, result.class_id):
            if int(cls) != self.manifest["person_id"]:
                continue
            box = np.clip(box / scale, 0, 1)
            if box[2] > box[0] and box[3] > box[1]:
                output.append({"bbox_xyxy_normalized": box.tolist(), "confidence": float(score), "class_name": "person"})
        return output


def create_detector(manifest_path, backend="onnx", provider=None):
    if backend == "python":
        return PythonDetector(manifest_path, provider or "cpu")
    if backend == "onnx":
        return OnnxDetector(manifest_path, provider or "CPUExecutionProvider")
    raise ValueError("Unknown detector backend.")
