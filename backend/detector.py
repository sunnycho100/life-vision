"""RF-DETR adapters (stock Nano, or a fine-tuned Small via the Python backend). Full-frame detection, no tracker and no NMS.

Contract checked against rfdetr 1.11.0 official predict/export implementation.
The NumPy resize implements bilinear half-pixel centers, antialias=False.
"""
from importlib.metadata import version
from pathlib import Path
import sysconfig
import numpy as np
from backend.common import digest, read_json, code_digest

RF_VERSION = "1.11.0"
MODELS = {"RFDETRNano": 384, "RFDETRSmall": 512}  # native resolution of each supported size
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
    if m.get("rfdetr_version") != RF_VERSION or MODELS.get(m.get("model")) != m.get("resolution"):
        raise ValueError("Expected pinned RF-DETR 1.11.0: Nano at 384 px or Small at 512 px.")
    if m.get("precision") != "fp32" or not isinstance(m.get("person_id"), int) or m["person_id"] < 0 or m.get("background_id") is not None:
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
                and report.get("graph_optimization") == optimization and report.get("adapter_sha256") == code_digest(__file__)):
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
        import rfdetr
        from rfdetr.assets.coco_classes import COCO_CLASSES
        if version("rfdetr") != RF_VERSION:
            raise ValueError("Install rfdetr==1.11.0 in the reference environment.")
        self.manifest = load_manifest(manifest_path)
        artifact = self.manifest["checkpoint"]
        weights = Path(manifest_path).parent / artifact["file"]
        # Stock checkpoints use COCO ids; a fine-tuned one names its own classes (Joanne's: person is 0).
        names = self.manifest.get("class_names") or COCO_CLASSES
        if digest(weights) != artifact["sha256"] or names.get(self.manifest["person_id"], names.get(str(self.manifest["person_id"]))) != "person":
            raise ValueError("Checkpoint checksum or person class mapping mismatch.")
        torch.set_num_threads(4)
        extra = {"num_classes": self.manifest["num_classes"]} if "num_classes" in self.manifest else {}
        self.model = getattr(rfdetr, self.manifest["model"])(pretrain_weights=str(weights.resolve()), device=provider, **extra)
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
        import torch
        return PythonDetector(manifest_path, provider or ("mps" if torch.backends.mps.is_available() else "cpu"))
    if backend == "onnx":
        return OnnxDetector(manifest_path, provider or "CPUExecutionProvider")
    raise ValueError("Unknown detector backend.")


class TiledDetector:
    """Full frame plus a grid of overlapping tiles, in one batched call, merged back to full-frame boxes.
    Far-away swimmers are too small at 384 px for the whole frame; each tile is seen at higher zoom.
    Idea from Joanne's model/detect_people_rfdetr_s.py. Python backend only (needs batched predict).
    ponytail: 3x2 grid costs about 4.5x a single full-frame pass on CPU."""

    def __init__(self, base, grid=(3, 2), overlap=0.25):
        if not isinstance(base, PythonDetector):
            raise ValueError("Tiling needs the Python backend (--backend python).")
        self.base, self.grid, self.overlap = base, tuple(grid), overlap
        self.manifest, self.runtime = base.manifest, {**base.runtime, "tiling": {"grid": list(grid), "overlap": overlap}}

    def windows(self, w, h):
        nx, ny = self.grid
        tw, th = int(w / (nx - (nx - 1) * self.overlap)), int(h / (ny - (ny - 1) * self.overlap))
        return [(min(int(ix * tw * (1 - self.overlap)), w - tw), min(int(iy * th * (1 - self.overlap)), h - th), tw, th)
                for iy in range(ny) for ix in range(nx)]

    def predict(self, rgb, threshold=.2):
        from backend.tracking import clean_boxes
        h, w = rgb.shape[:2]
        wins = [(0, 0, w, h)] + self.windows(w, h)
        results = self.base.model.predict([rgb] + [rgb[y:y + th, x:x + tw] for x, y, tw, th in wins[1:]], threshold=threshold)
        full, extra = [], []  # full-frame boxes are kept; tiles only add people the full frame missed
        for i, ((x, y, tw, th), r) in enumerate(zip(wins, results)):
            keep = r.class_id == self.manifest["person_id"]
            for b, s in zip(r.xyxy[keep], r.confidence[keep]):
                box = [(b[0] + x) / w, (b[1] + y) / h, (b[2] + x) / w, (b[3] + y) / h]
                (extra if i else full).append((box, float(s)))

        # Weak boxes (under 0.2) are kept for low-confidence tracking but must never remove a stronger box:
        # with a 0.1 floor they are numerous, and a real 0.7 swimmer containing two of them was dropped as a
        # "group box" while junk full-frame boxes hid good tile detections (median people per frame 29 -> 16).
        floor = max(threshold, .2)
        weak = [(b, s) for b, s in full + extra if s < floor]
        full, extra = [(b, s) for b, s in full if s >= floor], [(b, s) for b, s in extra if s >= floor]

        def covered(box):  # overlaps a full-frame person (e.g. half of a big swimmer cut at a tile seam)
            a = (box[2] - box[0]) * (box[3] - box[1])
            for f, _ in full:
                ix = max(0, min(box[2], f[2]) - max(box[0], f[0])); iy = max(0, min(box[3], f[3]) - max(box[1], f[1]))
                if ix * iy > .3 * a:
                    return True
            return False
        extra = [(b, s) for b, s in extra if not covered(b)]
        pairs = full + extra
        # Same person seen in several overlapping tiles collapses to one box.
        boxes, scores = clean_boxes(np.clip(np.asarray([b for b, _ in pairs]).reshape(-1, 4), 0, 1),
                                    [s for _, s in pairs], nms_iou=.5, inside=.8, part_ratio=1.0)
        strong = [b for b in boxes.tolist()]
        for b, s in sorted(weak, key=lambda x: -x[1]):  # weak boxes only fill gaps
            b = [min(max(v, 0.), 1.) for v in b]
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area > 0 and all(max(0, min(b[2], o[2]) - max(b[0], o[0])) * max(0, min(b[3], o[3]) - max(b[1], o[1])) <= .3 * area
                                for o in strong):
                strong.append(b); scores = np.append(scores, s)
        return [{"bbox_xyxy_normalized": [float(v) for v in b], "confidence": float(s), "class_name": "person"}
                for b, s in zip(strong, scores) if b[2] > b[0] and b[3] > b[1]]
