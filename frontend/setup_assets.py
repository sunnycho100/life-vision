"""Fetch optional 3D assets and the pinned YOLO evaluation baseline. No pose assets."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json

ROOT = Path(__file__).resolve().parent / "vendor"
ASSETS = {
    "yolov8n.onnx": "https://huggingface.co/webml/yolov8n/resolve/85bc8d7ab30d4065a41909d756c42819b67b4388/onnx/yolov8n.onnx",
    "three.module.js": "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js",
    "three.core.js": "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.core.js",
    "THREE-LICENSE.txt": "https://cdn.jsdelivr.net/npm/three@0.180.0/LICENSE",
}

if __name__ == "__main__":
    manifest = {}
    for name, url in ASSETS.items():
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"Downloading {name}", flush=True)
            with urlopen(url, timeout=90) as response:
                data = response.read()
            temp = target.with_suffix(target.suffix + ".part")
            temp.write_bytes(data)
            temp.replace(target)
        manifest[name] = {"url": url, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        if name == "yolov8n.onnx" and manifest[name]["sha256"] != "190ba5f1e61411a001683e349d6b2cdb0804c0dc67a5e34cd8ff6fd00ee54b4d":
            raise RuntimeError("YOLO detector checksum mismatch; remove the downloaded file and retry")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Browser assets ready.")
