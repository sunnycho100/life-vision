"""Fine-tune YOLO11n on sim frames (labels come free from the simulator's segmentation render).

Split by scenario, never by frame: the held-out scenario is the test set.

Usage: python sim/train_sim_detector.py [--test resurface] [--epochs 30]
Output: runs/sim_det/weights/best.pt
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = ROOT / "data" / "sim_scenarios" / "yolo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="resurface", help="scenario held out for validation")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    imgs = sorted((YOLO_DIR / "images").glob("*.jpg"))
    train = [str(p) for p in imgs if not p.name.startswith(args.test + "_")]
    val = [str(p) for p in imgs if p.name.startswith(args.test + "_")]
    assert train and val, "run sim/mujoco/scenarios.py first"
    (YOLO_DIR / "train.txt").write_text("\n".join(train))
    (YOLO_DIR / "val.txt").write_text("\n".join(val))
    (YOLO_DIR / "sim.yaml").write_text(f"path: {YOLO_DIR}\ntrain: train.txt\nval: val.txt\nnames: {{0: person}}\n")
    print(f"train {len(train)} frames, val {len(val)} frames (held-out scenario: {args.test})")

    from ultralytics import YOLO
    YOLO("yolo11n.pt").train(data=str(YOLO_DIR / "sim.yaml"), epochs=args.epochs, imgsz=960, batch=8,
                             device=args.device, project=str(ROOT / "runs"), name="sim_det", exist_ok=True,
                             patience=10, plots=False, verbose=False)


if __name__ == "__main__":
    main()
