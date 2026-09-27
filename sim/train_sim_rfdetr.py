"""Fine-tune RF-DETR Nano on the MuJoCo sim frames (same frames and split as train_sim_detector.py).

Split by scenario, never by frame: the held-out scenario is the validation set.

Usage: python sim/train_sim_rfdetr.py [--test resurface] [--epochs 20]
Output: runs/sim_rfdetr/checkpoint_best_total.pth (or the best checkpoint rfdetr writes)
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = ROOT / "data" / "sim_scenarios" / "yolo"
DS = ROOT / "data" / "sim_scenarios" / "rfdetr_ds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="resurface")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    shutil.rmtree(DS, ignore_errors=True)
    n = {"train": 0, "valid": 0}
    for img in sorted((YOLO_DIR / "images").glob("*.jpg")):
        split = "valid" if img.name.startswith(args.test + "_") else "train"
        for sub, src in (("images", img), ("labels", YOLO_DIR / "labels" / f"{img.stem}.txt")):
            (DS / split / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy(src, DS / split / sub / src.name)
        n[split] += 1
    assert n["train"] and n["valid"], "run sim/mujoco/scenarios.py first"
    (DS / "data.yaml").write_text("train: train/images\nval: valid/images\nnc: 1\nnames: ['person']\n")
    print(f"train {n['train']} frames, valid {n['valid']} frames (held-out scenario: {args.test})")

    from rfdetr import RFDETRNano
    RFDETRNano().train(dataset_dir=str(DS), epochs=args.epochs, batch_size=args.batch,
                       output_dir=str(ROOT / "runs" / "sim_rfdetr"))


if __name__ == "__main__":
    main()
