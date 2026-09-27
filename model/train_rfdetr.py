"""Fine-tune RF-DETR Small on the Isaac Sim pool frames (2 classes: swimming, underwater).

Same data and split as train_yolo.py: 261 frames from sim/isaac/make_training_set.ps1, a seeded
15% validation split, and the seed-0 demo clip held out as the test set.

Run from the repo root:
    model\\.venv\\Scripts\\python.exe model\\train_rfdetr.py
Checkpoints land in model/runs/pool_rfdetr/ (git ignores model/runs and model/_data).
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=[str(ROOT / "sim/isaac/_out_train")],
                    help="folders containing */yolo/images and */yolo/labels")
    ap.add_argument("--test", default=str(ROOT / "sim/isaac/_out_test0/yolo"), help="held-out seed-0 frames")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    ds = ROOT / "model/_data/pool_rfdetr"
    shutil.rmtree(ds, ignore_errors=True)
    images = sorted(p for d in args.train for p in Path(d).glob("*/yolo/images/*.jpg"))
    random.Random(0).shuffle(images)
    n_val = max(1, int(len(images) * args.val_frac))
    splits = {"valid": images[:n_val], "train": images[n_val:]}
    test = Path(args.test)
    if test.exists():
        splits["test"] = sorted((test / "images").glob("*.jpg"))
    for split, subset in splits.items():
        for sub in ("images", "labels"):
            (ds / split / sub).mkdir(parents=True, exist_ok=True)
        for img in subset:
            shutil.copy(img, ds / split / "images" / img.name)
            shutil.copy(img.parents[1] / "labels" / (img.stem + ".txt"), ds / split / "labels" / (img.stem + ".txt"))
    (ds / "data.yaml").write_text("train: train/images\nval: valid/images\n"
                                  + ("test: test/images\n" if "test" in splits else "")
                                  + "nc: 2\nnames: ['swimming', 'underwater']\n")
    print({k: len(v) for k, v in splits.items()}, flush=True)

    from rfdetr import RFDETRSmall

    RFDETRSmall().train(dataset_dir=str(ds), epochs=args.epochs, batch_size=args.batch,
                        grad_accum_steps=args.grad_accum, output_dir=str(ROOT / "model/runs/pool_rfdetr"),
                        num_workers=0)  # Windows data-loader workers crashed during YOLO training


if __name__ == "__main__":  # Windows re-imports this file in worker processes
    main()
