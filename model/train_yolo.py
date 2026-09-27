"""Fine-tune YOLO11n on the Isaac Sim pool frames (2 classes: swimming, underwater).

Training data comes from sim/isaac/make_training_set.ps1 (random stills + scripted clips with
seeds 1-3). The seed-0 demo clip is held out for testing, so the model never sees it.

Run from the repo root:
    model\\.venv\\Scripts\\python.exe model\\train_yolo.py
Weights land in model/runs/pool_yolo11n/weights/best.pt (git ignores model/runs and model/_data).
"""
import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", default=[str(ROOT / "sim/isaac/_out_train")],
                        help="folders containing */yolo/images and */yolo/labels")
    parser.add_argument("--test", default=str(ROOT / "sim/isaac/_out_test0/yolo"), help="held-out seed-0 frames")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960, help="bigger than 640 helps with small heads far away")
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    data = ROOT / "model/_data/pool_sim"
    if data.exists():
        shutil.rmtree(data)
    images = sorted(p for d in args.train for p in Path(d).glob("*/yolo/images/*.jpg"))
    random.Random(0).shuffle(images)
    n_val = max(1, int(len(images) * args.val_frac))
    for split, subset in (("val", images[:n_val]), ("train", images[n_val:])):
        for sub in ("images", "labels"):
            (data / split / sub).mkdir(parents=True, exist_ok=True)
        for img in subset:
            shutil.copy(img, data / split / "images" / img.name)
            shutil.copy(img.parents[1] / "labels" / (img.stem + ".txt"), data / split / "labels" / (img.stem + ".txt"))
    print(f"{len(images) - n_val} train / {n_val} val images")

    test = Path(args.test)
    yaml = data / "pool_sim.yaml"
    yaml.write_text(
        f"path: {data.as_posix()}\ntrain: train/images\nval: val/images\n"
        + (f"test: {(test / 'images').as_posix()}\n" if test.exists() else "")
        + "names:\n  0: swimming\n  1: underwater\n")

    model = YOLO("yolo11n.pt")  # COCO-pretrained; its "person" features transfer well
    model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=8, patience=25,
                project=str(ROOT / "model/runs"), name="pool_yolo11n", exist_ok=True,
                fliplr=0.5, degrees=5, mosaic=1.0, close_mosaic=10, workers=0, plots=True)  # workers=0: Windows loader workers crashed mid-run
    if test.exists():
        metrics = model.val(data=str(yaml), split="test", imgsz=args.imgsz, project=str(ROOT / "model/runs"),
                            name="pool_yolo11n_test", exist_ok=True)
        print("TEST mAP50:", round(metrics.box.map50, 3), "mAP50-95:", round(metrics.box.map, 3),
              "per class mAP50-95:", dict(zip(model.names.values(), [round(x, 3) for x in metrics.box.maps])))


if __name__ == "__main__":  # Windows data-loader workers re-import this file
    main()
