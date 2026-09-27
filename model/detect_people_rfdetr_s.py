"""Draw person boxes on one image with RF-DETR plus overlapping-tile inference for small people."""
import argparse

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image


def detect(model, img, thr):
    d = model.predict(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), threshold=thr)
    k = d.class_id == 1  # COCO person
    return d.xyxy[k], d.confidence[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--grid", type=int, nargs=2, default=[3, 2], help="tiles across, down")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--upscale", type=float, default=2.0)
    args = ap.parse_args()

    from rfdetr import RFDETRSmall
    model = RFDETRSmall()
    img = cv2.imread(args.image)
    H, W = img.shape[:2]

    boxes, scores = [], []
    b, s = detect(model, img, args.thr)
    boxes.append(b); scores.append(s)

    nx, ny = args.grid
    tw, th = int(W / (nx - (nx - 1) * args.overlap)), int(H / (ny - (ny - 1) * args.overlap))
    for iy in range(ny):
        for ix in range(nx):
            x0 = min(int(ix * tw * (1 - args.overlap)), W - tw)
            y0 = min(int(iy * th * (1 - args.overlap)), H - th)
            f = args.upscale
            tile = cv2.resize(img[y0:y0 + th, x0:x0 + tw], None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            b, s = detect(model, tile, args.thr)
            if len(b):
                boxes.append(b / f + [x0, y0, x0, y0]); scores.append(s)

    B = torch.tensor(np.concatenate(boxes), dtype=torch.float32)
    S = torch.tensor(np.concatenate(scores), dtype=torch.float32)
    keep = torchvision.ops.nms(B, S, 0.5).numpy()
    B, S = B[keep].numpy(), S[keep].numpy()
    # drop boxes mostly contained in a higher-scoring box (tile-edge fragments)
    order = np.argsort(-S); final = []
    for i in order:
        bi = B[i]; ai = (bi[2] - bi[0]) * (bi[3] - bi[1])
        dup = False
        for j in final:
            bj = B[j]; aj = (bj[2] - bj[0]) * (bj[3] - bj[1])
            iw = max(0, min(bi[2], bj[2]) - max(bi[0], bj[0])); ih = max(0, min(bi[3], bj[3]) - max(bi[1], bj[1]))
            inter = iw * ih
            if inter / max(min(ai, aj), 1e-6) > 0.8:
                dup = True; break
            # small tile-edge fragment sitting largely inside a much bigger box
            if min(ai, aj) < 0.35 * max(ai, aj) and inter / max(min(ai, aj), 1e-6) > 0.4:
                dup = True; break
        if not dup:
            final.append(i)

    out = img.copy()
    for i in final:
        x1, y1, x2, y2 = map(int, B[i])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, f"people: {len(final)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imwrite(args.out, out)
    print(f"people={len(final)}")


if __name__ == "__main__":
    main()
