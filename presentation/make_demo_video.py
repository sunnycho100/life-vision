"""Cut the hackathon demo video: title cards + Isaac Sim fly-through + CCTV clip + RF-DETR alarms.

Each segment is a card (text on a dark background) or a clip with a caption in the lower left.
Clips are resized to the output size and resampled to the output frame rate, with short
crossfades between segments.

Run from the repo root with the model venv (it has OpenCV and imageio-ffmpeg):
    model\\.venv\\Scripts\\python.exe presentation\\make_demo_video.py --plan presentation\\demo_plan.json
The plan is a JSON list of segments:
    {"card": ["Big title", "smaller line", ...], "seconds": 3}
    {"clip": "path.mp4", "caption": "text", "start": 0, "seconds": 10}   (start/seconds optional)
"""
import argparse
import json

import cv2
import imageio_ffmpeg
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1088)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--fade", type=float, default=0.5, help="crossfade seconds between segments")
args = ap.parse_args()
W, H, FPS = args.width, args.height, args.fps
BG, ACCENT, TEXT, DIM = (40, 24, 12), (255, 170, 40), (255, 255, 255), (200, 200, 200)  # BGR


def card(lines):
    """Title card: first line big, the rest smaller, centered, with an accent bar."""
    img = np.full((H, W, 3), BG, np.uint8)
    sizes = [2.3] + [1.15] * (len(lines) - 1)
    heights = [cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, z, 3)[0][1] for s, z in zip(lines, sizes)]
    gap = 34
    y = (H - (sum(heights) + gap * (len(lines) - 1))) // 2
    for i, (s, z, h) in enumerate(zip(lines, sizes, heights)):
        w = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, z, 3 if i == 0 else 2)[0][0]
        y += h
        cv2.putText(img, s, ((W - w) // 2, y), cv2.FONT_HERSHEY_DUPLEX, z, TEXT if i == 0 else DIM,
                    3 if i == 0 else 2, cv2.LINE_AA)
        if i == 0:
            cv2.rectangle(img, ((W - w) // 2, y + 18), ((W - w) // 2 + min(w, 260), y + 26), ACCENT, -1)
            y += 30
        y += gap
    return img


def caption(img, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    x, y = 28, H - 34
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 14, y - th - 18), (x + tw + 14, y + 14), (20, 20, 20), -1)
    img[:] = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
    cv2.rectangle(img, (x - 14, y - th - 18), (x - 8, y + 14), ACCENT, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, TEXT, 2, cv2.LINE_AA)


def clip_frames(seg):
    cap = cv2.VideoCapture(seg["clip"])
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = seg.get("start", 0.0)
    dur = seg.get("seconds", n_src / src_fps - start)
    frames = []
    ok, fr = cap.read()
    src_i = 0
    for k in range(int(round(dur * FPS))):  # nearest source frame for each output frame
        want = int((start + k / FPS) * src_fps)
        while ok and src_i < want:
            ok, fr = cap.read()
            src_i += 1
        if not ok:
            break
        img = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        if seg.get("caption"):
            caption(img, seg["caption"])
        frames.append(img)
    return frames


plan = json.load(open(args.plan))
writer = imageio_ffmpeg.write_frames(args.out, (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", quality=9)
writer.send(None)
n_fade = int(args.fade * FPS)
tail = []  # last frames of the previous segment, for the crossfade
total = 0
for seg in plan:
    frames = [card(seg["card"])] * int(seg.get("seconds", 3) * FPS) if "card" in seg else clip_frames(seg)
    k = min(n_fade, len(tail), len(frames))
    for j in range(k):
        a = (j + 1) / (k + 1)
        frames[j] = cv2.addWeighted(tail[len(tail) - k + j], 1 - a, frames[j], a, 0)
    for fr in frames[:len(frames) - n_fade] if len(frames) > n_fade else []:
        writer.send(np.ascontiguousarray(fr[:, :, ::-1]).tobytes())
        total += 1
    tail = frames[-n_fade:] if len(frames) > n_fade else frames
    print(f"{('card: ' + seg['card'][0]) if 'card' in seg else seg['clip']}: {len(frames) / FPS:.1f}s", flush=True)
for fr in tail:
    writer.send(np.ascontiguousarray(fr[:, :, ::-1]).tobytes())
    total += 1
writer.close()
print(f"wrote {args.out}: {total / FPS:.1f}s")
