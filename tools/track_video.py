"""Run RF-DETR Nano + PoolTracker over part of a video and write an annotated MP4 and a summary.

For checking the tracker without the web app (e.g. people diving in the wave-pool reference).

Usage (from the repo root):
  .venv/bin/python -m tools.track_video --start 85 --end 115 --hz 10 --out artifacts/track_85-115.mp4
"""
import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from backend.common import sampled_frames
from backend.detector import create_detector, TiledDetector
from backend.serve import DEFAULT_MODEL, REFERENCE_VIDEO
from backend.tracking import PoolTracker

COLORS = {"safe": (61, 220, 132), "missing": (154, 163, 171), "warning": (255, 176, 0), "alarm": (255, 77, 77)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(REFERENCE_VIDEO))
    ap.add_argument("--start", type=float, default=85)
    ap.add_argument("--end", type=float, default=115)
    ap.add_argument("--hz", type=float, default=10, help="analysis rate; the app uses 5")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", default="artifacts/track_preview.mp4")
    ap.add_argument("--no-tiles", action="store_true", help="full frame only (faster, misses far people)")
    ap.add_argument("--warn", type=float, default=5.0, help="seconds missing before yellow (demo: 3)")
    ap.add_argument("--alarm", type=float, default=12.0, help="seconds missing before red (demo: 8)")
    args = ap.parse_args()

    detector = create_detector(DEFAULT_MODEL, "python")
    if not args.no_tiles:
        detector = TiledDetector(detector)
    tracker = PoolTracker(args.hz, min_conf=args.conf, warn_s=args.warn, alarm_s=args.alarm)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    writer, log = None, []
    font = ImageFont.load_default()
    for _, _, _, t, rgb in sampled_frames(args.video, args.hz, args.start, args.end):
        h, w = rgb.shape[:2]
        if writer is None:
            writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                       "-s", f"{w}x{h}", "-r", str(args.hz), "-i", "-", "-pix_fmt", "yuv420p",
                                       "-vcodec", "libx264", str(out)], stdin=subprocess.PIPE)
        people = tracker.update(t, detector.predict(rgb, 0.2))
        img = Image.fromarray(rgb); d = ImageDraw.Draw(img)
        for p in people:
            b = p["bbox_xyxy_normalized"]; x0, y0, x1, y1 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
            c = COLORS["safe" if p["visible"] else p["level"]]
            if p["visible"]:
                d.rectangle([x0, y0, x1, y1], outline=c, width=3)
            else:  # dashed look: draw corners only
                L = max(6, (x1 - x0) / 4)
                for (ax, ay, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
                    d.line([ax, ay, ax + dx * L, ay], fill=c, width=3); d.line([ax, ay, ax, ay + dy * L], fill=c, width=3)
            text = f"Person {p['person_id']}" + ("" if p["visible"] else f" not seen {p['missing_s']:.1f}s")
            tw = d.textlength(text, font=font)
            d.rectangle([x0, y0 - 14, x0 + tw + 6, y0], fill=(0, 0, 0))
            d.text((x0 + 3, y0 - 13), text, fill=c, font=font)
        demo = "" if (args.warn, args.alarm) == (5.0, 12.0) else f"   DEMO THRESHOLDS: yellow {args.warn:g}s, red {args.alarm:g}s"
        banner = f"t = {int(t // 60)}:{t % 60:04.1f}   visible {sum(p['visible'] for p in people)}{demo}"
        d.rectangle([0, 0, d.textlength(banner, font=font) + 14, 22], fill=(0, 0, 0))
        d.text((6, 5), banner, fill=(255, 255, 255), font=font)
        writer.stdin.write(img.tobytes())
        log.append({"t": round(t, 2), "people": people})
    writer.stdin.close(); writer.wait()

    per = {}
    for fr in log:
        for p in fr["people"]:
            s = per.setdefault(p["person_id"], {"first": fr["t"], "last_visible": None, "visible_frames": 0,
                                                "continued": 0, "max_missing_s": 0.0, "levels": set()})
            if p["visible"]:
                s["visible_frames"] += 1; s["last_visible"] = fr["t"]; s["continued"] += p["continued"]
            else:
                s["max_missing_s"] = max(s["max_missing_s"], p["missing_s"]); s["levels"].add(p["level"])
    summary = {"video": args.video, "start": args.start, "end": args.end, "hz": args.hz, "warn_s": args.warn, "alarm_s": args.alarm, "frames": len(log),
               "person_ids_created": len(per),
               "reconnected_after_loss": sum(s["continued"] for s in per.values()),
               "short_lived_ids_under_1s": sum(s["visible_frames"] < args.hz for s in per.values()),
               "people": {pid: {**s, "levels": sorted(s["levels"])} for pid, s in sorted(per.items())}}
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "frames": log}))
    print(json.dumps({k: v for k, v in summary.items() if k != "people"}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
