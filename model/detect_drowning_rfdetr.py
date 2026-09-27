"""RF-DETR version of detect_drowning.py: fine-tuned RF-DETR Small (swimming / underwater) +
ByteTrack + the GREEN / ORANGE / RED rules in model/alert_rules.py.

Same logic as detect_drowning.py (YOLO11n), so the two can be compared directly:
  head under    the "underwater" class, majority-voted over the last second with hysteresis
  lost person   keeps their timer for --lost-keep s, and a new underwater track in almost the
                same spot inherits it
Tracking uses supervision's ByteTrack, because RF-DETR has no built-in tracker.

Run from the repo root (after model/train_rfdetr.py):
    model\\.venv\\Scripts\\python.exe model\\detect_drowning_rfdetr.py --video sim\\isaac\\_out_test0\\pool.mp4 ^
        --gt sim\\isaac\\_out_test0\\labels.jsonl
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import supervision as sv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import alert_rules  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)  # sv.ByteTrack deprecation notice
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--weights", default=str(ROOT / "model/runs/pool_rfdetr/checkpoint_best_ema.pth"))
parser.add_argument("--gt", help="labels.jsonl from pool_video.py, for scoring")
parser.add_argument("--out", help="annotated output video (default: <video>_rfdetr_alarms.mp4)")
parser.add_argument("--conf", type=float, default=0.4)
parser.add_argument("--lost-keep", type=float, default=3.0, help="seconds a lost underwater person keeps their timer")
parser.add_argument("--title", default="RF-DETR (fine-tuned) + ByteTrack")
args = parser.parse_args()

from rfdetr.detr import RFDETR  # noqa: E402

model = RFDETR.from_checkpoint(args.weights)
names = model.class_names
names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
UNDER = next(k for k, v in names.items() if v == "underwater")
print("classes:", names, flush=True)

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 15
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_path = args.out or str(Path(args.video).with_name(Path(args.video).stem + "_rfdetr_alarms.mp4"))
writer = imageio_ffmpeg.write_frames(out_path, (W, H), fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=8)
writer.send(None)
tracker = sv.ByteTrack(track_activation_threshold=args.conf, lost_track_buffer=int(3 * fps),
                       minimum_matching_threshold=0.8, frame_rate=int(round(fps)))
rules = alert_rules.StatusTracker(grace_s=0.0, dims=2)


class Person:
    def __init__(self, pid, t, box):
        self.pid, self.box, self.last_seen = pid, box, t
        self.st = {"status": "green", "reason": "ok", "seconds_under": 0.0}
        self.hist, self.under_now = [], False  # (t, class is underwater) over the last second

    def smoothed(self, t, is_under):
        """Majority vote over the last second, with hysteresis. Returns (under, since).
        Same as detect_drowning.py, so brief head dips while struggling don't count as going under."""
        self.hist = [(tt, u) for tt, u in self.hist if t - tt <= 1.0] + [(t, is_under)]
        frac = sum(u for _, u in self.hist) / len(self.hist)
        if not self.under_now and frac >= 0.5:
            self.under_now = True
        elif self.under_now and frac <= 0.2 and len(self.hist) >= 0.8 * fps:
            self.under_now = False
        since = next((tt for tt, u in self.hist if u), t) if self.under_now else None
        return self.under_now, since

    @property
    def under(self):
        return self.st["seconds_under"] > 0


people, alias, next_pid = {}, {}, [1]


def person_for(track_id, t, box, cls):
    """Map a tracker ID to a person. A new ID joins a lost underwater person only if it is also
    underwater and within about half a body of where they were lost."""
    if track_id in alias:
        return people[alias[track_id]]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best, best_d = None, None
    for p in people.values():
        if cls == UNDER and p.last_seen < t - 0.2 and p.under and t - p.last_seen < args.lost_keep:
            d = np.hypot(cx - (p.box[0] + p.box[2]) / 2, cy - (p.box[1] + p.box[3]) / 2)
            if d < 0.6 * max(p.box[3] - p.box[1], p.box[2] - p.box[0]) and (best_d is None or d < best_d):
                best, best_d = p, d
    if best is None:
        best = people[next_pid[0]] = Person(next_pid[0], t, box)
        next_pid[0] += 1
    alias[track_id] = best.pid
    return best


COL = alert_rules.COLORS_BGR
FS = W / 1920  # text scale follows resolution


def label(img, text, org, color, scale=0.7):
    """Text on a dark background so it stays readable over water and deck."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale * FS, max(1, int(2 * FS)))
    x, y = max(0, org[0]), max(th + 6, org[1])
    cv2.rectangle(img, (x - 3, y - th - 6), (x + tw + 3, y + base), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale * FS, color, max(1, int(2 * FS)))


log, f = [], 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = f / fps
    det = model.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), threshold=args.conf)
    det = det.with_nms(threshold=0.5, class_agnostic=True)  # one box per person, whichever class wins
    det = tracker.update_with_detections(det)
    seen, dets = set(), []
    for box, tid, cls, conf in zip(det.xyxy, det.tracker_id, det.class_id, det.confidence):
        cls = int(cls)
        p = person_for(int(tid), t, box.tolist(), cls)
        p.box, p.last_seen = box.tolist(), t
        x0, y0, x1, y1 = p.box
        under, since = p.smoothed(t, cls == UNDER)
        p.st = rules.update(p.pid, t, head_under=under, under_start=since,
                            pos=((x0 + x1) / 2, (y0 + y1) / 2), points=(x0, y0, x1, y1),
                            size=float(np.hypot(x1 - x0, y1 - y0)))
        seen.add(p.pid)
        dets.append((p, cls, float(conf)))
    for p in people.values():  # people the detector can't see this frame
        if p.pid not in seen:
            p.st = rules.update(p.pid, t, head_under=None if t - p.last_seen <= args.lost_keep else False)

    for p, cls, conf in dets:
        x0, y0, x1, y1 = map(int, p.box)
        c = COL[p.st["status"]]
        cv2.rectangle(frame, (x0, y0), (x1, y1), c, max(2, int((4 if p.st["status"] != "green" else 3) * FS)))
        txt = f"#{p.pid} {names[cls]} {conf:.2f}"
        if p.st["status"] != "green":
            txt += f"  {p.st['status'].upper()}: {p.st['reason']}"
        elif p.under:
            txt += f" {p.st['seconds_under']:.1f}s"
        label(frame, txt, (x0, y0 - 4), c)
    for p in people.values():  # lost to the detector while underwater, and already a concern
        if p.pid not in seen and p.under and p.st["status"] != "green":
            x0, y0, x1, y1 = map(int, p.box)
            cv2.rectangle(frame, (x0, y0), (x1, y1), COL[p.st["status"]], max(1, int(2 * FS)))
            label(frame, f"#{p.pid} not visible, {p.st['status'].upper()}: {p.st['reason']}", (x0, y1 + 26),
                  COL[p.st["status"]], 0.6)
    reds = [p for p in people.values() if p.st["status"] == "red"]
    oranges = [p for p in people.values() if p.st["status"] == "orange"]
    if reds:
        banner, bar = f"RED ALARM: #{reds[0].pid} {reds[0].st['reason']}", (0, 0, 190)
    elif oranges:
        banner, bar = f"ORANGE WATCH: #{oranges[0].pid} {oranges[0].st['reason']}", (0, 120, 210)
    else:
        banner, bar = "monitoring", (40, 40, 40)
    cv2.rectangle(frame, (0, 0), (W, int(64 * FS)), bar, -1)
    cv2.putText(frame, f"{args.title}   t = {t:5.1f}s   {banner}", (int(18 * FS), int(44 * FS)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05 * FS, (255, 255, 255), max(2, int(2 * FS)))
    writer.send(np.ascontiguousarray(frame[:, :, ::-1]).tobytes())
    log.append({"t": round(t, 3), "dets": [{"pid": p.pid, "box": [round(v, 1) for v in p.box], "cls": cls,
                                             "conf": round(conf, 3), "status": p.st["status"],
                                             "under": p.st["seconds_under"]} for p, cls, conf in dets],
                "status": {p.pid: p.st["status"] for p in people.values()}})
    f += 1
    if f % int(5 * fps) == 0:
        print(f"t={t:.0f}s", flush=True)
writer.close()
print(f"wrote {out_path} ({f} frames)")
(Path(out_path).with_suffix(".json")).write_text(json.dumps(log))

if args.gt:  # same scoring as detect_drowning.py
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

    gt = [json.loads(line) for line in open(args.gt)]
    has_status = any("status" in v for g in gt for v in g["people"].values())
    stats = {"swimming": [0, 0, 0], "underwater": [0, 0, 0]}  # found, class right, total
    first, truth, match_votes = {}, {}, {}
    for g, d in zip(gt, log):
        for gid, info in g["people"].items():
            box = info.get("body_xyxy")
            if box is None:
                continue
            name = "underwater" if info["head_state"] == "below" else "swimming"
            stats[name][2] += 1
            best = max(d["dets"], key=lambda x: iou(box, x["box"]), default=None)
            if best is not None and iou(box, best["box"]) >= 0.4:
                stats[name][0] += 1
                stats[name][1] += names[best["cls"]] == name
                match_votes.setdefault(gid, {}).setdefault(best["pid"], 0)
                match_votes[gid][best["pid"]] += 1
            tr = truth.setdefault(gid, {"scenario": info["scenario"]})
            if has_status:
                status = info["status"]
            else:
                s = info["seconds_below"]
                status = "red" if s >= alert_rules.RED_UNDER_S else ("orange" if s >= alert_rules.ORANGE_UNDER_S else "green")
            for level in ("orange", "red"):
                if level not in tr and (status == level or (level == "orange" and status == "red")):
                    tr[level] = g["t"]
    owner_of = {}  # each track ID counts only for the person it matched most often
    for gid, votes in match_votes.items():
        for pid, n in votes.items():
            if n > owner_of.get(pid, (None, 0))[1]:
                owner_of[pid] = (gid, n)
    for gid in match_votes:
        pids = [pid for pid, (g, _) in owner_of.items() if g == gid]
        for d in log:
            for level in ("orange", "red"):
                hit = any(d["status"].get(pid) in (("orange", "red") if level == "orange" else ("red",)) for pid in pids)
                if hit and level not in first.setdefault(gid, {}):
                    first[gid][level] = d["t"]
    print("\nDETECTION (IoU >= 0.4 against the full-body truth box)")
    for name, (found, right, total) in stats.items():
        print(f"  {name:10s}  found {found}/{total} ({found / max(total, 1):.0%})   "
              f"class correct when found {right / max(found, 1):.0%}")
    print("\nSTATUS (first time each person turned ORANGE and RED; rules in model/alert_rules.py)")
    print(f"  {'person':22s} {'true orange':>12s} {'model orange':>13s} {'true red':>10s} {'model red':>10s}")
    fmt = lambda v: "-" if v is None else f"{v:.1f}s"  # noqa: E731
    for gid in sorted(truth, key=int):
        tr, fm = truth[gid], first.get(gid, {})
        print(f"  #{gid} {tr['scenario']:19s} {fmt(tr.get('orange')):>12s} {fmt(fm.get('orange')):>13s} "
              f"{fmt(tr.get('red')):>10s} {fmt(fm.get('red')):>10s}")
