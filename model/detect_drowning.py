"""Run the fine-tuned YOLO11n + ByteTrack on a pool video and raise drowning warnings/alarms.

Per tracked person:
  - class each frame: swimming or underwater, with hysteresis (see Person.update)
  - underwater timer: runs while the class is underwater, and keeps running for up to
    --lost-keep seconds if the detector loses someone who was underwater
  - a new track inherits a lost person's timer only if it is also underwater and in almost the
    same spot (the same submerged person, briefly missed). A looser merge let swimmers passing
    by take over a sinking person's identity and timer, which caused false alarms in a busy pool
  - WARNING at --warn seconds under, ALARM at --alarm seconds, or at --still seconds if the
    person has also stopped moving

With --gt (labels.jsonl from sim/isaac/pool_video.py) it also scores detection and alarm timing.

Run from the repo root:
    model\\.venv\\Scripts\\python.exe model\\detect_drowning.py --video sim\\isaac\\_out_test0\\pool.mp4 ^
        --gt sim\\isaac\\_out_test0\\labels.jsonl
"""
import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--weights", default=str(ROOT / "model/runs/pool_yolo11n/weights/best.pt"))
parser.add_argument("--gt", help="labels.jsonl from pool_video.py, for scoring")
parser.add_argument("--out", help="annotated output video (default: <video>_detect.mp4)")
parser.add_argument("--conf", type=float, default=0.3)
parser.add_argument("--imgsz", type=int, default=960)
parser.add_argument("--warn", type=float, default=5.0)
parser.add_argument("--alarm", type=float, default=12.0)
parser.add_argument("--still", type=float, default=8.0, help="alarm sooner if underwater this long and not moving")
parser.add_argument("--lost-keep", type=float, default=3.0, help="seconds a lost underwater person keeps their timer")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 15
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_path = args.out or str(Path(args.video).with_name(Path(args.video).stem + "_detect.mp4"))
writer = imageio_ffmpeg.write_frames(out_path, (W, H), fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=8)
writer.send(None)

tracker_cfg = ROOT / "model/_data/pool_bytetrack.yaml"
tracker_cfg.parent.mkdir(parents=True, exist_ok=True)
tracker_cfg.write_text(  # keep lost tracks ~3 s so short dives keep their ID
    f"tracker_type: bytetrack\ntrack_high_thresh: 0.3\ntrack_low_thresh: 0.1\nnew_track_thresh: 0.4\n"
    f"track_buffer: {int(3 * fps)}\nmatch_thresh: 0.8\nfuse_score: True\n")
model = YOLO(args.weights)
UNDER = next((k for k, v in model.names.items() if v == "underwater"), None)
# Person-only models (stock YOLO, the MuJoCo-trained sim_yolo11n.pt) can't see "underwater".
# For them we fall back to the v1 rule: someone the detector loses is assumed to be underwater.
PERSON_ONLY = UNDER is None
CLASSES = [k for k, v in model.names.items() if v == "person"] if PERSON_ONLY and len(model.names) > 1 else None


class Person:
    def __init__(self, pid, t, box):
        self.pid, self.box, self.last_seen = pid, box, t
        self.hist = deque(maxlen=int(2 * fps))  # (t, class)
        self.centers = deque(maxlen=int(2 * fps) + 1)
        self.under_since = None
        self.status = "ok"

    def update(self, t, box, cls):
        self.box, self.last_seen = box, t
        self.hist.append((t, cls))
        self.centers.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[3] - box[1]))
        if PERSON_ONLY:
            self.under_since = None  # seen again = back up (v1 rule)
            return
        # Hysteresis: start the timer once half of the last second is "underwater" (backdated to
        # the first underwater frame), and only clear it after a second of mostly "swimming".
        # Without this, single-frame class flickers keep resetting the timer.
        recent = [(tt, c) for tt, c in self.hist if t - tt <= 1.0]
        frac = sum(c == UNDER for _, c in recent) / len(recent)
        if self.under_since is None and frac >= 0.5:
            self.under_since = next(tt for tt, c in recent if c == UNDER)
        elif self.under_since is not None and frac <= 0.2 and len(recent) >= 0.8 * fps:
            self.under_since = None

    def still(self):
        if len(self.centers) < self.centers.maxlen:
            return False
        (x0, y0, h), (x1, y1, _) = self.centers[0], self.centers[-1]
        return np.hypot(x1 - x0, y1 - y0) < 0.15 * max(h, 1)

    def seconds_under(self, t):
        return 0.0 if self.under_since is None else t - self.under_since

    def evaluate(self, t):
        s = self.seconds_under(t)
        self.status = "ALARM" if s >= args.alarm or (s >= args.still and self.still()) else (
            "WARNING" if s >= args.warn else "ok")
        return self.status


people, alias, next_pid = {}, {}, [1]


def person_for(track_id, t, box, cls):
    """Map a tracker ID to a person. A new ID joins a lost underwater person only if it is also
    underwater and within about half a body of where they were lost."""
    if track_id in alias:
        return people[alias[track_id]]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best, best_d = None, None
    for p in people.values():
        if (cls == UNDER and p.last_seen < t - 0.2 and p.under_since is not None
                and t - p.last_seen < args.lost_keep):
            d = np.hypot(cx - (p.box[0] + p.box[2]) / 2, cy - (p.box[1] + p.box[3]) / 2)
            if d < 0.6 * max(p.box[3] - p.box[1], p.box[2] - p.box[0]) and (best_d is None or d < best_d):
                best, best_d = p, d
    if best is None:
        best = people[next_pid[0]] = Person(next_pid[0], t, box)
        next_pid[0] += 1
    alias[track_id] = best.pid
    return best


COL = {"ok": (0, 200, 0), "WARNING": (0, 170, 255), "ALARM": (0, 0, 255)}
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
    res = model.track(frame, persist=True, tracker=str(tracker_cfg), conf=args.conf, imgsz=args.imgsz,
                      classes=CLASSES, agnostic_nms=True, verbose=False)[0]
    seen, dets = set(), []
    if res.boxes is not None and res.boxes.id is not None:
        for box, tid, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist(),
                                       res.boxes.cls.int().cpu().tolist(), res.boxes.conf.cpu().tolist()):
            p = person_for(tid, t, box.tolist(), cls)
            p.update(t, box.tolist(), cls)
            seen.add(p.pid)
            dets.append((p, cls, conf))
    for p in people.values():  # lost people keep their underwater timer running, for a while
        if PERSON_ONLY and p.pid not in seen and p.under_since is None and t - p.last_seen >= 0.3:
            p.under_since = p.last_seen  # v1 rule: disappeared = went under
        if not PERSON_ONLY and p.pid not in seen and t - p.last_seen > args.lost_keep:
            p.under_since = None  # gone too long: most likely surfaced under a new ID
        p.evaluate(t)

    for p, cls, conf in dets:
        x0, y0, x1, y1 = map(int, p.box)
        c = COL[p.status]
        cv2.rectangle(frame, (x0, y0), (x1, y1), c, max(2, int(3 * FS)))
        txt = f"#{p.pid} {model.names[cls]}"
        if p.seconds_under(t) > 0:
            txt += f" {p.seconds_under(t):.1f}s"
        if p.status != "ok":
            txt += f"  {p.status}"
        label(frame, txt, (x0, y0 - 4), c)
    for p in people.values():  # lost to the detector while underwater, and already a concern
        if p.pid not in seen and p.under_since is not None and p.status != "ok":
            x0, y0, x1, y1 = map(int, p.box)
            cv2.rectangle(frame, (x0, y0), (x1, y1), COL[p.status], max(1, int(2 * FS)))
            label(frame, f"#{p.pid} not visible, under {p.seconds_under(t):.1f}s  {p.status}", (x0, y1 + 26),
                  COL[p.status], 0.6)
    alarms = [p for p in people.values() if p.status == "ALARM"]
    banner = (f"ALARM: #{alarms[0].pid} underwater {alarms[0].seconds_under(t):.1f}s" if alarms
              else "monitoring")
    warns = [p for p in people.values() if p.status == "WARNING"]
    if not alarms and warns:
        banner = f"WARNING: #{warns[0].pid} underwater {warns[0].seconds_under(t):.1f}s"
    bar = (0, 0, 180) if alarms else ((0, 120, 200) if warns else (40, 40, 40))
    cv2.rectangle(frame, (0, 0), (W, int(64 * FS)), bar, -1)
    cv2.putText(frame, f"YOLO11n + ByteTrack   t = {t:5.1f}s   {banner}", (int(18 * FS), int(44 * FS)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05 * FS, (255, 255, 255), max(2, int(2 * FS)))
    writer.send(np.ascontiguousarray(frame[:, :, ::-1]).tobytes())
    log.append({"t": round(t, 3), "dets": [{"pid": p.pid, "box": [round(v, 1) for v in p.box], "cls": cls,
                                             "conf": round(conf, 3), "status": p.status,
                                             "under": round(p.seconds_under(t), 2)} for p, cls, conf in dets],
                "status": {p.pid: p.status for p in people.values()}})
    f += 1
writer.close()
print(f"wrote {out_path} ({f} frames)")
(Path(out_path).with_suffix(".json")).write_text(json.dumps(log))

if args.gt:
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

    gt = [json.loads(line) for line in open(args.gt)]
    stats = {"swimming": [0, 0, 0], "underwater": [0, 0, 0]}  # found, class right, total
    first = {}  # gt person -> {"WARNING": t, "ALARM": t} from the detector
    truth = {}  # gt person -> {"WARNING": t, "ALARM": t} from the true underwater time
    match_votes = {}
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
                stats[name][1] += (model.names[best["cls"]] == name) if not PERSON_ONLY else 0
                match_votes.setdefault(gid, {}).setdefault(best["pid"], 0)
                match_votes[gid][best["pid"]] += 1
            tr = truth.setdefault(gid, {"scenario": info["scenario"]})
            for level, thr in (("WARNING", args.warn), ("ALARM", args.alarm)):
                if info["seconds_below"] >= thr and level not in tr:
                    tr[level] = g["t"]
    # Each track ID counts only for the person it matched most often, so a swimmer passing
    # over someone on the floor can't pick up (or hand over) their alarm.
    owner_of = {}
    for gid, votes in match_votes.items():
        for pid, n in votes.items():
            if n > owner_of.get(pid, (None, 0))[1]:
                owner_of[pid] = (gid, n)
    for gid in match_votes:
        pids = [pid for pid, (g, _) in owner_of.items() if g == gid]
        for d in log:
            for level in ("WARNING", "ALARM"):
                hit = any(d["status"].get(pid) in ((level, "ALARM") if level == "WARNING" else ("ALARM",))
                          for pid in pids)
                if hit and level not in first.setdefault(gid, {}):
                    first[gid][level] = d["t"]
    print("\nDETECTION (IoU >= 0.4 against the full-body truth box)")
    for name, (found, right, total) in stats.items():
        print(f"  {name:10s}  found {found}/{total} ({found / max(total, 1):.0%})   "
              + (f"class correct when found {right / max(found, 1):.0%}" if not PERSON_ONLY else "(person-only model)"))
    print(f"\nALARMS (warn at {args.warn}s under, alarm at {args.alarm}s, or {args.still}s if still)")
    print(f"  {'person':22s} {'true warn':>10s} {'model warn':>11s} {'true alarm':>11s} {'model alarm':>12s}")
    fmt = lambda v: "-" if v is None else f"{v:.1f}s"  # noqa: E731
    for gid in sorted(truth, key=int):
        tr, fm = truth[gid], first.get(gid, {})
        print(f"  #{gid} {tr['scenario']:19s} {fmt(tr.get('WARNING')):>10s} {fmt(fm.get('WARNING')):>11s} "
              f"{fmt(tr.get('ALARM')):>11s} {fmt(fm.get('ALARM')):>12s}")
