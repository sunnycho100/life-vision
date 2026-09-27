"""Run the fine-tuned YOLO11n + ByteTrack on a pool video and color each person GREEN / ORANGE / RED.

The colors come from model/alert_rules.py, the same rules the sim ground truth uses:
  RED     head under 10 s, or under 5 s and not moving
  ORANGE  head under 5 s, or instinctive-drowning-response signs at the surface for 3 s
  GREEN   otherwise
Inputs are approximated from the video:
  head under    the YOLO "underwater" class, smoothed: "under" once half of the last second is
                underwater (timer backdated to the first such frame), "up" again only after the
                last second is at least 80% "swimming". Brief head dips while struggling at the
                surface don't count as going under.
  not moving    the box corners moved less than 15% of the box diagonal (RMS) over 2 s
  lost person   a person the detector loses while underwater keeps their timer for --lost-keep s
  posture and "mouth at the waterline" are not observable with the 2-class model, so the
                surface-distress rule (ORANGE O2) only shows in the sim ground truth for now

A new track inherits a lost person's timer only if it is also underwater and in almost the
same spot (the same submerged person, briefly missed). A looser merge let swimmers passing by
take over a sinking person's identity and timer, which caused false alarms in a busy pool.
The reverse also happens: a flickering second box on one underwater swimmer becomes a "lost"
record with its own timer. A lost underwater record that overlaps a visible underwater person
is treated as that person's duplicate and dropped (the visible one carries the timer). A lost
sinker overlapped by a swimmer passing above keeps their timer.

With --gt (labels.jsonl from sim/isaac/pool_video.py) it also scores detection and the times
each person first turned ORANGE and RED, against the ground truth.

Run from the repo root:
    model\\.venv\\Scripts\\python.exe model\\detect_drowning.py --video sim\\isaac\\_out_test0\\pool.mp4 ^
        --gt sim\\isaac\\_out_test0\\labels.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import alert_rules  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--weights", default=str(ROOT / "model/runs/pool_yolo11n/weights/best.pt"))
parser.add_argument("--gt", help="labels.jsonl from pool_video.py, for scoring")
parser.add_argument("--out", help="annotated output video (default: <video>_detect.mp4)")
parser.add_argument("--conf", type=float, default=0.3)
parser.add_argument("--imgsz", type=int, default=960)
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
rules = alert_rules.StatusTracker(grace_s=0.0, dims=2)  # class flicker is smoothed in Person.smoothed


class Person:
    def __init__(self, pid, t, box):
        self.pid, self.box, self.last_seen = pid, box, t
        self.st = {"status": "green", "reason": "ok", "seconds_under": 0.0}
        self.hist, self.under_now = [], False  # (t, class is underwater) over the last second

    def smoothed(self, t, is_under):
        """Majority vote over the last second, with hysteresis. Returns (under, since)."""
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


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


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
    res = model.track(frame, persist=True, tracker=str(tracker_cfg), conf=args.conf, imgsz=args.imgsz,
                      classes=CLASSES, agnostic_nms=True, verbose=False)[0]
    seen, dets = set(), []
    if res.boxes is not None and res.boxes.id is not None:
        for box, tid, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist(),
                                       res.boxes.cls.int().cpu().tolist(), res.boxes.conf.cpu().tolist()):
            p = person_for(tid, t, box.tolist(), cls)
            p.box, p.last_seen = box.tolist(), t
            x0, y0, x1, y1 = p.box
            under, since = p.smoothed(t, cls == UNDER) if not PERSON_ONLY else (False, None)
            p.st = rules.update(p.pid, t, head_under=under, under_start=since,
                                pos=((x0 + x1) / 2, (y0 + y1) / 2), points=(x0, y0, x1, y1),
                                size=float(np.hypot(x1 - x0, y1 - y0)))
            seen.add(p.pid)
            dets.append((p, cls, conf))
    visible_under = [q.box for q, c, _ in dets if c == UNDER]
    for p in people.values():  # people the detector can't see this frame
        if p.pid in seen:
            continue
        if PERSON_ONLY:  # v1 rule: disappeared = went under
            under = True if t - p.last_seen >= 0.3 else None
        elif p.under and any(iou(p.box, b) > 0.3 for b in visible_under):
            under = False  # a duplicate of an underwater person we can still see
        else:  # keep a lost underwater person's timer for a while, then assume they surfaced elsewhere
            under = None if t - p.last_seen <= args.lost_keep else False
        p.st = rules.update(p.pid, t, head_under=under)

    for p, cls, conf in dets:
        x0, y0, x1, y1 = map(int, p.box)
        c = COL[p.st["status"]]
        cv2.rectangle(frame, (x0, y0), (x1, y1), c, max(2, int((4 if p.st["status"] != "green" else 3) * FS)))
        txt = f"#{p.pid} {model.names[cls]}"
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
    cv2.putText(frame, f"YOLO11n + ByteTrack   t = {t:5.1f}s   {banner}", (int(18 * FS), int(44 * FS)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05 * FS, (255, 255, 255), max(2, int(2 * FS)))
    writer.send(np.ascontiguousarray(frame[:, :, ::-1]).tobytes())
    log.append({"t": round(t, 3), "dets": [{"pid": p.pid, "box": [round(v, 1) for v in p.box], "cls": cls,
                                             "conf": round(conf, 3), "status": p.st["status"],
                                             "under": p.st["seconds_under"]} for p, cls, conf in dets],
                "status": {p.pid: p.st["status"] for p in people.values()}})
    f += 1
writer.close()
print(f"wrote {out_path} ({f} frames)")
(Path(out_path).with_suffix(".json")).write_text(json.dumps(log))

if args.gt:
    gt = [json.loads(line) for line in open(args.gt)]
    has_status = any("status" in v for g in gt for v in g["people"].values())
    stats = {"swimming": [0, 0, 0], "underwater": [0, 0, 0]}  # found, class right, total
    first = {}  # gt person -> {"orange": t, "red": t} from the detector
    truth = {}  # gt person -> {"orange": t, "red": t} from the ground truth
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
            if has_status:  # ground truth colored by the same rules, from exact sim measurements
                status = info["status"]
            else:  # older labels: underwater time only
                s = info["seconds_below"]
                status = "red" if s >= alert_rules.RED_UNDER_S else ("orange" if s >= alert_rules.ORANGE_UNDER_S else "green")
            for level in ("orange", "red"):
                if level not in tr and (status == level or (level == "orange" and status == "red")):
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
            for level in ("orange", "red"):
                hit = any(d["status"].get(pid) in (("orange", "red") if level == "orange" else ("red",)) for pid in pids)
                if hit and level not in first.setdefault(gid, {}):
                    first[gid][level] = d["t"]
    print("\nDETECTION (IoU >= 0.4 against the full-body truth box)")
    for name, (found, right, total) in stats.items():
        print(f"  {name:10s}  found {found}/{total} ({found / max(total, 1):.0%})   "
              + (f"class correct when found {right / max(found, 1):.0%}" if not PERSON_ONLY else "(person-only model)"))
    print("\nSTATUS (first time each person turned ORANGE and RED; rules in model/alert_rules.py)"
          + ("" if has_status else "  [old labels: truth from underwater time only]"))
    print(f"  {'person':22s} {'true orange':>12s} {'model orange':>13s} {'true red':>10s} {'model red':>10s}")
    fmt = lambda v: "-" if v is None else f"{v:.1f}s"  # noqa: E731
    for gid in sorted(truth, key=int):
        tr, fm = truth[gid], first.get(gid, {})
        print(f"  #{gid} {tr['scenario']:19s} {fmt(tr.get('orange')):>12s} {fmt(fm.get('orange')):>13s} "
              f"{fmt(tr.get('red')):>10s} {fmt(fm.get('red')):>10s}")
