"""Person tracking on top of per-frame RF-DETR boxes: stable "Person N" IDs and a missing timer.

Pipeline per analyzed frame (all coordinates normalized 0..1):
  1. clean_boxes: NMS, drop one box around two people, drop small part boxes (head-only, arm-only)
  2. ByteTrack (supervision) gives short-term track IDs
  3. PoolTracker maps track IDs to people. A new track that appears near where a person was lost,
     within a few seconds, continues that person (same ID, same box history) instead of starting a
     new one. Taken from Sunny's MuJoCo edge logic and Rohan's detect_drowning.py.
  4. Missing timer (Rohan's person-only v1 rule): a person the detector loses is assumed to be under
     the water. Level: safe (visible), then warning, then alarm as the missing time grows. A person
     last seen touching the frame edge is treated as having left the view, not as missing.

Levels are for demo and review only. They are not a validated drowning signal.
"""
import numpy as np

DEFAULTS = {
    "min_conf": 0.3,        # boxes below this are ignored by the tracker
    "nms_iou": 0.6,
    "inside": 0.7,          # share of a box that must sit inside another box to count as contained
    "part_ratio": 0.4,      # a contained box smaller than this share of the bigger one is a part box
    "start_frames": 2,      # consecutive frames before a track gets an ID
    "lost_buffer_s": 5.0,   # ByteTrack keeps a lost track this long
    "stitch_s": 5.0,        # a new track can continue a person lost within this many seconds
    "stitch_scale": 3.0,    # ... if it appears within this many box sizes of where they were lost
    "stitch_min": 0.06,     # ... or within this share of the frame, so tiny far-away people get a usable radius.
                            # Wave pool clip: returning swimmers reappear 1.6-3 box sizes away, 0.5-4 s later.
    "resurface_speed": 0.5, # after stitch_s, a held person can still be reclaimed: the search radius grows by
                            # this many box sizes per second lost (swimmers drift), capped at resurface_max
    "resurface_max": 4.0,
    "back_show_s": 2.0,     # after a reliable person comes back from a gap of hold_s or more, flag them this long
    "mass_loss": 0.4,       # fewer boxes than this share of last frame's visible people (3+) = glitch frame, skipped
    "hold_s": 1.0,          # draw a lost person's last box this long as "missing" before any warning
    "warn_s": 5.0,          # missing this long: warning (yellow)
    "alarm_s": 12.0,        # missing this long: alarm (red)
    "forget_s": 20.0,       # stop showing a missing person after this long
    "edge": 0.02,           # last box within this distance of the frame edge = left the view
    "reliable_s": 1.5,      # only people tracked at least this long can start a missing timer ...
    "reliable_frac": 0.8,   # ... and seen in at least this share of frames over the last 2 s.
                            # Flickering far-away people are dropped when lost instead of raising warnings.
}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-12)


def clean_boxes(boxes, scores, nms_iou=0.6, inside=0.7, part_ratio=0.4):
    """1. NMS (RF-DETR sometimes returns one person twice). 2. Drop a box containing 2+ other boxes
    (one box around two overlapping people). 3. Drop a small box mostly inside a much bigger one."""
    boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
    if not len(boxes):
        return boxes, scores
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def frac_inside(i, j):
        b, o = boxes[i], boxes[j]
        ix = max(0, min(b[2], o[2]) - max(b[0], o[0])); iy = max(0, min(b[3], o[3]) - max(b[1], o[1]))
        return ix * iy / (area[i] + 1e-12)

    keep = []
    for i in np.argsort(-scores):
        if all(iou(boxes[i], boxes[k]) < nms_iou for k in keep):
            keep.append(int(i))
    keep = [j for j in keep if sum(frac_inside(i, j) >= inside for i in keep if i != j and area[i] < area[j]) < 2]
    keep = [i for i in keep if not any(frac_inside(i, j) >= inside and area[i] < part_ratio * area[j]
                                       for j in keep if j != i)]
    return boxes[keep], scores[keep]


def is_scene_cut(prev_rgb, rgb, threshold=15.0):
    """Mean absolute difference of small grayscale thumbnails; a hard cut changes most pixels at once.
    Wave-pool reference: the cut scores 24, ordinary frames at most 8.
    ponytail: fixed threshold on a 64x36 thumbnail, tune if a real pool camera trips it (e.g. auto exposure)."""
    if prev_rgb is None:
        return False
    def thumb(img):
        h, w = img.shape[:2]
        return img[:: max(1, h // 36), :: max(1, w // 64)].mean(axis=2)
    a, b = thumb(prev_rgb), thumb(rgb)
    n = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return float(np.abs(a[:n[0], :n[1]] - b[:n[0], :n[1]]).mean()) > threshold


class PoolTracker:
    def __init__(self, sample_hz, **overrides):
        import supervision as sv
        self.cfg = {**DEFAULTS, **overrides}
        self.sv = sv
        self.hz = sample_hz
        # supervision starts new tracks at track_activation_threshold + 0.1, so offset it to start at min_conf
        self.tracker = sv.ByteTrack(frame_rate=sample_hz, track_activation_threshold=self.cfg["min_conf"] - 0.1,
                                    lost_track_buffer=max(1, int(self.cfg["lost_buffer_s"] * 30)),  # supervision scales by frame_rate / 30
                                    minimum_consecutive_frames=self.cfg["start_frames"])
        self.person_of = {}   # tracker id -> person id
        self.people = {}      # person id -> {"box", "last_seen", "first_seen", "conf", "at_edge"}
        self.next_id = 1
        self.events = []      # went out of sight and came back: see update()
        self._last_visible, self._last_out = 0, []

    def scene_cut(self):
        """Camera cut or new shot: nobody from the old view is "missing". Forget them; IDs keep counting up."""
        self.person_of.clear(); self.people.clear()
        self.tracker.reset()
        self._last_visible, self._last_out = 0, []

    def _at_edge(self, b):
        e = self.cfg["edge"]
        return b[0] <= e or b[1] <= e or b[2] >= 1 - e or b[3] >= 1 - e

    def _stitch(self, t, box, visible):
        """Nearest person lost recently near this new box, or None."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best, best_d = None, None
        for pid, p in self.people.items():
            lost = t - p["last_seen"]
            if pid in visible or lost <= 0 or lost > self.cfg["forget_s"]:
                continue
            lb = p["box"]
            size = max(lb[2] - lb[0], lb[3] - lb[1])
            d = np.hypot(cx - (lb[0] + lb[2]) / 2, cy - (lb[1] + lb[3]) / 2)
            # Short gaps use the tight gate. Longer gaps (while the missing box is still shown) widen with time,
            # so a swimmer who resurfaces a little further away clears their own warning instead of leaving it
            # red until forget_s. Only brand-new tracks get here, so an already-tracked passer-by can't take it.
            reach = self.cfg["stitch_scale"] if lost <= self.cfg["stitch_s"] else min(
                self.cfg["stitch_scale"] + self.cfg["resurface_speed"] * (lost - self.cfg["stitch_s"]), self.cfg["resurface_max"])
            if d <= max(reach * size, self.cfg["stitch_min"]) and (best_d is None or d < best_d):
                best, best_d = pid, d
        return best

    def update(self, t, detections):
        """detections: [{"bbox_xyxy_normalized", "confidence"}]. Returns the people to draw this frame."""
        dets = [d for d in detections if (d.get("confidence") or 0) >= self.cfg["min_conf"]]
        boxes, scores = clean_boxes([d["bbox_xyxy_normalized"] for d in dets], [d["confidence"] for d in dets],
                                    self.cfg["nms_iou"], self.cfg["inside"], self.cfg["part_ratio"])
        # Mass-loss guard (Sam's monitoring.py rule): most people vanishing in one frame is a detector or camera
        # glitch, not a crowd going under. Skip the frame so no missing timer starts or advances.
        if self._last_visible >= 3 and len(boxes) < self.cfg["mass_loss"] * self._last_visible:
            return self._last_out
        tracked = self.tracker.update_with_detections(self.sv.Detections(
            xyxy=boxes.reshape(-1, 4), confidence=scores, class_id=np.zeros(len(boxes), int)))
        if len(tracked):
            tracked = tracked[tracked.tracker_id >= 0]  # tracks not yet confirmed come back as -1
        visible = {self.person_of[tid] for tid in tracked.tracker_id if tid in self.person_of}
        out = []
        for box, tid, conf in zip(tracked.xyxy.tolist(), tracked.tracker_id.tolist(), tracked.confidence.tolist()):
            continued = False
            if tid not in self.person_of:
                pid = self._stitch(t, box, visible)
                continued = pid is not None
                if continued:  # retire the person's old track, so it can't come back as a second copy
                    self.person_of = {k: v for k, v in self.person_of.items() if v != pid}
                    old = self.people[pid]
                    gap = t - old["last_seen"]
                    if gap >= self.cfg["hold_s"] and old.get("reliable", True):
                        # Went out of sight and came back: Sam's "person reappeared" event, visibility only.
                        old.update(back_after=round(gap, 2), back_until=t + self.cfg["back_show_s"])
                        self.events.append({"person_id": pid, "lost_at": round(old["last_seen"], 3), "back_at": round(t, 3),
                                            "gone_s": round(gap, 2), "last_box": old["box"], "back_box": [float(v) for v in box]})
                if pid is None:
                    pid, self.next_id = self.next_id, self.next_id + 1
                    self.people[pid] = {"first_seen": t}
                self.person_of[tid] = pid
                visible.add(pid)
            pid = self.person_of[tid]
            p = self.people[pid]
            p.update(box=[float(v) for v in box], last_seen=t, conf=float(conf), at_edge=self._at_edge(box))
            p.setdefault("seen", []).append(t)
            p["seen"] = [s for s in p["seen"] if s > t - 2.0]
            p.pop("reliable", None)
            out.append({"person_id": pid, "bbox_xyxy_normalized": p["box"], "confidence": round(p["conf"], 3),
                        "visible": True, "level": "safe", "missing_s": 0.0, "continued": continued,
                        "back_after_s": p["back_after"] if t <= p.get("back_until", -1) else None})
        for pid, p in list(self.people.items()):
            if pid in visible or "box" not in p:
                continue
            missing = t - p["last_seen"]
            if "reliable" not in p:  # decided once, at the moment the person is lost
                recent = [s for s in p.get("seen", []) if s > p["last_seen"] - 2.0]
                p["reliable"] = (p["last_seen"] - p["first_seen"] >= self.cfg["reliable_s"]
                                 and len(recent) >= self.cfg["reliable_frac"] * 2.0 * self.hz)
            if missing > self.cfg["forget_s"] or (p["at_edge"] and missing > self.cfg["hold_s"]) \
                    or (not p["reliable"] and missing > self.cfg["hold_s"]):
                continue  # gone for good, walked out of the view, or never tracked well enough to trust
            if missing < self.cfg["hold_s"]:
                level = "safe"
            else:
                level = "alarm" if missing >= self.cfg["alarm_s"] else "warning" if missing >= self.cfg["warn_s"] else "missing"
            out.append({"person_id": pid, "bbox_xyxy_normalized": p["box"], "confidence": round(p["conf"], 3),
                        "visible": False, "level": level, "missing_s": round(missing, 2), "continued": False})
        self._last_visible, self._last_out = len(visible), out
        return out


if __name__ == "__main__":  # smoke check: one person lost 2 s and back nearby keeps the same ID
    tr = PoolTracker(5, stitch_s=5.0)
    box = [0.40, 0.40, 0.46, 0.55]
    ids = []
    for f in range(60):
        t = f / 5
        if 4 <= t < 6:
            people = tr.update(t, [])
        else:
            shift = 0.01 if t >= 6 else 0.0
            b = [box[0] + shift, box[1], box[2] + shift, box[3]]
            dup = [b[0] + 0.001, b[1], b[2] + 0.001, b[3]]
            people = tr.update(t, [{"bbox_xyxy_normalized": b, "confidence": 0.9},
                                   {"bbox_xyxy_normalized": dup, "confidence": 0.5}])
        ids += [p["person_id"] for p in people if p["visible"]]
        if 4.5 <= t < 6:
            assert any(not p["visible"] for p in people), "lost person should be held"
    assert set(ids) == {1}, f"expected one person, got IDs {sorted(set(ids))}"
    for fr in range(200):  # no frame may show the same person ID twice
        t = 20 + fr / 5
        x = 0.2 + 0.002 * fr
        people = tr.update(t, [] if 5 <= fr % 20 < 9 else [{"bbox_xyxy_normalized": [x, 0.4, x + 0.06, 0.55], "confidence": 0.9}])
        seen = [p["person_id"] for p in people]
        assert len(seen) == len(set(seen)), f"duplicate person IDs at t={t}: {seen}"
    tr = PoolTracker(5)  # resurfacing after 8 s, 2.5 box sizes away, reclaims the same person
    far = [0.70, 0.40, 0.76, 0.55]
    got = set()
    for f in range(100):
        t = f / 5
        if t < 4:
            people = tr.update(t, [{"bbox_xyxy_normalized": far, "confidence": 0.9}])
        elif t < 12:
            people = tr.update(t, [])
        else:
            moved = [far[0] + 0.15, far[1], far[2] + 0.15, far[3]]
            people = tr.update(t, [{"bbox_xyxy_normalized": moved, "confidence": 0.9}])
            got |= {p["person_id"] for p in people}
    assert got == {1} and all(p["visible"] for p in people), f"resurfaced person should keep ID 1, got {sorted(got)}"
    assert len(tr.events) == 1 and tr.events[0]["person_id"] == 1 and 7.5 <= tr.events[0]["gone_s"] <= 8.5, tr.events
    tr = PoolTracker(5)  # mass loss: 4 people, then one frame with a single box, must not start missing timers
    crowd = [[0.1 + 0.2 * i, 0.4, 0.16 + 0.2 * i, 0.55] for i in range(4)]
    for f in range(20):
        people = tr.update(f / 5, [{"bbox_xyxy_normalized": b, "confidence": 0.9} for b in crowd])
    glitch = tr.update(4.0, [{"bbox_xyxy_normalized": crowd[0], "confidence": 0.9}])
    assert glitch is people and all(p["visible"] for p in glitch), "one-frame mass loss should be skipped"
    print("tracking smoke check ok")
