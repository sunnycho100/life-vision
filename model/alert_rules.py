"""GREEN / ORANGE / RED status for each person in the pool, from published norms.

One rule set for everything that colors a box:
  - sim ground truth (sim/isaac/pool_video.py): exact inputs from the simulator
  - the detector pipeline (model/detect_drowning.py): inputs approximated from video
Plain Python, no dependencies, so both the Isaac Sim and the model environments can import it.

GREEN   OK.
ORANGE  Watch: look at this person now.
RED     Alarm: get to this person now.

Rules, first match wins:

RED  R1  Head fully under water for 10 s or more.
         Lifeguard norm: the Ellis & Associates 10/20 rule gives a lifeguard 10 s to recognize an
         aquatic emergency and another 20 s to reach the person and begin care. This is stricter than
         the ASTM F3698-24 test, where a system must alarm for a fully submerged, motionless
         toddler dummy by 20 s. Physiology: someone who goes under holds their breath for at most
         about a minute, and most lose consciousness within about 2 minutes (NEJM 2012; StatPearls),
         so a 10 s alarm leaves time to reach them while they are still conscious.
     R2  Head fully under for 5 s or more AND not moving for the last 2 s.
         Motionless under water is the strongest single sign: Coral MYLO (ASTM F3698 compliant)
         alarms on "motionless, with the head under the surface", and the ASTM test case is a
         motionless submerged dummy.
ORANGE O1  Head fully under for 5 s or more.
         Early warning at half the RED time, so a watcher looks before the alarm. Our design
         choice, not a published number: tune it on real footage (kids hold their breath and
         dive on purpose).
     O2  Instinctive drowning response at the surface for 3 s or more: upright in the water, mouth
         at the waterline (brief dips under count; gaps under 0.7 s are bridged), and no headway.
         Pia (1974); Vittone, U.S. Coast Guard, "Drowning Doesn't Look Like Drowning": "head low in
         the water, mouth at water level", "not using legs, vertical", "trying to swim in a
         particular direction but not making headway". The surface struggle lasts only 20 to 60 s
         before the person goes under, so it has to be flagged within seconds.
     O3  Went under after showing O2: stays ORANGE (someone who was struggling and then slips
         under is not "fine" for the first 5 s under).
GREEN    Everything else.

RED and O3 latch: they clear only after the head has been clearly above water for 2 s.

What we measure or approximate (update() arguments):
  head_under  head fully below the local water surface (head center + head radius under the
              surface). Sim: exact. Detector: the YOLO "underwater" class. None = unknown (for
              example the tracker lost the person), which keeps a running underwater timer going.
  mouth_low   head low, mouth at the waterline: head center less than 0.75 head radius above the
              local surface, and not fully under. Sim only for now; None skips O2.
  upright     body axis (hip to neck) within 30 degrees of vertical. Sim only; None skips O2.
  pos         body position, for headway (sim: hip x, y in meters; detector: box center, pixels).
  points      flat list of body coordinates, for "not moving" (sim: hip, head, hands and feet in
              3D; detector: the box corners). Movement is the RMS displacement per point, so arms
              waving or a body sinking counts as moving. Defaults to pos.
  size        body size in the same units as pos (sim: height in m; detector: box diagonal in px).
              Movement thresholds are fractions of it, so the same rules work for both.
"""
from collections import deque
import math

RED_UNDER_S = 10.0         # R1
RED_STILL_UNDER_S = 5.0    # R2: under at least this long...
STILL_WINDOW_S = 2.0       # ...and body points moved (RMS) less than STILL_FRAC * size over this window
STILL_FRAC = 0.15
ORANGE_UNDER_S = 5.0       # O1
IDR_S = 3.0                # O2: distress signs held this long...
IDR_GAP_S = 0.7            # ...ignoring gaps up to this long (the head bobs)
DIP_S = 1.0                # a fully-under dip shorter than this still counts as "at the surface"
HEADWAY_WINDOW_S = 3.0     # O2: "no headway" = moved less than HEADWAY_FRAC * size over this window
HEADWAY_FRAC = 0.3
RECOVER_S = 2.0            # RED and O3 clear after the head is clearly above water this long
MOUTH_LOW_FRAC = 0.75      # mouth at the waterline: head center under this many head radii above water
UPRIGHT_DEG = 30.0
STALE_S = 0.5              # position history older than this can't prove someone is not moving

COLORS_BGR = {"green": (0, 200, 0), "orange": (0, 165, 255), "red": (0, 0, 255)}


class _Person:
    def __init__(self):
        self.under_since = None      # time the head went fully under
        self.last_under = None       # last time the head was seen fully under
        self.clear_since = None      # time the head came clearly back up (not under, not mouth-low)
        self.idr_since = None
        self.idr_last = None
        self.track = deque()         # (t, pos, points, size)
        self.red = False
        self.distress = False        # O2 was reached; latches as O3 if they go under


class StatusTracker:
    """Keeps per-person timers. Call update() once per frame for each person you know about,
    in time order."""

    def __init__(self, grace_s=0.0, dims=None):
        # grace_s: brief "above water" readings shorter than this don't reset the underwater
        # timer. 0 for exact sim labels; about 1 s for a detector whose class flickers.
        # dims: coordinates per point in `points` (3 for sim joints, 2 for box corners).
        self.grace_s = grace_s
        self.dims = dims
        self.people = {}

    def _moved(self, p, t, window, which):
        """Movement over the last `window` seconds as a fraction of body size (None until the
        history covers the window). which = 1 for pos (headway), 2 for points (stillness)."""
        old = [s for s in p.track if s[0] <= t - window]
        if not old or not p.track or t - p.track[-1][0] > STALE_S:
            return None  # no recent sighting: unknown, never "still"
        a, b, size = old[-1][which], p.track[-1][which], p.track[-1][3]
        if a is None or b is None:
            return None
        dims = 2 if which == 1 else (self.dims or 2)
        n = max(1, len(a) // dims)
        d = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / n)
        return d / max(size, 1e-6)

    def update(self, pid, t, head_under=None, mouth_low=None, upright=None, pos=None, points=None, size=1.0,
               under_start=None):
        """under_start: when a smoothed input only confirms "under" after a delay, the time the
        head actually went under, so the timer is backdated to it."""
        p = self.people.setdefault(pid, _Person())
        if pos is not None:
            pos = tuple(pos)
            p.track.append((t, pos, tuple(points) if points is not None else pos, size))
            while p.track and p.track[0][0] < t - max(STILL_WINDOW_S, HEADWAY_WINDOW_S) - 1.0:
                p.track.popleft()

        # Underwater timer. Unknown (None) keeps a running timer going: a lost underwater
        # person is still underwater until we see otherwise.
        if head_under:
            if p.under_since is None:
                p.under_since = t if under_start is None else min(t, under_start)
            p.last_under, p.clear_since = t, None
        elif head_under is False:
            if p.under_since is not None and t - p.last_under > self.grace_s:
                p.under_since = None
            clearly_up = not mouth_low
            if clearly_up and p.clear_since is None:
                p.clear_since = t
            elif not clearly_up:
                p.clear_since = None
        under_s = 0.0 if p.under_since is None else t - p.under_since
        recovered = p.clear_since is not None and t - p.clear_since >= RECOVER_S

        # Instinctive drowning response at the surface (O2). The mouth bobs in and out, so short
        # full dips count as "at the surface" and short gaps don't reset the timer. A deliberate
        # duck-under (longer than DIP_S) is not surface distress.
        headway = self._moved(p, t, HEADWAY_WINDOW_S, 1)
        low = bool(mouth_low or (head_under and under_s < DIP_S))
        if low and upright and headway is not None and headway < HEADWAY_FRAC:
            p.idr_since = t if p.idr_since is None else p.idr_since
            p.idr_last = t
        elif p.idr_since is not None and t - p.idr_last > IDR_GAP_S:
            p.idr_since = None
        idr_s = 0.0 if p.idr_since is None else t - p.idr_since

        still_moved = self._moved(p, t, STILL_WINDOW_S, 2)
        still = still_moved is not None and still_moved < STILL_FRAC

        if under_s >= RED_UNDER_S:
            p.red, status, reason = True, "red", f"under {under_s:.1f}s"
        elif under_s >= RED_STILL_UNDER_S and still:
            p.red, status, reason = True, "red", f"under {under_s:.1f}s, not moving"
        elif p.red and not recovered:
            status, reason = "red", f"under {under_s:.1f}s" if under_s else "not yet recovered"
        else:
            p.red = False
            if idr_s >= IDR_S:
                p.distress = True
            elif recovered:
                p.distress = False
            if under_s >= ORANGE_UNDER_S:
                status, reason = "orange", f"under {under_s:.1f}s"
            elif idr_s >= IDR_S:
                status, reason = "orange", f"drowning signs {idr_s:.1f}s"
            elif p.distress:
                status, reason = "orange", f"went under after distress, {under_s:.1f}s" if under_s else "distress"
            else:
                status, reason = "green", f"under {under_s:.1f}s" if under_s > 0 else "ok"
        return {"status": status, "reason": reason, "seconds_under": round(under_s, 2),
                "seconds_distress": round(idr_s, 2)}
