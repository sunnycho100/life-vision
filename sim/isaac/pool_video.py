"""Render a labeled backyard-pool video in Isaac Sim, from a home-style overhead camera.

Tested on Isaac Sim 6.1 (pip install), Windows 11, RTX 4070 Laptop 8 GB.

Six people, 20 s, chosen to show the cases a pool monitor has to tell apart:
  swimmer    front-crawl laps along the near side, body roll, smooth turns at the walls
  floater    floats on their back, face up, drifting (normal, never an alarm)
  treader    treads water and looks around (normal)
  diver      duck-dives at 3.6 s, swims about 1.8 m under water, resurfaces elsewhere at 7.6 s,
             then a short duck at 13 s (under about 3.5 s and 2 s: should not alarm)
  child      child-sized; dog-paddles, weakens silently, slips under at about 5 s and ends
             motionless on the bottom (the toddler case: ALARM)
  struggler  instinctive drowning response for 9 s, then goes limp and sinks (ALARM)

Arms, hands, legs, spine and head are animated procedurally (pool_anim.py), and the water
surface moves every frame (pool_scene.WaterSurface): ambient waves plus rings from each person.
Every frame gets exact labels, including how long each head has been underwater.

--seed 0 is the fixed demo layout. Other seeds shift positions, headings, timing, which
character plays which role, the sky and the camera, for training data that doesn't
repeat the test clip. --random re-poses everyone at random every frame (stills).

Run from the repo root (see sim/isaac/README.md):
    set OMNI_KIT_ACCEPT_EULA=YES
    sim\\isaac\\.venv\\Scripts\\python.exe sim\\isaac\\pool_video.py --seconds 20 --fps 30 --pathtrace 64 --subframes 1
Output in --out:
    pool.mp4             the clean video (what the model sees)
    pool_labeled.mp4     same video with true boxes, head state and underwater timers drawn in
    gt.txt               MOT format: frame,id,x,y,w,h,1,class,-1,-1 (full-body box, 1-based frame)
    labels.jsonl         one line per frame: every person's scenario, head height, state,
                         seconds underwater, full-body box and above-water box
    yolo/images, yolo/labels   with --yolo-every K: every K-th frame for detector training.
                         Classes: 0 = swimming (head above or partly above), 1 = underwater
"""
import argparse
import json
import math
import os
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--start", type=float, default=0.0, help="start time in the scenario, for quick checks")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=1920, help="multiple of 16 for H.264")
parser.add_argument("--height", type=int, default=1088, help="multiple of 16 for H.264")
parser.add_argument("--out", default="sim/isaac/_out_video")
parser.add_argument("--sky", default="/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr")
parser.add_argument("--seed", type=int, default=0, help="0 = fixed demo layout; others vary it")
parser.add_argument("--random", action="store_true", help="random people and poses every frame (stills)")
parser.add_argument("--yolo-every", type=int, default=0, metavar="K", help="also save every K-th frame for YOLO")
parser.add_argument("--subframes", type=int, default=4, help="RTX subframes per frame; more = cleaner, slower")
parser.add_argument("--pathtrace", type=int, default=0, metavar="SPP",
                    help="use the path tracer with this many samples per frame (e.g. 64): real refraction, "
                    "water color and caustics, but several times slower")
parser.add_argument("--fill", type=float, default=150.0, help="underwater fill light strength (0 = off)")
parser.add_argument("--gui", action="store_true", help="open the Isaac Sim window instead of headless")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

import cv2  # noqa: E402  (drawing only; this build has no video encoder)
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_anim as anim  # noqa: E402
from pool_scene import POOL_D, POOL_L, POOL_W, PoolScene  # noqa: E402

if args.pathtrace:
    import carb.settings  # noqa: E402

    st = carb.settings.get_settings()
    st.set("/rtx/rendermode", "PathTracing")
    st.set("/rtx/pathtracing/spp", args.pathtrace)
    st.set("/rtx/pathtracing/totalSpp", args.pathtrace)
    st.set("/rtx/pathtracing/optixDenoiser/enabled", True)

rng = random.Random(args.seed)
scene = PoolScene(simulation_app, fill=args.fill)
print("character heights (m):", [round(h, 2) for h in scene.heights], flush=True)


def smooth(a, b, t0, t1, t):
    """Ease from a to b between times t0 and t1."""
    if t <= t0:
        return a
    if t >= t1:
        return b
    u = (t - t0) / (t1 - t0)
    return a + (b - a) * (3 * u * u - 2 * u * u * u)


def bump(t0, t1, t):
    """0 -> 1 -> 0 over [t0, t1] (half a sine)."""
    return math.sin(math.pi * (t - t0) / (t1 - t0)) if t0 < t < t1 else 0.0


BOTTOM = -POOL_D + 0.2  # head-center height of an adult lying on the pool floor
wob = anim.wobble

# Each role returns: x, y (hip), head (head-center height), yaw, tilt, roll, pose, ripple amplitude (m).


def swimmer(t):
    """Front crawl along the near side. Speed eases down into each wall, the swimmer turns
    toward the middle of the pool while lifting the head, then pushes off."""
    x0, x1, y, v, turn = -2.8, 2.8, -1.35, 0.65, 1.8
    lap = (x1 - x0) / v
    u = (t + 1.0) % (2 * (lap + turn))  # start 1 s into the first lap

    def along(p):  # slower near the walls, faster mid-pool
        return p - 0.5 * math.sin(2 * math.pi * p) / (2 * math.pi)
    turning, s = 0.0, 0.0
    if u < lap:
        x, yaw = x0 + (x1 - x0) * along(u / lap), 0.0
    elif u < lap + turn:
        s = (u - lap) / turn
        x, yaw, turning = x1 + 0.15 * math.sin(math.pi * s), smooth(0, 180, 0, 1, s), math.sin(math.pi * s)
    elif u < 2 * lap + turn:
        x, yaw = x1 - (x1 - x0) * along((u - lap - turn) / lap), 180.0
    else:
        s = (u - 2 * lap - turn) / turn
        x, yaw, turning = x0 - 0.15 * math.sin(math.pi * s), smooth(180, 0, 0, 1, s), math.sin(math.pi * s)
    head = -0.05 + 0.025 * math.sin(2 * math.pi * 1.1 * t) + 0.08 * turning  # face in the water
    pose = anim.blend(anim.freestyle(t, 1), anim.tread(t, 1), 0.8 * turning)
    return (x, y + 0.03 * wob(t, 1, 0.3), head, yaw, 85 - 25 * turning, anim.freestyle_roll(t) * (1 - turning),
            pose, 0.012)


def floater(t):
    x, y = 1.9 + 0.15 * math.sin(0.25 * t), 0.35 + 0.1 * math.sin(0.19 * t + 1)
    head = -0.03 + 0.012 * math.sin(2 * math.pi * 0.25 * t)  # face out, back of the head in the water
    return x, y, head, 10 + 12 * math.sin(0.13 * t), -82, 5 * math.sin(0.4 * t), anim.backfloat(t, 2), 0.003


def treader(t):
    head = 0.13 + 0.035 * math.sin(2 * math.pi * 0.9 * t + 0.4 * wob(t, 4, 0.3))
    return (2.8 + 0.08 * math.sin(0.3 * t), 1.25 + 0.06 * math.sin(0.23 * t), head, 205 + 25 * wob(t, 4, 0.1),
            8 + 3 * wob(t, 5, 0.2), 0.0, anim.tread(t, 4), 0.008)


def diver(t):
    ax, ay, bx, by = -1.0, 1.3, 0.8, 1.15
    tread, down, glide = anim.tread(t, 3), anim.streamline(t), anim.underwater_swim(t)
    ripple = 0.008 + 0.03 * math.exp(-max(t - 3.9, 0) / 0.6) * (t > 3.6) + 0.02 * math.exp(-max(t - 7.3, 0) / 0.6) * (t > 7.0)
    if t < 3.6:  # treading at A
        return ax, ay, 0.12, 0.0, 8, 0.0, tread, 0.008
    if t < 4.2:  # duck dive, head first
        return (ax + 0.2 * smooth(0, 1, 3.6, 4.2, t), ay, smooth(0.12, -0.65, 3.6, 4.2, t), 0.0,
                smooth(8, 95, 3.6, 4.2, t), 0.0, anim.blend(tread, down, smooth(0, 1, 3.6, 3.9, t)), ripple)
    if t < 7.0:  # swim under water from A to B
        p = smooth(0, 1, 4.2, 7.0, t)
        yaw = math.degrees(math.atan2(by - ay, bx - ax - 0.2))
        return (ax + 0.2 + (bx - ax - 0.2) * p, ay + (by - ay) * p, -0.75 + 0.05 * math.sin(3 * t),
                yaw * smooth(0, 1, 4.2, 4.8, t), smooth(95, 82, 4.2, 5.0, t), 0.0,
                anim.blend(down, glide, smooth(0, 1, 4.2, 4.8, t)), ripple)
    yaw = math.degrees(math.atan2(by - ay, bx - ax - 0.2))
    bob = 0.02 * math.sin(2 * math.pi * 0.9 * t)
    if t < 7.6:  # surface at B
        return (bx, by, smooth(-0.75, 0.12 + bob, 7.0, 7.6, t), yaw, smooth(82, 8, 7.0, 7.6, t), 0.0,
                anim.blend(glide, tread, smooth(0, 1, 7.0, 7.6, t)), ripple)
    duck = smooth(0, 1, 13.0, 13.5, t) - smooth(0, 1, 15.3, 15.8, t)  # short playful duck under
    look = 20 * (wob(t, 3, 0.15) - wob(7.6, 3, 0.15)) * smooth(0, 1, 7.6, 9.0, t)
    return bx, by, 0.12 + bob - 0.62 * duck, yaw + look, 8, 0.0, tread, 0.008 + 0.015 * bump(12.9, 15.9, t)


CHILD_BOTTOM = -POOL_D + 0.12  # head-center height of a child lying on the floor


def child(t):
    """Toddler-style silent sink: no splashing, no calling out."""
    effort = smooth(1.0, 0.0, 3.0, 5.5, t)
    if t < 5.0:  # tiring: the face sinks to the waterline, the paddling fades
        head = smooth(0.05, -0.01, 3.0, 5.0, t) + 0.03 * math.sin(2 * math.pi * 1.3 * t) * effort
    else:  # slips under and sinks
        head = smooth(-0.01, CHILD_BOTTOM, 5.0, 9.0, t)
    tilt = smooth(25, 10, 3.0, 5.0, t) if t < 5.0 else smooth(10, 75, 5.0, 9.5, t)
    pose = anim.blend(anim.dogpaddle(t, 5, effort), anim.limp(t, 5), smooth(0, 1, 5.0, 6.5, t))
    drift = 0.05 * smooth(0, 1, 5, 9, t)
    return -2.3 + drift, 0.5, head, 235, tilt, 0.0, pose, 0.006 * effort


def struggler(t):
    """Instinctive drowning response for 9 s, then goes limp and sinks."""
    if t < 9:
        head = -0.03 + 0.11 * math.sin(2 * math.pi * 1.1 * t + 0.9 * wob(t, 6, 0.35))
        return 0.3, -0.15, head, 220 + 10 * wob(t, 6, 0.2), 5, 0.0, anim.struggle(t, 6), 0.022
    return (0.3, -0.15, smooth(-0.05, BOTTOM, 9, 13, t), 220 + 10 * wob(9, 6, 0.2), smooth(5, 78, 9, 13.5, t), 0.0,
            anim.blend(anim.struggle(t, 6), anim.limp(t, 6), smooth(0, 1, 9, 10.5, t)),
            0.022 * smooth(1, 0, 9, 11, t))


# (role, function, default character index, body scale)
ROLES = [("swimmer", swimmer, 0, 1.0), ("floater", floater, 5, 1.0), ("treader", treader, 3, 1.0),
         ("diver", diver, 1, 1.0), ("child", child, 4, 0.62), ("struggler", struggler, 2, 1.0)]

# Seed 0 keeps the demo layout. Other seeds shift each role in space and time and
# reassign characters, so training clips don't repeat the test clip.
cast = [r[2] for r in ROLES]
shift = [(0.0, 0.0, 0.0, 0.0)] * len(ROLES)
if args.seed:
    cast = rng.sample(range(len(scene.people)), len(ROLES))
    shift = [(rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3), rng.uniform(-40, 40), rng.uniform(-3, 4))
             for _ in ROLES]


def clamp_xy(x, y):
    return max(-POOL_L / 2 + 0.6, min(POOL_L / 2 - 0.6, x)), max(-POOL_W / 2 + 0.5, min(POOL_W / 2 - 0.5, y))


RANDOM_MOTIONS = {  # motion -> (tilt range, eye height range)
    "tread": ((0, 15), (-0.5, 0.25)), "struggle": ((5, 20), (-0.3, 0.15)), "relaxed": ((0, 10), (-0.6, 0.3)),
    "freestyle": ((80, 90), (-0.15, 0.05)), "limp": ((40, 88), (BOTTOM, -0.1)), "streamline": ((30, 80), (-1.3, -0.1)),
    "backfloat": ((-88, -70), (0.0, 0.1)), "dogpaddle": ((15, 40), (-0.1, 0.08)),
    "underwater_swim": ((70, 95), (-1.3, -0.3)),
}


def random_frame(t):
    """Stills mode: a random subset of people, each in a random motion, place, depth and size."""
    n = rng.randint(2, 5)
    chosen, spots, out, sources = rng.sample(range(len(scene.people)), n), [], {}, []
    for i in range(len(scene.people)):
        if i not in chosen:
            scene.park(i)
            continue
        for _ in range(50):
            x, y = rng.uniform(-POOL_L / 2 + 0.8, POOL_L / 2 - 0.8), rng.uniform(-POOL_W / 2 + 0.6, POOL_W / 2 - 0.6)
            if all(math.hypot(x - a, y - b) > 1.2 for a, b in spots):
                break
        spots.append((x, y))
        motion = rng.choice(list(RANDOM_MOTIONS))
        (t0, t1), (h0, h1) = RANDOM_MOTIONS[motion]
        scale = rng.uniform(0.58, 0.72) if rng.random() < 0.25 else 1.0
        pose = getattr(anim, motion)(rng.uniform(0, 30))
        roll = anim.freestyle_roll(rng.uniform(0, 30)) if motion == "freestyle" else 0.0
        head = rng.uniform(h0, h1)
        info = scene.set_person(i, x, y, head, rng.uniform(-180, 180), rng.uniform(t0, t1), pose, roll, scale)
        info["scenario"] = motion
        out[i] = info
        sources.append((x, y, rng.uniform(0.0, 0.02) if head > -0.2 else 0.0, rng.uniform(0, 6)))
    scene.set_water(t + 7 * args.seed, sources)
    return out


scene.set_light(rng, sky=None if args.seed else args.sky)
if not args.seed:
    scene.set_sun(52, -125, 3200)  # afternoon sun from behind the camera, so the walls it sees are lit

cam_pos, cam_look = (-POOL_L / 2 - 1.0, -POOL_W / 2 - 3.0, 4.2), (0.3, 0.3, -0.4)
if args.seed:
    cam_pos = tuple(c + rng.uniform(-0.6, 0.6) for c in cam_pos)
    cam_look = tuple(c + rng.uniform(-0.4, 0.4) for c in cam_look)
cam = rep.create.camera(position=cam_pos, look_at=cam_look, focal_length=18.0)
rp = rep.create.render_product(cam, (args.width, args.height))
annot = {k: rep.AnnotatorRegistry.get_annotator(k) for k in ["rgb", "bounding_box_2d_tight", "camera_params"]}
for a in annot.values():
    a.attach(rp)

os.makedirs(args.out, exist_ok=True)
if args.yolo_every:
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(args.out, "yolo", sub), exist_ok=True)


def video(name):
    """H.264 MP4 writer (plays in browsers and on phones). Send BGR frames."""
    w = imageio_ffmpeg.write_frames(os.path.join(args.out, name), (args.width, args.height), fps=args.fps,
                                    codec="libx264", pix_fmt_out="yuv420p", quality=8)
    w.send(None)
    return w


clean, labeled = video("pool.mp4"), video("pool_labeled.mp4")
gt = open(os.path.join(args.out, "gt.txt"), "w")
jl = open(os.path.join(args.out, "labels.jsonl"), "w")

FS, LW = 0.5 * args.width / 1280, max(1, args.width // 1280)  # text size and line width scale with resolution
COLORS = {"above": (0, 200, 0), "partial": (0, 170, 255), "below": (0, 0, 230)}  # BGR
roots = [str(p.GetPath()) for p in scene.people]
under_since = {}


def owner(prim_path):
    for i, r in enumerate(roots):
        if prim_path == r or prim_path.startswith(r + "/"):
            return i
    return None


def tight_boxes(data):
    """Visible-pixel boxes. The water counts as an occluder, so these cover only what is above the surface."""
    out = {}
    for pp, b in zip(data["info"]["primPaths"], data["data"]):
        i = owner(pp)
        if i is not None and b["x_max"] - b["x_min"] >= 2 and b["y_max"] - b["y_min"] >= 2:
            out[i] = [int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"])]
    return out


def refract(pts, cam, n=1.333):
    """Where the camera sees underwater points: replace each point below the surface (z < 0)
    with the spot on the surface where its light ray exits toward the camera (Snell's law,
    solved by bisection, against the mean surface z = 0). Submerged limbs look shallower."""
    pts = pts.copy()
    under = pts[:, 2] < 0
    if not under.any() or cam[2] <= 0:
        return pts
    p = pts[under]
    horiz = p[:, :2] - cam[:2]
    dist = np.linalg.norm(horiz, axis=1) + 1e-9
    h, d = cam[2], -p[:, 2]
    lo, hi = np.zeros_like(dist), dist.copy()
    for _ in range(40):  # r = horizontal distance from the camera to the exit point
        r = (lo + hi) / 2
        f = r / np.hypot(r, h) - n * (dist - r) / np.hypot(dist - r, d)
        lo, hi = np.where(f < 0, r, lo), np.where(f < 0, hi, r)
    r = (lo + hi) / 2
    exit_xy = cam[:2] + horiz * (r / dist)[:, None]
    pts[under] = np.column_stack([exit_xy, np.zeros(len(p))])
    return pts


def body_box(i, cp):
    """Full-body box from the skeleton projected through the camera, underwater parts included
    and shifted to where refraction makes them appear."""
    view = np.array(cp["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    proj = np.array(cp["cameraProjection"], dtype=np.float64).reshape(4, 4)
    cam = np.linalg.inv(view)[3, :3]
    pts = refract(np.array([[p[0], p[1], p[2]] for p in scene.body_points(i)]), cam)
    pts = np.column_stack([pts, np.ones(len(pts))])
    clip = pts @ view @ proj
    if np.any(clip[:, 3] <= 0):
        return None
    ndc = clip[:, :2] / clip[:, 3:4]
    px = (ndc[:, 0] + 1) / 2 * args.width
    py = (1 - ndc[:, 1]) / 2 * args.height
    x0, x1, y0, y1 = px.min(), px.max(), py.min(), py.max()
    pad = 0.04 * max(x1 - x0, y1 - y0) + 3  # joints sit inside the skin
    x0, y0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    x1, y1 = min(args.width - 1, int(x1 + pad)), min(args.height - 1, int(y1 + pad))
    return [x0, y0, x1, y1] if x1 - x0 > 4 and y1 - y0 > 4 else None


n_frames = int(round(args.seconds * args.fps))
for _ in range(3):  # warm-up renders; the first path-traced frame can show the editor grid
    rep.orchestrator.step(rt_subframes=args.subframes)
for f in range(n_frames):
    t = args.start + f / args.fps
    if args.random:
        people = random_frame(t)
        scene.set_light(rng)
    else:
        people, sources = {}, []
        for k, (name, fn, _, scale) in enumerate(ROLES):
            i = cast[k]
            dx, dy, dyaw, dt = shift[k]
            x, y, head, yaw, tilt, roll, pose, ripple = fn(max(0.0, t + dt))
            x, y = clamp_xy(x + dx, y + dy)
            info = scene.set_person(i, x, y, head, yaw + dyaw, tilt, pose, roll, scale)
            info["scenario"] = name
            people[i] = info
            sources.append((x, y, ripple, 1.7 * k))
        for i in range(len(scene.people)):
            if i not in cast:
                scene.park(i)
        scene.set_water(t + 7 * args.seed, sources)
    for i, info in people.items():
        if info["head_state"] == "below":
            under_since.setdefault(i, t)
        else:
            under_since.pop(i, None)
        info["seconds_below"] = round(t - under_since[i], 2) if i in under_since else 0.0

    for _ in range(2):
        simulation_app.update()
    rep.orchestrator.step(rt_subframes=args.subframes)

    frame = np.ascontiguousarray(annot["rgb"].get_data()[:, :, :3][:, :, ::-1])  # RGB -> BGR
    tight = tight_boxes(annot["bounding_box_2d_tight"].get_data())
    cp = annot["camera_params"].get_data()
    clean.send(np.ascontiguousarray(frame[:, :, ::-1]).tobytes())
    save_yolo = args.yolo_every and f % args.yolo_every == 0
    yolo_lines = []

    vis = frame.copy()
    for i, info in people.items():
        body = body_box(i, cp)
        info["body_xyxy"], info["above_water_xyxy"] = body, tight.get(i)
        if body is None:
            continue
        cls = 1 if info["head_state"] == "below" else 0
        x0, y0, x1, y1 = body
        gt.write(f"{f + 1},{i + 1},{x0},{y0},{x1 - x0},{y1 - y0},1,{cls},-1,-1\n")
        if save_yolo:
            yolo_lines.append(f"{cls} {(x0 + x1) / 2 / args.width:.6f} {(y0 + y1) / 2 / args.height:.6f} "
                              f"{(x1 - x0) / args.width:.6f} {(y1 - y0) / args.height:.6f}")
        c = COLORS[info["head_state"]]
        cv2.rectangle(vis, body[:2], body[2:], c, 2 * LW)
        label = f'#{i + 1} {info["scenario"]} {info["head_state"]}'
        if info["seconds_below"] > 0:
            label += f' {info["seconds_below"]:.1f}s under'
        cv2.putText(vis, label, (x0, max(int(28 * FS), y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, FS, (0, 0, 0), 3 * LW)
        cv2.putText(vis, label, (x0, max(int(28 * FS), y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, FS, c, LW)
    cv2.putText(vis, f"t = {t:5.1f}s  (ground truth)", (10, int(48 * FS)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * FS, (255, 255, 255), 2 * LW)
    labeled.send(np.ascontiguousarray(vis[:, :, ::-1]).tobytes())
    if save_yolo:
        stem = f"s{args.seed}{'r' if args.random else ''}_{f:05d}"
        cv2.imwrite(os.path.join(args.out, "yolo", "images", stem + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(args.out, "yolo", "labels", stem + ".txt"), "w") as fh:
            fh.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
    jl.write(json.dumps({"frame": f + 1, "t": round(t, 3), "people": {str(i + 1): v for i, v in people.items()}}) + "\n")
    jl.flush()
    if f % args.fps == 0:
        print(f"t={t:.0f}s frame {f + 1}/{n_frames}", flush=True)

clean.close()
labeled.close()
gt.close()
jl.close()
print("DONE", flush=True)
rep.orchestrator.wait_until_complete()
simulation_app.close()
