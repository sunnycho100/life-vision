"""Render a labeled backyard-pool video in Isaac Sim, from a home-style overhead camera.

Tested on Isaac Sim 6.1 (pip install), Windows 11, RTX 4070 Laptop 8 GB.

Five people follow scripted paths that match the MuJoCo motions in sim/mujoco:
  swimmer   laps along the pool, lying flat, face dipping (never "below" for long)
  treader   treads water in place, head above the surface
  diver     ducks under for about 4 s twice (should give a WARNING, not an ALARM)
  sinker    silent sink: goes under at 3 s, ends motionless on the bottom (ALARM case)
  struggler instinctive drowning response: head bobs in and out, then sinks at 12 s

Arms, hands, legs, spine and head are animated procedurally (pool_anim.py): front crawl,
sculling and eggbeater kick, streamline dive, the arm-pressing drowning response, and a
limp float. Every frame still gets exact labels, including how long
each head has been underwater, which is what the drowning timer needs to be tested on.

Run from the repo root (see sim/isaac/README.md):
    set OMNI_KIT_ACCEPT_EULA=YES
    sim\\isaac\\.venv\\Scripts\\python.exe sim\\isaac\\pool_video.py --seconds 20
Output in --out:
    pool.mp4             the clean video (what the model sees)
    pool_labeled.mp4     same video with true boxes, head state and underwater timers drawn in
    gt.txt               MOT format: frame,id,x,y,w,h,1,-1,-1,-1 (tight box, 1-based frame)
    labels.jsonl         one line per frame: every person's scenario, head height, state,
                         seconds underwater, tight and loose boxes
"""
import argparse
import json
import math
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--width", type=int, default=1920, help="multiple of 16 for H.264")
parser.add_argument("--height", type=int, default=1088, help="multiple of 16 for H.264")
parser.add_argument("--out", default="sim/isaac/_out_video")
parser.add_argument("--sky", default="/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr")
parser.add_argument("--subframes", type=int, default=4, help="RTX subframes per frame; more = cleaner, slower")
parser.add_argument("--pathtrace", type=int, default=0, metavar="SPP",
                    help="use the path tracer with this many samples per frame (e.g. 64): real refraction, "
                    "water color and caustics, but several times slower")
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

scene = PoolScene(simulation_app)
print("character heights (m):", [round(h, 2) for h in scene.heights], flush=True)


def smooth(a, b, t0, t1, t):
    """Ease from a to b between times t0 and t1."""
    if t <= t0:
        return a
    if t >= t1:
        return b
    u = (t - t0) / (t1 - t0)
    return a + (b - a) * (3 * u * u - 2 * u * u * u)


BOTTOM = -POOL_D + 0.2  # eye height of someone lying on the pool floor


def swimmer(t):
    span, speed = POOL_L - 2.5, 0.6
    d = (t * speed) % (2 * span)
    forward = d < span
    x = -span / 2 + (d if forward else 2 * span - d)
    head = -0.03 + 0.05 * math.sin(2 * math.pi * 1.1 * t)  # face in the water, lifting to breathe
    return x, -1.1, head, (0 if forward else 180), 85, anim.freestyle(t)


def treader(t):
    head = 0.12 + 0.04 * math.sin(2 * math.pi * 0.9 * t)
    return 2.6 + 0.1 * math.sin(0.3 * t), 1.1, head, 200 + 10 * math.sin(0.4 * t), 8, anim.tread(t)


def diver(t):
    head, tilt, w = 0.12, 8, 0.0
    for s in (4.0, 13.0):  # two dives, about 4 s under each
        if s - 0.6 <= t < s + 4:
            w = min(1.0, (t - s + 0.6) / 0.6)
            head, tilt = smooth(0.12, -0.8, s, s + 0.8, t), smooth(8, 70, s, s + 0.8, t)
        elif s + 4 <= t < s + 4.8:
            w = 1 - (t - s - 4) / 0.8
            head, tilt = smooth(-0.8, 0.12, s + 4, s + 4.8, t), smooth(70, 8, s + 4, s + 4.8, t)
    return -1.2 + 0.3 * math.sin(0.2 * t), 1.2, head, 30, tilt, anim.blend(anim.tread(t), anim.streamline(t), w)


def sinker(t):
    """Silent sink: stops treading, slips under without a struggle, ends limp on the bottom."""
    head = smooth(0.1, BOTTOM, 3.0, 9.0, t)
    tilt = smooth(8, 80, 4.0, 9.0, t)
    return 0.6, -0.1, head, 120, tilt, anim.blend(anim.tread(t), anim.limp(t), smooth(0, 1, 2.0, 4.5, t))


def struggler(t):
    """Instinctive drowning response for 12 s, then goes limp and sinks."""
    if t < 12:
        head = -0.02 + 0.14 * math.sin(2 * math.pi * 1.1 * t)  # mouth bobbing in and out
        return -2.8, -0.2, head, 300, 12, anim.struggle(t)
    return (-2.8, -0.2, smooth(-0.05, BOTTOM, 12, 17, t), 300, smooth(12, 80, 12, 17, t),
            anim.blend(anim.struggle(t), anim.limp(t), smooth(0, 1, 12, 13.5, t)))


SCENARIOS = [("swimmer", swimmer), ("treader", treader), ("diver", diver),
             ("sinker", sinker), ("struggler", struggler)]

scene.set_light(sky=args.sky)
scene.sun_rot.Set((30, 0, 40))

cam = rep.create.camera(position=(-POOL_L / 2 - 1.0, -POOL_W / 2 - 3.0, 4.2), look_at=(0.3, 0.3, -0.4),
                        focal_length=18.0)
rp = rep.create.render_product(cam, (args.width, args.height))
annot = {k: rep.AnnotatorRegistry.get_annotator(k) for k in ["rgb", "bounding_box_2d_tight", "bounding_box_2d_loose"]}
for a in annot.values():
    a.attach(rp)

os.makedirs(args.out, exist_ok=True)


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


def boxes(data):
    out = {}
    for pp, b in zip(data["info"]["primPaths"], data["data"]):
        i = owner(pp)
        if i is not None and b["x_max"] - b["x_min"] >= 2 and b["y_max"] - b["y_min"] >= 2:
            out[i] = [int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"])]
    return out


n_frames = int(args.seconds * args.fps)
for _ in range(3):  # warm-up renders; the first path-traced frame can show the editor grid
    rep.orchestrator.step(rt_subframes=args.subframes)
for f in range(n_frames):
    t = f / args.fps
    people = {}
    for i, (name, fn) in enumerate(SCENARIOS):
        x, y, head, yaw, tilt, pose = fn(t)
        info = scene.set_person(i, x, y, head, yaw, tilt, pose)
        if info["head_state"] == "below":
            under_since.setdefault(i, t)
        else:
            under_since.pop(i, None)
        info["scenario"] = name
        info["seconds_below"] = round(t - under_since[i], 2) if i in under_since else 0.0
        people[i] = info

    scene.set_ripples(t)
    for _ in range(2):
        simulation_app.update()
    rep.orchestrator.step(rt_subframes=args.subframes)

    frame = np.ascontiguousarray(annot["rgb"].get_data()[:, :, :3][:, :, ::-1])  # RGB -> BGR
    tight, loose = boxes(annot["bounding_box_2d_tight"].get_data()), boxes(annot["bounding_box_2d_loose"].get_data())
    clean.send(np.ascontiguousarray(frame[:, :, ::-1]).tobytes())

    vis = frame.copy()
    for i, info in people.items():
        info["tight_xyxy"], info["loose_xyxy"] = tight.get(i), loose.get(i)
        if i in tight:
            x0, y0, x1, y1 = tight[i]
            gt.write(f"{f + 1},{i + 1},{x0},{y0},{x1 - x0},{y1 - y0},1,-1,-1,-1\n")
        box = tight.get(i) or loose.get(i)
        if box:
            c = COLORS[info["head_state"]]
            cv2.rectangle(vis, box[:2], box[2:], c, (3 if i in tight else 1) * LW)
            label = f'#{i + 1} {info["scenario"]} {info["head_state"]}'
            if info["seconds_below"] > 0:
                label += f' {info["seconds_below"]:.1f}s under'
            cv2.putText(vis, label, (box[0], max(int(28 * FS), box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, FS, (0, 0, 0), 3 * LW)
            cv2.putText(vis, label, (box[0], max(int(28 * FS), box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, FS, c, LW)
    cv2.putText(vis, f"t = {t:5.1f}s  (ground truth)", (10, int(48 * FS)), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * FS, (255, 255, 255), 2 * LW)
    labeled.send(np.ascontiguousarray(vis[:, :, ::-1]).tobytes())
    jl.write(json.dumps({"frame": f + 1, "t": round(t, 3), "people": {str(i + 1): v for i, v in people.items()}}) + "\n")
    if f % args.fps == 0:
        print(f"t={t:.0f}s frame {f + 1}/{n_frames}", flush=True)

clean.close()
labeled.close()
gt.close()
jl.close()
rep.orchestrator.wait_until_complete()
simulation_app.close()
