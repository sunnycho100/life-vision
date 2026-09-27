"""Random still images of people in a pool, labeled for YOLO detector training.

Tested on Isaac Sim 6.1 (pip install), Windows 11, RTX 4070 Laptop 8 GB.

Idea (from sunnycho100's starter): for detector training we do not need animation.
Drop human characters at random positions and depths so the water surface cuts them at
different heights, render from an overhead and an underwater camera, and write
pixel-perfect boxes. For a moving, time-labeled clip use pool_video.py instead.

We place people from Python (not the Replicator randomizer graph) so we know each
person's exact head height every frame. That gives a free head_state label
(above / partial / below the surface), which real footage can't give us.

Run from the repo root (see sim/isaac/README.md):
    set OMNI_KIT_ACCEPT_EULA=YES
    sim\\isaac\\.venv\\Scripts\\python.exe sim\\isaac\\pool_replicator.py --frames 20
Output:
    <out>/<camera>/images/000000.png
    <out>/<camera>/labels/000000.txt    YOLO: class cx cy w h (normalized), class 0 = person
    <out>/<camera>/meta/000000.json     per person: head_state, head height, tight and loose box
"""
import argparse
import json
import math
import os
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=20)
parser.add_argument("--out", default="sim/isaac/_out_pool")
parser.add_argument("--res", type=int, default=640, help="square image size; keep small on 8 GB GPUs")
parser.add_argument("--people", type=int, default=4, help="max people in the pool per frame")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settle", type=int, default=3, help="app updates between moving people and capturing")
parser.add_argument("--gui", action="store_true", help="open the Isaac Sim window instead of headless")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_scene import POOL_L, POOL_W, PoolScene  # noqa: E402

random.seed(args.seed)
scene = PoolScene(simulation_app)
people = scene.people
print("character heights (m):", [round(h, 2) for h in scene.heights], flush=True)

# Cameras: overhead like a camera on the house, and one just under the surface.
cam_over = rep.create.camera(position=(0, -POOL_W / 2 - 1.5, 4.5), look_at=(0, 0, -0.3))
cam_under = rep.create.camera(position=(-POOL_L / 2 + 0.3, 0, -0.9), look_at=(0, 0, -0.9))
cams = {"overhead": cam_over, "underwater": cam_under}
rps = {name: rep.create.render_product(cam, (args.res, args.res)) for name, cam in cams.items()}
annot = {}
for name, rp in rps.items():
    annot[name] = {}
    for kind in ["rgb", "bounding_box_2d_tight", "bounding_box_2d_loose"]:
        a = rep.AnnotatorRegistry.get_annotator(kind)
        a.attach(rp)
        annot[name][kind] = a
    for sub in ["images", "labels", "meta"]:
        os.makedirs(os.path.join(args.out, name, sub), exist_ok=True)


def place_people():
    """Random x, y, yaw and depth for a random subset; park the rest. Returns labels by prim path."""
    n = random.randint(1, min(args.people, len(people)))
    chosen = random.sample(range(len(people)), n)
    spots, info = [], {}
    for i, p in enumerate(people):
        if i not in chosen:
            scene.park(i)
            continue
        for _ in range(50):  # keep people at least 0.9 m apart
            x = random.uniform(-POOL_L / 2 + 0.5, POOL_L / 2 - 0.5)
            y = random.uniform(-POOL_W / 2 + 0.5, POOL_W / 2 - 0.5)
            if all(math.hypot(x - a, y - b) > 0.9 for a, b in spots):
                break
        spots.append((x, y))
        # Head center from 0.6 m under the surface to 0.6 m above it (chest out).
        info[str(p.GetPath())] = scene.set_person(i, x, y, random.uniform(-0.6, 0.6), random.uniform(-180, 180))
    return info


def owner(prim_path, info):
    """Map a mesh prim path from the annotator back to its person_i root."""
    for root in info:
        if prim_path == root or prim_path.startswith(root + "/"):
            return root
    return None


for f in range(args.frames):
    info = place_people()
    scene.set_light(random)
    # Let the app settle the new poses before capturing. With fewer ticks, box data for
    # skinned characters that just moved can lag and people go missing from the labels.
    for _ in range(args.settle):
        simulation_app.update()
    rep.orchestrator.step(rt_subframes=4)
    for name in cams:
        a = annot[name]
        Image.fromarray(a["rgb"].get_data()[:, :, :3]).save(os.path.join(args.out, name, "images", f"{f:06d}.png"))
        tight = a["bounding_box_2d_tight"].get_data()
        loose = a["bounding_box_2d_loose"].get_data()
        # Tight box = visible pixels. The water counts as an occluder, so a tight box covers only
        # what is above the surface, and a fully submerged person has none. That is what the YOLO
        # labels use. The loose box (full body, occlusion ignored) goes in meta for every person.
        xyxy = lambda b: [float(b[k]) for k in ("x_min", "y_min", "x_max", "y_max")]  # noqa: E731
        tight_by_path = {pp: b for pp, b in zip(tight["info"]["primPaths"], tight["data"])}
        loose_by_path = {pp: b for pp, b in zip(loose["info"]["primPaths"], loose["data"])}
        lines, meta = [], []
        r = args.res
        for pp in sorted(set(tight_by_path) | set(loose_by_path)):
            root = owner(pp, info)
            if root is None:
                continue
            tb = tight_by_path.get(pp)
            tbox = xyxy(tb) if tb is not None else None
            if tbox and (tbox[2] - tbox[0] < 2 or tbox[3] - tbox[1] < 2):
                tbox = None
            if tbox:
                x0, y0, x1, y1 = tbox
                lines.append(f"0 {(x0 + x1) / 2 / r:.6f} {(y0 + y1) / 2 / r:.6f} {(x1 - x0) / r:.6f} {(y1 - y0) / r:.6f}")
            lb = loose_by_path.get(pp)
            meta.append({
                "prim": root,
                **info[root],
                "tight_xyxy": tbox,
                "loose_xyxy": xyxy(lb) if lb is not None else None,
            })
        with open(os.path.join(args.out, name, "labels", f"{f:06d}.txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        with open(os.path.join(args.out, name, "meta", f"{f:06d}.json"), "w") as fh:
            json.dump({"frame": f, "people": info, "boxes": meta}, fh, indent=1)
    print(f"frame {f + 1}/{args.frames}: {len(info)} people", flush=True)

rep.orchestrator.wait_until_complete()
simulation_app.close()
