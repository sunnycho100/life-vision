"""Camera fly-through of the pool scene, for the demo video. Runs pool_video.py unmodified and
moves its camera along a smooth path before every rendered frame: high and wide over the
backyard, down to the water, a circle around the pool, then into the home CCTV view that the
detector uses. The people, water and labels are exactly what pool_video.py produces.

With --gui the Isaac Sim window opens and its viewport follows the same path, so you can
watch the tour live (and screen-record it).

Run from the repo root (the extra arguments go to pool_video.py):
    set OMNI_KIT_ACCEPT_EULA=YES
    sim\\isaac\\.venv\\Scripts\\python.exe sim\\isaac\\pool_flythrough.py --seconds 12 ^
        --pathtrace 32 --subframes 1 --out sim\\isaac\\_out_demo\\flythrough
"""
import math
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
gui = "--gui" in argv
seconds = float(argv[argv.index("--seconds") + 1]) if "--seconds" in argv else 20.0
fps = int(argv[argv.index("--fps") + 1]) if "--fps" in argv else 30

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not gui})
import isaacsim  # noqa: E402

isaacsim.SimulationApp = lambda *a, **k: app  # pool_video.py reuses this app instead of starting a second one

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

# (seconds into the tour, camera position, look-at point). pool is 8 x 4 m, water surface at z = 0.
CCTV = ((-5.0, -5.0, 4.2), (0.3, 0.3, -0.4))  # the camera pool_video.py renders from
KEYS = [
    (0.00, (-16.0, -13.0, 10.0), (0.0, 0.0, 0.0)),   # high and wide: the whole backyard
    (0.22, (-8.5, -6.5, 4.5), (0.0, 0.0, -0.3)),     # coming down toward the pool
    (0.40, (-2.0, -5.2, 1.6), (0.5, 0.0, -0.5)),     # low along the long side, near the water
    (0.58, (5.5, -3.8, 2.2), (0.0, 0.3, -0.5)),      # around the far corner
    (0.74, (6.0, 3.6, 3.0), (0.0, 0.0, -0.5)),       # across the pool from behind
    (1.00, CCTV[0], CCTV[1]),                        # settle into the CCTV view
]


def catmull(p0, p1, p2, p3, u):
    return tuple(0.5 * ((2 * b) + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u * u + (-a + 3 * b - 3 * c + d) * u ** 3)
                 for a, b, c, d in zip(p0, p1, p2, p3))


def camera_at(s):
    """Smooth path through KEYS for s in [0, 1] (eased so it starts and lands gently)."""
    s = 0.5 - 0.5 * math.cos(math.pi * min(max(s, 0.0), 1.0))
    k = max(i for i in range(len(KEYS) - 1) if KEYS[i][0] <= s) if s < 1 else len(KEYS) - 2
    u = (s - KEYS[k][0]) / (KEYS[k + 1][0] - KEYS[k][0])
    idx = [max(0, k - 1), k, k + 1, min(len(KEYS) - 1, k + 2)]
    pos = catmull(*[KEYS[i][1] for i in idx], u)
    look = catmull(*[KEYS[i][2] for i in idx], u)
    return pos, look


state = {"n": -3, "cam": None}  # pool_video.py does 3 warm-up steps before frame 0
_step = rep.orchestrator.step


def step(*a, **k):
    stage = omni.usd.get_context().get_stage()
    if state["cam"] is None:  # the camera rep.create.camera made in pool_video.py
        cams = [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera) and str(p.GetPath()).startswith("/Replicator")]
        state["cam"] = cams[0]
        xf = UsdGeom.Xformable(state["cam"])
        xf.ClearXformOpOrder()
        state["op"] = xf.AddTransformOp()
        if gui:
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().camera_path = str(state["cam"].GetPath())
    pos, look = camera_at(max(state["n"], 0) / max(1, seconds * fps - 1))
    # USD cameras look down -Z with +Y up; SetLookAt builds the view matrix, so invert it. Replicator
    # puts the camera under a transformed parent, so convert the world pose into the parent's space.
    world = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1)).GetInverse()
    parent = UsdGeom.Xformable(state["cam"]).ComputeParentToWorldTransform(Usd.TimeCode.Default())
    state["op"].Set(world * parent.GetInverse())
    state["n"] += 1
    for _ in range(1 if not gui else 2):
        app.update()
    return _step(*a, **k)


rep.orchestrator.step = step
sys.argv = [os.path.join(HERE, "pool_video.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")
