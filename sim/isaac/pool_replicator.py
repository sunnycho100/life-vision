"""UNVERIFIED starter: synthetic pool images with human characters in Isaac Sim Replicator.

Not run. Isaac Sim does not install on macOS. Needs an NVIDIA RTX GPU (RT cores, 16 GB+ VRAM)
on Ubuntu 22.04/24.04 or Windows 10/11. API names follow Isaac Sim 4.5 / 5.x docs and may
need small fixes for the installed version.

Idea: for detector training we do not need animation. Drop human characters at random
positions and depths so the water surface cuts them at different heights, render from an
overhead and an underwater camera, and let Replicator write pixel-perfect 2D boxes.

Run (inside the Isaac Sim python):  ./python.sh pool_replicator.py
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402  (4.x: omni.isaac.core.utils.nucleus)

OUT_DIR = "_out_pool"
NUM_FRAMES = 200
POOL = (8.0, 4.0, 2.0)  # length, width, depth in meters, surface at z = 0

assets = get_assets_root_path()
# Placeholder paths: check the People asset folder in your Nucleus for real character USDs.
CHARACTERS = [
    assets + "/Isaac/People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd",
    assets + "/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
]

with rep.new_layer():
    # Pool shell and floor
    rep.create.cube(position=(0, 0, -POOL[2] - 0.05), scale=(POOL[0], POOL[1], 0.1),
                    semantics=[("class", "pool")])
    # Water surface: thin plane. TODO: assign a transparent water MDL (e.g. OmniGlass tinted blue).
    rep.create.plane(position=(0, 0, 0), scale=(POOL[0] / 2, POOL[1] / 2, 1),
                     semantics=[("class", "water")])

    people = rep.create.group([
        rep.create.from_usd(path, semantics=[("class", "person")]) for path in CHARACTERS
    ])

    sun = rep.create.light(light_type="distant", rotation=(-45, 0, 0))

    overhead = rep.create.camera(position=(0, -POOL[1] / 2 - 2, 3.5), look_at=(0, 0, 0))
    underwater = rep.create.camera(position=(0, -POOL[1] / 2 + 0.2, -1.0), look_at=(0, 0, -0.5))
    rps = [rep.create.render_product(overhead, (1280, 720)),
           rep.create.render_product(underwater, (1280, 720))]

    with rep.trigger.on_frame(num_frames=NUM_FRAMES):
        with people:
            # z between -1.6 (standing, head under) and -0.9 (chest out) for about 1.7 m characters
            rep.modify.pose(
                position=rep.distribution.uniform((-3, -1.5, -1.6), (3, 1.5, -0.9)),
                rotation=rep.distribution.uniform((0, 0, -180), (0, 0, 180)),
            )
        with sun:
            rep.modify.attribute("intensity", rep.distribution.uniform(1000, 6000))  # glare vs shade

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=OUT_DIR, rgb=True, bounding_box_2d_tight=True,
                      semantic_segmentation=True)
    writer.attach(rps)

rep.orchestrator.run_until_complete()
simulation_app.close()
