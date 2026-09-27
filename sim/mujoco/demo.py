"""10-second demo render with the nicer visuals (textures, shadows, body shapes, 720p).

Same physics as the test scenarios. Writes presentation/videos/demo_hq_10s.mp4.
Usage: python demo.py [--seconds 10] [--seed 1]
"""
import argparse
import shutil

import drown_sim as ds
import pool_scene as ps

CAST = [("stand", -0.8, 1.3, 20), ("normal_swim", -0.2, -1.2, 0), ("duck_under", 1.4, -0.6, 0),
        ("collapse", 1.8, 1.4, 90), ("float", 5.0, -1.0, 30), ("idr", 6.4, 1.2, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = ds.HERE.parents[1] / "data" / "sim_demo"
    s = ps.run(CAST, args.seconds, seed=args.seed, out_dir=out, name="demo_hq", gt=True, size=(1280, 720), hq=True)
    dst = ds.HERE.parents[1] / "presentation" / "videos" / "demo_hq_10s.mp4"
    shutil.copy(out / "videos" / "demo_hq.mp4", dst)
    for p in s["people"]:
        print(p["person"], p["motion"], "under %", p["pct_frames_head_underwater"], "first under", p["first_underwater_s"])
    print("wrote", dst)


if __name__ == "__main__":
    main()
