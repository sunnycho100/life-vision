"""End-to-end test scenarios with simulator ground truth.

Each scenario writes to data/sim_scenarios/:
  <name>.tracks_gt.json   contract A with exact boxes, head boxes, head above/below
  <name>.json             summary per person
  videos/<name>.mp4       the camera view (not committed)
  yolo/                   frames + labels every Nth frame, for training a sim detector (not committed)

Usage: python scenarios.py [--only NAME ...] [--yolo-every 5] [--seed 0]
"""
import argparse

import drown_sim as ds
import pool_scene as ps

OUT = ds.HERE.parents[1] / "data" / "sim_scenarios"

# name: (seconds, cast of (motion, x, y, heading deg), what the event engine should do)
SCENARIOS = {
    "baseline": (6, ps.DEFAULT_CAST,
                 "collapse -> red; everyone else green"),
    "resurface": (10, [("dive_resurface", 3.5, -0.5, 0), ("normal_swim", 0.0, 1.5, 0),
                       ("stand", -1.0, -1.5, 0), ("float", 6.5, 1.2, 0)],
                  "diver goes under about 5 s and comes up about 2 m away -> yellow, then merged back, never red"),
    "crossing": (8, [("normal_swim", -1.0, 0.25, 0), ("normal_swim", 2.2, -0.25, 180),
                     ("stand", 0.5, 1.6, 0), ("stand", 1.0, 1.6, 0)],
                 "two swimmers pass each other, two standers overlap -> IDs must not swap, all green"),
    "silent_sink_busy": (10, [("silent_sink", 6.0, -0.5, 0), ("normal_swim", -1.5, -1.5, 0),
                              ("stand", 0.0, 1.5, 0), ("float", 4.5, 1.5, 0), ("duck_under", 1.8, -0.8, 0)],
                         "silent sink in the deep end -> red; duck_under is under about 3 s -> yellow at most"),
    "entry": (6, [("fall_in", 1.0, 2.8, -90), ("normal_swim", -1.0, -1.0, 0), ("float", 5.0, 0.0, 0)],
              "person falls in from the deck, goes under briefly, then floats -> entry alert, short yellow at most"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--yolo-every", type=int, default=5, help="save a training frame every N frames (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for name, (seconds, cast, expect) in SCENARIOS.items():
        if args.only and name not in args.only:
            continue
        s = ps.run(cast, seconds, seed=args.seed, out_dir=OUT, name=name, gt=True, yolo_every=args.yolo_every)
        print(f"\n== {name} ({seconds} s): expect {expect}")
        for p in s["people"]:
            print(f"  p{p['person']} {p['motion']:15s} head z {p['head_z_start']:+.2f} -> {p['head_z_end']:+.2f}"
                  f"  under {p['pct_frames_head_underwater']:5.1f}%  first under {p['first_underwater_s']}  travel {p['travel_m']} m")


if __name__ == "__main__":
    main()
