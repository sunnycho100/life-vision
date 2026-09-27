"""Re-run only the tracker over a finished job's stored detections and rewrite its tracks.

Detection is the slow part (about 0.4 s per frame with tiling); tracking takes seconds. Use this after
changing backend/tracking.py, then reload the review page.

Usage: .venv/bin/python -m tools.retrack_job artifacts/poolside-tracking/jobs/<job id>
"""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from backend.common import read_json, sampled_frames
from backend.tracking import PoolTracker, is_scene_cut


def main(directory):
    directory = Path(directory)
    spec = read_json(directory / "manifest.json")
    db = sqlite3.connect(directory / "observations.sqlite3")
    rows = [(t, json.loads(p)) for t, p in db.execute("SELECT media_time, payload FROM observations ORDER BY media_time")]
    tracker = PoolTracker(spec["sample_hz"], **spec.get("tracking", {}))
    if not any("scene_cut" in obs for _, obs in rows):  # older jobs: find cuts from the video itself
        prev, cuts = None, set()
        for _, _, _, t, rgb in sampled_frames(spec["source"]["path"], spec["sample_hz"], spec["start"], spec["end"]):
            if is_scene_cut(prev, rgb):
                cuts.add(round(t, 3))
            prev = rgb
        for _, obs in rows:
            obs["scene_cut"] = round(obs["media_time"], 3) in cuts
        print("scene cuts at", sorted(cuts))
    levels, ids, dups = Counter(), set(), 0
    for t, obs in rows:
        if obs.get("scene_cut"):
            tracker.scene_cut()
        obs["tracks"] = tracker.update(obs["media_time"], obs["detections"])
        pids = [p["person_id"] for p in obs["tracks"]]
        dups += len(pids) != len(set(pids)); ids |= set(pids)
        levels.update(p["level"] for p in obs["tracks"] if not p["visible"])
        db.execute("UPDATE observations SET payload = ? WHERE media_time = ?", (json.dumps(obs, allow_nan=False), t))
    db.commit(); db.close()
    print(f"{len(rows)} frames | person IDs {len(ids)} | frames with duplicate IDs {dups} | held boxes by level {dict(levels)}")


if __name__ == "__main__":
    main(sys.argv[1])
