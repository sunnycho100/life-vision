"""Disposable analysis process. Never decodes media in the HTTP server process."""
import json
import sys
import time
import traceback
from pathlib import Path
from backend.common import digest, read_json, sampled_frames, write_json
from backend.detector import create_detector
from backend.jobs import observation_db


def run(directory):
    directory = Path(directory)
    spec = read_json(directory / "manifest.json")
    status = read_json(directory / "status.json")
    started = time.perf_counter()
    db = observation_db(directory / "observations.sqlite3")
    last_time = None
    try:
        source = spec["source"]
        if digest(source["path"]) != spec["source_sha256"]:
            raise ValueError("Source file changed after registration.")
        detector = create_detector(spec["model_manifest_path"], spec["backend"], spec["provider"])
        spec["runtime"] = detector.runtime
        write_json(directory / "manifest.json", spec)
        status.update(state="running", runtime=detector.runtime)
        write_json(directory / "status.json", status)
        durations = []
        for index, pts, base, t, rgb in sampled_frames(source["path"], spec["sample_hz"], spec["start"], spec["end"]):
            before = time.perf_counter()
            detections = detector.predict(rgb, spec["stored_threshold"])
            elapsed = (time.perf_counter() - before) * 1000
            durations.append(elapsed)
            observation = {"schema": "detections/1", "source_id": spec["source_id"], "job_id": spec["job_id"],
                           "frame_index": index, "pts": pts, "time_base": base, "media_time": t,
                           "status": "analyzed", "detections": detections, "inference_ms": elapsed}
            db.execute("INSERT INTO observations VALUES (?, ?)", (t, json.dumps(observation, allow_nan=False)))
            db.commit()
            last_time = t
            status.update(frames_analyzed=len(durations), last_media_time=t, inference_ms=elapsed,
                          progress=min(.999, (t - spec["start"]) / (spec["end"] - spec["start"])),
                          elapsed_seconds=time.perf_counter() - started)
            write_json(directory / "status.json", status)
        if not durations:
            raise ValueError("No timestamped frames in the requested interval.")
        elapsed = time.perf_counter() - started
        import numpy as np
        status.update(state="completed", incomplete=False, progress=1, elapsed_seconds=elapsed,
                      inference_ms_median=float(np.median(durations)), inference_ms_p95=float(np.percentile(durations, 95)),
                      processing_video_seconds_per_wall_second=(spec["end"] - spec["start"]) / elapsed)
    except Exception as error:
        traceback.print_exc()
        status.update(state="failed", incomplete=True, error=f"{type(error).__name__}: {error}", last_media_time=last_time)
    finally:
        db.close()
        write_json(directory / "status.json", status)


if __name__ == "__main__":
    run(sys.argv[1])
