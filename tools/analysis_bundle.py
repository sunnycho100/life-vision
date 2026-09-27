"""Share finished analyses (video + stored detections and tracks) so another machine can play them back
without re-running detection.

A job's manifest stores absolute paths (video, worker Python, model manifest) and the job fingerprint is a
hash over them, so copying artifacts/poolside/ to another machine makes the server start a fresh analysis.
`import` rewrites those fields for this machine and recomputes the fingerprint the same way JobManager.submit
does, so pressing Analyze on the same video returns the stored job.

Usage:
  .venv/bin/python -m tools.analysis_bundle export <job id> [<job id> ...] -o bundle.zip
  .venv/bin/python -m tools.analysis_bundle import bundle.zip

Run `import` with the same Python that runs backend/serve.py, and start the server with the same --backend and
--provider the jobs were made with (printed on import). The stored results are only reused if backend/detector.py,
worker.py, common.py and tracking.py are unchanged; otherwise the jobs still show up but Analyze starts over.
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import zipfile
from pathlib import Path

from backend.common import ROOT, code_digest, digest, read_json, write_json
from backend.detector import load_manifest

DATA = ROOT / "artifacts/poolside"
MODEL = ROOT / "artifacts/models/rfdetr-s-person-v1/manifest.json"
PIPELINE_FILES = ["detector.py", "worker.py", "common.py", "tracking.py"]
MACHINE_KEYS = {"job_id", "fingerprint", "model_manifest_path", "runtime"}


def fingerprint(spec):
    spec = {k: v for k, v in spec.items() if k not in MACHINE_KEYS}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def export(job_ids, out, data):
    jobs = [read_json(data / "jobs" / j / "manifest.json") for j in job_ids]
    for spec in jobs:
        state = read_json(data / "jobs" / spec["job_id"] / "status.json")["state"]
        if state != "completed":
            sys.exit(f"job {spec['job_id']} is {state}, only completed jobs can be exported")
    if len({s["model_manifest_sha256"] for s in jobs}) > 1:
        sys.exit("jobs use different model manifests, export them separately")
    model = Path(jobs[0]["model_manifest_path"])
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("bundle.json", json.dumps({"schema": "analysis-bundle/1", "jobs": job_ids}, indent=2))
        z.write(model, "model_manifest.json")
        for sid in sorted({s["source_id"] for s in jobs}):
            z.write(data / "sources" / f"{sid}.json", f"sources/{sid}.json")
            video = Path(read_json(data / "sources" / f"{sid}.json")["path"])
            z.write(video, f"sources/{sid}{video.suffix}", compress_type=zipfile.ZIP_STORED)
        for spec in jobs:
            job = data / "jobs" / spec["job_id"]
            db = sqlite3.connect(job / "observations.sqlite3")
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); db.close()
            for name in ["manifest.json", "status.json", "observations.sqlite3"]:
                z.write(job / name, f"jobs/{spec['job_id']}/{name}", compress_type=zipfile.ZIP_DEFLATED)
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.0f} MB): {len(jobs)} jobs")


def import_(bundle, data, model_path, worker_python):
    data, model_path = data.resolve(), model_path.resolve()
    (data / "sources").mkdir(parents=True, exist_ok=True)
    (data / "jobs").mkdir(parents=True, exist_ok=True)
    local_code = {name: code_digest(ROOT / "backend" / name) for name in PIPELINE_FILES}
    with zipfile.ZipFile(bundle) as z:
        theirs = json.loads(z.read("model_manifest.json"))
        if not model_path.is_file():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(z.read("model_manifest.json"))
            print(f"installed model manifest at {model_path}")
        mine = load_manifest(model_path)
        if mine["checkpoint"]["sha256"] != theirs["checkpoint"]["sha256"]:
            sys.exit(f"{model_path} points to different weights than the bundle; these results would not match it")
        if not (model_path.parent / mine["checkpoint"]["file"]).is_file():
            print(f"note: {mine['checkpoint']['file']} is missing next to {model_path}; playback works, new analyses need it")

        sources = {}
        for name in z.namelist():
            if name.startswith("sources/") and not name.endswith(".json"):
                sid = Path(name).stem
                video = data / "sources" / Path(name).name
                if not video.is_file():
                    with z.open(name) as src, video.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                if digest(video) != sid:
                    sys.exit(f"{video} does not match its sha256")
                source = json.loads(z.read(f"sources/{sid}.json"))
                source["path"] = str(video)
                write_json(data / "sources" / f"{sid}.json", source)
                sources[sid] = source

        for job_id in json.loads(z.read("bundle.json"))["jobs"]:
            job = data / "jobs" / job_id
            job.mkdir(exist_ok=True)
            for name in ["status.json", "observations.sqlite3"]:
                (job / name).write_bytes(z.read(f"jobs/{job_id}/{name}"))
            spec = json.loads(z.read(f"jobs/{job_id}/manifest.json"))
            spec.update(source=sources[spec["source_id"]], worker_python=worker_python, model_manifest=mine,
                        model_manifest_sha256=digest(model_path), model_manifest_path=str(model_path))
            spec["fingerprint"] = fingerprint(spec)
            write_json(job / "manifest.json", spec)
            stale = [n for n in PIPELINE_FILES if spec["pipeline_sha256"].get(n) != local_code[n]]
            print(f"job {job_id} | {Path(spec['source']['path']).name} {spec['start']:.0f}-{spec['end']:.0f} s | "
                  f"server flags: --backend {spec['backend']}" + (f" --provider {spec['provider']}" if spec["provider"] else "")
                  + (f" | backend/{', backend/'.join(stale)} changed since, Analyze will start over" if stale else " | reusable"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("jobs", nargs="+")
    e.add_argument("-o", "--out", required=True)
    i = sub.add_parser("import")
    i.add_argument("bundle")
    i.add_argument("--model-manifest", type=Path, default=MODEL)
    i.add_argument("--worker-python", default=sys.executable, help="the Python backend/serve.py runs with")
    for p in (e, i):
        p.add_argument("--data-dir", type=Path, default=DATA)
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.jobs, args.out, args.data_dir)
    else:
        import_(args.bundle, args.data_dir, args.model_manifest, args.worker_python)


if __name__ == "__main__":
    main()
