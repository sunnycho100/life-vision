"""Single-worker scheduler and indexed, bounded observation reads."""
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from backend.common import ROOT, digest, read_json, write_json, code_digest
from backend.detector import load_manifest


def observation_db(path):
    db = sqlite3.connect(path, timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS observations (media_time REAL PRIMARY KEY, payload TEXT NOT NULL)")
    return db


class JobManager:
    def __init__(self, root, manifest_path, backend="onnx", provider=None, worker_python=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(manifest_path).resolve()
        self.backend, self.provider = backend, provider
        self.worker_python = str(worker_python or sys.executable)
        self.lock = threading.RLock()
        self.process = None
        self.active_id = None
        for path in self.root.glob("*/status.json"):
            status = read_json(path)
            if status["state"] in {"starting", "running"}:
                status.update(state="interrupted", incomplete=True, error="Server stopped before this job completed.")
                write_json(path, status)

    def path(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise KeyError("Unknown job")
        path = self.root / job_id
        if not (path / "status.json").is_file():
            raise KeyError("Unknown job")
        return path

    def reap(self):
        if self.process is not None and self.process.poll() is not None:
            path = self.path(self.active_id)
            state = read_json(path / "status.json")
            if state["state"] in {"starting", "running"}:
                state.update(state="failed", incomplete=True, error=f"Analysis process exited ({self.process.returncode}). Check its worker.log.")
                write_json(path / "status.json", state)
            self.process = None
            self.active_id = None

    def submit(self, source, start=0., end=None):
        with self.lock:
            self.reap()
            model = load_manifest(self.manifest_path)
            if self.backend == "onnx" and not model.get("parity", {}).get("passed"):
                raise ValueError("ONNX reference parity has not passed. Use --backend python or run tools/prepare_rfdetr.py.")
            end = source["duration"] if end is None else end
            if not 0 <= start < end <= source["duration"] + .001:
                raise ValueError("Analysis interval must fall within this recording.")
            spec = {"schema": "detections/1", "source_id": source["id"], "source_sha256": source["sha256"],
                    "source": source, "start": start, "end": min(end, source["duration"]), "sample_hz": 5,
                    "sampling": "first distinct frame at/after target; source PTS minus stream start",
                    "stored_threshold": .1, "display_threshold": .5, "tiling": False,
                    "model_manifest": model, "model_manifest_sha256": digest(self.manifest_path),
                    "backend": self.backend, "provider": self.provider, "pipeline_version": 2,
                    "worker_python": self.worker_python}
            spec["pipeline_sha256"] = {name: code_digest(ROOT / "backend" / name) for name in ["detector.py", "worker.py", "common.py"]}
            fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            for path in self.root.glob("*/manifest.json"):
                old = read_json(path)
                if old.get("fingerprint") == fingerprint:
                    status = read_json(path.with_name("status.json"))
                    if status["state"] in {"completed", "running", "starting"}:
                        return status
            if self.process is not None:
                raise RuntimeError("Another analysis is running. Cancel it or wait for completion.")
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir()
            spec.update(job_id=job_id, fingerprint=fingerprint, model_manifest_path=str(self.manifest_path))
            write_json(directory / "manifest.json", spec)
            status = {"id": job_id, "source_id": source["id"], "state": "starting", "progress": 0,
                      "frames_analyzed": 0, "incomplete": True, "created": time.time(), "model": "RF-DETR Nano",
                      "backend": self.backend, "start": start, "end": end, "stored_threshold": .1}
            write_json(directory / "status.json", status)
            observation_db(directory / "observations.sqlite3").close()
            try:
                with (directory / "worker.log").open("wb") as log:
                    self.process = subprocess.Popen([self.worker_python, "-m", "backend.worker", str(directory)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except OSError as error:
                status.update(state="failed", error=str(error))
                write_json(directory / "status.json", status)
                raise ValueError("Could not start the configured analysis Python executable.") from error
            self.active_id = job_id
            return status

    def status(self, job_id):
        with self.lock:
            self.reap()
            return read_json(self.path(job_id) / "status.json")

    def cancel(self, job_id):
        with self.lock:
            self.reap()
            path = self.path(job_id)
            if self.active_id == job_id and self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                self.process = None
                self.active_id = None
                status = read_json(path / "status.json")
                status.update(state="cancelled", incomplete=True)
                write_json(path / "status.json", status)
            return read_json(path / "status.json")

    def observations(self, job_id, start, end):
        if not 0 <= start < end or end - start > 30:
            raise ValueError("Request an observation window of at most 30 seconds.")
        path = self.path(job_id)
        with sqlite3.connect(path / "observations.sqlite3", timeout=10) as db:
            rows = db.execute("SELECT payload FROM observations WHERE media_time >= ? AND media_time <= ? ORDER BY media_time",
                              (max(0, start - .3), end)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def close(self):
        with self.lock:
            if self.active_id:
                self.cancel(self.active_id)
