"""Recorded-video person detection. python -m backend.serve --video PATH"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import mimetypes
import re
import uuid
from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urllib.parse import urlsplit
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from backend.common import ROOT, REFERENCE_SHA, digest, probe, read_json, write_json
from backend.jobs import JobManager
from backend.review import install_review_routes

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
DEFAULT_MODEL = ROOT / "artifacts/models/rfdetr-nano/manifest.json"
REFERENCE_URL = "https://www.youtube.com/watch?v=PuAfTA2wf7o"
REFERENCE_VIDEO = ROOT / "artifacts/reference/wave-pool-PuAfTA2wf7o.mp4"


def fetch_reference(path=REFERENCE_VIDEO):
    """Download the wave pool reference once (H.264, which the browser player needs); later runs reuse the file."""
    if path.is_file():
        return path
    from yt_dlp import YoutubeDL
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading reference recording", flush=True)
    options = {"format": "bv*[vcodec^=avc1]+ba[ext=m4a]/b[vcodec^=avc1]", "merge_output_format": "mp4",
               "outtmpl": str(path.with_suffix(".%(ext)s")), "quiet": True, "noprogress": True}
    with YoutubeDL(options) as ydl:
        ydl.download([REFERENCE_URL])
    return path


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    start: float = Field(default=0, ge=0)
    end: float | None = Field(default=None, gt=0)


def create_app(video_path=None, data_dir=None, model_manifest=DEFAULT_MODEL, backend="onnx", provider=None,
               worker_python=None, max_upload_bytes=2 * 1024**3):
    data = Path(data_dir or ROOT / "artifacts/poolside").resolve()
    sources = data / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    manager = JobManager(data / "jobs", model_manifest, backend, provider, worker_python)
    video = Path(video_path).resolve() if video_path else None

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(app.state.review.close)
        await asyncio.to_thread(manager.close)

    app = FastAPI(title="Poolside person detection", lifespan=lifespan)
    app.state.jobs = manager

    @app.middleware("http")
    async def same_origin(request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Use the app on this server's origin."}, status_code=403)
        return await call_next(request)

    def public(source):
        return {k: v for k, v in source.items() if k != "path"}

    def register(path, sha=None):
        sha = sha or digest(path)
        metadata = probe(path)
        source = {**metadata, "id": sha, "sha256": sha, "path": str(path), "url": f"/api/sources/{sha}/video",
                  "mapping_end": min(95, metadata["duration"]) if sha == REFERENCE_SHA or path == video else metadata["duration"]}
        write_json(sources / (sha + ".json"), source)
        return source

    def get_source(source_id):
        if not re.fullmatch(r"[a-f0-9]{64}", source_id):
            raise HTTPException(404, "Unknown source")
        path = sources / (source_id + ".json")
        if not path.is_file():
            raise HTTPException(404, "Unknown source")
        return read_json(path)

    @app.get("/api/health")
    def health():
        ready = Path(model_manifest).is_file()
        return {"model": "RF-DETR Nano", "model_manifest_available": ready, "backend": backend,
                "message": "Ready to start analysis" if ready else "Run tools/prepare_rfdetr.py in the reference environment first."}

    @app.get("/api/reference")
    def reference():
        if not video or not video.is_file():
            return {"available": False}
        try:
            source = register(video)
            return {"available": True, **public(source), "source": REFERENCE_URL}
        except Exception as error:
            return {"available": False, "error": str(error), "sha256": digest(video)}

    @app.get("/media/reference")
    def reference_media():
        if not video or not video.is_file():
            raise HTTPException(404, "Reference recording unavailable")
        return FileResponse(video, media_type=mimetypes.guess_type(video)[0])

    @app.post("/api/sources", status_code=201)
    async def upload(request: Request):
        kind = request.headers.get("content-type", "").split(";")[0]
        if kind not in {"video/mp4", "video/webm"}:
            raise HTTPException(415, "Upload raw MP4 or WebM bytes, with video/mp4 or video/webm Content-Type.")
        temp = sources / (uuid.uuid4().hex + ".upload")
        sha, length = hashlib.sha256(), 0
        try:
            with temp.open("wb") as stream:
                async for chunk in request.stream():
                    length += len(chunk)
                    if length > max_upload_bytes:
                        raise HTTPException(413, "Recording exceeds the configured upload limit (default 2 GiB).")
                    sha.update(chunk)
                    stream.write(chunk)
            if not length:
                raise HTTPException(400, "Empty upload")
            try:
                metadata = await asyncio.to_thread(probe, temp)
            except Exception as error:
                raise HTTPException(422, f"Invalid recording: {error}") from error
            fingerprint = sha.hexdigest()
            suffix = ".webm" if "webm" in metadata["format"] or "matroska" in metadata["format"] else ".mp4"
            target = sources / (fingerprint + suffix)
            if not target.exists():
                temp.replace(target)
            source = await asyncio.to_thread(register, target, fingerprint)
            return public(source)
        finally:
            temp.unlink(missing_ok=True)

    @app.get("/api/sources/{source_id}")
    def source_info(source_id: str):
        return public(get_source(source_id))

    @app.get("/api/sources/{source_id}/video")
    def source_video(source_id: str):
        path = Path(get_source(source_id)["path"])
        if not path.is_file():
            raise HTTPException(404, "Source file no longer exists")
        return FileResponse(path, media_type=mimetypes.guess_type(path)[0] or "video/mp4")

    @app.get("/api/sources/{source_id}/jobs")
    def source_jobs(source_id: str):
        get_source(source_id)
        jobs = (read_json(path) for path in (data / "jobs").glob("*/status.json"))
        return sorted((j for j in jobs if j.get("source_id") == source_id), key=lambda j: j["created"], reverse=True)

    @app.post("/api/jobs", status_code=202)
    def create_job(body: JobRequest):
        try:
            return manager.submit(get_source(body.source_id), body.start, body.end)
        except FileNotFoundError as error:
            raise HTTPException(503, "Model artifacts missing. Run tools/prepare_rfdetr.py first.") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        try:
            return manager.status(job_id)
        except KeyError as error:
            raise HTTPException(404, "Unknown job") from error

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            return manager.cancel(job_id)
        except KeyError as error:
            raise HTTPException(404, "Unknown job") from error

    @app.get("/api/jobs/{job_id}/observations")
    def observations(job_id: str, start: float = Query(ge=0, allow_inf_nan=False), end: float = Query(gt=0, allow_inf_nan=False)):
        try:
            return {"job_id": job_id, "observations": manager.observations(job_id, start, end)}
        except KeyError as error:
            raise HTTPException(404, "Unknown job") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    install_review_routes(app, manager, get_source, data)
    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path)
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--host", default="127.0.0.1", help="Use a trusted LAN address to share this unauthenticated prototype.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["onnx", "python"], default="onnx")
    parser.add_argument("--provider")
    parser.add_argument("--worker-python", type=Path)
    args = parser.parse_args()
    if args.video is None:
        try:
            args.video = fetch_reference()
        except Exception as error:
            print(f"Reference download failed, continuing with upload only: {error}", flush=True)
    uvicorn.run(create_app(args.video, args.data_dir, args.model_manifest, args.backend, args.provider, args.worker_python),
                host=args.host, port=args.port)

