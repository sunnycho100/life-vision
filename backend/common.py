"""Shared storage and media contracts; no ML imports."""
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SHA = "66f488b08e2122e9d8195972a481af2931b0ea6586ed229208b02a195a488d2c"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def code_digest(path):
    # Git/editor CRLF conversion must not invalidate an otherwise identical adapter.
    return hashlib.sha256(Path(path).read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def probe(path):
    import av
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream found.")
        stream = container.streams.video[0]
        frame = next(container.decode(stream), None)
        if frame is None or frame.pts is None:
            raise ValueError("No decodable, timestamped video frame.")
        if abs(float(getattr(frame, "rotation", 0))) > .1:
            raise ValueError("Rotated video is not supported yet. Export an upright MP4 first.")
        if stream.sample_aspect_ratio and float(stream.sample_aspect_ratio) != 1:
            raise ValueError("Export this video with square pixels before upload.")
        duration = float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / av.time_base
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("A recording with a finite positive duration is required.")
        if stream.width * stream.height > 3840 * 2160:
            raise ValueError("Maximum supported frame size is 3840 × 2160 pixels.")
        if stream.codec_context.name not in {"h264", "vp8", "vp9", "av1"}:
            raise ValueError("Use H.264 MP4 or VP8/VP9/AV1 WebM for browser playback.")
        return {"width": stream.width, "height": stream.height, "duration": duration,
                "codec": stream.codec_context.name, "time_base": str(stream.time_base),
                "start_pts": stream.start_time if stream.start_time is not None else frame.pts,
                "average_rate": str(stream.average_rate), "format": container.format.name,
                "decoder": {"av": av.__version__, "libraries": {k: list(v) for k, v in av.library_versions.items()}}}


def sampled_frames(path, hz=5., start=0., end=None):
    """First *distinct* decoded frame at/after each sampling target, original PTS.

    Missing intervals remain gaps: low-FPS inputs never duplicate observations.
    media_time is relative to the video stream start (verified against the browser).
    """
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = stream.start_time
        target_index = 0
        previous = -math.inf
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError("Decoded frame has no presentation timestamp.")
            if origin is None:
                origin = frame.pts
            media_time = float((frame.pts - origin) * stream.time_base)
            if media_time < previous:
                raise ValueError("Non-monotonic video timestamps.")
            previous = media_time
            if end is not None and media_time >= end:
                break
            if media_time + 1e-7 < start + target_index / hz:
                continue
            yield index, frame.pts, str(stream.time_base), media_time, frame.to_ndarray(format="rgb24")
            target_index = max(target_index + 1, math.floor((media_time - start) * hz + 1e-7) + 1)
