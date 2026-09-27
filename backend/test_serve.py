import hashlib
from fastapi.testclient import TestClient
from backend.serve import create_app


def test_reference_ranges_and_fingerprint(tmp_path):
    content = bytes(range(256)) * 8
    video = tmp_path / "reference.mp4"
    video.write_bytes(content)
    client = TestClient(create_app(video, data_dir=tmp_path / "storage"))
    info = client.get("/api/reference").json()
    assert info["sha256"] == hashlib.sha256(content).hexdigest()
    full = client.get("/media/reference")
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"
    partial = client.get("/media/reference", headers={"Range": "bytes=20-119"})
    assert partial.status_code == 206
    assert partial.headers["content-range"] == "bytes 20-119/2048"
    assert partial.content == content[20:120]
    assert client.get("/media/reference", headers={"Range": "bytes=-100"}).content == content[-100:]
    assert client.get("/media/reference", headers={"Range": "bytes=9999-"}).status_code == 416


def test_missing_source_and_module_mime(tmp_path):
    client = TestClient(create_app(data_dir=tmp_path / "storage"))
    assert not client.get("/api/reference").json()["available"]
    assert client.get("/media/reference").status_code == 404
    assert client.get("/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/core.mjs").headers["content-type"].startswith("text/javascript")


def test_webm_mime(tmp_path):
    video = tmp_path / "reference.webm"
    video.write_bytes(b"test fixture")
    assert TestClient(create_app(video, data_dir=tmp_path / "storage")).get("/media/reference").headers["content-type"] == "video/webm"


def test_extra_videos_are_listed_as_presets(tmp_path):
    import subprocess
    clip = tmp_path / "rescue.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", str(clip)], check=True)
    client = TestClient(create_app(None, data_dir=tmp_path / "storage", extra_videos=[(clip, "Lifeguard rescue")]))
    [preset] = client.get("/api/presets").json()
    assert preset["label"] == "Lifeguard rescue" and preset["mapping_end"] == preset["duration"]
    assert client.get(preset["url"]).content == clip.read_bytes()
