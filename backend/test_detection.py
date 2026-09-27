from fractions import Fraction
import json
import sqlite3
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.common import digest, probe, sampled_frames, read_json, write_json
from backend.detector import preprocess, decode
from backend.jobs import JobManager, observation_db
from backend.serve import create_app


def make_video(path, pts=None):
    import av
    pts = pts or list(range(0, 1000, 40))
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height = 160, 96
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.options = {"bf": "0"}
        for i, pts_value in enumerate(pts):
            image = np.zeros((96, 160, 3), dtype=np.uint8)
            image[:, :, 0] = i * 5 % 256
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = pts_value, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_preprocess_normalization_channel_order_and_half_pixel():
    rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    result = preprocess(rgb, 1)
    expected = (np.full(3, .5) - [.485, .456, .406]) / [.229, .224, .225]
    np.testing.assert_allclose(result[0, :, 0, 0], expected, atol=1e-6)
    assert result.dtype == np.float32 and result.shape == (1, 3, 1, 1)


def test_decoder_keeps_150_people_without_nms_or_yolo_class_zero():
    boxes = np.tile([.5, .5, .1, .1], (300, 1)).astype(np.float32)
    logits = np.full((300, 91), -20., dtype=np.float32)
    logits[:150, 1] = 8
    logits[150:, 0] = 10
    result = decode(boxes, logits, .5)
    assert len(result) == 150
    assert all(d["class_name"] == "person" for d in result)
    # One query can score on multiple classes. Rank pairs before filtering persons.
    logits[:, 2] = 12
    assert decode(boxes, logits, .5) == []


def test_decoder_sparse_coco_does_not_drop_final_class_before_topk():
    boxes = np.array([[.5, .5, 1, 1]], dtype=np.float32)
    logits = np.full((1, 91), -20., dtype=np.float32)
    logits[0, 90], logits[0, 1] = 10, 8
    assert decode(boxes, logits, .5, num_select=1) == []


def test_vfr_sampling_uses_original_pts_and_never_duplicates(tmp_path):
    video = make_video(tmp_path / "vfr.mp4", [1000, 1033, 1100, 1250, 1460, 1500, 1710, 1840])
    info = probe(video)
    frames = list(sampled_frames(video))
    assert len({f[0] for f in frames}) == len(frames)
    assert [round(f[3], 3) for f in frames] == [0, .25, .46, .71, .84]
    assert info["width"] == 160
    selected = list(sampled_frames(video, start=.2, end=.7))
    assert [round(f[3], 3) for f in selected] == [.25, .46]


def test_upload_hash_range_validation_and_origin(tmp_path):
    video = make_video(tmp_path / "clip.mp4")
    content = video.read_bytes()
    with TestClient(create_app(data_dir=tmp_path / "storage", max_upload_bytes=len(content) + 1)) as client:
        response = client.post("/api/sources", content=content, headers={"Content-Type": "video/mp4"})
        assert response.status_code == 201, response.text
        source = response.json()
        assert source["sha256"] == digest(video) and "path" not in source
        assert client.get(source["url"], headers={"Range": "bytes=0-99"}).content == content[:100]
        duplicate = client.post("/api/sources", content=content, headers={"Content-Type": "video/mp4"}).json()
        assert duplicate["id"] == source["id"]
        assert client.post("/api/sources", content=b"broken", headers={"Content-Type": "video/mp4"}).status_code == 422
        assert client.post("/api/sources", content=content * 2, headers={"Content-Type": "video/mp4"}).status_code == 413
        assert client.post("/api/sources", content=b"", headers={"Content-Type": "video/mp4"}).status_code == 400
        assert client.post("/api/jobs", json={"source_id": source["id"], "start": 0, "end": 1}, headers={"Origin": "https://unrelated.example"}).status_code == 403
        assert client.get("/api/jobs/not-a-job").status_code == 404
        assert not list((tmp_path / "storage/sources").glob("*.upload"))


def manifest_fixture(path):
    write_json(path, {"model": "RFDETRNano", "rfdetr_version": "1.11.0", "resolution": 384,
                      "precision": "fp32", "person_id": 1, "background_id": None, "num_select": 300,
                      "parity": {"passed": True}})


def test_job_single_worker_cancel_and_restart(tmp_path, monkeypatch):
    manifest = tmp_path / "model.json"; manifest_fixture(manifest)
    class Process:
        returncode = None
        def __init__(self, *args, **kwargs): pass
        def poll(self): return self.returncode
        def terminate(self): self.returncode = -15
        def wait(self, timeout): return self.returncode
    monkeypatch.setattr("backend.jobs.subprocess.Popen", Process)
    manager = JobManager(tmp_path / "jobs", manifest)
    source = {"id": "a" * 64, "sha256": "a" * 64, "duration": 20}
    first = manager.submit(source, 0, 2)
    assert manager.submit(source, 0, 2)["id"] == first["id"]
    with pytest.raises(RuntimeError): manager.submit(source, 2, 4)
    assert manager.cancel(first["id"])["state"] == "cancelled"
    second = manager.submit(source, 2, 4)
    manager.close()
    path = manager.path(second["id"]) / "status.json"
    status = read_json(path); status["state"] = "running"; write_json(path, status)
    restarted = JobManager(tmp_path / "jobs", manifest)
    assert restarted.status(second["id"])["state"] == "interrupted"


def test_observation_window_empty_is_not_missing_and_more_than_100(tmp_path):
    manager = JobManager(tmp_path / "jobs", tmp_path / "absent.json")
    directory = manager.root / ("a" * 32); directory.mkdir()
    write_json(directory / "status.json", {"state": "completed"})
    db = observation_db(directory / "observations.sqlite3")
    for t, count in [(1., 150), (1.2, 0), (40., 2)]:
        db.execute("INSERT INTO observations VALUES (?,?)", (t, json.dumps({"media_time": t, "status": "analyzed", "detections": [{}] * count})))
    db.commit(); db.close()
    rows = manager.observations("a" * 32, 1, 2)
    assert [len(row["detections"]) for row in rows] == [150, 0]
    assert manager.observations("a" * 32, 2, 3) == []
    with pytest.raises(ValueError): manager.observations("a" * 32, 0, 60)
