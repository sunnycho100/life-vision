"""Safety of label gates, split isolation and real media/COCO conversion."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from backend.common import digest, read_json, write_json
from tools.rescue_dataset import (SCHEMA, prepare, validate_manifest, split_fingerprint,
                                  export_coco, export_temporal, contained)


def fixture(tmp_path):
    sources, frames = [], []
    for i, split in enumerate(("train", "valid", "test")):
        video = tmp_path / f"source-{i}.mp4"; video.write_bytes(bytes([i]))
        image = tmp_path / f"image-{i}.png"; Image.new("RGB", (100, 50), (i*50, 0, 0)).save(image)
        s = {"id": str(i), "sha256": digest(video), "filename": video.name, "duration": 10., "split": split,
             "review": {"reviewed": True, "reviewer": "Test reviewer", "same_event_group": str(i),
                        "rights_status": "unverified", "notes": "", "intervals": [], "target_keyframes": []}}
        sources.append(s)
        frames.append({"id": str(i), "source_id": str(i), "source_sha256": s["sha256"], "split": split,
                       "image": image.name, "image_sha256": digest(image), "width": 100, "height": 50, "time": 1.,
                       "reviewed": True, "reviewer": "Test reviewer", "ignore_regions": [],
                       "people": [{"bbox_xyxy_normalized": [.1, .2, .5, .8]}]})
        s["review"]["notes"] = "No event labels entered in this test"
    m = {"schema": SCHEMA, "manifest_id": hashlib.sha256(json.dumps(sorted(s["sha256"] for s in sources)).encode()).hexdigest(),
         "raw_directory": str(tmp_path), "sources": sources, "split_reviewed": True, "split_reviewer": "Test reviewer"}
    m["split_fingerprint"] = split_fingerprint(m)
    labels = {"schema": "pool-person-labels/1", "dataset_id": m["manifest_id"], "frames": frames}
    write_json(tmp_path / "manifest.json", m); write_json(tmp_path / "annotations.json", labels)
    return m, labels


def test_coco_real_dimensions_and_mapping(tmp_path):
    m, labels = fixture(tmp_path)
    assert export_coco(tmp_path / "annotations.json", tmp_path / "manifest.json", tmp_path / "coco") == 3
    data = read_json(tmp_path / "coco/train/_annotations.coco.json")
    assert data["annotations"][0]["bbox"] == pytest.approx([10, 10, 40, 30])
    assert data["annotations"][0]["category_id"] == 0
    assert data["categories"][0]["name"] == "person"


@pytest.mark.parametrize("field,value", [("reviewed", False), ("reviewer", ""), ("ignore_regions", [[.1,.1,.9,.9]])])
def test_unreviewed_or_ignored_frame_is_not_background(tmp_path, field, value):
    _, labels = fixture(tmp_path); labels["frames"][0][field] = value
    write_json(tmp_path / "annotations.json", labels)
    with pytest.raises(ValueError, match="Need reviewed frames in train"):
        export_coco(tmp_path / "annotations.json", tmp_path / "manifest.json", tmp_path / "coco")
    assert not (tmp_path / "coco").exists()


def test_group_leakage_and_changed_approval(tmp_path):
    m, _ = fixture(tmp_path)
    m["sources"][1]["review"]["same_event_group"] = "0"
    with pytest.raises(ValueError, match="crosses splits"):
        validate_manifest(m)
    m["sources"][1]["review"]["same_event_group"] = "new-group"
    with pytest.raises(ValueError, match="approval"):
        validate_manifest(m, require_split=True)


def test_duplicate_source_and_image_tamper(tmp_path):
    m, _ = fixture(tmp_path)
    m["sources"].append(copy.deepcopy(m["sources"][0]))
    with pytest.raises(ValueError, match="Duplicate source"):
        validate_manifest(m)
    (tmp_path / "image-0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        export_coco(tmp_path / "annotations.json", tmp_path / "manifest.json", tmp_path / "coco")


def test_temporal_evidence_and_uncertainty(tmp_path):
    m, _ = fixture(tmp_path)
    review = m["sources"][0]["review"]
    review["rescuer_entry"] = 3.5
    review["intervals"] = [{"start": 1., "end": 3., "label": "visible_distress", "target_id": "A"},
                           {"start": 3., "end": 4., "label": "not_visible", "target_id": "A"},
                           {"start": 4., "end": 5., "label": "rescue", "target_id": "A"}]
    review["target_keyframes"] = [{"media_time": 2., "target_id": "A", "bbox_xyxy_normalized": [.1,.2,.3,.4]}]
    write_json(tmp_path / "manifest.json", m)
    assert export_temporal(tmp_path / "manifest.json", tmp_path / "temporal.jsonl") == 3
    rows = [json.loads(s) for s in (tmp_path / "temporal.jsonl").read_text().splitlines()]
    assert [r["eligible_for_distress_research"] for r in rows] == [True, False, False]
    assert all(r["clinical_drowning_confirmed"] is None for r in rows)
    assert "overlaps_rescue_response" in rows[2]["exclusion_reasons"]
    review["intervals"][1]["start"] = 2.
    with pytest.raises(ValueError, match="Overlapping"):
        validate_manifest(m)


def test_no_event_labels_no_export(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError, match="No human-reviewed"):
        export_temporal(tmp_path / "manifest.json", tmp_path / "temporal.jsonl")


def test_path_escape():
    with pytest.raises(ValueError, match="escapes"):
        contained(Path.cwd(), "../outside")


def test_prepare_decodes_actual_pts_and_errors(tmp_path):
    import av
    import numpy as np
    from fractions import Fraction
    raw = tmp_path / "raw"; raw.mkdir()
    with av.open(str(raw / "real.mp4"), "w") as c:
        stream = c.add_stream("libx264", rate=10); stream.width=64; stream.height=48; stream.pix_fmt="yuv420p"
        stream.time_base = Fraction(1, 1000)
        for i, ms in enumerate([0, 100, 300, 400, 700, 800, 1000]):
            frame = av.VideoFrame.from_ndarray(np.full((48,64,3), i*30, dtype=np.uint8), format="rgb24")
            frame.pts=ms; frame.time_base=Fraction(1,1000)
            for packet in stream.encode(frame): c.mux(packet)
        for packet in stream.encode(): c.mux(packet)
    (raw / "corrupt.mp4").write_bytes(b"not a video")
    result = prepare(raw, tmp_path / "bundle", 3)
    assert len(result["sources"]) == 1 and len(result["errors"]) == 1
    labels = read_json(tmp_path / "bundle/annotations.json")
    times = [f["time"] for f in labels["frames"]]
    assert times == pytest.approx([.3, .7, 1.0])
    assert all(not f["reviewed"] for f in labels["frames"])
    assert result["split_reviewed"] is False


def test_proposals_cannot_migrate_into_heldout_set(tmp_path):
    _, labels = fixture(tmp_path)
    labels["frames"][1]["suggestions"] = {"human_verified": False}
    write_json(tmp_path / "annotations.json", labels)
    with pytest.raises(ValueError, match="independently labeled"):
        export_coco(tmp_path / "annotations.json", tmp_path / "manifest.json", tmp_path / "coco")


def test_unknown_rescue_entry_excludes_behavior_training(tmp_path):
    m, _ = fixture(tmp_path)
    review = m["sources"][0]["review"]
    review["intervals"] = [{"start": 1., "end": 3., "label": "visible_distress", "target_id": "A"}]
    review["target_keyframes"] = [{"media_time": 2., "target_id": "A", "bbox_xyxy_normalized": [.1,.2,.3,.4]}]
    write_json(tmp_path / "manifest.json", m)
    export_temporal(tmp_path / "manifest.json", tmp_path / "temporal.jsonl")
    row = json.loads((tmp_path / "temporal.jsonl").read_text())
    assert not row["eligible_for_distress_research"]
    assert row["exclusion_reasons"] == ["rescuer_entry_unknown"]
