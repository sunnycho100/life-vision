import pytest
from backend.common import read_json, write_json
from tools.evaluate_detection import measure, select, evaluate


def fixture_data():
    box = [.2, .2, .4, .4]
    frames, predictions = [], []
    for i in range(24):
        frames.append({"id": i, "split": "dev" if i < 16 else "locked", "reviewed": True,
                       "width": 1280, "height": 720, "image_sha256": str(i),
                       "pool_quad": [[0, 0], [1, 0], [1, 1], [0, 1]], "ignore_regions": [[.8, .8, 1, 1]],
                       "people": [{"bbox_xyxy_normalized": box, "in_pool": True, "tags": ["partial"]}]})
        predictions.append({"id": i, "detections": [
            {"bbox_xyxy_normalized": box, "confidence": .9},
            {"bbox_xyxy_normalized": box, "confidence": .8},
            {"bbox_xyxy_normalized": [.85, .85, .95, .95], "confidence": .85}]})
    return {"source_sha256": "source", "frames": frames}, {"source_sha256": "source", "model": "rfdetr", "frames": predictions,
                                                         "image_hashes": {str(i): str(i) for i in range(24)}}


def test_one_to_one_duplicates_ignore_and_partial_metrics():
    labels, predictions = fixture_data()
    result = measure(labels, predictions, "dev", .5)
    assert result["tp"] == 16 and result["fp"] == 16 and result["duplicates"] == 16
    assert result["ignored_predictions"] == 16 and result["precision"] == .5
    assert result["recall_groups"]["partial"] == {"matched": 16, "total": 16}


def test_unreviewed_labels_cannot_pass_accuracy_gate():
    labels, predictions = fixture_data()
    labels["frames"][0]["reviewed"] = False
    with pytest.raises(ValueError, match="manually reviewed"):
        measure(labels, predictions, "dev", .5)


def test_policy_frozen_before_locked_evaluation(tmp_path):
    labels, predictions = fixture_data()
    annotation = tmp_path / "labels.json"; candidates = tmp_path / "predictions.json"
    policy = tmp_path / "policy.json"; report = tmp_path / "report.json"
    write_json(annotation, labels); write_json(candidates, predictions)
    select(annotation, candidates, policy)
    selected = read_json(policy)
    assert selected["passed"] and selected["selected"]["threshold"] == .9
    evaluate(annotation, candidates, policy, report)
    assert read_json(report)["precision"] == 1
    with pytest.raises(ValueError, match="already exists"):
        evaluate(annotation, candidates, policy, report)
    with pytest.raises(ValueError, match="already frozen"):
        select(annotation, candidates, policy)


def test_evaluate_rejects_dev_labels_changed_after_policy_freeze(tmp_path):
    labels, predictions = fixture_data()
    annotation = tmp_path / "labels.json"; candidates = tmp_path / "predictions.json"
    policy = tmp_path / "policy.json"; report = tmp_path / "report.json"
    write_json(annotation, labels); write_json(candidates, predictions)
    select(annotation, candidates, policy)
    changed = read_json(annotation)
    changed["frames"][0]["people"][0]["in_pool"] = False
    write_json(annotation, changed)
    with pytest.raises(ValueError, match="Development labels changed"):
        evaluate(annotation, candidates, policy, report)


def test_no_positive_matches_is_a_failed_gate(tmp_path):
    labels, predictions = fixture_data()
    for frame in predictions["frames"]:
        frame["detections"] = [{"bbox_xyxy_normalized": [.6, .6, .7, .7], "confidence": .9}]
    a, p, out = (tmp_path / name for name in ["a.json", "p.json", "out.json"])
    write_json(a, labels); write_json(p, predictions)
    select(a, p, out)
    assert read_json(out)["passed"] is False
