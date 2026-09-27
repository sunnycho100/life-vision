import copy

import pytest

from backend.monitoring import process_observations


POOL = [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]
CONFIG = dict(corners=POOL, start=0, end=200)
BOX = [.4, .4, .5, .6]


def person(box=None, confidence=.9, **extra):
    return dict(bbox_xyxy_normalized=list(box or BOX), confidence=confidence, **extra)


def frame(time, detections=None, **extra):
    return {"media_time": round(time, 8), "status": "analyzed", "detections": [] if detections is None else detections, **extra}


def sequence(until, visible):
    return [frame(i / 5, [person()] if visible(i / 5) else []) for i in range(int(until * 5) + 1)]


def run(observations, **config):
    return process_observations(observations, {**CONFIG, **config})


def test_short_visibility_loss_is_not_an_incident():
    result = run(sequence(5, lambda t: t <= 1 or t >= 4))
    assert result["incidents"] == []
    assert result["frames"][-1]["tracks"][0]["track_id"] == "track-000001"
    assert result["frames"][-1]["tracks"][0]["state"] == "visible"


def test_one_prolonged_missing_event_reappearance_closes_and_ids_are_deterministic():
    observations = sequence(15, lambda t: t <= 1 or t >= 14)
    first = run(observations)
    assert first == run(copy.deepcopy(observations))
    assert len(first["incidents"]) == 1
    event = first["incidents"][0]
    assert event["kind"] == "visibility_lost" and event["evidence"] == "visibility-only"
    assert event["status"] == "pending" and event["severity"] == "urgent"
    assert event["end_time"] == 14 and event["resolution_reason"] == "person_reappeared"
    assert event["last_seen"] == 1


def test_low_confidence_can_maintain_confirmed_track_but_cannot_spawn_or_confirm():
    observations = [frame(i / 5, [person(confidence=.9 if i < 5 else .15), person([.6, .4, .7, .6], .15)]) for i in range(40)]
    result = run(observations)
    assert result["summary"]["tracks"] == 1
    assert result["incidents"] == []
    assert result["frames"][-1]["tracks"][0]["state"] == "visible"
    observations = [frame(0, [person()])] + [frame(i / 5, [person(confidence=.15)]) for i in range(1, 50)]
    assert run(observations)["incidents"] == []


def test_observed_pool_exit_does_not_start_missing_clock():
    observations = [frame(i / 5, [person([.70 + i * .01, .4, .80 + i * .01, .6])]) for i in range(20)]
    observations += [frame(i / 5) for i in range(20, 90)]
    result = run(observations)
    assert result["summary"]["tracks"] == 1
    assert result["incidents"] == []
    assert result["frames"][-1]["tracks"][0]["state"] == "outside"
    assert result["frames"][-1]["tracks"][0]["missing_seconds"] == 0


def test_timestamp_gap_does_not_count_and_final_observation_is_not_extended():
    observations = sequence(1, lambda t: True) + [frame(30), frame(30.2), frame(30.4)]
    result = run(observations)
    assert result["incidents"] == []
    assert result["frames"][-3]["quality"] == "degraded"
    assert result["frames"][-1]["tracks"][0]["missing_seconds"] == pytest.approx(.2)
    assert result["summary"]["last_media_time"] == 30.4


def test_unanalyzed_is_not_empty_and_breaks_evidence_run():
    observations = sequence(4, lambda t: t <= 1)
    observations += [dict(media_time=4.2, detections=[]), frame(4.4, status="failed")]
    observations += [frame(i / 5) for i in range(23, 41)]
    result = run(observations)
    assert result["incidents"] == []
    assert [f["quality"] for f in result["frames"][21:23]] == ["unavailable", "unavailable"]


def test_failed_status_overrides_analyzed_flag_and_does_not_mark_missing():
    observations = sequence(1, lambda t: True) + [frame(1.2, status="failed", analyzed=True)]
    last = run(observations)["frames"][-1]
    assert last["quality"] == "unavailable"
    assert last["tracks"][0]["state"] == "visible"
    assert last["tracks"][0]["last_seen"] == 1


def test_two_separate_losses_have_distinct_event_ids():
    result = run(sequence(16, lambda t: t <= 1 or 7 <= t <= 8 or t >= 15))
    assert len(result["incidents"]) == 2
    assert len({event["id"] for event in result["incidents"]}) == 2
    assert all("end_time" in event for event in result["incidents"])


def test_mass_loss_stays_degraded_until_people_recover():
    people = [person([x, .3, x + .05, .5]) for x in (.2, .4, .6)]
    observations = [frame(i / 5, people if i <= 5 or i >= 70 else []) for i in range(80)]
    result = run(observations)
    assert result["incidents"] == []
    assert result["frames"][20]["quality"] == "degraded"
    assert result["frames"][-1]["quality"] == "ok"


def test_mass_loss_can_recover_after_tracker_retirement():
    people = [person([x, .3, x + .05, .5]) for x in (.2, .4, .6)]
    observations = [frame(i / 5, people if i <= 5 or i >= 330 else []) for i in range(345)]
    result = run(observations)
    assert result["frames"][-1]["quality"] == "ok"
    assert result["incidents"] == []


def test_explicit_head_below_can_trigger_while_body_is_visible():
    result = run([frame(i / 5, [person(head_state="below" if i < 40 else "above")]) for i in range(45)])
    assert len(result["incidents"]) == 1
    event = result["incidents"][0]
    assert event["kind"] == "possible_submersion"
    assert event["evidence"] == "explicit-head-state"
    assert event["end_time"] == 8
    assert all(f["tracks"][0]["state"] == "visible" for f in result["frames"])


def test_duplicate_boxes_do_not_create_duplicate_tracks():
    result = run([frame(i / 5, [person(), person(confidence=.85)]) for i in range(10)])
    assert result["summary"]["tracks"] == 1
    assert len(result["frames"][-1]["tracks"]) == 1


def test_ambiguous_reappearance_does_not_clear_missing_identities():
    people = [person([.35, .4, .50, .6]), person([.45, .4, .60, .6])]
    observations = [frame(i / 5, people) for i in range(6)]
    observations += [frame(1.2, [person([.4, .4, .55, .6])])]
    result = run(observations)
    assert result["frames"][-1]["quality"] == "degraded"
    assert all(t["state"] == "missing" for t in result["frames"][-1]["tracks"])
    assert result["summary"]["tracks"] == 2


def test_150_people_and_streaming_sink_have_no_100_person_cap():
    people = [person([.12 + (i % 15) * .05, .12 + (i // 15) * .07,
                      .14 + (i % 15) * .05, .16 + (i // 15) * .07]) for i in range(150)]
    sink = []
    result = run([frame(i / 5, people) for i in range(5)], frame_sink=sink.append)
    assert result["frames"] == []
    assert len(sink[-1]["tracks"]) == 150 and result["summary"]["tracks"] == 150


def test_uncalibrated_and_mapping_interval_do_not_produce_incidents():
    assert run(sequence(20, lambda t: t < 1), corners=None)["incidents"] == []
    result = run(sequence(20, lambda t: t < 1), end=2)
    assert result["incidents"] == []
    assert result["frames"][-1]["quality"] == "uncalibrated"


def test_retired_track_does_not_resolve_incident():
    result = run(sequence(70, lambda t: t <= 1))
    assert len(result["incidents"]) == 1
    assert "end_time" not in result["incidents"][0]
    assert result["incidents"][0]["status"] == "pending"


@pytest.mark.parametrize("observations", [[frame(1), frame(1)], [frame(1), frame(.5)], [frame(float("nan"))]])
def test_invalid_media_timestamps_are_rejected(observations):
    with pytest.raises(ValueError):
        run(observations)
