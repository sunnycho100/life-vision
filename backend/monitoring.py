"""Deterministic recorded-media visibility review, not a drowning classifier.

Association is a conservative constant-velocity / IoU baseline, not ByteTrack.
High-score detections associate first; low scores only maintain confirmed tracks.
Ambiguous overlaps freeze evidence clocks. IDs are local to one recording/config.
"""
import hashlib
import math


DEFAULTS = dict(low_threshold=.10, high_threshold=.25, new_track_threshold=.25,
                min_confirm_seconds=.6, warn_seconds=5., urgent_seconds=12.,
                max_gap_seconds=.6, retire_seconds=60.)


def _center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def _iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union else 0.


def _quad(corners):
    if corners is None:
        return None
    if len(corners) != 4 or any(len(p) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in corners):
        raise ValueError("Pool corners must be four normalized points.")
    crosses = []
    for i in range(4):
        a, b, c = corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not all(v > 1e-8 for v in crosses):
        raise ValueError("Pool corners must form a convex clockwise quadrilateral in image coordinates.")
    return corners


def _inside(box, corners):
    x, y = _center(box)
    return all((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) >= -1e-9
               for a, b in zip(corners, corners[1:] + corners[:1]))


def _detections(raw, threshold):
    output = []
    for d in raw:
        box, score = d.get("bbox_xyxy_normalized"), d.get("confidence")
        if (box is None or len(box) != 4 or score is None or not math.isfinite(score)
                or not 0 <= score <= 1 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError("Invalid normalized detection.")
        if score >= threshold and d.get("class_name", "person") == "person":
            output.append(dict(d, bbox_xyxy_normalized=list(box)))
    output.sort(key=lambda d: (-d["confidence"], *d["bbox_xyxy_normalized"]))
    # RF query duplicates must not manufacture extra identities. This is only
    # near-identical box suppression, not a claim to full occlusion handling.
    unique = []
    for d in output:
        if not any(_iou(d["bbox_xyxy_normalized"], old["bbox_xyxy_normalized"]) > .95 for old in unique):
            unique.append(d)
    return unique


def _associate(tracks, detections, time, cfg):
    matches, used, ambiguous = {}, set(), set()
    blocked_tracks = set()
    for high in (True, False):
        edges = []
        for key, tr in tracks.items():
            if key in matches or key in blocked_tracks or (not high and not tr["confirmed"]):
                continue
            elapsed = time - tr["last_seen"]
            if elapsed > cfg["retire_seconds"]:
                continue
            box = tr["box"]
            if elapsed <= cfg["max_gap_seconds"]:
                dx, dy = [v * elapsed for v in tr["velocity"]]
                predicted = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
                gate = .15
            else:
                predicted, gate = box, .65
            for index, d in enumerate(detections):
                if index in used or index in ambiguous or (d["confidence"] >= cfg["high_threshold"]) != high:
                    continue
                score = _iou(predicted, d["bbox_xyxy_normalized"])
                if score >= gate:
                    edges.append((score, key, index))
        # A near tie in either direction is unresolved identity evidence. Do not
        # silently let greedy matching choose which missing swimmer reappeared.
        by_track, by_detection = {}, {}
        for edge in edges:
            by_track.setdefault(edge[1], []).append(edge)
            by_detection.setdefault(edge[2], []).append(edge)
        for group in [*by_track.values(), *by_detection.values()]:
            group.sort(reverse=True)
            if len(group) > 1 and group[0][0] - group[1][0] < .08:
                for score, key, index in group:
                    if group[0][0] - score < .08:
                        blocked_tracks.add(key)
                        ambiguous.add(index)
        for score, key, index in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
            if key not in matches and key not in blocked_tracks and index not in used and index not in ambiguous:
                matches[key] = index
                used.add(index)
    return matches, used, ambiguous


def process_observations(observations, config):
    """Consume strictly increasing media timestamps; never extrapolate the tail.

    A frame is analyzed only with status='analyzed' or analyzed=True and an
    explicit detections list (empty is valid). Invalid calibration is rejected;
    absent calibration yields uncalibrated frames without incidents.
    Optional frame_sink(frame) streams frames instead of retaining them.
    """
    cfg = {**DEFAULTS, **config}
    for key in DEFAULTS:
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}.")
    if not 0 <= cfg["low_threshold"] <= cfg["high_threshold"] <= cfg["new_track_threshold"] <= 1:
        raise ValueError("Confidence thresholds must be ordered between zero and one.")
    if cfg["urgent_seconds"] < cfg["warn_seconds"] or cfg["max_gap_seconds"] <= 0 or cfg["retire_seconds"] < cfg["max_gap_seconds"]:
        raise ValueError("Invalid timing thresholds.")
    corners = _quad(cfg.get("corners"))
    start, end = cfg.get("start", 0.), cfg.get("end", float("inf"))
    if not math.isfinite(start) or start < 0 or end <= start or math.isnan(end):
        raise ValueError("Invalid mapping interval.")
    tracks, incidents, frames = {}, [], []
    previous_time, previous_quality, next_id = None, None, 1
    mass_loss, mass_baseline, recovery_frames = False, 0, 0
    counts = dict(ok=0, unavailable=0, degraded=0, uncalibrated=0)
    total_tracks = 0
    sink = cfg.get("frame_sink", frames.append)

    def close(tr, time, reason):
        if tr["event"] is not None:
            tr["event"].update(end_time=time, resolution_reason=reason)
            tr["event"] = None
        tr["loss_start"], tr["missing_seconds"], tr["below_seconds"], tr["below_start"] = None, 0., 0., None

    for observation in observations:
        time = observation.get("media_time")
        if not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0 or (previous_time is not None and time <= previous_time):
            raise ValueError("Observations require strictly increasing finite media timestamps.")
        dt = 0. if previous_time is None else time - previous_time
        signal = observation.get("status") == "analyzed" if "status" in observation else observation.get("analyzed") is True
        analyzed = signal and isinstance(observation.get("detections"), list)
        quality = "ok" if analyzed else "unavailable"
        if analyzed and (observation.get("quality", "ok") != "ok" or dt > cfg["max_gap_seconds"] + 1e-8):
            quality = "degraded"
        if corners is None or not start <= time <= end:
            quality = "uncalibrated" if analyzed else "unavailable"
        detections = _detections(observation["detections"], cfg["low_threshold"]) if analyzed else []
        matches, used, ambiguous = _associate(tracks, detections, time, cfg) if analyzed else ({}, set(), set())
        eligible = [key for key, tr in tracks.items() if tr["confirmed"] and tr["in_pool"] and time - tr["last_seen"] <= cfg["retire_seconds"]]
        lost = sum(key not in matches for key in eligible)
        if quality == "ok" and len(eligible) >= 3 and lost / len(eligible) >= .6:
            if not mass_loss:
                mass_baseline = len(eligible)
            mass_loss = True
        if mass_loss:
            restored_count = sum(d["confidence"] >= cfg["high_threshold"] for d in detections)
            recovery_frames = recovery_frames + 1 if quality == "ok" and restored_count >= mass_baseline else 0
            if (eligible and lost / len(eligible) < .4) or recovery_frames >= 2:
                mass_loss, recovery_frames = False, 0
        if quality == "ok" and (mass_loss or ambiguous):
            quality = "degraded"
        reliable_delta = dt if quality == previous_quality == "ok" else 0.
        for key, tr in list(tracks.items()):
            if key in matches:
                d = detections[matches[key]]
                box = d["bbox_xyxy_normalized"]
                elapsed = time - tr["last_seen"]
                old_center, center = _center(tr["box"]), _center(box)
                tr["velocity"] = [(b - a) / elapsed for a, b in zip(old_center, center)] if 0 < elapsed <= cfg["max_gap_seconds"] else [0., 0.]
                tr.update(box=box, confidence=d["confidence"], last_seen=time,
                          in_pool=bool(corners and _inside(box, corners)))
                tr["confirm_seconds"] = tr["confirm_seconds"] + reliable_delta if elapsed <= cfg["max_gap_seconds"] + 1e-8 else 0.
                tr["confirmed"] |= tr["confirm_seconds"] + 1e-8 >= cfg["min_confirm_seconds"]
                if tr["event"] is not None and tr["event"]["kind"] == "visibility_lost":
                    close(tr, time, "person_reappeared")
                tr["loss_start"], tr["missing_seconds"] = None, 0.
                tr["state"], tr["reason"] = ("visible", "person_visible") if tr["in_pool"] else ("outside", "observed_pool_exit")
                if not tr["in_pool"]:
                    close(tr, time, "observed_pool_exit")
                elif d.get("head_state") == "below":
                    if tr["below_start"] is None:
                        tr["below_start"] = time
                    else:
                        tr["below_seconds"] += reliable_delta
                    tr["reason"] = "explicit_head_below"
                elif d.get("head_state") == "above":
                    close(tr, time, "head_observed_above")
                else:
                    tr["below_seconds"], tr["below_start"] = 0., None
            elif analyzed and tr["in_pool"]:
                tr["state"], tr["reason"] = "missing", "person_not_detected"
                tr["below_seconds"], tr["below_start"] = 0., None
                if tr["loss_start"] is None:
                    tr["loss_start"] = time - reliable_delta
                tr["missing_seconds"] += reliable_delta
            if quality != "ok":
                tr["reason"] = "analysis_unavailable" if quality == "unavailable" else "evidence_interrupted"
                # Require a fresh continuous run after an evidence interruption.
                tr["missing_seconds"], tr["below_seconds"] = 0., 0.
                tr["loss_start"], tr["below_start"] = None, None
            seconds = max(tr["missing_seconds"], tr["below_seconds"])
            if quality == "ok" and tr["confirmed"] and tr["in_pool"] and seconds + 1e-8 >= cfg["warn_seconds"] and seconds > 0:
                if tr["event"] is None:
                    below = tr["below_seconds"] > 0
                    kind = "possible_submersion" if below else "visibility_lost"
                    onset = tr["below_start"] if below else tr["loss_start"]
                    event_id = hashlib.sha256(f"{key}:{kind}:{onset:.9f}".encode()).hexdigest()[:20]
                    tr["event"] = dict(id=event_id, track_id=key, kind=kind,
                                       reason="explicit_head_below" if below else "person_visibility_lost",
                                       start_time=onset, trigger_time=time, last_seen=tr["last_seen"],
                                       last_position=_center(tr["box"]), severity="review", status="pending",
                                       evidence="explicit-head-state" if below else "visibility-only")
                    incidents.append(tr["event"])
                if seconds + 1e-8 >= cfg["urgent_seconds"]:
                    tr["event"]["severity"] = "urgent"
            # Retiring an identity never resolves its pending review incident.
            if time - tr["last_seen"] > cfg["retire_seconds"]:
                del tracks[key]
        for index, d in enumerate(detections):
            if index in used or index in ambiguous or d["confidence"] < cfg["new_track_threshold"]:
                continue
            key = f"track-{next_id:06d}"
            next_id += 1
            total_tracks += 1
            box = d["bbox_xyxy_normalized"]
            inside = bool(corners and _inside(box, corners))
            tracks[key] = dict(box=box, confidence=d["confidence"], velocity=[0., 0.], last_seen=time,
                               in_pool=inside, confirmed=cfg["min_confirm_seconds"] == 0,
                               confirm_seconds=0., state="visible" if inside else "outside", reason="new_detection",
                               missing_seconds=0., loss_start=None, below_seconds=0.,
                               below_start=time if d.get("head_state") == "below" else None, event=None)
        snapshot = dict(media_time=time, quality=quality, tracks=[
            dict(track_id=key, bbox_xyxy_normalized=list(tr["box"]), confidence=tr["confidence"],
                 state=tr["state"], last_seen=tr["last_seen"], missing_seconds=round(tr["missing_seconds"], 6), reason=tr["reason"],
                 review_severity=("urgent" if tr["event"]["severity"] == "urgent" else "warn") if tr["event"] else "none")
            for key, tr in tracks.items() if tr["confirmed"] or key in matches or tr["last_seen"] == time])
        sink(snapshot)
        counts[quality] += 1
        previous_time, previous_quality = time, quality
    return dict(frames=frames, incidents=incidents,
                summary=dict(frames=sum(counts.values()), tracks=total_tracks, incidents=len(incidents),
                             quality_counts=counts, last_media_time=previous_time,
                             tracker="conservative-constant-velocity-iou", diagnosis=False))
