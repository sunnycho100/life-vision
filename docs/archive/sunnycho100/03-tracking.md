# 03. Tracking and Alert Logic

updated: 2026-09-26 · author: sunnycho100 (agent research)

## Short answer
- Use **ByteTrack** first (built into Ultralytics, motion only, fast). Try **BoT-SORT with ReID** if IDs keep switching after people resurface.
- Do **not** rely on the tracker alone to know who is underwater. Put our own alert layer on top: **pool zone mask + per-ID underwater timer + head count check + distance gate**.
- Key change to our plan: in clear water an overhead camera can still **see** a submerged person, so "person disappeared" is not the same as "person is underwater". Use a `head` (above water) class as the main signal, and "disappeared" as a backup.

## Tracker comparison

| Tracker | Appearance (ReID) | Handles long disappearance | Speed | Notes |
|---|---|---|---|---|
| ByteTrack | No | Keeps lost tracks for `track_buffer` frames, uses low-score boxes to recover | Fastest | Default in Ultralytics. Kalman prediction drifts while the person is gone. |
| OC-SORT | No | Re-anchors a track on the real observation when it reappears, fixes drift | Fast | Best motion-only choice for non-linear motion (swimming). In Roboflow `trackers`. |
| BoT-SORT | Optional | Can match by appearance after occlusion | Slower with ReID | Built into Ultralytics. `with_reid: False` by default. Camera motion compensation not needed for a fixed camera. |
| DeepSORT | Yes (separate ReID model) | Appearance matching | Slowest | Older. Swimsuits and wet hair look alike, so ReID gains may be small (unverified for pools). |

Ultralytics default values (both trackers):
```yaml
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 30       # frames a lost track is kept
match_thresh: 0.8
# BoT-SORT only
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: False
gmc_method: sparseOptFlow   # set to none for a fixed camera
```

Recommended changes for the pool:
```yaml
track_buffer: 450      # 15 s at 30 FPS, so an ID survives a long dive
track_low_thresh: 0.05 # keep weak, half-submerged detections
new_track_thresh: 0.4  # fewer fake new IDs from splashes
gmc_method: none       # camera does not move
```
A long `track_buffer` also means more stale "zombie" tracks, which our alert layer handles.

## Alert layer design

### Inputs per frame
- Tracked `person` boxes with IDs
- `head` boxes (head above water)
- Pool zone polygon (drawn once per camera in setup)

### States per ID
`OUT_OF_POOL` → `SURFACE` (head seen) → `UNDER` (in pool, no head seen) → `WARNING` → `ALARM`

### Pseudocode
```python
WARN_AFTER = 3.0        # s under before a soft warning
ALARM_AFTER = 5.0       # s under before the alarm (tune on footage)
EXIT_MARGIN_PX = 40     # lost near the pool edge = probably climbed out
GATE_M = 3.0            # a new track within 3 m ...
GATE_S = 15.0           # ... within 15 s is probably the same person

def update(t, tracks, heads, pool):
    for tr in tracks:
        if not pool.contains(tr.foot_point):
            state[tr.id] = OUT_OF_POOL
            continue
        if any(head_inside(h, tr.box) for h in heads):
            state[tr.id] = SURFACE
            last_surface[tr.id] = t
        # person box visible but no head above water: still count as under

    for pid in ids_in_pool():
        if pid not in visible_ids(tracks) and near_edge(last_box[pid], pool, EXIT_MARGIN_PX):
            state[pid] = OUT_OF_POOL           # left the frame at the edge
            continue

        # distance gate: a new ID showed up close to where this one went under
        new = find_new_track(near=last_box[pid], within_m=GATE_M,
                             since=last_surface[pid], within_s=GATE_S)
        if new:
            merge(pid, new)                    # same person resurfaced
            continue

        under = t - last_surface[pid]
        if under > ALARM_AFTER and head_count(t) < head_count_before(pid):
            state[pid] = ALARM                 # someone is still missing
        elif under > WARN_AFTER:
            state[pid] = WARNING
```

### Why each piece
| Piece | Fixes |
|---|---|
| Pool zone mask | People lost for reasons that are not the water (walked off, out of frame) |
| Head class | Clear water: a submerged person is still visible, so "disappeared" is unreliable |
| Distance gate | Swims underwater and comes up nearby with a new ID |
| Head count check | Comes up far away with a new ID. If the count is back to normal, nobody is missing. |
| Warning before alarm | Normal breath holding. A soft warning first keeps false alarms tolerable. |

### Parameters to tune on our footage
| Parameter | Start | Tune by |
|---|---|---|
| `ALARM_AFTER` | 5 s | False alarms per hour vs alert delay on acted sessions |
| `WARN_AFTER` | 3 s | Parent feedback |
| `track_buffer` | 450 frames | ID switches per session |
| `GATE_M`, `GATE_S` | 3 m, 15 s | Resurfacing sessions |
| `EXIT_MARGIN_PX` | 40 px | Camera placement |

Known gap: two people go under together and one comes up. The head count then drops by one and the alarm fires, which is the correct behavior.

## Sources
- [Ultralytics tracking docs](https://docs.ultralytics.com/modes/track), [Ultralytics tracker configs](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)
- [Roboflow trackers comparison](https://trackers.roboflow.com/latest/trackers/comparison/)
- [MOT trackers in production 2026](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/multi-object-tracking-deepsort-bytetrack-ocsort)
- [ID reassignment in BoT-SORT and ByteTrack (Ultralytics discussion)](https://github.com/orgs/ultralytics/discussions/19784)
