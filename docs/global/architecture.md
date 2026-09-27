# Architecture

updated: 2026-09-26 · status: agreed plan for the 12-hour build, nothing built yet except the MuJoCo sim

## What the system does
A camera watches a pool. For every person in the pool, the system tracks whether their **head is above the water**. If a person's head is not seen above water for too long, their box turns yellow, then red, and the app sounds an alert. It also flags when someone enters the pool.

It does **not** diagnose drowning. It reports observable events: "head not above water for N s", "lost inside the pool", "entered the pool", "camera view degraded".

## Pipeline

```
 video file (or webcam later)
        |
  [1] Source            FileSource: frames + timestamps                      Sunny
        |
  [2] Detector          YOLO person boxes + head evidence                    Joanne
        |                (optional) YOLO-Pose keypoints                       Sam
  [3] Tracker           ByteTrack or BoT-SORT, short-term IDs                 Sunny
        |
        v   tracks.json  (contract A)
  [4] Event engine      pool zone, head timer, missing registry,             Rohan
        |                merge rule, entry alert, visibility warning
        v   results.json (contract B)
  [5] Frontend          4 feeds, activate animation, colored boxes,          David
                         timers, reasons, alerts, replay

  [S] Simulation        MuJoCo pool scenes -> demo videos + ground-truth      Sunny
                         tracks.json for testing the engine without a model
  [E] Evaluation        test clips, metrics, baseline comparison             Sam
```

For the hackathon everything runs **offline on prerecorded video**: the pipeline writes JSON files, and the frontend plays the video and draws the overlay in sync by timestamp. The same objects can later be streamed over a WebSocket without changing the frontend's rendering code.

Hardware: our five MacBooks. No NVIDIA GPU, so no Isaac Sim, TensorRT, or DeepStream in this build. Nano-size models on CPU or Apple MPS.

## Components

### [1] Source (Sunny)
- `FileSource` reads a video and yields `(frame, t)`, where `t` is the **media timestamp in seconds**. Never derive time from frame count alone.
- Interface from David's camera guide, so webcam, RTSP, Ring, and Nest can be added later: `connect()`, `start()`, `next_frame()`, `stop()`, `status()`.

### [2] Detector (Joanne, pose by Sam)
- **Person:** pretrained YOLO (YOLO11n or YOLO26n), class `person` only. Input 960 px (`rect=True`) so far swimmers stay visible.
- **Head evidence**, one of these, picked by Joanne in the first hours based on what works on our clips:
  1. A pretrained YOLO head detector (CrowdHuman-trained, license to check), a head box inside a person box counts.
  2. YOLO-Pose head keypoints (nose, eyes, ears) above a confidence threshold (Sam).
- Fine-tuning is optional and only if the pretrained detector clearly misses swimmers. Data for that: the Mibugi / YOLO11-LiB dataset (dataset README says CC BY 4.0), with splits rebuilt by source video.
- Big models (SAM 3, Grounding DINO) are for offline labeling only, never in the pipeline.

### [3] Tracker (Sunny)
- Ultralytics built-in tracker. Start with ByteTrack, compare BoT-SORT on the same clips.
- Tracker IDs are **not trusted through a dive**. Ultralytics drops a lost track after `track_buffer` frames (default 30, about 1 s). The event engine keeps its own person IDs.

### [4] Event engine (Rohan)
Per frame, per person:

| Signal | Meaning |
|---|---|
| `in_pool` | Bottom-center of the person box is inside the pool polygon |
| `head` | `above` (head evidence found), `below` (person seen in pool, no head evidence), `unknown` (low confidence, glare, blocked) |
| `moving` | Box center moved more than a fraction of its own size over the last 1 s |

**Person states**

```
 OUTSIDE --enters pool--> IN_WATER --head not above / lost in pool--> SUBMERGED
                            ^                                             |
                            |<------------- head above again -------------|
                                                                          | t >= T_warn
                                                                       WARNING (yellow)
                                                                          | t >= T_alarm, or
                                                                          | t >= T_still and not moving
                                                                       ALARM (red) -> user confirms or dismisses
```

**Thresholds** (config file, never hardcoded):

| | Real default | Demo value (shown with a "DEMO THRESHOLDS" badge) |
|---|---|---|
| `T_warn` | 5 s | 2 s |
| `T_alarm` | 12 s | 5 s |
| `T_still` (under and not moving) | 8 s | 4 s |
| Gap that does not reset the timer (`unknown`) | 1 s | 1 s |
| Head above needed to clear | 0.5 s | 0.5 s |

Real defaults come from MYLO (12-15 s), the ASTM F3698-24 test (20 s), and CPSC saying 20 s is too late.

**Missing registry**
- When a person's track is lost inside the pool, the engine keeps that person with last position, last box, and the time they were last seen. Their timer keeps running.
- **Merge rule:** a new track that first appears **inside** the pool within `R` pixels of a missing person's last position is the same person, and their timer stops.
- **Safeguard:** a track that enters from **outside** the pool never clears a missing person. If two missing people could match one new track, keep both open and mark the event "uncertain".
- Head count is supporting evidence only, never the reason to close an event.
- A person whose track ends at the pool edge after being seen crossing it is `OUTSIDE`, not missing.

**Scene-level alerts**
- **Entry:** a new person enters the pool polygon → yellow notice "entered pool". Shown only while monitoring is armed (explicit switch in the UI).
- **Visibility:** mean frame brightness below a threshold, or average detector confidence drops for several seconds → "monitoring degraded" banner. Required by ASTM F3698-24.
- **Distress (optional):** if Sam's pose features are ready, "repetitive arm motion + little forward progress" for several seconds → yellow with reason. Never red on its own.

### [5] Frontend (David)
- Start screen: 4 feeds (public pool clips and MuJoCo renders), no overlay.
- "Enable monitoring" → short activation animation (time-boxed, max about 1 hour of work), then overlays appear.
- Canvas over each video, scaled from source to display coordinates.
- Boxes: green, yellow, red, and dashed gray at the last known position for `missing`.
- Next to each box: person ID, timer, reasons.
- Alert panel, audible alert (needs an "enable audio" click first), notification setting (yellow + red, red only, on-screen only), confirm and false-alarm buttons, incident replay.
- Pool polygon: a click tool that saves `<video>.pool.json`, read by the engine.

### [S] Simulation (Sunny)
- `sim/mujoco/pool_scene.py`: 5 people (stand, swim, float, drowning response, collapse) in a pool with a 0.9 m shallow end and 2 m deep end. Renders a CCTV-style MP4.
- Two uses for the hackathon:
  1. Demo feed videos.
  2. A **ground-truth `tracks.json`**: the sim knows every head's height, so it can write contract A directly (projected into the camera view). Rohan can test the engine before any model works.
- Risk: YOLO may not see capsule bodies as `person`. Check in hour 1. If it fails, the sim is only used through ground-truth tracks.
- Isaac Sim is out of scope (needs an RTX 4080+ on Linux or Windows).

### [E] Evaluation (Sam)
- Test clips split by recording session, never by frame.
- Cases: normal swimming, going under and coming up elsewhere, new swimmer entering while someone is missing, exit at the pool edge, splash and glare, empty pool.
- Report: time to alarm, false alarms per monitored hour, missed events with counts (not only percentages), ID merge success.
- Compare against the plain "person missing" timer as the baseline.

## Data contracts

### A. `tracks.json` (pipeline → engine)
```json
{
  "video": "pool_01.mp4",
  "camera_id": "cam1",
  "fps": 30,
  "width": 1920,
  "height": 1080,
  "source": "yolo11n+bytetrack",
  "frames": [
    {
      "t": 12.40,
      "brightness": 0.62,
      "people": [
        {
          "track_id": 17,
          "bbox": [412, 180, 566, 403],
          "conf": 0.82,
          "head": "above",
          "head_conf": 0.71,
          "head_bbox": [460, 180, 510, 228],
          "keypoints": null
        }
      ]
    }
  ]
}
```
- `bbox` is `[x1, y1, x2, y2]` in source-video pixels.
- `head` is `above`, `below`, or `unknown` as defined by the detector owner. `keypoints` is optional (17 COCO points `[x, y, conf]`, missing points as `null`, never `0, 0`).
- `source` is `"sim-ground-truth"` for sim-generated files.

### B. `results.json` (engine → frontend)
```json
{
  "video": "pool_01.mp4",
  "camera_id": "cam1",
  "pool_polygon": [[120, 300], [1800, 300], [1850, 1000], [60, 1000]],
  "demo_thresholds": true,
  "thresholds": {"warn_s": 2, "alarm_s": 5, "still_s": 4},
  "frames": [
    {
      "t": 12.40,
      "monitoring": "ok",
      "people": [
        {
          "person_id": 3,
          "track_id": 17,
          "bbox": [412, 180, 566, 403],
          "state": "yellow",
          "timer_s": 2.4,
          "reasons": ["head not above water 2.4 s"]
        }
      ]
    }
  ],
  "events": [
    {
      "event_id": "e1",
      "camera_id": "cam1",
      "person_id": 3,
      "type": "submersion",
      "level": "red",
      "reason": "head not above water for 5.0 s (demo threshold)",
      "started_t": 10.0,
      "last_seen_t": 10.0,
      "last_seen_bbox": [412, 180, 566, 403],
      "uncertain": false,
      "resolved_t": null,
      "resolution": null
    }
  ]
}
```
- `state`: `green`, `yellow`, `red`, or `missing` (draw at last known box, dashed).
- `monitoring`: `ok` or `degraded`.
- `type`: `submersion`, `entry`, `visibility`, `distress`.
- `resolution`: `resurfaced`, `exited`, `merged`, `dismissed`, or `null`.
- Confidence scores are never shown as a "probability of drowning".

### C. `<video>.pool.json` (frontend → engine)
```json
{"video": "pool_01.mp4", "pool_polygon": [[120, 300], [1800, 300], [1850, 1000], [60, 1000]]}
```

## Code layout
```
backend/pipeline/   Source, detector wrapper, tracker, writes tracks.json      Sunny
backend/events/     Event engine, reads tracks.json + pool.json, writes results Rohan
model/detector/     Person and head detection, optional fine-tuning           Joanne
model/pose/         YOLO-Pose features for distress                          Sam
model/eval/         Test clip list, metrics script                           Sam
frontend/           Web app                                                  David
sim/mujoco/         Pool scenes, videos, ground-truth tracks                  Sunny
data/               Clip list and labels (no raw video in git)               Sam
```

## Limits we state in the demo
- No real drowning footage. Tests are acted, simulated, or dummy-based.
- Clear water: a submerged person can still be visible. That is why we time the head, not the body.
- Daylight only. IR night cameras cannot see underwater.
- Glare, heavy splash, and crowded pools degrade it.
- A supervision aid, not a replacement for watching children, fences, or lifeguards.
