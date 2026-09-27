# frontend

Owner: Person 5 (Frontend)

One page: live camera, alerts with sound, clip playback, "confirm" or "false alarm" button that sends the label back to the dataset.

## Implemented onboarding prototype

The current prototype connects a local recording, lets the operator mark four water-region corners and estimate pool dimensions, runs actual person/pose inference, and reveals a simple Three.js pool with synchronized markers and schematic upper-body joints. It includes footage, split, and 3D views, playback/seek controls, setup export, and timestamped tracking import. The broader alert dashboard described above remains integration work.

### Run from the repository root

```powershell
python -m pip install -r backend/requirements.txt
python frontend/setup_assets.py
python backend/serve.py --video "path/to/pool-reference.mp4"
```

Open **http://127.0.0.1:5173** in Chrome or Edge. Omit `--video` to use the local MP4/WebM import control instead. Dependencies and model weights download once into ignored `frontend/vendor/`; video files are not committed. The provided reference is the [2:09 wave-pool recording](https://www.youtube.com/watch?v=PuAfTA2wf7o). The reference preset stops its initial mapping at 95 seconds because the later camera view changes; inspect and recalibrate a new segment before continuing. `--video` uses this reference preset, so use the local-import control for unrelated recordings.

### Models and rendering

- **Person boxes:** pretrained COCO YOLOv8n, confidence threshold 0.25.
- **Joints:** pretrained YOLOv8n-pose; only confident upper-body points are displayed. A person with no reliable pose remains a location marker.
- **Inference:** ONNX Runtime Web 1.22.0, WebGPU with initialization fallback to single-thread WASM CPU. WebGPU uses a full frame plus four overlapping tiles; CPU uses a full frame. `?delegate=CPU` forces CPU initialization. Pinned export URLs and checkpoint hash checks are in `setup_assets.py`.
- **Optional comparison:** `?model=mediapipe` selects MediaPipe Pose Landmarker Lite. It produced fewer usable detections on this crowded clip during exploratory checks.
- **Motion/alarms:** there is no trained temporal distress model here. Temporary proximity-based IDs are for display only; Sunny's tracker and Rohan's event engine must provide authoritative identities and alarms.

The transition crossfades from footage through a video-textured water surface into a stylized pool. It is not recovered camera geometry. Four-point planar mapping provides approximate locations; dimensions/water level are operator estimates. Upper-body joints are a camera-facing schematic, not measured 3D anatomy or underwater depth. Occluded/unconfident joints are omitted. The pool depiction does not establish actual airway/submersion state.

### Timing and performance

Inference uses source-video timestamps, discards stale results after seeks, and hides outdated observations instead of showing an incorrect zero count. Automatic playback pacing slows the video when inference cannot keep up; very slow processing uses frame stepping. Do not describe this as guaranteed real-time operation at original recording speed.

A browser regression run on this laptop recorded **18 detections, 490 ms processing time, WebGPU, and 0.25× playback** on a sampled frame. These are runtime observations, not an accuracy benchmark; crowded swimmers are still missed and duplicates can occur. No four-camera throughput or drowning-event accuracy has been established.

### Tracking import contract

Use the tracking import control in the 3D stage. Coordinates are normalized to the original video frame; keypoints, when supplied, must contain all 17 COCO entries as `[x, y, confidence]`. The exact video SHA-256, dimensions, and duration must match. Export setup to obtain the video fingerprint.

```json
{
  "schema_version": 1,
  "video": {"sha256": "64-character-file-sha256", "width": 1280, "height": 720, "duration_sec": 129.0},
  "frames": [
    {"t_sec": 8.0, "people": [{"track_id": 1, "bbox_xyxy": [0.2, 0.3, 0.3, 0.5]}]},
    {"t_sec": 8.2, "people": []}
  ]
}
```

The example is a schema illustration; replace its fingerprint and values with measured data. Frames must be time-ordered. Files are limited to 50 MB, 10,000 frames and 100 people per frame. Observations older than 0.35 source seconds are hidden; there is no interpolation across missing observations. Importing results does not independently verify their accuracy. This sidecar is an observation format, separate from the team's incident/event schema.

### Validation and review

```powershell
python -m pip install pytest httpx playwright
python -m pytest backend/test_serve.py -q --basetemp artifacts/pytest-server
# With the reference server running and Chrome installed:
python -X utf8 frontend/browser_smoke.py
```

The server checks passed (3 tests: reference availability/fingerprint, media byte ranges, MIME handling). The browser check passed with real model inference, geometry/timestamp/ID checks, rapid paused seeks, invalidation after the view change, tracking import gaps and mobile layout; no page errors were recorded. Screenshots/results go into ignored `artifacts/`.

Claude Code was consulted in two read-only reviews using the CLI's `claude-opus-5-5` model and `xhigh` effort. Feedback informed timestamp handling, stale-result suppression, seek refresh, failure visibility, calibration validity and tile-boundary filtering. This was a code/design review, not an independent accuracy or safety certification. The paper audit is documented separately in [YOLO11-LiB analysis](../docs/samkwak188/yolo11-lib-paper-analysis.md).
