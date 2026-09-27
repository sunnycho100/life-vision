# Poolside person detection

The implemented milestone is **recorded video → RF-DETR Nano person boxes → timestamped replay**. The browser uploads recordings to a processing server. Pose models, skeletons, and temporary person IDs have been removed.

The 2D video is the default. The optional 3D view displays symbolic pins inside a manually calibrated pool rectangle. Pause and select a pin to inspect estimated X/Z coordinates from corner 1. Corner 1→2 is width; 1→4 is length. Mark the calibration as user-measured only when its dimensions and corners are known. The marker height is assumed at the water surface, and its size is symbolic. There is no drowning classifier, identity tracker, incident recorder, or alert workflow in this milestone. A separate YOLO/timer demo on upstream `main` is not connected to this browser pipeline.

## Run the installed application

From the repository root on this workstation:

```powershell
.venv\Scripts\python.exe -m backend.serve --video "..\footage-research\pool-reference.mp4"
```

Open **http://127.0.0.1:5173**. Connect the reference or upload an H.264 MP4 / WebM. Define the pool outline, choose the analysis interval, then select **Analyze recording**. Results can be reviewed during processing or replayed from the cache afterward.

The display confidence defaults to 0.50 and can be adjusted down to 0.20 without rerunning inference. These thresholds have not passed a human-labeled accuracy gate. Counts are person boxes, not a guaranteed census.

## Set up another machine

Use one server process per data directory. Model export requires a compatible PyTorch machine; CPU ONNX deployment does not require PyTorch or a browser GPU.

Reference/export environment:

```powershell
python -m venv .venv-reference
.venv-reference\Scripts\python.exe -m pip install -r backend/requirements-reference.txt
.venv-reference\Scripts\python.exe -m tools.prepare_rfdetr --video "path/to/pool-reference.mp4"
```

The preparation script uses five frames from the reference recording at approximately 8, 22, 45, 70, and 90 seconds. It downloads only the pinned official Nano checkpoint, verifies SHA-256, runs the official SDK, exports FP32 ONNX, and compares input tensors, raw outputs and decoded person boxes. Export and verification artifacts are written under ignored `artifacts/models/rfdetr-nano/`.

Deployment environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
.venv\Scripts\python.exe -m tools.prepare_rfdetr --verify-only
.venv\Scripts\python.exe -m backend.serve --video "path/to/pool-reference.mp4"
```

Copy the entire model artifact directory from the export machine before verification. The development requirements include SciPy for the parity/evaluation tools; ordinary serving only needs `backend/requirements.txt`. On POSIX, use the corresponding `.venv/bin/python` executable.

Native Windows ARM uses ONNX graph optimization **BASIC**; tested x64 uses **ALL**. This difference is intentional: default ARM fusion changed tied encoder proposals on one parity frame. The runtime refuses to load an artifact without a passing local parity report matching its model hash, adapter code, provider, ONNX Runtime version and platform. Run verification again after changing these.

If ONNX verification fails, the official Python backend remains available:

```powershell
.venv-reference\Scripts\python.exe -m backend.serve --backend python --video "path/to/pool-reference.mp4"
```

`--worker-python PATH` selects a different installed Python for the inference subprocess. `--model-manifest PATH` and `--data-dir PATH` select the model bundle and runtime storage. `--host LAN_ADDRESS` lets browsers on a trusted network connect; this prototype has no authentication and is not a public internet deployment.

Run `python frontend/setup_assets.py` to download optional Three.js assets and the pinned YOLO evaluation baseline. The normal video/detection view works without Three.js. No pose assets are downloaded or requested.

## Runtime behavior

- Official pretrained `RFDETRNano`, `rfdetr==1.11.0`, 384 × 384, FP32, person category resolved from the pinned COCO mapping.
- RF-specific RGB square resize, half-pixel bilinear interpolation without antialiasing, ImageNet normalization, sigmoid and global top-300 query/class selection. No full-frame NMS.
- One subprocess job at a time. Playback speed does not drive inference.
- Target sampling rate: 5 frames per video second, retaining each selected frame's actual presentation timestamp. Low-frame-rate inputs never duplicate observations.
- Cache: indexed SQLite observations on disk; at most three 10-second windows in the browser.
- A sampled result remains visible for at most 0.3 seconds. An analyzed empty frame displays **0**; missing/failed/stale analysis displays **—**.
- Pool membership uses box bottom-center as a provisional approximation. Detection continues after mapping expiry; pool counts and 3D markers require a valid mapping.
- The known reference view changes around 95 seconds. Its SHA-256, rather than the filename or `--video` flag, enables that preset.
- Uploads are streamed and hashed by the server, capped at 2 GiB. Media is served with byte-range support. Rotated video, non-square pixels, and frames larger than 3840 × 2160 are rejected.
- Cancelled or interrupted jobs retain their partial observations and are marked incomplete. The UI can start another analysis.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Model configuration and artifact availability |
| GET | `/api/reference` | Optional reference source metadata |
| POST | `/api/sources` | Raw MP4/WebM body with matching Content-Type |
| GET | `/api/sources/{source_id}` | Source metadata and SHA-256 |
| GET | `/api/sources/{source_id}/video` | Browser video, including Range requests |
| POST | `/api/jobs` | `{source_id, start, end}`; times in source seconds |
| GET | `/api/jobs/{job_id}` | Progress, runtime, errors, timing and completion |
| POST | `/api/jobs/{job_id}/cancel` | Stop worker, preserve incomplete results |
| GET | `/api/jobs/{job_id}/observations?start=8&end=18` | At most a 30-second window |

Observation shape:

```json
{
  "schema": "detections/1",
  "source_id": "video-sha256",
  "job_id": "server-job-id",
  "frame_index": 480,
  "pts": 480480,
  "time_base": "1/60000",
  "media_time": 8.008,
  "status": "analyzed",
  "detections": [
    {"bbox_xyxy_normalized": [0.2, 0.3, 0.3, 0.5], "confidence": 0.75, "class_name": "person"}
  ],
  "inference_ms": 140.0
}
```

Values above illustrate the schema, not actual detections. Frame index, PTS and media time are recorded from the decoded video. The job manifest includes source metadata/hashes, model hashes, package versions, sampling policy, runtime and code fingerprints.

Legacy `schema_version: 1` result files still import using video SHA-256, dimensions and duration, with `frames[].t_sec` and `people[].bbox_xyxy`. Former identity and keypoint fields are ignored. Missing legacy confidence is displayed as unknown. Imports are capped at 20 MiB with no 100-person truncation.

## Evaluation and annotation

Read [RF-DETR implementation and measured results](../docs/samkwak188/rfdetr-implementation.md).

The prepared bundle is `artifacts/evaluation/reference/`: 16 development frames, 8 locked later frames, image hashes, unreviewed annotation templates, and RF-DETR/YOLO candidate files. **These templates are not ground truth.**

Open **http://127.0.0.1:5173/annotate.html**, select that folder, draw person boxes, correct pool membership and visibility tags, inspect at 200%, and mark each completed frame reviewed. Download `annotations-reviewed.json` into the same folder. A second reviewer should independently annotate two frames to establish a consistent visible-extent policy. Locked frames should not be labeled from model suggestions.

```powershell
.venv\Scripts\python.exe -m tools.evaluate_detection select --labels artifacts/evaluation/reference/annotations-reviewed.json --candidates artifacts/evaluation/reference/rfdetr-full.json --output artifacts/evaluation/reference/rfdetr-policy.json
.venv\Scripts\python.exe -m tools.evaluate_detection evaluate --labels artifacts/evaluation/reference/annotations-reviewed.json --candidates artifacts/evaluation/reference/rfdetr-full.json --policy artifacts/evaluation/reference/rfdetr-policy.json --output artifacts/evaluation/reference/rfdetr-locked-report.json
```

The selector maximizes development recall subject to precision ≥90%, a nonzero true-positive count and threshold ≥0.20. It compares the two pool anchors using ground-truth boxes. The frozen policy cannot be overwritten; locked evaluation refuses changed predictions and an existing output. Run YOLO through the same commands with its own candidates and policy.

Full-frame and tiled candidate generation are available through `python -m tools.evaluate_detection predict --help`. Tiling is an offline experiment, not an automatically promoted production setting. It uses one full frame plus six overlapping square crops and cross-pass NMS at IoU 0.5 or 0.7; both detectors receive identical crops. Full-frame RF-DETR receives no NMS.

Metrics include one-to-one IoU-0.50 precision/recall, duplicate false positives, size/partial-person groups, ignored regions, pool-membership errors, and development AP50 over candidates above 0.05. The locked frames are from the same camera/session; another recording is required before claiming generalization.

## Verification

```powershell
.venv\Scripts\python.exe -m pytest backend tools/test_evaluate_detection.py -q --basetemp artifacts/pytest-rfdetr
.venv\Scripts\python.exe frontend/browser_smoke.py
.venv\Scripts\python.exe frontend/surface_geometry_smoke.py
```

The browser test needs Chrome, the reference server, prepared evaluation frames and verified model artifacts. It runs real inference and tests normal-speed replay, seek/cache behavior, presentation timestamps, more than 100 boxes, empty versus unavailable observations, mapping expiry, legacy import, mobile layout, no pose requests, and annotation export. Synthetic acceptance fixtures are explicitly separate from accuracy labels.

Screenshots and detailed results are saved under ignored `artifacts/`. Recordings, weights, and runtime environments are not committed.
