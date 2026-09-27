# Data, Model, Workflow, and Camera Integration Guide

Last verified: September 26, 2026

This document turns the team's dataset and camera research into an implementation plan. It separates what is useful for the 24-hour prototype from what belongs on the longer product roadmap.

## Recommended 24-hour build

Build one shared video pipeline, not separate large neural networks for each drowning pattern:

1. Read an uploaded or prerecorded video.
2. Detect people with a small YOLO model.
3. Track each person across frames.
4. Keep a rolling history for every track.
5. Run two explainable temporal rule sets:
   - **Passive submersion:** a person was visible in the pool, disappears without crossing the pool boundary, and remains missing.
   - **Active distress:** repeated upper-body movement, high local motion, and little forward progress persist together.
6. Return a risk state, timer, and reasons for each track.
7. Draw the results as green, yellow, or red boxes in the frontend.

The demo should use prerecorded footage and shortened, clearly labeled **demo thresholds**. It must not claim that the system is medically validated or that it specifically detects infant drowning.

## Source decision table

| Resource | What is available | Best use now | Important limitation | Decision |
|---|---|---|---|---|
| [Mibugi Drowning Detection repository](https://github.com/Mibugi/Drowning-detection) | A 2,000-image YOLO-format archive with `Drowning` and `Swimming` boxes; 1,400 train, 400 validation, and 200 test images. The dataset README states CC BY 4.0. The repository also includes YOLO11n and custom LiB-YOLO weights. | Fine-tune a normal, stock YOLO11n detector in the background, or use the images to test the data-loading and evaluation pipeline. | It is frame-based, not a temporal drowning dataset. Near-duplicate frames and source sequences cross the supplied splits, so the published split can overstate generalization. The custom LiB checkpoint depends on repository-specific Ultralytics modules. | **Use**, but rebuild the split by source video and train stock YOLO11n rather than making the custom checkpoint a dependency. |
| [YOLO11-LiB paper](https://www.mdpi.com/1424-8220/25/17/5552) | Architecture, reported accuracy, ablations, and experiment details associated with the Mibugi project. | Research citation and a reference for small-model design choices. | The reported metrics are not evidence that the model will transfer to our camera angle, pool, lighting, or actors. | **Reference only** for claims; reproduce our own results. |
| [Z5cc temporal drowning repository](https://github.com/Z5cc/drowning-detection) | Labels and reconstruction scripts for 111 roughly 25-second clips derived from 82 YouTube videos, plus temporal-model code and example GIFs. The repository includes 91 CVAT XML files with swim/drown boxes over time. | Borrow its central idea: classify a sequence of tracked observations rather than a single frame. If time permits, reconstruct a few available clips as an outside sanity check. | The videos are not included, links may disappear, the environment is old, trained weights are absent, and footage rights are unclear. Reproducing its full YOWOv2/ConvLSTM stack is too risky for the hackathon. | **Use the temporal design, not the full training stack.** |
| [H20Saver repository](https://github.com/kamalnath-git/drowning-detection-system) and [YOLO11x checkpoint](https://huggingface.co/EsonH/best.pt) | The author reports a 14,111-image, three-class dataset and publishes a large YOLO11x checkpoint. The repository includes training logs and prediction examples, but not the training images. | Give the checkpoint a time-boxed inference test on four of our clips: normal swimming, acted distress, submersion, and a difficult negative. | The weight file is about 114 MB, its training set is unavailable for inspection, its license/model card is unclear, and YOLO11x is unnecessarily heavy for five MacBooks. | **Backup only.** Stop after 30 minutes if setup or transfer performance is poor. |
| [SwimXYZ on Zenodo](https://zenodo.org/records/8399376) and [paper](https://arxiv.org/abs/2310.04360) | CC BY 4.0 synthetic swimming data: about 3.4 million frames, 11,520 five-second videos, 240 SMPL motions, multiple camera views, and pose annotations. | Future source of normal-swimming negatives and pose-sequence pretraining. | It contains normal strokes, not drowning, and its multi-gigabyte annotations and simulation workflow are too large for the current deadline. | **Do not download for the 24-hour build.** |
| [AHAR aquatic activity dataset](https://github.com/DingdongD/aquatic-activity-dataset) | Struggling, drowning, swimming, and waving sequences captured with 77 GHz millimeter-wave radar. | Possible future reference if the product ever adds radar hardware. | It is not RGB camera footage and does not fit the software-only camera product. Access and licensing are also unclear. | **Exclude.** |
| [Pool Detection Dataset preview](https://github.com/lonlonago/Pool-Detection-Dataset-for-Swimming-Pool-Drowning) and [Swimming Drowning Dataset preview](https://github.com/lonlonago/Swimming-Drowning-Detection-Dataset) | Commercial static-image datasets advertised with several thousand images and pool-related classes. | At most, visual comparison of camera angles and class definitions. | Paid, static rather than temporal, and provenance/licensing need due diligence. Purchasing them does not solve behavioral detection. | **Skip for the hackathon.** |

## How to use the useful data

### 1. Create a clean evaluation set before training

Record a small set of safe, staged clips with the same camera position planned for the presentation. Keep entire clips in only one split; never put neighboring frames from one clip into both training and test data.

Minimum test scenarios:

- Normal swimming across the pool.
- Treading water in one place.
- Splashing and playing.
- Floating still.
- Entering and leaving by the pool boundary.
- Safe, acted repetitive distress motion in shallow water.
- Safe, acted disappearance/submersion in shallow water.
- Empty pool, glare, shadows, and partial occlusion.

The recording must use shallow water, no breath-holding, and a dedicated observer who is not acting.

### 2. Establish a baseline before fine-tuning

Run a stock nano-size person detector on the team's clips. If it already maintains adequate person boxes, use it and spend the remaining time on tracking and temporal logic. Fine-tune on Mibugi only if the baseline consistently loses people in the water.

For a Mibugi experiment:

- Group frames by source clip before creating train/validation/test splits.
- Train stock YOLO11n at 416 or 640 pixels for a short 20–30 epoch experiment.
- Prefer a model that runs smoothly on one MacBook over a slightly more accurate large model.
- Report results on the team's held-out clips, not only on Mibugi's supplied test folder.
- Treat a frame-level `drowning` confidence as one input signal, never the final alert by itself.

### 3. Track people and calculate behavior over time

For each person, maintain a rolling window containing:

- Box center, width, height, and area.
- Detection confidence.
- Estimated speed and net forward progress.
- Total motion inside the box.
- Vertical oscillation and repeated-motion score.
- Whether the head or upper body is visible, if pose estimation works reliably.
- Time since the person was last visible.
- Whether the last visible position was inside the pool polygon.
- Per-frame drowning-model confidence, if available.

Start with transparent rules or a small logistic-regression/random-forest model. Do not attempt to train a new video transformer during the hackathon.

### 4. Use two temporal state machines

**Passive submersion**

`visible in pool -> temporarily missing -> yellow timer -> persistently missing -> red`

Cancel the timer if the person reappears or their track clearly exits through the pool boundary.

**Active distress**

`normal -> repetitive motion + low progress -> yellow persistence -> red persistence`

Use multiple signals together. Splashing alone or remaining in one position alone should not immediately trigger red.

### 5. Send a stable result contract to the frontend

The backend should emit one object per tracked person, for example:

```json
{
  "track_id": 3,
  "bbox": [412, 180, 566, 403],
  "state": "yellow",
  "risk_score": 0.64,
  "reasons": [
    "low forward movement",
    "repetitive upper-body motion"
  ],
  "warning_seconds": 2.4,
  "missing_seconds": 0.0
}
```

This contract lets the frontend progress with mocked data before the model is complete.

## Overlay design for existing cameras

The realistic product is an overlay in **our own web or mobile view**, fed by an authorized camera stream. We should not promise that green/yellow/red boxes can be injected into the native Ring or Google Home live-view screen. The camera provider supplies video; our application decodes the frames, runs detection, and draws an HTML canvas layer over its own video player.

```text
Ring / Nest / RTSP / file / webcam
                |
        camera-source adapter
                |
       normalized video frames
                |
      detector -> tracker -> rules
                |
       per-person JSON results
                |
     our video player + canvas overlay
                |
       alerts + incident history
```

Every source should implement the same internal operations:

- `connect()`
- `start()`
- `next_frame()`
- `stop()`
- `status()`

The prototype can implement `FileSource` first, then `WebcamSource`. Later connectors can replace the source without changing the detector, tracker, risk engine, or overlay.

## Ring integration: now officially feasible, but not a hackathon dependency

Ring now has an official [Ring Developer platform](https://developer.ring.com/) and [getting-started documentation](https://developer.amazon.com/docs/ring/get-started.html). Its documentation explicitly describes computer-vision applications, authorized device access, webhooks, and live video through WebRTC/WHEP.

A production Ring connector would:

1. Register the team and application in the Ring Developer Portal.
2. Obtain client credentials and keep all secrets on the backend.
3. Let the Ring account owner authorize selected cameras through Ring's account-linking flow.
4. Discover authorized devices and inspect their capabilities.
5. Receive motion/person webhooks as an efficient trigger.
6. Start a video-only WebRTC/WHEP session for the selected camera.
7. Decode frames into the same internal pipeline used by uploaded video.
8. Show the analyzed feed and colored overlay in our application.
9. Complete Ring privacy/security review and certification before a public release.

Ring's current getting-started guide says battery-powered live sessions are 30 seconds and line-powered sessions are 60 seconds. That makes Ring especially suitable for **event-triggered analysis**. We still need a device and developer credentials to test reconnection behavior, latency, rate limits, and whether continuous pool monitoring is practical. The customer also controls which cameras an app may access, and an eligible Ring subscription may be required.

Useful Ring sources:

- [Ring Developer overview](https://developer.ring.com/)
- [Getting started](https://developer.amazon.com/docs/ring/get-started.html)
- [Development guide](https://developer.amazon.com/docs/ring/develop.html)
- [Ring Appstore and customer authorization](https://ring.com/support/articles/e2sg9/Ring-appstore-third-party-integrations)

## Google Nest integration: feasible through Device Access

Google's [Device Access program](https://developers.google.com/nest/device-access) exposes Nest cameras through the Smart Device Management API. The exact stream protocol depends on the camera model and whether it is managed by the Nest app or Google Home app.

A production Nest connector would:

1. Register for Device Access and create a Google Cloud project.
2. Configure OAuth 2.0 and let the device owner grant access.
3. List the user's authorized devices and read each camera's supported protocol.
4. Request either a WebRTC or RTSP stream through the `CameraLiveStream` trait.
5. Decode the stream into the same internal pipeline used by uploaded video.
6. Renew or extend sessions where the device permits it.
7. Optionally use motion/person events from Google Cloud Pub/Sub to begin analysis.
8. Complete Google's commercial review before releasing a public consumer integration.

Google documents five-minute camera sessions. Wired cameras can extend eligible WebRTC sessions; battery devices have tighter limits. Modern Home-app cameras generally use WebRTC, while some legacy Nest-app cameras expose RTSP. The connector must inspect capabilities instead of assuming one protocol.

Useful Nest sources:

- [Device Access overview](https://developers.google.com/nest/device-access)
- [Get started](https://developers.google.com/nest/device-access/get-started)
- [Supported camera models and protocols](https://developers.google.com/nest/device-access/supported-devices)
- [CameraLiveStream trait](https://developers.google.com/nest/device-access/traits/device/camera-live-stream)
- [Camera events and Pub/Sub](https://developers.google.com/nest/device-access/api/events)
- [Product review requirements](https://developers.google.com/nest/device-access/project/review)

Google's product review guidance restricts apps that merely duplicate native Nest notifications. Our eventual commercial use case should be described as transparent pool-risk analysis with incident context, not a generic duplicate motion notification.

## Generic pool and IP cameras

For cameras or network video recorders that provide an authorized RTSP or ONVIF stream, an `RtspSource` is likely the simplest live integration. Support must be verified model by model; consumer cameras do not universally expose local RTSP. Credentials should be stored on the backend, video should be encrypted in transit where the device supports it, and raw footage should not be retained unless the user enables incident recording.

This is the best connector to implement after file/webcam input because it tests a real continuous stream without coupling the demo to Ring or Nest approval.

## Frontend next steps

### Must finish first

1. Build one page with file upload, video playback, and **Activate Monitoring**.
2. Put a transparent canvas exactly over the video and make it resize with the player.
3. Render mocked person results using green, yellow, and red boxes.
4. Add a short Matrix-inspired startup animation without delaying actual video processing.
5. Show risk reasons and a visible warning/critical timer next to each box.
6. Add notification choices: yellow + red, red only, or on-screen only.
7. Add a clear **Demo thresholds** badge whenever shortened presentation timers are active.
8. Add an alert panel containing timestamp, track ID, severity, and reasons.

### Connect to the backend next

1. Agree on the per-person JSON contract shown above.
2. Receive updates over WebSocket, with a mocked playback fallback.
3. Map source-video coordinates to displayed-video coordinates so boxes remain aligned at every window size.
4. Add hysteresis so a single noisy frame does not flicker between colors:
   - require roughly 0.5–1 second of concern before green becomes yellow;
   - require another 1–2 demo seconds before yellow becomes red;
   - require roughly 1 second of normal behavior before clearing an alert.
5. Keep the last good position briefly when a detection drops, and mark it as `missing` rather than instantly deleting the box.
6. Add a manual pool-polygon drawing step so disappearance logic knows the monitored water area.

### Presentation polish after the core works

- Preload two known-good demo clips and keep a screen recording as a fallback.
- Add an audible alert and a large red incident banner.
- Show the exact reasons for a state change; avoid an unexplained magic score.
- Add **Confirm incident** and **False alarm** buttons for the future feedback loop.
- Export or replay the annotated incident clip.
- Keep the safety disclaimer visible in settings/about and at the end of the presentation.

### Camera connector work after the hackathon

1. Implement `WebcamSource`.
2. Implement and test `RtspSource` with one known compatible camera.
3. Apply for Ring developer access and test an event-triggered WHEP connector on a real Ring device.
4. Register a Nest Device Access sandbox project and adapt Google's web application sample for one supported camera.
5. Measure end-to-end latency, reconnect reliability, frame rate, battery behavior, and false alerts for each source.
6. Design consent, retention, encryption, account deletion, and camera-access controls before any user pilot.

## Suggested team workflow

| Owner | Parallel task | Deliverable |
|---|---|---|
| Data | Create source-grouped Mibugi splits and organize safe team clips | Versioned labels and a held-out test set |
| Model | Run stock YOLO baseline, then one short YOLO11n experiment if necessary | Lightweight person/water detector and evaluation notes |
| Behavior | Implement per-track rolling features and the two state machines | Repeatable JSON results with reasons |
| Backend | Build file ingest, frame timing, tracker integration, WebSocket results, and incident buffer | One local service used by both model and frontend |
| Frontend | Build video/canvas overlay, controls, timers, alerts, and presentation polish | Reliable one-page demonstration |

Integrate against a single prerecorded clip early. Do not wait for training to finish: the backend can emit scripted JSON and the frontend can render it while model work continues.

## Definition of done for the presentation

- A known video loads and plays reliably.
- Activation animation runs.
- At least one person is tracked with a stable ID.
- The box visibly changes green -> yellow -> red from temporal evidence.
- The UI shows the timer and human-readable reasons.
- An alert follows the user's selected threshold.
- Passive-submersion and active-distress clips both demonstrate the shared pipeline.
- A backup annotated recording is available if live inference fails.
- The team states that this is a research prototype, not a certified lifesaving system.

