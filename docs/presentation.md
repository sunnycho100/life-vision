# We Fall, We Die

## A second set of eyes for the backyard pool

**Hackathon presentation · 7–9 minutes + questions**

> Opening: Play the first few seconds of the funny pool video with no overlay. Ask: “If this were your backyard, how long would it take you to notice that play had turned into danger?”

---

# 1. The problem

Backyard pools already have cameras.

What they usually do **not** have is an extra layer that understands what is happening in the water.

- Drowning can be quiet rather than theatrical.
- A person can disappear below the surface without crossing the pool boundary.
- Active distress can look like repetitive movement with little forward progress.
- Raw motion notifications create noise; they do not explain risk.

**Our question:** Can the cameras families already own become an explainable pool-risk assistant without requiring new hardware?

> Speaker note: Avoid claiming that the prototype “prevents drowning.” Say it identifies patterns that may require immediate attention.

---

# 2. Our product

**We Fall, We Die** analyzes a pool video stream, tracks each person, and turns behavior over time into three understandable states:

| State | Meaning | User experience |
|---|---|---|
| Green | Person is tracked and behavior appears normal | Quiet monitoring |
| Yellow | A concerning pattern is forming | Visible warning and timer |
| Red | The pattern persisted past a critical threshold | Alarm and “check the pool” action |

Every warning includes a timer and reasons such as:

- `Person missing inside pool: 2.7 s`
- `Repeated upper-body motion`
- `Low forward progress`

This is an assistant for the responsible adult—not a replacement for supervision, barriers, or lifeguards.

---

# 3. What we chose not to build

We began with the idea of two new neural networks: one for babies silently sinking and another for adults visibly struggling.

With five people, five MacBooks, and 24 hours, that would create two undertrained black boxes.

Instead, we chose:

1. **One shared person detector and tracker.**
2. **One history per person.**
3. **Two explainable temporal decision paths.**

We call the behaviors **passive submersion** and **active distress**, not “baby” and “adult.” Age-specific claims would require data we do not have.

> Decision principle: prioritize an end-to-end, explainable system over an impressive model that cannot be demonstrated reliably.

---

# 4. What powers the system

```text
Camera / uploaded video
          ↓
    frame ingestion
          ↓
 YOLO person detection
          ↓
 multi-object tracking ─── gives every person a stable ID
          ↓
 rolling feature window ── position, motion, visibility, time
          ↓
 temporal risk engine ──── passive + active behavior rules
          ↓
 WebSocket JSON results
          ↓
 video + canvas overlay ── green / yellow / red
          ↓
 alert, incident timeline, feedback
```

**Why this architecture works:** camera connectors can change without changing the detector, decision engine, or frontend.

---

# 5. Decision-making: passive submersion

```text
VISIBLE IN POOL
      ↓ person disappears inside pool boundary
TEMPORARILY MISSING
      ↓ warning time reached
YELLOW WARNING
      ↓ critical time reached
RED ALERT
```

The timer is canceled when:

- The person reappears.
- The person clearly exits through the pool boundary.
- A nearby new track is matched to the missing track.

The manually drawn pool polygon is important: it separates “went underwater” from “walked out of camera view.”

> Presentation thresholds are intentionally shortened and visibly labeled. They are not validated safety thresholds.

---

# 6. Decision-making: active distress

The system combines evidence across a rolling time window:

- High movement inside the person's box.
- Repeated upper-body or vertical motion.
- Little net movement through the pool.
- Persistence in approximately the same location.
- Optional frame-level drowning confidence.

```text
normal movement
      ↓ repeated motion + low forward progress
possible concern
      ↓ pattern persists
yellow warning
      ↓ multiple signals continue
red alert
```

No single splash creates a critical alert. Multiple signals must agree and persist.

---

# 7. Data and model strategy

| Source | How we use it |
|---|---|
| [Mibugi Drowning Detection](https://github.com/Mibugi/Drowning-detection) | Short YOLO11n fine-tuning experiment; rebuild splits by source video to reduce leakage |
| [Z5cc temporal project](https://github.com/Z5cc/drowning-detection) | Reference for sequence-based reasoning; not its full legacy training stack |
| [H20Saver checkpoint](https://huggingface.co/EsonH/best.pt) | Time-boxed backup inference test only; YOLO11x is too heavy for the core demo |
| Our staged footage | Camera-specific evaluation and live-demo material |
| [SwimXYZ](https://zenodo.org/records/8399376) | Future normal-swimming and pose-sequence data; too large for this hackathon |

Most public data contains still images or acted scenes. Therefore:

- We report results on whole held-out clips, not random neighboring frames.
- We do not claim clinical or real-world validation.
- We measure detection delay and false alarms, not just box accuracy.

---

# 8. Plugging into cameras people already own

The overlay lives in **our application**, not inside the native Ring or Google Home video screen.

```text
Ring / Nest / RTSP / webcam / file
                 ↓
        source-specific adapter
                 ↓
       normalized video frames
                 ↓
        shared analysis pipeline
                 ↓
       our player + canvas overlay
```

**Ring:** OAuth account linking, authorized device discovery, motion/person webhooks, then video-only WebRTC/WHEP. Current documented live sessions are short, making event-triggered analysis the initial use case.

**Google Nest:** Device Access + OAuth, inspect camera capabilities, then WebRTC or RTSP through the `CameraLiveStream` trait. Sessions require renewal or extension.

**Generic pool/IP cameras:** use an authorized RTSP/ONVIF feed when the model supports it.

For the hackathon, uploaded video is the reliable adapter. Webcam is next; RTSP, Ring, and Nest follow without changing the analysis or interface.

---

# 9. User journey

1. The user connects a supported camera or chooses a video.
2. They draw or confirm the pool boundary once.
3. They choose an alert preference:
   - yellow + red;
   - red only;
   - on-screen only.
4. They press **Activate monitoring**.
5. A deliberately cheesy Matrix-style scan signals activation.
6. People receive green tracking boxes.
7. Boxes transition to yellow and red only when temporal evidence persists.
8. An alert explains **who, why, and for how long**.
9. The user can confirm an incident or mark a false alarm, creating future labeled feedback.

---

# 10. Live demo

## Demo choreography

1. Open the dashboard before speaking; preload the team pool video.
2. Introduce the two teammates in the footage.
3. Press **Activate monitoring** and let the Matrix animation run.
4. Point out the two stable green IDs.
5. During the staged repetitive motion, Person 02 becomes yellow.
6. Read the explanation aloud: “Repeated upper-body motion; low forward progress.”
7. Person 02 turns red and the alarm triggers.
8. Change the notification preference to show user control.
9. Reset and briefly explain that the same interface can receive a real backend WebSocket stream.

## Important disclosure

The checked-in interface uses scripted results so the frontend and presentation can be developed before the model is connected. During the final presentation, clearly identify whether the run is:

- real model inference;
- a prerecorded real inference result; or
- the scripted interaction prototype.

Never describe scripted boxes as live AI inference.

---

# 11. Demo failure plan

| Failure | Recovery |
|---|---|
| Model misses a person | Use the known-good clip and fixed camera angle |
| Backend does not connect | Run the scripted frontend demo |
| Browser/audio problem | Show the exported annotated video |
| Wi-Fi or camera API fails | Use the local uploaded file; no internet is required |
| Live explanation runs long | Jump directly from activation to the yellow/red transition |

Have three artifacts ready on the presentation laptop:

1. Live/local model build.
2. Scripted interaction demo.
3. Screen recording of the successful end-to-end flow.

---

# 12. What we proved—and what comes next

## This hackathon

- One video-to-alert pipeline.
- Stable per-person visual state.
- Two explainable temporal behavior paths.
- Configurable notifications and incident history.
- A frontend that can receive real backend results without redesign.

## Next

- Replace scripted UI events with backend WebSocket results.
- Test stock YOLO11n and tracking on the team footage.
- Measure false alerts and detection delay on held-out clips.
- Add webcam, then one compatible RTSP camera.
- Prototype official Ring and Nest connectors using real developer credentials and devices.
- Design privacy, consent, retention, and account-deletion controls.

**Closing line:** The goal is not another camera. It is a camera that can tell you when the water stops looking normal—and explain why.

---

# Technical appendix: frontend contract

The backend sends one update per tracked person:

```json
{
  "track_id": 2,
  "bbox": [0.61, 0.33, 0.18, 0.43],
  "state": "yellow",
  "risk_score": 0.64,
  "reasons": [
    "repetitive upper-body motion",
    "low forward progress"
  ],
  "warning_seconds": 2.4,
  "missing_seconds": 0.0
}
```

Coordinates should either be normalized from 0–1, as above, or accompanied by the source resolution. The frontend maps them onto its displayed video and draws the overlay on a transparent canvas.

---

# Technical appendix: five-person final sprint

| Person | Final ownership |
|---|---|
| Data | Safe filming, labels, source-separated train/test clips |
| Model | Baseline detector, optional YOLO11n tuning, evaluation |
| Behavior | Tracker history, features, passive/active state machines |
| Backend | Video ingestion, WebSocket contract, rolling incident buffer |
| Frontend/presentation | Overlay, alerts, demo controls, screen recording, final talk |

Integrate against one known-good video as early as possible. Freeze the presentation build before the final hour; use the remaining time only for rehearsal and critical fixes.

