# Final implementation recommendation: 24-hour pool-monitoring prototype

> Updated direction: the user clarified the product as camera input → swimmer tracking → suspected-drowning review → saved incident clip → human disposition. Read the [current ML system decision](ml-system-decision.md) for the latest model choice, implementation audit, recording requirement and research findings. The plan below is preserved as the earlier 24-hour proposal; its initial repository inventory and YOLO11n selection predate the YOLOv8 browser prototype.

Date: 2026-09-26. Repository reviewed through `4daa9b7`.

**Confirmed constraints:** 24 hours, existing laptop, GPU unconfirmed. This is a concrete recommendation for the team, not a record that everyone has accepted the assignments. It consolidates all current notes under `docs/`; earlier research remains useful background where it conflicts with this narrower build scope.

## 1. Decision and honest product claim

Build a local application that processes **prerecorded, fixed-camera pool video**, tracks visible people, and presents explainable **armed-zone presence** and **lost visual contact** alerts with synchronized replay and sound.

Use **pretrained YOLO11n person detection, explicitly configured ByteTrack, and an incident registry outside the tracker**. Make upper-body pose a bounded experiment with YOLO11n-pose. Do not require training, a GPU, age classification, 3D reconstruction, or synthetic-data generation to finish the demo.

The demo claim is: **“We flag visible pool-zone presence and sustained loss of visual contact, and show the evidence for review.”** It is not a validated drowning detector. A visible submerged swimmer can remain detected; an unseen swimmer might be occluded rather than submerged. These are central limitations of the baseline, not problems solved by a timer.

**Hackathon judgment:** viable as a focused monitoring prototype. Generic boxes and a timer alone are a weak differentiation. The stronger demonstration is the complete incident workflow, explicit uncertainty, persistent unresolved incidents, and a measured comparison on held-out recordings. A claim of reliable toddler drowning detection would exceed the evidence and the build window.

## 2. How the team's research changes the plan

| Notes reviewed | Adopt | Change or defer |
|---|---|---|
| [Shared technical summary](../technical-summary.md) and [research history](../research-notes.md) | Home-pool focus, detect everyone, temporal events, one dashboard. | Replace “disappeared = underwater.” Defer factory use, live cameras, simulation-to-real classifier training, and multi-size training ablations. |
| Sunny: [summary](../sunnycho100/00-summary.md), [models](../sunnycho100/01-detection-model.md), [training](../sunnycho100/02-yolo-training-pipeline.md), [tracking](../sunnycho100/03-tracking.md), [MuJoCo](../sunnycho100/04-mujoco-sim.md), [Isaac](../sunnycho100/05-isaac-sim.md) | YOLO baseline, optional offline labeling, session splits, existing simulation work. | Do not clear incidents using population count or proximity alone. Do not infer an exit merely because loss occurred near the edge. Simulation is an optional separate illustration. |
| Rohan: [CV feasibility](../rsusarla3/cv-feasibility.md), [detection logic](../rsusarla3/detection-logic-research.md) | Clear-water failure case, separate incident registry, explicit unknown states, entry alerts, event-level evaluation. | Above/below-water inference needs its own validated evidence. Pose head points do not supply it. Vendor and standards timings do not validate our thresholds. Defer DeepStream and trained temporal classification. |
| Joanne: [baby detector research](../joannemiki57/baby-detector-research.md) | Detect all people first; evaluate small and partially visible people; consider an age attribute later. | Baby-specific thresholds and count rules add scope even when age is called an attribute. Defer age models, hardware purchases, multi-detector comparisons, and teacher ensembles. A pool-plane homography alone cannot recover vertical body height or off-plane shoulder dimensions. |
| Dpark: [frontend plan](../dpark/frontend-interface-plan.md) | Uploaded/recorded video, boxes, timelines, reasons, sound, review controls. | Replace `SAFE` with `VISIBLE` or `NO RULE TRIGGERED`. Defer consumer-camera integration, elaborate activation animations, and unvalidated danger scores. |
| Sam: [SAM / YOLO / pose analysis](sam%20analysis.md) | YOLO first, persistent incidents, visibility uncertainty, upper-body pose experiment. | Collapse the model shortlist to one baseline and one optional experiment for this deadline. MediaPipe and SAM 3D remain later alternatives. |

All contributor README files were also reviewed; they are folder introductions. Research tables and estimated FPS are not measurements of this project. This plan does not adopt every external factual claim in the notes.

## 3. What actually exists today

The repository contains research, directory placeholders, MuJoCo simulation code and generated sample sequences, and an Isaac script. It does **not** yet contain the working detection service, dashboard, tracking/event pipeline, dependency manifest, or measured pool-video evaluation.

The MuJoCo script generates scripted motions and 3D keypoints. Its geometric head-underwater flag is not an airway measurement or validated drowning label. The Isaac script has unverified execution and unfinished assets/rendering details. Neither establishes a working synthetic-to-real training pipeline. Preserve these contributions, but do not make the demo depend on completing them.

## 4. Frozen MVP scope and model choices

| Component | Decision | Completion criterion |
|---|---|---|
| Input | One local recorded video per job; fixed camera, daylight/lit pool. | File loads, dimensions/duration are reported, unsupported input returns an understandable error. |
| Setup | Draw a pool-region polygon; explicit armed mode; test audio. | Configuration is saved with the job. The UI clearly says whether alerts are armed. |
| Detection | `yolo11n.pt`, retain `person` detections. Start at 640 input; measure 960 if distant-person misses justify it. | Actual inference on the target laptop, with runtime and annotated example frames recorded. |
| Tracking | ByteTrack, selected explicitly rather than relying on package defaults. | Per-frame IDs plus recorded configuration; incident persistence survives tracker expiry. |
| Events | Armed-zone presence and sustained lost contact. | Deterministic event history with reason, timestamps, last-seen location, and review state. |
| Backend | Python + FastAPI; one processing job at a time; local JSON/JSONL outputs. | Upload/configure/process/status/results flow works without a cloud account or database. |
| Frontend | React + Vite, one page, video plus canvas overlay and incident list. | Playback and overlays use the same media clock; clicking an incident seeks to its evidence. |
| Pose | Optional `yolo11n-pose.pt` experiment, limited to two engineering hours after the baseline works. | Keep only if visible upper-body points are useful and runtime acceptable on development clips. |
| Deployment | Existing laptop; local processing. | Reproducible launch instructions and cached outputs from genuine inference for rehearsal. |

YOLO11 provides detection and pose checkpoints; pose uses 17 body landmarks. These are documented capabilities, not evidence of pool performance: [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/), [pose documentation](https://docs.ultralytics.com/tasks/pose/). Tracker selection and configuration are documented in [Ultralytics tracking](https://docs.ultralytics.com/modes/track/).

Pin package versions, checkpoint identity/hash, device, input size, sampling rate, and tracker settings once the baseline runs. Check applicable code, checkpoint, and dataset licenses before distributing them. Do not assume “offline teacher” use removes license obligations.

### Why not train babies versus adults immediately?

Detect children and adults as `person`. Land portraits do not represent distant, partly submerged swimmers. Separate age classes add labels and failure modes without resolving underwater status. A single person class still needs evaluation on children; adult-only footage cannot demonstrate child performance.

Later, compare an age attribute against a two-class baseline using representative, permissioned data. Keep `age_unknown`, and never suppress base monitoring because age is uncertain. Carrying an age label from deck to water also depends on correct identity association. Adult presence in the image does not establish active supervision.

### What pose can contribute in this build

Show sufficiently confident observed nose/shoulder/elbow/wrist estimates and their temporal availability. Treat hidden or uncertain joints as missing. Do not require visible legs or a complete skeleton to detect a person. Do not turn estimated hidden joints into evidence of submersion.

If the pose checkpoint supplies adequate person boxes, it may replace the detector after a development comparison. Otherwise retain the detector; omit pose if an additional pass is too costly. An optional overlay is not a validated distress feature. Do not train a drowning classifier in this window.

## 5. Event behavior: implement this contract

Keep **observation state**, **incident state**, **review state**, and **processing health** separate. Tracking IDs are association hypotheses, not permanent identity guarantees.

### A. Armed-zone presence

Use the box center inside the drawn polygon as the initial, explicitly approximate image-space membership rule. Test it on the chosen perspective; feet are often invisible. Display the polygon so boundary mistakes are reviewable.

When armed, a person observed inside for at least **0.5 seconds of media time** creates one presence event for that continuous visit. If already present when armed, report “person present when armed,” not an observed entry. Only label an entry when an outside-to-inside transition was observed. Avoid creating a new presence event every frame; track-ID changes remain a possible duplicate source to measure.

This is zone presence, not proof of physical water entry. Use it for everyone; do not condition it on inferred age or absence of an adult.

### B. Lost visual contact

1. After a person has been observed inside the pool region, retain a registry entry independently of the tracker.
2. On missing detections, record the last reliable observation timestamp and location. The UI says `CONTACT LOST`, not `UNDERWATER`.
3. At **3 seconds** since last reliable observation, show a warning. At **5 seconds**, create/escalate a lost-contact incident and sound the configured alert.
4. These are **demo defaults**, chosen to demonstrate the workflow. They are not medically validated thresholds or a claim of standards compliance. Tune only on development footage, then freeze before evaluation.
5. A continuous, unambiguous reacquisition can end the missing interval. The incident remains in history. A new nearby ID or restored head count is insufficient to establish that the missing person returned. Ambiguous reacquisition leaves the incident unresolved for review.
6. Observed movement out of the pool region onto the deck can end that visit. Merely disappearing near an edge cannot establish an exit. Boundary flicker should not repeatedly open/close visits; use the same 0.5-second membership confirmation and record ambiguous boundary loss.
7. Acknowledge/mute affects notification state only. Resolution is a separate action with a reason; it must not silently change the underlying observation evidence.

**Known missed case:** a person still detected under clear water does not trigger the missing-person rule. Report this as unsupported. Adding head detection or pose alone does not fix it. A future surface-state model needs `above` / `below` / `unknown` labels and separate evaluation.

### C. Time and processing failures

- Use source presentation timestamps, not inference wall time or an assumed frame count. Sparse analysis must retain the actual elapsed source time.
- Processing can be slower than playback. Preprocess the file, then replay results at original speed; display actual processing speed separately.
- Pausing playback freezes playback-driven sound/timer presentation. Seeking reconstructs state from saved results, rather than accumulating new timers or duplicate incidents.
- An incomplete/failed job is marked incomplete; retain its unresolved incidents. End-of-file does not mean a missing person is safe, and timers do not advance beyond available footage.
- Decode errors or a stopped worker produce a separate processing warning. Brightness or detector confidence alone cannot certify usable pool visibility.
- Exclude edited/camera-cut footage from continuous tracking evaluation, or split it manually into independent segments. Do not promise automatic cut detection in the MVP.

## 6. Data: the first dependency to settle

Within the first two hours, secure at least one usable **recorded** fixed-camera clip. Prefer an elevated corner view showing the pool and its exits, stable framing, enough resolution for heads, and several minutes of ordinary activity. Longer ordinary footage matters more for false-alert measurement than a short dramatic rescue clip.

Use permissioned existing recordings or ordinary consented swimming footage. Do not require children to stage danger, prolonged breath holds, or acted drowning. Use artificial detection gaps for event-engine tests, clearly labeled as injected tests; these do not validate visual recognition of submersion.

The prior footage search did not establish a permission-cleared, long, fixed-corner home-pool test set. Public availability is not permission to download, train on, redistribute, or present identifiable footage. A public dataset also needs its own terms checked. Do not make the schedule depend on finding a perfect dataset later.

Create `data/manifest.csv` with clip ID, local path, source/permission, duration, resolution, camera/session ID, split, and scenario notes. Keep raw video outside git. Split by recording session before extracting frames. Target separate development and held-out sessions; if only one session exists, call the result a same-session demonstration, not independent validation.

Label observable events: visible person, pool-region membership, last-seen time, observed exit, occlusion, ambiguous identity, and presence/lost-contact intervals. For a small detection audit, annotate approximately 100 diverse frames if time permits, including distant/partial people and negatives. This is a workload target, not a statistically sufficient sample size. Record unobservable or ambiguous examples rather than inventing labels.

### Training gate

No training is required for the MVP. Only attempt **one person-detector fine-tune** if the end-to-end baseline works, a GPU is confirmed, and reviewed pool labels plus an untouched evaluation split are ready by hour 12. Include deck/full-body and partial-body examples; do not restrict all labels to upper bodies. Compare with the pretrained baseline and keep the better complete pipeline. Do not attempt pose fine-tuning without keypoint annotations.

If those conditions fail, spend the time on footage quality, evaluation, event logic, and demo reliability. Do not start simultaneous RF-DETR, YOLO26, MediaPipe, SAM, and NVIDIA comparisons.

## 7. Integration contract and artifacts

```text
Recorded video + polygon + armed configuration
    -> decoder with source timestamps
    -> YOLO person detection -> ByteTrack
    -> persistent observation/incident registry
    -> frames.jsonl + events.json + run.json
    -> dashboard: synchronized video, overlay, reasons, audio, review
```

Agree on these fields before splitting work:

| Record | Required fields |
|---|---|
| Run | `job_id`, input hash, duration, dimensions, model/package versions, device, sampling/config, processing status and measured speed |
| Frame | `t_sec`, normalized `bbox_xyxy`, `track_id`, detection confidence, approximate zone membership; optional keypoints with confidence/missing mask |
| Incident | `event_id`, type, associated track references, `start_t_sec`, `trigger_t_sec`, last-seen box/time, severity, reason, open/resolved state, resolution reason |
| Review | `event_id`, acknowledged flag, reviewer disposition and timestamp; preserve original model/event outputs |

Use stable event IDs independent of tracker IDs. Absence from tracker output must not delete a registry incident. Store coordinates relative to original video dimensions so resizing the UI does not shift overlays.

Minimal API: `POST /jobs` (upload and config), `GET /jobs/{id}` (status/progress), `GET /jobs/{id}/results`, `GET /jobs/{id}/video` (seekable video), and `POST /jobs/{id}/events/{event_id}/review`. Poll job progress; WebSockets and a production database are unnecessary. Keep playback JSON indexed by timestamp; do not load unbounded frame histories into the browser.

Save run outputs locally in an ignored artifact directory. Commit the code, configuration examples, label guide, manifest without sensitive paths, and evaluation summary. Serve backend requests through the frontend development proxy to keep setup simple. For evidence replay, seek within the original video; clip export is optional.

## 8. Owners and 24-hour schedule

Suggested named assignments based on the research contributions; the team should confirm availability at kickoff. Reallocate the simulation role to integration because simulation is no longer on the critical path.

| Owner | Primary responsibility | First deliverable |
|---|---|---|
| Sam | Detector/tracker adapter and bounded pose experiment | Timestamped detections on the actual laptop, runtime measurements |
| Sunny | Video processing, backend, run artifacts, integration | One video job producing the agreed results contract |
| Joanne | Footage permissions, manifest, labels, small-person audit | Usable clips and session split before model experimentation |
| Rohan | Incident engine, scenario tests, evaluation | Registry/timer behavior on deterministic detection sequences |
| Dpark | Dashboard, overlay synchronization, audio and review | UI against the agreed fixture, then actual backend output |

| Hours | Deliverable / gate | If blocked |
|---|---|---|
| 0–2 | Confirm clip, machine, environment, polygon and JSON contract. Run one inference. | No usable footage: use a clearly labeled non-pool fixture for integration while sourcing footage; pool CV remains unvalidated. CPU only: proceed without training. |
| 2–6 | Vertical slice: video → actual boxes/IDs → saved results → dashboard. | Reduce analyzed frame rate/resolution or preprocess offline. Show honest processing progress. |
| 6–12 | Both event types, persistent registry, synchronized evidence, sound, review controls. | Cut optional visual polish and pose. Fix integration before adding models. |
| 12–16 | One bounded improvement: pose comparison **or** eligible fine-tune; run failure-case tests. | Keep pretrained detector. Freeze feature additions by hour 16. |
| 16–20 | Held-out evaluation, manual error review, performance report. | Report missing coverage and counts honestly; fix functional bugs, document any rerun after a change. |
| 20–24 | Rehearse full demo, clean launch instructions, cache verified outputs, prepare limitation slide. | Fall back to clearly labeled preprocessed replay generated by the same pipeline. |

## 9. Definition of done and measurements

The build is complete when a new supported local clip can be processed, displayed with correctly aligned results, and reviewed without editing source code. Both event rules must work, and limitations must be visible in the presentation.

Required deterministic event tests:

1. A tracked person disappears inside the region: warning/alert occur at configured source-time thresholds even after tracker expiry.
2. A different person enters while someone is missing: restored population count does not clear the incident.
3. A continuous observed deck exit ends the visit; loss at the image/pool edge alone does not.
4. Ambiguous reacquisition leaves the old incident unresolved; acknowledgement does not falsely resolve it.
5. Pause, seek, playback speed change, end-of-file, decode failure, and worker failure do not fabricate extra observation time or mark unresolved people safe.
6. Missing pose landmarks do not remove a person or clear an incident.

Report the following on untouched footage, with numerator, denominator, recording duration, and scenario coverage:

- Visible-person recall on the annotated sample, broken down by partial visibility and distance; child subgroup only if such data actually exists.
- Presence/lost-contact event recall against observable event labels, with misses listed. This is not drowning recall.
- False alerts per hour of ordinary footage: false alerts divided by evaluated hours. Short recordings yield unstable estimates; zero in a few minutes is not proof of reliability.
- Trigger delay from the labeled rule-condition onset, plus processing throughput and wall-clock availability of results. Precomputed playback timing is not live detection latency.
- ID switches, duplicate events, unresolved cases needing manual review, and representative failures.

Do not invent accuracy targets from public model benchmarks. If a deployment claim requires a target, choose it before a larger evaluation and collect enough independent data to assess it. For this hackathon, the acceptance bar is a functioning, measured prototype with disclosed coverage.

## 10. Demo script and next phase

In approximately three minutes:

1. Load a fixed-camera recording, show the region and armed/audio status.
2. Replay ordinary activity with tracked people and a presence event.
3. Show a genuine lost-contact event if suitable footage exists. Otherwise explicitly show an injected detection-gap test and distinguish it from real video performance.
4. Show that a new entrant cannot cancel an unresolved missing-person incident; open its evidence and review it.
5. Present measured runtime, false-alert counts and limitations, including the visible-submerged-person failure. Show optional pose only if it passed its experiment gate.

After the hackathon, prioritize representative recordings and a validated surface-visibility signal before adding age-dependent alerts. Then compare detectors on the same data, test live ingestion and health handling, and evaluate whether simulation improves real held-out results. Consider commercial model licensing and hardware only against an actual deployment requirement.

**Final recommendation:** ship the recorded-video incident workflow in 24 hours. Treat underwater-state estimation, child-specific behavior, and simulation transfer as the next research milestones. This recommendation is based on repository inspection and documented model capabilities; no new model benchmark or independent Claude review was performed for this plan.
