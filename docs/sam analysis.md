# SAM analysis: pool monitoring and hackathon feasibility

Date: 2026-09-26

Repository reviewed: `sunnycho100/we-fall-we-die` at `1b650fab3975e60f61b982d7221c6f875b4dd3ba`

Status: design review and proposed implementation plan; no measured model results.

## 1. Decision in brief

**Pursue a narrowly scoped pool-monitoring prototype. Do not present the current design as validated drowning detection.** The problem is meaningful and easy to explain. A five-person team could plausibly demonstrate a complete workflow in a 24–48-hour hackathon, assuming access to representative footage and working inference hardware. Reliable deployment around real swimmers is a substantially larger undertaking.

The strongest project is not the one with the most models. It is the one that demonstrates a specific improvement over a basic detector-and-timer baseline, explains uncertainty, and reports results honestly. The user requested a hackathon assessment, but the event duration is unknown: 24–48 hours is a planning assumption, not a repository fact. For a multi-week research project, revisit the deferred pose and simulation experiments.

| Question | Assessment | What would change the assessment |
|---|---|---|
| Is the problem valuable? | Yes: alerting a nearby responsible person can be useful. | User interviews and operational tests must establish that alerts are actionable. |
| Is the idea novel? | Limited novelty at category level; commercial camera-based pool alerts already exist. | Demonstrated improvements in difficult cases, installation, privacy, or reliability. |
| Is a hackathon demo feasible? | Yes, with one camera, existing models, and a small event engine. | Lack of representative footage or poor baseline visibility could block it. |
| Is the proposed timer a drowning detector? | No: detector disappearance is neither necessary nor sufficient evidence of submersion. | Additional observations and validation are required, not simply a longer timer. |
| Should SAM 3D be central? | Not initially. Its contribution to this task is unproven. | A controlled experiment showing useful gains at acceptable latency. |
| Should simulation be central? | Not for a weekend prototype. | A longer research timeline and a real-data evaluation that isolates its benefit. |

## 2. What was actually inspected

The reviewed commit contains thirteen Markdown documents and `.gitignore`, including five placeholder member-research READMEs. There is no application code, model checkpoint, dependency manifest, dataset, or executable test suite. Consequently, this review establishes design risks, not observed software defects or measured performance.

The main inputs are [the technical summary](technical-summary.md), [research notes](research-notes.md), and the folder READMEs.

There are two competing plans:

1. **Technical summary:** person detector → multi-object tracker → disappearance timer → alert.
2. **Research notes:** simulated motion → pose features → temporal classifier → real-data fine-tuning.

These require different labels, interfaces, and evaluation. Select one primary pipeline before assigning work. The recommendation here is the first pipeline with explicit uncertainty and event logic, while treating pose and simulation as optional experiments.

## 3. The central limitation: visibility is not submersion

The current proposal starts an underwater timer when a tracked person disappears. This conflates several different observations:

- A visible person can be missed because of glare, splashing, partial visibility, or detector error.
- Someone can disappear behind another swimmer or leave the camera view.
- A submerged person can remain visible and continue to receive a person detection.
- A person who was never detected has no established track and therefore no disappearance timer.
- Drowning-related distress can occur while someone remains visible; disappearance logic does not cover that case.

The pipeline therefore cannot infer “underwater for five seconds” merely from “no detection for five seconds.” Label this signal **time since reliable visual contact**, unless separate evidence supports a more specific interpretation.

Separating head visibility from whole-person detection may help, but is still a hypothesis. A head box does not establish that the mouth and nose are above water. Reflections, viewpoint, and an uncertain waterline remain important.

### Proposed fixes that need safeguards

| Proposed shortcut | Counterexample | Recommendation |
|---|---|---|
| Clear an old missing track when head count recovers | Another person enters while the original swimmer remains missing. | Maintain unresolved events per person. Use count only as supporting evidence. |
| Ignore disappearances at the pool edge | A swimmer can submerge beside the wall. | Close a track as an exit only when an exit transition is actually observed. |
| Associate the nearest new track | Two swimmers cross or surface close together. | Preserve ambiguity when multiple matches are plausible. |
| Use swimsuit or hair appearance as identity | Similar clothing, wet hair, glare, and partial views make appearance ambiguous. | Combine cues; do not assume appearance re-identification solves the problem. |
| Increase the timer until false alarms disappear | This delays genuine unresolved-event alerts and leaves the observation problem intact. | Evaluate the alert-delay/false-alarm tradeoff on held-out recordings. |

Image distance is not automatically physical distance. A “few meters” gate needs calibration; otherwise describe and tune the gate in image coordinates with perspective limitations documented.

## 4. Where SAM 3D fits

### Model purpose

| Model | Documented purpose | Relevance here |
|---|---|---|
| SAM 3D Objects | Reconstruct object geometry, texture, and layout from an image. | No clear need in the minimum pool-alert pipeline. |
| SAM 3D Body | Recover a 3D human mesh and pose from an image. | Possible pose-feature experiment, not direct evidence of drowning or airway position. |

Sources: [Meta SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), [Meta SAM 3D Body](https://github.com/facebookresearch/sam-3d-body).

A reconstructed hidden limb or body surface is a model estimate. It must not be treated as a measurement of what happened under the water. We have no benchmark establishing SAM 3D Body's accuracy, temporal stability, or runtime on this project's camera views.

### License and setup

Meta releases the code and weights under the custom SAM License. It grants royalty-free rights to use, modify, and distribute the materials subject to its terms, including redistribution conditions and use restrictions. Commercial use is generally permitted under those terms; it is not an unrestricted MIT/Apache-style release. Check dependent components separately before distribution. [Official license](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE)

The official **Objects** setup lists Linux and an NVIDIA GPU with at least 32 GB VRAM, with gated checkpoint access through Hugging Face. Do not transfer that hardware requirement to Body without checking Body's own setup and benchmarking it. [Objects setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md), [Body installation](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md).

### Decision rule

First build and evaluate a 2D baseline. Add SAM 3D Body only as a separate experiment on identical held-out clips. Measure event outcomes, end-to-end latency, and failure cases. Keep it only if the improvement is meaningful for the actual alert task. A compelling mesh visualization alone is not a detection improvement.

### Separate opportunity: 2D SAM for annotation

SAM 2 and SAM 3 are distinct from SAM 3D. Their image/video segmentation capabilities could help prepare masks or annotations offline, without putting them in the live alert pipeline. Whether they reduce labeling time on reflective pool footage is untested. Try a small timed comparison against manual annotation and review every generated mask. Segmentation still does not establish underwater status or danger. [SAM 2](https://github.com/facebookresearch/sam2), [SAM 3](https://github.com/facebookresearch/sam3).

## 5. Recommended hackathon scope

Suggested pitch:

> A pool-monitoring prototype that flags unexpected entry and prolonged loss of visual contact, shows the last known location, and provides an immediate incident replay.

This accurately describes observable events. It does not claim to diagnose drowning or replace supervision.

### Include

- One fixed camera or clearly labeled prerecorded stream.
- A manually drawn pool region and explicit exit regions.
- One existing person detector and tracker, selected by performance on representative clips.
- Per-person event state, timestamp-based timers, and explicit uncertainty.
- Separate entry-alert and swimming-monitoring modes.
- Local audible alert, last-seen marker, event reason, and short replay.
- Camera/stream health monitoring.
- A reproducible comparison against the simple disappearance-timer baseline.

### Defer

- Adult/child classification or inferred adult supervision.
- Drowning diagnosis, automated rescue claims, or guaranteed safety thresholds.
- SAM 3D, realistic water simulation, and custom temporal-model training.
- Multiple cameras, factory falls, and hands-in-machinery detection.
- Automatic retraining from dashboard feedback.
- Production camera-speaker integration unless already supported and tested.

Presence of an adult in the image does not establish that the adult is supervising. Use explicit arming controls instead of inferring responsibility from age or appearance.

For a tighter weekend scope, start with prerecorded inference and one primary mode. Add live streaming and a second mode only after the replay pipeline works. Entry mode can include a manually defined approach zone, allowing notification before water entry; evaluate nuisance alerts separately. Always display and log the armed state, and provide a re-arm reminder after disarming.

## 6. Architecture and event behavior

```text
Video source and timestamps
          |
          +--> Stream health checks --> Monitoring unavailable warning
          |
          +--> Short rolling video buffer
          |
     Detector and tracker
          |
     Pool and exit observations
          |
     Persistent per-person event state
          |
     Alert, last-seen marker, and replay
          |
     Human review labels, stored for later curation
```

### Suggested state semantics

| State | Meaning | Transition guidance |
|---|---|---|
| Visible | A person has reliable recent visual evidence. | Refresh last-seen time and location. |
| Temporarily obscured | Evidence is briefly missing or ambiguous. | Preserve history; avoid reacting to every missed frame. |
| Unresolved disappearance | Visual contact has not been restored within the configured interval. | Alert with an uncertainty-aware reason. |
| Observed exit | Evidence supports movement out through an exit region. | Close the active monitoring episode and record why. |
| Monitoring unavailable (orthogonal health status) | The source is stale, disconnected, or otherwise unusable. | Raise a health warning alongside existing person states; do not overwrite or clear unresolved events. |

Keep event identity separate from a tracker's temporary ID. A tracker can delete a lost track after a short buffer, but the unresolved event must survive. Avoid automatically resolving an event because the tracker created a new ID.

Use media timestamps to evaluate event timing in replayed footage. For live input, map trusted capture timestamps into a local monotonic time base where possible and distinguish capture, receipt, and processing time. If only receipt time is available, report that upstream latency is unknown. A separate wall-clock watchdog must continue while frames stop arriving; lack of new frames must not freeze pending-event handling. Preserve existing alerts and flag lost monitoring during unresolved events. Do not treat unobserved time as confirmed submersion. Do not derive elapsed time solely from frame count.

Camera health also includes scene coverage: a bumped camera invalidates drawn zones. A reference view or manual alignment check can expose displacement. Glare, darkness, and blur may make an active stream unusable; automatic detection of these conditions is additional work, not an assumed capability.

An acknowledgement means someone saw the alert, not that the person is safe. Record acknowledgement and resolution separately. Deduplicate repeated notifications into one incident while continuing to show elapsed time.

For the demo, browser or laptop audio is a practical target. Sound through a physical camera depends on its specific hardware and API and should not be promised without testing.

Include an explicit enable-audio interaction and test-alert button; verify behavior with the actual browser, speaker, and power settings. Keep live timers in the backend rather than relying on a background browser tab. Test alert delivery end to end. Track unresolved-event count and manual clearing burden as usability costs, not merely false-alert counts.

### Minimal event contract

Agree on `event_id`, `camera_id`, `track_reference`, `event_type`, `state`, `last_seen_timestamp`, `event_timestamp`, `last_seen_image_position`, `reason`, `clip_reference`, and acknowledgement/resolution timestamps. Specify time units and coordinate conventions. A detector confidence score must not be displayed as a probability of drowning.

## 7. Data and evaluation

### Labels should describe observable facts

Label pool entry, observed exit, last reliable visual contact, visible resurfacing, occlusion, uncertain identity, and unusable footage. Include an explicit uncertain/unobservable label. Do not label acted motion as medically confirmed drowning.

Replace the repository's blanket “upper body only” rule with representative camera-view coverage: heads, partial bodies, full bodies entering/exiting, crowded views, empty pools, reflections, toys, and difficult lighting. Full-body examples are not inherently useless; the issue is matching the target deployment distribution.

The statement “no real drowning videos exist” is too absolute. A defensible statement is: **this project has not established access to a suitable, licensed, ethically usable, representative dataset of real drowning events.** Public availability alone does not establish quality, permission, or suitability.

### Essential evaluation cases

| Case | What it tests |
|---|---|
| Normal exit from the pool | Whether observed exits prevent unnecessary alerts. |
| Two swimmers crossing | Identity changes and temporary occlusion. |
| Disappearance followed by resurfacing elsewhere | Whether reacquisition works without careless reassignment. |
| New entrant while another swimmer is missing | Whether head count incorrectly clears an unresolved event. |
| Splashing, glare, and reflections | Robustness to ordinary pool conditions. |
| Visible submerged person | The limitation of disappearance-only logic. |
| Person never detected before an event | Dependence on track initialization. |
| Frozen, delayed, or disconnected stream | Whether monitoring failure is exposed. |
| Empty pool and normal play | Background false-alert burden. |

These cases are coverage requirements, not instructions to act out dangerous events. Use permitted existing clips where available and dry-land stand-ins or injected track sequences for logic tests. A rescue manikin may support an equipment/visibility test under appropriate supervision, but is not evidence of human detection accuracy. Mark unavailable real-video cases as untested. Shortening a timer in a demo tests software behavior only and must be disclosed; it cannot validate a real-world safety threshold.

### Experimental protocol

1. Establish a simple detector/tracker/disappearance-timer baseline.
2. Choose parameters using development recordings only.
3. Freeze parameters before evaluating held-out recordings.
4. Split by recording session and, where possible, by people and camera placement. Adjacent frames from one video must not cross training and test boundaries.
5. Evaluate the baseline and improved system on exactly the same recordings.
6. Save event predictions and human labels so metrics can be reproduced.

Report event recall as a numerator and denominator, false alerts per monitored hour, alert-delay distribution, identity-related failures, processing latency, and source availability. Define event matching, duplicate handling, and alert onset before scoring. An undetected event is a miss, not a sample to omit from latency reporting without explanation.

Always report hours and event counts alongside percentages. Zero false alerts in a short recording is weak evidence of reliability. Report normal activity and staged events separately; performance on adults acting in one pool does not establish performance on toddlers or different pools.

These metrics measure the defined visibility/entry events, not medically confirmed drowning. Have two annotators independently label a subset, compare event and timestamp agreement, and resolve disagreements before scoring. Report appropriate uncertainty intervals when sample size allows meaningful interpretation. If independent sessions are unavailable, call the run an exploratory smoke test and state the leakage/generalization limitation. Preselect demonstration cases before final scoring and include an improved-system failure as well as a success.

Reviewed alerts alone create a biased dataset: they omit events the system missed. Sample ordinary footage and manually audit for misses as well. Store review feedback for curation; do not automatically retrain on every confirmation click.

## 8. Simulation: useful research, poor default critical path

The research notes propose MuJoCo motion and Isaac Sim rendering. Before investing in either, ask what failure the synthetic data is supposed to fix.

- Perfect simulated skeletons do not reproduce real pose-estimation errors, missing keypoints, or identity changes.
- Scripted “drowning” movement is a behavioral assumption, not validated drowning physiology.
- For a detector-and-timer pipeline, synthetic skeletons do not directly improve image detection without an additional training bridge.
- Water appearance, camera angle, partial visibility, and occlusion may dominate the task more than motion realism.

If the team later studies simulation, compare real-only versus real-plus-synthetic training on an unchanged real test set. Hold architecture and tuning effort comparable. The proposed real-data fraction curve is a useful research question, but synthetic benefit must be measured rather than assumed.

For the hackathon, deterministic track/event sequences can test state logic cheaply. Label these as software tests, not evidence of real-world perception accuracy.

## 9. Team plan and stopping rules

Assumption: five people and roughly 48 hours. Adjust to the actual event schedule.

| Owner | Primary deliverable |
|---|---|
| 1 | Representative footage, labels, difficult cases, and held-out split. |
| 2 | Detector/tracker baseline and visual overlays. |
| 3 | Event state, identity ambiguity handling, and evaluation script. |
| 4 | Stream ingest, health checks, rolling buffer, and event API. |
| 5 | Dashboard, audio, replay, and demo narrative. |

**First 4 hours:** agree on labels and API; run a baseline on target footage. If the camera cannot reliably observe relevant people, change the view or narrow the demonstration. Do not assume fine-tuning will rescue unusable observations.

Arrange footage access before the hackathon; the first four hours are not a realistic window to organize a safe pool shoot. Agree on baseline acceptance criteria before viewing evaluation results. If labeling is the bottleneck, have the frontend owner help once the mock-event UI works.

**Hours 4–16:** connect a complete vertical slice from video to dashboard alert. Frontend and backend can use clearly labeled mock events until inference is ready.

**Hours 16–32:** add exit evidence, persistent unresolved events, stream-health checks, and challenging clips. Evaluate against the baseline.

**Hours 32–40:** freeze scope and parameters; run the held-out evaluation. Drop optional features that have not shown task value.

**Final 8 hours:** fix integration failures, rehearse, prepare reproducible playback, and document results and limitations. Do not claim live inference if showing prerecorded output.

Proceed only if the team has representative footage, can run a baseline, and can explain what an alert means. If those conditions fail, pivot to a clearly limited entry/visibility monitoring demo rather than presenting unsupported drowning detection.

## 10. Hackathon strength and differentiation

Existing products make the broad category familiar. SwamCam advertises AI human-presence pool alerts; MYLO describes above-water and underwater cameras. These are vendor descriptions, not independent accuracy evidence. [SwamCam](https://www.theswamcam.com/), [MYLO](https://coralmylo.com/how-it-works/).

A strong presentation should show:

1. One concrete failure of the basic timer.
2. The same sequence handled better by the proposed event logic.
3. A genuine alert with last-seen location and replay.
4. A short, reproducible results table with denominators.
5. One unresolved limitation and the next experiment needed.

Local processing can be a useful privacy feature if implemented, but it is not automatically novel. Likewise, incident clips and dashboards support usability; they are not themselves a research contribution.

Use a parent-facing name that conveys care and assistance. The repository's current name may be memorable internally but is poorly suited to a sensitive consumer-safety pitch.

## 11. Recording safety and deployment boundaries

The technical summary proposes friends staying underwater for 10+ seconds, while the research notes say no breath holding. Remove the duration challenge from the recording plan. Prefer appropriately licensed existing footage, dry-land tracking/occlusion tests, and professionally supervised recordings without dangerous enactments. CDC specifically warns against prolonged underwater breath holding and emphasizes close supervision. [CDC prevention guidance](https://www.cdc.gov/drowning/prevention/index.html).

Do not stage drowning, use children as risk-event subjects, or treat shallow water as sufficient protection. A prototype is an additional experimental signal, not a replacement for supervision or established pool protections.

For any real recording, obtain venue permission and participant consent, use a qualified lifeguard dedicated to supervision, and agree on stop procedures. Minimize bystander capture. Identifiable children in ordinary footage require appropriate guardian permission. Prefer local storage, define retention/deletion, and do not upload identifiable footage to shared services without explicit permission. Exclude raw footage from version control and verify that exclusion before adding data.

For any later deployment, separately validate coverage, outages, alert delivery, operator response, privacy, retention, and applicable product requirements. A repository mention of a standard does not demonstrate compliance, and this review does not verify the standards claims in the older research notes.

## 12. Decisions to resolve before implementation

- Actual hackathon duration, judging criteria, and available hardware.
- Camera location, representative footage, and permission to use it.
- Whether the demo is entry monitoring, unresolved visual contact, or an explicitly separate research experiment.
- Detector/tracker choice based on a local benchmark and license review.
- Observable event definitions, alert thresholds, and resolution rules.
- Who responds to an alert and whether that response is realistically immediate.
- Whether processing is local, and what clips are retained.

## 13. Review provenance

This document distinguishes repository observations, externally documented capabilities, and proposed engineering choices. No benchmark, clinical validation, certification, or real-world safety claim is implied. External source links were checked during the review; implementation-specific results remain to be collected.

### Independent Claude review and disposition

The requested review completed through the local Claude Code CLI installed alongside the VS Code Claude Code extension, using explicit model `claude-opus-5-5` and `--effort xhigh`. The API result confirmed the canonical model `claude-opus-5-5`; the invocation specified the effort. This was a separate read-only CLI session, not an exchange inside an existing VS Code chat. Claude read the draft and the two planning documents. It did not independently verify external sources or run implementation benchmarks.

Accepted findings were incorporated: make the 48-hour assumption explicit; keep health status separate from person state; keep a live watchdog running through feed loss; distinguish capture/receipt/processing clocks; provide safe alternatives and untested labels for difficult scenarios; consider 2D SAM for offline annotation; expose armed state and audio readiness; measure manual-resolution burden; and tighten small-sample evaluation and annotation reliability.

Two qualifications matter. Claude questioned the hackathon premise because it was absent from the repository, but that premise came from the user's request; only the duration was assumed. Its suggestions for additional supervised underwater recordings were not adopted as required test collection: missing footage remains a documented coverage gap rather than a reason to stage risk. External SAM hardware and license statements remain grounded in the linked Meta sources, not Claude's review.
