# Research Comparison

updated: 2026-09-26

Summary of what each person committed under `docs/`, where we agree, and where we differ.
Rule: **3 of 5 agree → adopt. Split or everyone different → discuss.**
"n/a" means the person did not cover the topic, which is not a vote either way.

| Who | Doc | Focus |
|---|---|---|
| Sunny (sunnycho100) | `sunnycho100/00` to `05` | Detector, training pipeline, tracking, MuJoCo sim, Isaac Sim |
| Joanne (joannemiki57) | `baby-detector-research.md` | Baby vs adult, detector benchmarks, hardware, datasets |
| Sam (samkwak188) | `sam analysis.md` | Design review, SAM 3D, hackathon scope, evaluation, YOLO Pose vs MediaPipe |
| Rohan (rsusarla3) | `cv-feasibility.md`, `detection-logic-research.md` | Feasibility, NVIDIA tools, ASTM and CPSC standards, thresholds, trackers |
| David (dpark) | `frontend-interface-plan.md` | Product flow, risk levels, UI, camera roadmap |

## Adopt (3 or more agree)

| # | Decision | Sunny | Joanne | Sam | Rohan | David |
|---|---|---|---|---|---|---|
| 1 | **Detector uses one `person` class.** Everyone gets the same alarm rule. | Yes | Yes (age is a layer on top) | Yes | Yes | Yes (not baby-specific) |
| 2 | **"Person missing" is not the same as "underwater".** The timer needs a head-level signal, with "lost inside the pool" as a backup. | Yes (head class) | n/a | Yes (call it "time since reliable visual contact") | Yes (head above, below, unknown) | Partly (shows "head not visible" timer) |
| 3 | **The underwater timer lives outside the tracker.** Our own registry of missing people. Tracker IDs are not trusted through a dive. | Partly (long track buffer) | Yes | Yes | Yes | n/a |
| 4 | **Entry alert as a second feature.** Most toddler deaths start with an unnoticed entry (CPSC: 62%). | Yes (in summary) | Yes | Yes | Yes | n/a |
| 5 | **YOLO as the first baseline.** Compare others on the same footage afterwards. | Yes | Compare (RF-DETR-S vs YOLO26-s) | Yes | Yes | n/a |
| 6 | **Foundation models only offline, for labeling.** Never in the live pipeline. | Yes (Grounding DINO) | Yes (SAM 3, Qwen3-VL) | Yes (SAM 2/3) | n/a | n/a |
| 7 | **SAM 3 as the auto-labeler** (over Grounding DINO) | Grounding DINO | SAM 3 | SAM 2/3 | n/a | n/a |
| 8 | **Split train and test by recording session**, never by random frames | Yes | Yes | Yes | n/a | n/a |
| 9 | **Input resolution 960 px** for small, far heads | Yes | Yes | n/a | n/a | n/a |
| 10 | **Jetson Orin Nano** as the edge box | Yes | Yes | n/a | Yes | n/a |
| 11 | **Prerecorded video first**, live cameras later | n/a | n/a | Yes | Yes | Yes |
| 12 | **Thresholds are configurable**, and demo values are labeled as demo | Yes | Yes | Yes | Yes | Yes |
| 13 | **No long breath holds when recording.** Use a mannequin or dummy for the "on the bottom" test. | Yes | Yes (doll or dummy) | Yes | Yes | n/a |
| 14 | **Pose and "active distress" are a second layer**, not the core alarm | Yes | n/a | Yes | Yes | No (in the prototype) |
| 15 | **Simulation is parallel work**, not the critical path | Partly | Optional | Yes | Parallel | n/a |

Notes:
- #2 changes the technical summary. Core logic step 3 becomes "head not above water for N seconds, or lost inside the pool".
- #7 has only 3 votes cast, 2 of them for SAM 3. Weak majority, easy to switch.
- #9 and #10 have few voters but nobody disagrees.
- #13 contradicts `technical-summary.md` ("friends underwater for 10+ seconds"). That line should be removed.

## Discuss (split or unclear)

### A. Alarm thresholds
| | Warning | Alarm | Other |
|---|---|---|---|
| Sunny | 3 s | 5 s | |
| Rohan | 5 s | 12 s | 8 s if under and not moving |
| Joanne | | | Shorter for babies |
| Sam, David | | | Tune on held-out footage, configurable |

Rohan's numbers come from real products and the standard (MYLO 12-15 s, ASTM test 20 s, CPSC says 20 s is too late). Sunny's 5 s would fire on normal breath holding. **Suggestion:** start from Rohan's 5 / 12 / 8.

### B. When to add age (baby vs adult)
- **Joanne:** per-track age attribute now. Crop classifier, real-world size from the pool geometry, and classify on the deck then carry the label into the water.
- **Sunny:** v2.
- **Sam:** defer. Never suppress an alert because someone looks adult.
- **Rohan:** size only as a rough signal for entry alerts.

Everyone agrees on a single-class detector (#1). The question is whether the age layer is in the first build. Joanne's "classify on the deck" idea is the strongest argument for doing it early.

### C. "Child with no adult nearby" alert
- **Joanne, Rohan:** alert when a small person is in the pool area with no adult near.
- **Sam:** an adult in the frame does not mean someone is supervising. Use an explicit "armed" switch instead of guessing.

This decides how the entry alert (#4) is triggered.

### D. Head count and "came up somewhere else"
- **Sunny, Rohan, Joanne:** merge a new track near where someone went under, and clear the alarm if the head count recovers.
- **Sam:** risky. A new person entering the pool can wrongly clear a real missing person. Keep the event open, use the count only as supporting evidence, and calibrate distances.

3 votes for the merge rule, but Sam's counterexample is a missed-drowning case. **Suggestion:** adopt the merge, with Sam's safeguard that a track entering from outside the pool never clears a missing person.

### E. Tracker
- **Sunny:** ByteTrack with a long buffer.
- **Joanne:** BoT-SORT (ReID off) or OC-SORT. Pools look like DanceTrack, where ByteTrack does worse.
- **Rohan:** try BoT-SORT with ReID. NvDCF if we get an NVIDIA GPU.
- **Sam:** no pick. Event state must not depend on tracker IDs.

No majority. Low stakes because of #3. Benchmark ByteTrack vs BoT-SORT on our footage.

### F. Commercial model: YOLO vs RF-DETR
- **Joanne:** RF-DETR-N/S (Apache 2.0, better fine-tuning transfer). YOLO26 only for Pi + Hailo. Also flags that Objects365-pretrained weights are academic-only.
- **Sunny:** YOLO first, RF-DETR as the commercial fallback.
- **Sam, Rohan:** YOLO for speed, check licenses later.

Depends on whether we ever sell this. Decide together.

### G. Active distress detection in the prototype
- **David:** part of the deliverable (repetitive arm motion, no forward progress).
- **Sam, Rohan, Sunny:** stretch goal, after the head timer works.

Majority says stretch, but it's David's UI spec. Check what the yellow state shows if distress detection is not ready.

### H. How much simulation
- **Sunny:** MuJoCo motion and Isaac Sim images as a real part of the project.
- **Sam:** not on the critical path. Scripted drowning is an assumption, and sim skeletons lack real pose errors.
- **Rohan:** Isaac Sim + Replicator in parallel. Also found **SwimXYZ** (3.4M synthetic swimming frames with joints).
- **Joanne:** optional synthetic toddlers (Anny body model, image editing).

This ties to what we want on our resumes (the meeting said ML and simulation matter most). Decide how much time goes here.

### I. Project framing and timeline
- **Sam and David** both plan around a 24-48 hour hackathon. Nobody has confirmed the real timeline.
- **Sam:** don't call it drowning detection. Say "unexpected entry and prolonged loss of visual contact". Also suggests a friendlier name than "we-fall-we-die".
- **David:** keeps the name. Wants a deliberately cheesy Matrix animation, which got pushback in the meeting.

## New things only one person found
- **Joanne:** "classify on the deck, carry the label into the water". Also: **PoolScout** already sells toddler vs adult and "unattended toddler" alerts, the closest competitor. DETR models don't run on Hailo. DEIMv2 went non-commercial.
- **Sam:** event identity separate from tracker ID. Stream health and a "monitoring unavailable" state. Keep capture, receipt, and processing timestamps apart. Test cases like "new swimmer enters while another is missing".
- **Rohan:** ASTM F3698-24 details and the CPSC letter. Night limits (IR cameras can't see underwater). A "low visibility" warning that the standard requires. NVIDIA models (PeopleNet, BodyPoseNet, PoseClassificationNet). Datasets: SwimXYZ and CrowdHuman heads.
- **David:** camera roadmap (upload, webcam, RTSP, then Ring and Nest). Notification preferences. Incident replay and export.
- **Sunny:** MuJoCo sim with buoyancy (swim, float, drowning response, silent sink, 5-person pool scene). The YOLO11-LiB repo has public weights and a dataset.

## Proposed next steps
1. 10-minute call on A, B, C, D, H, and the real timeline (I).
2. Update `technical-summary.md` with the adopted decisions, especially #2 and #13.
3. Get footage: public datasets now (Roboflow pool sets, YOLO11-LiB, figshare underwater), plus one safe recording session.
4. Run pretrained YOLO person detection on that footage before any training. This is Sam's first step, and nobody disagrees.
