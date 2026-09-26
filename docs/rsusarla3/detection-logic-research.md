# Detection Logic Research: Timers, Standards, Datasets, Trackers

author: Rohan (@rsusarla3)
updated: 2026-09-26
status: research only, nothing measured yet. Builds on [cv-feasibility.md](cv-feasibility.md) and [samkwak188's SAM analysis](../samkwak188/sam%20analysis.md), which already explains why "disappeared from detector" is not the same as "underwater". This doc adds external evidence and a concrete logic proposal.

## TL;DR
1. **The industry signal is "head under the surface + not moving", not "person missing".** Coral MYLO alarms when a person is motionless with the head under the surface. Poseidon separates "drowning" from "just still".
2. **Real numbers to anchor our thresholds:** MYLO needs about 12-15 s motionless at the bottom. The ASTM F3698-24 test is a toddler dummy on the bottom for 20 s. CPSC staff say 20 s is **too late** to support a rescue.
3. **Entry alerts matter more than drowning alerts for toddlers.** In CPSC data, 62% of pool drownings of children under 5 happened when the adult lost track of where the child was. CPSC wants a toddler **entry** alert and a **perimeter** alert added to the standard.
4. **Ultralytics' ByteTrack deletes a lost track after 30 frames (1 s at 30 fps).** Someone underwater for 5 s loses their track ID completely. **The underwater timer must live outside the tracker.**
5. **Public data exists**, but almost none of it is overhead backyard footage with time-based labels. Use it to pretrain, and use our own footage for testing.
6. **Night is a hard limit.** Water absorbs near-infrared strongly, so a regular IR security camera can't see underwater at night. MYLO says it can't monitor in total darkness. Scope our demo to daylight or a lit pool.

---

## 1. How the real products decide "drowning"

| Product | Cameras | Trigger | Time to alarm |
|---|---|---|---|
| **Coral MYLO** (home, meets ASTM F3698-24) | Unit in one pool corner. Above-water and underwater cameras + water pressure sensor | 3 levels: **presence** (someone approaches or enters the water after it was unused for several minutes), **pre-drowning** (distress signs), **salient drowning** (motionless, head under the surface) | About **12-15 s** motionless at the bottom. Entry alert needs about half the body (waist down) in the water for **5-10 s** |
| **Poseidon** (public pools, Maytronics) | Underwater + overhead. Overhead uses an IR camera paired with a normal camera | Separates "drowning" from "still in the water" | Under 10 s to detect, then a configurable delay before alerting lifeguards |

MYLO's published limits, which we will share:
- Doesn't work in total darkness.
- Needs clear water. It warns the user when murk hides the far end of the pool.
- Designed for water at least 60 cm (2 ft) deep.
- One unit covers about 10 m (30 ft) of pool, with a 110-120° view.
- Pets can trigger false alarms (4 limbs).
- It has a "false alarm" button that teaches the system. That's the same idea as our frontend confirm/false-alarm loop.

**Takeaway:** use their 3-level structure. It maps onto our design directly and is easy to explain to judges.

## 2. The standard: ASTM F3698-24 and what CPSC thinks is missing

From the ASTM scope and the CPSC staff letter to ASTM (June 6, 2025):
- Covers systems that detect possible drowning and **signal within 30 s**. They must also **tell the user when visibility is low**.
- Drowning test (section 5.2.2.1): a toddler dummy (CAMI) completely submerged **on the pool bottom, not moving, for 20 s**, must set off the alarm.
- Entry test (section 5.2.1): the toddler dummy enters **feet first, upright, with head and shoulders above the water**.
- Loudness: pool unit 85 dBA at 9.8 ft, indoor unit 65 dBA at 3.2 ft. CPSC wants UL 2017 levels instead (85 dBA at 10 ft).

CPSC staff concerns:
- Children under 5, 2020-2022: **842** pool-related deaths. **62% (522)** happened when the adult lost contact with or knowledge of where the child was.
- A 20 s "motionless on the bottom" alarm "would not likely support a successful rescue", because a child in distress takes varying amounts of time to even reach the bottom.
- They want the ASTM F2208 toddler tests added: vertical feet-first entry, horizontal **roll-in**, and walking, crawling or running into the **perimeter** area.

**Takeaway for us:**
- Demo a **"visibility low" warning**. It's cheap (use frame brightness and detector confidence) and it's in the standard.
- Demo **entry and perimeter alerts**. That's where most toddler deaths are.
- Our drowning alarm should beat 20 s. Aim for a **warning at about 5-8 s** and an **alarm at about 10-15 s** of head-under, and tune from there. See section 5.

## 3. Drowning timing (for choosing thresholds)
- The **Instinctive Drowning Response** (Pia, 1974) usually lasts only about **20-60 s** before the person sinks. It is quiet: no waving, no shouting.
- Toddlers often skip the visible struggle altogether (see research-notes.md), so behavior models trained on adults acting it out will miss them. That's why the head timer is the backbone of the logic and the behavior classifier is only an add-on.
- A 20-60 s struggle window means a pre-drowning "distress" signal (upright body, head tilted back, arms pressing down, no forward progress) could come **before** submersion. That's the only way to beat the head timer. It's a stretch goal.

## 4. Public datasets

| Dataset | What | Use for us | Caveat |
|---|---|---|---|
| **Underwater Drowning Detection Dataset** (figshare 29497235) | 5,613 annotated **underwater** images from a controlled pool. Classes: Swimming / Struggling / Drowning. 640×640, YOLO labels | Pretrain a behavior classifier. Its "Struggling" class is our pre-drowning class | Underwater view, not our overhead one. Acted |
| **Roboflow "Swimming and Drowning Detection"** and "Pool Drowning Detection" | Mixed-angle images, Swimming / Drowning | Quick baseline. YOLOv8 got about 90% accuracy on it in one paper | Internet images from mixed views. Single frames, no time information |
| **YOLO11-LiB paper data** | 2,000 images → 4,800 augmented, Swimming / Drowning, overhead + eye level + underwater | Its label definitions are useful for our label guide (below) | The authors admit single-frame YOLO "cannot capture temporal features" of drowning |
| **SwimXYZ** | **3.4M synthetic frames**, 11,520 videos, 2D + 3D joints, SMPL motions, varied cameras and water | **P1 and P3:** pretrain pose models for bodies in water. Models fine-tuned on it did better on real swimming | Swimming strokes only, no drowning |
| **CrowdHuman** (heads) | 15k training images with **head boxes** | Pretrained YOLO head detectors exist (e.g. Owen718/Head-Detection-Yolov8) | Land crowds. Needs fine-tuning on heads at the water surface |

Label definitions from the YOLO11-LiB paper, which we can reuse in our label guide:
- **Swimming:** body streamlined and near-horizontal. Limbs rhythmic and coordinated. Head raised to breathe or dipped rhythmically. Moving in a clear direction.
- **Drowning:** body vertical or tilted upright. Not moving forward. Arms out to the sides or slapping the water. Head under for long periods. Little leg movement.

No public dataset combines **overhead backyard view + tracking IDs + head-under timing**. That gap is our contribution. Our own recorded clips are the test set.

## 5. Proposed detection logic (for the team to discuss)

### Signals per tracked person, every frame
- `head_state`: `above` / `below` / `unknown`. From a head detector plus a small above/below classifier on the head crop. `unknown` covers glare, a blocked view, or low confidence.
- `in_pool`: whether the body box center is inside the pool polygon, drawn once at setup.
- `motion`: how far the box center moved over the last ~1 s, normalized by box size.
- `size`: box height in meters, using the pool polygon for scale (the pool size is known). This is a rough small/large split for entry alerts only.

### Per-person state machine
```
            enters pool polygon
  OUTSIDE ───────────────────────▶ IN_WATER ◀────────────┐
     ▲                               │                   │ head above
     │ exits polygon at edge         │ head below        │ (any state)
     │ (not an alarm)                │ OR track lost     │
     │                               ▼ inside pool       │
     │                           SUBMERGED ──────────────┤
     │                               │ t ≥ T_warn        │
     │                               ▼                   │
     │                           WARNING  ───────────────┤
     │                               │ t ≥ T_alarm       │
     │                               │ (or t ≥ T_still   │
     │                               │  if not moving)   │
     │                               ▼                   │
     │                           ALARM ── human confirm/dismiss
```
Starting values, to tune on our footage:
- `T_warn` = 5 s: soft notification ("someone has been under for 5 s").
- `T_alarm` = 12 s: full alarm. That's between MYLO's 12-15 s and the ASTM 20 s test, and comes in under CPSC's "20 s is too late".
- `T_still` = 8 s: alarm sooner if the person is under **and** not moving. Motionless underwater is the strongest signal (MYLO, Poseidon).
- Short `unknown` gaps (under ~1 s) don't reset the timer. Only a confident `above` clears it.

### Scene-level rules
- **Entry alert:** a new track enters the pool polygon while no other tracks are near it. If `size` is small, it's high priority. This covers the CPSC 62% case.
- **Perimeter alert (optional):** a second polygon around the pool. A small person enters it with no large person nearby.
- **Head-count guard** (Known problem 1): when a new track appears within ~2 s and ~3 m of a SUBMERGED track's last position, merge it into the old ID and clear the timer. Only raise ALARM if the pool's head count is still lower than before the person went under.
- **Low-visibility warning:** if mean frame brightness is below a threshold, or average detector confidence drops for more than N seconds, tell the user "monitoring degraded". The standard requires this.

### The tracker problem (important for P3 and P4)
Ultralytics tracker defaults (`bytetrack.yaml`): `track_buffer: 30`, meaning a lost track is **deleted after 30 frames, about 1 s at 30 fps**. A person under for 5 s gets a **new ID** when they come up, and the old ID disappears from the tracker output. So:
- **Keep our own registry of "missing" IDs** with their last position, last box and the time they went missing. The timer runs on this registry, not on the tracker's output.
- We can raise `track_buffer` (e.g. 300 = 10 s) so IDs survive longer. The config comment warns this raises the risk of ID switches, so compare both settings on our footage.
- Ultralytics now also ships BoT-SORT (optional ReID), OC-SORT, Deep OC-SORT, FastTracker and TrackTrack. **BoT-SORT with `with_reid: True`** is the cheap first try for matching resurfacing people to their old ID.
- DeepStream's NvDCF (from cv-feasibility.md) also has occlusion handling built in, so it's worth comparing if we get an NVIDIA GPU.

## 6. Environment limits to state up front
- **Night:** water absorbs near-infrared (e.g. 850 nm) strongly. IR night-vision cameras see the surface but hardly anything below it. Scope the demo to daylight or a lit pool, and show the "monitoring degraded" warning at night.
- **Glare:** a high overhead mount plus a polarizing filter helps. Glare is the biggest source of false alarms reported in the review literature, along with ripples and splash.
- **Clear water cuts both ways:** we can see bodies on the bottom, which is good for "motionless" detection. But the detector keeps "seeing" a submerged person, so person presence can't be our underwater signal (same point as the SAM analysis).

## 7. Suggested evaluation (matches what the standard tests)
Record short clips (safely: shallow water, a spotter outside the water, no long breath holds) of:
1. Normal swimming, splashing, diving → count **false alarms per hour**.
2. A person goes under and comes up somewhere else → does the ID merge work?
3. Breath-hold play, 5-10 s → should get WARNING, not ALARM.
4. A **mannequin or weighted dummy** on the bottom (no people at risk) → **time to ALARM**, compared against ASTM's 20 s.
5. A small doll or child-sized dummy entering feet first and rolling in → **entry alert time**.
6. The same scenes at dusk → does the low-visibility warning fire?

Report: time to alarm (mean and max), false alarms per hour, and ID-merge success rate. Say clearly that there is no real drowning data.

## Sources
- ASTM F3698-24 scope: https://store.astm.org/f3698-24.html
- CPSC staff letter to ASTM F15.49 (June 6, 2025): https://www.cpsc.gov/s3fs-public/June-5-2025-CPSC-Letter-to-ASTM-F15-49-Computer-Vision-Pool-Alarms.pdf
- Coral MYLO FAQ: https://coralmylo.com/faq/ and how it works: https://coralmylo.com/how-it-works/
- Poseidon (Maytronics): https://www.maytronics.com.au/en-za/launch-of-poseidon-australia/
- Instinctive drowning response: https://en.wikipedia.org/wiki/Instinctive_drowning_response
- Underwater Drowning Detection Dataset: https://figshare.com/articles/dataset/Underwater_Drowning_Detection_Dataset/29497235
- YOLO11-LiB pool drowning model: https://pmc.ncbi.nlm.nih.gov/articles/PMC12431139/
- YOLOv8 drowning detection: https://www.etasr.com/index.php/ETASR/article/view/8834
- Multi-swimmer underwater drowning detection (ICIAP 2025): https://link.springer.com/chapter/10.1007/978-3-032-10192-1_48
- Review of drowning detection approaches: https://pmc.ncbi.nlm.nih.gov/articles/PMC10820385/
- SwimXYZ: https://arxiv.org/abs/2310.04360
- CrowdHuman: https://arxiv.org/pdf/1805.00123 and head detector: https://github.com/Owen718/Head-Detection-Yolov8
- Roboflow pool drowning data: https://universe.roboflow.com/pool-safety/pool-drowning-detection
- Ultralytics tracking docs: https://docs.ultralytics.com/modes/track/ and ByteTrack config: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml
- NIR absorption in water (underwater NIR imaging): https://arxiv.org/pdf/1402.1151
- NEPTUNE (above-water struggle detection, early work): https://arxiv.org/abs/1805.02530
