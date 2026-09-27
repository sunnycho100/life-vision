# Baby Detector Research

updated: 2026-09-26

Goal: in a pool with several adults and babies/toddlers, detect **only babies** accurately, with a small real-time object detector (limited compute).

## Recommendation (TL;DR)

1. **Do not train a detector that finds "babies only".** Train it on everyone, then decide the age per track:
   - Stage 1: a high-recall `person` detector (class-agnostic).
   - Stage 2: an age decision per track, fused from:
     - a small crop classifier (`baby_0_4` vs `other`; MobileNetV4 / EfficientNet-B0 / DINOv2-S);
     - real-world size from the pool homography (stature on the deck; shoulder and torso size in the water);
     - an asymmetric vote over the track, biased toward baby.
   - Baseline to compare against: the same detector trained 2-class (`baby`, `other_person`).
   - Why: the literature shows age cues are weak once in the water, and a detector-only 2-class model collapses when data is small (child AP50 0.39). The person prior from COCO transfers well, and the crop classifier retrains in minutes.
2. **Detector candidates**, chosen by the deployment hardware:
   - Jetson / GPU / Mac: **RF-DETR-N or S** (Apache 2.0, best fine-tuning transfer and occlusion handling). Comparison baseline: YOLO26-s.
   - Pi 5 + Hailo: **YOLO26-n or s** (DETR family does not compile on Hailo). AGPL. A permissive alternative is YOLOX-S.
   - Also worth trying: D-FINE-N/S (COCO weights, Apache 2.0).
   - Excluded: DEIMv2 / EdgeCrafter (non-commercial since 2026-08), YOLO-NAS weights (non-commercial), YOLO-World (GPL).
3. **Do not deploy SAM 3 / Grounding DINO / VLMs.** Use them only for **offline auto-labeling**. There is no zero-shot benchmark for "toddler vs adult", and indirect evidence (HumanRef attributes, CLIP age audits) says they are unreliable.
4. **Classify on the deck, then carry the label into the water.** On the deck the full body is visible, so stature separates toddlers from adults clearly (tallest 5.5-year-old 124 cm, shortest 17.5-year-old 150 cm).
5. **Alarm logic should not depend on re-ID.** Use the per-track lost timer, a baby-count check, and a "baby in the pool area with no adult" alert. The ASTM F3698-24 and CPSC direction is detecting water entry.
6. **Hardware and settings:** Jetson Orin Nano Super ($399), 960x544 input (`rect=True`), TensorRT FP16, BoT-SORT or OC-SORT.

## Relationship to v1 decision
`technical-summary.md` says "no adult vs baby split in v1". This design keeps that pipeline intact: the person detector + tracker + underwater timer stays the same for everyone. The age decision is an **attribute layer added on top**, so v1 does not change. Only the baby-specific rules sit on top: a shorter threshold, the "no adult present" alert, and the count check.

## Data and labeling pipeline
1. Record consented sessions from the real camera pose (lighting, splash, time of day). Sample 1-2 fps and deduplicate.
2. Offline pseudo-labels: SAM 3 (Ultralytics `SAM3SemanticPredictor`, "person" / "swimmer" / "head in water") or Grounded-SAM-2 produces **person tracks**. Union with a COCO person detector at low confidence.
3. Age votes per track, from an ensemble:
   - a Qwen3-VL-4B/8B crop VQA, given the candidate box (box hints raised HumanRef attributes from 54 to 81);
   - PINTO [DEIMv2-Wholebody34](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34) `child` attribute (child AP 0.57-0.79 at S and above; README says Apache 2.0 but the base is DEIMv2, so **use it as an offline teacher only**);
   - SAM 3 "toddler" scores;
   - homography size.
4. Tracks where the votes disagree go to a **human for a per-track decision** (CVAT or X-AnyLabeling). One decision labels hundreds of frames.
5. Optional synthetic data: insert toddlers into our own pool frames with Qwen-Image-Edit-2509 (Apache 2.0). Anny (Apache 2.0, all-ages body model) for Isaac Sim. SMIL / AGORA kids / BEDLAM are non-commercial. Pretrain on synthetic, fine-tune on real.
6. Pretrain on public child/adult sets (Ultralytics child-adult 9.6k, Daycare, CDD), then fine-tune on our pool data.
7. Evaluation: split by **session or day**, never random frames. Headline metric: **baby track recall at a fixed false alarms per hour**. Also time to first detection, and confusion on 5-10-year-olds.

## Next experiments
1. Film a 10-minute test at our pool, deck plus water, with an adult and a child actor (or a doll / CAMI-like dummy).
2. Measure zero-shot SAM 3 / Qwen3-VL / PINTO child on 100-200 frames. This sets how much the labeling pipeline can automate.
3. Train RF-DETR-S vs YOLO26-s at the same 960 input on the same data, and compare baby-track recall.
4. Build the homography and check how well the size feature separates babies from adults.

## Details

### Detectors (COCO, T4 TensorRT FP16)
| Model | COCO AP50:95 | RF100-VL AP50:95 | Latency | Params | License |
|---|---|---|---|---|---|
| YOLO26-N | 40.3 | 52.0 | 1.7 ms | 2.6M | AGPL-3.0 |
| YOLO26-S | 47.7 | 57.0 | 2.6 ms | 9.4M | AGPL-3.0 |
| RF-DETR-N | 48.4 | 57.7 | 2.3 ms | 30.5M | Apache 2.0 |
| RF-DETR-S | 53.0 | 60.2 | 3.5 ms | 32.1M | Apache 2.0 |
| D-FINE-N | 42.8 | 58.2 | 2.12 ms | 4M | Apache 2.0 (COCO-only weights) |
| D-FINE-S | 48.5 | 60.3 | 3.49 ms | 10M | Apache 2.0 (COCO-only weights) |
| YOLO11-S | 47.0 | 56.2 | 2.5 ms | 9.4M | AGPL-3.0 |
| DEIMv2-S | 50.9 | - | 5.78 ms | 9.7M | **Non-commercial since 2026-08-24** |

Sources: [RF-DETR benchmarks](https://rfdetr.roboflow.com/latest/learn/benchmarks/), [RF-DETR paper](https://arxiv.org/html/2511.09554), [D-FINE](https://github.com/Peterande/D-FINE), [DEIMv2 license](https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/LICENSE.md)

- RF100-VL measures how well a model fine-tunes to new domains. That matters more for us than COCO. DETR-family models (RF-DETR, D-FINE) beat YOLO at every size there.
- Small-data tests (50-200 images): RF-DETR-N beat YOLO on occluded objects and box tightness, but trained about 4x slower with about 3x the VRAM.
- Ultralytics (YOLO11/YOLO26) is AGPL-3.0. A closed commercial product needs an Enterprise license. Fine-tuned weights are still covered.
- DEIMv2 and EdgeCrafter moved to a non-commercial license on 2026-08-24 (verified in the repo commit "docs: update license information").
- Objects365 pretraining is "academic purpose only", and RF-DETR, YOLO26, and D-FINE `obj365` weights use it. Get legal review before a commercial launch. D-FINE `*_coco` weights are the cleanest.

### Edge hardware decides the family
- DETR-family models (RF-DETR, D-FINE, RT-DETR) need `grid_sample` / deformable attention, and **Hailo does not support these ops**. They also run slowly on the Pi CPU (RF-DETR-N about 2.4 FPS).
- Jetson Orin Nano: RF-DETR-N about 95-100 FPS as a raw TensorRT engine, about 25 FPS in a full Python pipeline.
- Pi 5 + Hailo-8L: YOLO26-n 111 FPS, YOLO26-s 66.6 FPS. Official HEF export exists.
- Permissive CNN fallback for Hailo: YOLOX-S (Apache 2.0, 95.8 FPS on Pi 5 + Hailo-8).

### Detector shortlist
1. **RF-DETR-N/S**: best transfer and occlusion handling, Apache 2.0. Needs Jetson, a GPU, or CoreML.
2. **YOLO26-n/s**: best edge path (Hailo, Pi, Android) and easiest tooling. Has a small-object assigner (STAL). AGPL.
3. **D-FINE-N/S with COCO weights**: RF-DETR-level transfer with 1/3 the params. Apache 2.0. Research-grade code.

### Foundation models (offline labeling only)
- SAM 3 (Meta, Nov 2025; SAM 3.1 Mar 2026): text prompt such as "child" returns boxes, masks, and track IDs for every instance in images and video. Too heavy for the camera, good for pseudo-labeling. [paper](https://arxiv.org/abs/2511.16719)

### Literature on child vs adult (summary)
- **No paper does "toddler vs adult, above-water pool camera".** Pool papers classify behavior (drowning / swimming, mostly with adult actors). Child-detection papers are on land.
- Pool papers almost always use YOLO n/s fine-tuned on small self-collected sets and report 86-98% mAP@50. These numbers are inflated: single site, and frames from the same video end up in both train and test.
- Three approaches:
  1. **One detector with 2 classes** (child / adult): cheapest. In-domain child detector 0.95 mAP@50 vs 0.61 off-the-shelf ([YOLOCDD](https://ieeexplore.ieee.org/document/11078264)). With too little data the child class collapses to AP50 about 0.39 ([ski-lift repo](https://github.com/JovanSk/ski-lift-safety-detection)).
  2. **Detector then crop classifier**: age from a body crop alone still works (MiVOLO body-only MAE 6.66 years). A child pipeline with track-level fusion worked ([arXiv 2608.14770](https://arxiv.org/pdf/2608.14770)).
  3. **Geometry / stature**: hand-made head/body ratios are weak (68-75%). But **stature** separates groups well: the tallest 5.5-year-old is 124 cm and the shortest 17.5-year-old is 150 cm ([CPSC](https://www.cpsc.gov/s3fs-public/pdfs/blk_media_childfromadult.pdf)). A pool-plane homography gives real-world size in meters.
- **Key insight: classify on the deck, then keep the label into the water.** On the deck the full body is visible and height separates toddlers from adults clearly. In the water only head and shoulders show, so the cues get weak. The tracker carries the label from the deck into the water.
- **PoolScout** already sells "toddlers (up to 4) vs other people/pets" and an "unattended toddler" alert (no adult in frame). This is the closest competitor to our goal.
- **ASTM F3698-24**: alarm within 30 s. The test uses a CAMI toddler dummy (water entry; motionless on the bottom for 20 s). CPSC is pushing to add perimeter-entry and roll-in tests, so **detecting a child entering the water is key**.
- Hard cases: an adult sitting or crouching on the steps, a child held by an adult (overlapping boxes), kids 5-10 years old, dolls, floats, pool robots, skin-tone bias.

### Datasets
- Open Images V7 has boxable classes `Boy`, `Girl`, `Man`, `Woman`, `Person` (no `Baby` box class). MIAP adds age presentation labels (young / middle / older) to person boxes. [MIAP](https://research.google/blog/a-step-toward-more-inclusive-people-annotations-in-the-open-images-extended-dataset/)
- Infant swimming dataset, 7000 images from 84 videos, infants in swim rings, used with YOLOv5 and Faster R-CNN. [Kaggle](https://www.kaggle.com/datasets/meizhiqiang/datasetof-infants-swimming)
- Roboflow Universe has many "drowning / swimming" datasets, mostly small and noisy. [search](https://universe.roboflow.com/search?q=class%3Adrowning)
- Child/adult 2-class sets (on land, good for pretraining): [Ultralytics child-adult](https://platform.ultralytics.com/joo-vyctor/datasets/child-adult-detection) 9,652 images, Roboflow "Child-Adult Classification" 9,498 images, [Roboflow Daycare](https://universe.roboflow.com/daycare/daycare) about 3,300 images (22k child boxes), [CDD](https://www.kaggle.com/datasets/samueldiop/child-detection-dataset) 1,928 images (child only). Check each license.
- LVIS folds baby/child into `person`. Objects365 and COCO have `person` only. CrowdHuman is non-commercial.
- The Kaggle infant set was partly scraped from Douyin / Xiaohongshu, so its license is doubtful. Research use only.
- **No public dataset shows toddlers in a backyard pool from a high camera with adults present.** We have to collect our own.

### Hardware (Sept 2026 prices)
| Option | Price | Runs | Notes |
|---|---|---|---|
| Jetson Orin Nano Super | $399 (was $249 before the July 2026 increase) | YOLO and DETR family (TensorRT) | Only cheap board where RF-DETR / D-FINE can be compared on-device. Python end-to-end @640: YOLO11n 19 ms, D-FINE-N 24 ms, RF-DETR-N 39 ms |
| Pi 5 4GB + AI HAT+ 26 TOPS (Hailo-8) | about $220 | CNN YOLO only | YOLO26n about 97 FPS on the Pi |
| Laptop (Intel NPU / Apple ANE) | $0 | YOLO; RF-DETR via CoreML (experimental) | Good for prototyping |
| Coral, RK3588 | - | - | Coral is discontinued. RK3588 has no DETR support |

### Training compute
- YOLO11s/26s @640, 3k images: about 2-3 h for 100 epochs on a T4 (estimate). 960 px is about 2.25x.
- RF-DETR-S: T4 batch 4 x grad_accum 4, roughly 3-8 h (estimate).
- Free: Kaggle 2xT4 about 30 h/week, Colab T4, Lightning AI about 80 h/month interruptible, Roboflow free tier (RF-DETR N/S/M training credits).

### Resolution for small babies
- A 40 px baby in a 1080p frame is about 24 px at 640 input, and a 20 px one about 13 px. That is too small.
- Start at **960 px with `rect=True`** (about 960x544). Mount the camera high and angled so babies stay above about 30 px.
- Only if small-baby recall is poor: add a P2 head (YOLO11-P2; no YOLO26 P2 yet) or ROI/SAHI tiling (about 6x compute at 1080p).

### Tracker
- Pool looks like DanceTrack: people look alike and move non-linearly. OC-SORT, Hybrid-SORT, and BoT-SORT beat ByteTrack there.
- Start with Ultralytics BoT-SORT (ReID off, camera-motion compensation off for a fixed camera) or OC-SORT, then try Hybrid-SORT via boxmot (AGPL).
- **No online tracker keeps an ID through 5-15 s fully underwater followed by coming up elsewhere.** So the alarm should not depend on re-ID:
  1. Per track: a baby track lost for N s, with its last box inside the pool region, raises the alarm.
  2. Count check: the number of babies seen in the last T s is greater than the number visible now, for N s.
  3. A new baby track within radius R, or the count recovering, clears the lost track.
- Class flicker: track one `person` class and treat baby/adult as a **per-track weighted vote**. Make it asymmetric: fewer votes are needed to lock "baby" than "adult".

### Open-vocab models (summary)
| Model | Use | License |
|---|---|---|
| SAM 3 / 3.1 | Offline person tracks and pseudo-labels. About 4 GB VRAM | SAM License (commercial OK, some uses prohibited) |
| Qwen3-VL 2B-8B | Age VQA on crops given boxes | Apache 2.0 |
| Grounding DINO 1.5/1.6, DINO-X | API only | Paid |
| OWLv2 / Florence-2 | Backup labelers | Apache 2.0 / MIT |
| YOLOE / YOLO-World | Fast, but unreliable for "toddler" zero-shot | AGPL / GPL |

Label tooling: X-AnyLabeling (local, SAM 3 video) or CVAT (SAM 3 text prompts), Autodistill (SAM 3 to RF-DETR / YOLO).

### Caveats
- Training time and 960 px FPS numbers are extrapolated estimates.
- Several dataset licenses (Kaggle, Roboflow, Figshare) need a manual check.
- Filming and synthesizing children needs consent, access control, and no public release.
- This is an assistive product, not a replacement for supervision or barriers.
