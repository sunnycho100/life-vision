# CV Feasibility: Underwater, Drowning, Babies vs Adults, NVIDIA Tools

author: Rohan
updated: 2026-09-26

Two questions:
1. Can we detect people going underwater and drowning, for both babies and adults?
2. Can we use NVIDIA models and tools for any of it?

Short answer: **yes to "underwater too long" for adults and kids, maybe to "drowning behavior", no to babies under 1 as a separate class.** NVIDIA has ready-made models for most of our pipeline.

---

## 1. What we can and can't detect

| Target | Feasible? | How | Main risk |
|---|---|---|---|
| Person in pool (adult) | Yes | Off-the-shelf person detector | Glare, splash, only upper body visible |
| Person in pool (toddler, ~1-4 yrs) | Yes, with care | Same detector, the COCO "person" class already includes children | Small in frame from a far camera. Keep camera high and close, or use a higher input resolution |
| Baby under 1 | Out of scope | n/a | Infant drownings are mostly bathtubs, not pools (see research-notes.md). Not our camera |
| **Head underwater too long** | **Yes, this is our v1** | Track heads, time how long each head is below the surface | See the "clear water problem" below |
| Drowning behavior (struggle, Instinctive Drowning Response) | Maybe, stretch goal | Pose keypoints + temporal classifier | Pose models are trained on land. Water hides the lower body and blurs arms. Our acted data is not real drowning |
| Toddler silent sinking | Yes, via the timer | No struggle to detect, so behavior models miss it. **The timer catches it** | Same as timer |
| Adult vs child split | Possible, not reliable | Size from calibrated pool geometry (known pool dimensions to meters) | Only upper body is visible, so height estimates are poor. Keep the v1 decision: same rule for everyone |
| Small person enters pool area, no adult nearby | Yes, easy | Detector + pool-zone polygon + size heuristic | This is the real toddler risk case (Known problem 4). Cheap, high value |

### Key point: the timer handles babies and adults the same way
Adults in trouble often struggle (arms pressing down, head tilted back). Toddlers often slip under silently. A behavior classifier trained on adult acting will miss the toddler case. A **"head not above water for N seconds"** timer does not care how the person went under, so it covers both. This backs up the v1 decision to not split adults and kids.

### The clear water problem (important for P3)
Our v1 logic says: "person disappears from the detector = underwater." In a **clear, calm home pool seen from above, the detector often still sees a submerged person** through the water. Then:
- A person lying on the bottom stays "detected" and **the timer never starts**. That is the worst failure: a missed drowning.
- With glare or splash, a person at the surface gets lost and the timer starts. That's a false alarm.

**Proposal:** do not time "person missing". Time **"head not above the surface"** instead.
- Detect **heads** as their own class, or take the head keypoint from a pose model.
- Label each head as `above` or `below` the water. Submerged heads look blurred, distorted, low contrast and blue-tinted, so a small classifier on the head crop should learn this. That's a good use for our own footage and Isaac Sim data.
- Timer per track ID: starts when the head goes `below` **or** the track is lost inside the pool zone. Stops when the head is `above` again.

This also helps with Known problem 1 (coming up somewhere else), because heads are what the head-count check should count.

### Camera placement matters more than the model
- **High, overhead, looking down** (mounted on the house, roof edge, or a pole): less glare, sees the bottom, less occlusion between swimmers. Best for us.
- **Low, side-on**: strong surface reflections, people block each other. Avoid.
- **Underwater** camera: best view of submerged bodies (this is what AngelEye and Coral do), but it means hardware in the pool. It's fine as a test rig for data, not the product.
- Polarizing filter on the lens cuts surface glare a lot. Cheap win.

### Honest limits to state in the demo
- We can't test on real drownings. All our test data is acted or simulated.
- Night, heavy splash, pool covers, and crowded pools will degrade detection.
- This is a supervision aid, not a replacement for watching kids. Home products are held to ASTM F3698-24.

---

## 2. NVIDIA tools we can use

All of these are free to download from **NVIDIA NGC** (catalog.ngc.nvidia.com). Check NGC for current versions before we commit to one.

### Pretrained models (TAO Toolkit / NGC)
| Model | What it does | Where it fits |
|---|---|---|
| **PeopleNet** (incl. the transformer version) | Detects person, face, bag. Trained on a lot of real CCTV-style footage | Person detector. Its **face** class is a decent proxy for "head above water" |
| **BodyPoseNet** | 2D body keypoints, multi-person | Head and arm keypoints for the head timer and struggle detection |
| **PoseClassificationNet** | Classifies actions from keypoint sequences | Exactly the "keypoints to temporal classifier" design in research-notes.md. We can fine-tune it on our sim + real keypoints |
| **ActionRecognitionNet** | Action recognition from video clips (RGB) | Alternative to pose-based. Needs more real data |
| **ReIdentificationNet** | Appearance embeddings to match the same person across tracks | Fix for Known problem 1 (resurfacing in another spot gets a new ID) |

**TAO Toolkit** is NVIDIA's fine-tuning toolkit for these. Fine-tune on our upper-body pool footage, then export to TensorRT.

### Tracking and deployment
- **DeepStream SDK**: a video pipeline (RTSP in, detect, track, output). It ships trackers out of the box: **IOU, NvSORT, NvDeepSORT, NvDCF**. NvDCF handles short occlusions well, which is what happens when someone ducks under briefly. This could be most of P4's backend.
- **TensorRT**: speeds up any model (including YOLO) on NVIDIA GPUs.
- **Jetson Orin Nano / Orin NX**: edge box that runs the whole thing next to the pool. This answers the "cloud vs on-device (privacy)" question: video never leaves the house.

### Synthetic data (P1)
- **Isaac Sim + Omniverse Replicator**: render labeled synthetic images and videos with automatic bounding boxes and keypoints. Human avatars and animation come from the `omni.anim.people` extension. This matches the "use human avatars, not robots" note in the technical summary.
- **NVIDIA Cosmos** (world foundation models, e.g. Cosmos Transfer): can restyle simple sim renders into more photoreal video. It could help close the "sim water looks fake" gap. This is a stretch goal and needs a big GPU.

### NVIDIA vs the usual open-source route
| | NVIDIA (PeopleNet + DeepStream) | Open source (Ultralytics YOLO + ByteTrack/BoT-SORT) |
|---|---|---|
| Time to first result | Slower: Docker, NGC, config files | Fast: `pip install`, a few lines of Python |
| Needs NVIDIA GPU | Yes, for DeepStream/TensorRT | No, runs on CPU/Mac too (slow) |
| Edge deployment | Best in class on Jetson | Works, via TensorRT export |
| Pose + action models | Included (BodyPoseNet, PoseClassificationNet) | YOLO-pose for keypoints, classifier is ours to write |
| License | NVIDIA model licenses, check for commercial use | Ultralytics is AGPL-3.0, which matters if we sell it |
| Demo "wow" | Isaac Sim + Jetson story is strong | Standard |

### Recommendation
1. **Now (hackathon speed):** Ultralytics YOLO (detect + pose) with its built-in ByteTrack on our test footage. Get the head timer working end to end first.
2. **In parallel, P1:** Isaac Sim + Replicator for labeled synthetic pool data with human avatars.
3. **If we have an NVIDIA GPU or Jetson:** swap in PeopleNet + DeepStream NvDCF and compare on the same footage. Use ReIdentificationNet for resurfacing.
4. **Stretch:** fine-tune PoseClassificationNet on sim + real keypoints for struggle detection.

Open question for the team: **who has an NVIDIA GPU (RTX card) or access to one?** Most of section 2 depends on it.

---

## 3. What this changes in the technical summary (proposed)
- Core logic step 3: time **"head not above water"**, not just "person missing".
- Add a **head class** (or head keypoint) and an **above/below water** label to the label guide.
- Add the **"small person in pool zone, no adult"** alert as a second feature.
- Detector/tracker shortlist: YOLO + ByteTrack vs PeopleNet + NvDCF.
