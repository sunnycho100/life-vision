# RF-DETR Nano: implementation, missed-person diagnosis, and tuning plan

Updated 2026-09-26. This supersedes the implementation status in the earlier planning notes. Findings below distinguish executed experiments, visual observations, research evidence, and proposed training. **Person detection is implemented; accurate detection of every swimmer is not established. No fine-tuning has run.**

## 1. Decision

Keep the verified pretrained RF-DETR Nano baseline. For immediate inspection, move the replay confidence slider from **0.50 to 0.20**: those candidates are already cached, so this does not require another analysis job. Treat the extra boxes as candidates for review. Do not call the resulting count a verified swimmer count.

The next accuracy experiment should compare **Nano versus Small, each full-frame and tiled**, with manually reviewed labels. The friend subsequently clarified that the 101-box image used **Small**, not Nano; the newly fetched code confirms that model choice. Fine-tune the best deployable baseline on complete, consistently labeled pool-person images after this comparison. Increasing Nano's input resolution alone did not produce a compelling candidate-count improvement in our local test. Small has not yet been executed in our local comparison.

The running RF-DETR browser milestone contains no joints, identities, drowning classifications, incident recording, or human-review queue. A separate synthetic YOLO/timer prototype now exists on upstream `main`; it is not integrated with this application. See section 9 for the branch-specific audit.

## 2. What runs now

**Recording upload → one background inference process → timestamped observations → normal-speed cached video replay.**

- FastAPI streams uploaded MP4/WebM files into server storage while hashing them. PyAV supplies decoded presentation timestamps; sampling targets five frames per video second.
- The browser draws 2D person boxes and confidence scores. Seeking retrieves cached observations; it does not start inference again. A sampled overlay expires after 0.3 seconds. An analyzed empty frame is zero; missing analysis is unavailable.
- Pool membership is separate from detection and currently uses box bottom-center inside the manual outline. Mapping expiry stops pool counts, not person detection. Optional 3D markers are approximate positions.
- Source/job hashes, preprocessing, model hashes and actual runtime are recorded. Cancellation preserves partial observations and marks the job incomplete.
- No pose/MediaPipe requests remain in the tested browser path. There is no application cap of 100 people, and a synthetic 150-box observation renders successfully.

Run instructions, API and annotation workflow: [frontend README](../../frontend/README.md). Local app: **http://127.0.0.1:5173**. Annotation tool: **http://127.0.0.1:5173/annotate.html**.

## 3. Audited model setup

| Item | Actual baseline |
|---|---|
| Model / package | Official COCO-pretrained `RFDETRNano`, `rfdetr==1.11.0` |
| Input | Native RGB frame, float32 / 255, square bilinear resize to 384 × 384, `antialias=False`, ImageNet normalization |
| Decoder | Sigmoid, official global top-300 query/class selection, then person filtering; no full-frame NMS |
| Class mapping | Pretrained sparse COCO **person = 1**, not YOLO's 0; no last-slot background removal |
| Stored / displayed confidence | Store >0.20; replay defaults to ≥0.50 and permits ≥0.20 |
| Deployment | ONNX Runtime 1.24.2 CPU, four threads, FP32; BASIC graph optimization on this Windows ARM machine |
| Official reference | Python 3.12 x64, PyTorch 2.10.0 CPU, RF-DETR 1.11.0, executed through Windows emulation |
| Resolution experiments | Official Python `predict(shape=(size,size))`; production ONNX remains fixed at 384 |

Checkpoint SHA-256: `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d`.

FP32 ONNX SHA-256: `fcc99ec4e85c88a553aa495b7df79892444990f25c2fb9d705066e8402f6035c`.

Reference video SHA-256: `66f488b08e2122e9d8195972a481af2931b0ea6586ed229208b02a195a488d2c`; 1280 × 720, 128.912 seconds, YouTube source `PuAfTA2wf7o`.

The five-frame reference comparison checks preprocessing, raw tensors, decoded scores and boxes. Native ARM preprocessing matched exactly; worst raw-logit difference was approximately 0.000439 and decoded box difference 0.00223 pixels. This is **numerical equivalence, not detection accuracy**. Default aggressive ONNX optimization failed the strict raw-output gate on ARM because encoder proposal ordering changed around ties; BASIC passed. The deployment refuses a runtime/adapter without a matching parity report. The x64 runtime independently passed using ALL optimization.

Sources: [pinned package release](https://pypi.org/project/rfdetr/1.11.0/), [official detection documentation](https://rfdetr.roboflow.com/latest/learn/run/detection/), [official export documentation](https://rfdetr.roboflow.com/latest/exports/onnx/). Installed 1.11.0 source and executed parity take precedence over examples from a newer documentation release.

## 4. Why people are missed

| Hypothesis | Evidence / status | Action |
|---|---|---|
| Confidence 0.50 rejects many candidates | **Confirmed:** first development frame has 3 boxes at 0.50, 12 at 0.35, 37 at 0.20. | Inspect 0.20 now; select the eventual threshold from labeled precision/recall. |
| Distant people lose detail in full-frame resizing | **Plausible and supported by the tiling response.** A hypothetical 10-pixel-wide head becomes 3 pixels wide after 1280→384 resizing. | Compare tiled inference on the same labeled frames; retain a full-frame pass for context. |
| Partial bodies, tubes, water and reflections differ from COCO examples | **Plausible domain shift**, not separately quantified. The contact sheet shows misses of clearly visible people as well as ambiguous water boxes. | Pool-specific visible-person labels and hard negatives; evaluate head-only and partial people separately. |
| Wrong RGB order, class mapping or ONNX decode | No discrepancy on the five tested reference frames. | Preserve parity checks for every export/runtime change. |
| A 100-person limit | Removed from the current path; model keeps its own 300-selection configuration. | Do not increase query count to explain a frame showing only 3 confident boxes. |
| Pool outline excludes people | Outline affects pool membership only; it does not suppress detection boxes. | Diagnose the global box count separately from the pool count. |
| Temporal gaps | Analysis is sampled at 5 Hz; rendering holds each sample for at most 0.3 seconds. | Distinguish an unavailable/stale observation from an analyzed frame with misses. No tracker fills gaps yet. |

The SDK's warnings about not loading separate DINOv2 weights are expected when loading the complete pretrained RF-DETR checkpoint; they are not evidence of random model weights. Its unoptimized-inference warning concerns speed. No half-precision change was made to chase recall.

## 5. Verifiable experiment loops already completed

### Loop A — verify integration before tuning

Frozen video/checkpoint hashes → official Python predictions → FP32 export → deployment parity → real browser boxes. Passed numerical and integration checks. The full 128.912-second clip completed **645 sampled observations in 139.01 seconds**, median inference **187.15 ms**, p95 **248.80 ms**. Whole-job throughput was **0.927 video seconds per wall-clock second**. Other checks ran during this pass; this is a recorded workstation measurement, not a controlled real-time or four-camera capacity claim.

### Loop B — confidence and tiling, fixed development frames

Six overlapping square crops plus a full-frame pass; each crop uses the same model. At 1280 × 720 the crop side is 493 pixels. Cross-pass NMS uses IoU 0.50; this suppression is separate from the unsuppressed full-frame RF output. Both model baselines receive identical crops. Inner crop-edge fragments are currently discarded; this heuristic may itself miss a person and must be audited.

| Experiment | Frame at 8.008 s: boxes at .20 / .35 / .50 | Total boxes across 16 dev frames at .20 / .35 / .50 | Mean stored time per dev frame |
|---|---:|---:|---:|
| RF-DETR full frame | 37 / 12 / 3 | 573 / 147 / 60 | 148.3 ms |
| RF-DETR full + six tiles | 95 / 35 / 12 | 1511 / 545 / 222 | 1090.9 ms |
| YOLOv8n full frame | 8 / 3 / 2 | 147 / 63 / 28 | 50.0 ms |
| YOLOv8n full + six tiles | 28 / 8 / 5 | 524 / 256 / 121 | 330.5 ms |

**These are candidate counts, not true-positive counts or unique identities.** Repeated observations of the same swimmer contribute repeatedly to totals. The RF tiled .20 image also contains overlapping foreground boxes and uncertain water detections. NMS cannot reliably distinguish all nested duplicates from two overlapping people. More boxes alone do not select a winner.

YOLO uses our pinned ONNX adapter with 640-pixel letterboxing and Pillow bilinear resize. It has not received the same official-SDK parity audit as RF-DETR, so treat it as a provisional baseline, not a definitive architecture ranking.

Reproduce the summary and qualitative image:

```powershell
.venv\Scripts\python.exe -m tools.summarize_detection_experiments --contact-sheet artifacts/evaluation/reference/rfdetr-contact-sheet.png
```

The generated `summary.json` includes source, model, prediction and annotation hashes. Only development frames are summarized; the eight locked frames are excluded.

### Loop C — larger input, same weights, official Python

One unmeasured warmup per shape; all 16 development frames; FP32 CPU, four threads. These timings use x64 Python emulation, unlike the native ARM ONNX timings above. Compare speeds **within this table**, not across the two runtimes.

| Nano input | First frame boxes at .20 / .35 / .50 | Dev totals at .20 / .35 / .50 | Median / mean inference |
|---|---:|---:|---:|
| 384 × 384 | 37 / 12 / 3 | 573 / 147 / 60 | 309.4 / 330.0 ms |
| 512 × 512 | 35 / 12 / 2 | 614 / 160 / 66 | 585.4 / 646.5 ms |
| 640 × 640 | 38 / 10 / 5 | 588 / 157 / 65 | 810.2 / 877.5 ms |

At .20, 512 added 7.2% candidates and 640 added 2.6% relative to 384, with substantially higher runtime. This neither proves nor disproves a recall improvement; labels are missing. It does rule out claiming that merely increasing resolution fixes this clip. Nano at 512 is **not** the RF-DETR Small checkpoint.

```powershell
artifacts\python-x64\python.exe -m tools.probe_rfdetr_resolution --output artifacts/evaluation/resolution-probe-new
```

The script uses only dev frames, checks their hashes, preserves existing experiment directories and saves actual predictions. Production ONNX/threshold defaults were not silently changed.

## 6. Research and repositories: what transfers to this project

This is a focused review of directly relevant primary sources, not an exhaustive survey of all repositories or papers.

| Source | Relevant finding | Practical choice / limitation |
|---|---|---|
| [RF-DETR paper](https://arxiv.org/html/2511.09554v1), [official repository](https://github.com/roboflow/rf-detr) | Studies resolution, queries and runtime tradeoffs; COCO and RF100-VL results. | Keep a pretrained architecture initially. Published GPU benchmarks do not establish crowded-pool recall or laptop speed. Our resolution ablation is the local evidence. |
| [SAHI, ICIP 2022](https://arxiv.org/abs/2202.06934), [official code](https://github.com/obss/sahi) | Sliced inference and sliced fine-tuning improve small-object detection in the tested aerial datasets. | Strong reason to test crops. It does not guarantee swimmer accuracy. Our simple six-crop implementation is SAHI-inspired, not a reproduction of every SAHI default or result. |
| [CrowdHuman paper](https://arxiv.org/abs/1805.00123), [dataset and terms](https://www.crowdhuman.org/download.html) | Provides separate head, visible-body and full-body annotations for crowded scenes. | A useful annotation design and research transfer source. Do not mix head-only and full-body boxes as if interchangeable. Data terms restrict use to noncommercial research/education and prohibit image redistribution; unsuitable as an assumed unrestricted product-training source. |
| [YOLO11-LiB pool paper](https://doi.org/10.3390/s25175552), [authors' release](https://github.com/Mibugi/Drowning-detection) | Pool-specific weights and two-class Swimming/Drowning image boxes; custom architecture. | Candidate data for person training after annotation/provenance review. The [existing local release audit](yolo11-lib-paper-analysis.md) found 2 exact test/train duplicate images and 4 validation/train duplicates. 1640 of 2000 images contain exactly one annotation: crowded-scene completeness is unproven. |
| [Z5cc drowning video repository](https://github.com/Z5cc/drowning-detection) | Provides a public-video manifest for temporal research. | Our exact source appears in its manifest. It cannot be treated as an independent test video for training that includes that source. Details and pinned reference are in [the earlier audit](ml-system-decision.md). |
| [Official RF-DETR dataset format](https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/), [training parameters](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/) | COCO/YOLO loading, learning rates, effective batch, resolution and validation options. | Use pinned 1.11.0 configuration validation. A custom single-class dataset changes the output class mapping; the current COCO adapter must not be reused blindly. |

The LiB paper's reported drowning AP is not person recall on this video. Re-splitting data cannot make an already pretrained checkpoint forget its original training examples. New training requires a rebuilt source-disjoint split; evaluation of existing weights requires separate sources.

## 7. Concrete fine-tuning recipe

### Define the target before collecting data

Train **one `person` class**, including deck occupants and swimmers. Pool membership remains geometry. Annotate the visible attributable extent consistently, including clearly visible submerged body parts; use a head-only box when only the head is observable. Do not invent submerged limbs. Record `head_only`, `partial`, `occluded` and size groups for evaluation. A head detector would be a separate later target with separate matching and person/head association.

Start with a proposed pilot of roughly **100–300 diverse, fully reviewed frames** from multiple permitted recordings, then examine learning curves and failure cases. This is a budgeting suggestion, not an evidence-based minimum or promise of accuracy. Crowd scenes can contain many thousands of boxes even at that frame count. Include distant heads, tubes, splash, glare, underwater visibility, deck people, empty water and hard negative objects.

Keep our 24-frame development/evaluation bundle out of training. Split new data by **source video and camera/session**, before augmentation or cropping. Prefer a completely different pool for the final generalization test. Hash exact duplicates, review near duplicates and source provenance, and assign all crops/adjacent frames from a session to one split. A filename-level random split is inadequate.

LiB's two human categories may be collapsed into `person` only after verifying and completing all visible-person boxes. Missing swimmers in a training image can be learned as background. Our pinned loader filters `iscrowd=1` boxes before training; an ignore flag is not a verified loss mask. Initially omit or crop out genuinely inseparable regions from training, while retaining explicit ignore regions for evaluation.

### Train a bounded pilot on a compatible GPU machine

The machine currently runs native ONNX inference but has no verified CUDA training environment. Do not plan the hackathon around ARM/x64-emulated CPU training. Use a compatible GPU machine once available; measure memory and epoch time with a one-epoch smoke run before estimating a full run.

The proposed [pilot configuration](../../configs/rfdetr-pool-person-pilot.json) holds Nano at 384 to isolate the effect of training data. It uses 30 maximum epochs, AdamW, the pinned default learning rates, physical batch 2 × accumulation 8 (effective 16 on one GPU), seed 42, EMA, and a step decay at epoch 20. Multiscale is disabled for this initial controlled comparison. It does not automatically run the test split. These are **starting settings**, not optimized hyperparameters or completed training.

Install `rfdetr[train]==1.11.0` with PyTorch/Torchvision compatible with the GPU driver in a separate environment. Prepare `train/`, `valid/`, `test/`, each with images and `_annotations.coco.json`. Use the same single category in all splits. After completing the source/annotation audit, the training call is:

```python
import json
from pathlib import Path
from importlib.metadata import version
import torch
from rfdetr import RFDETRNano

if __name__ == "__main__":  # required for safe Windows worker startup
    assert version("rfdetr") == "1.11.0"
    assert torch.cuda.is_available(), "Use the compatible training machine"
    settings = json.loads(Path("configs/rfdetr-pool-person-pilot.json").read_text())
    model = RFDETRNano(
        pretrain_weights="artifacts/models/rfdetr-nano/rf-detr-nano.pth",
        resolution=384,
        device="cuda",
    )
    model.train(dataset_dir="datasets/pool-person-reviewed", output_dir="artifacts/train/pool-pilot-42", **settings)
```

First run with `epochs=1` and a fresh output directory to check boxes/classes, loss, memory and validation. Then run the 30-epoch pilot. If memory fails, reduce physical batch to 1 and raise accumulation to 16. If validation worsens, inspect label completeness and distribution before changing architecture; one subsequent controlled experiment can lower both learning rates by half. Preserve configs, dataset hashes, logs and checkpoint hashes. A longer training budget or stronger augmentation needs evidence from the validation curve, not a fixed claim that 30 epochs suffice.

Do not use the production `person_id=1` adapter for these custom weights. Pinned custom/Roboflow loading remaps categories to contiguous zero-based labels; a single `person` class is normally output 0. Read the trained checkpoint's mapping, update the export/manifest contract, repeat Python/ONNX parity, then rerun accuracy and UI tests. The current strict adapter deliberately rejects incompatible manifests.

### Evaluate and promote in separate steps

1. Label all 16 existing dev frames and eight locked frames, without using predictions as unreviewed ground truth. Keep locked images/results out of tuning decisions. The bundled annotator starts with no model prelabels.
2. Compare pretrained full-frame, tiled and fine-tuned candidates with one-to-one IoU 0.50 matching. Report precision/recall, false positives, duplicates, tiny/partial/head-only misses, pool-membership errors and time. Consider two cross-tile merge settings (.50/.70) only on dev frames; inspect false merges in crowds.
3. Select the threshold ≥0.20 that maximizes dev recall subject to precision ≥90%. This is an initial engineering screen, not a safety standard. Record failure if no configuration qualifies. Freeze the model, preprocessing, merge and threshold before locked evaluation.
4. Retain the best accurate-enough configuration that fits the measured processing budget. Seven passes cost approximately seven times the inference work here. If tiles help, evaluate fewer far-pool crops as a new dev experiment before shipping; do not claim that optimization has been measured yet.
5. Evaluate another camera/source before calling the detector universal. Tracking may later improve continuity, but it cannot establish an unseen person's presence or drowning status.

## 8. Verification status and remaining work

Completed: official-reference parity, full-clip processing, real RF browser playback, seek/cache behavior, browser/decoder timestamp checks including variable-frame-rate input, cancellation and single-worker conflict, corrupt-upload handling, empty/unavailable distinction, 150-box rendering, mobile layout and no pose requests. The evaluator also rejects development-label changes after policy freeze.

Accuracy is **unmeasured**: all 24 current annotation templates remain `reviewed: false`. Threshold selection refuses these templates. There is no selected precision-qualified threshold, no locally trained pool checkpoint and no independent-camera result. The friend has now supplied the screenshot settings and upstream code, but the exact input image and benchmark artifacts have not been verified locally. Its `101` count is not ground truth.

Local evidence: `artifacts/models/rfdetr-nano/deployment-parity-win-arm64.json`, `artifacts/full-reference-job.json`, `artifacts/rfdetr-browser-results.json`, `artifacts/rfdetr-job-acceptance.json`, `artifacts/evaluation/reference/summary.json`, `artifacts/evaluation/resolution-probe/`. Large media, weights, environments and generated images remain outside Git. [Committed-size verification summary](rfdetr-verification.json) preserves the measurements and hashes without distributing the video.

Small reporting and regression-test tasks were delegated to the available GPT-6 Luna with the user's approval. No new Claude consultation or different model was claimed for this tuning pass.

## 9. Newly pushed model code and the friend's settings

Fetched and inspected `origin/main` at **`2dfd33d6da45ec795cd35e47a683ea48d9f59199`**, without merging it over the working `sam` branch. The repository has substantially different frontends on these branches. Remote `sam` also contains a newer reference-download change; this audit does not imply that all remote changes were merged into the running server.

### The 101-box image was a different pipeline

Verified source: [model/detect_people_rfdetr_s.py at the inspected commit](https://github.com/sunnycho100/we-fall-we-die/blob/2dfd33d6da45ec795cd35e47a683ea48d9f59199/model/detect_people_rfdetr_s.py).

| Setting | Running browser baseline | Friend's reported 101-box run | New script default |
|---|---|---|---|
| Model | Nano, 384 | Small, 512 | Small, 512 |
| Threshold | display .50, cached >.20 | .25 | .30 |
| Passes | 1 full frame | 1 full + 24 crops | 1 full + 6 crops |
| Grid / overlap | none | 6 × 4 / .25 | 3 × 2 / .25 |
| Crop interpolation | none | 3× OpenCV bicubic before SDK resize | 2× bicubic |
| Merge | no full-frame NMS | NMS .50, containment .80, fragment .35/.40 | same three filters |
| Device | explicit native ARM CPU ONNX | reported M4 MPS GPU | SDK auto-select; no explicit override |

The code matches the described BGR→RGB/PIL input and class-1 filter. Tile width is `int(W / (nx - (nx-1)*overlap))`, with an analogous height; origins step by tile size × `(1-overlap)`. This differs from our six **square** crops. To request her described settings in an environment with this upstream file and its dependencies:

```text
python model/detect_people_rfdetr_s.py FRAME.png --out OUTPUT.png --thr 0.25 --grid 6 4 --overlap 0.25 --upscale 3
```

This is a reproduction command, **not a statement that it was run locally**. The script does not pin package/checkpoint hashes, expose a device option, export boxes/scores/metadata, measure runtime, or validate failed image reads. Before comparing results, add those outputs and pin the environment. Its 3× interpolation does not create new visual detail; the SDK still resizes the crop to 512. Test upscale 1 versus 3 independently rather than assuming a threefold model resolution.

Containment can remove a real smaller swimmer overlapping a larger person: a contained small box can have low IoU yet 100% intersection-over-smaller-area. Likewise the fragment rule is geometric, not evidence of a duplicate. Keep the full unmerged candidates, compare NMS alone versus each extra filter on labeled dev frames, and report true-person removals. Do not copy the three-image tuning rounds as an independent evaluation result.

I found neither `detbench.py` nor `draw_people.py` by those exact names in the fetched `main`/`sam` trees. The pushed single-image script is named `detect_people_rfdetr_s.py`; the available `benchmark_isaac.py` is a **different** synthetic-video benchmark. The A/B/C sampling, rotation correction and COCOeval details are therefore friend-reported, not independently verified from those missing files.

### What to correct in the benchmark interpretation

- The reported 42 ms Nano / 70 ms Small M4 MPS times cannot rank against CPU YOLO/YOLOX/D-FINE/RT-DETRv2 times. Re-time **all models on explicit CPU** as the common baseline, then separately compare supported accelerators on the deployment machine. Record warmups, median/p95, decoding/preprocessing inclusion and full tiling/merging costs. Synchronize MPS around GPU timing; otherwise asynchronous work may distort a measurement. [PyTorch MPS synchronization](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html)
- Changing the description of the device does not retroactively change saved accuracy predictions, but a CPU rerun is not guaranteed bit-identical. Save per-image outputs and recompute metrics after changing runtime. We have not independently verified her accuracy numbers or splits.
- Prediction-assisted rotation is part of the evaluated pipeline. Because RF-DETR Small is one of the judges, this is model-dependent preprocessing. Report results with the correction and on a separately verified upright set; include orientation-stage cost in end-to-end timing. A fixed upright pool camera does not need four rotation trials.
- A .05 candidate floor truncates the low end of the precision/recall curve. Record it explicitly. Verify that every original category merged into `person` actually denotes a person and that all people are labeled.
- Standard `pycocotools.COCOeval` uses a 100-detection maximum for its standard detection summary. In crowded images this can cap reported recall despite retaining 300 candidates. Report standard AP@100 and a clearly named custom crowded-scene metric at a common larger maxDets across models; adapt summary extraction rather than silently comparing different definitions. [Official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)

### The existing underwater-timer prototype is useful, with specific gaps

Inspected [detect_drowning.py](https://github.com/sunnycho100/we-fall-we-die/blob/2dfd33d6da45ec795cd35e47a683ea48d9f59199/model/detect_drowning.py), [benchmark_isaac.py](https://github.com/sunnycho100/we-fall-we-die/blob/2dfd33d6da45ec795cd35e47a683ea48d9f59199/model/benchmark_isaac.py) and [model README](https://github.com/sunnycho100/we-fall-we-die/blob/2dfd33d6da45ec795cd35e47a683ea48d9f59199/model/README.md). Reuse the explicit state/timer idea and simulation test cases; do not equate this standalone rendered-video demo with an integrated reviewer workflow.

1. **Low-score recovery is cut off upstream:** detector `--conf` defaults to .30 while ByteTrack's low threshold is .10. Set the detector/storage floor at or below the tracking low threshold before expecting its second association stage to help. The browser currently stores only >.20, so a future tracker needing .10 candidates requires new analysis jobs; changing a slider cannot restore discarded boxes. [ByteTrack paper](https://arxiv.org/abs/2110.06864), [official implementation](https://github.com/FoundationVision/ByteTrack)
2. **Person-only fallback is not underwater evidence:** any disappearance starts its underwater timer; any visible person box clears it. This produces false alarms for occlusion/exits and misses a still-visible submerged body. Replace that fallback with `visibility=missing/unknown`; keep a separate, independently validated `head_state` signal. RF-DETR person confidence is neither head visibility nor drowning probability.
3. **Scene context is missing:** the script has no calibrated pool ROI or exit-state logic. Preserve a missing-person registry outside tracker expiry, but distinguish exits, camera movement/degradation and unresolved disappearances. Do not allow missing records to remain live indefinitely without disposition.
4. **Re-identification is proximity-only:** a nearby new ID may inherit an old underwater timer. Require time/motion/appearance consistency, one-to-one association and explicit ambiguity handling in crowds. A new unrelated swimmer must not clear the missing record.
5. **Time is derived from `frame_index / fps`:** replace it with decoded presentation timestamps before variable-frame-rate footage or dropped-frame processing. A fixed number of tracker frames must correspond to the actual analysis sampling rate.
6. **Frozen motion history can mislead:** a lost person's old center history remains available to the stillness rule. Missing observations are not new evidence of immobility; use that acceleration rule only with recent observed motion and independently supported underwater evidence.
7. **Reported synthetic results are limited:** the README's RF-S above/underwater detection and fine-tuned YOLO results are team-reported on the same synthetic scene family. They are useful pipeline evidence, not measured real-pool performance. `benchmark_isaac.py` uses one-to-one IoU matching at .50/.30; the timer script's separate diagnostic uses .40 and a different matching procedure. Do not mix their numbers.

The next integrated flow is detector candidates → tracker → persistent missing registry plus head/visibility evidence → configurable review-event policy → saved clip and human disposition. An event should say `visibility_lost`, `possible_submersion` or `view_degraded` with timestamps, reasons and the last reliable location. The existing 5/12/8-second values are unvalidated engineering/demo settings, not physiological drowning thresholds. Add replay tests for ordinary short dives, long occlusion, visible submerged bodies, crossing swimmers, exits, camera cuts and resurfacing before enabling alarms.

Reproduce the two narrow behavioral checks with `.venv\Scripts\python.exe -m tools.audit_upstream_model`. The tool reads the pinned Git objects, extracts only the duplicate-filter statements and `Person` class, and executes synthetic fixtures without loading any checkpoint. It confirms removal of a nested box surviving IoU .50 suppression, and resetting of an existing timer on a visible person-only update. `artifacts/upstream-model-audit.json` records source hashes and scope. These are implementation diagnostics, not measurements on real people.

## 10. Calibrated positions across the pool: implemented scope

The user selected **position across a calibrated pool surface**, not underwater depth. This is a planar geometry task: map an image anchor through the four-corner homography to `(X,Z)` in the supplied pool rectangle. The displayed vertical coordinate is the **assumed water height**, not a model measurement. The mathematical basis is the relationship between a physical plane and its image. [OpenCV homography explanation](https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html)

The UI now distinguishes approximate dimensions from user-measured calibration. In the optional 3D view, raised symbolic pins make positions visible, axis labels define the origin and scale, and selecting a marker reports its estimated X/Z coordinates. Corner 1 is the origin; 1→2 is width/X and 1→4 is length/Z. Repeated identical observations reuse marker geometry. No temporary identity labels have been added. Selection clears when the observation changes or calibration expires.

For a useful calibration, select the image of an actual rectangular patch on the same water plane, enter its measured width/length, and verify additional known points not used in the four-point fit. Report their position errors; four corner points fit exactly by construction and are not an independent accuracy test. A generic trapezoid drawn around an irregular wave pool does not establish metric accuracy. Camera movement, lens distortion, waves and a person's box bottom not lying on the water plane introduce errors. The current reference dimensions remain approximate; there is no measured positioning error for this recording.

Using a depth network is unnecessary for this selected scope. [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) offers relative depth and separately trained metric variants, but those do not automatically calibrate an overhead pool or measure distance below reflective water. If underwater depth becomes a requirement, evaluate calibrated multiple views/underwater sensing with appropriate refraction modeling and real ground truth as a separate project.

Geometry checks cover an oblique rectangle's corners and independently projected center, axis convention, invalid inputs and out-of-region anchors. These verify the implementation's geometry, not real-world calibration precision.
