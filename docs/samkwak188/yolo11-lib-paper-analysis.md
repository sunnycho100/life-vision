# YOLO11-LiB: precise paper extraction and release audit

Follow-up: [the current ML system decision](ml-system-decision.md) adds temporal-model research and confirms that the demo video appears in a separate public temporal dataset. That finding does not establish whether the same video appears in LiB's training data.

Reviewed 2026-09-26. Source: the supplied 19-page `sensors-25-05552-v2.pdf`, read in full; dataset and result figures/tables visually checked. Paper: **Wenhui Zhang, Lu Chen, and Jianchun Shi, “A Pool Drowning Detection Model Based on Improved YOLO,” Sensors 2025, 25, 5552**, published 5 September 2025. [DOI / article](https://doi.org/10.3390/s25175552).

This report separates **paper-reported results**, **independent inspection of released artifacts**, and **our engineering recommendations**. No YOLO11-LiB inference or training was run. The authors' implementation was read, and checkpoint metadata was inspected with `pickletools`; the checkpoint was not unpickled or executed.

## 1. What this paper actually gives us

**Useful:** a pool-specific two-class detector, released weights, custom architecture modules, and a downloadable labeled image dataset. These are worth evaluating as a domain-specific baseline.

**Not provided:** a motion classifier, a head/airway-above-water model, pose estimation, a tracker, persistent missing-person logic, a temporal alarm policy, a baby/adult classifier, 3D reconstruction, or validated drowning-event detection.

The model predicts bounding boxes with **`Drowning` or `Swimming` appearance labels from individual images**. Calling a box `Drowning` is a learned dataset classification, not proof of an emergency. The authors themselves explicitly acknowledge the inability of single-frame YOLO systems to capture temporal drowning dynamics and propose tracking/temporal modeling as future work (p. 17).

**Recommendation:** inspect and test the released checkpoint in an isolated experiment; do not replace the working demo pipeline or connect its class score directly to a red alarm. The released split has confirmed duplicate leakage, and the custom model has nontrivial deployment dependencies.

## 2. Model design, translated into implementation choices

The base is **YOLO11n detection**, with P3/P4/P5 detection outputs. The released full YAML ends in `Detect`, not a pose head. Its default `nc: 80` is a generic configuration value; the dataset and released checkpoint both identify two actual classes. Training code overrides the number of classes from the data configuration. Paper: pp. 6–8; release: [full model YAML](https://github.com/Mibugi/Drowning-detection/blob/a417d924b9d906dc50ab3078dd48b0c5c63a0f1c/Module/11/YOLO11-LAE-backbone-C3k2-GhostModule-backbone-C2PSA-iSCSA-FreqFusion-BiFPN.yaml).

| Component | What the authors change | Why relevant to pools | Practical consequence |
|---|---|---|---|
| **LGCBlock / A** | LAE downsampling plus GhostC3k2 using Ghost-style feature generation and dynamic convolution. Dynamic convolution uses four experts; the Ghost channel ratio is 2. | Attempts to preserve useful features with fewer parameters. | Lower parameter count does not ensure lower latency; the measured A-only model is slower than the baseline. |
| **C2PSAiSCSA / B** | Integrates spatial/channel attention with an inverted residual mobile block; local depthwise processing and residual paths. | Intended to emphasize heads/arms and suppress distracting water/background patterns. | These are internal image features, not exported anatomical joints or measured underwater status. B alone has an interesting measured speed/accuracy tradeoff. |
| **BiFF-Net / C** | Combines weighted bidirectional feature fusion with FreqFusion adaptive low/high-pass filtering and CARAFE upsampling. | Intended to preserve detail for small/distant targets and blurred boundaries. | Custom operators complicate ONNX/browser/edge export; inspect support before selecting it for deployment. |

The claimed **87.5% downsampling computation reduction** and **50% feature-extraction overhead reduction** refer to specific component designs, not a whole-model reduction of those sizes. Table 2's complete model goes from **6.3 to 6.2 GFLOPs**, only approximately **1.6% lower** (pp. 6–7, 12, 17).

For us, the transferable ideas are domain-specific data, small-target evaluation, and measuring the actual operator/runtime cost. Reimplementing all three modules during a hackathon is poor use of time.

## 3. Dataset described in the paper

Section 4.1, p. 9 reports:

- **2,000 original images**, expanded to **4,800** through augmentation.
- Sources: Roboflow plus internet videos/images; indoor/outdoor, above-water/underwater, overhead/eye-level/oblique views.
- Claimed diversity includes lighting, water clarity, ages, body types, skin tones, clothing, devices, and backgrounds. No quantitative subgroup counts or subgroup performance are supplied.
- Augmentations: rotation, scale, brightness, saturation, hue, zoom, blur, noise, and mosaic.
- **70/20/10 train/validation/test split**, with YOLO bounding-box annotations.
- Two categories: `Swimming` and `Drowning`.

The category descriptions mention rhythmic motion, prolonged head submersion, propulsion, and loss of control. **Those descriptions involve time or physiology that a still image cannot directly establish.** They should not be copied verbatim as ground truth for our event engine.

The paper does not document a source-video/session-disjoint split, unique swimmers/pools, annotation agreement, event-onset labels, normal-video hours, or how the augmented variants were assigned to splits. Do not assume augmentation happened only after a clean split.

## 4. What the public release actually contains

Inspected repository commit: **`a417d924b9d906dc50ab3078dd48b0c5c63a0f1c`** in [Mibugi/Drowning-detection](https://github.com/Mibugi/Drowning-detection/tree/a417d924b9d906dc50ab3078dd48b0c5c63a0f1c).

| Asset | Verified availability | Meaning for our project |
|---|---|---|
| [`Module/LiB-YOLO.pt`](https://github.com/Mibugi/Drowning-detection/blob/a417d924b9d906dc50ab3078dd48b0c5c63a0f1c/Module/LiB-YOLO.pt) | Present, **4,464,654 bytes**; two-class checkpoint metadata. | We do not have to retrain the proposed model merely to attempt evaluation. Loading still requires its custom classes. |
| `Module/yolo11n.pt` | Present, 5,613,764 bytes. | Do not assume this is a pool-fine-tuned baseline without inspecting its metadata. |
| `Module/11/*.yaml` | Baseline and ablation/full-model configurations. | Useful for understanding the architecture; not a complete installation recipe. |
| `Module/AddModules/*.py` | LAE, dynamic convolution, iSCSA, FreqFusion, BiFPN implementations. | Must be wired into the expected Ultralytics package/module paths. |
| `Module/train.py`, `predict.py`, `val.py` | Ultralytics-derived trainer/predictor/validator classes. | These are framework class implementations, not a complete reproducible experiment launcher with pinned dependencies. |
| [`datasets/Drowning Detect.v6-2000.yolov11.zip`](https://github.com/Mibugi/Drowning-detection/blob/a417d924b9d906dc50ab3078dd48b0c5c63a0f1c/datasets/Drowning%20Detect.v6-2000.yolov11.zip) | Present, **101,111,276 bytes**. | Actual downloadable original dataset, independently inspected below. |

The root README directs users to Roboflow for the expanded data. The archive identifies [Roboflow version 6](https://universe.roboflow.com/drowing-detect/drowning-detect-fvwia/dataset/6), exported 12 May 2025. **This checked archive is the 2,000-image original set, not a verified reproduction of the paper's 4,800-image training set.**

### Verified labels and counts

`data.yaml`: **class 0 = `Drowning`; class 1 = `Swimming`**. Always read these names from the actual checkpoint rather than assuming class order from prose.

| Split | Images | Drowning boxes | Swimming boxes | Total boxes |
|---|---:|---:|---:|---:|
| Train | 1,400 | 663 | 1,210 | 1,873 |
| Validation | 400 | 204 | 316 | 520 |
| Test | 200 | 102 | 155 | 257 |
| **Total** | **2,000** | **969** | **1,681** | **2,650** |

The archive describes **auto-orientation and stretch resizing to 640×640**, and says no augmentation was applied to this export. Preserve the distinction between that export and the paper's expanded training data. Stretching changes person proportions relative to our aspect-preserving video preprocessing.

The audit found **no empty annotation files** and **1,640/2,000 images with exactly one annotated box**. This is not proof of annotation completeness; it means we must not treat the dataset as established coverage for a wave pool containing dozens of people. Audit crowded images for unlabeled swimmers before collapsing both classes into a `person` training set. Add genuinely empty/negative pool scenes for our own evaluation and future training.

### Confirmed split leakage: independent audit

Comparing SHA-256 hashes of image bytes in the released ZIP found **6 exact-image groups crossing dataset splits**:

- **2 test images also appear byte-for-byte in train**: 2/200 = **1% of test images**.
- **4 validation images also appear byte-for-byte in train**: 4/400 = **1% of validation images**.

Examples:

| Test image | Byte-identical training image |
|---|---|
| `She-drowning_mp4-11_jpg.rf.688715fd9df78fd81a835d7df54c8e30.jpg` | `She-drowning_mp4-11_jpg.rf.ecc8526a26f2603a9544588065abd986.jpg` |
| `notdrowning_mp4-4_jpg.rf.332b815e7dd0d44aab37ccc3d22ffcbc.jpg` | `8_mp4-0004_jpg.rf.66f675246a5f358245bb0a09c1ade231.jpg` |

These pairs have slightly different box annotations despite identical image bytes. Filename text such as `notdrowning` is not the class label; both corresponding label files use class 0 in the second example. That discrepancy warrants manual review, not an automatic relabel based on the filename.

Removing the Roboflow `.rf.` suffix additionally reveals **46 filename-stem groups crossing splits**, involving **118 images**. **21 test images** and **26 validation images** share stems with training images. This is an additional provenance warning; matching stems alone do not prove pixel identity or quantify near-duplicate leakage.

**Scope of the finding:** this confirms contamination in the inspected public original-data split. It does not establish which exact augmented split produced every published metric, or how much any metric was inflated. It does mean the supplied test split is not a clean independent generalization benchmark.

Reproduce the exact-byte and stem checks with [the audit script](../../tools/audit_paper_dataset.py):

```powershell
python tools/audit_paper_dataset.py "path/to/Drowning Detect.v6-2000.yolov11.zip" --output audit.json
```

Archive SHA-256: `95932efb0e59ab7b54f82ae2640830a4fa91ffe881b19a9d68aebc3f10f9262d`.

## 5. Training setup and a checkpoint mismatch

Paper pp. 10–11 / Table 1:

| Setting | Reported value |
|---|---|
| GPU | NVIDIA RTX 3090 |
| CPU | Intel Xeon Platinum 8362; stated 15 cores, 2.80 GHz |
| OS / runtime | Windows 11, Python 3.10, PyTorch 2.0.0, CUDA 11.8 |
| Epochs / batch / image size | 300 / 16 / 640 |
| Optimizer / initial learning rate | SGD / 0.01 |
| Early stopping setting | 100 |

The text on p. 9 instead mentions **500 epochs**, while Table 1 and Figure 6 say **300**. This is an internal inconsistency.

The released `LiB-YOLO.pt` adds a concrete reproducibility issue. Its serialized metadata records:

- Ultralytics **8.3.82**, date **2025-05-29T21:57:53.701849**.
- `epochs = 500`, `patience = 20`, `batch = 16`, `imgsz = 640`.
- SGD, `lr0 = 0.01`, seed 0, and an AGPL-3.0 license string.
- Class names `{0: Drowning, 1: Swimming}`.
- Saved aggregate training metrics include `mAP50(B) = 0.89701` and `mAP50-95(B) = 0.58683`.

The aggregate mAP value must **not** be compared directly to the paper's drowning-only AP of 94.1%. For reference, the arithmetic mean of Table 2's two class APs is **89.85%**. The saved numbers are metadata, not independently reproduced measurements or evidence that the checkpoint is one of the five reported runs. A configured maximum of 500 epochs also does not prove 500 epochs actually completed.

Checkpoint SHA-256: `f9568af20305785641a089f7f9ab9496baa613e1ba3a2516e4e5b491950df619`.

## 6. Results that matter for model selection

Selected rows from **Table 2, p. 12**, all paper-reported on the authors' data/runtime:

| Model | Drowning precision % | Drowning recall % | Drowning AP50 % | Swimming AP50 % | GFLOPs | FPS | Parameters M | Size MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 85.7 | 88.8 | 91.8 | 82.4 | 6.3 | 189.5 | 2.58 | 5.26 |
| YOLO11s | 88.0 | 89.7 | 91.8 | 84.6 | 21.3 | 182.9 | 9.41 | 18.3 |
| YOLOv8n | 87.6 | 88.6 | 92.5 | 84.5 | 8.1 | 250.9 | 3.01 | 5.98 |
| YOLOv9t | 85.2 | 87.1 | 91.9 | 84.4 | 7.6 | 97.9 | 1.97 | 4.46 |
| ViTDet | 90.2 | 90.1 | 94.5 | 86.3 | 1,024.0 | 9.8 | 150.2 | 305 |
| **YOLO11-LiB** | **88.4** | **89.7** | **94.1** | **85.6** | **6.2** | **80.5** | **2.02** | **4.25** |

Relative to YOLO11n, calculations from that table give:

- Drowning AP50: **+2.3 percentage points**, not +2.3% relative; relative gain is approximately 2.5%.
- Drowning precision: **+2.7 points**; recall: **+0.9 points**.
- Swimming AP50: **+3.2 points**.
- Parameters: **21.7% fewer**; storage: **19.2% smaller**; GFLOPs: **1.6% lower**.
- Throughput: **57.5% lower**. The reciprocal-FPS frame-time equivalent is approximately **12.42 ms versus 5.28 ms**, or **2.35× longer**. This calculation is not an independently measured end-to-end latency.

**Therefore, “smaller” does not mean “faster.”** The paper's phrase “minor fluctuations” does not convey the size of its reported throughput reduction. The RTX 3090 FPS is not a prediction for our Windows ARM laptop, browser, CPU fallback, tiled input, or four simultaneous camera streams.

### What 94.1% does and does not mean

It is the paper's **drowning-class AP at box IoU 0.5**, obtained across a precision/recall curve. It is not general accuracy, event recall, a probability that a current swimmer is drowning, a 94.1% rescue success rate, or an operational false-alarm rate.

At the paper's reported operating point, drowning recall is **89.7%**. The complement is **10.3% missed labeled drowning instances under that evaluation**, not 10.3% missed drowning incidents in deployment. Precision 88.4% similarly cannot be converted into false alarms per hour without the video exposure, thresholds, and temporal event policy.

The authors report five runs, 95% confidence intervals and paired t-tests. For LiB they state DmAP50 **94.1 ± 0.3%**, precision **88.4 ± 0.4%**, recall **89.7 ± 0.5%**, and FPS **80.5 ± 2.5**. These are reported run-level statistics on their evaluation setup; they do not resolve source leakage or measure unseen-pool reliability.

## 7. Ablations: the useful engineering tradeoff

Table 3, p. 14; A = LGCBlock, B = C2PSAiSCSA, C = BiFF-Net:

| Variant | Drowning precision % | Recall % | Drowning AP50 % | Swimming AP50 % | GFLOPs | FPS | Parameters M |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 85.7 | 88.8 | 91.8 | 82.4 | 6.3 | 189.5 | 2.58 |
| +A | 86.6 | 87.9 | 92.0 | 83.8 | 5.9 | 118.8 | 2.21 |
| +B | 89.7 | 89.8 | 93.0 | 85.8 | 6.3 | 165.7 | 2.55 |
| +C | 84.9 | 90.1 | 91.7 | 85.4 | 6.8 | 108.6 | 2.43 |
| +A+B | 87.4 | 89.7 | 92.3 | 83.9 | 5.9 | 109.4 | 2.18 |
| +A+B+C | 88.4 | 89.7 | 94.1 | 85.6 | 6.2 | 80.5 | 2.02 |

Useful conclusions:

1. **B alone is worth a later comparison**: 93.0 AP50 at 165.7 FPS versus the full model's 94.1 at 80.5 FPS. It also has higher tabulated precision, recall and swimming AP than the full combination. No B-only pretrained checkpoint was identified in the inspected tree, so this is not automatically an immediate deployment option.
2. **C alone does not improve drowning AP50**: 91.7 versus 91.8. Its GFLOPs increase from 6.3 to 6.8, despite prose suggesting reduced consumption.
3. **Combining modules is not monotonically better**: A+B has lower drowning AP than B alone. The study does not include every pairwise combination, such as A+C and B+C.
4. The full model's highest AP among these ablations is not the same as being best at every metric. Choose for our measured recall/latency/false-event tradeoff.

## 8. Limitations and claims we should not repeat

| Claim or gap | Precise interpretation |
|---|---|
| “Outperforming all comparison models” in the conclusion | Table 2 gives ViTDet **94.5** drowning AP50 versus LiB **94.1**. The prose says the difference is not statistically significant; LiB is much smaller/faster than ViTDet. |
| Stronger small-object / glare / occlusion performance | Supported by selected qualitative examples and aggregate metrics. No quantitative AP-small, glare-only, occlusion-only, or distance-stratified table is supplied. |
| Good generalization because train/validation losses converge | Loss convergence does not establish source-independent generalization, especially given the released split overlap. |
| Practical multi-camera / Jetson deployment | Discussed on p. 16; no measured Jetson, CPU, browser, power, or four-stream result is presented. |
| Real-time drowning detection | Model FPS is measured; continuous incident sensitivity, alarm delay from real onset, false alerts/hour, and human-response effectiveness are not. |
| Children included | No child/infant-specific results establish toddler reliability. A small box is not an age label. |
| Joint/motion benefit | This model produces boxes/classes, not joints or temporal features. References to pose systems describe other work. |
| Above-water head / silent sinking | No airway/surface classifier or validated special-case recall is provided. A still image labeled `Drowning` is insufficient evidence. |

The paper's narrative also contains a 300-versus-500-epoch inconsistency, describes a “decline in precision” for A while its table's precision increases, and treats some large runtime changes as minor. These warrant verification, not allegations about author intent.

## 9. Practical integration barriers

The released checkpoint serializes custom classes under **`ultralytics.nn.Addmodules.*`**. The repository directory is named **`Module/AddModules`**. A standard, unmodified `pip install ultralytics` should not be assumed to resolve those exact custom module paths; case also matters on Linux.

The checked source imports:

- `timm` dynamic/conditional convolution components.
- `einops` rearrangement.
- `mmengine.model.BaseModule`.
- **`mmcv.ops.carafe`**, used by FreqFusion.

These are concrete reasons why the 4.25 MB model is **not a drop-in replacement for our browser ONNX models**. CARAFE support, custom dynamic convolution, export shape constraints, and runtime operator coverage must be tested. The release does not provide a verified ONNX/TensorRT/browser export, dependency lockfile, or complete package-registration instructions. [FreqFusion source](https://github.com/Mibugi/Drowning-detection/blob/a417d924b9d906dc50ab3078dd48b0c5c63a0f1c/Module/AddModules/FreqFusion.py), [dynamic convolution source](https://github.com/Mibugi/Drowning-detection/blob/a417d924b9d906dc50ab3078dd48b0c5c63a0f1c/Module/AddModules/DynamicConvModule.py), [iSCSA source](https://github.com/Mibugi/Drowning-detection/blob/a417d924b9d906dc50ab3078dd48b0c5c63a0f1c/Module/AddModules/iSCSA.py).

A practical route is **offline inference on a compatible Python environment → timestamped boxes/classes → the shared tracker/event pipeline and frontend replay**. Browser conversion is a separate optimization, not a prerequisite for evaluating detection quality.

### Artifact terms and provenance

The article is CC BY 4.0. The dataset export declares CC BY 4.0. Ultralytics-derived files contain AGPL-3.0 headers, and the checkpoint records an AGPL-3.0 string. No top-level repository license file was present in the inspected tree. These are different artifacts; do not label the entire release “permissive” based on the paper's license.

Some image filenames reference stock-media services such as iStock/Depositphotos. The export's license declaration does not independently verify each source's rights. Maintain provenance before redistributing training images or publicly presenting them. This inspection did not establish rights clearance.

## 10. What we should do next, in order

### Within the hackathon

1. **Keep the working baseline intact.** The onboarding prototype currently uses pretrained YOLOv8n detection and YOLOv8n-pose in the browser; this paper has not changed that implementation. It supports evaluating pool-specific training data, not blindly switching architectures.
2. **Time-box a checkpoint-loading experiment to 30–60 minutes** on a compatible machine. Record versions, exact checkpoint hash and preprocessing. Stop if custom dependencies consume the remaining integration window.
3. **Test a small, reviewed development subset** of the actual fixed-camera pool clip, including distant swimmers, partial bodies, floats, normal splashing and clear-water cases. Inspect every predicted class; a visually convincing box is not an incident label.
4. **Check source overlap first.** Internet pool clips may overlap the public dataset. Our exact 2:09 video was not proven absent from training during this audit. Do not call it an independent test until checked.
5. If LiB runs, export timestamped observations and compare against the baseline on the same data. Preserve a separate `appearance_class`/score; do not overwrite `head_state` or alarm state with it.
6. If the custom model is impractical, consider **a stock YOLO person detector fine-tuned on a cleaned, fully reviewed pool subset**, with both behavior classes remapped to `person` only after checking that all relevant people are annotated. This avoids custom architecture/export work but still requires compute and validation.

### Team ownership and event integration

- **Sam + Joanne:** model/data audit, class mapping, source grouping, preprocessing and a small measured comparison. Defer baby/adult classification.
- **Sunny:** stable tracking and the persistent missing-person registry outside the tracker. Keep these independent of the onboarding UI's temporary associations.
- **Rohan:** explicit event rules and scenario tests. The paper does not validate 5/12/8-second thresholds. Treat those as configurable experimental settings; actual submersion state requires evidence beyond a box label.
- **David:** show observation source, current/unknown status, reason, timer, sound and replay. The optional 3D view consumes observations and must not generate authoritative alarms.

Suggested observation fields: `camera_id`, `t_sec`, `track_id`, `bbox_xyxy`, `appearance_class`, `appearance_score`, `keypoints` when a separate pose model provides them, and observation/visibility quality. Suggested incident fields remain `event_id`, camera/person references, state, start/trigger time, last-seen time/position, reason and resolution. **Do not treat a per-image `Drowning` score as the event itself.**

### After the hackathon

Deduplicate and regroup by source video/session/pool before splitting; keep augmentation families within one split. Build a separate unambiguous normal-video set and quantify false alerts/hour, event-level misses, detection delay and identity failures. Report small/partial-person performance and uncertain/unobservable cases. Only then evaluate whether the paper's modules add value over a stock pool-fine-tuned detector.

## 11. Useful follow-up references from the paper

These were identified in the bibliography, not independently reproduced here:

- **He et al. (2023), infant drowning with YOLOv5 and Faster R-CNN**, reference 20: relevant to the deferred age/domain question, not evidence our model works on infants.
- **Gao et al. (2024), YOLOv8 drowning detection / pose discussion**, reference 27: worth reading for the motion branch; YOLO11-LiB itself does not implement it.
- **Jiang et al. (2025), Swimming-YOLO**, reference 25: potential comparison for pool-specific features.
- **Liu et al. (2025), lightweight outdoor drowning detection**, reference 26: potential deployment comparison.
- **Yu et al. (2024), LSM-YOLO**, reference 29: source for the adapted LAE component.

The paper's cited “ST-GCN” example (reference 28) is a **traffic-accident prediction** paper. It illustrates temporal graph modeling in another domain, not a ready-to-use skeleton drowning classifier. The discussed 1.2-second pose-system delay belongs to referenced work; it is not a measured YOLO11-LiB delay or a universal acceptable alarm threshold.

**Decision:** use this as a valuable pool-data/checkpoint lead and a research baseline. Its public assets make an experiment feasible; its split contamination, incomplete reproduction instructions, custom operators and lack of temporal validation prevent using the headline AP as a product claim.
