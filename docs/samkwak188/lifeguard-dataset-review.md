# Lifeguard clips: preparation, human review and training exports

## What is ready

The 19 local MP4s in `data/raw/lifeguard_rescue/under_60s` were fully decoded, fingerprinted with SHA-256 and paired with their `.info.json` source metadata. All passed decoding. Total duration is **875.719 seconds (14 minutes 35.7 seconds)**.

The prepared, **unreviewed** bundle is `data/processed/lifeguard-review-v1`:

- `rescue-manifest.json`: video provenance, hashes, timing, provisional split, human event-review fields.
- `annotations.json`: 76 empty annotation templates, four timestamped frames per source.
- `annotations-suggested.json`: editable model proposals for the first 12 training frames. These are not approved labels.
- 76 full-resolution PNGs and 19 timestamped contact sheets.
- `preparation-report.json`: actual counts and decoder errors (none).

Images were selected at 12.5%, 37.5%, 62.5% and 87.5% of each video's duration. Stored times are decoded presentation timestamps relative to video start, not frame-index/FPS estimates. This sparse starter set is for a first person-detector review, **not enough to label temporal behavior**.

Whole-source provisional splits are 13 train, 3 validation and 3 test videos, with seed 42. Exact file duplicates are rejected. A human must still check re-uploads, repeated events and edits before approving splits. Different files do not guarantee different events. These clips also share a pool/channel, so a source-disjoint result does not establish performance at another pool.

No training has run on these labels. No clinical drowning labels or model accuracy numbers have been fabricated.

## Start with the guided example

With the local application running, open:

- **Guided first exercise:** <http://127.0.0.1:5173/rescue-review-guide.html>
- **Video/event annotation:** <http://127.0.0.1:5173/rescue-review.html>
- **All-person frame annotation:** <http://127.0.0.1:5173/annotate.html>

If the application is not running, start it from the repository root:

```powershell
.venv\Scripts\python.exe -m backend.serve --port 5173
```

For event review, select `rescue-manifest.json` and the local `under_60s` video folder. No videos are uploaded by the review page. For person boxes, choose the entire prepared folder. The frame page prefers `annotations-reviewed.json`, then `annotations-suggested.json`, then the original `annotations.json`.

The guide uses training source **a2FD2QfT0d8**. Assistant inspection found a camera zoom, a lifeguard entering in approximately the first two seconds and approaching a swimmer before making contact around 9–10 seconds. These observations are **review prompts, not ground truth**. Start by finding the assisted swimmer around 10 seconds, then work backward through 8, 6 and 4 seconds. The person wearing a white shirt and the lifeguard are separate people. Confirm identity yourself.

The early lifeguard response makes this a useful lesson in uncertainty and rescue-context bias: much of the clip may be unsuitable for learning pre-rescue distress, even though its frames can help train person detection. Leave onset unknown if it precedes the clip or cannot be seen. There is no requirement to manufacture a positive label.

**First human task:** review this one clip, draw one or two visible target keyframes with a consistent ID such as `person-1`, mark a short interval `uncertain` if necessary, enter your notes and name, and download `rescue-manifest-reviewed.json` into the prepared folder. Tell the assistant when it is saved. We will validate that example before scaling up.

## Two different annotation products

### A. Person-detection data for RF-DETR

Draw a tight box around **every distinguishable visible person**, including swimmers, head-only people, partially occluded people and people on the deck. Include attributable submerged parts only when they are actually visible. Do not hallucinate a hidden full body or a box for someone who has disappeared.

Correct false proposals, duplicate boxes and missed people. A model's low score is not a human label. Because missed people would otherwise be treated as background, marking just the rescued person is insufficient for detector training. Inspect at 200% zoom. Add `head_only`, `partial`, `occluded` and `deck` tags where appropriate. Pool membership may remain unknown; it is not used to generate the COCO person class.

The frame must be marked entirely reviewed and have a reviewer name. Mark truly inseparable crowds as ignore regions. **The current training export excludes an entire frame containing an ignore region**, with a recorded reason: simply setting `iscrowd=1` is not a reliable ignored-background implementation in the current training pipeline. Choose clearer frames or later implement and verify ignore-aware training.

Suggestions use the already verified RF-DETR Nano FP32 ONNX model, threshold 0.20, a full-frame pass plus six overlapping square crops, and cross-pass IoU-0.5 NMS. This is an annotation convenience, not a selected production operating point. No proposals are generated on validation or test sources. Proposal metadata and scores remain available for audit. All suggested frames start unreviewed.

### B. Temporal, person-specific evidence

Use one human-assigned target ID only while identity is clear. Boxes at visible keyframes establish who an interval refers to. Do not interpolate identity or a submerged position through an ambiguous crowd.

| Label | What it means |
|---|---|
| `ordinary_swimming` | The named person's observed behavior appears ordinary during this interval; not a medical guarantee. |
| `visible_distress` | Observable behavior suggests distress; describe the sustained evidence in notes. |
| `submerged_visible` | The named person's submerged body is actually visible. This alone is not drowning. |
| `not_visible` | The named person cannot be seen. It does not prove submersion. |
| `uncertain` | Identity, behavior or visibility cannot be determined confidently. |
| `rescue` | Assistance/rescue is underway for the named person. |

Times are seconds relative to the source start. Intervals for one person must be nonoverlapping, with positive duration and valid bounds. Leave unreviewed time unknown; it does not automatically become negative data. Record possible distress-onset earliest/latest, first rescuer entry and first rescue contact when observable. Record zoom, cuts, slow motion and overlays in notes. Unknown boundaries remain null.

The temporal exporter preserves reviewed intervals as JSONL references to hashed original videos, with target keyframes and reviewer identity. It **does not manufacture cropped training tensors or a binary drowning dataset**. Only ordinary-swimming/visible-distress intervals with a visible target keyframe and an end no later than a known rescuer-entry time are marked eligible for further distress research. Rescue, uncertainty, missing identities and post-entry footage remain recorded but excluded from that eligibility. A second independent reviewer should check positive labels and onset bounds before training an action model.

## Reproducible commands

Run from repository root using the existing native environment. Prepared directories are versioned; the tool refuses to overwrite an existing bundle or export.

```powershell
# Already completed for v1. Use a NEW output directory to repeat.
.venv\Scripts\python.exe -m tools.rescue_dataset prepare --raw data/raw/lifeguard_rescue/under_60s --output data/processed/lifeguard-review-v2 --samples 4

# Already completed for v1. Suggestions never approve themselves.
.venv\Scripts\python.exe -m tools.rescue_dataset suggest --labels data/processed/lifeguard-review-v1/annotations.json --manifest data/processed/lifeguard-review-v1/rescue-manifest.json --model artifacts/models/rfdetr-nano/manifest.json --limit 12 --threshold 0.20

# Validate the downloaded human file without approving anything.
.venv\Scripts\python.exe -m tools.rescue_dataset validate --manifest data/processed/lifeguard-review-v1/rescue-manifest-reviewed.json
```

After a human has inspected the source groups and assigned repeated events to the same split, explicitly record split approval. Do not run this command merely to bypass the review requirement:

```powershell
.venv\Scripts\python.exe -m tools.rescue_dataset approve-splits --manifest data/processed/lifeguard-review-v1/rescue-manifest-reviewed.json --reviewer YOUR_NAME --output data/processed/lifeguard-review-v1/rescue-manifest-approved.json
```

If source assignments change, synchronize the affected frame `split` fields before exporting. Validation rejects stale split approvals and frame/source split mismatches. Source camera groups remain useful metadata; the starter exporter does not claim camera-disjoint splits.

After reviewing all people in enough frames from **each** split and saving `annotations-reviewed.json`:

```powershell
.venv\Scripts\python.exe -m tools.rescue_dataset export-coco --labels data/processed/lifeguard-review-v1/annotations-reviewed.json --manifest data/processed/lifeguard-review-v1/rescue-manifest-approved.json --output data/processed/lifeguard-coco-v1

.venv\Scripts\python.exe -m tools.rescue_dataset export-temporal --manifest data/processed/lifeguard-review-v1/rescue-manifest-approved.json --output data/processed/lifeguard-review-v1/temporal-reviewed-v1.jsonl
```

COCO export creates `train/`, `valid/`, `test/` images and `_annotations.coco.json` files. It verifies video/image hashes, frame dimensions, source splits and boxes; excludes pending/ignored frames; and fails when any split has no reviewed usable frame. The export report preserves exclusions and input hashes. A reviewed truly empty frame is valid; an unreviewed empty template is not.

**Class mapping:** this single-class export uses category `0: person`. The pretrained deployment adapter currently uses sparse COCO `1: person`. Fine-tuning requires a corresponding deployment mapping update, a fresh model manifest/hash and repeat SDK-versus-ONNX checks. Do not substitute a checkpoint into the old adapter without that work.

## What follows human review

1. Correct the first example together. Then review several detector frames from all three splits, including small/head-only people. Expand beyond the initial 76 frames around failures and important times; source assignments stay fixed.
2. Freeze a reviewed evaluation set. Measure the pretrained detector before training: recall/precision, small-person misses, duplicates and speed. Do not choose thresholds using the test set.
3. Train a small person-detection pilot using pinned RF-DETR 1.11.0 and the existing `configs/rfdetr-pool-person-pilot.json`, on a compatible processing machine. Compare against the baseline on unchanged reviewed evaluation frames. The local annotation workflow does not assume CUDA availability.
4. For temporal learning, obtain independently sampled normal footage: ordinary swimming, dives, underwater play, floating, occlusion, splashes and camera disruption. Rescue-only videos cannot estimate real-world false alarms per hour. Normal intervals from a rescue video remain in that video's split.
5. Review positive intervals twice and resolve target identity/onset disagreements. Decide the temporal model/input representation only after this label audit. Evaluate missed events, alert latency before rescue, false alerts per camera-hour, identity failures and coverage. Keep another pool/camera for external evaluation.

Rights/permission status is recorded as unverified because a public video URL does not itself specify dataset redistribution or model-training permissions. Resolve the applicable usage basis before publishing a derived dataset or deploying trained weights; preparation does not assert permission.

## Evidence and limits

RF-DETR's [official training documentation](https://rfdetr.roboflow.com/latest/learn/train/) describes COCO dataset training. This repository keeps package version 1.11.0 pinned; newer documentation is not a reason to silently upgrade the training environment.

Laxton et al., [“Search for a distressed swimmer in a dynamic, real-world environment”](https://irep.ntu.ac.uk/41673/1/1389681_Crundall.pdf), used this channel with permission, included non-distress controls, and ended stimulus clips before lifeguards entered the water. That supports separating observable distress from rescue-response shortcuts. It does not validate our detector or our new labels.

The previous Claude Code Opus 5.5 xhigh read-only review also recommended human verification, source/event grouping, onset uncertainty, normal controls and complete person annotations. Its review was advisory, not experimental evidence. This dataset workflow implements those data safeguards; it does not imply that outstanding tracking/incident-pipeline issues have been resolved.

Local media, labels and exports remain under ignored `data/raw` and `data/processed` directories. Review tools and this protocol are versionable; source videos are not committed automatically.
