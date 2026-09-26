# 01. Detection Model

updated: 2026-09-26 · author: sunnycho100 (agent research)

## Short answer
- **v1 detector:** a nano-size real-time detector (YOLO11n or YOLO26n) fine-tuned on our own pool footage. Best tooling, built-in tracking, fine for a portfolio.
- **If we ever sell it:** switch to **RF-DETR** (Apache 2.0) or buy the Ultralytics Enterprise License. Ultralytics models are AGPL-3.0 by default.
- **Open-vocabulary models** (Grounding DINO, LocateAnything-3B) are for **offline auto-labeling only**, never live.
- **Adult vs baby:** v1 uses one `person` class (matches our decision). Log box size as a cheap signal. Add a crop classifier in v2.

## Real-time detectors

| Model | License | Why use it | Watch out |
|---|---|---|---|
| YOLOv8 / YOLO11 / YOLO26 (Ultralytics) | AGPL-3.0, or paid Enterprise License | Easiest training, export (TensorRT, CoreML, TFLite), built-in ByteTrack and BoT-SORT | AGPL covers the code **and models trained with it**. A closed product must open-source the whole app or pay. |
| RF-DETR (Roboflow, ICLR 2026) | Apache 2.0 for base sizes; XL and 2XL under PML 1.0 | First real-time model over 60 mAP on COCO, leads RF100-VL (transfer to custom datasets), made for fine-tuning | Newer ecosystem, tracking is a separate library |
| RT-DETR (Baidu) | Apache 2.0 (original repo) | Transformer detector, no NMS | If used through the `ultralytics` package, AGPL applies (unverified how strictly for weights) |
| D-FINE | Apache 2.0 (unverified) | Precise boxes | RF-DETR Nano beats D-FINE Nano by 5.3 AP at similar latency |

Note: the Ultralytics GitHub title now lists a "YOLO27" as well (unverified, not checked in docs).

Pool-specific research exists and all of it is YOLO based: DrownACB-YOLO adds a "transition" label between swimming and drowned, Swimming-YOLO uses deformable convolution for crowded pools, and YOLO11-LiB reports 94.1% mAP on the drowning class at 2.02M parameters. All of these use pool or underwater datasets, not a home backyard setup.

## Open-vocabulary labelers (offline only)

| Model | License | Use |
|---|---|---|
| Grounding DINO | Apache 2.0 | Auto-label frames from a text prompt. Works with `autodistill` to go straight to a YOLO dataset. Large and slow. |
| LocateAnything-3B (NVIDIA Eagle) | Code Apache 2.0, **weights non-commercial** (research and evaluation only) | Stronger dense detection, 12.7 boxes/s on an H100. Research use only. |

## Adult vs baby

| Option | How | Pros | Cons |
|---|---|---|---|
| (a) Two detector classes | Train `adult` and `child` directly | One model, one pass | Needs many labeled kids in pools. Filming children needs parental consent. Top-down, half-submerged age is hard to see. |
| (b) Person detector + crop classifier | Detect `person`, then a tiny classifier on each crop says adult or child | Classifier can be trained on normal photos too. A YOLO-based head and body approach reached 95% (children) and 92.5% (adults) on public images. | Two models. Public accuracy is not pool accuracy (unverified for pools). |
| (c) Box size relative to pool | Pixel height of box vs known pool size | Free, no training | Breaks with camera angle, distance, and a crouching adult. Half-submerged bodies look small. |

**Recommendation:** v1 single `person` class, the alert rule applies to everyone. Record (c) as metadata so we can study it. v2 adds (b) once we have consented footage.

## Edge hardware (nano models)

| Hardware | YOLO11n speed | Notes |
|---|---|---|
| Jetson Orin Nano Super | 60+ FPS (INT8, TensorRT) | Best fit for a home box |
| Raspberry Pi 5 + Hailo-8L | about 30 FPS (reported for YOLOv8n) | Needs model calibration for the NPU |
| Raspberry Pi 5 CPU only | 5 to 10 FPS (ONNX INT8) | Too slow for multi-person tracking |
| Phone (CoreML or TFLite) | unverified | Nano models usually run in real time on recent phones; not measured here |

We need about 10 to 15 FPS for tracking and the underwater timer, so Orin Nano or Pi 5 + Hailo both work.

## Sources
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics), [Ultralytics License](https://www.ultralytics.com/license), [LibreYOLO on YOLO licenses](https://www.libreyolo.com/articles/yolo-commercial-license)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr), [RF-DETR vs alternatives](https://blog.roboflow.com/rf-detr-vs-alternatives/), [RT-DETR on Roboflow](https://playground.roboflow.com/models/baidu/rt-detr)
- [DrownACB-YOLO](https://journal.hep.com.cn/jdue/EN/10.19884/j.1672-5220.202406015), [Swimming-YOLO](https://link.springer.com/article/10.1007/s11760-024-03744-7), [Pool drowning detection with improved YOLO (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12431139/)
- [Grounding DINO](https://github.com/idea-research/groundingdino), [autodistill-grounding-dino](https://github.com/autodistill/autodistill-grounding-dino), [LocateAnything](https://github.com/NVlabs/Eagle/tree/main/Embodied)
- [Age group classifier with YOLO (IEEE)](https://ieeexplore.ieee.org/document/9937129/)
- [Jetson Orin Nano Super vs Pi 5 + Hailo-8L](https://www.myaihardware.com/compare-article/jetson-orin-nano-super-vs-raspberry-pi-5-hailo), [YOLO11 on Pi 5](https://vucense.com/dev-corner/yolov11-raspberry-pi-2026/), [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)
