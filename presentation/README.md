# Presentation assets

Everything that goes into the slides: videos, images, charts, and the numbers behind them. The talk itself is in [docs/presentation.md](../docs/presentation.md).

| Folder | What goes here |
|---|---|
| `videos/` | Short demo clips (MP4, H.264, keep each under about 5 MB) |
| `images/` | Stills and screenshots (PNG) |
| `charts/` | Charts made from `data/` |
| `data/` | CSV tables with the numbers shown on slides |
| `slides/` | Exported decks (PPTX or PDF) |

Name files by what they show, e.g. `before_stock_rfdetr_default_tracker.mp4`, not `video2.mp4`. When you add an asset, add a row below with where it came from, so anyone can regenerate it.

## Current assets

| File | Shows | Source |
|---|---|---|
| `videos/sim_raw_baseline.mp4` | Raw MuJoCo pool sim, 5 people, one collapses | `sim/mujoco/scenarios.py --only baseline` |
| `videos/sim_yolo11n_all_scenarios.mp4` | Sim-trained YOLO11n + ByteTrack on all 5 sim scenarios (40 s) | `sim/eval_detectors.py --detector sim/mujoco/finetuned/sim_yolo11n.pt --track --video --no-gt` |
| `videos/before_stock_rfdetr_default_tracker.mp4` | Before: stock RF-DETR-N + default tracker, 18 IDs for 4 people, misses the diver under water | `sim/eval_detectors.py --scenario resurface --detector rfdetr --track --video --no-gt` |
| `videos/after_finetuned_rfdetr_edge_logic.mp4` | After: RF-DETR-N fine-tuned on sim + edge logic, 4 IDs for 4 people | `sim/eval_detectors.py --scenario resurface --detector <sim_rfdetr checkpoint> --track --edge --video --no-gt` |
| `images/sim_pool_scene.png` | Still of the sim pool | frame at 5 s of `sim_raw_baseline.mp4` |
| `images/before_after_tracking.png` | Same frame (6 s), before vs after, side by side | frames of the two videos above |
| `images/isaac_sim_pool_editor.png` | Isaac Sim 6.1 editor with the backyard pool scene: 5 animated people, water volume, stage tree | screenshot of `sim/isaac/pool_scene.py` + `pool_anim.py` running live in the Isaac Sim window (RTX Real-Time) |
| `videos/demo_hq_10s.mp4` | 10 s higher-quality sim render (720p, shadows, textures, body shapes): stand, swim, duck under, collapse, float, drowning response | `sim/mujoco/demo.py` |
| `technical-story.md` | Full technical story: pipeline diagram, models tried, YOLO vs RF-DETR, fine-tuning, tracking, decisions, mistakes caught | written from the measured results |
| `data/sim_results.csv` | Detector and tracker results on the held-out sim scenario | [docs/global/simulation.md](../docs/global/simulation.md) |

## Rules for slides
- Sim numbers are sim only. Label them "on simulation" and never present them as real-world accuracy.
- Say "detects prolonged head submersion", not "detects drowning".
- Keep the safety line on the closing slide: a supervision aid, not a replacement for watching children.
