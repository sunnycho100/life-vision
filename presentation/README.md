# Presentation assets

Everything that goes into the slides: videos, images, charts, and the numbers behind them. Start with the [final deck](slides/final/life-vision-hackathon-deck.pptx) and [final presenter script](presenter-script-final.md). The earlier narrative remains in [docs/presentation.md](../docs/presentation.md) as supporting material.

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
| `slides/pool-assistant-draft.pptx` | Draft 10-slide deck, black and white SpaceX-style, speaker notes included, slide 8 is a demo placeholder | `npm install pptxgenjs && node presentation/slides/build_deck.js` |
| `slides/slide-context.md` | Per-slide content, visuals, speaker notes, timing, presenter split | written from the technical story |
| `videos/demo_hq_tracking_finetuned_rfdetr.mp4` | Fine-tuned RF-DETR-N + tracking on the HQ demo render (slide 8 fallback) | `sim/eval_detectors.py --scenario demo_hq --detector <sim_rfdetr checkpoint> --track --edge --video --no-gt` |
| `videos/isaac_pool_final_20260926-2151.mp4` | **Latest Isaac Sim generation (2026-09-26 21:51).** 20 s backyard pool, 30 fps, path-traced: lap swimmer, back floater, treader, diver who resurfaces elsewhere, child-sized silent sink, drowning response then sink | `sim/isaac/pool_video.py --seconds 20 --fps 30 --pathtrace 64 --subframes 1` (720p copy of the 1080p render) |
| `videos/isaac_pool_final_20260926-2151_groundtruth.mp4` | Same clip with the simulator's true boxes, head state and seconds underwater | `pool_labeled.mp4` from the same render |
| `videos/isaac_pool_final_20260926-2151_rfdetr.mp4` | Stock RF-DETR Small (COCO, no fine-tuning) + ByteTrack on the clip: finds 78% of people (57% of those fully under water) | `model/benchmark_isaac.py --detector rfdetr-s --track --clean` |
| `videos/isaac_pool_final_20260926-2151_yolo_alarms.mp4` | Our Isaac-trained YOLO11n + ByteTrack + underwater timer: WARNING at 5 s under, ALARM for the child at 13.5 s and the struggler at 16.9 s, no false alerts | `model/detect_drowning.py --video ... --gt ...` |
| `data/sim_results.csv` | Detector and tracker results on the held-out sim scenario | [docs/global/simulation.md](../docs/global/simulation.md) |
| `slides/final/life-vision-hackathon-deck.pptx` | Final 10-slide hackathon deck using the frontend palette and interaction language; includes editable chart data and speaker notes | generated with `slides/build_final_deck.mjs` from repository results and sources |
| `presenter-script-final.md` | Timed 5:30 talk track, live-demo checklist, transitions, and likely judge Q&A | written from the final deck and measured repository results |
| `images/generated/cover-pool-camera.png` | Cinematic pool-camera cover visual | generated with OpenAI ImageGen for this presentation |
| `images/generated/safe-pool-demo.png` | Safe two-person pool scene used to explain the live demo | generated with OpenAI ImageGen for this presentation |
| `images/generated/camera-pool-integration.png` | Unbranded pool-camera integration concept | generated with OpenAI ImageGen for this presentation |

## Presenting the final deck

1. Rehearse from `presenter-script-final.md`; the target runtime is 5 minutes 30 seconds.
2. Before presenting, open the local frontend and load the team's prerecorded pool clip.
3. On slide 9, switch to the frontend, activate monitoring, and narrate green → yellow → red as the timer advances.
4. State whether the run is live inference, prerecorded inference, or scripted output. Do not imply a scripted run is a live model result.
5. If the live demo fails, return to the deck and use the recorded simulation assets in `videos/`.

## Rules for slides
- Sim numbers are sim only. Label them "on simulation" and never present them as real-world accuracy.
- Say "detects prolonged head submersion", not "detects drowning".
- Keep the safety line on the closing slide: a supervision aid, not a replacement for watching children.
