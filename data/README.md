# data

Ground truth, datasets, and generated frames. Raw video is not stored in git (`*.mp4` is ignored), except where noted.

| Folder | What | From |
|---|---|---|
| `sim_scenarios/` | The five MuJoCo scenarios: `<name>.tracks_gt.json` (every person's exact box and whether their head is above or below the water), per-person summaries, and `eval/` with detector and tracker results | `sim/mujoco/scenarios.py`, `sim/eval_detectors.py` |
| `sim_demo/` | Ground truth for the higher-quality 10 s MuJoCo demo render (local only, git ignores it; a copy of the ground truth is in `sim_scenarios/demo_hq.tracks_gt.json`) | `sim/mujoco/demo.py` |
| `sim_samples/` | Early MuJoCo outputs: keypoints and summaries for single-person motions | `sim/mujoco/pool_scene.py`, `drown_sim.py` |
| `aquaperson_wavepool_v1/` | 80 real wave-pool frames from YouTube, one box per visible person (submerged parts not boxed). Used to fine-tune the RF-DETR Small the app runs | Joanne, hand-labeled |
| `veo_footage/` | Stills from CCTV-style pool scenarios generated with Google's Veo 3.1 (the videos stay local) | `sim/generate_cctv_pool.py`, `sim/generate_veo_scenario.py` |

Rules we follow:
- Every clip keeps its source and license.
- Split train and test by recording (or sim scenario), never by frame. Frames from one video are too alike to test on.
- Model predictions are never saved as ground truth. The rescue-video labels (`tools/rescue_dataset.py`) are reviewed by a person.
