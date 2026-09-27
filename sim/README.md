# sim

We couldn't film people drowning, so we simulated it. A simulator knows the exact answer for every frame (where each person is, whether their head is under water, for how long), so it can score a detector, a tracker, and the alarm logic without anyone labeling a single frame. It also gives us the cases we could never stage: a child sinking silently, someone collapsing, a diver coming up somewhere else.

We built the test bench in three steps, each more realistic than the last.

## 1. MuJoCo: a fast test bench on a laptop (Sunny)

[`mujoco/`](mujoco/). Capsule-body swimmers in a tinted pool, no GPU needed.

- `scenarios.py` renders five scripted scenarios with exact ground truth: `baseline` (one person collapses), `resurface` (a diver comes up 2 m away), `crossing` (two swimmers pass, IDs must not swap), `silent_sink_busy` (a quiet sink in a busy pool), and `entry` (someone falls in from the deck). Videos are in [`simulation_videos/`](simulation_videos/), ground truth in `data/sim_scenarios/`.
- `demo.py` renders a higher-quality 10 s scene with new textures and body shapes, to test how much a small visual change hurts.
- `pool_scene.py`, `drown_sim.py`: the first single-person motions (swim, float, instinctive drowning response, silent sink). Buoyancy includes air in the lungs, since MuJoCo has none of its own.

What we learned from it:
- Stock YOLO found a quarter of the simulated people and almost none fully under water. Fine-tuning on 180 sim frames (about 7 min on a MacBook) got it to 1.00 on a scenario it never saw.
- Stock RF-DETR Nano with a default tracker showed 18 IDs for 4 people. Fine-tuned RF-DETR Nano plus our tracking rules showed exactly 4, with no ID switches, on every scenario.
- In clear water a good detector keeps seeing a person lying on the bottom, so the box never disappears. The alarm has to watch the head, not just whether the person vanished.

```bash
python3 -m venv .venv && .venv/bin/pip install mujoco numpy pillow ultralytics "rfdetr[train]"
.venv/bin/python sim/mujoco/scenarios.py                                   # all five scenarios, about 2.5 min
.venv/bin/python sim/train_sim_detector.py --test resurface                # YOLO11n on sim frames
.venv/bin/python sim/train_sim_rfdetr.py --epochs 1                         # RF-DETR Nano on sim frames
.venv/bin/python sim/eval_detectors.py --scenario resurface --detector runs/sim_rfdetr/checkpoint_best_ema.pth --track --edge --video
```

Full results: [docs/global/simulation.md](../docs/global/simulation.md). The sim-trained YOLO11n is in [`mujoco/finetuned/`](mujoco/finetuned/).

## 2. Isaac Sim: realistic water and people (Rohan)

[`isaac/`](isaac/). A path-traced backyard pool in NVIDIA Isaac Sim 6.1 with moving water, refraction through the waves, and six animated people: a lap swimmer, a floater, a treader, a diver, a child who sinks silently, and an adult who shows the instinctive drowning response and then sinks. Every frame is labeled with each head's height against the local water surface.

A YOLO11n trained on 261 Isaac frames found 95% of people with their head fully under water (stock YOLO11n: 16%), and on the final clip raised both alarms within 0.6 s of the true times with no false alerts. Details in [model/README.md](../model/README.md). Needs a Windows machine with an RTX GPU.

## 3. Generated footage: Veo (Sunny)

`generate_cctv_pool.py` and `generate_veo_scenario.py` generate short CCTV-style pool scenarios (up to 15 s) with Google's Veo 3.1 (for example, a swimmer who goes under and comes back up), for testing the app on footage that looks like a real security camera. Stills are in `data/veo_footage/`. The scripts need Gemini API access set up locally.

## Other files

| File | What |
|---|---|
| `eval_detectors.py` | Scores any detector (and optionally the tracker with `--edge` rules) against sim ground truth, and writes annotated videos |
| `train_sim_detector.py`, `train_sim_rfdetr.py` | Fine-tune YOLO11n or RF-DETR Nano on MuJoCo frames |

Everything here is synthetic. A model that is perfect on sim has learned our sim bodies, not real people. The detector the app runs was fine-tuned on real wave-pool frames instead.
