# LifeVision

Every summer, a camera watches a pool and sees nothing. A toddler slips into the water while a parent looks away. There is no commotion, just silence. Drowning is the leading cause of death for children ages 1 to 4, and after decades of decline, U.S. drowning rates began rising again in 2020.

LifeVision turns a pool camera people already own into a second set of eyes. It finds every person in the water, keeps an ID on each one, and raises a warning when someone goes out of sight for too long. An adult still has to respond, but the silence doesn't go unnoticed.

## The problem we ran into first

Nobody has footage of kids drowning, and we couldn't stage it. So we couldn't just download a dataset and train on it. Instead we built the hard cases ourselves:

1. **Simulated them.** MuJoCo and NVIDIA Isaac Sim pools where people swim, float, dive, struggle, and sink, with exact labels from the simulator (every box, every head height, every second under water).
2. **Generated them.** CCTV-style pool scenarios made with Google's Veo 3.1, used as extra test footage.
3. **Labeled real pools.** Real wave-pool frames from YouTube, hand-boxed, to fine-tune the detector on real people.

Then we used the sims as a test bench with exact answers, and the real frames to fine-tune the model that runs in the app.

## What's in here

| Part | What it does | Where |
|---|---|---|
| Review app | Upload a pool video, draw the pool outline, run detection and tracking, and replay it with colored boxes and warnings | [`frontend/`](frontend/), [`backend/`](backend/) |
| Detector | RF-DETR Small fine-tuned on real wave-pool frames (runs on the Mac GPU) | [`model/`](model/), [`backend/detector.py`](backend/detector.py) |
| Tracker | ByteTrack plus our own ID rules, so a person keeps one ID when they go under and come back | [`backend/tracking.py`](backend/tracking.py) |
| MuJoCo sim | Five scripted pool scenarios with exact ground truth, runs on a laptop | [`sim/mujoco/`](sim/mujoco/) |
| Isaac Sim | Path-traced backyard pool with moving water, refraction, and six animated people | [`sim/isaac/`](sim/isaac/) |
| Veo scenarios | Generated CCTV pool footage | [`sim/generate_veo_scenario.py`](sim/generate_veo_scenario.py), [`data/veo_footage/`](data/veo_footage/) |
| Alert rules | One GREEN / ORANGE / RED rule set, based on lifeguard and pool-alarm standards | [`model/alert_rules.py`](model/alert_rules.py) |
| Presentation | Final deck, script, and demo videos | [`presentation/`](presentation/) |

## How it works

```
pool video ──> RF-DETR Small ──> tracker ──> warnings ──> review app
               (full frame +     (one ID     (missing     (boxes, IDs,
                6 tiles, GPU)     per person) 5 s, 12 s)   timers, replay)
```

1. **Detect.** Every sampled frame (10 per second) goes through the detector once whole and once as six overlapping tiles, so small far-away swimmers are still found.
2. **Track.** ByteTrack links boxes frame to frame. On top of that, a lost person's ID only carries over to a new box if the move is physically possible, so a lifeguard swimming past can't inherit a drowning swimmer's ID.
3. **Warn.** A person who disappears inside the pool outline is held at their last spot. After 5 s they turn yellow, after 12 s red. If a neighbour's box is covering their spot, the clock pauses instead of firing.
4. **Review.** The app shows it all in sync with the video, lists the incidents, and cuts a short clip of each one.

## What the simulations showed

All numbers below are on simulated video unless marked real. Good sim scores prove the pipeline works end to end, not that it works on real pools.

**MuJoCo (5 scenarios, held-out `resurface` scenario).** Stock detectors barely saw the simulated swimmers. Fine-tuning on frames the simulator labels for free fixed that in minutes.

| Setup | Recall | Head fully under | IDs shown for 4 people |
|---|---|---|---|
| Stock YOLO11n | 0.25 | 0.02 | |
| Stock RF-DETR Nano + default tracker | 0.79 | 0.60 | 18 |
| RF-DETR Nano fine-tuned on sim + our tracking rules | 1.00 | 1.00 | 4 |

It also showed the clear-water problem: a good detector keeps seeing someone lying on the bottom, so a "person disappeared" timer never starts. That's why the Isaac Sim work tracks head state, not just disappearance.

**Isaac Sim (held-out 20 s clip, 1,500 person boxes).** A YOLO11n trained on 261 Isaac frames found 95% of people with their head fully under water, against 16% for stock YOLO11n. On the final clip it raised the child's alarm at 13.5 s (true time about 13.4 s) and the struggler's at 16.9 s (about 17.5 s), with no false alerts.

**Real wave pool (Joanne's fine-tune, 16 validation frames).** RF-DETR Small went from F1 0.39 to 0.75 (recall 0.27 to 0.84). The validation frames come from the same video as part of the training set, so this is optimistic for other pools.

**In the app.** Switching from stock RF-DETR Nano on CPU to the fine-tuned RF-DETR Small on the Mac GPU cut 10 s of video from 88 s to 48 s of processing.

Details: [docs/global/simulation.md](docs/global/simulation.md) (MuJoCo), [model/README.md](model/README.md) (Isaac Sim and the real wave pool).

## Run the review app

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-reference.txt
.venv/bin/python backend/serve.py --backend python
```

Open http://127.0.0.1:5173. Setup for the fine-tuned weights and more options: [backend/README.md](backend/README.md). Using the app: [frontend/README.md](frontend/README.md).

## Repo layout

| Folder | What |
|---|---|
| `backend/` | FastAPI server, detection worker, tracker, incident review |
| `frontend/` | The review app (plain HTML, CSS, JS) and the annotation pages |
| `model/` | Isaac Sim detector training, alert rules, the real wave-pool fine-tune |
| `sim/` | MuJoCo and Isaac Sim scenes, sim detector training and evaluation, Veo generation |
| `data/` | Sim ground truth, the wave-pool dataset, generated frames (no raw video in git) |
| `tools/` | Model preparation, evaluation, rescue-video annotation, re-tracking |
| `docs/` | The plan and decisions (`global/`), earlier research (`archive/`) |
| `presentation/` | Deck, script, demo videos, charts |

## Team

| Person | GitHub | Worked on |
|---|---|---|
| Sunny | sunnycho100 | MuJoCo sim, sim-trained detectors, tracking rules, review app model and GPU integration |
| Rohan | rsusarla3 | Isaac Sim pool, Isaac-trained detectors, alert rules |
| Joanne | joannemiki57 | Real wave-pool dataset and RF-DETR Small fine-tune |
| Sam | samkwak188 | Review app, evaluation, rescue-video annotation |
| David | dpark | Presentation and design |

## Limits

- No real drowning footage. Everything we tested on is simulated, generated, or acted.
- The real fine-tune used 80 frames from wave-pool videos. Other pools, cameras, and lighting need their own labeled frames.
- Daylight only. Glare, heavy splash, and crowded pools lower accuracy.
- One `person` class. Telling children from adults is future work.
- Processing is still slower than real time (about 5 s per second of video on a MacBook).

## License

MIT, see [LICENSE](LICENSE). Ultralytics YOLO and the YOLO models trained here are AGPL-3.0. RF-DETR is Apache 2.0.
