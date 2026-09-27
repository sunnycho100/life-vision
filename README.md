# we-fall-we-die

A pool camera assistant for parents. It watches a pool through a camera, tracks every person in the water, and alerts when someone's head has not been seen above the water for too long. It also flags when someone enters the pool.

Status: 12-hour hackathon build. The plan is agreed, the MuJoCo simulation works, everything else is being built now.

> This is a research prototype and a supervision aid. It is not a certified lifesaving device and does not replace watching children, pool fences, or lifeguards.

## Start here
1. [docs/global/architecture.md](docs/global/architecture.md): how it works, data contracts, code layout
2. [docs/global/decisions.md](docs/global/decisions.md): what we agreed on and why
3. [docs/global/milestones.md](docs/global/milestones.md): what each person delivers, hour by hour
4. [presentation/slides/final/life-vision-team-refined.pptx](presentation/slides/final/life-vision-team-refined.pptx): final team presentation with the decision timeline, ownership, verified health context, editable charts, and speaker notes
5. [presentation/presenter-script-team-refined.md](presentation/presenter-script-team-refined.md): timed 5:30 team script, live-demo checklist, evidence caveats, and judge Q&A
6. [docs/presentation.md](docs/presentation.md): supporting presentation narrative and failure plan

Continuing in Cursor or another coding environment? Also read [`CURSOR_HANDOFF.md`](CURSOR_HANDOFF.md).

## Frontend interaction demo

Open [`frontend/index.html`](frontend/index.html) directly, or serve the repository and visit `/frontend/`. It accepts a pool video and demonstrates the Matrix-style activation, tracking overlays, green, yellow, and red decisions, timers, alerts, and incident history with scripted model results.

Old research lives in [docs/archive/](docs/archive/).

## How it works

```
video ──> detector ──> tracker ──> event engine ──> frontend
          (YOLO:       (short-     (head timer,     (4 feeds, colored
           person +     term IDs)   missing people,  boxes, timers,
           head)                    entry alert)     alerts, replay)
```

1. **Detect** people with a pretrained YOLO model, plus evidence of whether each head is above the water.
2. **Track** them across frames with ByteTrack.
3. **Decide** in our own event engine. It keeps its own person IDs, because tracker IDs don't survive someone going underwater. For each person in the pool:
   - Head above water: **green**.
   - Head not seen above water for `T_warn` (5 s, demo 2 s): **yellow**.
   - Still not seen after `T_alarm` (12 s, demo 5 s), or 8 s if they are not moving: **red** plus an audible alert.
   - Track lost inside the pool: shown as **missing** at the last known position, and the timer keeps running.
   - Someone who comes up nearby is matched back to the same person. Someone entering from outside the pool never clears a missing person.
4. **Show** it all in the web app: 4 camera feeds, an "enable monitoring" animation, boxes with timers and reasons, alert history, and incident replay.

Why the head and not the whole body: in clear water, a camera can still see a person lying on the bottom. A "person disappeared" timer would never start. Timing the head fixes that.

For the demo everything runs on prerecorded video. The pipeline writes JSON files, and the frontend plays the video and draws the results in sync.

## Team

| Person | GitHub | Owns | Folder |
|---|---|---|---|
| Joanne | joannemiki57 | Detector (person and head) | `model/detector/` |
| Sam | samkwak188 | Pose, evaluation, test data | `model/pose/`, `model/eval/`, `data/` |
| Sunny | sunnycho100 | Video pipeline, tracker, simulation | `backend/pipeline/`, `sim/` |
| Rohan | rsusarla3 | Event engine (alarm logic) | `backend/events/` |
| David | dpark | Frontend | `frontend/` |

## Repo layout

| Folder | What |
|---|---|
| `docs/global/` | Current plan |
| `docs/archive/` | Earlier research from each member |
| `backend/pipeline/` | Video source, detector wrapper, tracker, writes `tracks.json` |
| `backend/events/` | Event engine, writes `results.json` |
| `model/` | Detector, pose features, evaluation |
| `frontend/` | Web app |
| `sim/mujoco/` | MuJoCo pool simulation |
| `data/` | Clip list, labels, small sim samples (no raw video in git) |

## Data contracts (short version)

- **`tracks.json`** (pipeline → engine): per frame, each person's `track_id`, `bbox`, `conf`, and `head` (`above`, `below`, `unknown`).
- **`results.json`** (engine → frontend): per frame, each person's `person_id`, `bbox`, `state` (`green`, `yellow`, `red`, `missing`), `timer_s`, `reasons`, plus a list of events.
- **`<video>.pool.json`** (frontend → engine): the pool outline drawn on the video.

Full examples: [architecture.md#data-contracts](docs/global/architecture.md#data-contracts).

## Simulation

MuJoCo, runs on a Mac with no GPU.

```bash
python3 -m venv .venv && .venv/bin/pip install mujoco numpy
.venv/bin/python sim/mujoco/pool_scene.py      # 5 people, 5 s, writes data/sim_samples/ and an MP4 (needs ffmpeg)
.venv/bin/python sim/mujoco/drown_sim.py --video   # 4 single-person motions, 20 s each
```

- `pool_scene.py`: stand, swim, and collapse in a 0.9 m shallow end, and float and the instinctive drowning response in a 2 m deep end.
- Buoyancy includes air in the lungs, because MuJoCo has no buoyancy of its own.
- Used for demo feeds and for ground-truth tracks that test the event engine without a trained model.

## Limits
- No real drowning footage exists for us to test on. Tests are acted, simulated, or use a dummy.
- Daylight only. Night-vision IR cameras cannot see underwater.
- Glare, heavy splash, and crowded pools reduce accuracy.
- One `person` class. Telling babies from adults is planned for after the hackathon.
- Ultralytics YOLO is AGPL-3.0. That's fine for this project, but a commercial product would need a license or a switch to RF-DETR (Apache 2.0).
