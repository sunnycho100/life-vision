# we-fall-we-die

AI pool camera for parents. It tracks every person in a home pool and sounds an alarm when someone stays underwater too long.

Continuing this project in Cursor or another coding environment? Start with [`CURSOR_HANDOFF.md`](CURSOR_HANDOFF.md).

## Repo layout
| Folder | What | Owner |
|---|---|---|
| `docs/` | Technical summary, research notes | Everyone |
| `sim/` | MuJoCo and Isaac Sim synthetic data | Person 1 |
| `data/` | Label guide, dataset versions, conversion scripts (no raw video) | Person 2 |
| `model/` | Detector, tracker, fine-tuning, evaluation | Person 3 |
| `backend/` | Stream ingest, buffer, inference, incidents, alert API | Person 4 |
| `frontend/` | Parent dashboard and alerts | Person 5 |

## Start here
1. [docs/technical-summary.md](docs/technical-summary.md): what we have decided so far and what is still open.
2. [docs/research-notes.md](docs/research-notes.md): competitors, limitations, team plan.
3. [docs/dpark/data-model-camera-guide.md](docs/dpark/data-model-camera-guide.md): usable datasets and models, the 24-hour workflow, Ring/Nest integration findings, and frontend next steps.
4. [docs/presentation.md](docs/presentation.md): presentation narrative, technical architecture, live-demo script, and failure plan.

## Frontend interaction demo

Open [`frontend/index.html`](frontend/index.html) directly, or serve the repository and visit `/frontend/`. It accepts a pool video and demonstrates the Matrix-style activation, tracking overlays, explainable green/yellow/red decisions, timers, alerts, and incident history with scripted model results.

## Current step
Kickoff, all together: label guide, test detection on our own pool footage, agree on formats and API. See the summary for the list.
