# we-fall-we-die

AI pool camera for parents. It tracks every person in a home pool and sounds an alarm when someone stays underwater too long.

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

## Current step
Kickoff, all together: label guide, test detection on our own pool footage, agree on formats and API. See the summary for the list.
