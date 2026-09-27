# backend

Local server for the review app in [frontend/](../frontend/). `serve.py` runs a FastAPI server, `worker.py` runs RF-DETR Nano (`detector.py`) and the person tracker (`tracking.py`) over the video, and results are stored per job in SQLite. See [frontend/README.md](../frontend/README.md) for how to run it.

Contracts: [docs/global/architecture.md](../docs/global/architecture.md#data-contracts).
