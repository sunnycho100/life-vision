# backend

The local server behind the review app in [`frontend/`](../frontend/). It takes a pool video, runs the detector and tracker over it in a separate worker process, stores the results, and serves them back to the browser.

| File | What |
|---|---|
| `serve.py` | FastAPI server: sources (videos), jobs, observations, and the static frontend |
| `worker.py` | One analysis job: samples 10 frames per video second, detects, tracks, writes results to SQLite |
| `detector.py` | RF-DETR wrappers. Loads the fine-tuned RF-DETR Small or the stock Nano from a model manifest, plus full frame + 3x2 tiling |
| `tracking.py` | `PoolTracker`: ByteTrack plus our ID rules (plausible moves only, re-linking people who resurface) and the missing, warning, and alarm levels |
| `monitoring.py` | Incident review over a finished job: who was inside the pool outline, for how long they were out of sight, and when to warn |
| `review.py`, `clip_worker.py` | Review API and short incident clips |
| `common.py`, `jobs.py` | Video probing and sampling, job bookkeeping |

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-reference.txt
.venv/bin/python backend/serve.py --backend python
```

Open http://127.0.0.1:5173. Without `--video`, the server downloads the wave-pool reference video from YouTube once into `artifacts/reference/`.

Useful options:

| Option | What |
|---|---|
| `--video PATH` | The reference video shown first in the source list |
| `--extra-video PATH LABEL` | Another local video to list as a source (repeatable) |
| `--model-manifest PATH` | Which model to load (see below) |
| `--provider mps` or `cpu` | Where the model runs. Defaults to the Mac GPU (MPS) when available |
| `--port`, `--data-dir` | Port, and where jobs and uploads are stored (default `artifacts/poolside/`) |

## The model

By default the server uses Joanne's RF-DETR Small, fine-tuned on real wave-pool frames, if its files are in `artifacts/models/rfdetr-s-person-v1/`. Otherwise it falls back to stock RF-DETR Nano in `artifacts/models/rfdetr-nano/`.

To set up the fine-tuned model, download `rfdetr_s_person_best.pth` from the [`rfdetr-s-person-v1` release](https://github.com/sunnycho100/life-vision/releases/tag/rfdetr-s-person-v1) into `artifacts/models/rfdetr-s-person-v1/`, and save this next to it as `manifest.json`:

```json
{
  "label": "RF-DETR Small (fine-tuned)",
  "model": "RFDETRSmall", "rfdetr_version": "1.11.0", "resolution": 512, "precision": "fp32",
  "person_id": 0, "num_classes": 1, "class_names": {"0": "person"}, "background_id": null, "num_select": 300,
  "checkpoint": {"file": "rfdetr_s_person_best.pth",
                 "sha256": "<sha256 of the .pth file>"},
  "parity": {"passed": false}
}
```

Get the hash with `shasum -a 256 rfdetr_s_person_best.pth`. The server checks it before loading.

On a MacBook GPU, 10 s of video takes about 48 s to process (88 s with stock Nano on CPU). Most of the time goes to the six tiles per frame.

After changing `tracking.py`, re-run only the tracker over a finished job instead of re-detecting: `.venv/bin/python -m tools.retrack_job artifacts/poolside/jobs/<job id>`.

## Precomputed analyses

The [`analyses-v1` release](https://github.com/sunnycho100/life-vision/releases/tag/analyses-v1) has finished jobs for the full `NycwxaU4GPw` wave-pool video and two 10 s clips, so they play back without the 14 minutes of detection. Set up the fine-tuned model first, then:

```bash
.venv/bin/python -m tools.analysis_bundle import lifevision_analyses_v1.zip
S=artifacts/poolside/sources
.venv/bin/python backend/serve.py --backend python \
  --extra-video $S/80dc472660c3f7fd0c857c672647f5eaecbc595007281341533d165b9d2ef359.mp4 "Wave pool NycwxaU4GPw" \
  --extra-video $S/fcd076c383c0670e7df1ac5f24fee0c10f1f77271c7ecbcc0c5d0d787c363e03.mp4 "NycwxaU4GPw 35-45 s" \
  --extra-video $S/14c6355bf0ee95dc50b72173438156ca09fe3072ddb5c0c9368ad465e23ea2ef.mp4 "UaFwQMfQThE 35-45 s"
```

Pick a video in the source list and press Analyze: it returns the stored job. Uploading the same files works too. This only holds while `detector.py`, `worker.py`, `common.py` and `tracking.py` are unchanged. Share your own finished jobs with `.venv/bin/python -m tools.analysis_bundle export <job id> ... -o bundle.zip`.

## Tests

```bash
.venv/bin/pip install -r backend/requirements-dev.txt
.venv/bin/python -m pytest backend -q
```

Data contracts: [docs/global/architecture.md](../docs/global/architecture.md#data-contracts).
