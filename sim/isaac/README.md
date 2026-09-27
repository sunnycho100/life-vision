# sim/isaac

Synthetic backyard-pool data from Isaac Sim 6.1, with exact labels (boxes, track IDs, and
whether each head is above, partly under, or fully under the water).

| Script | Output |
|---|---|
| `pool_video.py` | A labeled MP4 from a home-style overhead camera. Five people follow scripted paths: swimmer, treader, diver (under about 4 s, twice), silent sinker, and struggler (bobbing, then sinks). Also writes `gt.txt` (MOT format) and `labels.jsonl` with seconds-underwater per person per frame. |
| `pool_replicator.py` | Random still images from an overhead and an underwater camera, with YOLO labels, for detector training. |
| `pool_scene.py` | Shared scene: tiled pool, paver deck, water volume, HDR sky, 6 human characters. |

## Status and limits
- Runs on Windows 11 with an RTX 4070 Laptop GPU (8 GB). NVIDIA's minimum is 16 GB, so keep the resolution moderate.
- **No swimming animations yet.** Bodies keep their default pose (mostly T-pose) and move as a whole. The characters' skeletons don't match NVIDIA's stock idle animation (`stand_idle_loop`).
- The water is a volume with NVIDIA's `Water.mdl`. It still looks too clear, with no ripples or caustics yet.
- Tight boxes treat the water as an occluder, so they cover only the part above the surface (the "upper body only" decision). These are what `gt.txt` and the YOLO labels use. Loose boxes are meant to cover the full body, but they are **misplaced for tilted bodies** (swimmer, sinker lying on the bottom), so don't train on them yet.
- Everything here is synthetic. Test models only on real footage.

## Setup (Windows)
1. Install **Python 3.12** (3.12.10 is the last 3.12 with a Windows installer). Isaac Sim 6.x does not install on 3.13+.
2. Turn on long paths in an Administrator PowerShell, then restart:
   `reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f`
3. From the repo root:
   ```
   py -3.12 -m venv sim\isaac\.venv
   sim\isaac\.venv\Scripts\python.exe -m pip install --upgrade pip
   sim\isaac\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
   sim\isaac\.venv\Scripts\python.exe -m pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com
   sim\isaac\.venv\Scripts\python.exe -m pip install imageio-ffmpeg
   ```
   About 10 GB of downloads. `imageio-ffmpeg` is needed because the OpenCV that ships with Isaac Sim has no video encoder.
4. Accept NVIDIA's license with `set OMNI_KIT_ACCEPT_EULA=YES` (cmd) or `$env:OMNI_KIT_ACCEPT_EULA="YES"` (PowerShell). The first launch compiles shaders for about 5 minutes.

## Run
```
sim\isaac\.venv\Scripts\python.exe sim\isaac\pool_video.py --seconds 20 --fps 15
sim\isaac\.venv\Scripts\python.exe sim\isaac\pool_replicator.py --frames 200
```
Add `--gui` to watch in the Isaac Sim window. Output goes to `sim/isaac/_out_*`, which git ignores.

A harmless error about `omni.scene.optimizer.core` ("filename too long") appears at startup when the repo is in a long path. We don't use that extension.
