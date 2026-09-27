# sim/isaac

The realistic half of our simulation work (Rohan). The MuJoCo sim showed the pipeline works, but its capsule bodies and flat water look nothing like a real pool. Here the pool is path-traced in NVIDIA Isaac Sim 6.1: the water surface moves every frame, light refracts through the waves, and six animated people swim, float, dive, struggle, and sink. Every frame comes with exact labels (boxes, track IDs, and whether each head is above, partly under, or fully under the water), so we can train detectors on it and score the alarms against the true times.

Results of the detectors trained here: [model/README.md](../../model/README.md#results-on-the-held-out-seed-0-clip-300-frames-1500-person-boxes-526-with-the-head-fully-under). Demo videos: [`presentation/videos/`](../../presentation/videos/).

| Script | Output |
|---|---|
| `pool_video.py` | A labeled 20 s MP4 (30 fps) from a home-style overhead camera. Six people: a lap swimmer (front crawl, body roll, smooth wall turns), a back floater, a treader, a diver who swims under water and resurfaces elsewhere (under about 3.5 s, then a 2 s duck), a child-sized silent sinker (alarm case), and an adult doing the instinctive drowning response and then sinking (alarm case). Writes `gt.txt` (MOT format, full-body boxes with class) and `labels.jsonl` (head height, the local water level, head state, GREEN / ORANGE / RED status and reason, seconds underwater per person per frame). `pool_labeled.mp4` colors each box with the shared rules in [`model/alert_rules.py`](../../model/alert_rules.py) (lifeguard 10/20 rule, ASTM F3698, drowning physiology, instinctive drowning response signs; see [model/README.md](../../model/README.md#box-colors-green--orange--red)). Underwater parts of each box are placed where the camera actually sees them: each joint's light ray is refracted through the rippled surface with Snell's law, using the local wave height and normal. `--seed N` varies layout, timing, cast, sky and camera for training clips; `--random` renders random stills; `--yolo-every K` exports YOLO frames; `--start` renders from a later time. |
| `pool_replicator.py` | Random still images from an overhead and an underwater camera, with YOLO labels, for detector training. |
| `pool_scene.py` | Shared scene: tiled pool, paver deck, HDR sky, 6 human characters with work gear (hats, vests, radios, badges, lab coats) hidden. The water is one closed mesh whose top surface moves every frame: ambient waves with real water dispersion, plus rings spreading from each person. People are placed and labeled by **head center**, with optional body roll and scale (0.62 = child). |
| `pool_anim.py` | Procedural body motion (arms, hands, legs, spine, head): front crawl with breathing, treading, back float, a child's dog paddle, streamline dive, underwater breaststroke, the drowning response, and a limp float. A per-person `wobble` keeps motions irregular. |
| `pool_flythrough.py` | Runs `pool_video.py` with a camera that tours the backyard and settles into the CCTV view, for the demo video. |
| `make_training_set.ps1` | Renders the 261-frame YOLO training set (81 random stills + 3 scripted clips, seeds 1-3). |

## Status and limits
- Runs on Windows 11 with an RTX 4070 Laptop GPU (8 GB). NVIDIA's minimum is 16 GB, so keep the resolution moderate.
- **Motion is procedural, not motion capture.** The stock animations don't match these skeletons, so `pool_anim.py` aims each bone at a direction every frame. It reads as swimming or struggling from a distance, but not up close. Fingers don't move.
- **Use `--pathtrace 32` (or 64 for a final cut) for realistic water** (refraction, tint, waterline, moving ripples). The default real-time renderer ignores the water's absorption, so the pool looks empty. Path tracing at 1080p takes about 1-2 s per frame on an RTX 4070 Laptop.
- `gt.txt` and the YOLO labels use **full-body boxes**: the skeleton projected through the camera, with underwater joints moved to where refraction makes them appear (Snell's law against the mean surface). `labels.jsonl` also keeps the above-water-only box from Isaac's annotator.
- Head state (`above`, `partial`, `below`) comes from the head center: `below` means the whole head is under the mean surface.
- The water mesh reaches 10 cm into the walls and floor. Stopping at the tile leaves a thin air gap that reflects underwater light back (total internal reflection) and turns the walls black.
- Two soft underwater fill lights (`--fill`, default 150) stand in for the light real pool water scatters. The path tracer can't carry direct sunlight through the rippling surface.
- Gloves stay on: the skin under them has no texture and renders white.
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
sim\isaac\.venv\Scripts\python.exe sim\isaac\pool_video.py --seconds 20 --fps 30 --pathtrace 64 --subframes 1
sim\isaac\.venv\Scripts\python.exe sim\isaac\pool_replicator.py --frames 200
```
Add `--gui` to watch in the Isaac Sim window. Output goes to `sim/isaac/_out_*`, which git ignores.

A harmless error about `omni.scene.optimizer.core` ("filename too long") appears at startup when the repo is in a long path. We don't use that extension.
