# Resume in Cursor: GREEN / ORANGE / RED boxes from safety norms (Isaac Sim)

updated: 2026-09-26 22:30 · owner: Rohan (@rsusarla3) · status: **committed and pushed**

## The task

Rohan asked: *"When we're labeling the red, green and orange boxes for the underwater detection, use correct norms and things you can approximate to make the detection boxes, and push it to GitHub."*

Done so far:
1. **Box colors follow published norms.** They no longer just say whether the head is above water. One shared rule set, `model/alert_rules.py`, colors both the sim ground truth and the detector output.
2. **Underwater boxes use the real water surface.** Each submerged joint's light ray is bent through the rippled surface with Snell's law, using the local wave height and normal, not a flat z = 0 plane.

## The rules (`model/alert_rules.py`)

| Color | Rule | Norm |
|---|---|---|
| RED | Head fully under **10 s** | Ellis & Associates 10/20 lifeguard rule (10 s to recognize); stricter than ASTM F3698-24 (20 s motionless dummy test); breath-hold ≤ ~1 min, unconscious within ~2 min (NEJM 2012) |
| RED | Head under **5 s and not moving** 2 s | Motionless under water = strongest sign (Coral MYLO, ASTM test case) |
| ORANGE | Head under **5 s** | Early warning at half the red time. Our choice; tune on real footage |
| ORANGE | **Drowning signs at the surface 3 s**: upright (< 30°), mouth at the waterline (head center < 0.75 head radius above water), no headway | Instinctive drowning response (Pia 1974; Vittone, USCG). The surface struggle lasts only 20–60 s |
| ORANGE | Went under after showing drowning signs | Latches until the head has been clearly up for 2 s |
| GREEN | Everything else | |

Full table, input definitions and sources: `model/README.md`, section "Box colors: GREEN / ORANGE / RED".

Inputs: the sim measures everything exactly. The detector approximates:
- **Head under:** the YOLO `underwater` class, majority over the last second, with the timer backdated.
- **Not moving:** box corners moved < 15% of the box diagonal in 2 s.
- **Posture and waterline:** not observable with the 2-class model, so the orange "drowning signs" rule only shows in the ground truth.

## Uncommitted changes (all mine, ready to commit)

| File | Change |
|---|---|
| `model/alert_rules.py` | **New.** `StatusTracker.update(pid, t, head_under, mouth_low, upright, pos, points, size, under_start)` returns `{status, reason, seconds_under, seconds_distress}`. Also `COLORS_BGR` and the threshold constants. No dependencies. |
| `model/detect_drowning.py` | Colors come from `alert_rules`. The class is smoothed (`Person.smoothed`, majority over 1 s). A lost underwater record that overlaps a visible underwater person is dropped as a duplicate. Scoring prints the first ORANGE and RED times vs the ground truth. |
| `sim/isaac/pool_scene.py` | `WaterSurface.sample(x, y)` returns the height and slopes. New helpers: `head_world`, `joint_world`, `body_tilt`, `key_points`, `scale_of`. |
| `sim/isaac/pool_video.py` | `refract()` uses the wave height and normals. Every box is colored by `alert_rules` with a reason label, and there's a legend. `labels.jsonl` gains `status`, `status_reason`, `surface_z`, `tilt_deg`, `seconds_distress`. |
| `model/README.md`, `sim/isaac/README.md` | Rules, norms, inputs and sources documented. |

Not mine, leave alone: `model/train_rfdetr.py`, `model/detect_drowning_rfdetr.py`, `presentation/make_demo_video.py` (the original session's demo work). Delete these scratch files, don't commit them: `sim/isaac/_out_final_20260926-2216.log`, `sim/isaac/_peek_*.jpg`.

## Results so far: final clip `sim/isaac/_out_final_20260926-2216/` (local, git-ignored)

Rendered with the new rules: 20 s, 30 fps, 1920×1088, path-traced. It contains `pool.mp4`, `pool_labeled.mp4` (ground truth with the new colors), `labels.jsonl`, `gt.txt`, `pool_rfdetr-s.mp4`, `pool_yolo_alarms.mp4`, and YOLO frames.

Ground truth (exact):

| Person | Orange | Red |
|---|---|---|
| struggler | 6.0 s (drowning signs) | 14.5 s (under 5 s, not moving) |
| child (silent sink) | 6.4 s (drowning signs) | 10.5 s (under 5 s, not moving) |
| swimmer, floater, treader, diver | never | never |

YOLO11n + tracker, **first run before the last fix**:
- Found 97% of swimming and 95% of underwater people; class correct 95% / 97%.
- Child red 10.5 s ✔, struggler red 14.7 s ✔, no alerts on the swimmer, floater or treader.
- **Diver: false orange 9.0 s and red 9.1 s.** Cause: a flickering duplicate box (track #9) kept a timer after it vanished, and its frozen history read as "not moving". Two fixes are in (stale history is never "still"; duplicates of a visible underwater person are dropped). **Not yet re-verified. That's step 1 below.**
- Orange from surface drowning signs can't come from the detector yet (see Inputs above).

RF-DETR-S zero-shot + ByteTrack on this clip (`benchmark_rfdetr-s_track.json`): precision 0.79, recall 0.77 at IoU 0.5 (head above 86%, head under 55%); at IoU 0.3, recall 0.88.

## What's left, in order

Run from the repo root in PowerShell.

1. **Re-verify the detector** (about 2 min on the GPU). Expect the diver to have no alerts, child red ~10.5 s, struggler red ~14.5–14.7 s:
   ```
   model\.venv\Scripts\python.exe model\detect_drowning.py --video sim\isaac\_out_final_20260926-2216\pool.mp4 --gt sim\isaac\_out_final_20260926-2216\labels.jsonl --out sim\isaac\_out_final_20260926-2216\pool_yolo_alarms.mp4
   ```
2. **Make the 720p presentation copies** (H.264, CRF 23; each stays under ~5 MB, per `presentation/README.md`):
   ```
   $ff = model\.venv\Scripts\python.exe -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
   $src = "sim\isaac\_out_final_20260926-2216"; $dst = "presentation\videos\isaac_pool_final_20260926-2216"
   & $ff -y -i "$src\pool.mp4"             -vf scale=1280:-2 -c:v libx264 -crf 23 -pix_fmt yuv420p "$dst.mp4"
   & $ff -y -i "$src\pool_labeled.mp4"     -vf scale=1280:-2 -c:v libx264 -crf 23 -pix_fmt yuv420p "${dst}_groundtruth.mp4"
   & $ff -y -i "$src\pool_rfdetr-s.mp4"    -vf scale=1280:-2 -c:v libx264 -crf 23 -pix_fmt yuv420p "${dst}_rfdetr.mp4"
   & $ff -y -i "$src\pool_yolo_alarms.mp4" -vf scale=1280:-2 -c:v libx264 -crf 23 -pix_fmt yuv420p "${dst}_yolo_alarms.mp4"
   ```
3. **Update `presentation/README.md`:** add rows for the four `isaac_pool_final_20260926-2216*` files and mark them as the latest generation. The 21:51 rows are currently marked "Latest"; change them to "previous".
4. **Add the step-1 results to `model/README.md`** as a new "Final clip (2026-09-26 22:16)" section, next to the 21:51 one.
5. **Commit and push as Rohan, with no Claude/AI attribution lines, ever:**
   ```
   git pull --rebase origin main
   git add model/alert_rules.py model/detect_drowning.py model/README.md sim/isaac/pool_scene.py sim/isaac/pool_video.py sim/isaac/README.md presentation/README.md presentation/videos/isaac_pool_final_20260926-2216*.mp4
   git commit -m "Box colors from safety norms (10/20 rule, ASTM F3698, drowning signs); wave-correct underwater boxes"
   git push origin main
   ```
   The repo's git identity is already set to `Rohan Susarla <rohansusarla3@gmail.com>`. A local `.git/hooks/commit-msg` hook strips any Claude co-author line as a backstop. Never use `--no-verify`.
6. **Tell the original Claude session** ("Crowning detection computer vision", cc5d4c) the commit hash. It's waiting to rebase its RF-DETR demo work on top.

## Environments (already installed on this laptop)

- **Isaac Sim 6.1:** `sim\isaac\.venv` (Python 3.12.10), also reachable as `C:\isim` (junction). Set `$env:OMNI_KIT_ACCEPT_EULA="YES"` first. Scripts: `sim\isaac\pool_video.py` (scripted clip; `--pathtrace 64 --subframes 1` for final quality, about 0.6 s per frame at 1080p) and `pool_replicator.py` (stills).
- **Models:** `model\.venv` has torch 2.11 + CUDA 12.8, ultralytics 8.4.163, rfdetr 1.11, imageio-ffmpeg. Trained weights are in `model\runs\pool_yolo11n\weights\best.pt` (git-ignored).
- GPU: RTX 4070 Laptop, 8 GB. Run one heavy job at a time.

## Gotchas

- The startup error about `omni.scene.optimizer.core` ("filename too long") is harmless.
- A path-traced run can hang at exit after the last frame. If `labels.jsonl` has every line and `DONE` is logged, kill the process.
- PowerShell mangles inline `python -c` scripts with quotes. Use a `.py` file or Git Bash.
- Teammates push often: always `git pull --rebase` before committing.
- The sim is sim-only. Say "on simulation", and "detects prolonged head submersion", not "detects drowning" (`presentation/README.md` rules).

## Open question for Rohan

In the video, the child and the struggler go limp on the pool floor within ~4 s of going under. Physiologically, a person is usually still conscious and moving for up to about a minute. Choose:
1. Physiologically accurate: weak movement under water. Red then comes from the 10 s timer, not the "not moving" rule.
2. Keep the compressed version and label the video as time-compressed.

Not answered yet; the current render uses the compressed version.

## Key files

| Path | What |
|---|---|
| `model/alert_rules.py` | The color rules (shared) |
| `model/detect_drowning.py` | YOLO11n + ByteTrack + rules → `*_yolo_alarms.mp4` and score |
| `model/benchmark_isaac.py` | Any detector (incl. zero-shot RF-DETR) scored by head above / under; `--track --clean` for the presentation overlay |
| `sim/isaac/pool_video.py` | Scripted 6-person clip + ground truth (seed 0 = demo; other seeds = training data) |
| `sim/isaac/pool_scene.py` | Pool, wave-surface water, characters, placement by head center |
| `sim/isaac/pool_anim.py` | Procedural swim, tread, float, dive, struggle and limp motions |
| `sim/isaac/make_training_set.ps1`, `model/train_yolo.py` | Training data (261 frames) and YOLO11n fine-tune |
| `docs/archive/rsusarla3/` | Earlier research: feasibility, detection logic, standards |
