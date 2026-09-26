# 04. MuJoCo Simulation

updated: 2026-09-26 · author: sunnycho100 (agent research)

Script: [`sim/mujoco/drown_sim.py`](../../sim/mujoco/drown_sim.py) · Output: [`data/sim_samples/`](../../data/sim_samples/)

## What it does
Simulates the DeepMind humanoid (40.8 kg, Apache 2.0 model from the MuJoCo repo) in a 2 m deep pool for four scripted motions, 20 seconds each, and saves per-frame 3D keypoints with labels. Runs headless on a Mac in about 2 seconds.

```bash
python3 -m venv .venv && .venv/bin/pip install mujoco numpy
.venv/bin/python sim/mujoco/drown_sim.py            # --seconds 20 --fps 30 --seed 0
```
Tested with MuJoCo 3.14.0, NumPy 2.5.3, Python 3.12.

Add `--video` to also render one MP4 per motion to `data/sim_samples/videos/` from a corner "cctv" camera (needs ffmpeg). The scene has a visual-only pool: white walls and a see-through blue water box, 10 x 5 m, 2 m deep. The pool is only drawn, it has no collision and does not change the physics (the keypoint output is identical with or without it). Videos are not committed (`*.mp4` is gitignored).

## Multi-person pool scene
[`sim/mujoco/pool_scene.py`](../../sim/mujoco/pool_scene.py) puts 5 people in one pool, 5 seconds by default (`--seconds` to change):

| Person | Motion | Where | Distress |
|---|---|---|---|
| 0 | stand (arms sway) | shallow end | 0 |
| 1 | normal_swim | shallow end | 0 |
| 2 | float | deep end | 0 |
| 3 | idr (instinctive drowning response) | deep end | 1 |
| 4 | collapse (stands 1 s, then folds under) | shallow end | 1 |

- Pool: shallow end 0.9 m (`--depth`, about waist deep so the upper body shows), deep end 2 m.
- Looks: random skin tone, hair, swimwear, and sometimes a rash guard per person (`--seed`). These are colors on the capsule body, not a real human mesh.
- Standing uses stiff hip and knee gains plus a scripted upright torque (no real balance controller).
- Output: `data/sim_samples/pool_scene.npz` (keypoints per person), `pool_scene.json`, `videos/pool_scene.mp4`.
- MuJoCo prints "Attach conflict" warnings when combining 5 copies of the model. They are harmless.

## Water model

| Part | How | Parameters |
|---|---|---|
| Drag | MuJoCo **ellipsoid** fluid model (`fluidshape="ellipsoid"` on every body geom). Includes viscous drag, lift, and added mass. | `density=1000` kg/m³, `viscosity=0.0009` Pa·s (water at about 25 C) |
| Buoyancy | MuJoCo has **no buoyancy**, so we add it. Each geom gets an upward force `rho * g * V * submerged_fraction`. | Water surface at z = 0 |
| Body density | Tissue slightly denser than water | `TISSUE_BUOYANCY = 0.98` |
| Lungs | Air in the lungs adds lift at the chest geoms. Full lungs float, empty lungs sink. | Lung air in liters, set per motion |
| Muscles | PD torques toward scripted joint targets | `kp = 60`, `kd = 4` |
| Swim thrust | Constant 25 N forward force (the fluid model alone barely moves a stroking arm) | Scripted, not physical |

The humanoid's mass equals its geom volume times 1000 kg/m³, so buoyancy is easy to reason about.

## Motions

| Motion | Start pose | Lung air | What the joints do | Distress label |
|---|---|---|---|---|
| `normal_swim` | Face down | 3.0 L | Alternating freestyle arms (0.8 Hz), flutter kick (2 Hz) | 0 |
| `float` | Face up | 3.0 L | Arms out, relaxed | 0 |
| `idr` | Upright | 3.0 → 0.5 L over 20 s | Instinctive drowning response: arms out to the side pressing down (1.2 Hz), head tilted back, weak legs | 1 |
| `silent_sink` | Upright | 1.0 → 0 L over 5 s | Limp, almost no movement | 1 |

## Output
`data/sim_samples/sim_samples.npz` (325 KB):
- `keypoints`: (4 motions, 600 frames, 15 keypoints, xyz) in meters
- `head_underwater`: (4, 600) true when the top of the head is below the surface
- `distress`: (4,) label per motion
- `motions`, `keypoint_names`, `fps`

Keypoints: head, torso, pelvis, both shoulders, elbows, hands, hips, knees, feet.

## Run result (seed 0, 20 s, 30 FPS)

| Motion | Distress | Head z start | Head z end | Head z min | % frames head underwater | First underwater (s) | Travel (m) |
|---|---|---|---|---|---|---|---|
| normal_swim | 0 | -0.051 | -0.064 | -0.089 | 0.0 | None | 5.03 |
| float | 0 | -0.046 | -0.024 | -0.081 | 0.0 | None | 0.327 |
| idr | 1 | 0.042 | -0.125 | -0.125 | 7.2 | 18.13 | 0.263 |
| silent_sink | 1 | 0.043 | -1.838 | -1.854 | 93.8 | 1.23 | 1.134 |

(Head z is the head center. The head radius is 0.09 m, so a center at -0.05 means the top of the head is still above water.)

What this shows:
- Swimmer and floater stay at the surface the whole time.
- IDR keeps the head at the surface for about 18 seconds, then goes under. Real IDR is commonly described as lasting about 20 to 60 seconds (unverified, not found in a primary source during this research), so this is in a believable range.
- Silent sink goes under in about 1 second and reaches the pool floor. This is the toddler case: no struggle, nothing for a "splashing" detector to see.

How we got here: the first version used one buoyancy number for the whole body. The swimmer slowly dove to 0.9 m and IDR went under in 1.3 s. Moving the extra lift to the chest (lungs) fixed both, which matches how real bodies float (chest up, legs down).

## Limitations
- Fluid drag applies everywhere, also above the water (MuJoCo has one global medium). Arms out of the water get too much drag.
- No splash, waves, or refraction. The rendered pool is a flat tinted box, good for checking motions and demos, not realistic enough for detector training.
- One adult-sized body. A child needs a scaled model (shorter limbs, bigger head ratio).
- Motions are hand-scripted sine waves, not motion capture. Our real footage should be used to check them.
- Keypoints are 3D world coordinates. To train on camera views we must project them into an overhead camera (next step).

## Next steps
1. Project keypoints into an overhead camera and a side camera (2D), to match what a pose model sees.
2. Randomize per run: lung air, stroke rate, body scale, start position, seed. Generate hundreds of sequences.
3. Add more normal motions that look like distress: breath-hold, playing dead, treading water, bobbing.
4. Export joint angles as BVH for Isaac Sim (see 05-isaac-sim.md).
5. Compare the sim IDR against our acted footage once we have it.

## Sources
- [MuJoCo fluid forces docs](https://mujoco.readthedocs.io/en/stable/computation/fluid.html), [fluid.rst source](https://github.com/google-deepmind/mujoco/blob/main/doc/computation/fluid.rst)
- [DeepMind humanoid model](https://github.com/google-deepmind/mujoco/tree/main/model/humanoid)
- Drowning behaviors (IDR, climbing ladder motion, backward stroke) from Carballo-Fazanes et al., as summarized in [this PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548417) (unverified which of the search results carried the summary)
