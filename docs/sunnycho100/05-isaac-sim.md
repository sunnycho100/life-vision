# 05. Isaac Sim

updated: 2026-09-26 · author: sunnycho100 (agent research)

Starter script: [`sim/isaac/pool_replicator.py`](../../sim/isaac/pool_replicator.py) (**unverified**, not run; Isaac Sim cannot install on a Mac)

## Short answer
- Use Isaac Sim for **labeled images of people partly in water**, to help the detector. Use MuJoCo for **motion**.
- Start with **static characters at random depths** (no animation). That already produces the "only upper body visible" images we need, with perfect boxes.
- Animating MuJoCo motion on a human character is possible but is a separate, harder project (retargeting). Do it only after the static version works.

## Hardware
| | Requirement |
|---|---|
| OS | Ubuntu 22.04 or 24.04, Windows 10 or 11. **macOS not supported.** |
| GPU | RTX with RT cores. Minimum GeForce RTX 4080, good RTX 5080, ideal RTX PRO 6000 Blackwell. **A100 and H100 are not supported** (no RT cores). |
| VRAM | 16 GB minimum |
| Driver | Linux 580.65.06, Windows 580.88 (recommended) |
| From a Mac | Run headless on a remote RTX machine and view it with the Isaac Sim WebRTC Streaming Client (works on macOS) |

Action item: find out whether any teammate or the university has an RTX 4080+ machine. Without one, this part stays on paper.

## Human avatars, not robots
- Isaac Sim ships human character assets (the People characters used by `omni.anim.people` and Isaac Replicator Agent). Use those so the detector sees people.
- Tag every character with the semantic class `person`. Replicator then writes the 2D boxes automatically.
- **Isaac Replicator Agent (IRA)** generates data of human characters walking and acting in 3D scenes from a config file. Useful for people **around** the pool (walking, sitting at the edge). Its built-in animations are land motions, not swimming.

## Replicator plan
1. Build a pool: floor, walls, a water surface plane at z = 0.
2. Load several human characters.
3. Each frame: move characters to random x, y and a random depth so the surface cuts them at the waist, chest, neck, or above the head. Random rotation.
4. Randomize light intensity (glare vs shade) and later water color and camera height.
5. Two cameras: overhead (like a home camera) and underwater.
6. `BasicWriter` with `rgb`, `bounding_box_2d_tight`, `semantic_segmentation`. Other annotators include depth, 3D boxes, normals, occlusion.
7. Convert the output boxes to YOLO format and mix them with real frames (see 02-yolo-training-pipeline.md). Always test on real frames only.

## Water rendering limits
- Isaac Sim has **no built-in underwater environment**. Water is a material on a surface. Community work (for example `isaac_underwater`) and OceanSim from the University of Michigan add underwater rendering on top.
- OceanSim uses physically based rendering for wavelength-dependent color loss, turbidity, and lighting. It targets underwater robots but could give our underwater camera a realistic look (unverified for pools).
- No splashes or surface waves unless we add particle fluids, which is slow.
- Transparent water plus ray tracing is expensive. Expect slow rendering per frame.
- Sim images will still look like sim. Mixing with real frames and testing only on real frames is required.

## MuJoCo motion onto a human character
Pipeline, all steps unverified for our setup:
1. Save MuJoCo joint angles per frame (our script saves positions today; add `qpos` export).
2. Write them as a **BVH** file using the DeepMind humanoid joint hierarchy.
3. **Retarget** to the character's skeleton. Isaac Sim needs animations compatible with the NVIDIA biped skeleton and provides a retarget script. Blender or tools like SOMA Retargeter (BVH, CSV, USD) and GMR (SMPL-X and BVH) can do this too.
4. Export as USD `SkelAnimation` and bind it to the character.
5. Render with the same Replicator setup, now with a label per frame (distress or not) from the MuJoCo run.

The MuJoCo humanoid has no neck, wrists, or fingers, so the retargeted character will look stiff in those joints.

## Sources
- [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [Replicator Agent: actor simulation and synthetic data](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/action_and_event_data_generation/tutorial_replicator_agent.html), [IRA customization and retargeting](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/action_and_event_data_generation/ext_replicator-agent/customization.html)
- [Replicator writers](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.replicator.writers/docs/index.html)
- [Synthetic data with Replicator (Medium)](https://medium.com/@soulina/generating-synthetic-training-data-for-object-detection-using-isaac-sims-replicator-a9a986ad63b4)
- [isaac_underwater](https://github.com/leonlime/isaac_underwater), [OceanSim (arXiv)](https://arxiv.org/html/2503.01074v1), [NVIDIA forum: no underwater environment](https://forums.developer.nvidia.com/t/create-water-with-dynamics-for-underwater-robots/266545)
- [SOMA Retargeter](https://github.com/MotomindKR/soma-retargeter), [GMR (LimX fork)](https://github.com/limxdynamics/GMR)
