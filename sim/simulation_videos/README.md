# Simulation videos

The five MuJoCo test scenarios we used to score detectors, the tracker, and the alarm logic against exact answers. Raw camera views, no boxes drawn. 960x540, 30 fps, corner CCTV camera.
Ground truth for each video (boxes, head above or below water) is in `data/sim_scenarios/<name>.tracks_gt.json`.

| Video | Length | What happens |
|---|---|---|
| `baseline.mp4` | 6 s | stand, swim, float, drowning response, and one person who collapses under the water |
| `resurface.mp4` | 10 s | a diver goes under about 5 s and comes up about 2 m away |
| `crossing.mp4` | 8 s | two swimmers pass each other, two standing people overlap |
| `silent_sink_busy.mp4` | 10 s | silent sink in the deep end in a busy pool, plus someone ducking under about 2.6 s |
| `entry.mp4` | 6 s | a person falls in from the deck |

Regenerate: `.venv/bin/python sim/mujoco/scenarios.py`. Details: [docs/global/simulation.md](../../docs/global/simulation.md).
