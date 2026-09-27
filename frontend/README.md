# frontend

The LifeVision review app. You give it a pool video, mark where the water is, and it plays the video back with a box and an ID on every person, turning yellow and red when someone stays out of sight. Plain HTML, CSS, and JavaScript, served by the [backend](../backend/). Start the server as shown there, then open http://127.0.0.1:5173.

## Using it

1. **Pick a source.** The wave-pool reference, any extra videos the server was started with (like the lifeguard rescue clip), or **Upload video** (the + box) for your own MP4 or WebM.
2. **Mark the pool area.** Drag the four corners around the water, clockwise from the far left. **Add corner** puts a new corner in the middle of the longest edge, so the outline can follow a curved or L-shaped pool. Only people inside the outline can raise a warning.
3. **Analyze.** The server runs the fine-tuned RF-DETR Small and the tracker. You can start reviewing while it runs. A video that was already analyzed opens straight into the review.
4. **Review.** Each person gets a number that sticks with them. Colors:

| Box | Meaning |
|---|---|
| Green | Seen |
| Grey, dashed | Just went out of sight (held at their last spot) |
| Yellow | Out of sight inside the pool for 5 s |
| Red | Out of sight for 12 s |
| Blue | Just came back after a long time out of sight |

Incidents are listed with a short clip of each. **Adjust pool mapping** lets you fix the outline without re-running detection.

The optional 3D view places people on an estimated pool plane. It needs exactly four corners, and positions are approximate.

## Files

| File | What |
|---|---|
| `index.html`, `app.js`, `style.css` | The review app |
| `core.mjs` | Geometry shared by the app: pool outline checks, point-in-outline, homography, camera fit |
| `twin.js` | The optional 3D view (Three.js) |
| `annotate.html` | Draw ground-truth person boxes on sampled frames for evaluation |
| `rescue-review.html`, `rescue-review-guide.html` | Label lifeguard rescue videos (who was rescued, when they were visible) |
| `setup_assets.py` | Downloads Three.js and other optional assets into `vendor/` |
| `browser_smoke.py`, `review_ui_smoke.py`, `surface_geometry_smoke.py` | Browser tests (need Chrome and a running server) |

## Notes

- Counts are person boxes, not a guaranteed headcount. Crowded frames can miss or double people.
- Warnings come from people going out of sight, not from a drowning classifier. A swimmer who ducks under on purpose will turn yellow too.
- Results are only as good as the pool outline and the camera staying still.
