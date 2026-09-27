# Cursor Handoff

## Repository

- GitHub: <https://github.com/sunnycho100/life-vision>
- Working branch: `main`
- The repository was formerly named `we-fall-we-die`; GitHub redirects the old URL.

Before making changes:

```bash
git switch main
git pull --ff-only
```

## Current product direction

**We Fall, We Die** is a research prototype for explainable pool-risk monitoring using cameras people may already own.

The 24-hour version should use one shared pipeline:

```text
video -> person detection -> multi-object tracking -> rolling history
      -> passive/active temporal rules -> JSON results -> frontend overlay
```

Two behavior paths:

1. **Passive submersion:** a person was visible inside the pool, disappears without leaving through the pool boundary, and remains missing.
2. **Active distress:** repetitive upper-body/local motion occurs with little forward progress and persists over time.

Do not make infant-specific or medically validated claims. This is not a certified lifesaving system.

## What is already implemented

### Interactive frontend prototype

Files:

- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`
- `frontend/README.md`

The frontend is dependency-free and currently includes:

- Local video upload.
- Simulated pool feed when no video is selected.
- Matrix-inspired activation animation.
- Canvas-based person boxes.
- Scripted green -> yellow -> red decisions.
- Risk reasons and persistence timers.
- Notification preferences.
- Audio alert and incident timeline.
- Responsive desktop/mobile layout.

Run it from the repository root:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/frontend/>.

Important: `stageFor()` in `frontend/app.js` generates scripted model results. These must be described as simulated inference until replaced with a real backend connection.

### Presentation

Read `docs/presentation.md`. It contains:

- The full presentation narrative.
- Technical architecture.
- Passive and active decision logic.
- Data/model strategy.
- Ring, Nest, RTSP, webcam, and uploaded-file integration plan.
- User journey.
- Live-demo choreography.
- Demo failure plan.
- Five-person final sprint.

### Research and implementation guide

Read `docs/dpark/data-model-camera-guide.md`. It contains:

- Dataset and checkpoint links.
- Verified use cases and limitations.
- Recommended 24-hour workflow.
- Ring and Google Nest API findings.
- Backend/frontend contract.
- Frontend priorities.

Also read:

- `docs/technical-summary.md`
- `docs/research-notes.md`
- `docs/dpark/frontend-interface-plan.md`

## Expected backend-to-frontend message

The frontend should eventually receive one update per person over WebSocket:

```json
{
  "track_id": 2,
  "bbox": [0.61, 0.33, 0.18, 0.43],
  "state": "yellow",
  "risk_score": 0.64,
  "reasons": [
    "repetitive upper-body motion",
    "low forward progress"
  ],
  "warning_seconds": 2.4,
  "missing_seconds": 0.0
}
```

The current demo uses normalized box coordinates in the order `[x, y, width, height]`.

## Recommended next implementation task

Connect the existing frontend to a mock or real WebSocket without breaking the scripted fallback:

1. Add a configurable WebSocket URL.
2. Validate incoming messages before rendering them.
3. Convert backend box coordinates to displayed-video coordinates.
4. Keep `stageFor()` as a clearly labeled demo mode.
5. Show connection state and automatically fall back to demo mode when requested by the presenter.
6. Test the complete flow with one known-good local pool video.

After that:

1. Add `WebcamSource`.
2. Add manual pool-polygon drawing.
3. Integrate the chosen detector/tracker.
4. Implement passive-submersion timers.
5. Implement active-distress rolling features.
6. Record an annotated fallback video before the presentation.

## Camera integration boundary

The colored overlay appears in this application's video player. Do not promise that boxes can be injected into the native Ring or Google Home interface.

- Ring production path: developer registration, OAuth/account linking, device discovery, webhooks, and WebRTC/WHEP.
- Nest production path: Device Access, Google Cloud/OAuth, capability discovery, and WebRTC or RTSP.
- Hackathon path: uploaded video first, webcam second, one compatible RTSP camera third.

## Safety and presentation rules

- Use shallow water and a dedicated observer for filming.
- Do not hold breath or simulate unconsciousness underwater.
- Clearly label shortened timers as demo thresholds.
- Never present scripted frontend detections as live model inference.
- Keep the disclaimer visible: this prototype does not replace supervision, pool barriers, lifeguards, or established water-safety practices.

## Suggested first prompt in Cursor

> Read `CURSOR_HANDOFF.md`, `docs/presentation.md`, and `docs/dpark/data-model-camera-guide.md`. Inspect the existing dependency-free frontend. Continue the project by adding a WebSocket data-source adapter while preserving the scripted demo as an explicit fallback. Before editing, summarize the existing architecture and propose the smallest safe implementation plan. Do not remove the current visual design or make unvalidated safety claims.

