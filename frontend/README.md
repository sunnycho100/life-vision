# Frontend demo

This dependency-free prototype demonstrates the complete user experience while the real detection backend is being built.

## Run it

Open `index.html` directly, or serve the repository locally:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000/frontend/`.

## Presentation flow

1. Optionally choose the team's pool video.
2. Press **Activate monitoring**.
3. The Matrix-inspired startup animation runs.
4. Two people appear with green tracking boxes.
5. Person 02 changes to yellow after about 4.5 seconds.
6. Person 02 changes to red after about 9 seconds and triggers an alert.
7. The sequence loops every 16 seconds.

The results are intentionally scripted. The page is a user-interface prototype, not a real drowning classifier. It is designed so the scripted `stageFor()` output in `app.js` can later be replaced with backend WebSocket messages using the contract in `docs/dpark/data-model-camera-guide.md`.

## Owner

Person 5 (Frontend): video/canvas overlay, alerts, clip playback, and feedback controls.
