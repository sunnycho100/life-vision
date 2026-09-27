# Frontend and Camera Integration Plan

## Product vision

**We Fall, We Die** is a software-only drowning-risk detection system designed to work with cameras that families may already own. The long-term goal is to connect to common camera ecosystems such as Ring, Google Nest, generic IP cameras, and dedicated backyard-pool cameras without requiring proprietary hardware.

Camera support will depend on the live-stream and developer access permitted by each manufacturer. Ring and Google Nest are therefore planned integrations, not guaranteed prototype capabilities.

## Hackathon prototype

The first version will process uploaded or prerecorded pool footage. A user selects a video and presses **Activate Monitoring**. The application then:

1. Plays an intentionally cheesy, Matrix-inspired activation animation with falling green characters.
2. Processes the video using person detection and tracking.
3. Draws a green bounding box around each detected person.
4. Maintains a short history of each person's position, visibility, and motion.
5. Changes the box from green to yellow to red as estimated risk increases.
6. Displays timers and the reasons behind warnings.
7. Produces an on-screen and audible alert when the selected notification threshold is reached.

## Risk levels

### Green: normal

The person is visible and their recent behavior does not match a drowning-risk pattern.

The overlay may display:

- `SAFE`
- Tracking ID
- Time in pool
- Detection confidence

### Yellow: warning

The system detects behavior that may require attention but is not yet classified as critical.

Possible triggers include:

- Unusual repetitive upper-body motion.
- Significant motion with little forward progress.
- Temporary loss of the person or their head.
- A person remaining submerged longer than the warning threshold.

The interface displays the active timer, risk score, and triggering conditions.

### Red: critical alert

The concerning behavior continues beyond the critical threshold, or several risk signals occur together.

Possible triggers include:

- A tracked person disappears within the pool and does not reappear.
- Active-distress motion continues for the configured duration.
- A warning timer reaches the critical threshold.
- Several signals combine into a high danger score.

## Notification preferences

Users will be able to choose when they are notified:

- Yellow and red alerts.
- Red alerts only.
- On-screen alerts without external notifications.

Warning and critical timers will be configurable. Demonstration thresholds may be shortened for presentation purposes, but the interface must label them as demo settings rather than validated safety thresholds.

## Detection modes

The prototype will use one shared person-detection and tracking pipeline with two temporal detection modes.

### Passive submersion

This mode looks for a person who:

- Was previously visible inside the pool.
- Becomes partially or completely undetectable.
- Did not appear to leave through the pool boundary.
- Remains missing beyond a configurable duration.

This represents passive or silent submersion. The prototype should not describe it as a baby-specific classifier because it will not be trained or validated on infant footage.

### Active distress

This mode looks for a combination of:

- Repetitive arm or upper-body motion.
- High local movement.
- Little meaningful forward progress.
- Continued activity in approximately the same location.
- Persistence beyond a configurable duration.

The application analyzes behavior across a sequence of frames rather than making a drowning decision from a single image.

## Interface elements

The demonstration interface should include:

- Video upload and playback.
- Activate and deactivate monitoring controls.
- Matrix-inspired startup animation.
- Manually defined pool-area overlay.
- Green, yellow, and red person boxes.
- Persistent person IDs where tracking permits.
- Warning and critical timers.
- Risk score and triggering reasons.
- Alert history.
- Notification preference controls.
- Annotated incident replay or export.

Example status messages:

- `Person detected`
- `Tracking person 1`
- `Low forward movement`
- `Repeated upper-body motion`
- `Head not visible: 1.8 s`
- `Person missing inside pool: 2.7 s`
- `Warning threshold reached`
- `Critical alert triggered`

## Camera integration roadmap

Development will proceed in stages:

1. Uploaded video files.
2. Local webcam or USB camera.
3. Generic IP-camera streams such as RTSP, where available.
4. Authorized, camera-specific connectors for Ring, Google Nest, and other supported systems.
5. Mobile and external notifications.
6. Continuous incident recording and review.

The 24-hour prototype will prioritize a stable prerecorded-video demonstration. Live camera input is optional if the core pipeline is completed early.

## Hackathon deliverable

The prototype should demonstrate:

- Prerecorded pool-video processing.
- Person detection and tracking.
- Green-to-yellow-to-red state transitions.
- Passive-submersion and active-distress rules.
- Configurable warning and critical timers.
- User-selectable notification thresholds.
- An audible or visual critical alert.
- Exported annotated demonstration footage.
- A deliberately playful activation animation.

## Safety limitation

This prototype is a research and presentation system, not a certified lifesaving device. It must not replace physical barriers, responsible adult supervision, lifeguards, or established water-safety practices. Its purpose is to demonstrate how existing cameras could provide an additional layer of awareness through transparent, explainable risk signals.
