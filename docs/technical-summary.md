# Technical Summary

updated: 2026-09-26

## What we are building
A **parent assistant** for home pools. A camera watches the pool, tracks everyone in it, and beeps when someone has been underwater too long.

- Home use first, not a lifeguard product.
- Later: public pools, and the original idea of fall detection in factories.

## Core logic (v1)
1. **Detect people** in the pool with a person detector. Each detection has a confidence score.
2. **Track them** with a MOT (multi-object tracking) tracker, so each person keeps an ID across frames.
3. **Underwater timer.** When a tracked person disappears from the detector (went under the water), start a timer for that ID.
4. **Alert.** If they are still gone after the threshold (about 5 seconds, to tune), sound the alarm through the camera and the app.
5. **Clear.** If the person is detected again, stop the timer.

We do **not** separate adults and babies in v1. The same rule applies to everyone. This also lets us generate training data with our own adult friends.

## Decided
- Home parent assistant first.
- Person detector + MOT tracker + underwater timer.
- No adult vs baby split in v1.
- We record our own footage (friends underwater for 10+ seconds, then coming up) to fine-tune.
- Training data should show **upper body only**, since that is what a pool camera sees. Full-body datasets include legs and will not match.
- Simulation with MuJoCo and Isaac Sim, for extra data and for a strong technical demo.

## Still open (decide together)
- **Which detector and which MOT tracker.** Everyone researches options, then we compare and pick.
- **Joint and motion analysis.** Whether to add pose (head, arms, hands, body) on top of the detector, for example to catch struggling before someone goes under.
- **Sim to real.** In Isaac Sim, use human avatars instead of robots so the detector sees people. MuJoCo gives the motion; the rendered body needs to look human.
- **Alert threshold** (5 seconds or other), and sound on camera vs phone vs both.
- **Cloud vs on-device** processing (privacy).

## Known problems
### 1. Person comes up somewhere else
An adult swims underwater for 10 seconds and comes up at another spot. The tracker gives them a new ID, and the old ID keeps counting as "missing", which causes a false alarm.

Ideas to try:
- **Head count check.** Only alarm if the number of people in the pool is still lower than before. If someone new appears, the missing person probably came back.
- **Re-ID.** Use appearance features (swimsuit color, hair) to match the new track to the old ID. More work.
- **Distance gate.** A new track that appears within a few meters within a few seconds is probably the same person.

### 2. Disappearing for other reasons
The detector can lose someone who is not underwater: glare, splashing, blocked by another person, left the camera view, got out of the pool.

Ideas: define the pool area on the camera image and only start timers for people lost inside it. Ignore people who leave through the edge.

### 3. Normal underwater play
Kids and adults hold their breath, dive, and play underwater on purpose. A fixed 5-second rule will fire a lot. We need to measure false alarms per hour on our own footage and tune the threshold.

### 4. The real toddler risk
Most toddler drownings happen when no adult is watching, often when the child was not supposed to be in the pool at all. "Parent is next to the pool" is not the main risk case. Worth also alerting when a small person enters the pool area with no adult present.

### 5. Data
- No real drowning videos exist to train on. Our footage is acted.
- Water breaks cameras: glare, reflection, splash, night.
- Recording safety: shallow water, someone trained watching from outside the water, no pushing breath holds.

## Team roles
| Person | Role |
|---|---|
| 1 | Simulation (MuJoCo, Isaac Sim) |
| 2 | Real data and data management |
| 3 | Model and evaluation |
| 4 | Backend |
| 5 | Frontend |

## Next steps (all together)
1. Each person researches detector and MOT tracker options. Compare and pick one.
2. Write the label guide: what counts as "underwater" and "alert".
3. Film a short test in a pool and run the chosen detector and tracker on it. Check: does it find upper bodies in water, and how often does it lose people?
4. Agree on data format and the backend to frontend API.
5. Split into roles.

Details on competitors and limitations: [research-notes.md](research-notes.md)
