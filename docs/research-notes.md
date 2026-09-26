# Factory Fall Detection: Technical Structure and Limitations

## Goal
Detect worker falls on existing factory CCTV, save the footage around the fall to a separate incident database, and alert staff in time to give treatment.

V2 (later): detect hands caught inside machines.

## Technical Structure

### 1. Video input
- Factory's existing CCTV or IP cameras.
- Live streams are pulled from the cameras or the NVR (usually RTSP).

### 2. Rolling buffer
- Each camera stream keeps the last 30 seconds in memory at all times.
- Nothing is saved unless a fall is detected.

### 3. Fall detection model
- An existing open-source model, fine-tuned if needed.
- Training and fine-tuning data: a Kaggle dataset of about 10k fall videos.
- Runs on every stream in real time and outputs a fall or no-fall decision.

### 4. Clip extraction
- When a fall is detected, the system keeps recording for 30 more seconds.
- The 30 seconds before and 30 seconds after are joined into one 60-second clip.

### 5. Incident database
- Clips are stored in a separate database, not the general CCTV archive.
- Each record holds the clip, camera, time, and detection confidence.
- General CCTV footage is rarely watched, so incidents need their own place to be found fast.

### 6. Alert
- Staff or the CCTV operator get notified right away with the clip.
- The goal is to reach the worker within the golden time.

### 7. Deployment (open decision)
- **Our server (cloud):** easier for us to run and update, but we see all customer video.
- **Customer site (on-prem or edge box):** video stays in the factory, but costs hardware and is harder to maintain.

## Flow
```
Factory CCTV -> Stream ingest -> Rolling 30s buffer -> Fall model
                                                        |
                                                   fall detected
                                                        |
                           Save 30s before + 30s after -> Incident DB -> Alert staff
```

## Limitations

### Ones we raised
1. **Manual review.** Someone still has to watch every 60-second clip to confirm the fall, which takes time from the CCTV operator.
2. **Privacy.** If the model runs on our server, we can access all factory video. This could stop customers from buying.

### Market
3. **Already exists.** Intenseye, Protex AI, Voxel, viAct and others sell fall detection on existing cameras, along with 50+ other hazards.
4. **Built into cameras.** Hikvision and Hanwha Vision ship fall detection in the camera firmware.
5. **Clip saving is not new.** Saving footage before and after an event is a standard NVR setting.
6. **Korea is crowded too.** LG U+, Intellivix, Superb AI and others sell AI CCTV for Serious Accident Punishment Act compliance.

### Technical
7. **Dataset gap.** Kaggle fall videos are mostly staged, indoors, at eye level. Factory cameras are high, far, blocked by machines, with bad lighting and workers in PPE. Accuracy will likely drop.
8. **False alarms.** Bending, kneeling, crawling under machines, and resting can look like falls. Too many false alerts and staff will ignore the system.
9. **Compute cost.** Running a model on every camera stream in real time needs GPUs, either on our side or at the factory.
10. **Camera coverage.** Blind spots and occlusion mean some falls will never be seen.

### Product
11. **Detection is after the fact.** Deadly falls (from height) are rarely saved by a faster alert. Buyers mostly pay for prevention and compliance.
12. **Golden time value is narrow.** On a busy floor a coworker sees the fall first. The alert matters most for lone workers, night shifts, and blind spots.
13. **Liability.** If a fall is missed and someone dies, we could be blamed for a product sold as "detects falls."
14. **Labor and legal.** In Korea, recording workers needs PIPA handling and may need consultation with worker representatives.

### V2 (hands in machines)
15. Detection alone is not enough. It only helps if it can stop the machine, which means connecting to machine safety controls and meeting safety certification.

## Open Questions
- What does a factory get from us that it doesn't get by turning on fall detection in a Hanwha or Hikvision camera it already owns?
- Cloud or edge?
- Which buyer first: factories directly, or camera and CCTV installers?

## Next Idea: Drowning Detection

### Why
- Drowning is the leading cause of death for children ages 1 to 4 in the US (CDC). Not infants under 1.
- Most of these happen in home pools during a short lapse in supervision, often when the child was not supposed to be swimming.

### Competitors
**Public pools (lifeguard assist)**
- **Poseidon:** cameras on the wall and above the pool. Alarms if someone is still at the bottom for about 10 seconds.
- **AngelEye:** underwater cameras built into pool lights. Alarm within 10 seconds. Public pools, water parks, hotels.
- **SwimEye:** underwater cameras plus pose estimation (OpenPose).
- **Lynxight:** AI on standard pool cameras, used in 16 countries.

**Backyard pools (home)**
- **Coral MYLO / Manta:** above-water and underwater camera, phone alerts. Meets ASTM F3698-24, a 2024 standard written specifically for computer-vision drowning detection in residential pools.
- **CamerEye, SwamCam, Pool Angel:** overhead AI cameras with phone alerts.

### Limitations
1. **Already a product category.** There is an ASTM standard for it, so the market is formed and certification is expected.
2. **No real drowning data.** Public research uses simulated drowning. Nobody can film real drownings, and toddler drownings are almost never on camera.
3. **Drowning is not always the same.** Adults often show the Instinctive Drowning Response (arms out, pressing down on the water, no shouting). Toddlers often slip under silently with no struggle, which looks very different.
4. **Water breaks vision.** Glare, reflections, splashing, refraction, and night. This is why most serious systems use an underwater camera, which means installing hardware in the pool.
5. **False alarms.** Kids holding their breath, playing dead, diving, and floating all look like drowning.
6. **Seconds matter.** Brain damage can start within minutes, so the alert must be near instant and someone must be close enough to act.
7. **Prevention may beat detection.** Fences, gate alarms, and "child entered pool area" alerts stop the child before they are in the water, and those already exist.
8. **Liability is higher than factories.** A missed drowning of a child is the worst possible failure for a consumer product.

### Shared Pattern With Fall Detection
Both ideas already exist as products. The model is not the hard part. Data, hardware placement, false alarms, and trust are.

## Team Plan: Drowning Detection (5 people)

### Approach
1. **Sim data:** generate drowning and normal swimming motion in simulation.
2. **Real data:** record ourselves acting out drowning and normal swimming.
3. **Model:** train on sim, fine-tune on our real videos, test only on real videos.

Use a **pose-based model** (video -> body keypoints -> temporal classifier) instead of raw pixels. Sim water looks fake, but skeleton motion transfers much better than pixels, so the sim-to-real gap is smaller.

### Roles
| Person | Role | Owns | Output |
|---|---|---|---|
| 1 | Simulation | MuJoCo humanoid in water (buoyancy, drag), scripted behaviors: instinctive drowning response, climbing ladder, silent sink, normal swim, float, dive, play. Isaac Sim visuals only if time allows | Labeled keypoint sequences, optional synthetic videos |
| 2 | Real data and data management | Recording sessions, camera rig, safety, labeling tool, storage, dataset versions, train and test splits, one data format for sim and real | Versioned, labeled dataset everyone pulls from |
| 3 | Model and evaluation | Pose extraction, temporal classifier, sim pretraining, real fine-tuning, metrics, false alarm tests | Trained model, ablation results, eval report |
| 4 | Backend | Camera stream ingest, rolling buffer, model inference, clip saving, incident database, alert API | Running service that turns a video stream into incidents |
| 5 | Frontend | One dashboard: live alerts, clip playback, "confirm" or "false alarm" button that sends the label back to the dataset | Working demo UI |

The confirm button closes the loop: every alert a person reviews becomes new labeled real data for P2 and P3.

### Kickoff (all 5 together, before splitting)
1. **Pick one goal and one number.** Example: "Detect drowning within X seconds with under Y false alarms per hour on our real test set."
2. **Study drowning together.** Read the instinctive drowning response research and watch public simulated drowning clips. Write a one-page label guide: classes, and exactly when "drowning" starts.
3. **Test the biggest risk first.** Film a few minutes of yourselves in a pool (phone overhead, cheap underwater camera) and run an off-the-shelf pose estimator on it. If it can't find bodies in water, the pose-based plan changes, so find out now.
4. **Agree on formats.** Keypoint and label format shared by sim and real, clip naming, backend to frontend API.
5. **Set up shared tools.** One GitHub repo, shared storage for video, a labeling tool, a weekly sync.
6. **Split into roles.**

### Phases
1. **Setup:** agree on labels (what counts as drowning), camera angles, data format, API between backend and frontend. Everyone.
2. **Build:** P1 and P2 produce data. P3 builds the model on a public dataset first. P4 builds the backend with a dummy model. P5 builds the dashboard against P4's API.
3. **Train:** P3 trains sim-only, real-only, and sim + real, and evaluates each on the real test set. P4 swaps in the real model.
4. **Demo:** live pool camera through the full system. Everyone writes up their part.

### Key experiment
Does sim data reduce how much real data we need? Train with 10%, 25%, 50%, 100% of real data, with and without sim pretraining. This curve is the result worth showing.

### Concerns
- **Safety when recording.** Shallow water only, a trained person watching who is not in the water, no breath holding. Shallow water blackout can happen to strong swimmers.
- **Our acting is not real drowning.** Adults acting look different from a toddler slipping under. The test set is still simulated, so say that clearly.
- **MuJoCo does not render water.** It approximates buoyancy and drag. It gives motion, not visuals.
- **Realistic water in Isaac Sim is hard.** Surface, refraction, and underwater look take time. Keep it optional for P1 and cut it if pose-based works.
- **Label disagreement.** Define exactly when "drowning" starts in a clip before anyone labels.
- **Frontend scope.** One page is enough. The model result is what matters for jobs, so do not let the UI eat the timeline.
