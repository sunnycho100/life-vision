# Life Vision presentation script

Target length: about 5 minutes 30 seconds, plus the live demo and questions.

## Slide 1 — Life Vision

**Presenter: David · 20 seconds**

“Imagine you are at a backyard pool with friends or family. For one moment, everyone looks away. The camera keeps recording, but recording alone cannot tell you when normal play has turned into danger. Life Vision adds that missing layer. It watches each person over time and warns the adult when someone’s head has stayed below the water too long.”

Transition: “The problem usually begins with a very ordinary gap in attention.”

## Slide 2 — The backyard pool use case

**Presenter: Rohan · 30 seconds**

“The Consumer Product Safety Commission reviewed reported pool and spa fatalities involving children under five. Sixty-one percent involved a gap in adult supervision, meaning an adult lost contact with the child long enough for the child to reach the water. Many of these homes already have a backyard camera. Our first use case is simple: turn that existing view into a second set of eyes, while keeping supervision and physical pool barriers as the primary protections.”

Transition: “That goal shaped the interaction we built.”

## Slide 3 — User interaction and alert states

**Presenter: David · 35 seconds**

“The user selects a prerecorded clip for the hackathon, then presses Activate Monitoring. We intentionally made activation theatrical, with a quick Matrix-style scan, so the audience can see exactly when analysis starts. Every person receives a stable ID and a green box. If a head remains below the surface, a timer starts. The box turns yellow at the warning threshold, then red if the condition persists. The interface always explains the decision, for example ‘head not above water for 3.1 seconds.’ The user chooses yellow and red alerts, red only, or on-screen alerts.”

Transition: “Behind that simple interaction is a pipeline with one job at each stage.”

## Slide 4 — Technical architecture

**Presenter: Sunny · 35 seconds**

“The source reads a video with media timestamps. The detector finds each person and the available evidence for the head. ByteTrack links detections across frames. Our own identity layer handles short losses and resurfacing, because tracker IDs alone do not survive every dive. The event engine owns the timers, missing-person registry, and entry alert. It writes results as timestamped JSON, and the frontend draws those results over the video. For the demo this all runs locally from files, so network or cloud failures cannot break the presentation.”

Transition: “The hardest part was getting honest test data without putting anyone at risk.”

## Slide 5 — Simulation test bench

**Presenter: Sunny · 35 seconds**

“We built a MuJoCo test bench with a shallow end, a deep end, buoyancy, and scripted behaviors such as swimming, resurfacing, collapse, entry, and a silent sink. The simulator gives us exact boxes and exact head height in every frame, so we can test the tracker and event logic without hand labeling. We also built an Isaac Sim scene with more realistic people, lighting, and water. Simulation lets us test dangerous cases safely, but it does not replace real pool footage.”

Transition: “Those exact labels let us compare model choices on the same held-out synthetic clip.”

## Slide 6 — Detector results on synthetic footage

**Presenter: Joanne · 40 seconds**

“This chart shows recall specifically when the head is fully below water on our held-out Isaac Sim clip. Stock YOLO11n found only sixteen percent. Stock RF-DETR-S found seventy-five percent. A YOLO model trained on the older MuJoCo characters transferred better at eighty-three percent. Our YOLO model fine-tuned on 261 Isaac frames reached ninety-five percent. Precision remained ninety-five percent. These are synthetic results from one scene and character set. They prove that our training and evaluation pipeline works. They do not predict real-world accuracy.”

Transition: “Even a strong detector did not solve the timer by itself.”

## Slide 7 — Tracking and identity

**Presenter: Sunny · 35 seconds**

“The default tracker produced eighteen IDs for four people. That makes a per-person timer useless. We added three layers: box cleanup removes duplicate and partial detections, an ID starts only after several consistent frames, and identity stitching gives a resurfacing person their previous ID when the timing and location agree. On the five MuJoCo scenarios, the fine-tuned detector plus these rules produced one ID per person with zero switches. We also learned that a submerged body remains visible in clear water, so the alarm must time the head rather than wait for the whole box to disappear.”

Transition: “The camera source can change without changing that decision logic.”

## Slide 8 — Existing-camera integration

**Presenter: Rohan · 35 seconds**

“Every camera enters through the same source adapter. The hackathon uses uploaded video first, then a webcam. A compatible IP camera can provide an RTSP stream. Ring now offers an official developer platform with authorized device access, webhooks, and WebRTC video. Google Nest provides WebRTC or RTSP through Device Access, depending on the model. The colored overlay appears in our application, not inside the native Ring or Google Home screen. Account authorization and commercial review come later.”

Transition: “Now we can show the experience that sits on top of the pipeline.”

## Slide 9 — Live demo

**Presenter: David · about 60 seconds plus interaction**

“This is our prerecorded pool clip. We use staged behavior in shallow water with a dedicated observer. When I activate monitoring, the scan starts and the system assigns each person an ID. Watch Person 02. The box starts green. As the head-submersion evidence persists, the timer appears and the box changes to yellow. At the demo alarm threshold, it turns red and the alert fires. The timeline records what changed and why.”

Then say one of the following, accurately:

- **Real pipeline:** “These overlays come from our detector, tracker, and event results.”
- **Recorded pipeline:** “This is a recording of our successful local pipeline run.”
- **Scripted interface:** “The interaction is scripted today so we can demonstrate the user experience while the real event engine is connected.”

Never describe scripted boxes as live model inference.

Transition: “The prototype works as a system, but the remaining limits matter.”

## Slide 10 — Limits and next test

**Presenter: Sam · 40 seconds**

“Every accuracy number in this presentation comes from synthetic footage. The simulator lacks real glare, splashing, occlusion, camera compression, and the diversity of real people. Our next experiment is real pool footage split by recording session. We will label whether each head is above or below the surface, fine-tune on the training sessions, and evaluate only on separate sessions. We will report detection delay, missed events, false warnings per monitored hour, and identity errors. Life Vision remains a supervision aid. It does not replace watching children, pool fences, or lifeguards.”

Closing: “We are not asking families to buy another camera. We are testing whether the camera they already have can recognize when the water stops looking normal.”

## Live demo checklist

1. Open `frontend/index.html` through the local server before presenting.
2. Load the known-good team video before the talk begins.
3. Enable browser audio with one click.
4. Reset the demo immediately before slide 9.
5. Keep the scripted interface available as the first fallback.
6. Keep a screen recording of the full successful flow as the second fallback.
7. Keep the annotated simulation clip as the third fallback.

## Likely judge questions

### Is this detecting drowning?

“No. The prototype detects observable risk signals, especially prolonged head submersion and loss of a tracked person inside the pool. Drowning is a medical outcome that this system cannot diagnose.”

### Why not alert when the person box disappears?

“In clear water, the detector can continue seeing a submerged body. Our simulation showed that the person box often remains while the head is below the surface, so head state is the more useful timer.”

### Why use simulation?

“It gives us exact labels and lets us test unsafe cases such as collapse and silent sinking. It validates the software pipeline, but it does not establish real-world accuracy.”

### Why not train separate baby and adult models?

“We do not have consented child footage or enough evidence for age-specific behavior models. The first version applies the same observable head-submersion rule to every person.”

### Will this work with Ring or Nest?

“Both now have official developer paths for authorized video access. We would ingest the stream into our application and draw the overlay there. The hackathon uses local video because certification, account linking, and device testing are separate product work.”

### What would you measure next?

“Detection delay, missed events, false warnings per monitored hour, and ID continuity on real footage separated by recording session.”
