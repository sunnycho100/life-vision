# Life Vision team presentation script

Target length: about 5 minutes 30 seconds, plus the live demo and questions.

## Slide 1 — Life Vision

**David · 15 seconds**

“A camera can record every second around a pool and still miss the moment that matters. Life Vision adds a local monitoring layer that tracks each person and explains when head-submersion evidence persists. Our goal is a second set of eyes for the responsible adult, using cameras families may already own.”

Transition: “The need becomes clear when we look at how quickly human attention can fail.”

## Slide 2 — The risk is fast, quiet, and easy to miss

**Rohan · 30 seconds**

“CDC reports that more children ages one to four die from drowning than from any other cause. CPSC found that sixty-one percent of reported pool and spa fatalities involving children under five were associated with a gap in adult supervision. In a published case series of 140 deaths at lifeguarded pools, other swimmers or bystanders noticed the victim first twice as often as lifeguards did. The authors stress that these deaths are uncommon and that lifeguards remain an important protection. Our point is that no observer can see every person every second.”

Transition: “That is why our design supports attention instead of pretending to replace it.”

## Slide 3 — The system times one observable condition

**Rohan · 30 seconds**

“The system tracks each person and watches whether the head is above or below the water. When the head goes below, a per-person timer starts. At the configurable warning threshold, the state becomes yellow. If that condition persists to the alarm threshold, the state becomes red and tells the user to check the pool. If the head comes back above water, the timer resets to green. Entry detection is a separate alert that the user must arm. For this demonstration, the thresholds are intentionally shortened and clearly labeled. This is prolonged head-submersion detection, not a drowning diagnosis.”

Transition: “That simple rule sits inside a pipeline the five workstreams built in parallel.”

## Slide 4 — Five workstreams met at one shared data contract

**Sunny · 35 seconds**

“We froze two JSON contracts before building. David built the overlay against hand-made results instead of waiting for the backend. Rohan built the event engine against simulator tracks. Joanne delivered person and head evidence into the same format. I connected the detector, tracker, and identity stitching. Sam evaluated each stage and tested whether pose could provide reliable head evidence. By hour six, simulator tracks could drive a red alert in the frontend. The last hours were reserved for integration, fixes, a backup recording, and rehearsal.”

Transition: “The simulator was what gave all five workstreams the same exact answers.”

## Slide 5 — Simulation covered cases the team could not film safely

**Rohan · 30 seconds**

“Sunny built a MuJoCo pool with buoyancy, a shallow end, a deep end, and five scripted scenarios. The simulator produced exact person boxes and exact head height without hand labeling. I extended the work in Isaac Sim with more realistic characters, moving water, and a held-out 300-frame clip. Sam could then score silent sinking, collapse, entry, and resurfacing without asking anyone to perform unsafe behavior. Simulation validates our pipeline. It does not establish real-pool accuracy.”

Transition: “Those labels let Joanne compare models on the same cases.”

## Slide 6 — Two experiments changed the detector plan

**Joanne · 40 seconds**

“Our first experiment used Sunny’s held-out MuJoCo scenario. Stock YOLOv8 found twelve percent of the people, YOLO11 found twenty-five percent, and RF-DETR Nano found seventy-nine percent. The second experiment used Rohan’s held-out Isaac clip and measured people whose heads were fully below water. Stock YOLO found sixteen percent. Stock RF-DETR-S found seventy-five percent. A YOLO model trained on the MuJoCo characters found eighty-three percent, and the Isaac-tuned YOLO reached ninety-five percent. So we kept lightweight YOLO for the hackathon after fine-tuning, while RF-DETR remained our strongest zero-shot candidate. These are synthetic results, not real-world claims.”

Transition: “A good detector still did not give us a reliable timer.”

## Slide 7 — Stable identities made the head timer possible

**Sunny · 30 seconds**

“Stock RF-DETR with the default tracker produced eighteen IDs for four people. A timer is useless if the same person keeps receiving a new identity. I added duplicate-box cleanup, delayed track confirmation, and stitching after short losses. That produced four IDs with zero switches across all five MuJoCo scenarios. The test bench also exposed a more important mistake: in clear water, a submerged body can remain visible. A missing-person timer may never start. Rohan’s engine therefore times head state and keeps missing-person logic only as backup.”

Transition: “David turned those internal states into something a user can understand.”

## Slide 8 — The demo explains the state change

**David · about 50 seconds plus interaction**

“This is our prerecorded team pool clip. I will activate monitoring, which starts the deliberately visible scan. Each person receives a stable ID and a green box. When head-submersion evidence persists, the timer appears and the box becomes yellow. At the demo threshold it becomes red, the alert fires, and the incident can be replayed. The user can choose yellow and red alerts, red only, or on-screen alerts.”

Then state exactly one of these:

- **Real pipeline:** “These overlays come from our detector, tracker, and event results.”
- **Recorded pipeline:** “This is a recording of a successful local pipeline run.”
- **Scripted interface:** “The interface is scripted today so we can demonstrate the experience while the final event connection is completed.”

Never call scripted boxes live model inference.

Transition: “The hardest part of the project was learning when our own evidence was wrong.”

## Slide 9 — Three mistakes changed the evaluation rules

**Sam · 30 seconds**

“A render batch omitted the water, which made early YOLO scores look far too good. A test script swapped color channels, and we discovered it only because an independent script disagreed. Our first head-under label was too strict and counted ordinary swimmers as submerged. We fixed each problem, rebuilt the labels from stored head height, and reran every number. This gives us more confidence in the process, but the accuracy is still synthetic-only.”

Transition: “The next honest test is real footage from a different recording session.”

## Slide 10 — Real footage is the next test

**Rohan · 40 seconds**

“Next, we record safe staged pool clips and label whether each head is above or below water. Joanne fine-tunes the detector. Sam separates training and testing by recording session and reports detection delay, missed events, false warnings per monitored hour, and identity errors. Sunny connects the pipeline, I validate the timer, and David replaces scripted events with real results. Uploaded video remains our reliable source. Webcam and RTSP follow, then authorized Ring and Nest adapters. The overlay stays in our application. Life Vision is a supervision aid. It does not replace watching children, pool barriers, or lifeguards.”

Closing: “Human attention will always matter around water. Our question is whether the camera already watching the pool can help attention fail less often.”

## Live demo checklist

1. Load the known-good video before the presentation.
2. Confirm browser audio with one click.
3. Reset the demo immediately before slide 8.
4. State whether the run is real, recorded, or scripted.
5. Keep the successful screen recording ready as the first fallback.
6. Keep the annotated simulation clip ready as the second fallback.

## Judge questions

### Are you detecting drowning?

“No. The system detects observable risk signals, especially prolonged head submersion and loss of a tracked person inside the pool. Drowning is a medical outcome that this system cannot diagnose.”

### Are you claiming lifeguards do not work?

“No. Lifeguards are an essential layer of protection. The study on our slide examined a case series of fatal incidents that occurred despite lifeguard presence. It does not estimate a general lifeguard failure rate. It shows why additional layers can still matter.”

### Why use one rule for babies and adults?

“Age-specific behavior models require data we do not have, especially consented child footage. Head-submersion duration is an observable signal that can be applied consistently while we collect better evidence.”

### Why did RF-DETR beat YOLO before fine-tuning?

“RF-DETR’s DINOv2 backbone transferred better to unfamiliar synthetic people. That explanation is plausible but not directly measured. Fine-tuning changed the result, so we compare both models on the same held-out clips.”

### Why is simulation useful if it does not prove real accuracy?

“It gives exact labels for the tracker and event engine, and it lets us test unsafe cases. It validates the software and evaluation pipeline before we collect real footage.”

### What would you measure on real footage?

“Detection delay, missed events, false warnings per monitored hour, and identity continuity. We would split data by recording session so neighboring frames cannot leak into both training and testing.”

## Source notes for presenters

- CDC states that drowning is the leading cause of death for U.S. children ages 1 to 4. Do not say “all children under five.”
- The 61% figure covers reported pool and spa fatalities involving children under age five in the CPSC 2024 report, using 2019–2021 cases classified for circumstance.
- The lifeguard finding comes from a case series of 140 fatal incidents at lifeguarded U.S. pools from 2000–2008. Say “twice as often in this case series,” not “lifeguards miss two-thirds of drownings.”
- Every detector and tracking percentage in the deck comes from synthetic footage.
