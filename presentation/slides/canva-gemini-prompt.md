# Prompt pack for Canva or Gemini

How to use:
1. Upload the images and videos listed in "Assets to upload" first.
2. Paste the **master prompt**.
3. Paste the **slide content** right after it (same message if the tool allows, otherwise the next one).
4. Use the **follow-up prompts** to fix anything that comes out wrong.

---

## Master prompt (paste first)

```
Create a 10-slide presentation for a student hackathon project. Audience: judges and engineers. Talk length: about 5 minutes.

Design direction:
- Inspired by SpaceX: near-black background (#000000 to #0B0C0E), white text, cool grays for secondary text, lots of empty space.
- Big, bold, uppercase titles with wide letter spacing, in a clean geometric sans (D-DIN, Barlow, or Inter).
- One visual per slide, large. Use my uploaded images and videos; do not add stock photos, clip art, cartoon icons, or emojis.
- Color only for meaning: green (#3DDC84) for safe, amber (#FFB000) for warning, red (#FF4D4D) for alert. Everything else stays black, white, and gray.
- Large number callouts where a slide has one key number.
- No decorative lines under titles, no colored side stripes, no gradients.

Content rules:
- Keep text short: a title plus at most 3 short bullets or one small table or one simple diagram per slide.
- Use my wording as written. Do not add claims, numbers, or marketing language.
- Every percentage is a simulation result. Keep the words "on simulation" where I wrote them.
- Put my speaker notes in the notes section, not on the slide.

The slide content follows.
```

---

## Slide content (paste after the master prompt)

```
SLIDE 1
Small label: POOL CAMERA ASSISTANT
Title: PARENTS LOSE TRACK FOR A FEW SECONDS
Big number: 62%
Under the number: of pool deaths of children under 5 happened when the adult lost track of where the child was.
Small source line: CPSC, 2020–2022
Visual: demo_hq_still.png, full height on the right half.
Notes: Most pool accidents with young kids don't happen because nobody was home. They happen in the few seconds when the adult looked away. A lot of families already have a camera on the backyard, but it only records, it doesn't tell you anything.

SLIDE 2
Small label: OUR IDEA
Title: TIME HOW LONG EACH HEAD IS UNDER
Diagram, left to right, three boxes with arrows:
  [HEAD ABOVE WATER] in green -> [5 S UNDER: WARNING] in amber -> [12 S UNDER: ALERT] in red
Under the diagram: Head comes back up, the timer resets. Also flags someone entering the pool.
Small line at the bottom: Pool alarm test standard uses 20 s. CPSC says 20 s is too late.
Visual: the diagram is the visual.
Notes: The camera tracks every person and times how long their head has been under the water. At 5 seconds the box turns yellow, at 12 seconds it turns red and the app alerts. The standard test for pool alarms is a dummy on the bottom for 20 seconds, and CPSC says that's too late, so we alert well before that.

SLIDE 3
Small label: HOW IT WORKS
Title: PIPELINE
Diagram, one row of 6 boxes with arrows:
  CAMERA VIDEO -> DETECTOR -> BOX CLEANUP -> TRACKER -> EVENT ENGINE -> APP
A dashed box below DETECTOR: SIMULATOR (exact labels for training and scoring)
Small line at the bottom: Runs locally on a laptop. The video never leaves the device.
Notes: The detector finds people in each frame, and the tracker keeps the same ID on each person over time, so each person gets their own timer. The event engine runs the timers and the entry alert, and the app draws the boxes. Everything runs locally, so the pool video never leaves the device.

SLIDE 4
Small label: NO DROWNING FOOTAGE
Title: WE BUILT A TEST BENCH
Bullets:
- Physics simulator (MuJoCo) with a shallow and deep end, and buoyancy
- Swim, float, dive, silent sink, collapse, fall in from the deck
- Exact labels for every frame, no hand labeling
Visual: sim_raw_baseline.mp4 (or sim_pool_scene.png) on the right. Optional small inset: isaac_sim_pool_editor.png labeled "Isaac Sim version".
Notes: We can't film real drowning, and we shouldn't ask anyone to fake it underwater. So we built a physics sim where we script people swimming, sinking, and collapsing, and the sim tells us exactly where every person and head is. We train on 4 scenarios and test on a 5th the models never see.

SLIDE 5
Small label: STOCK MODELS, ON SIMULATION
Title: YOLO BARELY SEES OUR PEOPLE
Bar chart, "people found" (white bars) and "fully under water" (gray bars):
  YOLO11n: 25%, 2%
  YOLOv8n: 12%, 31%
  RF-DETR Nano: 79%, 60%
Two small cards on the right:
  YOLO: CNN trained on everyday photos. Guesses boxes everywhere, then removes duplicates.
  RF-DETR: Transformer with a DINOv2 backbone. Each of 300 queries claims at most one object.
Notes: We tried two YOLO models and RF-DETR, all out of the box. YOLO found 25% of our people at most. RF-DETR found 79% with no training on our data. Our best explanation is its DINOv2 backbone, which learned general shapes from a much wider set of images.

SLIDE 6
Small label: RF-DETR NANO, ON SIMULATION
Title: FINE-TUNING ON SIM FRAMES
Big number: 79% -> 100%
Under the number: people found on a scenario it never saw, after 1 epoch (about 2 minutes on a laptop)
Small table:
  | Test video           | Stock | Fine-tuned |
  | Held-out scenario    | 79%   | 100%       |
  | New-looking render   | 76%   | 93%        |
Small line: A render with new textures cost accuracy. Real footage will be a bigger change.
Notes: One epoch of fine-tuning took RF-DETR to 100% on the scenario it never saw, including people fully under water. On a new render with textures and body shapes it never saw, it dropped to 93%. Even a small visual change costs accuracy, which is why real footage is our next step.

SLIDE 7
Small label: TRACKING
Title: A GOOD DETECTOR STILL BROKE THE IDS
Big number: 18 -> 4
Next to it: IDs on screen for 4 people. Zero ID switches after our fixes.
Visual: before_after_tracking.png across the slide, with labels "BEFORE" on the left half and "AFTER" on the right half.
Notes: With the default tracker we got 18 IDs for 4 people, and a timer is useless if a person keeps getting a new ID. We removed duplicate boxes, only started an ID after 3 frames, and gave people their old ID back if they reappeared nearby. That got us to 4 IDs for 4 people with zero switches.

SLIDE 8
Small label: DEMO (PLACEHOLDER)
Title: LIVE DEMO
Left, a dashed empty frame with the text: App recording goes here.
Three short lines beside it:
- 4 camera feeds, then Enable monitoring
- Green, yellow, red boxes with timer and reason
- Alert and incident replay
Visual: demo_hq_tracking_finetuned_rfdetr.mp4 on the right as the fallback, labeled "Fallback: tracking on simulation".
Notes: When we turn on monitoring, every person gets a box. When someone's head stays under, their box turns yellow with a timer, then red, and the parent can replay what happened.

SLIDE 9
Small label: HONEST NUMBERS
Title: LIMITS, AND MISTAKES WE CAUGHT
Three numbered lines:
01 All numbers are on simulation: simple bodies, flat water, no splash or glare.
02 We caught 2 bugs that made scores look too good, fixed them, and re-measured everything.
03 Daylight only. One person class, no child vs adult yet.
Visual: sim_pool_scene.png, smaller, on the right.
Notes: None of this is real-world accuracy yet. Our first scores were too good because the water wasn't drawn in some renders, and later we found the test images were color-swapped. We fixed both and re-measured every number on these slides.

SLIDE 10
Small label: NEXT
Title: NEXT STEPS
Three numbered cards in a row:
01 RECORD: Real pool footage, label each head above or below water
02 FINE-TUNE: Same method, tested only on real clips
03 CONNECT: Head timer drives the app's colors and alerts
Bottom line, bold: A supervision aid that detects prolonged head submersion. It does not replace watching children, pool fences, or lifeguards.
Small line: Sunny · Joanne · Sam · Rohan · David
Visual: demo_hq_10s.mp4 as a dark, muted background loop, or no image.
Notes: Next we record real footage, label whether each head is above or below the water, and fine-tune the same way, testing only on real clips. Then we connect the head timer so the boxes turn yellow and red on their own. This is a second set of eyes for the parent, not a replacement for watching. Thanks.
```

---

## Assets to upload (all in `presentation/`)

| File | Slide |
|---|---|
| `images/demo_hq_still.png` | 1 |
| `videos/sim_raw_baseline.mp4` or `images/sim_pool_scene.png` | 4 |
| `images/isaac_sim_pool_editor.png` | 4 (optional inset) |
| `images/before_after_tracking.png` | 7 |
| `videos/demo_hq_tracking_finetuned_rfdetr.mp4` | 8 |
| `images/sim_pool_scene.png` | 9 |
| `videos/demo_hq_10s.mp4` | 10 (optional) |

Canva and Gemini may not accept videos everywhere. If not, use the matching still: `images/demo_hq_tracking_still.png` for slide 8.

---

## Follow-up prompts (use if needed)

```
Make it more minimal: remove any icons or decorations you added, keep only my text and my images.
```
```
Slide 5: turn the numbers into a clean column chart, white bars for "people found" and gray bars for "fully under water", values on top of the bars, no chart border.
```
```
Make every title the same style: uppercase, bold, wide letter spacing, left aligned, same position on every slide.
```
```
The text is too long on slide [N]. Shorten it to at most 3 short lines without changing the numbers.
```
```
Use the same near-black background on every slide. Only use green, amber, and red on slide 2.
```
```
Do not change any numbers or add new claims. Restore my exact wording on slide [N].
```
