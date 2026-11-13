# LifeVision deck: style and structure prompt

Use this as the prompt (or the brief) when building or regenerating the LifeVision slides. It describes the look we want, taken from a reference startup deck, and maps each of our 11 slides onto one of its layouts. Match the structure, not the reference's exact design.

## Look and feel

- **Canvas:** 16:9. Each slide is a white card with large rounded corners (about 24 px at 1920 wide) and a soft shadow, sitting on a light background with a gentle accent gradient at one edge.
- **Whitespace:** generous. Content takes about 60% of the card. Never more than 4 text lines of body copy on a slide.
- **Type:** one clean sans-serif (Inter, Helvetica Neue, or similar).
  - Kicker: small uppercase label above the title, letter-spaced, grey (for example `THE PROBLEM`, `OUR PRODUCT`, `HOW WE BUILT IT`).
  - Title: bold, 2 lines at most, very large. One key phrase in the title is set in the accent color (for example "Drowning is **silent**").
  - Body: short lines in grey, with 1 or 2 key words per line bolded in the accent color.
- **Color:** one accent color used sparingly, for the highlighted title phrase, key words, icons, rings, and shapes. Everything else is near-black, grey, and white.
  - LifeVision accent: pool teal `#0f766e` (light tint `#e6f4f1`).
  - Alert color, only where we talk about warnings: amber `#f59e0b` for yellow, `#ef4444` for red.
- **Bullets:** no plain dots. Use a small accent icon in front of each line: a cross for problems, a check for what we do, an arrow for steps.
- **Page number:** tiny grey number bottom right.
- **No** emojis, clip art, or stock icon packs. Logos only where we name the tool (NVIDIA, Google DeepMind).

## Layouts to reuse (from the reference)

| Name | Structure |
|---|---|
| **A. Split with image** | Left half: kicker, 2-line title with accent phrase, 3 icon bullets. Right half: a full-height photo or video frame bleeding to the card edge. |
| **B. Product mockup** | Left: kicker, title, 1 short sentence, 3 check bullets. Right: a laptop or phone mockup showing our app, placed on a large accent circle that bleeds off the card. |
| **C. Big numbers** | Kicker and title across the top. Two or three large stats underneath, each with a short grey caption. Stats can sit in a partial ring (progress arc in the accent color) or be joined by an arrow to show a change (before → after). |
| **D. Stacked stats** | A large soft accent circle on one side holding the headline number, a bracket, and 2 or 3 smaller numbers stacked on the other side. |
| **E. Chart + callout** | A simple line or bar chart on the left, a big accent arrow or "Up to 10×" style callout on the right. |
| **F. Concentric circles** | 2 or 3 nested circles in accent tints, largest to smallest, each labeled with a number and a short caption. Title and one line of text on the left. |
| **G. Full diagram** | Kicker and one-line title, then a single diagram across the whole card. |

## Our 11 slides mapped to layouts

| # | Kicker | Title (accent phrase in bold) | Layout | Visual |
|---|---|---|---|---|
| 1 | LIFEVISION | A pool camera that **notices** | A | Cover image `images/generated/cover-pool-camera.png`, team names small at the bottom |
| 2 | THE PROBLEM | Drowning is **silent** | C | Big stats: leading cause of death ages 1 to 4, rising again since 2020. Cross bullets for why we chose drowning over factory falls |
| 3 | OUR PRODUCT | Uses the camera you **already own** | B | Laptop mockup of the review app with boxes and one yellow box, on a teal circle |
| 4 | HOW IT WORKS | The **full pipeline** | G | `pipeline.png` (data → labels → models tried → fine-tuned → home camera). Models tried but not used in grey |
| 5 | THE GAP | Bridging the gap between **simulation and real pools** | A | Right: Isaac Sim still. Bullets: no dataset exists, can't film it, simulated children and adults. NVIDIA and Google DeepMind logos |
| 6 | SIMULATION | Data we **could not film** | C | MuJoCo and Isaac stills side by side, one before → after stat (stock YOLO11n 16% → fine-tuned 95% of people fully under water, on simulation) |
| 7 | FINE-TUNING | Trained on a **real wave pool** | E | Accuracy chart on the left (teammate's diagram), callout on the right with the before → after score |
| 8 | TRACKING | **One ID** per person | C | `images/before_after_tracking.png`, stat: 18 IDs → 4 IDs for 4 people. Dotted-box explanation in one line |
| 9 | LIVE DEMO | See it **run** | B | Mockup frame only; switch to the app. Recorded backup ready |
| 10 | DEPLOYMENT | Two ways to **bring it home** | D | Circle holds the timer (yellow 5 s, red 12 s in the demo); stacked options: runs locally (footage stays home), our cloud (cheaper, faster), camera provider APIs |
| 11 | WHAT'S NEXT | From pools to **every camera** | F | Concentric circles: pools today → home cameras → factories. Closing line: a supervision aid, not a replacement |

## Rules for the text

- Titles say what the slide is about. No slogans.
- Every number on a slide from simulation is labeled "on simulation".
- Say "detects prolonged time out of sight", not "detects drowning".
- No em dashes. No slash shorthand ("X/Y"); write "X and Y".
