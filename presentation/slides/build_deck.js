// Builds presentation/slides/pool-assistant-draft.pptx (10 slides, SpaceX-style black and white).
// Content and speaker notes follow slide-context.md. Numbers are on simulation only.
// Usage: npm install pptxgenjs && node presentation/slides/build_deck.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const P = path.resolve(__dirname, "..");
const img = (f) => path.join(P, "images", f);
const vid = (f) => path.join(P, "videos", f);
const b64 = (f) => "image/png;base64," + fs.readFileSync(f).toString("base64");

// Palette: black dominates, white text, grays for secondary, color only for alert states.
const C = {
  bg: "000000", panel: "111214", line: "3A3D42", white: "FFFFFF", gray: "A7A9AC", dim: "6E7075",
  green: "3DDC84", amber: "FFB000", red: "FF4D4D",
};
const FONT = "Arial";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Pool camera assistant";

function base(title, kicker) {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  if (kicker) {
    s.addText(kicker, { x: 0.5, y: 0.3, w: 9, h: 0.3, fontFace: FONT, fontSize: 10, color: C.dim,
      charSpacing: 4, margin: 0, isTextBox: true });
  }
  s.addText(title, { x: 0.5, y: kicker ? 0.58 : 0.4, w: 9, h: 0.6, fontFace: FONT, fontSize: 26, bold: true,
    color: C.white, charSpacing: 3, margin: 0, isTextBox: true });
  return s;
}

function caption(s, text, y = 5.1) {
  s.addText(text, { x: 0.5, y, w: 9, h: 0.3, fontFace: FONT, fontSize: 10, color: C.dim, margin: 0, isTextBox: true });
}

function box(s, x, y, w, h, text, opts = {}) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: opts.fill || C.panel },
    line: { color: opts.line || C.line, width: 1, dashType: opts.dash || "solid" } });
  s.addText(text, { x, y, w, h, fontFace: FONT, fontSize: opts.size || 11, bold: opts.bold !== false,
    color: opts.color || C.white, align: "center", valign: "middle", charSpacing: opts.cs || 1,
    margin: 4, isTextBox: true });
}

function arrow(s, x1, y, x2) {
  s.addShape(pres.shapes.LINE, { x: x1, y, w: x2 - x1, h: 0, line: { color: C.gray, width: 1.25, endArrowType: "triangle" } });
}

// 1. Hook
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addImage({ path: img("demo_hq_still.png"), x: 5.0, y: 0, w: 5.0, h: 5.625,
    sizing: { type: "cover", w: 5.0, h: 5.625 } });
  s.addText("POOL CAMERA ASSISTANT", { x: 0.5, y: 0.45, w: 4.3, h: 0.3, fontFace: FONT, fontSize: 10,
    color: C.dim, charSpacing: 4, margin: 0, isTextBox: true });
  s.addText("PARENTS LOSE TRACK FOR A FEW SECONDS", { x: 0.5, y: 0.8, w: 4.3, h: 1.0, fontFace: FONT,
    fontSize: 21, bold: true, color: C.white, charSpacing: 3, margin: 0, valign: "top", isTextBox: true });
  s.addText("62%", { x: 0.5, y: 2.0, w: 4.3, h: 1.1, fontFace: FONT, fontSize: 72, bold: true,
    color: C.white, margin: 0, isTextBox: true });
  s.addText("of pool deaths of children under 5 happened when the adult lost track of where the child was.",
    { x: 0.5, y: 3.15, w: 4.1, h: 0.8, fontFace: FONT, fontSize: 14, color: C.gray, margin: 0, valign: "top", isTextBox: true });
  s.addText("CPSC, 2020–2022. Going under is often quiet, and many backyards already have a camera on the pool.",
    { x: 0.5, y: 4.55, w: 4.1, h: 0.6, fontFace: FONT, fontSize: 10, color: C.dim, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Most pool accidents with young kids don't happen because nobody was home. They happen in the few seconds when the adult looked away. A lot of families already have a camera on the backyard, but it only records, it doesn't tell you anything.");
}

// 2. Idea: head timer
{
  const s = base("TIME HOW LONG EACH HEAD IS UNDER", "OUR IDEA");
  const y = 2.1, h = 1.1, w = 2.4;
  box(s, 0.5, y, w, h, "HEAD ABOVE WATER", { line: C.green, color: C.green });
  arrow(s, 0.5 + w + 0.1, y + h / 2, 3.5 - 0.1 + 0.05);
  box(s, 3.55, y, w, h, "5 S UNDER\nWARNING", { line: C.amber, color: C.amber });
  arrow(s, 3.55 + w + 0.1, y + h / 2, 6.6 - 0.05);
  box(s, 6.6, y, w + 0.4, h, "12 S UNDER\nALERT", { line: C.red, color: C.red });
  s.addText("Head comes back up: the timer resets and the box turns green again.", { x: 0.5, y: 3.55, w: 9, h: 0.35,
    fontFace: FONT, fontSize: 14, color: C.gray, margin: 0, isTextBox: true });
  s.addText("Also flags someone entering the pool.", { x: 0.5, y: 3.95, w: 9, h: 0.35,
    fontFace: FONT, fontSize: 14, color: C.gray, margin: 0, isTextBox: true });
  caption(s, "Pool alarm standard test: a dummy on the bottom for 20 s. CPSC says 20 s is too late, so we alert well before.");
  s.addNotes("The camera tracks every person and times how long their head has been under the water. At 5 seconds the box turns yellow, at 12 seconds it turns red and the app alerts. The industry test for pool alarms is a dummy on the bottom for 20 seconds, and CPSC says 20 seconds is too late, so we set our numbers well under that. It's a supervision aid that detects prolonged head submersion, not a certified lifesaving device.");
}

// 3. Pipeline
{
  const s = base("PIPELINE", "HOW IT WORKS");
  const steps = ["CAMERA\nVIDEO", "DETECTOR", "BOX\nCLEANUP", "TRACKER +\nID STITCHING", "EVENT\nENGINE", "APP"];
  const w = 1.2, gap = 0.36, y = 1.75, h = 0.95;
  steps.forEach((t, i) => {
    const x = 0.5 + i * (w + gap);
    box(s, x, y, w, h, t, { size: 10 });
    if (i < steps.length - 1) arrow(s, x + w + 0.04, y + h / 2, x + w + gap - 0.04);
  });
  box(s, 2.06, 3.35, 4.3, 0.75, "SIMULATOR: EXACT BOXES AND HEAD HEIGHT\nFOR TRAINING AND SCORING",
    { dash: "dash", size: 10, color: C.gray, fill: C.bg });
  s.addShape(pres.shapes.LINE, { x: 2.3, y: 3.35, w: 0, h: -0.6, line: { color: C.gray, width: 1, dashType: "dash", endArrowType: "triangle" } });
  s.addText("Detector finds people. Tracker keeps one ID per person, so each gets its own timer. Event engine runs the timers and the entry alert.",
    { x: 0.5, y: 4.35, w: 9, h: 0.5, fontFace: FONT, fontSize: 12, color: C.gray, margin: 0, isTextBox: true });
  caption(s, "Everything runs locally on a MacBook. The pool video never leaves the device.");
  s.addNotes("The detector finds people in each frame, and the tracker keeps the same ID on each person over time, so each person gets their own timer. The event engine runs the timers and the entry alert, and the app draws the boxes. Everything runs locally on a MacBook, so the pool video never leaves the device.");
}

// 4. Test bench
{
  const s = base("WE BUILT A TEST BENCH", "NO DROWNING FOOTAGE, SO: SIMULATION");
  s.addText([
    { text: "MuJoCo physics sim: shallow and deep end, buoyancy added (full lungs float, empty lungs sink)", options: { bullet: true, breakLine: true } },
    { text: "Swim, float, dive, silent sink, collapse, fall in from the deck", options: { bullet: true, breakLine: true } },
    { text: "Exact boxes and head height for every frame, no hand labeling", options: { bullet: true } },
  ], { x: 0.5, y: 1.5, w: 4.0, h: 2.8, fontFace: FONT, fontSize: 14, color: C.white, paraSpaceAfter: 10,
    valign: "top", margin: 0, isTextBox: true });
  s.addMedia({ type: "video", path: vid("sim_raw_baseline.mp4"), cover: b64(img("sim_baseline_still.png")),
    x: 4.9, y: 1.5, w: 4.6, h: 2.59 });
  caption(s, "Train on 4 scenarios, test on a 5th the models never see: a diver under about 5 s who comes up 2 m away.");
  s.addNotes("We can't film real drowning, and we shouldn't ask anyone to fake it underwater. So we built a physics sim where we script people swimming, sinking, and collapsing, and the sim tells us exactly where every person and head is. We train on 4 scenarios and test on a 5th the models never see, where a diver goes under for about 5 seconds and comes up 2 meters away. Rohan also built a second version in Isaac Sim with more realistic people and water.");
}

// 5. Stock models
{
  const s = base("YOLO BARELY SEES OUR PEOPLE", "STOCK MODELS, ON SIMULATION");
  s.addChart(pres.charts.BAR, [
    { name: "People found", labels: ["YOLO11n", "YOLOv8n", "RF-DETR Nano"], values: [25, 12, 79] },
    { name: "Fully under water", labels: ["YOLO11n", "YOLOv8n", "RF-DETR Nano"], values: [2, 31, 60] },
  ], { x: 0.4, y: 1.35, w: 5.4, h: 3.55, barDir: "col", barGapWidthPct: 60,
    chartColors: [C.white, C.dim], showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.white,
    dataLabelFontSize: 11, dataLabelFormatCode: '0"%"', catAxisLabelColor: C.gray, valAxisLabelColor: C.dim,
    catAxisLabelFontSize: 11, valAxisLabelFontSize: 9, valAxisMaxVal: 100, valAxisLabelFormatCode: '0"%"',
    valGridLine: { color: "222326", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendColor: C.gray, legendFontSize: 10 });
  const card = (y, head, body) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 6.1, y, w: 3.4, h: 1.55, fill: { color: C.panel }, line: { color: C.line, width: 1 } });
    s.addText(head, { x: 6.3, y: y + 0.15, w: 3.0, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: C.white, charSpacing: 2, margin: 0, isTextBox: true });
    s.addText(body, { x: 6.3, y: y + 0.5, w: 3.0, h: 0.95, fontFace: FONT, fontSize: 11, color: C.gray, margin: 0, valign: "top", isTextBox: true });
  };
  card(1.4, "YOLO", "CNN trained on COCO photos. Guesses boxes everywhere, then removes duplicates (NMS). About 2.6 M parameters.");
  card(3.15, "RF-DETR", "Transformer on a DINOv2 backbone. 300 learned queries, each claims at most one object. About 30 M parameters.");
  caption(s, "Held-out scenario, on simulation. No training on our data.");
  s.addNotes("We tried two YOLO models, Grounding DINO, and RF-DETR, all out of the box. YOLO is a CNN trained on COCO photos: it guesses boxes everywhere and filters duplicates after. RF-DETR is a transformer with a DINOv2 backbone that learned general shapes from a much wider set of images, and each of its 300 queries claims at most one object. That's our best explanation for why it found 79% of people with no training on our data, while YOLO found 25% at most.");
}

// 6. Fine-tuning
{
  const s = base("FINE-TUNING ON SIM FRAMES", "RF-DETR NANO, ON SIMULATION");
  s.addText("79% → 100%", { x: 0.5, y: 1.45, w: 4.2, h: 1.0, fontFace: FONT, fontSize: 48, bold: true, color: C.white, margin: 0, isTextBox: true });
  s.addText("people found on the scenario it never saw, after 1 epoch (about 2 min on a MacBook, 180 sim frames)",
    { x: 0.5, y: 2.5, w: 4.0, h: 0.9, fontFace: FONT, fontSize: 13, color: C.gray, margin: 0, valign: "top", isTextBox: true });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: "1C1D20" } } });
  const cell = (t, strong) => ({ text: t, options: { color: strong ? C.white : C.gray, bold: !!strong } });
  s.addTable([
    [hdr("Setup"), hdr("Test video"), hdr("Found"), hdr("Fully under")],
    [cell("Stock"), cell("held-out scenario"), cell("79%"), cell("60%")],
    [cell("Fine-tuned", true), cell("held-out scenario"), cell("100%", true), cell("100%", true)],
    [cell("Stock"), cell("new render"), cell("76%"), cell("34%")],
    [cell("Fine-tuned", true), cell("new render"), cell("93%", true), cell("71%", true)],
  ], { x: 4.9, y: 1.5, w: 4.6, colW: [1.15, 1.55, 0.9, 1.0], rowH: 0.42, fontFace: FONT, fontSize: 11,
    border: { type: "solid", color: C.line, pt: 0.75 }, fill: { color: C.bg }, valign: "middle" });
  s.addText("A new render with textures and body shapes the model never saw costs accuracy. Real footage will be a bigger change.",
    { x: 4.9, y: 3.85, w: 4.6, h: 0.7, fontFace: FONT, fontSize: 11, color: C.gray, margin: 0, valign: "top", isTextBox: true });
  caption(s, "On simulation only. Not real-world accuracy.");
  s.addNotes("We fine-tuned on 180 sim frames. One epoch, about 2 minutes on a MacBook, took RF-DETR to 100% on the scenario it never saw, including people fully under water. But on a new sim render with textures and body shapes it never saw, it dropped to 93%, and 71% for people fully under. Even a small visual change costs accuracy, which is why real footage is our next step.");
}

// 7. Tracking
{
  const s = base("A GOOD DETECTOR STILL BROKE THE IDS", "TRACKING");
  s.addText("18 → 4", { x: 0.5, y: 1.35, w: 3.2, h: 0.9, fontFace: FONT, fontSize: 48, bold: true, color: C.white, margin: 0, isTextBox: true });
  s.addText("IDs on screen for 4 people. Zero ID switches across all 5 scenarios.", { x: 3.8, y: 1.45, w: 5.7, h: 0.75,
    fontFace: FONT, fontSize: 14, color: C.gray, margin: 0, valign: "middle", isTextBox: true });
  s.addImage({ path: img("before_after_tracking.png"), x: 0.5, y: 2.45, w: 9.0, h: 2.53 });
  s.addText("STOCK + DEFAULT TRACKER", { x: 0.6, y: 2.52, w: 3.5, h: 0.28, fontFace: FONT, fontSize: 9, bold: true,
    color: C.white, fill: { color: C.bg }, charSpacing: 2, margin: 3, isTextBox: true });
  s.addText("FINE-TUNED + OUR RULES", { x: 5.1, y: 2.52, w: 3.3, h: 0.28, fontFace: FONT, fontSize: 9, bold: true,
    color: C.white, fill: { color: C.bg }, charSpacing: 2, margin: 3, isTextBox: true });
  caption(s, "Rules: remove duplicate and part boxes, start an ID after 3 frames, give back an old ID if the person reappears nearby within 5 s.");
  s.addNotes("With the default tracker we got 18 IDs for 4 people, and a timer is useless if a person keeps getting a new ID. We added rules: remove duplicate and part boxes, only start an ID after 3 frames, and give a person their old ID back if they reappear nearby within 5 seconds. That got us to 4 IDs for 4 people with zero switches across all 5 scenarios. We also saw that in clear water the body stays visible under the surface, so the alarm has to time the head, not wait for the box to disappear.");
}

// 8. Demo placeholder
{
  const s = base("DEMO", "PLACEHOLDER: FRONTEND RECORDING GOES HERE");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.4, w: 4.3, h: 2.95, fill: { color: C.bg }, line: { color: C.gray, width: 1, dashType: "dash" } });
  s.addText([
    { text: "4 camera feeds, then Enable monitoring", options: { bullet: true, breakLine: true } },
    { text: "Green, yellow, red boxes with timer and reason", options: { bullet: true, breakLine: true } },
    { text: "Alert panel and incident replay", options: { bullet: true } },
  ], { x: 0.75, y: 1.65, w: 3.8, h: 2.45, fontFace: FONT, fontSize: 13, color: C.gray, paraSpaceAfter: 10, valign: "middle", margin: 0, isTextBox: true });
  s.addMedia({ type: "video", path: vid("demo_hq_tracking_finetuned_rfdetr.mp4"), cover: b64(img("demo_hq_tracking_still.png")),
    x: 5.1, y: 1.4, w: 4.4, h: 2.475 });
  s.addText("FALLBACK: SIM TRACKING, FINE-TUNED RF-DETR", { x: 5.1, y: 3.95, w: 4.4, h: 0.3, fontFace: FONT, fontSize: 9,
    bold: true, color: C.dim, charSpacing: 2, margin: 0, isTextBox: true });
  caption(s, "If the recording shows scripted boxes, label it \"scripted demo\". Demo thresholds carry a DEMO THRESHOLDS badge.");
  s.addNotes("PLACEHOLDER. Full demo: Here are four feeds. When we turn on monitoring, every person gets a box. When someone's head stays under, their box turns yellow with a timer, then red, and the alert panel lets the parent replay what happened. Fallback if the app is not ready: The app isn't ready to show, so here is the tracking running on our sim. Each person keeps one ID the whole time, which is what the timer needs.");
}

// 9. Limits
{
  const s = base("LIMITS, AND MISTAKES WE CAUGHT", "HONEST NUMBERS");
  const rows = [
    ["01", "All numbers are on simulation: simple bodies, flat water, no splash or glare."],
    ["02", "Caught 2 bugs that made scores look too good (water not drawn, color-swapped test images) and re-measured everything."],
    ["03", "Daylight only, one person class, no child vs adult yet."],
  ];
  rows.forEach(([n, t], i) => {
    const y = 1.5 + i * 1.05;
    s.addText(n, { x: 0.5, y, w: 0.7, h: 0.6, fontFace: FONT, fontSize: 24, bold: true, color: C.dim, margin: 0, isTextBox: true });
    s.addText(t, { x: 1.25, y, w: 3.9, h: 0.85, fontFace: FONT, fontSize: 13, color: C.white, margin: 0, valign: "top", isTextBox: true });
  });
  s.addImage({ path: img("sim_pool_scene.png"), x: 5.5, y: 1.5, w: 4.0, h: 2.25 });
  caption(s, "None of this is real-world accuracy yet.");
  s.addNotes("None of this is real-world accuracy yet. Our first YOLO scores were way too good, and it turned out the water wasn't drawn in some renders. Later a second script disagreed with the first and we found we were feeding the models color-swapped images, so we fixed it and re-measured every number on these slides.");
}

// 10. Next steps
{
  const s = base("NEXT STEPS", "WHAT COMES AFTER THE HACKATHON");
  const steps = [["01", "RECORD", "Real pool footage, label each head above or below water"],
    ["02", "FINE-TUNE", "RF-DETR the same way, test only on real clips"],
    ["03", "CONNECT", "Head timer and alerts drive the app's colors"]];
  steps.forEach(([n, h, t], i) => {
    const x = 0.5 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.5, w: 2.8, h: 2.1, fill: { color: C.panel }, line: { color: C.line, width: 1 } });
    s.addText(n, { x: x + 0.25, y: 1.7, w: 2.3, h: 0.45, fontFace: FONT, fontSize: 22, bold: true, color: C.dim, margin: 0, isTextBox: true });
    s.addText(h, { x: x + 0.25, y: 2.2, w: 2.3, h: 0.35, fontFace: FONT, fontSize: 14, bold: true, color: C.white, charSpacing: 2, margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.25, y: 2.6, w: 2.3, h: 0.85, fontFace: FONT, fontSize: 12, color: C.gray, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("A supervision aid that detects prolonged head submersion. It does not replace watching children, pool fences, or lifeguards.",
    { x: 0.5, y: 3.95, w: 9.0, h: 0.6, fontFace: FONT, fontSize: 13, bold: true, color: C.white, margin: 0, isTextBox: true });
  caption(s, "Sunny · Joanne · Sam · Rohan · David");
  s.addNotes("Next we record real footage, label whether each head is above or below the water, and fine-tune the same way, testing only on real clips. Then we connect the head timer so the boxes turn yellow and red on their own. This is a second set of eyes for the parent, not a replacement for watching. Thanks.");
}

const out = path.join(__dirname, "pool-assistant-draft.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));
