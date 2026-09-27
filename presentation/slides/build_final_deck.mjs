import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = process.cwd();
const CODEX_HOME = process.env.CODEX_HOME ?? path.join(process.env.HOME ?? "", ".codex");
const SKILL_DIR = process.env.PRESENTATIONS_SKILL_DIR
  ?? path.join(CODEX_HOME, "plugins", "cache", "openai-primary-runtime", "presentations", "26.921.11914", "skills", "presentations");
const TMP_DIR = path.join(workspaceDir, ".codex-build", "presentation");
const FINAL_PPTX = path.join(workspaceDir, "presentation", "slides", "final", "life-vision-hackathon-deck.pptx");
const RUNTIME_PYTHON = process.env.RUNTIME_PYTHON ?? "python3";

const { applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const W = 1280;
const H = 720;
const FONT = "Avenir Next";
const C = {
  bg: "#071411",
  panel: "#0B1F1A",
  panel2: "#102822",
  line: "#24443A",
  text: "#EDFDF6",
  muted: "#8AACA0",
  green: "#41F28D",
  yellow: "#FFD447",
  red: "#FF4E5C",
  blue: "#64B7FF",
  ink: "#04120D",
};

const IMG = path.join(workspaceDir, "presentation", "images");
const imagePaths = {
  cover: path.join(IMG, "generated", "cover-pool-camera.png"),
  people: path.join(IMG, "generated", "safe-pool-demo.png"),
  camera: path.join(IMG, "generated", "camera-pool-integration.png"),
  tracking: path.join(IMG, "before_after_tracking.png"),
  sim: path.join(IMG, "sim_pool_scene.png"),
  isaac: path.join(IMG, "isaac_sim_pool_editor.png"),
  demo: path.join(IMG, "demo_hq_tracking_still.png"),
};

const blobs = Object.fromEntries(
  await Promise.all(Object.entries(imagePaths).map(async ([key, file]) => [key, new Uint8Array(await fs.readFile(file))])),
);

const pres = Presentation.create({ slideSize: { width: W, height: H } });

function rect(slide, x, y, width, height, fill, radius = 0, lineFill = "none", lineWidth = 0) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left: x, top: y, width, height },
    fill,
    line: { fill: lineFill, width: lineWidth },
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function text(slide, value, x, y, width, height, options = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width, height },
    fill: options.fill ?? "none",
    line: { fill: options.lineFill ?? "none", width: options.lineWidth ?? 0 },
    ...(options.radius ? { borderRadius: options.radius } : {}),
  });
  box.text = value;
  box.text.style = {
    typeface: FONT,
    fontSize: options.size ?? 24,
    bold: options.bold ?? false,
    color: options.color ?? C.text,
    autoFit: "shrinkText",
    verticalAlignment: options.valign ?? "middle",
    textAlign: options.align ?? "left",
  };
  return box;
}

function image(slide, key, x, y, width, height, alt, fit = "cover", radius = 0, crop) {
  return slide.images.add({
    blob: blobs[key],
    contentType: "image/png",
    alt,
    fit,
    position: { left: x, top: y, width, height },
    ...(radius ? { geometry: "roundRect", borderRadius: radius } : {}),
    ...(crop ? { crop } : {}),
  });
}

function topRule(slide, section, page) {
  text(slide, section.toUpperCase(), 58, 31, 520, 26, { size: 13, bold: true, color: C.green });
  text(slide, String(page).padStart(2, "0"), 1170, 31, 52, 26, { size: 13, bold: true, color: C.muted, align: "right" });
  rect(slide, 58, 69, 1164, 1, C.line);
}

function title(slide, value, y = 92, width = 1120) {
  return text(slide, value, 58, y, width, 76, { size: 42, bold: true, color: C.text, valign: "top" });
}

function footer(slide, value = "RESEARCH PROTOTYPE · SYNTHETIC RESULTS WHERE NOTED") {
  text(slide, value, 58, 684, 1000, 18, { size: 9, color: C.muted });
}

function note(slide, script, sources = []) {
  const sourceBlock = sources.length ? `\n\nSources:\n${sources.map((source) => `- ${source}`).join("\n")}` : "";
  slide.speakerNotes.textFrame.setText(`${script}${sourceBlock}`);
}

function node(slide, label, x, y, width, state = "normal") {
  const color = state === "green" ? C.green : state === "yellow" ? C.yellow : state === "red" ? C.red : C.line;
  const fill = state === "normal" ? C.panel2 : C.panel;
  const shape = rect(slide, x, y, width, 74, fill, 14, color, state === "normal" ? 1 : 2);
  text(slide, label, x + 12, y + 10, width - 24, 54, { size: 17, bold: true, color: state === "normal" ? C.text : color, align: "center" });
  return shape;
}

function arrow(slide, from, to) {
  return slide.shapes.connect(from, to, {
    kind: "straight",
    fromSide: "right",
    toSide: "left",
    line: { fill: C.muted, width: 2 },
    tail: { type: "arrow", width: "med", length: "med" },
  });
}

// Slide 1: cover
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  image(s, "cover", 0, 0, W, H, "Concept image of a backyard pool viewed from above", "cover");
  rect(s, 0, 0, 535, H, C.bg);
  rect(s, 58, 92, 38, 38, C.green, 10);
  rect(s, 66, 111, 5, 11, C.ink, 2);
  rect(s, 75, 101, 5, 21, C.ink, 2);
  rect(s, 84, 106, 5, 16, C.ink, 2);
  text(s, "LIFE VISION", 58, 173, 430, 74, { size: 54, bold: true, color: C.text });
  text(s, "Explainable pool-risk monitoring\nfor cameras families already own", 58, 255, 410, 94, { size: 25, color: C.green, valign: "top" });
  text(s, "Hackathon prototype", 58, 594, 250, 30, { size: 14, bold: true, color: C.text });
  text(s, "David · Rohan · Sunny · Joanne · Sam", 58, 629, 420, 25, { size: 12, color: C.muted });
  note(s,
    "Imagine you are at a backyard pool with friends or family. For one moment, everyone looks away. The camera keeps recording, but recording alone cannot tell you when normal play has turned into danger. Life Vision adds that missing layer. It watches each person over time and warns the adult when someone's head has stayed below the water too long.",
    ["Concept image generated for this presentation. It does not depict a real incident."]
  );
}

// Slide 2: problem and use case
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "The use case", 2);
  title(s, "A short gap in supervision can become a pool emergency");
  text(s, "61%", 58, 192, 300, 118, { size: 82, bold: true, color: C.green, valign: "top" });
  text(s, "of reported pool and spa fatalities involving children under five were associated with a gap in adult supervision", 62, 312, 420, 116, { size: 22, color: C.text, valign: "top" });
  text(s, "FIRST USE CASE", 62, 490, 170, 22, { size: 11, bold: true, color: C.muted });
  text(s, "Backyard pool · existing camera · responsible adult nearby", 62, 518, 430, 62, { size: 18, bold: true, color: C.text, valign: "top" });
  image(s, "people", 566, 180, 656, 420, "Two adults safely swimming in a backyard pool during a staged demo", "cover", 20);
  footer(s, "CPSC 2024 REPORT · REPORTED FATALITIES, 2019–2021");
  note(s,
    "The Consumer Product Safety Commission reviewed reported pool and spa fatalities involving children under five. Sixty-one percent involved a gap in adult supervision, meaning an adult lost contact with the child long enough for the child to reach the water. Many of these homes already have a backyard camera. Our first use case is simple: turn that existing view into a second set of eyes, while keeping supervision and physical pool barriers as the primary protections.",
    [
      "U.S. CPSC, Pool or Spa Submersion: Estimated Nonfatal Drowning Injuries and Reported Drownings, 2024 Report, Table 14: https://www.cpsc.gov/s3fs-public/Pool-or-Spa-Submersion-Estimated-Nonfatal-Drowning-Injuries-and-Reported-Drownings-2024-Report.pdf",
      "Pool image generated for this presentation. It depicts safe staged activity, not a real incident.",
    ]
  );
}

// Slide 3: interaction
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "User interaction", 3);
  title(s, "Every alert shows who, why, and for how long");
  const green = node(s, "GREEN\ntracked normally", 58, 220, 238, "green");
  const yellow = node(s, "YELLOW\nwarning timer", 375, 220, 238, "yellow");
  const red = node(s, "RED\ncheck the pool", 692, 220, 238, "red");
  arrow(s, green, yellow);
  arrow(s, yellow, red);
  text(s, "Head evidence changes", 286, 184, 102, 32, { size: 11, color: C.muted, align: "center" });
  text(s, "Condition persists", 603, 184, 102, 32, { size: 11, color: C.muted, align: "center" });
  rect(s, 987, 195, 235, 148, C.panel2, 16, C.line, 1);
  text(s, "PERSON 02", 1007, 213, 195, 24, { size: 12, bold: true, color: C.text });
  text(s, "HEAD NOT ABOVE", 1007, 250, 195, 22, { size: 11, bold: true, color: C.yellow });
  text(s, "3.1 s", 1007, 277, 195, 42, { size: 28, bold: true, color: C.text });
  text(s, "1  Choose video or camera", 58, 410, 330, 42, { size: 19, bold: true, color: C.text });
  text(s, "2  Activate monitoring", 465, 410, 310, 42, { size: 19, bold: true, color: C.text });
  text(s, "3  Select alert preference", 852, 410, 370, 42, { size: 19, bold: true, color: C.text });
  text(s, "A brief Matrix-style scan marks the moment analysis starts. Demo thresholds are always labeled.", 58, 494, 1164, 62, { size: 20, color: C.muted, align: "center" });
  footer(s);
  note(s,
    "The user selects a prerecorded clip for the hackathon, then presses Activate Monitoring. We intentionally made activation theatrical, with a quick Matrix-style scan, so the audience can see exactly when analysis starts. Every person receives a stable ID and a green box. If a head remains below the surface, a timer starts. The box turns yellow at the warning threshold, then red if the condition persists. The interface always explains the decision, for example head not above water for 3.1 seconds. The user chooses yellow and red alerts, red only, or on-screen alerts.",
    ["Repository frontend: frontend/index.html, frontend/styles.css, frontend/app.js"]
  );
}

// Slide 4: architecture
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Technical architecture", 4);
  title(s, "One pipeline separates perception from the safety decision");
  const labels = ["VIDEO\nsource", "PERSON + HEAD\ndetector", "TRACK +\nID stitching", "EVENT\nengine", "APP\noverlay"];
  const xs = [58, 292, 526, 760, 994];
  const nodes = labels.map((label, index) => node(s, label, xs[index], 238, 170, index === 4 ? "green" : "normal"));
  for (let i = 0; i < nodes.length - 1; i++) arrow(s, nodes[i], nodes[i + 1]);
  rect(s, 348, 432, 584, 92, C.panel, 16, C.blue, 1);
  text(s, "SIMULATION TEST BENCH", 372, 448, 232, 24, { size: 12, bold: true, color: C.blue });
  text(s, "training frames · exact boxes · head height · scoring", 372, 478, 520, 26, { size: 17, color: C.text });
  s.shapes.add({ geometry: "line", position: { left: 640, top: 322, width: 0, height: 98 }, fill: "none", line: { fill: C.blue, width: 2 } });
  text(s, "Prerecorded video and timestamped JSON keep the hackathon demo local and reliable.", 214, 574, 852, 42, { size: 20, color: C.muted, align: "center" });
  footer(s);
  note(s,
    "The source reads a video with media timestamps. The detector finds each person and the available evidence for the head. ByteTrack links detections across frames. Our own identity layer handles short losses and resurfacing, because tracker IDs alone do not survive every dive. The event engine owns the timers, missing-person registry, and entry alert. It writes results as timestamped JSON, and the frontend draws those results over the video. For the demo this all runs locally from files, so network or cloud failures cannot break the presentation.",
    ["Repository architecture: docs/global/architecture.md"]
  );
}

// Slide 5: simulation
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Test data", 5);
  title(s, "Simulation gives exact answers for cases we cannot film safely");
  image(s, "sim", 58, 188, 548, 308, "MuJoCo swimming pool simulation", "cover", 18);
  image(s, "isaac", 674, 188, 548, 308, "Isaac Sim editor showing a realistic pool scene", "cover", 18, { left: 0.05, top: 0.02, right: 0.25, bottom: 0.04 });
  text(s, "MUJOCO", 58, 516, 170, 24, { size: 12, bold: true, color: C.green });
  text(s, "Physics, buoyancy, exact boxes and head height", 58, 548, 520, 54, { size: 19, color: C.text, valign: "top" });
  text(s, "ISAAC SIM", 674, 516, 170, 24, { size: 12, bold: true, color: C.blue });
  text(s, "More realistic people, lighting, water and held-out video", 674, 548, 520, 54, { size: 19, color: C.text, valign: "top" });
  footer(s, "SAFE TESTING · AUTOMATIC LABELS · SYNTHETIC ONLY");
  note(s,
    "We built a MuJoCo test bench with a shallow end, a deep end, buoyancy, and scripted behaviors such as swimming, resurfacing, collapse, entry, and a silent sink. The simulator gives us exact boxes and exact head height in every frame, so we can test the tracker and event logic without hand labeling. We also built an Isaac Sim scene with more realistic people, lighting, and water. Simulation lets us test dangerous cases safely, but it does not replace real pool footage.",
    ["Repository simulation details: docs/global/simulation.md", "Isaac results: model/README.md"]
  );
}

// Slide 6: results chart
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Model evidence", 6);
  title(s, "Fine-tuning improved underwater recall on held-out synthetic footage");
  const chart = s.charts.add("bar", {
    position: { left: 58, top: 190, width: 790, height: 400 },
    categories: ["Stock YOLO11n", "Stock RF-DETR-S", "MuJoCo tuned YOLO", "Isaac tuned YOLO"],
    series: [{
      name: "Recall, head fully under",
      values: [16, 75, 83, 95],
      fill: C.green,
      points: [
        { idx: 0, fill: C.muted },
        { idx: 1, fill: C.blue },
        { idx: 2, fill: C.yellow },
        { idx: 3, fill: C.green },
      ],
      valuesFormatCode: '0"%"',
    }],
    hasLegend: false,
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 62, varyColors: true },
    xAxis: { min: 0, max: 100, numberFormatCode: '0"%"', textStyle: { typeface: FONT, fontSize: 12, fill: C.muted }, majorGridlines: { line: { fill: C.line, width: 1 } } },
    yAxis: { textStyle: { typeface: FONT, fontSize: 14, fill: C.text } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: FONT, fontSize: 15, bold: true, fill: C.text }, numberFormatCode: '0"%"' },
    chartFill: C.bg,
    chartLine: { fill: "none", width: 0 },
    plotAreaFill: C.bg,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  rect(s, 902, 218, 320, 192, C.panel2, 18, C.line, 1);
  text(s, "95%", 930, 238, 260, 66, { size: 50, bold: true, color: C.green });
  text(s, "underwater recall", 932, 305, 250, 28, { size: 17, bold: true, color: C.text });
  text(s, "Precision: 95%\n261 Isaac training frames\nHeld-out seed-0 clip", 932, 347, 250, 72, { size: 15, color: C.muted, valign: "top" });
  text(s, "Synthetic-only result. Same scene and character set. This validates the pipeline, not real-world accuracy.", 902, 460, 320, 94, { size: 17, bold: true, color: C.yellow, valign: "top" });
  footer(s, "HELD-OUT ISAAC SIM CLIP · 526 BOXES WITH HEAD FULLY UNDER WATER");
  note(s,
    "This chart shows recall specifically when the head is fully below water on our held-out Isaac Sim clip. Stock YOLO11n found only sixteen percent. Stock RF-DETR-S found seventy-five percent. A YOLO model trained on the older MuJoCo characters transferred better at eighty-three percent. Our YOLO model fine-tuned on 261 Isaac frames reached ninety-five percent. Precision remained ninety-five percent. These are synthetic results from one scene and character set. They prove that our training and evaluation pipeline works. They do not predict real-world accuracy.",
    ["Repository measured results: model/README.md, held-out seed-0 clip"]
  );
}

// Slide 7: tracking
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Tracking", 7);
  title(s, "A strong detector still failed when the identity changed");
  text(s, "18", 58, 183, 130, 74, { size: 58, bold: true, color: C.red });
  text(s, "IDs for four people", 184, 197, 250, 38, { size: 20, bold: true, color: C.text });
  text(s, "became", 461, 197, 90, 38, { size: 18, color: C.muted, align: "center" });
  text(s, "4", 580, 183, 92, 74, { size: 58, bold: true, color: C.green });
  text(s, "IDs with zero switches", 672, 197, 300, 38, { size: 20, bold: true, color: C.text });
  image(s, "tracking", 58, 286, 1164, 300, "Before and after tracking comparison on the synthetic pool scenario", "contain", 14);
  text(s, "DEFAULT TRACKER", 76, 302, 220, 28, { size: 12, bold: true, color: C.text, fill: C.bg, radius: 5 });
  text(s, "BOX CLEANUP + ID STITCHING", 672, 302, 292, 28, { size: 12, bold: true, color: C.green, fill: C.bg, radius: 5 });
  footer(s, "FIVE MUJOCO SCENARIOS · SYNTHETIC ONLY");
  note(s,
    "The default tracker produced eighteen IDs for four people. That makes a per-person timer useless. We added three layers: box cleanup removes duplicate and partial detections, an ID starts only after several consistent frames, and identity stitching gives a resurfacing person their previous ID when the timing and location agree. On the five MuJoCo scenarios, the fine-tuned detector plus these rules produced one ID per person with zero switches. We also learned that a submerged body remains visible in clear water, so the alarm must time the head rather than wait for the whole box to disappear.",
    ["Repository tracking results: docs/global/simulation.md", "Visual: presentation/images/before_after_tracking.png"]
  );
}

// Slide 8: integrations
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  image(s, "camera", 0, 0, W, H, "Generic outdoor camera beside a backyard pool", "cover");
  rect(s, 0, 0, 650, H, C.bg);
  topRule(s, "Camera integration", 8);
  title(s, "One source adapter supports local video today and camera APIs later", 96, 560);
  const items = [
    ["NOW", "Uploaded video", C.green],
    ["NEXT", "Webcam and RTSP", C.yellow],
    ["LATER", "Ring WebRTC/WHEP", C.blue],
    ["LATER", "Nest WebRTC or RTSP", C.blue],
  ];
  items.forEach(([tag, label, color], index) => {
    const y = 292 + index * 61;
    text(s, tag, 58, y, 76, 24, { size: 10, bold: true, color });
    text(s, label, 146, y - 5, 380, 34, { size: 20, bold: true, color: C.text });
  });
  text(s, "The overlay appears in our application. The native Ring or Google Home screen remains unchanged.", 58, 572, 512, 70, { size: 18, color: C.muted, valign: "top" });
  footer(s, "AUTHORIZED ACCESS · ACCOUNT LINKING · PRIVACY REVIEW");
  note(s,
    "Every camera enters through the same source adapter. The hackathon uses uploaded video first, then a webcam. A compatible IP camera can provide an RTSP stream. Ring now offers an official developer platform with authorized device access, webhooks, and WebRTC video. Google Nest provides WebRTC or RTSP through Device Access, depending on the model. The colored overlay appears in our application, not inside the native Ring or Google Home screen. Account authorization and commercial review come later.",
    [
      "Ring Developer Experience: https://developer.amazon.com/docs/ring/get-started.html",
      "Google Nest supported devices: https://developers.google.com/nest/device-access/supported-devices",
      "Camera concept image generated for this presentation. It does not show a branded product.",
    ]
  );
}

// Slide 9: demo
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Live demo", 9);
  title(s, "Watch the state change, not just the box");
  image(s, "demo", 58, 188, 805, 452, "Synthetic pool scene with green person tracking boxes", "cover", 18);
  rect(s, 892, 188, 330, 452, C.panel, 18, C.line, 1);
  text(s, "DEMO RUN", 920, 214, 270, 24, { size: 12, bold: true, color: C.green });
  text(s, "01", 920, 270, 48, 30, { size: 20, bold: true, color: C.muted });
  text(s, "Activate monitoring", 974, 267, 214, 38, { size: 18, bold: true, color: C.text });
  text(s, "02", 920, 341, 48, 30, { size: 20, bold: true, color: C.muted });
  text(s, "Green becomes yellow", 974, 338, 214, 38, { size: 18, bold: true, color: C.yellow });
  text(s, "03", 920, 412, 48, 30, { size: 20, bold: true, color: C.muted });
  text(s, "Timer reaches red", 974, 409, 214, 38, { size: 18, bold: true, color: C.red });
  text(s, "04", 920, 483, 48, 30, { size: 20, bold: true, color: C.muted });
  text(s, "Explain and replay", 974, 480, 214, 38, { size: 18, bold: true, color: C.text });
  text(s, "STATE THE MODE", 920, 554, 270, 21, { size: 10, bold: true, color: C.green });
  text(s, "Real · recorded · scripted", 920, 578, 270, 28, { size: 16, color: C.text });
  footer(s, "SWITCH TO THE LOCAL FRONTEND NOW");
  note(s,
    "This is our prerecorded pool clip. We use staged behavior in shallow water with a dedicated observer. When I activate monitoring, the scan starts and the system assigns each person an ID. Watch Person 02. The box starts green. As the head-submersion evidence persists, the timer appears and the box changes to yellow. At the demo alarm threshold, it turns red and the alert fires. The timeline records what changed and why. State accurately whether the run uses real model inference, a recording of a pipeline run, or scripted interface results. Never describe scripted boxes as live model inference.",
    ["Frontend: frontend/index.html", "Fallback visual: presentation/images/demo_hq_tracking_still.png"]
  );
}

// Slide 10: limits and next steps
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "What comes next", 10);
  title(s, "The next test is real footage, separated by recording session");
  text(s, "LIMITS TODAY", 58, 202, 330, 24, { size: 12, bold: true, color: C.red });
  text(s, "Synthetic accuracy only", 58, 248, 440, 42, { size: 24, bold: true, color: C.text });
  text(s, "No real glare, splash, compression or crowded-pool occlusion", 58, 300, 470, 66, { size: 18, color: C.muted, valign: "top" });
  text(s, "One person class", 58, 395, 440, 42, { size: 24, bold: true, color: C.text });
  text(s, "No age-specific claim and no medical diagnosis", 58, 447, 470, 58, { size: 18, color: C.muted, valign: "top" });

  rect(s, 622, 196, 2, 360, C.line);
  text(s, "NEXT EXPERIMENT", 684, 202, 330, 24, { size: 12, bold: true, color: C.green });
  text(s, "Record and label real pool clips", 684, 248, 480, 42, { size: 24, bold: true, color: C.text });
  text(s, "Fine-tune on training sessions and test only on separate sessions", 684, 300, 480, 66, { size: 18, color: C.muted, valign: "top" });
  text(s, "Measure operational errors", 684, 395, 480, 42, { size: 24, bold: true, color: C.text });
  text(s, "Detection delay · missed events · false warnings per hour · ID continuity", 684, 447, 480, 64, { size: 18, color: C.muted, valign: "top" });
  rect(s, 58, 588, 1164, 64, C.panel2, 14, C.green, 1);
  text(s, "A supervision aid. It does not replace watching children, pool fences, or lifeguards.", 84, 599, 1112, 42, { size: 20, bold: true, color: C.text, align: "center" });
  footer(s, "LIFE VISION · SECOND SET OF EYES, NOT A SAFETY GUARANTEE");
  note(s,
    "Every accuracy number in this presentation comes from synthetic footage. The simulator lacks real glare, splashing, occlusion, camera compression, and the diversity of real people. Our next experiment is real pool footage split by recording session. We will label whether each head is above or below the surface, fine-tune on the training sessions, and evaluate only on separate sessions. We will report detection delay, missed events, false warnings per monitored hour, and identity errors. Life Vision remains a supervision aid. It does not replace watching children, pool fences, or lifeguards. We are not asking families to buy another camera. We are testing whether the camera they already have can recognize when the water stops looking normal.",
    ["Repository decisions: docs/global/decisions.md", "Evaluation plan: docs/global/architecture.md"]
  );
}

const candidatePath = path.join(TMP_DIR, "life-vision-candidate.pptx");
await (await PresentationFile.exportPptx(pres)).save(candidatePath);

for (let i = 0; i < pres.slides.length; i++) {
  const slide = pres.slides.getItemAt(i);
  const preview = await pres.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const requirements = {
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [6],
  materializeLiteralChartWorkbooks: true,
};
const fontPolicy = { basis: "design", families: [FONT] };
const expectedSlideSizeEmu = "12192000,6858000";

await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", expectedSlideSizeEmu,
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [6],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(TMP_DIR, "life-vision-hackathon-deck.pptx.validation.json"),
});

console.log(FINAL_PPTX);
