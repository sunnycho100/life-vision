import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = process.cwd();
const CODEX_HOME = process.env.CODEX_HOME ?? path.join(process.env.HOME ?? "", ".codex");
const SKILL_DIR = process.env.PRESENTATIONS_SKILL_DIR
  ?? path.join(CODEX_HOME, "plugins", "cache", "openai-primary-runtime", "presentations", "26.921.11914", "skills", "presentations");
const TMP_DIR = path.join(workspaceDir, ".codex-build", "presentation", "team-refined-v2");
const FINAL_PPTX = path.join(workspaceDir, "presentation", "slides", "final", "life-vision-team-refined.pptx");
const RUNTIME_PYTHON = process.env.RUNTIME_PYTHON ?? "python3";

const { applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const W = 1280;
const H = 720;
const FONT = "Arial";
const C = {
  bg: "#F8FAFC",
  white: "#FFFFFF",
  pale: "#EEF5F6",
  paleBlue: "#EEF4FA",
  line: "#D7E1E7",
  text: "#17324D",
  muted: "#5E7182",
  teal: "#087F72",
  blue: "#2B6FAE",
  amber: "#A86E00",
  red: "#C43D4D",
  ink: "#FFFFFF",
};

const IMG = path.join(workspaceDir, "presentation", "images");
const imagePaths = {
  cover: path.join(IMG, "generated", "cover-pool-camera.png"),
  safePool: path.join(IMG, "generated", "safe-pool-demo.png"),
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

function line(slide, x, y, width, height, color = C.line, lineWidth = 2) {
  return slide.shapes.add({
    geometry: "line",
    position: { left: x, top: y, width, height },
    fill: "none",
    line: { fill: color, width: lineWidth },
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
  text(slide, section.toUpperCase(), 58, 28, 560, 28, { size: 13, bold: true, color: C.teal });
  text(slide, String(page).padStart(2, "0"), 1170, 28, 52, 28, { size: 13, bold: true, color: C.muted, align: "right" });
  rect(slide, 58, 67, 1164, 1, C.line);
}

function title(slide, value, y = 90, width = 1164, size = 42) {
  return text(slide, value, 58, y, width, 92, { size, bold: true, color: C.text, valign: "top" });
}

function footer(slide, value = "SYNTHETIC RESULTS ARE LABELED") {
  text(slide, value, 58, 683, 1080, 16, { size: 9, color: C.muted });
}

function note(slide, script, sources = []) {
  const sourceBlock = sources.length ? `\n\nSources:\n${sources.map((source) => `- ${source}`).join("\n")}` : "";
  slide.speakerNotes.textFrame.setText(`${script}${sourceBlock}`);
}

function connector(slide, from, to, color = C.muted) {
  return slide.shapes.connect(from, to, {
    kind: "straight",
    fromSide: "right",
    toSide: "left",
    line: { fill: color, width: 2 },
    tail: { type: "arrow", width: "med", length: "med" },
  });
}

function pipelineNode(slide, label, owner, x, state = "normal") {
  const fill = state === "accent" ? C.pale : C.white;
  const border = state === "accent" ? C.teal : C.line;
  const shape = rect(slide, x, 215, 190, 108, fill, 16, border, state === "accent" ? 2 : 1);
  text(slide, label, x + 14, 229, 162, 48, { size: 18, bold: true, color: C.text, align: "center" });
  text(slide, owner, x + 14, 280, 162, 22, { size: 11, bold: true, color: state === "accent" ? C.teal : C.blue, align: "center" });
  return shape;
}

// Slide 1: cover
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  image(s, "cover", 700, 64, 522, 592, "Aerial concept image of a backyard pool at night", "cover", 24);
  rect(s, 638, 64, 2, 592, C.line);
  rect(s, 58, 92, 40, 40, C.teal, 11);
  rect(s, 67, 111, 5, 12, C.ink, 2);
  rect(s, 76, 101, 5, 22, C.ink, 2);
  rect(s, 85, 106, 5, 17, C.ink, 2);
  text(s, "LIFE VISION", 58, 175, 510, 78, { size: 54, bold: true, color: C.text });
  text(s, "A second set of eyes\nfor the backyard pool", 58, 256, 500, 104, { size: 29, color: C.teal, valign: "top" });
  text(s, "Hackathon prototype", 58, 576, 260, 30, { size: 14, bold: true, color: C.text });
  text(s, "David, Rohan, Sunny, Joanne and Sam", 58, 613, 520, 26, { size: 13, color: C.muted });
  note(s,
    "David, 15 seconds. A camera can record every second around a pool and still miss the moment that matters. Life Vision adds a local monitoring layer that tracks each person and explains when head-submersion evidence persists. Our goal is a second set of eyes for the responsible adult, using cameras families may already own.",
    ["Concept image generated for this presentation. It does not depict a real incident."]
  );
}

// Slide 2: health and human-attention facts
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Why this matters", 2);
  title(s, "The risk is fast, quiet, and easy to miss");

  const facts = [
    ["#1", "cause of death for U.S. children ages 1 to 4", C.red],
    ["61%", "of reported pool and spa fatalities under age 5 involved a gap in adult supervision", C.teal],
    ["2×", "In one U.S. case series of fatal incidents at lifeguarded pools, bystanders first noticed victims twice as often as lifeguards", C.blue],
  ];
  facts.forEach(([stat, label, color], index) => {
    const y = 195 + index * 137;
    text(s, stat, 58, y, 148, 90, { size: 58, bold: true, color, valign: "top" });
    text(s, label, 210, y + 4, 385, 92, { size: index === 2 ? 19 : 22, bold: true, color: C.text, valign: "top" });
    if (index < 2) rect(s, 58, y + 110, 536, 1, C.line);
  });
  image(s, "safePool", 656, 194, 566, 390, "Two adults safely swimming in a staged backyard pool scene", "cover", 22);
  text(s, "Technology can support attention. It cannot replace supervision or trained rescue staff.", 656, 604, 566, 46, { size: 18, bold: true, color: C.text, align: "center" });
  footer(s, "CDC 2026 · CPSC 2024 · LIFEGUARDED-POOL CASE SERIES 2000–2008");
  note(s,
    "Rohan, 30 seconds. Three facts shaped our problem statement. CDC reports that more children ages one to four die from drowning than from any other cause. CPSC found that sixty-one percent of reported pool and spa fatalities involving children under five were associated with a gap in adult supervision. A published case series identified 140 deaths in pools with lifeguards from 2000 through 2008, and victims were first identified twice as often by other swimmers or bystanders as by lifeguards. The authors emphasize that these deaths are uncommon and lifeguards remain important. Our point is that no human observer can see every person every second, so technology can add another layer without replacing supervision.",
    [
      "CDC, Drowning Facts: https://www.cdc.gov/drowning/data-research/facts/index.html",
      "U.S. CPSC, Pool or Spa Submersion, 2024 Report, Table 14: https://www.cpsc.gov/s3fs-public/Pool-or-Spa-Submersion-Estimated-Nonfatal-Drowning-Injuries-and-Reported-Drownings-2024-Report.pdf",
      "Pelletier and Gilchrist, Fatalities in swimming pools with lifeguards: USA, 2000–2008, Injury Prevention: https://pubmed.ncbi.nlm.nih.gov/21270060/",
      "CDC, Summer Swim Safety: https://www.cdc.gov/drowning/prevention/summer-swim-safety.html",
      "Pool image generated for this presentation. It depicts safe staged activity.",
    ]
  );
}

// Slide 3: scope evolution and decisions
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Scope decision", 3);
  title(s, "One observable rule replaced two age-based algorithms");

  text(s, "ORIGINAL IDEA", 58, 202, 330, 24, { size: 12, bold: true, color: C.muted });
  const baby = rect(s, 58, 242, 242, 104, C.white, 16, C.line, 1);
  text(s, "BABY MODEL", 78, 258, 202, 28, { size: 18, bold: true, color: C.text, align: "center" });
  text(s, "sink and disappear", 78, 292, 202, 30, { size: 16, color: C.muted, align: "center" });
  const adult = rect(s, 334, 242, 242, 104, C.white, 16, C.line, 1);
  text(s, "ADULT MODEL", 354, 258, 202, 28, { size: 18, bold: true, color: C.text, align: "center" });
  text(s, "repetitive distress motion", 354, 292, 202, 30, { size: 16, color: C.muted, align: "center" });

  text(s, "ADOPTED BUILD", 694, 202, 330, 24, { size: 12, bold: true, color: C.teal });
  rect(s, 694, 242, 528, 104, C.pale, 16, C.teal, 2);
  text(s, "ONE PERSON CLASS", 722, 258, 472, 28, { size: 21, bold: true, color: C.teal, align: "center" });
  text(s, "time how long the head is not above water", 722, 292, 472, 32, { size: 19, color: C.text, align: "center" });

  line(s, 632, 230, 0, 140, C.line, 2);
  text(s, "WHY THE TEAM CHANGED COURSE", 58, 408, 430, 24, { size: 12, bold: true, color: C.blue });
  text(s, "All five", 58, 448, 130, 28, { size: 17, bold: true, color: C.teal });
  text(s, "One class, configurable thresholds, and safe filming rules", 190, 445, 446, 36, { size: 18, color: C.text });
  text(s, "Joanne", 58, 502, 130, 28, { size: 17, bold: true, color: C.blue });
  text(s, "Age classification stays a later feature because child data requires consent", 190, 499, 446, 50, { size: 18, color: C.text });
  text(s, "Sunny, Sam, Rohan", 674, 448, 206, 28, { size: 17, bold: true, color: C.blue });
  text(s, "Active-distress pose becomes an optional second layer", 884, 445, 338, 42, { size: 18, color: C.text });
  text(s, "David", 674, 502, 206, 28, { size: 17, bold: true, color: C.teal });
  text(s, "Every warning must show who, why, and for how long", 884, 499, 338, 50, { size: 18, color: C.text });
  text(s, "Decision principle: build one explainable end-to-end system before adding specialized behavior models.", 58, 602, 1164, 42, { size: 20, bold: true, color: C.text, align: "center" });
  footer(s, "DECISION RECORD · 3 OF 5 SUPPORT REQUIRED");
  note(s,
    "Rohan, 30 seconds. We began with two algorithms: one for a baby who silently sinks and another for an adult showing repetitive distress motion. The team rejected that split for the hackathon. All five members supported one person class and configurable thresholds. Joanne's age classification idea moves to a later phase because we do not have consented child footage. Sunny, Sam, and Rohan kept active-distress pose as an optional second layer. David required every alert to explain who triggered it, why, and for how long. The core build therefore times one observable condition: the head is not above water.",
    ["Repository decision record: docs/global/decisions.md", "Original project narrative: docs/presentation.md"]
  );
}

// Slide 4: ownership pipeline and build timeline
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Team execution", 4);
  title(s, "Five workstreams met at one shared data contract");
  const nodes = [
    pipelineNode(s, "VIDEO\nsource", "David", 58),
    pipelineNode(s, "PERSON + HEAD\ndetector", "Joanne", 300),
    pipelineNode(s, "TRACK +\nID stitching", "Sunny", 542),
    pipelineNode(s, "EVENT\nengine", "Rohan", 784),
    pipelineNode(s, "APP +\nreplay", "David", 1026, "accent"),
  ];
  for (let i = 0; i < nodes.length - 1; i++) connector(s, nodes[i], nodes[i + 1]);
  text(s, "Sam tested head evidence and owned evaluation across detector, tracker, and event output.", 238, 350, 804, 38, { size: 18, bold: true, color: C.blue, align: "center" });

  line(s, 102, 489, 1076, 0, C.line, 3);
  const timeline = [
    ["H1", "Freeze JSON\ncontracts", "Everyone"],
    ["H3", "Sim tracks feed\nengine and UI", "Sunny, Rohan and David"],
    ["H6", "First end-to-end\nred alert", "Integration"],
    ["H9", "Detector and tracker\nreplace fake data", "Joanne, Sunny and Sam"],
    ["H12", "Freeze, rehearse,\nkeep a backup", "Everyone"],
  ];
  timeline.forEach(([hour, label, owner], index) => {
    const x = 78 + index * 238;
    rect(s, x + 72, 476, 26, 26, index === 2 ? C.teal : C.blue, 13);
    text(s, hour, x, 520, 170, 26, { size: 14, bold: true, color: index === 2 ? C.teal : C.blue, align: "center" });
    text(s, label, x, 550, 170, 52, { size: 17, bold: true, color: C.text, align: "center" });
    text(s, owner, x, 608, 170, 24, { size: 11, color: C.muted, align: "center" });
  });
  footer(s, "BUILD AGAINST FAKE DATA FIRST · NOBODY WAITS FOR ANOTHER WORKSTREAM");
  note(s,
    "Sunny, 35 seconds. The team could work in parallel because we froze two JSON contracts first. David built the overlay against hand-made results. Rohan built the event engine against Sunny's simulator tracks. Joanne delivered person and head evidence into the same contract. Sam evaluated each stage instead of waiting for one final accuracy number. By hour six, simulated tracks could drive a red alert in the frontend. By hour nine, real detector output replaced the fake data. The final hours were reserved for fixing, recording a fallback, and rehearsing.",
    ["Repository architecture: docs/global/architecture.md", "12-hour milestone plan: docs/global/milestones.md"]
  );
}

// Slide 5: simulation
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Safe test bench", 5);
  title(s, "Simulation covered cases the team could not film safely");
  image(s, "sim", 58, 194, 548, 312, "MuJoCo pool simulation with multiple scripted people", "cover", 18);
  image(s, "isaac", 674, 194, 548, 312, "Isaac Sim editor showing a realistic pool scene", "cover", 18, { left: 0.05, top: 0.02, right: 0.25, bottom: 0.04 });
  text(s, "SUNNY · MUJOCO", 58, 526, 240, 24, { size: 12, bold: true, color: C.teal });
  text(s, "Five scenarios with exact boxes, head height, and identity", 58, 556, 520, 56, { size: 19, bold: true, color: C.text, valign: "top" });
  text(s, "ROHAN · ISAAC SIM", 674, 526, 240, 24, { size: 12, bold: true, color: C.blue });
  text(s, "More realistic people, moving water, and a held-out 300-frame clip", 674, 556, 520, 56, { size: 19, bold: true, color: C.text, valign: "top" });
  text(s, "Safe cases: silent sink, collapse, entry, and resurfacing elsewhere", 58, 632, 1164, 30, { size: 17, color: C.muted, align: "center" });
  footer(s, "SIMULATION PROVIDES EXACT ANSWERS · IT DOES NOT ESTABLISH REAL-WORLD ACCURACY");
  note(s,
    "Rohan, 30 seconds. Sunny built a MuJoCo test bench with a shallow end, deep end, buoyancy, and five scenarios. It produced exact person boxes and exact head height without hand labeling. Rohan extended the work in Isaac Sim with more realistic characters, lighting, moving water, and a held-out clip. Simulation let the team test silent sinking, collapse, entry, and resurfacing elsewhere without putting anyone at risk. It also gave Sam exact answers for scoring. These environments validate the software pipeline, not real-world accuracy.",
    ["MuJoCo test bench: docs/global/simulation.md", "Isaac Sim results: model/README.md"]
  );
}

// Slide 6: detector decisions and results
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Model decisions", 6);
  title(s, "Two experiments changed the detector plan");
  text(s, "STOCK MODELS ON HELD-OUT MUJOCO", 58, 188, 520, 24, { size: 12, bold: true, color: C.teal });
  text(s, "People found", 58, 215, 520, 28, { size: 19, bold: true, color: C.text });
  const stockChart = s.charts.add("bar", {
    position: { left: 58, top: 246, width: 520, height: 292 },
    categories: ["YOLOv8n", "YOLO11n", "RF-DETR Nano"],
    series: [{
      name: "Recall",
      values: [12, 25, 79],
      fill: C.teal,
      points: [
        { idx: 0, fill: C.muted },
        { idx: 1, fill: C.blue },
        { idx: 2, fill: C.teal },
      ],
      valuesFormatCode: '0"%"',
    }],
    hasLegend: false,
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 64, varyColors: true },
    xAxis: { min: 0, max: 100, numberFormatCode: '0"%"', textStyle: { typeface: FONT, fontSize: 11, fill: C.muted }, majorGridlines: { line: { fill: C.line, width: 1 } } },
    yAxis: { textStyle: { typeface: FONT, fontSize: 13, fill: C.text } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: FONT, fontSize: 14, bold: true, fill: C.text }, numberFormatCode: '0"%"' },
    chartFill: C.bg,
    chartLine: { fill: "none", width: 0 },
    plotAreaFill: C.bg,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(stockChart, { fontFamily: FONT });

  line(s, 638, 190, 0, 360, C.line, 2);
  text(s, "HEAD FULLY UNDER ON HELD-OUT ISAAC", 682, 188, 540, 24, { size: 12, bold: true, color: C.blue });
  text(s, "People found", 682, 215, 540, 28, { size: 19, bold: true, color: C.text });
  const isaacChart = s.charts.add("bar", {
    position: { left: 682, top: 246, width: 540, height: 292 },
    categories: ["Stock YOLO11n", "Stock RF-DETR-S", "MuJoCo-tuned YOLO", "Isaac-tuned YOLO"],
    series: [{
      name: "Recall",
      values: [16, 75, 83, 95],
      fill: C.teal,
      points: [
        { idx: 0, fill: C.muted },
        { idx: 1, fill: C.blue },
        { idx: 2, fill: C.amber },
        { idx: 3, fill: C.teal },
      ],
      valuesFormatCode: '0"%"',
    }],
    hasLegend: false,
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 54, varyColors: true },
    xAxis: { min: 0, max: 100, numberFormatCode: '0"%"', textStyle: { typeface: FONT, fontSize: 11, fill: C.muted }, majorGridlines: { line: { fill: C.line, width: 1 } } },
    yAxis: { textStyle: { typeface: FONT, fontSize: 12, fill: C.text } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: FONT, fontSize: 14, bold: true, fill: C.text }, numberFormatCode: '0"%"' },
    chartFill: C.bg,
    chartLine: { fill: "none", width: 0 },
    plotAreaFill: C.bg,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(isaacChart, { fontFamily: FONT });

  rect(s, 58, 573, 1164, 68, C.pale, 14, C.line, 1);
  text(s, "Decision: keep lightweight YOLO for the hackathon pipeline. Retain RF-DETR as the strongest zero-shot candidate for unfamiliar footage.", 82, 586, 1116, 42, { size: 19, bold: true, color: C.text, align: "center" });
  footer(s, "ALL VALUES ON SIMULATION · SAME SCENE OR CHARACTER SET WHERE NOTED");
  note(s,
    "Joanne, 40 seconds. The teammate deck captured the first turning point. On the held-out MuJoCo scenario, stock YOLOv8n found twelve percent of people and YOLO11n found twenty-five percent. Stock RF-DETR Nano found seventy-nine percent. The second experiment used Rohan's held-out Isaac clip and focused on people whose heads were fully under water. Stock YOLO found sixteen percent, stock RF-DETR-S found seventy-five percent, the MuJoCo-tuned YOLO found eighty-three percent, and the Isaac-tuned YOLO reached ninety-five percent. That led to a practical split: use lightweight YOLO in the hackathon pipeline after fine-tuning, while treating RF-DETR as the stronger zero-shot candidate. Every number here is synthetic and does not predict real-pool performance.",
    ["MuJoCo detector results: docs/global/simulation.md", "Isaac held-out results: model/README.md"]
  );
}

// Slide 7: tracking and head timer
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Tracking decision", 7);
  title(s, "Stable identities made the head timer possible");
  image(s, "tracking", 58, 196, 760, 350, "Before and after comparison of person tracking on a synthetic pool scenario", "contain", 16);
  text(s, "18", 858, 204, 132, 84, { size: 66, bold: true, color: C.red });
  text(s, "IDs for four people", 986, 224, 236, 40, { size: 20, bold: true, color: C.text });
  line(s, 866, 307, 342, 0, C.line, 2);
  text(s, "4", 858, 332, 132, 84, { size: 66, bold: true, color: C.teal });
  text(s, "IDs after cleanup and stitching", 986, 342, 236, 66, { size: 20, bold: true, color: C.text, valign: "top" });
  text(s, "0 switches", 858, 440, 350, 42, { size: 24, bold: true, color: C.blue });
  text(s, "across all five MuJoCo scenarios", 858, 482, 350, 48, { size: 17, color: C.muted, valign: "top" });
  rect(s, 58, 579, 1164, 64, C.paleBlue, 14, C.blue, 1);
  text(s, "Clear water kept submerged bodies visible. The team changed the alarm from “person missing” to “head not above water.”", 82, 592, 1116, 40, { size: 19, bold: true, color: C.text, align: "center" });
  footer(s, "HELD-OUT MUJOCO RESURFACE SCENARIO · SYNTHETIC ONLY");
  note(s,
    "Sunny, 30 seconds. Detection alone did not give us a usable timer. Stock RF-DETR with the default tracker produced eighteen IDs for four people. Sunny added duplicate-box cleanup, delayed track confirmation, and identity stitching after short losses. The result was four IDs with zero switches across all five MuJoCo scenarios. The test bench also exposed a more important design mistake: in clear water, a submerged body can stay visible, so a missing-person timer never starts. Rohan's event logic therefore times the head state and keeps the missing registry only as backup.",
    ["Tracking results and edge logic: docs/global/simulation.md", "Team decision record: docs/global/decisions.md", "Visual: presentation/images/before_after_tracking.png"]
  );
}

// Slide 8: product demo
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Product experience", 8);
  title(s, "The demo explains the state change, not just the box");
  image(s, "demo", 58, 194, 790, 436, "Synthetic pool scene with person tracking boxes", "cover", 18);
  rect(s, 880, 194, 342, 436, C.white, 18, C.line, 1);
  text(s, "DAVID · FRONTEND", 906, 216, 290, 24, { size: 12, bold: true, color: C.teal });
  const states = [
    ["GREEN", "tracked normally", C.teal],
    ["YELLOW", "head timer running", C.amber],
    ["RED", "check the pool", C.red],
  ];
  states.forEach(([name, label, color], index) => {
    const y = 263 + index * 76;
    rect(s, 906, y + 5, 12, 46, color, 6);
    text(s, name, 934, y, 112, 27, { size: 16, bold: true, color });
    text(s, label, 934, y + 27, 246, 28, { size: 16, color: C.text });
  });
  line(s, 906, 502, 290, 0, C.line, 1);
  text(s, "USER CONTROL", 906, 520, 290, 22, { size: 11, bold: true, color: C.blue });
  text(s, "Alert preference\nTimer reason and incident replay", 906, 546, 290, 52, { size: 17, color: C.text, valign: "top" });
  text(s, "DEMO MODE MUST BE DISCLOSED", 906, 602, 290, 20, { size: 10, bold: true, color: C.red });
  footer(s, "ACTIVATE MONITORING · SHOW GREEN, YELLOW, RED · REPLAY THE EVENT");
  note(s,
    "David, about 50 seconds plus interaction. Start with the prerecorded team pool clip already loaded. Press Activate Monitoring and point out the intentionally visible scan. Each person gets a stable ID and a green box. When the head-submersion evidence persists, the timer appears and the box becomes yellow. At the demo threshold it becomes red, the alert sounds, and the incident can be replayed. Show that the user can select yellow and red alerts, red only, or on-screen alerts. State the mode accurately: real model inference, a recording of a successful pipeline run, or the scripted interaction prototype. Never describe scripted boxes as live inference.",
    ["Frontend: frontend/index.html, frontend/styles.css, frontend/app.js", "Fallback visual: presentation/images/demo_hq_tracking_still.png"]
  );
}

// Slide 9: mistakes and limits
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Evaluation discipline", 9);
  title(s, "Three mistakes changed the evaluation rules");
  const rows = [
    ["01", "Water was missing in one render batch", "Scores looked unrealistically high", "Render evidence before scoring"],
    ["02", "A test script swapped color channels", "Two scripts disagreed", "Cross-check with an independent path"],
    ["03", "The first head label was too strict", "Swimmers counted as submerged", "Rebuild labels from stored head height"],
  ];
  rows.forEach(([num, issue, signal, rule], index) => {
    const y = 194 + index * 124;
    text(s, num, 58, y, 66, 56, { size: 26, bold: true, color: C.red });
    text(s, issue, 132, y, 340, 56, { size: 20, bold: true, color: C.text, valign: "top" });
    text(s, signal, 502, y, 282, 56, { size: 18, color: C.muted, valign: "top" });
    rect(s, 812, y - 2, 410, 68, C.pale, 14, C.line, 1);
    text(s, rule, 834, y + 8, 366, 46, { size: 18, bold: true, color: C.teal, valign: "top" });
    if (index < 2) rect(s, 58, y + 84, 1164, 1, C.line);
  });
  rect(s, 58, 584, 1164, 68, C.paleBlue, 14, C.blue, 1);
  text(s, "The team fixed each issue and reran every published number. Accuracy remains synthetic-only.", 82, 598, 1116, 40, { size: 20, bold: true, color: C.text, align: "center" });
  footer(s, "SAM · EVALUATION AND FAILURE CASES");
  note(s,
    "Sam, 30 seconds. The team caught three errors that changed how we evaluate the system. A render batch omitted the water, which made early YOLO scores look far too good. A test script swapped color channels, and we found it only because an independent script disagreed. The first head-under label was too strict and misclassified ordinary swimmers. We fixed each problem, rebuilt the labels from stored head height, and reran every number. The remaining limitation is still fundamental: all accuracy results come from synthetic scenes with simplified water, people, and motion.",
    ["Corrections and known limits: docs/global/simulation.md", "Technical story: presentation/technical-story.md"]
  );
}

// Slide 10: next experiment and camera path
{
  const s = pres.slides.add();
  s.background.fill = C.bg;
  topRule(s, "Next step", 10);
  title(s, "Real footage is the next test, not a victory lap");
  text(s, "VALIDATE", 58, 204, 150, 24, { size: 12, bold: true, color: C.teal });
  text(s, "Record safe staged pool clips", 58, 238, 510, 34, { size: 23, bold: true, color: C.text });
  text(s, "Label each head above or below water. Split train and test by recording session.", 58, 278, 510, 66, { size: 18, color: C.muted, valign: "top" });
  text(s, "MEASURE", 58, 374, 150, 24, { size: 12, bold: true, color: C.blue });
  text(s, "Report operational errors", 58, 408, 510, 34, { size: 23, bold: true, color: C.text });
  text(s, "Detection delay, missed events, false warnings per monitored hour, and identity errors.", 58, 448, 510, 66, { size: 18, color: C.muted, valign: "top" });
  text(s, "CONNECT", 58, 544, 150, 24, { size: 12, bold: true, color: C.amber });
  text(s, "Uploaded video now\nWebcam and RTSP next\nRing and Nest later", 58, 568, 570, 70, { size: 19, bold: true, color: C.text, valign: "top" });
  image(s, "camera", 674, 196, 548, 374, "Generic unbranded outdoor camera beside a backyard pool", "cover", 22);
  text(s, "The overlay stays in our application. Camera APIs only change the source adapter.", 674, 592, 548, 42, { size: 18, bold: true, color: C.text, align: "center" });
  rect(s, 58, 648, 1164, 42, C.pale, 12, C.teal, 1);
  text(s, "A supervision aid. It does not replace watching children, pool fences, or lifeguards.", 78, 654, 1124, 28, { size: 17, bold: true, color: C.text, align: "center" });
  note(s,
    "Rohan, 40 seconds. The next experiment uses real pool footage recorded safely. Joanne will label each head above or below water and fine-tune the detector. Sam will split by recording session and report detection delay, missed events, false warnings per monitored hour, and identity errors. Sunny will connect the pipeline, Rohan will validate the head timer, and David will replace scripted frontend events with real results. Uploaded video remains the reliable demo source. Webcam and RTSP follow, then authorized Ring and Nest adapters. The overlay stays in our application. Life Vision remains a supervision aid and does not replace watching children, pool barriers, or lifeguards.",
    [
      "Next-step evaluation plan: docs/global/architecture.md",
      "Ring Developer Experience: https://developer.amazon.com/docs/ring/get-started.html",
      "Google Nest supported devices: https://developers.google.com/nest/device-access/supported-devices",
      "CDC prevention guidance: https://www.cdc.gov/drowning/prevention/index.html",
      "Camera concept image generated for this presentation. It does not show a branded product.",
    ]
  );
}

const candidatePath = path.join(TMP_DIR, "life-vision-team-refined-candidate.pptx");
await (await PresentationFile.exportPptx(pres)).save(candidatePath);

for (let i = 0; i < pres.slides.length; i++) {
  const slide = pres.slides.getItemAt(i);
  const preview = await pres.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, `preview-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await preview.arrayBuffer()));
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
  receiptPath: path.join(TMP_DIR, "life-vision-team-refined.pptx.validation.json"),
});

console.log(FINAL_PPTX);
