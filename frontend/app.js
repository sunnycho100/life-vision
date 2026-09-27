const video = document.querySelector("#video");
const videoUpload = document.querySelector("#videoUpload");
const placeholder = document.querySelector("#placeholder");
const overlay = document.querySelector("#overlayCanvas");
const ctx = overlay.getContext("2d");
const matrixCanvas = document.querySelector("#matrixCanvas");
const matrixCtx = matrixCanvas.getContext("2d");
const activateButton = document.querySelector("#activateButton");
const activateLabel = document.querySelector("#activateLabel");
const resetButton = document.querySelector("#resetButton");
const videoShell = document.querySelector("#videoShell");
const personList = document.querySelector("#personList");
const emptyState = document.querySelector("#emptyState");
const eventLog = document.querySelector("#eventLog");
const clearEvents = document.querySelector("#clearEvents");
const liveBadge = document.querySelector("#liveBadge");
const feedState = document.querySelector("#feedState");
const systemStatus = document.querySelector("#systemStatus");
const criticalBanner = document.querySelector("#criticalBanner");
const peopleCount = document.querySelector("#peopleCount");
const riskCount = document.querySelector("#riskCount");
const sourceTag = document.querySelector("#sourceTag");

let running = false;
let startedAt = 0;
let animationFrame = 0;
let selectedAlert = "red";
let emittedEvents = new Set();
let objectUrl = null;
let alarmPlayed = false;

const colors = {
  green: "#41f28d",
  yellow: "#ffd447",
  red: "#ff4e5c",
};

const demoPeople = [
  {
    id: 1,
    name: "PERSON 01",
    box: [0.17, 0.28, 0.21, 0.49],
  },
  {
    id: 2,
    name: "PERSON 02",
    box: [0.61, 0.33, 0.18, 0.43],
  },
];

function stageFor(elapsedSeconds) {
  const phase = elapsedSeconds % 16;

  if (phase < 4.5) {
    return {
      phase,
      people: demoPeople.map((person, index) => ({
        ...person,
        state: "green",
        risk: 0.08 + index * 0.04,
        reason: "Normal movement · forward progress detected",
        timer: 0,
      })),
    };
  }

  if (phase < 9) {
    const warningTimer = phase - 4.5;
    return {
      phase,
      people: [
        { ...demoPeople[0], state: "green", risk: 0.11, reason: "Normal movement · forward progress detected", timer: 0 },
        {
          ...demoPeople[1],
          state: "yellow",
          risk: Math.min(0.74, 0.43 + warningTimer * 0.065),
          reason: "Repeated upper-body motion · low forward progress",
          timer: warningTimer,
        },
      ],
    };
  }

  const criticalTimer = phase - 9;
  return {
    phase,
    people: [
      { ...demoPeople[0], state: "green", risk: 0.1, reason: "Normal movement · forward progress detected", timer: 0 },
      {
        ...demoPeople[1],
        state: "red",
        risk: Math.min(0.98, 0.83 + criticalTimer * 0.025),
        reason: "Distress pattern persisted · intervention recommended",
        timer: criticalTimer + 4.5,
      },
    ],
  };
}

function resizeCanvas() {
  const rect = videoShell.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  overlay.width = Math.round(rect.width * ratio);
  overlay.height = Math.round(rect.height * ratio);
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  matrixCanvas.width = Math.round(window.innerWidth * ratio);
  matrixCanvas.height = Math.round(window.innerHeight * ratio);
  matrixCanvas.style.width = `${window.innerWidth}px`;
  matrixCanvas.style.height = `${window.innerHeight}px`;
  matrixCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawCornerBox(x, y, width, height, state, label, risk, timer) {
  const color = colors[state];
  const corner = Math.min(28, width * 0.18);
  ctx.lineWidth = state === "red" ? 3 : 2;
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = state === "red" ? 15 : 7;
  ctx.beginPath();
  ctx.moveTo(x + corner, y); ctx.lineTo(x, y); ctx.lineTo(x, y + corner);
  ctx.moveTo(x + width - corner, y); ctx.lineTo(x + width, y); ctx.lineTo(x + width, y + corner);
  ctx.moveTo(x, y + height - corner); ctx.lineTo(x, y + height); ctx.lineTo(x + corner, y + height);
  ctx.moveTo(x + width - corner, y + height); ctx.lineTo(x + width, y + height); ctx.lineTo(x + width, y + height - corner);
  ctx.stroke();
  ctx.shadowBlur = 0;

  const title = `${label}  ·  ${Math.round(risk * 100)}%`;
  const timerText = state === "green" ? "TRACKING" : `${timer.toFixed(1)}s`;
  ctx.font = "700 11px ui-monospace, monospace";
  const titleWidth = ctx.measureText(title).width;
  const timerWidth = ctx.measureText(timerText).width;
  ctx.fillStyle = color;
  ctx.fillRect(x, Math.max(0, y - 24), titleWidth + 16, 22);
  ctx.fillStyle = "#04120d";
  ctx.fillText(title, x + 8, Math.max(15, y - 9));

  ctx.fillStyle = "rgba(4, 18, 13, .85)";
  ctx.fillRect(x + width - timerWidth - 16, y + height + 2, timerWidth + 16, 21);
  ctx.fillStyle = color;
  ctx.fillText(timerText, x + width - timerWidth - 8, y + height + 17);
}

function drawOverlay(state) {
  const rect = videoShell.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  state.people.forEach((person, index) => {
    const motion = Math.sin(performance.now() / 600 + index) * 0.008;
    const [bx, by, bw, bh] = person.box;
    drawCornerBox(
      (bx + motion) * rect.width,
      (by + motion * 0.25) * rect.height,
      bw * rect.width,
      bh * rect.height,
      person.state,
      person.name,
      person.risk,
      person.timer,
    );
  });
}

function personCard(person) {
  const stateLabel = person.state === "green" ? "NORMAL" : person.state === "yellow" ? "WARNING" : "CRITICAL";
  const timerLabel = person.state === "green" ? "stable" : `${person.timer.toFixed(1)}s active`;
  return `
    <article class="person-card ${person.state}">
      <div class="person-top">
        <span class="person-name">${person.name}</span>
        <span class="state-chip">${stateLabel}</span>
      </div>
      <p class="person-reason">${person.reason}</p>
      <div class="risk-row">
        <div class="risk-bar"><span style="--risk: ${person.risk * 100}%"></span></div>
        <span>${Math.round(person.risk * 100)}% · ${timerLabel}</span>
      </div>
    </article>`;
}

function renderPeople(state) {
  emptyState.hidden = true;
  personList.innerHTML = state.people.map(personCard).join("");
  peopleCount.textContent = state.people.length;
  riskCount.textContent = state.people.filter((person) => person.state !== "green").length;
  const critical = state.people.some((person) => person.state === "red");
  const warning = state.people.some((person) => person.state === "yellow");
  const shouldAlert = critical || (warning && selectedAlert === "yellow");
  criticalBanner.classList.toggle("visible", critical);

  if (shouldAlert && !alarmPlayed && selectedAlert !== "screen") {
    playAlarm();
    alarmPlayed = true;
  }
  if (!shouldAlert) alarmPlayed = false;
}

function logEvent(key, message, severity = "green") {
  if (emittedEvents.has(key)) return;
  emittedEvents.add(key);
  if (eventLog.querySelector(".muted-event")) eventLog.innerHTML = "";
  const item = document.createElement("li");
  item.className = severity;
  item.innerHTML = `<time>${new Date().toLocaleTimeString([], { hour12: false })}</time><span>${message}</span>`;
  eventLog.prepend(item);
}

function updateEvents(phase) {
  if (phase < 0.25) logEvent("start", "Monitoring activated · two people acquired");
  if (phase >= 4.5) logEvent("warning", "Person 02 entered warning · repeated motion with low progress", "yellow");
  if (phase >= 9) logEvent("critical", "Person 02 entered critical state · persistence threshold reached", "red");
}

function tick(now) {
  if (!running) return;
  const elapsed = (now - startedAt) / 1000;
  const state = stageFor(elapsed);
  drawOverlay(state);
  renderPeople(state);
  updateEvents(state.phase);

  if (state.phase < 0.2 && elapsed > 15) emittedEvents = new Set();
  animationFrame = requestAnimationFrame(tick);
}

function playAlarm() {
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audio = new AudioContext();
    [0, 0.16, 0.32].forEach((delay) => {
      const oscillator = audio.createOscillator();
      const gain = audio.createGain();
      oscillator.type = "square";
      oscillator.frequency.value = 740;
      gain.gain.setValueAtTime(0.0001, audio.currentTime + delay);
      gain.gain.exponentialRampToValueAtTime(0.09, audio.currentTime + delay + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + delay + 0.11);
      oscillator.connect(gain).connect(audio.destination);
      oscillator.start(audio.currentTime + delay);
      oscillator.stop(audio.currentTime + delay + 0.12);
    });
  } catch (error) {
    console.info("Audio alert unavailable", error);
  }
}

function runMatrixIntro() {
  matrixCanvas.classList.add("active");
  const glyphs = "01人水危険SAFEWATCH";
  const size = 18;
  const columns = Math.ceil(window.innerWidth / size);
  const drops = Array.from({ length: columns }, () => Math.random() * -30);
  const start = performance.now();

  function frame(now) {
    matrixCtx.fillStyle = "rgba(3, 14, 10, .18)";
    matrixCtx.fillRect(0, 0, window.innerWidth, window.innerHeight);
    matrixCtx.fillStyle = "#41f28d";
    matrixCtx.font = `${size}px ui-monospace, monospace`;
    drops.forEach((drop, index) => {
      const character = glyphs[Math.floor(Math.random() * glyphs.length)];
      matrixCtx.fillText(character, index * size, drop * size);
      drops[index] = drop * size > window.innerHeight && Math.random() > 0.96 ? 0 : drop + 1;
    });

    if (now - start < 1100) requestAnimationFrame(frame);
    else {
      matrixCanvas.classList.remove("active");
      setTimeout(() => matrixCtx.clearRect(0, 0, window.innerWidth, window.innerHeight), 250);
    }
  }
  requestAnimationFrame(frame);
}

function startMonitoring() {
  running = true;
  startedAt = performance.now();
  emittedEvents = new Set();
  alarmPlayed = false;
  videoShell.classList.add("monitoring");
  liveBadge.classList.add("active");
  liveBadge.textContent = "ANALYZING";
  feedState.textContent = "LIVE";
  systemStatus.textContent = "MONITORING ACTIVE";
  activateLabel.textContent = "Pause monitoring";
  activateButton.querySelector(".button-icon").textContent = "Ⅱ";
  if (video.src) video.play().catch(() => {});
  runMatrixIntro();
  cancelAnimationFrame(animationFrame);
  animationFrame = requestAnimationFrame(tick);
}

function pauseMonitoring() {
  running = false;
  cancelAnimationFrame(animationFrame);
  video.pause();
  videoShell.classList.remove("monitoring");
  liveBadge.classList.remove("active");
  liveBadge.textContent = "PAUSED";
  feedState.textContent = "PAUSED";
  systemStatus.textContent = "MONITORING PAUSED";
  activateLabel.textContent = "Resume monitoring";
  activateButton.querySelector(".button-icon").textContent = "▶";
}

function resetDemo() {
  running = false;
  cancelAnimationFrame(animationFrame);
  video.pause();
  video.currentTime = 0;
  emittedEvents = new Set();
  alarmPlayed = false;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  personList.innerHTML = "";
  emptyState.hidden = false;
  peopleCount.textContent = "0";
  riskCount.textContent = "0";
  criticalBanner.classList.remove("visible");
  videoShell.classList.remove("monitoring");
  liveBadge.classList.remove("active");
  liveBadge.textContent = "OFFLINE";
  feedState.textContent = "STANDBY";
  systemStatus.textContent = "SYSTEM READY";
  activateLabel.textContent = "Activate monitoring";
  activateButton.querySelector(".button-icon").textContent = "▶";
  eventLog.innerHTML = `<li class="muted-event"><time>—</time><span>No events yet. Activate monitoring to start the demonstration.</span></li>`;
}

videoUpload.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (!file) return;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  video.src = objectUrl;
  video.style.display = "block";
  placeholder.style.display = "none";
  sourceTag.textContent = `${file.name.toUpperCase()} · LOCAL VIDEO`;
  resetDemo();
  logEvent("upload", `Video ready · ${file.name}`);
});

activateButton.addEventListener("click", () => running ? pauseMonitoring() : startMonitoring());
resetButton.addEventListener("click", resetDemo);
clearEvents.addEventListener("click", () => {
  eventLog.innerHTML = `<li class="muted-event"><time>—</time><span>Event history cleared.</span></li>`;
  emittedEvents = new Set();
});

document.querySelectorAll("[data-alert]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-alert]").forEach((item) => item.classList.remove("selected"));
    button.classList.add("selected");
    selectedAlert = button.dataset.alert;
    logEvent(`preference-${selectedAlert}`, `Alert preference changed · ${button.textContent}`);
  });
});

window.addEventListener("resize", resizeCanvas);
setInterval(() => {
  document.querySelector("#clock").textContent = new Date().toLocaleTimeString([], { hour12: false });
}, 1000);

resizeCanvas();
resetDemo();
