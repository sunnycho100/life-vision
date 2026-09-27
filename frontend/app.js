import {validQuad, homography, project, inside, boxAnchor, timeLabel, parseTracking, observationAt, ObservationCache} from './core.mjs';

const $ = id => document.getElementById(id);
const video = $('video'), stage = $('stage'), overlay = $('overlay'), context = overlay.getContext('2d');
const panels = ['source', 'calibrate', 'scan', 'twin'];
let phase = 'source', corners = [[.085, .08], [.95, .14], [.96, .91], [.22, .91]];
let source = null, reference, config = null, job = null, lastJob = null, twin = null, importedFrames = null;
let sourceGeneration = 0, frameTime = 0, currentDetections = [], threshold = .2, refreshPending = false, pollBusy = false;
let reviewAnalysis = null, reviewBusy = false, reviewSubmitting = false, incidents = [], selectedIncident = null, currentTracks = [], trackRenderKey = '';
let soundEnabled = false, alertAudio = null;
const seenEvents = new Set();
let detectionTracks = null;
const handles = [];
const cache = new ObservationCache(async (start, end) => {
  const response = await api('/api/jobs/' + job.id + '/observations?start=' + start + '&end=' + end);
  return response.observations;
});
const trackCache = new ObservationCache(async (start, end) => {
  const response = await api('/api/jobs/' + job.id + '/tracks?start=' + start + '&end=' + end);
  return response.frames.map(frame => ({...frame, status: ['ok', 'uncalibrated'].includes(frame.quality) ? 'analyzed' : frame.quality}));
});

async function api(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail));
  return body;
}
function message(text) {$('message').textContent = text; $('message').hidden = !text; if (text) setMenu(true);}
function setMenu(open) {document.body.classList.toggle('menu-open', open); $('menu-toggle').setAttribute('aria-expanded', open);}
$('menu-toggle').onclick = () => setMenu(!document.body.classList.contains('menu-open'));
function setPhase(next) {
  phase = next;
  panels.forEach((name, i) => {
    $(name + '-panel').hidden = name !== next;
    const step = document.querySelector('[data-step="' + name + '"]');
    step.classList.toggle('active', name === next);
    step.classList.toggle('done', i < panels.indexOf(next));
  });
  $('scan-effect').hidden = next !== 'scan';
  $('handles').hidden = next !== 'calibrate';
  $('play-button').disabled = !source;
  $('timeline').disabled = !source;
  if (next !== 'twin') stage.dataset.view = 'video';
  message('');
}
function videoRect() {
  const w = $('video-pane').clientWidth, h = $('video-pane').clientHeight;
  const ratio = video.videoWidth / video.videoHeight || 16 / 9;
  let width = w, height = w / ratio;
  if (height > h) {height = h; width = h * ratio;}
  return {x: (w - width) / 2, y: (h - height) / 2, width, height};
}
function mappingValid() {return config && video.currentTime >= config.start - 1e-6 && video.currentTime < config.end;}
function showPosition(position) {
  $('surface-position').textContent = position
    ? `Estimated position: X ${position.x.toFixed(1)} m · Z ${position.z.toFixed(1)} m from corner 1. ${position.calibration === 'user-measured' ? 'User-measured plane' : 'Approximate dimensions'}. Height is assumed at the water surface; marker size is symbolic.`
    : 'Pause and select a 3D marker to inspect its estimated position across the water.';
}
function clearDisplay() {currentDetections = []; currentTracks = []; detectionTracks = null; renderTrackStates(null); twin?.updatePeople([]); showPosition(null); $('person-count').textContent = '—'; $('pool-count').textContent = '—'; $('hud').hidden = true;}
// A lost box counts as "possibly under water" only if the person was last seen inside the mapped pool
// and was big enough to detect reliably. Far-away swimmers (box under 5% of frame height) flicker in and out
// of detection on their own, which raised 5 false alerts in the first 30 s of the wave pool clip.
// ponytail: fixed size floor tuned on one camera; lower it when the detector sees small people reliably.
const MIN_UNDER_HEIGHT = .05;
function possiblyUnder(track) {
  const b = track.bbox_xyxy_normalized;
  return !track.visible && mappingValid() && b[3] - b[1] >= MIN_UNDER_HEIGHT && inside(boxAnchor(b), config.corners);
}
// Came back after going out of sight inside the pool (same size and pool rules as possiblyUnder).
function cameBack(track) {
  const b = track.bbox_xyxy_normalized;
  return track.visible && track.back_after_s != null && mappingValid() && b[3] - b[1] >= MIN_UNDER_HEIGHT && inside(boxAnchor(b), config.corners);
}
const TRACK_COLORS = {safe: '#3ddc84', missing: '#9aa3ab', warning: '#ffb000', alarm: '#ff4d4d', back: '#6fd3ff'};
function draw() {
  const width = $('video-pane').clientWidth, height = $('video-pane').clientHeight, dpr = Math.min(devicePixelRatio, 2);
  if (overlay.width !== Math.round(width * dpr) || overlay.height !== Math.round(height * dpr)) {
    overlay.width = Math.round(width * dpr); overlay.height = Math.round(height * dpr);
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0); context.clearRect(0, 0, width, height);
  if (!source) return;
  const r = videoRect(), point = p => [r.x + p[0] * r.width, r.y + p[1] * r.height];
  if (phase === 'calibrate') { // outline only while placing corners; review shows just the person boxes
    context.beginPath(); corners.forEach((p, i) => {const xy = point(p); i ? context.lineTo(...xy) : context.moveTo(...xy);}); context.closePath();
    context.strokeStyle = '#abf2d8'; context.fillStyle = '#81f3cf09'; context.lineWidth = 1.3; context.stroke(); context.fill();
  }
  handles.forEach((handle, i) => {const p = point(corners[i]); handle.style.left = p[0] + 'px'; handle.style.top = p[1] + 'px';});
  if (detectionTracks) { // tracked people: stable IDs, lost people held at their last box
    for (const track of detectionTracks) {
      const b = track.bbox_xyxy_normalized, a = point(b.slice(0, 2)), z = point(b.slice(2));
      const under = possiblyUnder(track);
      const back = cameBack(track);
      const color = TRACK_COLORS[track.visible ? (back ? 'back' : 'safe') : under ? track.level : 'missing'] ?? TRACK_COLORS.missing;
      context.strokeStyle = color; context.fillStyle = color; context.lineWidth = track.visible ? 2 : 1.6;
      context.setLineDash(track.visible ? [] : [5, 4]);
      context.strokeRect(a[0], a[1], z[0] - a[0], z[1] - a[1]);
      context.setLineDash([]);
      if (!track.visible && (!under || track.level === 'safe')) continue; // dashed outline only; label once missing 1 s+ in the pool
      const text = track.visible
        ? track.person_id + (back ? ' · back after ' + track.back_after_s.toFixed(1) + 's' : '')
        : 'Person ' + track.person_id + ' · possibly under water ' + track.missing_s.toFixed(1) + 's';
      context.font = '600 11px system-ui, sans-serif';
      const w = context.measureText(text).width, y = Math.max(r.y + 14, a[1] - 4);
      context.fillStyle = '#000000b0'; context.fillRect(a[0] - 2, y - 11, w + 4, 14);
      context.fillStyle = color; context.fillText(text, a[0], y);
    }
    overlay.dataset.boxCount = detectionTracks.filter(tr => tr.visible).length;
    return;
  }
  for (const detection of currentDetections) {
    const b = detection.bbox_xyxy_normalized, a = point(b.slice(0, 2)), z = point(b.slice(2));
    const color = detection.review_severity === 'urgent' ? '#ff8d8d' : detection.review_severity === 'warn' ? '#f0c971' : '#9be6ce';
    context.strokeStyle = color; context.fillStyle = color; context.lineWidth = 1.4;
    context.strokeRect(a[0], a[1], z[0] - a[0], z[1] - a[1]);
    context.font = '10px monospace';
    const label = detection.track_id != null ? trackLabel(detection) : detection.confidence === null ? 'person' : Math.round(detection.confidence * 100) + '%';
    context.fillText(label, a[0], Math.max(r.y + 10, a[1] - 3));
  }
  overlay.dataset.boxCount = currentDetections.length;
}
for (let i = 0; i < 4; i++) {
  const button = document.createElement('button'); button.className = 'corner'; button.textContent = i + 1;
  button.setAttribute('aria-label', 'Water reference corner ' + (i + 1) + '; use arrow keys to adjust');
  let dragging = false;
  button.addEventListener('pointerdown', e => {dragging = true; button.setPointerCapture(e.pointerId); e.preventDefault();});
  button.addEventListener('pointermove', e => {
    if (!dragging) return;
    const box = $('video-pane').getBoundingClientRect(), r = videoRect();
    corners[i] = [Math.max(0, Math.min(1, (e.clientX - box.left - r.x) / r.width)), Math.max(0, Math.min(1, (e.clientY - box.top - r.y) / r.height))];
  });
  button.addEventListener('pointerup', () => dragging = false); button.addEventListener('pointercancel', () => dragging = false);
  button.addEventListener('keydown', e => {
    const offsets = {ArrowLeft: [-.005, 0], ArrowRight: [.005, 0], ArrowUp: [0, -.005], ArrowDown: [0, .005]};
    if (offsets[e.key]) {e.preventDefault(); corners[i] = corners[i].map((v, j) => Math.max(0, Math.min(1, v + offsets[e.key][j])));}
  });
  $('handles').append(button); handles.push(button);
}
function once(target, event, timeout = 20000) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {clearTimeout(timer); target.removeEventListener(event, done); target.removeEventListener('error', error);};
    const done = () => {cleanup(); resolve();};
    const error = () => {cleanup(); reject(Error('Browser could not decode this recording. Use an H.264 MP4 or WebM.'));};
    const timer = setTimeout(() => {cleanup(); reject(Error('Video loading timed out.'));}, timeout);
    target.addEventListener(event, done, {once: true}); target.addEventListener('error', error, {once: true});
  });
}
async function connect(info, isReference = false) {
  const generation = ++sourceGeneration;
  video.pause(); clearDisplay(); cache.clear(); resetReview(); importedFrames = null; job = null; config = null; source = null;
  $('resume-inference').hidden = true; $('zoom-warning').hidden = true; $('last-analysis').hidden = true; lastJob = null;
  const ready = once(video, 'loadeddata'); video.src = info.url; video.load(); await ready;
  if (generation !== sourceGeneration) return;
  if (!Number.isFinite(video.duration) || Math.abs(video.duration - info.duration) > .15 || video.videoWidth !== info.width || video.videoHeight !== info.height) throw Error('Browser and decoder disagree on video geometry or duration. Re-export as an upright H.264 MP4.');
  source = info; video.playbackRate = 1; $('speed-button').textContent = '1×';
  $('video-empty').hidden = true; $('timeline').max = video.duration; $('time-total').textContent = timeLabel(Math.ceil(video.duration));
  $('mapping-end').value = Math.min(info.mapping_end, video.duration); $('mapping-end').max = video.duration;
  $('analysis-start').value = 0; $('analysis-end').value = Math.min(info.duration, video.duration);
  $('viewer-title').textContent = isReference ? 'REFERENCE / WAVE POOL' : 'UPLOADED RECORDING';
  $('viewer-meta').textContent = info.width + ' × ' + info.height + ' · RECORDED';
  corners = isReference ? [[.085, .08], [.95, .14], [.96, .91], [.22, .91]] : [[.15, .2], [.85, .2], [.9, .85], [.1, .85]];
  $('measured-plane').checked = false;
  const target = Math.min(isReference ? 8 : .1, video.duration / 2);
  if (Math.abs(video.currentTime - target) > .001) {const seeked = once(video, 'seeked'); video.currentTime = target; await seeked;}
  frameTime = video.currentTime;
  setPhase('calibrate'); $('scan-button').textContent = 'Analyze recording'; $('tracking-status').textContent = 'Define the pool region, then analyze the recording.';
  const past = await api('/api/sources/' + info.id + '/jobs').catch(() => []);
  if (generation !== sourceGeneration) return;
  lastJob = past.find(j => j.state === 'completed') ?? null;
  $('last-analysis').hidden = !lastJob;
  if (lastJob) $('last-analysis').textContent = 'Review last analysis · ' + timeLabel(Math.round(lastJob.start)) + '–' + timeLabel(Math.round(lastJob.end));
}
$('reference-button').onclick = () => connect(reference, true).catch(e => message(e.message));
$('video-file').onchange = async e => {
  const file = e.target.files[0]; if (!file) return;
  $('reference-button').disabled = true; message('Uploading recording to this server…');
  try {
    const kind = file.name.toLowerCase().endsWith('.webm') ? 'video/webm' : 'video/mp4';
    const info = await api('/api/sources', {method: 'POST', headers: {'Content-Type': kind}, body: file});
    await connect(info);
  } catch (error) {message(error.message);}
  finally {e.target.value = ''; $('reference-button').disabled = !reference?.available;}
};
$('back-button').onclick = async () => {
  try {if (job && ['running', 'starting'].includes(job.state)) await api('/api/jobs/' + job.id + '/cancel', {method: 'POST'});}
  catch (e) {message(e.message); return;}
  sourceGeneration++; video.pause(); source = null; job = null; cache.clear(); resetReview(); clearDisplay(); setPhase('source');
};
function readConfig() {
  const width = Number($('pool-width').value), length = Number($('pool-length').value), depth = Number($('pool-depth').value), water = Number($('water-height').value), end = Number($('mapping-end').value);
  if (!validQuad(corners)) throw Error('Place four separated corners clockwise around a convex water region.');
  if (![width, length, depth, water, end].every(Number.isFinite) || width < 2 || width > 100 || length < 2 || length > 100 || depth < .3 || depth > 10 || water < .1 || water > depth) throw Error('Use a water height above zero and no higher than the wall; width/length 2–100 m and wall height 0.3–10 m.');
  if (end <= video.currentTime + .05 || end > video.duration + .1) throw Error('Mapping end must be after the current frame and within the recording.');
  if (source.sha256 === reference?.sha256 && source.mapping_end < source.duration && video.currentTime < source.mapping_end && end > source.mapping_end) throw Error('End this mapping before the reference camera view changes.');
  return {width, length, depth, water, measured: $('measured-plane').checked, corners: structuredClone(corners), start: video.currentTime, end, aspect: video.videoWidth / video.videoHeight};
}
$('scan-button').onclick = async () => {
  try {
    config = readConfig(); twin?.configure(config); showPosition(null); $('calibration-kind').textContent = config.measured ? 'User-measured plane' : 'Approximate'; video.pause();
    if (job || importedFrames) {setPhase('twin'); await setView('video'); await refresh(); return;}
    const start = Number($('analysis-start').value), end = Number($('analysis-end').value);
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start || end > source.duration + .001) throw Error('Set an analysis interval within the recording.');
    $('scan-button').disabled = true;
    job = await api('/api/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({source_id: source.id, start, end})});
    cache.clear(); resetReview(); setPhase(job.state === 'completed' ? 'twin' : 'scan'); showJob();
    if (job.state === 'completed') await openReview(); else await pollJob();
  } catch (error) {message(error.message);}
  finally {$('scan-button').disabled = false;}
};
function showJob() {
  if (!job) return;
  const running = ['starting', 'running'].includes(job.state), pct = Math.round(job.progress * 100);
  const details = job.state + ' · ' + pct + '% · ' + job.frames_analyzed + ' frames';
  $('scan-status').textContent = details; $('job-status').textContent = details;
  $('scan-progress-bar').style.width = pct + '%';
  $('scan-geometry').textContent = '✓ Pool outline saved';
  $('scan-model').textContent = 'RF-DETR Nano · ' + (job.runtime?.provider ?? job.backend);
  $('scan-detections').textContent = job.frames_analyzed + ' frames analyzed';
  $('delegate').textContent = job.runtime?.provider ?? job.backend;
  $('cancel-job').hidden = !running; $('cancel-scan').disabled = !running;
  $('new-analysis').hidden = running;
  $('region-size').textContent = config ? config.width + ' × ' + config.length + ' m' : 'Not mapped';
  if (job.error) message(job.error);
  showReviewAnalysis();
}
async function pollJob() {
  if (!job || pollBusy || !['starting', 'running'].includes(job.state)) return;
  pollBusy = true; const id = job.id, generation = sourceGeneration;
  try {
    const updated = await api('/api/jobs/' + id);
    if (generation !== sourceGeneration || job?.id !== id) return;
    const completedNow = updated.state === 'completed' && job.state !== 'completed';
    job = updated;
    if (completedNow) cache.clear(); // Final fetch includes rows committed after earlier polling.
    showJob(); await refresh();
    if (!['starting', 'running'].includes(job.state) && phase === 'scan') await openReview();
  } catch (error) {message(error.message);}
  finally {pollBusy = false;}
}
async function cancel() {
  if (!job) return;
  try {job = await api('/api/jobs/' + job.id + '/cancel', {method: 'POST'}); setPhase('twin'); await setView('video'); showJob();}
  catch (error) {message(error.message);}
}
$('cancel-scan').onclick = cancel; $('cancel-job').onclick = cancel;
$('new-analysis').onclick = () => {
  sourceGeneration++; video.pause(); video.currentTime = config?.start ?? 0; job = null; importedFrames = null; cache.clear(); resetReview(); clearDisplay();
  setPhase('calibrate'); $('scan-button').textContent = 'Analyze recording';
};
async function openReview() {
  setPhase('twin'); await setView('video'); showJob();
  if (!job || importedFrames) return;
  const target = video.currentTime < job.start || video.currentTime >= job.end ? job.start : video.currentTime;
  await cache.ensure(target, job.state === 'completed');
  const frames = cache.frames(target);
  if (video.paused && !observationAt(frames, target)) {
    const next = frames.find(f => f.media_time >= target && f.media_time - target <= .2);
    if (next) video.currentTime = next.media_time + .001;
  }
  await refresh();
}
$('last-analysis').onclick = async () => {
  try {config = readConfig(); twin?.configure(config); video.pause(); job = lastJob; cache.clear(); resetReview(); await openReview();}
  catch (error) {message(error.message);}
};
$('review-progress').onclick = () => openReview().catch(e => message(e.message));
async function refresh() {
  if (!source || !job || importedFrames || refreshPending) return;
  refreshPending = true;
  try {
    if (reviewAnalysis?.state === 'completed') await trackCache.ensure(video.currentTime, true);
    else await cache.ensure(video.currentTime, job.state === 'completed');
  }
  catch (error) {message(error.message);}
  finally {refreshPending = false;}
}
function updateDisplay() {
  if (!source || video.seeking) {clearDisplay(); return;}
  const time = video.paused ? video.currentTime : frameTime;
  const tracked = !importedFrames && reviewAnalysis?.state === 'completed';
  const frames = importedFrames ?? (tracked ? trackCache.frames(time) : cache.frames(time));
  const observation = observationAt(frames, time);
  currentTracks = tracked && observation ? observation.tracks : [];
  currentDetections = observation ? (tracked ? currentTracks.filter(t => ['visible', 'outside'].includes(t.state) && Math.abs(t.last_seen - observation.media_time) < .001) : observation.detections).filter(d => d.confidence === null || d.confidence >= threshold) : [];
  renderTrackStates(tracked ? observation : null);
  detectionTracks = !tracked ? observation?.tracks ?? null : null;
  if (detectionTracks) {
    detectionTracks = detectionTracks.filter(t => !t.visible || t.confidence >= threshold);
    currentDetections = detectionTracks.filter(t => t.visible);
  }
  $('person-count').textContent = observation ? currentDetections.length : '—';
  $('inference-time').textContent = observation?.inference_ms ? Math.round(observation.inference_ms) : '—';
  $('pool-count').textContent = observation && mappingValid() ? currentDetections.filter(d => inside(boxAnchor(d.bbox_xyxy_normalized), config.corners)).length : '—';
  $('hud').hidden = !observation;
  $('hud-people').textContent = currentDetections.length;
  const alerts = (detectionTracks ?? []).filter(tr => possiblyUnder(tr) && ['warning', 'alarm'].includes(tr.level));
  $('hud-alert').hidden = !alerts.length;
  $('hud-alert').className = alerts.some(tr => tr.level === 'alarm') ? 'alarm' : 'warning';
  $('hud-alert').textContent = alerts.map(tr => 'Person ' + tr.person_id + ' possibly under water ' + Math.floor(tr.missing_s) + 's').join(' · ');
  const returns = (detectionTracks ?? []).filter(cameBack);
  $('hud-back').hidden = !returns.length;
  $('hud-back').textContent = returns.map(tr => 'Person ' + tr.person_id + ' back after ' + tr.back_after_s.toFixed(1) + 's').join(' · ');
  if (phase === 'twin' || phase === 'scan') $('tracking-status').textContent = observation
    ? currentDetections.length + ' person boxes · ' + (importedFrames ? 'imported observations' : tracked ? 'persistent track replay' : 'RF-DETR Nano') + (mappingValid() ? '' : ' · pool outline needs calibration')
    : 'Analysis unavailable at this time';
  if (twin && stage.dataset.view !== 'video') twin.updatePeople(mappingValid() ? currentDetections : []);
  $('zoom-warning').hidden = stage.dataset.view === 'video' || !config || mappingValid();
}
async function setView(view) {
  showPosition(null);
  if (view !== 'video') {
    try {
      if (!config) throw Error('Map the pool before opening the optional 3D view.');
      if (!twin) {const {PoolTwin} = await import('./twin.js'); twin = new PoolTwin($('three-canvas'), video, showPosition);}
      twin.configure(config);
    } catch (error) {message('3D view unavailable: ' + error.message); view = 'video';}
  }
  stage.dataset.view = view;
  $('zoom-warning').hidden = view === 'video' || !config || mappingValid();
  document.querySelectorAll('button[data-view]').forEach(button => button.classList.toggle('selected', button.dataset.view === view));
}
document.querySelectorAll('button[data-view]').forEach(button => button.onclick = () => setView(button.dataset.view));
function recalibrate() {video.pause(); setPhase('calibrate'); $('mapping-end').value = video.currentTime >= source.mapping_end ? source.duration : source.mapping_end; $('scan-button').textContent = 'Save pool mapping';}
$('recalibrate-button').onclick = recalibrate; $('zoom-recalibrate').onclick = recalibrate;
$('reset-camera').onclick = () => twin?.resetCamera();
document.querySelectorAll('[data-preset]').forEach(button => button.onclick = () => twin?.preset(button.dataset.preset));
$('play-button').onclick = () => video.paused ? video.play().catch(e => message(e.message)) : video.pause();
$('timeline').oninput = e => {video.currentTime = Number(e.target.value);};
$('speed-button').onclick = () => {const rates = [1, .5, .25, 2]; video.playbackRate = rates[(rates.indexOf(video.playbackRate) + 1) % rates.length]; $('speed-button').textContent = video.playbackRate + '×';};
$('mute-button').onclick = () => video.muted = !video.muted;
$('confidence').oninput = e => {threshold = Number(e.target.value); $('threshold-value').textContent = threshold.toFixed(2);};
video.addEventListener('seeking', clearDisplay);
video.addEventListener('seeked', () => {frameTime = video.currentTime; refresh();});
video.addEventListener('error', () => message('Video playback failed. Try an H.264 MP4 or WebM recording.'));
if (video.requestVideoFrameCallback) {
  const presented = (_, meta) => {frameTime = meta.mediaTime; video.requestVideoFrameCallback(presented);};
  video.requestVideoFrameCallback(presented);
} else video.addEventListener('timeupdate', () => frameTime = video.currentTime);
$('tracking-button').onclick = () => $('tracking-file').click();
$('calibration-import').onclick = () => $('tracking-file').click();
$('tracking-file').onchange = async e => {
  const file = e.target.files[0]; if (!file || !source) return;
  try {
    if (file.size > 20 * 1024 * 1024) throw Error('Result imports are limited to 20 MiB.');
    importedFrames = parseTracking(JSON.parse(await file.text()), source);
    if (!config) config = readConfig();
    $('resume-inference').hidden = !job;
    setPhase('twin'); await setView('video'); $('delegate').textContent = 'Imported box results';
  } catch (error) {message(error.message);}
  e.target.value = '';
};
$('resume-inference').onclick = () => {importedFrames = null; $('resume-inference').hidden = true; showJob(); refresh();};
$('export-button').onclick = () => {
  const data = {schema: 'poolside-setup/2', source_sha256: source.sha256, model: 'RF-DETR Nano', job_id: job?.id ?? null, confidence: threshold, membership_anchor: 'bottom-center', mapping: config};
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'poolside-setup.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
function trackLabel(track) {
  const state = track.state === 'missing' ? 'lost' : track.state === 'outside' ? 'visible outside' : track.state === 'visible' ? 'visible' : 'unknown';
  return '#' + track.track_id + ' · ' + state + (track.missing_seconds > 0 ? ' · ' + track.missing_seconds.toFixed(1) + 's missing' : '') + (track.reason ? ' · ' + track.reason.replaceAll('_', ' ') : '');
}
function renderTrackStates(frame) {
  const key = frame ? frame.media_time + ':' + frame.quality : 'unavailable';
  if (trackRenderKey === key) return;
  trackRenderKey = key;
  const tracks = frame?.tracks ?? [], visible = tracks.filter(t => ['visible', 'outside'].includes(t.state)).length;
  $('track-states').textContent = frame ? visible + ' visible · ' + tracks.filter(t => t.state === 'missing').length + ' lost · quality: ' + frame.quality + '. Lost positions are not displayed as current detections.' : 'Track states unknown: analysis unavailable at this time.';
  $('track-details').replaceChildren(...tracks.map(track => {
    const chip = document.createElement('span'); chip.className = 'track-chip'; chip.dataset.severity = track.review_severity ?? 'none'; chip.textContent = trackLabel(track); return chip;
  }));
}
function resetReview() {
  reviewAnalysis = null; incidents = []; selectedIncident = null; seenEvents.clear(); trackCache.clear(); trackRenderKey = ''; currentTracks = [];
  $('event-review').hidden = true; $('incident-detail').hidden = true; $('incident-list').replaceChildren();
  $('incident-video').pause(); $('incident-video').removeAttribute('src'); $('incident-video').load();
  showReviewAnalysis();
}
function reviewRange() {return config && job ? {start: Math.max(config.start, job.start), end: Math.min(config.end, job.end)} : null;}
function showReviewAnalysis() {
  const state = reviewAnalysis?.state ?? 'not_started', range = reviewRange();
  const running = ['queued', 'running'].includes(state);
  $('build-review').disabled = reviewSubmitting || !job || job.state !== 'completed' || !!importedFrames || state !== 'not_started' || !range || range.end <= range.start;
  $('build-review').textContent = reviewSubmitting ? 'Starting review analysis…' : 'Build review events';
  $('review-range').textContent = reviewAnalysis?.config ? 'Saved review interval: ' + timeLabel(reviewAnalysis.config.start) + '–' + timeLabel(reviewAnalysis.config.end) + '. Configuration is fixed for this job; run a new analysis to change it.' : range && range.end > range.start ? 'Review interval: ' + timeLabel(range.start) + '–' + timeLabel(range.end) + ' (pool mapping ∩ analyzed footage). Missing visibility: review after 5s; urgent after 12s.' : 'The pool mapping must overlap the analyzed interval.';
  $('review-analysis-status').textContent = state === 'not_started' ? importedFrames ? 'Return to server results to build review events.' : job?.state === 'completed' ? 'Ready to build tracks and visibility-loss review candidates.' : 'Complete person analysis to build persistent tracks and visibility-loss review events.' : 'Review analysis: ' + state + (Number.isFinite(reviewAnalysis.progress) ? ' · ' + Math.round(reviewAnalysis.progress * 100) + '%' : '') + (reviewAnalysis.error ? ' · ' + reviewAnalysis.error : '');
  $('review-analysis-progress').hidden = !running;
  if (Number.isFinite(reviewAnalysis?.progress)) $('review-analysis-progress').value = reviewAnalysis.progress;
  else $('review-analysis-progress').removeAttribute('value');
  $('event-review').hidden = state === 'not_started';
}
$('build-review').onclick = async () => {
  const range = reviewRange(); if (!range || !job || reviewSubmitting) return;
  const id = job.id, generation = sourceGeneration;
  reviewSubmitting = true; showReviewAnalysis();
  try {
    const result = await api('/api/jobs/' + id + '/review-analysis', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({corners: config.corners, ...range, low_threshold: .1, high_threshold: .25, new_track_threshold: .25, warn_seconds: 5, urgent_seconds: 12})});
    if (generation !== sourceGeneration || job?.id !== id) return;
    reviewAnalysis = result; trackCache.clear(); await pollReview(); await refresh();
  } catch (error) {message(error.message);}
  finally {reviewSubmitting = false; showReviewAnalysis();}
};
async function pollReview() {
  if (!job || job.state !== 'completed' || reviewBusy) return;
  reviewBusy = true; const id = job.id, generation = sourceGeneration;
  try {
    const result = await api('/api/jobs/' + id + '/review-analysis');
    if (generation !== sourceGeneration || job?.id !== id) return;
    const completedNow = result.state === 'completed' && reviewAnalysis?.state !== 'completed';
    reviewAnalysis = result; showReviewAnalysis();
    if (completedNow) {trackCache.clear(); clearDisplay(); await refresh();}
    if (result.state === 'not_started') return;
    const response = await api('/api/jobs/' + id + '/incidents');
    if (generation !== sourceGeneration || job?.id !== id) return;
    let newlyPending = false;
    for (const incident of response.incidents) {
      if (!seenEvents.has(incident.id) && (incident.review?.decision ?? 'needs_review') === 'needs_review') newlyPending = true;
      seenEvents.add(incident.id);
    }
    incidents = response.incidents;
    if (newlyPending && soundEnabled) playAlert();
    renderIncidents();
    if (selectedIncident) renderIncidentDetail(false);
  } catch (error) {$('review-analysis-status').textContent = 'Review service error: ' + error.message;}
  finally {reviewBusy = false;}
}
function renderIncidents() {
  const pending = incidents.filter(i => (i.review?.decision ?? 'needs_review') === 'needs_review').length;
  $('incident-count').textContent = pending + ' need review · ' + incidents.length + ' total events';
  const filter = $('incident-filter').value;
  const filtered = incidents.filter(i => filter === 'all' || (i.review?.decision ?? 'needs_review') === (filter === 'pending' ? 'needs_review' : filter));
  $('incident-list').replaceChildren(...filtered.map(incident => {
    const button = document.createElement('button'); button.className = 'incident-card'; button.dataset.incidentId = incident.id; button.dataset.severity = incident.severity === 'urgent' ? 'urgent' : 'warn';
    button.setAttribute('aria-pressed', String(incident.id === selectedIncident));
    button.textContent = '#' + incident.track_id + ' · ' + timeLabel(incident.trigger_time) + ' · ' + incident.severity + ' concern';
    const detail = document.createElement('small'); detail.textContent = (incident.reason ?? incident.kind).replaceAll('_', ' ') + ' · ' + (incident.review?.decision ?? 'needs_review').replaceAll('_', ' '); button.append(detail);
    button.onclick = () => {selectedIncident = incident.id; renderIncidents(); renderIncidentDetail(true);}; return button;
  }));
  if (!filtered.length) {const empty = document.createElement('p'); empty.textContent = incidents.length ? 'No events match this filter.' : 'No review candidates available. This does not establish swimmer safety.'; $('incident-list').append(empty);}
}
function renderIncidentDetail(replaceNote) {
  const incident = incidents.find(i => i.id === selectedIncident); $('incident-detail').hidden = !incident; if (!incident) return;
  $('incident-title').textContent = 'Track #' + incident.track_id + ' · ' + incident.severity + ' review concern';
  $('incident-reason').textContent = (incident.reason ?? incident.kind).replaceAll('_', ' ') + ' · started ' + timeLabel(incident.start_time) + ' · triggered ' + timeLabel(incident.trigger_time) + ' · evidence: ' + (incident.evidence ?? 'visibility-only') + (incident.resolution_reason ? ' · ' + incident.resolution_reason.replaceAll('_', ' ') : '');
  $('incident-decision').textContent = 'Review: ' + (incident.review?.decision ?? 'needs_review').replaceAll('_', ' ') + (incident.review?.updated_at ? ' · saved ' + new Date(incident.review.updated_at * 1000).toLocaleString() : '');
  if (replaceNote) {$('incident-note').value = incident.review?.note ?? ''; $('review-save-status').textContent = '';}
  const ready = incident.clip?.state === 'ready', clipUrl = '/api/jobs/' + job.id + '/incidents/' + encodeURIComponent(incident.id) + '/clip';
  $('download-clip').hidden = !ready; $('incident-video').hidden = !ready;
  $('clip-status').textContent = 'Saved clip: ' + (incident.clip?.state ?? 'pending') + (incident.clip?.partial ? ' · partial context at recording boundary' : '') + (incident.clip?.error ? ' · ' + incident.clip.error : '');
  if (ready) {
    $('download-clip').href = clipUrl; $('download-clip').download = 'review-' + incident.id + '.mp4';
    if ($('incident-video').getAttribute('src') !== clipUrl) {$('incident-video').src = clipUrl; $('incident-video').load();}
  } else {$('incident-video').pause(); $('incident-video').removeAttribute('src');}
}
$('incident-filter').onchange = renderIncidents;
$('seek-incident').onclick = async () => {
  const incident = incidents.find(i => i.id === selectedIncident); if (!incident) return;
  video.pause(); await setView('video'); video.currentTime = Math.max(0, incident.clip?.start ?? incident.start_time); $('stage').scrollIntoView({behavior: 'smooth', block: 'center'});
};
document.querySelectorAll('[data-decision]').forEach(button => button.onclick = async () => {
  if (!selectedIncident || !job) return;
  const id = selectedIncident, jobId = job.id, generation = sourceGeneration;
  const buttons = [...document.querySelectorAll('[data-decision]')]; buttons.forEach(b => b.disabled = true);
  try {
    const result = await api('/api/jobs/' + jobId + '/incidents/' + encodeURIComponent(id), {method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({decision: button.dataset.decision, note: $('incident-note').value})});
    if (generation !== sourceGeneration || job?.id !== jobId) return;
    incidents = incidents.map(i => i.id === id ? result : i); renderIncidents(); if (selectedIncident === id) {renderIncidentDetail(false); $('review-save-status').textContent = 'Review saved.';}
  } catch (error) {$('review-save-status').textContent = 'Could not save review: ' + error.message;}
  finally {buttons.forEach(b => b.disabled = false);}
});
function playAlert() {
  if (!alertAudio || alertAudio.state !== 'running') return;
  const oscillator = alertAudio.createOscillator(), gain = alertAudio.createGain();
  oscillator.connect(gain); gain.connect(alertAudio.destination); oscillator.frequency.value = 660;
  gain.gain.setValueAtTime(.06, alertAudio.currentTime); gain.gain.exponentialRampToValueAtTime(.001, alertAudio.currentTime + .3);
  oscillator.start(); oscillator.stop(alertAudio.currentTime + .3);
}
$('enable-alerts').onclick = async () => {
  try {
    if (!soundEnabled) {alertAudio ??= new (window.AudioContext || window.webkitAudioContext)(); await alertAudio.resume(); incidents.forEach(i => seenEvents.add(i.id));}
    soundEnabled = !soundEnabled; $('enable-alerts').textContent = soundEnabled ? 'Disable sound' : 'Enable sound'; $('enable-alerts').setAttribute('aria-pressed', String(soundEnabled));
  } catch (error) {message('Sound unavailable: ' + error.message);}
};
function animate(now) {
  updateDisplay(); draw();
  if (twin && stage.dataset.view !== 'video') twin.render(now);
  $('time-current').textContent = timeLabel(video.currentTime); $('timeline').value = video.currentTime;
  $('play-button').textContent = video.paused ? '▶' : 'Ⅱ';
  requestAnimationFrame(animate);
}
stage.dataset.view = 'video'; requestAnimationFrame(animate);
setInterval(() => {pollJob(); refresh();}, 500);
setInterval(pollReview, 2000);
try {
  reference = await api('/api/reference'); $('reference-button').disabled = !reference.available;
  const health = await api('/api/health');
  if (!health.model_manifest_available) message(health.message);
} catch (error) {message('Server unavailable: ' + error.message);}

