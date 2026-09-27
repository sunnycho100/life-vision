// Geometry helpers shared by calibration and optional 3D markers.
export const boxAnchor = b => [(b[0] + b[2]) / 2, b[3]];
export const timeLabel = t => `${Math.floor((t || 0) / 60)}:${String(Math.floor((t || 0) % 60)).padStart(2, '0')}`;

export function validDetection(d, legacy = false) {
  const b = d.bbox_xyxy_normalized;
  if (!Array.isArray(b) || b.length !== 4 || !b.every(v => Number.isFinite(v) && v >= 0 && v <= 1) || b[2] <= b[0] || b[3] <= b[1]) throw Error('Invalid normalized person box.');
  if (!(legacy && d.confidence === null) && (!Number.isFinite(d.confidence) || d.confidence < 0 || d.confidence > 1)) throw Error('Invalid detection confidence.');
  if (d.class_name !== 'person') throw Error('Only person observations are supported.');
  return {bbox_xyxy_normalized: b, confidence: d.confidence, class_name: 'person'};
}

export function parseTracking(data, source) {
  if (data.schema_version !== 1 || !data.video || !source.sha256 || data.video.sha256 !== source.sha256) throw Error('Results must match the video SHA-256 fingerprint.');
  if (data.video.width !== source.width || data.video.height !== source.height || !Number.isFinite(data.video.duration_sec) || Math.abs(data.video.duration_sec - source.duration) > .1) throw Error('Video dimensions or duration do not match.');
  if (!Array.isArray(data.frames) || !data.frames.length) throw Error('Expected timestamped frames.');
  let previous = -1;
  return data.frames.map(frame => {
    if (!Number.isFinite(frame.t_sec) || frame.t_sec <= previous || frame.t_sec < 0 || frame.t_sec > source.duration || !Array.isArray(frame.people)) throw Error('Invalid frame timestamp or person array.');
    previous = frame.t_sec;
    // Legacy identity and keypoint fields deliberately have no role in person-only replay.
    const detections = frame.people.map(p => validDetection({bbox_xyxy_normalized: p.bbox_xyxy, confidence: p.confidence ?? null, class_name: 'person'}, true));
    return {media_time: frame.t_sec, status: 'analyzed', detections};
  });
}

export function observationAt(frames, time) {
  let low = 0, high = frames.length - 1, result = null;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (frames[middle].media_time <= time + 1e-6) {result = frames[middle]; low = middle + 1;} else high = middle - 1;
  }
  return result && time - result.media_time <= .3 && result.status === 'analyzed' ? result : null;
}

export class ObservationCache {
  constructor(loader, maxWindows = 3) {this.loader = loader; this.maxWindows = maxWindows; this.windows = new Map(); this.pending = new Map(); this.generation = 0;}
  clear() {this.generation++; this.windows.clear(); this.pending.clear();}
  frames(time) {return this.windows.get(Math.floor(time / 10))?.frames ?? [];}
  async ensure(time, completed = false) {
    const key = Math.floor(time / 10), found = this.windows.get(key);
    if (found && (completed || performance.now() - found.loaded < 1000)) {
      this.windows.delete(key); this.windows.set(key, found); return;
    }
    if (this.pending.has(key)) return this.pending.get(key);
    const generation = this.generation;
    const promise = this.loader(key * 10, key * 10 + 10).then(frames => {
      if (generation !== this.generation) return;
      this.windows.delete(key); this.windows.set(key, {frames, loaded: performance.now()});
      while (this.windows.size > this.maxWindows) this.windows.delete(this.windows.keys().next().value);
    }).finally(() => {if (generation === this.generation) this.pending.delete(key);});
    this.pending.set(key, promise); return promise;
  }
}
export function validQuad(q) {
  if (q.length !== 4 || q.some(p => p.length !== 2 || p.some(v => !Number.isFinite(v) || v < 0 || v > 1))) return false;
  const crosses = q.map((a, i) => {
    const b = q[(i + 1) % 4], c = q[(i + 2) % 4];
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
  });
  const area = Math.abs(q.reduce((s, a, i) => s + a[0] * q[(i + 1) % 4][1] - a[1] * q[(i + 1) % 4][0], 0)) / 2;
  return area > .025 && crosses.every(v => v > .001);
}

function solve(matrix, rhs) {
  const a = matrix.map((r, i) => [...r, rhs[i]]), n = rhs.length;
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let j = i + 1; j < n; j++) if (Math.abs(a[j][i]) > Math.abs(a[pivot][i])) pivot = j;
    [a[i], a[pivot]] = [a[pivot], a[i]];
    if (Math.abs(a[i][i]) < 1e-10) throw new Error('Reference points are too close or collinear.');
    const d = a[i][i];
    for (let k = i; k <= n; k++) a[i][k] /= d;
    for (let j = 0; j < n; j++) if (j !== i) {
      const m = a[j][i];
      for (let k = i; k <= n; k++) a[j][k] -= m * a[i][k];
    }
  }
  return a.map(r => r[n]);
}

export function homography(from, to) {
  const a = [], b = [];
  from.forEach(([x, y], i) => {
    const [u, v] = to[i];
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(u, v);
  });
  return [...solve(a, b), 1];
}

export function project(h, x, y) {
  const d = h[6] * x + h[7] * y + h[8];
  if (Math.abs(d) < 1e-8) return null;
  return [(h[0] * x + h[1] * y + h[2]) / d, (h[3] * x + h[4] * y + h[5]) / d];
}

export function inside(p, quad) {
  return quad.every((a, i) => {
    const b = quad[(i + 1) % 4];
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0;
  });
}

export function surfacePosition(detection, config) {
  const {bbox_xyxy_normalized: box} = detection ?? {};
  const {corners, width, length, water, measured} = config ?? {};
  if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite) || box.some(v => v < 0 || v > 1) || box[2] <= box[0] || box[3] <= box[1]) return null;
  if (!Array.isArray(corners) || corners.length !== 4 || corners.some(p => !Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite)) ||
      !validQuad(corners) || ![width, length, water].every(Number.isFinite) || width <= 0 || length <= 0 || water < 0) return null;
  const anchor = boxAnchor(box);
  if (!inside(anchor, corners)) return null;
  try {
    const h = homography(corners, [[0, 0], [width, 0], [width, length], [0, length]]);
    const [x, y] = project(h, ...anchor) ?? [];
    if (![x, y].every(Number.isFinite)) return null;
    return {x, y: water, z: y, method: 'water-plane-homography', height_assumed: true, units: 'm', calibration: measured ? 'user-measured' : 'approximate'};
  } catch { return null; }
}

