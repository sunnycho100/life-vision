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
  constructor(loader, maxWindows = 4) {this.loader = loader; this.maxWindows = maxWindows; this.windows = new Map(); this.pending = new Map(); this.generation = 0;}
  clear() {this.generation++; this.windows.clear(); this.pending.clear();}
  // Frames from the current 10 s window plus its loaded neighbors, so a box never drops at a window edge.
  frames(time) {
    const key = Math.floor(time / 10);
    return [key - 1, key, key + 1].flatMap(k => this.windows.get(k)?.frames ?? []).sort((a, b) => a.media_time - b.media_time);
  }
  async ensure(time, completed = false) {
    const key = Math.floor(time / 10);
    if (time - key * 10 > 5) this.load(key + 1, completed); // prefetch the next window before playback reaches it
    return this.load(key, completed);
  }
  async load(key, completed = false) {
    const found = this.windows.get(key);
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


// Best-fit camera over the mapped water, in the 3D view's axes (x right, y up, water at y = 0).
// ponytail: pattern search over position, height and lens (about 48 to 104 degree field of view), aimed at the image center with no roll.
// Mapped corners often outline visible water rather than a true rectangle, so an exact solve can fail; this always returns the closest fit.
export function cameraPose(corners, width, length, aspect) {
  const world = [[-width / 2, -length / 2], [width / 2, -length / 2], [width / 2, length / 2], [-width / 2, length / 2]];
  const map = homography(corners, world), aim = project(map, .5, .5), near = project(map, .5, .95);
  const image = corners.map(([x, y]) => [(x - .5) * aspect, y - .5]);
  const norm = v => {const n = Math.hypot(...v); return v.map(x => x / n);};
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const error = ([x, z, height, f]) => {
    if (height < .3 || f < .7 || f > 2) return Infinity;
    const eye = [x, height, z], forward = norm([aim[0] - x, -height, aim[1] - z]);
    const right = norm(cross(forward, [0, 1, 0])), down = cross(forward, right);
    return world.reduce((sum, [wx, wz], i) => {
      const p = [wx - eye[0], -eye[1], wz - eye[2]], depth = dot(p, forward);
      if (depth <= 0) return Infinity;
      return sum + (f * dot(p, right) / depth - image[i][0]) ** 2 + (f * dot(p, down) / depth - image[i][1]) ** 2;
    }, 0);
  };
  const size = Math.max(width, length), back = norm([near[0] - aim[0], near[1] - aim[1]]);
  let best = [near[0] + back[0] * size * .5, near[1] + back[1] * size * .5, size * .4, 1.27], score = error(best);
  const steps = [size / 4, size / 4, size / 4, .25];
  for (let round = 0; round < 60 && steps[0] > 1e-4; round++) {
    let improved = false;
    for (let k = 0; k < 4; k++) for (const sign of [1, -1]) {
      const trial = best.slice(); trial[k] += sign * steps[k];
      const e = error(trial);
      if (e < score) {best = trial; score = e; improved = true;}
    }
    if (!improved) steps.forEach((_, k) => steps[k] /= 2); else round--;
  }
  return {x: best[0], z: best[1], height: best[2], focal: best[3], aim, fitError: Math.sqrt(score / 4)};
}
