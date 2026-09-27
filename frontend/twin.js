import * as THREE from './vendor/three.module.js';
import {homography, surfacePosition, cameraPose} from './core.mjs';

export class PoolTwin {
  constructor(canvas, video, onSelect = () => {}) {
    this.canvas = canvas; this.video = video;
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, .1, 1000);
    this.scene.add(new THREE.HemisphereLight(0xc5eeef, 0x183d44, 3));
    const sun = new THREE.DirectionalLight(0xd6fff1, 3); sun.position.set(-15, 25, 15); this.scene.add(sun);
    this.pool = new THREE.Group(); this.scene.add(this.pool);
    this.people = new THREE.Group(); this.scene.add(this.people);
    this.videoTexture = new THREE.VideoTexture(video);
    this.videoTexture.colorSpace = THREE.SRGBColorSpace;
    this.theta = .55; this.phi = .82; this.distance = 60; this.revealStart = null;
    this.markerGeometry = new THREE.RingGeometry(.35, .41, 32);
    this.pinGeometry = new THREE.SphereGeometry(.23, 12, 8);
    this.stemGeometry = new THREE.CylinderGeometry(.045, .045, .45, 6);
    this.sharedGeometry = new Set([this.markerGeometry, this.pinGeometry, this.stemGeometry]);
    this.onSelect = onSelect; this.lastPeople = null;
    this.raycaster = new THREE.Raycaster();
    this.materials = [0xbdfce0, 0xf2d39b, 0x93deff, 0xb8b6ff, 0xffc5b8].map(color => new THREE.MeshBasicMaterial({color}));
    let drag = null, moved = 0;
    canvas.addEventListener('pointerdown', e => {drag = [e.clientX, e.clientY]; moved = 0; canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      moved += Math.hypot(e.clientX - drag[0], e.clientY - drag[1]);
      this.theta -= (e.clientX - drag[0]) * .008;
      this.phi = THREE.MathUtils.clamp(this.phi + (e.clientY - drag[1]) * .005, .23, 1.35);
      drag = [e.clientX, e.clientY];
    });
    canvas.addEventListener('pointerup', e => {
      if (drag && moved < 5) {
        const rect = canvas.getBoundingClientRect();
        this.raycaster.setFromCamera(new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1, 1 - (e.clientY - rect.top) / rect.height * 2), this.camera);
        const hit = this.raycaster.intersectObjects(this.people.children, true)[0];
        this.onSelect(hit?.object.userData.surfacePosition ?? null);
      }
      drag = null;
    });
    canvas.addEventListener('pointercancel', () => drag = null);
    canvas.addEventListener('wheel', e => {e.preventDefault(); this.distance = THREE.MathUtils.clamp(this.distance * Math.exp(e.deltaY * .001), this.size * .5, this.size * 6);}, {passive: false});
    this.resize = new ResizeObserver(() => this.resizeCanvas()); this.resize.observe(canvas);
  }

  resizeCanvas() {
    const width = this.canvas.clientWidth, height = this.canvas.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false); this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
  }

  clearGroup(group) {
    for (const child of [...group.children]) {
      child.traverse(o => {
        if (o.geometry && !this.sharedGeometry.has(o.geometry)) o.geometry.dispose();
        if (o.material && !this.materials.includes(o.material)) {o.material.map?.dispose(); o.material.dispose();}
      });
      group.remove(child);
    }
  }

  configure(config) {
    this.config = config; this.clearGroup(this.pool); this.clearGroup(this.people); this.lastPeople = null; this.onSelect(null); this.canvas.dataset.markerCount = '0';
    const {width: w, length: l, depth: d, water: h, corners} = config;
    this.size = Math.max(w, l); this.pose = cameraPose(corners, w, l, config.aspect || this.video.videoWidth / this.video.videoHeight || 16 / 9); this.resetCamera();
    this.map = homography(corners, [[-w / 2, -l / 2], [w / 2, -l / 2], [w / 2, l / 2], [-w / 2, l / 2]]);
    const forward = homography([[0, 0], [1, 0], [1, 1], [0, 1]], corners);
    const tile = new THREE.MeshStandardMaterial({color: 0x2c5960, roughness: .8, side: THREE.DoubleSide});
    const floor = new THREE.Mesh(new THREE.BoxGeometry(w, .14, l), tile); floor.position.y = -.07; this.pool.add(floor);
    const wall = new THREE.MeshStandardMaterial({color: 0x9aafa9, roughness: .6, transparent: true, opacity: .24, depthWrite: false});
    [[w + .24, d, .16, 0, d / 2, -l / 2], [w + .24, d, .16, 0, d / 2, l / 2], [.16, d, l, -w / 2, d / 2, 0], [.16, d, l, w / 2, d / 2, 0]].forEach(([x, y, z, px, py, pz]) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(x, y, z), wall.clone()); mesh.position.set(px, py, pz); this.pool.add(mesh);
    });
    wall.dispose();
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, d, l)), new THREE.LineBasicMaterial({color: 0x89b3b0, transparent: true, opacity: .6}));
    edges.position.y = d / 2; this.pool.add(edges);
    const grid = [];
    for (let i = 0; i <= 20; i++) {
      const x = -w / 2 + w * i / 20, z = -l / 2 + l * i / 20;
      grid.push(x, .01, -l / 2, x, .01, l / 2, -w / 2, .01, z, w / 2, .01, z);
    }
    this.pool.add(new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(grid, 3)), new THREE.LineBasicMaterial({color: 0x5b9197, transparent: true, opacity: .3})));
    this.waterMaterial = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
      uniforms: {clock: {value: 0}, footage: {value: this.videoTexture}, textureMix: {value: 0},
        imageMap: {value: new THREE.Matrix3().set(...forward)}},
      vertexShader: 'varying vec2 vUv; void main(){vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
      fragmentShader: `uniform float clock; uniform float textureMix; uniform sampler2D footage; uniform mat3 imageMap; varying vec2 vUv;
        void main(){ vec2 poolUv=vec2(vUv.x,1.-vUv.y); vec3 mapped=imageMap*vec3(poolUv,1.); vec2 imageUv=mapped.xy/mapped.z;
        float waves=sin(vUv.x*105.+clock)*sin(vUv.y*75.-clock*.7)*.025;
        float grid=pow(abs(sin(vUv.x*62.83)*sin(vUv.y*62.83)),18.)*.035;
        vec3 water=vec3(.13,.47,.48)+waves+grid;
        vec3 source=texture2D(footage,vec2(imageUv.x,1.-imageUv.y)).rgb;
        gl_FragColor=vec4(mix(water,source,textureMix),mix(.58,.96,textureMix)); }`,
    });
    const water = new THREE.Mesh(new THREE.PlaneGeometry(w, l), this.waterMaterial);
    water.rotation.x = -Math.PI / 2; water.position.y = h; this.pool.add(water);
    const rim = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-w / 2, h, -l / 2), new THREE.Vector3(w / 2, h, -l / 2), new THREE.Vector3(w / 2, h, l / 2), new THREE.Vector3(-w / 2, h, l / 2),
    ]), new THREE.LineBasicMaterial({color: 0x9debd7})); this.pool.add(rim);
    const label = (text, x, z) => {
      const canvas = document.createElement('canvas'); canvas.width = 512; canvas.height = 64;
      const ctx = canvas.getContext('2d'); ctx.fillStyle = '#c5ffeb'; ctx.font = '28px sans-serif'; ctx.fillText(text, 4, 43);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: new THREE.CanvasTexture(canvas), depthTest: false}));
      sprite.position.set(x, h + .15, z); sprite.scale.set(this.size * .27, this.size * .034, 1); this.pool.add(sprite);
    };
    label('1: origin (0, 0)', -w / 2, -l / 2);
    label(`X: ${w} m`, w / 2, -l / 2);
    label(`Z: ${l} m`, -w / 2, l / 2);
    this.pool.add(this.cameraMarker(this.pose, w, l, h));
  }

  // Estimated recording camera: body, height label, and view lines to the mapped water corners.
  cameraMarker(pose, w, l, h) {
    const group = new THREE.Group(), color = 0xf2d39b, eye = new THREE.Vector3(pose.x, h + pose.height, pose.z);
    const body = new THREE.Group(); body.position.copy(eye);
    const s = this.size / 30, solid = new THREE.MeshStandardMaterial({color, roughness: .5});
    body.add(new THREE.Mesh(new THREE.BoxGeometry(.9 * s, .7 * s, 1.3 * s), solid));
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(.22 * s, .3 * s, .5 * s, 20), solid);
    lens.rotation.x = -Math.PI / 2; lens.position.z = -.85 * s; body.add(lens);
    body.lookAt(pose.aim[0], h, pose.aim[1]); body.rotateY(Math.PI); // lookAt points +z; the lens sits on -z
    group.add(body);
    const line = (points, opacity) => new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({color, transparent: true, opacity}));
    [[-w / 2, -l / 2], [w / 2, -l / 2], [w / 2, l / 2], [-w / 2, l / 2]].forEach(([x, z]) => group.add(line([eye, new THREE.Vector3(x, h, z)], .35)));
    const label = document.createElement('canvas'); label.width = 384; label.height = 64;
    const ctx = label.getContext('2d'); ctx.font = '600 26px Inter, Arial, sans-serif'; ctx.fillStyle = '#f2d39b'; ctx.textAlign = 'center';
    ctx.fillText('Camera · ~' + Math.round(pose.height) + ' m up (est.)', 192, 40);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: new THREE.CanvasTexture(label), transparent: true, depthTest: false}));
    sprite.scale.set(9 * s, 1.5 * s, 1); sprite.position.set(eye.x, eye.y + 1.6 * s, eye.z); group.add(sprite);
    return group;
  }

  resetCamera() {
    // Orbit around the midpoint of pool and camera, viewed from the side so both stay in frame.
    const {x, z, height} = this.pose;
    this.target = new THREE.Vector3(x / 2, this.config.water + height / 3, z / 2);
    this.theta = Math.atan2(x, z) - 1.1; this.phi = .62;
    this.distance = Math.max(this.size * 1.95, Math.hypot(x, z, height) * 2.1);
  }
  preset(name) { // Top, Front, or from the estimated recording camera
    const water = this.config.water, {x, z, height, aim} = this.pose;
    if (name === 'camera') {
      this.target.set(aim[0], water, aim[1]);
      const dx = x - aim[0], dz = z - aim[1];
      this.theta = Math.atan2(dx, dz); this.phi = THREE.MathUtils.clamp(Math.atan2(height, Math.hypot(dx, dz)), .23, 1.35);
      this.distance = Math.hypot(dx, dz, height) - 2.5 * this.size / 30; // just in front of the lens, not inside the camera model
    } else {
      this.target.set(0, water, 0); this.theta = 0;
      this.phi = name === 'top' ? 1.35 : .3; this.distance = this.size * (name === 'top' ? 1.7 : 1.9);
    }
  }
  reveal() {this.revealStart = performance.now();}

  updatePeople(detections) {
    const signature = JSON.stringify(detections);
    if (signature === this.lastPeople) return;
    this.lastPeople = signature; this.onSelect(null);
    this.clearGroup(this.people);
    if (!this.config) return;
    for (const detection of detections) {
      const loc = surfacePosition(detection, this.config);
      if (!loc) continue;
      const x = loc.x - this.config.width / 2, z = loc.z - this.config.length / 2;
      const ring = new THREE.Mesh(this.markerGeometry, this.materials[0]);
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(x, loc.y + .025, z);
      const pin = new THREE.Mesh(this.pinGeometry, this.materials[0]); pin.position.set(x, loc.y + .5, z);
      const stem = new THREE.Mesh(this.stemGeometry, this.materials[0]); stem.position.set(x, loc.y + .25, z);
      for (const marker of [ring, pin, stem]) {marker.userData.surfacePosition = loc; this.people.add(marker);}
    }
    this.canvas.dataset.markerCount = this.people.children.length / 3;
  }

  render(now) {
    if (!this.config) return;
    const reveal = this.revealStart === null ? 1 : Math.min(1, (now - this.revealStart) / 2300);
    const eased = reveal * reveal * (3 - 2 * reveal);
    this.waterMaterial.uniforms.textureMix.value = 1 - eased;
    this.waterMaterial.uniforms.clock.value = now / 1300;
    const phi = this.phi - (1 - eased) * .2;
    // Preserve horizontal coverage when the split/mobile viewport becomes narrow.
    const distance = this.distance * Math.max(1, 1 / this.camera.aspect);
    this.camera.position.set(distance * Math.sin(this.theta) * Math.cos(phi), distance * Math.sin(phi), distance * Math.cos(this.theta) * Math.cos(phi)).add(this.target);
    this.camera.lookAt(this.target);
    this.renderer.render(this.scene, this.camera);
  }
}
