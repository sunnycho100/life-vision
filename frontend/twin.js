import * as THREE from './vendor/three.module.js';
import {project, homography, inside, boxAnchor} from './core.mjs';

export class PoolTwin {
  constructor(canvas, video) {
    this.canvas = canvas;
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
    this.materials = [0xbdfce0, 0xf2d39b, 0x93deff, 0xb8b6ff, 0xffc5b8].map(color => new THREE.MeshBasicMaterial({color}));
    let drag = null;
    canvas.addEventListener('pointerdown', e => {drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      this.theta -= (e.clientX - drag[0]) * .008;
      this.phi = THREE.MathUtils.clamp(this.phi + (e.clientY - drag[1]) * .005, .23, 1.35);
      drag = [e.clientX, e.clientY];
    });
    canvas.addEventListener('pointerup', () => drag = null);
    canvas.addEventListener('pointercancel', () => drag = null);
    canvas.addEventListener('wheel', e => {e.preventDefault(); this.distance = THREE.MathUtils.clamp(this.distance * Math.exp(e.deltaY * .001), this.size * .8, this.size * 4);}, {passive: false});
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
        if (o.geometry && o.geometry !== this.markerGeometry) o.geometry.dispose();
        if (o.material && !this.materials.includes(o.material)) o.material.dispose();
      });
      group.remove(child);
    }
  }

  configure(config) {
    this.config = config; this.clearGroup(this.pool); this.clearGroup(this.people);
    const {width: w, length: l, depth: d, water: h, corners} = config;
    this.size = Math.max(w, l); this.resetCamera();
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
  }

  resetCamera() {this.theta = .52; this.phi = .82; this.distance = this.size * 1.95;}
  reveal() {this.revealStart = performance.now();}

  updatePeople(detections) {
    this.clearGroup(this.people);
    if (!this.config) return;
    for (const detection of detections) {
      const [x, y] = boxAnchor(detection.bbox_xyxy_normalized);
      if (!inside([x, y], this.config.corners)) continue;
      const loc = project(this.map, x, y);
      if (!loc) continue;
      const ring = new THREE.Mesh(this.markerGeometry, this.materials[0]);
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(loc[0], this.config.water + .025, loc[1]);
      this.people.add(ring);
    }
  }

  render(now) {
    if (!this.config) return;
    const reveal = this.revealStart === null ? 1 : Math.min(1, (now - this.revealStart) / 2300);
    const eased = reveal * reveal * (3 - 2 * reveal);
    this.waterMaterial.uniforms.textureMix.value = 1 - eased;
    this.waterMaterial.uniforms.clock.value = now / 1300;
    const phi = this.phi - (1 - eased) * .2;
    this.camera.position.set(this.distance * Math.sin(this.theta) * Math.cos(phi), this.distance * Math.sin(phi), this.distance * Math.cos(this.theta) * Math.cos(phi));
    this.camera.lookAt(0, this.config.water, 0);
    this.renderer.render(this.scene, this.camera);
  }
}
