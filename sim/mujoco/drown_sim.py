"""Scripted drowning and swimming motions for the DeepMind humanoid in MuJoCo.

Runs headless. For each motion it simulates 20 s and saves per-frame 3D keypoints,
a per-frame "head fully underwater" flag, and a distress label to data/sim_samples/.

Water model:
  - Drag: MuJoCo ellipsoid fluid model with water density and viscosity.
  - Buoyancy: MuJoCo has no buoyancy, so we add it per geom from the submerged fraction.
    Tissue is slightly denser than water (TISSUE_BUOYANCY < 1). Air in the lungs adds lift
    at the chest, set per motion in liters. Full lungs float, empty lungs sink. The humanoid's
    mass equals its geom volume times 1000 kg/m^3, so buoyancy factor 1 would be neutral.
  - Muscles: joint PD torques toward scripted targets, applied directly to the joints.

Usage: python drown_sim.py [--seconds 20] [--fps 30] [--seed 0] [--video]
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[1] / "data" / "sim_samples"

WATER_DENSITY = 1000.0  # kg/m^3
WATER_VISCOSITY = 0.0009  # Pa*s at about 25 C
SURFACE_Z = 0.0  # water surface height
POOL_DEPTH = 2.0  # pool floor at z = -2
TIMESTEP = 1 / 240
POOL_CENTER_X = 3.0  # pool spans x -2..8, y -2.5..2.5 (swimmer heads toward +x)
POOL_HALF = (5.0, 2.5)
TISSUE_BUOYANCY = 0.98  # body without lung air sinks slowly
LUNG_GEOMS = ("torso", "waist_upper")  # chest geoms that carry the lung air

KEYPOINTS = ["head", "torso", "pelvis",
             "upper_arm_right", "lower_arm_right", "hand_right",
             "upper_arm_left", "lower_arm_left", "hand_left",
             "thigh_right", "shin_right", "foot_right",
             "thigh_left", "shin_left", "foot_left"]

# Root orientations (w, x, y, z)
UPRIGHT = [1, 0, 0, 0]
PRONE = [0.7071, 0, 0.7071, 0]    # face down, for swimming
SUPINE = [0.7071, 0, -0.7071, 0]  # face up, for floating


def build_model():
    xml = (HERE / "humanoid.xml").read_text()
    xml = xml.replace(
        '<option timestep="0.005"/>',
        f'<option timestep="{TIMESTEP}" density="{WATER_DENSITY}" viscosity="{WATER_VISCOSITY}"/>')
    # ellipsoid fluid model on every body geom
    xml = xml.replace('<geom type="capsule" condim="1"',
                      '<geom type="capsule" fluidshape="ellipsoid" condim="1"')
    xml = xml.replace('<geom name="floor" size="0 0 .05" type="plane"',
                      f'<geom name="floor" pos="0 0 {-POOL_DEPTH}" size="0 0 .05" type="plane"')
    # Visual-only pool: walls and a see-through water volume (no collision, no physics)
    cx, (hx, hy), hz = POOL_CENTER_X, POOL_HALF, POOL_DEPTH / 2
    vis = 'contype="0" conaffinity="0" group="0"'
    pool = f"""
    <geom name="water" type="box" pos="{cx} 0 {-hz}" size="{hx} {hy} {hz}" rgba="0.15 0.55 0.85 0.35" {vis}/>
    <geom type="box" pos="{cx} {hy + .1} {-hz + .1}" size="{hx + .2} .1 {hz + .1}" rgba=".9 .9 .88 1" {vis}/>
    <geom type="box" pos="{cx} {-hy - .1} {-hz + .1}" size="{hx + .2} .1 {hz + .1}" rgba=".9 .9 .88 1" {vis}/>
    <geom type="box" pos="{cx + hx + .1} 0 {-hz + .1}" size=".1 {hy} {hz + .1}" rgba=".9 .9 .88 1" {vis}/>
    <geom type="box" pos="{cx - hx - .1} 0 {-hz + .1}" size=".1 {hy} {hz + .1}" rgba=".9 .9 .88 1" {vis}/>
    <light name="sun" directional="true" pos="{cx} 0 6" dir="0 0 -1" diffuse=".6 .6 .6" castshadow="false"/>
    <camera name="cctv" pos="{cx - hx - .5} {-hy - .5} 2.5" xyaxes="1 -1 0 .45 .45 .77" fovy="55"/>"""
    xml = xml.replace('<light name="spotlight"', pool + '\n    <light name="spotlight"', 1)
    # the model's body-following light goes under the surface and blacks out the water
    xml = xml.replace('<light name="top" pos="0 0 2" mode="trackcom"/>', '')
    return mujoco.MjModel.from_xml_string(xml)


def geom_volume(m, g):
    r = m.geom_size[g, 0]
    sphere = 4 / 3 * np.pi * r ** 3
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
        return sphere
    return np.pi * r * r * 2 * m.geom_size[g, 1] + sphere  # capsule


def apply_buoyancy(m, d, geoms, volumes, lung_liters, lung_vol):
    """Add upward force rho*g*V*frac at each geom center, plus lung air at the chest."""
    g = -m.opt.gravity[2]
    for gid, vol in zip(geoms, volumes):
        if m.geom(gid).name.rsplit("/", 1)[-1] in LUNG_GEOMS:  # lung air spread over chest geoms by volume
            vol_eff = TISSUE_BUOYANCY * vol + lung_liters / 1000 * vol / lung_vol
        else:
            vol_eff = TISSUE_BUOYANCY * vol
        r = m.geom_size[gid, 0]
        # vertical half-extent: radius plus the capsule segment's vertical reach
        half = r + abs(m.geom_size[gid, 1] * d.geom_xmat[gid, 8]) if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SPHERE else r
        zc = d.geom_xpos[gid, 2]
        # ponytail: linear submerged fraction, exact integral over the capsule if accuracy matters
        frac = np.clip((SURFACE_Z - (zc - half)) / (2 * half), 0.0, 1.0)
        if frac > 0:
            force = np.array([0.0, 0.0, WATER_DENSITY * g * vol_eff * frac])
            mujoco.mj_applyFT(m, d, force, np.zeros(3), d.geom_xpos[gid],
                              m.geom_bodyid[gid], d.qfrc_applied)


def lerp(a, b, s):
    return a + (b - a) * s


# Each motion: start pose, buoyancy B(t), joint targets(t), label
def targets_swim(t, rng):
    ph = 2 * np.pi * 0.8 * t  # 0.8 strokes/s per arm
    s = 0.5 * (1 + np.sin(ph))
    kick = 0.3 * np.sin(2 * np.pi * 2.0 * t)
    return {
        "shoulder1_right": lerp(-1.4, 1.0, s), "shoulder2_right": lerp(0.7, -1.4, s),
        "shoulder1_left": lerp(-1.4, 1.0, 1 - s), "shoulder2_left": lerp(0.7, -1.4, 1 - s),
        "hip_y_right": kick, "hip_y_left": -kick, "knee_right": -0.2, "knee_left": -0.2,
    }


def targets_float(t, rng):
    return {"shoulder1_right": -0.7, "shoulder2_right": -1.2,
            "shoulder1_left": -0.7, "shoulder2_left": -1.2,
            "hip_x_right": -0.2, "hip_x_left": -0.2}


def targets_idr(t, rng):
    # Instinctive drowning response: arms out to the side pressing down on the water,
    # body upright, head tilted back, no useful kick.
    press = 0.5 * (1 + np.sin(2 * np.pi * 1.2 * t))
    return {"shoulder2_right": -1.4, "shoulder2_left": -1.4,
            "shoulder1_right": lerp(-0.9, 0.1, press), "shoulder1_left": lerp(-0.9, 0.1, press),
            "elbow_right": -0.3, "elbow_left": -0.3,
            "abdomen_y": -0.3,
            "knee_right": -0.3 * press, "knee_left": -0.3 * (1 - press)}


def targets_sink(t, rng):
    return {"shoulder2_right": -1.0, "shoulder2_left": -1.0,
            "shoulder1_right": 0.5, "shoulder1_left": 0.5}


MOTIONS = {
    # name: (root quat, root z, lung air liters(t), targets, distress label, forward thrust N)
    # ponytail: swim thrust is a constant scripted force, the fluid model alone barely propels the stroke
    "normal_swim": (PRONE, -0.05, lambda t: 3.0, targets_swim, 0, 25.0),
    "float": (SUPINE, -0.05, lambda t: 3.0, targets_float, 0, 0.0),
    "idr": (UPRIGHT, -0.15, lambda t: lerp(3.0, 0.5, min(t / 20, 1)), targets_idr, 1, 0.0),
    "silent_sink": (UPRIGHT, -0.15, lambda t: lerp(1.0, 0.0, min(t / 5, 1)), targets_sink, 1, 0.0),
}


def run_motion(m, name, seconds, fps, seed, video=None):
    quat, z0, lungs, targets_fn, label, thrust = MOTIONS[name]
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    d.qpos[0:3] = [0, 0, z0]
    d.qpos[3:7] = quat
    mujoco.mj_forward(m, d)

    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
    volumes = [geom_volume(m, g) for g in geoms]
    lung_vol = sum(v for g, v in zip(geoms, volumes) if m.geom(g).name.rsplit("/", 1)[-1] in LUNG_GEOMS)
    joint_names = [m.joint(j).name for j in range(1, m.njnt)]  # skip free root
    qadr = {n: m.joint(n).qposadr[0] for n in joint_names}
    vadr = {n: m.joint(n).dofadr[0] for n in joint_names}
    kp, kd = 60.0, 4.0
    body_ids = [m.body(k).id for k in KEYPOINTS]
    head_r = m.geom_size[m.geom("head").id, 0]

    steps_per_frame = round(1 / (fps * TIMESTEP))
    n_frames = int(seconds * fps)
    kps = np.zeros((n_frames, len(KEYPOINTS), 3), np.float32)
    under = np.zeros(n_frames, bool)

    for f in range(n_frames):
        for _ in range(steps_per_frame):
            d.qfrc_applied[:] = 0
            tgt = targets_fn(d.time, rng)
            for n in joint_names:  # joints not listed are held at 0 (relaxed pose)
                err = tgt.get(n, 0.0) - d.qpos[qadr[n]]
                d.qfrc_applied[vadr[n]] += kp * err - kd * d.qvel[vadr[n]]
            apply_buoyancy(m, d, geoms, volumes, lungs(d.time), lung_vol)
            if thrust:  # along the body's head direction, kept horizontal
                fwd = d.body("torso").xmat.reshape(3, 3)[:, 2].copy(); fwd[2] = 0
                mujoco.mj_applyFT(m, d, thrust * fwd / (np.linalg.norm(fwd) + 1e-9), np.zeros(3),
                                  d.body("torso").xpos, m.body("torso").id, d.qfrc_applied)
            mujoco.mj_step(m, d)
        kps[f] = d.xpos[body_ids]
        if video:
            video.update_scene(d, camera="cctv")
            video.pipe.stdin.write(video.render().tobytes())
        under[f] = d.body("head").xpos[2] + head_r < SURFACE_Z
        if not np.isfinite(kps[f]).all():
            raise RuntimeError(f"{name}: simulation diverged at frame {f}")

    return kps, under, label


def open_video(m, path, fps, w=960, h=540):
    """Offscreen renderer that streams raw frames into ffmpeg (needs ffmpeg on PATH)."""
    import subprocess
    path.parent.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(m, height=h, width=w)
    r.pipe = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", str(path)],
        stdin=subprocess.PIPE)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true", help="also write an MP4 per motion from the cctv camera")
    args = ap.parse_args()

    m = build_model()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"fps": args.fps, "seconds": args.seconds, "keypoints": KEYPOINTS, "motions": {}}
    all_kps, all_under, labels = [], [], []
    for name in MOTIONS:
        video = open_video(m, OUT_DIR / "videos" / f"{name}.mp4", args.fps) if args.video else None
        kps, under, label = run_motion(m, name, args.seconds, args.fps, args.seed, video)
        if video:
            video.pipe.stdin.close(); video.pipe.wait()
        all_kps.append(kps); all_under.append(under); labels.append(label)
        head_z = kps[:, 0, 2]
        summary["motions"][name] = {
            "distress_label": label,
            "frames": len(kps),
            "head_z_start": round(float(head_z[0]), 3),
            "head_z_end": round(float(head_z[-1]), 3),
            "head_z_min": round(float(head_z.min()), 3),
            "pct_frames_head_underwater": round(100 * float(under.mean()), 1),
            "first_underwater_s": (round(float(np.argmax(under)) / args.fps, 2) if under.any() else None),
            "horizontal_travel_m": round(float(np.linalg.norm(kps[-1, 1, :2] - kps[0, 1, :2])), 3),
        }

    np.savez_compressed(OUT_DIR / "sim_samples.npz",
                        keypoints=np.stack(all_kps), head_underwater=np.stack(all_under),
                        distress=np.array(labels), motions=np.array(list(MOTIONS)),
                        keypoint_names=np.array(KEYPOINTS), fps=args.fps)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["motions"], indent=2))


if __name__ == "__main__":
    main()
