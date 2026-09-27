"""People in a pool with a shallow and deep end, rendered from a corner CCTV camera.

Reuses the water model and motions from drown_sim.py. Each person gets random skin tone,
hair, and swimwear colors. Optionally writes ground truth from the simulator:
  - tracks_gt.json (contract A in docs/global/architecture.md): per-frame visible box,
    head box, and head above/below water for every person, from a segmentation render
  - YOLO-format frames and labels for training a detector on sim images

Usage: python pool_scene.py [--seconds 5] [--fps 30] [--seed 0] [--depth 0.9] [--no-video]
Output: data/sim_samples/pool_scene.npz, pool_scene.json, videos/pool_scene.mp4
Scenarios with ground truth: see scenarios.py
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import drown_sim as ds


# Extra motions for the scene. Standing people start with feet on the shallow floor.
def targets_stand(t, rng):
    sway = 0.15 * np.sin(2 * np.pi * 0.4 * t)
    return {"shoulder1_right": 0.6 + sway, "shoulder1_left": 0.6 - sway,
            "shoulder2_right": -0.3, "shoulder2_left": -0.3}


SQUAT = {"knee_right": -2.2, "knee_left": -2.2, "hip_y_right": -1.8, "hip_y_left": -1.8,
         "abdomen_y": 0.5, "shoulder2_right": -1.0, "shoulder2_left": -1.0}


def targets_collapse(t, rng):
    # Stands for 1 s, then goes limp and folds down under the surface
    return targets_stand(t, rng) if t < 1.0 else SQUAT


DEEP_SQUAT = {**SQUAT, "knee_right": -2.6, "knee_left": -2.6, "hip_y_right": -2.2, "hip_y_left": -2.2}


def targets_duck(t, rng):
    # Breath-hold play: ducks under from 2 s to 5 s, then stands back up
    return DEEP_SQUAT if 2.0 <= t < 5.0 else targets_stand(t, rng)


def targets_fall(t, rng):
    return targets_stand(t, rng) if t < 1.0 else {"shoulder1_right": -1.0, "shoulder1_left": -1.0}


def between(a, b, f):
    return lambda t: np.asarray(f, float) if a <= t < b else None


# Each motion: joint targets(t), start pose, lungs in liters(t), forward thrust (N),
# distress label, when the upright balance torque is on, and an extra world force(t) at the torso.
# start z: a number, "stand" (feet on the shallow floor), or "deck" (feet on the pool deck).
ALWAYS, NEVER = (lambda t: True), (lambda t: False)
SCENE_MOTIONS = {
    "stand":       dict(targets=targets_stand, quat=ds.UPRIGHT, z="stand", lungs=lambda t: 3.0, balance=ALWAYS),
    "normal_swim": dict(targets=ds.targets_swim, quat=ds.PRONE, z=-0.05, lungs=lambda t: 3.0, thrust=25.0),
    "float":       dict(targets=ds.targets_float, quat=ds.SUPINE, z=-0.05, lungs=lambda t: 3.0),
    "idr":         dict(targets=ds.targets_idr, quat=ds.UPRIGHT, z=-0.15, label=1,
                        lungs=lambda t: ds.lerp(3.0, 0.5, min(t / 20, 1))),
    "collapse":    dict(targets=targets_collapse, quat=ds.UPRIGHT, z="stand", label=1,
                        lungs=lambda t: 1.0 if t < 1 else 0.0, balance=lambda t: t < 1.0),
    "silent_sink": dict(targets=ds.targets_sink, quat=ds.UPRIGHT, z=-0.15, label=1,
                        lungs=lambda t: ds.lerp(1.0, 0.0, min(t / 5, 1))),
    # Breath-hold play, should be a warning at most, never an alarm
    # (exhales and pushes down, like a kid ducking under on purpose)
    "duck_under":  dict(targets=targets_duck, quat=ds.UPRIGHT, z="stand", balance=lambda t: not 2.0 <= t < 5.0,
                        lungs=lambda t: 0.5 if 2.0 <= t < 5.0 else 3.0, force=between(2.0, 5.0, [0, 0, -350])),
    # Dives at 1 s, swims underwater, comes back up about 2 m away around 6-7 s
    "dive_resurface": dict(targets=ds.targets_swim, quat=ds.PRONE, z=-0.05, thrust=30.0,
                           lungs=lambda t: 0.3 if 1.0 <= t < 6.0 else 3.0,
                           force=lambda t: between(1.0, 2.5, [0, 0, -120])(t) if t < 6 else between(6.0, 7.5, [0, 0, 150])(t)),
    # Stands on the deck at the pool edge, gets pushed in at 1 s (entry test)
    "fall_in":     dict(targets=targets_fall, quat=ds.UPRIGHT, z="deck", lungs=lambda t: 3.0,
                        balance=lambda t: t < 1.0, force=None),
}
for _m in SCENE_MOTIONS.values():
    _m.setdefault("label", 0); _m.setdefault("thrust", 0.0); _m.setdefault("balance", NEVER); _m.setdefault("force", None)

# Default cast: (motion, x, y, heading deg). Pool spans x -2..8, y -2.5..2.5,
# shallow end x < SHALLOW_END_X, deep end beyond.
DEFAULT_CAST = [("stand", -0.5, 1.2, 0), ("normal_swim", 0.3, -1.3, 0), ("float", 5.0, -0.8, 0),
                ("idr", 6.5, 1.3, 0), ("collapse", 1.5, 1.2, 90)]
SHALLOW_END_X = 3.0
DEEP = 2.0
DECK_Z = 0.2  # deck and rim height above the water
LEG_KP, LEG_KD = 200.0, 6.0  # stiff legs so standing people do not tip over (ankles too light for this)
LEG_JOINTS = ("hip", "knee", "abdomen")
# ponytail: scripted upright torque instead of a real balance controller, swap for one if standing matters
BAL_KP, BAL_KD = 400.0, 40.0
FALL_PUSH_N = 250.0

SKIN = [(0.98, 0.84, 0.72), (0.91, 0.72, 0.58), (0.78, 0.57, 0.42), (0.58, 0.40, 0.28), (0.36, 0.24, 0.16)]
HAIR = [(0.08, 0.06, 0.05), (0.30, 0.18, 0.10), (0.75, 0.60, 0.35), (0.45, 0.20, 0.08)]
SUIT = [(0.10, 0.10, 0.12), (0.85, 0.15, 0.15), (0.10, 0.30, 0.75), (0.95, 0.55, 0.10), (0.15, 0.55, 0.35)]
SUIT_GEOMS = ("waist_lower", "butt")
TOP_GEOMS = ("torso", "waist_upper")  # rash guard for some people
WATER_GROUP = 2  # drawn by default, hidden in the segmentation render so bodies under water still count as visible


def yaw_quat(deg):
    return np.array([np.cos(np.radians(deg) / 2), 0, 0, np.sin(np.radians(deg) / 2)])


def build_scene(shallow, rng, cast):
    spec = mujoco.MjSpec()
    spec.option.timestep = ds.TIMESTEP
    spec.option.density = ds.WATER_DENSITY
    spec.option.viscosity = ds.WATER_VISCOSITY
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    wb = spec.worldbody
    cx, (hx, hy), hz = ds.POOL_CENTER_X, ds.POOL_HALF, DEEP / 2
    BOX = mujoco.mjtGeom.mjGEOM_BOX
    tile = [0.55, 0.75, 0.85, 1]
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[cx, 0, -DEEP],
                size=[hx, hy, 0.05], rgba=tile)
    x0 = cx - hx  # shallow end: a solid block raising the floor to -shallow
    wb.add_geom(name="shallow_floor", type=BOX, pos=[(x0 + SHALLOW_END_X) / 2, 0, -(shallow + DEEP) / 2],
                size=[(SHALLOW_END_X - x0) / 2, hy, (DEEP - shallow) / 2], rgba=tile)
    vis = dict(contype=0, conaffinity=0)
    wb.add_geom(name="water", type=BOX, pos=[cx, 0, -hz], size=[hx, hy, hz],
                rgba=[0.15, 0.55, 0.85, 0.35], group=WATER_GROUP, **vis)
    wall = [0.92, 0.92, 0.9, 1]
    for pos, size in [([cx, hy + .1, -hz + .1], [hx + .2, .1, hz + .1]),
                      ([cx, -hy - .1, -hz + .1], [hx + .2, .1, hz + .1]),
                      ([cx + hx + .1, 0, -hz + .1], [.1, hy, hz + .1]),
                      ([cx - hx - .1, 0, -hz + .1], [.1, hy, hz + .1])]:
        wb.add_geom(type=BOX, pos=pos, size=size, rgba=wall, **vis)
    # Walkable deck along the far long side (for the fall-in entry test)
    wb.add_geom(name="deck", type=BOX, pos=[cx, hy + 0.2 + 1.5, DECK_Z / 2], size=[hx + 0.2, 1.5, DECK_Z / 2],
                rgba=[0.8, 0.78, 0.72, 1])
    wb.add_geom(type=BOX, pos=[cx, 0, -0.01 - DEEP - 0.2], size=[hx + 6, hy + 6, 0.01],
                rgba=[0.35, 0.35, 0.33, 1], **vis)  # ground below the rim
    wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[cx, 0, 6], dir=[0, 0, -1],
                 diffuse=[0.7, 0.7, 0.7], castshadow=0)
    wb.add_camera(name="cctv", pos=[cx - hx - .5, -hy - .5, 2.5], fovy=55,
                  xyaxes=[1, -1, 0, .45, .45, .77])

    looks = []
    for i, (name, x, y, heading) in enumerate(cast):
        child = mujoco.MjSpec.from_file(str(ds.HERE / "humanoid.xml"))
        for k in list(child.keys):
            child.delete(k)
        skin, hair, suit = (SKIN[rng.integers(len(SKIN))], HAIR[rng.integers(len(HAIR))],
                            SUIT[rng.integers(len(SUIT))])
        rash = rng.random() < 0.4
        for g in child.geoms:
            if g.name == "floor":
                continue
            g.fluid_ellipsoid = 1  # same ellipsoid fluid model as drown_sim
            g.material = ""
            color = suit if g.name in SUIT_GEOMS or (rash and g.name in TOP_GEOMS) else skin
            g.rgba = [*color, 1]
        # hair cap: slightly bigger sphere pushed back, face stays visible
        child.body("head").add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.094, 0, 0],
                                    pos=[-0.02, 0, 0.02], rgba=[*hair, 1], density=0, **vis)
        frame = wb.add_frame(pos=[x, y, 0], quat=yaw_quat(heading))
        frame.attach_body(child.body("torso"), f"p{i}/", "")
        looks.append({"person": i, "motion": name, "skin": skin, "hair": hair, "suit": suit, "rash_guard": rash})
    return spec.compile(), looks


def box_of(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:  # ponytail: fixed pixel floor for "visible", scale with resolution if it changes
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def run(cast, seconds, fps=30, seed=0, depth=0.9, out_dir=None, name="pool_scene",
        video=True, gt=False, yolo_every=0, size=(960, 540)):
    """Simulate a cast. Returns summary dict. Writes video, npz, and optionally ground truth."""
    out_dir = Path(out_dir or ds.OUT_DIR)
    rng = np.random.default_rng(seed)
    m, looks = build_scene(depth, rng, cast)
    d = mujoco.MjData(m)

    people = []
    for i, (motion, x, y, heading) in enumerate(cast):
        mo = SCENE_MOTIONS[motion]
        pre = f"p{i}/"
        root = m.joint(pre + "root")
        a = root.qposadr[0]
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, yaw_quat(heading), np.array(mo["quat"], float))
        z = {"stand": -depth + 1.29, "deck": DECK_Z + 1.29}.get(mo["z"], mo["z"])
        assert mo["z"] != "stand" or x < SHALLOW_END_X, f"{motion}: standing people belong in the shallow end"
        d.qpos[a:a + 3] = [x, y, z]
        d.qpos[a + 3:a + 7] = q
        geoms = [g for g in range(m.ngeom)
                 if (m.geom(g).name or "").startswith(pre) and m.geom_contype[g] != 0]
        vols = [ds.geom_volume(m, g) for g in geoms]
        joints = [j for j in range(m.njnt) if m.joint(j).name.startswith(pre) and j != root.id]
        bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(pre)}
        people.append(dict(
            name=motion, mo=mo, root_dof=root.dofadr[0], torso=m.body(pre + "torso").id,
            geoms=geoms, vols=vols, heading=heading,
            lung_vol=sum(v for g, v in zip(geoms, vols) if m.geom(g).name.rsplit("/", 1)[-1] in ds.LUNG_GEOMS),
            joints=[(m.joint(j).name[len(pre):], m.jnt_qposadr[j], m.jnt_dofadr[j]) for j in joints],
            kp_ids=[m.body(pre + k).id for k in ds.KEYPOINTS], head=m.body(pre + "head").id,
            all_geoms=np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies]),
            head_geoms=np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(pre + "head").id])))
    mujoco.mj_forward(m, d)

    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    vid = ds.open_video(m, out_dir / "videos" / f"{name}.mp4", fps, w, h) if video else None
    seg = None
    if gt or yolo_every:
        seg = mujoco.Renderer(m, height=h, width=w)
        seg.enable_segmentation_rendering()
        seg_opt = mujoco.MjvOption()
        seg_opt.geomgroup[WATER_GROUP] = 0
        rgb = vid or mujoco.Renderer(m, height=h, width=w)
    steps = round(1 / (fps * ds.TIMESTEP))
    T, P = int(seconds * fps), len(people)
    kps = np.zeros((P, T, len(ds.KEYPOINTS), 3), np.float32)
    under = np.zeros((P, T), bool)
    head_r = 0.09
    kp, kd = 60.0, 4.0
    frames_gt = []
    if yolo_every:
        for sub in ("images", "labels"):
            (out_dir / "yolo" / sub).mkdir(parents=True, exist_ok=True)
    for f in range(T):
        for _ in range(steps):
            d.qfrc_applied[:] = 0
            t = d.time
            for p in people:
                mo = p["mo"]
                tgt = mo["targets"](t, None)
                for jn, qa, va in p["joints"]:
                    k, c = (LEG_KP, LEG_KD) if jn.startswith(LEG_JOINTS) else (kp, kd)
                    d.qfrc_applied[va] += k * (tgt.get(jn, 0.0) - d.qpos[qa]) - c * d.qvel[va]
                ds.apply_buoyancy(m, d, p["geoms"], p["vols"], mo["lungs"](t), p["lung_vol"])
                if mo["balance"](t):
                    R = d.xmat[p["torso"]].reshape(3, 3)
                    wv = R @ d.qvel[p["root_dof"] + 3:p["root_dof"] + 6]  # angular velocity, world frame
                    torque = BAL_KP * np.cross(R[:, 2], [0, 0, 1]) - BAL_KD * wv
                    mujoco.mj_applyFT(m, d, np.zeros(3), torque, d.xpos[p["torso"]], p["torso"], d.qfrc_applied)
                force = mo["force"](t) if mo["force"] else None
                if p["name"] == "fall_in" and 1.0 <= t < 1.8:  # push toward the pool, facing direction
                    force = FALL_PUSH_N * np.array([np.cos(np.radians(p["heading"])), np.sin(np.radians(p["heading"])), 0])
                if force is not None:
                    mujoco.mj_applyFT(m, d, force, np.zeros(3), d.xpos[p["torso"]], p["torso"], d.qfrc_applied)
                if mo["thrust"]:
                    fwd = d.xmat[p["torso"]].reshape(3, 3)[:, 2].copy(); fwd[2] = 0
                    mujoco.mj_applyFT(m, d, mo["thrust"] * fwd / (np.linalg.norm(fwd) + 1e-9), np.zeros(3),
                                      d.xpos[p["torso"]], p["torso"], d.qfrc_applied)
            mujoco.mj_step(m, d)
        for i, p in enumerate(people):
            kps[i, f] = d.xpos[p["kp_ids"]]
            under[i, f] = d.xpos[p["head"], 2] + head_r < ds.SURFACE_Z
        if not np.isfinite(kps[:, f]).all():
            raise RuntimeError(f"simulation diverged at frame {f}")
        img = None
        if vid or seg:
            r = vid or rgb
            r.update_scene(d, camera="cctv")
            img = r.render()
            if vid:
                vid.pipe.stdin.write(img.tobytes())
        if seg:
            seg.update_scene(d, camera="cctv", scene_option=seg_opt)
            ids = seg.render()
            geom_ids = np.where(ids[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, ids[..., 0], -1)
            ppl = []
            for i, p in enumerate(people):
                bbox = box_of(np.isin(geom_ids, p["all_geoms"]))
                if bbox is None:
                    continue
                hz_ = d.xpos[p["head"], 2]
                ppl.append({"track_id": i, "bbox": bbox, "conf": 1.0,
                            # below only when the whole head is under the surface
                            "head": "below" if hz_ + head_r < ds.SURFACE_Z else "above",
                            "head_conf": 1.0, "head_bbox": box_of(np.isin(geom_ids, p["head_geoms"])),
                            "keypoints": None, "motion": p["name"], "head_z": round(float(hz_), 3)})
            frames_gt.append({"t": round(f / fps, 4), "brightness": round(float(img.mean() / 255), 3), "people": ppl})
            if yolo_every and f % yolo_every == 0:
                stem = f"{name}_{f:05d}"
                from PIL import Image
                Image.fromarray(img).save(out_dir / "yolo" / "images" / f"{stem}.jpg", quality=90)
                lines = [f"0 {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}"
                         for b in (pp["bbox"] for pp in ppl)]
                (out_dir / "yolo" / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    if vid:
        vid.pipe.stdin.close(); vid.pipe.wait()

    if gt:
        (out_dir / f"{name}.tracks_gt.json").write_text(json.dumps({
            "video": f"videos/{name}.mp4", "camera_id": "sim-cctv", "fps": fps, "width": w, "height": h,
            "source": "sim-ground-truth", "frames": frames_gt}))
    np.savez_compressed(out_dir / f"{name}.npz", keypoints=kps, head_underwater=under,
                        distress=np.array([p["mo"]["label"] for p in people]),
                        motions=np.array([p["name"] for p in people]),
                        keypoint_names=np.array(ds.KEYPOINTS), fps=fps, depth=depth)
    summary = {"name": name, "seconds": seconds, "fps": fps, "shallow_depth_m": depth, "deep_depth_m": DEEP, "people": [
        {**looks[i], "distress": p["mo"]["label"],
         "head_z_start": round(float(kps[i, 0, 0, 2]), 3), "head_z_end": round(float(kps[i, -1, 0, 2]), 3),
         "pct_frames_head_underwater": round(100 * float(under[i].mean()), 1),
         "first_underwater_s": round(float(np.argmax(under[i])) / fps, 2) if under[i].any() else None,
         "travel_m": round(float(np.linalg.norm(kps[i, -1, 1, :2] - kps[i, 0, 1, :2])), 2)}
        for i, p in enumerate(people)]}
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=float, default=0.9, help="shallow end depth in m (0.9 = about waist deep); deep end is 2 m")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    s = run(DEFAULT_CAST, args.seconds, args.fps, args.seed, args.depth, video=not args.no_video)
    for p in s["people"]:
        print(p["person"], p["motion"], "distress", p["distress"], "head z", p["head_z_start"], "->",
              p["head_z_end"], "underwater %", p["pct_frames_head_underwater"])


if __name__ == "__main__":
    main()
