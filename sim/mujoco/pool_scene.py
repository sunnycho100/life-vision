"""Five people in a shallow pool, each doing a different motion, rendered from a corner camera.

Reuses the water model and motions from drown_sim.py. Each person gets random skin tone,
hair, and swimwear colors so the renders look less like identical robots.

Usage: python pool_scene.py [--seconds 5] [--fps 30] [--seed 0] [--depth 0.9] [--no-video]
Output: data/sim_samples/pool_scene.npz, pool_scene.json, videos/pool_scene.mp4
"""
import argparse
import json

import mujoco
import numpy as np

import drown_sim as ds

# Extra motions for the scene. Standing people start with feet on the shallow floor.
def targets_stand(t, rng):
    sway = 0.15 * np.sin(2 * np.pi * 0.4 * t)
    return {"shoulder1_right": 0.6 + sway, "shoulder1_left": 0.6 - sway,
            "shoulder2_right": -0.3, "shoulder2_left": -0.3}


def targets_collapse(t, rng):
    # Stands for 1 s, then goes limp and folds down under the surface
    if t < 1.0:
        return targets_stand(t, rng)
    return {"knee_right": -2.0, "knee_left": -2.0, "hip_y_right": -1.5, "hip_y_left": -1.5,
            "abdomen_y": 0.4, "shoulder2_right": -1.0, "shoulder2_left": -1.0}


# name: (targets, start quat, start z ("stand" = feet on floor), lung liters(t), thrust N, distress)
SCENE_MOTIONS = {
    "stand":       (targets_stand, ds.UPRIGHT, "stand", lambda t: 3.0, 0.0, 0),
    "normal_swim": (ds.targets_swim, ds.PRONE, -0.05, lambda t: 3.0, 25.0, 0),
    "float":       (ds.targets_float, ds.SUPINE, -0.05, lambda t: 3.0, 0.0, 0),
    "idr":         (ds.targets_idr, ds.UPRIGHT, -0.15, lambda t: ds.lerp(3.0, 0.5, min(t / 20, 1)), 0.0, 1),
    "collapse":    (targets_collapse, ds.UPRIGHT, "stand", lambda t: 1.0 if t < 1 else 0.0, 0.0, 1),
}
# (x, y, heading deg) per person, pool spans x -2..8, y -2.5..2.5
# Shallow end x < SHALLOW_END_X (stand, swim, collapse), deep end beyond (float, idr)
PLACES = [(-0.5, 1.2, 0), (0.3, -1.3, 0), (5.0, -0.8, 0), (6.5, 1.3, 0), (1.5, 1.2, 90)]
SHALLOW_END_X = 3.0
DEEP = 2.0
LEG_KP, LEG_KD = 200.0, 6.0  # stiff legs so standing people do not tip over (ankles too light for this)
LEG_JOINTS = ("hip", "knee", "abdomen")
# ponytail: scripted upright torque instead of a real balance controller, swap for one if standing matters
BALANCED = {"stand": lambda t: True, "collapse": lambda t: t < 1.0}
BAL_KP, BAL_KD = 400.0, 40.0

SKIN = [(0.98, 0.84, 0.72), (0.91, 0.72, 0.58), (0.78, 0.57, 0.42), (0.58, 0.40, 0.28), (0.36, 0.24, 0.16)]
HAIR = [(0.08, 0.06, 0.05), (0.30, 0.18, 0.10), (0.75, 0.60, 0.35), (0.45, 0.20, 0.08)]
SUIT = [(0.10, 0.10, 0.12), (0.85, 0.15, 0.15), (0.10, 0.30, 0.75), (0.95, 0.55, 0.10), (0.15, 0.55, 0.35)]
SUIT_GEOMS = ("waist_lower", "butt")
TOP_GEOMS = ("torso", "waist_upper")  # rash guard for some people


def build_scene(shallow, rng):
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
                rgba=[0.15, 0.55, 0.85, 0.35], **vis)
    wall = [0.92, 0.92, 0.9, 1]
    for pos, size in [([cx, hy + .1, -hz + .1], [hx + .2, .1, hz + .1]),
                      ([cx, -hy - .1, -hz + .1], [hx + .2, .1, hz + .1]),
                      ([cx + hx + .1, 0, -hz + .1], [.1, hy, hz + .1]),
                      ([cx - hx - .1, 0, -hz + .1], [.1, hy, hz + .1])]:
        wb.add_geom(type=BOX, pos=pos, size=size, rgba=wall, **vis)
    wb.add_geom(type=BOX, pos=[cx, 0, -0.01 - DEEP - 0.2], size=[hx + 6, hy + 6, 0.01],
                rgba=[0.35, 0.35, 0.33, 1], **vis)  # deck below the rim
    wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[cx, 0, 6], dir=[0, 0, -1],
                 diffuse=[0.7, 0.7, 0.7], castshadow=0)
    wb.add_camera(name="cctv", pos=[cx - hx - .5, -hy - .5, 2.5], fovy=55,
                  xyaxes=[1, -1, 0, .45, .45, .77])

    looks = []
    for i, (name, place) in enumerate(zip(SCENE_MOTIONS, PLACES)):
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
        frame = wb.add_frame(pos=[place[0], place[1], 0],
                             quat=[np.cos(np.radians(place[2]) / 2), 0, 0, np.sin(np.radians(place[2]) / 2)])
        frame.attach_body(child.body("torso"), f"p{i}/", "")
        looks.append({"person": i, "motion": name, "skin": skin, "hair": hair, "suit": suit, "rash_guard": rash})
    return spec.compile(), looks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=float, default=0.9, help="shallow end depth in m (0.9 = about waist deep); deep end is 2 m")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    m, looks = build_scene(args.depth, rng)
    d = mujoco.MjData(m)

    people = []
    for i, (name, (tfn, quat, z0, lungs, thrust, label)) in enumerate(SCENE_MOTIONS.items()):
        pre = f"p{i}/"
        root = m.joint(pre + "root")
        a = root.qposadr[0]
        x, y, head = PLACES[i]
        yaw = [np.cos(np.radians(head) / 2), 0, 0, np.sin(np.radians(head) / 2)]
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array(yaw, float), np.array(quat, float))
        d.qpos[a:a + 3] = [x, y, (-args.depth + 1.29) if z0 == "stand" else z0]
        assert z0 != "stand" or x < SHALLOW_END_X, f"{name}: standing people belong in the shallow end"
        d.qpos[a + 3:a + 7] = q
        geoms = [g for g in range(m.ngeom)
                 if m.geom(g).name.startswith(pre) and m.geom_contype[g] != 0]
        vols = [ds.geom_volume(m, g) for g in geoms]
        joints = [j for j in range(m.njnt) if m.joint(j).name.startswith(pre) and j != root.id]
        people.append(dict(
            name=name, tfn=tfn, root_dof=root.dofadr[0], lungs=lungs, thrust=thrust, label=label, torso=m.body(pre + "torso").id,
            geoms=geoms, vols=vols,
            lung_vol=sum(v for g, v in zip(geoms, vols) if m.geom(g).name.rsplit("/", 1)[-1] in ds.LUNG_GEOMS),
            joints=[(m.joint(j).name[len(pre):], m.jnt_qposadr[j], m.jnt_dofadr[j]) for j in joints],
            kp_ids=[m.body(pre + k).id for k in ds.KEYPOINTS], head=m.body(pre + "head").id))
    mujoco.mj_forward(m, d)

    ds.OUT_DIR.mkdir(parents=True, exist_ok=True)
    video = None if args.no_video else ds.open_video(m, ds.OUT_DIR / "videos" / "pool_scene.mp4", args.fps)
    steps = round(1 / (args.fps * ds.TIMESTEP))
    T, P = int(args.seconds * args.fps), len(people)
    kps = np.zeros((P, T, len(ds.KEYPOINTS), 3), np.float32)
    under = np.zeros((P, T), bool)
    head_r = 0.09
    kp, kd = 60.0, 4.0
    for f in range(T):
        for _ in range(steps):
            d.qfrc_applied[:] = 0
            for p in people:
                tgt = p["tfn"](d.time, None)
                for jn, qa, va in p["joints"]:
                    k, c = (LEG_KP, LEG_KD) if jn.startswith(LEG_JOINTS) else (kp, kd)
                    d.qfrc_applied[va] += k * (tgt.get(jn, 0.0) - d.qpos[qa]) - c * d.qvel[va]
                ds.apply_buoyancy(m, d, p["geoms"], p["vols"], p["lungs"](d.time), p["lung_vol"])
                if BALANCED.get(p["name"], lambda t: False)(d.time):
                    R = d.xmat[p["torso"]].reshape(3, 3)
                    w = R @ d.qvel[p["root_dof"] + 3:p["root_dof"] + 6]  # angular velocity, world frame
                    torque = BAL_KP * np.cross(R[:, 2], [0, 0, 1]) - BAL_KD * w
                    mujoco.mj_applyFT(m, d, np.zeros(3), torque, d.xpos[p["torso"]], p["torso"], d.qfrc_applied)
                if p["thrust"]:
                    fwd = d.xmat[p["torso"]].reshape(3, 3)[:, 2].copy(); fwd[2] = 0
                    mujoco.mj_applyFT(m, d, p["thrust"] * fwd / (np.linalg.norm(fwd) + 1e-9), np.zeros(3),
                                      d.xpos[p["torso"]], p["torso"], d.qfrc_applied)
            mujoco.mj_step(m, d)
        for i, p in enumerate(people):
            kps[i, f] = d.xpos[p["kp_ids"]]
            under[i, f] = d.xpos[p["head"], 2] + head_r < ds.SURFACE_Z
        if not np.isfinite(kps[:, f]).all():
            raise RuntimeError(f"simulation diverged at frame {f}")
        if video:
            video.update_scene(d, camera="cctv")
            video.pipe.stdin.write(video.render().tobytes())
    if video:
        video.pipe.stdin.close(); video.pipe.wait()

    np.savez_compressed(ds.OUT_DIR / "pool_scene.npz", keypoints=kps, head_underwater=under,
                        distress=np.array([p["label"] for p in people]),
                        motions=np.array([p["name"] for p in people]),
                        keypoint_names=np.array(ds.KEYPOINTS), fps=args.fps, depth=args.depth)
    summary = {"seconds": args.seconds, "fps": args.fps, "shallow_depth_m": args.depth, "deep_depth_m": DEEP, "people": [
        {**looks[i], "distress": p["label"],
         "head_z_start": round(float(kps[i, 0, 0, 2]), 3), "head_z_end": round(float(kps[i, -1, 0, 2]), 3),
         "pct_frames_head_underwater": round(100 * float(under[i].mean()), 1)}
        for i, p in enumerate(people)]}
    (ds.OUT_DIR / "pool_scene.json").write_text(json.dumps(summary, indent=2))
    for p in summary["people"]:
        print(p["person"], p["motion"], "distress", p["distress"], "head z", p["head_z_start"], "->",
              p["head_z_end"], "underwater %", p["pct_frames_head_underwater"])


if __name__ == "__main__":
    main()
