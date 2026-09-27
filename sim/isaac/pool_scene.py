"""Shared backyard-pool scene for the Isaac Sim scripts.

Import this only after SimulationApp has been created (it imports omni and pxr).
Used by pool_replicator.py (random still images) and pool_video.py (scripted video).
"""
import math
import random

import numpy as np
import omni.kit.commands
import omni.usd
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from pool_anim import Rig, relaxed

POOL_L, POOL_W, POOL_D = 8.0, 4.0, 1.8  # meters; water surface at z = 0
HEAD_R = 0.11  # rough adult head radius, meters; head height is measured at the head center

# Work clothes nobody swims in. Matching mesh parts are hidden on every character.
HIDE_PARTS = ("hardhat", "policehat", "policeradio", "sunglasses", "idbadge", "stethoscope", "labcoat",
              "safetyvest", "vest", "gloves", "reflective__pants")

# Ambient surface waves: (amplitude m, wavelength m, direction deg, phase). Speeds follow the
# capillary-gravity dispersion of real water, so short ripples and longer swells move differently.
WAVES = [(0.0035, 1.30, 20, 0.0), (0.0028, 0.85, 75, 1.1), (0.0022, 0.55, -35, 2.3),
         (0.0018, 0.40, 130, 0.4), (0.0013, 0.28, 200, 5.0), (0.0010, 0.20, -110, 3.3)]

CHARACTERS = [
    "/Isaac/People/Characters/F_Business_02/F_Business_02.usd",
    "/Isaac/People/Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd",
    "/Isaac/People/Characters/female_adult_police_02/female_adult_police_02.usd",
    "/Isaac/People/Characters/male_adult_police_04/male_adult_police_04.usd",
    "/Isaac/People/Characters/F_Medical_01/F_Medical_01.usd",
    "/Isaac/People/Characters/M_Medical_01/M_Medical_01.usd",
]

SKIES = [f"/NVIDIA/Assets/Skies/{s}_4k.hdr" for s in [
    "Clear/noon_grass", "Clear/qwantani", "Clear/sunflowers", "Clear/syferfontein_18d_clear",
    "Clear/white_cliff_top", "Clear/mealie_road", "Cloudy/kloofendal_48d_partly_cloudy",
    "Cloudy/lakeside", "Cloudy/champagne_castle_1", "Cloudy/table_mountain_1"]]


def head_state(head_z, r=HEAD_R):
    """Label from the head center height relative to the surface (z = 0)."""
    if head_z + r <= 0.0:
        return "below"
    if head_z - r >= 0.0:
        return "above"
    return "partial"


def _dispersion(k):
    return math.sqrt(9.81 * k + 7.3e-5 * k ** 3)


class WaterSurface:
    """The pool water as one closed mesh: a fine grid on top that moves every frame (ambient
    waves plus rings spreading from each person), straight sides, flat bottom. The mean level
    stays at z = 0. Normals come from the height field so refraction and reflections wobble
    like real water."""

    def __init__(self, stage, path, mat, length, width, depth, res=0.05, inset=0.002, tile=1.5):
        nx, ny = int(round(length / res)), int(round(width / res))
        xs = np.linspace(-length / 2 + inset, length / 2 - inset, nx + 1)
        ys = np.linspace(-width / 2 + inset, width / 2 - inset, ny + 1)
        self.X, self.Y = np.meshgrid(xs, ys)  # shape (ny + 1, nx + 1)
        self.dx, self.dy = xs[1] - xs[0], ys[1] - ys[0]
        edge = np.minimum.reduce([self.X - xs[0], xs[-1] - self.X, self.Y - ys[0], ys[-1] - self.Y])
        u = np.clip(edge / 0.15, 0, 1)
        self.taper = u * u * (3 - 2 * u)  # waves fade out at the walls, so the rim stays at z = 0
        self.tile = tile

        idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
        top = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
        ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])  # counterclockwise
        n_grid, n_ring = idx.size, len(ring)
        ring_xy = np.column_stack([self.X.ravel()[ring], self.Y.ravel()[ring]])
        z_bot = -depth + inset
        self.static = np.vstack([np.column_stack([ring_xy, np.zeros(n_ring)]),
                                 np.column_stack([ring_xy, np.full(n_ring, z_bot)])]).astype(np.float32)
        top_ring, bot_ring = n_grid + np.arange(n_ring), n_grid + n_ring + np.arange(n_ring)
        nxt = np.roll(np.arange(n_ring), -1)
        sides = np.stack([top_ring, bot_ring, bot_ring[nxt], top_ring[nxt]], -1)
        bottom = bot_ring[::-1]
        self.top_fv = top.ravel()  # face-vertex -> grid vertex, for normals and UVs of the top
        indices = np.concatenate([top.ravel(), sides.ravel(), bottom])
        counts = np.concatenate([np.full(len(top), 4), np.full(n_ring, 4), [n_ring]])

        # Constant normals per face-vertex: sides point outward, bottom points down.
        d = np.roll(ring_xy, -1, axis=0) - ring_xy
        side_n = np.column_stack([d[:, 1], -d[:, 0], np.zeros(n_ring)])
        side_n /= np.linalg.norm(side_n, axis=1, keepdims=True) + 1e-12
        self.static_normals = np.vstack([np.repeat(side_n, 4, axis=0),
                                         np.tile([0.0, 0.0, -1.0], (n_ring, 1))]).astype(np.float32)
        self.static_st = np.zeros((4 * n_ring + n_ring, 2), np.float32)

        self.mesh = UsdGeom.Mesh.Define(stage, path)
        self.mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts.astype(np.int32)))
        self.mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices.astype(np.int32)))
        self.mesh.CreateSubdivisionSchemeAttr("none")
        self.mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        self.st = UsdGeom.PrimvarsAPI(self.mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                                UsdGeom.Tokens.faceVarying)
        UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(mat)
        self.update(0.0)

    def height(self, t, sources=()):
        """sources: (x, y, amplitude m, phase) per person; rings with a 0.35 m wavelength."""
        X, Y = self.X, self.Y
        h = np.zeros_like(X)
        for amp, lam, ang, ph in WAVES:
            k = 2 * math.pi / lam
            a = math.radians(ang)
            h += amp * np.sin(k * (X * math.cos(a) + Y * math.sin(a)) - _dispersion(k) * t + ph)
        k = 2 * math.pi / 0.35
        w = _dispersion(k)
        for sx, sy, amp, ph in sources:
            if amp <= 0:
                continue
            r = np.hypot(X - sx, Y - sy)
            env = np.exp(-np.maximum(r - 0.25, 0) / 0.7) * np.clip((r - 0.12) / 0.15, 0, 1)
            h += amp * env * np.sin(k * r - w * t + ph)
        return h * self.taper

    def update(self, t, sources=(), scroll=(0.02, 0.012)):
        z = self.height(t, sources)
        pts = np.column_stack([self.X.ravel(), self.Y.ravel(), z.ravel()]).astype(np.float32)
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.vstack([pts, self.static])))
        gy, gx = np.gradient(z, self.dy, self.dx)
        n = np.column_stack([-gx.ravel(), -gy.ravel(), np.ones(z.size)])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        normals = np.vstack([n[self.top_fv], self.static_normals]).astype(np.float32)
        self.mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
        uv = np.column_stack([self.X.ravel() / self.tile + scroll[0] * t, self.Y.ravel() / self.tile + scroll[1] * t])
        st = np.vstack([uv[self.top_fv], self.static_st]).astype(np.float32)
        self.st.Set(Vt.Vec2fArray.FromNumpy(st))


class PoolScene:
    def __init__(self, app):
        self.app = app
        self.assets = get_assets_root_path()
        omni.usd.get_context().new_stage()
        self.stage = stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")

        # Pool shell: floor, 4 walls, and a paved deck around it at the water level.
        tile = self.mdl_material("/World/Looks/Tile", "/NVIDIA/Materials/Base/Stone/Porcelain_Tile_4.mdl")
        deck = self.mdl_material("/World/Looks/Deck", "/NVIDIA/Materials/Base/Masonry/Brick_Pavers.mdl", 0.5)
        t = 0.2
        self.box("/World/Pool/Floor", (0, 0, -POOL_D - t / 2), (POOL_L + 2 * t, POOL_W + 2 * t, t), tile)
        for name, c, s in [
            ("WallN", (0, POOL_W / 2 + t / 2, -POOL_D / 2), (POOL_L + 2 * t, t, POOL_D)),
            ("WallS", (0, -POOL_W / 2 - t / 2, -POOL_D / 2), (POOL_L + 2 * t, t, POOL_D)),
            ("WallE", (POOL_L / 2 + t / 2, 0, -POOL_D / 2), (t, POOL_W, POOL_D)),
            ("WallW", (-POOL_L / 2 - t / 2, 0, -POOL_D / 2), (t, POOL_W, POOL_D)),
        ]:
            self.box(f"/World/Pool/{name}", c, s, tile)
        d = 3.0
        for name, c, s in [
            ("DeckN", (0, POOL_W / 2 + d / 2), (POOL_L + 2 * d, d)),
            ("DeckS", (0, -POOL_W / 2 - d / 2), (POOL_L + 2 * d, d)),
            ("DeckE", (POOL_L / 2 + d / 2, 0), (d, POOL_W)),
            ("DeckW", (-POOL_L / 2 - d / 2, 0), (d, POOL_W)),
        ]:
            self.box(f"/World/Deck/{name}", (c[0], c[1], 0.05), (s[0], s[1], 0.1), deck)

        # Water: the whole pool is a volume with NVIDIA's water material, so light
        # refracts at the surface and fades to blue with depth. A thin slab does not.
        water = self.mdl_material("/World/Looks/Water", "/NVIDIA/Materials/Base/Natural/Water.mdl", uv=False)
        # Clear pool water: absorption = -ln(color) * depth * 100 per meter, here about 0.49 (red),
        # 0.09 (green) and 0.04 (blue) per meter, close to real water. Red fades first, so deep
        # parts and submerged limbs turn turquoise. Only shows with path tracing.
        ws = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Water/Shader"))
        ws.CreateInput("transmission_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.88, 0.95))
        ws.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.007)
        self.water = WaterSurface(stage, "/World/Water", water, POOL_L, POOL_W, POOL_D)

        self.sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        self.sun_xf = self.sun.AddTransformOp()
        self.sun.CreateAngleAttr(0.53)
        self.sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
        self.sky.CreateIntensityAttr(1000)
        self.sky.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
        self.sky_tex = self.sky.CreateTextureFileAttr()

        # Characters under parent Xforms we own (the character files carry their own
        # transform ops). One matrix op per person, rebuilt every frame by set_person.
        self.people, self.ops = [], []
        for i, rel in enumerate(CHARACTERS):
            parent = UsdGeom.Xform.Define(stage, f"/World/People/person_{i}")
            self.ops.append(parent.AddTransformOp())
            self.ops[-1].Set(Gf.Matrix4d(1))
            stage.DefinePrim(f"/World/People/person_{i}/body").GetReferences().AddReference(self.assets + rel)
            self.people.append(parent.GetPrim())
        app.update()
        for p in self.people:
            for q in Usd.PrimRange(p):
                if q.IsA(UsdGeom.Mesh) and any(k in q.GetName().lower() for k in HIDE_PARTS):
                    UsdGeom.Imageable(q).MakeInvisible()

        try:
            from isaacsim.core.utils.semantics import add_labels

            for p in self.people:
                add_labels(p, labels=["person"], instance_name="class")
        except ImportError:
            from isaacsim.core.utils.semantics import add_update_semantics

            for p in self.people:
                add_update_semantics(p, "person")

        # A rig per person, plus where its skeleton sits inside the person's own space.
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        self.rigs, self.skel_xf, self.hip, self.yaw0, self.side_axis, self.heights = [], [], [], [], [], []
        self.up_axis = []
        for i, p in enumerate(self.people):
            rig = Rig(stage, p, f"/World/People/person_{i}/anim")
            xf = UsdGeom.Xformable(rig.skel_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            model = rig.fk(rig.rest)
            fwd = xf.TransformDir(rig.F)
            self.rigs.append(rig)
            self.skel_xf.append(xf)
            self.hip.append(xf.Transform(rig.pos(model, "Hip")))
            self.yaw0.append(math.degrees(math.atan2(fwd[1], fwd[0])))
            self.side_axis.append(xf.TransformDir(rig.L).GetNormalized())
            self.up_axis.append(xf.TransformDir(rig.U).GetNormalized())
            cache.Clear()
            self.heights.append(cache.ComputeWorldBound(p).ComputeAlignedRange().GetMax()[2])
        self.last = [None] * len(self.people)
        for i in range(len(self.people)):
            self.park(i)

    def mdl_material(self, path, rel_url, scale=1.0, uv=True):
        """NVIDIA MDL material from the asset server. With uv=True, textures are projected
        in world space so tiles keep their real size on stretched cubes."""
        name = rel_url.rsplit("/", 1)[-1][:-4]
        omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url=self.assets + rel_url, mtl_name=name,
                                  mtl_path=path)
        if uv:
            shader = UsdShade.Shader(self.stage.GetPrimAtPath(path + "/Shader"))
            shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(scale, scale))
        return UsdShade.Material(self.stage.GetPrimAtPath(path))

    def set_water(self, t, sources=()):
        """Move the water surface to time t. sources: (x, y, amplitude m, phase) per active person."""
        self.water.update(t, sources)

    def set_ripples(self, t):
        """Ambient waves only (kept for pool_replicator.py)."""
        self.water.update(t)

    def box(self, path, center, size, mat):
        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(cube)
        xf.SetTranslate(Gf.Vec3d(*center))
        xf.SetScale(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
        return cube

    def head_center(self, i, model):
        """Head center in the person's own space: on the head's axis, at eye level."""
        rig, xf = self.rigs[i], self.skel_xf[i]
        h, n, e = (xf.Transform(rig.pos(model, "Head")), xf.Transform(rig.pos(model, "NeckTwist02")),
                   xf.Transform(rig.eyes(model)))
        axis = (h - n).GetNormalized()
        return h + axis * Gf.Dot(e - h, axis), axis

    def set_person(self, i, x, y, head_z, yaw=0.0, tilt=0.0, pose=None, roll=0.0, scale=1.0):
        """Pose person i and place them so their head center is at height head_z.
        x, y: hip position. yaw: direction the chest faces, degrees.
        tilt: 0 = upright, 90 = lying face down, -90 = lying face up.
        roll: rotation about the body's long axis, degrees (freestyle body roll).
        scale: body size (0.6 is about a 1 m tall child). pose: dict from pool_anim (default relaxed)."""
        rig = self.rigs[i]
        model = rig.pose(pose or relaxed())
        center, _ = self.head_center(i, model)
        # Roll about the upright body's long axis first, then tilt: the same as rolling about
        # the long axis of the tilted body.
        m = (Gf.Matrix4d().SetTranslate(-self.hip[i]) * Gf.Matrix4d().SetScale(scale)
             * Gf.Matrix4d().SetRotate(Gf.Rotation(self.up_axis[i], roll))
             * Gf.Matrix4d().SetRotate(Gf.Rotation(self.side_axis[i], tilt))
             * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw - self.yaw0[i])))
        c = m.Transform(center)
        m = m * Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, head_z - c[2]))
        self.ops[i].Set(m)
        self.last[i] = (model, self.skel_xf[i] * m, m)
        return {"head_z": round(head_z, 3), "head_state": head_state(head_z, HEAD_R * scale)}

    def body_points(self, i):
        """World positions of person i's joints in their current pose, plus a point at the
        top of the head, for projecting a full-body box (underwater parts included)."""
        model, to_world, m = self.last[i]
        pts = [to_world.Transform(mj.ExtractTranslation()) for mj in model]
        center, axis = self.head_center(i, model)
        center, axis = m.Transform(center), m.TransformDir(axis)
        pts += [center + axis * 0.12, center + Gf.Vec3d(0, 0, 0.1)]  # crown, and the top of the head
        return pts

    def park(self, i):
        """Move an unused person far away. Toggling visibility instead makes the box
        annotators miss a person for the first frame they reappear."""
        self.ops[i].Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(1000 + 10 * i, 1000, 0)))

    def set_sun(self, elev, azim, intensity=3000):
        """Sun elevation and compass direction it shines from (degrees; azimuth 0 = +x, 90 = +y)."""
        e, a = math.radians(elev), math.radians(azim)
        toward = -Gf.Vec3d(math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
        self.sun_xf.Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, -1), toward)))
        self.sun.CreateIntensityAttr().Set(intensity)

    def set_light(self, rng=random, sky=None):
        self.set_sun(rng.uniform(25, 80), rng.uniform(-180, 180), rng.uniform(1500, 4000))
        self.sky_tex.Set(Sdf.AssetPath(self.assets + (sky or rng.choice(SKIES))))
