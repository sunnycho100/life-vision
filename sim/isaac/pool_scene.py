"""Shared backyard-pool scene for the Isaac Sim scripts.

Import this only after SimulationApp has been created (it imports omni and pxr).
Used by pool_replicator.py (random still images) and pool_video.py (scripted video).
"""
import math
import random

import omni.kit.commands
import omni.usd
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from pool_anim import Rig, relaxed

POOL_L, POOL_W, POOL_D = 8.0, 4.0, 1.8  # meters; water surface at z = 0
HEAD_R = 0.11  # rough head radius, meters; head height is measured at the eyes

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


def head_state(head_z):
    """Label from the head center height relative to the surface (z = 0)."""
    if head_z + HEAD_R <= 0.0:
        return "below"
    if head_z - HEAD_R >= 0.0:
        return "above"
    return "partial"


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
        # Pool-cyan tint: absorption = -ln(color) * depth * 100 per meter. At 0.012, submerged
        # limbs turn blue-green within about 1 m. Only shows with path tracing.
        ws = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Water/Shader"))
        ws.CreateInput("transmission_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.85, 0.88))
        ws.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.012)
        self.water_st = self.water_box("/World/Water", water)

        self.sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        self.sun_rot = self.sun.AddRotateXYZOp()
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

    def water_box(self, path, mat):
        """The pool volume as a mesh whose UVs tile the ripple normal map about every 1.5 m.
        A Cube's UVs stretch one texture over each whole face. Returns the UV primvar so
        ripples can be scrolled over time."""
        x, y, z = POOL_L / 2, POOL_W / 2, POOL_D
        pts = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z), (-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0)]
        faces = [(4, 5, 6, 7), (0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*pts[i]) for f in faces for i in f])
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr(list(range(24)))
        mesh.CreateSubdivisionSchemeAttr("none")
        self._water_pts = [pts[i] for f in faces for i in f]
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.faceVarying)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        self.water_mesh_st = st
        self.set_ripples(0.0)
        return st

    def set_ripples(self, t, tile=1.5, speed=(0.05, 0.03)):
        """Scroll the ripple texture so the surface moves (t in seconds)."""
        self.water_mesh_st.Set([Gf.Vec2f(px / tile + speed[0] * t, py / tile + speed[1] * t)
                                for px, py, _ in self._water_pts])

    def box(self, path, center, size, mat):
        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(cube)
        xf.SetTranslate(Gf.Vec3d(*center))
        xf.SetScale(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
        return cube

    def set_person(self, i, x, y, head_z, yaw=0.0, tilt=0.0, pose=None):
        """Pose person i and place them so their eyes are at height head_z.
        x, y: hip position. yaw: direction the chest faces, degrees.
        tilt: 0 = upright, 90 = lying face down. pose: dict from pool_anim (default relaxed)."""
        rig = self.rigs[i]
        model = rig.pose(pose or relaxed())
        eyes = self.skel_xf[i].Transform(rig.eyes(model))
        m = (Gf.Matrix4d().SetTranslate(-self.hip[i])
             * Gf.Matrix4d().SetRotate(Gf.Rotation(self.side_axis[i], tilt))
             * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw - self.yaw0[i])))
        e = m.Transform(eyes)
        m = m * Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, head_z - e[2]))
        self.ops[i].Set(m)
        self.last[i] = (model, self.skel_xf[i] * m)
        return {"head_z": round(head_z, 3), "head_state": head_state(head_z)}

    def body_points(self, i):
        """World positions of person i's joints in their current pose, plus a point at the
        top of the head, for projecting a full-body box (underwater parts included)."""
        model, to_world = self.last[i]
        pts = [to_world.Transform(mj.ExtractTranslation()) for mj in model]
        rig = self.rigs[i]
        eyes, neck = to_world.Transform(rig.eyes(model)), to_world.Transform(rig.pos(model, "NeckTwist02"))
        up = (eyes - neck).GetNormalized()
        pts.append(eyes + up * 0.14)
        return pts

    def park(self, i):
        """Move an unused person far away. Toggling visibility instead makes the box
        annotators miss a person for the first frame they reappear."""
        self.ops[i].Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(1000 + 10 * i, 1000, 0)))

    def set_light(self, rng=random, sky=None):
        elev = rng.uniform(25, 80)
        self.sun_rot.Set(Gf.Vec3f(90 - elev, 0, rng.uniform(-180, 180)))
        self.sun.CreateIntensityAttr().Set(rng.uniform(1500, 4000))
        self.sky_tex.Set(Sdf.AssetPath(self.assets + (sky or rng.choice(SKIES))))
