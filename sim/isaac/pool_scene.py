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

POOL_L, POOL_W, POOL_D = 8.0, 4.0, 1.8  # meters; water surface at z = 0
HEAD_R = 0.11  # rough head radius, meters
HEAD_DOWN = 0.12  # head center sits this far below the top of the head

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
        self.box("/World/Water", (0, 0, -POOL_D / 2), (POOL_L, POOL_W, POOL_D), water)

        self.sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        self.sun_rot = self.sun.AddRotateXYZOp()
        self.sun.CreateAngleAttr(0.53)
        self.sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
        self.sky.CreateIntensityAttr(1000)
        self.sky.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
        self.sky_tex = self.sky.CreateTextureFileAttr()

        # Characters under parent Xforms we own (the character files carry their own
        # double-precision transform ops). Op order: move, yaw, tilt, then shift the
        # body so the pivot is its middle, not its feet.
        self.people, self.ops = [], []
        for i, rel in enumerate(CHARACTERS):
            parent = UsdGeom.Xform.Define(stage, f"/World/People/person_{i}")
            ops = (parent.AddTranslateOp(), parent.AddRotateZOp(), parent.AddRotateXOp(),
                   parent.AddTranslateOp(opSuffix="pivot"))
            self.ops.append(ops)
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

        # Height of each character in its default pose (feet at local z = 0).
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        self.heights = []
        for p, ops in zip(self.people, self.ops):
            for op in ops:
                op.Set(Gf.Vec3d(0, 0, 0) if op.GetOpType() == UsdGeom.XformOp.TypeTranslate else 0.0)
            cache.Clear()
            self.heights.append(cache.ComputeWorldBound(p).ComputeAlignedRange().GetMax()[2])
        for i in range(len(self.people)):
            self.ops[i][3].Set(Gf.Vec3d(0, 0, -self.heights[i] / 2))
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

    def box(self, path, center, size, mat):
        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(cube)
        xf.SetTranslate(Gf.Vec3d(*center))
        xf.SetScale(Gf.Vec3f(*size))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
        return cube

    def set_person(self, i, x, y, head_z, yaw=0.0, tilt=0.0):
        """Place person i so their head center is at height head_z.
        tilt: 0 = upright, 90 = lying flat (face down when yaw points the way they swim)."""
        reach = self.heights[i] / 2 - HEAD_DOWN  # body middle to head center
        center_z = head_z - math.cos(math.radians(tilt)) * reach
        move, rz, rx, _ = self.ops[i]
        move.Set(Gf.Vec3d(x, y, center_z))
        rz.Set(yaw)
        rx.Set(tilt)
        return {"head_z": round(head_z, 3), "head_top_z": round(head_z + HEAD_DOWN, 3),
                "head_state": head_state(head_z)}

    def park(self, i):
        """Move an unused person far away. Toggling visibility instead makes the box
        annotators miss a person for the first frame they reappear."""
        self.ops[i][0].Set(Gf.Vec3d(1000 + 10 * i, 1000, 0))

    def set_light(self, rng=random, sky=None):
        elev = rng.uniform(25, 80)
        self.sun_rot.Set(Gf.Vec3f(90 - elev, 0, rng.uniform(-180, 180)))
        self.sun.CreateIntensityAttr().Set(rng.uniform(1500, 4000))
        self.sky_tex.Set(Sdf.AssetPath(self.assets + (sky or rng.choice(SKIES))))
