"""Procedural body motion for the Isaac Sim people (all share one Reallusion skeleton).

There are no swimming animations in the asset library, and the stock walk/idle clips
don't match these skeletons. So each frame we aim bones (upper arm, forearm, hand,
thigh, calf, spine, neck) at directions given in the body's own frame:
    f = forward (where the chest faces), l = the character's left, u = toward the head.
A pose is a dict {joint: (f, l, u)}. Motions below are functions of time.
"""
import math

from pxr import Gf, Usd, UsdSkel, Vt

# joint -> joint that marks the end of its bone
AIMS = {
    "L_Upperarm": "L_Forearm", "L_Forearm": "L_Hand", "L_Hand": "L_Mid1",
    "R_Upperarm": "R_Forearm", "R_Forearm": "R_Hand", "R_Hand": "R_Mid1",
    "L_Thigh": "L_Calf", "L_Calf": "L_Foot", "R_Thigh": "R_Calf", "R_Calf": "R_Foot",
    "Spine01": "NeckTwist01", "NeckTwist01": "Head",
}


class Rig:
    """Drives one character's skeleton through a SkelAnimation we author every frame."""

    def __init__(self, stage, root_prim, anim_path):
        self.skel_prim = next(p for p in Usd.PrimRange(root_prim) if p.IsA(UsdSkel.Skeleton))
        skel = UsdSkel.Skeleton(self.skel_prim)
        self.joints = list(skel.GetJointsAttr().Get())
        self.short = [j.rsplit("/", 1)[-1] for j in self.joints]
        self.idx = {n: i for i, n in enumerate(self.short)}
        topo = UsdSkel.Topology(self.joints)
        self.parent = [topo.GetParent(i) for i in range(len(self.joints))]
        self.rest = [Gf.Matrix4d(m) for m in skel.GetRestTransformsAttr().Get()]
        self.anim = UsdSkel.Animation.Define(stage, anim_path)
        self.anim.CreateJointsAttr(self.joints)
        UsdSkel.BindingAPI.Apply(self.skel_prim).CreateAnimationSourceRel().SetTargets([self.anim.GetPath()])
        self._paths = {}

        # Body frame in skeleton space, from the rest pose.
        m = self.fk(self.rest)
        up = (self.pos(m, "Head") - self.pos(m, "Hip")).GetNormalized()
        left = self.pos(m, "L_Upperarm") - self.pos(m, "R_Upperarm")
        left = (left - up * Gf.Dot(left, up)).GetNormalized()
        self.F, self.L, self.U = Gf.Cross(left, up).GetNormalized(), left, up

    def pos(self, m, name):
        return m[self.idx[name]].ExtractTranslation()

    def fk(self, local):
        out = []
        for i, lm in enumerate(local):
            p = self.parent[i]
            out.append(lm * out[p] if p >= 0 else Gf.Matrix4d(lm))
        return out

    def _path(self, j, c):
        """Joints from c up to (not including) j, child first."""
        key = (j, c)
        if key not in self._paths:
            path = []
            while c != j:
                path.append(c)
                c = self.parent[c]
            self._paths[key] = path
        return self._paths[key]

    def pose(self, dirs):
        """Apply a pose. Returns skeleton-space joint transforms for labels."""
        local = list(self.rest)
        model = [None] * len(local)
        for i in range(len(local)):
            p = self.parent[i]
            mp = model[p] if p >= 0 else Gf.Matrix4d(1)
            mj = local[i] * mp
            d = dirs.get(self.short[i])
            if d is not None:
                target = self.F * d[0] + self.L * d[1] + self.U * d[2]
                if target.GetLength() > 1e-6:
                    chain = Gf.Matrix4d(1)
                    for k in self._path(i, self.idx[AIMS[self.short[i]]]):
                        chain = chain * local[k]
                    pj = mj.ExtractTranslation()
                    bone = (chain * mj).ExtractTranslation() - pj
                    rot = Gf.Matrix4d().SetRotate(Gf.Rotation(bone, target))
                    mj = mj * Gf.Matrix4d().SetTranslate(-pj) * rot * Gf.Matrix4d().SetTranslate(pj)
                    local[i] = mj * mp.GetInverse()
            model[i] = mj
        self.anim.SetTransforms(Vt.Matrix4dArray(local), Usd.TimeCode.Default())
        return model

    def eyes(self, model):
        return (self.pos(model, "L_Eye") + self.pos(model, "R_Eye")) / 2


# ---- motions -------------------------------------------------------------------------

def _both(side_fn):
    """Build a pose from a function of the side sign s (+1 left, -1 right)."""
    pose = {}
    for s, pre in ((1, "L_"), (-1, "R_")):
        for joint, d in side_fn(s).items():
            pose[pre + joint] = d
    return pose


def blend(a, b, w):
    """Mix two poses; w = 0 gives a, 1 gives b."""
    w = max(0.0, min(1.0, w))
    return {k: tuple(x + (y - x) * w for x, y in zip(a[k], b[k])) for k in a}


def relaxed(t=0.0):
    p = _both(lambda s: {
        "Upperarm": (0.1, 0.25 * s, -1), "Forearm": (0.35, 0.15 * s, -1), "Hand": (0.3, 0.1 * s, -1),
        "Thigh": (0.05, 0.08 * s, -1), "Calf": (0.0, 0.05 * s, -1)})
    p.update({"Spine01": (0.0, 0, 1), "NeckTwist01": (0.05, 0, 1)})
    return p


def tread(t):
    """Treading water: hands scull side to side, legs cycle (eggbeater)."""
    w = 2 * math.pi * 0.9

    def side(s):
        a = math.sin(w * t + (0 if s > 0 else math.pi))
        b = 2 * math.pi * 0.8 * t + (0 if s > 0 else math.pi)
        return {"Upperarm": (0.55, 0.85 * s, -0.45), "Forearm": (0.75, (0.55 + 0.45 * a) * s, -0.25),
                "Hand": (0.8, (0.3 + 0.6 * a) * s, -0.45),
                "Thigh": (0.55 + 0.25 * math.sin(b), 0.35 * s, -0.8),
                "Calf": (-0.35 + 0.45 * math.cos(b), 0.25 * s, -1)}
    p = _both(side)
    p.update({"Spine01": (0.05, 0, 1), "NeckTwist01": (0.05, 0, 1)})
    return p


def freestyle(t):
    """Front crawl, in the body frame of someone lying face down (f points at the pool floor)."""
    w = 2 * math.pi * 0.55

    def side(s):
        ph = w * t + (0 if s > 0 else math.pi)
        c, sn = math.cos(ph), math.sin(ph)
        upper = (0.9 * sn, 0.2 * s, c)  # overhead -> pull under the body -> hip -> recover over the water
        fore = (0.9 * sn + 0.3, 0.15 * s, c) if sn >= 0 else (0.5, 0.2 * s, 0.8 * c)
        k = 2 * math.pi * 1.6 * t + (0 if s > 0 else math.pi)
        return {"Upperarm": upper, "Forearm": fore, "Hand": (fore[0] + 0.2, fore[1], fore[2]),
                "Thigh": (0.25 * math.sin(k), 0.08 * s, -1), "Calf": (-0.1 + 0.25 * math.sin(k - 0.8), 0.05 * s, -1)}
    p = _both(side)
    breathe = max(0.0, math.sin(w * t)) ** 4  # turn the head to breathe every other stroke
    p.update({"Spine01": (0.0, 0, 1), "NeckTwist01": (0.35, 0.45 * breathe, 1)})
    return p


def struggle(t):
    """Instinctive drowning response: arms out to the sides pressing down on the water,
    head tilted back, little useful leg action."""
    w = 2 * math.pi * 1.1
    a = math.sin(w * t)
    p = _both(lambda s: {
        "Upperarm": (0.25, 1.0 * s, 0.05 + 0.45 * a), "Forearm": (0.3, 0.9 * s, -0.1 + 0.55 * math.sin(w * t - 0.6)),
        "Hand": (0.35, 0.6 * s, -0.7 + 0.3 * math.sin(w * t - 1.0)),
        "Thigh": (0.15 + 0.1 * math.sin(0.7 * w * t + s), 0.12 * s, -1), "Calf": (-0.15, 0.05 * s, -1)})
    p.update({"Spine01": (-0.15, 0, 1), "NeckTwist01": (-0.55, 0, 1)})
    return p


def limp(t):
    """Unconscious: arms drift up and forward, head hangs, knees loose."""
    b = 0.08 * math.sin(2 * math.pi * 0.2 * t)
    p = _both(lambda s: {
        "Upperarm": (0.45, 0.55 * s, -0.1 + b), "Forearm": (0.65, 0.35 * s, 0.15 + b), "Hand": (0.6, 0.2 * s, 0.0),
        "Thigh": (0.35, 0.12 * s, -1), "Calf": (-0.25, 0.05 * s, -1)})
    p.update({"Spine01": (0.25, 0, 1), "NeckTwist01": (0.7, 0, 1)})
    return p


def streamline(t):
    """Arms locked overhead, dolphin kick: a deliberate dive."""
    k = 2 * math.pi * 1.5 * t
    p = _both(lambda s: {
        "Upperarm": (0.1, 0.08 * s, 1), "Forearm": (0.05, 0.03 * s, 1), "Hand": (0.05, 0.0, 1),
        "Thigh": (0.3 * math.sin(k), 0.05 * s, -1), "Calf": (-0.1 + 0.3 * math.sin(k - 0.7), 0.0, -1)})
    p.update({"Spine01": (0.1, 0, 1), "NeckTwist01": (0.2, 0, 1)})
    return p
