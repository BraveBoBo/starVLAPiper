"""
可切换平滑器工厂:从 ServoConfig(含 .chunk_smoother/.target_blender/.joint_dims/.dof) 构造。
"""

from .chunk import PassthroughChunkSmoother, WindowChunkSmoother, BSplineChunkSmoother
from .blender import EnsembleBlender, LatestBlender


def make_chunk_smoother(config):
    cs = getattr(config, "chunk_smoother", None) or {}
    method = cs.get("method", "none")
    dof = getattr(config, "dof", 14)
    jd = getattr(config, "joint_dims", None)
    if method == "none":
        return PassthroughChunkSmoother(dof, jd)
    if method in ("mean", "gauss"):
        return WindowChunkSmoother(dof, jd, cs.get("window_size", 5), method)
    if method == "bspline":
        return BSplineChunkSmoother(dof, jd, cs.get("n_ctrl", 8))
    raise ValueError(f"未知 chunk_smoother.method={method}")


def make_target_blender(config):
    tb = getattr(config, "target_blender", None) or {}
    method = tb.get("method", "ensemble")
    dof = getattr(config, "dof", 14)
    jd = getattr(config, "joint_dims", None)
    if method == "ensemble":
        decay = tb.get("ensemble_exp_decay", getattr(config, "ensemble_exp_decay", 0.5))
        return EnsembleBlender(dof, jd, decay)
    if method == "latest":
        return LatestBlender(dof, jd)
    if method == "bspline_refit":
        raise NotImplementedError("bspline_refit 为二期:需维护'已执行 past'缓冲(用 BSplineFitter.refit_prefix_w)")
    raise ValueError(f"未知 target_blender.method={method}")
