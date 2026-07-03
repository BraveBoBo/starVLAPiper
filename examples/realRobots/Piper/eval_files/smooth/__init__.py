"""piper 执行端可切换动作平滑器包。

- 阶段② ChunkSmoother(块后处理) / 阶段③ TargetBlender(块边界续接),都继承 BaseSmoother。
- robot_config.yaml 选 method;默认 none + ensemble = 原 temporal ensemble。
- 自检:python -c "import sys; sys.path.insert(0,'deployment/piper'); from smooth import check; check()"
"""

from .base import BaseSmoother
from .chunk import (ChunkSmoother, PassthroughChunkSmoother,
                    WindowChunkSmoother, BSplineChunkSmoother)
from .blender import TargetBlender, EnsembleBlender, LatestBlender, interp_chunk
from .factory import make_chunk_smoother, make_target_blender

__all__ = ["BaseSmoother", "ChunkSmoother", "TargetBlender",
           "PassthroughChunkSmoother", "WindowChunkSmoother", "BSplineChunkSmoother",
           "EnsembleBlender", "LatestBlender", "interp_chunk",
           "make_chunk_smoother", "make_target_blender", "check"]


def check():
    """离线自检:passthrough==identity、夹爪维不被平滑、Ensemble 与原逻辑一致、latest 取最新块。"""
    from collections import deque

    import numpy as np

    JD = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    chunk = np.random.RandomState(0).randn(16, 14).astype(np.float32)
    chunk[:, 6] = 1.0; chunk[:, 13] = 0.0

    assert np.array_equal(PassthroughChunkSmoother(14, JD).smooth(chunk), chunk)
    for make in (lambda: WindowChunkSmoother(14, JD, 5, "gauss"),
                 lambda: BSplineChunkSmoother(14, JD, 8)):
        try:
            sm = make()
        except Exception as e:
            print(f"  [skip] {e.__class__.__name__}: ABPolicy/scipy 不可用,跳过该项")
            continue
        out = sm.smooth(chunk)
        assert out.shape == chunk.shape
        assert np.allclose(out[:, 6], 1.0) and np.allclose(out[:, 13], 0.0), "夹爪维被平滑了!"

    hist = deque([{"data": chunk, "birth_model_step": 0},
                  {"data": chunk + 0.1, "birth_model_step": 2}])

    def ref(history, frac, dof, decay):
        ps = np.zeros(dof); ws = 0.0; n = len(history)
        for idx, ch in enumerate(history):
            rel = frac - ch["birth_model_step"]; mi = len(ch["data"]) - 1
            rel = max(0.0, min(rel, mi)); f = int(np.floor(rel)); c = min(f + 1, mi); a = rel - f
            w = np.exp(-decay * ((n - 1) - idx))
            ps += w * ((1 - a) * ch["data"][f] + a * ch["data"][c]); ws += w
        return ps / ws

    assert np.allclose(EnsembleBlender(14, JD, 0.5).target(hist, 3.4), ref(hist, 3.4, 14, 0.5))
    assert np.allclose(LatestBlender(14, JD).target(hist, 3.4), interp_chunk(hist[-1], 3.4))
    print("smooth package check OK")
