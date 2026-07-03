"""
阶段② 块后处理:ChunkSmoother  (T,D) -> (T,D)。
只平滑 joint_dims,其余维(夹爪)旁路。robot_server 调 .smooth(委托 apply)。
"""

from abc import abstractmethod

import numpy as np

from .base import BaseSmoother


class ChunkSmoother(BaseSmoother):
    name = "chunk-base"

    def apply(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32)
        out = chunk.copy()
        out[:, self.joint_dims] = self.apply_joints(chunk[:, self.joint_dims])
        return out

    def smooth(self, chunk):           # robot_server 调用入口
        return self.apply(chunk)

    @abstractmethod
    def apply_joints(self, y):         # (T, n_joint) -> (T, n_joint)
        raise NotImplementedError


class PassthroughChunkSmoother(ChunkSmoother):
    name = "none"

    def apply_joints(self, y):
        return y


class WindowChunkSmoother(ChunkSmoother):
    """滑窗低通(mean/gauss)。
    # ponytail: 逻辑 vendor 自 ABPolicy utils/utils.py:ActionSmoother(~15 行);
    # 直接 import 它会拖入 torch/matplotlib/torchvision,机器人机不该为滑窗装这些。
    """
    name = "window"

    def __init__(self, dof=14, joint_dims=None, window_size=5, mode="gauss", sigma=None):
        super().__init__(dof, joint_dims)
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size 必须为正奇数")
        if mode == "mean":
            kernel = np.ones(window_size, np.float32) / window_size
        elif mode == "gauss":
            sigma = sigma or window_size / 2.0
            idx = np.arange(window_size) - window_size // 2
            kernel = np.exp(-0.5 * (idx / sigma) ** 2).astype(np.float32)
            kernel /= kernel.sum()
        else:
            raise ValueError("mode 只能是 mean/gauss")
        self.window_size = window_size
        self.kernel = kernel[:, None]   # (W, 1)

    def apply_joints(self, y):
        y = np.asarray(y, dtype=np.float32)
        pad = self.window_size // 2
        padded = np.pad(y, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(y)
        for t in range(y.shape[0]):
            out[t] = np.sum(padded[t:t + self.window_size] * self.kernel, axis=0)
        return out


class BSplineChunkSmoother(ChunkSmoother):
    """B-spline fit→rebuild(C2 连续),复用 ABPolicy BSplineFitter(需 scipy)。"""
    name = "bspline"

    def __init__(self, dof=14, joint_dims=None, n_ctrl=8, k=3):
        super().__init__(dof, joint_dims)
        from .ABPolicy import BSplineFitter   # 复制自 ABPolicy(子包懒导出), 需 scipy
        self.bspline_fitter_cls = BSplineFitter
        self.n_ctrl, self.k = n_ctrl, k
        self.fitter = None

    def reset(self):
        self.fitter = None

    def apply_joints(self, y):
        T = y.shape[0]
        if self.fitter is None or self.fitter.T != T:
            self.fitter = self.bspline_fitter_cls(T=T, k=self.k, n_ctrl=self.n_ctrl)
        ctrl = self.fitter.fit(y)
        y_hat, _ = self.fitter.rebuild(ctrl)
        return np.asarray(y_hat, dtype=np.float32)
