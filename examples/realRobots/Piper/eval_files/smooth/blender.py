"""
阶段③ 块边界续接:TargetBlender  (chunk_history, fractional_step) -> target (D,)。
robot_server 调 .target(委托 apply)。
"""

from abc import abstractmethod

import numpy as np

from .base import BaseSmoother


def interp_chunk(chunk, fractional_step):
    """块内按分数步线性插值取点(钳位在 [0, max_idx])。"""
    rel = fractional_step - chunk["birth_model_step"]
    max_idx = len(chunk["data"]) - 1
    rel = max(0.0, min(rel, max_idx))
    f = int(np.floor(rel)); c = min(f + 1, max_idx); a = rel - f
    return (1 - a) * chunk["data"][f] + a * chunk["data"][c]


class TargetBlender(BaseSmoother):
    name = "blender-base"

    @abstractmethod
    def apply(self, chunk_history, fractional_step):
        raise NotImplementedError

    def target(self, chunk_history, fractional_step):   # robot_server 调用入口
        return self.apply(chunk_history, fractional_step)


class EnsembleBlender(TargetBlender):
    """temporal ensemble:多块分数步插值 + 按块新旧指数加权(= 原 _ensemble_step_target)。"""
    name = "ensemble"

    def __init__(self, dof=14, joint_dims=None, ensemble_exp_decay=0.5):
        super().__init__(dof, joint_dims)
        self.decay = ensemble_exp_decay

    def apply(self, chunk_history, fractional_step):
        pos_sum = np.zeros(self.dof)
        weight_sum = 0.0
        n = len(chunk_history)
        for idx, chunk in enumerate(chunk_history):
            pos = interp_chunk(chunk, fractional_step)
            w = np.exp(-self.decay * ((n - 1) - idx))
            pos_sum += w * pos
            weight_sum += w
        return pos_sum / weight_sum if weight_sum > 0 else None


class LatestBlender(TargetBlender):
    """只取最新块的分数步插值点(无融合,对比基线)。"""
    name = "latest"

    def apply(self, chunk_history, fractional_step):
        if not chunk_history:
            return None
        return interp_chunk(chunk_history[-1], fractional_step)
