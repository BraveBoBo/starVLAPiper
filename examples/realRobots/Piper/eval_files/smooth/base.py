"""
动作平滑器基类。

piper 执行端的平滑分两个阶段(块后处理 / 块边界续接),接口不同,但共享:
- dof / joint_dims:平滑只作用关节维,其余维(夹爪)旁路;
- name:方法标识(给日志/config 用);
- reset():episode 切换时清内部状态(有 fitter/缓冲的子类重写);
- from_config(config):由 robot_config 的对应段构造(子类重写)。

各阶段的抽象(ChunkSmoother / TargetBlender)继承本类,在各自模块里定义具体的
apply 签名与实现。
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseSmoother(ABC):
    name: str = "base"

    def __init__(self, dof: int = 14, joint_dims: Optional[List[int]] = None):
        self.dof = int(dof)
        self.joint_dims = list(joint_dims) if joint_dims is not None else list(range(self.dof))

    def reset(self) -> None:
        """有内部状态(如 B-spline fitter、已执行 past 缓冲)的子类重写。"""
        pass

    @classmethod
    def from_config(cls, config):
        """从 robot_config 的对应段构造;子类按需重写。"""
        return cls(dof=getattr(config, "dof", 14), joint_dims=getattr(config, "joint_dims", None))

    @abstractmethod
    def apply(self, *args, **kwargs):
        """具体平滑;子类按阶段定义签名(块→块 / 历史+时刻→目标点)。"""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, dof={self.dof}, joint_dims={self.joint_dims})"
