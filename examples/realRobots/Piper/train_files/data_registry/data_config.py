"""Piper (aloha-agilex) fold 数据集 — data config / robot_type / mixture。

照 examples/Robotwin 的 AgilexDataConfig,但针对 fold 数据的两个差异:
1. 布局是 [L_arm6, L_grip, R_arm6, R_grip](夹爪 interleaved, 与 info.json names 一致),
   而非 Robotwin 的 [L_joints, R_joints, L_grip, R_grip](夹爪末尾)。
2. fold 夹爪是连续值(0~0.09), 全 14 维 min_max —— **不是 binary**。

被 starVLA/dataloader/gr00t_lerobot/registry.py 自动发现并 merge。
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class PiperFoldDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.cam_d455", "video.cam_left_wrist", "video.cam_right_wrist"]
    # fold 布局: [左臂6, 左夹爪, 右臂6, 右夹爪] —— 顺序须与 meta/modality.json 切分、部署端拼接一致
    state_keys = ["state.left_arm", "state.left_gripper", "state.right_arm", "state.right_gripper"]
    action_keys = ["action.left_arm", "action.left_gripper", "action.right_arm", "action.right_gripper"]
    state_key_dims = {"state.left_arm": 6, "state.left_gripper": 1, "state.right_arm": 6, "state.right_gripper": 1}
    action_key_dims = {"action.left_arm": 6, "action.left_gripper": 1, "action.right_arm": 6, "action.right_gripper": 1}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))   # chunk 长度, 与 yaml 的 action_horizon 一致

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # fold 夹爪连续(0~0.09) → 全 14 维 min_max(关节也 min_max), 不用 binary
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={
                "state.left_arm": "min_max", "state.left_gripper": "min_max",
                "state.right_arm": "min_max", "state.right_gripper": "min_max",
            }),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={
                "action.left_arm": "min_max", "action.left_gripper": "min_max",
                "action.right_arm": "min_max", "action.right_gripper": "min_max",
            }),
        ])


ROBOT_TYPE_CONFIG_MAP = {"piper_fold": PiperFoldDataConfig()}

# mixture: (数据集子目录, 权重, robot_type) —— data_root_dir/fold 即数据集
DATASET_NAMED_MIXTURES = {
    "fold": [("fold", 1.0, "piper_fold")],
}
