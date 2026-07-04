# ELEPHANT K-frame history -- dataset subclass (upstream datasets.py unchanged).
# Inherits LeRobotSingleDataset and only overrides _pack_sample: the parent takes
# frame 0 per camera (datasets.py::_pack_sample); here the video delta_indices are
# K-long ([-(K-1)S, ..., -S, 0], set by the Elephant DataConfig), and the K frames
# are packed per the Elephant framework contract:
#   sample["history_images"] : list of K image groups, oldest first, current last;
#                              each group = [n_view PILs] (same layout as "image")
#   sample["image"]          : the CURRENT frame group (== history_images[-1])
# K == 1 degenerates to the parent format (no history_images key).
# Instantiated via the make_dataset factory hook in lerobot_datasets.py.
import numpy as np
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


class ElephantLeRobotSingleDataset(LeRobotSingleDataset):
    """LeRobotSingleDataset + K-frame history groups for ELEPHANT moment memory."""

    def _pack_sample(self, data: dict) -> dict:
        # ===== body mirrors LeRobotSingleDataset._pack_sample; only the video section differs =====
        video_keys = self.modality_keys["video"]
        num_frames = len(data[video_keys[0]])  # K; all video keys share delta_indices
        frame_groups = []
        for frame_idx in range(num_frames):  # oldest first
            group = []
            for video_key in video_keys:
                frame = data[video_key][frame_idx]
                group.append(Image.fromarray(frame).resize((224, 224)))
            frame_groups.append(group)

        language = data[self.modality_keys["language"][0]][0]
        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1).astype(np.float16)

        sample = {
            "action": action,
            "image": frame_groups[-1],  # current frame group
            "lang": language,
            "robot_tag": self.tag,
        }
        if num_frames > 1:
            sample["history_images"] = frame_groups

        if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
            state = []
            for state_key in self.modality_keys.get("state", []):
                state.append(data[state_key])
            if state:
                sample["state"] = np.concatenate(state, axis=1).astype(np.float16)

        return sample
