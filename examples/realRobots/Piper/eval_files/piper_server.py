"""
Piper 版 starVLA policy server —— 继承扩展,原始 deployment/model_server 代码零改动。

比原始 server_policy.py 多一项能力:当请求带 normalize_state=true 时,server 端用
训练同款 transform 对输入 state 归一化(各 state key 按自己的 normalization_mode:
min_max / binary / q99 / mean_std 自动正确),于是 client 只需传 raw state。

实现方式:
- PiperNormProcessor(PolicyNormProcessor): 补 apply_state(对称 unapply_actions)
- PiperPolicyWrapper(PolicyServerWrapper): _get_processor 改用 PiperNormProcessor;
  predict_action 加 normalize_state 开关(默认 False, 不影响其它 client)

用法(同原始 server_policy.py):
    python deployment/piper/piper_server.py --ckpt_path X --port 10093 --use_bf16
"""

import logging

import numpy as np
import torch

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.server_policy import build_argparser
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


class PiperNormProcessor(PolicyNormProcessor):
    """补 apply_state:正向归一化 state,镜像 unapply_actions(只是 state_keys + 正向 apply)。"""

    def apply_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        squeeze = state.ndim == 1
        if squeeze:
            state = state[None, :]                       # -> (N, D)

        # 按 state_keys 拆成 {key: tensor[..., dim_k]}(对称 unapply_actions 的拆分)
        data = {}
        cursor = 0
        for key in self._state_keys:
            dim_k = self._state_key_dims.get(key, 1)
            data[key] = torch.as_tensor(state[..., cursor:cursor + dim_k], dtype=torch.float32)
            cursor += dim_k
        if cursor != state.shape[-1]:
            raise ValueError(
                f"state dim 拼接不一致: sum={cursor} != {state.shape[-1]}; state_keys={self._state_keys}"
            )

        out = self._transform.apply(data)               # transform 只含 state/action 的 StateActionTransform

        parts = []
        for key in self._state_keys:
            v = out[key]
            parts.append(v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
        res = np.concatenate(parts, axis=-1).astype(np.float32)
        return res[0] if squeeze else res


class PiperPolicyWrapper(PolicyServerWrapper):
    """让 processor 用 PiperNormProcessor;predict_action 支持 normalize_state(默认关)。"""

    def _get_processor(self, unnorm_key):
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PiperNormProcessor(self._ckpt_path, unnorm_key=unnorm_key)
        return self._norm_processors[cache_key]

    def predict_action(self, examples, unnorm_key=None, normalize_state=False, **kwargs):
        if normalize_state:
            proc = self._get_processor(unnorm_key if unnorm_key is not None else self._default_unnorm_key)
            examples = [
                ({**ex, "state": proc.apply_state(np.asarray(ex["state"]))} if "state" in ex else ex)
                for ex in examples
            ]
        return super().predict_action(examples, unnorm_key=unnorm_key, **kwargs)


def main(args):
    wrapper = PiperPolicyWrapper(ckpt_path=args.ckpt_path, device="cuda", use_bf16=args.use_bf16)
    logging.info("Piper policy server on :%d  metadata=%s", args.port, wrapper.metadata)
    server = WebsocketPolicyServer(
        policy=wrapper, host="0.0.0.0", port=args.port,
        idle_timeout=args.idle_timeout, metadata=wrapper.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
