# Copyright 2026 starVLA community. All rights reserved.
"""
ELEPHANT Framework -- long-memory VLA ("an elephant never forgets")

Extension-only HAMLET-style variant of QwenPI_v3. Learnable moment tokens are
appended to the VLM sequence tail as an independent nn.Parameter via
ElephantQwen3Interface (inputs_embeds concat -- tokenizer/embedding table
untouched, HAMLET style, arXiv:2510.00695). Their post-LLM hidden states are
aggregated across a short history window, then the memory-augmented current
moment tokens are exposed to the layer-wise Action DiT.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.modules.vlm.elephant_qwen3 import ElephantQwen3Interface
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


class BlockCausalMomentMemory(nn.Module):
    """Aggregate K sets of moment tokens with bidirectional-in-block, causal-across-block attention."""

    def __init__(
        self,
        dim: int,
        num_moment_tokens: int,
        memory_window: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = int(dim)
        self.num_moment_tokens = int(num_moment_tokens)
        self.memory_window = int(memory_window)
        self.seq_len = self.num_moment_tokens * self.memory_window

        layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.dim * mlp_ratio),
            dropout=float(dropout),
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.final_norm = nn.LayerNorm(self.dim)

        positions = torch.arange(self.memory_window, dtype=torch.long).repeat_interleave(self.num_moment_tokens)
        allow = positions.unsqueeze(0) <= positions.unsqueeze(1)
        mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.float32)
        mask.masked_fill_(~allow, float("-inf"))
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, moment_sequence: torch.Tensor) -> torch.Tensor:
        if moment_sequence.dim() != 3:
            raise ValueError(f"moment_sequence must be [B, K*n_q, D], got {tuple(moment_sequence.shape)}.")
        if moment_sequence.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {moment_sequence.shape[1]}.")
        if moment_sequence.shape[2] != self.dim:
            raise ValueError(f"Expected hidden dim {self.dim}, got {moment_sequence.shape[2]}.")

        # Compute in the module's own dtype (fp32) for stability and to keep
        # nn.TransformerEncoder's fused fast path happy under bf16 autocast callers;
        # cast back to the caller's dtype so downstream concat stays consistent.
        in_dtype = moment_sequence.dtype
        x = moment_sequence.to(self.final_norm.weight.dtype)
        mask = self.attn_mask.to(device=x.device, dtype=x.dtype)
        return self.final_norm(self.encoder(x, mask=mask)).to(in_dtype)

    def current_slice(self, memory_output: torch.Tensor) -> torch.Tensor:
        return memory_output[:, -self.num_moment_tokens :, :]


@FRAMEWORK_REGISTRY.register("Elephant")
class Elephant(Qwen_PI_v3):
    """
    QwenPI_v3 with VLM-internal moment tokens.

    Training input contract when memory_window > 1:
        - ``example[moment_memory.history_image_key]``: list of K image groups,
          oldest first and current last. Each image group has the same format as
          ``example["image"]``.
    Inference uses a rolling cache from current-frame calls.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

        memory_cfg = self.config.framework.moment_memory
        self.use_moment_memory = bool(memory_cfg.enabled)
        self.num_moment_tokens = int(memory_cfg.num_moment_tokens)
        self.moment_memory_window = int(memory_cfg.memory_window)
        self.moment_history_image_key = str(memory_cfg.history_image_key)
        self.moment_reset_key = str(memory_cfg.reset_key)

        if self.use_moment_memory:
            # HAMLET-style: moment tokens live as an independent nn.Parameter inside the VLM
            # interface (appended to inputs_embeds tail); tokenizer/embedding table untouched,
            # so checkpoints stay size-compatible with the base VLM and the tokens remain
            # trainable even when the VLM is frozen. base_interface reuse avoids reloading
            # the already-built backbone.
            self.qwen_vl_interface = ElephantQwen3Interface(
                self.config,
                n_moment_tokens=self.num_moment_tokens,
                base_interface=self.qwen_vl_interface,
            )
            self.moment_memory_transformer = BlockCausalMomentMemory(
                dim=self.action_dit_hidden_dim,
                num_moment_tokens=self.num_moment_tokens,
                memory_window=self.moment_memory_window,
                num_layers=int(memory_cfg.memory_num_layers),
                num_heads=int(memory_cfg.num_heads),
                mlp_ratio=float(memory_cfg.mlp_ratio),
                dropout=float(memory_cfg.dropout),
            )
        else:
            self.moment_memory_transformer = None
        # Rolling inference cache, single shared cache => single-session/serial-env only.
        # Parallel envs would cross-contaminate; per-session isolation (keyed cache or a
        # recurrent state per session) is planned for the fla infinite-memory phase.
        self._moment_memory_cache: Optional[torch.Tensor] = None

    def _encode_flat_vl_with_moment_tokens(
        self,
        batch_images: List,
        instructions: List[str],
    ) -> tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # ElephantQwen3Interface appends the moment tokens to the inputs_embeds tail and
            # returns every layer's hidden states with the moment tail included, plus the
            # extended attention_mask.
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs.get("pixel_values", None),
                image_grid_thw=qwen_inputs.get("image_grid_thw", None),
            )
            vl_embs_list = list(qwenvl_outputs.hidden_states[-self.num_action_dit_layers :])
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
        return vl_embs_list, qwenvl_outputs.attention_mask

    def _build_moment_training_window(
        self,
        examples: List[dict],
        current_images: List,
        instructions: List[str],
    ) -> tuple[List, List[str], int, int]:
        batch_size = len(examples)
        if self.moment_memory_window == 1:
            return current_images, instructions, batch_size, 1

        flat_images = []
        flat_instructions = []
        for idx, (example, instruction) in enumerate(zip(examples, instructions)):
            if self.moment_history_image_key not in example:
                raise ValueError(
                    f"`framework.name=Elephant` requires "
                    f"`{self.moment_history_image_key}` with K={self.moment_memory_window} image groups "
                    f"for training."
                )
            history_images = example[self.moment_history_image_key]
            if len(history_images) != self.moment_memory_window:
                raise ValueError(
                    f"`{self.moment_history_image_key}` for example {idx} must have "
                    f"{self.moment_memory_window} image groups, got {len(history_images)}."
                )
            flat_images.extend(history_images)
            flat_instructions.extend([instruction] * self.moment_memory_window)
        return flat_images, flat_instructions, batch_size, self.moment_memory_window

    def _apply_moment_memory_to_flat_rows(
        self,
        flat_vl_embs_list: List[torch.Tensor],
        flat_attention_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        use_cache: bool,
        reset_memory: Optional[torch.Tensor] = None,
    ) -> tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        flat_rows, seq_len, hidden_dim = flat_vl_embs_list[-1].shape
        window_rows = flat_rows // batch_size
        if flat_rows != batch_size * window_rows:
            raise ValueError(f"Expected flat rows to be divisible by batch size {batch_size}, got {flat_rows}.")
        if window_rows not in (1, self.moment_memory_window):
            raise ValueError(
                f"Moment memory got K={window_rows}; expected 1 or {self.moment_memory_window}."
            )

        last_layer = flat_vl_embs_list[-1].view(batch_size, window_rows, seq_len, hidden_dim)
        moment_current = last_layer[:, -1, -self.num_moment_tokens :, :]

        # NOTE: when moment_memory_window == 1, inference (use_cache=True) also lands in this
        # branch because window_rows==K_target==1; that is computationally equivalent (memory
        # over the current frame only) and the rolling cache stays unused by design.
        if window_rows == self.moment_memory_window:
            moment_all = last_layer[:, :, -self.num_moment_tokens :, :]
            memory_input = moment_all.reshape(
                batch_size,
                self.moment_memory_window * self.num_moment_tokens,
                hidden_dim,
            )
        elif use_cache:
            memory_input = self._update_moment_memory_cache(moment_current, reset_memory)
        else:
            memory_input = moment_current.repeat(1, self.moment_memory_window, 1)

        memory_output = self.moment_memory_transformer(memory_input)
        memory_current = self.moment_memory_transformer.current_slice(memory_output)

        current_vl_embs_list = []
        for layer_hidden in flat_vl_embs_list:
            current_hidden = layer_hidden.view(batch_size, window_rows, seq_len, hidden_dim)[:, -1, :, :]
            current_hidden = torch.cat((current_hidden[:, : -self.num_moment_tokens, :], memory_current), dim=1)
            current_vl_embs_list.append(current_hidden)

        if flat_attention_mask is None:
            current_attention_mask = None
        else:
            current_attention_mask = flat_attention_mask.view(batch_size, window_rows, seq_len)[:, -1, :]
        return current_vl_embs_list, current_attention_mask

    def _update_moment_memory_cache(
        self,
        moment_current: torch.Tensor,
        reset_memory: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, num_moment_tokens, hidden_dim = moment_current.shape
        cache_len = self.moment_memory_window * self.num_moment_tokens
        default_cache = moment_current.repeat(1, self.moment_memory_window, 1)

        if self._moment_memory_cache is None or self._moment_memory_cache.shape[0] != batch_size:
            self._moment_memory_cache = default_cache
            return self._moment_memory_cache

        shifted_cache = torch.cat((self._moment_memory_cache[:, num_moment_tokens:, :], moment_current), dim=1)
        if reset_memory is not None and reset_memory.any():
            reset_mask = reset_memory.to(device=moment_current.device, dtype=torch.bool).view(batch_size, 1, 1)
            reset_mask = reset_mask.expand(batch_size, cache_len, hidden_dim)
            self._moment_memory_cache = torch.where(reset_mask, default_cache, shifted_cache)
        else:
            self._moment_memory_cache = shifted_cache
        return self._moment_memory_cache

    def reset_moment_memory(self) -> None:
        self._moment_memory_cache = None

    def _encode_vl_hidden_states_with_moment_memory(
        self,
        batch_images: List,
        instructions: List[str],
        *,
        examples: List[dict],
        use_cache: bool,
        reset_memory: Optional[torch.Tensor] = None,
    ) -> tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_moment_memory:
            return self._encode_vl_hidden_states(batch_images, instructions)

        if use_cache:
            flat_images = batch_images
            flat_instructions = instructions
            batch_size = len(batch_images)
        else:
            flat_images, flat_instructions, batch_size, _ = self._build_moment_training_window(
                examples,
                batch_images,
                instructions,
            )

        flat_vl_embs_list, flat_attention_mask = self._encode_flat_vl_with_moment_tokens(
            flat_images,
            flat_instructions,
        )
        return self._apply_moment_memory_to_flat_rows(
            flat_vl_embs_list,
            flat_attention_mask,
            batch_size=batch_size,
            use_cache=use_cache,
            reset_memory=reset_memory,
        )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )
        state = None

        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states_with_moment_memory(
            batch_images,
            instructions,
            examples=examples,
            use_cache=False,
        )
        base_hidden = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 16) if self.config and self.config.trainer else 4
            )

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                None,
                encoder_attention_mask=backbone_attention_mask,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )
        state = None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        reset_memory = kwargs.get("reset_memory", None)
        if reset_memory is None and self.use_moment_memory:
            reset_memory = [bool(example.get(self.moment_reset_key, False)) for example in examples]
        if reset_memory is not None:
            reset_memory = torch.as_tensor(reset_memory, device=self.qwen_vl_interface.model.device, dtype=torch.bool)

        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states_with_moment_memory(
            batch_images,
            instructions,
            examples=examples,
            use_cache=True,
            reset_memory=reset_memory,
        )
        base_hidden = vl_embs_list[-1]
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list,
                None,
                encoder_attention_mask=backbone_attention_mask,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
