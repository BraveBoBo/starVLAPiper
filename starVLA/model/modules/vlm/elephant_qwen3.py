# ELEPHANT moment-token VLM interface (HAMLET-style mechanism, arXiv:2510.00695) -- subclass, original QWen3.py unchanged
# Ported from smooth/starVLA (verified 2026-06-30) with one adaptation for QwenPI_v3:
#   language_model is called with output_hidden_states=True and ALL layer hidden states are
#   returned (each (B, S+n_q, H), moment tail included) so the layer-wise Action DiT can consume
#   hidden_states[-num_action_dit_layers:]. Slicing off / picking the moment tail is the caller's job.
#
# Mechanism (HAMLET, arXiv:2510.00695; same family as CronusVLA arXiv:2506.19816):
#   n_q learnable moment query tokens (independent nn.Parameter -- NOT tokenizer special tokens,
#   the embedding table is untouched) are appended at the sequence tail of inputs_embeds; their
#   post-LLM hidden states summarize "what happened now". Trainable even when the VLM is frozen.
#
# splice/rope/hidden-extraction behavior copied from HF transformers 5.3:
#   site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py::Qwen3VLModel.forward (~L1269-1373)
#   helper: get_image_features / get_placeholder_mask / get_rope_index all on Qwen3VLModel.
#
# ⚠️ Version dependency (verified on transformers==5.3.0):
#   This file's forward directly calls Qwen3-VL internal implementation details (not stable public API):
#     Qwen3VLModel.{get_image_features, get_placeholder_mask, get_rope_index, language_model}
#     and the .pooler_output / .deepstack_features fields returned by get_image_features.
#   → When upgrading transformers or swapping the Qwen3-VL implementation, re-verify these against
#     modeling_qwen3_vl.py (especially mrope position and deepstack pass-through).
#   Also compatible with Qwen3.5 (model_type qwen3_5, e.g. Qwen3.5-0.8B): it exposes the same
#   get_image_features/get_placeholder_mask/get_rope_index but has NO deepstack -- handled by the
#   conditional deepstack pass-through below (verified against transformers 5.3 modeling_qwen3_5).
import types

import torch
import torch.nn as nn

from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface


class ElephantQwen3Interface(_QWen3_VL_Interface):
    """Qwen3-VL + n_q learnable moment query tokens. When base_interface is not None, reuse its already-loaded model."""

    def __init__(self, config=None, n_moment_tokens: int = 4, base_interface=None, **kwargs):
        if base_interface is not None:
            # Reuse the caller's already-built _QWen3_VL_Interface (avoid reloading the 4B weights)
            nn.Module.__init__(self)
            self.model = base_interface.model
            self.processor = base_interface.processor
            self.config = base_interface.config
        else:
            super().__init__(config, **kwargs)

        self.n_q = int(n_moment_tokens)
        hidden = self.model.config.text_config.hidden_size
        # Learnable moment query tokens (fp32 master weights, cast in forward); still trainable when VLM is frozen
        self.moment_tokens = nn.Parameter(torch.empty(self.n_q, hidden))
        nn.init.normal_(self.moment_tokens, std=0.02)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                image_grid_thw=None, mm_token_type_ids=None, **kwargs):
        assert input_ids is not None, "input_ids required for placeholder mask (matched by token id)"
        assert attention_mask is not None, "attention_mask required to locate real tail for moment tokens"
        assert self.n_q > 0, "n_moment_tokens must be > 0"
        # vlm = Qwen3VLModel (splice/rope live in this layer); line numbers refer to modeling_qwen3_vl.py
        vlm = self.model.model
        cfg = self.model.config
        B = input_ids.shape[0]
        device = input_ids.device

        # Whole hand-written Qwen forward under one bf16 autocast (embed/vision/splice/language_model consistent dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # 1) text embed (modeling L1299)
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

            # 2) vision splice -- copied from Qwen3VLModel.forward L1299-1345
            visual_pos_masks = None
            deepstack_visual_embeds = None
            if pixel_values is not None:
                vout = vlm.get_image_features(pixel_values, image_grid_thw, return_dict=True)  # L1158
                image_embeds = torch.cat(vout.pooler_output, dim=0).to(inputs_embeds.dtype)
                image_mask, _ = vlm.get_placeholder_mask(  # matched by input_ids==image_token_id, L1183
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)  # L1309
                visual_pos_masks = image_mask[..., 0]  # (B,S) bool
                # Qwen3-VL has deepstack features; Qwen3.5 (model_type qwen3_5) dropped them.
                deepstack_visual_embeds = getattr(vout, "deepstack_features", None)

            # 3) 3D mrope position_ids (3,B,S) -- get_rope_index L1037; build mm_token_type_ids on the fly if missing
            if mm_token_type_ids is None:
                mm_token_type_ids = torch.zeros_like(input_ids)
                mm_token_type_ids[input_ids == cfg.image_token_id] = 1
                mm_token_type_ids[input_ids == cfg.video_token_id] = 2
            position_ids, _ = vlm.get_rope_index(
                input_ids, mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw, attention_mask=attention_mask,
            )

            # 4) append n_q moment tokens at the tail; extend mask/visual_pos_masks/position_ids accordingly
            moment = self.moment_tokens.unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, moment], dim=1)  # (B, S+n_q, H)

            ones = attention_mask.new_ones(B, self.n_q)
            attention_mask = torch.cat([attention_mask, ones], dim=1)  # (B, S+n_q)

            if visual_pos_masks is not None:
                pad = visual_pos_masks.new_zeros(B, self.n_q)  # moment is non-visual -> deepstack count unchanged
                visual_pos_masks = torch.cat([visual_pos_masks, pad], dim=1)

            # Moment tokens follow each sample's last real token. This is more stable
            # than taking a global max over 3D M-RoPE positions because visual tokens
            # carry spatial coordinates that need not match the text tail position.
            valid = attention_mask.bool()
            seq_idx = torch.arange(input_ids.shape[1], device=device).unsqueeze(0).expand(B, -1)
            last_valid_idx = seq_idx.masked_fill(~valid[:, : input_ids.shape[1]], 0).amax(dim=1)  # (B,)
            batch_idx = torch.arange(B, device=device)
            last_pos = position_ids[:, batch_idx, last_valid_idx]  # (3,B)
            offs = torch.arange(1, self.n_q + 1, device=device)
            moment_pos = last_pos.unsqueeze(-1) + offs.view(1, 1, -1)  # (3,B,n_q)
            position_ids = torch.cat([position_ids, moment_pos], dim=2)  # (3, B, S+n_q)

            # 5) call language_model directly (already through final RMSNorm).
            #    output_hidden_states=True: the layer-wise Action DiT consumes hidden_states[-N:].
            #    deepstack kwargs only exist on Qwen3-VL's language_model; Qwen3.5 takes neither.
            lm_kwargs = {}
            if deepstack_visual_embeds is not None:
                lm_kwargs["visual_pos_masks"] = visual_pos_masks
                lm_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds
            out = vlm.language_model(
                input_ids=None, inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, position_ids=position_ids,
                use_cache=False, output_hidden_states=True,
                **lm_kwargs,
            )
        h = out.last_hidden_state  # (B, S+n_q, H)
        return types.SimpleNamespace(
            hidden_states=out.hidden_states,          # tuple of (B, S+n_q, H), moment tail INCLUDED
            attention_mask=attention_mask,            # extended (B, S+n_q)
            last_hidden_state=h[:, : -self.n_q, :],   # real token hidden (moment tail stripped)
            moment_tokens=h[:, -self.n_q :, :],       # (B, n_q, H) current-frame memory
        )
