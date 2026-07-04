# ELEPHANT phase-2 memory: unbounded recurrent moment memory + QFormer readout.
# Deliberately thin glue -- all heavy lifting is reused, nothing hand-rolled:
#   * mixer  = official fla GatedDeltaNet layer (projections, A_log/dt_bias init,
#     short-conv, gated RMSNorm, o_proj all inside; chunk kernel in training,
#     fused_recurrent automatically for short eval inputs)
#   * state  = official fla Cache (one instance spans all layers via layer_idx;
#     holds fp32 recurrent_state + short-conv conv_state)
#   * readout = starVLA's existing CrossAttentionBlock (imported, unchanged)
# Verified on this machine (fla 0.5.2, triton 3.2, RTX 4080): full-sequence chunk
# forward and step-by-step Cache inference produce bit-identical tail outputs.
from typing import Optional, Tuple

import torch
import torch.nn as nn

from starVLA.model.modules.projector.QFormer import CrossAttentionBlock


def _require_fla():
    try:
        from fla.layers import GatedDeltaNet
        from fla.models.utils import Cache
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "ELEPHANT recurrent memory requires flash-linear-attention: "
            "pip install flash-linear-attention==0.5.2"
        ) from e
    return GatedDeltaNet, Cache


class RecurrentMomentMemory(nn.Module):
    """Unbounded recurrent memory over moment tokens.

    A pre-norm residual stack of official fla ``GatedDeltaNet`` layers. Training
    runs the whole (B, K*n_q, d) window through the chunk kernel with no state;
    inference feeds one step (B, n_q, d) at a time with a persistent fla ``Cache``
    -- O(1) state, unbounded history. No FFN between mixers (the mixer already has
    a gated output projection).
    """

    def __init__(
        self,
        dim: int,
        num_moment_tokens: int,
        num_layers: int = 2,
        num_heads: int = 2,
        head_dim: int = 128,
        expand_v: float = 2.0,
    ):
        super().__init__()
        GatedDeltaNet, Cache = _require_fla()
        self._cache_cls = Cache
        self.dim = int(dim)
        self.num_moment_tokens = int(num_moment_tokens)
        self.norms = nn.ModuleList([nn.RMSNorm(self.dim) for _ in range(int(num_layers))])
        self.mixers = nn.ModuleList(
            [
                GatedDeltaNet(
                    hidden_size=self.dim,
                    num_heads=int(num_heads),
                    head_dim=int(head_dim),
                    expand_v=float(expand_v),
                    mode="chunk",
                    layer_idx=i,
                )
                for i in range(int(num_layers))
            ]
        )

    def new_cache(self):
        return self._cache_cls()

    def forward(
        self,
        x: torch.Tensor,
        cache=None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[object]]:
        """x: (B, T, dim). Returns (y (B, T, dim), cache|None)."""
        if use_cache and cache is None:
            cache = self.new_cache()
        for norm, mixer in zip(self.norms, self.mixers):
            h, _, cache = mixer(norm(x), past_key_values=cache, use_cache=use_cache)
            x = x + h
        return x, (cache if use_cache else None)

    def current_slice(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -self.num_moment_tokens :, :]

    def state_batch_size(self, cache) -> Optional[int]:
        """Batch size of the cached recurrent state, or None if empty/uninitialised."""
        if cache is None:
            return None
        try:
            layer_state = cache[0]
        except (IndexError, KeyError):
            return None
        if layer_state is None:
            return None
        rs = layer_state.get("recurrent_state")
        return None if rs is None else int(rs.shape[0])

    def reset_state_rows(self, cache, rows: torch.Tensor) -> None:
        """Zero the given batch rows (bool mask) in every layer's recurrent/conv state."""
        if cache is None:
            return
        for i in range(len(self.mixers)):
            try:
                layer_state = cache[i]
            except (IndexError, KeyError):
                continue
            if layer_state is None:
                continue
            rs = layer_state.get("recurrent_state")
            if rs is not None:
                rs[rows] = 0
            cs = layer_state.get("conv_state")
            if cs is not None:
                for c in cs:
                    if c is not None:
                        c[rows] = 0


class QFormerReadout(nn.Module):
    """m learnable queries cross-attend the memory's current-step output.

    Layers are starVLA's existing ``CrossAttentionBlock`` (query cross-attends
    KV + MLP residual); mask is always None here (all KV tokens are valid).
    """

    def __init__(self, dim: int, num_queries: int = 4, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.num_queries = int(num_queries)
        self.query_tokens = nn.Parameter(torch.empty(self.num_queries, int(dim)))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(int(dim), int(num_heads)) for _ in range(int(num_layers))]
        )
        self.final_norm = nn.LayerNorm(int(dim))

    def forward(self, memory_out: torch.Tensor) -> torch.Tensor:
        """memory_out: (B, n_kv, dim) -> (B, num_queries, dim)."""
        q = self.query_tokens.unsqueeze(0).expand(memory_out.shape[0], -1, -1).to(memory_out.dtype)
        for blk in self.layers:
            q = blk(q, memory_out, None)
        return self.final_norm(q)
