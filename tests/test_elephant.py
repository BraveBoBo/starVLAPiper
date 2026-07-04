# Unit tests for Elephant (HAMLET-style moment tokens, no special tokens).
# CPU-only, no model weights: the VLM interface is mocked via base_interface reuse;
# framework methods under test are invoked unbound with a fake `self`.
# Run: python tests/test_qwenpi_v3_moment_memory.py   (or pytest -q tests/test_qwenpi_v3_moment_memory.py)
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.model.framework.VLM4A.Elephant import (  # noqa: E402
    BlockCausalMomentMemory,
    Elephant,
)
from starVLA.model.modules.vlm.elephant_qwen3 import ElephantQwen3Interface  # noqa: E402


def _fake_base_interface(hidden_size: int = 64):
    """Minimal stand-in for an already-built _QWen3_VL_Interface."""

    class _Tok:
        def __len__(self):
            return 151936

    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden_size)))
    return SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=_Tok()), config=None)


# ---------------------------------------------------------------- T1: moment tokens Parameter
def test_t1_moment_tokens_parameter_init():
    base = _fake_base_interface(hidden_size=64)
    tok_len_before = len(base.processor.tokenizer)
    iface = ElephantQwen3Interface(base_interface=base, n_moment_tokens=4)

    assert isinstance(iface.moment_tokens, torch.nn.Parameter)
    assert iface.moment_tokens.shape == (4, 64)
    assert iface.moment_tokens.requires_grad
    std = iface.moment_tokens.detach().float().std().item()
    assert 0.015 < std < 0.025, f"init std {std} not ~0.02 (init-bug regression)"
    # tokenizer/embedding untouched: same tokenizer object, same length, model untouched
    assert len(base.processor.tokenizer) == tok_len_before
    assert iface.model is base.model  # backbone reused, not reloaded


# ---------------------------------------------------------------- T3: block-causal mask
def test_t3_block_causal_mask():
    n_q, K = 2, 3
    mem = BlockCausalMomentMemory(
        dim=32, num_moment_tokens=n_q, memory_window=K,
        num_layers=1, num_heads=4, mlp_ratio=2.0, dropout=0.0,
    ).eval()

    # positions = [0,0,1,1,2,2]; allow[i,j] = pos[j] <= pos[i]
    expected_allow = torch.tensor(
        [[pj <= pi for pj in (0, 0, 1, 1, 2, 2)] for pi in (0, 0, 1, 1, 2, 2)]
    )
    got_allow = mem.attn_mask == 0  # 0 = allow, -inf = blocked
    assert torch.equal(got_allow, expected_allow), "mask must be bidirectional-in-block, causal-across-block"

    # behavioral causality: perturbing the LAST block must not change earlier blocks' outputs
    torch.manual_seed(0)
    x1 = torch.randn(2, K * n_q, 32)
    x2 = x1.clone()
    # NOTE: must be a non-constant perturbation -- LayerNorm is invariant to a
    # constant shift (mean subtraction), so `+= 1.0` leaves outputs identical.
    x2[:, -n_q:, :] += torch.randn(2, n_q, 32)
    with torch.no_grad():
        y1, y2 = mem(x1), mem(x2)
    assert torch.allclose(y1[:, : -n_q, :], y2[:, : -n_q, :], atol=1e-5), "future block leaked into the past"
    assert not torch.allclose(y1[:, -n_q:, :], y2[:, -n_q:, :]), "current block should change"
    # current_slice helper
    assert mem.current_slice(y1).shape == (2, n_q, 32)


# ---------------------------------------------------------------- T4: K-frame training window
def _fake_framework_self(**over):
    fake = SimpleNamespace(
        moment_memory_window=3,
        moment_history_image_key="history_images",
        num_moment_tokens=2,
        _moment_memory_cache=None,
    )
    for k, v in over.items():
        setattr(fake, k, v)
    return fake


def test_t4_training_window_expansion():
    fake = _fake_framework_self()
    fn = Elephant._build_moment_training_window

    # K frames flatten oldest-first, instruction repeated per frame
    examples = [
        {"history_images": [f"s{i}_t{t}" for t in range(3)]} for i in range(2)
    ]
    imgs, instrs, bsz, k = fn(fake, examples, ["curA", "curB"], ["instA", "instB"])
    assert bsz == 2 and k == 3
    assert imgs == ["s0_t0", "s0_t1", "s0_t2", "s1_t0", "s1_t1", "s1_t2"]
    assert instrs == ["instA"] * 3 + ["instB"] * 3

    # missing key -> ValueError
    with pytest.raises(ValueError):
        fn(fake, [{}], ["c"], ["i"])
    # wrong length -> ValueError
    with pytest.raises(ValueError):
        fn(fake, [{"history_images": ["only_one"]}], ["c"], ["i"])

    # K == 1 passthrough
    fake1 = _fake_framework_self(moment_memory_window=1)
    imgs, instrs, bsz, k = fn(fake1, [{"any": 1}], ["cur"], ["inst"])
    assert (imgs, instrs, bsz, k) == (["cur"], ["inst"], 1, 1)


# ---------------------------------------------------------------- T5: rolling cache
def test_t5_rolling_cache():
    fn = Elephant._update_moment_memory_cache
    n_q, K, d = 2, 3, 8
    fake = _fake_framework_self(num_moment_tokens=n_q, moment_memory_window=K)

    step1 = torch.randn(2, n_q, d)
    cache1 = fn(fake, step1, None)
    assert cache1.shape == (2, K * n_q, d)
    assert torch.equal(cache1, step1.repeat(1, K, 1)), "first call replicates current step"

    step2 = torch.randn(2, n_q, d)
    cache2 = fn(fake, step2, None)
    assert torch.equal(cache2, torch.cat([cache1[:, n_q:, :], step2], dim=1)), "shift-append"

    # per-row reset: row 0 resets to replicate(step3), row 1 keeps shifting
    step3 = torch.randn(2, n_q, d)
    cache3 = fn(fake, step3, torch.tensor([True, False]))
    assert torch.equal(cache3[0], step3[0].repeat(K, 1))
    assert torch.equal(cache3[1], torch.cat([cache2[1, n_q:, :], step3[1]], dim=0))

    # explicit reset helper
    Elephant.reset_moment_memory(fake)
    assert fake._moment_memory_cache is None


# ---------------------------------------------------------------- T6: enabled=False no-op routing
def test_t6_disabled_noop_routing():
    calls = {}

    def _base_encode(images, instructions):
        calls["args"] = (images, instructions)
        return ["vl_embs"], "mask"

    fake = _fake_framework_self(use_moment_memory=False, _encode_vl_hidden_states=_base_encode)
    out = Elephant._encode_vl_hidden_states_with_moment_memory(
        fake, ["img"], ["inst"], examples=[{}], use_cache=False
    )
    assert out == (["vl_embs"], "mask"), "disabled path must defer to the base encoder"
    assert calls["args"] == (["img"], ["inst"])


# ---------------------------------------------------------------- T2: real-model splice (GPU, slow)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU + local Qwen3-VL weights")
def test_t2_interface_shapes_gpu():
    # Exercised more fully by the smoke script; here only the contract via a real forward
    # is asserted if someone runs pytest on a GPU machine with the base VLM available.
    pytest.skip("covered by scripts/smoke run (Task 5); enable manually when needed")


if __name__ == "__main__":
    test_t1_moment_tokens_parameter_init()
    test_t3_block_causal_mask()
    test_t4_training_window_expansion()
    test_t5_rolling_cache()
    test_t6_disabled_noop_routing()
    print("ALL CPU TESTS PASSED (T1, T3, T4, T5, T6)")
