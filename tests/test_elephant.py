# Unit tests for ELEPHANT (HAMLET-style moment tokens + fla unbounded recurrent memory
# + QFormer readout). CPU tests mock the VLM via base_interface reuse and invoke framework
# methods unbound with a fake `self`; recurrent-memory kernel tests need a GPU (fla/triton).
# Run: pytest -q tests/test_elephant.py   (GPU tests auto-skip without CUDA)
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.model.framework.VLM4A.Elephant import Elephant  # noqa: E402
from starVLA.model.modules.action_model.elephant_memory import (  # noqa: E402
    QFormerReadout,
    RecurrentMomentMemory,
)
from starVLA.model.modules.vlm.elephant_qwen3 import ElephantQwen3Interface  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="fla kernels need CUDA")


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


# ---------------------------------------------------------------- T3': chunk == stepwise (GPU)
@needs_gpu
def test_t3_chunk_vs_stepwise_equivalence():
    torch.manual_seed(0)
    D, B, nq, K = 128, 2, 4, 4
    mem = RecurrentMomentMemory(dim=D, num_moment_tokens=nq, num_layers=2, num_heads=2, head_dim=64)
    mem = mem.cuda().to(torch.bfloat16).eval()
    x = torch.randn(B, K * nq, D, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        y_full, _ = mem(x)  # one chunk pass over the whole window (training path)
        cache = None
        outs = []
        for t in range(K):  # step-by-step with carried state (inference path)
            y_t, cache = mem(x[:, t * nq : (t + 1) * nq, :], cache=cache, use_cache=True)
            outs.append(y_t)
    tail_full = mem.current_slice(y_full).float()
    tail_step = outs[-1].float()
    assert torch.allclose(tail_full, tail_step, atol=1e-2, rtol=1e-2), (
        f"chunk vs stepwise tail mismatch: {(tail_full - tail_step).abs().max().item()}"
    )


# ---------------------------------------------------------------- T5': state evolution + reset (GPU)
@needs_gpu
def test_t5_state_evolution_and_reset():
    torch.manual_seed(1)
    D, B, nq = 128, 2, 4
    mem = RecurrentMomentMemory(dim=D, num_moment_tokens=nq, num_layers=2, num_heads=2, head_dim=64)
    mem = mem.cuda().to(torch.bfloat16).eval()

    with torch.no_grad():
        x1 = torch.randn(B, nq, D, device="cuda", dtype=torch.bfloat16)
        y1, cache = mem(x1, cache=None, use_cache=True)
        assert mem.state_batch_size(cache) == B
        rs_after1 = cache[0]["recurrent_state"].clone()

        # same input again: carried state must change the output vs the fresh call
        y2, cache = mem(x1, cache=cache, use_cache=True)
        assert not torch.allclose(y1.float(), y2.float()), "carried state should influence the output"
        assert not torch.equal(rs_after1, cache[0]["recurrent_state"]), "state must evolve"

        # per-row reset: zero row 0, keep row 1
        mem.reset_state_rows(cache, torch.tensor([True, False], device="cuda"))
        assert cache[0]["recurrent_state"][0].abs().max().item() == 0.0
        assert cache[0]["recurrent_state"][1].abs().max().item() > 0.0


# ---------------------------------------------------------------- T4: K-frame training window
def _fake_framework_self(**over):
    fake = SimpleNamespace(
        moment_memory_window=3,
        moment_history_image_key="history_images",
        num_moment_tokens=2,
        _moment_memory_state=None,
    )
    for k, v in over.items():
        setattr(fake, k, v)
    return fake


def test_t4_training_window_expansion():
    fake = _fake_framework_self()
    fn = Elephant._build_moment_training_window

    examples = [{"history_images": [f"s{i}_t{t}" for t in range(3)]} for i in range(2)]
    imgs, instrs, bsz, k = fn(fake, examples, ["curA", "curB"], ["instA", "instB"])
    assert bsz == 2 and k == 3
    assert imgs == ["s0_t0", "s0_t1", "s0_t2", "s1_t0", "s1_t1", "s1_t2"]
    assert instrs == ["instA"] * 3 + ["instB"] * 3

    with pytest.raises(ValueError):
        fn(fake, [{}], ["c"], ["i"])
    with pytest.raises(ValueError):
        fn(fake, [{"history_images": ["only_one"]}], ["c"], ["i"])

    fake1 = _fake_framework_self(moment_memory_window=1)
    imgs, instrs, bsz, k = fn(fake1, [{"any": 1}], ["cur"], ["inst"])
    assert (imgs, instrs, bsz, k) == (["cur"], ["inst"], 1, 1)


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


# ---------------------------------------------------------------- T7: QFormer readout (CPU)
def test_t7_qformer_readout():
    torch.manual_seed(2)
    D, B, nq, m = 64, 2, 4, 4
    readout = QFormerReadout(dim=D, num_queries=m, num_layers=2, num_heads=8)
    assert len(readout.layers) == 2
    kv = torch.randn(B, nq, D)
    out = readout(kv)
    assert out.shape == (B, m, D)
    # gradient reaches the learnable queries
    out.sum().backward()
    assert readout.query_tokens.grad is not None and readout.query_tokens.grad.abs().sum() > 0


# ---------------------------------------------------------------- T2: real-model splice (GPU, slow)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU + local Qwen3-VL weights")
def test_t2_interface_shapes_gpu():
    pytest.skip("covered by the Bridge warm-start smoke script; enable manually when needed")


if __name__ == "__main__":
    test_t1_moment_tokens_parameter_init()
    test_t4_training_window_expansion()
    test_t6_disabled_noop_routing()
    test_t7_qformer_readout()
    if torch.cuda.is_available():
        test_t3_chunk_vs_stepwise_equivalence()
        test_t5_state_evolution_and_reset()
    print("ALL TESTS PASSED")
