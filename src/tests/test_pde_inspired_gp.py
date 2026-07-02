"""Tests for PDEInspiredGPPath and the unchanged VectorValuedGP reference.

Run from src/:
    conda run -n corrflow python -m pytest tests/test_pde_inspired_gp.py -v

All 20 required tests are implemented in order.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from modules.pde_gp_path import VectorValuedGP, PDEInspiredGPPath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(B=2, L=8, d=16, seed=0, dtype=torch.float32):
    """Return (t, source_noise, target) with deterministic content."""
    g = torch.Generator().manual_seed(seed)
    source_noise = torch.randn(B, L, d, generator=g, dtype=dtype)
    target       = torch.randn(B, L, d, generator=g, dtype=dtype)
    t            = torch.rand(B, generator=g).clamp(0.05, 0.95)
    return t.to(dtype), source_noise, target


def _default_model(**kw) -> PDEInspiredGPPath:
    defaults = dict(log_length_scale=0.0, trainable_length_scale=False,
                    time_scale=1.0, context_scale=1.0, denominator_eps=1e-6)
    defaults.update(kw)
    return PDEInspiredGPPath(**defaults)


# ---------------------------------------------------------------------------
# Test 1: VectorValuedGP output is identical before and after adding the new
#         class (runs the reference on fixed inputs and checks reproducibility).
# ---------------------------------------------------------------------------

def test_1_vector_valued_gp_unchanged():
    """VectorValuedGP produces identical outputs on repeated calls."""
    B, d = 4, 32
    g = torch.Generator().manual_seed(99)
    t          = torch.rand(B, generator=g)
    z_driver_0 = torch.randn(B, d, generator=g)
    z_driver_1 = torch.randn(B, d, generator=g)
    z_driven_0 = torch.randn(B, d, generator=g)
    z_driven_1 = torch.randn(B, d, generator=g)

    model = VectorValuedGP(log_length_scale=0.0)

    out1_z, out1_v = model(t, z_driver_0, z_driver_1, z_driven_0, z_driven_1)
    out2_z, out2_v = model(t, z_driver_0, z_driver_1, z_driven_0, z_driven_1)

    assert torch.allclose(out1_z, out2_z), "VectorValuedGP z not reproducible"
    assert torch.allclose(out1_v, out2_v), "VectorValuedGP v not reproducible"
    assert torch.isfinite(out1_z).all()
    assert torch.isfinite(out1_v).all()


# ---------------------------------------------------------------------------
# Test 2: Output shapes
# ---------------------------------------------------------------------------

def test_2_shapes():
    B, L, d, q = 2, 8, 16, 3
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    path, velocity, aux = model(t, src, tgt, q)

    assert path.shape     == (B, L, d), f"path shape {path.shape}"
    assert velocity.shape == (B, L, d), f"velocity shape {velocity.shape}"

    aux_scalar_keys = [
        "k0", "k1", "endpoint_kernel", "denominator", "well_conditioned",
        "w0", "w1", "dw0_dt", "dw1_dt",
        "distance_to_source", "distance_to_target", "endpoint_distance", "is_finite",
    ]
    for key in aux_scalar_keys:
        assert key in aux, f"aux missing key '{key}'"
        assert aux[key].shape == (B, L), \
            f"aux['{key}'] shape {aux[key].shape}, expected ({B}, {L})"


# ---------------------------------------------------------------------------
# Test 3: First token uses linear interpolation
# ---------------------------------------------------------------------------

def test_3_first_token_linear():
    B, L, d, q = 3, 6, 8, 2
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    path, velocity, _ = model(t, src, tgt, q)

    expected_z0 = (1.0 - t[:, None]) * src[:, 0, :] + t[:, None] * tgt[:, 0, :]
    expected_v0 = tgt[:, 0, :] - src[:, 0, :]

    assert torch.allclose(path[:, 0, :], expected_z0, atol=1e-5), \
        "First token path not linear"
    assert torch.allclose(velocity[:, 0, :], expected_v0, atol=1e-5), \
        "First token velocity not x - eps"


# ---------------------------------------------------------------------------
# Test 4: Source endpoint (t=0 → z = source_noise for every token)
# ---------------------------------------------------------------------------

def test_4_source_endpoint():
    B, L, d, q = 2, 8, 16, 3
    _, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    t_zero = torch.zeros(B)
    path, _, _ = model(t_zero, src, tgt, q)

    assert torch.allclose(path.float(), src.float(), atol=1e-4), \
        f"t=0 path not equal to source_noise; max diff = {(path - src).abs().max()}"


# ---------------------------------------------------------------------------
# Test 5: Target endpoint (t=1 → z = target for every token)
# ---------------------------------------------------------------------------

def test_5_target_endpoint():
    B, L, d, q = 2, 8, 16, 3
    _, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    t_one = torch.ones(B)
    path, _, _ = model(t_one, src, tgt, q)

    assert torch.allclose(path.float(), tgt.float(), atol=1e-4), \
        f"t=1 path not equal to target; max diff = {(path - tgt).abs().max()}"


# ---------------------------------------------------------------------------
# Test 6: Analytic velocity matches finite differences (including later tokens)
# ---------------------------------------------------------------------------

def test_6_analytic_velocity_vs_finite_diff():
    B, L, d, q = 2, 10, 16, 3
    t0, src, tgt = _make_inputs(B, L, d)
    t0 = torch.full((B,), 0.4)
    h  = 1e-4
    model = _default_model()

    path_plus,  _, _ = model(t0 + h, src, tgt, q)
    path_minus, _, _ = model(t0 - h, src, tgt, q)
    v_fd              = (path_plus - path_minus) / (2.0 * h)

    _, v_analytic, _ = model(t0, src, tgt, q)

    # Check ALL tokens (including later ones which have recursive drivers)
    abs_err = (v_analytic - v_fd).abs()
    assert abs_err.max() < 5e-3, \
        f"Analytic velocity vs FD: max abs error {abs_err.max():.2e} (later tokens included)"


# ---------------------------------------------------------------------------
# Test 7: q=1 uses only the immediately previous token
# ---------------------------------------------------------------------------

def test_7_q1_single_context():
    B, L, d = 2, 6, 8
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    # Build path with q=1
    path_q1, vel_q1, _ = model(t, src, tgt, q=1)

    # Manually construct token 1 (index 1) using only token 0
    src_ctx = src[:, 0:1, :]
    tgt_ctx = tgt[:, 0:1, :]
    z_ctx   = path_q1[:, 0:1, :].detach()
    v_ctx   = vel_q1[:, 0:1, :].detach()

    z_manual, v_manual, _ = model._step(
        t=t,
        source_i=src[:, 1, :],
        target_i=tgt[:, 1, :],
        source_context=src_ctx,
        target_context=tgt_ctx,
        path_context=z_ctx,
        velocity_context=v_ctx,
    )

    assert torch.allclose(path_q1[:, 1, :], z_manual, atol=1e-5), \
        "q=1: token 1 path doesn't match manual single-context computation"


# ---------------------------------------------------------------------------
# Test 8: Multiple context sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [2, 4, 100])
def test_8_multiple_q(q):
    B, L, d = 2, 8, 16
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    path, velocity, aux = model(t, src, tgt, q=q)

    assert path.shape     == (B, L, d)
    assert velocity.shape == (B, L, d)
    assert torch.isfinite(path).all(),     f"path not finite for q={q}"
    assert torch.isfinite(velocity).all(), f"velocity not finite for q={q}"


# ---------------------------------------------------------------------------
# Test 9: Causality — modifying token j doesn't affect tokens before j
# ---------------------------------------------------------------------------

def test_9_causality():
    B, L, d, q = 2, 8, 16, 3
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    path_orig, _, _ = model(t, src, tgt, q)

    # Modify source and target of token j=4
    j = 4
    src_mod = src.clone()
    tgt_mod = tgt.clone()
    src_mod[:, j, :] += 5.0
    tgt_mod[:, j, :] += 5.0

    path_mod, _, _ = model(t, src_mod, tgt_mod, q)

    # Tokens before j must be identical
    assert torch.allclose(path_orig[:, :j, :], path_mod[:, :j, :], atol=1e-6), \
        "Modifying token j changed tokens before j (causality violated)"


# ---------------------------------------------------------------------------
# Test 10: Context window restriction [max(seg_start, i-q), i)
# ---------------------------------------------------------------------------

def test_10_context_window():
    """Token i directly accesses only context indices [max(seg_start, i-q), i).

    We verify this by intercepting what source/target slices are passed to _step
    and confirming they never reference indices outside the causal window.
    """
    B, L, d, q = 2, 10, 8, 3
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    # Record the (source_context, target_context) shapes + values passed to _step
    # at each token position.
    call_log: list = []
    original_step = model._step

    def recording_step(t, source_i, target_i, source_context, target_context,
                        path_context, velocity_context):
        call_log.append({
            "c": source_context.shape[1],
            "src_ctx": source_context.detach(),
        })
        return original_step(t=t, source_i=source_i, target_i=target_i,
                             source_context=source_context, target_context=target_context,
                             path_context=path_context, velocity_context=velocity_context)

    model._step = recording_step
    model(t, src, tgt, q)

    # call_log[k] corresponds to token k+1 (index 1, 2, … L-1; token 0 skips _step)
    for k, entry in enumerate(call_log):
        token_i = k + 1  # segment-local index (no split_at here)
        expected_c = min(token_i, q)
        assert entry["c"] == expected_c, \
            f"Token {token_i}: expected context length {expected_c}, got {entry['c']}"

        # Verify the source context values match the correct slice of src
        ctx_start = max(0, token_i - q)
        expected_src = src[:, ctx_start:token_i, :].float()
        assert torch.allclose(entry["src_ctx"], expected_src, atol=1e-6), \
            (f"Token {token_i}: source_context values don't match "
             f"src[:, {ctx_start}:{token_i}, :]")


# ---------------------------------------------------------------------------
# Test 11: Genuine driver dependence (context_scale > 0)
# ---------------------------------------------------------------------------

def test_11_driver_dependence():
    B, L, d, q = 2, 6, 8, 2
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model(context_scale=1.0)

    path_orig, _, _ = model(t, src, tgt, q)

    # Change previous tokens while keeping token i=2's own endpoints fixed
    src_mod = src.clone()
    tgt_mod = tgt.clone()
    src_mod[:, 0, :] += 3.0
    tgt_mod[:, 0, :] += 3.0
    src_mod[:, 1, :] += 3.0
    tgt_mod[:, 1, :] += 3.0

    path_mod, _, _ = model(t, src_mod, tgt_mod, q)

    # Token 2 must have changed (it uses tokens 0 and 1 as context)
    assert not torch.allclose(path_orig[:, 2, :], path_mod[:, 2, :], atol=1e-5), \
        "Token 2 path unchanged after modifying its context tokens (context_scale=1.0)"


# ---------------------------------------------------------------------------
# Test 12: Context-scale ablation (context_scale=0 → GP independent of context)
# ---------------------------------------------------------------------------

def test_12_context_scale_zero():
    B, L, d, q = 2, 6, 8, 2
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model(context_scale=0.0)

    path_orig, _, _ = model(t, src, tgt, q)

    # Changing context tokens should NOT change later tokens
    src_mod = src.clone()
    src_mod[:, 0, :] += 10.0
    src_mod[:, 1, :] += 10.0

    path_mod, _, _ = model(t, src_mod, tgt, q)

    assert torch.allclose(path_orig[:, 2, :], path_mod[:, 2, :], atol=1e-5), \
        "With context_scale=0, token 2 should not depend on context tokens"


# ---------------------------------------------------------------------------
# Test 13: Length-scale behavior — finite outputs for multiple values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("log_ell", [-1.0, 0.0, 1.0])
def test_13_length_scale(log_ell):
    B, L, d, q = 2, 8, 16, 3
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model(log_length_scale=log_ell)

    path, velocity, _ = model(t, src, tgt, q)

    assert torch.isfinite(path).all(),     f"Non-finite path at log_ell={log_ell}"
    assert torch.isfinite(velocity).all(), f"Non-finite velocity at log_ell={log_ell}"

    # Different length scales should produce different intermediate paths
    # (Check a non-first, non-boundary token)
    model2 = _default_model(log_length_scale=log_ell + 1.0)
    path2, _, _ = model2(t, src, tgt, q)
    assert not torch.allclose(path[:, 2, :], path2[:, 2, :], atol=1e-4), \
        "Different length scales gave identical intermediate paths"


# ---------------------------------------------------------------------------
# Test 14: Large embedding dimension
# ---------------------------------------------------------------------------

def test_14_large_d():
    B, L, d, q = 2, 6, 512, 2
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    path, velocity, _ = model(t, src, tgt, q)

    assert torch.isfinite(path).all(),     "Non-finite path with d=512"
    assert torch.isfinite(velocity).all(), "Non-finite velocity with d=512"


# ---------------------------------------------------------------------------
# Test 15: Reduced-precision inputs (float16 / bfloat16)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_15_reduced_precision(dtype):
    B, L, d, q = 2, 6, 16, 2
    t, src, tgt = _make_inputs(B, L, d, dtype=dtype)
    model = _default_model()

    path, velocity, _ = model(t, src, tgt, q)

    assert torch.isfinite(path).all(),     f"Non-finite path with dtype={dtype}"
    assert torch.isfinite(velocity).all(), f"Non-finite velocity with dtype={dtype}"


# ---------------------------------------------------------------------------
# Test 16: Ill-conditioned endpoint kernel triggers linear fallback
# ---------------------------------------------------------------------------

def test_16_ill_conditioned_fallback():
    """Large length scale makes endpoint_kernel ≈ 1, denominator ≈ 0."""
    B, L, d, q = 2, 6, 8, 2
    t, src, tgt = _make_inputs(B, L, d)

    # log_ell=3 → ell²=exp(6)≈403; distances small → endpoint_kernel near 1
    model = _default_model(log_length_scale=3.0, denominator_eps=0.01)

    path, velocity, aux = model(t, src, tgt, q)

    # Must not produce NaN/Inf despite ill-conditioning
    assert torch.isfinite(path).all(),     "Non-finite path despite linear fallback"
    assert torch.isfinite(velocity).all(), "Non-finite velocity despite linear fallback"

    # well_conditioned should be False for at least some non-first tokens
    wc = aux["well_conditioned"][:, 1:]  # exclude first token (always well-conditioned placeholder)
    assert not wc.all(), "Expected at least some ill-conditioned tokens with log_ell=3"


# ---------------------------------------------------------------------------
# Test 17: Determinism
# ---------------------------------------------------------------------------

def test_17_determinism():
    B, L, d, q = 2, 8, 16, 3
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    path1, vel1, _ = model(t, src, tgt, q)
    path2, vel2, _ = model(t, src, tgt, q)

    assert torch.allclose(path1, path2), "Path not deterministic"
    assert torch.allclose(vel1,  vel2),  "Velocity not deterministic"


# ---------------------------------------------------------------------------
# Test 18: Segment reset — segments are independent
# ---------------------------------------------------------------------------

def test_18_segment_reset():
    B, L, d, q = 2, 10, 8, 3
    split_at = 5
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    path_orig, _, _ = model(t, src, tgt, q, split_at=split_at)

    # Modify first segment; second segment should be unchanged
    src_mod = src.clone()
    src_mod[:, :split_at, :] += 100.0
    path_mod1, _, _ = model(t, src_mod, tgt, q, split_at=split_at)
    assert torch.allclose(
        path_orig[:, split_at:, :], path_mod1[:, split_at:, :], atol=1e-5
    ), "Changing first segment affected second segment"

    # Modify second segment; first segment should be unchanged
    tgt_mod = tgt.clone()
    tgt_mod[:, split_at:, :] += 100.0
    path_mod2, _, _ = model(t, src, tgt_mod, q, split_at=split_at)
    assert torch.allclose(
        path_orig[:, :split_at, :], path_mod2[:, :split_at, :], atol=1e-5
    ), "Changing second segment affected first segment"


# ---------------------------------------------------------------------------
# Test 19: Trainable length scale receives gradients
# ---------------------------------------------------------------------------

def test_19_trainable_length_scale_gradient():
    B, L, d, q = 2, 6, 8, 2
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model(trainable_length_scale=True)

    assert isinstance(model.log_length_scale, nn.Parameter), \
        "log_length_scale should be nn.Parameter when trainable"

    path, velocity, _ = model(t, src, tgt, q)
    loss = path.square().mean() + velocity.square().mean()
    loss.backward()

    grad = model.log_length_scale.grad
    assert grad is not None,              "log_length_scale.grad is None"
    assert torch.isfinite(grad).all(),    "log_length_scale.grad is non-finite"
    assert grad.abs() > 0,               "log_length_scale.grad is zero"


# ---------------------------------------------------------------------------
# Test 20: Finite velocity loss
# ---------------------------------------------------------------------------

def test_20_finite_velocity_loss():
    B, L, d, q = 2, 16, 512, 2
    t, src, tgt = _make_inputs(B, L, d)
    model = _default_model()

    path, velocity, _ = model(t, src, tgt, q)

    dummy_velocity = torch.zeros_like(velocity)
    loss = (dummy_velocity - velocity).square().mean()

    assert torch.isfinite(loss), f"Velocity loss is non-finite: {loss}"
    assert loss > 0,             "Velocity loss is zero (unexpected)"
