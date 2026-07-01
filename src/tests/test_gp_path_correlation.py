"""Tests for GP path correlation functions (deterministic, path-GP variant).

Run from src/:
    conda run -n corrflow python -m pytest tests/test_gp_path_correlation.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils.sampling_utils import (
    make_gp_context_feature,
    gp_kernel_fn,
    build_gp_path_correlation_at_t,
    sample_gp_path_correlation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, L, d, q = 2, 8, 16, 4
SIGMA = 2.0


def _rand(shape, seed=0):
    rng = jax.random.PRNGKey(seed)
    return jax.random.normal(rng, shape)


@pytest.fixture
def batch():
    x0    = _rand((B, L, d), seed=0)
    eps   = _rand((B, L, d), seed=1)
    t_val = jnp.array(0.4)
    return x0, eps, t_val


# ---------------------------------------------------------------------------
# 1-3: make_gp_context_feature shape and content
# ---------------------------------------------------------------------------

def test_make_gp_context_feature_shape():
    source_i = _rand((B, d))
    target_i = _rand((B, d))
    linear_i = _rand((B, d))
    buf_z    = _rand((B, q, d))
    t        = jnp.array(0.3)
    mask     = jnp.ones((B, q))

    feat = make_gp_context_feature(source_i, target_i, linear_i, buf_z, t, mask)
    assert feat.shape == (B, 1 + 3 * d + q + q * d), \
        f"Expected shape (B, {1+3*d+q+q*d}), got {feat.shape}"


def test_make_gp_context_feature_t_first():
    """First column of feature equals the supplied t value."""
    source_i = _rand((B, d))
    target_i = _rand((B, d))
    linear_i = _rand((B, d))
    buf_z    = _rand((B, q, d))
    t        = jnp.array(0.7)
    mask     = jnp.ones((B, q))

    feat = make_gp_context_feature(source_i, target_i, linear_i, buf_z, t, mask)
    np.testing.assert_allclose(feat[:, 0], 0.7, atol=1e-6)


def test_make_gp_context_feature_normalized_parts():
    """The 3d embedding columns (source, target, linear) should be unit-norm."""
    source_i = _rand((B, d))
    target_i = _rand((B, d))
    linear_i = _rand((B, d))
    buf_z    = _rand((B, q, d))
    t        = jnp.array(0.5)
    mask     = jnp.ones((B, q))

    feat = make_gp_context_feature(source_i, target_i, linear_i, buf_z, t, mask)

    for start in [1, 1 + d, 1 + 2 * d]:
        block = feat[:, start:start + d]
        norms = jnp.linalg.norm(block, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5, err_msg=f"block at col {start} not unit-norm")


# ---------------------------------------------------------------------------
# 4-9: gp_kernel_fn
# ---------------------------------------------------------------------------

def test_gp_kernel_rbf_self():
    """RBF: K(x, x) == 1."""
    f = _rand((B, 10))
    k = gp_kernel_fn(f, f, kernel_type="rbf", lengthscale=1.0)
    np.testing.assert_allclose(k, 1.0, atol=1e-6)


def test_gp_kernel_rbf_decay():
    """RBF: K(x, y) < K(x, x) when x != y."""
    f1 = _rand((B, 10), seed=0)
    f2 = _rand((B, 10), seed=1)
    k_self  = gp_kernel_fn(f1, f1, kernel_type="rbf", lengthscale=1.0)
    k_cross = gp_kernel_fn(f1, f2, kernel_type="rbf", lengthscale=1.0)
    assert jnp.all(k_cross < k_self)


def test_gp_kernel_rbf_symmetry():
    """RBF: K(x, y) == K(y, x)."""
    f1 = _rand((B, 10), seed=2)
    f2 = _rand((B, 10), seed=3)
    np.testing.assert_allclose(
        gp_kernel_fn(f1, f2, "rbf", 1.0),
        gp_kernel_fn(f2, f1, "rbf", 1.0),
        atol=1e-6,
    )


def test_gp_kernel_exponential_self():
    """Exponential: K(x, x) == 1."""
    f = _rand((B, 10))
    k = gp_kernel_fn(f, f, kernel_type="exponential", lengthscale=1.0)
    np.testing.assert_allclose(k, 1.0, atol=1e-5)


def test_gp_kernel_exponential_symmetry():
    """Exponential: K(x, y) == K(y, x)."""
    f1 = _rand((B, 10), seed=4)
    f2 = _rand((B, 10), seed=5)
    np.testing.assert_allclose(
        gp_kernel_fn(f1, f2, "exponential", 1.0),
        gp_kernel_fn(f2, f1, "exponential", 1.0),
        atol=1e-5,
    )


def test_gp_kernel_lengthscale_effect():
    """Larger lengthscale → larger (closer to 1) kernel value for same feature distance."""
    f1 = jnp.ones((1, 10))
    f2 = jnp.zeros((1, 10))
    k_small = gp_kernel_fn(f1, f2, "rbf", lengthscale=0.5)
    k_large = gp_kernel_fn(f1, f2, "rbf", lengthscale=2.0)
    assert jnp.all(k_large > k_small), "larger ℓ should give larger K"


# ---------------------------------------------------------------------------
# 10-16: build_gp_path_correlation_at_t
# ---------------------------------------------------------------------------

def test_build_output_shape(batch):
    x0, eps, t = batch
    z_path, aux = build_gp_path_correlation_at_t(
        x0, eps, t, gp_q=q, gp_lengthscale=1.0, gp_path_strength=0.5,
    )
    assert z_path.shape == (B, L, d), f"Expected {(B,L,d)}, got {z_path.shape}"
    assert isinstance(aux, dict)


def test_build_endpoint_t0(batch):
    """At t=0 the bridge factor 4t(1-t)=0, so z_path == ε·σ exactly."""
    x0, eps, _ = batch
    z_path, _ = build_gp_path_correlation_at_t(
        x0, eps, jnp.array(0.0), gp_q=q, gp_lengthscale=1.0, gp_path_strength=1.0,
    )
    expected = eps * SIGMA
    np.testing.assert_allclose(z_path, expected, atol=1e-5)


def test_build_endpoint_t1(batch):
    """At t=1 the bridge factor 4t(1-t)=0, so z_path == x0 exactly."""
    x0, eps, _ = batch
    z_path, _ = build_gp_path_correlation_at_t(
        x0, eps, jnp.array(1.0), gp_q=q, gp_lengthscale=1.0, gp_path_strength=1.0,
    )
    np.testing.assert_allclose(z_path, x0, atol=1e-5)


def test_build_first_token_is_linear(batch):
    """Token 0 has no context, so z_path[:,0,:] == m_0(t) = t*x0 + (1-t)*ε*σ."""
    x0, eps, t = batch
    z_path, _ = build_gp_path_correlation_at_t(
        x0, eps, t, gp_q=q, gp_lengthscale=1.0, gp_path_strength=2.0,
    )
    m0 = t * x0[:, 0, :] + (1.0 - t) * eps[:, 0, :] * SIGMA
    np.testing.assert_allclose(z_path[:, 0, :], m0, atol=1e-5)


def test_build_zero_strength_gives_linear(batch):
    """gp_path_strength=0: z_path == linear midpoint everywhere."""
    x0, eps, t = batch
    z_path, _ = build_gp_path_correlation_at_t(
        x0, eps, t, gp_q=q, gp_lengthscale=1.0, gp_path_strength=0.0,
    )
    t_e = t.reshape(1, 1, 1)
    expected = t_e * x0 + (1.0 - t_e) * eps * SIGMA
    np.testing.assert_allclose(z_path, expected, atol=1e-5)


def test_build_cond_seq_mask(batch):
    """Conditioned positions (mask=1) are exactly x0."""
    x0, eps, t = batch
    cond_mask = jnp.zeros((B, L, 1), dtype=x0.dtype)
    cond_mask = cond_mask.at[:, :3, :].set(1.0)  # first 3 tokens conditioned

    z_path, _ = build_gp_path_correlation_at_t(
        x0, eps, t, gp_q=q, gp_lengthscale=1.0, gp_path_strength=1.0,
        cond_seq_mask=cond_mask,
    )
    np.testing.assert_allclose(z_path[:, :3, :], x0[:, :3, :], atol=1e-5,
                                err_msg="conditioned tokens should equal x0")


def test_build_split_at_resets_buffer(batch):
    """split_at resets the GP buffer: token split_at sees no context from tokens < split_at."""
    x0, eps, t = batch
    split = 4

    # With split_at, token at split_at has empty context → should equal linear midpoint
    z_split, _ = build_gp_path_correlation_at_t(
        x0, eps, t, gp_q=q, gp_lengthscale=1.0, gp_path_strength=2.0,
        split_at=split,
    )
    m_split = t * x0[:, split, :] + (1.0 - t) * eps[:, split, :] * SIGMA
    np.testing.assert_allclose(z_split[:, split, :], m_split, atol=1e-5,
                                err_msg="first token after split should equal linear midpoint")


# ---------------------------------------------------------------------------
# 17-20: sample_gp_path_correlation
# ---------------------------------------------------------------------------

def test_sample_output_shapes(batch):
    x0, eps, t = batch
    z_path, v_path, aux = sample_gp_path_correlation(
        x0, eps, t, gp_q=q, gp_lengthscale=1.0, gp_path_strength=0.5,
    )
    assert z_path.shape == (B, L, d)
    assert v_path.shape == (B, L, d)
    assert isinstance(aux, dict)


def test_sample_velocity_finite_difference(batch):
    """v_path ≈ (z(t+h) - z(t-h)) / 2h via central differences."""
    x0, eps, t = batch
    h = 1e-4

    z_path, v_path, _ = sample_gp_path_correlation(
        x0, eps, t, gp_q=q, gp_lengthscale=1.0, gp_path_strength=0.3,
    )
    z_fwd, _ = build_gp_path_correlation_at_t(
        x0, eps, jnp.broadcast_to(t + h, (B,)),
        gp_q=q, gp_lengthscale=1.0, gp_path_strength=0.3,
    )
    z_bwd, _ = build_gp_path_correlation_at_t(
        x0, eps, jnp.broadcast_to(t - h, (B,)),
        gp_q=q, gp_lengthscale=1.0, gp_path_strength=0.3,
    )
    v_fd = (z_fwd - z_bwd) / (2.0 * h)
    np.testing.assert_allclose(v_path, v_fd, rtol=1e-3, atol=1e-3,
                                err_msg="jvp velocity should match finite-difference")


def test_sample_zero_strength_velocity_is_linear(batch):
    """gp_path_strength=0: v_path == x0 - ε·σ (linear OT velocity) everywhere."""
    x0, eps, t = batch
    _, v_path, _ = sample_gp_path_correlation(
        x0, eps, t, gp_q=q, gp_lengthscale=1.0, gp_path_strength=0.0,
    )
    v_linear = x0 - eps * SIGMA
    np.testing.assert_allclose(v_path, v_linear, atol=1e-5)


def test_sample_endpoint_velocity_not_nan(batch):
    """Velocity should be finite at t near the endpoints (not NaN/Inf)."""
    x0, eps, _ = batch
    for t_val in [jnp.array(0.01), jnp.array(0.99)]:
        _, v_path, _ = sample_gp_path_correlation(
            x0, eps, t_val, gp_q=q, gp_lengthscale=1.0, gp_path_strength=0.5,
        )
        assert jnp.all(jnp.isfinite(v_path)), f"v_path has non-finite values at t={t_val}"
