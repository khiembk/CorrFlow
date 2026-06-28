"""Unit tests for the path-correlation GP implementation.

Run from the src/ directory:
    cd src && python -m pytest tests/test_gp_path.py -v

All tests use small synthetic tensors (B=2, L=8, d=4, q=2) for speed.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from utils.sampling_utils import (
    _bridge_scale,
    _path_kernel_fn,
    _path_context_feature_single,
    _build_gp_path_single,
    build_path_at_t,
    sample_gp_path_correlation,
    sample_gp_noise_correlation,
    sample_gp_path,
)


# ---- fixtures / helpers -----------------------------------------------

B, L, d, gp_q = 2, 8, 4, 2
LENGTHSCALE = 1.0
KERNEL = 'rbf'
JITTER = 1e-4
NOISE_SCALE = 2.0

RNG = jax.random.PRNGKey(0)


def _make_inputs(rng=RNG, b=B, l=L, d_=d):
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    x0 = jax.random.normal(k1, (b, l, d_))
    src = jax.random.normal(k2, (b, l, d_))
    pth = jax.random.normal(k3, (b, l, d_))
    t   = jax.random.uniform(k4, (b,), minval=0.05, maxval=0.95)
    mask = jnp.zeros((b, l, 1))  # no conditioning
    return x0, src, pth, t, mask


# ---- Test 1: output shapes -------------------------------------------

def test_01_shape():
    """sample_gp_path_correlation returns (B,L,d), (B,L,d), None."""
    x0, src, pth, t, mask = _make_inputs()
    z, v, aux = sample_gp_path_correlation(
        x0, src, pth, t, gp_q=gp_q, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE, cond_seq_mask=mask,
    )
    assert z.shape == (B, L, d), f"z shape {z.shape}"
    assert v.shape == (B, L, d), f"v shape {v.shape}"
    assert aux is None


# ---- Test 2: build_path_at_t matches sample_gp_path_correlation z ----

def test_02_build_path_matches():
    """build_path_at_t must return the same z as sample_gp_path_correlation."""
    x0, src, pth, t, mask = _make_inputs()
    z_corr, _, _ = sample_gp_path_correlation(
        x0, src, pth, t, gp_q=gp_q, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE, cond_seq_mask=mask,
    )
    z_build = build_path_at_t(
        x0, src, pth, t, gp_q=gp_q, gp_lengthscale=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE, cond_seq_mask=mask,
    )
    np.testing.assert_allclose(z_corr, z_build, atol=1e-5)


# ---- Test 3: first token is exact linear path ------------------------

def test_03_first_token_linear():
    """Token 0 must satisfy z_t^0 = m_0(t) = (1-t)*eps^0*σ + t*x^0."""
    x0, src, pth, t, mask = _make_inputs()
    z, _, _ = sample_gp_path_correlation(
        x0, src, pth, t, gp_q=gp_q, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE,
    )
    t_e = t[:, None]  # (B, 1)
    m0_expected = (1.0 - t_e) * src[:, 0, :] * NOISE_SCALE + t_e * x0[:, 0, :]  # (B, d)
    np.testing.assert_allclose(z[:, 0, :], m0_expected, atol=1e-5,
                                err_msg="Token 0 must be the exact linear interpolation.")


# ---- Test 4: z at t=0 equals source_noise * noise_scale -------------

def test_04_endpoint_t0():
    """z_gp^i(t=0) = eps^i * noise_scale for all tokens i."""
    x0, src, pth, _, mask = _make_inputs()
    t_zero = jnp.zeros((B,))  # t = 0 exactly
    z, _, _ = sample_gp_path_correlation(
        x0, src, pth, t_zero, gp_q=gp_q, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE,
    )
    expected = src * NOISE_SCALE  # (B, L, d)
    np.testing.assert_allclose(z, expected, atol=1e-5,
                                err_msg="At t=0, z must equal source_noise * noise_scale.")


# ---- Test 5: z at t=1 equals x0 -------------------------------------

def test_05_endpoint_t1():
    """z_gp^i(t=1) = x^i for all tokens i."""
    x0, src, pth, _, mask = _make_inputs()
    t_one = jnp.ones((B,))
    z, _, _ = sample_gp_path_correlation(
        x0, src, pth, t_one, gp_q=gp_q, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE,
    )
    np.testing.assert_allclose(z, x0, atol=1e-5,
                                err_msg="At t=1, z must equal x0.")


# ---- Test 6: JVP velocity matches finite differences -----------------

def test_06_velocity_fd():
    """v_gp = dz/dt must match a central finite difference at t=0.4."""
    x0, src, pth, _, mask = _make_inputs()
    t0 = jnp.full((B,), 0.4)
    h = 1e-4

    def z_at(t_val):
        t_b = jnp.full((B,), t_val)
        z, _, _ = sample_gp_path_correlation(
            x0, src, pth, t_b, gp_q=gp_q, gp_rho=LENGTHSCALE,
            gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE,
        )
        return z

    z_hi = z_at(0.4 + h)
    z_lo = z_at(0.4 - h)
    v_fd = (z_hi - z_lo) / (2 * h)  # central difference, (B, L, d)

    _, v_jvp, _ = sample_gp_path_correlation(
        x0, src, pth, t0, gp_q=gp_q, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE,
    )

    # float32 central-difference noise: cancellation on values ~5 with h=1e-4
    # gives absolute error ~2h * 5 * eps_f32 / (2h) ≈ 5e-7/1e-4 * (num_scan_ops) ~ 5e-3.
    # The JVP is exact; the FD is noisy.  5e-3 is a safe float32 FD tolerance.
    np.testing.assert_allclose(v_jvp, v_fd, atol=5e-3,
                                err_msg="JVP velocity must match central FD at t=0.4.")


# ---- Test 7: bridge scale properties --------------------------------

def test_07_bridge_scale():
    """s(0)=0, s(1)=0, s(0.5)=1."""
    assert float(_bridge_scale(jnp.float32(0.0))) == pytest.approx(0.0)
    assert float(_bridge_scale(jnp.float32(1.0))) == pytest.approx(0.0)
    assert float(_bridge_scale(jnp.float32(0.5))) == pytest.approx(1.0)
    # Monotone increase then decrease
    assert float(_bridge_scale(jnp.float32(0.3))) < float(_bridge_scale(jnp.float32(0.5)))
    assert float(_bridge_scale(jnp.float32(0.7))) < float(_bridge_scale(jnp.float32(0.5)))


# ---- Test 8: causality -----------------------------------------------

def test_08_causality():
    """Changing token j's inputs must not affect z[:, :j, :] for any j > 0."""
    x0, src, pth, t, _ = _make_inputs()
    j = 4  # perturb token 4

    z_orig, _, _ = sample_gp_path_correlation(
        x0, src, pth, t, gp_q=gp_q, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE,
    )

    # Perturb future inputs
    x0_perturbed = x0.at[:, j:, :].set(jax.random.normal(RNG, x0[:, j:, :].shape) * 5)
    src_perturbed = src.at[:, j:, :].set(jax.random.normal(RNG, src[:, j:, :].shape) * 5)
    pth_perturbed = pth.at[:, j:, :].set(jax.random.normal(RNG, pth[:, j:, :].shape) * 5)

    z_perturbed, _, _ = sample_gp_path_correlation(
        x0_perturbed, src_perturbed, pth_perturbed, t,
        gp_q=gp_q, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE,
    )

    np.testing.assert_allclose(
        z_orig[:, :j, :], z_perturbed[:, :j, :], atol=1e-5,
        err_msg=f"Tokens 0..{j-1} must be unaffected when token {j}+ is perturbed."
    )


# ---- Test 9: no-context fallback (gp_q=1) ----------------------------

def test_09_no_context_fallback():
    """With gp_q=1, token 0 still equals the linear interpolation."""
    x0, src, pth, t, _ = _make_inputs()
    z, _, _ = sample_gp_path_correlation(
        x0, src, pth, t, gp_q=1, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE,
    )
    t_e = t[:, None]
    m0 = (1.0 - t_e) * src[:, 0, :] * NOISE_SCALE + t_e * x0[:, 0, :]
    np.testing.assert_allclose(z[:, 0, :], m0, atol=1e-5)


# ---- Test 10: segment reset with split_at ----------------------------

def test_10_split_at():
    """With split_at, token split_at is unaffected by tokens 0..split_at-1."""
    x0, src, pth, t, _ = _make_inputs(l=8)
    split = 4

    z, _, _ = sample_gp_path_correlation(
        x0, src, pth, t, gp_q=gp_q, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE, split_at=split,
    )

    # Token split_at must be the linear interp (first in its segment, no context)
    t_e = t[:, None]
    m_split = ((1.0 - t_e) * src[:, split, :] * NOISE_SCALE
               + t_e * x0[:, split, :])
    np.testing.assert_allclose(z[:, split, :], m_split, atol=1e-5,
                                err_msg="First token of second segment must be linear interp.")

    # Perturbing first segment must not change second segment
    src_perturbed = src.at[:, :split, :].set(
        jax.random.normal(RNG, src[:, :split, :].shape) * 10
    )
    z2, _, _ = sample_gp_path_correlation(
        x0, src_perturbed, pth, t, gp_q=gp_q, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE, split_at=split,
    )
    np.testing.assert_allclose(z[:, split:, :], z2[:, split:, :], atol=1e-5,
                                err_msg="Second segment must be independent of first.")


# ---- Test 11: gradient flows through v_gp ----------------------------

def test_11_gradient_flows():
    """A loss on v_gp must produce non-zero gradients w.r.t. x0."""
    x0, src, pth, t, _ = _make_inputs()

    def loss_fn(x0_):
        _, v, _ = sample_gp_path_correlation(
            x0_, src, pth, t, gp_q=gp_q, gp_rho=LENGTHSCALE,
            gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE,
        )
        return jnp.sum(v ** 2)

    grads = jax.grad(loss_fn)(x0)
    assert grads is not None
    assert not jnp.all(grads == 0), "Gradients through v_gp w.r.t. x0 must be non-zero."
    assert jnp.isfinite(grads).all(), "Gradients must be finite."


# ---- Test 12: conditioning mask pins tokens to x0 -------------------

def test_12_cond_mask():
    """Conditioned tokens (mask=1) must satisfy z == x0."""
    x0, src, pth, t, _ = _make_inputs()
    # Condition tokens 2 and 5
    mask = jnp.zeros((B, L, 1))
    mask = mask.at[:, 2, 0].set(1.0)
    mask = mask.at[:, 5, 0].set(1.0)

    z, _, _ = sample_gp_path_correlation(
        x0, src, pth, t, gp_q=gp_q, gp_rho=LENGTHSCALE,
        gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE, cond_seq_mask=mask,
    )
    np.testing.assert_allclose(z[:, 2, :], x0[:, 2, :], atol=1e-5,
                                err_msg="Conditioned token 2 must equal x0.")
    np.testing.assert_allclose(z[:, 5, :], x0[:, 5, :], atol=1e-5,
                                err_msg="Conditioned token 5 must equal x0.")


# ---- Test 13: numerical stability near endpoints ---------------------

def test_13_stability_near_endpoints():
    """z and v must be finite at t very close to 0 and 1."""
    x0, src, pth, _, _ = _make_inputs()

    for t_val in [1e-6, 1e-4, 1.0 - 1e-4, 1.0 - 1e-6]:
        t_b = jnp.full((B,), t_val)
        z, v, _ = sample_gp_path_correlation(
            x0, src, pth, t_b, gp_q=gp_q, gp_rho=LENGTHSCALE,
            gp_kernel=KERNEL, gp_jitter=JITTER, noise_scale=NOISE_SCALE,
        )
        assert jnp.isfinite(z).all(), f"z has non-finite values at t={t_val}"
        assert jnp.isfinite(v).all(), f"v has non-finite values at t={t_val}"


# ---- Test 14: kernel symmetry ----------------------------------------

def test_14_kernel_symmetry():
    """K(fa, fb) must equal K(fb, fa) for both RBF and exponential."""
    k1, k2 = jax.random.split(RNG, 2)
    F = 1 + gp_q * d
    fa = jax.random.normal(k1, (F,))
    fb = jax.random.normal(k2, (F,))

    for kernel in ('rbf', 'exponential'):
        kab = _path_kernel_fn(fa, fb, LENGTHSCALE, kernel)
        kba = _path_kernel_fn(fb, fa, LENGTHSCALE, kernel)
        np.testing.assert_allclose(
            float(kab), float(kba), atol=1e-6,
            err_msg=f"Kernel {kernel} is not symmetric."
        )
        # Self-kernel = 1
        kaa = _path_kernel_fn(fa, fa, LENGTHSCALE, kernel)
        np.testing.assert_allclose(float(kaa), 1.0, atol=1e-6,
                                    err_msg=f"Kernel {kernel}: K(fa,fa) must equal 1.")


# ---- Test 15 (bonus): alias matches original -------------------------

def test_15_alias():
    """sample_gp_noise_correlation must be sample_gp_path."""
    assert sample_gp_noise_correlation is sample_gp_path


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-v"])
