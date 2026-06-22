#!/usr/bin/env python
"""Diagnose whether the GP path produces noise different from i.i.d.

Checks:
  1. Mean kernel value between adjacent context features (if ~0, GP ≈ i.i.d.)
  2. Adjacent-token cosine similarity in GP vs i.i.d. noise
  3. Cross-token correlation magnitude
  4. Posterior mu magnitude vs sigma (how much conditioning actually shifts eps)

Usage:
    cd src
    python diagnose_gp.py
"""

import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

from utils.sampling_utils import sample_gp_path

# ---- config ----
B = 8
L = 128
d = 512
gp_q = 4
gp_rho = 10.0
noise_scale = 2.0
t_val = 0.5   # also test at t=0 (pure noise) and t=0.5
seed = 0

rng = jax.random.PRNGKey(seed)

def run_diag(t_val, label):
    rng_n, rng_x = jax.random.split(jax.random.PRNGKey(seed + int(t_val * 1000)))
    iid_noise = jax.random.normal(rng_n, (B, L, d))
    x0 = jax.random.normal(rng_x, (B, L, d)) * 0.2   # small x0
    t = jnp.full((B,), t_val)

    z_gp, eps_gp, _ = sample_gp_path(
        x0=x0, noise=iid_noise, t=t,
        gp_q=gp_q, gp_rho=gp_rho, noise_scale=noise_scale,
    )

    # iid baseline: eps = innovations (before GP conditioning)
    iid_eps = iid_noise  # shape (B, L, d)

    # 1. Cosine similarity between adjacent tokens
    def adj_cos_sim(arr):
        # arr: (B, L, d)
        a = arr[:, :-1, :]   # (B, L-1, d)
        b = arr[:, 1:,  :]
        dot = jnp.sum(a * b, axis=-1)
        na = jnp.linalg.norm(a, axis=-1)
        nb = jnp.linalg.norm(b, axis=-1)
        return dot / (na * nb + 1e-8)

    cos_gp  = adj_cos_sim(eps_gp)   # (B, L-1)
    cos_iid = adj_cos_sim(iid_eps)  # (B, L-1)

    print(f"\n=== t={label} ===")
    print(f"Adjacent cosine sim  GP : mean={float(cos_gp.mean()):.4f}  std={float(cos_gp.std()):.4f}")
    print(f"Adjacent cosine sim IID : mean={float(cos_iid.mean()):.4f}  std={float(cos_iid.std()):.4f}")

    # 2. L2 distance between adjacent eps
    def adj_l2(arr):
        a = arr[:, :-1, :]
        b = arr[:, 1:,  :]
        return jnp.linalg.norm(a - b, axis=-1)

    l2_gp  = adj_l2(eps_gp)
    l2_iid = adj_l2(iid_eps)
    print(f"Adjacent L2 dist     GP : mean={float(l2_gp.mean()):.4f}")
    print(f"Adjacent L2 dist    IID : mean={float(l2_iid.mean()):.4f}")

    # 3. Estimate kernel value between adjacent context features
    #    Feature for position i: unit-norm stack of previous q z_t values (B, gp_q*d)
    #    We approximate by taking pairs of consecutive z_gp latents and computing RBF
    fd = gp_q * d
    # Build sliding windows from z_gp
    z_np = np.array(z_gp)   # (B, L, d)
    kernels = []
    for i in range(gp_q, L - 1):
        win_a = z_np[:, i - gp_q:i, :]    # (B, gp_q, d)
        win_b = z_np[:, i - gp_q + 1:i + 1, :]
        # unit-norm each latent in window
        norm_a = np.linalg.norm(win_a, axis=-1, keepdims=True) + 1e-8
        norm_b = np.linalg.norm(win_b, axis=-1, keepdims=True) + 1e-8
        fa = (win_a / norm_a).reshape(B, -1)   # (B, gp_q*d)
        fb = (win_b / norm_b).reshape(B, -1)
        sq_dist = np.sum((fa - fb) ** 2, axis=-1)   # (B,)
        k = np.exp(-sq_dist / gp_rho)
        kernels.append(k)
    kernels = np.concatenate(kernels)
    print(f"RBF kernel (adjacent context features): mean={kernels.mean():.2e}  max={kernels.max():.2e}")

    # 4. Mean / std of eps_gp vs iid
    print(f"eps_gp  mean={float(eps_gp.mean()):.4f}  std={float(eps_gp.std()):.4f}")
    print(f"iid_eps mean={float(iid_eps.mean()):.4f}  std={float(iid_eps.std()):.4f}")

    # 5. Max deviation: how much does GP conditioning shift eps?
    diff = jnp.abs(eps_gp - iid_eps)
    print(f"Mean |eps_gp - iid_eps| per dim: {float(diff.mean()):.4f}  max: {float(diff.max()):.4f}")

for t_val, label in [(0.0, "0.0"), (0.5, "0.5"), (0.9, "0.9")]:
    run_diag(t_val, label)

print("\n--- INTERPRETATION ---")
print("If RBF kernel ~ 0: GP context features are nearly orthogonal → GP ≈ i.i.d. (no real conditioning)")
print("If adj cosine sim GP >> IID: GP is genuinely introducing cross-token correlation")
print("If |eps_gp - iid_eps| ~ 0: GP conditioning has no effect → gp_rho too small for feature dimension")
