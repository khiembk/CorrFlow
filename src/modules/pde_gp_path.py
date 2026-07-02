"""PDE-inspired causal GP probability path for token sequences.

Mathematical formulation
------------------------
For token i with causal context C_i = {max(1, i-q), ..., i-1}, c_i = |C_i|:

    R_{i,r}²(t) = τ²(t-r)² + (κ²/c_i·d) Σ_{j∈C_i} ||z_t^j - e_r^j||²
    R_{i,01}²   = τ²        + (κ²/c_i·d) Σ_{j∈C_i} ||x^j - ε^j||²

    k_{i,r}(t)  = exp(-R_{i,r}²(t) / 2ℓ²)
    a_i         = exp(-R_{i,01}² / 2ℓ²)

    w_{i,0}(t)  = (k0 - a·k1) / (1 - a²)
    w_{i,1}(t)  = (k1 - a·k0) / (1 - a²)

    z_t^i       = w_{i,0}·ε^i + w_{i,1}·x^i

    dR_{i,r}²/dt = 2τ²(t-r) + (2κ²/c_i·d) Σ_{j∈C_i} <z_t^j - e_r^j, v_t^j>
    dk_{i,r}/dt  = -k_{i,r}/(2ℓ²) · dR_{i,r}²/dt
    dw_{i,0}/dt  = (dk0 - a·dk1) / (1 - a²)
    dw_{i,1}/dt  = (dk1 - a·dk0) / (1 - a²)
    v_t^i        = dw_{i,0}·ε^i + dw_{i,1}·x^i

For the first token of each segment (no context):
    z_t^i = (1-t)·ε^i + t·x^i,   v_t^i = x^i - ε^i

Note: cross-dimensional posterior covariance is not modelled explicitly.
Scalar weights w_{i,0}, w_{i,1} are shared across all d output dimensions
under the separable covariance interpretation Cov[g(s), g(s')] = k(s,s')·B.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# ===========================================================================
# Existing reference implementation — do not modify
# ===========================================================================

class VectorValuedGP(nn.Module):
    """
    Vector-valued Gaussian Process mapping (t, z_driver) -> z_driven.

    Conditions on exactly 2 training points per sample:

        x_0 = (t=0, z_driver_0) -> y_0 = z_driven_0
        x_1 = (t=1, z_driver_1) -> y_1 = z_driven_1

    The query point is:

        x_t = (t, z_driver_t)

    The posterior mean has the form:

        z_driven_t = w0(t) * z_driven_0 + w1(t) * z_driven_1

    The instantaneous velocity is:

        u_t = dw0_dt(t) * z_driven_0 + dw1_dt(t) * z_driven_1
    """

    def __init__(self, log_length_scale: float = 0.0):
        super().__init__()

        self.alpha = float(
            torch.tensor(log_length_scale)
            .mul(2)
            .exp()
            .item()
        )

    def forward(
        self,
        t: torch.Tensor,
        z_driver_0: torch.Tensor,
        z_driver_1: torch.Tensor,
        z_driven_0: torch.Tensor,
        z_driven_1: torch.Tensor,
    ):
        B = z_driver_0.shape[0]
        t_s = t.view(B)

        d = z_driver_0[0].numel()

        with torch.no_grad():
            D_sq = (
                (z_driver_1 - z_driver_0)
                .flatten(1)
                .square()
                .sum(dim=1)
                / d
            )

            one_plus_D = 1.0 + D_sq
            ell_sq = self.alpha * one_plus_D

            k0 = torch.exp(
                -t_s.square()
                * one_plus_D
                / (2.0 * ell_sq)
            )

            k1 = torch.exp(
                -(1.0 - t_s).square()
                * one_plus_D
                / (2.0 * ell_sq)
            )

            a = torch.exp(
                -one_plus_D
                / (2.0 * ell_sq)
            )

            denom = 1.0 - a.square() + 1e-8

            w0 = (k0 - a * k1) / denom
            w1 = (k1 - a * k0) / denom

            dk0_dt = (
                -k0
                * t_s
                * one_plus_D
                / ell_sq
            )

            dk1_dt = (
                k1
                * (1.0 - t_s)
                * one_plus_D
                / ell_sq
            )

            dw0_dt = (
                dk0_dt - a * dk1_dt
            ) / denom

            dw1_dt = (
                dk1_dt - a * dk0_dt
            ) / denom

        shape = (B,) + (1,) * (
            z_driven_0.ndim - 1
        )

        z_driven_t = (
            w0.view(shape) * z_driven_0
            + w1.view(shape) * z_driven_1
        )

        u_ins_driven = (
            dw0_dt.view(shape) * z_driven_0
            + dw1_dt.view(shape) * z_driven_1
        )

        return z_driven_t, u_ins_driven


# ===========================================================================
# New implementation: PDE-inspired causal GP path
# ===========================================================================

class PDEInspiredGPPath(nn.Module):
    """
    Causal deterministic GP path for token sequences.

    Token i is driven jointly by the previous q token trajectories.

    The GP has two observations:

        (time=0, previous-token source endpoints)
            -> token i source endpoint

        (time=1, previous-token target endpoints)
            -> token i target endpoint

    The intermediate query uses the recursively generated previous-token
    paths at the current time.

    The path and instantaneous velocity are calculated analytically.
    """

    def __init__(
        self,
        log_length_scale: float = 0.0,
        trainable_length_scale: bool = False,
        time_scale: float = 1.0,
        context_scale: float = 1.0,
        denominator_eps: float = 1e-6,
    ):
        super().__init__()

        if time_scale <= 0:
            raise ValueError(f"time_scale must be > 0, got {time_scale}")
        if context_scale < 0:
            raise ValueError(f"context_scale must be >= 0, got {context_scale}")
        if denominator_eps <= 0:
            raise ValueError(f"denominator_eps must be > 0, got {denominator_eps}")

        self.time_scale = float(time_scale)
        self.context_scale = float(context_scale)
        self.denominator_eps = float(denominator_eps)

        log_ell_t = torch.tensor(float(log_length_scale))
        if trainable_length_scale:
            self.log_length_scale = nn.Parameter(log_ell_t)
        else:
            self.register_buffer("log_length_scale", log_ell_t)

    def _length_scale_sq(self) -> torch.Tensor:
        log_ell = torch.clamp(self.log_length_scale, min=-5.0, max=3.0)
        return torch.exp(2.0 * log_ell)

    # -------------------------------------------------------------------
    # Internal per-token GP step
    # -------------------------------------------------------------------

    def _step(
        self,
        t: torch.Tensor,
        source_i: torch.Tensor,
        target_i: torch.Tensor,
        source_context: torch.Tensor,
        target_context: torch.Tensor,
        path_context: torch.Tensor,
        velocity_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute path z_i and velocity v_i for one token given causal context.

        Args:
            t:                Shape (B,).
            source_i:         Shape (B, d)  — ε^i.
            target_i:         Shape (B, d)  — x^i.
            source_context:   Shape (B, c, d) — ε for context tokens.
            target_context:   Shape (B, c, d) — x for context tokens.
            path_context:     Shape (B, c, d) — z_t for context tokens.
            velocity_context: Shape (B, c, d) — dz/dt for context tokens.

        Returns:
            z_i:   Shape (B, d).
            v_i:   Shape (B, d).
            aux_i: Per-token diagnostic dict.
        """
        _, c, d = source_context.shape

        compute_dtype = torch.float32

        t               = t.to(compute_dtype)
        source_i        = source_i.to(compute_dtype)
        target_i        = target_i.to(compute_dtype)
        source_context  = source_context.to(compute_dtype)
        target_context  = target_context.to(compute_dtype)
        path_context    = path_context.to(compute_dtype)
        velocity_context = velocity_context.to(compute_dtype)

        ell_sq = self._length_scale_sq().to(device=t.device, dtype=compute_dtype)

        normalizer = float(c * d)

        time_scale_sq    = self.time_scale ** 2
        context_scale_sq = self.context_scale ** 2

        # Differences for distance computation
        diff_0  = path_context - source_context   # (B, c, d): z_t^j - ε^j
        diff_1  = path_context - target_context   # (B, c, d): z_t^j - x^j
        diff_01 = target_context - source_context  # (B, c, d): x^j - ε^j

        # Context distance terms (normalized by c*d)
        context_dist_0  = diff_0.square().sum(dim=(1, 2))  / normalizer  # (B,)
        context_dist_1  = diff_1.square().sum(dim=(1, 2))  / normalizer  # (B,)
        context_dist_01 = diff_01.square().sum(dim=(1, 2)) / normalizer  # (B,)

        # Full squared distances  (sections 5 / 7)
        dist_0  = time_scale_sq * t.square()         + context_scale_sq * context_dist_0
        dist_1  = time_scale_sq * (1.0 - t).square() + context_scale_sq * context_dist_1
        dist_01 = time_scale_sq                      + context_scale_sq * context_dist_01

        # Kernel values
        k0              = torch.exp(-dist_0  / (2.0 * ell_sq))
        k1              = torch.exp(-dist_1  / (2.0 * ell_sq))
        endpoint_kernel = torch.exp(-dist_01 / (2.0 * ell_sq))

        # Denominator: 1 - a²; check numerical conditioning
        denominator_raw  = 1.0 - endpoint_kernel.square()
        well_conditioned = denominator_raw > self.denominator_eps

        # Safe denominator — replace ill-conditioned entries with 1 so we
        # don't divide by zero; those tokens will use the linear fallback.
        safe_denominator = torch.where(
            well_conditioned,
            denominator_raw,
            torch.ones_like(denominator_raw),
        )

        # GP posterior-mean weights (section 8)
        w0_gp = (k0 - endpoint_kernel * k1) / safe_denominator
        w1_gp = (k1 - endpoint_kernel * k0) / safe_denominator

        # --- Analytic velocity (section 10) ---
        # Time derivatives of context distances
        # dR²_{i,0}/dt = 2τ²t + (2κ²/c·d) Σ_j <z_t^j - ε^j, v_t^j>
        inner_0 = (diff_0 * velocity_context).sum(dim=(1, 2))  # (B,)
        inner_1 = (diff_1 * velocity_context).sum(dim=(1, 2))  # (B,)

        dist_0_dt = (
            2.0 * time_scale_sq * t
            + 2.0 * context_scale_sq * inner_0 / normalizer
        )

        dist_1_dt = (
            2.0 * time_scale_sq * (t - 1.0)
            + 2.0 * context_scale_sq * inner_1 / normalizer
        )

        # dk/dt = -k/(2ℓ²) * dR²/dt  (endpoint_kernel has no t-dependence)
        dk0_dt = -k0 * dist_0_dt / (2.0 * ell_sq)
        dk1_dt = -k1 * dist_1_dt / (2.0 * ell_sq)

        dw0_dt_gp = (dk0_dt - endpoint_kernel * dk1_dt) / safe_denominator
        dw1_dt_gp = (dk1_dt - endpoint_kernel * dk0_dt) / safe_denominator

        # Linear fallback for ill-conditioned endpoint kernels (section 14)
        w0     = torch.where(well_conditioned, w0_gp,     1.0 - t)
        w1     = torch.where(well_conditioned, w1_gp,     t)
        dw0_dt = torch.where(well_conditioned, dw0_dt_gp, -torch.ones_like(t))
        dw1_dt = torch.where(well_conditioned, dw1_dt_gp,  torch.ones_like(t))

        z_i = w0[:, None] * source_i + w1[:, None] * target_i
        v_i = dw0_dt[:, None] * source_i + dw1_dt[:, None] * target_i

        # Finite-value fallback: replace any remaining NaN/Inf with linear path
        finite = (
            torch.isfinite(z_i).all(dim=-1)
            & torch.isfinite(v_i).all(dim=-1)
        )

        linear_i          = (1.0 - t[:, None]) * source_i + t[:, None] * target_i
        linear_velocity_i = target_i - source_i

        z_i = torch.where(finite[:, None], z_i, linear_i)
        v_i = torch.where(finite[:, None], v_i, linear_velocity_i)

        aux_i = {
            "k0":               k0,
            "k1":               k1,
            "endpoint_kernel":  endpoint_kernel,
            "denominator":      denominator_raw,
            "well_conditioned": well_conditioned,
            "w0":               w0,
            "w1":               w1,
            "dw0_dt":           dw0_dt,
            "dw1_dt":           dw1_dt,
            "distance_to_source": dist_0,
            "distance_to_target": dist_1,
            "endpoint_distance":  dist_01,
            "is_finite":          finite,
        }

        return z_i, v_i, aux_i

    # -------------------------------------------------------------------
    # Segment processing helper
    # -------------------------------------------------------------------

    def _process_segment(
        self,
        t: torch.Tensor,
        source_noise: torch.Tensor,
        target: torch.Tensor,
        seg_start: int,
        seg_end: int,
        q: int,
    ) -> Tuple[list, list, list]:
        """Left-to-right construction of one contiguous segment [seg_start, seg_end)."""
        B = source_noise.shape[0]
        device = t.device

        paths: list      = []
        velocities: list = []
        token_aux: list  = []

        for i in range(seg_start, seg_end):
            local_index = i - seg_start

            source_i = source_noise[:, i, :].to(torch.float32)
            target_i = target[:, i, :].to(torch.float32)

            if local_index == 0:
                # First token in segment: ordinary linear interpolation (section 4)
                z_i = (1.0 - t[:, None]) * source_i + t[:, None] * target_i
                v_i = target_i - source_i

                # Placeholder diagnostics documenting the linear first-token path
                # (section 19).  These are NOT GP kernel values.
                aux_i = {
                    "k0":               torch.ones(B,  device=device, dtype=torch.float32),
                    "k1":               torch.zeros(B, device=device, dtype=torch.float32),
                    "endpoint_kernel":  torch.zeros(B, device=device, dtype=torch.float32),
                    "denominator":      torch.ones(B,  device=device, dtype=torch.float32),
                    "well_conditioned": torch.ones(B,  device=device, dtype=torch.bool),
                    "w0":               1.0 - t,
                    "w1":               t.clone(),
                    "dw0_dt":          -torch.ones(B, device=device, dtype=torch.float32),
                    "dw1_dt":           torch.ones(B, device=device, dtype=torch.float32),
                    "distance_to_source": torch.zeros(B, device=device, dtype=torch.float32),
                    "distance_to_target": torch.zeros(B, device=device, dtype=torch.float32),
                    "endpoint_distance":  torch.zeros(B, device=device, dtype=torch.float32),
                    "is_finite":          torch.ones(B, device=device, dtype=torch.bool),
                }

            else:
                # Gather causal context: tokens [max(seg_start, i-q), i)
                context_start  = max(seg_start, i - q)
                context_length = i - context_start

                source_context   = source_noise[:, context_start:i, :]
                target_context   = target[:, context_start:i, :]
                path_context     = torch.stack(paths[-context_length:],      dim=1)
                velocity_context = torch.stack(velocities[-context_length:], dim=1)

                z_i, v_i, aux_i = self._step(
                    t=t,
                    source_i=source_i,
                    target_i=target_i,
                    source_context=source_context,
                    target_context=target_context,
                    path_context=path_context,
                    velocity_context=velocity_context,
                )

            paths.append(z_i)
            velocities.append(v_i)
            token_aux.append(aux_i)

        return paths, velocities, token_aux

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------

    def forward(
        self,
        t: torch.Tensor,
        source_noise: torch.Tensor,
        target: torch.Tensor,
        q: int,
        split_at: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            t:            One timestep per batch element. Accepted shapes include
                          (B,), (B, 1), and (B, 1, 1).
            source_noise: Source token endpoints, shape (B, L, d).
            target:       Target token endpoints, shape (B, L, d).
            q:            Maximum number of previous tokens forming the joint driver.
            split_at:     Optional sequence boundary. When supplied, reset the
                          recursive driver at this position.

        Returns:
            path:     Shape (B, L, d).
            velocity: Shape (B, L, d).
            aux:      Sequence-level diagnostic dictionary; each value shape (B, L).
        """
        # --- Input validation ---
        if source_noise.shape != target.shape:
            raise ValueError(
                f"source_noise and target must have the same shape, "
                f"got {source_noise.shape} vs {target.shape}"
            )
        if source_noise.ndim != 3:
            raise ValueError(
                f"source_noise must be 3-D (B, L, d), got ndim={source_noise.ndim}"
            )
        if q < 1:
            raise ValueError(f"q must be >= 1, got {q}")

        B, L, d = source_noise.shape
        t = t.reshape(B).to(torch.float32)

        if not torch.isfinite(t).all():
            raise ValueError("t contains non-finite values")
        if not ((t >= 0.0).all() and (t <= 1.0).all()):
            raise ValueError("All entries of t must be in [0, 1]")

        if split_at is not None and not (0 < split_at < L):
            raise ValueError(
                f"split_at must satisfy 0 < split_at < L={L}, got split_at={split_at}"
            )

        # --- Build path(s) ---
        if split_at is None:
            paths, velocities, token_aux = self._process_segment(
                t, source_noise, target, 0, L, q
            )
        else:
            paths_src, vel_src, aux_src = self._process_segment(
                t, source_noise, target, 0, split_at, q
            )
            paths_tgt, vel_tgt, aux_tgt = self._process_segment(
                t, source_noise, target, split_at, L, q
            )
            paths      = paths_src + paths_tgt
            velocities = vel_src   + vel_tgt
            token_aux  = aux_src   + aux_tgt

        path     = torch.stack(paths,      dim=1)  # (B, L, d)
        velocity = torch.stack(velocities, dim=1)  # (B, L, d)

        # Aggregate per-token scalar diagnostics into (B, L) tensors
        aux: Dict[str, torch.Tensor] = {}
        for key in token_aux[0]:
            aux[key] = torch.stack([a[key] for a in token_aux], dim=1)

        return path, velocity, aux


# ===========================================================================
# Configuration dispatcher
# ===========================================================================

def build_pde_gp_path(cfg: dict) -> PDEInspiredGPPath:
    """Instantiate PDEInspiredGPPath from a config dict.

    Expected YAML structure::

        path_type: pde_inspired_gp
        pde_inspired_gp:
          q: 2
          log_length_scale: 0.0
          trainable_length_scale: false
          time_scale: 1.0
          context_scale: 1.0
          denominator_eps: 1.0e-6

    The ``q`` key is consumed by the caller (``forward``); the remaining keys
    are passed to the constructor.
    """
    return PDEInspiredGPPath(
        log_length_scale=cfg.get("log_length_scale", 0.0),
        trainable_length_scale=cfg.get("trainable_length_scale", False),
        time_scale=cfg.get("time_scale", 1.0),
        context_scale=cfg.get("context_scale", 1.0),
        denominator_eps=cfg.get("denominator_eps", 1e-6),
    )
