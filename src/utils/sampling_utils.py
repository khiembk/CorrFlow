from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array


# ============================================
# Noise Schedulers (how to compute z from x0 and noise)
# ============================================

def add_noise(x0, noise, t, config, cond_seq_mask=None):
    """Flow-matching interpolation z = t*x0 + (1-t)*noise*scale, preserving cond tokens."""
    t_expanded = t.reshape(-1, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z
    return z


# ============================================
# Time Schedulers (how to sample t)
# ============================================

def sample_timesteps(
    rng,
    batch_size,
    P_mean=-0.8,
    P_std=0.8,
    time_schedule='logit_normal',
):
    """Sample timesteps using various time schedules.

    Args:
        rng: JAX random key
        batch_size: Number of samples
        P_mean: Mean for logit-normal distribution
        P_std: Std for logit-normal distribution
        time_schedule: 'logit_normal' or 'uniform'

    Returns:
        Sampled timesteps in [0, 1]
    """
    if time_schedule == 'logit_normal':
        # Biased toward middle timesteps via sigmoid(N(P_mean, P_std)).
        z = jax.random.normal(rng, (batch_size,)) * P_std + P_mean
        return jax.nn.sigmoid(z)
    if time_schedule == 'uniform':
        return jax.random.uniform(rng, (batch_size,))
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(
    rng, n_steps: int, time_schedule: str = "logit_normal",
    P_mean: float = -0.8, P_std: float = 0.8,
) -> Array:
    """Return a length-(n_steps+1) array of t values in [0, 1] for a sampling run.

    - "uniform": evenly-spaced linspace from 0 to 1 (deterministic).
    - "logit_normal": sorted logit-normal samples with 0 / 1 endpoints (random).
    """
    if time_schedule == "uniform":
        return jnp.linspace(0.0, 1.0, n_steps + 1)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(
            rng, batch_size=n_steps - 1,
            P_mean=P_mean, P_std=P_std, time_schedule=time_schedule,
        )
        return jnp.concatenate([jnp.array([0.0]), jnp.sort(steps), jnp.array([1.0])])
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


# ============================================
# CFG Scale Sampling (how to sample cfg scale)
# ============================================

def sample_cfg_scale(rng, batch_size, cfg_min=0.0, cfg_max=3.0):
    """Sample CFG scale from log-uniform distribution in [cfg_min, cfg_max]."""
    u = jax.random.uniform(rng, (batch_size,))
    a = jnp.float32(1.0 + cfg_min)
    b = jnp.float32(1.0 + cfg_max)
    return a * jnp.exp(u * jnp.log(b / a)) - 1.0


# ============================================
# Conditioning helpers (preserve clean tokens during sampling)
# ============================================

def restore_cond(z_updated, cond_seq, cond_seq_mask):
    """Restore clean conditioning tokens in z after a denoising step."""
    mask = cond_seq_mask
    target_ndim = max(z_updated.ndim, cond_seq.ndim)
    while mask.ndim < target_ndim:
        mask = mask[..., None]
    return jnp.where(mask > 0, cond_seq, z_updated)


def restore_vx(v, x, cond_seq, cond_seq_mask):
    """Restore cond positions: x → clean cond_seq, v → 0 (cond tokens don't move)."""
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, jnp.zeros_like(cond_seq), cond_seq_mask)
    return v, x


# ============================================
# Flow-matching forward passes (with optional self-cond / CFG)
# ============================================

def net_out_to_v_x(net_out, z, t, t_eps=5e-2):
    """Convert x_pred network output to v and x.

    When the model returns a tuple (denoised_output, decoder_logits),
    decoder logits are discarded here (used separately in training).
    """
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.reshape(-1, 1, 1)
    x = net_out
    v = (x - z) / jnp.maximum(1.0 - t_reshaped, t_eps)
    return v, x


@partial(jax.jit, static_argnums=(0, 5, 6))
def _forward_sample_self_cond(
    model_apply_fn, model_params, z, t_batch, x_pred_prev, config,
    self_cond_cfg_scale, cond_seq, cond_seq_mask,
):
    """Forward pass with self-conditioning."""
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    _restore_vx = partial(restore_vx, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(jnp.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_cond = jnp.concatenate([z, x_pred_prev], axis=-1)
        self_cond_scale_batch = jnp.full((z.shape[0],), self_cond_cfg_scale)
        net_out_cond = model_apply_fn(
            {"params": model_params}, z_input_cond, t_batch, deterministic=True,
            self_cond_cfg_scale=self_cond_scale_batch,
        )
        v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
        return _restore_vx(v_cond, x_cond)

    # No self-conditioning
    if self_cond_prob == 0:
        net_out = model_apply_fn(
            {"params": model_params}, z, t_batch, deterministic=True,
        )
        v, x = net_out_to_v_x(net_out, z, t_batch, t_eps)
        return _restore_vx(v, x)

    # Combined unconditional and conditional forward pass
    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = restore_cond(jnp.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_uncond = jnp.concatenate([z, z_uncond], axis=-1)
        net_out_uncond = model_apply_fn(
            {"params": model_params}, z_input_uncond, t_batch, deterministic=True,
        )
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
        v_uncond, x_uncond = _restore_vx(v_uncond, x_uncond)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond

    z_input_cond = jnp.concatenate([z, x_pred_prev], axis=-1)
    net_out_cond = model_apply_fn(
        {"params": model_params}, z_input_cond, t_batch, deterministic=True,
    )
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
    v_cond, x_cond = _restore_vx(v_cond, x_cond)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return _restore_vx(v_out, x_out)


@partial(jax.jit, static_argnums=(0, 5, 6, 7))
def _forward_sample(
    model_apply_fn, model_params, z, t_batch, x_pred_prev, config,
    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
):
    """Forward pass with optional self-conditioning and CFG."""
    v_cond, x_cond = _forward_sample_self_cond(
        model_apply_fn, model_params, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond

    # Unconditional forward: zero out cond prefix, no self-cond state, no restore
    z_uncond = restore_cond(z, jnp.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = (
        None if x_pred_prev is None
        else restore_cond(x_pred_prev, jnp.zeros_like(x_pred_prev), cond_seq_mask)
    )
    v_uncond, x_uncond = _forward_sample_self_cond(
        model_apply_fn, model_params, z_uncond, t_batch, x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=jnp.zeros_like(cond_seq), cond_seq_mask=cond_seq_mask,
    )

    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


@partial(jax.jit, static_argnums=(0, 6, 7, 8))
def _ode_step(
    model_apply_fn, model_params, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
):
    """Single ODE (Euler) step for sampling."""
    t_batch = jnp.full((z.shape[0],), t)
    v_pred, x_pred = _forward_sample(
        model_apply_fn=model_apply_fn, model_params=model_params,
        z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z + (t_next - t) * v_pred, x_pred


@partial(jax.jit, static_argnums=(0, 6, 7, 8))
def _sde_step(
    model_apply_fn, model_params, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, gamma, rng,
):
    """Per-step SDE-style sampler with hybrid (t-and-step) noise scaling.

    t_back = t * (1 - gamma * h), where h = t_next - t. alpha = 1 - gamma*h is the
    signal-preservation fraction, constant in t. gamma=0 degenerates to a plain ODE step.
    Uniform-N-step equivalence with old multiplicative gamma_old: gamma_hybrid = gamma_old * N.
    """
    h = t_next - t
    alpha = jnp.clip(1.0 - gamma * h, 0.0, 1.0)
    t_back = alpha * t
    eps = jax.random.normal(rng, z.shape) * config.denoiser_noise_scale
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    t_batch = jnp.full((z.shape[0],), t_back)
    v_pred, x_pred = _forward_sample(
        model_apply_fn=model_apply_fn, model_params=model_params,
        z=z_back, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z_back + (t_next - t_back) * v_pred, x_pred


# ============================================
# GP Path (Stage 2: correlation-aware path)
# ============================================

def _gp_tables(gp_q: int, gp_rho: float, gp_kernel: str = 'rbf'):
    """Precompute GP posterior weights for context sizes 0..gp_q (NumPy, static).

    Supports two kernels parameterised by gp_rho (= adjacent-token correlation):
      'rbf':         K(i,j) = gp_rho^(|i-j|^2)  — non-Markov; q>1 adds new info.
      'exponential': K(i,j) = gp_rho^|i-j|       — Markov; q>1 collapses to q=1.

    For each context size c returns:
      alpha[c]: shape (gp_q,) — right-aligned GP posterior weights;
                alpha[c, gp_q-c:] holds the c actual weights (oldest→newest),
                alpha[c, :gp_q-c] = 0 (padding for shorter contexts at sequence start).
      sigma[c]: scalar — GP posterior std.

    Called once at trace time; results become JAX constants.
    """
    if gp_kernel == 'rbf':
        def K(d):
            return gp_rho ** (int(d) ** 2)
    elif gp_kernel == 'exponential':
        def K(d):
            return gp_rho ** abs(int(d))
    else:
        raise ValueError(f"Unknown gp_kernel '{gp_kernel}'. Choose 'rbf' or 'exponential'.")

    alpha = np.zeros((gp_q + 1, gp_q))
    sigma = np.zeros(gp_q + 1)
    sigma[0] = 1.0  # no context → prior: eps_gp = noise

    for c in range(1, gp_q + 1):
        # k_cross[j] = K(current, context[j]); context goes oldest→newest,
        # distances c, c-1, ..., 1 from the current token.
        k_cross = np.array([K(c - j) for j in range(c)])            # (c,)

        # Kernel matrix of the c context tokens.
        K_ctx = np.array([[K(abs(a - b)) for b in range(c)] for a in range(c)])
        K_inv = np.linalg.inv(K_ctx + 1e-8 * np.eye(c))

        alpha_c = k_cross @ K_inv                                    # (c,)
        sigma[c] = np.sqrt(max(1.0 - float(k_cross @ K_inv @ k_cross), 0.0))

        # Right-align into (gp_q,) so it lines up with the sliding buffer.
        alpha[c, gp_q - c:] = alpha_c

    return alpha, sigma




def sample_gp_path(x0, noise, t, gp_q, gp_rho, gp_kernel='rbf',
                   noise_scale=1.0, cond_seq_mask=None, split_at=None):
    """Construct the GP-correlated path and velocity target for Stage 2 training.

    Each token's correlated noise eps_gp^i is drawn from the GP posterior conditioned
    on the previous min(i, gp_q) tokens' already-correlated noise:

        eps_gp^i = alpha^{c_i} · [eps_gp^{i-c_i}, ..., eps_gp^{i-1}]
                   + sigma^{c_i} · noise^i

    where c_i = min(i, gp_q) and (alpha^c, sigma^c) are the GP posterior parameters
    for context size c under the chosen kernel (default: RBF).

    With RBF, K(i,j) = gp_rho^(|i-j|^2): adjacent-token correlation = gp_rho,
    two-step = gp_rho^4, so farther tokens decorrelate faster than exponential.
    This is non-Markov — context tokens beyond the immediate neighbour contribute
    independent information, making gp_q > 1 meaningful.

    The GP path and velocity:
        z_gp^i_t = t * x^i + (1-t) * eps_gp^i * noise_scale
        v_gp^i   = x^i - eps_gp^i * noise_scale

    Token 0 (no context): eps_gp^0 = noise^0 (standard linear path).

    Args:
        x0:            (B, L, d) clean embeddings.
        noise:         (B, L, d) i.i.d. N(0,I) GP innovation noise.
        t:             (B,) timesteps in [0, 1].
        gp_q:          Context window size (>= 1).
        gp_rho:        Adjacent-token correlation for the chosen kernel.
        gp_kernel:     'rbf' (default) or 'exponential'.
        noise_scale:   Global noise scale (matches denoiser_noise_scale in config).
        cond_seq_mask: Optional (B, L, 1) mask — cond tokens pinned to clean x0.
        split_at:      Optional int — if set, run two independent GP scans:
                       tokens [0, split_at) form one path, [split_at, L) form another.
                       Each segment resets the sliding buffer at its own position 0,
                       so no GP correlation bleeds across the boundary.
                       Intended for bilingual sequences (source | target).

    Returns:
        z_gp:   (B, L, d) GP-path latent at time t.
        v_gp:   (B, L, d) GP velocity target (= x0 - eps_gp * noise_scale).
        eps_gp: (B, L, d) GP-correlated effective noise.
    """
    B, L, d = x0.shape

    # ------------------------------------------------------------------ kernel
    # Autoregressive GP over interpolants z^i_t.
    #
    # Notation (flow matching):
    #   z^i_1 = x_i          (clean embedding at t=1)
    #   z^i_0 ~ N(0, σ²I)    (noise at t=0)
    #   z^i_t = t·z^i_1 + (1-t)·z^i_0   (interpolant)
    #
    # GP model:  y = z^i_t ~ GP( x = [t, z^{i-q}_t, ..., z^{i-1}_t] )
    #
    # The GP input for token i is its CONTEXT WINDOW — the previous q
    # interpolants at the same time t.  Two positions are "similar" when
    # their context windows are similar, making their noise correlated.
    #
    # Kernel:  K(f_a, f_b) = exp(-‖f_a - f_b‖² / gp_rho)
    #   where f = unit-normalise each z in the window, then concatenate.
    #   Unit-normalising keeps each slot's contribution in [0,2], so the
    #   full window distance is in [0, 4·q] and gp_rho is interpretable.
    #   Diagonal K(f,f) = 0  ← empty window, K(f,f) = 1 ← filled window.
    #   Wait: K(f_i, f_i) = exp(0) = 1  always, so diagonal = 1. ✓
    #
    # Carry:  (buf_eps, buf_z, buf_f, idx)
    #   buf_eps: (q, B, d)     — GP noise eps_j for last q positions
    #   buf_z:   (q, B, d)     — interpolants z^j_t for last q positions
    #   buf_f:   (q, B, q*d)   — context features f_j stored when j was processed
    #   idx:     scalar int    — step counter
    #
    # At step i:
    #   f_i = concat( unit_norm(buf_z[0]), ..., unit_norm(buf_z[q-1]) )
    #        = the context window CURRENTLY in buf_z  (shape B × q·d)
    #   Cross-kernel  k_cross[k] = K(f_i, buf_f[k])
    #   Context kernel K_ctx[k,l] = K(buf_f[k], buf_f[l])
    #   GP posterior → eps_i ~ N(mu, sigma²)
    #   z^i_t = t·x_i + (1-t)·eps_i·noise_scale
    #   Push (eps_i, z^i_t, f_i) into sliding buffers.
    #
    # Token 0 (no context): all buffers zero, active mask zeros out
    # k_cross → alpha=0 → mu=0, sigma=1 → eps_0 ~ N(0,I) → linear interp. ✓

    fd = gp_q * d   # feature dimension

    def _feat(zw):
        """Unit-normalise each latent in window and concatenate.
        zw: (gp_q, B, d) → (B, gp_q*d)."""
        zn = zw / (jnp.linalg.norm(zw, axis=-1, keepdims=True) + 1e-8)
        return zn.transpose(1, 0, 2).reshape(B, fd)

    def _k(fa, fb):
        """RBF on concatenated context features.  fa, fb: (B, gp_q*d). → (B,)."""
        sq_dist = jnp.sum((fa - fb) ** 2, axis=-1)   # (B,), ∈ [0, 4*gp_q]
        return jnp.exp(-sq_dist / gp_rho)             # (B,)

    # ------------------------------------------------------------------ scan
    def step_fn(carry, inputs):
        buf_eps, buf_z, buf_f, idx = carry
        noise_i, x_i = inputs                         # (B, d) each

        c = jnp.minimum(idx, jnp.int32(gp_q))
        active = jnp.arange(gp_q) >= (gp_q - c)     # (gp_q,) bool

        # Context feature for current token i: concatenation of current window
        f_i = _feat(buf_z)                            # (B, gp_q*d)

        # Cross-kernel: K(f_i, stored feature of each context position)
        k_cross = jax.vmap(lambda fb_q: _k(f_i, fb_q))(buf_f)  # (gp_q, B)
        k_cross = k_cross * active[:, None]

        # Context kernel: K between stored features of context positions
        K_ctx = jax.vmap(
            lambda fp: jax.vmap(lambda fq: _k(fp, fq))(buf_f)
        )(buf_f)                                       # (gp_q, gp_q, B)

        act2d = (active[:, None] * active[None, :]).astype(jnp.float32)
        eye_q = jnp.eye(gp_q)
        K_ctx = K_ctx * act2d[:, :, None] + (eye_q * (1.0 - act2d))[:, :, None]
        K_ctx = K_ctx + 1e-2 * eye_q[:, :, None]     # diagonal=1 → cond ≤ q/ε

        K_ctx_b   = K_ctx.transpose(2, 0, 1)          # (B, gp_q, gp_q)
        k_cross_b = k_cross.transpose(1, 0)            # (B, gp_q)
        alpha     = jax.vmap(jnp.linalg.solve)(K_ctx_b, k_cross_b)   # (B, gp_q)

        # Posterior variance; prior K(f_i, f_i) = 1.0
        var   = 1.0 - jnp.einsum('bq,bq->b', alpha, k_cross_b)   # (B,)
        sigma = jnp.sqrt(jnp.maximum(var, 0.0))                   # (B,)

        mu    = jnp.einsum('bq,qbd->bd', alpha, buf_eps)    # (B, d)
        eps_i = mu + sigma[:, None] * noise_i                # (B, d)

        # Interpolant z^i_t — stored as context for future tokens
        z_i = t[:, None] * x_i + (1.0 - t[:, None]) * eps_i * noise_scale

        new_buf_eps = jnp.concatenate([buf_eps[1:], eps_i[None]], axis=0)
        new_buf_z   = jnp.concatenate([buf_z[1:],   z_i[None]],  axis=0)
        new_buf_f   = jnp.concatenate([buf_f[1:],   f_i[None]],  axis=0)
        return (new_buf_eps, new_buf_z, new_buf_f, idx + 1), eps_i

    init_carry = (
        jnp.zeros((gp_q, B, d),  dtype=x0.dtype),    # buf_eps
        jnp.zeros((gp_q, B, d),  dtype=x0.dtype),    # buf_z
        jnp.zeros((gp_q, B, fd), dtype=x0.dtype),    # buf_f
        jnp.array(0, dtype=jnp.int32),
    )
    noise_T = noise.transpose(1, 0, 2)             # (L, B, d)
    x0_T    = x0.transpose(1, 0, 2)               # (L, B, d)

    if split_at is not None:
        # Two independent GP paths: source [0, split_at) and target [split_at, L).
        # Fresh buffer reset at each language boundary.
        _, eps_src_T = jax.lax.scan(
            step_fn, init_carry, (noise_T[:split_at], x0_T[:split_at])
        )
        _, eps_tgt_T = jax.lax.scan(
            step_fn, init_carry, (noise_T[split_at:], x0_T[split_at:])
        )
        eps_gp = jnp.concatenate(
            [eps_src_T, eps_tgt_T], axis=0
        ).transpose(1, 0, 2)                       # (B, L, d)
    else:
        _, eps_gp_T = jax.lax.scan(step_fn, init_carry, (noise_T, x0_T))
        eps_gp = eps_gp_T.transpose(1, 0, 2)      # (B, L, d)

    # Build path and velocity
    t_e  = t.reshape(-1, 1, 1)
    z_gp = t_e * x0 + (1 - t_e) * eps_gp * noise_scale
    v_gp = x0 - eps_gp * noise_scale

    if cond_seq_mask is not None:
        z_gp = cond_seq_mask * x0 + (1 - cond_seq_mask) * z_gp

    return z_gp, v_gp, eps_gp

# ============================================
# Stage 2 — conditional velocity wrapper
# ============================================

def make_corr_model_apply_fn(backbone_apply_fn, corr_apply_fn, corr_params, d_model, t_eps):
    """Return a drop-in backbone apply_fn that adds the CorrNetwork velocity correction.

    During ODE/SDE sampling the caller does:
        net_out = model_apply_fn(params, z_input, t, ...)
        v, x_pred = net_out_to_v_x(net_out, z, t)   →  v = (x_pred - z) / max(1-t, t_eps)

    We replace x_pred with x_pred_modified so that net_out_to_v_x yields v_con:
        v_ind  = (x_pred - z) / max(1-t, t_eps)
        v_con  = v_ind + phi_eta(v_ind, t)
        x_pred_modified = v_con * max(1-t, t_eps) + z

    The decode step uses only decoder_logits (the second element of the tuple) and
    ignores x_pred — so it is unaffected by this modification.

    Self-conditioning: z_input may be [z, x_pred_prev] concatenated along the last
    dim (shape B×L×2d). We always extract z = z_input[..., :d_model] which is safe
    for both the conditioned (2d) and unconditioned (d) cases.
    """
    def wrapped_apply_fn(params, z_input, t, **kwargs):
        net_out = backbone_apply_fn(params, z_input, t, **kwargs)

        # net_out_to_v_x takes net_out[0] when it's a tuple.
        x_pred = net_out[0] if isinstance(net_out, tuple) else net_out
        decoder_logits = net_out[1] if isinstance(net_out, tuple) else None

        # Extract z (first d_model dims — correct for both self-cond and no-self-cond)
        z = z_input[..., :d_model]

        # Compute v_ind, apply phi_eta, back-project to x_pred space
        t_denom = jnp.maximum(1.0 - t[:, None, None], t_eps)
        v_ind = (x_pred - z) / t_denom
        delta_v = corr_apply_fn({"params": corr_params}, v_ind, t)
        v_con = v_ind + delta_v
        x_pred_modified = v_con * t_denom + z

        if decoder_logits is None:
            return x_pred_modified
        return x_pred_modified, decoder_logits

    return wrapped_apply_fn
