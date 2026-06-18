"""Per-device pmap'd training step for Stage 2 of CorrFlow.

Stage 2 trains the correlation network phi_eta while the Stage 1 backbone
theta remains frozen.  The GP-correlated path replaces the independent
linear path from Stage 1:

    z_gp_t = t * x0 + (1 - t) * eps_gp * sigma        (GP interpolant)
    v_gp   = x0 - eps_gp * sigma                        (GP velocity target)

    v_ind  = sg[(D_theta(z_gp_t, t) - z_gp_t) / (1-t)] (frozen backbone)
    delta_v = phi_eta(v_ind, t)                          (correlation network)
    v_con  = v_ind + delta_v

    L_con  = E[ mean_i ||v_con^i - v_gp^i||^2 ]

Only phi_eta's parameters (corr_state.params) receive gradients; backbone_params
are closures treated as constants by JAX autodiff and additionally wrapped in
stop_gradient for explicitness.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from utils.train_utils import TrainState
from utils.encoder_utils import encode_text
from utils.sampling_utils import (
    sample_timesteps, sample_gp_path, net_out_to_v_x,
)


Array = jnp.ndarray


def train_step_stage2(
    corr_state: TrainState,
    backbone_params: Dict,
    encoder_params: Dict,
    backbone_apply_fn,
    encoder_apply_fn,
    batch: Dict[str, Array],
    config,
) -> Tuple[TrainState, Dict[str, float]]:
    """Perform a single Stage 2 training step (correlation network only).

    Arg order: positional JAX arrays first (corr_state, backbone_params, encoder_params),
    then static callables/config (backbone_apply_fn, encoder_apply_fn, config) which are
    baked in via functools.partial before pmap so they don't cross device boundaries.

    Args:
        corr_state:          TrainState for phi_eta (params + optimizer + EMA).
        backbone_params:     Frozen Stage 1 model params (use EMA params for stability).
        encoder_params:      Frozen T5 encoder params.
        backbone_apply_fn:   Backbone model's apply function (baked in via partial).
        encoder_apply_fn:    T5 encoder apply function (baked in via partial).
        batch:               Input batch dict from the dataloader.
        config:              Config object with all hyperparameters (baked in via partial).

    Returns:
        Updated corr_state and a metrics dict with 'loss' and 'l2_loss'.
    """
    t_eps = config.t_eps
    latent_mean, latent_std = config.latent_mean, config.latent_std

    new_dropout_rng, current_step_rng = jax.random.split(corr_state.dropout_rng, 2)
    current_step_rng = jax.random.fold_in(
        current_step_rng, jax.lax.axis_index(axis_name="batch")
    )
    t_rng, noise_rng = jax.random.split(current_step_rng, 2)

    # ------------------------------------------------------------------ encode
    x0 = encode_text(
        input_ids=batch["input_ids"],
        attention_mask=batch["encoder_attention_mask"],
        encoder_apply_fn=encoder_apply_fn,
        encoder_params=encoder_params,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )  # (B, L, d)

    batch_size = x0.shape[0]

    # ------------------------------------------------------------------ masks
    attention_mask = batch["attention_mask"]
    cond_seq_mask = batch["cond_seq_mask"][:, :, None]       # (B, L, 1)
    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = jnp.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - batch["cond_seq_mask"])     # (B, L), excludes cond tokens

    # ------------------------------------------------------------------ timesteps + GP path
    t = sample_timesteps(
        t_rng, batch_size,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
    )  # (B,)

    noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)  # (B, L, d)

    z_gp, v_gp, _eps_gp = sample_gp_path(
        x0, noise, t,
        gp_q=config.gp_q,
        gp_rho=config.gp_rho,
        gp_kernel=config.gp_kernel,
        noise_scale=config.denoiser_noise_scale,
        cond_seq_mask=cond_seq_mask,
    )  # z_gp: (B, L, d), v_gp: (B, L, d)

    # ------------------------------------------------------------------ backbone input
    # Run the frozen backbone without self-conditioning: zero out the self-cond slot.
    if config.self_cond_prob > 0:
        z_gp_input = jnp.concatenate([z_gp, jnp.zeros_like(z_gp)], axis=-1)  # (B, L, 2d)
    else:
        z_gp_input = z_gp  # (B, L, d)

    # Neutral CFG scale (= 1) for the self-cond CFG token, if the backbone uses one.
    if config.num_self_cond_cfg_tokens > 0:
        sc_cfg_scale = jnp.ones((batch_size,), dtype=x0.dtype)
    else:
        sc_cfg_scale = None

    # ------------------------------------------------------------------ helpers
    def reduce_token_loss(per_token_loss, mask):
        mask = mask.astype(per_token_loss.dtype)
        safe = jnp.where(mask > 0, per_token_loss, jnp.zeros_like(per_token_loss))
        return (safe * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    # ------------------------------------------------------------------ loss
    def loss_fn(corr_params):
        # --- frozen backbone forward pass (no gradients flow through this) ---
        backbone_out = backbone_apply_fn(
            {"params": jax.lax.stop_gradient(backbone_params)},
            z_gp_input, t,
            deterministic=True,
            self_cond_cfg_scale=sc_cfg_scale,
            decoder_step_active=jnp.array(False),
        )
        # backbone always returns (x_pred, decoder_logits); decode the tuple safely.
        v_ind_raw, _x_pred = net_out_to_v_x(backbone_out, z_gp, t, t_eps)
        # v_ind = (D_theta(z_gp_t, t) - z_gp_t) / max(1-t, t_eps)  — stop-gradient
        v_ind = jax.lax.stop_gradient(v_ind_raw)

        # --- correlation network predicts per-token velocity correction ---
        delta_v = corr_state.apply_fn(
            {"params": corr_params},
            v_ind, t,
        )  # (B, L, d)

        # --- composed velocity and GP-target loss ---
        v_con = v_ind + delta_v
        per_dim_loss = (v_con - v_gp) ** 2
        loss = reduce_token_loss(jnp.mean(per_dim_loss, axis=-1), loss_mask)
        return loss, ()

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(corr_state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")

    new_corr_state = corr_state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)

    # ------------------------------------------------------------------ EMA
    def ema_update(ema_params, params, decay):
        return jax.tree_util.tree_map(
            lambda e, p: e * decay + p * (1 - decay), ema_params, params,
        )

    is_optimizer_step = (new_corr_state.step % config.grad_accum_steps) == 0
    new_ema_params1 = jax.lax.cond(
        is_optimizer_step,
        lambda: ema_update(corr_state.ema_params1, new_corr_state.params, config.ema_decay1),
        lambda: corr_state.ema_params1,
    )
    new_corr_state = new_corr_state.replace(
        ema_params1=new_ema_params1, dropout_rng=new_dropout_rng,
    )

    metrics = {"loss": loss, "l2_loss": loss}
    return new_corr_state, metrics
