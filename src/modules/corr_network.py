"""Correlation network phi_eta for Stage 2 of CorrFlow.

Architecture: causal 1D depthwise-separable convolution with AdaLN time conditioning.

For each token i, the network sees only the hidden representations of the previous
gp_q tokens (strictly causal) and outputs a velocity correction delta_v^i such that:

    v_con^i = v_ind^i + delta_v^i

The causal property is enforced structurally by left-padding the input with gp_q
zeros before each depthwise conv (kernel size = gp_q, VALID padding), so the
receptive field of output[i] is exactly input[i-gp_q : i].
"""

import jax.numpy as jnp
import flax.linen as nn

from modules.layers import (
    RMSNorm, TimestepEmbedder,
    DEFAULT_KERNEL_INIT, DEFAULT_BIAS_INIT, ZERO_INIT,
)


class CausalConvBlock(nn.Module):
    """AdaLN → causal depthwise conv (kernel=gp_q) → pointwise → silu → residual."""
    hidden_dim: int
    gp_q: int

    @nn.compact
    def __call__(self, x, gamma, beta):
        # x:     (B, L, hidden_dim)
        # gamma: (B, hidden_dim)  — AdaLN scale derived from time embedding
        # beta:  (B, hidden_dim)  — AdaLN shift derived from time embedding
        residual = x
        L = x.shape[1]

        # AdaLN: RMSNorm then FiLM modulation
        x = RMSNorm(self.hidden_dim, name='norm')(x)
        x = (1.0 + gamma[:, None, :]) * x + beta[:, None, :]

        # Causal depthwise conv:
        #   left-pad gp_q zeros so output[i] receives only input[i-gp_q : i].
        #   padded shape (B, L+gp_q, h); VALID conv with kernel gp_q yields
        #   (B, L+1, h); slice [:, :L, :] drops the extra trailing position.
        x_pad = jnp.pad(x, [(0, 0), (self.gp_q, 0), (0, 0)])
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.gp_q,),
            padding='VALID',
            feature_group_count=self.hidden_dim,   # depthwise: one filter per channel
            use_bias=True,
            kernel_init=DEFAULT_KERNEL_INIT,
            bias_init=DEFAULT_BIAS_INIT,
            name='dw_conv',
        )(x_pad)[:, :L, :]

        # Pointwise channel mixing
        x = nn.Dense(
            self.hidden_dim, use_bias=True,
            kernel_init=DEFAULT_KERNEL_INIT,
            bias_init=DEFAULT_BIAS_INIT,
            name='pw',
        )(x)
        x = nn.silu(x)

        return x + residual


class CorrNetwork(nn.Module):
    """Correlation network phi_eta for Stage 2 of CorrFlow.

    Maps stop-gradient independent velocities and the current timestep to a
    per-token velocity correction:

        delta_v = CorrNetwork(sg[v_ind], t)   shape (B, L, d)
        v_con^i = v_ind^i + delta_v^i

    The output layer is zero-initialised so all corrections are zero at the start
    of Stage 2 training, preserving the already-trained Stage 1 behaviour.

    Args:
        hidden_dim:  Width of the internal representations.
        num_layers:  Number of CausalConvBlocks.
        gp_q:        Causal conv kernel size — must match the GP path's gp_q config.
    """
    hidden_dim: int = 256
    num_layers: int = 2
    gp_q: int = 2

    @nn.compact
    def __call__(self, v_ind, t):
        # v_ind: (B, L, d) — stop-gradient independent velocities
        # t:     (B,)      — current timestep in [0, 1]
        d = v_ind.shape[-1]

        # Time embedding → shared AdaLN (gamma, beta) for all blocks.
        # TimestepEmbedder: sinusoidal encoding → 2-layer MLP → (B, hidden_dim).
        t_emb = TimestepEmbedder(self.hidden_dim, name='t_emb')(t)
        film = nn.Dense(
            2 * self.hidden_dim, use_bias=True,
            kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT,
            name='film_proj',
        )(nn.silu(t_emb))                                  # (B, 2 * hidden_dim)
        gamma, beta = jnp.split(film, 2, axis=-1)          # (B, hidden_dim) each

        # Input projection: embed d-dim velocities into hidden_dim space.
        x = nn.Dense(
            self.hidden_dim, use_bias=True,
            kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT,
            name='in_proj',
        )(v_ind)                                            # (B, L, hidden_dim)

        # Stacked causal conv blocks.
        for i in range(self.num_layers):
            x = CausalConvBlock(
                hidden_dim=self.hidden_dim,
                gp_q=self.gp_q,
                name=f'block_{i}',
            )(x, gamma, beta)

        # Output projection: zero-init so delta_v = 0 at training start.
        x = RMSNorm(self.hidden_dim, name='norm_out')(x)
        delta_v = nn.Dense(
            d, use_bias=True,
            kernel_init=ZERO_INIT, bias_init=ZERO_INIT,
            name='out_proj',
        )(x)                                                # (B, L, d)

        return delta_v


# ---------------------------------------------------------------------------
# Size variants  (hidden_dim, num_layers)
# ---------------------------------------------------------------------------
def CorrNetwork_S(**kwargs): return CorrNetwork(hidden_dim=128, num_layers=2, **kwargs)
def CorrNetwork_M(**kwargs): return CorrNetwork(hidden_dim=256, num_layers=2, **kwargs)
def CorrNetwork_XL(**kwargs): return CorrNetwork(hidden_dim=512, num_layers=2, **kwargs)
def CorrNetwork_L(**kwargs): return CorrNetwork(hidden_dim=512, num_layers=4, **kwargs)


class EmCorrNetwork(nn.Module):
    """Embedding-augmented CorrNetwork (EmCorrNet).

    Extends CorrNetwork by concatenating the current latent state z_t
    with the independent velocity v_ind as input, giving the network
    direct access to the actual path position at training time (z_gp_t)
    and ODE state at inference time.

        input:    [v_ind ‖ z_t]   (B, L, 2d)
        in_proj:  2d → hidden_dim
        blocks:   N × CausalConvBlock  (identical to CorrNetwork)
        out_proj: hidden_dim → d       (zero-init)

    At training:  z_t = z_gp_t  (the GP-curved path latent).
    At inference: z_t = z        (the current ODE state, updated each step).
    """
    hidden_dim: int = 256
    num_layers: int = 2
    gp_q: int = 2

    @nn.compact
    def __call__(self, v_ind, z_t, t):
        # v_ind: (B, L, d) — stop-gradient independent velocities
        # z_t:   (B, L, d) — current latent state (z_gp at train, ODE state at inference)
        # t:     (B,)      — current timestep in [0, 1]
        d = v_ind.shape[-1]

        t_emb = TimestepEmbedder(self.hidden_dim, name='t_emb')(t)
        film = nn.Dense(
            2 * self.hidden_dim, use_bias=True,
            kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT,
            name='film_proj',
        )(nn.silu(t_emb))
        gamma, beta = jnp.split(film, 2, axis=-1)

        # Input: concat v_ind and z_t → (B, L, 2d) then project to hidden_dim.
        x = nn.Dense(
            self.hidden_dim, use_bias=True,
            kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT,
            name='in_proj',
        )(jnp.concatenate([v_ind, z_t], axis=-1))

        for i in range(self.num_layers):
            x = CausalConvBlock(
                hidden_dim=self.hidden_dim,
                gp_q=self.gp_q,
                name=f'block_{i}',
            )(x, gamma, beta)

        x = RMSNorm(self.hidden_dim, name='norm_out')(x)
        delta_v = nn.Dense(
            d, use_bias=True,
            kernel_init=ZERO_INIT, bias_init=ZERO_INIT,
            name='out_proj',
        )(x)

        return delta_v


def EmCorrNetwork_S(**kwargs): return EmCorrNetwork(hidden_dim=128, num_layers=2, **kwargs)
def EmCorrNetwork_M(**kwargs): return EmCorrNetwork(hidden_dim=256, num_layers=2, **kwargs)
def EmCorrNetwork_L(**kwargs): return EmCorrNetwork(hidden_dim=512, num_layers=4, **kwargs)
def EmCorrNetwork_XL(**kwargs): return EmCorrNetwork(hidden_dim=512, num_layers=4, **kwargs)


class CNPPathNet(nn.Module):
    """Conditional-Neural-Process-style, trainable alternative to the fixed GP/Clayton
    noise-path constructions in sampling_utils.py.

    Unlike those (hand-specified kernel -> posterior mean/variance), this network
    *learns* a causal context encoder end-to-end, trained with its own auxiliary
    objective: predict each token's clean embedding x0_i from the strictly-previous
    tokens' embeddings x0_{<i} (standard teacher-forced autoregressive NLL — a
    well-posed objective since x0 is real data, unlike the latent noise path itself).

    The predicted uncertainty (log_sigma_i) is then repurposed as a *learned,
    per-token, data-dependent* AR(1) correlation strength: tokens the network finds
    highly predictable (low sigma) get strongly-correlated injected noise; tokens
    it finds surprising (high sigma) get noise close to independent. This replaces
    the GP's fixed distance/kernel-based correlation strength with one driven by
    the actual local predictability of the text.

    Reuses CausalConvBlock from CorrNetwork (same architecture family) with the
    AdaLN gamma/beta fixed at zero (no time-conditioning needed here — this network
    only ever looks at x0, not the flow-matching timestep).
    """
    hidden_dim: int = 128
    num_layers: int = 2
    gp_q: int = 4
    log_sigma_min: float = -5.0
    log_sigma_max: float = 5.0

    @nn.compact
    def __call__(self, x0):
        # x0: (B, L, d) clean embeddings.
        B, L, d = x0.shape

        # Causal shift: position i's input is x0_{i-1} (zero-padded at i=0), so the
        # network can never see x0_i itself before predicting it.
        x0_shifted = jnp.concatenate(
            [jnp.zeros((B, 1, d), dtype=x0.dtype), x0[:, :-1, :]], axis=1,
        )

        x = nn.Dense(
            self.hidden_dim, use_bias=True,
            kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT,
            name='in_proj',
        )(x0_shifted)

        zero_cond = jnp.zeros((B, self.hidden_dim), dtype=x0.dtype)
        for i in range(self.num_layers):
            x = CausalConvBlock(
                hidden_dim=self.hidden_dim, gp_q=self.gp_q, name=f'block_{i}',
            )(x, zero_cond, zero_cond)

        x = RMSNorm(self.hidden_dim, name='norm_out')(x)
        mu = nn.Dense(
            d, use_bias=True, kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT,
            name='mu_proj',
        )(x)
        log_sigma = nn.Dense(
            d, use_bias=True, kernel_init=DEFAULT_KERNEL_INIT, bias_init=ZERO_INIT,
            name='log_sigma_proj',
        )(x)
        log_sigma = jnp.clip(log_sigma, self.log_sigma_min, self.log_sigma_max)

        return mu, log_sigma


def CNPPathNet_S(**kwargs): return CNPPathNet(hidden_dim=128, num_layers=2, **kwargs)
def CNPPathNet_M(**kwargs): return CNPPathNet(hidden_dim=256, num_layers=2, **kwargs)


CNPPathNet_models = {
    'CNPPathNet-S': CNPPathNet_S,
    'CNPPathNet-M': CNPPathNet_M,
}


CorrNetwork_models = {
    'CorrNet-S':    CorrNetwork_S,
    'CorrNet-M':    CorrNetwork_M,
    'CorrNet-XL':   CorrNetwork_XL,
    'CorrNet-L':    CorrNetwork_L,
    'EmCorrNet-S':  EmCorrNetwork_S,
    'EmCorrNet-M':  EmCorrNetwork_M,
    'EmCorrNet-L':  EmCorrNetwork_L,
    'EmCorrNet-XL': EmCorrNetwork_XL,
}
