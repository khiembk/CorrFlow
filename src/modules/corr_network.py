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

CorrNetwork_models = {
    'CorrNet-S': CorrNetwork_S,
    'CorrNet-M': CorrNetwork_M,
    'CorrNet-XL': CorrNetwork_XL,
    'CorrNet-L': CorrNetwork_L,
}
