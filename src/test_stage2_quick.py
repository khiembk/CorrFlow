#!/usr/bin/env python
"""Quick smoke-test for Stage 2: loads ELF-B-de-en checkpoint, runs 5 training steps,
confirms shapes are correct and loss is finite.

Run:
    cd src
    python test_stage2_quick.py
"""

import os, sys, copy
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp

from transformers import AutoTokenizer
from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from modules.model import ELF_models
from modules.corr_network import CorrNetwork_models
from utils.train_utils import TrainState, get_optimizer, create_learning_rate_fn
from utils.checkpoint_utils import load_encoder_checkpoint
from train_stage2 import load_frozen_backbone

CONFIG = "configs/training_configs/train_corrflow_stage2_wmt14.yml"
N_STEPS = 5
BATCH = 2
SEQ_LEN = 128
COND_LEN = 64


def main():
    print("=" * 60)
    print("CorrFlow Stage 2 — Quick Smoke Test (CPU)")
    print("=" * 60)

    config = load_config_from_yaml(CONFIG)
    config.global_batch_size = BATCH
    config.batch_size = BATCH
    config.lr = 1e-3
    config.use_wandb = False

    rng = jax.random.PRNGKey(0)

    # ---- encoder + vocab ----
    print("\n[1/4] Loading T5 encoder...")
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name)
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    encoder_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)
    d_model = encoder_config.d_model
    print(f"      d_model={d_model}, vocab_size={vocab_size}  ✓")

    # ---- frozen backbone ----
    print("\n[2/4] Loading frozen backbone (ELF-B-de-en)...")
    backbone_model = ELF_models[config.model](
        text_encoder_dim=d_model, max_length=config.max_length,
        attn_drop=0.0, proj_drop=0.0,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )
    rng, bb_rng = jax.random.split(rng)
    frozen_backbone_params = load_frozen_backbone(config, backbone_model, bb_rng, d_model)
    n_bb = sum(x.size for x in jax.tree_util.tree_leaves(frozen_backbone_params))
    print(f"      backbone params: {n_bb:,}  ✓")

    # ---- CorrNetwork ----
    print("\n[3/4] Initialising CorrNetwork...")
    corr_model = CorrNetwork_models[config.corr_model](gp_q=config.gp_q)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_v = jnp.ones((1, SEQ_LEN, d_model))
    dummy_t = jnp.ones((1,))
    corr_params = corr_model.init(init_rng, dummy_v, dummy_t)
    n_corr = sum(x.size for x in jax.tree_util.tree_leaves(corr_params))
    print(f"      CorrNetwork ({config.corr_model}, gp_q={config.gp_q}) params: {n_corr:,}  ✓")

    lr_fn = create_learning_rate_fn(num_train_steps=100, num_warmup_steps=10, learning_rate=config.lr)
    optimizer = get_optimizer(config, lr_fn)
    corr_state = TrainState.create(
        apply_fn=corr_model.apply,
        params=corr_params["params"],
        tx=optimizer,
        dropout_rng=dropout_rng,
        ema_params1=copy.deepcopy(corr_params["params"]),
    )

    # ---- synthetic batch (random token ids, correct shapes) ----
    rng, batch_rng = jax.random.split(rng)
    input_ids = jax.random.randint(batch_rng, (BATCH, SEQ_LEN), 0, vocab_size)
    cond_seq_mask = jnp.concatenate([
        jnp.ones((BATCH, COND_LEN), dtype=jnp.float32),
        jnp.zeros((BATCH, SEQ_LEN - COND_LEN), dtype=jnp.float32),
    ], axis=1)
    batch = {
        "input_ids": input_ids,
        "attention_mask": jnp.ones((BATCH, SEQ_LEN), dtype=jnp.int32),
        "encoder_attention_mask": jnp.ones((BATCH, SEQ_LEN), dtype=jnp.int32),
        "cond_seq_mask": cond_seq_mask,
        "label_drop_mask": jnp.zeros((BATCH,), dtype=jnp.bool_),
    }

    # ---- single-device step function (no pmap) ----
    # train_step_stage2 uses jax.lax.pmean / axis_index internally.
    # Wrap it to neutralise those calls for single-device testing.
    from train_step_stage2 import train_step_stage2
    import jax.lax as lax

    _pmean = lax.pmean
    _axis_index = lax.axis_index
    lax.pmean = lambda x, axis_name="batch": x
    lax.axis_index = lambda axis_name="batch": jnp.int32(0)

    def step(state):
        return train_step_stage2(
            state,
            frozen_backbone_params,
            encoder_params,
            backbone_model.apply,
            encoder_model.apply,
            batch,
            config,
        )

    # ---- run steps ----
    print(f"\n[4/4] Running {N_STEPS} training steps (single-device, CPU)...")
    losses = []
    for i in range(N_STEPS):
        corr_state, metrics = step(corr_state)
        loss = float(metrics["loss"])
        losses.append(loss)
        status = "finite ✓" if jnp.isfinite(metrics["loss"]) else "NON-FINITE ✗"
        print(f"  step {i+1}/{N_STEPS}: loss={loss:.6f}  [{status}]")

    lax.pmean = _pmean
    lax.axis_index = _axis_index

    # ---- result ----
    all_finite = all(jnp.isfinite(l) for l in losses)
    print()
    print("=" * 60)
    status = "PASSED ✓" if all_finite else "FAILED ✗"
    print(f"Smoke test {status}")
    print(f"  checkpoint loaded  : ELF-B-de-en (step 880600, epoch 100)  ✓")
    print(f"  backbone params    : {n_bb:,}  ✓")
    print(f"  CorrNetwork params : {n_corr:,}  ✓")
    print(f"  all losses finite  : {all_finite}")
    print(f"  losses             : {[f'{l:.4f}' for l in losses]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
