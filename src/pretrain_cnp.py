#!/usr/bin/env python
"""Standalone pretraining for CNPPathNet — phase 1 of the two-phase CNP path.

Trains the causal context-encoder (CNPPathNet) on its own, well-posed objective:
predict each token's clean embedding x0_i from the strictly-previous tokens'
embeddings x0_{<i} (teacher-forced autoregressive Gaussian NLL). No backbone,
no CorrNetwork, no flow-matching machinery — just a small causal-conv network
learning the local predictability structure of real text embeddings.

The resulting checkpoint is loaded FROZEN into Stage 2 (train_stage2.py with
use_cnp_gp=true, cnp_checkpoint=<this run's output_dir>), exactly mirroring how
Stage 2 already freezes the Stage 1 backbone.

Run:
    cd src
    python pretrain_cnp.py --config configs/training_configs/pretrain_cnp_wmt14.yml
"""

import argparse
import os
import sys
from functools import partial

import jax
try:
    jax.distributed.initialize()
except (RuntimeError, ValueError):
    pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax.numpy as jnp
import numpy as np
import copy
from flax import jax_utils
from flax.training.common_utils import shard
from tqdm import tqdm
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder
from modules.corr_network import CNPPathNet_models
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import save_checkpoint, load_encoder_checkpoint
from utils.train_utils import TrainState, get_optimizer, create_learning_rate_fn
from configs.config import load_config_from_yaml, apply_config_overrides
from utils.data_utils import get_dataloader, load_dataset, get_pad_token_id
from utils.encoder_utils import encode_text

import logging
logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO, force=True,
)
for _name in ("absl", "orbax", "tensorstore"):
    logging.getLogger(_name).setLevel(logging.ERROR)
sys.stdout.reconfigure(line_buffering=True)


def cnp_train_step(state, encoder_params, encoder_apply_fn, batch, config):
    latent_mean, latent_std = config.latent_mean, config.latent_std

    x0 = encode_text(
        input_ids=batch["input_ids"],
        attention_mask=batch["encoder_attention_mask"],
        encoder_apply_fn=encoder_apply_fn,
        encoder_params=encoder_params,
        latent_mean=latent_mean, latent_std=latent_std,
    )  # (B, L, d)
    attention_mask = batch["attention_mask"]  # (B, L), 1 = real token

    def loss_fn(params):
        mu, log_sigma = state.apply_fn({"params": params}, x0)
        sigma = jnp.exp(log_sigma)
        per_dim_nll = 0.5 * ((x0 - mu) / sigma) ** 2 + log_sigma
        per_token_nll = jnp.mean(per_dim_nll, axis=-1)  # (B, L)
        mask = attention_mask.astype(per_token_nll.dtype)
        loss = (per_token_nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        return loss, ()

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")

    new_state = state.apply_gradients(grads=grads)
    new_ema = jax.tree_util.tree_map(
        lambda e, p: e * config.ema_decay1 + p * (1 - config.ema_decay1),
        state.ema_params1, new_state.params,
    )
    new_state = new_state.replace(ema_params1=new_ema)
    return new_state, {"loss": loss}


def run_pretrain(config):
    log_for_0("=" * 60)
    log_for_0("CNPPathNet Pretraining (Phase 1 of two-phase CNP path)")
    log_for_0("=" * 60)

    rng = jax.random.PRNGKey(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    train_dataset, _ = load_dataset(config)

    encoder_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)
    encoder_params = jax_utils.replicate(encoder_params)
    d_model = encoder_config.d_model

    num_hosts = jax.process_count()
    local_batch_size = config.global_batch_size // num_hosts

    cnp_model = CNPPathNet_models[config.cnp_model](gp_q=config.gp_q)
    rng, init_rng = jax.random.split(rng)
    dummy_x0 = jnp.ones((1, config.max_length, d_model))
    cnp_params_init = cnp_model.init(init_rng, dummy_x0)
    log_for_0("\n" + cnp_model.tabulate(init_rng, dummy_x0))

    num_train_steps = config.max_steps if config.max_steps is not None else 10_000
    lr_fn = create_learning_rate_fn(
        num_train_steps=num_train_steps, num_warmup_steps=config.warmup_steps,
        learning_rate=config.blr * config.global_batch_size / 256,
    )
    optimizer = get_optimizer(config, lr_fn)
    rng, dropout_rng = jax.random.split(rng)
    state = TrainState.create(
        apply_fn=cnp_model.apply, params=cnp_params_init["params"], tx=optimizer,
        dropout_rng=dropout_rng, ema_params1=copy.deepcopy(cnp_params_init["params"]),
    )
    state = jax_utils.replicate(state)

    num_local_devices = jax.local_device_count()
    p_train_step = jax.pmap(
        partial(cnp_train_step, encoder_apply_fn=encoder_model.apply, config=config),
        axis_name="batch", donate_argnums=(0,),
    )

    dataloader = get_dataloader(
        train_dataset, batch_size=local_batch_size, shuffle=True, drop_last=True,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
    )

    log_for_0(f"Steps: {num_train_steps}, local_batch={local_batch_size}, devices={num_local_devices}")
    global_step = 0
    pbar = tqdm(total=num_train_steps, desc="CNP pretrain")
    done = False
    while not done:
        for batch in dataloader:
            # Only convert real numpy arrays — some datasets (e.g. XSum) also carry
            # raw "input"/"target" text columns through the collate_fn for BLEU/ROUGE
            # eval elsewhere; those must stay out of the pmapped/sharded batch.
            batch = {k: jnp.array(v) for k, v in batch.items() if isinstance(v, np.ndarray)}
            batch = shard(batch)
            state, metrics = p_train_step(state, encoder_params, batch=batch)
            global_step += 1
            pbar.update(1)
            if global_step % config.log_freq == 0:
                loss_val = float(jax_utils.unreplicate(metrics["loss"]))
                log_for_0(f"Step {global_step}: cnp_nll={loss_val:.4f}")
            if global_step >= num_train_steps:
                done = True
                break
    pbar.close()

    save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
    log_for_0(f"CNPPathNet pretraining complete. Checkpoint: {config.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="CNPPathNet standalone pretraining.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config_override", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
    run_pretrain(config)


if __name__ == "__main__":
    main()
