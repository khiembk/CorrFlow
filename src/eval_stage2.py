#!/usr/bin/env python
"""Standalone eval for CorrFlow Stage 2 checkpoint.

Loads the frozen backbone + CorrNetwork checkpoint and runs generation
(with optional corr_infer mode). No training is performed.

Usage:
    cd src
    python eval_stage2.py --config configs/training_configs/train_corrflow_stage2_wmt14.yml
    python eval_stage2.py --config configs/training_configs/train_corrflow_stage2_wmt14.yml \
        --config_override corr_infer=false   # compare i.i.d. vs GP noise
"""

import argparse
import copy
import logging
import os
import sys

import jax
try:
    jax.distributed.initialize()
except (RuntimeError, ValueError):
    pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax.numpy as jnp
from flax import jax_utils
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder
from modules.model import ELF_models
from modules.corr_network import CorrNetwork_models
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import load_encoder_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.train_utils import TrainState, get_optimizer, create_learning_rate_fn
from configs.config import load_config_from_yaml, apply_config_overrides
from utils.data_utils import load_dataset, get_pad_token_id
from generation import run_generation
from train_stage2 import load_frozen_backbone

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
for _name in ("absl", "orbax", "tensorstore", "flax.training.checkpoints"):
    logging.getLogger(_name).setLevel(logging.ERROR)
sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description="CorrFlow Stage 2 — eval only.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific CorrNetwork checkpoint. Default: latest in output_dir.")
    return parser.parse_args()


def run_eval(config, ckpt_path=None):
    log_for_0("=" * 60)
    log_for_0("CorrFlow Stage 2 — Eval Only")
    log_for_0("=" * 60)
    log_for_0(f"corr_infer : {config.corr_infer}")
    log_for_0(f"gp_q={config.gp_q}, gp_rho={config.gp_rho}")
    log_for_0(f"CorrNetwork: {config.corr_model}")
    log_for_0("=" * 60)

    rng = jax.random.PRNGKey(config.seed)

    # Tokenizer + data
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    _, eval_dataset = load_dataset(config)
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size

    # Encoder
    encoder_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)
    encoder_params = jax_utils.replicate(encoder_params)
    d_model = encoder_config.d_model
    log_for_0(f"Encoder d_model: {d_model}")

    # Batch sizing (for eval only; local_batch_size drives generation loop)
    num_hosts = jax.process_count()
    if config.global_batch_size is not None:
        local_batch_size = config.global_batch_size // num_hosts
    elif config.batch_size is not None:
        local_batch_size = config.batch_size * jax.local_device_count()
    else:
        local_batch_size = 64

    # Frozen backbone
    backbone_model = ELF_models[config.model](
        text_encoder_dim=d_model, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )
    rng, backbone_rng = jax.random.split(rng)
    frozen_backbone_params = load_frozen_backbone(config, backbone_model, backbone_rng, d_model)
    frozen_backbone_params = jax_utils.replicate(frozen_backbone_params)

    # CorrNetwork — init then load checkpoint
    log_for_0(f"Initialising CorrNetwork ({config.corr_model}, gp_q={config.gp_q})...")
    corr_model = CorrNetwork_models[config.corr_model](gp_q=config.gp_q)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_v_ind = jnp.ones((1, config.max_length, d_model))
    dummy_t = jnp.ones((1,))
    corr_params_init = corr_model.init(init_rng, dummy_v_ind, dummy_t)

    dummy_lr_fn = create_learning_rate_fn(num_train_steps=1, num_warmup_steps=0, learning_rate=1e-3)
    dummy_optimizer = get_optimizer(config, dummy_lr_fn)
    corr_state = TrainState.create(
        apply_fn=corr_model.apply,
        params=corr_params_init["params"],
        tx=dummy_optimizer,
        dropout_rng=dropout_rng,
        ema_params1=copy.deepcopy(corr_params_init["params"]),
    )

    resolve_path = ckpt_path or find_latest_checkpoint(config.output_dir) or config.output_dir
    log_for_0(f"Loading CorrNetwork checkpoint from {resolve_path}...")
    corr_state, resume_step = load_checkpoint(resolve_path, corr_state)
    log_for_0(f"Loaded CorrNetwork checkpoint at step {resume_step}")

    corr_unreplicated = jax_utils.unreplicate(
        jax_utils.replicate(corr_state)
    ) if False else corr_state  # corr_state is already unreplicated here

    # Build a backbone TrainState for generation (not replicated for .apply_fn; replicate for pmap)
    dummy_lr_fn2 = create_learning_rate_fn(num_train_steps=1, num_warmup_steps=0, learning_rate=1e-3)
    dummy_optimizer2 = get_optimizer(config, dummy_lr_fn2)
    rng, gen_dropout_rng = jax.random.split(rng)
    backbone_state = TrainState.create(
        apply_fn=backbone_model.apply,
        params=jax_utils.unreplicate(frozen_backbone_params),
        tx=dummy_optimizer2,
        dropout_rng=gen_dropout_rng,
        ema_params1=jax_utils.unreplicate(frozen_backbone_params),
    )
    backbone_state = backbone_state.replace(
        epoch=corr_state.epoch,
        step=corr_state.step,
    )
    backbone_state_rep = jax_utils.replicate(backbone_state)

    rng, gen_rng = jax.random.split(rng)
    run_generation(
        state=backbone_state_rep,
        encoder_params=encoder_params,
        encoder_apply_fn=encoder_model.apply,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        config=config,
        rng=gen_rng,
        local_batch_size=local_batch_size,
        corr_apply_fn=corr_model.apply,
        corr_params=corr_state.ema_params1,
    )


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
    run_eval(config, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()
