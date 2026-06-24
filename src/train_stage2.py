#!/usr/bin/env python
"""Training script for Stage 2 of CorrFlow: correlation network φ_η.

Loads the frozen Stage 1 backbone (EMA params), initialises the CorrNetwork,
then trains φ_η to match the GP-path velocity target:

    v_con^i = v_ind^i + phi_eta(sg[v_ind], t)
    L_con   = E[ mean_i ||v_con^i - v_gp^i||^2 ]

Run:
    cd src
    python train_stage2.py --config configs/training_configs/train_corrflow_stage2_owt.yml
"""

import argparse
import copy
import yaml
import logging
import os
import sys
import time
from functools import partial

# Initialize JAX distributed BEFORE importing other JAX modules.
import jax
try:
    jax.distributed.initialize()
except (RuntimeError, ValueError):
    pass  # Single-host run, or already initialized.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training.common_utils import get_metrics, shard
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb

from modules.t5_encoder import get_encoder
from modules.model import ELF_models
from modules.corr_network import CorrNetwork_models
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import (
    save_checkpoint, load_encoder_checkpoint, load_checkpoint,
    find_latest_checkpoint,
)
from utils.train_utils import (
    TrainState, prefetch_to_device, get_optimizer, create_learning_rate_fn,
)
from configs.config import (
    load_config_from_yaml, apply_config_overrides, load_sampling_configs, SamplingConfig,
)
from utils.data_utils import get_dataloader, prepare_batch, load_dataset, get_pad_token_id
from train_step_stage2 import train_step_stage2
from generation import run_generation


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)
for _name in ("absl", "orbax", "tensorstore", "flax.training.checkpoints"):
    logging.getLogger(_name).setLevel(logging.ERROR)
sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description="CorrFlow Stage 2: train the correlation network φ_η.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument(
        "--config_override", action="append", default=[],
        help="Override config values (field_name=value). Can be specified multiple times.",
    )
    return parser.parse_args()


# ============================================
# Frozen backbone loader
# ============================================

def _load_ema_params_orbax(checkpoint_dir: str):
    """Load ema_params1 from an orbax ocdbt checkpoint (e.g. HuggingFace ELF checkpoints).

    These checkpoints store arrays in ocdbt shards under a checkpoint_N/ subdirectory.
    We restore the raw pytree (target=None) and extract ema_params1.
    """
    import orbax.checkpoint as ocp

    ckpt_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
    # Find the latest checkpoint_N subdirectory.
    subdirs = sorted(
        [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint_N directory found in {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, subdirs[-1])
    log_for_0(f"Loading orbax ocdbt checkpoint from {ckpt_path}...")

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_path)   # returns raw nested dict
    if "ema_params1" not in restored:
        raise ValueError(f"'ema_params1' not found in orbax checkpoint at {ckpt_path}")
    log_for_0("Loaded frozen backbone EMA params (orbax ocdbt format).")
    return jax.tree_util.tree_map(jnp.array, restored["ema_params1"])


def load_frozen_backbone(config, backbone_model, rng, d_model):
    """Initialise the backbone model, restore Stage 1 EMA weights, return frozen params.

    Supports two checkpoint formats automatically:
      1. Msgpack format written by our save_checkpoint (Stage 1 runs).
         Uses a full TrainState template so opt_state shapes match for deserialization.
      2. Orbax ocdbt format used by the official HuggingFace ELF checkpoints
         (e.g. embedded-language-flows/ELF-B-de-en).
         Detected when our msgpack load fails; requires orbax-checkpoint installed.

    Returns the ema_params1 pytree (frozen backbone weights).
    """
    if not config.backbone_checkpoint:
        raise ValueError(
            "backbone_checkpoint must be set in the Stage 2 config "
            "(path to the Stage 1 output directory or HF checkpoint)."
        )

    log_for_0(f"Initialising backbone ({config.model}) for checkpoint loading...")
    input_dim = 2 * d_model if config.self_cond_prob > 0 else d_model
    dummy_x = jnp.ones((1, config.max_length, input_dim))
    dummy_t = jnp.ones((1,))
    dummy_sc_cfg = jnp.ones((1,)) if config.num_self_cond_cfg_tokens > 0 else None

    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    init_params = backbone_model.init(
        init_rng, x=dummy_x, t=dummy_t, deterministic=True,
        self_cond_cfg_scale=dummy_sc_cfg,
    )
    total_backbone_params = sum(x.size for x in jax.tree_util.tree_leaves(init_params))
    log_for_0(f"Backbone parameters: {total_backbone_params:,}")

    # --- Format 1: msgpack (our save_checkpoint format) ---
    # Build a TrainState with a dummy Muon optimizer so the opt_state structure
    # matches the Stage 1 checkpoint exactly (needed for deserialization).
    dummy_lr_fn = create_learning_rate_fn(
        num_train_steps=1, num_warmup_steps=0, learning_rate=config.lr or 1e-3,
    )
    dummy_optimizer = get_optimizer(config, dummy_lr_fn)
    backbone_state = TrainState.create(
        apply_fn=backbone_model.apply,
        params=init_params["params"],
        tx=dummy_optimizer,
        dropout_rng=dropout_rng,
        ema_params1=init_params["params"],
    )

    log_for_0(f"Loading backbone checkpoint from {config.backbone_checkpoint}...")
    try:
        backbone_state, ckpt_step = load_checkpoint(config.backbone_checkpoint, backbone_state)
        log_for_0(f"Loaded frozen backbone EMA params (msgpack format, step {ckpt_step}).")
        return backbone_state.ema_params1
    except Exception as e:
        log_for_0(f"Msgpack load failed ({e}); trying orbax ocdbt format...")

    # --- Format 2: orbax ocdbt (HuggingFace ELF checkpoints) ---
    return _load_ema_params_orbax(config.backbone_checkpoint)


# ============================================
# Main training loop
# ============================================

def run_training(config):
    log_for_0("=" * 60)
    log_for_0("CorrFlow Stage 2 — Correlation Network Training (JAX/Flax)")
    log_for_0("=" * 60)
    log_for_0(f"Backbone : {config.model}  [frozen, EMA from Stage 1]")
    log_for_0(f"  checkpoint : {config.backbone_checkpoint}")
    log_for_0(f"CorrNetwork: {config.corr_model}")
    log_for_0(f"GP path    : q={config.gp_q}, rho={config.gp_rho}, kernel={config.gp_kernel}")
    log_for_0(f"Encoder    : {config.encoder_model_name}")
    log_for_0(f"Data       : {config.data_path}")
    log_for_0(f"Output dir : {config.output_dir}")
    log_for_0(f"JAX devices: {jax.device_count()}")
    log_for_0(f"JAX backend: {jax.default_backend()}")
    log_for_0("=" * 60)

    if config.use_wandb and jax.process_index() == 0:
        wandb_config = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
        wandb_tags = config.wandb_tag.split(",") if config.wandb_tag else None
        wandb.init(
            project=config.wandb_project, entity=config.wandb_entity,
            name=config.wandb_run_name, id=config.wandb_run_name, resume=config.wandb_resume,
            tags=wandb_tags, config=wandb_config, dir="/tmp",
            settings=wandb.Settings(start_method="thread"),
        )
        log_for_0(f"Wandb initialized: {wandb.run.url}")

    rng = jax.random.PRNGKey(config.seed)

    # ------------------------------------------------------------------ tokenizer / data
    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")

    train_dataset, eval_dataset = load_dataset(config)

    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size

    # ------------------------------------------------------------------ frozen encoder
    log_for_0(f"Loading encoder ({config.encoder_model_name})...")
    encoder_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)
    encoder_params = jax_utils.replicate(encoder_params)
    d_model = encoder_config.d_model
    log_for_0(f"Encoder d_model: {d_model}")

    # ------------------------------------------------------------------ batch / step sizing
    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    num_hosts = jax.process_count()

    if config.global_batch_size is not None:
        total_batch_size = config.global_batch_size
        local_batch_size = total_batch_size // num_hosts
        config.batch_size = local_batch_size
    elif config.batch_size is not None:
        total_batch_size = config.batch_size * num_devices
        local_batch_size = config.batch_size * num_local_devices
        config.global_batch_size = total_batch_size
    else:
        raise ValueError("Either global_batch_size or batch_size must be specified.")

    steps_per_epoch = len(train_dataset) // total_batch_size
    num_train_steps = steps_per_epoch * config.epochs
    if config.max_steps is not None:
        num_train_steps = min(num_train_steps, config.max_steps)
    if config.warmup_steps >= 0:
        num_warmup_steps = config.warmup_steps
    elif config.warmup_epochs is not None:
        num_warmup_steps = int(config.warmup_epochs * steps_per_epoch)
    else:
        num_warmup_steps = 0

    grad_accum_steps = config.grad_accum_steps
    num_optimizer_steps = num_train_steps // grad_accum_steps
    num_warmup_optimizer_steps = num_warmup_steps // grad_accum_steps

    if config.lr is None or config.lr <= 0:
        config.lr = config.blr * (total_batch_size * grad_accum_steps) / 256

    log_for_0(
        f"Hosts={num_hosts}, local_devices={num_local_devices}, total_devices={num_devices} | "
        f"batch local={local_batch_size}, total={total_batch_size} | "
        f"steps/epoch={steps_per_epoch}, total={num_train_steps}, warmup={num_warmup_steps}, lr={config.lr:.2e}"
    )

    # ------------------------------------------------------------------ frozen backbone
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
    log_for_0("Frozen backbone replicated to all devices.")

    # ------------------------------------------------------------------ CorrNetwork (trainable)
    log_for_0(f"Creating CorrNetwork ({config.corr_model}, gp_q={config.gp_q})...")
    corr_model = CorrNetwork_models[config.corr_model](gp_q=config.gp_q)

    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    dummy_v_ind = jnp.ones((1, config.max_length, d_model))
    dummy_t = jnp.ones((1,))
    corr_params = corr_model.init(init_rng, dummy_v_ind, dummy_t)
    log_for_0("\n" + corr_model.tabulate(init_rng, dummy_v_ind, dummy_t))
    total_corr_params = sum(x.size for x in jax.tree_util.tree_leaves(corr_params))
    log_for_0(f"CorrNetwork trainable parameters: {total_corr_params:,}")

    lr_schedule = create_learning_rate_fn(
        num_train_steps=num_optimizer_steps, num_warmup_steps=num_warmup_optimizer_steps,
        learning_rate=config.lr, schedule=config.lr_schedule, min_lr=config.min_lr,
    )
    optimizer = get_optimizer(config, lr_schedule, grad_accum_steps=grad_accum_steps)
    corr_state = TrainState.create(
        apply_fn=corr_model.apply,
        params=corr_params["params"],
        tx=optimizer,
        dropout_rng=dropout_rng,
        ema_params1=copy.deepcopy(corr_params["params"]),
    )

    # ------------------------------------------------------------------ auto-resume Stage 2
    if not config.resume:
        auto_ckpt = find_latest_checkpoint(config.output_dir)
        if auto_ckpt:
            config.resume = config.output_dir
            log_for_0(f"Auto-resuming Stage 2 from {auto_ckpt}")

    start_epoch, resume_step = 0, 0
    resume_epoch_fractional = 0.0
    if config.resume:
        try:
            ckpt_path = config.resume
            if "checkpoint_" not in ckpt_path:
                ckpt_path = find_latest_checkpoint(ckpt_path) or ckpt_path
            corr_state, resume_step = load_checkpoint(ckpt_path, corr_state)
            resume_epoch_fractional = float(corr_state.epoch)
            start_epoch = int(corr_state.epoch)
            log_for_0(f"Resumed Stage 2 from step {resume_step} (epoch {resume_epoch_fractional:.2f})")
        except Exception as e:
            log_for_0(f"Stage 2 checkpoint load failed: {e}")
            log_for_0("Starting Stage 2 from scratch.")

    corr_state = jax_utils.replicate(corr_state)

    p_train_step = jax.pmap(
        partial(
            train_step_stage2,
            backbone_apply_fn=backbone_model.apply,
            encoder_apply_fn=encoder_model.apply,
            config=config,
        ),
        axis_name="batch",
        donate_argnums=(0,),   # donate corr_state buffers; backbone/encoder params are kept
    )

    # ------------------------------------------------------------------ output dir + config snapshot
    os.makedirs(config.output_dir, exist_ok=True)
    config_dict = {
        k: ([vars(sc) for sc in v] if isinstance(v, list) and v and isinstance(v[0], SamplingConfig) else v)
        for k, v in vars(config).items()
    }
    with open(os.path.join(config.output_dir, "config.yml"), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    train_dataloader = get_dataloader(
        train_dataset, batch_size=local_batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
    )

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    # ============================================
    # Training Loop
    # ============================================
    log_for_0("\n" + "=" * 60)
    log_for_0("Starting Stage 2 Training")
    log_for_0(
        f"Steps/epoch={steps_per_epoch}, epochs={config.epochs}, total={steps_per_epoch * config.epochs} | "
        f"save every {config.save_freq} epoch(s)"
    )
    log_for_0("=" * 60)

    if resume_step > 0:
        global_step = resume_step
        steps_to_skip_in_epoch = resume_step - start_epoch * steps_per_epoch
    else:
        global_step = start_epoch * steps_per_epoch
        steps_to_skip_in_epoch = 0

    last_log_step = global_step
    train_metrics = []
    last_log_time = time.time()
    last_save_epoch = resume_epoch_fractional if resume_step > 0 else float(start_epoch)

    for epoch in range(start_epoch, config.epochs):
        log_for_0(f"\nEpoch {epoch + 1}/{config.epochs}")

        if epoch > start_epoch:
            del train_loader, train_iterator
            train_metrics = []
            jax.clear_caches()

        train_dataloader.sampler.set_epoch(epoch)
        train_iterator = iter(train_dataloader)
        train_loader = prefetch_to_device(train_iterator, size=4)

        initial_pbar = (resume_step - start_epoch * steps_per_epoch) if (epoch == start_epoch and resume_step > 0) else 0
        epoch_pbar = tqdm(
            total=steps_per_epoch, desc=f"Epoch {epoch + 1}", initial=initial_pbar,
            mininterval=1.0, disable=jax.process_index() != 0,
        )

        for step_in_epoch, batch in enumerate(train_loader):
            is_first_step = step_in_epoch == 0 and epoch == start_epoch
            if is_first_step:
                log_for_0("Performing first Stage 2 step (XLA compilation, may take a while)...")
            if epoch == start_epoch and step_in_epoch < steps_to_skip_in_epoch:
                continue

            rng, batch_rng = jax.random.split(rng, 2)
            batch = prepare_batch(batch, config, rng=batch_rng)
            batch = {k: v for k, v in batch.items() if isinstance(v, (np.ndarray, jnp.ndarray))}
            batch = shard(batch)

            corr_state, metrics = p_train_step(
                corr_state, frozen_backbone_params, encoder_params, batch=batch,
            )

            if is_first_step:
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
                log_for_0("First Stage 2 step complete.")

            train_metrics.append(metrics)
            global_step += 1
            epoch_pbar.update(1)

            if config.max_steps is not None and global_step >= config.max_steps:
                break

            if global_step % config.log_freq == 0:
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), corr_state.params)
                gathered = get_metrics(train_metrics)
                avg_loss = float(jnp.mean(gathered["loss"]))
                avg_l2 = float(jnp.mean(gathered["l2_loss"]))
                now = time.time()
                sps = (global_step - last_log_step) / max(now - last_log_time, 1e-8)
                current_lr = lr_schedule((global_step - 1) // grad_accum_steps)

                postfix = {
                    "step": f"{global_step}", "loss": f"{avg_loss:.4f}",
                    "l2": f"{avg_l2:.4f}", "sps": f"{sps:.1f}", "lr": f"{current_lr:.2e}",
                }
                log_for_0(postfix)
                epoch_pbar.set_postfix(**postfix)

                if jax.process_index() == 0:
                    tqdm.write(
                        f"INFO - engine - Step {global_step}: loss={avg_loss:.4f}, "
                        f"l2={avg_l2:.4f}, lr={current_lr:.2e}, sps={sps:.2f}"
                    )
                    if config.use_wandb:
                        current_epoch_progress = epoch + (step_in_epoch + 1) / steps_per_epoch
                        try:
                            wandb.log({
                                "train_loss": avg_loss, "train_l2_loss": avg_l2,
                                "lr": current_lr, "epoch": current_epoch_progress,
                                "step": global_step,
                            }, step=global_step)
                        except Exception:
                            pass

                train_metrics = []
                last_log_step = global_step
                last_log_time = now

            # Intra-epoch checkpointing (fractional save_freq, e.g. 0.5)
            if 0 < config.save_freq < 1:
                progress = epoch + (global_step - epoch * steps_per_epoch) / steps_per_epoch
                if progress - last_save_epoch >= config.save_freq:
                    save_checkpoint(corr_state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
                    log_for_0(f"Saved Stage 2 checkpoint at epoch {progress:.2f} (step {global_step})")
                    last_save_epoch = progress

        epoch_pbar.close()
        current_epoch = epoch + 1
        corr_state = jax_utils.replicate(jax_utils.unreplicate(corr_state).replace(epoch=current_epoch))

        if config.save_freq >= 1 and current_epoch % config.save_freq == 0:
            save_checkpoint(corr_state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
            log_for_0(f"Saved Stage 2 checkpoint at epoch {current_epoch} (step {global_step})")

        # ---- Generation eval (v_con = v_ind + phi_eta; decode step unchanged) ----
        # For unconditional tasks (eval_dataset is None), still run eval to get PPL/entropy.
        do_eval = (
            config.sampling_configs
            and config.eval_freq > 0
            and current_epoch % config.eval_freq == 0
        )
        if do_eval:
            log_for_0(f"Running Stage 2 generation eval (epoch {current_epoch})...")
            corr_unreplicated = jax_utils.unreplicate(corr_state)

            # Build a minimal state wrapping the frozen backbone so _setup_generation
            # extracts backbone.apply and backbone EMA params for the ODE trajectory.
            # corr EMA params are passed separately and applied via make_corr_model_apply_fn.
            dummy_lr_fn = create_learning_rate_fn(
                num_train_steps=1, num_warmup_steps=0, learning_rate=1e-3,
            )
            dummy_optimizer = get_optimizer(config, dummy_lr_fn)
            rng, gen_rng, gen_dropout_rng = jax.random.split(rng, 3)
            backbone_state_for_gen = TrainState.create(
                apply_fn=backbone_model.apply,
                params=jax_utils.unreplicate(frozen_backbone_params),
                tx=dummy_optimizer,
                dropout_rng=gen_dropout_rng,
                ema_params1=jax_utils.unreplicate(frozen_backbone_params),
            )
            backbone_state_for_gen = backbone_state_for_gen.replace(
                epoch=corr_unreplicated.epoch,
                step=corr_unreplicated.step,
            )
            backbone_state_for_gen_rep = jax_utils.replicate(backbone_state_for_gen)

            gen_batch_size = config.generation_batch_size or local_batch_size
            rng = run_generation(
                state=backbone_state_for_gen_rep,
                encoder_params=encoder_params,
                encoder_apply_fn=encoder_model.apply,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                config=config,
                rng=gen_rng,
                local_batch_size=gen_batch_size,
                corr_apply_fn=corr_model.apply,
                corr_params=corr_unreplicated.ema_params1,
            )

        if config.max_steps is not None and global_step >= config.max_steps:
            log_for_0(f"Reached max_steps={config.max_steps}, stopping training.")
            break

    log_for_0("\n" + "=" * 60)
    log_for_0("Stage 2 Training Complete")
    log_for_0("=" * 60)
    save_checkpoint(corr_state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
    log_for_0(f"Final Stage 2 checkpoint saved to {config.output_dir}")
    if config.use_wandb and jax.process_index() == 0:
        wandb.finish()


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} config override(s)")
    run_training(config)


if __name__ == "__main__":
    main()
