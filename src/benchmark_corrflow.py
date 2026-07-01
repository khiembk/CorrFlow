"""Benchmark CorrFlow Stage 1 generation time (no PPL scoring)."""
import os, time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np

# Must run from src/
from configs.config import load_config
from utils.checkpoint_utils import restore_checkpoint
from utils.sampling_utils import ODE_sampler, SDE_sampler
from utils.train_utils import TrainState
from modules.model import ELF_models

CKPT = "outputs/corrflow_stage1-lm1b/checkpoint_996904"
CONFIG = "configs/training_configs/train_corrflow_stage1_lm1b.yml"
BATCH = 128
WARMUP = 1
RUNS = 3

config = load_config(CONFIG)
model_cls = ELF_models[config.model]
model = model_cls(config=config)

print(f"Loading checkpoint from {CKPT}...")
state = restore_checkpoint(CKPT, target=None, step=None)
params = state["ema_params"]
print(f"Params loaded. JAX devices: {jax.local_device_count()}")
print(f"Batch size: {BATCH}, Length: {config.max_length}")

# Build a single-device generate fn
key = jax.random.PRNGKey(42)
devices = jax.local_devices()
num_devices = len(devices)
per_device_batch = BATCH // num_devices

def generate_batch(params, key, num_steps, method="sde"):
    noise = jax.random.normal(key, (num_devices, per_device_batch, config.max_length, config.encoder_hidden_size))
    noise_sharded = jax.device_put_sharded(list(noise), devices)
    params_replicated = jax.device_put_replicated(params, devices)

    def _gen(params, noise):
        if method == "sde":
            return SDE_sampler(
                model=model, params=params, noise=noise,
                num_steps=num_steps, sigma=config.sigma,
                self_cond=True, self_cond_cfg_scale=3.0,
                sde_gamma=1.0 if num_steps == 64 else 1.5,
                time_schedule="logit_normal",
                P_mean=config.denoiser_p_mean,
                P_std=config.denoiser_p_std,
            )
        else:
            return ODE_sampler(
                model=model, params=params, noise=noise,
                num_steps=num_steps, sigma=config.sigma,
                self_cond=True, self_cond_cfg_scale=3.0,
                time_schedule="logit_normal",
                P_mean=config.denoiser_p_mean,
                P_std=config.denoiser_p_std,
            )

    pmap_gen = jax.pmap(_gen)
    out = pmap_gen(params_replicated, noise_sharded)
    jax.block_until_ready(out)
    return out

# Warmup (triggers XLA compilation)
print("\nWarming up (compiles XLA)...")
for _ in range(WARMUP):
    key, subkey = jax.random.split(key)
    generate_batch(params, subkey, num_steps=32)
print("Warmup done.\n")

for num_steps in [32, 64]:
    print(f"--- {num_steps} steps (SDE) ---")
    times = []
    for i in range(RUNS):
        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()
        generate_batch(params, subkey, num_steps=num_steps)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s for {BATCH} samples "
              f"({elapsed/BATCH*1000:.1f} ms/sample, "
              f"{elapsed/num_steps*1000:.1f} ms/step)")

    avg = sum(times) / len(times)
    print(f"  Avg: {avg:.2f}s | {avg/BATCH*1000:.1f} ms/sample | "
          f"{BATCH/avg:.1f} samples/s\n")
