# CorrFlow — Correlation Flow Matching for LLM

ICLR 2026 submission. Semi-autoregressive flow matching for language generation.

## Conda environment

```bash
conda activate corrflow
# env at /home.na1/ad.wsu.edu/khiem.tran/pvt/miniconda3/envs/corrflow
```

## Project structure

```
src/
  train.py                          # Stage 1 training driver
  train_step.py                     # Stage 1 per-step loss (denoiser L2 + decoder CE)
  train_stage2.py                   # Stage 2 training driver
  train_step_stage2.py              # Stage 2 per-step loss (GP path + CorrNetwork L2)
  generation.py                     # Inference / sampling (Stage 1 + 2)
  eval.py                           # Standalone eval for Stage 1 checkpoints
  eval_stage2.py                    # Standalone eval for Stage 2 checkpoints (--baseline flag)
  preprocess_lm1b.py                # Tokenize raw LM1B → HuggingFace arrow format
  modules/
    model.py                        # ELF_models dict (ELF-B/M/L backbones)
    layers.py                       # Transformer layers
    t5_encoder.py                   # Frozen T5 encoder
    corr_network.py                 # CorrNetwork φ_η (CorrNet-S/M/L/XL variants)
  configs/
    config.py                       # Config dataclass + YAML loader
    training_configs/               # Per-run YAML configs
      train_corrflow_stage1_owt.yml
      train_corrflow_stage1_lm1b.yml
      train_corrflow_stage2_owt.yml
      train_corrflow_stage2_wmt14.yml
      train_corrflow_stage2_wmt14_rho{2,3,4}_blr5e4.yml  # sweep configs
      train_corrflow_stage2_xsum.yml
    sampling_configs/               # Sampling sweep configs
      uncond_sampling_configs.yml   # for OWT / LM1B (unconditional)
      cond_sampling_configs.yml     # for WMT14 / XSum (conditional)
  utils/
    sampling_utils.py               # add_noise, sample_timesteps, sample_gp_path, ODE/SDE samplers
    train_utils.py                  # TrainState, optimizer, LR schedule
    data_utils.py                   # Dataloader, batch prep
    encoder_utils.py                # encode_text
    checkpoint_utils.py             # save/load checkpoints
    generation_utils.py             # decode helpers
    logging_utils.py                # log_for_0
    metrics_utils.py                # PPL and BLEU metrics
papers/our/                         # Paper source (LaTeX + PDF)
requirements.txt
```

## Datasets

| Dataset | Task | Location | Size | Notes |
|---------|------|----------|------|-------|
| OpenWebText (OWT) | Unconditional LM | `/home.na1/.../data/owt_train_t5` | 9.7M docs | max_length=1024 |
| WMT14 de→en | Translation | `/home.na1/.../data/wmt14_de-en_*_t5` | 4.5M train | max_length=128, max_input_length=64 |
| XSum | Summarization | `/home.na1/.../data/xsum_*_t5` | — | max_length=1088, max_input_length=1024 |
| LM1B | Unconditional LM | `/home.na1/.../data/lm1b_{train,test}_t5` | 30.3M sentences | max_length=128, avg ~25 tokens/sent |

### LM1B preprocessing

Raw data downloaded from statmt.org (1.7 GB tar.gz). Tokenized with T5-small tokenizer:

```bash
cd src
python preprocess_lm1b.py \
  --lm1b_dir /path/to/1-billion-word-.../ \
  --output_train /home.na1/ad.wsu.edu/khiem.tran/pvt/data/lm1b_train_t5 \
  --output_test  /home.na1/ad.wsu.edu/khiem.tran/pvt/data/lm1b_test_t5 \
  --max_length 128
```

Output format: `{'input_ids': List[int32], 'sequence_length': int64}` — matches OWT format.

## Two-stage training algorithm

### Stage 1 — Independent velocity field `v_ind,θ` (ELF recipe)

Train the base flow-matching model. This is equivalent to ELF training (arXiv:2605.10938).

**Math:**
- Linear OT path: `z_t = t·x + (1−t)·ε·σ`, σ=2.0
- Denoiser x-prediction: `D_θ(z_t, t) = E[x | z_t, t]`
- Velocity: `v_ind,θ(z_t, t) = (D_θ(z_t, t) − z_t) / (1−t)`
- Loss: `L_ind = E[Σ_i ||v_ind,θ(z_t,t) − v_lin||²]`, where `v_lin = x − ε`

**Training details (ELF):**
- Logit-normal time sampling: `t = sigmoid(N(−1.5, 0.8))`
- Decoder CE branch (`decoder_prob=0.2`) for token prediction at t=1
- Self-conditioning (`self_cond_prob=0.5`)
- Muon optimizer, `blr=1e-3`, `global_batch_size=512`
- Loss mask: always uses `attention_mask` (real tokens only) — **do not use `jnp.ones_like`**

**OWT config:** `src/configs/training_configs/train_corrflow_stage1_owt.yml`

```bash
cd src && python train.py --config configs/training_configs/train_corrflow_stage1_owt.yml
```

**LM1B config:** `src/configs/training_configs/train_corrflow_stage1_lm1b.yml`
- `global_batch_size: 128`, `grad_accum_steps: 4` (effective batch = 512)
- `warmup_steps: 5000`, `warmup_epochs: -1`
- Eval: `online_eval: true`, `eval_ppl_model: gpt2-large`

```bash
cd src && python train.py --config configs/training_configs/train_corrflow_stage1_lm1b.yml
```

**Outputs:** `outputs/corrflow_stage1-owt/`, `outputs/corrflow_stage1-lm1b/`

### Stage 2 — Correlation network `φ_η`

Freeze θ; train `φ_η` so that `v_con = v_ind + φ_η(...)` matches the GP-path velocity.

**Math:**
- Construct GP path `z^gp_t` and GP velocity `v_gp`
- `v_con,i = v_ind,i(z^gp_t, t) + φ_η^i({sg[v_ind,j(z^gp_t, t)]}_{j∈C_i}, t)`
- Loss: `L_con = E[Σ_i ||v_con,i − v_gp,i||²]`
- Context window: `C_i = {j | max(1, i−q) ≤ j < i}`

**Architecture (Option A — causal conv + AdaLN):**
- Input proj → N × (AdaLN + causal depthwise conv, kernel=q + pointwise + silu + residual) → zero-init output proj
- Time conditioning: sinusoidal embedding → MLP → (γ, β) FiLM applied as `(1+γ)·RMSNorm(x) + β`
- Strictly causal: output[i] sees only input[i−q : i] via left-zero-padding before each conv
- Variants: `CorrNet-S` (h=128, 2L), `CorrNet-M` (h=256, 2L), `CorrNet-L` (h=512, 4L), `CorrNet-XL` (h=512, 4L+extra)

**Best hyperparams (WMT14 sweep):** `gp_rho=3.0`, `gp_q=4`, `gp_kernel=content`, `blr=1e-3` → BLEU 26.16

```bash
# WMT14
cd src && python train_stage2.py --config configs/training_configs/train_corrflow_stage2_wmt14.yml

# XSum
cd src && python train_stage2.py --config configs/training_configs/train_corrflow_stage2_xsum.yml

# OWT (unconditional — eval_data_path: null, uses GEN.PPL + Entropy metrics)
cd src && python train_stage2.py --config configs/training_configs/train_corrflow_stage2_owt.yml
```

**Outputs:** `outputs/corrflow_stage2-{owt,wmt14,xsum}/`

## Key notation

| Symbol | Meaning |
|--------|---------|
| `x = encode(s)` | T5 embedding of token sequence, shape `(B, L, d)` |
| `ε ~ N(0, I)` | Source noise |
| `z_t` | Noisy latent at time t |
| `v_lin = x − ε` | Linear OT velocity target |
| `v_gp` | GP-path velocity target (Stage 2) |
| `D_θ` | Denoiser head (x-prediction) |
| `v_ind,θ` | Independent velocity field (Stage 1 model) |
| `φ_η` | Correlation network (Stage 2) |
| `sg[·]` | Stop-gradient |
| `q` | Causal context window size |

## Running commands

```bash
# Stage 1 training
cd src && python train.py --config configs/training_configs/train_corrflow_stage1_owt.yml
cd src && python train.py --config configs/training_configs/train_corrflow_stage1_lm1b.yml

# Stage 2 training (requires completed Stage 1 — backbone_checkpoint in YAML)
cd src && python train_stage2.py --config configs/training_configs/train_corrflow_stage2_wmt14.yml

# Override config values
python train.py --config ... --config_override epochs=10 --config_override global_batch_size=256

# Standalone eval (Stage 1)
python eval.py --config configs/training_configs/train_corrflow_stage1_lm1b.yml \
  --checkpoint_path outputs/corrflow_stage1-lm1b/checkpoint_50000

# Standalone eval (Stage 2)
python eval_stage2.py --config configs/training_configs/train_corrflow_stage2_wmt14.yml \
  --checkpoint_path outputs/corrflow_stage2-wmt14/

# Baseline eval (backbone only, no CorrNet)
python eval_stage2.py --config configs/training_configs/train_corrflow_stage2_wmt14.yml \
  --baseline
```

## Experimental results

### WMT14 de→en — Stage 2 hyperparameter sweep

| gp_rho | blr | BLEU (best epoch) |
|--------|-----|-------------------|
| 3.0 | 1e-3 | **26.16** ← best |
| 4.0 | 5e-4 | 26.02 |
| 2.0 | 5e-4 | 26.06 |
| 3.0 | 5e-4 | 26.05 |

Best config: `gp_rho=3.0`, `blr=1e-3`. All rho values (2/3/4) perform similarly at 5e-4; higher blr wins.

### LM1B — Stage 1 (ELF-B, 104M params, trained from scratch)

| Checkpoint | Steps | GEN.PPL (32-step SDE) | GEN.PPL (64-step SDE) | Entropy (64-step) |
|------------|-------|----------------------|-----------------------|-------------------|
| epoch 1 (partial) | 50k | — | 61.9 | 2.36 |
| epoch 2 | 287k | — | 117.9 | 3.98 |
| epoch 3 | 523k | — | 101.7 | 4.00 |
| epoch 5 | 996k | **130.5** | **95.3** | 4.03 |

PPL steadily improves; entropy stabilizes ~4. Eval checkpoint: `outputs/corrflow_stage1-lm1b/checkpoint_996904`.

### LM1B — CorrFlow vs FLM comparison

FLM checkpoint: `david3684/FLM-B-LM1B` (HF), 30522 vocab, BERT tokenizer, 1M training steps.
FLM eval repo: `/home.na1/ad.wsu.edu/khiem.tran/pvt/flm/`, env: `flm` conda env.
Converted ckpt: `/home.na1/ad.wsu.edu/khiem.tran/pvt/checkpoints/FLM-B-LM1B/lm1b_flm.ckpt`

| Model | Steps | Gen.PPL | Entropy |
|-------|-------|---------|---------|
| FLM (paper) | 32 | 152.01 | 4.40 |
| FLM (ours) | 32 | 154.34 | 4.41 |
| **CorrFlow Stage 1** | **32** | **130.5** | **4.03** |
| FLM (paper) | 64 | 126.51 | 4.36 |
| FLM (ours) | 64 | 125.24 | 4.37 |
| **CorrFlow Stage 1** | **64** | **95.3** | **3.98** |

CorrFlow outperforms FLM by ~24 PPL at 32 steps and ~30 PPL at 64 steps. FLM paper numbers reproduced within 1.5%.

## Important implementation notes

### Loss mask — CRITICAL
`train_step.py` must use `attention_mask` for the loss mask in ALL cases:
```python
loss_mask = attention_mask * (1 - batch["cond_seq_mask"])
```
**Never** use `jnp.ones_like(attention_mask)` even when `pad_token == "eos"`. For short-sentence datasets (LM1B: avg 25 tokens padded to 128), uniform masking causes ~80% of the loss to fall on EOS-padding positions, collapsing the decoder to output EOS everywhere.

### Generation dispatch — unconditional vs conditional
`generation.py` and `eval.py` route to `test_generation_uncond` vs `test_generation_cond` based on:
```python
is_conditional = eval_dataset is not None and config.max_input_length is not None
```
Do **not** dispatch on `eval_dataset is None` alone — unconditional tasks (LM1B, OWT) may have a non-null `eval_data_path` (test set for PPL), but are still unconditional because `max_input_length` is unset.

### max_steps
Both `train.py` and `train_stage2.py` support `max_steps: N` in config to hard-cap total training steps regardless of `epochs`. Useful for large datasets (OWT: 15k steps; LM1B Stage 1: removed after initial debugging). The epoch break fires **after** the eval block so metrics always log.

### grad_accum_steps
LM1B uses `global_batch_size: 128` + `grad_accum_steps: 4` → effective batch = 512 (same as OWT). Set `warmup_epochs: -1` and `warmup_steps: 5000` so warmup is step-based not epoch-based (otherwise a 236k-step epoch makes warmup too long).

### Stage 2 unconditional eval
`do_eval` in `train_stage2.py` gates on `config.sampling_configs` and `eval_freq`, not on `eval_dataset is not None`. OWT has no eval dataset but uses `run_generation` → `test_generation_uncond` → PPL/entropy evaluation.

### generation_batch_size
Config field `generation_batch_size` overrides `global_batch_size` only for generation eval. Used for XSum (L=1088) and OWT Stage 2 (L=1024) to prevent OOM during generation without shrinking the training batch.

## Notes

- "ELF paper" = arXiv:2605.10938, Hu et al. 2026 — the base embedding-flow model this builds on
- `train_step.py` implements Stage 1 exactly; `train_step_stage2.py` implements Stage 2
- The model backbone (`ELF-B/M/L`) is reused for Stage 1; Stage 2 adds `φ_η` (`corr_network.py`) on top
- JAX distributed init is handled in both drivers; single-host runs work without flags
- `backbone_checkpoint` in Stage 2 config points to the Stage 1 output dir; EMA params are loaded
- Pre-trained ELF-B checkpoints: HuggingFace `embedded-language-flows` org (ELF-B-owt, ELF-B-de-en, etc.)
