# CorrFlow — Correlation Flow Matching for LLM

ICLR 2026 submission. Semi-autoregressive flow matching for language generation.

## Conda environment

```bash
conda activate corrflow
# env at /scratch/user/u.kt348068/.conda/envs/corrflow
```

## Project structure

```
src/
  train.py                          # Stage 1 training driver
  train_step.py                     # Stage 1 per-step loss (denoiser L2 + decoder CE)
  train_stage2.py                   # Stage 2 training driver
  train_step_stage2.py              # Stage 2 per-step loss (GP path + CorrNetwork L2)
  generation.py                     # Inference / sampling
  eval.py                           # Evaluation
  modules/
    model.py                        # ELF_models dict (ELF-B/M/L backbones)
    layers.py                       # Transformer layers
    t5_encoder.py                   # Frozen T5 encoder
    corr_network.py                 # CorrNetwork φ_η (CorrNet-S/M/L variants)
  configs/
    config.py                       # Config dataclass + YAML loader
    training_configs/               # Per-run YAML configs
      train_corrflow_stage1_owt.yml
      train_corrflow_stage2_owt.yml
    sampling_configs/               # Sampling sweep configs
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

**Config:** `src/configs/training_configs/train_corrflow_stage1_owt.yml`

```bash
cd src
python train.py --config configs/training_configs/train_corrflow_stage1_owt.yml
```

**Output:** `outputs/corrflow_stage1-owt/`

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
- Variants: `CorrNet-S` (h=128, 2L), `CorrNet-M` (h=256, 2L), `CorrNet-L` (h=512, 4L)

**Config:** `src/configs/training_configs/train_corrflow_stage2_owt.yml`

```bash
cd src
python train_stage2.py --config configs/training_configs/train_corrflow_stage2_owt.yml
```

**Output:** `outputs/corrflow_stage2-owt/`

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
# Stage 1 training (independent velocity field v_ind,θ)
cd src && python train.py --config configs/training_configs/train_corrflow_stage1_owt.yml

# Stage 2 training (correlation network φ_η — requires completed Stage 1)
cd src && python train_stage2.py --config configs/training_configs/train_corrflow_stage2_owt.yml

# Override config values
python train.py --config configs/training_configs/train_corrflow_stage1_owt.yml \
  --config_override epochs=10 --config_override global_batch_size=256

# Generation / eval
python generation.py
python eval.py
```

## Notes

- "ELF paper" = arXiv:2605.10938, Hu et al. 2026 — the base embedding-flow model this builds on
- `train_step.py` implements Stage 1 exactly; `train_step_stage2.py` implements Stage 2
- The model backbone (`ELF-B/M/L`) is reused for Stage 1; Stage 2 adds `φ_η` (`corr_network.py`) on top
- JAX distributed init is handled in both drivers; single-host runs work without flags
- `backbone_checkpoint` in Stage 2 config points to the Stage 1 output dir; EMA params are loaded
