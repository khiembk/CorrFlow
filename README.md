
**"Rethinking Flow-Based Language Models as Many-Body Dynamical Systems"**

The paper proposes a correlation-aware extension of flow-based language modeling (implemented here as **CorrFlow**) with two key ideas:

1. **Correlation-aware probability path (GP path).**
  Instead of transporting each token independently, token trajectories are conditioned on a local left context (window size `q`) using a GP-style correlated path.

2. **Correlation network (`CorrNetwork`).**
  A lightweight residual network learns a context-induced correction to the base velocity field:
   
  \[
  v_{\text{cond}} = v_{\text{base}} + \phi
  \]
   
  where `v_base` is from the frozen Stage-1 flow backbone, and `\phi` is trained in Stage 2.

### Two-stage training in this repo

- **Stage 1 (base flow backbone):** `src/train.py`
- **Stage 2 (CorrFlow residual/correlation module):** `src/train_stage2.py`
- **Stage 2 evaluation:** `src/eval_stage2.py`

This follows the paper's recipe: train a standard flow model first, then freeze it and train only the correlation module to match correlated velocity targets.


### Minimal CorrFlow run examples

```bash
cd src

# Stage 1 (example)
python train.py --config configs/training_configs/train_corrflow_stage1_owt.yml

# Stage 2 (example)
python train_stage2.py --config configs/training_configs/train_corrflow_stage2_owt.yml

# Eval Stage 2 checkpoint
python eval_stage2.py --config configs/training_configs/train_corrflow_stage2_owt.yml
```

> Notes:
> - Stage 2 configs require `backbone_checkpoint` to point to a trained Stage-1 backbone (or compatible ELF checkpoint).
> - Correlation controls are in config keys like `gp_q`, `gp_rho` / `gp_path_strength`, and `corr_model`.

## Initialization

Install the dependencies (JAX+TPUs) and log in to WandB to track your experiments if needed.

```bash
pip install -r requirements.txt
wandb login YOUR_WANDB_API_KEY
```

#### Sanity Check

1. **Get the checkpoint.** All pre-trained checkpoints are on HuggingFace under [`embedded-language-flows`](https://huggingface.co/embedded-language-flows) and are pulled automatically via `--checkpoint_path <hf-repo-id>` — no manual download needed. To use a locally trained checkpoint, pass the path to the specific checkpoint file, e.g. `--checkpoint_path outputs/elf_b-owt/checkpoint_19000`.

2. **(Optional) Tweak the config.** The provided `configs/training_configs/train_owt_ELF-{B,M,L}.yml` already point at the correct HuggingFace data + T5 encoder, so they run as-is. You may want to edit:
    - `output_dir` — where samples and logs are written
    - `wandb_entity` — set to your entity, or set `use_wandb: false` to disable
    - `sampling_configs_path` — defaults to `configs/sampling_configs/uncond_sampling_configs.yml` (32-step SDE + 64-step SDE, both with self-conditioning CFG); swap for your preferred schedule if needed

3. **Launch evaluation.**

**Unconditional generation:**
```bash
cd src/

  # ELF-B (105M)
  python eval.py \
      --config configs/training_configs/train_owt_ELF-B.yml \
      --checkpoint_path embedded-language-flows/ELF-B-owt

  # ELF-M (342M) — smaller batch to fit the bigger model
  python eval.py \
      --config configs/training_configs/train_owt_ELF-M.yml \
      --checkpoint_path embedded-language-flows/ELF-M-owt \
      --config_override global_batch_size=64

  # ELF-L (652M)
  python eval.py \
      --config configs/training_configs/train_owt_ELF-L.yml \
      --checkpoint_path embedded-language-flows/ELF-L-owt \
      --config_override global_batch_size=64
```
The evaluator generates 1,000 samples and reports Gen. PPL (under a pretrained GPT-2 Large) and unigram entropy. Expected: Gen. PPL ≈ 24 and entropy ≈ 5.15 for ELF-B at 32 SDE steps.

**Conditional generation:**
```bash
cd src/

# XSum (summarization)
python eval.py \
    --config configs/training_configs/train_xsum_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-xsum

# WMT14 De-En (translation)
python eval.py \
    --config configs/training_configs/train_de-en_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-de-en
```
The evaluator runs on each task's **validation** set and reports BLEU for WMT14 De-En and ROUGE-1/2/L for XSum. Expected: BLEU ≈ 26.7 on De-En; ROUGE-1/2/L ≈ 36.3 / 12.5 / 28.1 on XSum. Note that the paper numbers are computed on the **test** sets, so validation scores here may differ slightly.

