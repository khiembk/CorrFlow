# Task: Diagnose and Fix MAUVE Evaluation in CorrFlow

## 1. Background and objective

Repository: https://github.com/khiembk/CorrFlow

The current MAUVE evaluation produces unexpectedly low scores. In particular, pretrained ELF obtains approximately 0.11 on the current evaluation, whereas published ELF evaluations report substantially higher scores under different protocols. We need to determine whether the discrepancy is caused by evaluation configuration, reference construction, text-length mismatch, feature extraction, or actual generation quality.

The objective is to implement a reproducible, fair MAUVE evaluation for ELF and CorrFlow, supporting OpenWebText (OWT) and LM1B. Do not optimize the evaluation to maximize the score. A low score that survives the diagnostics is a valid result.

**Constraints:**
- Do not retrain models, modify checkpoints, change the flow path, or change the sampling solver.
- Preserve the existing generation length (OWT: 1024; LM1B: 128) and sampling configuration.
- Do not modify existing generative perplexity or entropy evaluation.
- Reuse saved generated JSONL files whenever possible. Generate additional samples only if needed for a pre-specified evaluation protocol.
- Do not change training datasets or preprocessing for existing checkpoints.
- Apply exactly the same metric protocol and reference set to ELF and CorrFlow at each NFE.

## 2. Audit the current implementation before editing

Inspect the current versions of these files (and follow the actual call sites if the paths have changed):

- `src/utils/metrics_utils.py`
- `src/generation.py`
- `src/configs/config.py` and dataset-specific configuration files
- `src/preprocess_lm1b.py` and OWT dataset-loading/preprocessing code

Trace the complete path: generated token IDs -> decoded strings -> saved JSONL -> reference token IDs -> decoded strings -> MAUVE inputs -> feature extraction -> clustering -> reported value.

Check the following suspected issues against the actual code:

1. MAUVE defaults to GPT-2 Large with `max_text_length=1024`, independent of the dataset or metric configuration.
2. References are taken from the first N dataset rows instead of a randomly selected, reproducible reference pool.
3. Generated and reference texts have different lengths after T5 decoding and GPT-2 retokenization.
4. Reference and generation counts are coupled by truncating both lists to their minimum size.
5. LM1B may contain single sentences instead of packed 128-token blocks; old preprocessing/checkpoints may also differ in EOS conventions.
6. The metric may be unintentionally using padding, special tokens, the wrong tokenizer, duplicated generations, the wrong data split, or stale cached reference features.

Report which issues are confirmed, which are hypotheses, and which are not present. Do not assume that any one of them explains the score until tested.

## 3. Add independent MAUVE configuration

Introduce configurable fields following the repository's existing configuration conventions. Suggested configuration:

```python
eval_mauve_featurize_model = "gpt2-large"
eval_mauve_max_length = 768    # OWT length-controlled diagnostic
# LM1B: choose a dataset-specific value after measuring lengths.
eval_mauve_num_samples = 1000
eval_mauve_num_references = 1000
eval_mauve_seed = 42
eval_mauve_strict_length = False
eval_mauve_num_buckets = "auto"
eval_mauve_scaling_factor = 5
eval_mauve_batch_size = 8
```

Allow these values to be overridden from the CLI/config files. Keep MAUVE evaluation length independent of model generation length and make the legacy 1024-token protocol explicitly reproducible. Do not silently change the meaning of historical MAUVE results; identify the protocol in the output.

## 4. Correct reference preparation

- Load the correct dataset and split for the checkpoint being evaluated. Prefer held-out data for new evaluations. If reproducing a published protocol using a different split, identify it explicitly.
- Decode stored token IDs with the original dataset/model tokenizer (e.g., the corresponding T5 tokenizer), not the MAUVE featurizer tokenizer.
- Verify `sequence_length`, padding masks, EOS handling, and special-token removal against the dataset format. Apply equivalent intentional decoding conventions to reference and generated text; do not silently clean generated text more aggressively.
- Sample references randomly without replacement with a fixed seed. For strict length matching, select from eligible references or scan a deterministic shuffled pool until the requested count is met.
- Save the fixed reference pool/selected references with dataset identifiers, split, seed, tokenizer, selection indices, and length settings. Reuse it across ELF and CorrFlow and across NFE values.
- Do not discard references merely because there are fewer generations. MAUVE permits unequal sample counts. Record actual counts and keep them identical across models for the controlled comparison.
- Detect empty strings and duplicates. Report them; do not deduplicate or filter away poor outputs silently.

## 5. Implement GPT-2 token-length diagnostics and two evaluation modes

A T5 sequence of 1024 tokens is not necessarily a GPT-2 sequence of 1024 tokens. Measure lengths using the exact tokenizer associated with the MAUVE feature extractor, after decoding the original token IDs.

For generated and reference samples separately, report count, empty-text count, duplicate rate, minimum, maximum, mean, standard deviation, median, percentiles (5, 25, 75, 95), and fraction reaching each threshold in {128, 256, 384, 512, 768, 1024} for OWT. Use appropriate shorter thresholds for LM1B.

Support both modes:

### Mode A: Unfiltered / truncation-only MAUVE

Use all selected samples and truncate texts to at most K featurizer tokens. Shorter texts remain shorter. Preserve short outputs rather than silently excluding them. If the library cannot featurize empty strings, define and report a consistent minimal representation for them, or report the technical exclusion explicitly with counts.

### Mode B: Strict length-controlled MAUVE

1. Tokenize generated and reference text using the featurizer tokenizer, with truncation disabled and explicit special-token conventions.
2. Retain texts with at least K featurizer tokens.
3. Take exactly the first K tokens of each eligible text.
4. Select the requested numbers of generated/reference samples without replacement using fixed seeds.
5. Pass the token sequences directly to MAUVE if supported by the installed version; verify token tensor shape and dtype against the actual API. Otherwise use validated precomputed features. Do not decode and re-encode truncated IDs unless the round-trip is verified.
6. Assert the retained sequence lengths are exactly K and report generated/reference retention rates and requested versus actual counts.

For OWT, begin with K=768 as a published-style diagnostic, then test other lengths. Example API pattern (adapt to installed `mauve-text`):

```python
result = mauve.compute_mauve(
    p_tokens=reference_tokens,
    q_tokens=generated_tokens,
    featurize_model_name="gpt2-large",
    max_text_length=K,
    device_id=device_id,
    num_buckets=num_buckets,
    mauve_scaling_factor=5,
    seed=seed,
    verbose=True,
)
```

Verify the package name/version, argument support, feature pooling, token limit, device placement, and clustering behavior. Keep p=reference and q=generated consistently.

If there are not enough eligible samples, fail with a clear diagnostic or explicitly run a smaller, common sample count. Do not silently sample with replacement, switch protocols, or selectively generate until only high-quality outputs remain. Report the retention fraction against all original attempted generations.

**Important:** Mode B measures the distribution conditional on having at least K tokens. It is a diagnostic for length confounding, not an unconditional replacement for Mode A. Always report both modes and retention rates.

## 6. Add an offline evaluation script

Implement a standalone script, e.g., `src/eval_mauve.py`, that can read saved generation files such as `all_generated_{epoch}_{step}.jsonl` without loading the flow model or rerunning inference.

Suggested interface (adapt to repository conventions):

```bash
python -m src.eval_mauve \
  --generated_file /path/to/elf_generations.jsonl \
  --dataset owt \
  --reference_path /path/to/reference_dataset \
  --featurizer gpt2-large \
  --max_lengths 128 256 384 512 768 1024 \
  --num_samples 1000 \
  --num_references 1000 \
  --length_mode both \
  --seed 42 \
  --output_dir results/mauve/elf
```

Support reading an existing fixed reference-text file and evaluating multiple seeds. Inspect the actual JSONL schema; never assume its text field name. Do not accidentally evaluate prompts, metadata, reference answers, or JSON serialization instead of generated text.

Save results to JSON and CSV with model/checkpoint, dataset/split, NFE, generation length, feature extractor and revision, MAUVE version, K, length mode, sample counts, retention rates, length statistics, random seeds, effective bucket counts, scaling factor, and score. Cache features if useful, but include all feature-affecting settings and sample identities in the cache key. For GPT-2 last-token features, different truncation lengths need separately computed features (or equivalent features at the correct positions).

## 7. Run diagnostic experiments on pretrained ELF first

Use the exact same saved generation pool and fixed reference pool where possible. Keep the extractor, sample count, seed and clustering settings fixed when isolating length effects.

| ID | Feature extractor | Evaluation length | Mode | Purpose |
|---|---|---:|---|---|
| E0 | GPT-2 Large | 1024 | Legacy | Reproduce current ~0.11 score exactly |
| E1 | GPT-2 Large | 128, 256, 384, 512, 768, 1024 | A | Test truncation sensitivity |
| E2 | GPT-2 Large | 128, 256, 384, 512, 768, 1024 | B | Test exact length alignment |
| E3 | GPT-2 Large | 768 | B | Human vs. independent human references |
| E4 | GPT-2 Large | 768 | B | Identical reference set vs. itself |
| E5 | GPT-2 Large | 768 | B | Human vs. token-shuffled human text (negative control) |
| E6 | ModernBERT Large | 512 | Verified protocol | Optional feature-extractor comparison |

For E3 use independent disjoint samples, equal counts, and the same length conditioning. E4 should be close to 1 with identical features and correct clustering. E3 is a finite-sample ceiling estimate, not necessarily 1. E5 should generally be lower than E3; do not impose an arbitrary numeric threshold.

If E0 cannot reproduce 0.11, stop and identify differences in exact inputs/settings before interpreting improvements. To isolate reference selection, repeat E0 with randomly sampled references while changing no other setting. Check representative generated and reference texts manually and verify existing Gen. PPL/entropy are consistent with the expected pretrained ELF baseline under the same sampler.

### Optional official ELF-style evaluation

Inspect the official ELF paper and evaluation code before reproducing its protocol. Prior investigation identified a ModernBERT Large feature extractor and a 512-token window in a reported ELF evaluation, but verify extractor, pooling, tokenizer, sample counts, reference data, preprocessing, and quantization settings from the source. Do not claim an exact reproduction if these differ. If MAUVE's default feature extractor does not support ModernBERT correctly, use the official feature extraction implementation and supply `p_features` and `q_features` to MAUVE. Keep this protocol separately labeled from GPT-2 MAUVE.

## 8. Handle LM1B separately

Inspect whether the actual LM1B dataset is composed of individual sentences or packed token blocks, and check whether existing checkpoints were trained with EOS tokens and what termination strategy they use.

If this repository uses individual sentences, do not directly compare scores with published packed-block LM1B results. Do not use 768 tokens for LM1B or blindly adopt a 96-token cutoff: inspect the length distribution and retention first. Test reasonable thresholds such as 16, 32, 64, and 96 GPT-2 tokens where supported. A protocol should be set ahead of comparing the models, based on dataset construction and adequate common coverage rather than whichever length gives CorrFlow a better score.

Do not change LM1B packing, append EOS to existing checkpoints' training data, or retrain as part of this MAUVE fix. Report any such historical mismatch separately.

## 9. Final comparison and reporting

After diagnostics, freeze one main evaluation protocol per dataset before comparing CorrFlow against ELF. Report unfiltered MAUVE alongside any strict length-controlled MAUVE. Use identical reference samples, feature extractor, length, sample counts, clustering settings, seeds, and preprocessing across models and NFE budgets.

For initial debugging, use 1,000 samples. For final results, consider 5,000 or more if feasible and multiple seeds. Distinguish independent generation seeds from metric clustering seeds; do not present clustering-only variation as model-generation uncertainty. Use an explicit common bucket count for the final comparison if appropriate, and report the effective bucket count and human-human calibration. Avoid cherry-picking lengths or seeds.

Expected table format (values below for MAUVE/retention must come from actual runs):

| Dataset | Model | NFE | Gen. PPL | MAUVE-A | MAUVE-B | Gen. retention | Ref. retention |
|---|---|---:|---:|---:|---:|---:|---:|
| OWT | ELF | 32 | existing result | TBD | TBD | TBD | TBD |
| OWT | CorrFlow | 32 | existing result | TBD | TBD | TBD | TBD |
| OWT | ELF | 64 | existing result | TBD | TBD | TBD | TBD |
| OWT | CorrFlow | 64 | existing result | TBD | TBD | TBD | TBD |

## 10. Acceptance criteria and deliverables

- Provide a concise root-cause analysis supported by measured ablations, not assumptions.
- Implement the configuration, reference selection, both length modes, offline CLI, and diagnostic reporting with minimal changes to existing generation/training code.
- Add tests for strict token lengths, short/empty inputs, reference reproducibility, unequal sample counts, insufficient eligible samples, JSONL parsing, and preservation of original generation data. Mock expensive feature extraction in unit tests where appropriate.
- Provide exact commands for OWT and LM1B, including the old protocol, offline length sweep, sanity checks, and the final ELF versus CorrFlow comparison.
- Provide a list of modified files, test results, and a diagnostic results table. If checkpoints, saved generations, dependencies, or GPU resources are unavailable, implement and test what is possible and provide runnable commands; do not fabricate MAUVE values.
- Explain whether the low ELF MAUVE is due to length mismatch, reference preparation, feature extraction, generation quality, or a combination. If the evidence is inconclusive, state what still needs testing.

**First priority: reproduce ELF's 0.11 with the legacy evaluator, then run the offline length sweep and strict 768-token comparison on OWT. Do not retrain or change the model to fix an evaluation issue.**
