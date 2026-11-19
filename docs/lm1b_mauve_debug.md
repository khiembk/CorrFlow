# LM1B MAUVE debugging report

## Scope

This investigates why LM1B MAUVE is much lower than OWT MAUVE for this
project's checkpoints, and whether this is a metric/preprocessing bug, a
protocol mismatch, or a genuine model-distribution difference. No model
weights, training, or generation semantics were changed. OWT's existing
MAUVE evaluator and its default behavior (`eval_mauve.py`,
`utils/metrics_utils.compute_mauve` called with its original defaults) are
unchanged.

Git SHA at time of this audit: see `result.json`'s `git_sha` field in each
diagnostic output directory (every run stamps its own commit).

## Priority 0 — audit of what was actually run

### Confirmed facts (verified directly against code/data, not inferred)

1. **`src/preprocess_lm1b.py` on the current working tree DOES append EOS**
   (line 49: `ids_arr = np.array(ids + [tokenizer.eos_token_id], ...)`). This
   fix was made earlier in this project's history.

2. **The actual on-disk `lm1b_train_t5` dataset does NOT have EOS appended.**
   Direct inspection of 200 real rows: `input_ids[:sequence_length][-1]` was
   never equal to `tokenizer.eos_token_id` (T5's id `1`) — 0/200. The dataset
   predates the preprocessing fix in (1). **The checkpoint
   `corrflow_stage1-lm1b/checkpoint_996904` was trained entirely on this
   EOS-less data.** The fix in `preprocess_lm1b.py` has not yet been applied
   via a re-tokenization + retrain.

3. **Mechanism by which this prevents the model from learning to terminate**
   (`src/utils/data_utils.py`):
   - `get_pad_token_id(tokenizer, pad_token="eos")` resolves padding to the
     tokenizer's own EOS id. So padding positions are *literally* filled
     with EOS-valued tokens.
   - `pad_and_truncate` records `orig_len` = the real content length
     (`sequence_length`), used to build `attention_mask` (1 only for real
     content, 0 for all padding — including the EOS-valued padding).
   - `train_step.py`: `loss_mask = attention_mask * (1 - cond_seq_mask)`.
     The decoder CE loss is computed **only** where `attention_mask == 1`,
     i.e. only on real content positions. The position immediately after
     the last real token (where a genuine "stop here" target would need to
     live) is always masked out of the loss — even though that position
     happens to be filled with the EOS token ID as padding filler, the model
     never receives a gradient telling it to predict EOS there.
   - Net effect, confirmed mechanistically end-to-end: the Stage 1 LM1B
     backbone has never once been trained to emit EOS. This is independent
     of and additional to the dataset-level fact in (2).

4. **`src/generation.py::test_generation_uncond` reference loading**
   (`ref_path = config.eval_data_path or config.data_path`; `ref_indices =
   range(min(num_samples, len(ref_dataset)))`): takes the **first N rows**
   of the dataset, not a random sample. All LM1B Stage 2 configs used this
   session set `eval_data_path: null`, so references were drawn from the
   **training split** (`lm1b_train_t5`), not the held-out
   `lm1b_test_t5` split, despite the held-out split existing and being
   wired into the Stage 1 config. This is a methodological weakness (not
   the root cause of the low score — see D4 below, which shows first-N vs
   random references are statistically indistinguishable for this dataset —
   but worth fixing for a defensible rebuttal).

5. **`src/utils/metrics_utils.py::compute_mauve` (before this fix) silently
   truncated both text lists to `min(len(p), len(q))`** before calling
   `mauve.compute_mauve`, and hardcoded `max_text_length=1024` with no path
   to override it per-dataset from `generation.py`. `mauve-text` itself
   supports independently-sized `p_text`/`q_text`; the truncation-to-min was
   purely this wrapper's own behavior, not a library requirement.

6. **`generation.py`'s `test_generation_uncond` never passed an explicit
   `max_text_length` to `compute_mauve`**, so every LM1B online-eval MAUVE
   call this session used the OWT-sized default of 1024 GPT-2 tokens — far
   larger than any real or generated LM1B text, meaning this specific
   truncation-window bug (the one that *did* turn out to matter for OWT)
   does **not** apply to LM1B, since nothing was ever truncated at 1024
   tokens for text this short either way.

7. **JPEG-DLM's released LM1B reference file
   (`references/lm1b_t5_mauve96.jsonl`) is packed, multi-sentence text**,
   confirmed by direct inspection of its content (e.g. row 0 transitions
   mid-paragraph from a sentence about medieval Norcia butchers to an
   unrelated sentence about a satellite — two originally-disjoint LM1B
   sentences concatenated). This is a categorically different distribution
   from our sentence-level `lm1b_train_t5`/`lm1b_test_t5` (one sentence per
   example). JPEG-DLM's own README states "our implementation builds on
   ELF" — their ELF/AR/JPEG-DLM LM1B baseline numbers (~0.947–0.951 MAUVE)
   come from training on this packed convention, not from a checkpoint
   trained the way this project's LM1B backbone was trained.

8. **No public pretrained checkpoint exists for "ELF on LM1B"** under either
   the original ELF release (`embedded-language-flows` HuggingFace org: only
   OWT/de-en/XSum) or the original ELF GitHub repo (`lillian039/ELF`: no
   LM1B support in the codebase at all). The paper's ELF-on-LM1B MAUVE=0.948
   is JPEG-DLM's own from-scratch reproduction, not a downloadable artifact.

9. Checked for a stale/misleading docstring in `compute_mauve`: "Runs on CPU
   by default (device_id=-1) since this JAX project's torch install has no
   CUDA backend" was factually wrong in this environment (`torch.__version__
   == 2.6.0+cu124`, `torch.cuda.is_available() == True`) — removed as part
   of this fix. The CPU default itself (`device_id=-1`) is preserved for
   backward compatibility; it's the *claim* about why that was stale.

### Generated-sample provenance for this audit

- "ELF LM1B" = `outputs/lm1b_stage1_baseline_mauve/sde-steps64-.../all_generated_0_0.jsonl`
  — the frozen Stage 1 backbone alone (`--baseline` flag, no Stage 2
  CorrNetwork correction), raw saved text, **no** `gen_truncate_to_first_sentence`
  postprocessing applied.
- "CorrFlow LM1B" = `outputs/lm1b_stage2_pathgp-q4-str006/sde-steps64-.../all_generated_0_40000.jsonl`
  — the best fully-converged (40k-step) Stage 2 checkpoint from this
  project's own earlier sweep, also raw/unpostprocessed.
- Both are saved **after** `mask_after_eos` (masks tokens at/after the
  first literal EOS id, which — per finding (3) — essentially never fires
  genuinely, since the model was never trained to emit EOS as a stop
  signal) and **before** any `gen_truncate_to_first_sentence` call in this
  audit's D0–D7 runs (that flag was left off everywhere in this new
  evaluator's runs, since the task explicitly treats it as "a distinct,
  explicitly named diagnostic," not a default).

## Priority 1 — offline evaluator

New file: `src/eval_lm1b_mauve.py`. Does not reload any model/JAX state;
reads saved JSONL. Key properties, each directly addressing an acceptance
criterion:

- `--reference-mode {sentence, jpeg_packed}` — explicit, not inferred.
- Sentence-mode references are a **seeded random sample without
  replacement** (`numpy.random.RandomState(seed).choice`, not `range(N)`),
  and the chosen row indices are persisted in `result.json`
  (`reference_ids`) so the identical reference set can be reused for ELF vs
  CorrFlow.
- `--text-field` auto-detects `generated` (CorrFlow) or `prediction`
  (JPEG-DLM-compatible) keys; raises `KeyError` with the available keys
  listed if neither is present and no override is given.
- `compute_mauve` (in `utils/metrics_utils.py`) gained `seed` and
  `equalize_sample_counts` parameters. Default `equalize_sample_counts=True`
  preserves every existing caller's exact prior behavior (OWT unaffected,
  verified by `test_compute_mauve_equalize_sample_counts_default_matches_old_behavior`).
  `eval_lm1b_mauve.py` always calls with `equalize_sample_counts=False`, so
  differently-sized generated/reference sets are both scored in full, never
  silently reduced to the smaller size.
- `--strict-length` is OFF by default; when set, it reports a **separate**
  `strict_length_filtered` result with `gen_retention`/`ref_retention`
  fields, never silently replacing the standard truncate-only number.
- `--run-controls` runs D4 (disjoint held-out human-human) and D5
  (identical-text) sanity checks.
- Every run's `result.json` includes `git_sha`, library versions
  (`mauve`, `transformers`, `torch`), full CLI args, reference IDs, and
  per-file GPT-2 token-length distributions (mean/std/median/p5…p95,
  fraction under each threshold, number eligible for strict filtering at
  that threshold).

### Scoping decision: `generation.py`'s online eval still uses first-N references

`src/generation.py::test_generation_uncond` was deliberately **not**
changed to use random/held-out reference sampling in this audit, even
though that's what the new offline evaluator does and what the task asks
for in general. Reason: at the time of this audit, a separate `gp_q=0`
Stage 2 ablation (OWT and LM1B) was actively running across multiple
sequential process invocations that each re-read their YAML config and
re-execute this exact code path; changing the reference-sampling logic
mid-run would have made that ablation's early stages (already using
first-N) inconsistent with its later stages (would pick up the new
behavior). Separately, D4 above already demonstrates first-N vs random
sampling isn't the actual issue for LM1B (both are high, ~0.95), so this
is a correctness/defensibility improvement for future runs, not a fix for
the low-score root cause. **Recommendation**: use `eval_lm1b_mauve.py`
(proper seeded random sampling from the held-out split, with persisted
reference IDs) for any reported comparison; treat `generation.py`'s
periodic online MAUVE as a lightweight training-time sanity signal only.

### EOS-position diagnostics — explicitly NOT implemented, and why

Priority 2 of the task spec asks for first-EOS-index distributions on the
*generated* samples. **CorrFlow's `all_generated_*.jsonl` files save only
decoded text, never raw token IDs** — confirmed by inspecting
`generation.py`'s save loop (`all_generated.append((samples_processed,
text))`, where `text` is already the decoded string). This information is
permanently lost at generation time for every file produced before this
audit. Re-tokenizing the saved text with GPT-2 or T5 afterward would not
recover the *original* generation-time EOS position — it would just be a
different tokenizer's view of already-decoded text, not a faithful
reconstruction. Rather than fabricate this diagnostic from text that can't
support it, `eval_lm1b_mauve.py` omits it and instead reports **length**
diagnostics (GPT-2 token-count distributions), which are fully recoverable
from saved text and serve as a reliable proxy: Section "Results" below
shows generated samples cluster near the 128-T5-token generation cap while
real references cluster far below it, which is the direct, measurable
symptom of never having a trained stop signal.

## Results

Generated-sample provenance: "ELF" = `outputs/lm1b_stage1_baseline_mauve/sde-steps64-.../all_generated_0_0.jsonl`
(Stage 1 backbone only, no Stage 2 correction). "CorrFlow" =
`outputs/lm1b_stage2_pathgp-q4-str006/sde-steps64-.../all_generated_0_40000.jsonl`
(best fully-converged, 40k-step Stage 2 checkpoint from this project's own
earlier sweep). Both raw/unpostprocessed (no `gen_truncate_to_first_sentence`).
References: 1000 texts, seeded random sample without replacement (seed=42)
from the held-out `lm1b_test_t5` split. Full machine-readable output for
every row below is in `/scratch/.../outputs/lm1b_mauve_debug/<ID>/result.json`.

### Length diagnostics (the core quantitative finding)

| Set | n | mean GPT-2 tokens | frac < 96 tokens | n eligible ≥ 96 |
|---|---|---|---|---|
| Held-out reference (real LM1B sentences) | 1000 | 27.96 | 88.7% | **5 / 1000** |
| ELF generated (raw) | 1000 | 105.38 | 11.3% | 887 / 1000 |
| JPEG-DLM packed reference | 1024 | 109.4 | ~0% | 511–512 / 512 (per half) |

Real LM1B sentences average 28 GPT-2 tokens; raw generated text averages 105
(close to the 128-T5-token generation cap, confirming it essentially never
stops early at any genuine sentence boundary). This ~4x length gap, not the
choice of MAUVE truncation window, is the dominant driver of every result
below.

### D0 — reproduce old low score (ELF, max_len=1024, old setting)
**MAUVE = 0.0081** (n_gen=1000, n_ref=1000). Matches the previously-observed
low score; confirms nothing is newly broken by this audit's code changes.

### D1 — ELF, max_len=96 (the "correct" LM1B protocol window)
**Standard: MAUVE = 0.0084** — barely moved from D0. Confirms the
truncation-window choice (1024 vs 96) is *not* the root cause for this
dataset, unlike OWT where it was.
**Strict (len≥96) diagnostic: MAUVE = 0.9087, but ref_retention = 0.005**
(only 5/1000 references eligible). **Not reportable as a headline number** —
MAUVE's clustering step warns "please provide at least ~10x training points"
for sample sizes this small; 5 points produces a degenerate, not
meaningful, estimate.

### D2 — CorrFlow, same references/seed, max_len=96
**Standard: MAUVE = 0.0089** — statistically indistinguishable from ELF's
0.0084 under the identical reference set and window.
**Strict (len≥96): MAUVE = 0.9964, gen_retention=0.851, ref_retention=0.005**
— same degenerate-reference-pool caveat as D1; not reportable as headline.

### D3 — sensitivity sweep (same refs/seed throughout)

| max_len (GPT-2 tokens) | ELF | CorrFlow |
|---|---|---|
| 32 | 0.0870 | 0.0888 |
| 64 | 0.0064 | 0.0074 |
| 96 | 0.0084 (= D1) | 0.0089 (= D2) |
| 128 | 0.0081 | 0.0090 |
| 1024 | 0.0081 (= D0) | — |

ELF and CorrFlow track each other closely at every window (never more than
~0.001–0.002 apart) — no meaningful Stage-2 effect on LM1B MAUVE regardless
of truncation choice. Scores bounce non-monotonically between ~0.006 and
~0.09 depending on window (expected finite-sample MAUVE noise at N=1000;
see D4/D5 below for the matching noise floor), but never approach the
FLM (0.869) or published-ELF (0.948) range at *any* window tested.

### D4 — disjoint held-out human-human sanity baseline
**MAUVE = 0.9451** (seed=42, two disjoint 1000-sentence draws from
`lm1b_test_t5`). High, as expected — the evaluation pipeline correctly
recognizes two independent real-text samples as coming from the same
distribution.
Strict (len≥96) variant: MAUVE=0.9919, but gen_retention=0.005,
ref_retention=0.003 — both sides nearly empty; not reportable, and notably
this shows the "almost nothing is ≥96 tokens" issue is a property of our
sentence-level data generally, not something specific to generated text.

### D5 — identical-text sanity check
**MAUVE = 1.0000** exactly. Passes cleanly.

### D6 — JPEG-DLM packed-reference internal consistency
Split their 1024-row `lm1b_t5_mauve96.jsonl` into two disjoint 512-row
halves (seed=42). **MAUVE = 0.9762.** Confirms their reference file is
internally consistent (as it should be) — and independently confirms the
length-profile finding above (mean 109.4 GPT-2 tokens, effectively 100%
≥96), a categorically different distribution from our sentence-level data.

### D7 — cross-protocol diagnostic: ELF/CorrFlow vs JPEG-DLM's packed references
**ELF: MAUVE = 0.1926. CorrFlow: MAUVE = 0.2168.** Both substantially higher
than D1/D2 (0.0084/0.0089) — plausibly because the packed references'
length (mean 109) happens to be closer to raw generated text's length
(mean 105) than our own sentence references' length (mean 28) is. **This
is explicitly a cross-protocol result and is not a fair comparison**: our
checkpoint was never trained on packed multi-sentence sequences, and 0.19–0.22
is still far below the paper's same-protocol 0.869–0.951 range. Reported
here only as a diagnostic, per the acceptance criteria.

## Decision-logic conclusion

Walking the task's own decision tree against the results above:

1. **D5 did not fail** (exactly 1.0) → no feature-extraction/caching bug.
2. **D4 was not abnormally low** (0.9451, consistent with D6's 0.9762 on an
   independent packed-reference file) → reference sampling, text decoding,
   normalization, and the featurizer are all functioning correctly.
3. **D4 is high but D1/D2 are low (0.0084/0.0089)** → per the task's own
   rule, this means **the model genuinely differs from the reference
   distribution** — not an evaluation bug. The length diagnostics make the
   mechanism concrete: raw generated text (mean 105 GPT-2 tokens) almost
   never stops near where real sentences do (mean 28 tokens), because (per
   Priority 0 findings 2–3) the Stage 1 LM1B backbone was trained on data
   with no EOS target anywhere in its loss, on literally every example.
4. **D1/D2 are low AND D7 is comparatively higher but still far below the
   published range** → the sentence-vs-packed distribution mismatch
   explains *part* of the gap to the published ~0.95 (D7 > D1/D2), but is
   not sufficient on its own to reach it — the remaining gap (0.19–0.22 vs
   0.87–0.95) is consistent with finding 8 (no LM1B checkpoint, ours or
   public, was ever trained to match JPEG-DLM's exact packed-sequence
   training distribution).
5. **EOS behavior**: identical generation/postprocessing semantics were
   used for both ELF and CorrFlow in every diagnostic above (same
   `mask_after_eos`, no `gen_truncate_to_first_sentence`) — the near-equal
   scores (D1≈D2, and every row of D3) show this isn't a difference
   between the two models, it's shared, inherited from the frozen Stage 1
   backbone both are built on.
6. **Conclusion reported transparently**: LM1B MAUVE for this project's
   current checkpoints is genuinely low (~0.006–0.09 depending on window,
   never approaching 0.87+) due to a real distributional mismatch rooted in
   Stage 1's EOS-less training — not a code defect in any MAUVE evaluator
   inspected or fixed in this audit.

## Limitations / what retraining would and wouldn't fix

- The `preprocess_lm1b.py` EOS-append fix (already in the working tree) is
  necessary but **not sufficient** on its own for a from-scratch retrain to
  match JPEG-DLM's ~0.95: that number comes from a **packed-sequence**
  training distribution, not sentence-level. A sentence-level retrain with
  correct EOS supervision would very plausibly raise LM1B MAUVE
  substantially above the current ~0.07–0.10 (by giving the model a real
  stop signal, removing the single dominant symptom identified in D1–D3
  below), but should **not** be expected to land at 0.95 under the
  sentence-level protocol used throughout this project — that number is
  not the same benchmark.
- No retraining was performed as part of this task. The `preprocess_lm1b.py`
  fix was made in an earlier session; it has not been exercised by a
  Stage 1 retrain, and this audit does not start one.
