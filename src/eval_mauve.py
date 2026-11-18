#!/usr/bin/env python
"""Standalone, offline MAUVE evaluation for CorrFlow/ELF unconditional generation.

Reads an existing `all_generated_*.jsonl` file (no model, no JAX, no GPU
inference needed) and a reference dataset, then reports:
  1. GPT-2-token length diagnostics for both generated and reference text
     (count, empty, duplicates, min/max/mean/std/median/percentiles, and
     coverage at common thresholds).
  2. Mode A ("unfiltered"): truncate every text to at most K GPT-2 tokens,
     keep shorter texts as-is. This is what compute_mauve(..., max_text_length=K)
     already does internally when passing raw text.
  3. Mode B ("strict length-controlled"): keep only texts with >= K GPT-2
     tokens, take exactly the first K tokens of each. Reports retention rate.

References are selected either by taking the first N dataset rows (legacy,
`--ref_sampling first`) or a seeded random sample without replacement
(`--ref_sampling random`), so the "first-N vs random reference" hypothesis
can be tested directly instead of assumed.

Usage:
    cd src
    python eval_mauve.py \
        --generated_file outputs/corrflow_stage2-owt/sde-steps64-.../all_generated_0_0.jsonl \
        --reference_path embedded-language-flows/openwebtext-t5 \
        --tokenizer_name t5-small \
        --max_lengths 256 512 768 1024 \
        --num_samples 1000 --num_references 1000 \
        --length_mode both --ref_sampling both --seed 42
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.data_utils import load_dataset_split  # noqa: E402


def load_generated_texts(path, text_field="generated"):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_field not in obj:
                raise KeyError(f"'{text_field}' not found in JSONL row keys: {list(obj.keys())}")
            texts.append(obj[text_field])
    return texts


def decode_reference_texts(dataset, indices, tokenizer):
    texts = []
    for idx in indices:
        row = dataset[int(idx)]
        seq_len = int(row["sequence_length"])
        ids = np.asarray(row["input_ids"])[:seq_len]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


def report_length_stats(name, texts, gpt2_tokenizer, thresholds):
    lens = [len(gpt2_tokenizer.encode(t)) for t in texts]
    empty = sum(1 for t in texts if not t.strip())
    dup = len(texts) - len(set(texts))
    lens_arr = np.array(lens)
    print(f"\n--- Length diagnostics: {name} (GPT-2 tokens) ---")
    print(f"  count={len(texts)} empty={empty} duplicates={dup}")
    print(f"  min={lens_arr.min()} max={lens_arr.max()} mean={lens_arr.mean():.1f} "
          f"std={lens_arr.std():.1f} median={np.median(lens_arr):.1f}")
    for p in [5, 25, 75, 95]:
        print(f"  p{p}={np.percentile(lens_arr, p):.1f}")
    for k in thresholds:
        frac = float(np.mean(lens_arr >= k))
        print(f"  frac(len>={k}) = {frac:.3f}")
    return lens_arr


def mode_a_tokens(texts, gpt2_tokenizer, K):
    """Truncation-only: keep every text, cap at K tokens."""
    import torch
    out = []
    for t in texts:
        ids = gpt2_tokenizer.encode(t, truncation=True, max_length=K)
        if len(ids) == 0:
            ids = [gpt2_tokenizer.eos_token_id]
        out.append(torch.tensor(ids, dtype=torch.long).view(1, -1))
    return out


def mode_b_tokens(texts, gpt2_tokenizer, K):
    """Strict length-controlled: keep only texts with >= K tokens, take first K."""
    import torch
    out = []
    kept = 0
    for t in texts:
        ids = gpt2_tokenizer.encode(t)
        if len(ids) >= K:
            out.append(torch.tensor(ids[:K], dtype=torch.long).view(1, -1))
            kept += 1
    return out, kept


def run_mauve(p_tokens, q_tokens, featurize_model_name, device_id, num_buckets,
              mauve_scaling_factor, seed, verbose=False):
    import mauve as mauve_lib
    out = mauve_lib.compute_mauve(
        p_tokens=p_tokens, q_tokens=q_tokens,
        featurize_model_name=featurize_model_name, device_id=device_id,
        num_buckets=num_buckets, mauve_scaling_factor=mauve_scaling_factor,
        seed=seed, verbose=verbose,
    )
    return float(out.mauve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated_file", required=True)
    ap.add_argument("--text_field", default="generated")
    ap.add_argument("--reference_path", required=True)
    ap.add_argument("--tokenizer_name", default="t5-small", help="Tokenizer used to decode reference input_ids")
    ap.add_argument("--featurizer", default="gpt2-large")
    ap.add_argument("--max_lengths", type=int, nargs="+", default=[128, 256, 384, 512, 768, 1024])
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--num_references", type=int, default=1000)
    ap.add_argument("--length_mode", choices=["a", "b", "both"], default="both")
    ap.add_argument("--ref_sampling", choices=["first", "random", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--num_buckets", default="auto")
    ap.add_argument("--mauve_scaling_factor", type=int, default=5)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    gpt2_tok = AutoTokenizer.from_pretrained(args.featurizer)
    ref_tok = AutoTokenizer.from_pretrained(args.tokenizer_name)

    gen_texts = load_generated_texts(args.generated_file, args.text_field)
    gen_texts = gen_texts[: args.num_samples]
    print(f"Loaded {len(gen_texts)} generated texts from {args.generated_file}")

    ref_dataset = load_dataset_split(args.reference_path)
    print(f"Loaded reference dataset: {len(ref_dataset)} rows from {args.reference_path}")

    ref_sampling_modes = ["first", "random"] if args.ref_sampling == "both" else [args.ref_sampling]
    ref_text_sets = {}
    for mode in ref_sampling_modes:
        if mode == "first":
            idxs = np.arange(min(args.num_references, len(ref_dataset)))
        else:
            rng = np.random.RandomState(args.seed)
            idxs = rng.choice(len(ref_dataset), size=min(args.num_references, len(ref_dataset)), replace=False)
        ref_text_sets[mode] = decode_reference_texts(ref_dataset, idxs, ref_tok)
        print(f"Decoded {len(ref_text_sets[mode])} reference texts (ref_sampling={mode})")

    thresholds = sorted(set(args.max_lengths))
    report_length_stats("generated", gen_texts, gpt2_tok, thresholds)
    for mode, texts in ref_text_sets.items():
        report_length_stats(f"reference[{mode}]", texts, gpt2_tok, thresholds)

    results = []
    length_modes = ["a", "b"] if args.length_mode == "both" else [args.length_mode]
    for ref_mode, ref_texts in ref_text_sets.items():
        for K in args.max_lengths:
            for lm in length_modes:
                if lm == "a":
                    p_tok = mode_a_tokens(ref_texts, gpt2_tok, K)
                    q_tok = mode_a_tokens(gen_texts, gpt2_tok, K)
                    p_ret, q_ret = len(p_tok) / len(ref_texts), len(q_tok) / len(gen_texts)
                else:
                    p_tok, p_kept = mode_b_tokens(ref_texts, gpt2_tok, K)
                    q_tok, q_kept = mode_b_tokens(gen_texts, gpt2_tok, K)
                    p_ret, q_ret = p_kept / len(ref_texts), q_kept / len(gen_texts)
                    if len(p_tok) < 2 or len(q_tok) < 2:
                        print(f"[K={K} mode={lm} ref={ref_mode}] insufficient eligible samples "
                              f"(p={len(p_tok)}, q={len(q_tok)}); skipping")
                        continue
                score = run_mauve(
                    p_tok, q_tok, args.featurizer, args.device_id,
                    args.num_buckets, args.mauve_scaling_factor, args.seed,
                )
                print(f"[K={K:4d} mode={lm.upper()} ref={ref_mode:6s}] "
                      f"MAUVE={score:.4f}  gen_retention={q_ret:.3f}  ref_retention={p_ret:.3f}")
                results.append({
                    "K": K, "mode": lm, "ref_sampling": ref_mode, "mauve": score,
                    "gen_retention": q_ret, "ref_retention": p_ret,
                    "gen_n": len(q_tok), "ref_n": len(p_tok),
                })

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
