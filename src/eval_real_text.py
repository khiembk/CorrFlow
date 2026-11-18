#!/usr/bin/env python
"""Measure PPL / entropy / MAUVE of REAL text on a dataset, as a ceiling
reference for comparing against generated samples.

Takes two disjoint random samples of real text: sample A is scored for
GEN-PPL + entropy (via GPT-2-large, same harness as generated-text eval);
MAUVE is computed between A (as "generated"/q) and B (as "reference"/p),
giving the real-vs-real ceiling under the identical length-controlled
protocol used elsewhere (Mode A truncation at K GPT-2 tokens).

Usage:
    cd src
    python eval_real_text.py --dataset_path embedded-language-flows/openwebtext-t5 \
        --tokenizer_name t5-small --ppl_max_length 1024 --mauve_k 768 --num_samples 1000
    python eval_real_text.py --dataset_path /path/to/lm1b_train_t5 \
        --tokenizer_name t5-small --ppl_max_length 128 --mauve_k 128 --num_samples 1000
"""
import argparse
import os
import sys

import jax
try:
    jax.distributed.initialize()
except (RuntimeError, ValueError):
    pass

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.data_utils import load_dataset_split  # noqa: E402
from utils.metrics_utils import Metrics as PPLMetrics  # noqa: E402
from eval_mauve import mode_a_tokens, run_mauve, report_length_stats  # noqa: E402


def decode_texts(dataset, indices, tokenizer):
    texts = []
    for idx in indices:
        row = dataset[int(idx)]
        seq_len = int(row["sequence_length"])
        ids = np.asarray(row["input_ids"])[:seq_len]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--tokenizer_name", default="t5-small")
    ap.add_argument("--eval_ppl_model", default="gpt2-large")
    ap.add_argument("--ppl_max_length", type=int, required=True)
    ap.add_argument("--eval_ppl_batch_size", type=int, default=32)
    ap.add_argument("--mauve_k", type=int, required=True)
    ap.add_argument("--mauve_featurizer", default="gpt2-large")
    ap.add_argument("--mauve_device_id", type=int, default=0)
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name)

    ds = load_dataset_split(args.dataset_path)
    print(f"Loaded dataset: {len(ds)} rows from {args.dataset_path}")

    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(ds), size=2 * args.num_samples, replace=False)
    idx_a, idx_b = idxs[: args.num_samples], idxs[args.num_samples:]

    text_a = decode_texts(ds, idx_a, tok)
    text_b = decode_texts(ds, idx_b, tok)
    print(f"Decoded {len(text_a)} (sample A) + {len(text_b)} (sample B) real texts")

    # --- PPL + entropy (sample A, real text, under the generated-text harness) ---
    print("\n" + "=" * 70)
    print("  GEN-PPL / ENTROPY ON REAL TEXT (sample A)")
    print("=" * 70)
    ppl_metrics = PPLMetrics(
        gen_ppl_eval_model_name_or_path=args.eval_ppl_model,
        eval_ppl_batch_size=args.eval_ppl_batch_size,
        eval_context_size=args.ppl_max_length,
    )
    ppl_results = ppl_metrics.record_generative_perplexity(
        text_samples=text_a, max_length=args.ppl_max_length, retokenize=True,
    )
    print(f"Real-text PPL: {ppl_results['ppl']:.4f}")
    print(f"Real-text mean entropy: {ppl_results['mean_entropy']:.4f}")

    # --- MAUVE (real A vs real B), length-controlled Mode A at K tokens ---
    print("\n" + "=" * 70)
    print(f"  MAUVE: real_A vs real_B (Mode A, K={args.mauve_k})")
    print("=" * 70)
    from transformers import AutoTokenizer as _AT
    gpt2_tok = _AT.from_pretrained(args.mauve_featurizer)
    report_length_stats("real_A", text_a, gpt2_tok, [args.mauve_k])
    report_length_stats("real_B", text_b, gpt2_tok, [args.mauve_k])
    p_tok = mode_a_tokens(text_b, gpt2_tok, args.mauve_k)
    q_tok = mode_a_tokens(text_a, gpt2_tok, args.mauve_k)
    mauve_score = run_mauve(
        p_tok, q_tok, args.mauve_featurizer, args.mauve_device_id,
        "auto", 5, args.seed,
    )
    print(f"MAUVE(real_A, real_B) = {mauve_score:.4f}")

    print("\n" + "=" * 70)
    print("  SUMMARY (real text ceiling)")
    print("=" * 70)
    print(f"  PPL        = {ppl_results['ppl']:.4f}")
    print(f"  Entropy    = {ppl_results['mean_entropy']:.4f}")
    print(f"  MAUVE      = {mauve_score:.4f}  (K={args.mauve_k})")


if __name__ == "__main__":
    main()
