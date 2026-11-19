#!/usr/bin/env python
"""Offline LM1B MAUVE + length/EOS diagnostics evaluator.

Reads already-saved `all_generated_*.jsonl` files (no model reload, no JAX,
no retraining) and scores them against a reference set under one of two
EXPLICITLY DISTINCT protocols:

  --reference-mode sentence     (DEFAULT, primary protocol)
      Our actual sentence-level LM1B dataset (held-out test split by
      default). This is the correct, apples-to-apples protocol for a
      checkpoint trained on sentence-level data (see
      docs/lm1b_mauve_debug.md for why).

  --reference-mode jpeg_packed
      JPEG-DLM's released packed-sequence LM1B reference file. Scores
      against a DIFFERENT distribution than our sentence-trained checkpoint
      was ever trained on. Results under this mode are always reported
      under a "cross_protocol" key and must NEVER be presented as a
      same-protocol comparison to the paper's ~0.95 headline number.

Does not touch training, model weights, or OWT's existing evaluator
(generation.py / utils/metrics_utils.compute_mauve's default behavior is
unchanged -- this script only adds optional, explicit parameters).

Usage (D0/D1 -- reproduce old score, then correct LM1B truncation):
    python eval_lm1b_mauve.py \
        --generated-files outputs/.../all_generated_0_0.jsonl \
        --reference-mode sentence \
        --reference-path /path/to/lm1b_test_t5 \
        --max-text-length 1024 --num-references 1000 --seed 42 \
        --output-dir results/lm1b_mauve/D0_old_setting

    python eval_lm1b_mauve.py \
        --generated-files outputs/.../all_generated_0_0.jsonl \
        --reference-mode sentence \
        --reference-path /path/to/lm1b_test_t5 \
        --max-text-length 96 --num-references 1000 --seed 42 \
        --output-dir results/lm1b_mauve/D1_correct_truncation

Usage (D4 -- human-human sanity baseline, disjoint subsets):
    python eval_lm1b_mauve.py --run-controls \
        --reference-path /path/to/lm1b_test_t5 \
        --max-text-length 96 --num-references 1000 --seed 42 \
        --output-dir results/lm1b_mauve/D4_human_human

Usage (D7 -- cross-protocol diagnostic against JPEG-DLM's packed refs):
    python eval_lm1b_mauve.py \
        --generated-files outputs/.../all_generated_0_0.jsonl \
        --reference-mode jpeg_packed \
        --reference-path third_party/jpeg-dlm/references/lm1b_t5_mauve96.jsonl \
        --max-text-length 96 --num-references 1000 --seed 42 \
        --output-dir results/lm1b_mauve/D7_cross_protocol_jpeg_refs
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.data_utils import load_dataset_split  # noqa: E402
from utils.metrics_utils import compute_mauve  # noqa: E402


# ============================================================
# Loading: generated predictions (auto-detect key, explicit override allowed)
# ============================================================

def load_predictions(path, text_field=None):
    """Load one JSONL file of generated samples. Auto-detects the `generated`
    (CorrFlow native) or `prediction` (JPEG-DLM-compatible) key; raises a
    clear error if neither is present and no explicit --text-field is given."""
    texts, blanks = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_field is not None:
                if text_field not in obj:
                    raise KeyError(
                        f"{path}:{line_no}: requested --text-field '{text_field}' "
                        f"not found; available keys: {list(obj.keys())}"
                    )
                field = text_field
            elif "generated" in obj:
                field = "generated"
            elif "prediction" in obj:
                field = "prediction"
            else:
                raise KeyError(
                    f"{path}:{line_no}: neither 'generated' nor 'prediction' key "
                    f"found; available keys: {list(obj.keys())}. Pass --text-field "
                    f"explicitly if this file uses a different schema."
                )
            text = obj[field]
            if not isinstance(text, str) or not text.strip():
                blanks += 1
            texts.append(text if isinstance(text, str) else "")
    return texts, blanks


# ============================================================
# Loading: references
# ============================================================

def load_sentence_references(data_path, tokenizer, num_references, seed, pool_cap=None):
    """Random, without-replacement sample of real LM1B sentences, decoded with
    the SAME tokenizer used at generation time (T5, NOT the GPT-2 MAUVE
    featurizer). Returns (texts, chosen_row_indices) so the exact reference
    set is reproducible and persistable."""
    ds = load_dataset_split(data_path)
    pool_n = len(ds) if pool_cap is None else min(pool_cap, len(ds))
    rng = np.random.RandomState(seed)
    indices = rng.choice(pool_n, size=min(num_references, pool_n), replace=False)
    texts = []
    for idx in indices:
        row = ds[int(idx)]
        seq_len = int(row["sequence_length"])
        ids = np.asarray(row["input_ids"])[:seq_len]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts, indices.tolist()


def load_jpeg_packed_references(jsonl_path, field="reference"):
    texts, indices = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(str(obj[field]))
            indices.append(obj.get("index"))
    return texts, indices


# ============================================================
# Diagnostics
# ============================================================

def length_stats(name, texts, gpt2_tokenizer, thresholds=(32, 64, 96, 128)):
    nonempty = [t for t in texts if isinstance(t, str) and t.strip()]
    blanks = len(texts) - len(nonempty)
    lens = np.array([len(gpt2_tokenizer.encode(t)) for t in nonempty]) if nonempty else np.array([])
    dup = len(nonempty) - len(set(nonempty))
    stats = {
        "name": name,
        "total": len(texts),
        "nonempty": len(nonempty),
        "blank": blanks,
        "duplicates": dup,
    }
    if len(lens) > 0:
        stats.update({
            "mean": float(lens.mean()), "std": float(lens.std()), "median": float(np.median(lens)),
            "min": int(lens.min()), "max": int(lens.max()),
            **{f"p{p}": float(np.percentile(lens, p)) for p in (5, 10, 25, 50, 75, 90, 95)},
            **{f"frac_lt_{t}": float(np.mean(lens < t)) for t in thresholds},
            **{f"n_eligible_ge_{t}": int(np.sum(lens >= t)) for t in thresholds},
        })
    return stats, nonempty


def print_stats(stats):
    print(f"--- {stats['name']} ---")
    for k, v in stats.items():
        if k == "name":
            continue
        print(f"  {k}: {v}")


# ============================================================
# MAUVE scoring (standard + strict-length diagnostic)
# ============================================================

def score_mauve(gen_texts, ref_texts, featurizer, max_text_length, device, seed, strict=False):
    """strict=True: keep only texts with >= max_text_length GPT-2 tokens, else
    pad-free truncation-only scoring (compute_mauve's normal behavior)."""
    if not strict:
        result = compute_mauve(
            generated_texts=gen_texts, reference_texts=ref_texts,
            featurize_model_name=featurizer, device_id=device,
            max_text_length=max_text_length, seed=seed,
            equalize_sample_counts=False,
        )
        result["mode"] = "standard_truncate"
        result["gen_retention"] = 1.0
        result["ref_retention"] = 1.0
        return result

    from transformers import AutoTokenizer
    gpt2_tok = AutoTokenizer.from_pretrained(featurizer)
    def filt(texts):
        kept = [t for t in texts if len(gpt2_tok.encode(t)) >= max_text_length]
        return kept
    gen_kept, ref_kept = filt(gen_texts), filt(ref_texts)
    if len(gen_kept) < 2 or len(ref_kept) < 2:
        return {
            "mauve": float("nan"), "mode": "strict_length_filtered",
            "num_generated": len(gen_kept), "num_reference": len(ref_kept),
            "gen_retention": len(gen_kept) / max(len(gen_texts), 1),
            "ref_retention": len(ref_kept) / max(len(ref_texts), 1),
            "error": "fewer than 2 eligible samples on one side; refusing to report",
        }
    result = compute_mauve(
        generated_texts=gen_kept, reference_texts=ref_kept,
        featurize_model_name=featurizer, device_id=device,
        max_text_length=max_text_length, seed=seed,
        equalize_sample_counts=False,
    )
    result["mode"] = "strict_length_filtered"
    result["gen_retention"] = len(gen_kept) / max(len(gen_texts), 1)
    result["ref_retention"] = len(ref_kept) / max(len(ref_texts), 1)
    return result


# ============================================================
# Metadata / provenance
# ============================================================

def git_sha():
    try:
        return subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def library_versions():
    versions = {}
    for mod in ("mauve", "transformers", "torch"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not available"
    return versions


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generated-files", nargs="*", default=[],
                     help="One or more all_generated_*.jsonl files. Omit when --run-controls.")
    ap.add_argument("--text-field", default=None,
                     help="Override auto-detected key ('generated' or 'prediction').")
    ap.add_argument("--reference-mode", choices=["sentence", "jpeg_packed"], default="sentence")
    ap.add_argument("--reference-path", required=True,
                     help="Arrow dataset path/HF id (sentence mode) or JSONL path (jpeg_packed mode).")
    ap.add_argument("--tokenizer", default="t5-small",
                     help="Tokenizer for decoding reference arrow rows (generation tokenizer, NOT the MAUVE featurizer).")
    ap.add_argument("--featurizer", default="gpt2-large")
    ap.add_argument("--max-text-length", type=int, default=96,
                     help="GPT-2 token truncation window for MAUVE featurization "
                          "(96 = LM1B protocol default; sweep with repeated calls for 32/64/128).")
    ap.add_argument("--num-generated", type=int, default=1000)
    ap.add_argument("--num-references", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    ap.add_argument("--batch-size", type=int, default=1, help="Unused placeholder for parity with other evaluators.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--strict-length", action="store_true",
                     help="Also report the strict length-filtered diagnostic (OFF by default). "
                          "Never the headline number -- always reported alongside retention rate.")
    ap.add_argument("--lowercase", action="store_true",
                     help="Lowercase BOTH generated and reference text identically before scoring "
                          "(explicit protocol choice, not silently applied).")
    ap.add_argument("--run-controls", action="store_true",
                     help="Run D4 (disjoint held-out human-human) and D5 (identical-text) sanity checks "
                          "instead of scoring --generated-files.")
    args = ap.parse_args()
    if not args.run_controls and not args.generated_files:
        ap.error("--generated-files is required unless --run-controls is set")
    return args


def device_to_mauve_id(device):
    if device == "cpu":
        return -1
    if device.startswith("cuda"):
        return 0 if ":" not in device else int(device.split(":")[1])
    raise ValueError(f"Unrecognized --device {device!r}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    from transformers import AutoTokenizer
    ref_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    gpt2_tok = AutoTokenizer.from_pretrained(args.featurizer)
    device_id = device_to_mauve_id(args.device)

    report = {
        "git_sha": git_sha(),
        "library_versions": library_versions(),
        "args": vars(args),
    }

    def maybe_lower(texts):
        return [t.lower() for t in texts] if args.lowercase else texts

    if args.run_controls:
        # D4: disjoint held-out subsets A / B (human-human sanity baseline).
        # D5: identical texts vs itself (near-1 identity check).
        ds = load_dataset_split(args.reference_path)
        n_each = args.num_references
        rng = np.random.RandomState(args.seed)
        all_idx = rng.choice(len(ds), size=2 * n_each, replace=False)
        idx_a, idx_b = all_idx[:n_each], all_idx[n_each:]
        assert set(idx_a.tolist()).isdisjoint(set(idx_b.tolist())), "D4 subsets must be disjoint"

        def decode(indices):
            texts = []
            for idx in indices:
                row = ds[int(idx)]
                seq_len = int(row["sequence_length"])
                ids = np.asarray(row["input_ids"])[:seq_len]
                texts.append(ref_tokenizer.decode(ids, skip_special_tokens=True))
            return texts

        texts_a, texts_b = maybe_lower(decode(idx_a)), maybe_lower(decode(idx_b))
        stats_a, nonempty_a = length_stats("D4_subset_A", texts_a, gpt2_tok)
        stats_b, nonempty_b = length_stats("D4_subset_B", texts_b, gpt2_tok)
        print_stats(stats_a)
        print_stats(stats_b)

        d4 = score_mauve(nonempty_a, nonempty_b, args.featurizer, args.max_text_length, device_id, args.seed)
        print(f"D4 (disjoint human-human, max_len={args.max_text_length}): MAUVE={d4['mauve']:.4f}")

        d5 = score_mauve(nonempty_a, nonempty_a, args.featurizer, args.max_text_length, device_id, args.seed)
        print(f"D5 (identical texts, max_len={args.max_text_length}): MAUVE={d5['mauve']:.4f} (expect near 1, not exactly 1)")

        report.update({
            "experiment": "run_controls_D4_D5",
            "reference_ids_subset_A": idx_a.tolist(),
            "reference_ids_subset_B": idx_b.tolist(),
            "length_stats": {"subset_A": stats_a, "subset_B": stats_b},
            "D4_disjoint_human_human": d4,
            "D5_identical_text_sanity": d5,
        })
        if args.strict_length:
            d4_strict = score_mauve(nonempty_a, nonempty_b, args.featurizer, args.max_text_length, device_id, args.seed, strict=True)
            report["D4_strict_length_filtered"] = d4_strict
            print(f"D4 strict (len>={args.max_text_length}): MAUVE={d4_strict['mauve']:.4f} "
                  f"gen_retention={d4_strict['gen_retention']:.3f} ref_retention={d4_strict['ref_retention']:.3f}")

        with open(os.path.join(args.output_dir, "result.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved: {os.path.join(args.output_dir, 'result.json')}")
        return

    # --- Load references ---
    if args.reference_mode == "sentence":
        ref_texts, ref_ids = load_sentence_references(
            args.reference_path, ref_tokenizer, args.num_references, args.seed,
        )
    else:
        ref_texts, ref_ids = load_jpeg_packed_references(args.reference_path)
        ref_texts = ref_texts[:args.num_references]
        ref_ids = ref_ids[:args.num_references]
    ref_texts = maybe_lower(ref_texts)
    ref_stats, ref_nonempty = length_stats(f"reference[{args.reference_mode}]", ref_texts, gpt2_tok)
    print_stats(ref_stats)

    report.update({
        "reference_mode": args.reference_mode,
        "reference_path": args.reference_path,
        "reference_ids": ref_ids,
        "reference_length_stats": ref_stats,
        "per_file": [],
    })
    if args.reference_mode == "jpeg_packed":
        report["cross_protocol_warning"] = (
            "reference_mode=jpeg_packed: scoring against JPEG-DLM's PACKED multi-sentence "
            "reference distribution, NOT the same distribution this sentence-trained "
            "checkpoint was trained/evaluated on elsewhere in this report. This result is "
            "a cross-protocol diagnostic only and must not be presented as comparable to "
            "the paper's same-protocol ~0.95 headline MAUVE."
        )

    for path in args.generated_files:
        gen_texts, blanks = load_predictions(path, args.text_field)
        gen_texts = maybe_lower(gen_texts[:args.num_generated])
        gen_stats, gen_nonempty = length_stats(os.path.basename(path), gen_texts, gpt2_tok)
        gen_stats["blank_in_raw_file"] = blanks
        print_stats(gen_stats)

        standard = score_mauve(gen_nonempty, ref_nonempty, args.featurizer, args.max_text_length, device_id, args.seed)
        print(f"[{os.path.basename(path)}] standard MAUVE (max_len={args.max_text_length}): {standard['mauve']:.4f} "
              f"(n_gen={standard['num_generated']}, n_ref={standard['num_reference']})")

        entry = {
            "file": path,
            "generated_length_stats": gen_stats,
            "standard": standard,
        }
        if args.strict_length:
            strict = score_mauve(gen_nonempty, ref_nonempty, args.featurizer, args.max_text_length, device_id, args.seed, strict=True)
            print(f"[{os.path.basename(path)}] strict MAUVE (len>={args.max_text_length}): {strict['mauve']:.4f} "
                  f"gen_retention={strict['gen_retention']:.3f} ref_retention={strict['ref_retention']:.3f}")
            entry["strict_length_filtered"] = strict
        report["per_file"].append(entry)

    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {os.path.join(args.output_dir, 'result.json')}")


if __name__ == "__main__":
    main()
