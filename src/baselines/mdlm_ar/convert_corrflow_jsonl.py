#!/usr/bin/env python
"""Convert CorrFlow/ELF's saved generation JSONL (`{"id": ..., "generated": ...}`,
written by generation.py's test_generation_uncond) into the `{"prediction": ...}`
format JPEG-DLM's gen_ppl.py/mauve_score.py expect (via read_predictions /
read_texts, which both do `json.loads(line)["prediction"]`).

Preserves text exactly — no normalization, whitespace changes, or truncation.
Does not touch CorrFlow's original file or generation logic.

Usage:
    python convert_corrflow_jsonl.py \
        --input outputs/corrflow_stage2-owt/sde-steps64-.../all_generated_0_15000.jsonl \
        --output outputs/baselines/converted/corrflow_stage2owt_steps64_predictions.jsonl
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CorrFlow/ELF all_generated_*.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-field", default="generated",
                     help="Field name holding the generated text in the input file")
    args = ap.parse_args()

    n_in, n_out, n_blank = 0, 0, 0
    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            obj = json.loads(line)
            text = obj[args.text_field]
            if not text.strip():
                n_blank += 1
            f_out.write(json.dumps({
                "prediction": text,
                "source_id": obj.get("id"),
                "source_file": args.input,
            }, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Converted {n_out}/{n_in} lines from {args.input} -> {args.output}")
    print(f"Blank predictions: {n_blank} (preserved as-is, not filtered)")


if __name__ == "__main__":
    main()
