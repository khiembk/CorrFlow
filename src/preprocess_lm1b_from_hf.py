"""Preprocess LM1B into HuggingFace arrow format (T5 tokenizer IDs), sourcing
text from the cached `dvruette/lm1b` HF dataset instead of raw statmt.org
shards (same official 1-billion-word benchmark train/heldout split: 30,301,028
train / 306,688 test sentences) since statmt.org serves the raw tarball at a
throttled ~135KB/s (~3.3h for 1.7GB), while the HF copy is already cached.

Output format matches preprocess_lm1b.py exactly:
  {'input_ids': List[int32], 'sequence_length': int64}

Usage:
    python preprocess_lm1b_from_hf.py \
        --output_train /path/to/lm1b_train_t5 \
        --output_test  /path/to/lm1b_test_t5 \
        --max_length 128
"""

import argparse
import os
import sys

import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def tokenize_split(hf_split, tokenizer, max_length, split_name):
    """Tokenize and append EOS so the dataloader's attention_mask covers the
    stop position — without it, the decoder CE loss never sees a "terminate
    here" target and generation never learns to emit EOS (runs to max_length)."""
    all_input_ids, all_seq_lens = [], []
    batch_size = 10000
    total = len(hf_split)
    for i in range(0, total, batch_size):
        batch = hf_split[i : i + batch_size]["text"]
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 1,
            padding=False,
        )
        for ids in enc["input_ids"]:
            if len(ids) == 0:
                continue
            ids_arr = np.array(ids + [tokenizer.eos_token_id], dtype=np.int32)
            all_input_ids.append(ids_arr)
            all_seq_lens.append(len(ids_arr))
        if (i // batch_size) % 50 == 0:
            print(f"[{split_name}] {i:,}/{total:,} ({len(all_input_ids):,} sentences so far)", flush=True)
    return all_input_ids, all_seq_lens


def save_split(all_input_ids, all_seq_lens, output_path):
    print(f"\nBuilding arrow dataset ({len(all_input_ids):,} examples)...", flush=True)
    ds = Dataset.from_dict({
        "input_ids": [ids.tolist() for ids in all_input_ids],
        "sequence_length": all_seq_lens,
    })
    print(f"Saving to {output_path}...", flush=True)
    ds.save_to_disk(output_path)
    print(f"Done. {len(ds):,} examples saved to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_train", required=True)
    parser.add_argument("--output_test", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--tokenizer", default="t5-small")
    parser.add_argument("--hf_cache_dir", default="/scratch/khiem.tran@ad.wsu.edu/hf_cache")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print("Loading dvruette/lm1b from HF cache...", flush=True)
    ds = load_dataset("dvruette/lm1b", cache_dir=args.hf_cache_dir)
    print(f"train={len(ds['train']):,} test={len(ds['test']):,}", flush=True)

    test_ids, test_lens = tokenize_split(ds["test"], tokenizer, args.max_length, "test")
    save_split(test_ids, test_lens, args.output_test)

    train_ids, train_lens = tokenize_split(ds["train"], tokenizer, args.max_length, "train")
    save_split(train_ids, train_lens, args.output_train)

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
