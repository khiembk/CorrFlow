"""Preprocess LM1B into HuggingFace arrow format (T5 tokenizer IDs).

Reads raw LM1B training/heldout shards, tokenizes with T5-small tokenizer,
filters by max_length, and saves as arrow dataset matching the OWT-t5 format:
  {'input_ids': List[int32], 'sequence_length': int64}

Usage:
    python preprocess_lm1b.py \
        --lm1b_dir /path/to/1-billion-word-.../  \
        --output_train /path/to/lm1b_train_t5 \
        --output_test  /path/to/lm1b_test_t5 \
        --max_length 128
"""

import argparse
import glob
import os
import sys

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def tokenize_shard(shard_path, tokenizer, max_length):
    input_ids_list, seq_len_list = [], []
    with open(shard_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    batch_size = 10000
    for i in range(0, len(lines), batch_size):
        batch = lines[i : i + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        for ids in enc["input_ids"]:
            if len(ids) == 0:
                continue
            ids_arr = np.array(ids, dtype=np.int32)
            input_ids_list.append(ids_arr)
            seq_len_list.append(len(ids_arr))
    return input_ids_list, seq_len_list


def process_split(shard_paths, tokenizer, max_length, output_path, split_name):
    all_input_ids, all_seq_lens = [], []
    total = len(shard_paths)
    for idx, shard in enumerate(sorted(shard_paths)):
        print(f"[{split_name}] Shard {idx+1}/{total}: {os.path.basename(shard)}", flush=True)
        ids_list, lens_list = tokenize_shard(shard, tokenizer, max_length)
        all_input_ids.extend(ids_list)
        all_seq_lens.extend(lens_list)
        print(f"  → {len(ids_list):,} sentences (total so far: {len(all_input_ids):,})", flush=True)

    print(f"\nBuilding arrow dataset ({len(all_input_ids):,} examples)...", flush=True)
    ds = Dataset.from_dict({
        "input_ids": [ids.tolist() for ids in all_input_ids],
        "sequence_length": all_seq_lens,
    })
    ds = ds.cast_column("input_ids", ds.features["input_ids"])
    print(f"Saving to {output_path}...", flush=True)
    ds.save_to_disk(output_path)
    print(f"Done. {len(ds):,} examples saved to {output_path}", flush=True)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm1b_dir", required=True,
                        help="Path to unpacked LM1B root dir (contains training-*/heldout-* subdirs)")
    parser.add_argument("--output_train", required=True)
    parser.add_argument("--output_test", required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--tokenizer", default="t5-small")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    train_shards = glob.glob(
        os.path.join(args.lm1b_dir, "training-monolingual.tokenized.shuffled", "news.en-*")
    )
    test_shards = glob.glob(
        os.path.join(args.lm1b_dir, "heldout-monolingual.tokenized.shuffled", "news.en.heldout-*")
    )

    print(f"Found {len(train_shards)} train shards, {len(test_shards)} test shards", flush=True)
    print(f"max_length={args.max_length}", flush=True)

    process_split(train_shards, tokenizer, args.max_length, args.output_train, "train")
    process_split(test_shards,  tokenizer, args.max_length, args.output_test,  "test")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
