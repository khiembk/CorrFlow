#!/usr/bin/env python
"""Unconditional generation from the LM1B AR checkpoint trained by
train_lm1b.py. Separate from generate.py (OWT baseline) because that script
loads MDLM's own Lightning checkpoint format via Hydra + model=small-ar,
data=openwebtext-split, while train_lm1b.py saves a raw state_dict with its
own metadata (vocab_size, max_length, tokenizer_name) and uses T5's
tokenizer/vocab, not BERT's.

No KV cache, same as generate.py: recomputes the full growing prefix at
every step (correctness-first, not throughput-optimized).

T5 has no native BOS token (tokenizer.bos_token_id is None), and
train_lm1b.py's training data has no BOS prepended either -- position 0 is
always a real content token, never a predicted target (the causal shift is
log_probs[:, :-1] vs input_ids[:, 1:], so position 0 is context-only).
Unconditional generation therefore seeds each sequence by sampling a
reference the first-token's empirical marginal distribution from the
training set, computed once and cached, rather than a fixed BOS id.

Usage:
    conda activate mdlm-ar
    python generate_lm1b.py \
        --mdlm-repo /scratch/.../third_party/mdlm \
        --checkpoint /scratch/.../ar_lm1b_run/checkpoint_80000.pt \
        --data-path /home.na1/.../data/lm1b_train_t5 \
        --output-dir outputs/baselines/ar_lm1b_gen \
        --num-samples 1000 --batch-size 32 \
        --sampler nucleus --top-p 0.9 --temperature 1.0 \
        --seed 0 --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdlm-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", required=True,
                     help="Same LM1B arrow dataset used in training, for the first-token seed distribution.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--sampler", choices=["nucleus", "full"], default="nucleus")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed-pool-size", type=int, default=50000,
                     help="How many training rows to sample first-token stats from.")
    args = ap.parse_args()
    if args.temperature <= 0:
        raise ValueError(f"--temperature must be > 0, got {args.temperature}")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError(f"--top-p must be in (0, 1], got {args.top_p}")
    return args


def load_checkpoint_and_model(mdlm_repo, checkpoint_path, device):
    sys.path.insert(0, mdlm_repo)
    import omegaconf
    from models.autoregressive import AR

    ckpt = torch.load(checkpoint_path, map_location=device)
    vocab_size = ckpt["vocab_size"]
    max_length = ckpt["max_length"]
    arch = ckpt["model_arch"]

    model_config = omegaconf.OmegaConf.create({
        "model": {
            "name": "small", "type": "ddit", "hidden_size": arch["hidden_size"],
            "cond_dim": 128, "length": max_length, "n_blocks": arch["n_blocks"],
            "n_heads": arch["n_heads"], "scale_by_sigma": True, "dropout": 0.1,
            "causal": arch["causal"], "tie_word_embeddings": False,
        },
    })
    mask_index = vocab_size
    model = AR(model_config, vocab_size=vocab_size + 1, mask_index=mask_index)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, ckpt


def empirical_first_token_distribution(data_path, tokenizer_name, pool_size, seed):
    """Sample pool_size training rows' input_ids[0] (the real first content
    token of each sentence) and return (unique_tokens, probs) for sampling."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from datasets import load_dataset as hf_load_dataset, load_from_disk, DatasetDict
    try:
        ds = hf_load_dataset(data_path)
    except Exception:
        ds = load_from_disk(data_path)
    if isinstance(ds, DatasetDict):
        ds = ds[list(ds.keys())[0]]

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=min(pool_size, len(ds)), replace=False)
    first_tokens = [int(ds[int(i)]["input_ids"][0]) for i in idx]
    unique, counts = np.unique(first_tokens, return_counts=True)
    probs = counts / counts.sum()
    return unique, probs


def top_p_filter(probs, top_p):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cum_probs - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
    sorted_probs[:, 0] = torch.where(
        sorted_probs[:, 0] == 0, probs.gather(-1, sorted_idx[:, :1]).squeeze(-1), sorted_probs[:, 0]
    )
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)


@torch.no_grad()
def generate_batch(model, seed_tokens, max_length, sampler, top_p, temperature, device, generator):
    x = seed_tokens.unsqueeze(1).to(device)  # (batch, 1)
    for _ in range(max_length - 1):
        log_probs = model(x, None)[:, -1, :]
        if sampler == "full":
            noise = -torch.log(-torch.log(
                torch.rand(log_probs.shape, generator=generator, device=device).clamp_min(1e-20)
            ).clamp_min(1e-20))
            next_token = (log_probs / temperature + noise).argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(log_probs / temperature, dim=-1)
            probs = top_p_filter(probs, top_p)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        x = torch.cat([x, next_token], dim=1)
    return x


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    model, ckpt = load_checkpoint_and_model(args.mdlm_repo, args.checkpoint, args.device)
    max_length = ckpt["max_length"]
    tokenizer_name = ckpt["tokenizer_name"]
    print(f"Loaded checkpoint at step {ckpt['step']}, max_length={max_length}, tokenizer={tokenizer_name}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Computing empirical first-token distribution from {args.seed_pool_size} training rows...")
    seed_vocab, seed_probs = empirical_first_token_distribution(
        args.data_path, tokenizer_name, args.seed_pool_size, args.seed,
    )
    print(f"First-token distribution: {len(seed_vocab)} unique tokens")

    os.makedirs(args.output_dir, exist_ok=True)
    predictions_path = os.path.join(args.output_dir, f"predictions_seed{args.seed}.jsonl")

    rng = np.random.RandomState(args.seed)
    n_generated = 0
    t0 = time.time()
    with open(predictions_path, "w", encoding="utf-8") as f_pred:
        while n_generated < args.num_samples:
            cur_bsz = min(args.batch_size, args.num_samples - n_generated)
            seed_ids = rng.choice(seed_vocab, size=cur_bsz, p=seed_probs)
            seed_tensor = torch.tensor(seed_ids, dtype=torch.long)

            tokens = generate_batch(
                model, seed_tensor, max_length, args.sampler, args.top_p,
                args.temperature, args.device, generator,
            )
            tokens_cpu = tokens.cpu().tolist()
            for row in tokens_cpu:
                text = tokenizer.decode(row, skip_special_tokens=True)
                f_pred.write(json.dumps({"prediction": text}) + "\n")
            n_generated += cur_bsz
            elapsed = time.time() - t0
            print(f"Generated {n_generated}/{args.num_samples} ({elapsed:.1f}s elapsed)", flush=True)

    metadata = {
        "checkpoint": args.checkpoint, "checkpoint_step": ckpt["step"],
        "tokenizer_name": tokenizer_name, "max_length": max_length,
        "sampler": args.sampler, "top_p": args.top_p, "temperature": args.temperature,
        "seed": args.seed, "num_samples": args.num_samples,
        "seed_pool_size": args.seed_pool_size,
    }
    with open(os.path.join(args.output_dir, f"metadata_seed{args.seed}.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {args.num_samples} predictions to {predictions_path}")


if __name__ == "__main__":
    main()
