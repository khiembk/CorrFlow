#!/usr/bin/env python
"""Train an AR baseline on LM1B from scratch, using CorrFlow's own
T5-tokenized `lm1b_train_t5` dataset (reused directly, no retokenization)
and MDLM's `models.autoregressive.AR` backbone class (same architecture
family as the OWT `ar.ckpt`: causal transformer, RoPE, flash-attn).

No pretrained OWT checkpoint is reused here — per the OWT/LM1B provenance
investigation (see docs/AR_BASELINE_JPEG_DLM.md), the GPT-2-vocab ar.ckpt
cannot be loaded onto a T5-vocab model, and JPEG-DLM has no public LM1B AR
checkpoint either. This is a genuine from-scratch training run, scaled down
from CorrFlow Stage 1's own ~1M-step LM1B recipe to a faster rebuttal-timeline
budget (~100-200k steps) per explicit user decision.

Does NOT use MDLM's Hydra/Lightning train loop — that pipeline assumes its
own HF-streaming data format and OWT-specific wrapping/packing conventions.
This is a plain PyTorch training loop instead, so it can run against
CorrFlow's already-tokenized local data directly.

Loss masking follows the SAME critical lesson already learned the hard way
for LM1B in this project (see CLAUDE.md "Loss mask — CRITICAL"): only
pad_token==eos is used as padding, and the loss is masked to real tokens via
attention_mask, never `ones_like`.

Usage:
    conda activate mdlm-ar
    python train_lm1b.py \
        --mdlm-repo /path/to/mdlm \
        --data-path /path/to/lm1b_train_t5 \
        --tokenizer-name t5-small \
        --output-dir outputs/baselines/ar_lm1b \
        --max-steps 150000 --batch-size 256 --max-length 128 \
        --lr 3e-4 --warmup-steps 2000 \
        --save-every 10000 --log-every 100 --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdlm-repo", required=True)
    ap.add_argument("--data-path", required=True, help="CorrFlow lm1b_train_t5 path (HF hub id or local dir)")
    ap.add_argument("--tokenizer-name", default="t5-small")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=150_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=10_000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    return ap.parse_args()


def load_lm1b_dataset(path):
    """Minimal, jax-free reimplementation of CorrFlow's load_dataset_split,
    since utils/data_utils.py imports jax (not installed in this torch env).
    Same fallback behavior: HF hub id first, then local on-disk Arrow."""
    from datasets import load_dataset as hf_load_dataset, load_from_disk, DatasetDict
    try:
        ds = hf_load_dataset(path)
    except Exception:
        ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        assert len(splits) == 1, f"Expected a single split at {path}, got {splits}"
        ds = ds[splits[0]]
    return ds


class LM1BTorchDataset(torch.utils.data.Dataset):
    """Wraps the HF arrow dataset ({'input_ids', 'sequence_length'}) for
    plain PyTorch DataLoader iteration. Padding/truncation happens in the
    collate_fn, mirroring CorrFlow's utils/data_utils.py:pad_and_truncate."""

    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        return np.asarray(row["input_ids"], dtype=np.int64), int(row["sequence_length"])


def make_collate_fn(max_length, pad_token_id):
    def collate_fn(batch):
        input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_length), dtype=torch.float32)
        for i, (ids, seq_len) in enumerate(batch):
            n = min(seq_len, max_length, len(ids))
            input_ids[i, :n] = torch.from_numpy(ids[:n].copy())
            attention_mask[i, :n] = 1.0
        return input_ids, attention_mask
    return collate_fn


def build_ar_model(mdlm_repo, vocab_size, max_length, device):
    sys.path.insert(0, mdlm_repo)
    import omegaconf
    from models.autoregressive import AR

    model_config = omegaconf.OmegaConf.create({
        "model": {
            "name": "small",
            "type": "ddit",
            "hidden_size": 768,
            "cond_dim": 128,
            "length": max_length,
            "n_blocks": 12,
            "n_heads": 12,
            "scale_by_sigma": True,
            "dropout": 0.1,
            "causal": True,
            "tie_word_embeddings": False,
        },
    })
    mask_index = vocab_size  # no native mask token in T5's vocab; add one unused slot
    # DDIT.__init__ expects the FULL config (reads config.model.xxx), matching how
    # MDLM's own Diffusion class passes its whole hydra config through, not just
    # the model sub-tree.
    model = AR(model_config, vocab_size=vocab_size + 1, mask_index=mask_index)
    return model.to(device), model_config


def lr_lambda(step, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 1.0


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    pad_token_id = tokenizer.eos_token_id  # pad_token: eos, matching CorrFlow's LM1B config
    vocab_size = len(tokenizer)

    print(f"Tokenizer: {args.tokenizer_name}, vocab_size={vocab_size}, pad_token_id(eos)={pad_token_id}")

    ds = load_lm1b_dataset(args.data_path)
    print(f"Loaded LM1B dataset: {len(ds)} examples from {args.data_path}")
    torch_ds = LM1BTorchDataset(ds)
    collate_fn = make_collate_fn(args.max_length, pad_token_id)
    loader = torch.utils.data.DataLoader(
        torch_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, drop_last=True, persistent_workers=True,
    )

    model, model_config = build_ar_model(args.mdlm_repo, vocab_size, args.max_length, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AR model built: {n_params:,} parameters, vocab_size={vocab_size + 1} (incl. unused mask slot)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda(step, args.warmup_steps),
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"]
        print(f"Resumed from {args.resume} at step {start_step}")

    def save_checkpoint(step):
        path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "vocab_size": vocab_size,
            "mask_index": vocab_size,
            "max_length": args.max_length,
            "tokenizer_name": args.tokenizer_name,
            "model_arch": {"hidden_size": 768, "n_blocks": 12, "n_heads": 12, "causal": True},
        }, path)
        print(f"Saved checkpoint: {path}")

    model.train()
    step = start_step
    t0 = time.time()
    running_loss = 0.0
    running_tokens = 0
    log_loss_sum, log_steps = 0.0, 0

    data_iter = iter(loader)
    while step < args.max_steps:
        try:
            input_ids, attention_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            input_ids, attention_mask = next(data_iter)

        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)

        log_probs = model(input_ids, None)  # (B, L, vocab+1), log_softmax'd, mask slot clamped
        # Standard causal LM shift: predict token[t+1] from tokens[:t+1].
        shift_log_probs = log_probs[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]  # CRITICAL: real-token mask, never ones_like

        token_nll = -shift_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        masked_nll = token_nll * shift_mask
        n_tokens = shift_mask.sum().clamp_min(1.0)
        loss = masked_nll.sum() / n_tokens

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        log_loss_sum += loss.item()
        log_steps += 1
        running_tokens += int(n_tokens.item())
        step += 1

        if step % args.log_every == 0:
            avg_loss = log_loss_sum / max(log_steps, 1)
            elapsed = time.time() - t0
            print(f"step {step}/{args.max_steps} | loss {avg_loss:.4f} | ppl {np.exp(avg_loss):.2f} "
                  f"| lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s elapsed "
                  f"| {step / max(elapsed, 1e-6):.2f} steps/s", flush=True)
            log_loss_sum, log_steps = 0.0, 0

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(step)

    with open(os.path.join(args.output_dir, "train_metadata.json"), "w") as f:
        json.dump({
            "data_path": args.data_path,
            "tokenizer_name": args.tokenizer_name,
            "vocab_size": vocab_size,
            "max_length": args.max_length,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "total_wall_time_seconds": time.time() - t0,
            "n_params": n_params,
            "architecture": "MDLM models.autoregressive.AR (12 blocks, hidden 768, 12 heads, causal, RoPE)",
            "pretrained_init": None,
            "note": "Trained from scratch on T5-tokenized LM1B; no OWT ar.ckpt weights reused "
                    "(incompatible vocab/distribution; see docs/AR_BASELINE_JPEG_DLM.md).",
        }, f, indent=2)
    print("Training complete.")


if __name__ == "__main__":
    main()
