#!/usr/bin/env python
"""Train a standard HuggingFace GPT2LMHeadModel from scratch on CorrFlow's
own T5-tokenized OWT dataset (`embedded-language-flows/openwebtext-t5`,
reused directly, no retokenization).

Rationale (rebuttal AR baseline #2, alongside the existing MDLM-paper
~130M-param AR baseline): that existing baseline reuses MDLM's OWN released
pretrained ar.ckpt (trained by the MDLM authors on their own OWT
tokenization/pipeline) -- it is not trained by this project and uses a
different architecture family (MDLM's DiT-style causal backbone) and a
different tokenizer (GPT-2 BPE, 50257 vocab) than CorrFlow's own ELF-B
backbone (T5 tokenizer, 32128 vocab). This script trains the LITERAL
HuggingFace GPT2LMHeadModel architecture (standard GPT-2-small config:
n_layer=12, n_head=12, n_embd=768) from scratch, on the EXACT SAME
data/tokenizer ELF-B/CorrFlow were trained on. Two things this buys:
  1. GPT-2-small's dims (768/12/12) are nearly identical to ELF-B's
     (hidden=768, depth=12, heads=12, ~104M params) -- an
     architecture-matched AR baseline, not just a same-size one.
  2. Holding data/tokenizer fixed isolates the training OBJECTIVE
     (autoregressive CE vs CorrFlow's flow-matching + correlation
     correction) as the only varying factor versus ELF-B/CorrFlow.

IMPORTANT caveat for the rebuttal write-up: this is GPT-2-small's
ARCHITECTURE trained from scratch here, NOT the original off-the-shelf
GPT-2 checkpoint (which was pretrained on WebText with GPT-2's own BPE
tokenizer). Do not present this number as "the GPT-2 paper's GPT-2-small
score" -- it is a from-scratch reproduction under this project's own
data/tokenizer, for a controlled same-data comparison.

Same loss-masking convention as this project's LM1B AR baseline
(train_lm1b.py) and CorrFlow itself: pad_token=eos, loss masked to real
tokens via attention_mask, never `ones_like`.

Usage:
    conda activate mdlm-ar
    python train_gpt2_owt.py \
        --data-path embedded-language-flows/openwebtext-t5 \
        --tokenizer-name t5-small \
        --output-dir outputs/baselines/gpt2_owt \
        --max-steps 95000 --batch-size 32 --max-length 1024 \
        --lr 6e-4 --warmup-steps 2000 \
        --save-every 10000 --log-every 100 --device cuda:0
"""
import argparse
import json
import os
import time

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, help="CorrFlow openwebtext-t5 path (HF hub id or local dir)")
    ap.add_argument("--tokenizer-name", default="t5-small")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=95_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=10_000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    ap.add_argument("--grad-checkpointing", action="store_true", default=False,
                     help="Trade compute for memory. Usually unnecessary with bf16 autocast at batch<=64.")
    return ap.parse_args()


def load_owt_dataset(path):
    """Minimal loader matching CorrFlow's utils/data_utils.load_dataset_split
    (reimplemented jax-free here, same as train_lm1b.py:load_lm1b_dataset)."""
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


class OWTTorchDataset(torch.utils.data.Dataset):
    """Wraps the HF arrow dataset ({'input_ids', 'sequence_length'}) for
    plain PyTorch DataLoader iteration. Padding/truncation in collate_fn."""

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


def build_gpt2_small(vocab_size, max_length, device, grad_checkpointing=False):
    from transformers import GPT2Config, GPT2LMHeadModel
    # transformers==4.38.2 (this env's pin) does not support SDPA/flash-attn
    # for GPT2 -- only "eager", which is O(B*H*L^2) memory and OOMs at
    # seq_len=1024 for batch sizes beyond ~16-32. Gradient checkpointing
    # trades recompute for memory, enabling a larger effective batch via
    # more grad_accum_steps at a fixed micro-batch size instead.
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_embd=768, n_layer=12, n_head=12,
        bos_token_id=None, eos_token_id=None,  # resolved via the tokenizer at generation time, not baked in here
    )
    model = GPT2LMHeadModel(config)
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model.to(device), config


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
    pad_token_id = tokenizer.eos_token_id  # pad_token: eos, matching CorrFlow's OWT config
    vocab_size = len(tokenizer)
    print(f"Tokenizer: {args.tokenizer_name}, vocab_size={vocab_size}, pad_token_id(eos)={pad_token_id}")

    ds = load_owt_dataset(args.data_path)
    print(f"Loaded OWT dataset: {len(ds)} examples from {args.data_path}")
    torch_ds = OWTTorchDataset(ds)
    collate_fn = make_collate_fn(args.max_length, pad_token_id)
    loader = torch.utils.data.DataLoader(
        torch_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, drop_last=True, persistent_workers=True,
    )

    model, model_config = build_gpt2_small(vocab_size, args.max_length, args.device, args.grad_checkpointing)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2-small built: {n_params:,} parameters, vocab_size={vocab_size}")

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
            "max_length": args.max_length,
            "tokenizer_name": args.tokenizer_name,
            "model_arch": {"n_embd": 768, "n_layer": 12, "n_head": 12},
        }, path)
        print(f"Saved checkpoint: {path}")

    model.train()
    step = start_step
    t0 = time.time()
    log_loss_sum, log_steps = 0.0, 0
    accum_steps = max(args.grad_accum_steps, 1)

    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    while step < args.max_steps:
        try:
            input_ids, attention_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            input_ids, attention_mask = next(data_iter)

        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask=attention_mask).logits  # (B, L, vocab)
            # Standard causal LM shift: predict token[t+1] from tokens[:t+1].
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]  # CRITICAL: real-token mask, never ones_like

            log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
            token_nll = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            masked_nll = token_nll * shift_mask
            n_tokens = shift_mask.sum().clamp_min(1.0)
            loss = masked_nll.sum() / n_tokens

        (loss / accum_steps).backward()
        micro_step += 1

        if micro_step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            log_loss_sum += loss.item()
            log_steps += 1

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
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "total_wall_time_seconds": time.time() - t0,
            "n_params": n_params,
        }, f, indent=2)
    print("Training complete.")


if __name__ == "__main__":
    main()
