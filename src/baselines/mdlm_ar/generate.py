#!/usr/bin/env python
"""Unconditional generation from MDLM's pretrained AR checkpoint (`ar.ckpt`),
with JPEG-DLM-style nucleus (top-p) sampling as an option alongside MDLM's
own upstream full-categorical Gumbel-max sampler.

Must be run inside the `mdlm-ar` conda environment (see
docs/AR_BASELINE_JPEG_DLM.md) with `--mdlm-repo` pointing at a pinned
checkout of https://github.com/kuleshov-group/mdlm — this imports MDLM's
own diffusion.py/dataloader.py rather than reimplementing the model.

No KV cache: mirrors MDLM's own `Diffusion._ar_sampler` exactly, which
recomputes the full growing prefix at every step. This is correctness-first
and intentionally not optimized for throughput (see docs for the tradeoff).

Usage:
    conda activate mdlm-ar
    python generate.py \
        --mdlm-repo /path/to/mdlm \
        --checkpoint /path/to/ar.ckpt \
        --output-dir outputs/baselines/mdlm_ar_owt1024 \
        --num-samples 64 --sequence-length 1024 --batch-size 8 \
        --sampler nucleus --top-p 0.9 --temperature 1.0 \
        --seed 0 --device cuda:0
"""
import argparse
import json
import os
import platform
import subprocess
import sys

import torch
import time


def git_commit(repo_dir):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdlm-repo", required=True,
                     help="Path to a checkout of github.com/kuleshov-group/mdlm")
    ap.add_argument("--checkpoint", required=True, help="Path to ar.ckpt")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-samples", type=int, default=1024)
    ap.add_argument("--sequence-length", type=int, default=1024,
                     help="Model positions INCLUDING the leading BOS token")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sampler", choices=["nucleus", "full"], default="nucleus",
                     help="'nucleus' = JPEG-DLM-style top-p; "
                          "'full' = MDLM upstream full-categorical Gumbel-max")
    ap.add_argument("--top-p", type=float, default=0.9,
                     help="Nucleus sampling threshold. This is OUR declared "
                          "experimental choice, NOT a verified JPEG-DLM setting "
                          "(the paper does not disclose their exact top-p).")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.temperature <= 0:
        raise ValueError(f"--temperature must be > 0, got {args.temperature}")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError(f"--top-p must be in (0, 1], got {args.top_p}")
    return args


def build_model(mdlm_repo, checkpoint_path, sequence_length, device):
    sys.path.insert(0, mdlm_repo)
    import hydra
    with hydra.initialize_config_dir(config_dir=f"{mdlm_repo}/configs", version_base=None):
        config = hydra.compose(
            config_name="config",
            overrides=[
                "model=small-ar",
                "parameterization=ar",
                "backbone=ar",
                f"model.length={sequence_length}",
                "data=openwebtext-split",
                f"eval.checkpoint_path={checkpoint_path}",
                "mode=sample_eval",
                "loader.eval_batch_size=1",
                "loader.batch_size=1",
            ],
        )
    import dataloader
    import diffusion
    tokenizer = dataloader.get_tokenizer(config)
    model = diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path, tokenizer=tokenizer, config=config, strict=True,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer, config


def top_p_filter(probs, top_p):
    """Nucleus filtering: keep the minimal prefix (by descending prob) whose
    cumulative probability reaches top_p; renormalize; always keep >=1 token.
    probs: (batch, vocab), already non-negative and summing to ~1 per row."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    # First index where cumulative prob >= top_p is the last one to KEEP.
    # Everything strictly after it is masked out.
    sorted_mask = cum_probs - sorted_probs > top_p  # token pushes cumsum past top_p AND wasn't needed to reach it
    sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
    # Always keep at least the top-1 token.
    sorted_probs[:, 0] = torch.where(
        sorted_probs[:, 0] == 0, probs.gather(-1, sorted_idx[:, :1]).squeeze(-1), sorted_probs[:, 0]
    )
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    filtered_probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return filtered_probs


@torch.no_grad()
def generate_batch(model, tokenizer, batch_size, sequence_length, sampler,
                    top_p, temperature, device, generator):
    num_pred_tokens = sequence_length - 1
    x = torch.full((batch_size, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
    for _ in range(num_pred_tokens):
        log_probs = model.forward(x, None)[:, -1, :]  # (batch, vocab) — already log_softmax'd
        if sampler == "full":
            # MDLM upstream default: full-categorical Gumbel-max, sampled online
            # (not precomputing a [batch, steps, vocab] noise tensor).
            noise = -torch.log(-torch.log(
                torch.rand(log_probs.shape, generator=generator, device=device).clamp_min(1e-20)
            ).clamp_min(1e-20))
            next_token = (log_probs / temperature + noise).argmax(dim=-1, keepdim=True)
        else:
            scaled_log_probs = log_probs / temperature
            probs = torch.softmax(scaled_log_probs, dim=-1)
            probs = top_p_filter(probs, top_p)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        x = torch.cat([x, next_token], dim=1)
    return x  # (batch, sequence_length), token ids including leading BOS


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    model, tokenizer, config = build_model(
        args.mdlm_repo, args.checkpoint, args.sequence_length, args.device,
    )

    out_dir = os.path.join(args.output_dir, f"top_p_{args.top_p}" if args.sampler == "nucleus" else "full",
                            f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    predictions_path = os.path.join(out_dir, f"predictions_seed{args.seed}.jsonl")
    tokens_path = os.path.join(out_dir, f"tokens_seed{args.seed}.jsonl")

    n_generated = 0
    internal_eos_count = 0
    t0 = time.time()
    with open(predictions_path, "w", encoding="utf-8") as f_pred, \
         open(tokens_path, "w", encoding="utf-8") as f_tok:
        while n_generated < args.num_samples:
            cur_bsz = min(args.batch_size, args.num_samples - n_generated)
            tokens = generate_batch(
                model, tokenizer, cur_bsz, args.sequence_length,
                args.sampler, args.top_p, args.temperature, args.device, generator,
            )
            tokens_cpu = tokens.cpu().tolist()
            for i, ids in enumerate(tokens_cpu):
                sample_id = n_generated + i
                # Exclude only the leading BOS when exporting text, per spec.
                body_ids = ids[1:]
                if tokenizer.eos_token_id in body_ids:
                    internal_eos_count += 1
                # Do NOT truncate at internal EOS by default (packed-data boundary,
                # not necessarily "stop generating" — see docs for discussion).
                text = tokenizer.decode(body_ids, skip_special_tokens=False)
                f_pred.write(json.dumps({
                    "prediction": text,
                    "sample_id": sample_id,
                    "seed": args.seed,
                    "sampler": args.sampler,
                    "top_p": args.top_p if args.sampler == "nucleus" else None,
                    "temperature": args.temperature,
                    "num_gpt2_tokens_raw": len(body_ids),
                }, ensure_ascii=False) + "\n")
                f_tok.write(json.dumps({"sample_id": sample_id, "token_ids": ids}) + "\n")
            n_generated += cur_bsz
            print(f"[seed {args.seed}] generated {n_generated}/{args.num_samples} "
                  f"({time.time() - t0:.1f}s elapsed)", flush=True)

    metadata = {
        "checkpoint_path": args.checkpoint,
        "mdlm_repo": args.mdlm_repo,
        "mdlm_commit": git_commit(args.mdlm_repo),
        "corrflow_commit": git_commit(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "sequence_length_including_bos": args.sequence_length,
        "num_predicted_tokens_per_sample": args.sequence_length - 1,
        "tokenizer": "gpt2 (BPE), bos_token_id == eos_token_id == 50256",
        "decode_policy": "exclude leading BOS only; internal EOS/<|endoftext|> "
                          "tokens are NOT used as an early-stop signal and are "
                          "NOT stripped from exported text by default — they may "
                          "represent packed-document boundaries, not a true end.",
        "sampler": args.sampler,
        "top_p": args.top_p if args.sampler == "nucleus" else None,
        "top_p_is_verified_jpeg_dlm_setting": False,
        "temperature": args.temperature,
        "kv_cache": False,
        "samples_with_internal_eos": internal_eos_count,
        "device": args.device,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "gpu_name": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None,
        "generation_wall_time_seconds": time.time() - t0,
        "predictions_path": predictions_path,
        "tokens_path": tokens_path,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Done. {n_generated} samples written to {predictions_path}")
    print(f"Samples containing an internal EOS token: {internal_eos_count}/{n_generated}")
    print(f"Metadata: {os.path.join(out_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
