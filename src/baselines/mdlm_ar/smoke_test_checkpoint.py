#!/usr/bin/env python
"""Validation Step A/architecture check for the MDLM AR checkpoint.

Builds the exact Hydra config MDLM's own main.py would build for
`model=small-ar parameterization=ar backbone=ar model.length=1024`, loads
`ar.ckpt` via Lightning's `load_from_checkpoint` (strict by default — any
missing/unexpected keys raise), and checks:
  - architecture matches small-ar.yaml (12 blocks, hidden 768, 12 heads)
  - tokenizer is GPT-2 BPE, vocab size sane, bos==eos (GPT-2's native setup)
  - a forward pass on a tiny prefix produces finite logits that softmax to 1

Must be run with `--mdlm-repo` pointing at a checkout of
https://github.com/kuleshov-group/mdlm (pinned commit recorded in
metadata.json), since this imports MDLM's own `diffusion.py`/`dataloader.py`
rather than reimplementing them.

Usage:
    conda activate mdlm-ar
    python smoke_test_checkpoint.py \
        --mdlm-repo /home.na1/.../CorrFlow/third_party/mdlm \
        --checkpoint /home.na1/.../checkpoints/mdlm/ar.ckpt
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdlm-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, args.mdlm_repo)
    import hydra
    import torch

    with hydra.initialize_config_dir(config_dir=f"{args.mdlm_repo}/configs", version_base=None):
        config = hydra.compose(
            config_name="config",
            overrides=[
                "model=small-ar",
                "parameterization=ar",
                "backbone=ar",
                "model.length=1024",
                "data=openwebtext-split",
                f"eval.checkpoint_path={args.checkpoint}",
                "mode=sample_eval",
                "loader.eval_batch_size=2",
                "loader.batch_size=2",
            ],
        )

    print("=" * 70)
    print("Config: model=%s backbone=%s parameterization=%s length=%s"
          % (config.model.name, config.backbone, config.parameterization, config.model.length))
    print("Architecture check vs small-ar.yaml: n_blocks=%s hidden_size=%s n_heads=%s causal=%s"
          % (config.model.n_blocks, config.model.hidden_size, config.model.n_heads, config.model.causal))
    assert config.model.n_blocks == 12, config.model.n_blocks
    assert config.model.hidden_size == 768, config.model.hidden_size
    assert config.model.n_heads == 12, config.model.n_heads
    assert config.model.causal is True
    print("Architecture matches small-ar.yaml (12 blocks, hidden 768, 12 heads, causal). OK")
    print("=" * 70)

    import dataloader
    import diffusion

    tokenizer = dataloader.get_tokenizer(config)
    print(f"Tokenizer: {type(tokenizer).__name__}, vocab_size={tokenizer.vocab_size}, "
          f"bos_token_id={tokenizer.bos_token_id}, eos_token_id={tokenizer.eos_token_id}")
    assert tokenizer.bos_token_id == tokenizer.eos_token_id, (
        "Expected GPT-2's native BOS==EOS (<|endoftext|>, id 50256); "
        "mismatch means a different tokenizer config was loaded.")
    print("Confirmed: GPT-2 tokenizer, BOS==EOS (both <|endoftext|>) as expected.")
    print("=" * 70)

    print(f"Loading checkpoint strictly from {args.checkpoint} ...")
    model = diffusion.Diffusion.load_from_checkpoint(
        args.checkpoint, tokenizer=tokenizer, config=config, strict=True,
    )
    model = model.to(args.device)
    model.eval()
    print("Checkpoint loaded with STRICT=True — no missing/unexpected keys "
          "(load_from_checkpoint would have raised otherwise).")
    print(f"Backbone class: {type(model.backbone).__name__}")
    n_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"Backbone parameter count: {n_params:,}")
    print("=" * 70)

    with torch.no_grad():
        prefix = torch.full((2, 5), tokenizer.bos_token_id, dtype=torch.long, device=args.device)
        log_probs = model.forward(prefix, None)
        print(f"Forward pass output shape: {tuple(log_probs.shape)} "
              f"(expect [2, 5, vocab_size={model.vocab_size}])")
        assert log_probs.shape == (2, 5, model.vocab_size)
        assert torch.isfinite(log_probs).all(), "Non-finite logits in forward pass!"
        probs = log_probs.exp()
        row_sums = probs.sum(-1)
        print(f"Next-token prob row sums (should be ~1.0): "
              f"min={row_sums.min().item():.6f} max={row_sums.max().item():.6f}")
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
        mask_logit = log_probs[..., model.mask_index]
        print(f"Logit at mask_index ({model.mask_index}) — should be ~-inf/clamped: "
              f"{mask_logit.min().item():.1f} to {mask_logit.max().item():.1f}")

    print("=" * 70)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
