"""Unit tests for nucleus/temperature sampling logic in generate.py.

These test pure tensor logic only — no GPU, no MDLM checkpoint, no MDLM
repo import required. Run with:

    conda activate mdlm-ar   # or any env with torch installed
    python -m pytest src/baselines/mdlm_ar/tests/test_sampling.py -v

Checkpoint-dependent checks (architecture, strict loading, finite logits)
live in smoke_test_checkpoint.py and are NOT run here — they require the
2.6GB ar.ckpt and a CUDA device, and are marked "not run" in the
reproducibility report when those aren't available.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate import top_p_filter, parse_args  # noqa: E402


def test_top_p_filter_sums_to_one():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(8, 100), dim=-1)
    filtered = top_p_filter(probs, top_p=0.9)
    sums = filtered.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_top_p_filter_keeps_at_least_one_token():
    # Extremely peaked distribution: top_p smaller than the top token's own prob.
    probs = torch.zeros(1, 10)
    probs[0, 3] = 0.6
    probs[0, 4] = 0.3
    probs[0, 5] = 0.1
    filtered = top_p_filter(probs, top_p=0.01)
    assert (filtered > 0).sum().item() >= 1
    assert filtered[0, 3].item() > 0  # the top token must survive


def test_top_p_one_reproduces_full_distribution():
    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(4, 50), dim=-1)
    filtered = top_p_filter(probs, top_p=1.0)
    assert torch.allclose(filtered, probs, atol=1e-5)


def test_top_p_boundary_crossing_token_is_included():
    # Hand-constructed distribution: probs 0.5, 0.3, 0.15, 0.05 (descending already).
    # top_p=0.7: cumulative after token0=0.5 (<0.7), after token1=0.8 (>=0.7).
    # Token1 is the one whose inclusion CROSSES the 0.7 threshold -> must be kept.
    # Token2 (cumulative 0.95) and token3 (cumulative 1.0) must be dropped.
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    filtered = top_p_filter(probs, top_p=0.7)
    assert filtered[0, 0].item() > 0, "top token must survive"
    assert filtered[0, 1].item() > 0, "boundary-crossing token must survive"
    assert filtered[0, 2].item() == 0, "token beyond the crossing point must be dropped"
    assert filtered[0, 3].item() == 0, "token beyond the crossing point must be dropped"
    assert abs(filtered.sum().item() - 1.0) < 1e-6


def test_top_p_filter_batched_independent_rows():
    # Each row should be filtered independently of the others.
    probs = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.34, 0.33, 0.33],
    ])
    filtered = top_p_filter(probs, top_p=0.5)
    assert filtered[0, 0].item() > 0 and filtered[0, 1].item() == 0
    # Row 1 is near-uniform; top_p=0.5 should need 2 tokens to cross 0.5.
    assert (filtered[1] > 0).sum().item() >= 2


def test_temperature_rejects_nonpositive(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "generate.py", "--mdlm-repo", "x", "--checkpoint", "y",
        "--output-dir", "z", "--temperature", "0",
    ])
    with pytest.raises(ValueError):
        parse_args()
    monkeypatch.setattr(sys, "argv", [
        "generate.py", "--mdlm-repo", "x", "--checkpoint", "y",
        "--output-dir", "z", "--temperature", "-1.0",
    ])
    with pytest.raises(ValueError):
        parse_args()


def test_top_p_rejects_out_of_range(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "generate.py", "--mdlm-repo", "x", "--checkpoint", "y",
        "--output-dir", "z", "--top-p", "0",
    ])
    with pytest.raises(ValueError):
        parse_args()
    monkeypatch.setattr(sys, "argv", [
        "generate.py", "--mdlm-repo", "x", "--checkpoint", "y",
        "--output-dir", "z", "--top-p", "1.5",
    ])
    with pytest.raises(ValueError):
        parse_args()


def test_default_args_are_declared_not_verified(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "generate.py", "--mdlm-repo", "x", "--checkpoint", "y", "--output-dir", "z",
    ])
    args = parse_args()
    assert args.sampler == "nucleus"
    assert args.top_p == 0.9
    assert args.temperature == 1.0
    assert args.sequence_length == 1024
