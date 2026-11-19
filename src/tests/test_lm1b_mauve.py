"""Unit/integration tests for eval_lm1b_mauve.py and the compute_mauve changes.

CPU-only, no GPU or real GPT-2/mauve-text model required for most tests
(mocked feature extraction where real MAUVE would be too expensive). Run:

    cd src
    python -m pytest tests/test_lm1b_mauve.py -v
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_lm1b_mauve import load_predictions, length_stats  # noqa: E402
from utils.metrics_utils import compute_mauve  # noqa: E402


def _write_jsonl(tmp_path, rows):
    path = os.path.join(tmp_path, "gen.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


class _FakeTokenizer:
    """Deterministic whitespace tokenizer standing in for GPT-2's real one."""
    def encode(self, text):
        return text.split()


# ---------------------------------------------------------------- key detection
def test_load_predictions_generated_key(tmp_path):
    path = _write_jsonl(tmp_path, [{"generated": "hello world"}, {"generated": "foo"}])
    texts, blanks = load_predictions(str(path))
    assert texts == ["hello world", "foo"]
    assert blanks == 0


def test_load_predictions_prediction_key(tmp_path):
    path = _write_jsonl(tmp_path, [{"prediction": "hello"}, {"prediction": ""}])
    texts, blanks = load_predictions(str(path))
    assert texts == ["hello", ""]
    assert blanks == 1


def test_load_predictions_explicit_text_field(tmp_path):
    path = _write_jsonl(tmp_path, [{"custom_key": "hi"}])
    texts, blanks = load_predictions(str(path), text_field="custom_key")
    assert texts == ["hi"]


def test_load_predictions_unknown_key_raises(tmp_path):
    path = _write_jsonl(tmp_path, [{"foo": "bar"}])
    with pytest.raises(KeyError):
        load_predictions(str(path))


def test_load_predictions_explicit_field_missing_raises(tmp_path):
    path = _write_jsonl(tmp_path, [{"generated": "hi"}])
    with pytest.raises(KeyError):
        load_predictions(str(path), text_field="does_not_exist")


# ---------------------------------------------------------------- length stats
def test_length_stats_counts_blank_and_duplicates():
    tok = _FakeTokenizer()
    texts = ["a b c", "", "a b c", "   ", "d e f g h i j k"]
    stats, nonempty = length_stats("test", texts, tok, thresholds=(3, 8))
    assert stats["total"] == 5
    assert stats["blank"] == 2
    assert stats["nonempty"] == 3
    assert stats["duplicates"] == 1  # "a b c" appears twice
    assert nonempty == ["a b c", "a b c", "d e f g h i j k"]


def test_length_stats_threshold_fractions():
    tok = _FakeTokenizer()
    texts = ["a", "a b", "a b c d"]  # lengths 1, 2, 4
    stats, _ = length_stats("test", texts, tok, thresholds=(2, 3))
    assert stats["frac_lt_2"] == pytest.approx(1 / 3)
    assert stats["n_eligible_ge_2"] == 2
    assert stats["n_eligible_ge_3"] == 1


# ---------------------------------------------------------------- compute_mauve wrapper
def test_compute_mauve_equalize_sample_counts_default_matches_old_behavior(monkeypatch):
    """Default (equalize_sample_counts=True) must still silently truncate to
    min(len(p), len(q)) -- this is the exact behavior every existing OWT call
    site already depends on; changing the default would be a regression."""
    calls = {}

    class FakeOut:
        mauve = 0.5

    def fake_compute_mauve(p_text, q_text, **kwargs):
        calls["p_text"] = p_text
        calls["q_text"] = q_text
        return FakeOut()

    import types
    fake_mauve_module = types.SimpleNamespace(compute_mauve=fake_compute_mauve)
    monkeypatch.setitem(sys.modules, "mauve", fake_mauve_module)

    gen = ["g1", "g2", "g3"]
    ref = ["r1", "r2"]
    result = compute_mauve(generated_texts=gen, reference_texts=ref, device_id=-1)
    assert len(calls["p_text"]) == 2  # truncated to min(3, 2)
    assert len(calls["q_text"]) == 2
    assert result["mauve"] == 0.5
    assert result["num_generated"] == 2
    assert result["num_reference"] == 2


def test_compute_mauve_independent_counts_when_disabled(monkeypatch):
    class FakeOut:
        mauve = 0.7

    def fake_compute_mauve(p_text, q_text, **kwargs):
        return FakeOut()

    import types
    monkeypatch.setitem(sys.modules, "mauve", types.SimpleNamespace(compute_mauve=fake_compute_mauve))

    gen = ["g1", "g2", "g3", "g4"]
    ref = ["r1", "r2"]
    result = compute_mauve(
        generated_texts=gen, reference_texts=ref, device_id=-1, equalize_sample_counts=False,
    )
    assert result["num_generated"] == 4
    assert result["num_reference"] == 2


def test_compute_mauve_nan_on_insufficient_samples():
    result = compute_mauve(generated_texts=["only one"], reference_texts=["a", "b"], device_id=-1)
    assert np.isnan(result["mauve"])


# ---------------------------------------------------------------- reference sampling reproducibility
def test_random_reference_sampling_is_reproducible_and_disjoint():
    rng1 = np.random.RandomState(42)
    idx1 = rng1.choice(10000, size=100, replace=False)
    rng2 = np.random.RandomState(42)
    idx2 = rng2.choice(10000, size=100, replace=False)
    assert np.array_equal(idx1, idx2)

    rng3 = np.random.RandomState(42)
    idx_a = rng3.choice(10000, size=200, replace=False)
    idx_a1, idx_a2 = idx_a[:100], idx_a[100:]
    assert set(idx_a1.tolist()).isdisjoint(set(idx_a2.tolist()))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
