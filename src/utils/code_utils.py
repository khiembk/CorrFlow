"""Python-as-text utilities for TinyGSM / GSM8K (used when `config.code_target` is set).

Codec. The T5 sentencepiece vocabulary cannot represent Python: it drops every
whitespace run (newlines, indentation) and maps ``{ } < \\ ^ ~ ` `` to ``<unk>``.
`escape_code` rewrites source into text that survives the tokenizer using the
``<extra_id_*>`` sentinels, and `unescape_code` / `decode_sample` invert it
exactly, so generated samples decode back to runnable Python. T5 decoding emits
one space after every sentinel and drops any space before one, so `escape_code`
never leaves a meaningful raw space next to a sentinel (such spaces become
`_SPACE`) and `unescape_code` can strip whitespace around sentinels unconditionally.

Execution. On GSM8K a solution counts only if the program runs and
`simple_math_problem()` returns the gold answer; BLEU/ROUGE against a bare
numeric answer carries no signal. Each program runs in its own fresh interpreter
under CPU/memory/wall-clock limits, since model output is arbitrary code. Threads
only wait on those subprocesses, so the (JAX) parent process is never forked.
"""

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np

# ============================================
# Codec: Python source <-> T5-tokenizable text
# ============================================
# Sentinel assignment. Only ever append to this table -- changing an existing
# entry invalidates every dataset tokenized with an older version.
_NEWLINE = "<extra_id_0>"
_INDENT = "<extra_id_1>"  # one 4-space indentation unit
_SPACE = "<extra_id_2>"   # a space the tokenizer would not preserve

_CHAR_SENTINELS: Dict[str, str] = {
    "{": "<extra_id_3>",
    "}": "<extra_id_4>",
    "<": "<extra_id_5>",
    "\\": "<extra_id_6>",
    "^": "<extra_id_7>",
    "~": "<extra_id_8>",
    "`": "<extra_id_9>",
    "\t": "<extra_id_10>",
    "\r": "<extra_id_11>",
}
_SENTINEL_CHARS = {v: k for k, v in _CHAR_SENTINELS.items()}

INDENT_UNIT = "    "  # 4 spaces == one _INDENT sentinel

_ANY_SENTINEL_RE = re.compile(r"<extra_id_\d+>")
# Whitespace the decoder inserted around sentinels; safe to drop because
# `escape_code` never leaves a meaningful raw space next to one.
_DECODE_WS_RE = re.compile(r"\s*(<extra_id_\d+>)\s*")


def _escape_line(line: str) -> str:
    """Escape one line (no newlines inside): indentation, then per-char."""
    stripped = line.lstrip(" ")
    n_lead = len(line) - len(stripped)
    units, remainder = divmod(n_lead, len(INDENT_UNIT))
    prefix = _INDENT * units + _SPACE * remainder

    # Classify every character first, so the decision for a space can look at
    # what its neighbours became without any replacement corrupting another
    # (e.g. escaping '<' must not rewrite the '<' of an inserted sentinel).
    pieces: List[str] = [_CHAR_SENTINELS.get(ch, ch) for ch in stripped]
    is_sentinel = [ch in _CHAR_SENTINELS for ch in stripped]

    out: List[str] = []
    for i, ch in enumerate(stripped):
        if ch != " ":
            out.append(pieces[i])
            continue
        # A raw space survives only between two raw, non-space characters --
        # anywhere else the tokenizer would move or swallow it.
        prev_raw = i > 0 and stripped[i - 1] != " " and not is_sentinel[i - 1]
        next_raw = i + 1 < len(stripped) and stripped[i + 1] != " " and not is_sentinel[i + 1]
        out.append(" " if (prev_raw and next_raw) else _SPACE)
    return prefix + "".join(out)


def escape_code(code: str) -> str:
    """Rewrite `code` so a T5 tokenizer round-trips it losslessly."""
    # A literal sentinel in the source would break the inverse. TinyGSM has
    # none, but fail loudly rather than corrupt silently.
    if _ANY_SENTINEL_RE.search(code):
        raise ValueError("Source already contains an <extra_id_*> sentinel; cannot escape losslessly.")
    return _NEWLINE.join(_escape_line(line) for line in code.split("\n"))


def unescape_code(escaped: str) -> str:
    """Inverse of `escape_code`, tolerant of truncated model output."""
    escaped = _DECODE_WS_RE.sub(r"\1", escaped)
    text = escaped.replace(_NEWLINE, "\n").replace(_INDENT, INDENT_UNIT).replace(_SPACE, " ")
    for sentinel, ch in _SENTINEL_CHARS.items():
        text = text.replace(sentinel, ch)
    return text


def decode_sample(tokenizer, ids) -> str:
    """Decode generated ids back to Python source.

    `skip_special_tokens=True` would drop the ``<extra_id_*>`` sentinels along
    with the padding, taking every newline and brace with them, so the padding
    tokens are stripped by hand instead.
    """
    text = tokenizer.decode(ids, skip_special_tokens=False)
    for special in (tokenizer.pad_token, tokenizer.eos_token, tokenizer.unk_token):
        if special:
            text = text.replace(special, "")
    return unescape_code(text)


def roundtrip_ok(code: str, tokenizer) -> bool:
    """True if `code` survives escape -> tokenize -> decode -> unescape."""
    try:
        escaped = escape_code(code)
    except ValueError:
        return False
    ids = tokenizer(escaped, add_special_tokens=False)["input_ids"]
    return unescape_code(tokenizer.decode(ids)) == code


# ============================================
# Execution-based scoring
# ============================================
FUNC_NAME = "simple_math_problem"

# Executed in a fresh interpreter: read source on stdin, run it, print the value
# returned by FUNC_NAME behind a marker so we can ignore whatever else it prints.
_RUNNER = r"""
import sys, json, resource
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
ns = {}
exec(sys.stdin.read(), ns)
print("__RESULT__" + json.dumps(str(ns["%s"]())))
""" % FUNC_NAME


def run_program(code: str, timeout: float = 10.0):
    """Run one program; return the value it returned (as a string), or None."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER],
            input=code, capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            try:
                return json.loads(line[len("__RESULT__"):])
            except json.JSONDecodeError:
                return None
    return None


def is_correct(pred, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except (TypeError, ValueError):
        return str(pred).strip() == str(gold).strip()


def bootstrap_ci(correct, n_resamples: int = 1000, seed: int = 0):
    """Mean and 95% percentile bootstrap CI over problems."""
    arr = np.asarray(correct, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    return float(draws.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def execution_accuracy(programs, golds, num_proc: int = 16, timeout: float = 10.0) -> dict:
    """Execute each program and compare its return value with the gold answer."""
    with ThreadPoolExecutor(max_workers=max(1, num_proc)) as pool:
        values = list(pool.map(lambda code: run_program(code, timeout), programs))
    correct = [1.0 if is_correct(v, g) else 0.0 for v, g in zip(values, golds)]
    n = max(len(correct), 1)
    _, lo, hi = bootstrap_ci(correct)
    return {
        "exec_accuracy": sum(correct) / n,
        "exec_rate": sum(v is not None for v in values) / n,
        "exec_accuracy_ci_low": lo,
        "exec_accuracy_ci_high": hi,
        "n": len(correct),
    }
