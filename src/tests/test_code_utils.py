"""Unit tests for utils/code_utils.py (TinyGSM/GSM8K escape codec + execution
scoring). CPU-only, no model/GPU required. Run:

    cd src
    python -m pytest tests/test_code_utils.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.code_utils import (  # noqa: E402
    escape_code, unescape_code, run_program, is_correct, execution_accuracy,
)


SAMPLE = '''def simple_math_problem():
    x = 3
    y = 5
    return x + y
'''


def test_escape_unescape_roundtrip():
    assert unescape_code(escape_code(SAMPLE)) == SAMPLE


def test_escape_handles_braces_and_special_chars():
    code = 'def simple_math_problem():\n    d = {"a": 1}\n    return d["a"]\n'
    assert unescape_code(escape_code(code)) == code


def test_escape_rejects_literal_sentinel():
    with pytest.raises(ValueError):
        escape_code("x = '<extra_id_0>'")


def test_run_program_correct_value():
    # The runner subprocess does str(result) before JSON-encoding, so the
    # return value is always a string -- is_correct() handles the numeric
    # comparison via float() conversion, not run_program() itself.
    assert run_program(SAMPLE) == "8"


def test_run_program_handles_exception():
    assert run_program("def simple_math_problem():\n    return 1 / 0\n") is None


def test_run_program_handles_timeout():
    code = "def simple_math_problem():\n    while True:\n        pass\n"
    assert run_program(code, timeout=1.0) is None


def test_is_correct_numeric_tolerance():
    assert is_correct(8, "8")
    assert is_correct(8.00001, "8")
    assert not is_correct(7, "8")
    assert not is_correct(None, "8")


def test_execution_accuracy_aggregates_correctly():
    programs = [SAMPLE, SAMPLE, "def simple_math_problem():\n    return 1 / 0\n"]
    golds = ["8", "7", "0"]
    result = execution_accuracy(programs, golds, num_proc=2, timeout=5.0)
    assert result["n"] == 3
    assert result["exec_accuracy"] == pytest.approx(1 / 3)
    assert result["exec_rate"] == pytest.approx(2 / 3)


def test_execution_accuracy_empty_input():
    result = execution_accuracy([], [], num_proc=2, timeout=5.0)
    assert result["n"] == 0
    assert result["exec_accuracy"] == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
