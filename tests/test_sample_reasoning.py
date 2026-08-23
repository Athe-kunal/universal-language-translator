"""Tests for the pure row-mapping logic in data_gen.sample_reasoning."""

from data_gen.sample_reasoning import (
    map_natural_reasoning_row,
    map_openthoughts3_row,
    map_opencodereasoning_row,
)


def test_map_opencodereasoning_row():
    row = {
        "input": "Given an array, find the maximum subarray sum.",
        "output": "<think>\nLet's use Kadane's algorithm.\n</think>\n\n```python\nprint('ok')\n```\n",
        "solution": "print('ok')\n",
        "source": "codeforces",
        "dataset": "code_contests",
        "difficulty": "5",
        "license": "cc-by-4.0",
    }
    mapped = map_opencodereasoning_row(row)
    assert mapped is not None
    assert mapped.source == "opencodereasoning"
    assert mapped.question == row["input"]
    assert mapped.reference_answer == row["solution"]
    assert "<think>" not in mapped.cot_answer and "</think>" not in mapped.cot_answer
    assert "Kadane's algorithm" in mapped.cot_answer
    assert mapped.metadata == {
        "platform": "codeforces",
        "origin_dataset": "code_contests",
        "difficulty": "5",
        "license": "cc-by-4.0",
    }


def test_map_opencodereasoning_row_missing_output():
    assert map_opencodereasoning_row({"input": "q", "output": None}) is None
    assert map_opencodereasoning_row({"input": "q", "output": ""}) is None


def test_map_openthoughts3_row_missing_turns_returns_none():
    row = {"conversations": [{"from": "human", "value": "hi"}]}
    assert map_openthoughts3_row(row) is None


def test_map_natural_reasoning_row_no_responses_returns_none():
    row = {"question": "q", "reference_answer": "a", "responses": []}
    assert map_natural_reasoning_row(row) is None
