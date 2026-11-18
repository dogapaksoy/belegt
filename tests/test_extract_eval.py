import json
import zlib

from belegt.data.synth import generate
from belegt.extract_eval import build_prompt, parse_output, score, value_correct
from belegt.noise import degrade
from belegt.types import ValueType


def test_parse_output_handles_fences_thinking_and_chatter():
    raw = (
        "<think>hmm</think>Sure! Here you go:\n```json\n"
        '[{"field": "frist", "value": "2026-03-15", "quote": "bis zum 15.03.2026"}]\n```'
    )
    assert parse_output(raw) == [
        {"field": "frist", "value": "2026-03-15", "quote": "bis zum 15.03.2026"}
    ]


def test_parse_output_salvages_objects_from_broken_json():
    raw = '[{"field": "betrag", "value": "12.50", "quote": "x"}, {"field": "frist", "val'
    assert parse_output(raw) == [{"field": "betrag", "value": "12.50", "quote": "x"}]


def test_value_correct_normalizes_formats():
    assert value_correct(ValueType.DATE, "15.03.2026", "2026-03-15")
    assert value_correct(ValueType.AMOUNT, "1.234,56 €", "1234.56")
    assert value_correct(ValueType.ID, "34 / 815 / 08150", "34/815/08150")
    assert not value_correct(ValueType.DATE, "2026-03-16", "2026-03-15")
    assert not value_correct(ValueType.AMOUNT, "", "1234.56")


def test_prompt_lists_requested_fields():
    prompt = build_prompt("Brief", [("frist", ValueType.DATE), ("betrag", ValueType.AMOUNT)])
    assert "- frist (date as YYYY-MM-DD)" in prompt and "- betrag" in prompt


def _fake_generations(records, noise, corrupt_field=None):
    gens = []
    for rec in records:
        doc = degrade(rec["document"], noise, seed=zlib.crc32(rec["id"].encode()))
        answer = []
        for c in rec["claims"]:
            if c["corruption"] is None and c["type"] != "text":
                value = c["value"]
                if c["field"] == corrupt_field and c["type"] == "date":
                    value = "2030-01-01"  # hallucinated date with a real quote
                answer.append({"field": c["field"], "value": value, "quote": c["quote"]})
        assert doc  # degradation is deterministic per id
        gens.append({"id": rec["id"], "seconds": 1.0, "output": json.dumps(answer)})
    return gens


def test_perfect_extraction_scores_all_correct_and_unflagged():
    records = generate(10)
    s = score("fake", _fake_generations(records, 0.0), noise=0.0, seed=0)
    assert s.fields > 0 and s.correct == s.fields and s.correct_flagged == 0


def test_hallucinated_values_are_counted_and_flagged():
    records = generate(10)
    s = score("fake", _fake_generations(records, 0.0, corrupt_field="frist"), noise=0.0, seed=0)
    assert s.wrong == 10 and s.wrong_flagged == 10 and s.unflagged_errors == 0
