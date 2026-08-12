"""T6, T7 — strict structured-output parsing."""
from analysis.parsing import parse_forced_choice_text, parse_json_number


def test_t6_valid_outputs():
    assert parse_json_number('{"score": 72}', "score", 0, 100) == (72.0, "ok")
    assert parse_json_number('noise {"score": 0.5} tail', "score", 0, 100)[1] == "ok"
    assert parse_forced_choice_text("A") == ("A", "ok")
    assert parse_forced_choice_text("B\n") == ("B", "ok")


def test_t7_malformed_flagged_not_coerced():
    for raw in ("", "score: 72", '{"score": "high"}', '{"other": 1}',
                '{"score": 150}', '{"score": true}'):
        value, status = parse_json_number(raw, "score", 0, 100)
        assert value is None
        assert status.startswith("parse_error")
    for raw in ("Maybe A", "The answer is B", "AB", ""):
        value, status = parse_forced_choice_text(raw)
        assert value is None
        assert status.startswith("parse_error")
