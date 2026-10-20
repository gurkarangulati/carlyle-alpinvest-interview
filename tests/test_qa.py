"""Unit tests for the Q&A agent's code-only parts (no LLM, no network).

Run: uv run pytest tests/test_qa.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extraction"))

from qa_agent import (
    MAX_TOOL_CALLS,
    agent,
    bad_operands,
    clean_final,
    number_supported,
    route_after_agent,
    route_after_verify,
    verify_answer,
)
from qa_scoring import dollar_amounts, grade, in_text, predicted_value
from qa_tools import calculate, make_chunks, search_filing, use_texts

# ---------- calculate: safe arithmetic ----------


@pytest.mark.parametrize(("expr", "expected"), [
    ("1 + 2 * 3", 7), ("(7940 - 7830) / 7830 * 100", (7940 - 7830) / 7830 * 100),
    ("-5 + 2", -3), ("2.5 * 4", 10),
])
def test_calculate(expr, expected):
    assert calculate(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')", "open('x')", "x + 1", "2 ** 10", "1,234 + 1",
    "[1, 2]", "1 / 0", "1 +", "9" * 300,
])
def test_calculate_rejects_anything_else(expr):
    with pytest.raises(ValueError):
        calculate(expr)


# ---------- chunks and search ----------

FILING = {
    "section_1": "We make widgets. We had 1,200 employees as of December 31, 2018.",
    "section_1A": "Our largest customer, Acme Corp, accounted for 35% of net sales.",
    "section_7": "x " * 800 + "Net sales were $412.3 million in fiscal 2018, up from $380.1 million.",
    "section_8": "", "section_15": "",
}


def test_chunk_ids_say_where_text_came_from():
    chunks = make_chunks(FILING)
    assert list(chunks)[:3] == ["item1-000", "item1a-000", "item7-000"]
    assert "item7-001" in chunks  # Item 7 is longer than one chunk


def test_search_finds_the_right_chunk():
    use_texts({(1, 2018): FILING})
    hits = search_filing(1, 2018, "largest customer", k=2)
    assert hits[0]["chunk_id"] == "item1a-000"
    assert search_filing(1, 2018, "net sales fiscal 2018", k=1)[0]["chunk_id"] == "item7-001"
    assert search_filing(1, 2018, "", k=3) == []


# ---------- verify ----------

CHUNKS = {"item7-001": "Net sales were $412.3 million in fiscal 2018, up from $380.1 million.",
          "item1-000": "We had 1,200 employees."}
KNOWN = set(CHUNKS) | {"item7-002"}


def final(answer, cites, answerable=True, confidence="high"):
    return {"answer": answer, "citations": cites, "answerable": answerable,
            "confidence": confidence}


def check(f, calcs=(), fin=()):
    return verify_answer(f, "What was revenue?", CHUNKS, KNOWN, list(calcs), list(fin))


def test_grounded_answer_passes():
    assert check(final("Net sales were $412.3 million in fiscal 2018.", ["item7-001"])) == []


def test_rounded_number_is_supported():
    assert number_supported("412", [412.3])
    assert not number_supported("413", [412.3])


def test_citation_must_exist_and_be_retrieved():
    assert any("does not exist" in p for p in check(final("x", ["item9-999"])))
    assert any("not retrieved" in p for p in check(final("x", ["item7-002"])))


def test_number_not_in_cited_chunk_fails():
    problems = check(final("Net sales were $415.0 million.", ["item7-001"]))
    assert any("numbers not" in p for p in problems)


def test_list_markers_are_not_numbers():
    answer = "Main points: (1) net sales were $412.3 million; (2) growth.\n3. more"
    assert check(final(answer, ["item7-001"])) == []


def test_number_from_uncited_chunk_fails():
    # 1,200 is retrieved, but only item7-001 is cited.
    assert any("numbers not" in p for p in check(final("1,200 employees.", ["item7-001"])))


def test_calculation_output_is_allowed_if_inputs_were_found():
    growth = (412.3 - 380.1) / 380.1 * 100
    f = final("Net sales grew 8.5%.", ["item7-001"])
    assert check(f, calcs=[("(412.3 - 380.1) / 380.1 * 100", growth)]) == []


def test_calculation_with_invented_inputs_fails():
    f = final("Net sales grew 1.4%.", ["item7-001"])
    problems = check(f, calcs=[("(7940 - 7830) / 7830 * 100", 1.4049)])
    assert any("calculation" in p for p in problems)
    assert bad_operands("(412.3 - 380.1) / 380.1 * 100", [412.3, 380.1]) == []
    assert bad_operands("412300000 - 1", [412.3]) == []   # $412.3 million written in full


def test_abstaining_needs_no_citation():
    assert check(final("Not in the document.", [], answerable=False)) == []


def test_answer_needs_a_citation_and_valid_fields():
    problems = check(final("Revenue was large.", [], confidence="sure"))
    assert any("at least one citation" in p for p in problems)
    assert any("confidence" in p for p in problems)
    assert check(None) == ["no final answer was given"]


def test_clean_final_fixes_string_fields():
    f = clean_final({"answer": "a", "citations": '["item7-001"]', "answerable": "true"})
    assert f["citations"] == ["item7-001"] and f["answerable"] is True


# ---------- routing and the tool cap ----------

def test_routes():
    assert route_after_agent({"final": {"answer": "x"}}) == "verify"
    assert route_after_agent({"final": None, "pending": [{"name": "calculate"}]}) == "tools"
    assert route_after_agent({"status": "needs_human_review"}) == "needs_human_review"
    assert route_after_verify({"problems": []}) == "done"
    assert route_after_verify({"problems": ["x"], "revisions": 0}) == "revise"
    assert route_after_verify({"problems": ["x"], "revisions": 1}) == "needs_human_review"


def test_agent_is_forced_to_answer_at_the_cap(monkeypatch):
    seen = {}

    def fake_turn(system, messages, tools, tool_choice, max_tokens):
        seen["choice"] = tool_choice
        return {"message": {"role": "assistant", "content": []}, "tokens_in": 1, "tokens_out": 1}

    monkeypatch.setattr("llm.converse_turn", fake_turn)
    agent({"messages": [], "tool_calls": MAX_TOOL_CALLS})
    assert seen["choice"] == {"tool": {"name": "final_answer"}}
    agent({"messages": [], "tool_calls": 0})
    assert seen["choice"] == {"any": {}}


# ---------- grading, abstention included ----------

def test_reading_numbers_from_answers():
    assert dollar_amounts("Revenue was $7.94 billion, up from $7,830 million.") == \
        [7.94e9, 7.83e9]
    assert predicted_value("revenue", "In fiscal 2018 revenue was $412.3 million.") == 412.3e6
    assert predicted_value("growth", "Revenue decreased 5.2% versus 2017.") == -5.2
    assert predicted_value("growth", "Revenue changed by -5.2%.") == -5.2
    assert in_text([412.3e6], 412_345_678) and not in_text([400e6], 412_345_678)


def test_grade_counts_correct_abstention():
    assert grade("revenue", None, 1e9, abstained=True, findable=False, tol=1) == (True, None)
    assert grade("revenue", None, 1e9, abstained=True, findable=True, tol=1) == (False, None)
    assert grade("revenue", 1.004e9, 1e9, abstained=False, findable=True, tol=1)[0]
    assert not grade("growth", 12.0, 13.5, abstained=False, findable=True, tol=1)[0]
    assert grade("growth", 12.0, 13.5, abstained=False, findable=True, tol=5)[0]
