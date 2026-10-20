"""The extraction flow as a LangGraph graph: locate -> extract -> normalize -> check -> analyze.

One filing:  uv run python extraction/graph.py --cik 1667313 --year 2018
Diagram:     uv run python extraction/graph.py --mermaid
(The full evaluation over the sample is extraction/evaluate.py.)

Each node is a plain function: it reads the state (a typed dict) and returns only
the keys it changes. LangGraph merges that into the state and follows the edges.

    locate     code  pick the ~8,000 characters most likely to hold the statements
    extract    LLM   fill a JSON tool: each value as written, its scale, a verbatim quote
    normalize  code  "(1,234)" + "thousands" -> -1,234,000
    check      code  quote is really in the text, numbers plausible, year consistent
    analyze    code + LLM  ratios, sector outlier flags, a 2-3 sentence note, grounding check

After `check`, a conditional edge decides: failed and not yet retried -> back to
`locate` (next-best windows, wider); failed again -> `needs_human_review`;
passed -> `analyze`. Every path ends in `done` or `needs_human_review`, with a reason.
"""

import argparse
import json
import operator
import re
from typing import Annotated, TypedDict

import llm
from analysis import (
    NOTE_PROMPT,
    compute_ratios,
    flag_outliers,
    note_facts,
    ungrounded_numbers,
)
from checks import (
    apply_scale,
    evidence_found,
    number_in_quote,
    plausibility_problems,
    year_problem,
)
from common import FIELDS, SECTIONS
from langgraph.graph import END, START, StateGraph

MAX_RETRIES = 1
CHUNK = 2_000             # locate scores the text in pieces of this size
BUDGET = 8_000            # characters sent to the LLM on the first try
RETRY_BUDGET = 12_000     # wider on the retry

# What makes a piece of text likely to hold our numbers, with a weight for each.
# Important: EDGAR-CORPUS stripped the HTML tables out of the filings. The statement
# headings survive, but the rows of numbers do not. So the numbers we can find are in
# sentences, mostly in Item 7 (MD&A): "Revenues increased 8% to $412.3 million."
# The sentence patterns therefore weigh the most; the headings and scale words help
# point at the right neighbourhood.
_MONEY = r"[^.]{0,80}\$\s?\(?[\d.,]+"   # "... $412.3" within the same sentence
KEYWORDS = {
    r"(total |net )?(revenues?|net sales)" + _MONEY: 5,
    r"net (income|loss|earnings)" + _MONEY: 5,
    r"total assets" + _MONEY: 5,
    r"(operating (income|loss)|income from operations)" + _MONEY: 2,
    r"cash and cash equivalents" + _MONEY: 1,
    r"\$\s?[\d.,]+ (million|billion)": 1,
    r"consolidated statements? of (operations|income|earnings)": 1,
    r"consolidated balance sheets?": 1,
    r"total revenues?|net sales|net (income|loss)|total assets": 1,
    r"in thousands|in millions": 2,
}


class FilingState(TypedDict, total=False):
    # inputs
    cik: int
    year: int
    name: str
    sector: str
    sections: dict[str, str]          # section name -> text
    prior_revenue: float | None       # from XBRL, for growth
    sector_ref: dict                  # sector medians from analysis.sector_reference()
    # locate
    retries: int
    used_chunks: list[str]
    window: str
    # extract / normalize
    extracted: dict                   # the model's raw tool input
    values: dict[str, float | None]   # dollars
    # check
    problems: list[str]
    # analyze
    ratios: dict
    flags: dict
    facts: dict
    note: str
    # ends
    status: str                       # "done" or "needs_human_review"
    reason: str
    # running totals: the reducer (operator.add) adds each node's value to the total
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]
    trace: Annotated[list[str], operator.add]   # which nodes ran, in order


# ---------- locate ----------

def score_chunk(text: str) -> int:
    """Keyword score; each keyword counts up to 3 times so one long list can't dominate."""
    low = text.lower()
    return sum(w * min(3, len(re.findall(p, low))) for p, w in KEYWORDS.items())


def ranked_chunks(sections: dict[str, str]) -> list[tuple[int, str, int]]:
    """Every CHUNK-sized piece of the sections, best first: (score, section, start)."""
    chunks = []
    for name in SECTIONS:
        text = sections.get(name, "")
        for start in range(0, len(text), CHUNK):
            chunks.append((score_chunk(text[start:start + CHUNK]), name, start))
    return sorted(chunks, key=lambda c: (-c[0], SECTIONS.index(c[1]), c[2]))


def locate(state: FilingState) -> dict:
    """Pick the best windows, up to 8,000 characters.

    On a retry: keep the best window, add the piece right after it, and fill the
    rest with next-best windows not shown yet, up to 12,000 characters.
    """
    retry = bool(state.get("problems"))
    retries = state.get("retries", 0) + retry
    used = set(state.get("used_chunks", []))
    budget = RETRY_BUDGET if retry else BUDGET
    ranked = ranked_chunks(state["sections"])
    picked = []
    if retry and ranked:
        # Keep the best piece, and widen it: a passage often runs past a chunk edge.
        picked += [(ranked[0][1], ranked[0][2]), (ranked[0][1], ranked[0][2] + CHUNK)]
    for score, name, start in ranked:
        if len(picked) * CHUNK >= budget:
            break
        if score > 0 and f"{name}:{start}" not in used and (name, start) not in picked:
            picked.append((name, start))
    picked.sort(key=lambda p: (SECTIONS.index(p[0]), p[1]))  # back in reading order
    parts = [f"[{n} chars {s}-{s + CHUNK}]\n{state['sections'][n][s:s + CHUNK]}"
             for n, s in picked if state["sections"][n][s:s + CHUNK]]
    return {"window": "\n\n".join(parts), "retries": retries,
            "used_chunks": sorted(used | {f"{n}:{s}" for n, s in picked}),
            "trace": ["locate"]}


# ---------- extract ----------

EXTRACT_PROMPT = """You read excerpts from a company's 10-K annual report and record its key
financial figures by calling the tool.

The excerpts come from a text version of the filing in which the tables were removed,
so most figures appear inside sentences, e.g. "Revenues increased 8% to $412.3 million
in fiscal 2018" (value "412.3", scale "millions").

Rules:
- Use the most recent fiscal year in the excerpt. Never use a prior-year figure.
- For each figure fill three keys, e.g. revenue_value, revenue_scale, revenue_evidence.
- value: copy the number exactly as written, e.g. "412.3" or "(1,234)". A loss is
  negative: for "net loss of $4.2 million" write "-4.2".
- scale: what the number is in. "$412.3 million" is "millions"; a figure under an
  "(in thousands)" heading is "thousands"; a full amount like "$12,345,678" is "units".
- evidence: copy a short verbatim snippet (under 150 characters) from the excerpt that
  contains the label and the number. Do not paraphrase.
- revenue: total revenues, or net sales. net_income: net income (loss) attributable to
  the company. total_assets: total assets at the fiscal year end.
- If a figure is not in the excerpt, use null for value and evidence. Do not guess."""

# A flat schema: three plain keys per field (revenue_value, revenue_scale,
# revenue_evidence). We first tried one nested object per field, and the model
# sometimes returned the object as a string of XML-like text. Flat keys are
# filled more reliably, and the check still validates every one.
_PER_FIELD = {
    "value": {"type": ["string", "null"], "description": "the number exactly as written"},
    "scale": {"type": "string", "enum": ["units", "thousands", "millions", "billions"]},
    "evidence": {"type": ["string", "null"], "description": "verbatim quote with the number"},
}
EXTRACT_TOOL = {
    "name": "record_financials",
    "description": "Record the key figures for the most recent fiscal year.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {
            "fiscal_year_end": {"type": "string", "description": "YYYY-MM-DD"},
            **{f"{f}_{part}": spec for f in FIELDS for part, spec in _PER_FIELD.items()},
        },
        "required": ["fiscal_year_end", *(f"{f}_{part}" for f in FIELDS for part in _PER_FIELD)],
    }},
}


def field(extracted: dict, f: str) -> dict:
    """The model's answer for one field as {"value", "scale", "evidence"}."""
    return {part: extracted.get(f"{f}_{part}") for part in _PER_FIELD}


def extract_call(window: str, feedback: list[str] | None = None) -> dict:
    """One structured-output call. Shared with the single-call baseline in evaluate.py."""
    text = f"10-K excerpt:\n\n{window}"
    if feedback:  # on a retry, tell the model what went wrong last time
        text += "\n\nA previous answer failed these checks; avoid them:\n- " + "\n- ".join(feedback)
    return llm.converse(EXTRACT_PROMPT, text, tool=EXTRACT_TOOL, max_tokens=1200)


def extract(state: FilingState) -> dict:
    try:
        r = extract_call(state["window"], state.get("problems"))
    except Exception as e:  # noqa: BLE001 (llm.py already retried): send to review
        return {"extracted": {}, "status": "needs_human_review",
                "reason": f"LLM call failed: {type(e).__name__}: {str(e)[:150]}",
                "trace": ["extract"]}
    return {"extracted": r["output"], "tokens_in": r["tokens_in"],
            "tokens_out": r["tokens_out"], "trace": ["extract"]}


# ---------- normalize ----------

def normalize_fields(extracted: dict) -> dict[str, float | None]:
    """Each field -> dollars. Anything malformed becomes None (the check will notice)."""
    values = {}
    for f in FIELDS:
        item = field(extracted, f)
        values[f] = apply_scale(item["value"], item["scale"])
    return values


def normalize(state: FilingState) -> dict:
    return {"values": normalize_fields(state.get("extracted") or {}), "trace": ["normalize"]}


# ---------- check ----------

def find_problems(extracted: dict, values: dict, window: str, year: int) -> list[str]:
    """All three checks. Empty list = passed."""
    problems = []
    for f in FIELDS:
        item = field(extracted, f)
        if values.get(f) is None:
            continue  # nothing claimed, nothing to verify
        if not evidence_found(item.get("evidence"), window):
            problems.append(f"{f}: evidence quote is not in the text")
        elif not number_in_quote(item.get("value"), item.get("evidence")):
            problems.append(f"{f}: the value {item.get('value')!r} is not in its quote")
    problems += plausibility_problems(values)
    if (p := year_problem(extracted.get("fiscal_year_end"), year)):
        problems.append(p)
    if all(values.get(f) is None for f in ("revenue", "net_income", "total_assets")):
        problems.append("no revenue, net income or total assets found")
    return problems


def check(state: FilingState) -> dict:
    if state.get("status"):  # extract already failed hard; nothing to check
        return {"trace": ["check"]}
    problems = find_problems(state["extracted"], state["values"], state["window"], state["year"])
    return {"problems": problems, "trace": ["check"]}


def route_after_check(state: FilingState) -> str:
    """The conditional edge: retry once, else send failures to a human."""
    if state.get("status"):
        return "needs_human_review"
    if not state["problems"]:
        return "analyze"
    return "locate" if state.get("retries", 0) < MAX_RETRIES else "needs_human_review"


# ---------- analyze ----------

def analyze(state: FilingState) -> dict:
    ratios = compute_ratios(state["values"], state.get("prior_revenue"))
    flags = flag_outliers(ratios, state.get("sector_ref") or {})
    facts = note_facts(state.get("name", ""), state.get("sector", ""), state["year"],
                       state["values"], flags)
    out = {"ratios": ratios, "flags": flags, "facts": facts, "trace": ["analyze"]}
    try:
        r = llm.converse(NOTE_PROMPT, json.dumps(facts, indent=1), max_tokens=300)
    except Exception as e:  # noqa: BLE001 (a missing note should not crash the run)
        return out | {"note": "", "reason": f"note call failed: {type(e).__name__}"}
    out |= {"note": r["output"].strip(), "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"]}
    if (bad := ungrounded_numbers(out["note"], facts)):
        out["reason"] = f"analyst note has numbers not in the computed set: {bad}"
    return out


def route_after_analyze(state: FilingState) -> str:
    return "needs_human_review" if state.get("reason") else "done"


# ---------- end states ----------

def done(state: FilingState) -> dict:
    return {"status": "done", "reason": "all checks passed", "trace": ["done"]}


def needs_human_review(state: FilingState) -> dict:
    reason = state.get("reason") or "checks failed after retry: " + "; ".join(state["problems"])
    return {"status": "needs_human_review", "reason": reason, "trace": ["needs_human_review"]}


def build_graph():
    g = StateGraph(FilingState)
    for node in (locate, extract, normalize, check, analyze, done, needs_human_review):
        g.add_node(node.__name__, node)
    g.add_edge(START, "locate")
    g.add_edge("locate", "extract")
    g.add_edge("extract", "normalize")
    g.add_edge("normalize", "check")
    g.add_conditional_edges("check", route_after_check,
                            ["analyze", "locate", "needs_human_review"])
    g.add_conditional_edges("analyze", route_after_analyze, ["done", "needs_human_review"])
    g.add_edge("done", END)
    g.add_edge("needs_human_review", END)
    return g.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mermaid", action="store_true", help="print the graph diagram")
    parser.add_argument("--cik", type=int)
    parser.add_argument("--year", type=int)
    args = parser.parse_args()
    graph = build_graph()
    if args.mermaid or not args.cik:
        print(graph.get_graph().draw_mermaid())
    else:
        from analysis import sector_reference
        from common import load_companies, load_texts

        row = load_companies().set_index(["cik", "year"]).loc[(args.cik, args.year)]
        texts = load_texts({(args.cik, args.year)})[(args.cik, args.year)]
        state = graph.invoke({"cik": args.cik, "year": args.year, "name": row["name"],
                              "sector": row["sector"], "sections": texts,
                              "sector_ref": sector_reference().get(row["sector"], {})})
        for k in ("trace", "status", "reason", "values", "ratios", "note",
                  "tokens_in", "tokens_out"):
            print(f"{k:11s} {state.get(k)}")
