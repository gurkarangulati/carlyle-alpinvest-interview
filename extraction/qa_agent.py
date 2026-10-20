"""Document Q&A agent: ask one 10-K a question, get a sourced answer or "not in the document".

One question:  uv run python extraction/qa_agent.py --cik 23217 --year 2018 --question "What was revenue growth?"
Diagram:       uv run python extraction/qa_agent.py --mermaid
(The evaluation is extraction/qa_eval.py.)

Unlike graph.py, where code fixes the order of steps, here the MODEL chooses: each
turn it picks a tool (search the filing, get the extracted financials, calculate),
reads the result, and decides what to do next, until it calls `final_answer`.
That is what makes it an agent. Code keeps it safe:

  - a cap of 6 tool calls; after that the model is forced to answer;
  - `final_answer` is a tool with a fixed schema (answer, citations, confidence,
    answerable), and the answer is checked in code (`verify`);
  - one revision if a check fails, then `needs_human_review` with the reason;
  - every tool call is recorded in `trace` with its arguments and a short result.

Graph:  agent -> (tools -> agent)* -> verify -> done
                                            \\-> revise -> agent (once)
                                            \\-> needs_human_review
"""

import argparse
import json
import operator
import re
from typing import Annotated, TypedDict

import llm
from common import load_texts
from langgraph.graph import END, START, StateGraph
from qa_tools import (
    QA_SECTIONS,
    calculate,
    filing_index,
    get_financials,
    search_filing,
    use_texts,
)

MAX_TOOL_CALLS = 6
MAX_REVISIONS = 1
CONFIDENCE = ["high", "medium", "low"]
NOT_FOUND = "Not in the document."

SYSTEM = f"""You answer an analyst's question about one company's 10-K annual report, using tools.

- search_filing finds passages ("chunks") of the filing. Each has a chunk_id. If the first
  search misses, try other words (e.g. "net sales" for revenue, "customers accounted for"
  for customer concentration).
- get_financials returns revenue, net income and other figures already extracted from this
  filing's text, in US dollars. They are extracted, not official, so prefer a cited passage.
- calculate does arithmetic. Use it for every computation (growth, ratios). Write plain
  numbers without commas or $ signs, in the form (new - old) / old * 100. Use only
  numbers you found with the other tools.
- The filing's tables were removed, so figures appear only inside sentences.
- When ready, call final_answer. Cite the chunk_ids that support the answer. Every number
  in the answer must appear in a cited chunk or come from calculate; copy numbers as
  written in the passage.
- If the filing does not contain the answer, set answerable to false and answer
  "{NOT_FOUND}" Do not guess.
- You have at most {MAX_TOOL_CALLS} tool calls before you must answer."""


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description, "inputSchema": {"json": {
        "type": "object", "properties": properties, "required": required}}}


TOOLS = [
    _tool("search_filing", "Search this filing. Returns the best-matching chunks with ids.",
          {"query": {"type": "string"},
           "k": {"type": "integer", "description": "how many chunks, 1-8 (default 5)"}},
          ["query"]),
    _tool("get_financials", "Revenue, net income, total assets etc. extracted from this "
          "filing's text (US dollars). Not official XBRL data.", {}, []),
    _tool("calculate", "Arithmetic on plain numbers: + - * / and brackets.",
          {"expression": {"type": "string"}}, ["expression"]),
    _tool("final_answer", "Give the final answer with citations.",
          {"answer": {"type": "string"},
           "citations": {"type": "array", "items": {"type": "string"},
                         "description": "chunk_ids that support the answer"},
           "confidence": {"type": "string", "enum": CONFIDENCE},
           "answerable": {"type": "boolean",
                          "description": "false if the filing does not contain the answer"}},
          ["answer", "citations", "confidence", "answerable"]),
]


class QAState(TypedDict, total=False):
    cik: int
    year: int
    question: str
    messages: list                     # the Bedrock conversation so far
    pending: list                      # tool calls the model asked for in its last turn
    tool_calls: int                    # tools run so far (final_answer not counted)
    retrieved: dict[str, str]          # chunk_id -> text, everything search returned
    calcs: list[tuple[str, float]]     # every calculate call: (expression, result)
    fin_values: list[float]            # values get_financials returned
    final: dict | None                 # the model's final_answer input
    problems: list[str]
    revisions: int
    status: str                        # "done" or "needs_human_review"
    reason: str
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]
    trace: Annotated[list[dict], operator.add]   # every tool call: tool, args, short result
    path: Annotated[list[str], operator.add]     # nodes visited, in order


# ---------- verify: plain-code checks on the final answer ----------

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
LIST_MARKER = re.compile(r"\(\d{1,2}\)|^\s*\d{1,2}[.)]\s", re.MULTILINE)   # "(3)" or "3. " in lists


def answer_numbers(answer: str) -> list[str]:
    """The numbers an answer states, leaving out list markers like "(3)"."""
    return NUMBER.findall(LIST_MARKER.sub(" ", answer))


def clean_final(final) -> dict:
    """Undo common format slips before checking: Bedrock does not enforce the schema,
    and arrays or booleans sometimes arrive as strings."""
    if not isinstance(final, dict):
        return {}
    out = dict(final)
    cites = out.get("citations")
    if isinstance(cites, str):
        try:
            cites = json.loads(cites)
        except ValueError:
            cites = [c.strip() for c in cites.split(",") if c.strip()]
    out["citations"] = cites
    if isinstance(out.get("answerable"), str):
        out["answerable"] = {"true": True, "false": False}.get(out["answerable"].lower())
    return out


def number_supported(token: str, allowed: list[float]) -> bool:
    """Is this number (as written in the answer) one of the allowed numbers, maybe rounded?

    "7.9" is supported by 7.94 (rounded to the answer's 1 decimal); "8.1" is not.
    """
    x = float(token.replace(",", ""))
    decimals = len(token.split(".")[1]) if "." in token else 0
    return any(abs(round(abs(y), decimals) - x) < 1e-9 for y in allowed)


def allowed_numbers(source_text: str, calc_results: list[float], fin_values: list[float]):
    """Every number the answer may use: in the cited text, a calculation, or the
    extracted financials (also in thousands, millions, billions)."""
    nums = [float(n.replace(",", "")) for n in NUMBER.findall(source_text)]
    nums += [v for c in calc_results for v in (c, c * 100)]   # 0.126 may be written 12.6%
    nums += [v / s for v in fin_values for s in (1, 1e3, 1e6, 1e9)]
    return nums


def bad_operands(expression: str, allowed: list[float]) -> list[str]:
    """Numbers in a calculation that came from nowhere. Without this check the model
    could calculate with invented inputs, and the result would count as supported.
    1, 100 and 1000 are allowed constants; "$7.94 billion" may be written 7940 or 7940000000.
    """
    scaled = [y * s for y in allowed for s in (1, 1e3, 1e6, 1e9)]
    return [n for n in NUMBER.findall(expression)
            if float(n.replace(",", "")) not in (1, 100, 1000) and not number_supported(n, scaled)]


def verify_answer(final, question: str, retrieved: dict[str, str], known_ids: set[str],
                  calcs: list[tuple[str, float]], fin_values: list[float]) -> list[str]:
    """All checks on a final answer. Empty list = passed."""
    final = clean_final(final)
    if not final:
        return ["no final answer was given"]
    problems = []
    # A calculation counts as a source only if its inputs were found in the filing.
    seen = allowed_numbers(" ".join(retrieved.values()) + " " + question, [], fin_values)
    calc_results = []
    for expr, value in calcs:
        if (bad := bad_operands(expr, seen)):
            problems.append(f"calculation {expr!r} uses numbers not found in the filing: {bad}")
        else:
            calc_results.append(value)
            seen.append(value)  # a later calculation may build on this one
    answer, cites = final.get("answer"), final.get("citations")
    if not isinstance(answer, str) or not answer.strip():
        problems.append("the answer is empty")
        answer = ""
    if not isinstance(final.get("answerable"), bool):
        problems.append("answerable must be true or false")
    if final.get("confidence") not in CONFIDENCE:
        problems.append(f"confidence must be one of {CONFIDENCE}")
    if not isinstance(cites, list) or not all(isinstance(c, str) for c in cites):
        problems.append("citations must be a list of chunk ids")
        cites = []
    for c in cites:
        if c not in known_ids:
            problems.append(f"citation {c} does not exist in this filing")
        elif c not in retrieved:
            problems.append(f"citation {c} was not retrieved in this run")
    if final.get("answerable") is True and not cites:
        problems.append("an answer needs at least one citation")
    cited = " ".join(retrieved[c] for c in cites if c in retrieved)
    allowed = allowed_numbers(cited + " " + question, calc_results, fin_values)
    if (bad := [n for n in answer_numbers(answer) if not number_supported(n, allowed)]):
        problems.append(f"numbers not in a cited chunk or a calculation: {bad}")
    return problems


# ---------- nodes ----------

def agent(state: QAState) -> dict:
    """One LLM turn. The model picks a tool; past the cap it must pick final_answer."""
    forced = state.get("tool_calls", 0) >= MAX_TOOL_CALLS
    choice = {"tool": {"name": "final_answer"}} if forced else {"any": {}}
    try:
        r = llm.converse_turn(SYSTEM, state["messages"], TOOLS, choice, max_tokens=1200)
    except Exception as e:  # noqa: BLE001 (llm.py already retried): send to review
        return {"status": "needs_human_review", "pending": [], "final": None,
                "reason": f"LLM call failed: {type(e).__name__}: {str(e)[:120]}",
                "path": ["agent"]}
    msg = r["message"]
    uses = [c["toolUse"] for c in msg.get("content", []) if "toolUse" in c]
    final = next((u["input"] for u in uses if u["name"] == "final_answer"), None)
    return {"messages": state["messages"] + [msg], "pending": uses, "final": final,
            "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"], "path": ["agent"]}


def run_tool(state: QAState, name: str, args: dict) -> tuple[str, str, dict]:
    """Run one tool. Returns (result text for the model, short text for the trace,
    state updates). Raises ValueError for bad input; the caller reports it to the model."""
    cik, year = state["cik"], state["year"]
    if name == "search_filing":
        hits = search_filing(cik, year, str(args.get("query", "")), args.get("k", 5) or 5)
        text = "\n\n".join(f"[{h['chunk_id']}]\n{h['text']}" for h in hits) or "no matches"
        retrieved = state.get("retrieved", {}) | {h["chunk_id"]: h["text"] for h in hits}
        return text, f"{len(hits)} chunks: {[h['chunk_id'] for h in hits]}", \
            {"retrieved": retrieved}
    if name == "get_financials":
        fin = get_financials(cik, year)
        values = list(fin.get("values", {}).values())
        return json.dumps(fin), f"status {fin.get('status')}, values {fin.get('values')}", \
            {"fin_values": state.get("fin_values", []) + values}
    if name == "calculate":
        expr = str(args.get("expression", ""))
        value = calculate(expr)
        return str(value), f"{expr} = {value:.6g}", \
            {"calcs": state.get("calcs", []) + [(expr, value)]}
    raise ValueError(f"unknown tool {name}")


def tools(state: QAState) -> dict:
    """Run the tools the model asked for and send the results back to it."""
    updates, results, trace = {}, [], []
    count = state.get("tool_calls", 0)
    for use in state["pending"]:
        name, args = use["name"], use.get("input") or {}
        ok = count < MAX_TOOL_CALLS
        if not ok:
            text = short = "tool budget used up: call final_answer now"
        else:
            count += 1
            try:
                text, short, upd = run_tool(state | updates, name, args)
                updates |= upd
            except (ValueError, TypeError) as e:
                ok, text, short = False, f"error: {e}", f"error: {e}"
        results.append({"toolResult": {"toolUseId": use["toolUseId"],
                                       "content": [{"text": text}],
                                       "status": "success" if ok else "error"}})
        trace.append({"tool": name, "args": args, "result": short[:300]})
    return updates | {"messages": state["messages"] + [{"role": "user", "content": results}],
                      "pending": [], "tool_calls": count, "trace": trace, "path": ["tools"]}


def verify(state: QAState) -> dict:
    known = set(filing_index(state["cik"], state["year"])[0])
    problems = verify_answer(state.get("final"), state["question"], state.get("retrieved", {}),
                             known, state.get("calcs", []), state.get("fin_values", []))
    return {"problems": problems, "path": ["verify"]}


def revise(state: QAState) -> dict:
    """Send the failed checks back to the model as the result of its final_answer call."""
    feedback = ("Rejected by the checks:\n- " + "\n- ".join(state["problems"]) +
                "\nFix these and call final_answer again. You may search again if needed.")
    pending = state.get("pending") or []
    if pending:
        content = [{"toolResult": {"toolUseId": u["toolUseId"], "status": "error",
                                   "content": [{"text": feedback if u["name"] == "final_answer"
                                                else "not run"}]}} for u in pending]
    else:  # the model answered in plain text instead of calling a tool
        content = [{"text": feedback}]
    return {"messages": state["messages"] + [{"role": "user", "content": content}],
            "pending": [], "final": None, "revisions": state.get("revisions", 0) + 1,
            "trace": [{"tool": "verify", "args": {}, "result": "; ".join(state["problems"])[:300]}],
            "path": ["revise"]}


def done(state: QAState) -> dict:
    return {"status": "done", "reason": "checks passed", "path": ["done"]}


def needs_human_review(state: QAState) -> dict:
    reason = state.get("reason") or "checks failed after revision: " + "; ".join(state["problems"])
    return {"status": "needs_human_review", "reason": reason, "path": ["needs_human_review"]}


# ---------- edges ----------

def route_after_agent(state: QAState) -> str:
    if state.get("status"):
        return "needs_human_review"
    if state.get("final") is not None:
        return "verify"
    return "tools" if state.get("pending") else "verify"  # plain text: verify will reject it


def route_after_verify(state: QAState) -> str:
    if not state["problems"]:
        return "done"
    return "revise" if state.get("revisions", 0) < MAX_REVISIONS else "needs_human_review"


def build_qa_graph():
    g = StateGraph(QAState)
    for node in (agent, tools, verify, revise, done, needs_human_review):
        g.add_node(node.__name__, node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "verify", "needs_human_review"])
    g.add_edge("tools", "agent")
    g.add_conditional_edges("verify", route_after_verify, ["done", "revise", "needs_human_review"])
    g.add_edge("revise", "agent")
    g.add_edge("done", END)
    g.add_edge("needs_human_review", END)
    return g.compile()


_GRAPH = None


def answer_question(cik: int, year: int, question: str) -> dict:
    """Run the agent on one question. The filing's text must be registered (use_texts)."""
    global _GRAPH
    _GRAPH = _GRAPH or build_qa_graph()
    start = {"cik": cik, "year": year, "question": question, "tool_calls": 0, "revisions": 0,
             "retrieved": {}, "calcs": [], "fin_values": [],
             "messages": [{"role": "user", "content": [{"text":
                          f"Question: {question}\n(10-K of CIK {cik}, fiscal year {year}.)"}]}]}
    # Each tool round is two steps (agent, tools); 40 is far above the worst case (~18).
    return _GRAPH.invoke(start, {"recursion_limit": 40})


def print_result(state: dict) -> None:
    final = clean_final(state.get("final")) or {}
    print(f"\nanswer      {final.get('answer')}")
    print(f"answerable  {final.get('answerable')}   confidence {final.get('confidence')}")
    print(f"status      {state.get('status')}  ({state.get('reason')})")
    print("citations:")
    for c in final.get("citations") or []:
        snippet = (state.get("retrieved", {}).get(c) or "(not retrieved)")[:200]
        print(f"  [{c}] {' '.join(snippet.split())}")
    print("trace:")
    for i, t in enumerate(state.get("trace", []), 1):
        print(f"  {i}. {t['tool']}({json.dumps(t['args'])}) -> {t['result'][:150]}")
    print(f"path        {' > '.join(state.get('path', []))}")
    print(f"tokens      in {state.get('tokens_in')}  out {state.get('tokens_out')}  "
          f"cost ${llm.cost(state.get('tokens_in', 0), state.get('tokens_out', 0)):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mermaid", action="store_true", help="print the graph diagram")
    parser.add_argument("--cik", type=int)
    parser.add_argument("--year", type=int)
    parser.add_argument("--question")
    args = parser.parse_args()
    if args.mermaid or not args.question:
        print(build_qa_graph().get_graph().draw_mermaid())
    else:
        use_texts(load_texts({(args.cik, args.year)}, list(QA_SECTIONS)))
        print_result(answer_question(args.cik, args.year, args.question))
