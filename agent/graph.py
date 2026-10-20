"""The analyst agent: answer a question from the data lake, with sources, or say it isn't there.

One question:  uv run python agent/graph.py --question "Which sector has the most holdings in fund X?"
Diagram:       uv run python agent/graph.py --mermaid
Local lake by default; LAKE_BACKEND=athena (plus ATHENA_* env) to query AWS.

It is the same design as extraction/qa_agent.py, widened from one 10-K to a whole lake:
the MODEL picks tools turn by turn (resolve a name, run SQL, search documents,
calculate) until it calls `final_answer`; CODE keeps it safe:

  - at most 6 tool calls, then final_answer is forced;
  - every SQL query passes the guard in tools.py (one SELECT, lake tables only, LIMIT 200);
  - the final answer is a tool call with a fixed schema, checked in code (`verify`):
      * every citation was produced in THIS run (a chunk it retrieved, or sql:<query_id>
        of a query it ran);
      * every number in the answer appears in a tool result or a checked calculation;
      * an entity answer is a company whose cik a tool returned in this run;
  - one revision if a check fails, then `needs_human_review` with the reason;
  - every step (LLM turn, tool call, verify) goes into `trace`.

Graph:  agent -> (tools -> agent)* -> verify -> done
                                           \\-> revise -> agent (once)
                                           \\-> needs_human_review
"""

import argparse
import json
import math
import operator
import re
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

if __name__ == "__main__":  # run as a script: make `agent` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as _pkg  # noqa: F401, E402  (puts extraction/ on sys.path)

import llm  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from qa_agent import (MAX_REVISIONS, MAX_TOOL_CALLS, NUMBER, answer_numbers,  # noqa: E402
                      bad_operands, clean_final, number_supported)

from agent import tools as T  # noqa: E402
from agent.schema import describe_tables  # noqa: E402

NOT_IN_LAKE = "This is not in the data lake."
RESULT_CHARS = 6000          # how much of one tool result the model sees

SYSTEM = f"""You are a financial data analyst. You answer questions about public companies
and private-equity funds using ONLY the data lake, through tools.

The lake (SQL tables; write Presto/Trino SQL, plain table names, no schema prefix):
{describe_tables()}

How to work:
- Company names: call resolve_company first and filter SQL by the cik it returns, never by
  name text (names are spelled many ways). If it abstains, say the company is ambiguous or
  not in the lake.
- run_sql runs ONE SELECT over the tables above (max 200 rows). Prefer the gold tables for
  financials and fund exposure. Money is in US dollars. Each result has a query_id.
- search_docs finds passages of 10-Ks and fund reports (filter by cik or fund_id). Use it
  for questions about text (risks, strategy, descriptions). Each passage has a chunk_id.
- calculate does arithmetic on plain numbers (no commas or $). Use it for every computation,
  only with numbers a tool returned.
- When ready, call final_answer:
  * answer: one or two plain sentences.
  * answer_value: the number (in the lake's units, e.g. US dollars, not millions; percents
    as the table stores them), the list for a "which ones" question (CIKs as integers when
    asked for CIKs, fund_ids like "F01" for funds), the fund_id when the answer is one
    fund, a short string, or null.
  * entity_cik: when the answer is one company, its cik.
  * citations: "sql:<query_id>" for queries and chunk_ids for passages that support it.
  * Every number in the answer must come from a tool result or calculate.
- If the lake does not hold the answer (other companies, other years, data it never had),
  set answerable to false and answer "{NOT_IN_LAKE}" followed by what is missing. Do not guess.
- You have at most {MAX_TOOL_CALLS} tool calls before you must answer.

Values in the lake (match them exactly in WHERE clauses):
{{values}}"""

VALUE_QUERIES = {
    "sector (all tables)": "SELECT DISTINCT sector FROM companies ORDER BY 1",
    "funds (fund_id: fund_name)": "SELECT fund_id || ': ' || fund_name FROM funds ORDER BY 1",
    "company_financials.fiscal_year": "SELECT DISTINCT fiscal_year FROM company_financials ORDER BY 1",
    "metrics.metric": "SELECT DISTINCT metric FROM metrics ORDER BY 1",
    "doc_chunks.doc_type": "SELECT DISTINCT doc_type FROM doc_chunks ORDER BY 1",
}
_system = None


def system_prompt() -> str:
    """SYSTEM plus the lake's categorical values, read once through the backend.

    Without them the model guesses 'Energy' where the lake says 'energy' and gets 0 rows.
    """
    global _system
    if _system is None:
        lines = []
        for label, sql in VALUE_QUERIES.items():
            try:
                rows = T.get_backend().query(sql, max_rows=60).rows
                lines.append(f"- {label}: " + ", ".join(str(r[0]) for r in rows))
            except Exception as e:  # noqa: BLE001: the prompt still works without hints
                lines.append(f"- {label}: (unavailable: {type(e).__name__})")
        _system = SYSTEM.replace("{values}", "\n".join(lines))
    return _system


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description, "inputSchema": {"json": {
        "type": "object", "properties": properties, "required": required}}}


TOOLS = [
    _tool("resolve_company", "Find a company's cik from a name, former name or ticker. "
          "Returns cik, name, score (0-100) and decision (match or abstain).",
          {"name": {"type": "string"},
           "as_of": {"type": "string", "description": "optional YYYY-MM-DD"}}, ["name"]),
    _tool("run_sql", "Run one read-only SELECT (Presto/Trino SQL) over the lake tables. "
          "Returns query_id, columns and up to 200 rows.",
          {"query": {"type": "string"}}, ["query"]),
    _tool("search_docs", "Search 10-K and fund-report passages. Returns chunk_ids and text.",
          {"query": {"type": "string"},
           "cik": {"type": "integer", "description": "only this company's 10-K passages"},
           "fund_id": {"type": "string", "description": "only this fund's report passages"},
           "k": {"type": "integer", "description": "how many passages, 1-8 (default 5)"}},
          ["query"]),
    _tool("describe_tables", "The lake's tables and columns (already in your instructions).",
          {}, []),
    _tool("calculate", "Arithmetic on plain numbers: + - * / and brackets.",
          {"expression": {"type": "string"}}, ["expression"]),
    _tool("final_answer", "Give the final answer with citations.",
          {"answer": {"type": "string"},
           "answer_value": {"type": ["number", "string", "array", "null"],
                            "items": {"type": ["string", "number"]},
                            "description": "number, list of names/ids, short string, or null"},
           "entity_cik": {"type": "integer",
                          "description": "the cik when the answer is one company"},
           "citations": {"type": "array", "items": {"type": "string"},
                         "description": "sql:<query_id> and/or chunk_ids"},
           "answerable": {"type": "boolean",
                          "description": "false if the lake does not hold the answer"}},
          ["answer", "answer_value", "citations", "answerable"]),
]


class AgentState(TypedDict, total=False):
    question: str
    messages: list                     # the Bedrock conversation so far
    pending: list                      # tool calls the model asked for in its last turn
    tool_calls: int                    # tools run so far (final_answer not counted)
    chunks: dict[str, str]             # chunk_id -> text, every passage seen this run
    queries: dict[str, str]            # query_id -> the SQL that ran, every query this run
    values: list[float]                # every number a tool returned (SQL cells, scores...)
    texts: list[str]                   # tool result texts (numbers inside count too)
    calcs: list[tuple[str, float]]     # every calculate call: (expression, result)
    ciks: dict[int, str]               # cik -> name for every company a tool returned
    bytes_scanned: int
    final: dict | None                 # the model's final_answer input
    problems: list[str]
    revisions: int
    status: str                        # "done" or "needs_human_review"
    reason: str
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]
    trace: Annotated[list[dict], operator.add]
    path: Annotated[list[str], operator.add]


# The LLM call, as a module variable so tests can swap in a fake.
converse_turn = llm.converse_turn


# ---------- verify: plain-code checks on the final answer ----------

def _number_like(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def normalize_final(final) -> dict:
    """clean_final from qa_agent, plus answer_value/entity_cik slips (strings for lists)."""
    out = clean_final(final)
    if not out:
        return out
    v = out.get("answer_value")
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                v = json.loads(s)
            except ValueError:
                pass
        else:
            try:
                v = float(s.replace(",", "").replace("$", ""))
            except ValueError:
                pass
    out["answer_value"] = v
    cik = out.get("entity_cik")
    if isinstance(cik, str) and cik.strip().isdigit():
        out["entity_cik"] = int(cik)
    elif isinstance(cik, float) and cik.is_integer():
        out["entity_cik"] = int(cik)
    return out


IDENTIFIER = re.compile(r"\b(?=[\w-]*[A-Za-z])(?=[\w-]*\d)[\w-]+\b")


def stated_numbers(answer: str) -> list[str]:
    """Numbers the answer states, skipping ids that mix letters and digits (F04, 10-K)."""
    return answer_numbers(IDENTIFIER.sub(" ", answer))


def allowed_numbers(state: AgentState, calc_results: list[float]) -> list[float]:
    """Every number the answer may use, at the scales people write money and shares in:
    as is, in thousands/millions/billions, and fractions as percents (0.153 -> 15.3)."""
    base = list(state.get("values", []))
    for text in state.get("texts", []) + [state.get("question", "")]:
        base += [float(n.replace(",", "")) for n in NUMBER.findall(text)]
    base += calc_results
    return [v * s for v in base for s in (1, 1e-3, 1e-6, 1e-9, 100)]


def value_supported(v: float, allowed: list[float]) -> bool:
    """answer_value is a raw number: it must equal an allowed one, give or take rounding."""
    return any(abs(v - a) <= max(1e-9, 0.005 * abs(a)) for a in allowed)


def verify_answer(final, state: AgentState) -> list[str]:
    """All checks on a final answer. Empty list = passed."""
    final = normalize_final(final)
    if not final:
        return ["no final answer was given"]
    problems = []

    # Calculations count as a source only if their inputs came from tools.
    seen = allowed_numbers(state, [])
    calc_results = []
    for expr, value in state.get("calcs", []):
        if (bad := bad_operands(expr, seen)):
            problems.append(f"calculation {expr!r} uses numbers no tool returned: {bad}")
        else:
            calc_results.append(value)
            seen += [value * s for s in (1, 1e-3, 1e-6, 1e-9, 100)]

    answer, cites, value = final.get("answer"), final.get("citations"), final.get("answer_value")
    if not isinstance(answer, str) or not answer.strip():
        problems.append("the answer is empty")
        answer = ""
    answerable = final.get("answerable")
    if not isinstance(answerable, bool):
        problems.append("answerable must be true or false")
    if not (value is None or isinstance(value, (str, list)) or _number_like(value)):
        problems.append("answer_value must be a number, a list, a string or null")

    # 1. Citations: produced in this run.
    if not isinstance(cites, list) or not all(isinstance(c, str) for c in cites):
        problems.append("citations must be a list of strings")
        cites = []
    for c in cites:
        if c.startswith("sql:"):
            if c[4:] not in state.get("queries", {}):
                problems.append(f"citation {c} is not a query run in this session")
        elif c not in state.get("chunks", {}):
            problems.append(f"citation {c} is not a passage retrieved in this session")
    if answerable is True and not cites:
        problems.append("an answer needs at least one citation")

    # 2. Numbers: in a tool result or a checked calculation.
    allowed = seen
    if (bad := [n for n in stated_numbers(answer) if not number_supported(n, allowed)]):
        problems.append(f"numbers not returned by any tool or calculation: {bad}")
    if _number_like(value) and answerable and not value_supported(float(value), allowed):
        problems.append(f"answer_value {value} was not returned by any tool or calculation")

    # 3. Entities: a company in the answer must be one a tool returned in this run.
    ciks = state.get("ciks", {})
    cik = final.get("entity_cik")
    if cik is not None:
        if not isinstance(cik, int) or isinstance(cik, bool):
            problems.append("entity_cik must be an integer")
        elif cik not in ciks:
            problems.append(f"entity_cik {cik} was not returned by resolve_company or a query")
    names = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    for name in names:
        if answerable and isinstance(name, str) and (hit := _company(name)) and hit not in ciks:
            problems.append(f"{name!r} looks like company cik {hit}, which no tool returned "
                            "in this run: resolve it or query it first")
    return problems


def _company(name: str) -> int | None:
    """The cik a name confidently matches in the lake, else None (sectors, fund ids...)."""
    try:
        r = T.resolve_company(name)
    except Exception:  # noqa: BLE001 (a matcher problem must not crash verify)
        return None
    return r["cik"] if r["decision"] == "match" and r["score"] >= 95 else None


# ---------- nodes ----------

def agent(state: AgentState) -> dict:
    """One LLM turn. The model picks a tool; past the cap it must pick final_answer."""
    forced = state.get("tool_calls", 0) >= MAX_TOOL_CALLS
    choice = {"tool": {"name": "final_answer"}} if forced else {"any": {}}
    try:
        r = converse_turn(system_prompt(), state["messages"], TOOLS, choice, max_tokens=1500)
    except Exception as e:  # noqa: BLE001 (llm.py already retried): send to review
        return {"status": "needs_human_review", "pending": [], "final": None,
                "reason": f"LLM call failed: {type(e).__name__}: {str(e)[:160]}",
                "trace": [{"step": "agent", "error": f"{type(e).__name__}"}],
                "path": ["agent"]}
    msg = r["message"]
    uses = [c["toolUse"] for c in msg.get("content", []) if "toolUse" in c]
    final = next((u["input"] for u in uses if u["name"] == "final_answer"), None)
    return {"messages": state["messages"] + [msg], "pending": uses, "final": final,
            "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
            "trace": [{"step": "agent", "forced": forced, "asked_for": [u["name"] for u in uses],
                       "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
                       "seconds": round(r.get("seconds", 0.0), 2)}],
            "path": ["agent"]}


def _cells(rows: list[list]) -> list[float]:
    return [float(v) for row in rows for v in row if _number_like(v)]


def run_tool(state: AgentState, name: str, args: dict) -> tuple[str, str, dict]:
    """Run one tool. Returns (result text for the model, short text for the trace,
    state updates). Raises ValueError for bad input; the caller reports it to the model."""
    if name == "run_sql":
        r = T.run_sql(str(args.get("query", "")))
        payload = {"query_id": r["query_id"], "columns": r["columns"], "rows": r["rows"],
                   "truncated": r["truncated"]}
        text = json.dumps(payload, default=str)
        if len(text) > RESULT_CHARS:
            text = text[:RESULT_CHARS] + ' ... (cut: ask for fewer rows or columns)"'
        ciks = dict(state.get("ciks", {}))
        cols = [c.lower() for c in r["columns"]]
        if "cik" in cols:
            ic = cols.index("cik")
            iname = next((cols.index(c) for c in ("name", "matched_name") if c in cols), None)
            for row in r["rows"]:
                if _number_like(row[ic]):
                    ciks[int(row[ic])] = row[iname] if iname is not None else ciks.get(int(row[ic]))
        chunks = dict(state.get("chunks", {}))
        if "chunk_id" in cols and "text" in cols:  # passages read through SQL count too
            for row in r["rows"]:
                chunks[str(row[cols.index("chunk_id")])] = str(row[cols.index("text")])
        upd = {"queries": state.get("queries", {}) | {r["query_id"]: r["sql"]},
               "values": state.get("values", []) + _cells(r["rows"]),
               "ciks": ciks, "chunks": chunks,
               "bytes_scanned": state.get("bytes_scanned", 0) + r["bytes_scanned"]}
        short = f"sql:{r['query_id']} {len(r['rows'])} rows; {r['sql'][:160]}"
        return text, short, upd
    if name == "resolve_company":
        r = T.resolve_company(str(args.get("name", "")), args.get("as_of"))
        upd = {"values": state.get("values", []) + [c["score"] for c in r["candidates"]]}
        if r["cik"] is not None:
            upd["ciks"] = state.get("ciks", {}) | {r["cik"]: r["name"]}
            upd["values"] += [float(r["cik"])]
        return json.dumps(r), f"{r['decision']} cik={r['cik']} {r['name']} ({r['score']})", upd
    if name == "search_docs":
        cik = args.get("cik")
        hits = T.search_docs(str(args.get("query", "")), int(cik) if cik not in (None, "") else None,
                             args.get("fund_id") or None, args.get("k", 5) or 5)
        text = "\n\n".join(f"[{h['chunk_id']}] ({h['doc_type']}, cik {h['cik']}, fund "
                           f"{h['fund_id']}, FY {h['fiscal_year']}, {h['section']})\n{h['text']}"
                           for h in hits) or "no matches"
        upd = {"chunks": state.get("chunks", {}) | {h["chunk_id"]: h["text"] for h in hits},
               "texts": state.get("texts", []) + [h["text"] for h in hits]}
        return text[:RESULT_CHARS * 2], f"{len(hits)} chunks: {[h['chunk_id'] for h in hits]}", upd
    if name == "calculate":
        expr = str(args.get("expression", ""))
        value = T.calculate(expr)
        return str(value), f"{expr} = {value:.6g}", \
            {"calcs": state.get("calcs", []) + [(expr, value)]}
    if name == "describe_tables":
        return T.describe_tables(), "schema", {}
    raise ValueError(f"unknown tool {name}")


def tools(state: AgentState) -> dict:
    """Run the tools the model asked for and send the results back to it."""
    updates, results, trace = {}, [], []
    count = state.get("tool_calls", 0)
    for use in state["pending"]:
        name, args = use["name"], use.get("input") or {}
        t0 = time.perf_counter()
        ok = count < MAX_TOOL_CALLS and name != "final_answer"
        if name == "final_answer":   # asked together with other tools: answer after them
            text = short = "final_answer must be called alone, after the other tools"
        elif not ok:
            text = short = "tool budget used up: call final_answer now"
        else:
            count += 1
            try:
                text, short, upd = run_tool(state | updates, name, args)
                updates |= upd
            except Exception as e:  # noqa: BLE001 (SQL errors, bad args): tell the model
                ok, text, short = False, f"error: {e}"[:1000], f"error: {e}"[:300]
        results.append({"toolResult": {"toolUseId": use["toolUseId"],
                                       "content": [{"text": text}],
                                       "status": "success" if ok else "error"}})
        trace.append({"step": "tool", "tool": name, "args": args, "ok": ok,
                      "result": short[:300], "seconds": round(time.perf_counter() - t0, 3)})
    return updates | {"messages": state["messages"] + [{"role": "user", "content": results}],
                      "pending": [], "final": None, "tool_calls": count, "trace": trace,
                      "path": ["tools"]}


def verify(state: AgentState) -> dict:
    problems = verify_answer(state.get("final"), state)
    return {"problems": problems, "path": ["verify"],
            "trace": [{"step": "verify", "problems": problems}]}


def revise(state: AgentState) -> dict:
    """Send the failed checks back to the model as the result of its final_answer call."""
    feedback = ("Rejected by the checks:\n- " + "\n- ".join(state["problems"]) +
                "\nFix these and call final_answer again. You may use tools again if needed.")
    pending = state.get("pending") or []
    if pending:
        content = [{"toolResult": {"toolUseId": u["toolUseId"], "status": "error",
                                   "content": [{"text": feedback if u["name"] == "final_answer"
                                                else "not run"}]}} for u in pending]
    else:  # the model answered in plain text instead of calling a tool
        content = [{"text": feedback}]
    return {"messages": state["messages"] + [{"role": "user", "content": content}],
            "pending": [], "final": None, "revisions": state.get("revisions", 0) + 1,
            "path": ["revise"]}


def done(state: AgentState) -> dict:
    return {"status": "done", "reason": "checks passed", "path": ["done"]}


def needs_human_review(state: AgentState) -> dict:
    reason = state.get("reason") or "checks failed after revision: " + "; ".join(state["problems"])
    return {"status": "needs_human_review", "reason": reason, "path": ["needs_human_review"]}


# ---------- edges ----------

def route_after_agent(state: AgentState) -> str:
    if state.get("status"):
        return "needs_human_review"
    if state.get("final") is not None and len(state.get("pending", [])) == 1:
        return "verify"
    return "tools" if state.get("pending") else "verify"  # plain text: verify will reject it


def route_after_verify(state: AgentState) -> str:
    if not state["problems"]:
        return "done"
    return "revise" if state.get("revisions", 0) < MAX_REVISIONS else "needs_human_review"


def build_graph():
    g = StateGraph(AgentState)
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


def answer_question(question: str) -> dict:
    """Run the agent on one question; returns the full final state."""
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    start = {"question": question, "tool_calls": 0, "revisions": 0, "chunks": {}, "queries": {},
             "values": [], "texts": [], "calcs": [], "ciks": {}, "bytes_scanned": 0,
             "messages": [{"role": "user", "content": [{"text": f"Question: {question}"}]}]}
    # Each tool round is two steps (agent, tools); 40 is far above the worst case (~20).
    return _GRAPH.invoke(start, {"recursion_limit": 40})


def ask(question: str) -> dict:
    """Run one question and return a JSON-ready summary (what app.py and evaluate.py use)."""
    t0 = time.perf_counter()
    state = answer_question(question)
    final = normalize_final(state.get("final")) or {}
    tin, tout = state.get("tokens_in", 0), state.get("tokens_out", 0)
    return {"question": question, "answer": final.get("answer"),
            "answer_value": final.get("answer_value"), "entity_cik": final.get("entity_cik"),
            "citations": final.get("citations") or [], "answerable": final.get("answerable"),
            "status": state.get("status"), "reason": state.get("reason"),
            "tool_calls": state.get("tool_calls", 0), "revisions": state.get("revisions", 0),
            "tokens_in": tin, "tokens_out": tout, "cost_usd": round(llm.cost(tin, tout), 6),
            "bytes_scanned": state.get("bytes_scanned", 0),
            "seconds": round(time.perf_counter() - t0, 2),
            "queries": state.get("queries", {}), "path": state.get("path", []),
            "trace": state.get("trace", [])}


def print_result(r: dict) -> None:
    print(f"\nanswer      {r['answer']}")
    print(f"value       {r['answer_value']!r}   entity_cik {r['entity_cik']}")
    print(f"answerable  {r['answerable']}")
    print(f"status      {r['status']}  ({r['reason']})")
    print(f"citations   {r['citations']}")
    for qid, sql in r["queries"].items():
        print(f"  sql:{qid}  {sql}")
    print("trace:")
    for i, t in enumerate(r["trace"], 1):
        if t["step"] == "tool":
            print(f"  {i:2}. tool {t['tool']}({json.dumps(t['args'])[:140]}) -> {t['result'][:140]}")
        elif t["step"] == "agent":
            print(f"  {i:2}. agent asked for {t.get('asked_for')} "
                  f"({t.get('tokens_in')} in / {t.get('tokens_out')} out, {t.get('seconds')}s)")
        else:
            print(f"  {i:2}. verify problems={t.get('problems')}")
    print(f"path        {' > '.join(r['path'])}")
    print(f"tool calls  {r['tool_calls']}   tokens in {r['tokens_in']} out {r['tokens_out']}   "
          f"cost ${r['cost_usd']:.4f}   {r['seconds']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mermaid", action="store_true", help="print the graph diagram")
    parser.add_argument("--question", "-q")
    parser.add_argument("--json", action="store_true", help="print the raw JSON result")
    args = parser.parse_args()
    if args.mermaid or not args.question:
        print(build_graph().get_graph().draw_mermaid())
    else:
        result = ask(args.question)
        print(json.dumps(result, indent=2, default=str) if args.json else "", end="")
        if not args.json:
            print_result(result)
