"""Tests for the lake analyst agent (agent/): SQL guard, backend, verify, call cap, scoring.

No AWS and no LLM: a tiny DuckDB lake is built from agent/schema.py, and the LLM is a
fake that replays scripted tool calls.  Run:  uv run python -m pytest -q tests/test_agent.py
"""

import sys
from pathlib import Path

import duckdb
import pytest

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))

import agent  # noqa: E402,F401  (puts extraction/ on sys.path)
from agent import backend as B  # noqa: E402
from agent import evaluate as E  # noqa: E402
from agent import graph as G  # noqa: E402
from agent import tools as T  # noqa: E402
from agent.schema import DUCKDB_TYPE, TABLES  # noqa: E402

ROWS = {
    "companies": [(320193, "Apple Inc.", "Technology", 3571, "Apple Computer Inc | AAPL"),
                  (1652044, "Alphabet Inc.", "Technology", 7370, "Google Inc | GOOGL"),
                  (21344, "Coca-Cola Co", "Consumer", 2080, "KO")],
    "funds": [("F1", "Granite Growth Fund I", 2018, 500e6, 400e6, 450e6, "2023-12-31", "f1.pdf")],
    "holdings": [("F1", "Apple Computer", 320193, "Apple Inc.", 97.0, "match", "Technology",
                  10e6, 25e6, 0.5, "2023-12-31", "f1.pdf", 4),
                 ("F1", "Mystery Widgets LLC", None, None, 41.0, "abstain", "Industrials",
                  5e6, 4e6, 2.0, "2023-12-31", "f1.pdf", 5)],
    "metrics": [(320193, 2020, "revenue", 274515e6, "xbrl", None, "ok", "aapl-2020")],
    "doc_chunks": [("aapl-2020-risk-000", "aapl-2020", "10-K", 320193, None, 2020, "item1a",
                    "Supply chain disruption in Asia is a principal risk to iPhone production."),
                   ("f1-2023-000", "f1-2023", "fund_report", None, "F1", None, "overview",
                    "Granite Growth Fund I invests in growth-stage technology companies.")],
    "company_financials": [(320193, "Apple Inc.", "Technology", 2019, 260174e6, 55256e6, 338516e6, None),
                           (320193, "Apple Inc.", "Technology", 2020, 274515e6, 57411e6, 323888e6, 5.51),
                           (21344, "Coca-Cola Co", "Consumer", 2020, 33014e6, 7747e6, 87296e6, -11.4)],
    "fund_exposure": [("F1", "Granite Growth Fund I", "Technology", 1, 10e6, 25e6, 5.56)],
    "metric_reconciliation": [(320193, 2020, "revenue", 274515e6, 274515e6, 0.0, True)],
}


def make_lake(path: Path) -> Path:
    """The fixture lake: tables in silver/gold schemas, like a medallion layout."""
    con = duckdb.connect(str(path))
    for schema in ("silver", "gold"):
        con.execute(f"CREATE SCHEMA {schema}")
    for table, (layer, _, cols) in TABLES.items():
        con.execute(f"CREATE TABLE {layer}.{table} ("
                    + ", ".join(f"{c} {DUCKDB_TYPE[t]}" for c, t, _ in cols) + ")")
        marks = ", ".join("?" for _ in cols)
        con.executemany(f"INSERT INTO {layer}.{table} VALUES ({marks})", ROWS[table])
    con.execute("CREATE TABLE main.secrets (k VARCHAR, v VARCHAR)")   # must stay unreachable
    con.execute("INSERT INTO main.secrets VALUES ('key', 'hunter2')")
    con.close()
    return path


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    path = make_lake(tmp_path_factory.mktemp("lake") / "lake.duckdb")
    B.set_backend(B.LocalDuckDB(path))
    T.reset_caches()
    yield path
    B.set_backend(None)
    T.reset_caches()


# ---------------------------------------------------------------------------
# SQL guard
# ---------------------------------------------------------------------------

ATTACKS = [
    # more than one statement
    "SELECT * FROM companies; DROP TABLE companies",
    "SELECT 1; SELECT 2",
    "SELECT * FROM companies;DELETE FROM holdings",
    "SELECT * FROM companies -- harmless\n; DROP TABLE funds",
    "SELECT * FROM companies /* ; */ ; UPDATE funds SET nav_usd = 0",
    # not a SELECT
    "DROP TABLE companies",
    "DELETE FROM holdings",
    "INSERT INTO companies VALUES (1, 'x', 'y', 1, '')",
    "UPDATE funds SET nav_usd = 0",
    "CREATE TABLE x AS SELECT * FROM companies",
    "CREATE VIEW v AS SELECT 1",
    "ALTER TABLE companies ADD COLUMN x INT",
    "TRUNCATE TABLE holdings",
    "MERGE INTO funds USING companies ON 1=1 WHEN MATCHED THEN DELETE",
    "SET enable_external_access = true",
    "PRAGMA table_info('companies')",
    "ATTACH '/tmp/x.db' AS x",
    "COPY companies TO '/tmp/out.csv'",
    "SHOW TABLES",
    "DESCRIBE companies",
    "EXPLAIN SELECT * FROM companies",
    "INSTALL httpfs",
    "GRANT SELECT ON companies TO public",
    "CALL system.sync_partition_metadata('a', 'b', 'FULL')",
    "UNLOAD (SELECT * FROM companies) TO 's3://bucket/x' WITH (format = 'PARQUET')",
    # DML hidden inside a SELECT
    "SELECT * INTO stolen FROM companies",
    "WITH x AS (DELETE FROM holdings RETURNING *) SELECT * FROM x",
    "SELECT * FROM companies FOR UPDATE",
    # tables outside the allowlist
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM information_schema.columns WHERE table_name = 'companies'",
    "SELECT * FROM pg_catalog.pg_tables",
    "SELECT * FROM secrets",
    "SELECT * FROM main.secrets",
    "SELECT * FROM lake.main.secrets",
    "SELECT * FROM silver.companies",
    "SELECT * FROM awsdatacatalog.other_db.users",
    "SELECT * FROM \"information_schema\".\"tables\"",
    "SELECT cik FROM companies UNION SELECT k FROM secrets",
    "SELECT cik FROM companies UNION ALL SELECT table_name FROM information_schema.tables",
    "SELECT name FROM companies WHERE cik IN (SELECT cik FROM other_schema.t)",
    "WITH companies AS (SELECT * FROM secrets) SELECT * FROM companies",
    "SELECT * FROM (WITH secrets AS (SELECT 1 AS k) SELECT * FROM secrets) t CROSS JOIN secrets",
    "SELECT * FROM INFORMATION_SCHEMA.TABLES",
    "SELECT * FROM companies WHERE cik = (SELECT max(cik) FROM secrets)",
    "SELECT * FROM companies c JOIN system.runtime.queries q ON 1=1",
    # table functions and engine extras
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('s3://other-bucket/*.parquet')",
    "SELECT * FROM read_text('/etc/hosts')",
    "SELECT * FROM query('SELECT * FROM secrets')",
    "SELECT * FROM duckdb_tables()",
    "SELECT * FROM duckdb_settings()",
    "SELECT getenv('AWS_SECRET_ACCESS_KEY')",
    "SELECT current_setting('s3_secret_access_key')",
    "SELECT * FROM glob('/home/*')",
    "SELECT * FROM 's3://bucket/file.parquet'",
    "SELECT * FROM '/etc/passwd'",
    # junk
    "",
    "   ",
    "SELECT * FROM",
    "SELEC * FROM companies",
    "SELECT " + "1+" * 3000 + "1",
]


@pytest.mark.parametrize("query", ATTACKS)
def test_guard_rejects(query):
    with pytest.raises(T.SQLRejected):
        T.guard_sql(query)


SAFE = [
    "SELECT * FROM companies",
    "select name, sector from companies where cik = 320193",
    "SELECT sector, count(*) AS n FROM holdings GROUP BY sector ORDER BY n DESC",
    "SELECT c.name, f.revenue FROM company_financials f JOIN companies c ON c.cik = f.cik",
    "WITH t AS (SELECT cik, revenue FROM company_financials WHERE fiscal_year = 2020) "
    "SELECT * FROM t ORDER BY revenue DESC LIMIT 3",
    "SELECT cik FROM companies UNION SELECT cik FROM holdings",
    "SELECT name, rank() OVER (ORDER BY revenue DESC) FROM company_financials",
    "SELECT approx_percentile(revenue, 0.5) FROM company_financials",
    "SELECT * FROM companies /* a comment */ WHERE cik = 1 -- trailing comment",
    "SELECT * FROM companies;",
    "SELECT 'DROP TABLE companies' AS s FROM companies",
    "SELECT \"name\" FROM \"companies\"",
    "WITH secrets AS (SELECT cik FROM companies) SELECT * FROM secrets",   # a CTE, not a table
    "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 3) SELECT * FROM n",
]


@pytest.mark.parametrize("query", SAFE)
def test_guard_allows(query):
    out = T.guard_sql(query)
    assert "LIMIT" in out.upper()
    assert "--" not in out and "/*" not in out          # comments never reach the engine


@pytest.mark.parametrize("query,expected", [
    ("SELECT * FROM companies", 200),                    # added
    ("SELECT * FROM companies LIMIT 5000", 200),         # lowered
    ("SELECT * FROM companies LIMIT 10", 10),            # kept
    ("SELECT cik FROM companies UNION SELECT cik FROM holdings LIMIT 999", 200),
])
def test_guard_limit(query, expected):
    tree = __import__("sqlglot").parse_one(T.guard_sql(query), read="duckdb")
    assert int(tree.args["limit"].expression.this) == expected


def test_guard_rewrites_for_engine():
    assert "APPROX_QUANTILE" in T.guard_sql("SELECT approx_percentile(revenue, 0.5) "
                                            "FROM company_financials", dialect="duckdb")
    assert "APPROX_PERCENTILE" in T.guard_sql("SELECT approx_percentile(revenue, 0.5) "
                                              "FROM company_financials", dialect="trino")


# ---------------------------------------------------------------------------
# Backend and tools on the fixture lake
# ---------------------------------------------------------------------------

def test_backend_reads_views(lake):
    r = B.get_backend().query("SELECT name FROM companies ORDER BY cik")
    assert [x[0] for x in r.rows] == ["Coca-Cola Co", "Apple Inc.", "Alphabet Inc."]
    assert r.query_id.startswith("local-")


def test_backend_is_read_only_and_sealed(lake):
    be = B.get_backend()
    with pytest.raises(duckdb.Error):
        be.query("DELETE FROM lake.silver.companies")       # attached read-only
    with pytest.raises(duckdb.Error):
        be.query("SELECT * FROM read_csv('/etc/hosts')")    # external access off
    with pytest.raises(duckdb.Error):
        be.query("SET enable_external_access = true")       # configuration locked


def test_backend_caps_rows(lake):
    r = B.get_backend().query("SELECT * FROM range(500)", max_rows=200)
    assert len(r.rows) == 200 and r.truncated


def test_run_sql_end_to_end(lake):
    r = T.run_sql("SELECT name, revenue FROM company_financials WHERE fiscal_year = 2020 "
                  "ORDER BY revenue DESC")
    assert r["rows"][0] == ["Apple Inc.", 274515e6] and r["sql"].endswith("LIMIT 200")
    with pytest.raises(T.SQLRejected):
        T.run_sql("SELECT * FROM secrets")


def test_resolve_company(lake):
    r = T.resolve_company("Apple Computer, Inc.")
    assert r["decision"] == "match" and r["cik"] == 320193
    assert T.resolve_company("KO")["cik"] == 21344                    # alias as ticker
    assert T.resolve_company("Zzyzx Quantum Holdings")["cik"] is None  # abstain


def test_search_docs(lake):
    hits = T.search_docs("supply chain risk iPhone")
    assert hits[0]["chunk_id"] == "aapl-2020-risk-000"
    assert T.search_docs("supply chain", fund_id="F1") == [] or \
        all(h["fund_id"] == "F1" for h in T.search_docs("supply chain", fund_id="F1"))
    assert all(h["cik"] == 320193 for h in T.search_docs("risk", cik=320193))


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def state_with(**kw):
    base = {"question": "What was Apple's revenue in 2020?", "chunks": {"c1": "Revenue rose."},
            "queries": {"q1": "SELECT ..."}, "values": [274515e6, 2020.0], "texts": [],
            "calcs": [], "ciks": {320193: "Apple Inc."}}
    return base | kw


def final(**kw):
    return {"answer": "Apple's revenue in 2020 was $274.5 billion.", "answer_value": 274515e6,
            "citations": ["sql:q1"], "answerable": True} | kw


def test_verify_passes_good_answer(lake):
    assert G.verify_answer(final(), state_with()) == []


def test_verify_citation_must_come_from_this_run(lake):
    probs = G.verify_answer(final(citations=["sql:q999", "made-up-chunk"]), state_with())
    assert any("q999" in p for p in probs) and any("made-up-chunk" in p for p in probs)
    assert any("citation" in p for p in G.verify_answer(final(citations=[]), state_with()))


def test_verify_numbers_must_come_from_tools(lake):
    probs = G.verify_answer(final(answer="Revenue was $301.2 billion.", answer_value=301.2e9),
                            state_with())
    assert any("301.2" in p for p in probs) and any("answer_value" in p for p in probs)


def test_verify_calculation_inputs_checked(lake):
    good = state_with(values=[274515e6, 260174e6], calcs=[("(274515 - 260174) / 260174 * 100",
                                                          5.512)])
    assert G.verify_answer(final(answer="Revenue grew 5.5% in 2020.", answer_value=5.512), good) == []
    bad = state_with(calcs=[("(300000 - 250000) / 250000 * 100", 20.0)])
    probs = G.verify_answer(final(answer="Revenue grew 20%.", answer_value=20.0), bad)
    assert any("calculation" in p for p in probs)


def test_verify_entity_must_be_resolved(lake):
    probs = G.verify_answer(final(answer="Coca-Cola.", answer_value="Coca-Cola Co",
                                  entity_cik=21344), state_with())
    assert any("entity_cik 21344" in p for p in probs)
    assert any("cik 21344" in p for p in probs if "looks like" in p)
    ok = state_with(ciks={21344: "Coca-Cola Co"}, values=[])
    assert G.verify_answer(final(answer="Coca-Cola.", answer_value="Coca-Cola Co",
                                 entity_cik=21344), ok) == []


def test_verify_unanswerable_needs_no_citation(lake):
    f = {"answer": "This is not in the data lake: no 2031 data.", "answer_value": None,
         "citations": [], "answerable": False}
    assert G.verify_answer(f, state_with(question="Apple's revenue in 2031?")) == []


def test_verify_handles_string_slips(lake):
    f = final(citations='["sql:q1"]', answerable="true", answer_value="274515000000")
    assert G.verify_answer(f, state_with()) == []
    assert G.verify_answer(None, state_with()) == ["no final answer was given"]


# ---------------------------------------------------------------------------
# The graph with a fake LLM: call cap, forced answer, revise, review
# ---------------------------------------------------------------------------

class FakeLLM:
    """Replays a script of turns. Each turn is a list of (tool name, input)."""

    def __init__(self, turns):
        self.turns, self.calls, self.choices = list(turns), 0, []

    def __call__(self, system, messages, tools, choice, max_tokens=1000):
        self.calls += 1
        self.choices.append(choice)
        if "tool" in choice:                     # forced final_answer
            turn = [("final_answer", final_from_last(messages))]
        else:
            turn = self.turns.pop(0) if self.turns else [("final_answer", {})]
        content = [{"toolUse": {"toolUseId": f"t{self.calls}-{i}", "name": n, "input": a}}
                   for i, (n, a) in enumerate(turn)]
        return {"message": {"role": "assistant", "content": content},
                "tokens_in": 100, "tokens_out": 20, "seconds": 0.01}


def final_from_last(messages):
    """For the forced answer: cite the latest query id seen in a tool result."""
    import json as _json
    for m in reversed(messages):
        for c in m.get("content", []):
            text = c.get("toolResult", {}).get("content", [{}])[0].get("text", "")
            if text.startswith("{"):
                qid = _json.loads(text)["query_id"]
                return {"answer": "Apple is in the lake.", "answer_value": None,
                        "citations": [f"sql:{qid}"], "answerable": True}
    return {"answer": "x", "answer_value": None, "citations": [], "answerable": True}


def test_call_cap_forces_final_answer(lake, monkeypatch):
    sql = ("run_sql", {"query": "SELECT name FROM companies"})
    fake = FakeLLM([[sql]] * 20)                  # a model that never stops querying
    monkeypatch.setattr(G, "converse_turn", fake)
    r = G.ask("Name the companies.")
    assert r["tool_calls"] == G.MAX_TOOL_CALLS
    assert fake.choices[-1] == {"tool": {"name": "final_answer"}}
    assert r["status"] == "done"
    assert sum(1 for t in r["trace"] if t["step"] == "tool") == G.MAX_TOOL_CALLS


def test_parallel_calls_past_cap_are_refused(lake, monkeypatch):
    sql = ("run_sql", {"query": "SELECT 1 AS one FROM companies"})
    fake = FakeLLM([[sql] * 10])                  # 10 tools in one turn
    monkeypatch.setattr(G, "converse_turn", fake)
    r = G.ask("x")
    tools_run = [t for t in r["trace"] if t["step"] == "tool"]
    assert sum(t["ok"] for t in tools_run) == G.MAX_TOOL_CALLS
    assert any("budget" in t["result"] for t in tools_run)


def test_revise_once_then_human_review(lake, monkeypatch):
    bad = ("final_answer", {"answer": "Revenue was $999 billion.", "answer_value": 999e9,
                            "citations": ["sql:nope"], "answerable": True})
    fake = FakeLLM([[bad], [bad]])
    monkeypatch.setattr(G, "converse_turn", fake)
    r = G.ask("What was Apple's revenue?")
    assert r["status"] == "needs_human_review" and r["revisions"] == 1
    assert r["path"] == ["agent", "verify", "revise", "agent", "verify", "needs_human_review"]
    assert "999" in r["reason"]


def test_good_run_is_done_with_trace(lake, monkeypatch):
    turns = [[("resolve_company", {"name": "Apple"})],
             [("run_sql", {"query": "SELECT revenue FROM company_financials "
                                    "WHERE cik = 320193 AND fiscal_year = 2020"})]]
    fake = FakeLLM(turns)

    def script(system, messages, tools, choice, max_tokens=1000):
        if fake.calls == 2:  # after the SQL result: answer from it
            import json as _json
            qid = _json.loads(messages[-1]["content"][0]["toolResult"]["content"][0]["text"])[
                "query_id"]
            fake.turns.append([("final_answer", {
                "answer": "Apple's 2020 revenue was $274.5 billion.", "answer_value": 274515e6,
                "entity_cik": 320193, "citations": [f"sql:{qid}"], "answerable": True})])
        return fake(system, messages, tools, choice, max_tokens)

    monkeypatch.setattr(G, "converse_turn", script)
    r = G.ask("What was Apple's revenue in 2020?")
    assert r["status"] == "done", r["reason"]
    assert [t["tool"] for t in r["trace"] if t["step"] == "tool"] == ["resolve_company", "run_sql"]
    assert r["cost_usd"] > 0 and r["tool_calls"] == 2


def test_llm_failure_goes_to_review(lake, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bedrock down")
    monkeypatch.setattr(G, "converse_turn", boom)
    r = G.ask("x")
    assert r["status"] == "needs_human_review" and "bedrock down" in r["reason"]


# ---------------------------------------------------------------------------
# Evaluation scoring
# ---------------------------------------------------------------------------

def res(**kw):
    return {"answer": "", "answer_value": None, "citations": [], "answerable": True,
            "status": "done", "entity_cik": None} | kw


def test_score_number():
    assert E.score({"answer_kind": "number", "expected": 656978000.0, "tolerance": {"rel": 0.005}},
                   res(answer_value=657e6))["correct"]
    assert E.score({"answer_kind": "number", "expected": 5.5, "tolerance": {"abs": 0.1}},
                   res(answer_value="5.45"))["correct"]
    assert not E.score({"answer_kind": "number", "expected": 3, "tolerance": {"abs": 0}},
                       res(answer_value=4))["correct"]
    q = {"answer_kind": "number", "expected": 100.0, "tolerance": 0.01}
    assert E.score(q, res(answer_value=100.5))["correct"]
    assert not E.score(q, res(answer_value=103))["correct"]
    assert not E.score(q, res(answer_value=None))["correct"]
    assert E.score(q | {"tolerance": 0}, res(answer_value=100.0))["correct"]


def test_score_set():
    q = {"answer_kind": "set", "expected": ["Apple Inc.", "Alphabet Inc"]}
    s = E.score(q, res(answer_value=["alphabet inc.", "APPLE INC"]))
    assert s["correct"] and s["precision"] == 1 and s["recall"] == 1
    s = E.score(q, res(answer_value=["Apple Inc.", "Coca-Cola"]))
    assert not s["correct"] and s["precision"] == 0.5 and s["recall"] == 0.5


def test_score_entity():
    q = {"answer_kind": "entity", "expected": 320193}
    assert E.score(q, res(entity_cik=320193))["correct"]
    assert not E.score(q, res(entity_cik=21344))["correct"]
    assert E.score({"answer_kind": "entity", "expected": {"cik": 320193}},
                   res(entity_cik=320193))["correct"]
    assert E.score({"answer_kind": "entity", "expected": {"cik": 320193}},
                   res(answer_value=320193))["correct"]
    fund = {"answer_kind": "entity", "expected": {"fund_id": "F04", "fund_name": "Tidewrack Fund"}}
    assert E.score(fund, res(answer_value="F04"))["correct"]
    assert E.score(fund, res(answer_value="Tidewrack Fund"))["correct"]
    assert not E.score(fund, res(answer_value="F05"))["correct"]


def test_score_set_of_ciks_and_empty():
    q = {"answer_kind": "set", "expected": [1800, 899866]}
    assert E.score(q, res(answer_value=["899866", 1800]))["correct"]
    empty = {"answer_kind": "set", "expected": []}
    assert E.score(empty, res(answer_value=[]))["correct"]
    assert not E.score(empty, res(answer_value=["F01"]))["correct"]


def test_score_text():
    q = {"answer_kind": "text", "expected": "Discusses supply chain risk (look for 'supply').",
         "expected_sources": {"chunk_prefixes": ["aapl-2020"], "keywords": ["supply"]}}
    good = res(answer="The main risk is supply chain disruption.",
               citations=["aapl-2020-risk-000"])
    assert E.score(q, good)["correct"]
    assert not E.score(q, good | {"citations": ["f1-2023-000"]})["correct"]   # wrong source
    assert not E.score(q, good | {"answer": "Currency moves."})["correct"]     # no keyword
    assert not E.score(q, good | {"status": "needs_human_review"})["correct"]  # invalid cites


def test_score_unanswerable():
    q = {"answer_kind": "unanswerable", "expected": None}
    assert E.score(q, res(answerable=False))["correct"]
    assert not E.score(q, res(answerable=True))["correct"]


# ---------------------------------------------------------------------------
# The real lake, when the lake module has built it: its Parquet matches the contract
# ---------------------------------------------------------------------------

LAKE_DIR = ML / "data" / "lake"


@pytest.mark.skipif(not (LAKE_DIR / "silver").exists(), reason="lake not built yet")
@pytest.mark.parametrize("table", list(TABLES))
def test_real_parquet_matches_contract(table):
    import pyarrow.parquet as pq

    layer, _, cols = TABLES[table]
    path = LAKE_DIR / layer / table / "data.parquet"
    assert path.exists(), f"missing {path}"
    schema = pq.read_schema(path)
    arrow = {"bigint": "int64", "int": "int32", "double": "double", "string": "string",
             "date": "date32[day]", "boolean": "bool"}
    got = {f.name: str(f.type) for f in schema}
    for c, t, _ in cols:
        assert c in got, f"{table}.{c} missing from Parquet"
        ok = got[c] == arrow[t] or (t == "string" and got[c] in ("large_string", "string_view")) \
            or (got[c] == "null")
        assert ok, f"{table}.{c}: Parquet {got[c]} vs Glue {t}"


def test_verify_ignores_ids_with_digits(lake):
    f = final(answer="Fund F04 (see its 10-K style report) holds $274.5 billion.")
    assert G.verify_answer(f, state_with()) == []
