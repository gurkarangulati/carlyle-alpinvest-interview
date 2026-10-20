"""The analyst agent's tools, as plain Python functions (graph.py wires them to the LLM).

    resolve_company(name, as_of)   name -> cik with the matching module's hybrid matcher
    run_sql(query)                 ONE guarded SELECT over the lake tables (the safety point)
    describe_tables()              the schema with one line per column
    search_docs(query, cik, fund_id, k)   TF-IDF over doc_chunks
    calculate(expression)          safe arithmetic (reused from extraction/qa_tools.py)

Every tool reads the lake through backend.get_backend(), so the same code runs on the
laptop (DuckDB) and in AWS (Athena).
"""

import logging
import re
import threading
from decimal import Decimal

import sqlglot
from sqlglot import exp

from agent.backend import MAX_ROWS, get_backend
from agent.schema import ALLOWED_TABLES
from agent.schema import describe_tables as _describe

from qa_tools import MAX_K, calculate  # noqa: F401  (re-exported: the safe calculator)

logging.getLogger("sqlglot").setLevel(logging.ERROR)  # it warns on SHOW etc.; we reject those

# ---------------------------------------------------------------------------
# run_sql: the SQL guard
# ---------------------------------------------------------------------------

MAX_SQL_CHARS = 4000

# Statement kinds and clauses that are never allowed anywhere in the tree. Looked up by
# name because sqlglot adds and renames classes between versions.
_FORBIDDEN_NAMES = [
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "Command", "Copy",
    "Set", "Pragma", "Use", "Describe", "Transaction", "Commit", "Rollback", "Grant",
    "LoadData", "Into", "Lock", "TruncateTable", "Attach", "Detach", "Install", "Summarize",
    "Export", "Show", "Analyze", "Cache", "Uncache", "Refresh",
]
FORBIDDEN_NODES = tuple(c for n in _FORBIDDEN_NAMES if isinstance(c := getattr(exp, n, None), type))
# Plain functions the model may call that sqlglot does not know by a typed class.
# Everything else that parses as an unknown ("anonymous") function is rejected, which
# blocks engine extras such as DuckDB's getenv(), query(), duckdb_tables(), read_text().
SAFE_ANONYMOUS = {"approx_percentile", "approx_distinct", "date_diff", "date_add",
                  "date_format", "regexp_like", "strpos", "split_part", "format",
                  "round", "truncate", "try", "typeof"}


class SQLRejected(ValueError):
    """The query broke a guard rule. The message tells the model what to fix."""


def guard_sql(query: str, dialect: str = "duckdb", limit: int = MAX_ROWS) -> str:
    """Check one model-written query and return the SQL to actually run.

    Rules (each a test in tests/test_agent.py):
      1. one statement only (no `;` chains), at most MAX_SQL_CHARS characters;
      2. it must be a SELECT (or SELECTs joined by UNION/INTERSECT/EXCEPT), WITH allowed;
      3. no DDL/DML/session commands anywhere inside (INSERT, DROP, SET, SELECT INTO ...);
      4. every table is an allowlisted lake table or a CTE defined in the query, with no
         schema/catalog prefix (so no information_schema, pg_catalog, other databases);
      5. no table functions (read_csv, query(...)) and no unknown functions;
      6. LIMIT is enforced: added if missing, lowered to `limit` if higher.

    We never run the text the model wrote. We run sqlglot's re-generation of the parsed
    tree, so comments are dropped and the SQL is rewritten for the engine's dialect.
    """
    if not isinstance(query, str) or not query.strip():
        raise SQLRejected("empty query")
    if len(query) > MAX_SQL_CHARS:
        raise SQLRejected(f"query longer than {MAX_SQL_CHARS} characters")
    try:
        statements = [s for s in sqlglot.parse(query, read="trino")
                      if s is not None and not isinstance(s, exp.Semicolon)]
    except sqlglot.errors.SqlglotError as e:
        raise SQLRejected(f"could not parse as Trino SQL: {str(e).splitlines()[0][:200]}") from e
    if len(statements) != 1:
        raise SQLRejected(f"exactly one statement allowed, got {len(statements)}")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SQLRejected(f"only SELECT is allowed, got {type(tree).__name__.upper()}")

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SQLRejected(f"{type(node).__name__.upper()} is not allowed")
        if isinstance(node, exp.Select) and (node.args.get("into") or node.args.get("locks")):
            raise SQLRejected("SELECT INTO / FOR UPDATE are not allowed")
        if isinstance(node, exp.Anonymous) and node.name.lower() not in SAFE_ANONYMOUS:
            raise SQLRejected(f"function {node.name}() is not allowed")
        if isinstance(node, exp.Func) and type(node).__name__.lower().startswith("read"):
            raise SQLRejected(f"{node.sql_name()} is not allowed")

    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SQLRejected("table functions are not allowed; select from the lake tables")
        if table.args.get("db") or table.args.get("catalog"):
            raise SQLRejected(f"use plain table names; {table.sql('trino')} names a schema")
        name = table.name.lower()
        if name not in ALLOWED_TABLES and not _cte_in_scope(table, name):
            raise SQLRejected(f"table {table.name!r} is not in the lake; "
                              f"allowed: {', '.join(sorted(ALLOWED_TABLES))}")

    current = tree.args.get("limit")
    value = current.expression if current is not None else None
    if not (isinstance(value, exp.Literal) and value.is_int and int(value.this) <= limit):
        tree = tree.limit(limit)  # replaces a missing, too-large or non-literal LIMIT
    return tree.sql(dialect=dialect, comments=False)


def _cte_in_scope(table: exp.Table, name: str) -> bool:
    """Is `name` a CTE visible where this table is used? We walk up from the table and
    look at each enclosing WITH. A CTE defined inside one subquery does not count
    elsewhere, so `(WITH secrets AS (...) ...) JOIN secrets` cannot sneak a real table in."""
    node = table.parent
    while node is not None:
        with_ = (node.args.get("with_") or node.args.get("with")) \
            if isinstance(node, exp.Expression) else None
        if with_ is not None and any(c.alias_or_name.lower() == name for c in with_.expressions):
            return True
        node = node.parent
    return False


def _jsonable(v):
    if isinstance(v, Decimal):
        return float(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def run_sql(query: str) -> dict:
    """Guard, run, and return {"query_id", "sql", "columns", "rows", "truncated"}.

    Raises SQLRejected (rule broken) or the engine's error (bad column etc.); the graph
    sends either message back to the model so it can fix the query.
    """
    backend = get_backend()
    safe = guard_sql(query, dialect=backend.dialect)
    r = backend.query(safe, max_rows=MAX_ROWS)  # the engine also stops at MAX_ROWS
    return {"query_id": r.query_id, "sql": safe, "columns": r.columns,
            "rows": [[_jsonable(v) for v in row] for row in r.rows],
            "truncated": r.truncated, "bytes_scanned": r.bytes_scanned}


def describe_tables() -> str:
    return _describe()


# ---------------------------------------------------------------------------
# resolve_company: the matching module over the lake's companies table
# ---------------------------------------------------------------------------

_matcher = None
_matcher_lock = threading.Lock()
TICKER = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")


def load_matcher():
    """Build the hybrid matcher from `companies` (name + ' | '-joined aliases), once.

    The lake's aliases carry no dates, so `as_of` cannot exclude a name here; it is
    passed through for when the lake adds valid_from/valid_to. Aliases that look like
    tickers (AAPL) are indexed as tickers: exact lookup, never fuzzy.
    """
    global _matcher
    with _matcher_lock:
        if _matcher is None:
            import json

            from matching.matcher import Matcher
            rows = get_backend().query("SELECT cik, name, aliases FROM companies",
                                       max_rows=None).rows
            universe = []
            for cik, name, aliases in rows:
                parts = [a.strip() for a in (aliases or "").split("|") if a.strip()]
                universe.append({
                    "cik": int(cik), "name": name,
                    "former_names": [{"name": a} for a in parts if not TICKER.match(a)
                                     and a.lower() != str(name).lower()],
                    "tickers": [a for a in parts if TICKER.match(a)]})
            from agent import ML
            t_path = ML / "matching" / "results" / "hybrid_thresholds.json"
            t = json.loads(t_path.read_text()) if t_path.exists() else {}
            _matcher = Matcher(universe, t.get("min_score", 90.0), t.get("min_margin", 3.0))
    return _matcher


def resolve_company(name: str, as_of: str | None = None) -> dict:
    """{"query", "cik", "name", "score", "decision", "candidates"}.

    decision "match": use cik. "abstain": the name is unknown or ambiguous; cik is None
    and the top candidates are shown so the model can ask or try another spelling.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name is required")
    m = load_matcher().match(name.strip()[:200], k=3, as_of=as_of)
    top = m["candidates"][0] if m["candidates"] else None
    ok = m["decision"] == "match" and top is not None
    return {"query": name, "cik": top["cik"] if ok else None, "name": top["name"] if ok else None,
            "score": top["score"] if top else 0.0, "decision": m["decision"],
            "candidates": [{"cik": c["cik"], "name": c["name"], "score": c["score"]}
                           for c in m["candidates"]]}


# ---------------------------------------------------------------------------
# search_docs: TF-IDF over doc_chunks (same pattern as extraction/qa_tools.py)
# ---------------------------------------------------------------------------

_index = None
_index_lock = threading.Lock()


def docs_index():
    """(rows, vectorizer, matrix) over every chunk, built once per process.

    Loaded through the backend (trusted fixed SQL, not model-written), so in AWS the
    chunks come from S3 via Athena. Bigrams catch phrases like "net sales".
    """
    global _index
    with _index_lock:
        if _index is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            r = get_backend().query("SELECT chunk_id, doc_id, doc_type, cik, fund_id, "
                                    "fiscal_year, section, text FROM doc_chunks", max_rows=None)
            rows = r.records()
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            matrix = vec.fit_transform([x["text"] or "" for x in rows]) if rows else None
            _index = (rows, vec, matrix)
    return _index


def search_docs(query: str, cik: int | None = None, fund_id: str | None = None,
                k: int = 5) -> list[dict]:
    """Top-k chunks, optionally only one company's or one fund's: [{chunk_id, ..., text}]."""
    import numpy as np
    from sklearn.metrics.pairwise import linear_kernel

    rows, vec, matrix = docs_index()
    if matrix is None or not str(query).strip():
        return []
    k = max(1, min(int(k or 5), MAX_K))
    scores = linear_kernel(vec.transform([str(query)]), matrix).ravel()
    keep = np.ones(len(rows), dtype=bool)
    if cik is not None:
        keep &= np.array([x["cik"] is not None and int(x["cik"]) == int(cik) for x in rows])
    if fund_id:
        keep &= np.array([str(x["fund_id"] or "").lower() == str(fund_id).lower() for x in rows])
    scores = np.where(keep, scores, 0.0)
    best = scores.argsort()[::-1][:k]
    out = []
    for i in best:
        if scores[i] <= 0:
            break
        x = rows[i]
        out.append({"chunk_id": x["chunk_id"], "doc_id": x["doc_id"], "doc_type": x["doc_type"],
                    "cik": x["cik"], "fund_id": x["fund_id"], "fiscal_year": x["fiscal_year"],
                    "section": x["section"], "text": x["text"]})
    return out


def reset_caches() -> None:
    """Forget the matcher, the search index and the prompt's value list (tests swap lakes)."""
    global _matcher, _index
    _matcher = _index = None
    import agent.graph as g
    g._system = None
