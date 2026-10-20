"""One query interface, two engines. The agent never knows which one it is talking to.

    LAKE_BACKEND=local   LocalDuckDB: data/lake/lake.duckdb on this laptop (default)
    LAKE_BACKEND=athena  Athena: the same tables as Parquet on S3, catalogued in Glue

Both take the same SQL. The agent writes Presto/Trino SQL (Athena's dialect); the
SQL guard in tools.py parses it with sqlglot as Trino and re-writes it for the engine
(`dialect`), so things like approx_percentile or date functions are translated for
DuckDB. Keep queries to plain Trino SQL: SELECT, JOIN, GROUP BY, window functions,
CASE, CAST, COALESCE, NULLIF. Engine-only features (DuckDB's QUALIFY or read_csv,
Athena's UNLOAD) are not allowed by the guard anyway.

Each engine returns QueryResult(query_id, columns, rows, bytes_scanned). The query id
is what the agent cites as `sql:<query_id>`.
"""

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agent import ML
from agent.schema import ALLOWED_TABLES

LAKE_DB = ML / "data" / "lake" / "lake.duckdb"
QUERY_TIMEOUT_S = 30
MAX_ROWS = 200


@dataclass
class QueryResult:
    query_id: str
    columns: list[str]
    rows: list[tuple]
    bytes_scanned: int = 0
    seconds: float = 0.0
    truncated: bool = False
    meta: dict = field(default_factory=dict)

    def records(self) -> list[dict]:
        return [dict(zip(self.columns, r)) for r in self.rows]


class LocalDuckDB:
    """The lake file, opened READ ONLY, behind views with the plain table names.

    We attach the file read-only to an in-memory database and create one view per
    allowlisted table in the in-memory `main` schema. That way:
      - the SQL is identical to Athena's (`FROM companies`, no schema prefix), whether
        the lake writer put tables in `main` or in `silver`/`gold` schemas;
      - nothing can be written to the lake file;
      - external access (reading files, http) is switched off after setup, and the
        configuration is locked so a query cannot switch it back on.
    """

    dialect = "duckdb"

    def __init__(self, path: Path | str | None = None):
        import duckdb  # imported here: the cloud image does not need it

        self.path = Path(path or os.environ.get("LAKE_DB") or LAKE_DB)
        if not self.path.exists():
            raise FileNotFoundError(f"lake not found: {self.path} (build it with the lake module)")
        self.con = duckdb.connect(":memory:")
        self.con.execute(f"ATTACH '{self.path}' AS lake (READ_ONLY)")
        found = self.con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_catalog = 'lake'").fetchall()
        where = {}
        for schema, table in found:  # prefer gold, then silver, then main if a name repeats
            rank = {"gold": 0, "silver": 1, "main": 2}.get(schema, 3)
            if table in ALLOWED_TABLES and rank < where.get(table, (9, ""))[0]:
                where[table] = (rank, schema)
        for table, (_, schema) in where.items():
            self.con.execute(f'CREATE VIEW "{table}" AS SELECT * FROM lake."{schema}"."{table}"')
        self.tables = sorted(where)
        self.con.execute("SET enable_external_access = false")
        self.con.execute("SET lock_configuration = true")
        self._lock = threading.Lock()

    def query(self, sql: str, max_rows: int | None = MAX_ROWS) -> QueryResult:
        t0 = time.perf_counter()
        with self._lock:
            cur = self.con.cursor()
        # A timer interrupts a runaway query (a big cross join) instead of hanging the agent.
        timer = threading.Timer(QUERY_TIMEOUT_S, cur.interrupt)
        timer.start()
        try:
            cur.execute(sql)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall() if max_rows is None else cur.fetchmany(max_rows + 1)
        finally:
            timer.cancel()
            cur.close()
        truncated = max_rows is not None and len(rows) > max_rows
        return QueryResult(f"local-{uuid.uuid4().hex[:10]}", columns, rows[:max_rows],
                           0, time.perf_counter() - t0, truncated)


class Athena:
    """Athena: start the query, poll until it finishes, page through the results.

    Configuration comes from the environment (the AgentCore Runtime sets these):
      ATHENA_DATABASE   the Glue database holding the lake tables
      ATHENA_WORKGROUP  the workgroup (it caps bytes scanned per query)
      ATHENA_OUTPUT     s3://bucket/prefix/ for result files
    """

    dialect = "trino"   # Athena engine v3 is Trino

    def __init__(self):
        import boto3
        from botocore.config import Config

        self.database = os.environ["ATHENA_DATABASE"]
        self.workgroup = os.environ.get("ATHENA_WORKGROUP", "primary")
        self.output = os.environ.get("ATHENA_OUTPUT")
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        self.client = boto3.client("athena", region_name=region,
                                   config=Config(retries={"total_max_attempts": 1}))
        self.tables = sorted(ALLOWED_TABLES)

    def _call(self, fn, **kwargs):
        """Retry throttling and 5xx only (same rule as llm.py), up to 4 attempts."""
        import llm

        for attempt in range(4):
            try:
                return fn(**kwargs)
            except Exception as e:
                if attempt == 3 or not llm.transient(e):
                    raise
                time.sleep(2 ** attempt * 0.5)

    def query(self, sql: str, max_rows: int | None = MAX_ROWS) -> QueryResult:
        t0 = time.perf_counter()
        start = {"QueryString": sql, "WorkGroup": self.workgroup,
                 "QueryExecutionContext": {"Database": self.database}}
        if self.output:
            start["ResultConfiguration"] = {"OutputLocation": self.output}
        qid = self._call(self.client.start_query_execution, **start)["QueryExecutionId"]

        delay = 0.25
        while True:
            ex = self._call(self.client.get_query_execution, QueryExecutionId=qid)["QueryExecution"]
            state = ex["Status"]["State"]
            if state == "SUCCEEDED":
                break
            if state in ("FAILED", "CANCELLED"):
                reason = ex["Status"].get("StateChangeReason", state)
                raise RuntimeError(f"Athena query {state.lower()}: {reason}")
            if time.perf_counter() - t0 > QUERY_TIMEOUT_S:
                self._call(self.client.stop_query_execution, QueryExecutionId=qid)
                raise TimeoutError(f"Athena query took over {QUERY_TIMEOUT_S}s; stopped")
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)

        columns, types, rows, token, first = [], [], [], None, True
        while True:
            kwargs = {"QueryExecutionId": qid, "MaxResults": 1000}
            if token:
                kwargs["NextToken"] = token
            page = self._call(self.client.get_query_results, **kwargs)
            if first:
                info = page["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
                columns = [c["Name"] for c in info]
                types = [c["Type"] for c in info]
            data = page["ResultSet"]["Rows"]
            if first:
                data = data[1:]   # the first row of the first page is the header
                first = False
            rows += [tuple(_convert(cell.get("VarCharValue"), t)
                           for cell, t in zip(r["Data"], types)) for r in data]
            token = page.get("NextToken")
            if not token or (max_rows is not None and len(rows) > max_rows):
                break
        truncated = max_rows is not None and len(rows) > max_rows
        scanned = ex.get("Statistics", {}).get("DataScannedInBytes", 0)
        return QueryResult(qid, columns, rows[:max_rows] if max_rows is not None else rows,
                           scanned, time.perf_counter() - t0, truncated)


def _convert(value: str | None, athena_type: str):
    """Athena returns every cell as a string; turn numbers back into numbers."""
    if value is None:
        return None
    t = athena_type.lower()
    try:
        if t in ("tinyint", "smallint", "integer", "int", "bigint"):
            return int(value)
        if t in ("double", "float", "real") or t.startswith("decimal"):
            return float(value)
        if t == "boolean":
            return value.lower() == "true"
    except ValueError:
        return value
    return value


_backend = None
_backend_lock = threading.Lock()


def get_backend():
    """The engine for this process, chosen once by LAKE_BACKEND (default local)."""
    global _backend
    with _backend_lock:
        if _backend is None:
            kind = os.environ.get("LAKE_BACKEND", "local").lower()
            if kind == "local":
                _backend = LocalDuckDB()
            elif kind == "athena":
                _backend = Athena()
            else:
                raise ValueError(f"LAKE_BACKEND must be local or athena, not {kind!r}")
    return _backend


def set_backend(backend) -> None:
    """Swap the engine (tests use a fixture lake)."""
    global _backend
    with _backend_lock:
        _backend = backend
