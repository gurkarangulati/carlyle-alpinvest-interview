"""The three tools the Q&A agent may call, as plain Python functions.

    search_filing(query, k)  TF-IDF search over ~1,500-character chunks of one filing
    get_financials()         numbers from the extraction graph (extracted, NOT XBRL)
    calculate(expression)    safe arithmetic, parsed with `ast` (never eval)

XBRL truth is never a tool: it is the answer key the agent is scored against.

The tools work on one filing at a time. `use_texts()` hands them the filing texts,
so a script can load the big jsonl files once and share them across threads.
"""

import ast
import json
import operator
import threading

from common import OUT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

QA_SECTIONS = {"section_1": "item1", "section_1A": "item1a", "section_7": "item7",
               "section_8": "item8", "section_15": "item15"}
CHUNK = 1_500
MAX_K = 8

_texts: dict[tuple[int, int], dict[str, str]] = {}
_index: dict[tuple[int, int], tuple] = {}
_lock = threading.Lock()


def use_texts(texts: dict) -> None:
    """Register filing texts: {(cik, year): {section: text}}."""
    _texts.update(texts)


# ---------- chunks and search ----------

def make_chunks(sections: dict[str, str]) -> dict[str, str]:
    """{chunk_id: text}. The id says where it came from: "item7-012" = Item 7, 13th chunk."""
    chunks = {}
    for section, short in QA_SECTIONS.items():
        text = sections.get(section, "")
        for n, start in enumerate(range(0, len(text), CHUNK)):
            chunks[f"{short}-{n:03d}"] = text[start:start + CHUNK]
    return chunks


def filing_index(cik: int, year: int) -> tuple:
    """(chunk ids, chunk texts, vectorizer, matrix) for one filing, built once.

    TF-IDF scores a chunk high when it shares the query's words, and more so for
    words that are rare in this filing. Bigrams catch phrases like "net sales".
    """
    key = (cik, year)
    with _lock:
        if key not in _index:
            chunks = make_chunks(_texts.get(key, {}))
            ids, docs = list(chunks), list(chunks.values())
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            matrix = vec.fit_transform(docs) if docs else None
            _index[key] = (ids, docs, vec, matrix)
    return _index[key]


def search_filing(cik: int, year: int, query: str, k: int = 5) -> list[dict]:
    """Top-k chunks for the query: [{"chunk_id", "text"}], best first."""
    ids, docs, vec, matrix = filing_index(cik, year)
    if matrix is None or not str(query).strip():
        return []
    k = max(1, min(int(k), MAX_K))
    scores = linear_kernel(vec.transform([query]), matrix).ravel()
    best = scores.argsort()[::-1][:k]
    return [{"chunk_id": ids[i], "text": docs[i]} for i in best if scores[i] > 0]


def chunk_text(cik: int, year: int, chunk_id: str) -> str | None:
    ids, docs, _, _ = filing_index(cik, year)
    return dict(zip(ids, docs)).get(chunk_id)


# ---------- get_financials ----------

def get_financials(cik: int, year: int) -> dict:
    """The extraction graph's numbers for this filing, if they passed its checks.

    Read from the extraction runs' saved outputs (extraction/out/graph/ from a full
    run, else extraction/out/smoke/graph/). If neither has this filing, run the
    extraction graph now. Values that failed the extraction checks are withheld.
    """
    label = "extracted from this filing's text by the extraction graph; NOT XBRL"
    for folder in (OUT / "graph", OUT / "smoke" / "graph"):
        path = folder / f"{cik}_{year}.json"
        if path.exists():
            state = json.loads(path.read_text())
            break
    else:
        state = _run_extraction(cik, year)
    if state.get("status") != "done":
        return {"source": label, "status": state.get("status"),
                "values": {}, "note": "extraction failed its checks; use search_filing"}
    values = {f: v for f, v in (state.get("values") or {}).items() if v is not None}
    extracted = state.get("extracted") or {}
    return {"source": label, "status": "done", "units": "US dollars",
            "fiscal_year_end": extracted.get("fiscal_year_end"), "values": values,
            "evidence": {f: extracted.get(f"{f}_evidence") for f in values}}


def _run_extraction(cik: int, year: int) -> dict:
    from analysis import sector_reference  # imported here: only needed on a cache miss
    from common import load_companies
    from graph import build_graph

    row = load_companies().set_index(["cik", "year"]).loc[(cik, year)]
    return build_graph().invoke({
        "cik": cik, "year": year, "name": row["name"], "sector": row["sector"],
        "sections": _texts.get((cik, year), {}),
        "sector_ref": sector_reference().get(row["sector"], {})})


# ---------- calculate ----------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression: str) -> float:
    """Evaluate + - * / and brackets on plain numbers. Anything else raises ValueError.

    We walk the parsed expression ourselves and allow only numbers and these
    operators, so text like "__import__('os')" can never run.
    """
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"not allowed: {type(node).__name__}")

    if len(expression) > 200:
        raise ValueError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError("not an arithmetic expression (write numbers without commas)") from e
    try:
        return walk(tree)
    except ZeroDivisionError as e:
        raise ValueError("division by zero") from e
