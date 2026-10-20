"""The matcher: clean up names, index every alias, rank candidate companies.

Try one name:  uv run python -m matching.matcher "Walt Disney Co"
Point in time: uv run python -m matching.matcher "Walt Disney Co" --as-of 2015-06-30

Methods (evaluate.py compares them on the same queries):
  exact          exact match on normalized names
  fuzzy          rapidfuzz token_set_ratio against every normalized alias
  tfidf          TF-IDF on character 3-5-grams, cosine nearest neighbour
  tfidf_noalias  tfidf, but former names hidden (shows what the aliases add)
  hybrid         tfidf finds 20 companies, rapidfuzz re-scores them,
                 and the matcher abstains when the best score or the margin is low

Every method returns, per query, a ranked list of candidates (one per CIK),
each with a score from 0 to 100.
"""

import argparse
import re
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer

METHODS = ["exact", "fuzzy", "tfidf", "tfidf_noalias", "hybrid"]

# Word swaps applied after lowercasing. Long forms and short forms map to one
# spelling, so "Holdings" and "Hldgs" compare as equal.
SYNONYMS = {
    "incorporated": "inc", "corporation": "corp", "company": "co", "limited": "ltd",
    "intl": "international", "int'l": "international",
    "hldgs": "holdings", "hldg": "holding", "grp": "group",
    "mfg": "manufacturing", "svcs": "services", "svc": "service",
    "natl": "national", "sys": "systems", "cos": "companies", "assoc": "associates",
}
# Legal forms. They are dropped from the END of a name: "Acme Inc" and
# "Acme Corp" are almost always the same business in a GP report.
LEGAL_SUFFIXES = {"inc", "corp", "co", "ltd", "llc", "lp", "llp", "plc",
                  "sa", "ag", "nv", "se", "bv", "gmbh", "and"}
# A ticker looks like "AAPL", "BRK-B" or "BF.A": short, upper case, no spaces.
TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z])?$")


def normalize(name: str) -> str:
    """Turn a company name into a comparable key.

    "The Walt Disney Company" -> "walt disney"
    "AIR PRODUCTS & CHEMICALS INC /DE/" -> "air products and chemicals"
    """
    s = name.lower()
    s = re.sub(r"[/\\][a-z]{2,5}([/\\]|$)", " ", s)  # SEC tags: /DE/, \NEW\, /NJ
    s = s.replace("&", " and ")
    s = re.sub(r"['.]", "", s)  # U.S. -> us, Macy's -> macys (no gap left behind)
    s = re.sub(r"[^a-z0-9]+", " ", s)  # every other symbol becomes a space
    words = [SYNONYMS.get(w, w) for w in s.split()]
    if len(words) > 1 and words[0] == "the":
        words = words[1:]
    while len(words) > 1 and words[-1] in LEGAL_SUFFIXES:  # "acme holdings inc" -> "acme holdings"
        words.pop()
    return " ".join(words)


@dataclass
class Alias:
    """One name a company has gone by, and when."""
    cik: int
    text: str  # as SEC wrote it
    norm: str  # normalize(text)
    kind: str  # "current" or "former"
    valid_from: str  # "YYYY-MM-DD"; "" = unknown, treated as always
    valid_to: str  # "YYYY-MM-DD"; "" = still valid


def build_aliases(universe: list[dict]) -> list[Alias]:
    """Every current and former name, with the dates it was valid."""
    aliases = []
    for c in universe:
        ends = [f["to"][:10] for f in c["former_names"] if f.get("to")]
        # The current name starts when the last former name stopped.
        aliases.append(Alias(c["cik"], c["name"], normalize(c["name"]), "current",
                             max(ends, default=""), ""))
        for f in c["former_names"]:
            aliases.append(Alias(c["cik"], f["name"], normalize(f["name"]), "former",
                                 (f.get("from") or "")[:10], (f.get("to") or "")[:10]))
    return aliases


def decide(cands: list[dict], min_score: float, min_margin: float) -> str:
    """Answer only when the best candidate is good AND clearly ahead of the next one.

    A low score means nothing looks right. A small margin means two companies
    look equally right (e.g. two "First National Bank"s): guessing would be a coin flip.
    """
    if not cands or cands[0]["score"] < min_score:
        return "abstain"
    runner_up = cands[1]["score"] if len(cands) > 1 else 0.0
    return "match" if cands[0]["score"] - runner_up >= min_margin else "abstain"


class Matcher:
    """Holds the alias index and the TF-IDF matrix. Build once, query many times."""

    def __init__(self, universe: list[dict], min_score=90.0, min_margin=3.0):
        self.names = {c["cik"]: c["name"] for c in universe}
        self.aliases = build_aliases(universe)
        self.norms = [a.norm for a in self.aliases]
        self.alias_cik = np.array([a.cik for a in self.aliases])
        self.is_former = np.array([a.kind == "former" for a in self.aliases])
        self.valid_from = np.array([a.valid_from or "0000" for a in self.aliases])
        self.valid_to = np.array([a.valid_to or "9999" for a in self.aliases])
        self.min_score, self.min_margin = min_score, min_margin

        self.exact = {}  # normalized name -> alias positions
        self.by_cik = {}  # CIK -> alias positions
        for i, a in enumerate(self.aliases):
            self.exact.setdefault(a.norm, []).append(i)
            self.by_cik.setdefault(a.cik, []).append(i)
        self.tickers = {}  # "AAPL" -> CIKs
        for c in universe:
            for t in c["tickers"]:
                self.tickers.setdefault(t.upper(), set()).add(c["cik"])

        # Character n-grams survive typos and spacing: "acme" and "acem" still
        # share " ac" and "acm". char_wb keeps n-grams inside word boundaries.
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.tfidf = self.vectorizer.fit_transform(self.norms)  # rows are L2-normalized

    # ---- helpers -------------------------------------------------------------

    def _valid_mask(self, as_of: str | None, use_former=True) -> np.ndarray:
        """True for aliases we may match: right date range, former names allowed."""
        mask = np.ones(len(self.aliases), dtype=bool)
        if not use_former:
            mask &= ~self.is_former
        if as_of:  # ISO dates compare correctly as strings
            mask &= (self.valid_from <= as_of) & (as_of <= self.valid_to)
        return mask

    def _collapse(self, scores: np.ndarray, k: int) -> list[dict]:
        """Alias scores -> the top k companies, each with its best-scoring alias."""
        out, seen = [], set()
        # Look at extra aliases (50 per slot): one company can own several close names.
        top = np.argpartition(-scores, min(len(scores) - 1, 50 * k))[: 50 * k]
        for i in top[np.argsort(-scores[top])]:
            if scores[i] <= 0 or len(out) == k:
                break
            cik = int(self.alias_cik[i])
            if cik not in seen:
                seen.add(cik)
                out.append(self._cand(cik, i, float(scores[i])))
        return out

    def _cand(self, cik: int, alias_i: int | None, score: float, kind=None) -> dict:
        a = self.aliases[alias_i] if alias_i is not None else None
        return {"cik": cik, "name": self.names[cik], "alias": a.text if a else None,
                "alias_kind": kind or a.kind, "score": round(score, 1)}

    def _ticker_hit(self, query: str) -> list[dict] | None:
        """Tickers are codes, not names: look them up exactly, never fuzzily."""
        q = query.strip()
        if TICKER_RE.match(q) and q in self.tickers:
            return [self._cand(c, None, 100.0, "ticker") | {"alias": q}
                    for c in sorted(self.tickers[q])]
        return None

    # ---- the five methods: each takes a batch of normalized queries ---------

    def _exact(self, norms, k, mask):
        out = []
        for n in norms:
            hits = [i for i in self.exact.get(n, []) if mask[i]]
            scores = np.zeros(len(self.aliases))
            scores[hits] = 100.0
            out.append(self._collapse(scores, k))
        return out

    def _fuzzy(self, norms, k, mask):
        # cdist scores every query against every alias (C++ speed, all CPU cores).
        m = process.cdist(norms, self.norms, scorer=fuzz.token_set_ratio, workers=-1)
        return [self._collapse(np.where(mask, row, 0.0), k) for row in m]

    def _tfidf_scores(self, norms, mask) -> np.ndarray:
        """Cosine similarity x 100. Rows are L2-normalized, so a dot product is the cosine."""
        sims = (self.vectorizer.transform(norms) @ self.tfidf.T).toarray() * 100
        return np.where(mask, sims, 0.0)

    def _tfidf(self, norms, k, mask):
        return [self._collapse(row, k) for row in self._tfidf_scores(norms, mask)]

    def _hybrid(self, norms, k, mask):
        """Cheap recall first (TF-IDF top 20 companies), careful scoring second."""
        out = []
        for n, row in zip(norms, self._tfidf_scores(norms, mask)):
            shortlist = {c["cik"] for c in self._collapse(row, 20)}
            cands = []
            for cik in shortlist:  # re-score every valid alias of each shortlisted company
                scored = [(rescore(n, self.norms[i]), i) for i in self.by_cik[cik] if mask[i]]
                score, best = max(scored)
                cands.append(self._cand(cik, best, score))
            out.append(sorted(cands, key=lambda c: -c["score"])[:k])
        return out

    # ---- public API ----------------------------------------------------------

    def search_many(self, queries: list[str], method="hybrid", k=5,
                    as_of: str | None = None) -> list[list[dict]]:
        """Rank candidates for many queries at once (TF-IDF is faster in batches)."""
        mask = self._valid_mask(as_of, use_former=method != "tfidf_noalias")
        run = {"exact": self._exact, "fuzzy": self._fuzzy, "tfidf": self._tfidf,
               "tfidf_noalias": self._tfidf, "hybrid": self._hybrid}[method]
        results = [None] * len(queries)
        names_todo = []
        for i, q in enumerate(queries):
            hit = self._ticker_hit(q)
            if hit:
                results[i] = hit[:k]
            else:
                names_todo.append(i)
        for start in range(0, len(names_todo), 256):  # chunks keep the dense matrix small
            chunk = names_todo[start:start + 256]
            ranked = run([normalize(queries[i]) for i in chunk], k, mask)
            for i, cands in zip(chunk, ranked):
                results[i] = cands
        return results

    def search(self, query: str, method="hybrid", k=5, as_of=None) -> list[dict]:
        return self.search_many([query], method, k, as_of)[0]

    def match(self, query: str, k=5, as_of=None) -> dict:
        """The hybrid answer plus its decision: what the API returns."""
        cands = self.search(query, "hybrid", k, as_of)
        return {"query": query, "decision": decide(cands, self.min_score, self.min_margin),
                "candidates": cands}


def rescore(a: str, b: str) -> float:
    """Average of two rapidfuzz scores, 0-100.

    token_set_ratio forgives missing words ("disney" vs "walt disney" = 100),
    which also makes "apple" tie with "apple hospitality reit". token_sort_ratio
    punishes missing words. The average forgives a little, not completely.
    """
    return (fuzz.token_set_ratio(a, b) + fuzz.token_sort_ratio(a, b)) / 2


def load_matcher() -> Matcher:
    """Build the matcher from the cached SEC files, with thresholds tuned on dev if present."""
    import json

    from matching.common import THRESHOLDS, load_universe

    t = json.loads(THRESHOLDS.read_text()) if THRESHOLDS.exists() else {}
    return Matcher(load_universe(), t.get("min_score", 90.0), t.get("min_margin", 3.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--as-of", help="YYYY-MM-DD: only names valid on that date")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    m = load_matcher()
    print(f"normalized: {normalize(args.name)!r}")
    for method in METHODS:
        top = m.search(args.name, method, args.k, args.as_of)
        print(f"\n{method}")
        for c in top:
            print(f"  {c['score']:6.1f}  {c['cik']:>8}  {c['name']}  (via {c['alias_kind']}: "
                  f"{c['alias']})")
    print(f"\nhybrid decision: {m.match(args.name, args.k, args.as_of)['decision']}")
