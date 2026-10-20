"""Find company names inside documents, then link each one to a CIK.

Demo:  uv run python -m matching.mentions          (20 business sections)
       uv run python -m matching.mentions --docs 5

Step 1, find: a plain regex. A mention is a run of Capitalized Words that ends in a
legal suffix: Inc, Corp, Corporation, LLC, Ltd, plc, Co., Holdings, L.P.
Step 2, link: each mention goes through the hybrid matcher (match or abstain).

This is a QUALITATIVE demo, not a scored result: there are no labels for mentions.
To score it, hand-label ~100 mentions (right span? right CIK?) and report
precision and recall of the finder and accuracy of the linker separately.
"""

import argparse
import re

import pandas as pd

from matching.common import DATA, RESULTS
from matching.matcher import Matcher, decide

# One capitalized word ("Acme", "Coca-Cola", "AT&T"), or initials ("U.S.").
CAP = r"(?:(?:[A-Z]\.){2,}|[A-Z][\w'&\-]*)"
# Words may be joined by spaces or by small words: "Bank of America", "Procter & Gamble".
NAME = rf"{CAP}(?:\s+(?:(?:&|and|of|the|de)\s+)?{CAP})*"
SUFFIX = r"(?:Inc\.?|Corp\.?|Corporation|LLC|L\.L\.C\.|Ltd\.?|plc|PLC|Co\.|Holdings|L\.P\.)"
MENTION = re.compile(rf"\b{NAME},?\s+{SUFFIX}(?!\w)")
# Capitalized words that start sentences or headings, not company names.
STOP_START = {"In", "On", "As", "At", "By", "For", "From", "With", "Our", "We", "Its",
              "Under", "Through", "And", "Of", "See", "Since", "During", "Pursuant",
              "Item", "Business", "Overview", "General", "Background", "Description",
              "Introduction", "Company", "History", "Corporate", "The"}
MAX_TEXT = 20_000  # characters; longer documents should be chunked by the caller


def is_heading(words: list[str]) -> bool:
    """An ALL-CAPS first word before mixed-case words is a heading, not part of the name."""
    return words[0].isupper() and len(words[0]) > 1 and not all(w.isupper() for w in words)


def find_mentions(text: str) -> list[dict]:
    """Every regex hit, with leading sentence words trimmed off. Duplicates kept once."""
    out, seen = [], set()
    for m in MENTION.finditer(text):
        words, start = m.group().split(), m.start()
        # Trim heading and sentence words: "BUSINESS General Overview Acme Inc" -> "Acme Inc".
        while len(words) > 2 and (words[0].capitalize() in STOP_START or is_heading(words)):
            start += len(words[0]) + text[start + len(words[0]):].index(words[1])
            words = words[1:]
        mention = text[start:m.end()]
        if mention not in seen:
            seen.add(mention)
            out.append({"mention": mention, "start": start, "end": m.end()})
    return out


def link_mentions(matcher: Matcher, text: str, as_of: str | None = None) -> list[dict]:
    """Find mentions, then match each one with the hybrid method."""
    found = find_mentions(text)
    ranked = matcher.search_many([f["mention"] for f in found], "hybrid", 3, as_of)
    for f, cands in zip(found, ranked):
        f["decision"] = decide(cands, matcher.min_score, matcher.min_margin)
        f["match"] = cands[0] if cands and f["decision"] == "match" else None
        f["candidates"] = cands
    return found


def main():
    from matching.matcher import load_matcher

    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=20, help="how many business sections")
    args = parser.parse_args()

    docs = (pd.read_csv(DATA / "companies.csv").drop_duplicates("cik")
            .sample(args.docs, random_state=14))
    matcher = load_matcher()
    rows = []
    for d in docs.itertuples():
        for f in link_mentions(matcher, d.business):
            top = f["candidates"][0] if f["candidates"] else {}
            rows.append({"doc_cik": d.cik, "doc_name": d.name, "mention": f["mention"],
                         "start": f["start"], "decision": f["decision"],
                         "match_cik": top.get("cik"), "match_name": top.get("name"),
                         "score": top.get("score"),
                         "is_filer": f["decision"] == "match" and top.get("cik") == d.cik})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "mentions_demo.csv", index=False)

    print(f"{len(docs)} business sections, {len(out)} mentions found, "
          f"{(out['decision'] == 'match').mean():.0%} linked, the rest abstained")
    # Most Item 1 sections name the filer itself in the first lines. That gives a weak
    # sanity check (not a score): how often do we find and link the filer's own name?
    filer_found = out.groupby("doc_cik")["is_filer"].any().reindex(docs["cik"], fill_value=False)
    print(f"documents where the filer's own name was found and linked to its CIK: "
          f"{filer_found.sum()} of {len(docs)}")
    with pd.option_context("display.width", 160, "display.max_colwidth", 45):
        print(out[["doc_name", "mention", "decision", "match_name", "score"]].head(30)
              .to_string(index=False))
    print(f"saved {RESULTS / 'mentions_demo.csv'}")


if __name__ == "__main__":
    main()
