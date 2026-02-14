#!/usr/bin/env python3
"""Build an expanded evaluation dataset with multi-relevance labels."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "eval-mixed"
TARGET = ROOT / "data" / "eval-expanded"

MULTI_QUERIES = [
    {
        "query": "Which documents in this corpus are IETF RFCs?",
        "relevant": [
            {"id": "rfc822", "score": 1.0},
            {"id": "rfc5322", "score": 1.0},
            {"id": "rfc8446", "score": 1.0},
            {"id": "rfc9110", "score": 1.0},
            {"id": "rfc9112", "score": 1.0},
            {"id": "rfc9113", "score": 1.0},
            {"id": "rfc9595", "score": 1.0},
        ],
    },
    {
        "query": "Which documents are literary narratives rather than technical standards?",
        "relevant": [
            {"id": "a_scandal_in_bohemia", "score": 1.0},
            {"id": "the_cask_of_amontillado", "score": 1.0},
            {"id": "the_gift_of_the_magi", "score": 1.0},
            {"id": "the_yellow_wallpaper", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs define email message syntax and header formats?",
        "relevant": [
            {"id": "rfc5322", "score": 2.0},
            {"id": "rfc822", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs form the HTTP/1.1 and HTTP/2 core set?",
        "relevant": [
            {"id": "rfc9110", "score": 2.0},
            {"id": "rfc9112", "score": 1.5},
            {"id": "rfc9113", "score": 1.5},
        ],
    },
    {
        "query": "Which documents focus on transport or application protocol security guidance?",
        "relevant": [
            {"id": "rfc8446", "score": 2.0},
            {"id": "rfc9113", "score": 1.0},
            {"id": "rfc9595", "score": 1.0},
        ],
    },
    {
        "query": "Which documents are historical speeches or literary classics from before 1920?",
        "relevant": [
            {"id": "gettysburg_address", "score": 2.0},
            {"id": "the_gift_of_the_magi", "score": 1.0},
            {"id": "the_cask_of_amontillado", "score": 1.0},
            {"id": "a_scandal_in_bohemia", "score": 1.0},
        ],
    },
    {
        "query": "Which documents include normative protocol language such as MUST and SHOULD?",
        "relevant": [
            {"id": "rfc8446", "score": 1.0},
            {"id": "rfc9110", "score": 1.0},
            {"id": "rfc9112", "score": 1.0},
            {"id": "rfc9113", "score": 1.0},
            {"id": "rfc9595", "score": 1.0},
            {"id": "rfc5322", "score": 1.0},
            {"id": "rfc822", "score": 1.0},
        ],
    },
    {
        "query": "Which documents mention message headers directly?",
        "relevant": [
            {"id": "rfc5322", "score": 2.0},
            {"id": "rfc822", "score": 2.0},
            {"id": "rfc9110", "score": 1.0},
            {"id": "rfc9112", "score": 1.0},
        ],
    },
    {
        "query": "Which documents are centered on Sherlock Holmes or detective reasoning?",
        "relevant": [
            {"id": "a_scandal_in_bohemia", "score": 2.0},
        ],
    },
    {
        "query": "Which documents are centered on domestic life and household economics?",
        "relevant": [
            {"id": "the_gift_of_the_magi", "score": 2.0},
            {"id": "the_yellow_wallpaper", "score": 1.0},
        ],
    },
    {
        "query": "Which documents discuss HTTP semantics, caching, or method behavior?",
        "relevant": [
            {"id": "rfc9110", "score": 2.0},
            {"id": "rfc9112", "score": 1.0},
            {"id": "rfc9113", "score": 1.0},
        ],
    },
    {
        "query": "Which documents are likely to contain ABNF grammar definitions?",
        "relevant": [
            {"id": "rfc5322", "score": 1.5},
            {"id": "rfc822", "score": 1.5},
            {"id": "rfc9110", "score": 1.0},
            {"id": "rfc9112", "score": 1.0},
            {"id": "rfc9113", "score": 1.0},
            {"id": "rfc8446", "score": 1.0},
            {"id": "rfc9595", "score": 1.0},
        ],
    },
    {
        "query": "Which documents are primarily prose fiction with named characters?",
        "relevant": [
            {"id": "a_scandal_in_bohemia", "score": 1.0},
            {"id": "the_cask_of_amontillado", "score": 1.0},
            {"id": "the_gift_of_the_magi", "score": 1.0},
            {"id": "the_yellow_wallpaper", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs are in the 9000 series?",
        "relevant": [
            {"id": "rfc9110", "score": 1.0},
            {"id": "rfc9112", "score": 1.0},
            {"id": "rfc9113", "score": 1.0},
            {"id": "rfc9595", "score": 1.0},
        ],
    },
    {
        "query": "Which documents contain protocol state machine behavior or sequencing rules?",
        "relevant": [
            {"id": "rfc8446", "score": 1.5},
            {"id": "rfc9112", "score": 1.0},
            {"id": "rfc9113", "score": 1.0},
            {"id": "rfc9595", "score": 1.0},
        ],
    },
    {
        "query": "Which documents are likely to discuss interoperability and conformance?",
        "relevant": [
            {"id": "rfc8446", "score": 1.0},
            {"id": "rfc9110", "score": 1.0},
            {"id": "rfc9112", "score": 1.0},
            {"id": "rfc9113", "score": 1.0},
            {"id": "rfc9595", "score": 1.0},
            {"id": "rfc5322", "score": 1.0},
        ],
    },
    {
        "query": "Which documents focus on personal identity, voice, or self-expression?",
        "relevant": [
            {"id": "the_yellow_wallpaper", "score": 2.0},
            {"id": "a_scandal_in_bohemia", "score": 1.0},
            {"id": "the_gift_of_the_magi", "score": 1.0},
        ],
    },
    {
        "query": "Which documents mention bohemia or bohemian contexts?",
        "relevant": [
            {"id": "a_scandal_in_bohemia", "score": 2.0},
        ],
    },
    {
        "query": "Which documents are likely to define wire-level framing details?",
        "relevant": [
            {"id": "rfc9112", "score": 1.5},
            {"id": "rfc9113", "score": 1.5},
            {"id": "rfc8446", "score": 1.0},
        ],
    },
    {
        "query": "Which documents include references to ceremony, rituals, or performative acts?",
        "relevant": [
            {"id": "the_cask_of_amontillado", "score": 2.0},
            {"id": "a_scandal_in_bohemia", "score": 1.0},
        ],
    },
    {
        "query": "Which documents are suitable sources for questions about US Civil War-era rhetoric?",
        "relevant": [
            {"id": "gettysburg_address", "score": 2.0},
        ],
    },
    {
        "query": "Which documents are likely to mention headers named From, To, and Date?",
        "relevant": [
            {"id": "rfc5322", "score": 2.0},
            {"id": "rfc822", "score": 2.0},
        ],
    },
    {
        "query": "Which documents have most content about internet architecture rather than narrative plot?",
        "relevant": [
            {"id": "rfc822", "score": 1.0},
            {"id": "rfc5322", "score": 1.0},
            {"id": "rfc8446", "score": 1.0},
            {"id": "rfc9110", "score": 1.0},
            {"id": "rfc9112", "score": 1.0},
            {"id": "rfc9113", "score": 1.0},
            {"id": "rfc9595", "score": 1.0},
        ],
    },
    {
        "query": "Which documents are likely to discuss connection management or session behavior?",
        "relevant": [
            {"id": "rfc9112", "score": 1.5},
            {"id": "rfc9113", "score": 1.5},
            {"id": "rfc8446", "score": 1.0},
        ],
    },
    {
        "query": "Which documents have strong focus on human relationships and emotional tradeoffs?",
        "relevant": [
            {"id": "the_gift_of_the_magi", "score": 2.0},
            {"id": "the_yellow_wallpaper", "score": 1.0},
            {"id": "a_scandal_in_bohemia", "score": 1.0},
        ],
    },
    {
        "query": "Which documents involve formal sections titled Security Considerations?",
        "relevant": [
            {"id": "rfc5322", "score": 1.0},
            {"id": "rfc8446", "score": 1.0},
            {"id": "rfc9110", "score": 1.0},
            {"id": "rfc9112", "score": 1.0},
            {"id": "rfc9113", "score": 1.0},
            {"id": "rfc9595", "score": 1.0},
        ],
    },
    {
        "query": "Which documents are about standards evolution or obsoleting prior RFCs?",
        "relevant": [
            {"id": "rfc5322", "score": 1.0},
            {"id": "rfc9110", "score": 1.0},
            {"id": "rfc9112", "score": 1.0},
            {"id": "rfc9113", "score": 1.0},
        ],
    },
    {
        "query": "Which documents are ideal for testing retrieval over very long technical texts?",
        "relevant": [
            {"id": "rfc9110", "score": 1.0},
            {"id": "rfc9112", "score": 1.0},
            {"id": "rfc9113", "score": 1.0},
            {"id": "rfc8446", "score": 1.0},
            {"id": "rfc9595", "score": 1.0},
            {"id": "a_scandal_in_bohemia", "score": 1.0},
        ],
    },
]


def _read_source_queries() -> list[dict[str, object]]:
    queries_path = SOURCE / "queries.jsonl"
    rows: list[dict[str, object]] = []
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "relevant_ids" in row:
            rows.append(row)
    return rows


def main() -> None:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    (TARGET / "documents").mkdir(parents=True, exist_ok=True)

    # Copy document corpus
    for source_file in sorted((SOURCE / "documents").glob("*")):
        if source_file.is_file():
            shutil.copy2(source_file, TARGET / "documents" / source_file.name)

    # Preserve v1 query set, then append multi-relevance v2 additions.
    queries = _read_source_queries()
    for item in MULTI_QUERIES:
        queries.append(item)

    queries_path = TARGET / "queries.jsonl"
    with queries_path.open("w", encoding="utf-8") as handle:
        for row in queries:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(queries)} queries to {queries_path}")


if __name__ == "__main__":
    main()
