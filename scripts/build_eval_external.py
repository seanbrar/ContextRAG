#!/usr/bin/env python3
"""Build the external RFC holdout dataset used for generalization checks."""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "eval-external"

RFC_URLS = {
    "rfc2119": "https://www.rfc-editor.org/rfc/rfc2119.txt",
    "rfc3986": "https://www.rfc-editor.org/rfc/rfc3986.txt",
    "rfc6265": "https://www.rfc-editor.org/rfc/rfc6265.txt",
    "rfc6750": "https://www.rfc-editor.org/rfc/rfc6750.txt",
    "rfc7230": "https://www.rfc-editor.org/rfc/rfc7230.txt",
    "rfc7231": "https://www.rfc-editor.org/rfc/rfc7231.txt",
    "rfc7540": "https://www.rfc-editor.org/rfc/rfc7540.txt",
    "rfc8259": "https://www.rfc-editor.org/rfc/rfc8259.txt",
    "rfc8441": "https://www.rfc-editor.org/rfc/rfc8441.txt",
    "rfc9114": "https://www.rfc-editor.org/rfc/rfc9114.txt",
}

QUERIES = [
    {"query": "Which RFC defines the requirement keywords MUST, SHOULD, and MAY for standards documents?", "relevant_ids": ["rfc2119"]},
    {"query": "Which RFC defines generic URI components such as scheme, authority, path, query, and fragment?", "relevant_ids": ["rfc3986"]},
    {"query": "Which RFC defines HTTP state management with Cookie and Set-Cookie headers?", "relevant_ids": ["rfc6265"]},
    {"query": "Which RFC defines OAuth 2.0 Bearer Token usage in HTTP Authorization headers?", "relevant_ids": ["rfc6750"]},
    {"query": "Which RFC defines HTTP/1.1 message syntax and routing rules?", "relevant_ids": ["rfc7230"]},
    {"query": "Which RFC defines HTTP/1.1 method semantics and status codes?", "relevant_ids": ["rfc7231"]},
    {"query": "Which RFC defines HTTP/2 framing, streams, and multiplexing?", "relevant_ids": ["rfc7540"]},
    {"query": "Which RFC defines JSON syntax and interoperability constraints?", "relevant_ids": ["rfc8259"]},
    {"query": "Which RFC defines bootstrapping WebSockets with extended CONNECT over HTTP/2?", "relevant_ids": ["rfc8441"]},
    {"query": "Which RFC defines HTTP/3 over QUIC and obsoletes RFC 7540?", "relevant_ids": ["rfc9114"]},
    {"query": "Which RFC focuses on Set-Cookie attributes like Secure, HttpOnly, and SameSite rather than bearer tokens?", "relevant_ids": ["rfc6265"]},
    {"query": "Which RFC defines bearer-token error codes such as invalid_token and insufficient_scope?", "relevant_ids": ["rfc6750"]},
    {"query": "Which RFC defines request-target forms such as origin-form and absolute-form?", "relevant_ids": ["rfc7230"]},
    {"query": "Which RFC is the primary source for semantics of GET, HEAD, POST, PUT, and DELETE?", "relevant_ids": ["rfc7231"]},
    {"query": "Which RFC introduces HPACK for HTTP header compression?", "relevant_ids": ["rfc7540"]},
    {"query": "Which RFC defines JSON number grammar and parser behavior?", "relevant_ids": ["rfc8259"]},
    {"query": "Which RFC adds the :protocol pseudo-header for WebSockets over HTTP/2?", "relevant_ids": ["rfc8441"]},
    {"query": "Which RFC maps HTTP semantics to QUIC transport behavior for HTTP/3?", "relevant_ids": ["rfc9114"]},
    {"query": "Which RFC should be cited for user-agent cookie storage and return behavior?", "relevant_ids": ["rfc6265"]},
    {"query": "Which RFC defines URI normalization and relative reference resolution?", "relevant_ids": ["rfc3986"]},
    {
        "query": "Which RFCs define HTTP protocol evolution from HTTP/1.1 text messaging to HTTP/2 and HTTP/3?",
        "relevant": [
            {"id": "rfc7230", "score": 1.5},
            {"id": "rfc7231", "score": 1.0},
            {"id": "rfc7540", "score": 1.5},
            {"id": "rfc9114", "score": 2.0},
        ],
    },
    {
        "query": "Which RFCs should be consulted for HTTP authentication and session-state mechanisms?",
        "relevant": [
            {"id": "rfc6750", "score": 2.0},
            {"id": "rfc6265", "score": 2.0},
            {"id": "rfc7231", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs are most relevant for URI and API identifier syntax used on the web?",
        "relevant": [
            {"id": "rfc3986", "score": 2.0},
            {"id": "rfc6750", "score": 1.0},
            {"id": "rfc7231", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs define HTTP semantics versus wire-format and framing details?",
        "relevant": [
            {"id": "rfc7231", "score": 2.0},
            {"id": "rfc7230", "score": 1.5},
            {"id": "rfc7540", "score": 1.0},
            {"id": "rfc9114", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs are likely to contain ABNF-like grammar rules for internet syntax?",
        "relevant": [
            {"id": "rfc3986", "score": 2.0},
            {"id": "rfc7230", "score": 1.5},
            {"id": "rfc7231", "score": 1.0},
            {"id": "rfc8259", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs cover transition to binary or QUIC-based HTTP transport compared with HTTP/1.1?",
        "relevant": [
            {"id": "rfc7540", "score": 2.0},
            {"id": "rfc9114", "score": 2.0},
            {"id": "rfc8441", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs define pseudo-header behavior in modern HTTP versions?",
        "relevant": [
            {"id": "rfc7540", "score": 2.0},
            {"id": "rfc8441", "score": 1.5},
            {"id": "rfc9114", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs are concise standards guidance documents rather than full protocol suites?",
        "relevant": [
            {"id": "rfc2119", "score": 2.0},
            {"id": "rfc8441", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs are about data representation or identifier syntax rather than transport framing?",
        "relevant": [
            {"id": "rfc3986", "score": 2.0},
            {"id": "rfc8259", "score": 2.0},
            {"id": "rfc6265", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs should be prioritized for connection management and routing behavior in HTTP?",
        "relevant": [
            {"id": "rfc7230", "score": 2.0},
            {"id": "rfc7540", "score": 1.0},
            {"id": "rfc9114", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs include notable security considerations for authentication tokens or cookies?",
        "relevant": [
            {"id": "rfc6750", "score": 2.0},
            {"id": "rfc6265", "score": 1.5},
            {"id": "rfc9114", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs define keyword-style compliance language and normative requirements?",
        "relevant": [
            {"id": "rfc2119", "score": 2.0},
            {"id": "rfc7230", "score": 1.0},
            {"id": "rfc7231", "score": 1.0},
            {"id": "rfc7540", "score": 1.0},
            {"id": "rfc9114", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs define WebSocket bootstrapping over HTTP/2 and the HTTP/2 context it depends on?",
        "relevant": [
            {"id": "rfc8441", "score": 2.0},
            {"id": "rfc7540", "score": 1.5},
        ],
    },
    {
        "query": "Which RFCs are strongest sources for HTTP status-code semantics and interoperability expectations?",
        "relevant": [
            {"id": "rfc7231", "score": 2.0},
            {"id": "rfc9114", "score": 1.0},
            {"id": "rfc7540", "score": 1.0},
        ],
    },
    {
        "query": "Which RFCs are easy hard negatives for JSON parsing tasks because they focus on HTTP transport?",
        "relevant": [
            {"id": "rfc8259", "score": 2.0},
            {"id": "rfc7230", "score": 0.5},
            {"id": "rfc7540", "score": 0.5},
        ],
    },
    {
        "query": "Which RFCs are easy hard negatives for cookie queries because they describe authentication bearer tokens?",
        "relevant": [
            {"id": "rfc6265", "score": 2.0},
            {"id": "rfc6750", "score": 1.5},
        ],
    },
]


def main() -> None:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    docs_dir = TARGET / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    for doc_id, url in sorted(RFC_URLS.items()):
        destination = docs_dir / f"{doc_id}.txt"
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())

    queries_path = TARGET / "queries.jsonl"
    with queries_path.open("w", encoding="utf-8") as handle:
        for row in QUERIES:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    readme = (
        "External holdout benchmark for ContextRAG.\n\n"
        "Purpose:\n"
        "- Provide an out-of-corpus split to check whether findings from `data/eval-mixed`\n"
        "  and `data/eval-expanded` hold on a distinct RFC set.\n"
        "- Increase protocol difficulty with contrastive prompts and hard-negative phrasing.\n\n"
        "Contents:\n"
        "- `documents/`: 10 RFC texts not used in the mixed/expanded corpora\n"
        f"- `queries.jsonl`: {len(QUERIES)} total queries\n"
        "- 20 single-label queries (`relevant_ids`)\n"
        "- 16 multi-label/graded queries (`relevant`)\n\n"
        "Schema:\n"
        "- Legacy line shape: `{\"query\": \"...\", \"relevant_ids\": [\"doc1\", ...]}`\n"
        "- v2 line shape: `{\"query\": \"...\", \"relevant\": [{\"id\": \"doc1\", \"score\": 2.0}, ...]}`\n\n"
        "Sources (retrieved from rfc-editor.org):\n"
        "- RFC 2119, RFC 3986, RFC 6265, RFC 6750, RFC 7230, RFC 7231,\n"
        "  RFC 7540, RFC 8259, RFC 8441, RFC 9114\n\n"
        "Validation:\n\n"
        "```bash\n"
        "uv run contextrag validate-dataset --dataset data/eval-external\n"
        "```\n"
    )
    (TARGET / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {len(QUERIES)} queries to {queries_path}")


if __name__ == "__main__":
    main()
