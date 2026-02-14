import pytest

from contextrag.eval.query_schema import load_and_validate_queries, validate_query_record


def test_validate_query_record_legacy():
    record = validate_query_record(
        {"query": "What is RFC 9110?", "relevant_ids": ["rfc9110", "rfc9112"]},
        line_no=1,
    )
    assert record["query"] == "What is RFC 9110?"
    assert record["relevant_ids"] == ["rfc9110", "rfc9112"]


def test_validate_query_record_v2_relevant_array():
    record = validate_query_record(
        {
            "query": "Compare specs",
            "relevant": [
                {"id": "rfc9110", "score": 2.0},
                {"id": "rfc9112", "score": 1.0},
            ],
        },
        line_no=1,
    )
    assert record["relevant_ids"] == ["rfc9110", "rfc9112"]
    assert record["relevant_scores"]["rfc9110"] == 2.0


def test_validate_query_record_rejects_duplicates():
    with pytest.raises(ValueError, match="must not contain duplicates"):
        validate_query_record(
            {"query": "q", "relevant_ids": ["doc1", "doc1"]},
            line_no=3,
        )


def test_load_and_validate_queries_rejects_invalid_json(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text("{invalid json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_and_validate_queries(path)


def test_validate_query_record_relevant_scores_subset():
    with pytest.raises(ValueError, match="must also appear in 'relevant_ids'"):
        validate_query_record(
            {
                "query": "q",
                "relevant_ids": ["doc1"],
                "relevant_scores": {"doc2": 1.0},
            },
            line_no=1,
        )
