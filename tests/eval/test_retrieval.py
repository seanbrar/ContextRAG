from contextrag.eval.retrieval import (build_lexical_index, rank_bm25,
                                       rank_hybrid_rrf,
                                       rerank_by_token_overlap)


def test_rank_bm25_prefers_matching_document():
    docs = ["http semantics methods", "tls handshake key schedule"]
    ids = ["doc_http", "doc_tls"]
    index = build_lexical_index(documents=docs, ids=ids)
    ranked = rank_bm25(index, "http methods", n_results=2)
    assert ranked[0] == "doc_http"


def test_rank_hybrid_rrf_combines_sources():
    ranked = rank_hybrid_rrf(
        dense_ids=["a", "b", "c"],
        lexical_ids=["c", "a", "d"],
        n_results=3,
        rrf_k=10,
    )
    assert ranked[0] in {"a", "c"}
    assert len(ranked) == 3


def test_rerank_by_token_overlap_uses_query_overlap():
    token_sets = {
        "a": {"http", "cache"},
        "b": {"tls", "cipher"},
    }
    reranked = rerank_by_token_overlap(
        query="http caching semantics",
        candidate_ids=["b", "a"],
        token_sets=token_sets,
        n_results=2,
    )
    assert reranked[0] == "a"
