EMBEDDING_COSTS_PER_MILLION: dict[str, float] = {
    # OpenRouter models
    "qwen/qwen3-embedding-8b": 0.01,
    "thenlper/gte-base": 0.005,
    # OpenAI models (direct or via OpenRouter)
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def get_embedding_cost_per_million(model: str) -> float | None:
    """Get cost per million tokens for a model, or None if unknown."""
    if model in EMBEDDING_COSTS_PER_MILLION:
        return EMBEDDING_COSTS_PER_MILLION[model]
    for known_model, cost in EMBEDDING_COSTS_PER_MILLION.items():
        if model.endswith(known_model) or known_model.endswith(model):
            return cost
    return None
