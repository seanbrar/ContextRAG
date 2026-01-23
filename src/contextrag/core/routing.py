from contextrag.core.constants import MEDIUM_MAX_TOKENS, SHORT_MAX_TOKENS


def route_bucket(
    token_count: int,
    short_max: int = SHORT_MAX_TOKENS,
    medium_max: int = MEDIUM_MAX_TOKENS,
) -> str:
    if token_count <= short_max:
        return "short"
    if token_count <= medium_max:
        return "medium"
    return "long"
