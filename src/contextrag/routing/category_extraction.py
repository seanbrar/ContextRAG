from __future__ import annotations

import logging
import re
from pathlib import Path

from contextrag.core.logging import get_logger


logger = get_logger(__name__)

__all__ = ["extract_categories_from_file"]


def extract_categories_from_file(file_path: str | Path) -> tuple[str, int]:
    """Extract unique categories from a file containing category listings."""
    pattern = r"\| Categories: (.+?)\n"
    path = Path(file_path)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "File not found. Please check the file path.", 0

    extracted = re.findall(pattern, text)
    categories = ", ".join(extracted).split(", ") if extracted else []
    unique_categories = list(set(categories))
    result = ", ".join(unique_categories)
    return result, len(unique_categories)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    file_path = "output_2024-01-11T05-16-42.txt"
    categories, total_number = extract_categories_from_file(file_path)
    logger.info("Categories: %s", categories)
    logger.info("Total Number of Categories: %s", total_number)
