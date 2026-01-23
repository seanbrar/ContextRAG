from __future__ import annotations

from contextrag.legacy import category_extraction as _legacy

extract_categories_from_file = _legacy.extract_categories_from_file

__all__ = ["extract_categories_from_file"]


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    file_path = "output_2024-01-11T05-16-42.txt"
    categories, total_number = extract_categories_from_file(file_path)
    logger = _legacy.logger
    logger.info("Categories: %s", categories)
    logger.info("Total Number of Categories: %s", total_number)
