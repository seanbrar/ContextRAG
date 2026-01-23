from __future__ import annotations

import logging

from contextrag.legacy import category_extraction


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    file_path = "output_2024-01-11T05-16-42.txt"
    categories, total_number = category_extraction.extract_categories_from_file(
        file_path
    )
    logger = category_extraction.logger
    logger.info("Categories: %s", categories)
    logger.info("Total Number of Categories: %s", total_number)
