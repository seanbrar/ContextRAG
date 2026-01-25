from __future__ import annotations

import argparse
import os

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from contextrag.config import load_config
from contextrag.core import tokenizer
from contextrag.core.cache import load_json_cache, save_json_cache
from contextrag.core.io import checksum as checksum_text
from contextrag.embeddings.provider import build_embedding_function
from contextrag.ingest.markdown_processing import preprocess_similarity_text

__all__ = [
    "main",
    "parse_arguments",
    "load_json_cache",
    "save_json_cache",
    "read_markdown_files",
    "preprocess_text",
    "count_tokens",
    "compute_similarity",
    "group_similar_files",
    "print_file_groupings",
]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Markdown File Grouping")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the embeddings instead of file groupings",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Specify the output file path (default: None)",
    )
    parser.add_argument(
        "folder_path",
        type=str,
        nargs="?",
        default="markdown_grouping/markdown",
        help="Path to the folder containing markdown files",
    )
    return parser.parse_args()


def read_markdown_files(folder_path: str):
    """Read all markdown files from the specified folder."""
    markdown_files: dict[str, str] = {}
    checksums: dict[str, str] = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                content = file.read()
            checksum = checksum_text(content)
            markdown_files[filename] = content
            checksums[filename] = checksum
    return markdown_files, checksums


def preprocess_text(text: str) -> str:
    """Preprocess markdown text by removing attachments and cleaning formatting."""
    return preprocess_similarity_text(text)


def count_tokens(text: str) -> int:
    """Count tokens in a text string using the shared tokenizer."""
    return tokenizer.count_tokens(text)


def compute_similarity(files_dict, checksums, cache):
    """Compute similarity between files using chromaroute embeddings."""
    embeddings = []
    config = load_config()
    embedding_function = build_embedding_function(config=config)
    for filename, content in files_dict.items():
        checksum = checksums[filename]
        if checksum in cache:
            embeddings.append(cache[checksum])
        else:
            processed_text = preprocess_text(content)
            token_count = count_tokens(processed_text)

            if token_count <= 8000:
                embedding = embedding_function([processed_text])[0]
                embeddings.append(embedding)
                cache[checksum] = embedding
            else:
                print(
                    f"Skipped {filename} due to excessive token count ({token_count} tokens)."
                )

    normalized_embeddings = normalize(np.array(embeddings))
    similarity_matrix = cosine_similarity(normalized_embeddings)
    return similarity_matrix


def group_similar_files(similarity_matrix, threshold: float = 0.6):
    """Group files based on their similarity scores."""
    groups: dict[int, list[int]] = {}
    for i, row in enumerate(similarity_matrix):
        similar_files = [j for j in range(i + 1, len(row)) if row[j] > threshold]
        if similar_files:
            groups[i] = similar_files
    return groups


def print_file_groupings(files_dict, groups, output_file=None):
    """Print or save the groupings of similar files."""
    filenames = list(files_dict.keys())
    output = []

    for file_index in range(len(filenames)):
        if file_index in groups:
            output.append(f"File: {filenames[file_index]} is similar to:")
            for similar_file_index in groups[file_index]:
                output.append(f" - {filenames[similar_file_index]}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as file_handle:
            file_handle.write("\n".join(output))
    else:
        for line in output:
            print(line)


def main(folder_path, debug, output_file):
    """Main function to process and group similar markdown files."""
    load_dotenv()
    cache_file = "embeddings_cache.json"
    embeddings_cache = load_json_cache(cache_file)

    markdown_files, checksums = read_markdown_files(folder_path)
    similarity_matrix = compute_similarity(markdown_files, checksums, embeddings_cache)
    save_json_cache(cache_file, embeddings_cache)

    groups = group_similar_files(similarity_matrix, threshold=0.9)
    print_file_groupings(markdown_files, groups, output_file=output_file)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.folder_path, args.debug, args.output)
