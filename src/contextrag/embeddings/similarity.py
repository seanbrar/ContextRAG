import argparse
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from contextrag.core.cache import load_json_cache, save_json_cache
from contextrag.core.io import checksum as checksum_text
from contextrag.core import tokenizer
from contextrag.ingest.markdown_processing import preprocess_similarity_text

load_dotenv()


def parse_arguments():
    """Parse command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
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


def read_markdown_files(folder_path):
    """Read all markdown files from the specified folder.

    Args:
        folder_path (str): Path to the folder containing markdown files.

    Returns:
        tuple: A tuple containing:
            - dict: Mapping of filenames to their contents
            - dict: Mapping of filenames to their checksums
    """
    markdown_files = {}
    checksums = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r") as file:
                content = file.read()
            checksum = checksum_text(content)
            markdown_files[filename] = content
            checksums[filename] = checksum
    return markdown_files, checksums


def preprocess_text(text):
    """Preprocess markdown text by removing attachments and cleaning formatting.

    Args:
        text (str): Raw markdown text to process.

    Returns:
        str: Cleaned and preprocessed text.
    """
    return preprocess_similarity_text(text)


def count_tokens(text):
    """Count tokens in a text string using the shared tokenizer.

    Args:
        text (str): Text to count tokens for.

    Returns:
        int: Number of tokens in the text.
    """
    return tokenizer.count_tokens(text)


def compute_similarity(files_dict, checksums, cache):
    """Compute similarity between files using OpenAI embeddings.

    Args:
        files_dict (dict): Mapping of filenames to their contents.
        checksums (dict): Mapping of filenames to their checksums.
        cache (dict): Cache of previously computed embeddings.

    Returns:
        numpy.ndarray: Similarity matrix of file comparisons.
    """
    embeddings = []
    client = OpenAI()
    for filename, content in files_dict.items():
        checksum = checksums[filename]
        if checksum in cache:
            embeddings.append(cache[checksum])
        else:
            processed_text = preprocess_text(content)
            token_count = count_tokens(processed_text)

            if token_count <= 8000:
                response = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=processed_text,
                    encoding_format="float",
                    dimensions=3072,
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                cache[checksum] = embedding
            else:
                print(
                    f"Skipped {filename} due to excessive token count ({token_count} tokens)."
                )

    # Normalize the embeddings before computing the cosine similarity
    normalized_embeddings = normalize(np.array(embeddings))
    similarity_matrix = cosine_similarity(normalized_embeddings)
    return similarity_matrix


def group_similar_files(similarity_matrix, threshold=0.6):
    """Group files based on their similarity scores.

    Args:
        similarity_matrix (numpy.ndarray): Matrix of similarity scores.
        threshold (float, optional): Minimum similarity score to consider files similar.
            Defaults to 0.6.

    Returns:
        dict: Mapping of file indices to lists of similar file indices.
    """
    groups = {}
    for i, row in enumerate(similarity_matrix):
        # similar_files = [j for j, sim in enumerate(row) if sim > threshold and i != j]
        # Only consider files with an index greater than the current file
        # to avoid duplicating pairings (i.e., if i is similar to j, don't list j as similar to i)
        similar_files = [j for j in range(i + 1, len(row)) if row[j] > threshold]
        if similar_files:  # Only add if there are any similar files
            groups[i] = similar_files
    return groups


def print_file_groupings(files_dict, groups, output_file=None):
    """Print or save the groupings of similar files.

    Args:
        files_dict (dict): Mapping of filenames to their contents.
        groups (dict): Mapping of file indices to lists of similar file indices.
        output_file (str, optional): Path to save the output. If None, prints to stdout.
    """
    filenames = list(files_dict.keys())
    output = []

    for file_index in range(len(filenames)):
        if file_index in groups:
            output.append(f"File: {filenames[file_index]} is similar to:")
            for similar_file_index in groups[file_index]:
                output.append(f" - {filenames[similar_file_index]}")
        else:
            # Note files with no similar files explicitly
            # output.append(f"File: {filenames[file_index]} has no similar files.")
            pass

    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(output))
    else:
        for line in output:
            print(line)


def main(folder_path, debug, output_file):
    """Main function to process and group similar markdown files.

    Args:
        folder_path (str): Path to the folder containing markdown files.
        debug (bool): Whether to print debug information.
        output_file (str): Path to save the output file.
    """
    cache_file = "embeddings_cache.json"
    embeddings_cache = load_json_cache(cache_file)

    # Step 1: Read markdown files from the specified folder
    markdown_files, checksums = read_markdown_files(folder_path)

    # Step 2 and 3 are combined: Compute similarity between files
    similarity_matrix = compute_similarity(markdown_files, checksums, embeddings_cache)
    save_json_cache(cache_file, embeddings_cache)

    # Step 4: Group similar files based on the similarity matrix
    # You can adjust the threshold as needed
    groups = group_similar_files(similarity_matrix, threshold=0.9)

    # Step 5: Print the file groupings to the specified output file
    print_file_groupings(markdown_files, groups, output_file=output_file)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.folder_path, args.debug, args.output)
