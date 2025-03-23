import argparse
import hashlib
import json
import os
import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from utils.tokenizer import count_tokens

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def calculate_checksum(file_content):
    return hashlib.sha256(file_content.encode()).hexdigest()


def load_embeddings_cache(cache_file):
    try:
        with open(cache_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def update_embeddings_cache(cache_file, cache_data):
    with open(cache_file, "w") as file:
        json.dump(cache_data, file)


def parse_arguments():
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
    markdown_files = {}
    checksums = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r") as file:
                content = file.read()
                checksum = calculate_checksum(content)
                markdown_files[filename] = content
                checksums[filename] = checksum
    return markdown_files, checksums


def preprocess_text(text):
    # Complex logic
    # Remove attachments subheader and everything below
    text = re.split(r"\n## Attachments:", text, maxsplit=1)[0]

    # Remove inline attachments
    text = re.sub(r"^.*!\[.*?\]\(attachments/.*?\).*$", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^.*\[!\[.*?\]\(attachments/.*?\)\]\(attachments/.*?\).*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove trailing spaces from every line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Remove lines that only consist of spaces or tab characters
    text = re.sub(r"^[\t ]+$", "", text, flags=re.MULTILINE)

    # Reduce excessive line breaks to two
    # Account for lines with spaces but no other content
    text = re.sub(r"(\n[ \t]*){3,}", "\n\n", text)

    # Simple logic
    # Remove markdown formatting (basic example)
    # text = re.sub(r'\#.*', '', text)  # Remove headers
    text = re.sub(r"\!\[.*\]\(.*\)", "", text)  # Remove images
    # text = re.sub(r'\[.*\]\(.*\)', '', text)  # Remove links

    return text.strip()


def count_tokens(text):
    if not isinstance(text, str):
        raise TypeError(f"Expected a string, but received {type(text).__name__}")
    encoding = count_tokens.get_encoding("cl100k_base")

    token_count = len(encoding.encode(text))

    return token_count


def compute_similarity(files_dict, checksums, cache):
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
    cache_file = "embeddings_cache.json"
    embeddings_cache = load_embeddings_cache(cache_file)

    # Step 1: Read markdown files from the specified folder
    markdown_files, checksums = read_markdown_files(folder_path)

    # Step 2 and 3 are combined: Compute similarity between files
    similarity_matrix = compute_similarity(markdown_files, checksums, embeddings_cache)
    update_embeddings_cache(cache_file, embeddings_cache)

    # Step 4: Group similar files based on the similarity matrix
    # You can adjust the threshold as needed
    groups = group_similar_files(similarity_matrix, threshold=0.9)

    # Step 5: Print the file groupings to the specified output file
    print_file_groupings(markdown_files, groups, output_file=output_file)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.folder_path, args.debug, args.output)
