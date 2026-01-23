from __future__ import annotations

from contextrag.legacy import similarity as _legacy

parse_arguments = _legacy.parse_arguments
load_json_cache = _legacy.load_json_cache
save_json_cache = _legacy.save_json_cache
OpenAI = _legacy.OpenAI
read_markdown_files = _legacy.read_markdown_files
preprocess_text = _legacy.preprocess_text
count_tokens = _legacy.count_tokens
group_similar_files = _legacy.group_similar_files
print_file_groupings = _legacy.print_file_groupings

__all__ = [
    "main",
    "parse_arguments",
    "load_json_cache",
    "save_json_cache",
    "OpenAI",
    "read_markdown_files",
    "preprocess_text",
    "count_tokens",
    "compute_similarity",
    "group_similar_files",
    "print_file_groupings",
]


def main(folder_path, debug, output_file):
    _legacy.load_json_cache = load_json_cache
    _legacy.save_json_cache = save_json_cache
    _legacy.OpenAI = OpenAI
    _legacy.read_markdown_files = read_markdown_files
    _legacy.preprocess_text = preprocess_text
    _legacy.count_tokens = count_tokens
    _legacy.compute_similarity = compute_similarity
    _legacy.group_similar_files = group_similar_files
    _legacy.print_file_groupings = print_file_groupings
    return _legacy.main(folder_path, debug, output_file)


def compute_similarity(files_dict, checksums, cache):
    _legacy.OpenAI = OpenAI
    _legacy.count_tokens = count_tokens
    return _legacy.compute_similarity(files_dict, checksums, cache)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.folder_path, args.debug, args.output)
