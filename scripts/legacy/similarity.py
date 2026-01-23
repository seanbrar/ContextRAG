from __future__ import annotations

from contextrag.legacy import similarity


if __name__ == "__main__":
    args = similarity.parse_arguments()
    similarity.main(args.folder_path, args.debug, args.output)
