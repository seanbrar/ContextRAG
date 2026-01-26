import logging

from contextrag.ingest.html_to_markdown import HTMLToMarkdownConverter


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    folder_path = "./html"
    options = {
        # "ignore_links": True,
        # "wrap_links": False,
        # "ignore_images": True,
    }
    converter = HTMLToMarkdownConverter(folder_path, options)
    converter.convert_all_files(use_target_folder=True)


if __name__ == "__main__":
    main()
