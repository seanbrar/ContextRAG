import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
import html2text
from contextrag.core.routing import route_bucket
from contextrag.core.tokenizer import count_tokens


class HTMLToMarkdownConverter:
    def __init__(self, folder_path: str, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the HTML to Markdown converter.

        :param folder_path: Path to the folder containing HTML files.
        :param options: Optional configuration for conversion.
        """
        self.folder_path = Path(folder_path)
        self.options = options or {}

        # Default HTML2Text configuration
        self.html2text_converter = html2text.HTML2Text()
        for option, value in self.options.items():
            setattr(self.html2text_converter, option, value)

    def _read_html_file(self, file_path: Path) -> Optional[str]:
        """
        Read and return HTML content from a file.

        :param file_path: Path to the HTML file.
        :return: HTML content as a string.
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError) as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return None

    def _html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML content to Markdown.

        :param html_content: HTML content as a string.
        :return: Markdown content as a string.
        """
        return self.html2text_converter.handle(html_content)

    def _write_markdown_file(
        self, file_path: Path, content: str, use_target_folder: bool = False
    ):
        """
        Write Markdown content to a file, optionally in a target folder based on content length.

        :param file_path: Path to the Markdown file.
        :param content: Markdown content to write.
        :param use_target_folder: Whether to use the target folder system based on content length.
        """
        if use_target_folder:
            target_folder = self._get_target_folder(content)
            target_folder.mkdir(exist_ok=True)  # Create the folder if it doesn't exist
            file_path = (
                target_folder / file_path.name
            )  # Update the file path to the target folder

        try:
            file_path.write_text(content, encoding="utf-8")
        except (FileNotFoundError, PermissionError) as e:
            logging.error(f"Error writing file {file_path}: {e}")

    def _remove_html_footer(self, html_content: str) -> str:
        """
        Remove the footer from HTML content.

        :param html_content: HTML content as a string.
        :return: HTML content without the footer.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        footer = soup.find("div", {"id": "footer", "role": "contentinfo"})
        if footer:
            footer.decompose()
        return str(soup)

    def _get_target_folder(self, content: str) -> Path:
        """
        Determine the target folder based on the content length.

        :param content: Markdown content.
        :return: Path to the target folder.
        """

        bucket = route_bucket(count_tokens(str(content)))
        return self.folder_path / bucket
        
    def convert_all_files(self, use_target_folder: bool = False):
        """
        Convert all HTML files in the folder to Markdown, optionally sorting them into target folders.

        :param use_target_folder: Whether to use the target folder system based on content length.
        """
        for filename in self.folder_path.glob("*.html"):
            html_content = self._read_html_file(filename)
            if html_content is not None:
                html_without_footer = self._remove_html_footer(html_content)
                markdown_content = self._html_to_markdown(html_without_footer)
                md_filename = filename.with_suffix(".md")
                self._write_markdown_file(
                    md_filename, markdown_content, use_target_folder
                )
