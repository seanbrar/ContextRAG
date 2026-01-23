from __future__ import annotations

from datetime import datetime
import logging
import os
import re

from dotenv import load_dotenv

from contextrag.core.logging import get_logger
from contextrag.core.routing import route_bucket
from contextrag.core.tokenizer import count_tokens
from contextrag.ingest.markdown_processing import preprocess_similarity_text
from contextrag.providers.openai_chat import ChatManager, ChatModels


logger = get_logger(__name__)

__all__ = [
    "main",
    "read_markdown_files",
    "ChatManager",
    "ChatModels",
    "count_tokens",
    "datetime",
    "preprocess_similarity_text",
]


def read_markdown_files(folder_path: str = "markdown_grouping/markdown") -> dict:
    """Read markdown files from the specified directory."""
    markdown_files: dict[str, str] = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                markdown_files[filename] = file.read()
    return markdown_files


def main() -> None:
    """Process markdown files and assign AI-generated categories."""
    load_dotenv()
    chat_manager = ChatManager()
    markdown_files = read_markdown_files()
    system_message = (
        "You are a helpful assistant. Your goal is to analyze markdown files from "
        "a corporate knowledge base. The files you will examine are highly technical, "
        "and may focus on hardware, software, or a mixture of the two. Focus on "
        "technical information rather than metadata such as author and creation date."
    )

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_filename = f"output_{timestamp}.txt"

    with open(output_filename, "w", encoding="utf-8") as output_file:
        for filename, content in markdown_files.items():
            content = preprocess_similarity_text(content)
            token_count = count_tokens(content)

            bucket = route_bucket(token_count)
            if bucket == "short":
                model = ChatModels.GPT_3_5_TURBO_1106
            elif bucket == "medium":
                model = ChatModels.GPT_3_5_TURBO_16K
            else:
                logger.warning(
                    "Skipped %s due to excessive token count (%s tokens).",
                    filename,
                    token_count,
                )
                continue

            user_message = (
                "Please read the following markdown file carefully. "
                "Summarize the main points and topics in this document. Afterwards, "
                "identify 3-5 primary technical categories using the following notation "
                '(encased in triple quotes):\n\n'
                '"""Categories: Item 1, Item 2, Item 3, etc."""\n\n'
                "FILE:\n```markdown\n"
                f"{content}\n```"
            )

            response = chat_manager.complete(
                model,
                user_message,
                system_message,
                temperature=0,
            )

            message = response.choices[0].message.content
            categories = re.findall(r'"""(.*?)"""|```(.*?)```', message, re.DOTALL)

            if categories:
                for match in categories:
                    extracted_categories = match[0] if match[0] else match[1]
                    output_file.write(
                        f"Filename: {filename} | {extracted_categories.strip()}\n"
                    )
            else:
                output_file.write(f"Filename: {filename} | No categories found.\n")

            chat_manager.reset()
            logger.info("Processed: %s", filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
