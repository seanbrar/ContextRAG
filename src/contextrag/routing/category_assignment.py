from datetime import datetime
import logging
import os
import re

from dotenv import load_dotenv
from contextrag.providers.openai_chat import ChatManager, ChatModels
from contextrag.core.routing import route_bucket
from contextrag.core.tokenizer import count_tokens
from contextrag.ingest.markdown_processing import preprocess_similarity_text
from contextrag.core.logging import get_logger

load_dotenv()

logger = get_logger(__name__)


def read_markdown_files(folder_path: str = "markdown_grouping/markdown"):
    """Read markdown files from the specified directory.

    Args:
        folder_path (str, optional): Path to the folder containing markdown files.
            Defaults to "markdown_grouping/markdown".

    Returns:
        dict: Mapping of filenames to their contents.
    """
    markdown_files = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r") as file:
                content = file.read()
                markdown_files[filename] = content
    return markdown_files


def main():
    """Main function to process markdown files and assign categories.

    This function:
    1. Reads markdown files from the default directory
    2. Processes each file using OpenAI's chat models
    3. Extracts technical categories using AI analysis
    4. Saves the results to a timestamped output file
    """
    chat_manager = ChatManager()
    markdown_files: dict = read_markdown_files()
    system_message = "You are a helpful assistant. Your goal is to analyze markdown files from a corporate knowledge base. The files you will examine are highly technical, and may focus on hardware, software, or a mixture of the two. Focus on technical information rather than metadata such as author and creation date."

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_filename = f"output_{timestamp}.txt"

    with open(output_filename, "w") as output_file:
        for filename, content in markdown_files.items():
            content = preprocess_similarity_text(content)
            token_count: int = count_tokens(content)

            # Determine the model based on the token count
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

            # Generate the summary using the selected model
            user_message = f'Please read the following markdown file carefully. Summarize the main points and topics in this document. Afterwards, identify 3-5 primary technical categories using the following notation (encased in triple quotes):\n\n"""Categories: Item 1, Item 2, Item 3, etc."""\n\nFILE:\n```markdown\n{content}\n```'

            response = chat_manager.complete(
                model,
                user_message,
                system_message,
                temperature=0,
            )

            # Extract the assistant's message
            message = response.choices[0].message.content

            # Extract categories using regex
            categories = re.findall(r'"""(.*?)"""|```(.*?)```', message, re.DOTALL)

            # Write results to file
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
