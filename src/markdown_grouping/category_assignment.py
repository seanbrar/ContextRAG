from datetime import datetime
import os
import re

from dotenv import load_dotenv
from utils.chat import ChatManager, ChatModels
from utils.tokenizer import count_tokens
from markdown_grouping.file_grouping import preprocess_text

load_dotenv()


def read_markdown_files(folder_path: str = "markdown_grouping/markdown"):
    markdown_files = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r") as file:
                content = file.read()
                markdown_files[filename] = content
    return markdown_files


def main():
    chat_manager = ChatManager()
    markdown_files: dict = read_markdown_files()
    system_message = "You are a helpful assistant. Your goal is to analyze markdown files from a corporate knowledge base. The files you will examine are highly technical, and may focus on hardware, software, or a mixture of the two. Focus on technical information rather than metadata such as author and creation date."

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_filename = f"output_{timestamp}.txt"

    with open(output_filename, "w") as output_file:
        for filename, content in markdown_files.items():
            content = preprocess_text(content)
            token_count: int = count_tokens(content)

            # Determine the model based on the token count
            if token_count <= 3500:
                model = ChatModels.GPT_3_5_TURBO_1106
            elif 3500 < token_count < 15000:
                model = ChatModels.GPT_3_5_TURBO_16K
            else:
                print(
                    f"Skipped {filename} due to excessive token count ({token_count} tokens)."
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
            print(f"Processed: {filename}")


if __name__ == "__main__":
    main()
