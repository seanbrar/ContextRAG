import re
import logging
from typing import Optional
import os

# Setup logging with external configuration option
logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
logging.basicConfig(level=logging_level)

# Constants
TABLE_TYPE_1_PATTERN = "---"
TABLE_TYPE_2_PATTERN = "|"
NORMALIZED_SPACING_REGEX = r"\s{2,}"
SIMPLE_TASK_TYPE_REGEX = r"!\[(.*?)\]"
COMPLEX_TASK_TYPE_REGEX = r"\[!\[(.*?)\]\(.*?\)\]"
PRIORITY_REGEX = r"icons/priorities/(.*?).svg"
ROW_REGEX = r"(\[EW-\d+\].*?Fixed)"
ISSUE_COUNTER_REGEX = r"\[\s*(\d+\s*issue[s]?)\s*\]\(https://[^)]+\)"
KEY_REGEX = r"\[EW-\d+\]"
COMPANY_NAME = os.getenv("COMPANY_NAME")


def is_table_start(line: str, next_line: Optional[str] = None) -> bool:
    """
    Detect the start of a table in Markdown format.
    Type 1 table has '---', Type 2 table has '|' in the next line.

    :param line: The current line being checked.
    :param next_line: The next line in the text.
    :return: True if the line indicates the start of a table, False otherwise.
    """
    return (
        TABLE_TYPE_2_PATTERN in line
        and next_line
        and (TABLE_TYPE_1_PATTERN in next_line or TABLE_TYPE_2_PATTERN in next_line)
    )


def is_table_end(line: str) -> bool:
    """
    Detect the end of a table based on line content in Markdown format.

    :param line: The current line being checked.
    :return: True if the line indicates the end of a table, False otherwise.
    """
    return not line.strip() or TABLE_TYPE_2_PATTERN not in line


# Precompile regular expressions
task_type_regex_simple = re.compile(SIMPLE_TASK_TYPE_REGEX)
task_type_regex_complex = re.compile(COMPLEX_TASK_TYPE_REGEX)
priority_regex = re.compile(PRIORITY_REGEX)
row_regex = re.compile(ROW_REGEX, flags=re.DOTALL)
issue_counter_regex = re.compile(ISSUE_COUNTER_REGEX)
key_regex = re.compile(KEY_REGEX)


def extract_task_type(url: str) -> Optional[str]:
    """
    Extract task type from the provided URL string using regular expressions.

    :param url: The URL string to extract task type from.
    :return: The extracted task type as a string, or None if not found or an error occurs.
    """
    try:
        match = task_type_regex_simple.search(url)
        if match:
            return match.group(1)

        match = task_type_regex_complex.search(url)
        if match:
            return match.group(1)
    except re.error as e:
        logging.error(f"Regex error extracting task type from URL '{url}': {e}")
    return None


def extract_priority(url: str) -> Optional[str]:
    """
    Extract priority from the provided URL string.

    :param url: The URL string to extract priority from.
    :return: The extracted priority as a string, or None if not found or an error occurs.
    """
    try:
        match = priority_regex.search(url)
        if match:
            return match.group(1).capitalize()
    except re.error as e:
        logging.error(f"Regex error extracting priority from URL '{url}': {e}")
    return None


def normalize_spacing(entry: str) -> str:
    """
    Normalize spacing between fields in an entry by replacing multiple
    consecutive spaces with a single space.

    :param entry: The entry string to normalize.
    :return: The normalized entry string.
    """
    return re.sub(NORMALIZED_SPACING_REGEX, " ", entry)


# Constants for Regular Expressions
TASK_TYPE_REPLACEMENT_PATTERN = r"\[!\[.*?\]\(.*?\)\]\(.*?\)"
PRIORITY_REPLACEMENT_PATTERN = r"!\[.*?\]\(.*?icons/priorities/.*?.svg\)"
SUMMARY_REPLACEMENT_PATTERN = r"\[\s?(?!EW-)(.*?)\s?\]\(https://.*?\)"
URL_FORMAT_PATTERN = r"\[(EW-\d+)\]\(https://{COMPANY_NAME}\.atlassian\.net/browse/(EW-\d+)\)"


def process_task_type(entry: str) -> str:
    """
    Replace task type placeholders in the entry with actual task types.

    :param entry: The entry string to process.
    :return: The entry string with task type placeholders replaced.
    """
    task_type = extract_task_type(entry)
    if task_type:
        entry = re.sub(TASK_TYPE_REPLACEMENT_PATTERN, task_type, entry)
    return entry


def process_priority(entry: str) -> str:
    """
    Replace priority placeholders in the entry with actual priorities.

    :param entry: The entry string to process.
    :return: The entry string with priority placeholders replaced.
    """
    priority = extract_priority(entry)
    if priority:
        entry = re.sub(PRIORITY_REPLACEMENT_PATTERN, priority, entry)
    return entry


def process_summary(entry: str) -> str:
    """
    Process the summary section of an entry.

    :param entry: The entry string to process.
    :return: The processed entry string with summary information updated.
    """
    return re.sub(SUMMARY_REPLACEMENT_PATTERN, r"\1", entry)


def ensure_correct_url_format(entry: str) -> str:
    """
    Ensure the key field has the correct URL format in an entry.

    :param entry: The entry string to process.
    :return: The entry string with URL format corrected if necessary.
    """
    return re.sub(
        URL_FORMAT_PATTERN,
        rf"[\1](https://{COMPANY_NAME}.atlassian.net/browse/\1)",
        entry,
    )


def process_buffer_for_replacements(buffer: list) -> str:
    """
    Process a buffer of lines for various replacements and normalization.

    :param buffer: A list of lines representing the buffer to be processed.
    :return: The processed buffer as a single string.
    """
    entry = "".join(buffer).strip()
    entry = process_task_type(entry)
    entry = process_priority(entry)
    entry = process_summary(entry)
    entry = ensure_correct_url_format(entry)
    return normalize_spacing(entry)


def join_entries_on_single_line(text: str) -> str:
    """
    Join multiple entries into a single line.

    :param text: The text containing multiple entries.
    :return: The text with entries joined into a single line.
    """
    # Split the text into rows and process each row
    rows = [
        process_buffer_for_replacements(row.replace("\n", " "))
        for row in row_regex.findall(text)
    ]

    # Replace the original multi-line rows in text with processed single-line rows
    for idx, row in enumerate(row_regex.findall(text)):
        text = text.replace(row, rows[idx])

    return text


def process_issue_counter(content: str) -> str:
    """
    Replace the verbose issue counter with a simplified version.

    :param content: The string content containing issue counters.
    :return: The content with simplified issue counters.
    """
    return re.sub(issue_counter_regex, r"\1", content)


def add_url_to_key(entry: str) -> str:
    """
    Add the URL back to the key field in an entry, if missing.

    :param entry: The entry string where the URL needs to be added.
    :return: The entry with the URL added.
    """
    key_match = key_regex.search(entry)
    if key_match:
        key_value = key_match.group().strip("[]")
        url_pattern = f"\[{key_value}\]\(https://{COMPANY_NAME}.atlassian.net/browse/{key_value}\)"
        if not re.search(url_pattern, entry):
            key_pattern = r"\[" + re.escape(key_value) + r"\]"
            new_url_pattern = f"[{key_value}](https://{COMPANY_NAME}.atlassian.net/browse/{key_value})"
            entry = re.sub(key_pattern, new_url_pattern, entry)
    return entry


def transform_entries(text: str) -> str:
    """
    Transform entries in the text.

    :param text: The string text containing multiple entries.
    :return: The transformed text with processed entries.
    """
    lines = text.split("\n")
    entry_buffer = []
    entries = []

    for line in lines:
        if should_flush_buffer(line, entry_buffer):
            entries.append(flush_buffer(entry_buffer))
            entries.append(process_line(line))
        else:
            entry_buffer.append(line)

    entries.append(flush_buffer(entry_buffer))  # Process any remaining buffer
    return "\n".join(entries)


def process_line(line: str) -> str:
    """
    Process a single line of text.

    :param line: The line to be processed.
    :return: The processed line.
    """
    stripped_line = line.strip()
    return (
        add_url_to_key(stripped_line)
        if key_regex.search(stripped_line)
        else stripped_line
    )


def should_flush_buffer(line: str, buffer: list) -> bool:
    """
    Determine whether the current buffer should be flushed based on the line content.

    :param line: The current line being read.
    :param buffer: The current buffer of lines.
    :return: True if the buffer should be flushed, False otherwise.
    """
    return is_non_data_entry(line) or is_data_entry_start(line, buffer)


def flush_buffer(buffer: list) -> str:
    """
    Process and clear the current buffer, returning the processed text.

    :param buffer: The buffer to be flushed.
    :return: The processed buffer as a string.
    """
    processed_text = process_buffer_for_replacements(buffer)
    buffer.clear()  # Clearing the buffer
    return processed_text


def process_buffer_if_needed(buffer: list) -> str:
    """
    Process the buffer if it contains data, otherwise return an empty string.

    :param buffer: The buffer to be processed.
    :return: The processed buffer or an empty string.
    """
    return process_buffer_for_replacements(buffer) if buffer else ""


def is_non_data_entry(line: str) -> bool:
    """
    Check if the line is a non-data entry such as a header or label.

    :param line: The line to be checked.
    :return: True if it's a non-data entry, False otherwise.
    """
    return line.startswith("## ") or "![](" in line or "Key | Summary | T |" in line


def is_data_entry_start(line: str, buffer: list) -> bool:
    """
    Check if the line is the start of a data entry.

    :param line: The line to be checked.
    :param buffer: The current buffer.
    :return: True if it's the start of a data entry, False otherwise.
    """
    return key_regex.match(line) and buffer


def finalize_entries(text: str) -> str:
    """
    Finalize the entries in the text.

    :param text: The string text containing entries.
    :return: The text with entries finalized.
    """
    lines = text.split("\n")
    entries = [
        add_url_to_key(line) if key_regex.search(line) else line for line in lines
    ]
    return "\n".join(entries)


def process_content(content: str, content_type: str) -> str:
    """
    Process the entire content from start to end, applying transformations
    based on the content type.

    :param content: The markdown content to be processed.
    :param content_type: The type of content (e.g., 'release_notes', 'documentation').
    :return: The fully processed content.
    """
    if content_type == "release_notes":
        return process_release_notes_content(content)
    elif content_type == "documentation":
        # Process documentation or other types of content (not yet implemented)
        # ...
        pass
    return content


def process_release_notes_content(content: str) -> str:
    """
    Process the entire release notes content.

    :param content: The release notes content to be processed.
    :return: The processed content.
    """
    content = join_entries_on_single_line(content)
    content = process_issue_counter(content)
    content = finalize_entries(content)
    return content
