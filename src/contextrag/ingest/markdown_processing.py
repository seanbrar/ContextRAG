import re


def modify_markdown(content):
    """
    Modify Markdown content by applying a series of transformations.

    :param content: Markdown content as a string.
    :return: Modified Markdown content.
    """
    content = remove_above_first_header_if_level_one(content)
    content = remove_inline_attachments(content)
    content = remove_attachments_header_and_first_line(content)
    content = convert_indented_blocks_to_code(content)
    content = reduce_excessive_line_breaks(content)
    return content.strip()


def remove_above_first_header(content):
    """
    Remove everything above the first Markdown header in the content. A Markdown
    header is defined as a line starting with '#', optionally preceded by whitespace.

    :param content: Markdown content as a string.
    :return: Markdown content without the section above the first header.
    """
    # Validate input
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    # Precompile regex pattern for the first header
    first_header_pattern = re.compile(r"^(?:\s*)#", flags=re.MULTILINE)

    # Search for the first header and remove content above it if found
    match = first_header_pattern.search(content)
    if match:
        return content[match.start() :]
    else:
        return content


def remove_above_first_header_if_level_one(content):
    """
    Remove everything above the first Markdown header only if it is a level-one header.

    :param content: Markdown content as a string.
    :return: Markdown content without the section above the first level-one header.
    """
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    first_header_pattern = re.compile(r"^(?:\s*)(#+)\s", flags=re.MULTILINE)
    match = first_header_pattern.search(content)
    if match and len(match.group(1)) == 1:
        return content[match.start() :]
    return content


def remove_attachments_header_and_first_line(content):
    """
    Remove the 'Attachments' header line and the line that immediately follows it.

    :param content: Markdown content as a string.
    :return: Markdown content without the 'Attachments' header and first line.
    """
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    pattern = re.compile(r"(^|\n)## Attachments:\n[^\n]*(?:\n|$)")
    return re.sub(pattern, r"\1", content)


def remove_attachments_section(content):
    """
    Remove the 'Attachments' subheader and everything that follows.

    :param content: Markdown content as a string.
    :return: Markdown content without the 'Attachments' section.
    """
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    # Split on the first occurrence of '\n## Attachments:' and keep the part before it.
    return re.split(r"\n## Attachments:", content, maxsplit=1)[0]


def remove_inline_attachments(content):
    """
    Remove inline attachments from the Markdown content.

    :param content: Markdown content as a string.
    :return: Markdown content without inline attachments.
    """
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    nested_pattern = re.compile(r"\[!\[.*?\]\(attachments/.*?\)\]\(attachments/.*?\)")
    standard_pattern = re.compile(r"!\[.*?\]\(attachments/.*?\)")

    content = re.sub(nested_pattern, "", content)
    return re.sub(standard_pattern, "", content)


def remove_attachments(content):
    """
    Remove the 'Attachments' section and inline attachments from the Markdown content.

    This function performs two main tasks:
    1. Removes the section starting with '## Attachments:' and everything that follows.
    2. Removes lines containing inline attachments, both in the standard and nested formats.

    :param content: Markdown content as a string.
    :return: Markdown content without the 'Attachments' section and inline attachments.
    """
    # Validate input
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    # Precompile regex patterns for efficiency
    attachments_section_pattern = re.compile(r"\n## Attachments:.*", flags=re.DOTALL)
    standard_inline_attachment_pattern = re.compile(
        r"^.*!\[.*?\]\(attachments/.*?\).*$", flags=re.MULTILINE
    )
    nested_inline_attachment_pattern = re.compile(
        r"^.*\[!\[.*?\]\(attachments/.*?\)\]\(attachments/.*?\).*$", flags=re.MULTILINE
    )

    # Remove 'Attachments' section
    content = re.sub(attachments_section_pattern, "", content)

    # Remove standard and nested inline attachments
    content = re.sub(standard_inline_attachment_pattern, "", content)
    content = re.sub(nested_inline_attachment_pattern, "", content)

    return content


def clean_up_lines(content):
    """
    Clean up lines in the Markdown content by performing two actions:
    1. Removing trailing spaces or tabs from every line.
    2. Removing lines that consist only of spaces or tabs.

    :param content: Markdown content as a string.
    :return: Markdown content with cleaned-up lines.
    """
    # Validate input
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    # Precompile regex patterns for efficiency
    trailing_spaces_pattern = re.compile(r"[ \t]+$", flags=re.MULTILINE)
    only_spaces_lines_pattern = re.compile(r"^[\t ]+$", flags=re.MULTILINE)

    # Remove trailing spaces or tabs from every line
    content = re.sub(trailing_spaces_pattern, "", content)
    # Remove lines that consist only of spaces or tab characters
    content = re.sub(only_spaces_lines_pattern, "", content)

    return content


def convert_indented_blocks_to_code(content):
    """
    Convert blocks of text indented by four spaces into Markdown code blocks.
    This transformation enhances readability and formatting in Markdown-rendered content.

    :param content: Markdown content as a string.
    :return: Markdown content with indented text blocks converted to code blocks.
    """
    # Validate input
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    if not content:
        return content

    # If the content starts with an indented line, wrap the whole block.
    if content.startswith("    "):
        stripped = content.rstrip("\n")
        return f"```\n{stripped}\n```\n"

    indented_blocks_pattern = re.compile(r"(?:^ {4}.*(?:\n|$))+", flags=re.MULTILINE)

    def wrap_block(match):
        block = match.group(0).rstrip("\n")
        return f"```\n{block}\n```\n"

    return re.sub(indented_blocks_pattern, wrap_block, content)


def reduce_excessive_line_breaks(content):
    """
    Reduce instances of more than two consecutive line breaks in the Markdown content.
    This function ensures that the spacing in the rendered Markdown does not have
    excessive whitespace, which can improve readability.

    :param content: Markdown content as a string.
    :return: Markdown content with reduced excessive line breaks.
    """
    # Validate input
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    # Precompile regex pattern for excessive line breaks
    excessive_line_breaks_pattern = re.compile(r"(\n[ \t]*){3,}", flags=re.MULTILINE)

    # Reduce excessive line breaks to two
    return re.sub(excessive_line_breaks_pattern, "\n\n", content)
