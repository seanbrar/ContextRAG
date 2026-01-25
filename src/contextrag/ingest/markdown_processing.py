import re

_FIRST_HEADER_PATTERN = re.compile(r"^(?:\s*)(#+)\s", flags=re.MULTILINE)
_ATTACHMENTS_HEADER_LINE_PATTERN = re.compile(
    r"(^|\n)## Attachments:\n[^\n]*(?:\n|$)"
)
_INLINE_ATTACHMENTS_NESTED_PATTERN = re.compile(
    r"\[!\[.*?\]\(attachments/.*?\)\]\(attachments/.*?\)"
)
_INLINE_ATTACHMENTS_PATTERN = re.compile(r"!\[.*?\]\(attachments/.*?\)")
_TRAILING_SPACES_PATTERN = re.compile(r"[ \t]+$", flags=re.MULTILINE)
_ONLY_SPACES_LINES_PATTERN = re.compile(r"^[\t ]+$", flags=re.MULTILINE)
_INDENTED_BLOCKS_PATTERN = re.compile(r"(?:^ {4}.*(?:\n|$))+", flags=re.MULTILINE)
_EXCESSIVE_LINE_BREAKS_PATTERN = re.compile(
    r"(\n[ \t]*){3,}", flags=re.MULTILINE
)
_HEADER_PATTERN = re.compile(r"^#+.*$", flags=re.MULTILINE)
_IMAGE_PATTERN = re.compile(r"\!\[.*?\]\(.*?\)")
_LINK_PATTERN = re.compile(r"\[.*?\]\(.*?\)")


def _ensure_str(content: str) -> str:
    if not isinstance(content, str):
        raise ValueError("Content must be a string")
    return content


def modify_markdown(content: str) -> str:
    """
    Modify Markdown content by applying a series of transformations.

    :param content: Markdown content as a string.
    :return: Modified Markdown content.
    """
    content = _ensure_str(content)
    content = remove_above_first_header_if_level_one(content)
    content = remove_inline_attachments(content)
    content = remove_attachments_header_and_first_line(content)
    content = convert_indented_blocks_to_code(content)
    content = reduce_excessive_line_breaks(content)
    return content.strip()


def remove_above_first_header_if_level_one(content: str) -> str:
    """
    Remove everything above the first Markdown header only if it is a level-one header.

    :param content: Markdown content as a string.
    :return: Markdown content without the section above the first level-one header.
    """
    content = _ensure_str(content)

    match = _FIRST_HEADER_PATTERN.search(content)
    if match and len(match.group(1)) == 1:
        return content[match.start() :]
    return content


def remove_attachments_header_and_first_line(content: str) -> str:
    """
    Remove the 'Attachments' header line and the line that immediately follows it.

    :param content: Markdown content as a string.
    :return: Markdown content without the 'Attachments' header and first line.
    """
    content = _ensure_str(content)

    return _ATTACHMENTS_HEADER_LINE_PATTERN.sub(r"\1", content)


def remove_attachments_section(content: str) -> str:
    """
    Remove the 'Attachments' subheader and everything that follows.

    :param content: Markdown content as a string.
    :return: Markdown content without the 'Attachments' section.
    """
    content = _ensure_str(content)

    # Split on the first occurrence of '\n## Attachments:' and keep the part before it.
    return re.split(r"\n## Attachments:", content, maxsplit=1)[0]


def remove_inline_attachments(content: str) -> str:
    """
    Remove inline attachments from the Markdown content.

    :param content: Markdown content as a string.
    :return: Markdown content without inline attachments.
    """
    content = _ensure_str(content)

    content = _INLINE_ATTACHMENTS_NESTED_PATTERN.sub("", content)
    return _INLINE_ATTACHMENTS_PATTERN.sub("", content)


def clean_up_lines(content: str) -> str:
    """
    Clean up lines in the Markdown content by performing two actions:
    1. Removing trailing spaces or tabs from every line.
    2. Removing lines that consist only of spaces or tabs.

    :param content: Markdown content as a string.
    :return: Markdown content with cleaned-up lines.
    """
    # Validate input
    content = _ensure_str(content)

    # Precompile regex patterns for efficiency
    # Remove trailing spaces or tabs from every line
    content = _TRAILING_SPACES_PATTERN.sub("", content)
    # Remove lines that consist only of spaces or tab characters
    content = _ONLY_SPACES_LINES_PATTERN.sub("", content)

    return content


def convert_indented_blocks_to_code(content: str) -> str:
    """
    Convert blocks of text indented by four spaces into Markdown code blocks.
    This transformation enhances readability and formatting in Markdown-rendered content.

    :param content: Markdown content as a string.
    :return: Markdown content with indented text blocks converted to code blocks.
    """
    # Validate input
    content = _ensure_str(content)

    if not content:
        return content

    # If the content starts with an indented line, wrap the whole block.
    if content.startswith("    "):
        stripped = content.rstrip("\n")
        return f"```\n{stripped}\n```\n"

    def wrap_block(match: re.Match[str]) -> str:
        block = match.group(0).rstrip("\n")
        return f"```\n{block}\n```\n"

    return _INDENTED_BLOCKS_PATTERN.sub(wrap_block, content)


def reduce_excessive_line_breaks(content: str) -> str:
    """
    Reduce instances of more than two consecutive line breaks in the Markdown content.
    This function ensures that the spacing in the rendered Markdown does not have
    excessive whitespace, which can improve readability.

    :param content: Markdown content as a string.
    :return: Markdown content with reduced excessive line breaks.
    """
    # Validate input
    content = _ensure_str(content)

    # Precompile regex pattern for excessive line breaks
    # Reduce excessive line breaks to two
    return _EXCESSIVE_LINE_BREAKS_PATTERN.sub("\n\n", content)


def strip_basic_markdown_formatting(content: str) -> str:
    """
    Remove basic Markdown formatting to leave plain text.

    This is intentionally conservative and mirrors the minimal formatting removal
    used by embedding preprocessing.
    """
    content = _ensure_str(content)

    content = _HEADER_PATTERN.sub("", content)
    content = _IMAGE_PATTERN.sub("", content)
    content = _LINK_PATTERN.sub("", content)
    return content


def preprocess_similarity_text(content: str) -> str:
    """
    Preprocess Markdown for embedding similarity comparisons.

    This mirrors the historical behavior in embeddings/similarity.py so tests and
    comparisons remain stable.
    """
    content = remove_attachments_section(content)
    content = remove_inline_attachments(content)
    content = clean_up_lines(content)
    content = reduce_excessive_line_breaks(content)
    content = strip_basic_markdown_formatting(content)
    return content
