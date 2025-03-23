import pytest
from data_processing.markdown_processing import (
    modify_markdown,
    remove_above_first_header,
    remove_attachments_section,
    remove_inline_attachments,
    clean_up_lines,
    convert_indented_blocks_to_code,
    reduce_excessive_line_breaks,
)

# 1. Function-Specific Tests


# Test remove_above_first_header
@pytest.mark.parametrize(
    "input_content,expected_output",
    [
        ("Some text\n# Header\nContent", "# Header\nContent"),
        ("# Header\nContent", "# Header\nContent"),
        ("Content without header", "Content without header"),
        ("## Lesser Header\nContent", "## Lesser Header\nContent"),
        ("### Another Level\nContent", "### Another Level\nContent"),
        ("", ""),  # Test empty input
    ],
)
def test_remove_above_first_header(input_content, expected_output):
    """
    Test the remove_above_first_header function with various scenarios:
    - Text before and after the first header.
    - Only header and content.
    - Content without any headers.
    - Headers with different levels.
    - Empty input.
    """
    result = remove_above_first_header(input_content)
    assert result == expected_output
    assert isinstance(result, str)  # Check return type


# Test remove_attachments_section
@pytest.mark.parametrize(
    "input_content,expected_output",
    [
        ("Content\n## Attachments:\nAttachment", "Content"),
        ("Content without attachments", "Content without attachments"),
        ("## Attachments: in a sentence", "## Attachments: in a sentence"),
        (
            "Content\n## Attachments:\nAttachment\n## Attachments:\nAnother Attachment",
            "Content",
        ),
        ("", ""),  # Test empty input
    ],
)
def test_remove_attachments_section(input_content, expected_output):
    """
    Test the remove_attachments_section function with various scenarios:
    - Content with an 'Attachments' section.
    - Content without an 'Attachments' section.
    - 'Attachments' appearing in a sentence.
    - Multiple 'Attachments' sections.
    - Empty input.
    """
    result = remove_attachments_section(input_content)
    assert result == expected_output
    assert isinstance(result, str)  # Check return type


# Test remove_inline_attachments
@pytest.mark.parametrize(
    "input_content,expected_output",
    [
        ("Content with ![Image](attachments/image.jpg)", "Content with "),
        ("No inline attachments here", "No inline attachments here"),
        ("Nested ![Image](attachments/image.jpg) inline", "Nested  inline"),
        ("![Another](attachments/file.jpg) format", " format"),
        ("", ""),  # Test empty input
    ],
)
def test_remove_inline_attachments(input_content, expected_output):
    """
    Test the remove_inline_attachments function with various scenarios:
    - Content with different formats of inline attachments.
    - Nested inline attachments.
    - No inline attachments.
    - Different inline attachment format.
    - Empty input.
    """
    result = remove_inline_attachments(input_content)
    assert result == expected_output
    assert isinstance(result, str)  # Check return type


# Test clean_up_lines
@pytest.mark.parametrize(
    "input_content,expected_output",
    [
        ("Line with space at end ", "Line with space at end"),
        ("Line with tab at end\t", "Line with tab at end"),
        ("    ", ""),
        ("Line with mixed spaces and tabs \t", "Line with mixed spaces and tabs"),
        ("", ""),  # Test empty input
    ],
)
def test_clean_up_lines(input_content, expected_output):
    """
    Test the clean_up_lines function with various scenarios:
    - Lines with trailing spaces or tabs.
    - Line consisting only of spaces or tabs.
    - Line with mixed spaces and tabs.
    - Empty input.
    """
    result = clean_up_lines(input_content)
    assert result == expected_output
    assert isinstance(result, str)  # Check return type


# Test convert_indented_blocks_to_code
@pytest.mark.parametrize(
    "input_content,expected_output",
    [
        (
            "    Indented line\nAnother line",
            "```\n    Indented line\nAnother line\n```\n",
        ),
        ("Not indented", "Not indented"),
        (
            "    Indented\n\n    Another indented",
            "```\n    Indented\n\n    Another indented\n```\n",
        ),
        (
            "Mixed indentation\n    Indented",
            "Mixed indentation\n```\n    Indented\n```\n",
        ),
        ("", ""),  # Test empty input
    ],
)
def test_convert_indented_blocks_to_code(input_content, expected_output):
    """
    Test the convert_indented_blocks_to_code function with various scenarios:
    - Indented text blocks.
    - Mixed indented and non-indented blocks.
    - Multiple indented blocks.
    - Empty input.
    """
    result = convert_indented_blocks_to_code(input_content)
    assert result == expected_output
    assert isinstance(result, str)  # Check return type


# Test reduce_excessive_line_breaks
@pytest.mark.parametrize(
    "input_content,expected_output",
    [
        ("Line\n\n\nAnother line", "Line\n\nAnother line"),
        ("Line\nAnother line", "Line\nAnother line"),
        ("Line\n\n\n\n    \n\nAnother line", "Line\n\nAnother line"),
        ("Line\n    \n\n\nAnother line", "Line\n\nAnother line"),
        ("", ""),  # Test empty input
    ],
)
def test_reduce_excessive_line_breaks(input_content, expected_output):
    """
    Test the reduce_excessive_line_breaks function with various scenarios:
    - More than two consecutive line breaks.
    - Line breaks with spaces or tabs.
    - No excessive line breaks.
    - Empty input.
    """
    result = reduce_excessive_line_breaks(input_content)
    assert result == expected_output
    assert isinstance(result, str)  # Check return type


# 2. Integration Test


@pytest.mark.parametrize(
    "input_md,expected_md",
    [
        # Test case 1: Basic scenario with header and code block
        (
            "Text before header\n"
            "# Header\n"
            "Content with ![Image](attachments/image.jpg)\n"
            "## Attachments:\n"
            "Attachment content\n"
            "    Indented code block\n"
            "Extra\n\n\n"
            "line breaks",
            "# Header\n"
            "Content with \n"
            "```\n"
            "    Indented code block\n"
            "```\n"
            "Extra\n\n"
            "line breaks",
        ),
        # Test case 2: No modifications needed
        (
            "This is a regular paragraph.\n" "## Subheader\n" "Another paragraph.\n",
            "This is a regular paragraph.\n" "## Subheader\n" "Another paragraph.",
        ),
        # Test case 3: Empty input
        ("", ""),
        # Test case 4: Header without content
        ("# Empty Header\n" "## Subheader\n", "# Empty Header\n" "## Subheader"),
        # Test case 5: Code block only
        (
            "Code block:\n" "    Indented code block\n",
            "Code block:\n" "```\n" "    Indented code block\n" "```",
        ),
        # Add more test cases as needed
    ],
)
def test_modify_markdown_integration(input_md, expected_md):
    """
    Integration test for modify_markdown function with various Markdown scenarios:
    - Multiple elements combined.
    - Different element combinations.
    - Edge cases.
    - Empty input.
    """
    assert modify_markdown(input_md).strip() == expected_md.strip()
