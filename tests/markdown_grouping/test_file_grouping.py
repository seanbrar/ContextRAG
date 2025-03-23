import pytest
from markdown_grouping.file_grouping import preprocess_text


def test_preprocess_text():
    # Input text with markdown formatting
    input_text = "# Title\nThis is a [link](http://example.com) and an ![image](image.png).\nSome text."

    # Expected output text after preprocessing
    expected_output = "\nThis is a  and an .\nSome text."

    # Call the preprocess_text function
    processed_text = preprocess_text(input_text)

    # Check if the processed_text matches the expected_output
    assert processed_text == expected_output
