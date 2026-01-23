from contextrag.routing.category_extraction import extract_categories_from_file


def test_extract_categories_from_file(tmp_path):
    path = tmp_path / "out.txt"
    path.write_text(
        "Filename: a | Categories: Foo, Bar\nFilename: b | Categories: Bar, Baz\n",
        encoding="utf-8",
    )
    categories, count = extract_categories_from_file(path)
    assert count == 3
    for item in ["Foo", "Bar", "Baz"]:
        assert item in categories


def test_extract_categories_from_file_missing():
    categories, count = extract_categories_from_file("missing.txt")
    assert count == 0
    assert "File not found" in categories
