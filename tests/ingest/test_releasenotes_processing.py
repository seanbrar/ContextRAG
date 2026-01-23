import os

import pytest

from contextrag.ingest import releasenotes_processing as rp


def test_extract_task_type_and_priority():
    assert rp.extract_task_type("![Bug](foo)") == "Bug"
    assert rp.extract_task_type("[![Task](x)](y)") == "Task"
    assert rp.extract_priority("icons/priorities/high.svg") == "High"


def test_process_release_notes_content_adds_urls(monkeypatch):
    monkeypatch.setattr(rp, "COMPANY_NAME", "example")
    content = "[EW-123] Fixed\n[ 2 issues ](https://example.com)\n"
    processed = rp.process_release_notes_content(content)
    assert "2 issues" in processed
    assert "https://example.atlassian.net/browse/EW-123" in processed


def test_add_url_to_key_no_duplicate(monkeypatch):
    monkeypatch.setattr(rp, "COMPANY_NAME", "example")
    entry = "[EW-555](https://example.atlassian.net/browse/EW-555)"
    assert rp.add_url_to_key(entry) == entry


def test_is_table_start_and_end():
    assert rp.is_table_start("| head |", "| --- |") is True
    assert rp.is_table_end("| row |") is False
    assert rp.is_table_end("") is True


def test_normalize_spacing_collapses_runs():
    assert rp.normalize_spacing("A  B   C") == "A B C"


def test_process_summary_strips_link():
    assert rp.process_summary("[Summary](https://example.com)") == "Summary"


def test_ensure_correct_url_format_expands_placeholder(monkeypatch):
    monkeypatch.setattr(rp, "COMPANY_NAME", "example")
    entry = "[EW-1](https://{COMPANY_NAME}.atlassian.net/browse/EW-1)"
    expected = "[EW-1](https://example.atlassian.net/browse/EW-1)"
    assert rp.ensure_correct_url_format(entry) == expected


def test_process_buffer_for_replacements(monkeypatch):
    monkeypatch.setattr(rp, "COMPANY_NAME", "example")
    entry = (
        "[EW-1](https://{COMPANY_NAME}.atlassian.net/browse/EW-1) "
        "[My summary](https://example.com) "
        "[![Bug](x)](y) "
        "![Priority](https://example.com/icons/priorities/high.svg) Fixed"
    )
    processed = rp.process_buffer_for_replacements([entry])
    assert "My summary" in processed
    assert "Bug" in processed
    assert "High" in processed
    assert "https://example.atlassian.net/browse/EW-1" in processed


def test_process_issue_counter_simplifies():
    content = "[ 2 issues ](https://example.com)"
    assert rp.process_issue_counter(content) == "2 issues"


def test_should_flush_buffer_for_header_or_new_entry():
    assert rp.should_flush_buffer("## Header", []) is True
    assert rp.should_flush_buffer("[EW-1] entry", ["buffered"])


def test_process_content_documentation_returns_input():
    assert rp.process_content("content", "documentation") == "content"


def test_add_url_to_key_inserts_when_missing(monkeypatch):
    monkeypatch.setattr(rp, "COMPANY_NAME", "example")
    entry = "[EW-9] Fixed"
    result = rp.add_url_to_key(entry)
    assert "https://example.atlassian.net/browse/EW-9" in result


def test_finalize_entries_adds_urls(monkeypatch):
    monkeypatch.setattr(rp, "COMPANY_NAME", "example")
    content = "[EW-9] Fixed\nNo key here"
    result = rp.finalize_entries(content)
    assert "https://example.atlassian.net/browse/EW-9" in result
