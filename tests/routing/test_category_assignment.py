from contextrag.routing import category_assignment


def test_read_markdown_files_filters_md(tmp_path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    files = category_assignment.read_markdown_files(str(tmp_path))
    assert list(files.keys()) == ["a.md"]


def test_main_processes_and_skips(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    files = {
        "short.md": "short",
        "medium.md": "medium",
        "long.md": "long",
    }

    def fake_count_tokens(text):
        return {"short": 10, "medium": 4000, "long": 20000}[text]

    class FakeChatManager:
        def __init__(self):
            self.calls = []
            self.resets = 0

        def complete(self, model, user_message, system_message, temperature=0):
            self.calls.append(model)
            if "short" in user_message:
                content = '"""Categories: Alpha, Beta"""'
            else:
                content = "No categories here."

            class Response:
                choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]

            return Response()

        def reset(self):
            self.resets += 1

    class FakeDateTime:
        @staticmethod
        def now():
            class FakeNow:
                def strftime(self, fmt):
                    return "2024-01-01T00-00-00"

            return FakeNow()

    monkeypatch.setattr(category_assignment, "read_markdown_files", lambda: files)
    monkeypatch.setattr(category_assignment, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(category_assignment, "preprocess_text", lambda text: text)
    monkeypatch.setattr(category_assignment, "ChatManager", FakeChatManager)
    monkeypatch.setattr(category_assignment, "datetime", FakeDateTime)

    category_assignment.main()
    output = (tmp_path / "output_2024-01-01T00-00-00.txt").read_text(
        encoding="utf-8"
    )
    assert "short.md" in output
    assert "Categories: Alpha, Beta" in output
    assert "medium.md" in output
    assert "No categories found" in output
    assert "Skipped long.md due to excessive token count" in capsys.readouterr().out
