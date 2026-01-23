import hashlib
import json
from pathlib import Path
from typing import Iterable


def iter_files(root: Path, extensions: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{json.dumps(row)}\n")


def checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
