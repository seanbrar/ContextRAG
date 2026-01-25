#!/usr/bin/env python3
import argparse
import urllib.error
import urllib.request
from pathlib import Path

RFC_URL = "https://www.rfc-editor.org/rfc/rfc{rfc_id}.txt"


def download_rfc(rfc_id: str, output_dir: Path) -> None:
    url = RFC_URL.format(rfc_id=rfc_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"rfc{rfc_id}.txt"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to download RFC {rfc_id}: {exc}") from exc

    target_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download RFC text files.")
    parser.add_argument(
        "--rfcs",
        required=True,
        help="Comma-separated RFC numbers (e.g., 822,9110,9595).",
    )
    parser.add_argument(
        "--output",
        default="data/rfc/raw",
        help="Output directory for RFC text files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    rfcs = [item.strip() for item in args.rfcs.split(",") if item.strip()]
    if not rfcs:
        raise SystemExit("Provide at least one RFC number via --rfcs.")

    for rfc_id in rfcs:
        download_rfc(rfc_id, output_dir)
        print(f"Downloaded RFC {rfc_id}")


if __name__ == "__main__":
    main()
