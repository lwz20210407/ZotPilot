"""Export PP-DocLayout formula predictions as Label Studio preannotations."""

from __future__ import annotations

import argparse
from pathlib import Path

from zotpilot.feature_extraction.formula_gold import export_label_studio_tasks, write_label_studio_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--item-key", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    tasks = export_label_studio_tasks(
        args.pdf,
        args.cache,
        args.images,
        item_key=args.item_key,
        dpi=args.dpi,
    )
    write_label_studio_tasks(tasks, args.output)


if __name__ == "__main__":
    main()
