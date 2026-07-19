"""Evaluate a PP-DocLayout region cache against a tier-1 annotation file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zotpilot.feature_extraction.vision_layout.formula_region_metrics import evaluate_formula_regions


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute formula-region IoU precision/recall without touching Chroma.")
    parser.add_argument("annotation", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--item-key")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--draft-coordinate-space",
        choices=("auto", "pdf", "normalized_1000"),
        default="auto",
        help="Use annotation metadata when available; otherwise preserve the legacy PDF-point default.",
    )
    parser.add_argument("--include-inline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    report = evaluate_formula_regions(
        annotation,
        cache,
        item_key=args.item_key,
        iou_threshold=args.iou_threshold,
        draft_coordinate_space=args.draft_coordinate_space,
        display_only=not args.include_inline,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
