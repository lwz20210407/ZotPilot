"""Evaluate a reviewed formula Gold document against a PP-DocLayout cache."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zotpilot.feature_extraction.formula_gold_metrics import (
    evaluate_coco_map,
    evaluate_formula_gold,
    evaluate_formula_gold_corpus,
    to_coco,
)


def main() -> int:
    args = _arguments()
    gold = _json_object(Path(args.gold))
    if args.item_cache:
        if args.cache or args.item_key:
            raise ValueError("Use either --cache/--item-key or repeated --item-cache ITEM_KEY=PATH")
        if args.coco_map:
            raise ValueError("--coco-map currently supports one item only; run it per document")
        cache_paths = parse_item_caches(args.item_cache)
        report = evaluate_formula_gold_corpus(
            gold,
            {item_key: _json_object(cache_path) for item_key, cache_path in cache_paths.items()},
            iou_threshold=args.iou_threshold,
        )
        summary = {
            "output": str(args.output),
            "documents": report["document_count"],
            "precision": report["overall"]["precision"],
            "recall": report["overall"]["recall"],
        }
    else:
        if not args.cache or not args.item_key:
            raise ValueError("--cache and --item-key are required unless --item-cache is used")
        cache = _json_object(Path(args.cache))
        report = evaluate_formula_gold(gold, cache, item_key=args.item_key, iou_threshold=args.iou_threshold)
        if args.coco_map:
            coco_gold, predictions = to_coco(gold, cache, item_key=args.item_key)
            report["coco_map"] = evaluate_coco_map(coco_gold, predictions)
        summary = {
            "output": str(args.output),
            "precision": report["overall"]["precision"],
            "recall": report["overall"]["recall"],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Reviewed ZotPilot Label Studio Gold JSON")
    parser.add_argument("--cache", help="Source-bound PP-DocLayout cache JSON")
    parser.add_argument("--item-key", help="Zotero item key to evaluate")
    parser.add_argument(
        "--item-cache",
        action="append",
        default=[],
        metavar="ITEM_KEY=PATH",
        help="Repeat to aggregate explicit item-keyed visual-layout caches",
    )
    parser.add_argument("--output", required=True, help="Metrics report JSON")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="One-to-one matching IoU threshold")
    parser.add_argument("--coco-map", action="store_true", help="Also calculate optional pycocotools mAP")
    return parser.parse_args()


def _json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_item_caches(values: list[str]) -> dict[str, Path]:
    """Parse explicit item/cache bindings without inferring identity from paths."""
    parsed: dict[str, Path] = {}
    for value in values:
        item_key, separator, path_value = value.partition("=")
        item_key = item_key.strip()
        path_value = path_value.strip()
        if not separator or not item_key or not path_value:
            raise ValueError("Each --item-cache value must use ITEM_KEY=PATH")
        if item_key in parsed:
            raise ValueError(f"Duplicate item key in --item-cache: {item_key}")
        parsed[item_key] = Path(path_value)
    if not parsed:
        raise ValueError("At least one --item-cache ITEM_KEY=PATH value is required")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
