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
    to_coco,
)


def main() -> int:
    args = _arguments()
    gold = _json_object(Path(args.gold))
    cache = _json_object(Path(args.cache))
    report = evaluate_formula_gold(gold, cache, item_key=args.item_key, iou_threshold=args.iou_threshold)
    if args.coco_map:
        coco_gold, predictions = to_coco(gold, cache, item_key=args.item_key)
        report["coco_map"] = evaluate_coco_map(coco_gold, predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "precision": report["overall"]["precision"],
                "recall": report["overall"]["recall"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Reviewed ZotPilot Label Studio Gold JSON")
    parser.add_argument("--cache", required=True, help="Source-bound PP-DocLayout cache JSON")
    parser.add_argument("--item-key", required=True, help="Zotero item key to evaluate")
    parser.add_argument("--output", required=True, help="Metrics report JSON")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="One-to-one matching IoU threshold")
    parser.add_argument("--coco-map", action="store_true", help="Also calculate optional pycocotools mAP")
    return parser.parse_args()


def _json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
