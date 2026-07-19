"""Evaluate reviewed formula-region gold against a PP-DocLayout cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zotpilot.feature_extraction.formula_gold_metrics import evaluate_coco_map, evaluate_formula_gold, to_coco


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--item-key", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--coco-ground-truth", type=Path)
    parser.add_argument("--coco-predictions", type=Path)
    parser.add_argument("--coco-evaluate", action="store_true")
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    report = evaluate_formula_gold(gold, cache, item_key=args.item_key, iou_threshold=args.iou)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if bool(args.coco_ground_truth) != bool(args.coco_predictions):
        parser.error("--coco-ground-truth and --coco-predictions must be supplied together")
    if args.coco_ground_truth:
        coco_gold, coco_predictions = to_coco(gold, cache, item_key=args.item_key)
        args.coco_ground_truth.write_text(json.dumps(coco_gold, ensure_ascii=False, indent=2), encoding="utf-8")
        args.coco_predictions.write_text(json.dumps(coco_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.coco_evaluate:
            report["coco"] = evaluate_coco_map(coco_gold, coco_predictions)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.coco_evaluate:
        parser.error("--coco-evaluate requires --coco-ground-truth and --coco-predictions")


if __name__ == "__main__":
    main()
