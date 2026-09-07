"""Evaluate PP-DocLayout formula-number bindings against reviewed Gold."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zotpilot.feature_extraction.formula_ocr import FormulaCandidate
from zotpilot.feature_extraction.vision_layout.equation_number_binding import (
    bind_pp_doclayout_equation_numbers,
    evaluate_equation_number_binding_gold,
)
from zotpilot.feature_extraction.vision_layout.formula_layout_filter import (
    classify_visual_formula_candidates,
    load_visual_page_columns,
)
from zotpilot.feature_extraction.vision_layout.pp_doclayout_candidate_cache import (
    load_pp_doclayout_formula_number_regions,
    load_pp_doclayout_formula_regions,
)


def main() -> int:
    args = _arguments()
    pdf_path = Path(args.pdf)
    regions = load_pp_doclayout_formula_regions(
        pdf_path,
        args.cache,
        min_confidence=args.min_confidence,
    )
    candidates = [
        FormulaCandidate(
            page_num=region.page_num,
            bbox=region.bbox,
            raw_text="",
            confidence=region.confidence,
            source="pp_doclayout_region",
            bbox_coordinate_space="pdf",
            equation_number_status="unbound",
            source_artifact_hash=region.source_artifact_hash,
        )
        for region in regions
    ]
    layout = classify_visual_formula_candidates(
        pdf_path, candidates,
        load_pp_doclayout_formula_number_regions(pdf_path, args.cache, min_confidence=args.min_confidence),
        min_confidence=min(args.min_confidence, 0.5),
        number_assisted_floor=min(args.min_confidence, 0.5),
        cached_columns_by_page=load_visual_page_columns(pdf_path, args.cache),
    )
    binding = bind_pp_doclayout_equation_numbers(
        pdf_path,
        layout.candidates,
        args.cache,
        min_confidence=args.min_confidence,
        columns_by_page=layout.columns_by_page,
    )
    report = evaluate_equation_number_binding_gold(
        _json_object(Path(args.gold)),
        binding,
        item_key=args.item_key,
        iou_threshold=args.iou_threshold,
    )
    report["binding_diagnostics"] = {
        "candidate_count": len(binding.candidates),
        "bound_numbers": list(binding.bound_numbers),
        "rejected_number_region_count": binding.rejected_number_region_count,
        "sequence_gap_numbers": list(binding.sequence_gap_numbers),
        "duplicate_numbers": list(binding.duplicate_numbers),
        "non_monotonic_pairs": [list(pair) for pair in binding.non_monotonic_pairs],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "precision": report["precision"],
                "recall": report["recall"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="Original source PDF")
    parser.add_argument("--gold", required=True, help="Reviewed Formula Gold JSON")
    parser.add_argument("--cache", action="append", required=True, help="Source-bound PP-DocLayout cache JSON")
    parser.add_argument("--item-key", required=True, help="Zotero item key in reviewed Gold")
    parser.add_argument("--output", required=True, help="Number-binding report JSON")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="PP-DocLayout confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Formula-block matching IoU threshold")
    return parser.parse_args()


def _json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
