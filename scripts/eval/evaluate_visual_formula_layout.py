"""Evaluate cached PP-DocLayout regions through display filtering and number binding.

This command is intentionally read-only: it consumes an existing PDF and
source-bound PP-DocLayout cache, does not load a layout model, and never opens
ZotPilot's vector store.  It produces a compact JSON report suitable for
cross-document review before recognition or indexing is enabled.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from zotpilot.feature_extraction.formula_ocr import FormulaCandidate, PpDocLayoutFormulaCandidateProvider
from zotpilot.feature_extraction.vision_layout.pp_doclayout_candidate_cache import (
    load_pp_doclayout_formula_number_regions,
    load_pp_doclayout_formula_regions,
)


def main() -> int:
    args = _arguments()
    cache_paths = tuple(Path(path) for path in args.cache)
    detector_floor = min(args.min_confidence, 0.5)
    raw_regions = load_pp_doclayout_formula_regions(
        args.pdf,
        cache_paths,
        min_confidence=detector_floor,
    )
    strict_regions = load_pp_doclayout_formula_regions(
        args.pdf,
        cache_paths,
        min_confidence=args.min_confidence,
    )
    number_regions = load_pp_doclayout_formula_number_regions(
        args.pdf,
        cache_paths,
        min_confidence=args.min_confidence,
    )
    candidates = PpDocLayoutFormulaCandidateProvider().extract_candidates(
        args.pdf,
        cache_paths=cache_paths,
        min_confidence=args.min_confidence,
        max_formulas_per_doc=0,
        max_formulas_per_page=0,
        max_candidates_per_doc=0,
    )
    report = {
        "mode": "read_only_pp_doclayout_layout_evaluation",
        "pdf": str(args.pdf),
        "cache_paths": [str(path) for path in cache_paths],
        "min_confidence": args.min_confidence,
        "candidate_detector_floor": detector_floor,
        "raw_formula_region_count": len(raw_regions),
        "strict_formula_region_count": len(strict_regions),
        "raw_formula_number_region_count": len(number_regions),
        "display_candidate_count": len(candidates),
        "bound_equation_number_count": sum(bool(candidate.equation_number) for candidate in candidates),
        "unnumbered_display_candidate_count": sum(not candidate.equation_number for candidate in candidates),
        "quality_flag_counts": _quality_flag_counts(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "raw_formula_regions": len(raw_regions),
                "display_candidates": report["display_candidate_count"],
                "bound_numbers": report["bound_equation_number_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _quality_flag_counts(candidates: Iterable[FormulaCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        for flag in getattr(candidate, "quality_flags", ()):
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, help="Original source PDF")
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        help="Source-bound PP-DocLayout cache JSON; repeat when needed",
    )
    parser.add_argument("--output", type=Path, required=True, help="Read-only evaluation report JSON")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
