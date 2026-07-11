"""Candidate-level consensus review for formula parser outputs."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from typing import Any

from .formula_semantic_evidence import normalize_equation_number

STRUCTURED_SOURCE_GROUPS = {
    "docling",
    "marker",
    "mineru_cache",
    "monkeyocr",
    "ocrflux",
    "olmocr",
    "pdf_extract_kit",
    "pix2text",
}

_FORMULA_SIGNAL_RE = re.compile(
    r"(?:[=<>≤≥≈∑∫√]|\\(?:frac|sum|int|sqrt|sigma|epsilon|varepsilon|eta|theta)|[α-ωΑ-Ω])",
    re.IGNORECASE,
)
_SIGNATURE_GREEK_REPLACEMENTS = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "η": "eta",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Δ": "delta",
    "∆": "delta",
    "Σ": "sigma",
}
_SIGNATURE_LATEX_REPLACEMENTS = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\varepsilon": "epsilon",
    r"\eta": "eta",
    r"\theta": "theta",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\nu": "nu",
    r"\xi": "xi",
    r"\pi": "pi",
    r"\rho": "rho",
    r"\sigma": "sigma",
    r"\tau": "tau",
    r"\phi": "phi",
    r"\chi": "chi",
    r"\psi": "psi",
    r"\omega": "omega",
    r"\Delta": "delta",
    r"\Sigma": "sigma",
    r"\cdot": "*",
    r"\times": "*",
    r"\leq": "<=",
    r"\le": "<=",
    r"\geq": ">=",
    r"\ge": ">=",
    r"\approx": "≈",
}


def build_formula_candidate_consensus(candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Group candidate previews into formula-level consensus clusters."""
    candidate_rows = [_candidate_row(candidate) for candidate in candidates if isinstance(candidate, Mapping)]
    if not candidate_rows:
        return {
            "mode": "no_candidate_preview",
            "preview_candidate_count": 0,
            "cluster_count": 0,
            "multi_provider_cluster_count": 0,
            "single_provider_cluster_count": 0,
            "conflict_cluster_count": 0,
            "supported_cluster_count": 0,
            "review_cluster_count": 0,
            "single_provider_review_cluster_count": 0,
            "conflict_review_cluster_count": 0,
            "ocr_fallback_cluster_count": 0,
            "unlocated_candidate_count": 0,
            "clusters": [],
        }

    clusters: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        cluster = _matching_cluster(clusters, candidate)
        if cluster is None:
            clusters.append({"candidates": [candidate]})
        else:
            cluster["candidates"].append(candidate)

    cluster_rows = [_summarize_cluster(index + 1, cluster["candidates"]) for index, cluster in enumerate(clusters)]
    conflict_count = sum(1 for cluster in cluster_rows if cluster["conflict_flags"])
    supported_count = sum(1 for cluster in cluster_rows if cluster["cluster_route"] == "supported_candidate")
    return {
        "mode": "candidate_preview_consensus",
        "preview_candidate_count": len(candidate_rows),
        "cluster_count": len(cluster_rows),
        "multi_provider_cluster_count": sum(
            1 for cluster in cluster_rows
            if int(cluster["provider_group_count"]) >= 2 or int(cluster["parser_label_count"]) >= 2
        ),
        "single_provider_cluster_count": sum(
            1 for cluster in cluster_rows
            if int(cluster["provider_group_count"]) == 1
        ),
        "conflict_cluster_count": conflict_count,
        "supported_cluster_count": supported_count,
        "review_cluster_count": len(cluster_rows) - supported_count,
        "single_provider_review_cluster_count": sum(
            1 for cluster in cluster_rows
            if cluster["cluster_route"] == "single_provider_review"
        ),
        "conflict_review_cluster_count": sum(
            1 for cluster in cluster_rows
            if cluster["cluster_route"] == "conflict_review"
        ),
        "ocr_fallback_cluster_count": sum(
            1 for cluster in cluster_rows
            if "ocr_fallback_required" in cluster["review_flags"]
        ),
        "unlocated_candidate_count": sum(
            1 for candidate in candidate_rows
            if not candidate["page_num"] or len(candidate["bbox"]) != 4
        ),
        "clusters": cluster_rows,
    }


def source_group(source: str) -> str:
    """Return a stable parser family label for a candidate source string."""
    if source == "text_layer" or source.startswith("pdf_text"):
        return "pdf_text_layer"
    if source.startswith("mineru_"):
        return "mineru_cache"
    if source.startswith("pdf_extract_kit_"):
        return "pdf_extract_kit"
    if source.startswith("pix2text_"):
        return "pix2text"
    if source.startswith("marker_"):
        return "marker"
    if source.startswith("docling_"):
        return "docling"
    if source.startswith("monkeyocr_"):
        return "monkeyocr"
    if source.startswith("ocrflux_"):
        return "ocrflux"
    if source.startswith("olmocr_"):
        return "olmocr"
    return "other"


def _candidate_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    source = str(candidate.get("source", "") or "")
    latex = str(candidate.get("latex_preview", "") or "")
    raw_text = str(candidate.get("raw_text_preview", "") or "")
    equation_number = str(candidate.get("equation_number", "") or "")
    return {
        "candidate_index": _int_value(candidate.get("candidate_index")),
        "page_num": _int_value(candidate.get("page_num")),
        "bbox": _bbox_value(candidate.get("bbox")),
        "source": source,
        "source_group": source_group(source),
        "parser_label": str(candidate.get("parser_label", "") or ""),
        "equation_number": equation_number,
        "normalized_equation_number": normalize_equation_number(equation_number),
        "latex_signature": _formula_signature(latex or raw_text),
        "has_latex": bool(candidate.get("has_latex")),
        "needs_ocr": bool(candidate.get("needs_ocr")),
        "confidence": _float_value(candidate.get("confidence")),
        "latex_preview": latex,
        "raw_text_preview": raw_text,
    }


def _matching_cluster(clusters: list[dict[str, Any]], candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    for cluster in clusters:
        existing_candidates = cluster["candidates"]
        if any(_same_formula_candidate(candidate, existing) for existing in existing_candidates):
            return cluster
    return None


def _same_formula_candidate(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_number = str(left.get("normalized_equation_number", "") or "")
    right_number = str(right.get("normalized_equation_number", "") or "")
    if left_number and right_number and left_number == right_number:
        return True
    left_page = _int_value(left.get("page_num"))
    right_page = _int_value(right.get("page_num"))
    if left_page and left_page == right_page:
        left_bbox = _bbox_value(left.get("bbox"))
        right_bbox = _bbox_value(right.get("bbox"))
        if len(left_bbox) == 4 and len(right_bbox) == 4 and _bbox_iou(left_bbox, right_bbox) >= 0.2:
            return True
        left_signature = str(left.get("latex_signature", "") or "")
        right_signature = str(right.get("latex_signature", "") or "")
        if left_signature and right_signature and left_signature == right_signature:
            return True
        if _formula_signatures_similar(left_signature, right_signature):
            return True
    return False


def _summarize_cluster(index: int, candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(str(candidate["source"]) for candidate in candidates)
    group_counts = Counter(str(candidate["source_group"]) for candidate in candidates)
    parser_label_counts = Counter(
        str(candidate["parser_label"])
        for candidate in candidates
        if candidate["parser_label"]
    )
    page_nums = sorted({int(candidate["page_num"]) for candidate in candidates if int(candidate["page_num"]) > 0})
    equation_numbers = [
        str(candidate["equation_number"])
        for candidate in candidates
        if str(candidate["equation_number"] or "")
    ]
    normalized_numbers = {
        str(candidate["normalized_equation_number"])
        for candidate in candidates
        if str(candidate["normalized_equation_number"] or "")
    }
    signatures = {
        str(candidate["latex_signature"])
        for candidate in candidates
        if str(candidate["latex_signature"] or "")
    }
    flags = _cluster_review_flags(
        candidates,
        group_counts=group_counts,
        page_nums=page_nums,
        normalized_numbers=normalized_numbers,
        signatures=signatures,
    )
    conflict_flags = [
        flag for flag in flags
        if flag in {"equation_number_conflict", "same_number_multiple_pages", "latex_signature_conflict"}
    ]
    cluster_route = _cluster_route(flags, conflict_flags)
    return {
        "cluster_id": f"formula_cluster_{index:04d}",
        "cluster_route": cluster_route,
        "candidate_count": len(candidates),
        "source_counts": dict(sorted(source_counts.items())),
        "source_group_counts": dict(sorted(group_counts.items())),
        "parser_label_counts": dict(sorted(parser_label_counts.items())),
        "source_group_count": len([count for count in group_counts.values() if count]),
        "provider_group_count": len([count for count in group_counts.values() if count]),
        "parser_label_count": len([count for count in parser_label_counts.values() if count]),
        "structured_provider_group_count": sum(
            1 for group, count in group_counts.items()
            if count and group in STRUCTURED_SOURCE_GROUPS
        ),
        "page_nums": page_nums,
        "equation_numbers": sorted(set(equation_numbers)),
        "primary_equation_number": sorted(set(equation_numbers))[0] if equation_numbers else "",
        "has_latex_count": sum(1 for candidate in candidates if candidate["has_latex"]),
        "needs_ocr_count": sum(1 for candidate in candidates if candidate["needs_ocr"]),
        "representative_latex_preview": _first_non_empty(
            str(candidate["latex_preview"] or "")
            for candidate in candidates
        ),
        "representative_raw_text_preview": _first_non_empty(
            str(candidate["raw_text_preview"] or "")
            for candidate in candidates
        ),
        "review_flags": flags,
        "conflict_flags": conflict_flags,
        "candidate_indices": [int(candidate["candidate_index"]) for candidate in candidates],
        "candidate_details": [
            {
                "candidate_index": int(candidate["candidate_index"]),
                "parser_label": str(candidate["parser_label"] or ""),
                "source": str(candidate["source"] or ""),
                "source_group": str(candidate["source_group"] or ""),
                "page_num": int(candidate["page_num"]),
                "equation_number": str(candidate["equation_number"] or ""),
                "has_latex": bool(candidate["has_latex"]),
                "needs_ocr": bool(candidate["needs_ocr"]),
                "latex_preview": str(candidate["latex_preview"] or ""),
                "raw_text_preview": str(candidate["raw_text_preview"] or ""),
            }
            for candidate in candidates
        ],
    }


def _cluster_route(flags: list[str], conflict_flags: list[str]) -> str:
    if conflict_flags:
        return "conflict_review"
    if "multi_provider_agreement" in flags:
        return "supported_candidate"
    if "single_provider_only" in flags:
        return "single_provider_review"
    return "manual_review"


def _cluster_review_flags(
    candidates: list[Mapping[str, Any]],
    *,
    group_counts: Mapping[str, int],
    page_nums: list[int],
    normalized_numbers: set[str],
    signatures: set[str],
) -> list[str]:
    flags: list[str] = []
    parser_label_count = len({
        str(candidate.get("parser_label", "") or "")
        for candidate in candidates
        if str(candidate.get("parser_label", "") or "")
    })
    if len([count for count in group_counts.values() if count]) >= 2 or parser_label_count >= 2:
        flags.append("multi_provider_agreement")
    else:
        flags.append("single_provider_only")
    if not any(group in STRUCTURED_SOURCE_GROUPS for group in group_counts):
        flags.append("no_structured_parser_evidence")
    if len(normalized_numbers) > 1:
        flags.append("equation_number_conflict")
    if len(normalized_numbers) == 1 and len(page_nums) > 1:
        flags.append("same_number_multiple_pages")
    if len(signatures) > 1 and len([count for count in group_counts.values() if count]) >= 2:
        flags.append("latex_signature_conflict")
    if any(not int(candidate["page_num"]) for candidate in candidates):
        flags.append("missing_page")
    if any(len(candidate["bbox"]) != 4 for candidate in candidates):
        flags.append("missing_bbox")
    if any(not str(candidate["normalized_equation_number"] or "") for candidate in candidates):
        flags.append("missing_equation_number")
    if any(bool(candidate["needs_ocr"]) for candidate in candidates):
        flags.append("ocr_fallback_required")
    return flags


def _formula_signature(value: str) -> str:
    compact = re.sub(r"\s+", "", value or "").lower()
    compact = compact.strip("$")
    compact = _normalize_formula_signature(compact)
    if len(compact) < 4:
        return ""
    if len(compact) > 320:
        return ""
    if not _FORMULA_SIGNAL_RE.search(compact):
        return ""
    return compact


def _normalize_formula_signature(value: str) -> str:
    normalized = value or ""
    replacements = sorted(
        _SIGNATURE_LATEX_REPLACEMENTS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for latex, replacement in replacements:
        normalized = normalized.replace(latex.lower(), replacement)
    for greek, replacement in _SIGNATURE_GREEK_REPLACEMENTS.items():
        normalized = normalized.replace(greek, replacement)
    normalized = re.sub(r"\\(?:left|right|mathrm|mathbf|mathit|text|operatorname)", "", normalized)
    normalized = normalized.replace("¼", "=").replace("þ", "+").replace("−", "-")
    return normalized


def _formula_signatures_similar(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = _formula_signature_tokens(left)
    right_tokens = _formula_signature_tokens(right)
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        return False
    overlap = len(set(left_tokens) & set(right_tokens))
    jaccard = overlap / max(len(set(left_tokens) | set(right_tokens)), 1)
    if overlap >= 4 and jaccard >= 0.72:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.86 and jaccard >= 0.58


def _formula_signature_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+|[=<>≤≥≈+\-*/_^{}/]+", value or "")


def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    left_x0, left_y0, left_x1, left_y1 = _ordered_bbox(left)
    right_x0, right_y0, right_x1, right_y1 = _ordered_bbox(right)
    intersection_width = max(0.0, min(left_x1, right_x1) - max(left_x0, right_x0))
    intersection_height = max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0))
    intersection_area = intersection_width * intersection_height
    if intersection_area <= 0:
        return 0.0
    left_area = max(0.0, left_x1 - left_x0) * max(0.0, left_y1 - left_y0)
    right_area = max(0.0, right_x1 - right_x0) * max(0.0, right_y1 - right_y0)
    union_area = left_area + right_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def _ordered_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _bbox_value(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    try:
        return tuple(float(part) for part in value)
    except (TypeError, ValueError):
        return ()


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _first_non_empty(values: Iterable[str]) -> str:
    for value in values:
        if value:
            return value
    return ""
