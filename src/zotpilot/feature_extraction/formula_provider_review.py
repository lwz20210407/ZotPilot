"""Read-only provider cross-review for formula candidate reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

STRUCTURED_PROVIDER_GROUPS = {
    "docling",
    "marker",
    "mineru_cache",
    "monkeyocr",
    "ocrflux",
    "olmocr",
    "pdf_extract_kit",
    "pix2text",
}


def build_formula_provider_cross_review(estimate: Mapping[str, Any]) -> dict[str, Any]:
    """Build a read-only cross-provider review report from a formula estimate."""
    rows = [
        _provider_cross_review_row(
            row,
            provider=str(estimate.get("provider", "") or ""),
            route_info=_route_info_for_row(estimate, row),
        )
        for row in _list_value(estimate.get("results"))
        if isinstance(row, Mapping)
    ]
    route_counts = Counter(str(row.get("quality_route", "") or "unknown") for row in rows)
    provider_group_totals: Counter[str] = Counter()
    for row in rows:
        provider_group_totals.update(_dict_value(row.get("provider_evidence")).get("source_group_counts", {}))
    external_call_count = sum(_int_value(row.get("simpletex_external_call_count")) for row in rows)
    return {
        "mode": "read_only_provider_cross_review",
        "paper_count": len(rows),
        "candidate_count": _int_value(estimate.get("candidate_count")),
        "provider": str(estimate.get("provider", "") or ""),
        "candidate_provider": str(estimate.get("candidate_provider", "") or ""),
        "simpletex_role": (
            "fallback_recognizer_only"
            if str(estimate.get("provider", "") or "") == "simpletex"
            else "not_selected"
        ),
        "external_call_count": external_call_count,
        "route_counts": dict(sorted(route_counts.items())),
        "provider_group_totals": dict(sorted(provider_group_totals.items())),
        "semantic_unmatched_reference_paper_count": sum(
            1
            for row in rows
            if _int_value(_dict_value(row.get("semantic_evidence")).get("unmatched_reference_count")) > 0
        ),
        "write_blocked": bool(estimate.get("write_blocked")),
        "write_review_required": bool(estimate.get("write_review_required")),
        "readonly_index_changed": bool(estimate.get("readonly_index_changed")),
        "rows": rows,
    }


def _provider_cross_review_row(
    row: Mapping[str, Any],
    *,
    provider: str,
    route_info: Mapping[str, Any],
) -> dict[str, Any]:
    audit = _dict_value(row.get("candidate_audit"))
    source_counts = _dict_value(audit.get("source_counts"))
    source_group_counts = _source_group_counts(source_counts)
    semantic_evidence = _semantic_summary(row)
    simpletex_external_call_count = (
        _int_value(row.get("estimated_external_calls"))
        if provider == "simpletex"
        else 0
    )
    quality_route = str(route_info.get("quality_route", "") or _fallback_quality_route(row))
    review_flags = _review_flags(
        audit=audit,
        semantic_evidence=semantic_evidence,
        source_group_counts=source_group_counts,
        simpletex_external_call_count=simpletex_external_call_count,
    )
    return {
        "item_key": str(row.get("item_key", "") or ""),
        "title": str(row.get("title", "") or ""),
        "quality_route": quality_route,
        "route_reason": str(route_info.get("route_reason", "") or ""),
        "candidate_count": _int_value(row.get("candidate_count")),
        "provider_evidence": {
            "source_counts": dict(sorted(source_counts.items())),
            "source_group_counts": dict(sorted(source_group_counts.items())),
            "source_group_count": len([count for count in source_group_counts.values() if count]),
            "structured_parser_candidate_count": sum(
                count
                for group, count in source_group_counts.items()
                if group in STRUCTURED_PROVIDER_GROUPS
            ),
            "pdf_text_layer_candidate_count": _int_value(source_group_counts.get("pdf_text_layer")),
            "ocr_fallback_candidate_count": _int_value(audit.get("ocr_needed_count")),
            "cached_latex_candidate_count": _int_value(audit.get("cached_latex_count")),
        },
        "traceability": {
            "page_min": _int_value(audit.get("page_min")),
            "page_max": _int_value(audit.get("page_max")),
            "page_count_with_candidates": _int_value(audit.get("page_count_with_candidates")),
            "page_tagged_count": _int_value(audit.get("page_tagged_count")),
            "page_missing_count": _int_value(audit.get("page_missing_count")),
            "bbox_present_count": _int_value(audit.get("bbox_present_count")),
            "bbox_missing_count": _int_value(audit.get("bbox_missing_count")),
            "numbered_count": _int_value(audit.get("numbered_count")),
            "unnumbered_count": _int_value(audit.get("unnumbered_count")),
            "first_equation_number": str(audit.get("first_equation_number", "") or ""),
            "last_equation_number": str(audit.get("last_equation_number", "") or ""),
            "equation_number_warnings": _list_value(audit.get("equation_number_warnings")),
        },
        "semantic_evidence": semantic_evidence,
        "review_flags": sorted(set(review_flags)),
        "simpletex_external_call_count": simpletex_external_call_count,
        "write_recommendation": _write_recommendation(quality_route, review_flags),
    }


def _source_group_counts(source_counts: Mapping[str, Any]) -> Counter[str]:
    groups: Counter[str] = Counter()
    for source, count in source_counts.items():
        groups[_source_group(str(source))] += _int_value(count)
    return groups


def _source_group(source: str) -> str:
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


def _semantic_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _dict_value(row.get("semantic_formula_evidence"))
    return {
        "source": str(evidence.get("source", "") or ""),
        "mode": str(evidence.get("mode", "") or ""),
        "evidence_count": _int_value(row.get("semantic_formula_evidence_count", evidence.get("evidence_count"))),
        "unmatched_reference_count": _int_value(
            row.get("semantic_formula_unmatched_reference_count", evidence.get("unmatched_reference_count"))
        ),
        "unmatched_reference_numbers": _list_value(
            row.get("semantic_formula_unmatched_reference_numbers", evidence.get("unmatched_reference_numbers"))
        ),
        "reference_match_status": str(
            row.get("semantic_formula_reference_match_status", evidence.get("reference_match_status", "")) or ""
        ),
        "reference_coverage_ratio": _float_value(
            row.get("semantic_formula_reference_coverage_ratio", evidence.get("reference_coverage_ratio", 0.0))
        ),
        "review_flags": _list_value(
            row.get("semantic_formula_review_flags", evidence.get("review_flags"))
        ),
    }


def _review_flags(
    *,
    audit: Mapping[str, Any],
    semantic_evidence: Mapping[str, Any],
    source_group_counts: Mapping[str, int],
    simpletex_external_call_count: int,
) -> list[str]:
    flags = [str(flag) for flag in _list_value(audit.get("equation_number_warnings"))]
    flags.extend(str(flag) for flag in _list_value(semantic_evidence.get("review_flags")))
    if _int_value(semantic_evidence.get("unmatched_reference_count")):
        flags.append("semantic_unmatched_references")
    candidate_count = _int_value(audit.get("candidate_count"))
    if candidate_count and len([count for count in source_group_counts.values() if count]) <= 1:
        flags.append("single_detection_source")
    if candidate_count and not any(group in STRUCTURED_PROVIDER_GROUPS for group in source_group_counts):
        flags.append("no_structured_parser_evidence")
    if candidate_count and _int_value(source_group_counts.get("pdf_text_layer")) == candidate_count:
        flags.append("pdf_text_layer_only")
    if _int_value(audit.get("bbox_missing_count")):
        flags.append("missing_bbox")
    if _int_value(audit.get("page_missing_count")):
        flags.append("missing_page")
    if _int_value(audit.get("ocr_needed_count")):
        flags.append("ocr_fallback_required")
    if simpletex_external_call_count:
        flags.append("simpletex_external_fallback")
    return flags


def _write_recommendation(quality_route: str, review_flags: list[str]) -> str:
    if quality_route == "auto_candidate" and not _blocking_review_flags(review_flags):
        return "auto_candidate_after_reviewed_batch"
    if quality_route == "no_formula_candidate":
        return "skip_no_formula_candidate"
    if quality_route in {"skipped", "failed", "deferred_high_density"}:
        return quality_route
    return "manual_review_queue"


def _blocking_review_flags(review_flags: list[str]) -> set[str]:
    blocking_prefixes = (
        "cached_latex_",
        "duplicate_equation",
        "fallback_truncated",
        "large_equation",
        "missing_equation",
        "missing_bbox",
        "missing_page",
        "no_structured_parser_evidence",
        "pdf_text_layer_only",
        "semantic_unmatched",
        "structured_cache_high_density",
        "text_layer_high_density",
    )
    return {
        flag for flag in review_flags
        if flag.startswith(blocking_prefixes)
    }


def _route_info_for_row(estimate: Mapping[str, Any], row: Mapping[str, Any]) -> Mapping[str, Any]:
    item_key = str(row.get("item_key", "") or "")
    for route_row in _list_value(estimate.get("formula_quality_route_summary")):
        if isinstance(route_row, Mapping) and str(route_row.get("item_key", "") or "") == item_key:
            return route_row
    return {}


def _fallback_quality_route(row: Mapping[str, Any]) -> str:
    if row.get("status") == "skipped":
        return "skipped"
    if row.get("error"):
        return "failed"
    if _int_value(row.get("candidate_count")) == 0:
        return "no_formula_candidate"
    if row.get("default_batch_status") == "deferred_high_density":
        return "deferred_high_density"
    return "auto_candidate"


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


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
