"""Read-only comparison of formula candidates from multiple parser estimates."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .formula_candidate_consensus import build_formula_candidate_consensus, source_group


def build_formula_external_parser_comparison(named_reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compare formula candidates for the same papers across parser estimate reports."""
    estimates = {
        label: _unwrap_estimate(report)
        for label, report in named_reports.items()
        if isinstance(report, Mapping)
    }
    rows = _comparison_rows(estimates)
    parser_labels = sorted(estimates)
    return {
        "mode": "read_only_external_parser_comparison",
        "parser_count": len(estimates),
        "parser_labels": parser_labels,
        "readonly_index_changed": any(bool(estimate.get("readonly_index_changed")) for estimate in estimates.values()),
        "readonly_index_changed_by_parser": {
            label: bool(estimate.get("readonly_index_changed"))
            for label, estimate in sorted(estimates.items())
        },
        "request_complete_by_parser": {
            label: bool(estimate.get("request_complete", True))
            for label, estimate in sorted(estimates.items())
        },
        "paper_count": len(rows),
        "candidate_count_by_parser": {
            label: _int_value(estimate.get("candidate_count"))
            for label, estimate in sorted(estimates.items())
        },
        "previewed_paper_count": sum(1 for row in rows if row["preview_candidate_count"] > 0),
        "consensus_cluster_count": sum(
            _int_value(_dict_value(row.get("candidate_consensus")).get("cluster_count"))
            for row in rows
        ),
        "multi_provider_cluster_count": sum(
            _int_value(_dict_value(row.get("candidate_consensus")).get("multi_provider_cluster_count"))
            for row in rows
        ),
        "conflict_cluster_count": sum(
            _int_value(_dict_value(row.get("candidate_consensus")).get("conflict_cluster_count"))
            for row in rows
        ),
        "supported_cluster_count": sum(
            _int_value(_dict_value(row.get("candidate_consensus")).get("supported_cluster_count"))
            for row in rows
        ),
        "review_cluster_count": sum(
            _int_value(_dict_value(row.get("candidate_consensus")).get("review_cluster_count"))
            for row in rows
        ),
        "single_provider_review_cluster_count": sum(
            _int_value(_dict_value(row.get("candidate_consensus")).get("single_provider_review_cluster_count"))
            for row in rows
        ),
        "conflict_review_cluster_count": sum(
            _int_value(_dict_value(row.get("candidate_consensus")).get("conflict_review_cluster_count"))
            for row in rows
        ),
        "ocr_fallback_cluster_count": sum(
            _int_value(_dict_value(row.get("candidate_consensus")).get("ocr_fallback_cluster_count"))
            for row in rows
        ),
        "manual_review_paper_count": sum(
            1 for row in rows
            if row["write_recommendation"] == "manual_review_queue"
        ),
        "partial_review_paper_count": sum(
            1 for row in rows
            if row["write_recommendation"] == "partial_cross_parser_support_review_extras"
        ),
        "review_required_paper_count": sum(
            1 for row in rows
            if row["write_recommendation"] in {
                "manual_review_queue",
                "partial_cross_parser_support_review_extras",
                "single_parser_candidate_review",
            }
        ),
        "comparison_flag_counts": _comparison_flag_counts(rows),
        "write_recommendation_counts": _write_recommendation_counts(rows),
        "parser_candidate_summary": _parser_candidate_summary(rows, parser_labels),
        "rows": rows,
    }


def _comparison_rows(estimates: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    item_keys: set[str] = set()
    rows_by_label: dict[str, dict[str, Mapping[str, Any]]] = {}
    for label, estimate in estimates.items():
        row_map = _row_map(estimate)
        rows_by_label[label] = row_map
        item_keys.update(row_map)

    rows: list[dict[str, Any]] = []
    labels = sorted(estimates)
    for item_key in sorted(item_keys):
        parser_rows = {
            label: rows_by_label.get(label, {}).get(item_key, {})
            for label in labels
        }
        rows.append(_comparison_row(item_key=item_key, parser_rows=parser_rows, parser_labels=labels))
    return rows


def _comparison_row(
    *,
    item_key: str,
    parser_rows: Mapping[str, Mapping[str, Any]],
    parser_labels: list[str],
) -> dict[str, Any]:
    title = _first_non_empty(str(row.get("title", "") or "") for row in parser_rows.values())
    previews = [
        {**preview, "parser_label": label}
        for label, row in parser_rows.items()
        for preview in _list_value(row.get("candidate_preview"))
        if isinstance(preview, Mapping)
    ]
    consensus = build_formula_candidate_consensus(previews)
    parser_summaries = {
        label: _parser_row_summary(row)
        for label, row in parser_rows.items()
        if row
    }
    missing_parser_labels = [
        label for label in parser_labels
        if label not in parser_summaries
    ]
    candidate_counts = [
        int(summary["candidate_count"])
        for summary in parser_summaries.values()
    ]
    preview_counts = [
        int(summary["preview_candidate_count"])
        for summary in parser_summaries.values()
    ]
    flags = _comparison_flags(
        parser_count=len(parser_summaries),
        expected_parser_count=len(parser_labels),
        candidate_counts=candidate_counts,
        preview_counts=preview_counts,
        opaque_parser_count=sum(
            1 for summary in parser_summaries.values()
            if _parser_summary_has_opaque_text_layer_evidence(summary)
        ),
        number_only_pdf_fallback_parser_count=sum(
            1 for summary in parser_summaries.values()
            if _parser_summary_has_number_only_pdf_fallback_evidence(summary)
        ),
        weak_text_layer_fragment_parser_count=sum(
            1 for summary in parser_summaries.values()
            if _parser_summary_has_weak_text_layer_fragment_evidence(summary)
        ),
        preview_candidate_count=_int_value(consensus.get("preview_candidate_count")),
        consensus=consensus,
    )
    return {
        "item_key": item_key,
        "title": title,
        "parser_count": len(parser_summaries),
        "missing_parser_labels": missing_parser_labels,
        "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
        "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
        "candidate_count_total": sum(candidate_counts),
        "preview_candidate_count": _int_value(consensus.get("preview_candidate_count")),
        "candidate_consensus": consensus,
        "parser_summaries": parser_summaries,
        "comparison_flags": flags,
        "write_recommendation": _write_recommendation(flags, candidate_counts, consensus),
    }


def _parser_row_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    audit = _dict_value(row.get("candidate_audit"))
    source_counts = _dict_value(audit.get("source_counts"))
    previews = [
        preview for preview in _list_value(row.get("candidate_preview"))
        if isinstance(preview, Mapping)
    ]
    return {
        "candidate_count": _int_value(row.get("candidate_count")),
        "source_group_counts": dict(sorted(_source_group_counts(source_counts).items())),
        "quality_route": str(row.get("quality_route", "") or ""),
        "status": str(row.get("status", "") or ""),
        "reason": str(row.get("reason", "") or ""),
        "preview_candidate_count": len(previews),
        "opaque_text_layer_candidate_count": sum(
            1 for preview in previews
            if _looks_like_opaque_text_layer_candidate_preview(preview)
        ),
        "number_only_pdf_fallback_candidate_count": sum(
            1 for preview in previews
            if _looks_like_number_only_pdf_fallback_candidate_preview(preview)
        ),
        "weak_text_layer_fragment_candidate_count": sum(
            1 for preview in previews
            if _looks_like_weak_text_layer_fragment_candidate_preview(preview)
        ),
    }


def _comparison_flags(
    *,
    parser_count: int,
    expected_parser_count: int,
    candidate_counts: list[int],
    preview_counts: list[int],
    opaque_parser_count: int,
    number_only_pdf_fallback_parser_count: int,
    weak_text_layer_fragment_parser_count: int,
    preview_candidate_count: int,
    consensus: Mapping[str, Any],
) -> list[str]:
    flags: list[str] = []
    if parser_count < expected_parser_count:
        flags.append("missing_parser_result")
    if candidate_counts and max(candidate_counts) != min(candidate_counts):
        flags.append("candidate_count_mismatch")
    if any(count == 0 for count in candidate_counts) and sum(candidate_counts) > 0:
        flags.append("parser_without_candidates")
    if sum(candidate_counts) > 0 and preview_candidate_count == 0:
        flags.append("candidate_preview_missing")
    if any(
        preview_count < candidate_count
        for candidate_count, preview_count in zip(candidate_counts, preview_counts, strict=False)
    ):
        flags.append("candidate_preview_truncated")
    if _int_value(consensus.get("conflict_cluster_count")):
        flags.append("candidate_consensus_conflicts")
    if (
        parser_count >= 2
        and all(count > 0 for count in candidate_counts)
        and preview_candidate_count > 0
        and _int_value(consensus.get("multi_provider_cluster_count")) == 0
    ):
        if opaque_parser_count:
            flags.append("opaque_text_layer_candidate_evidence")
        elif number_only_pdf_fallback_parser_count:
            flags.append("number_only_pdf_fallback_evidence")
        elif weak_text_layer_fragment_parser_count:
            flags.append("weak_text_layer_fragment_evidence")
        else:
            flags.append("no_cross_parser_candidate_overlap")
    return flags


def _write_recommendation(
    flags: list[str],
    candidate_counts: list[int],
    consensus: Mapping[str, Any],
) -> str:
    if not candidate_counts or sum(candidate_counts) == 0:
        return "skip_no_formula_candidate"
    conflict_count = _int_value(consensus.get("conflict_cluster_count"))
    multi_provider_count = _int_value(consensus.get("multi_provider_cluster_count"))
    hard_review_flags = {
        "missing_parser_result",
        "candidate_preview_missing",
        "candidate_preview_truncated",
        "candidate_consensus_conflicts",
        "no_cross_parser_candidate_overlap",
    }
    if conflict_count > 0 or any(flag in hard_review_flags for flag in flags):
        if (
            conflict_count == 0
            and "no_cross_parser_candidate_overlap" in flags
            and _has_structured_single_parser_review_candidate(consensus)
        ):
            return "single_parser_candidate_review"
        return "manual_review_queue"
    if "parser_without_candidates" in flags or "opaque_text_layer_candidate_evidence" in flags:
        return "single_parser_candidate_review"
    if "number_only_pdf_fallback_evidence" in flags or "weak_text_layer_fragment_evidence" in flags:
        return "single_parser_candidate_review"
    if flags == ["candidate_count_mismatch"] and multi_provider_count > 0:
        return "partial_cross_parser_support_review_extras"
    if flags:
        return "manual_review_queue"
    if multi_provider_count > 0:
        return "candidate_supported_by_cross_parser_review"
    return "manual_review_queue"


def _has_structured_single_parser_review_candidate(consensus: Mapping[str, Any]) -> bool:
    for cluster in _list_value(consensus.get("clusters")):
        if not isinstance(cluster, Mapping):
            continue
        if _int_value(cluster.get("structured_provider_group_count")) > 0:
            return True
    return False


def _parser_summary_has_opaque_text_layer_evidence(summary: Mapping[str, Any]) -> bool:
    preview_count = _int_value(summary.get("preview_candidate_count"))
    if preview_count <= 0:
        return False
    source_counts = _dict_value(summary.get("source_group_counts"))
    if set(source_counts) != {"pdf_text_layer"}:
        return False
    opaque_count = _int_value(summary.get("opaque_text_layer_candidate_count"))
    threshold = 1 if preview_count <= 3 else max(5, int(preview_count * 0.25))
    return opaque_count >= threshold


def _parser_summary_has_number_only_pdf_fallback_evidence(summary: Mapping[str, Any]) -> bool:
    preview_count = _int_value(summary.get("preview_candidate_count"))
    if preview_count <= 0:
        return False
    count = _int_value(summary.get("number_only_pdf_fallback_candidate_count"))
    return count > 0


def _parser_summary_has_weak_text_layer_fragment_evidence(summary: Mapping[str, Any]) -> bool:
    preview_count = _int_value(summary.get("preview_candidate_count"))
    if preview_count <= 0 or preview_count > 3:
        return False
    source_counts = _dict_value(summary.get("source_group_counts"))
    if set(source_counts) != {"pdf_text_layer"}:
        return False
    count = _int_value(summary.get("weak_text_layer_fragment_candidate_count"))
    return count == preview_count


def _looks_like_opaque_text_layer_candidate_preview(preview: Mapping[str, Any]) -> bool:
    if source_group(str(preview.get("source", "") or "")) != "pdf_text_layer":
        return False
    raw_text = str(preview.get("raw_text_preview", "") or "")
    if not raw_text:
        return False
    private_use_count = sum(1 for char in raw_text if "\ue000" <= char <= "\uf8ff")
    pdf_encoded_count = sum(1 for char in raw_text if char in "¼ðþÞ")
    control_count = sum(1 for char in raw_text if ord(char) < 32 and char not in "\t\r\n")
    return private_use_count >= 2 or pdf_encoded_count >= 2 or control_count >= 2


def _looks_like_number_only_pdf_fallback_candidate_preview(preview: Mapping[str, Any]) -> bool:
    source = str(preview.get("source", "") or "")
    if not source.startswith("pdf_text_equation_number"):
        return False
    raw_text = str(preview.get("raw_text_preview", "") or "").strip()
    equation_number = str(preview.get("equation_number", "") or "").strip()
    if not raw_text or not equation_number or str(preview.get("latex_preview", "") or "").strip():
        return False
    number = equation_number.strip("()（）")
    return bool(number and raw_text.strip(" \t\r\n()（）ðÞ") == number)


def _looks_like_weak_text_layer_fragment_candidate_preview(preview: Mapping[str, Any]) -> bool:
    if source_group(str(preview.get("source", "") or "")) != "pdf_text_layer":
        return False
    if str(preview.get("latex_preview", "") or "").strip() or str(preview.get("equation_number", "") or "").strip():
        return False
    raw_text = str(preview.get("raw_text_preview", "") or "").strip()
    if not raw_text or len(raw_text) > 120:
        return False
    normalized = re.sub(r"\s+", "", raw_text)
    if re.search(r"(?:=|¼|þ|≈|≤|≥|≠|<|>|\\(?:leq?|geq?|approx|sim))", normalized):
        return False
    return normalized.endswith("/") or len(re.findall(r"/\s*(?:mm|cm|m|s|pa|mpa|gpa|%)", raw_text, re.IGNORECASE)) >= 1


def _row_map(estimate: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    route_rows = {
        str(route.get("item_key", "") or ""): route
        for route in _list_value(estimate.get("formula_quality_route_summary"))
        if isinstance(route, Mapping)
    }
    for row in _list_value(estimate.get("results")):
        if not isinstance(row, Mapping):
            continue
        item_key = str(row.get("item_key", "") or "")
        if not item_key:
            continue
        route_row = route_rows.get(item_key, {})
        merged = dict(row)
        if route_row:
            merged["quality_route"] = str(route_row.get("quality_route", "") or "")
            merged["route_reason"] = str(route_row.get("route_reason", "") or "")
        rows[item_key] = merged
    return rows


def _source_group_counts(source_counts: Mapping[str, Any]) -> Counter[str]:
    groups: Counter[str] = Counter()
    for source, count in source_counts.items():
        groups[source_group(str(source))] += _int_value(count)
    return groups


def _comparison_flag_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(str(flag) for flag in _list_value(row.get("comparison_flags")))
    return dict(sorted(counts.items()))


def _write_recommendation_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("write_recommendation", "") or "unknown")] += 1
    return dict(sorted(counts.items()))


def _parser_candidate_summary(
    rows: list[dict[str, Any]],
    parser_labels: list[str],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for label in parser_labels:
        row_count = 0
        candidate_count = 0
        preview_candidate_count = 0
        opaque_text_layer_candidate_count = 0
        number_only_pdf_fallback_candidate_count = 0
        weak_text_layer_fragment_candidate_count = 0
        quality_routes: Counter[str] = Counter()
        source_groups: Counter[str] = Counter()
        for row in rows:
            parser_summary = _dict_value(_dict_value(row.get("parser_summaries")).get(label))
            if not parser_summary:
                continue
            row_count += 1
            candidate_count += _int_value(parser_summary.get("candidate_count"))
            preview_candidate_count += _int_value(parser_summary.get("preview_candidate_count"))
            opaque_text_layer_candidate_count += _int_value(parser_summary.get("opaque_text_layer_candidate_count"))
            number_only_pdf_fallback_candidate_count += _int_value(
                parser_summary.get("number_only_pdf_fallback_candidate_count")
            )
            weak_text_layer_fragment_candidate_count += _int_value(
                parser_summary.get("weak_text_layer_fragment_candidate_count")
            )
            quality_route = str(parser_summary.get("quality_route", "") or "unknown")
            quality_routes[quality_route] += 1
            source_groups.update(_dict_value(parser_summary.get("source_group_counts")))
        summaries[label] = {
            "paper_count": row_count,
            "candidate_count": candidate_count,
            "preview_candidate_count": preview_candidate_count,
            "opaque_text_layer_candidate_count": opaque_text_layer_candidate_count,
            "number_only_pdf_fallback_candidate_count": number_only_pdf_fallback_candidate_count,
            "weak_text_layer_fragment_candidate_count": weak_text_layer_fragment_candidate_count,
            "quality_route_counts": dict(sorted(quality_routes.items())),
            "source_group_counts": dict(sorted(source_groups.items())),
        }
    return summaries


def _unwrap_estimate(report: Mapping[str, Any]) -> Mapping[str, Any]:
    estimate = report.get("estimate")
    return estimate if isinstance(estimate, Mapping) else report


def _first_non_empty(values: Any) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
