"""Read-only comparison of formula candidates from multiple parser estimates."""

from __future__ import annotations

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
    return {
        "mode": "read_only_external_parser_comparison",
        "parser_count": len(estimates),
        "parser_labels": sorted(estimates),
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
        "manual_review_paper_count": sum(
            1 for row in rows
            if row["write_recommendation"] == "manual_review_queue"
        ),
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
    flags = _comparison_flags(
        parser_count=len(parser_summaries),
        expected_parser_count=len(parser_labels),
        candidate_counts=candidate_counts,
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
    return {
        "candidate_count": _int_value(row.get("candidate_count")),
        "source_group_counts": dict(sorted(_source_group_counts(source_counts).items())),
        "quality_route": str(row.get("quality_route", "") or ""),
        "status": str(row.get("status", "") or ""),
        "reason": str(row.get("reason", "") or ""),
        "preview_candidate_count": len(_list_value(row.get("candidate_preview"))),
    }


def _comparison_flags(
    *,
    parser_count: int,
    expected_parser_count: int,
    candidate_counts: list[int],
    preview_candidate_count: int,
    consensus: Mapping[str, Any],
) -> list[str]:
    flags: list[str] = []
    if parser_count < expected_parser_count:
        flags.append("missing_parser_result")
    if candidate_counts and max(candidate_counts) != min(candidate_counts):
        flags.append("candidate_count_mismatch")
    if sum(candidate_counts) > 0 and preview_candidate_count == 0:
        flags.append("candidate_preview_missing")
    if _int_value(consensus.get("conflict_cluster_count")):
        flags.append("candidate_consensus_conflicts")
    if (
        parser_count >= 2
        and preview_candidate_count > 0
        and _int_value(consensus.get("multi_provider_cluster_count")) == 0
    ):
        flags.append("no_cross_parser_candidate_overlap")
    return flags


def _write_recommendation(
    flags: list[str],
    candidate_counts: list[int],
    consensus: Mapping[str, Any],
) -> str:
    if not candidate_counts or sum(candidate_counts) == 0:
        return "skip_no_formula_candidate"
    if flags:
        return "manual_review_queue"
    if _int_value(consensus.get("multi_provider_cluster_count")) > 0:
        return "candidate_supported_by_cross_parser_review"
    return "manual_review_queue"


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
