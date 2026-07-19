"""Read-only consensus for formula-candidate evidence.

Parser labels are useful diagnostics, but they are not independent evidence.
For example, the ``auto`` and ``text_layer`` paths may both read the same
PyMuPDF text layer.  This module therefore routes candidates by source family,
artifact provenance, and evidence modality rather than label count.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

STRUCTURED_SOURCE_GROUPS = frozenset(
    {
        "docling",
        "mineru_cache",
        "pdf_extract_kit",
        "pp_doclayout",
    }
)
_EVIDENCE_MODALITY_BY_SOURCE_GROUP = {
    "pdf_text_layer": "text_layer",
    "mineru_cache": "structured_cache",
    "pdf_extract_kit": "vision_layout",
    "pp_doclayout": "vision_layout",
    "docling": "vision_layout",
}
_NUMBER_RE = re.compile(r"[^0-9A-Za-z.\-]+")


def source_group(source: str) -> str:
    """Return a stable source family for a formula-candidate source string."""
    normalized = (source or "").strip().lower()
    if normalized == "text_layer" or normalized.startswith("pdf_text"):
        return "pdf_text_layer"
    if normalized.startswith("mineru_"):
        return "mineru_cache"
    if normalized.startswith("pp_doclayout_"):
        return "pp_doclayout"
    if normalized.startswith("pdf_extract_kit_") or normalized.startswith("doclayout_yolo_"):
        return "pdf_extract_kit"
    if normalized.startswith("docling_"):
        return "docling"
    return "other"


def evidence_modality(source_group_name: str) -> str:
    """Return the evidence modality represented by a source family."""
    return _EVIDENCE_MODALITY_BY_SOURCE_GROUP.get(source_group_name, "unknown")


def build_formula_candidate_consensus(candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Cluster candidate previews and return conservative read-only routes.

    ``supported_candidate`` never means writable.  It only says that two
    independent source families support the same candidate region.  A cluster
    backed only by the PDF text layer always remains in review, even when it
    was emitted under multiple parser labels.
    """
    rows = [_candidate_row(index, candidate) for index, candidate in enumerate(candidates)]
    clusters: list[list[dict[str, Any]]] = []
    for row in rows:
        target = next((cluster for cluster in clusters if _belongs_to_cluster(row, cluster)), None)
        if target is None:
            clusters.append([row])
        else:
            target.append(row)
    summaries = [_summarize_cluster(index, cluster) for index, cluster in enumerate(clusters)]
    return {
        "mode": "candidate_preview_consensus",
        "preview_candidate_count": len(rows),
        "cluster_count": len(summaries),
        "supported_cluster_count": sum(
            summary["cluster_route"] == "supported_candidate" for summary in summaries
        ),
        "single_provider_review_cluster_count": sum(
            summary["cluster_route"] == "single_provider_review" for summary in summaries
        ),
        "manual_review_cluster_count": sum(
            summary["cluster_route"] == "manual_review" for summary in summaries
        ),
        "clusters": summaries,
    }


def _candidate_row(index: int, candidate: Mapping[str, Any]) -> dict[str, Any]:
    source = str(candidate.get("source", "") or "")
    group = source_group(source)
    artifact_hash = str(candidate.get("source_artifact_hash", "") or "").strip().lower()
    return {
        "candidate_index": index,
        "parser_label": str(candidate.get("parser_label", "") or ""),
        "source": source,
        "source_group": group,
        "evidence_modality": evidence_modality(group),
        "source_artifact_hash": artifact_hash,
        "page_num": _positive_int(candidate.get("page_num")),
        "bbox": _bbox(candidate.get("bbox")),
        "equation_number": str(candidate.get("equation_number", "") or ""),
    }


def _belongs_to_cluster(row: Mapping[str, Any], cluster: list[dict[str, Any]]) -> bool:
    return any(_same_formula(row, existing) for existing in cluster)


def _same_formula(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["page_num"] <= 0 or left["page_num"] != right["page_num"]:
        return False
    left_number = _normalize_number(str(left["equation_number"]))
    right_number = _normalize_number(str(right["equation_number"]))
    if left_number and right_number and left_number == right_number:
        return True
    left_bbox = left["bbox"]
    right_bbox = right["bbox"]
    return len(left_bbox) == 4 and len(right_bbox) == 4 and _bbox_iou(left_bbox, right_bbox) >= 0.3


def _summarize_cluster(index: int, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    source_groups = {str(candidate["source_group"]) for candidate in candidates}
    parser_labels = {str(candidate["parser_label"]) for candidate in candidates if candidate["parser_label"]}
    modalities = {str(candidate["evidence_modality"]) for candidate in candidates}
    provenance_keys = {
        str(candidate["source_artifact_hash"]) or f"group:{candidate['source_group']}"
        for candidate in candidates
    }
    structured_group_count = len(source_groups & STRUCTURED_SOURCE_GROUPS)
    provider_group_count = len(source_groups)
    independent_source_count = len(provenance_keys)
    has_independent_agreement = provider_group_count >= 2 and independent_source_count >= 2
    flags = []
    if has_independent_agreement:
        flags.append("multi_provider_agreement")
    else:
        flags.append("single_provider_only")
    if structured_group_count == 0:
        flags.append("no_structured_parser_evidence")
    if len(provenance_keys) < provider_group_count:
        flags.append("shared_source_artifact")
    route = _cluster_route(flags)
    source_group_counts = Counter(candidate["source_group"] for candidate in candidates)
    parser_label_counts = Counter(
        candidate["parser_label"] for candidate in candidates if candidate["parser_label"]
    )
    modality_counts = Counter(candidate["evidence_modality"] for candidate in candidates)
    return {
        "cluster_id": f"formula_cluster_{index:04d}",
        "cluster_route": route,
        "candidate_count": len(candidates),
        "source_group_counts": dict(sorted(source_group_counts.items())),
        "parser_label_counts": dict(sorted(parser_label_counts.items())),
        "evidence_modality_counts": dict(sorted(modality_counts.items())),
        "provider_group_count": provider_group_count,
        "parser_label_count": len(parser_labels),
        "evidence_modality_count": len(modalities),
        "independent_source_count": independent_source_count,
        "structured_provider_group_count": structured_group_count,
        "review_flags": flags,
        "candidate_indices": [int(candidate["candidate_index"]) for candidate in candidates],
    }


def _cluster_route(flags: list[str]) -> str:
    if "no_structured_parser_evidence" in flags or "shared_source_artifact" in flags:
        return "single_provider_review"
    if "multi_provider_agreement" in flags:
        return "supported_candidate"
    return "manual_review"


def _positive_int(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return result if result > 0 else 0


def _bbox(value: Any) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return ()
    try:
        x0, y0, x1, y1 = (float(part) for part in value)
    except (TypeError, ValueError):
        return ()
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else ()


def _normalize_number(value: str) -> str:
    return _NUMBER_RE.sub("", value or "").lower()


def _bbox_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)
