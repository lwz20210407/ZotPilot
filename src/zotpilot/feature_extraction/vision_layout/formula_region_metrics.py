"""Offline formula-region detection metrics for visual-layout caches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def evaluate_formula_regions(
    annotation: Mapping[str, Any],
    cache: Mapping[str, Any],
    *,
    item_key: str | None = None,
    iou_threshold: float = 0.5,
    draft_coordinate_space: str = "pdf",
    display_only: bool = True,
) -> dict[str, Any]:
    """Match formula regions against a tier-1 annotation or its draft proposals."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    document = _annotation_document(annotation, item_key)
    page_sizes = _cache_page_sizes(cache)
    gold, reference_kind = _annotation_formulas(
        document,
        page_sizes=page_sizes,
        draft_coordinate_space=draft_coordinate_space,
    )
    resolved_draft_coordinate_space = _resolve_draft_coordinate_space(document, draft_coordinate_space)
    predicted = _cache_formula_regions(cache, display_only=display_only)
    matches = _match_regions(gold, predicted, iou_threshold=iou_threshold)
    matched_gold = {gold_index for gold_index, _predicted_index, _iou in matches}
    matched_predicted = {predicted_index for _gold_index, predicted_index, _iou in matches}
    true_positive = len(matches)
    false_positive = len(predicted) - len(matched_predicted)
    false_negative = len(gold) - len(matched_gold)
    return {
        "mode": "formula_region_iou",
        "reference_kind": reference_kind,
        "item_key": str(document.get("item_key", "")),
        "iou_threshold": iou_threshold,
        "draft_coordinate_space": resolved_draft_coordinate_space if reference_kind == "draft_proposal" else None,
        "display_only": display_only,
        "gold_formula_count": len(gold),
        "predicted_formula_count": len(predicted),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "matches": [
            {
                "gold_index": gold_index,
                "predicted_index": predicted_index,
                "page_num": gold[gold_index]["page_num"],
                "iou": round(iou, 6),
            }
            for gold_index, predicted_index, iou in matches
        ],
        "unmatched_gold": [gold[index] for index in range(len(gold)) if index not in matched_gold],
        "unmatched_predicted": [predicted[index] for index in range(len(predicted)) if index not in matched_predicted],
    }


def _annotation_document(annotation: Mapping[str, Any], item_key: str | None) -> Mapping[str, Any]:
    documents = annotation.get("documents", [])
    if not isinstance(documents, list):
        raise ValueError("annotation.documents must be a list")
    matches = [
        document
        for document in documents
        if isinstance(document, Mapping) and (item_key is None or str(document.get("item_key", "")) == item_key)
    ]
    if len(matches) != 1:
        raise ValueError("annotation must resolve to exactly one document")
    return matches[0]


def _annotation_formulas(
    document: Mapping[str, Any],
    *,
    page_sizes: Mapping[int, tuple[float, float]],
    draft_coordinate_space: str,
) -> tuple[list[dict[str, Any]], str]:
    accepted = _region_rows(document.get("formulas"), bbox_key="bbox", page_sizes=page_sizes, coordinate_space="pdf")
    if accepted:
        return accepted, "gold"
    resolved_coordinate_space = _resolve_draft_coordinate_space(document, draft_coordinate_space)
    return (
        _region_rows(
            document.get("proposed_formulas"),
            bbox_key="bbox",
            page_sizes=page_sizes,
            coordinate_space=resolved_coordinate_space,
        ),
        "draft_proposal",
    )


def _resolve_draft_coordinate_space(document: Mapping[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    declared = str(document.get("proposed_bbox_coordinate_space", "") or "").strip().lower()
    return declared if declared in {"pdf", "normalized_1000"} else "pdf"


def _cache_formula_regions(cache: Mapping[str, Any], *, display_only: bool) -> list[dict[str, Any]]:
    pages = cache.get("pages", [])
    if not isinstance(pages, list):
        raise ValueError("cache.pages must be a list")
    regions: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        page_num = _positive_int(page.get("page_num"))
        if not page_num:
            continue
        page_regions = page.get("regions", [])
        if not isinstance(page_regions, list):
            continue
        formula_regions = [
            region
            for region in page_regions
            if isinstance(region, Mapping) and str(region.get("cls", "")).replace(" ", "_") == "formula"
        ]
        number_regions = [
            region
            for region in page_regions
            if isinstance(region, Mapping)
            and str(region.get("cls", "")).replace(" ", "_") in {"formula_number", "equation_number"}
        ]
        page_size = _page_size(page)
        bound_formula_indices = _bound_formula_indices(formula_regions, number_regions)
        for index, region in enumerate(formula_regions):
            bbox = _bbox(region.get("bbox_pt"))
            if bbox is None:
                continue
            if display_only and not _is_display_formula_region(bbox, index in bound_formula_indices, page_size):
                continue
            regions.append({"page_num": page_num, "bbox": bbox, "conf": _float(region.get("conf"))})
    return regions


def _region_rows(
    value: Any,
    *,
    bbox_key: str,
    page_sizes: Mapping[int, tuple[float, float]],
    coordinate_space: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        page_num = _positive_int(item.get("page_num"))
        bbox = _bbox(item.get(bbox_key))
        if page_num and bbox is not None:
            if coordinate_space == "normalized_1000":
                page_size = page_sizes.get(page_num)
                if page_size is None:
                    continue
                bbox = _normalized_1000_to_pdf(bbox, page_size)
            elif coordinate_space != "pdf":
                raise ValueError(f"Unsupported annotation coordinate space: {coordinate_space}")
            rows.append({"page_num": page_num, "bbox": bbox, "equation_number": str(item.get("equation_number", ""))})
    return rows


def _match_regions(
    gold: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = [
        (iou, gold_index, predicted_index)
        for gold_index, gold_region in enumerate(gold)
        for predicted_index, predicted_region in enumerate(predicted)
        if gold_region["page_num"] == predicted_region["page_num"]
        if (iou := bbox_iou(gold_region["bbox"], predicted_region["bbox"])) >= iou_threshold
    ]
    matches: list[tuple[int, int, float]] = []
    used_gold: set[int] = set()
    used_predicted: set[int] = set()
    for iou, gold_index, predicted_index in sorted(candidates, reverse=True):
        if gold_index in used_gold or predicted_index in used_predicted:
            continue
        matches.append((gold_index, predicted_index, iou))
        used_gold.add(gold_index)
        used_predicted.add(predicted_index)
    return matches


def _cache_page_sizes(cache: Mapping[str, Any]) -> dict[int, tuple[float, float]]:
    pages = cache.get("pages", [])
    if not isinstance(pages, list):
        return {}
    return {
        page_num: page_size
        for page in pages
        if isinstance(page, Mapping)
        if (page_num := _positive_int(page.get("page_num")))
        if (page_size := _page_size(page)) is not None
    }


def _page_size(page: Mapping[str, Any]) -> tuple[float, float] | None:
    value = page.get("page_size_pt")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def _is_display_formula_region(
    bbox: tuple[float, float, float, float],
    has_bound_number: bool,
    page_size: tuple[float, float] | None,
) -> bool:
    if has_bound_number:
        return True
    if page_size is None:
        return False
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width / page_size[0] >= 0.09 and height / page_size[1] >= 0.014


def _bound_formula_indices(
    formula_regions: Sequence[Mapping[str, Any]],
    number_regions: Sequence[Mapping[str, Any]],
) -> set[int]:
    """Return the one-to-one visual formula/number assignments for one page."""
    pairs: list[tuple[float, float, int, int]] = []
    for formula_index, formula_region in enumerate(formula_regions):
        formula = _bbox(formula_region.get("bbox_pt"))
        if formula is None:
            continue
        for number_index, number_region in enumerate(number_regions):
            number = _bbox(number_region.get("bbox_pt"))
            if number is None or not _is_number_in_same_display_band(formula, number):
                continue
            formula_center = (formula[1] + formula[3]) / 2
            number_center = (number[1] + number[3]) / 2
            pairs.append(
                (
                    abs(formula_center - number_center),
                    abs(number[0] - formula[2]),
                    formula_index,
                    number_index,
                )
            )
    bound: set[int] = set()
    used_numbers: set[int] = set()
    for _distance, _gap, formula_index, number_index in sorted(pairs):
        if formula_index in bound or number_index in used_numbers:
            continue
        bound.add(formula_index)
        used_numbers.add(number_index)
    return bound


def _is_number_in_same_display_band(
    formula: tuple[float, float, float, float],
    number: tuple[float, float, float, float] | None,
) -> bool:
    if number is None:
        return False
    formula_center = (formula[1] + formula[3]) / 2
    number_center = (number[1] + number[3]) / 2
    formula_height = formula[3] - formula[1]
    number_height = number[3] - number[1]
    if abs(formula_center - number_center) > max(formula_height, number_height) * 1.25:
        return False
    return number[0] >= (formula[0] + formula[2]) / 2


def _normalized_1000_to_pdf(
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    width, height = page_size
    return (
        round(x0 * width / 1000, 6),
        round(y0 * height / 1000, 6),
        round(x1 * width / 1000, 6),
        round(y1 * height / 1000, 6),
    )


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    left_x0, left_y0, left_x1, left_y1 = left
    right_x0, right_y0, right_x1, right_y1 = right
    x0, y0 = max(left_x0, right_x0), max(left_y0, right_y0)
    x1, y1 = min(left_x1, right_x1), min(left_y1, right_y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    left_area = (left_x1 - left_x0) * (left_y1 - left_y0)
    right_area = (right_x1 - right_x0) * (right_y1 - right_y0)
    return intersection / (left_area + right_area - intersection)


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _positive_int(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return result if result > 0 else 0


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
