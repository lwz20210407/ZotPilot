"""Deterministic formula-region metrics and COCO interchange."""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

_CATEGORIES = {"formula": 1, "formula_number": 2}


def evaluate_formula_gold(
    gold: Mapping[str, Any],
    cache: Mapping[str, Any],
    *,
    item_key: str,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Return one-to-one per-class P/R diagnostics for one source-bound PDF."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    document = _gold_document(gold, item_key, source_pdf_sha256=str(cache.get("source_pdf_sha256", "")))
    _validate_cache_source(cache, document)
    gold_regions = _gold_regions(document)
    reviewed_pages = {int(page["page_num"]) for page in document["pages"]}
    predicted_regions = [region for region in _cache_regions(cache) if region["page_num"] in reviewed_pages]
    by_class = {
        label: _match_class(gold_regions, predicted_regions, label, iou_threshold)
        for label in _CATEGORIES
    }
    return {
        "mode": "formula_gold_iou",
        "item_key": item_key,
        "source_pdf_sha256": document["source_pdf_sha256"],
        "iou_threshold": iou_threshold,
        "reviewed_page_numbers": sorted(reviewed_pages),
        "region_metric_scope": "formula_including_inline_and_formula_number_on_reviewed_pages",
        "by_class": by_class,
        "overall": _aggregate(by_class.values()),
        "formula_layout": _formula_layout_metrics(gold_regions, predicted_regions, iou_threshold),
        "formula_latex_coverage": _formula_latex_coverage(document),
        "formula_number_coverage": _formula_number_coverage(document),
    }


def evaluate_formula_gold_corpus(
    gold: Mapping[str, Any],
    caches_by_item_key: Mapping[str, Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggregate source-bound reviewed-Gold metrics across selected documents.

    The caller supplies the explicit ``item_key -> visual cache`` mapping
    rather than allowing filename matching.  Each row still goes through
    :func:`evaluate_formula_gold`, which rejects an attachment hash mismatch.
    This keeps multi-journal benchmark reports from accidentally combining an
    original PDF annotation with a translated or replaced attachment.
    """
    if not caches_by_item_key:
        raise ValueError("At least one item-keyed visual-layout cache is required")
    documents = {
        item_key: evaluate_formula_gold(gold, cache, item_key=item_key, iou_threshold=iou_threshold)
        for item_key, cache in sorted(caches_by_item_key.items())
    }
    reports = list(documents.values())
    return {
        "mode": "formula_gold_iou_corpus",
        "iou_threshold": iou_threshold,
        "document_count": len(documents),
        "documents": documents,
        "overall": _aggregate(report["overall"] for report in reports),
        "by_class": {
            label: _aggregate(report["by_class"][label] for report in reports)
            for label in _CATEGORIES
        },
        "formula_latex_coverage": _aggregate_coverage(
            (report["formula_latex_coverage"] for report in reports),
            "latex_annotated_count",
        ),
        "formula_number_coverage": _aggregate_coverage(
            (report["formula_number_coverage"] for report in reports),
            "number_annotated_count",
        ),
    }


def to_coco(
    gold: Mapping[str, Any],
    cache: Mapping[str, Any],
    *,
    item_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert reviewed gold and PP-DocLayout cache rows to COCO instances."""
    document = _gold_document(gold, item_key, source_pdf_sha256=str(cache.get("source_pdf_sha256", "")))
    _validate_cache_source(cache, document)
    images = []
    image_ids: dict[int, int] = {}
    for index, page in enumerate(document["pages"], start=1):
        page_num = int(page["page_num"])
        width, height = page["page_size_pt"]
        image_ids[page_num] = index
        images.append({"id": index, "file_name": f"{item_key}-{page_num:04d}.png", "width": width, "height": height})
    annotations = []
    for annotation_id, region in enumerate(_gold_regions(document), start=1):
        if region["page_num"] not in image_ids:
            continue
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_ids[region["page_num"]],
                "category_id": _CATEGORIES[region["cls"]],
                "bbox": _xywh(region["bbox_pt"]),
                "area": _area(region["bbox_pt"]),
                "iscrowd": 0,
            }
        )
    predictions = []
    for region in _cache_regions(cache):
        if region["page_num"] not in image_ids:
            continue
        predictions.append(
            {
                "image_id": image_ids[region["page_num"]],
                "category_id": _CATEGORIES[region["cls"]],
                "bbox": _xywh(region["bbox_pt"]),
                "score": region["confidence"],
            }
        )
    return (
        {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": identifier, "name": label} for label, identifier in _CATEGORIES.items()],
        },
        predictions,
    )


def to_coco_corpus(
    gold: Mapping[str, Any],
    caches_by_item_key: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Combine explicitly item-bound Gold/cache pairs into one COCO dataset.

    ``to_coco`` validates each pair independently.  This wrapper only remaps
    per-document COCO image and annotation IDs to a shared namespace so
    canonical mAP can be calculated once across the benchmark corpus.
    """
    if not caches_by_item_key:
        raise ValueError("At least one item-keyed visual-layout cache is required")
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    for item_key, cache in sorted(caches_by_item_key.items()):
        document_coco, document_predictions = to_coco(gold, cache, item_key=item_key)
        image_ids: dict[int, int] = {}
        for image in document_coco["images"]:
            original_id = int(image["id"])
            image_ids[original_id] = next_image_id
            images.append({**image, "id": next_image_id})
            next_image_id += 1
        for annotation in document_coco["annotations"]:
            annotations.append(
                {
                    **annotation,
                    "id": next_annotation_id,
                    "image_id": image_ids[int(annotation["image_id"])],
                }
            )
            next_annotation_id += 1
        predictions.extend(
            {
                **prediction,
                "image_id": image_ids[int(prediction["image_id"])],
            }
            for prediction in document_predictions
        )
    return (
        {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": identifier, "name": label} for label, identifier in _CATEGORIES.items()],
        },
        predictions,
    )


def evaluate_coco_map(coco_gold: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Evaluate COCO mAP using the optional ``pycocotools`` package.

    Keeping this optional makes annotation export usable in lightweight ZotPilot
    installations while still providing canonical COCO metrics in the review
    environment.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError(
            "COCO mAP requires optional pycocotools; install it in the isolated validation environment."
        ) from error
    with tempfile.TemporaryDirectory(prefix="zotpilot-formula-coco-") as temporary_directory:
        root = Path(temporary_directory)
        gold_path = root / "gold.json"
        prediction_path = root / "predictions.json"
        gold_path.write_text(json.dumps(coco_gold), encoding="utf-8")
        prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
        coco_gt = COCO(str(gold_path))
        coco_dt = coco_gt.loadRes(str(prediction_path))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "map": float(evaluator.stats[0]),
        "map50": float(evaluator.stats[1]),
        "map75": float(evaluator.stats[2]),
        "mar100": float(evaluator.stats[8]),
    }


def _gold_document(
    gold: Mapping[str, Any],
    item_key: str,
    *,
    source_pdf_sha256: str = "",
) -> dict[str, Any]:
    documents = gold.get("documents")
    if not isinstance(documents, list):
        raise ValueError("gold documents must be a list")
    matches = [
        document
        for document in documents
        if isinstance(document, Mapping) and document.get("item_key") == item_key
    ]
    source_hash = str(source_pdf_sha256).strip().lower()
    if source_hash:
        matches = [
            document
            for document in matches
            if str(document.get("source_pdf_sha256", "")).strip().lower() == source_hash
        ]
    if len(matches) != 1:
        raise ValueError(f"gold must contain exactly one document for {item_key} and the cache source PDF")
    document = dict(matches[0])
    pages = document.get("pages")
    if not isinstance(pages, list):
        raise ValueError("gold document pages must be a list")
    document["pages"] = [dict(page) for page in pages if isinstance(page, Mapping)]
    return document


def _validate_cache_source(cache: Mapping[str, Any], document: Mapping[str, Any]) -> None:
    if str(cache.get("generator", "")) != "pp_doclayout":
        raise ValueError("Expected a PP-DocLayout cache")
    if str(cache.get("source_pdf_sha256", "")).lower() != str(document.get("source_pdf_sha256", "")).lower():
        raise ValueError("Gold document and visual-layout cache use different source PDFs")


def _gold_regions(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    regions = []
    for page in document.get("pages", []):
        page_num = _positive_int(page.get("page_num"))
        if page_num is None:
            continue
        for region in page.get("regions", []):
            if not isinstance(region, Mapping):
                continue
            label = str(region.get("cls", "")).strip().lower().replace(" ", "_")
            bbox = _bbox(region.get("bbox_pt"))
            if label in _CATEGORIES and bbox is not None:
                regions.append(
                    {
                        "page_num": page_num,
                        "cls": label,
                        "bbox_pt": bbox,
                        "layout": _layout(region.get("layout")) if label == "formula" else "unknown",
                    }
                )
    return regions


def _formula_latex_coverage(document: Mapping[str, Any]) -> dict[str, Any]:
    """Report whether reviewed boxes are also ready for CDM content evaluation."""
    formulas = []
    for page in document.get("pages", []):
        if not isinstance(page, Mapping):
            continue
        page_num = _positive_int(page.get("page_num"))
        if page_num is None:
            continue
        for region in page.get("regions", []):
            if not isinstance(region, Mapping) or str(region.get("cls", "")).strip() != "formula":
                continue
            if _bbox(region.get("bbox_pt")) is None:
                continue
            formulas.append((page_num, str(region.get("latex", "")).strip()))
    annotated = [page_num for page_num, latex in formulas if latex]
    return {
        "formula_count": len(formulas),
        "latex_annotated_count": len(annotated),
        "latex_coverage": _ratio(len(annotated), len(formulas)),
        "pages_missing_latex": sorted({page_num for page_num, latex in formulas if not latex}),
    }


def _formula_number_coverage(document: Mapping[str, Any]) -> dict[str, Any]:
    """Report formula-level number labels required for binding/sequence checks."""
    formulas = []
    for page in document.get("pages", []):
        if not isinstance(page, Mapping):
            continue
        page_num = _positive_int(page.get("page_num"))
        if page_num is None:
            continue
        for region in page.get("regions", []):
            if not isinstance(region, Mapping) or str(region.get("cls", "")).strip() != "formula":
                continue
            if _bbox(region.get("bbox_pt")) is None:
                continue
            formulas.append((page_num, str(region.get("equation_number", "")).strip()))
    annotated = [(page_num, number) for page_num, number in formulas if number]
    duplicate_numbers = sorted(
        number
        for number in {number for _, number in annotated}
        if sum(1 for _, candidate in annotated if candidate == number) > 1
    )
    return {
        "formula_count": len(formulas),
        "number_annotated_count": len(annotated),
        "number_coverage": _ratio(len(annotated), len(formulas)),
        "pages_missing_number": sorted({page_num for page_num, number in formulas if not number}),
        "duplicate_numbers": duplicate_numbers,
    }


def _cache_regions(cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    regions = []
    for page in cache.get("pages", []):
        if not isinstance(page, Mapping):
            continue
        page_num = _positive_int(page.get("page_num"))
        if page_num is None:
            continue
        for region in page.get("regions", []):
            if not isinstance(region, Mapping):
                continue
            label = str(region.get("cls", "")).strip().lower().replace(" ", "_")
            bbox = _bbox(region.get("bbox_pt"))
            confidence = _score(region.get("conf"))
            if label in _CATEGORIES and bbox is not None and confidence is not None:
                regions.append(
                    {
                        "page_num": page_num,
                        "cls": label,
                        "bbox_pt": bbox,
                        "confidence": confidence,
                        "layout": _layout(region.get("layout")) if label == "formula" else "unknown",
                    }
                )
    return regions


def _match_class(
    gold_regions: list[dict[str, Any]],
    predicted_regions: list[dict[str, Any]],
    label: str,
    threshold: float,
) -> dict[str, Any]:
    gold_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    predicted_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for region in gold_regions:
        if region["cls"] == label:
            gold_by_page[region["page_num"]].append(region)
    for region in predicted_regions:
        if region["cls"] == label:
            predicted_by_page[region["page_num"]].append(region)
    matches = []
    unmatched_gold: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []
    for page_num in sorted(set(gold_by_page) | set(predicted_by_page)):
        candidates = [
            (_iou(gold_region["bbox_pt"], predicted_region["bbox_pt"]), gold_index, predicted_index)
            for gold_index, gold_region in enumerate(gold_by_page[page_num])
            for predicted_index, predicted_region in enumerate(predicted_by_page[page_num])
        ]
        used_gold: set[int] = set()
        used_predicted: set[int] = set()
        for iou, gold_index, predicted_index in sorted(candidates, reverse=True):
            if iou < threshold or gold_index in used_gold or predicted_index in used_predicted:
                continue
            used_gold.add(gold_index)
            used_predicted.add(predicted_index)
            matches.append({"page_num": page_num, "iou": round(iou, 6)})
        unmatched_gold.extend(
            {"page_num": page_num, "bbox_pt": list(region["bbox_pt"])}
            for index, region in enumerate(gold_by_page[page_num])
            if index not in used_gold
        )
        unmatched_predictions.extend(
            {"page_num": page_num, "bbox_pt": list(region["bbox_pt"]), "confidence": region.get("confidence")}
            for index, region in enumerate(predicted_by_page[page_num])
            if index not in used_predicted
        )
    gold_count = sum(len(rows) for rows in gold_by_page.values())
    predicted_count = sum(len(rows) for rows in predicted_by_page.values())
    true_positive = len(matches)
    return {
        "gold_count": gold_count,
        "predicted_count": predicted_count,
        "true_positive": true_positive,
        "false_positive": predicted_count - true_positive,
        "false_negative": gold_count - true_positive,
        "precision": _ratio(true_positive, predicted_count),
        "recall": _ratio(true_positive, gold_count),
        "matches": matches,
        "unmatched_gold": unmatched_gold,
        "unmatched_predictions": unmatched_predictions,
    }


def _formula_layout_metrics(
    gold_regions: list[dict[str, Any]],
    predicted_regions: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate display/inline only when both sides provide a layout label."""
    gold_formula_regions = [region for region in gold_regions if region["cls"] == "formula"]
    predicted_formula_regions = [region for region in predicted_regions if region["cls"] == "formula"]
    labels = ("display", "inline")
    gold_labelled = sum(region["layout"] in labels for region in gold_formula_regions)
    predicted_labelled = sum(region["layout"] in labels for region in predicted_formula_regions)
    unavailable_reasons = []
    if gold_labelled < len(gold_formula_regions):
        unavailable_reasons.append("gold_formula_layout_missing")
    if predicted_labelled < len(predicted_formula_regions):
        unavailable_reasons.append("predicted_formula_layout_missing")
    coverage = {
        "available": not unavailable_reasons,
        "unavailable_reasons": unavailable_reasons,
        "gold_labelled_count": gold_labelled,
        "predicted_labelled_count": predicted_labelled,
        "gold_unknown_count": len(gold_formula_regions) - gold_labelled,
        "predicted_unknown_count": len(predicted_formula_regions) - predicted_labelled,
    }
    if unavailable_reasons:
        return {**coverage, "by_layout": {}}
    by_layout = {
        label: _match_class(
            [dict(region, cls=f"formula_{label}") for region in gold_formula_regions if region["layout"] == label],
            [
                dict(region, cls=f"formula_{label}")
                for region in predicted_formula_regions
                if region["layout"] == label
            ],
            f"formula_{label}",
            threshold,
        )
        for label in labels
    }
    return {
        **coverage,
        "by_layout": by_layout,
    }


def _aggregate(values: Any) -> dict[str, Any]:
    rows = list(values)
    gold_count = sum(int(row["gold_count"]) for row in rows)
    predicted_count = sum(int(row["predicted_count"]) for row in rows)
    true_positive = sum(int(row["true_positive"]) for row in rows)
    return {
        "gold_count": gold_count,
        "predicted_count": predicted_count,
        "true_positive": true_positive,
        "false_positive": predicted_count - true_positive,
        "false_negative": gold_count - true_positive,
        "precision": _ratio(true_positive, predicted_count),
        "recall": _ratio(true_positive, gold_count),
    }


def _aggregate_coverage(rows: Any, annotated_key: str) -> dict[str, Any]:
    values = list(rows)
    formula_count = sum(int(row["formula_count"]) for row in values)
    annotated_count = sum(int(row[annotated_key]) for row in values)
    return {
        "formula_count": formula_count,
        annotated_key: annotated_count,
        "coverage": _ratio(annotated_count, formula_count),
    }


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(part) for part in value)
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 0 <= score <= 1 else None


def _layout(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"display", "inline"} else "unknown"


def _xywh(bbox: tuple[float, float, float, float]) -> list[float]:
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def _area(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    return intersection / (_area(left) + _area(right) - intersection)
