"""Model-assisted formula-region annotation interchange for Label Studio."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pymupdf

_ANNOTATABLE_CLASSES = frozenset({"formula", "formula_number"})


def export_label_studio_tasks(
    pdf_path: Path | str,
    cache_path: Path | str,
    image_dir: Path | str,
    *,
    item_key: str,
    dpi: int = 150,
) -> list[dict[str, Any]]:
    """Render pages and export PP-DocLayout predictions as Label Studio tasks.

    Boxes are converted from PDF points to Label Studio's percentage coordinate
    system.  The source PDF hash is retained on every task, so annotations from
    a translated or otherwise different attachment cannot silently be reused.
    """
    pdf = Path(pdf_path)
    payload = _read_cache(cache_path, pdf)
    output_dir = Path(image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(str(pdf))
    try:
        tasks = []
        for page_data in _cache_pages(payload):
            page_num = _positive_int(page_data.get("page_num"))
            page_size = _page_size(page_data.get("page_size_pt"))
            if page_num is None or page_size is None or page_num > document.page_count:
                continue
            page = document[page_num - 1]
            image_path = output_dir / f"{item_key}-{page_num:04d}.png"
            page.get_pixmap(dpi=dpi, alpha=False).save(str(image_path))
            prediction = _label_studio_prediction(page_data, page_size)
            tasks.append(
                {
                    "data": {
                        "image": image_path.resolve().as_uri(),
                        "item_key": item_key,
                        "page_num": page_num,
                        "source_pdf_sha256": str(payload["source_pdf_sha256"]),
                    },
                    "meta": {
                        "page_size_pt": list(page_size),
                        "cache_generator": str(payload.get("generator", "")),
                        "cache_model": str(payload.get("model", "")),
                    },
                    "predictions": [prediction],
                }
            )
    finally:
        document.close()
    return tasks


def write_label_studio_tasks(tasks: Iterable[Mapping[str, Any]], output_path: Path | str) -> Path:
    """Write Label Studio tasks deterministically for import or review."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(tasks), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def import_label_studio_gold(tasks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Convert reviewed regions and optional per-region LaTeX into gold JSON."""
    documents: dict[tuple[str, str], dict[str, Any]] = {}
    for task in tasks:
        data = _mapping(task.get("data"))
        meta = _mapping(task.get("meta"))
        item_key = str(data.get("item_key", "")).strip()
        source_hash = str(data.get("source_pdf_sha256", "")).strip().lower()
        page_num = _positive_int(data.get("page_num"))
        page_size = _page_size(meta.get("page_size_pt"))
        if not item_key or not source_hash or page_num is None or page_size is None:
            continue
        annotation = _completed_annotation(task.get("annotations"))
        if annotation is None:
            continue
        key = (item_key, source_hash)
        document = documents.setdefault(
            key,
            {
                "item_key": item_key,
                "source_pdf_sha256": source_hash,
                "pages": [],
            },
        )
        document["pages"].append(
            {
                "page_num": page_num,
                "page_size_pt": list(page_size),
                "regions": _annotation_regions(annotation, page_size),
            }
        )
    for document in documents.values():
        document["pages"].sort(key=lambda page: int(page["page_num"]))
    return {
        "schema_version": 1,
        "generator": "zotpilot_label_studio_gold",
        "documents": sorted(documents.values(), key=lambda document: str(document["item_key"])),
    }


def _read_cache(cache_path: Path | str, pdf_path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read visual-layout cache: {cache_path}") from error
    if not isinstance(payload, Mapping) or str(payload.get("generator", "")) != "pp_doclayout":
        raise ValueError("Expected a PP-DocLayout cache")
    expected_hash = str(payload.get("source_pdf_sha256", "")).strip().lower()
    actual_hash = _sha256(pdf_path)
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError("Visual-layout cache does not belong to the supplied PDF")
    return payload


def _cache_pages(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    pages = payload.get("pages")
    return [page for page in pages if isinstance(page, Mapping)] if isinstance(pages, list) else []


def _label_studio_prediction(page: Mapping[str, Any], page_size: tuple[float, float]) -> dict[str, Any]:
    width, height = page_size
    result = []
    scores: list[float] = []
    for index, region in enumerate(page.get("regions", [])):
        if not isinstance(region, Mapping):
            continue
        label = str(region.get("cls", "")).strip().lower().replace(" ", "_")
        bbox = _bbox(region.get("bbox_pt"), page_size)
        confidence = _confidence(region.get("conf"))
        if label not in _ANNOTATABLE_CLASSES or bbox is None or confidence is None:
            continue
        x0, y0, x1, y1 = bbox
        result.append(
            {
                "id": f"pp-doclayout-{page.get('page_num', 0)}-{index}",
                "from_name": "formula_region",
                "to_name": "image",
                "type": "rectanglelabels",
                "score": confidence,
                "value": {
                    "x": 100 * x0 / width,
                    "y": 100 * y0 / height,
                    "width": 100 * (x1 - x0) / width,
                    "height": 100 * (y1 - y0) / height,
                    "rotation": 0,
                    "rectanglelabels": [label],
                },
                "original_width": int(round(width)),
                "original_height": int(round(height)),
            }
        )
        scores.append(confidence)
    return {
        "model_version": "pp_doclayout",
        "score": _mean(scores),
        "result": result,
    }


def _completed_annotation(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, list):
        return None
    for annotation in reversed(value):
        if isinstance(annotation, Mapping) and not bool(annotation.get("was_cancelled")):
            result = annotation.get("result")
            if isinstance(result, list):
                return annotation
    return None


def _annotation_regions(annotation: Mapping[str, Any], page_size: tuple[float, float]) -> list[dict[str, Any]]:
    width, height = page_size
    latex_by_region = _text_by_region(annotation, "formula_latex")
    number_by_region = _text_by_region(annotation, "equation_number")
    regions = []
    for result in annotation.get("result", []):
        if not isinstance(result, Mapping) or result.get("type") != "rectanglelabels":
            continue
        value = _mapping(result.get("value"))
        labels = value.get("rectanglelabels")
        label = str(labels[0]).strip().lower().replace(" ", "_") if isinstance(labels, list) and labels else ""
        if label not in _ANNOTATABLE_CLASSES:
            continue
        try:
            x = float(value["x"]) * width / 100
            y = float(value["y"]) * height / 100
            region_width = float(value["width"]) * width / 100
            region_height = float(value["height"]) * height / 100
        except (KeyError, TypeError, ValueError):
            continue
        if region_width <= 0 or region_height <= 0:
            continue
        region_id = str(result.get("id", "")).strip()
        regions.append(
            {
                "cls": label,
                "bbox_pt": [round(x, 3), round(y, 3), round(x + region_width, 3), round(y + region_height, 3)],
                "layout": "unknown",
                "equation_number": number_by_region.get(region_id, "") if label == "formula" else "",
                "latex": latex_by_region.get(region_id, "") if label == "formula" else "",
            }
        )
    return regions


def _text_by_region(annotation: Mapping[str, Any], from_name: str) -> dict[str, str]:
    """Read one Label Studio per-region TextArea without trusting global text."""
    values: dict[str, str] = {}
    for result in annotation.get("result", []):
        if not isinstance(result, Mapping) or str(result.get("from_name", "")) != from_name:
            continue
        parent_id = _parent_region_id(result)
        text_values = _mapping(result.get("value")).get("text")
        if not parent_id or not isinstance(text_values, list):
            continue
        latex = "\n".join(str(item).strip() for item in text_values if str(item).strip()).strip()
        if latex:
            values[parent_id] = latex
    return values


def _parent_region_id(result: Mapping[str, Any]) -> str:
    for key in ("parentID", "parent_id", "parentId"):
        value = str(result.get(key, "")).strip()
        if value:
            return value
    return ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _page_size(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        width, height = (float(part) for part in value)
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def _bbox(value: Any, page_size: tuple[float, float]) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(part) for part in value)
    except (TypeError, ValueError):
        return None
    width, height = page_size
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0 or x1 > width or y1 > height:
        return None
    return (x0, y0, x1, y1)


def _confidence(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 0 <= score <= 1 else None


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
