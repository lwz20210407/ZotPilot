"""Model-assisted formula-region annotation interchange for Label Studio."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import pymupdf

_ANNOTATABLE_CLASSES = frozenset({"formula", "formula_number"})


def export_label_studio_tasks(
    pdf_path: Path | str,
    cache_path: Path | str,
    image_dir: Path | str,
    *,
    item_key: str,
    dpi: int = 150,
    label_studio_local_files_root: Path | str | None = None,
    image_url_root: Path | str | None = None,
    image_url_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Render pages and export PP-DocLayout predictions as Label Studio tasks.

    Boxes are converted from PDF points to Label Studio's percentage coordinate
    system.  The source PDF hash is retained on every task, so annotations from
    a translated or otherwise different attachment cannot silently be reused.
    When ``label_studio_local_files_root`` is supplied, generated page images
    use Label Studio's local-files URL instead of a browser ``file://`` URL.
    ``image_url_root`` and ``image_url_prefix`` instead produce an absolute
    URL below a caller-managed static server, for Label Studio deployments
    whose local-files endpoint requires API-header authentication.
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
                        "image": _label_studio_image_url(
                            image_path,
                            label_studio_local_files_root,
                            image_url_root,
                            image_url_prefix,
                        ),
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


def _label_studio_image_url(
    image_path: Path,
    local_files_root: Path | str | None,
    image_url_root: Path | str | None,
    image_url_prefix: str | None,
) -> str:
    resolved_image = image_path.resolve()
    if (image_url_root is None) != (image_url_prefix is None):
        raise ValueError("image_url_root and image_url_prefix must be provided together")
    if local_files_root is not None and image_url_root is not None:
        raise ValueError("Use either Label Studio local-files URLs or a static image URL, not both")
    if local_files_root is None:
        if image_url_root is None:
            return resolved_image.as_uri()
        relative_path = _relative_image_path(resolved_image, image_url_root)
        return f"{str(image_url_prefix).rstrip('/')}/{quote(relative_path.as_posix(), safe='/')}"
    relative_path = _relative_image_path(resolved_image, local_files_root)
    return f"/data/local-files/?d={quote(relative_path.as_posix(), safe='/')}"


def _relative_image_path(image_path: Path, root: Path | str) -> Path:
    try:
        return image_path.relative_to(Path(root).resolve())
    except ValueError as error:
        raise ValueError("Generated page image is outside the configured image URL root") from error


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


def summarize_label_studio_review(tasks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Report readiness without confusing predictions with completed review."""
    task_count = 0
    preannotated_task_count = 0
    reviewed_task_count = 0
    browser_file_image_url_task_count = 0
    static_http_image_url_task_count = 0
    label_studio_image_url_task_count = 0
    missing_image_url_task_count = 0
    unsupported_image_url_task_count = 0
    reviewed_documents: set[tuple[str, str]] = set()
    formula_region_count = 0
    latex_annotated_count = 0
    equation_number_annotated_count = 0
    layout_annotated_count = 0
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_count += 1
        image_url_kind = _review_image_url_kind(_mapping(task.get("data")).get("image"))
        if image_url_kind == "browser_file":
            browser_file_image_url_task_count += 1
        elif image_url_kind == "static_http":
            static_http_image_url_task_count += 1
        elif image_url_kind == "label_studio_local":
            label_studio_image_url_task_count += 1
        elif image_url_kind == "missing":
            missing_image_url_task_count += 1
        else:
            unsupported_image_url_task_count += 1
        predictions = task.get("predictions")
        if isinstance(predictions, list) and any(isinstance(item, Mapping) for item in predictions):
            preannotated_task_count += 1
        annotation = _completed_annotation(task.get("annotations"))
        data = _mapping(task.get("data"))
        meta = _mapping(task.get("meta"))
        page_size = _page_size(meta.get("page_size_pt"))
        if annotation is None or page_size is None:
            continue
        reviewed_task_count += 1
        item_key = str(data.get("item_key", "")).strip()
        source_hash = str(data.get("source_pdf_sha256", "")).strip().lower()
        if item_key and source_hash:
            reviewed_documents.add((item_key, source_hash))
        for region in _annotation_regions(annotation, page_size):
            if region["cls"] != "formula":
                continue
            formula_region_count += 1
            latex_annotated_count += bool(region["latex"])
            equation_number_annotated_count += bool(region["equation_number"])
            layout_annotated_count += region["layout"] in {"display", "inline"}
    return {
        "mode": "formula_label_studio_review_readiness",
        "task_count": task_count,
        "preannotated_task_count": preannotated_task_count,
        "reviewed_task_count": reviewed_task_count,
        "unreviewed_task_count": task_count - reviewed_task_count,
        "browser_file_image_url_task_count": browser_file_image_url_task_count,
        "static_http_image_url_task_count": static_http_image_url_task_count,
        "label_studio_image_url_task_count": label_studio_image_url_task_count,
        "missing_image_url_task_count": missing_image_url_task_count,
        "unsupported_image_url_task_count": unsupported_image_url_task_count,
        "review_image_url_ready": (
            task_count > 0
            and browser_file_image_url_task_count == 0
            and missing_image_url_task_count == 0
            and unsupported_image_url_task_count == 0
        ),
        "reviewed_document_count": len(reviewed_documents),
        "formula_region_count": formula_region_count,
        "latex_annotated_count": latex_annotated_count,
        "equation_number_annotated_count": equation_number_annotated_count,
        "layout_annotated_count": layout_annotated_count,
        "latex_coverage": _ratio(latex_annotated_count, formula_region_count),
        "equation_number_coverage": _ratio(equation_number_annotated_count, formula_region_count),
        "layout_coverage": _ratio(layout_annotated_count, formula_region_count),
    }


def _review_image_url_kind(value: Any) -> str:
    """Classify task images without attempting network access in readiness checks."""
    image_url = str(value or "").strip()
    if not image_url:
        return "missing"
    parsed = urlparse(image_url)
    if parsed.scheme.lower() == "file":
        return "browser_file"
    if parsed.scheme.lower() in {"http", "https"}:
        return "static_http"
    if not parsed.scheme and image_url.startswith("/data/local-files/"):
        return "label_studio_local"
    return "unsupported"


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
    layout_by_region = _choices_by_region(annotation, "formula_layout")
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
                "layout": layout_by_region.get(region_id, "unknown") if label == "formula" else "unknown",
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


def _choices_by_region(annotation: Mapping[str, Any], from_name: str) -> dict[str, str]:
    """Read one per-region Label Studio single-choice control."""
    values: dict[str, str] = {}
    for result in annotation.get("result", []):
        if not isinstance(result, Mapping) or str(result.get("from_name", "")) != from_name:
            continue
        parent_id = _parent_region_id(result)
        choices = _mapping(result.get("value")).get("choices")
        if not parent_id or not isinstance(choices, list) or len(choices) != 1:
            continue
        choice = str(choices[0]).strip().lower()
        if choice in {"display", "inline"}:
            values[parent_id] = choice
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


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
