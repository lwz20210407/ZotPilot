"""PP-DocLayout runner that emits PDF-point visual-layout caches.

The PaddleOCR import happens only when ``export_pp_doclayout_cache`` runs, so
ZotPilot's normal installation does not require PaddlePaddle.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pymupdf

SCHEMA_VERSION = 1
GENERATOR = "pp_doclayout"


def export_pp_doclayout_cache(
    pdf_path: Path | str,
    output_path: Path | str,
    *,
    model_name: str = "PP-DocLayout_plus-L",
    model_dir: Path | str | None = None,
    dpi: int = 200,
    confidence_threshold: float | None = 0.5,
    device: str = "cpu",
    engine: str | None = "onnxruntime",
    max_pages: int | None = None,
) -> Path:
    """Run PP-DocLayout and atomically write a deterministic region cache."""
    try:
        from paddleocr import LayoutDetection
    except ImportError as error:
        raise RuntimeError("PP-DocLayout requires the optional paddleocr dependency.") from error

    pdf = Path(pdf_path)
    output = Path(output_path)
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive when provided")

    output.parent.mkdir(parents=True, exist_ok=True)
    model = LayoutDetection(
        model_name=model_name,
        model_dir=str(model_dir) if model_dir else None,
        device=device,
        threshold=confidence_threshold,
        engine=engine,
    )
    with tempfile.TemporaryDirectory(prefix="zotpilot-pp-doclayout-") as temporary_directory:
        pages = _predict_pages(pdf, model, Path(temporary_directory), dpi=dpi, max_pages=max_pages)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "model": model_name,
        "source_pdf_sha256": _sha256(pdf),
        "render_dpi": dpi,
        "confidence_threshold": confidence_threshold,
        "pages": pages,
    }
    _write_json_atomically(output, payload)
    return output


def _predict_pages(
    pdf_path: Path,
    model: Any,
    image_dir: Path,
    *,
    dpi: int,
    max_pages: int | None,
) -> list[dict[str, object]]:
    document = pymupdf.open(str(pdf_path))
    pages: list[dict[str, object]] = []
    try:
        for page_index in range(document.page_count):
            if max_pages is not None and page_index >= max_pages:
                break
            page: Any = document[page_index]
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            image_path = image_dir / f"page-{page_index + 1:04d}.png"
            pixmap.save(str(image_path))
            results = list(model.predict(str(image_path), batch_size=1, layout_nms=True))
            if len(results) != 1:
                raise RuntimeError(f"Expected one PP-DocLayout result for page {page_index + 1}, got {len(results)}")
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            pages.append(
                {
                    "page_num": page_index + 1,
                    "page_size_pt": [page_width, page_height],
                    "image_size_px": [pixmap.width, pixmap.height],
                    "coordinate_space": "pdf",
                    "regions": _pdf_regions(
                        results[0],
                        image_width=float(pixmap.width),
                        image_height=float(pixmap.height),
                        page_width=page_width,
                        page_height=page_height,
                    ),
                }
            )
    finally:
        document.close()
    return pages


def _pdf_regions(
    result: Any,
    *,
    image_width: float,
    image_height: float,
    page_width: float,
    page_height: float,
) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    for box in _result_boxes(result):
        coordinate = box.get("coordinate")
        if not isinstance(coordinate, list) or len(coordinate) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(value) for value in coordinate)
        except (TypeError, ValueError):
            continue
        bbox = [
            round(x0 * page_width / image_width, 3),
            round(y0 * page_height / image_height, 3),
            round(x1 * page_width / image_width, 3),
            round(y1 * page_height / image_height, 3),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        regions.append(
            {
                "cls": str(box.get("label", "")),
                "bbox_pt": bbox,
                "conf": round(float(box.get("score", 0.0)), 6),
            }
        )
    return regions


def _result_boxes(result: Any) -> list[Mapping[str, Any]]:
    payload = _result_mapping(result)
    body = payload.get("res", payload)
    boxes = body.get("boxes", []) if isinstance(body, Mapping) else []
    return [box for box in boxes if isinstance(box, Mapping)] if isinstance(boxes, list) else []


def _result_mapping(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        return result
    for attribute in ("res", "json"):
        value = getattr(result, attribute, None)
        value = value() if callable(value) else value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = None
        if isinstance(value, Mapping):
            return value
    raise TypeError(f"Unsupported PaddleOCR result type: {type(result)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
