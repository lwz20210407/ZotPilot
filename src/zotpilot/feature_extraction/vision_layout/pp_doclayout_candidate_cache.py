"""Read formula-region evidence from PP-DocLayout caches.

The runner owns model execution and cache creation.  This module intentionally
only validates and reads that cache format so normal ZotPilot operation has no
PaddleOCR dependency.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PP_DOCLAYOUT_CACHE_GENERATOR = "pp_doclayout"
PP_DOCLAYOUT_FORMULA_LABELS = frozenset({"formula", "display_formula", "isolated"})
PP_DOCLAYOUT_FORMULA_NUMBER_LABELS = frozenset({"formula_number", "equation_number"})


@dataclass(frozen=True)
class PpDocLayoutFormulaRegion:
    """One validated display-formula region in PDF-point coordinates."""

    page_num: int
    bbox: tuple[float, float, float, float]
    confidence: float
    source_artifact_hash: str


@dataclass(frozen=True)
class PpDocLayoutFormulaNumberRegion:
    """One visual equation-number region, kept separate from formula blocks."""

    page_num: int
    bbox: tuple[float, float, float, float]
    confidence: float
    source_artifact_hash: str


@dataclass(frozen=True)
class PpDocLayoutColumnBlock:
    """One source-bound page-column block in PDF-point coordinates."""

    page_num: int
    bbox: tuple[float, float, float, float]
    source: str
    source_artifact_hash: str


def load_pp_doclayout_formula_regions(
    pdf_path: Path | str,
    cache_paths: Iterable[Path | str],
    *,
    min_confidence: float = 0.6,
) -> list[PpDocLayoutFormulaRegion]:
    """Return display-formula regions from caches bound to ``pdf_path``.

    A malformed cache or one produced for another PDF is ignored.  Formula
    number regions remain in the cache for the later geometry-binding stage;
    this detection-only reader deliberately does not bind them.
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    source_hash = _sha256(Path(pdf_path))
    regions: list[PpDocLayoutFormulaRegion] = []
    for cache_path in sorted({Path(path) for path in cache_paths}, key=lambda path: str(path).lower()):
        payload = _read_payload(cache_path)
        if payload is None or not _matches_source_pdf(payload, source_hash):
            continue
        artifact_hash = _sha256(cache_path)
        if not artifact_hash:
            continue
        regions.extend(
            _regions_from_payload(
                payload,
                artifact_hash=artifact_hash,
                min_confidence=min_confidence,
            )
        )
    return _dedupe_regions(regions)


def load_pp_doclayout_formula_number_regions(
    pdf_path: Path | str,
    cache_paths: Iterable[Path | str],
    *,
    min_confidence: float = 0.6,
) -> list[PpDocLayoutFormulaNumberRegion]:
    """Return visual ``formula_number`` regions from source-bound caches."""
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    source_hash = _sha256(Path(pdf_path))
    regions: list[PpDocLayoutFormulaNumberRegion] = []
    for cache_path in sorted({Path(path) for path in cache_paths}, key=lambda path: str(path).lower()):
        payload = _read_payload(cache_path)
        if payload is None or not _matches_source_pdf(payload, source_hash):
            continue
        artifact_hash = _sha256(cache_path)
        if not artifact_hash:
            continue
        regions.extend(
            _number_regions_from_payload(
                payload,
                artifact_hash=artifact_hash,
                min_confidence=min_confidence,
            )
        )
    return _dedupe_number_regions(regions)


def load_pp_doclayout_column_blocks(
    pdf_path: Path | str,
    cache_paths: Iterable[Path | str],
) -> list[PpDocLayoutColumnBlock]:
    """Return validated cached page columns bound to ``pdf_path``.

    Column blocks are optional derived layout evidence.  Callers must fall
    back to text projection if no source-bound block cache is available.
    """
    source_hash = _sha256(Path(pdf_path))
    blocks: list[PpDocLayoutColumnBlock] = []
    for cache_path in sorted({Path(path) for path in cache_paths}, key=lambda path: str(path).lower()):
        payload = _read_payload(cache_path)
        if payload is None or not _matches_source_pdf(payload, source_hash):
            continue
        artifact_hash = _sha256(cache_path)
        if not artifact_hash:
            continue
        blocks.extend(_column_blocks_from_payload(payload, artifact_hash=artifact_hash))
    return _dedupe_column_blocks(blocks)


def _read_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    if str(value.get("generator", "")).strip().lower() != PP_DOCLAYOUT_CACHE_GENERATOR:
        return None
    return value


def _matches_source_pdf(payload: Mapping[str, Any], source_hash: str) -> bool:
    expected = str(payload.get("source_pdf_sha256", "")).strip().lower()
    return bool(expected and source_hash and expected == source_hash.lower())


def _regions_from_payload(
    payload: Mapping[str, Any],
    *,
    artifact_hash: str,
    min_confidence: float,
) -> list[PpDocLayoutFormulaRegion]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return []
    regions: list[PpDocLayoutFormulaRegion] = []
    for page in pages:
        if not isinstance(page, Mapping) or str(page.get("coordinate_space", "")).lower() != "pdf":
            continue
        page_num = _positive_int(page.get("page_num"))
        page_size = _page_size(page.get("page_size_pt"))
        page_regions = page.get("regions")
        if page_num is None or page_size is None or not isinstance(page_regions, list):
            continue
        for region in page_regions:
            if not isinstance(region, Mapping):
                continue
            label = str(region.get("cls", "")).strip().lower().replace(" ", "_")
            confidence = _confidence(region.get("conf"))
            bbox = _bbox(region.get("bbox_pt"), page_size)
            if (
                label not in PP_DOCLAYOUT_FORMULA_LABELS
                or confidence is None
                or confidence < min_confidence
                or bbox is None
            ):
                continue
            regions.append(
                PpDocLayoutFormulaRegion(
                    page_num=page_num,
                    bbox=bbox,
                    confidence=confidence,
                    source_artifact_hash=artifact_hash,
                )
            )
    return regions


def _number_regions_from_payload(
    payload: Mapping[str, Any],
    *,
    artifact_hash: str,
    min_confidence: float,
) -> list[PpDocLayoutFormulaNumberRegion]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return []
    regions: list[PpDocLayoutFormulaNumberRegion] = []
    for page in pages:
        if not isinstance(page, Mapping) or str(page.get("coordinate_space", "")).lower() != "pdf":
            continue
        page_num = _positive_int(page.get("page_num"))
        page_size = _page_size(page.get("page_size_pt"))
        page_regions = page.get("regions")
        if page_num is None or page_size is None or not isinstance(page_regions, list):
            continue
        for region in page_regions:
            if not isinstance(region, Mapping):
                continue
            label = str(region.get("cls", "")).strip().lower().replace(" ", "_")
            confidence = _confidence(region.get("conf"))
            bbox = _bbox(region.get("bbox_pt"), page_size)
            if (
                label not in PP_DOCLAYOUT_FORMULA_NUMBER_LABELS
                or confidence is None
                or confidence < min_confidence
                or bbox is None
            ):
                continue
            regions.append(
                PpDocLayoutFormulaNumberRegion(
                    page_num=page_num,
                    bbox=bbox,
                    confidence=confidence,
                    source_artifact_hash=artifact_hash,
                )
            )
    return regions


def _column_blocks_from_payload(
    payload: Mapping[str, Any],
    *,
    artifact_hash: str,
) -> list[PpDocLayoutColumnBlock]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return []
    blocks: list[PpDocLayoutColumnBlock] = []
    for page in pages:
        if not isinstance(page, Mapping) or str(page.get("coordinate_space", "")).lower() != "pdf":
            continue
        page_num = _positive_int(page.get("page_num"))
        page_size = _page_size(page.get("page_size_pt"))
        page_blocks = page.get("blocks")
        if page_num is None or page_size is None or not isinstance(page_blocks, list):
            continue
        for block in page_blocks:
            if not isinstance(block, Mapping) or str(block.get("cls", "")).strip().lower() != "column":
                continue
            bbox = _bbox(block.get("bbox_pt"), page_size)
            if bbox is None:
                continue
            blocks.append(
                PpDocLayoutColumnBlock(
                    page_num=page_num,
                    bbox=bbox,
                    source=str(block.get("source", "")).strip(),
                    source_artifact_hash=artifact_hash,
                )
            )
    return blocks


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


def _confidence(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if 0 <= result <= 1 else None


def _bbox(value: Any, page_size: tuple[float, float]) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(part) for part in value)
    except (TypeError, ValueError):
        return None
    page_width, page_height = page_size
    if x0 < 0 or y0 < 0 or x1 > page_width or y1 > page_height or x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _dedupe_regions(regions: list[PpDocLayoutFormulaRegion]) -> list[PpDocLayoutFormulaRegion]:
    unique = {
        (region.page_num, region.bbox, region.source_artifact_hash): region
        for region in regions
    }
    return sorted(
        unique.values(),
        key=lambda region: (region.page_num, region.bbox[1], region.bbox[0], -region.confidence),
    )


def _dedupe_number_regions(
    regions: list[PpDocLayoutFormulaNumberRegion],
) -> list[PpDocLayoutFormulaNumberRegion]:
    unique = {
        (region.page_num, region.bbox, region.source_artifact_hash): region
        for region in regions
    }
    return sorted(
        unique.values(),
        key=lambda region: (region.page_num, region.bbox[1], region.bbox[0], -region.confidence),
    )


def _dedupe_column_blocks(blocks: list[PpDocLayoutColumnBlock]) -> list[PpDocLayoutColumnBlock]:
    unique = {
        (block.page_num, block.bbox, block.source_artifact_hash): block
        for block in blocks
    }
    return sorted(
        unique.values(),
        key=lambda block: (block.page_num, block.bbox[0], block.bbox[1], block.bbox[2]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()
