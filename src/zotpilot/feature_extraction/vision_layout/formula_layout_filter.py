"""Classify visual math regions before equation-number binding.

PP-DocLayout detects both isolated display equations and small inline or
figure-local mathematical fragments.  This module keeps that detector output
intact, but makes the next stage conservative: only regions with display
geometry are eligible for equation-number binding.  It is deliberately a
read-only PDF geometry pass and has no OCR or vector-store dependency.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pymupdf

from ..formula_ocr import FormulaCandidate
from .pp_doclayout_candidate_cache import PpDocLayoutFormulaNumberRegion


@dataclass(frozen=True)
class PageColumn:
    """A horizontal reading column in PDF-point coordinates."""

    page_num: int
    index: int
    x0: float
    x1: float


@dataclass(frozen=True)
class FormulaLayoutAnalysis:
    """Display-filtered candidates plus the page columns used for binding."""

    candidates: tuple[FormulaCandidate, ...]
    columns_by_page: Mapping[int, tuple[PageColumn, ...]]


def classify_visual_formula_candidates(
    pdf_path: Path | str,
    candidates: Iterable[FormulaCandidate],
    number_regions: Iterable[PpDocLayoutFormulaNumberRegion],
    *,
    min_confidence: float = 0.6,
    number_assisted_floor: float = 0.5,
) -> FormulaLayoutAnalysis:
    """Attach a conservative layout kind and retain only display candidates.

    A nearby visual equation-number box is the strongest display signal.  For
    unnumbered candidates, an inline decision needs prose on both sides of the
    same text line; otherwise a sufficiently wide isolated region is retained
    as ``display``.  Small ambiguous visual fragments remain ``unknown`` and
    are kept out of OCR/indexing until a later review stage.
    """
    if not 0 <= number_assisted_floor <= min_confidence <= 1:
        raise ValueError("number_assisted_floor must be between 0 and min_confidence")
    original = list(candidates)
    number_by_page: dict[int, list[PpDocLayoutFormulaNumberRegion]] = defaultdict(list)
    for region in number_regions:
        number_by_page[region.page_num].append(region)
    words_by_page, page_widths = _page_words(pdf_path)
    columns_by_page = {
        page_num: _infer_page_columns(page_num, page_width, words)
        for page_num, (page_width, words) in words_by_page.items()
    }
    # Empty-text PDFs can still have visual formula candidates.  Use a single
    # full-width column so the geometry gate remains defined.
    for candidate in original:
        columns_by_page.setdefault(
            candidate.page_num,
            (PageColumn(candidate.page_num, 0, 0.0, page_widths.get(candidate.page_num, candidate.bbox[2])),),
        )

    classified: list[FormulaCandidate] = []
    for candidate in original:
        if candidate.source != "pp_doclayout_region":
            classified.append(candidate)
            continue
        page_width = page_widths.get(candidate.page_num, candidate.bbox[2])
        words = words_by_page.get(candidate.page_num, (page_width, ()))[1]
        layout_kind, number_assisted = _classify_layout_kind(
            candidate,
            number_by_page.get(candidate.page_num, ()),
            words,
            page_width,
        )
        if candidate.confidence < min_confidence and not (
            number_assisted and candidate.confidence >= number_assisted_floor
        ):
            layout_kind = "unknown"
        flags = list(candidate.quality_flags)
        flags.append(f"layout_{layout_kind}")
        if number_assisted and candidate.confidence < min_confidence:
            flags.append("low_detector_confidence_number_assisted")
        classified.append(replace(candidate, layout_kind=layout_kind, quality_flags=tuple(dict.fromkeys(flags))))
    return FormulaLayoutAnalysis(
        candidates=tuple(candidate for candidate in classified if candidate.layout_kind == "display"),
        columns_by_page=columns_by_page,
    )


def _page_words(
    pdf_path: Path | str,
) -> tuple[dict[int, tuple[float, tuple[tuple[float, float, float, float], ...]]], dict[int, float]]:
    result: dict[int, tuple[float, tuple[tuple[float, float, float, float], ...]]] = {}
    widths: dict[int, float] = {}
    try:
        document = pymupdf.open(str(pdf_path))
    except (OSError, RuntimeError):
        return result, widths
    try:
        for page_index in range(document.page_count):
            page = document[page_index]
            width = float(page.rect.width)
            widths[page_index + 1] = width
            words = tuple(
                (float(word[0]), float(word[1]), float(word[2]), float(word[3]))
                for word in page.get_text("words")
                if float(word[2]) > float(word[0]) and float(word[3]) > float(word[1])
            )
            result[page_index + 1] = (width, words)
    finally:
        document.close()
    return result, widths


def _infer_page_columns(
    page_num: int,
    page_width: float,
    words: tuple[tuple[float, float, float, float], ...],
) -> tuple[PageColumn, ...]:
    """Infer a two-column gutter from text occupancy, else use one column."""
    if page_width <= 0 or len(words) < 24:
        return (PageColumn(page_num, 0, 0.0, page_width),)
    bins = 32
    occupancy = [0] * bins
    for x0, _y0, x1, _y1 in words:
        start = max(0, min(bins - 1, int(x0 / page_width * bins)))
        end = max(0, min(bins - 1, int(max(x0, x1 - 0.001) / page_width * bins)))
        for index in range(start, end + 1):
            occupancy[index] += 1
    candidates = range(int(bins * 0.36), int(bins * 0.64) + 1)
    gutter_bin = min(candidates, key=lambda index: occupancy[index])
    split = (gutter_bin + 0.5) * page_width / bins
    left_words = sum(1 for x0, _y0, x1, _y1 in words if (x0 + x1) / 2 < split - page_width * 0.08)
    right_words = sum(1 for x0, _y0, x1, _y1 in words if (x0 + x1) / 2 > split + page_width * 0.08)
    center_density = occupancy[gutter_bin]
    side_density = min(sum(occupancy[:gutter_bin]), sum(occupancy[gutter_bin + 1 :]))
    if left_words >= 12 and right_words >= 12 and side_density > 0 and center_density * 12 <= side_density:
        return (
            PageColumn(page_num, 0, 0.0, split),
            PageColumn(page_num, 1, split, page_width),
        )
    return (PageColumn(page_num, 0, 0.0, page_width),)


def _classify_layout_kind(
    candidate: FormulaCandidate,
    number_regions: Iterable[PpDocLayoutFormulaNumberRegion],
    words: tuple[tuple[float, float, float, float], ...],
    page_width: float,
)-> tuple[str, bool]:
    if any(_is_number_neighbor(candidate.bbox, region.bbox, page_width) for region in number_regions):
        return "display", True
    if _has_prose_on_both_sides(candidate.bbox, words):
        return "inline", False
    width = candidate.bbox[2] - candidate.bbox[0]
    if page_width > 0 and width >= page_width * 0.28:
        return "display", False
    return "unknown", False


def _is_number_neighbor(
    formula: tuple[float, float, float, float],
    number: tuple[float, float, float, float],
    page_width: float,
) -> bool:
    formula_height = max(formula[3] - formula[1], 1.0)
    number_height = max(number[3] - number[1], 1.0)
    vertical_limit = max(formula_height, number_height) * 1.25
    return (
        number[0] >= formula[2] - 2.0
        and number[0] - formula[2] <= page_width * 0.8
        and abs((formula[1] + formula[3]) / 2 - (number[1] + number[3]) / 2) <= vertical_limit
    )


def _has_prose_on_both_sides(
    formula: tuple[float, float, float, float],
    words: tuple[tuple[float, float, float, float], ...],
) -> bool:
    x0, y0, x1, y1 = formula
    vertical_overlap = max(y1 - y0, 1.0) * 0.45
    left = False
    right = False
    for word_x0, word_y0, word_x1, word_y1 in words:
        if min(y1, word_y1) - max(y0, word_y0) < vertical_overlap:
            continue
        if 0 <= x0 - word_x1 <= 48:
            left = True
        if 0 <= word_x0 - x1 <= 48:
            right = True
        if left and right:
            return True
    return False
