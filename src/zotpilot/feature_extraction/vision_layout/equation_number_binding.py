"""Bind PP-DocLayout equation-number regions to visual formula blocks."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pymupdf

from ..formula_ocr import FormulaCandidate
from .formula_layout_filter import PageColumn
from .pp_doclayout_candidate_cache import (
    PpDocLayoutFormulaNumberRegion,
    load_pp_doclayout_formula_number_regions,
)

_NUMBER_LABEL_RE = re.compile(
    r"^\s*[（(]\s*(?P<number>(?:[A-Za-z](?:[.:]\s*)?)?\d+(?:[.\-]\d+)*(?:[A-Za-z])?)\s*[）)]\s*$"
)
# A display formula can be short while its number remains flush with the page
# right edge.  PR-3 intentionally covers single-column geometry; PR-3.5 adds
# explicit column boundaries before this permissive bound is used for multi-
# column pages.
_MAX_HORIZONTAL_GAP_FRACTION = 0.8
# Detector boxes may include the formula's trailing whitespace, leaving only a
# few PDF points before a genuine right-edge number.  The maximum-gap and
# one-token/one-block constraints prevent cross-column matches; tolerate up to
# two points of detector-box overlap at the formula tail.
_MIN_HORIZONTAL_GAP = -2.0


@dataclass(frozen=True)
class EquationNumberBindingResult:
    """Bound candidates and diagnostics from a non-destructive geometry pass."""

    candidates: tuple[FormulaCandidate, ...]
    bound_numbers: tuple[str, ...]
    rejected_number_region_count: int
    sequence_gap_numbers: tuple[str, ...]
    duplicate_numbers: tuple[str, ...]
    non_monotonic_pairs: tuple[tuple[str, str], ...]


def bind_pp_doclayout_equation_numbers(
    pdf_path: Path | str,
    candidates: Iterable[FormulaCandidate],
    cache_paths: Iterable[Path | str],
    *,
    min_confidence: float = 0.6,
    columns_by_page: Mapping[int, tuple[PageColumn, ...]] | None = None,
) -> EquationNumberBindingResult:
    """Bind only visual number boxes that contain a standalone equation label.

    Text such as ``Eq. (5)`` is rejected because it does not fill a dedicated
    formula-number region with only a parenthesized identifier.  Missing or
    ambiguous bindings remain unnumbered; this function never invents IDs.
    """
    original = list(candidates)
    number_regions = load_pp_doclayout_formula_number_regions(
        pdf_path,
        cache_paths,
        min_confidence=min_confidence,
    )
    tokens = _read_number_tokens(pdf_path, number_regions)
    matches = _binding_matches(original, tokens, pdf_path, columns_by_page=columns_by_page or {})
    bound_candidates = list(original)
    used_candidates: set[int] = set()
    used_tokens: set[int] = set()
    bound_numbers: list[str] = []
    for _score, candidate_index, token_index in matches:
        if candidate_index in used_candidates or token_index in used_tokens:
            continue
        candidate = bound_candidates[candidate_index]
        if candidate.equation_number:
            continue
        token = tokens[token_index]
        bound_candidates[candidate_index] = replace(
            candidate,
            equation_number=token.number,
            equation_number_status="detected_region",
        )
        used_candidates.add(candidate_index)
        used_tokens.add(token_index)
        bound_numbers.append(token.number)
    diagnosed, gaps, duplicates, non_monotonic = _annotate_sequence_quality(bound_candidates)
    return EquationNumberBindingResult(
        candidates=tuple(diagnosed),
        bound_numbers=tuple(bound_numbers),
        rejected_number_region_count=max(len(number_regions) - len(tokens), 0) + max(len(tokens) - len(used_tokens), 0),
        sequence_gap_numbers=tuple(gaps),
        duplicate_numbers=tuple(duplicates),
        non_monotonic_pairs=tuple(non_monotonic),
    )


@dataclass(frozen=True)
class _NumberToken:
    number: str
    page_num: int
    bbox: tuple[float, float, float, float]
    confidence: float
    page_width: float


def _read_number_tokens(
    pdf_path: Path | str,
    regions: Iterable[PpDocLayoutFormulaNumberRegion],
) -> list[_NumberToken]:
    by_page: dict[int, list[PpDocLayoutFormulaNumberRegion]] = defaultdict(list)
    for region in regions:
        by_page[region.page_num].append(region)
    tokens: list[_NumberToken] = []
    try:
        document = pymupdf.open(str(pdf_path))
    except (OSError, RuntimeError):
        return tokens
    try:
        for page_num, page_regions in by_page.items():
            if not 1 <= page_num <= document.page_count:
                continue
            page = document[page_num - 1]
            for region in page_regions:
                number = _equation_number_from_region_text(_text_in_bbox(page, region.bbox))
                if not number:
                    continue
                tokens.append(
                    _NumberToken(
                        number=number,
                        page_num=page_num,
                        bbox=region.bbox,
                        confidence=region.confidence,
                        page_width=float(page.rect.width),
                    )
                )
    finally:
        document.close()
    return tokens


def _text_in_bbox(page: pymupdf.Page, bbox: tuple[float, float, float, float]) -> str:
    region = pymupdf.Rect(*bbox)
    words = page.get_text("words")
    values = [
        str(word[4])
        for word in words
        if _intersection_ratio(region, pymupdf.Rect(*word[:4])) >= 0.5
    ]
    return "".join(values)


def _equation_number_from_region_text(text: str) -> str:
    match = _NUMBER_LABEL_RE.fullmatch(text or "")
    if match is None:
        return ""
    number = re.sub(r"\s+", "", match.group("number"))
    return f"({number})"


def _binding_matches(
    candidates: list[FormulaCandidate],
    tokens: list[_NumberToken],
    pdf_path: Path | str,
    *,
    columns_by_page: Mapping[int, tuple[PageColumn, ...]],
) -> list[tuple[float, int, int]]:
    try:
        document = pymupdf.open(str(pdf_path))
    except (OSError, RuntimeError):
        return []
    try:
        page_widths = {
            index + 1: float(document[index].rect.width)
            for index in range(document.page_count)
        }
    finally:
        document.close()
    matches: list[tuple[float, int, int]] = []
    strict_match_tokens: set[int] = set()
    for candidate_index, candidate in enumerate(candidates):
        if candidate.source != "pp_doclayout_region" or candidate.equation_number:
            continue
        for token_index, token in enumerate(tokens):
            if _can_bind(
                candidate,
                token,
                page_widths.get(candidate.page_num, 0.0),
                columns_by_page.get(candidate.page_num, ()),
            ):
                matches.append((_binding_score(candidate, token), candidate_index, token_index))
                strict_match_tokens.add(token_index)
    # A short full-width equation can start in the left margin even on a page
    # whose body is otherwise two-column.  Permit that outer-margin fallback
    # only when no same-column candidate exists for that number token; this
    # prevents a left-column formula from stealing a real right-column label.
    for candidate_index, candidate in enumerate(candidates):
        if candidate.source != "pp_doclayout_region" or candidate.equation_number:
            continue
        for token_index, token in enumerate(tokens):
            if token_index in strict_match_tokens:
                continue
            page_width = page_widths.get(candidate.page_num, 0.0)
            if (
                _can_bind(candidate, token, page_width, ())
                and _is_outer_margin_fallback(candidate.bbox, token.bbox, page_width)
            ):
                matches.append((_binding_score(candidate, token) + 25.0, candidate_index, token_index))
    return sorted(matches)


def _can_bind(
    candidate: FormulaCandidate,
    token: _NumberToken,
    page_width: float,
    columns: tuple[PageColumn, ...],
) -> bool:
    if candidate.page_num != token.page_num or page_width <= 0:
        return False
    candidate_x0, candidate_y0, candidate_x1, candidate_y1 = candidate.bbox
    token_x0, token_y0, token_x1, token_y1 = token.bbox
    if token_x0 < candidate_x1 + _MIN_HORIZONTAL_GAP:
        return False
    if token_x0 - candidate_x1 > page_width * _MAX_HORIZONTAL_GAP_FRACTION:
        return False
    candidate_height = max(candidate_y1 - candidate_y0, 1.0)
    token_height = max(token_y1 - token_y0, 1.0)
    vertical_limit = max(candidate_height, token_height) * 1.25
    if abs(_center_y(candidate.bbox) - _center_y(token.bbox)) > vertical_limit:
        return False
    if columns and not _same_column(candidate.bbox, token.bbox, columns):
        return False
    return token_x1 > token_x0 and candidate_x1 > candidate_x0


def _same_column(
    candidate: tuple[float, float, float, float],
    token: tuple[float, float, float, float],
    columns: tuple[PageColumn, ...],
) -> bool:
    if len(columns) < 2:
        return True
    if any(candidate[0] < column.x0 < candidate[2] for column in columns[1:]):
        return True
    candidate_center = (candidate[0] + candidate[2]) / 2
    candidate_column = next((column for column in columns if column.x0 <= candidate_center <= column.x1), None)
    if candidate_column is None:
        return False
    # A detector's number box can straddle the gutter even though it is the
    # right-edge label of the left column.  Use the label's left edge and a
    # small gutter tolerance instead of its center for that boundary case.
    gutter_tolerance = max(18.0, (candidate_column.x1 - candidate_column.x0) * 0.10)
    return candidate_column.x0 - gutter_tolerance <= token[0] <= candidate_column.x1 + gutter_tolerance


def _is_outer_margin_fallback(
    candidate: tuple[float, float, float, float],
    token: tuple[float, float, float, float],
    page_width: float,
) -> bool:
    return page_width > 0 and candidate[0] <= page_width * 0.18 and token[0] >= page_width * 0.8


def _binding_score(candidate: FormulaCandidate, token: _NumberToken) -> float:
    vertical_distance = abs(_center_y(candidate.bbox) - _center_y(token.bbox))
    horizontal_gap = token.bbox[0] - candidate.bbox[2]
    return vertical_distance + horizontal_gap * 0.01 - token.confidence * 0.001


def _annotate_sequence_quality(
    candidates: list[FormulaCandidate],
) -> tuple[list[FormulaCandidate], list[str], list[str], list[tuple[str, str]]]:
    updated = list(candidates)
    seen: set[tuple[str, tuple[int, ...]]] = set()
    previous_key: tuple[str, tuple[int, ...]] | None = None
    previous_number = ""
    gaps: list[str] = []
    duplicates: list[str] = []
    non_monotonic: list[tuple[str, str]] = []
    ordered_indices = sorted(
        range(len(updated)),
        key=lambda i: (updated[i].page_num, updated[i].bbox[1], updated[i].bbox[0]),
    )
    for index in ordered_indices:
        candidate = updated[index]
        key = _number_key(candidate.equation_number)
        if key is None:
            continue
        flags = list(candidate.quality_flags)
        if key in seen:
            flags.append("equation_number_duplicate")
            duplicates.append(candidate.equation_number)
        elif previous_key is not None and key[0] == previous_key[0] and key[1][:-1] == previous_key[1][:-1]:
            if key[1][-1] < previous_key[1][-1]:
                flags.append("equation_number_non_monotonic")
                non_monotonic.append((previous_number, candidate.equation_number))
            elif key[1][-1] > previous_key[1][-1] + 1:
                flags.append("equation_number_gap_before")
                prefix = ".".join(str(part) for part in key[1][:-1])
                gaps.extend(
                    f"({f'{prefix}.' if prefix else ''}{number})"
                    for number in range(previous_key[1][-1] + 1, key[1][-1])
                )
        if flags != list(candidate.quality_flags):
            updated[index] = replace(candidate, quality_flags=tuple(dict.fromkeys(flags)))
        seen.add(key)
        previous_key = key
        previous_number = candidate.equation_number
    return updated, list(dict.fromkeys(gaps)), list(dict.fromkeys(duplicates)), list(dict.fromkeys(non_monotonic))


def _number_key(value: str) -> tuple[str, tuple[int, ...]] | None:
    match = _NUMBER_LABEL_RE.fullmatch(value or "")
    if match is None:
        return None
    normalized = re.sub(r"\s+", "", match.group("number"))
    section = ""
    if normalized[:1].isalpha():
        section, normalized = normalized[0].upper(), normalized[1:]
        normalized = normalized[1:] if normalized.startswith(".") else normalized
    parts = normalized.split(".")
    if not parts or not all(part.isdigit() for part in parts):
        return None
    return section, tuple(int(part) for part in parts)


def _center_y(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def _intersection_ratio(left: pymupdf.Rect, right: pymupdf.Rect) -> float:
    intersection = left & right
    if intersection.is_empty or right.get_area() <= 0:
        return 0.0
    return float(intersection.get_area()) / float(right.get_area())
