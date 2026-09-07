"""Bind PP-DocLayout equation-number regions to visual formula blocks."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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

    source_pdf_sha256: str
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
    diagnosed, gaps, duplicates, non_monotonic = _annotate_sequence_quality(
        bound_candidates, columns_by_page=columns_by_page or {},
    )
    return EquationNumberBindingResult(
        source_pdf_sha256=_sha256(Path(pdf_path)),
        candidates=tuple(diagnosed),
        bound_numbers=tuple(bound_numbers),
        rejected_number_region_count=max(len(number_regions) - len(tokens), 0) + max(len(tokens) - len(used_tokens), 0),
        sequence_gap_numbers=tuple(gaps),
        duplicate_numbers=tuple(duplicates),
        non_monotonic_pairs=tuple(non_monotonic),
    )


def evaluate_equation_number_binding_gold(
    gold: Mapping[str, Any],
    result: EquationNumberBindingResult,
    *,
    item_key: str,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Measure formula-to-number bindings against reviewed, source-bound Gold.

    A correct-looking number is not enough: it must be assigned to the Gold
    formula block at the same page and IoU-matched location.  This separates
    a wrong adjacent label, a missed label, and a spurious number on a
    non-formula candidate, which are different failure modes for retrieval.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    document = _gold_document_for_binding(gold, item_key, result.source_pdf_sha256)
    gold_rows = _gold_numbered_formula_rows(document)
    matches = _match_gold_formulas(gold_rows, list(result.candidates), iou_threshold)
    matched_candidates = {candidate_index for _, candidate_index, _ in matches}
    by_gold = {gold_index: (candidate_index, iou) for gold_index, candidate_index, iou in matches}
    true_positive = 0
    wrong_numbers: list[dict[str, Any]] = []
    missing_numbers: list[dict[str, Any]] = []
    for gold_index, gold_row in enumerate(gold_rows):
        match = by_gold.get(gold_index)
        if match is None:
            missing_numbers.append({**_gold_identity(gold_row), "reason": "formula_not_detected"})
            continue
        candidate_index, iou = match
        candidate = result.candidates[candidate_index]
        predicted_number = _normalize_equation_number(candidate.equation_number)
        if not predicted_number:
            missing_numbers.append(
                {
                    **_gold_identity(gold_row),
                    "reason": "formula_number_unbound",
                    "iou": round(iou, 6),
                }
            )
        elif predicted_number == gold_row["equation_number"]:
            true_positive += 1
        else:
            wrong_numbers.append(
                {
                    **_gold_identity(gold_row),
                    "predicted_equation_number": candidate.equation_number,
                    "iou": round(iou, 6),
                }
            )
    spurious_numbers = [
        {
            "page_num": candidate.page_num,
            "bbox_pt": list(candidate.bbox),
            "equation_number": candidate.equation_number,
        }
        for index, candidate in enumerate(result.candidates)
        if index not in matched_candidates and _normalize_equation_number(candidate.equation_number)
    ]
    predicted_number_count = sum(
        bool(_normalize_equation_number(candidate.equation_number)) for candidate in result.candidates
    )
    gold_count = len(gold_rows)
    return {
        "mode": "formula_number_binding_gold_iou",
        "item_key": item_key,
        "source_pdf_sha256": result.source_pdf_sha256,
        "iou_threshold": iou_threshold,
        "gold_numbered_formula_count": gold_count,
        "predicted_numbered_formula_count": predicted_number_count,
        "true_positive": true_positive,
        "wrong_number_count": len(wrong_numbers),
        "missing_number_count": len(missing_numbers),
        "spurious_number_count": len(spurious_numbers),
        "precision": _ratio(true_positive, predicted_number_count),
        "recall": _ratio(true_positive, gold_count),
        "wrong_numbers": wrong_numbers,
        "missing_numbers": missing_numbers,
        "spurious_numbers": spurious_numbers,
    }


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


def _binding_score(candidate: FormulaCandidate, token: _NumberToken) -> float:
    vertical_distance = abs(_center_y(candidate.bbox) - _center_y(token.bbox))
    horizontal_gap = token.bbox[0] - candidate.bbox[2]
    return vertical_distance + horizontal_gap * 0.01 - token.confidence * 0.001


def _annotate_sequence_quality(
    candidates: list[FormulaCandidate],
    *,
    columns_by_page: Mapping[int, tuple[PageColumn, ...]],
) -> tuple[list[FormulaCandidate], list[str], list[str], list[tuple[str, str]]]:
    updated = list(candidates)
    seen: set[tuple[str, tuple[int, ...]]] = set()
    previous_key: tuple[str, tuple[int, ...]] | None = None
    previous_number = ""
    gaps: list[str] = []
    duplicates: list[str] = []
    non_monotonic: list[tuple[str, str]] = []
    # Mixed page-wide and column-local formulas need reading-order bands.
    # Until those bands are available, retain bindings but abstain on sequence.
    ambiguous_pages = {
        candidate.page_num for candidate in updated
        if len(columns_by_page.get(candidate.page_num, ())) > 1
        and _candidate_column(candidate, columns_by_page[candidate.page_num]) is None
    }
    ordered_indices = sorted(
        range(len(updated)),
        key=lambda i: (
            updated[i].page_num,
            _candidate_column(updated[i], columns_by_page.get(updated[i].page_num, ())) or 0.0,
            updated[i].bbox[1], updated[i].bbox[0],
        ),
    )
    for index in ordered_indices:
        candidate = updated[index]
        key = _number_key(candidate.equation_number)
        if key is None:
            continue
        flags = list(candidate.quality_flags)
        if candidate.page_num in ambiguous_pages:
            flags.append("equation_number_reading_order_ambiguous")
            updated[index] = replace(candidate, quality_flags=tuple(dict.fromkeys(flags)))
            previous_key = None
            previous_number = ""
            continue
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


def _candidate_column(candidate: FormulaCandidate, columns: tuple[PageColumn, ...]) -> float | None:
    if len(columns) < 2:
        return 0.0
    ordered = sorted(columns, key=lambda column: column.x0)
    if any(candidate.bbox[0] < column.x0 < candidate.bbox[2] for column in ordered[1:]):
        return None
    center = (candidate.bbox[0] + candidate.bbox[2]) / 2
    return next((column.x0 for column in ordered if column.x0 <= center <= column.x1), None)


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


def _gold_document_for_binding(
    gold: Mapping[str, Any],
    item_key: str,
    source_pdf_sha256: str,
) -> Mapping[str, Any]:
    documents = gold.get("documents")
    if not isinstance(documents, list) or not source_pdf_sha256:
        raise ValueError("Gold documents and a source-bound binding result are required")
    matches = [
        document
        for document in documents
        if isinstance(document, Mapping)
        and str(document.get("item_key", "")) == item_key
        and str(document.get("source_pdf_sha256", "")).strip().lower() == source_pdf_sha256.lower()
    ]
    if len(matches) != 1:
        raise ValueError("Gold must contain exactly one item/source-PDF document for the binding result")
    return matches[0]


def _gold_numbered_formula_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    pages = document.get("pages")
    for page in pages if isinstance(pages, list) else []:
        if not isinstance(page, Mapping):
            continue
        page_num = _positive_int(page.get("page_num"))
        regions = page.get("regions")
        if page_num is None or not isinstance(regions, list):
            continue
        for region in regions:
            if not isinstance(region, Mapping) or str(region.get("cls", "")).strip() != "formula":
                continue
            bbox = _bbox_from_value(region.get("bbox_pt"))
            equation_number = _normalize_equation_number(str(region.get("equation_number", "")))
            if bbox is None or not equation_number:
                continue
            rows.append({"page_num": page_num, "bbox_pt": bbox, "equation_number": equation_number})
    return sorted(rows, key=lambda row: (row["page_num"], row["bbox_pt"][1], row["bbox_pt"][0]))


def _match_gold_formulas(
    gold_rows: list[dict[str, Any]],
    candidates: list[FormulaCandidate],
    threshold: float,
) -> list[tuple[int, int, float]]:
    proposals = []
    for gold_index, gold_row in enumerate(gold_rows):
        for candidate_index, candidate in enumerate(candidates):
            if candidate.page_num != gold_row["page_num"]:
                continue
            iou = _bbox_iou(gold_row["bbox_pt"], candidate.bbox)
            if iou >= threshold:
                proposals.append((iou, gold_index, candidate_index))
    used_gold: set[int] = set()
    used_candidates: set[int] = set()
    matches = []
    for iou, gold_index, candidate_index in sorted(proposals, reverse=True):
        if gold_index in used_gold or candidate_index in used_candidates:
            continue
        used_gold.add(gold_index)
        used_candidates.add(candidate_index)
        matches.append((gold_index, candidate_index, iou))
    return matches


def _gold_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "page_num": row["page_num"],
        "bbox_pt": list(row["bbox_pt"]),
        "equation_number": row["equation_number"],
    }


def _normalize_equation_number(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = re.sub(r"\s+", "", normalized).replace("（", "(").replace("）", ")")
    return normalized.upper()


def _bbox_from_value(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(part) for part in value)
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
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


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()
