"""Read-only formula evidence mined from ZotPilot's own semantic chunks.

This module deliberately treats semantic chunks as review evidence, not as
verified formulas. A text chunk that mentions ``Eq. (7)`` can help reviewers
find missed candidates, but it is not enough to create a formula index row.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_FULLWIDTH_TRANS = str.maketrans(
    {
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "（": "(",
        "）": ")",
        "－": "-",
        "—": "-",
        "＿": "_",
        "．": ".",
    }
)

_FORMULA_SIGNAL_RE = re.compile(
    r"(?:"
    r"[=<>≤≥≈∝±×÷∑∫√]|"
    r"\\(?:frac|sum|int|sqrt|partial|dot|bar|tilde|hat|sigma|epsilon|varepsilon|theta|eta|mu)|"
    r"[α-ωΑ-ΩεσμηθκλφψΩ]|"
    r"\b(?:J|I|K|G|E|D|Y|f|g|h)_[0-9a-zA-Z]+\b"
    r")",
    re.IGNORECASE,
)

_FORMULA_RELATION_RE = re.compile(r"(?:=|≤|≥|≈|∝|<|>|\\leq|\\geq|\\approx|\\sim)", re.IGNORECASE)
_TEX_COMMAND_RE = re.compile(r"\\(?:frac|sum|int|sqrt|partial|dot|bar|tilde|hat|begin|end)\b")
_EXPLICIT_REF_WINDOW_RE = re.compile(
    r"(?:\b(?:eqs?|equations?|formulae?|formulas?)\.?|公式|方程|式)\s*[:：]?\s*.{0,140}",
    re.IGNORECASE,
)
_PAREN_NUMBER_RE = re.compile(r"\(\s*([0-9]+(?:[.\-_][0-9]+)*(?:[a-z])?)\s*\)", re.IGNORECASE)
_EXPLICIT_MARKER_NUMBER_RE = re.compile(
    r"(?:\b(?:eqs?|equations?|formulae?|formulas?)\.?|公式|方程|式)\s*"
    r"(?:[\(:：]\s*)?"
    r"([0-9]+(?:[.\-_][0-9]+)*(?:[a-z])?)",
    re.IGNORECASE,
)
_TRAILING_NUMBER_RE = re.compile(r"\(\s*([0-9]+(?:[.\-_][0-9]+)*(?:[a-z])?)\s*\)\s*$", re.IGNORECASE)
_DOI_OR_REFERENCE_RE = re.compile(
    r"(?:\bdoi\b|https?://|references\b|bibliography\b|^\s*\[[0-9]+\]|参考文献)",
    re.IGNORECASE,
)


def _looks_like_citation_year_number(value: str) -> bool:
    """Return True for plain parenthetical years that should not be equation IDs."""
    normalized = normalize_equation_number(value)
    if not re.fullmatch(r"\d{4}", normalized):
        return False
    year = int(normalized)
    return 1800 <= year <= 2099


@dataclass(frozen=True)
class FormulaSemanticEvidence:
    """A formula-like clue from an existing ZotPilot semantic chunk."""

    chunk_id: str
    chunk_type: str
    page_num: int
    chunk_index: int
    section: str
    score: int
    equation_numbers: tuple[str, ...]
    formula_hint: str
    text_preview: str


def normalize_equation_number(value: str) -> str:
    """Normalize an equation number for matching, without claiming semantic equivalence."""
    text = (value or "").translate(_FULLWIDTH_TRANS).strip().lower()
    text = text.strip("()[]{}")
    text = re.sub(r"\s+", "", text)
    text = text.replace("_", "-")
    return text


def format_equation_number(value: str) -> str:
    """Return a display form such as ``(4.12)`` from a normalized number."""
    normalized = normalize_equation_number(value)
    return f"({normalized})" if normalized else ""


def extract_equation_references(text: str) -> list[str]:
    """Extract explicit equation references from text chunks.

    We intentionally require either an Eq./formula marker or a formula-like
    trailing number. Plain parenthetical prose such as ``(2024)`` is ignored.
    """
    normalized_text = (text or "").translate(_FULLWIDTH_TRANS)
    seen: set[str] = set()
    numbers: list[str] = []

    def add(raw: str) -> None:
        number = normalize_equation_number(raw)
        if _looks_like_citation_year_number(number):
            return
        if number and number not in seen:
            seen.add(number)
            numbers.append(format_equation_number(number))

    for window_match in _EXPLICIT_REF_WINDOW_RE.finditer(normalized_text):
        window = window_match.group(0)
        window_numbers: list[tuple[int, str]] = []
        for match in _EXPLICIT_MARKER_NUMBER_RE.finditer(window):
            window_numbers.append((match.start(1), match.group(1)))
        for match in _PAREN_NUMBER_RE.finditer(window):
            window_numbers.append((match.start(1), match.group(1)))
        for _position, raw_number in sorted(window_numbers):
            add(raw_number)

    trailing = _TRAILING_NUMBER_RE.search(normalized_text)
    if trailing and _FORMULA_SIGNAL_RE.search(normalized_text):
        add(trailing.group(1))

    return numbers


def looks_formulaish_semantic_text(text: str) -> bool:
    """Return True when a semantic chunk contains formula-like signals."""
    if not (text or "").strip():
        return False
    normalized_text = text.translate(_FULLWIDTH_TRANS)
    return bool(_FORMULA_SIGNAL_RE.search(normalized_text) or _EXPLICIT_REF_WINDOW_RE.search(normalized_text))


def formula_semantic_evidence_score(text: str) -> int:
    """Score a semantic chunk as formula evidence for review ranking."""
    normalized_text = (text or "").translate(_FULLWIDTH_TRANS)
    if not normalized_text.strip():
        return 0
    score = 0
    if _FORMULA_RELATION_RE.search(normalized_text):
        score += 4
    if _TEX_COMMAND_RE.search(normalized_text):
        score += 3
    if re.search(r"[α-ωΑ-ΩεσμηθκλφψΩ]", normalized_text):
        score += 2
    references = extract_equation_references(normalized_text)
    if references:
        score += min(4, 1 + len(references))
    if re.search(r"\b(?:stress|strain|damage|fracture|yield|constitutive|equation|model)\b", normalized_text, re.I):
        score += 1
    if re.search(r"(?:应力|应变|损伤|断裂|屈服|本构|方程|公式|模型)", normalized_text):
        score += 1
    if _DOI_OR_REFERENCE_RE.search(normalized_text) and not _FORMULA_RELATION_RE.search(normalized_text):
        score -= 2
    return max(score, 0)


def extract_formula_hint(text: str, *, limit: int = 260) -> str:
    """Extract a compact formula-like window for review output."""
    normalized_text = re.sub(r"\s+", " ", (text or "").translate(_FULLWIDTH_TRANS)).strip()
    if not normalized_text:
        return ""
    match = _FORMULA_SIGNAL_RE.search(normalized_text)
    if match:
        start = max(match.start() - 90, 0)
        end = min(match.end() + 170, len(normalized_text))
        hint = normalized_text[start:end].strip()
    else:
        hint = normalized_text[:limit]
    return hint[:limit] + ("..." if len(hint) > limit else "")


def build_formula_semantic_evidence(
    chunks: Iterable[object],
    *,
    max_evidence: int = 8,
    text_preview_chars: int = 220,
) -> list[FormulaSemanticEvidence]:
    """Build ranked formula evidence rows from existing ZotPilot chunks."""
    evidence_rows: list[FormulaSemanticEvidence] = []
    for chunk in chunks:
        text = str(getattr(chunk, "text", "") or "")
        score = formula_semantic_evidence_score(text)
        if score <= 0:
            continue
        metadata = getattr(chunk, "metadata", {}) or {}
        chunk_type = str(metadata.get("chunk_type", "") or "")
        if chunk_type == "formula":
            continue
        equation_numbers = tuple(extract_equation_references(text))
        evidence_rows.append(
            FormulaSemanticEvidence(
                chunk_id=str(getattr(chunk, "id", "") or ""),
                chunk_type=chunk_type,
                page_num=int(metadata.get("page_num", 0) or 0),
                chunk_index=int(metadata.get("chunk_index", -1) or -1),
                section=str(metadata.get("section", "") or ""),
                score=score,
                equation_numbers=equation_numbers,
                formula_hint=extract_formula_hint(text),
                text_preview=(re.sub(r"\s+", " ", text).strip()[:text_preview_chars]),
            )
        )
    return sorted(
        evidence_rows,
        key=lambda row: (row.score, row.page_num, -row.chunk_index),
        reverse=True,
    )[:max(max_evidence, 0)]


def summarize_formula_semantic_evidence(
    chunks: Iterable[object],
    *,
    candidate_equation_numbers: Iterable[str] = (),
    max_evidence: int = 8,
) -> dict[str, object]:
    """Return a JSON-safe review summary for formula semantic evidence."""
    evidence = build_formula_semantic_evidence(chunks, max_evidence=max_evidence)
    candidate_numbers = {
        number
        for raw in candidate_equation_numbers
        if (number := normalize_equation_number(raw))
    }
    reference_numbers: list[str] = []
    seen_references: set[str] = set()
    for row in evidence:
        for display_number in row.equation_numbers:
            normalized_number = normalize_equation_number(display_number)
            if normalized_number and normalized_number not in seen_references:
                seen_references.add(normalized_number)
                reference_numbers.append(format_equation_number(normalized_number))

    unmatched = [
        format_equation_number(number)
        for number in seen_references
        if number not in candidate_numbers
    ]
    matched = [
        format_equation_number(number)
        for number in seen_references
        if number in candidate_numbers
    ]
    return {
        "source": "zotpilot_chroma_chunks",
        "mode": "read_only_review_evidence",
        "evidence_count": len(evidence),
        "equation_reference_numbers": reference_numbers,
        "candidate_equation_numbers": [
            format_equation_number(number)
            for number in sorted(candidate_numbers)
        ],
        "matched_reference_numbers": matched,
        "unmatched_reference_numbers": unmatched,
        "unmatched_reference_count": len(unmatched),
        "top_evidence": [
            {
                "chunk_id": row.chunk_id,
                "chunk_type": row.chunk_type,
                "page_num": row.page_num,
                "chunk_index": row.chunk_index,
                "section": row.section,
                "score": row.score,
                "equation_numbers": list(row.equation_numbers),
                "formula_hint": row.formula_hint,
                "text_preview": row.text_preview,
            }
            for row in evidence
        ],
        "review_note": (
            "Semantic chunks are only review evidence. They can indicate missed or suspicious "
            "formula candidates, but they are not treated as verified formulas."
        ),
    }
