from zotpilot.feature_extraction.formula_semantic_evidence import (
    extract_equation_references,
    extract_formula_hint,
    formula_semantic_evidence_score,
    summarize_formula_semantic_evidence,
)
from zotpilot.models import StoredChunk


def test_extract_equation_references_requires_explicit_marker_or_formula_tail():
    assert extract_equation_references("Substituting Eqs. (5.38)-(5.40) into Eq. (5.37).") == [
        "(5.38)",
        "(5.40)",
        "(5.37)",
    ]
    assert extract_equation_references(r"\sigma = E\varepsilon \quad (2)") == ["(2)"]
    assert extract_equation_references("The year (2024) is not an equation reference.") == []
    assert extract_equation_references("The equation follows Rice (1976) and is given by Eq. (7).") == [
        "(7)"
    ]


def test_formula_semantic_evidence_summary_flags_unmatched_references():
    chunks = [
        StoredChunk(
            id="DOC1_chunk_0003",
            text=r"The constitutive update uses Eq. (2), where \sigma = E\varepsilon.",
            metadata={"chunk_type": "text", "page_num": 4, "chunk_index": 3, "section": "methods"},
        ),
        StoredChunk(
            id="DOC1_chunk_0004",
            text="References [17] discuss a related model.",
            metadata={"chunk_type": "text", "page_num": 10, "chunk_index": 4, "section": "references"},
        ),
    ]

    summary = summarize_formula_semantic_evidence(chunks, candidate_equation_numbers=["(1)", "(3)"])

    assert summary["source"] == "zotpilot_chroma_chunks"
    assert summary["mode"] == "read_only_review_evidence"
    assert summary["evidence_count"] == 1
    assert summary["equation_reference_numbers"] == ["(2)"]
    assert summary["unmatched_reference_numbers"] == ["(2)"]
    assert summary["unmatched_reference_count"] == 1
    assert summary["top_evidence"][0]["chunk_id"] == "DOC1_chunk_0003"
    assert "not treated as verified formulas" in summary["review_note"]


def test_formula_hint_and_score_prioritize_math_signal():
    text = "The damage equation can be written as D = 1 - exp(-a epsilon_p)."

    assert formula_semantic_evidence_score(text) >= 5
    assert "D = 1" in extract_formula_hint(text)
