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
    assert extract_equation_references("The model follows Aerens et al. (2011c), not an equation.") == []
    assert extract_equation_references("The equation follows Rice (1976) and is given by Eq. (7).") == [
        "(7)"
    ]
    assert extract_equation_references("The term is formalized in Eq. 2 (Ref 35).") == ["(2)"]


def test_extract_equation_references_ignores_non_equation_numbers_in_marker_window():
    assert extract_equation_references("The projectile diameter is (7.62m) and velocity is (0.5) km/s.") == []
    assert extract_equation_references("The projectile is defined by formula (3.2), diameter (7.62m).") == [
        "(3.2)"
    ]
    assert extract_equation_references("The fitted range is (0.4-0.6) for this condition.") == []
    assert extract_equation_references("代入式（4-8）即可得到 m 值，试样尺寸为 (0.5) mm。") == ["(4-8)"]
    assert extract_equation_references("将拟合得到的系数代入公式（4-4），材料编号 (316) 不应视为公式。") == [
        "(4-4)"
    ]
    assert extract_equation_references("铺层方式0、方式1、方式2、方式3和方式4均不是公式编号。") == []
    assert extract_equation_references("这种形式(1)和计算模式(2)不应视为公式。") == []
    assert extract_equation_references("The generalized equation 131 is discussed in the next line.") == []
    assert extract_equation_references("The resultant formula 266 is a line-number artifact.") == []
    assert extract_equation_references("Equation (10) indicates the equivalent stress for 261 components.") == [
        "(10)"
    ]


def test_extract_equation_references_keeps_connected_equation_lists():
    assert extract_equation_references("式（4-9）、（4-10）、（4-11）给出更新过程。") == [
        "(4-9)",
        "(4-10)",
        "(4-11)",
    ]
    assert extract_equation_references("The residual follows Eqs. (6) and (7).") == ["(6)", "(7)"]


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
    assert summary["reference_match_status"] == "no_match"
    assert summary["reference_coverage_ratio"] == 0.0
    assert summary["review_flags"] == [
        "semantic_unmatched_references",
        "semantic_no_candidate_reference_overlap",
    ]
    assert summary["top_evidence"][0]["chunk_id"] == "DOC1_chunk_0003"
    assert "not treated as verified formulas" in summary["review_note"]


def test_formula_semantic_evidence_matches_collapsed_chapter_number_aliases():
    chunks = [
        StoredChunk(
            id="DOC1_chunk_0005",
            text="将拟合得到的系数代入式（49），可获得流变失稳判据。",
            metadata={"chunk_type": "text", "page_num": 5, "chunk_index": 5, "section": "results"},
        ),
        StoredChunk(
            id="DOC1_chunk_0006",
            text="Luo 等建立了相变关系，如式（316）所示： beta = f(T)。",
            metadata={"chunk_type": "text", "page_num": 6, "chunk_index": 6, "section": "results"},
        ),
    ]

    summary = summarize_formula_semantic_evidence(chunks, candidate_equation_numbers=["(4-9)", "(3-16)"])

    assert set(summary["equation_reference_numbers"]) == {"(49)", "(316)"}
    assert set(summary["matched_reference_numbers"]) == {"(49)", "(316)"}
    assert summary["unmatched_reference_numbers"] == []
    assert summary["reference_match_status"] == "all_matched"
    assert summary["reference_coverage_ratio"] == 1.0


def test_formula_semantic_evidence_expands_same_prefix_equation_ranges():
    chunks = [
        StoredChunk(
            id="DOC1_chunk_0007",
            text="The update follows Eqs. (4.3.47-4.3.51) and Eq. (4-9).",
            metadata={"chunk_type": "text", "page_num": 7, "chunk_index": 7, "section": "methods"},
        ),
    ]

    summary = summarize_formula_semantic_evidence(
        chunks,
        candidate_equation_numbers=["(4.3.47)", "(4.3.48)", "(4.3.49)", "(4-9)"],
    )

    assert summary["equation_reference_numbers"] == [
        "(4.3.47)",
        "(4.3.48)",
        "(4.3.49)",
        "(4.3.50)",
        "(4.3.51)",
        "(4-9)",
    ]
    assert summary["matched_reference_numbers"] == ["(4.3.47)", "(4.3.48)", "(4.3.49)", "(4-9)"]
    assert summary["unmatched_reference_numbers"] == ["(4.3.50)", "(4.3.51)"]


def test_formula_semantic_evidence_tracks_formula_like_chunks_without_numbers():
    chunks = [
        StoredChunk(
            id="DOC1_chunk_0007",
            text=r"The calibrated relation is \sigma = A + B\varepsilon^n.",
            metadata={"chunk_type": "text", "page_num": 7, "chunk_index": 7, "section": "results"},
        ),
    ]

    summary = summarize_formula_semantic_evidence(chunks, candidate_equation_numbers=[])

    assert summary["evidence_count"] == 1
    assert summary["equation_reference_numbers"] == []
    assert summary["unmatched_reference_count"] == 0
    assert summary["reference_match_status"] == "no_references"
    assert summary["formula_evidence_without_reference_count"] == 1
    assert summary["review_flags"] == ["formula_like_evidence_without_equation_numbers"]


def test_formula_hint_and_score_prioritize_math_signal():
    text = "The damage equation can be written as D = 1 - exp(-a epsilon_p)."

    assert formula_semantic_evidence_score(text) >= 5
    assert "D = 1" in extract_formula_hint(text)
