import hashlib
import json

import pymupdf

from zotpilot.feature_extraction.formula_ocr import FormulaCandidate
from zotpilot.feature_extraction.vision_layout.equation_number_binding import (
    _equation_number_from_region_text,
    bind_pp_doclayout_equation_numbers,
)


def _write_numbered_pdf(path, labels):
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    for index, label in enumerate(labels):
        y = 120 + index * 80
        page.insert_text((60, y), f"x_{index + 1} = y_{index + 1}", fontsize=12)
        page.insert_text((500, y), label, fontsize=12)
    document.save(path)
    document.close()


def _write_cache(path, pdf_path, labels):
    regions = []
    for index, _label in enumerate(labels):
        y = 105 + index * 80
        regions.extend(
            [
                {"cls": "formula", "bbox_pt": [50, y, 350, y + 30], "conf": 0.95},
                {"cls": "formula_number", "bbox_pt": [490, y, 550, y + 30], "conf": 0.95},
            ]
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "pp_doclayout",
                "source_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                "pages": [
                    {
                        "page_num": 1,
                        "page_size_pt": [612, 792],
                        "coordinate_space": "pdf",
                        "regions": regions,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _candidates(labels):
    return [
        FormulaCandidate(
            page_num=1,
            bbox=(50, 105 + index * 80, 350, 135 + index * 80),
            raw_text="",
            confidence=0.95,
            source="pp_doclayout_region",
            equation_number_status="unbound",
        )
        for index, _label in enumerate(labels)
    ]


def test_visual_number_regions_bind_to_the_formula_on_the_same_horizontal_band(tmp_path):
    labels = ["(1)", "(2)"]
    pdf_path = tmp_path / "paper.pdf"
    cache_path = tmp_path / "pp_doclayout_layout.json"
    _write_numbered_pdf(pdf_path, labels)
    _write_cache(cache_path, pdf_path, labels)

    result = bind_pp_doclayout_equation_numbers(pdf_path, _candidates(labels), [cache_path])

    assert result.bound_numbers == ("(1)", "(2)")
    assert [candidate.equation_number for candidate in result.candidates] == ["(1)", "(2)"]
    assert all(candidate.equation_number_status == "detected_region" for candidate in result.candidates)
    assert result.rejected_number_region_count == 0


def test_visual_number_binding_accepts_a_close_non_overlapping_tail_number(tmp_path):
    labels = ["(1)"]
    pdf_path = tmp_path / "paper.pdf"
    cache_path = tmp_path / "pp_doclayout_layout.json"
    _write_numbered_pdf(pdf_path, labels)
    _write_cache(cache_path, pdf_path, labels)
    candidates = [
        FormulaCandidate(
            page_num=1,
            bbox=(50, 105, 488, 135),
            raw_text="",
            confidence=0.95,
            source="pp_doclayout_region",
        )
    ]

    result = bind_pp_doclayout_equation_numbers(pdf_path, candidates, [cache_path])

    assert result.bound_numbers == ("(1)",)


def test_visual_number_binding_accepts_a_short_single_column_formula_with_right_edge_number(tmp_path):
    labels = ["(1)"]
    pdf_path = tmp_path / "paper.pdf"
    cache_path = tmp_path / "pp_doclayout_layout.json"
    _write_numbered_pdf(pdf_path, labels)
    _write_cache(cache_path, pdf_path, labels)
    candidates = [
        FormulaCandidate(
            page_num=1,
            bbox=(50, 105, 180, 135),
            raw_text="",
            confidence=0.95,
            source="pp_doclayout_region",
        )
    ]

    result = bind_pp_doclayout_equation_numbers(pdf_path, candidates, [cache_path])

    assert result.bound_numbers == ("(1)",)


def test_visual_number_binding_rejects_an_equation_reference_not_a_standalone_label(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((60, 120), "x = y", fontsize=12)
    page.insert_text((490, 120), "Eq. (5)", fontsize=12)
    document.save(pdf_path)
    document.close()
    cache_path = tmp_path / "pp_doclayout_layout.json"
    _write_cache(cache_path, pdf_path, ["Eq. (5)"])

    result = bind_pp_doclayout_equation_numbers(pdf_path, _candidates(["Eq. (5)"]), [cache_path])

    assert result.bound_numbers == ()
    assert result.candidates[0].equation_number == ""
    assert result.rejected_number_region_count == 1


def test_visual_number_binding_marks_sequence_gaps_without_inventing_a_formula(tmp_path):
    labels = ["(1)", "(3)"]
    pdf_path = tmp_path / "paper.pdf"
    cache_path = tmp_path / "pp_doclayout_layout.json"
    _write_numbered_pdf(pdf_path, labels)
    _write_cache(cache_path, pdf_path, labels)

    result = bind_pp_doclayout_equation_numbers(pdf_path, _candidates(labels), [cache_path])

    assert result.sequence_gap_numbers == ("(2)",)
    assert result.candidates[1].equation_number == "(3)"
    assert "equation_number_gap_before" in result.candidates[1].quality_flags


def test_equation_number_parser_accepts_appendix_numbers_but_not_prose_references():
    assert _equation_number_from_region_text("(A2e)") == "(A2e)"
    assert _equation_number_from_region_text("(A.2e)") == "(A.2e)"
    assert _equation_number_from_region_text("Eq. (2)") == ""
