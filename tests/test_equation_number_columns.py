import hashlib
import json

import pymupdf

from zotpilot.feature_extraction.formula_ocr import FormulaCandidate
from zotpilot.feature_extraction.vision_layout.equation_number_binding import bind_pp_doclayout_equation_numbers
from zotpilot.feature_extraction.vision_layout.formula_layout_filter import PageColumn


def _bind(tmp_path, specifications):
    pdf = tmp_path / "paper.pdf"
    cache = tmp_path / "pp_doclayout_layout.json"
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    candidates, regions = [], []
    for bbox, number_x, number in specifications:
        page.insert_text((number_x, bbox[1] + 20), number, fontsize=12)
        candidates.append(FormulaCandidate(1, bbox, "", 0.95, source="pp_doclayout_region"))
        regions.append({"cls": "formula_number", "conf": 0.95,
                        "bbox_pt": [number_x - 2, bbox[1], number_x + 30, bbox[3]]})
    document.save(pdf)
    document.close()
    cache.write_text(json.dumps({
        "generator": "pp_doclayout", "source_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "pages": [{"page_num": 1, "page_size_pt": [612, 792], "coordinate_space": "pdf", "regions": regions}],
    }), encoding="utf-8")
    return bind_pp_doclayout_equation_numbers(
        pdf, candidates, [cache],
        columns_by_page={1: (PageColumn(1, 0, 0, 306), PageColumn(1, 1, 306, 612))},
    )


def test_sequence_checks_follow_columns_instead_of_interleaving_rows(tmp_path):
    result = _bind(tmp_path, [
        ((60, 100, 240, 130), 272, "(1)"),
        ((350, 100, 530, 130), 562, "(3)"),
        ((60, 200, 240, 230), 272, "(2)"),
        ((350, 200, 530, 230), 562, "(4)"),
    ])
    assert [c.equation_number for c in result.candidates] == ["(1)", "(3)", "(2)", "(4)"]
    assert result.sequence_gap_numbers == ()
    assert result.non_monotonic_pairs == ()
    assert all(not c.quality_flags for c in result.candidates)


def test_proven_gutter_spanning_formula_can_bind_right_margin(tmp_path):
    result = _bind(tmp_path, [((60, 100, 530, 130), 562, "(1)")])
    assert result.bound_numbers == ("(1)",)
    assert "equation_number_reading_order_ambiguous" in result.candidates[0].quality_flags
    assert result.sequence_gap_numbers == ()


def test_real_within_column_gap_is_still_reported(tmp_path):
    result = _bind(tmp_path, [
        ((60, 100, 240, 130), 272, "(1)"),
        ((60, 200, 240, 230), 272, "(3)"),
    ])
    assert result.sequence_gap_numbers == ("(2)",)
