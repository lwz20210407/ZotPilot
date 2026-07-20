import hashlib
import json

import pymupdf

from zotpilot.feature_extraction.formula_ocr import FormulaCandidate
from zotpilot.feature_extraction.vision_layout.equation_number_binding import (
    bind_pp_doclayout_equation_numbers,
)
from zotpilot.feature_extraction.vision_layout.formula_layout_filter import (
    PageColumn,
    classify_visual_formula_candidates,
)
from zotpilot.feature_extraction.vision_layout.pp_doclayout_candidate_cache import (
    load_pp_doclayout_formula_number_regions,
)


def _write_two_column_pdf(path):
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    for index in range(16):
        y = 70 + index * 24
        page.insert_text((50, y), f"left prose row {index} with ordinary words", fontsize=9)
        page.insert_text((340, y), f"right prose row {index} with ordinary words", fontsize=9)
    page.insert_text((72, 520), "a = b", fontsize=12)
    page.insert_text((272, 520), "(1)", fontsize=12)
    page.insert_text((360, 520), "c = d", fontsize=12)
    page.insert_text((562, 520), "(2)", fontsize=12)
    page.insert_text((145, 600), "prose", fontsize=10)
    page.insert_text((205, 600), "x+y", fontsize=10)
    page.insert_text((250, 600), "continues", fontsize=10)
    document.save(path)
    document.close()


def _write_cache(path, pdf_path):
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
                        "regions": [
                            {"cls": "formula", "bbox_pt": [60, 500, 240, 530], "conf": 0.95},
                            {"cls": "formula", "bbox_pt": [350, 500, 530, 530], "conf": 0.95},
                            {"cls": "formula", "bbox_pt": [200, 588, 245, 610], "conf": 0.95},
                            {"cls": "formula_number", "bbox_pt": [262, 500, 300, 530], "conf": 0.95},
                            {"cls": "formula_number", "bbox_pt": [552, 500, 600, 530], "conf": 0.95},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _visual_candidates():
    return [
        FormulaCandidate(1, (60, 500, 240, 530), "", 0.95, source="pp_doclayout_region"),
        FormulaCandidate(1, (350, 500, 530, 530), "", 0.95, source="pp_doclayout_region"),
        FormulaCandidate(1, (200, 588, 245, 610), "", 0.95, source="pp_doclayout_region"),
    ]


def test_layout_filter_keeps_numbered_display_and_excludes_inline_math(tmp_path):
    pdf_path = tmp_path / "two-column.pdf"
    cache_path = tmp_path / "layout.json"
    _write_two_column_pdf(pdf_path)
    _write_cache(cache_path, pdf_path)
    numbers = load_pp_doclayout_formula_number_regions(pdf_path, [cache_path])

    analysis = classify_visual_formula_candidates(pdf_path, _visual_candidates(), numbers)

    assert [candidate.layout_kind for candidate in analysis.candidates] == ["display", "display"]
    assert len(analysis.columns_by_page[1]) == 2


def test_number_binding_stays_in_the_same_detected_column(tmp_path):
    pdf_path = tmp_path / "two-column.pdf"
    cache_path = tmp_path / "layout.json"
    _write_two_column_pdf(pdf_path)
    _write_cache(cache_path, pdf_path)
    numbers = load_pp_doclayout_formula_number_regions(pdf_path, [cache_path])
    analysis = classify_visual_formula_candidates(pdf_path, _visual_candidates(), numbers)

    result = bind_pp_doclayout_equation_numbers(
        pdf_path,
        analysis.candidates,
        [cache_path],
        columns_by_page=analysis.columns_by_page,
    )

    assert [candidate.equation_number for candidate in result.candidates] == ["(1)", "(2)"]


def test_two_column_analysis_allows_outer_margin_fallback_only_without_same_column_label(tmp_path):
    pdf_path = tmp_path / "two-column.pdf"
    cache_path = tmp_path / "layout.json"
    _write_two_column_pdf(pdf_path)
    _write_cache(cache_path, pdf_path)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["pages"][0]["regions"] = [
        region for region in payload["pages"][0]["regions"]
        if not (region["cls"] == "formula_number" and region["bbox_pt"][0] < 400)
    ]
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    # A short page-wide display formula can begin in the left margin while its
    # genuine number is flush with the outer page edge.
    result = bind_pp_doclayout_equation_numbers(
        pdf_path,
        [FormulaCandidate(1, (60, 500, 240, 530), "", 0.95, source="pp_doclayout_region")],
        [cache_path],
        columns_by_page={
            1: (PageColumn(1, 0, 0, 306), PageColumn(1, 1, 306, 612)),
        },
    )
    assert result.bound_numbers == ("(2)",)


def test_layout_filter_prefers_source_bound_cached_columns_when_available(tmp_path):
    pdf_path = tmp_path / "two-column.pdf"
    cache_path = tmp_path / "layout.json"
    _write_two_column_pdf(pdf_path)
    _write_cache(cache_path, pdf_path)
    numbers = load_pp_doclayout_formula_number_regions(pdf_path, [cache_path])
    cached_columns = {
        1: (
            PageColumn(1, 0, 0, 200),
            PageColumn(1, 1, 200, 400),
            PageColumn(1, 2, 400, 612),
        )
    }

    analysis = classify_visual_formula_candidates(
        pdf_path,
        _visual_candidates(),
        numbers,
        cached_columns_by_page=cached_columns,
    )

    assert analysis.columns_by_page[1] == cached_columns[1]
