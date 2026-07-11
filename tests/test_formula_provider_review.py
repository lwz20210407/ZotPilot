from zotpilot.feature_extraction.formula_provider_review import build_formula_provider_cross_review


def test_provider_cross_review_summarizes_structured_sources_and_simpletex_fallback():
    estimate = {
        "provider": "simpletex",
        "candidate_provider": "auto",
        "candidate_count": 4,
        "write_blocked": True,
        "results": [
            {
                "item_key": "DOC1",
                "title": "Paper",
                "candidate_count": 4,
                "estimated_external_calls": 1,
                "candidate_audit": {
                    "candidate_count": 4,
                    "source_counts": {
                        "mineru_content_list": 2,
                        "pdf_extract_kit_formula_detection": 1,
                        "text_layer": 1,
                    },
                    "ocr_needed_count": 1,
                    "cached_latex_count": 3,
                    "page_min": 2,
                    "page_max": 4,
                    "page_count_with_candidates": 3,
                    "page_tagged_count": 4,
                    "page_missing_count": 0,
                    "bbox_present_count": 3,
                    "bbox_missing_count": 1,
                    "numbered_count": 3,
                    "unnumbered_count": 1,
                    "first_equation_number": "(1)",
                    "last_equation_number": "(3)",
                    "equation_number_warnings": ["missing_equation_number_gap"],
                },
                "semantic_formula_evidence_count": 2,
                "semantic_formula_unmatched_reference_count": 1,
                "semantic_formula_unmatched_reference_numbers": ["(2)"],
                "semantic_formula_reference_match_status": "partial_match",
                "semantic_formula_reference_coverage_ratio": 0.5,
                "semantic_formula_review_flags": ["semantic_unmatched_references"],
            }
        ],
        "formula_quality_route_summary": [
            {
                "item_key": "DOC1",
                "quality_route": "review_queue",
                "route_reason": "missing_equation_number_gap",
            }
        ],
    }

    review = build_formula_provider_cross_review(estimate)

    assert review["simpletex_role"] == "fallback_recognizer_only"
    assert review["external_call_count"] == 1
    assert review["provider_group_totals"] == {
        "mineru_cache": 2,
        "pdf_extract_kit": 1,
        "pdf_text_layer": 1,
    }
    row = review["rows"][0]
    assert row["provider_evidence"]["structured_parser_candidate_count"] == 3
    assert row["traceability"]["bbox_missing_count"] == 1
    assert row["semantic_evidence"]["unmatched_reference_numbers"] == ["(2)"]
    assert set(row["review_flags"]) >= {
        "missing_equation_number_gap",
        "missing_bbox",
        "ocr_fallback_required",
        "semantic_unmatched_references",
        "simpletex_external_fallback",
    }
    assert row["write_recommendation"] == "manual_review_queue"


def test_provider_cross_review_keeps_clear_structured_candidate_auto_writable():
    estimate = {
        "provider": "local",
        "candidate_provider": "auto",
        "candidate_count": 1,
        "results": [
            {
                "item_key": "DOC2",
                "title": "Clean Paper",
                "candidate_count": 1,
                "candidate_audit": {
                    "candidate_count": 1,
                    "source_counts": {"mineru_content_list": 1},
                    "ocr_needed_count": 0,
                    "cached_latex_count": 1,
                    "page_min": 1,
                    "page_max": 1,
                    "page_count_with_candidates": 1,
                    "page_tagged_count": 1,
                    "page_missing_count": 0,
                    "bbox_present_count": 1,
                    "bbox_missing_count": 0,
                    "numbered_count": 1,
                    "unnumbered_count": 0,
                    "first_equation_number": "(1)",
                    "last_equation_number": "(1)",
                    "equation_number_warnings": [],
                },
            }
        ],
        "formula_quality_route_summary": [
            {
                "item_key": "DOC2",
                "quality_route": "auto_candidate",
                "route_reason": "candidate_quality_clear",
            }
        ],
    }

    review = build_formula_provider_cross_review(estimate)

    row = review["rows"][0]
    assert row["quality_route"] == "auto_candidate"
    assert row["provider_evidence"]["source_group_counts"] == {"mineru_cache": 1}
    assert row["write_recommendation"] == "auto_candidate_after_reviewed_batch"


def test_provider_cross_review_blocks_pdf_text_layer_only_auto_write():
    estimate = {
        "provider": "local",
        "candidate_provider": "auto",
        "candidate_count": 1,
        "results": [
            {
                "item_key": "DOC3",
                "title": "Text Layer Only Paper",
                "candidate_count": 1,
                "candidate_audit": {
                    "candidate_count": 1,
                    "source_counts": {"text_layer": 1},
                    "ocr_needed_count": 0,
                    "cached_latex_count": 1,
                    "page_min": 1,
                    "page_max": 1,
                    "page_count_with_candidates": 1,
                    "page_tagged_count": 1,
                    "page_missing_count": 0,
                    "bbox_present_count": 1,
                    "bbox_missing_count": 0,
                    "numbered_count": 1,
                    "unnumbered_count": 0,
                    "first_equation_number": "(1)",
                    "last_equation_number": "(1)",
                    "equation_number_warnings": [],
                },
            }
        ],
        "formula_quality_route_summary": [
            {
                "item_key": "DOC3",
                "quality_route": "auto_candidate",
                "route_reason": "candidate_quality_clear",
            }
        ],
    }

    review = build_formula_provider_cross_review(estimate)

    row = review["rows"][0]
    assert set(row["review_flags"]) >= {
        "no_structured_parser_evidence",
        "pdf_text_layer_only",
    }
    assert row["write_recommendation"] == "manual_review_queue"
