from zotpilot.feature_extraction.formula_external_parser_comparison import (
    build_formula_external_parser_comparison,
)


def test_external_parser_comparison_accepts_cross_parser_consensus():
    mineru_estimate = {
        "candidate_count": 1,
        "results": [
            {
                "item_key": "DOC1",
                "title": "Paper",
                "candidate_count": 1,
                "candidate_audit": {"source_counts": {"mineru_content_list": 1}},
                "candidate_preview": [
                    {
                        "candidate_index": 0,
                        "page_num": 3,
                        "source": "mineru_content_list",
                        "equation_number": "(1)",
                        "bbox": [10, 20, 300, 48],
                        "latex_preview": r"\sigma = E\varepsilon",
                        "has_latex": True,
                        "needs_ocr": False,
                    }
                ],
            }
        ],
    }
    pdf_extract_kit_estimate = {
        "candidate_count": 1,
        "results": [
            {
                "item_key": "DOC1",
                "title": "Paper",
                "candidate_count": 1,
                "candidate_audit": {"source_counts": {"pdf_extract_kit_formula_recognition": 1}},
                "candidate_preview": [
                    {
                        "candidate_index": 0,
                        "page_num": 3,
                        "source": "pdf_extract_kit_formula_recognition",
                        "equation_number": "(1)",
                        "bbox": [11, 21, 299, 49],
                        "latex_preview": r"\sigma = E\varepsilon",
                        "has_latex": True,
                        "needs_ocr": False,
                    }
                ],
            }
        ],
    }

    comparison = build_formula_external_parser_comparison(
        {
            "mineru": mineru_estimate,
            "pdf_extract_kit": pdf_extract_kit_estimate,
        }
    )

    assert comparison["mode"] == "read_only_external_parser_comparison"
    assert comparison["parser_count"] == 2
    assert comparison["paper_count"] == 1
    assert comparison["multi_provider_cluster_count"] == 1
    assert comparison["conflict_cluster_count"] == 0
    assert comparison["comparison_flag_counts"] == {}
    assert comparison["write_recommendation_counts"] == {
        "candidate_supported_by_cross_parser_review": 1
    }
    assert comparison["parser_candidate_summary"] == {
        "mineru": {
            "paper_count": 1,
            "candidate_count": 1,
            "preview_candidate_count": 1,
            "quality_route_counts": {"unknown": 1},
            "source_group_counts": {"mineru_cache": 1},
        },
        "pdf_extract_kit": {
            "paper_count": 1,
            "candidate_count": 1,
            "preview_candidate_count": 1,
            "quality_route_counts": {"unknown": 1},
            "source_group_counts": {"pdf_extract_kit": 1},
        },
    }
    row = comparison["rows"][0]
    assert row["comparison_flags"] == []
    assert row["write_recommendation"] == "candidate_supported_by_cross_parser_review"
    cluster = row["candidate_consensus"]["clusters"][0]
    assert cluster["parser_label_counts"] == {"mineru": 1, "pdf_extract_kit": 1}
    assert cluster["source_group_counts"] == {"mineru_cache": 1, "pdf_extract_kit": 1}
    assert cluster["candidate_details"][0]["parser_label"] == "mineru"
    assert cluster["candidate_details"][1]["parser_label"] == "pdf_extract_kit"


def test_external_parser_comparison_routes_conflicts_to_manual_review():
    first_estimate = {
        "candidate_count": 1,
        "results": [
            {
                "item_key": "DOC2",
                "title": "Conflicting Paper",
                "candidate_count": 1,
                "candidate_audit": {"source_counts": {"mineru_content_list": 1}},
                "candidate_preview": [
                    {
                        "candidate_index": 0,
                        "page_num": 7,
                        "source": "mineru_content_list",
                        "equation_number": "(3)",
                        "bbox": [10, 20, 300, 48],
                        "latex_preview": r"D = 1 - \exp(-a\varepsilon_p)",
                        "has_latex": True,
                        "needs_ocr": False,
                    }
                ],
            }
        ],
    }
    second_estimate = {
        "candidate_count": 2,
        "results": [
            {
                "item_key": "DOC2",
                "title": "Conflicting Paper",
                "candidate_count": 2,
                "candidate_audit": {"source_counts": {"pdf_extract_kit_formula_detection": 2}},
                "candidate_preview": [
                    {
                        "candidate_index": 0,
                        "page_num": 7,
                        "source": "pdf_extract_kit_formula_detection",
                        "equation_number": "(4)",
                        "bbox": [11, 20, 299, 49],
                        "raw_text_preview": r"D = 1 - exp(-a epsilon_p)",
                        "has_latex": False,
                        "needs_ocr": True,
                    },
                    {
                        "candidate_index": 1,
                        "page_num": 9,
                        "source": "pdf_extract_kit_formula_detection",
                        "equation_number": "(5)",
                        "bbox": [11, 70, 299, 99],
                        "raw_text_preview": r"\eta = \sigma_m / \sigma_eq",
                        "has_latex": False,
                        "needs_ocr": True,
                    },
                ],
            }
        ],
    }

    comparison = build_formula_external_parser_comparison(
        {"mineru": first_estimate, "pdf_extract_kit": second_estimate}
    )

    assert comparison["conflict_cluster_count"] == 1
    assert comparison["manual_review_paper_count"] == 1
    row = comparison["rows"][0]
    assert set(row["comparison_flags"]) >= {
        "candidate_count_mismatch",
        "candidate_consensus_conflicts",
    }
    assert row["write_recommendation"] == "manual_review_queue"
    assert comparison["comparison_flag_counts"] == {
        "candidate_consensus_conflicts": 1,
        "candidate_count_mismatch": 1,
    }
    assert comparison["write_recommendation_counts"] == {"manual_review_queue": 1}


def test_external_parser_comparison_tracks_readonly_index_change():
    comparison = build_formula_external_parser_comparison(
        {
            "stable": {"candidate_count": 0, "readonly_index_changed": False, "results": []},
            "changed": {"candidate_count": 0, "readonly_index_changed": True, "results": []},
        }
    )

    assert comparison["readonly_index_changed"] is True
    assert comparison["readonly_index_changed_by_parser"] == {
        "changed": True,
        "stable": False,
    }
