from zotpilot.feature_extraction.formula_candidate_consensus import (
    build_formula_candidate_consensus,
    evidence_modality,
    source_group,
)


def test_same_text_layer_under_auto_and_text_labels_is_not_independent_support():
    consensus = build_formula_candidate_consensus(
        [
            {
                "parser_label": "auto",
                "source": "text_layer",
                "source_artifact_hash": "pdf-hash",
                "page_num": 4,
                "bbox": [50, 100, 300, 140],
                "equation_number": "(1)",
            },
            {
                "parser_label": "text",
                "source": "text_layer",
                "source_artifact_hash": "pdf-hash",
                "page_num": 4,
                "bbox": [50, 100, 300, 140],
                "equation_number": "(1)",
            },
        ]
    )

    cluster = consensus["clusters"][0]
    assert cluster["parser_label_count"] == 2
    assert cluster["provider_group_count"] == 1
    assert cluster["independent_source_count"] == 1
    assert "no_structured_parser_evidence" in cluster["review_flags"]
    assert cluster["cluster_route"] == "single_provider_review"
    assert consensus["supported_cluster_count"] == 0


def test_text_layer_and_pp_doclayout_are_independent_candidate_evidence():
    consensus = build_formula_candidate_consensus(
        [
            {
                "parser_label": "text",
                "source": "text_layer",
                "source_artifact_hash": "pdf-hash",
                "page_num": 4,
                "bbox": [50, 100, 300, 140],
                "equation_number": "(1)",
            },
            {
                "parser_label": "pp_doclayout",
                "source": "pp_doclayout_region",
                "source_artifact_hash": "layout-cache-hash",
                "page_num": 4,
                "bbox": [55, 98, 305, 142],
                "equation_number": "",
            },
        ]
    )

    cluster = consensus["clusters"][0]
    assert cluster["source_group_counts"] == {"pdf_text_layer": 1, "pp_doclayout": 1}
    assert cluster["evidence_modality_counts"] == {"text_layer": 1, "vision_layout": 1}
    assert cluster["provider_group_count"] == 2
    assert cluster["independent_source_count"] == 2
    assert cluster["cluster_route"] == "supported_candidate"


def test_same_artifact_cannot_count_as_two_independent_provider_votes():
    consensus = build_formula_candidate_consensus(
        [
            {
                "parser_label": "first",
                "source": "mineru_content_list",
                "source_artifact_hash": "shared-cache",
                "page_num": 2,
                "bbox": [50, 100, 300, 140],
            },
            {
                "parser_label": "second",
                "source": "pp_doclayout_region",
                "source_artifact_hash": "shared-cache",
                "page_num": 2,
                "bbox": [55, 98, 305, 142],
            },
        ]
    )

    cluster = consensus["clusters"][0]
    assert cluster["provider_group_count"] == 2
    assert cluster["independent_source_count"] == 1
    assert "shared_source_artifact" in cluster["review_flags"]
    assert cluster["cluster_route"] == "single_provider_review"


def test_pp_doclayout_source_has_vision_layout_modality():
    assert source_group("pp_doclayout_region") == "pp_doclayout"
    assert evidence_modality("pp_doclayout") == "vision_layout"
