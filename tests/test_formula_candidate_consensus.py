from itertools import permutations

import pytest

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


def _numbered_evidence(source, number, *, page=1, bbox=(50, 100, 300, 140)):
    return {"source": source, "source_artifact_hash": source,
            "equation_number": number, "page_num": page, "bbox": bbox}


def test_overlapping_explicit_number_disagreement_requires_review():
    report = build_formula_candidate_consensus([
        _numbered_evidence("text_layer", "(1)"),
        _numbered_evidence("pp_doclayout_region", "(2)"),
    ])
    assert report["cluster_count"] == 2
    assert report["supported_cluster_count"] == 0
    assert report["manual_review_cluster_count"] == 2
    assert all("equation_number_conflict" in c["review_flags"] for c in report["clusters"])


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_unnumbered_bridge_cannot_hide_number_disagreement(order):
    rows = [
        _numbered_evidence("text_layer", "(1)"),
        _numbered_evidence("pp_doclayout_region", ""),
        _numbered_evidence("mineru_content_list", "(2)"),
    ]
    report = build_formula_candidate_consensus(rows[i] for i in order)
    assert report["cluster_count"] == 2
    assert report["supported_cluster_count"] == 0
    assert report["manual_review_cluster_count"] == 2


def test_number_formatting_variants_still_support_each_other():
    report = build_formula_candidate_consensus([
        _numbered_evidence("text_layer", "(A.1)"),
        _numbered_evidence("pp_doclayout_region", "A.1"),
    ])
    assert report["supported_cluster_count"] == 1


@pytest.mark.parametrize("page,bbox", [(2, (50, 100, 300, 140)), (1, (50, 200, 300, 240))])
def test_separate_formulas_do_not_create_number_conflicts(page, bbox):
    report = build_formula_candidate_consensus([
        _numbered_evidence("text_layer", "(1)"),
        _numbered_evidence("pp_doclayout_region", "(2)", page=page, bbox=bbox),
    ])
    assert report["cluster_count"] == 2
    assert all("equation_number_conflict" not in c["review_flags"] for c in report["clusters"])
