from zotpilot.feature_extraction.vision_layout.formula_region_metrics import bbox_iou, evaluate_formula_regions


def test_formula_region_metrics_matches_one_to_one_and_marks_draft_reference():
    annotation = {
        "documents": [
            {
                "item_key": "ITEM123",
                "proposed_formulas": [
                    {"page_num": 1, "bbox": [10, 10, 30, 30], "equation_number": "(1)"},
                    {"page_num": 2, "bbox": [10, 10, 30, 30], "equation_number": "(2)"},
                ],
            }
        ]
    }
    cache = {
        "pages": [
            {
                "page_num": 1,
                "regions": [
                    {"cls": "formula", "bbox_pt": [10, 10, 30, 30], "conf": 0.9},
                    {"cls": "text", "bbox_pt": [40, 40, 50, 50], "conf": 0.9},
                ],
            },
            {"page_num": 2, "regions": [{"cls": "formula", "bbox_pt": [50, 50, 70, 70], "conf": 0.9}]},
        ]
    }

    report = evaluate_formula_regions(annotation, cache, display_only=False)

    assert report["reference_kind"] == "draft_proposal"
    assert report["true_positive"] == 1
    assert report["false_positive"] == 1
    assert report["false_negative"] == 1
    assert report["precision"] == 0.5
    assert report["recall"] == 0.5


def test_formula_region_iou_handles_disjoint_and_identical_boxes():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_formula_region_metrics_can_scale_normalized_draft_coordinates():
    annotation = {
        "documents": [
            {"item_key": "ITEM123", "proposed_formulas": [{"page_num": 1, "bbox": [500, 500, 600, 600]}]}
        ]
    }
    cache = {
        "pages": [
            {
                "page_num": 1,
                "page_size_pt": [600, 800],
                "regions": [
                    {"cls": "formula", "bbox_pt": [300, 400, 360, 480], "conf": 0.9},
                    {"cls": "formula_number", "bbox_pt": [500, 420, 510, 440], "conf": 0.9},
                ],
            }
        ]
    }

    report = evaluate_formula_regions(annotation, cache, draft_coordinate_space="normalized_1000")

    assert report["true_positive"] == 1
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0


def test_formula_region_metrics_auto_detects_normalized_draft_coordinates_from_annotation_metadata():
    annotation = {
        "documents": [
            {
                "item_key": "ITEM123",
                "proposed_bbox_coordinate_space": "normalized_1000",
                "proposed_formulas": [{"page_num": 1, "bbox": [500, 500, 600, 600]}],
            }
        ]
    }
    cache = {
        "pages": [
            {
                "page_num": 1,
                "page_size_pt": [600, 800],
                "regions": [{"cls": "formula", "bbox_pt": [300, 400, 360, 480], "conf": 0.9}],
            }
        ]
    }

    report = evaluate_formula_regions(annotation, cache, draft_coordinate_space="auto", display_only=False)

    assert report["draft_coordinate_space"] == "normalized_1000"
    assert report["true_positive"] == 1
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
