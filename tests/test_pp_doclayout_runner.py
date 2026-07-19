from zotpilot.feature_extraction.vision_layout.pp_doclayout_runner import _pdf_regions


def test_pp_doclayout_regions_convert_pixels_to_pdf_points():
    result = {
        "res": {
            "boxes": [
                {"label": "formula", "score": 0.9, "coordinate": [100, 200, 500, 280]},
                {"label": "formula_number", "score": 0.8, "coordinate": [900, 200, 1000, 280]},
            ]
        }
    }

    regions = _pdf_regions(
        result,
        image_width=1224,
        image_height=1584,
        page_width=612,
        page_height=792,
    )

    assert regions == [
        {"cls": "formula", "bbox_pt": [50.0, 100.0, 250.0, 140.0], "conf": 0.9},
        {"cls": "formula_number", "bbox_pt": [450.0, 100.0, 500.0, 140.0], "conf": 0.8},
    ]
