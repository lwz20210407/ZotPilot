import pytest

from zotpilot.feature_extraction.formula_gold import import_label_studio_gold, summarize_label_studio_review


def _task(link="id"):
    return {
        "data": {"item_key": "EXAMPLE1", "source_pdf_sha256": "a" * 64, "page_num": 1},
        "meta": {"page_size_pt": [600, 800]},
        "annotations": [{"id": 7, "was_cancelled": False, "result": [
            {"id": "formula-1", "type": "rectanglelabels", "value": {
                "x": 10, "y": 20, "width": 40, "height": 10, "rectanglelabels": ["formula"]}},
            {link: "formula-1", "type": "textarea", "from_name": "formula_latex", "value": {"text": ["x=y"]}},
            {link: "formula-1", "type": "textarea", "from_name": "equation_number", "value": {"text": ["(1)"]}},
            {link: "formula-1", "type": "choices", "from_name": "formula_layout", "value": {"choices": ["display"]}},
        ]}],
    }


@pytest.mark.parametrize("link", ["id", "parentID", "parent_id", "parentId"])
def test_standard_and_legacy_region_links_preserve_reviewed_fields(link):
    task = _task(link)
    region = import_label_studio_gold([task])["documents"][0]["pages"][0]["regions"][0]
    assert region["latex"] == "x=y"
    assert region["equation_number"] == "(1)"
    assert region["layout"] == "display"
    summary = summarize_label_studio_review([task])
    assert summary["latex_annotated_count"] == summary["equation_number_annotated_count"] == 1
    assert summary["layout_annotated_count"] == 1


def test_global_or_unknown_region_controls_are_not_attached_to_formula():
    task = _task()
    for result in task["annotations"][0]["result"][1:]:
        result["id"] = "global-note"
        result["parentID"] = "absent-region"
    region = import_label_studio_gold([task])["documents"][0]["pages"][0]["regions"][0]
    assert region["latex"] == region["equation_number"] == ""
    assert region["layout"] == "unknown"


def test_same_region_id_wins_over_hierarchical_parent_link():
    task = _task()
    results = task["annotations"][0]["result"]
    results.append({"id": "parent-region", "type": "rectanglelabels", "value": {
        "x": 50, "y": 40, "width": 20, "height": 10, "rectanglelabels": ["formula"]}})
    for result in results[1:4]:
        result["parentID"] = "parent-region"
    regions = import_label_studio_gold([task])["documents"][0]["pages"][0]["regions"]
    assert regions[0]["latex"] == "x=y"
    assert regions[1]["latex"] == ""


def test_prediction_and_cancelled_review_do_not_supply_gold():
    task = _task()
    task["predictions"] = task.pop("annotations")
    assert import_label_studio_gold([task])["documents"] == []
    task["annotations"] = task.pop("predictions")
    task["annotations"][0]["was_cancelled"] = True
    assert import_label_studio_gold([task])["documents"] == []
