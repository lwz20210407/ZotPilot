from copy import deepcopy

import pytest

from zotpilot.feature_extraction.formula_gold_metrics import evaluate_formula_gold, to_coco


def _inputs():
    region = {"cls": "formula", "bbox_pt": [10, 10, 40, 30], "layout": "display"}
    gold = {"documents": [{
        "item_key": "ITEM0001", "source_pdf_sha256": "a" * 64,
        "pages": [{"page_num": 1, "page_size_pt": [100, 100], "regions": [region]}],
    }]}
    cache = {"generator": "pp_doclayout", "source_pdf_sha256": "a" * 64,
             "pages": [{"page_num": 1, "regions": [{**region, "conf": 0.9}]}]}
    return gold, cache


@pytest.mark.parametrize("side", ["gold", "predicted"])
def test_missing_layout_is_unavailable_even_with_some_labelled_regions(side):
    gold, cache = _inputs()
    regions = gold["documents"][0]["pages"][0]["regions"] if side == "gold" else cache["pages"][0]["regions"]
    unknown = {**regions[0], "bbox_pt": [10, 50, 40, 70]}
    unknown.pop("layout")
    regions.append(unknown)
    layout = evaluate_formula_gold(gold, cache, item_key="ITEM0001")["formula_layout"]
    assert layout["available"] is False
    assert layout["unavailable_reasons"] == [f"{side}_formula_layout_missing"]
    assert layout[f"{side}_unknown_count"] == 1
    assert layout["by_layout"] == {}


def test_true_empty_predictions_still_report_display_misses():
    gold, cache = _inputs()
    cache["pages"][0]["regions"] = []
    layout = evaluate_formula_gold(gold, cache, item_key="ITEM0001")["formula_layout"]
    assert layout["available"] is True
    assert layout["by_layout"]["display"]["false_negative"] == 1
    assert layout["by_layout"]["display"]["recall"] == 0.0


def test_display_metric_is_distinct_from_pooled_inline_recall():
    gold, cache = _inputs()
    gold["documents"][0]["pages"][0]["regions"].append(
        {"cls": "formula", "bbox_pt": [10, 50, 40, 70], "layout": "inline"}
    )
    report = evaluate_formula_gold(gold, cache, item_key="ITEM0001")
    assert report["formula_layout"]["available"] is True
    assert report["formula_layout"]["by_layout"]["display"]["recall"] == 1.0
    assert report["by_class"]["formula"]["recall"] == 0.5


def test_unreviewed_page_predictions_do_not_become_false_positives():
    gold, cache = _inputs()
    unreviewed_page = deepcopy(cache["pages"][0])
    unreviewed_page["page_num"] = 2
    unreviewed_page["regions"][0].pop("layout")
    cache["pages"].append(unreviewed_page)
    report = evaluate_formula_gold(gold, cache, item_key="ITEM0001")
    _, predictions = to_coco(gold, cache, item_key="ITEM0001")
    assert report["reviewed_page_numbers"] == [1]
    assert report["overall"]["predicted_count"] == len(predictions) == 1
    assert report["overall"]["precision"] == 1.0
    assert report["formula_layout"]["available"] is True
