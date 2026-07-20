import hashlib
import importlib.util
import json

import pymupdf
import pytest

from zotpilot.feature_extraction.formula_gold import (
    export_label_studio_tasks,
    import_label_studio_gold,
)
from zotpilot.feature_extraction.formula_gold_metrics import evaluate_coco_map, evaluate_formula_gold, to_coco


def _write_pdf(path):
    document = pymupdf.open()
    document.new_page(width=612, height=792)
    document.save(path)
    document.close()


def _write_cache(path, pdf_path):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "pp_doclayout",
                "model": "PP-DocLayout_plus-L",
                "source_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                "pages": [
                    {
                        "page_num": 1,
                        "page_size_pt": [612, 792],
                        "coordinate_space": "pdf",
                        "regions": [
                            {"cls": "formula", "bbox_pt": [61.2, 79.2, 306, 158.4], "conf": 0.9},
                            {"cls": "formula_number", "bbox_pt": [500, 100, 530, 130], "conf": 0.8},
                            {"cls": "text", "bbox_pt": [10, 10, 100, 30], "conf": 0.99},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_export_uses_percentage_coordinates_and_keeps_source_hash(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    cache_path = tmp_path / "cache.json"
    _write_pdf(pdf_path)
    _write_cache(cache_path, pdf_path)

    tasks = export_label_studio_tasks(pdf_path, cache_path, tmp_path / "images", item_key="ITEM0001")

    assert len(tasks) == 1
    task = tasks[0]
    assert task["data"]["item_key"] == "ITEM0001"
    assert task["data"]["source_pdf_sha256"] == hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    result = task["predictions"][0]["result"]
    assert [row["value"]["rectanglelabels"] for row in result] == [["formula"], ["formula_number"]]
    assert result[0]["value"]["x"] == 10.0
    assert result[0]["value"]["width"] == 40.0
    assert (tmp_path / "images" / "ITEM0001-0001.png").is_file()


def test_import_recovers_pdf_point_gold_boxes_from_completed_annotation(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    cache_path = tmp_path / "cache.json"
    _write_pdf(pdf_path)
    _write_cache(cache_path, pdf_path)
    task = export_label_studio_tasks(pdf_path, cache_path, tmp_path / "images", item_key="ITEM0001")[0]
    task["annotations"] = [
        {
            "was_cancelled": False,
            "result": [
                {
                    "id": "formula-1",
                    "type": "rectanglelabels",
                    "value": {
                        "x": 10,
                        "y": 10,
                        "width": 40,
                        "height": 10,
                        "rectanglelabels": ["formula"],
                    },
                }
            ],
        }
    ]

    gold = import_label_studio_gold([task])

    page = gold["documents"][0]["pages"][0]
    assert page["regions"] == [
        {
            "cls": "formula",
            "bbox_pt": [61.2, 79.2, 306.0, 158.4],
            "layout": "unknown",
            "equation_number": "",
            "latex": "",
        }
    ]


def test_export_rejects_a_cache_for_a_different_pdf(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    other_pdf = tmp_path / "other.pdf"
    cache_path = tmp_path / "cache.json"
    _write_pdf(pdf_path)
    _write_pdf(other_pdf)
    _write_cache(cache_path, other_pdf)

    try:
        export_label_studio_tasks(pdf_path, cache_path, tmp_path / "images", item_key="ITEM0001")
    except ValueError as error:
        assert "does not belong" in str(error)
    else:
        raise AssertionError("expected source-PDF validation failure")


def test_metrics_match_reviewed_gold_and_export_coco(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    cache_path = tmp_path / "cache.json"
    _write_pdf(pdf_path)
    _write_cache(cache_path, pdf_path)
    task = export_label_studio_tasks(pdf_path, cache_path, tmp_path / "images", item_key="ITEM0001")[0]
    task["annotations"] = [
        {
            "was_cancelled": False,
            "result": [
                {
                    "id": "formula-1",
                    "type": "rectanglelabels",
                    "value": {
                        "x": 10,
                        "y": 10,
                        "width": 40,
                        "height": 10,
                        "rectanglelabels": ["formula"],
                    },
                },
                {
                    "id": "number-1",
                    "type": "rectanglelabels",
                    "value": {
                        "x": 81.699346,
                        "y": 12.626263,
                        "width": 4.901961,
                        "height": 3.787879,
                        "rectanglelabels": ["formula_number"],
                    },
                },
                {
                    "from_name": "formula_latex",
                    "parentID": "formula-1",
                    "type": "textarea",
                    "value": {"text": [r"x = y"]},
                },
            ],
        }
    ]
    gold = import_label_studio_gold([task])
    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    report = evaluate_formula_gold(gold, cache, item_key="ITEM0001")
    coco_gold, predictions = to_coco(gold, cache, item_key="ITEM0001")

    assert report["overall"]["true_positive"] == 2
    assert report["overall"]["precision"] == 1.0
    assert report["overall"]["recall"] == 1.0
    assert report["formula_latex_coverage"]["latex_coverage"] == 1.0
    assert len(coco_gold["images"]) == 1
    assert len(coco_gold["annotations"]) == 2
    assert len(predictions) == 2
    assert gold["documents"][0]["pages"][0]["regions"][0]["latex"] == r"x = y"


def test_coco_map_explains_missing_optional_dependency():
    if importlib.util.find_spec("pycocotools") is not None:
        pytest.skip("exercise the missing-dependency path only")
    try:
        evaluate_coco_map({"images": [], "annotations": [], "categories": []}, [])
    except RuntimeError as error:
        assert "pycocotools" in str(error)
    else:
        raise AssertionError("expected optional dependency error")


@pytest.mark.skipif(importlib.util.find_spec("pycocotools") is None, reason="optional validation dependency")
def test_coco_map_evaluates_a_perfect_prediction():
    gold = {
        "images": [{"id": 1, "file_name": "page.png", "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "formula"}, {"id": 2, "name": "formula_number"}],
    }
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99}]

    report = evaluate_coco_map(gold, predictions)

    assert report["map50"] == pytest.approx(1.0)
    assert report["mar100"] == pytest.approx(1.0)
