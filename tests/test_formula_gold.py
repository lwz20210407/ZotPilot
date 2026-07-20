import hashlib
import importlib.util
import json

import pymupdf
import pytest

from zotpilot.feature_extraction.formula_gold import (
    export_label_studio_tasks,
    import_label_studio_gold,
)
from zotpilot.feature_extraction.formula_gold_metrics import (
    evaluate_coco_map,
    evaluate_formula_gold,
    evaluate_formula_gold_corpus,
    to_coco,
    to_coco_corpus,
)


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


def test_export_can_use_label_studio_local_file_urls(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    cache_path = tmp_path / "cache.json"
    _write_pdf(pdf_path)
    _write_cache(cache_path, pdf_path)

    tasks = export_label_studio_tasks(
        pdf_path,
        cache_path,
        tmp_path / "images",
        item_key="ITEM0001",
        label_studio_local_files_root=tmp_path,
    )

    assert tasks[0]["data"]["image"] == "/data/local-files/?d=images/ITEM0001-0001.png"


def test_export_can_use_a_local_static_image_url_prefix(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    cache_path = tmp_path / "cache.json"
    _write_pdf(pdf_path)
    _write_cache(cache_path, pdf_path)

    tasks = export_label_studio_tasks(
        pdf_path,
        cache_path,
        tmp_path / "images",
        item_key="ITEM0001",
        image_url_root=tmp_path,
        image_url_prefix="http://127.0.0.1:8092",
    )

    assert tasks[0]["data"]["image"] == "http://127.0.0.1:8092/images/ITEM0001-0001.png"


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
                {
                    "from_name": "equation_number",
                    "parentID": "formula-1",
                    "type": "textarea",
                    "value": {"text": ["(1)"]},
                },
                {
                    "from_name": "formula_layout",
                    "parentID": "formula-1",
                    "type": "choices",
                    "value": {"choices": ["display"]},
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
    assert report["formula_number_coverage"]["number_coverage"] == 1.0
    assert report["formula_number_coverage"]["duplicate_numbers"] == []
    assert report["formula_layout"]["gold_labelled_count"] == 1
    assert report["formula_layout"]["by_layout"]["display"]["recall"] == 0.0
    assert len(coco_gold["images"]) == 1
    assert len(coco_gold["annotations"]) == 2
    assert len(predictions) == 2
    assert gold["documents"][0]["pages"][0]["regions"][0]["latex"] == r"x = y"
    assert gold["documents"][0]["pages"][0]["regions"][0]["equation_number"] == "(1)"
    assert gold["documents"][0]["pages"][0]["regions"][0]["layout"] == "display"


def test_coco_map_explains_missing_optional_dependency():
    if importlib.util.find_spec("pycocotools") is not None:
        pytest.skip("exercise the missing-dependency path only")
    try:
        evaluate_coco_map({"images": [], "annotations": [], "categories": []}, [])
    except RuntimeError as error:
        assert "pycocotools" in str(error)
    else:
        raise AssertionError("expected optional dependency error")


def test_metrics_select_the_gold_document_bound_to_the_visual_cache_source(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    cache_path = tmp_path / "cache.json"
    _write_pdf(pdf_path)
    _write_cache(cache_path, pdf_path)
    source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    gold = {
        "documents": [
            {
                "item_key": "ITEM0001",
                "source_pdf_sha256": "b" * 64,
                "pages": [{"page_num": 1, "page_size_pt": [612, 792], "regions": []}],
            },
            {
                "item_key": "ITEM0001",
                "source_pdf_sha256": source_hash,
                "pages": [
                    {
                        "page_num": 1,
                        "page_size_pt": [612, 792],
                        "regions": [{"cls": "formula", "bbox_pt": [61.2, 79.2, 306, 158.4]}],
                    }
                ],
            },
        ]
    }

    report = evaluate_formula_gold(gold, cache, item_key="ITEM0001")

    assert report["source_pdf_sha256"] == source_hash
    assert report["by_class"]["formula"]["recall"] == 1.0


def test_corpus_metrics_aggregate_only_source_bound_item_caches():
    first_hash = "a" * 64
    second_hash = "b" * 64
    gold = {
        "documents": [
            {
                "item_key": "ITEM0001",
                "source_pdf_sha256": first_hash,
                "pages": [
                    {
                        "page_num": 1,
                        "page_size_pt": [100, 100],
                        "regions": [{"cls": "formula", "bbox_pt": [10, 10, 30, 30]}],
                    }
                ],
            },
            {
                "item_key": "ITEM0002",
                "source_pdf_sha256": second_hash,
                "pages": [
                    {
                        "page_num": 1,
                        "page_size_pt": [100, 100],
                        "regions": [{"cls": "formula", "bbox_pt": [10, 10, 30, 30]}],
                    }
                ],
            },
        ]
    }
    caches = {
        "ITEM0001": {
            "generator": "pp_doclayout",
            "source_pdf_sha256": first_hash,
            "pages": [{"page_num": 1, "regions": [{"cls": "formula", "bbox_pt": [10, 10, 30, 30], "conf": 1}]}],
        },
        "ITEM0002": {
            "generator": "pp_doclayout",
            "source_pdf_sha256": second_hash,
            "pages": [{"page_num": 1, "regions": [{"cls": "formula", "bbox_pt": [50, 50, 70, 70], "conf": 1}]}],
        },
    }

    report = evaluate_formula_gold_corpus(gold, caches)

    assert report["document_count"] == 2
    assert report["overall"] == {
        "gold_count": 2,
        "predicted_count": 2,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
    }
    assert set(report["documents"]) == {"ITEM0001", "ITEM0002"}


def test_corpus_coco_uses_unique_images_and_keeps_item_bound_predictions():
    first_hash = "a" * 64
    second_hash = "b" * 64
    gold = {
        "documents": [
            {
                "item_key": "ITEM0001",
                "source_pdf_sha256": first_hash,
                "pages": [
                    {
                        "page_num": 1,
                        "page_size_pt": [100, 100],
                        "regions": [{"cls": "formula", "bbox_pt": [10, 10, 30, 30]}],
                    }
                ],
            },
            {
                "item_key": "ITEM0002",
                "source_pdf_sha256": second_hash,
                "pages": [
                    {
                        "page_num": 1,
                        "page_size_pt": [100, 100],
                        "regions": [{"cls": "formula", "bbox_pt": [10, 10, 30, 30]}],
                    }
                ],
            },
        ]
    }
    caches = {
        "ITEM0001": {
            "generator": "pp_doclayout",
            "source_pdf_sha256": first_hash,
            "pages": [{"page_num": 1, "regions": [{"cls": "formula", "bbox_pt": [10, 10, 30, 30], "conf": 1}]}],
        },
        "ITEM0002": {
            "generator": "pp_doclayout",
            "source_pdf_sha256": second_hash,
            "pages": [{"page_num": 1, "regions": [{"cls": "formula", "bbox_pt": [10, 10, 30, 30], "conf": 1}]}],
        },
    }

    coco_gold, predictions = to_coco_corpus(gold, caches)

    assert [image["file_name"] for image in coco_gold["images"]] == ["ITEM0001-0001.png", "ITEM0002-0001.png"]
    assert [image["id"] for image in coco_gold["images"]] == [1, 2]
    assert [annotation["image_id"] for annotation in coco_gold["annotations"]] == [1, 2]
    assert [prediction["image_id"] for prediction in predictions] == [1, 2]


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
