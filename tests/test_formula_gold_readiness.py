import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_writes_label_studio_review_readiness(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    output_path = tmp_path / "readiness.json"
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "data": {"item_key": "ITEM0001", "source_pdf_sha256": "a" * 64},
                    "meta": {"page_size_pt": [100, 100]},
                    "predictions": [{"result": []}],
                    "annotations": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval/formula_gold_review_readiness.py",
            "--tasks",
            str(tasks_path),
            "--output",
            str(output_path),
        ],
        cwd=root,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["preannotated_task_count"] == 1
    assert report["reviewed_task_count"] == 0


def test_gold_import_cli_rejects_unreviewed_predictions_but_keeps_reviewed_negative_pages(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    output_path = tmp_path / "gold.json"
    task = {
        "data": {"item_key": "ITEM0001", "page_num": 1, "source_pdf_sha256": "a" * 64},
        "meta": {"page_size_pt": [100, 100]},
        "predictions": [{"result": []}],
        "annotations": [],
    }
    tasks_path.write_text(json.dumps([task]), encoding="utf-8")
    root = Path(__file__).parents[1]
    command = [
        sys.executable,
        "scripts/eval/import_formula_labelstudio.py",
        str(tasks_path),
        str(output_path),
    ]

    unreviewed = subprocess.run(
        command,
        cwd=root,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert unreviewed.returncode != 0
    assert not output_path.exists()

    alternate_output = tmp_path / "alternate-gold.json"
    alternate_unreviewed = subprocess.run(
        [
            sys.executable,
            "scripts/label_studio/import_formula_gold.py",
            "--tasks",
            str(tasks_path),
            "--output",
            str(alternate_output),
        ],
        cwd=root,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert alternate_unreviewed.returncode != 0
    assert not alternate_output.exists()

    task["annotations"] = [{"was_cancelled": False, "result": []}]
    tasks_path.write_text(json.dumps([task]), encoding="utf-8")
    reviewed = subprocess.run(
        command,
        cwd=root,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert reviewed.returncode == 0, reviewed.stderr
    gold = json.loads(output_path.read_text(encoding="utf-8"))
    assert gold["documents"][0]["pages"][0]["regions"] == []
