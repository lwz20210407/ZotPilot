import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_builds_balanced_formula_gold_review_queue(tmp_path):
    tasks = [
        _task("ITEM0001", 1, formulas=3, numbers=3),
        _task("ITEM0001", 2, formulas=1, numbers=0),
        _task("ITEM0001", 3, formulas=0, numbers=0),
        _task("ITEM0002", 1, formulas=4, numbers=2),
        _task("ITEM0002", 2, formulas=0, numbers=0),
    ]
    tasks_path = tmp_path / "tasks.json"
    output_path = tmp_path / "queue.json"
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval/build_formula_gold_review_queue.py",
            "--tasks",
            str(tasks_path),
            "--output",
            str(output_path),
            "--per-document",
            "1",
            "--empty-pages-per-document",
            "1",
        ],
        cwd=root,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    queue = json.loads(output_path.read_text(encoding="utf-8"))
    assert [(task["data"]["item_key"], task["data"]["page_num"]) for task in queue] == [
        ("ITEM0001", 1),
        ("ITEM0001", 3),
        ("ITEM0002", 1),
        ("ITEM0002", 2),
    ]


def _task(item_key, page_num, *, formulas, numbers):
    regions = [
        {"from_name": "formula_region", "value": {"rectanglelabels": ["formula"]}}
        for _ in range(formulas)
    ]
    regions.extend(
        {"from_name": "formula_region", "value": {"rectanglelabels": ["formula_number"]}}
        for _ in range(numbers)
    )
    return {
        "data": {"item_key": item_key, "page_num": page_num, "source_pdf_sha256": "a" * 64},
        "predictions": [{"result": regions}],
        "annotations": [],
    }
