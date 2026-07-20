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
