import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_item_cache_arguments_require_explicit_nonempty_item_key_and_path():
    module = _load_script()

    assert module.parse_item_caches(["ITEM0001=C:/cache/one.json"]) == {"ITEM0001": Path("C:/cache/one.json")}
    with pytest.raises(ValueError, match="ITEM_KEY=PATH"):
        module.parse_item_caches(["missing-separator"])
    with pytest.raises(ValueError, match="Duplicate"):
        module.parse_item_caches(["ITEM0001=one.json", "ITEM0001=two.json"])


def test_cli_aggregates_explicit_item_caches(tmp_path):
    first_hash = "a" * 64
    second_hash = "b" * 64
    gold_path = tmp_path / "gold.json"
    first_cache = tmp_path / "first-cache.json"
    second_cache = tmp_path / "second-cache.json"
    output_path = tmp_path / "report.json"
    gold_path.write_text(
        json.dumps(
            {
                "documents": [
                    _document("ITEM0001", first_hash),
                    _document("ITEM0002", second_hash),
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_cache(first_cache, first_hash)
    _write_cache(second_cache, second_hash)
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval/evaluate_formula_gold.py",
            "--gold",
            str(gold_path),
            "--item-cache",
            f"ITEM0001={first_cache}",
            "--item-cache",
            f"ITEM0002={second_cache}",
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
    assert report["document_count"] == 2
    assert report["overall"]["precision"] == 1.0
    assert report["overall"]["recall"] == 1.0


def _document(item_key, source_hash):
    return {
        "item_key": item_key,
        "source_pdf_sha256": source_hash,
        "pages": [
            {
                "page_num": 1,
                "page_size_pt": [100, 100],
                "regions": [{"cls": "formula", "bbox_pt": [10, 10, 30, 30]}],
            }
        ],
    }


def _write_cache(path, source_hash):
    path.write_text(
        json.dumps(
            {
                "generator": "pp_doclayout",
                "source_pdf_sha256": source_hash,
                "pages": [
                    {
                        "page_num": 1,
                        "regions": [{"cls": "formula", "bbox_pt": [10, 10, 30, 30], "conf": 1}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _load_script():
    script = Path(__file__).parents[1] / "scripts" / "eval" / "evaluate_formula_gold.py"
    spec = importlib.util.spec_from_file_location("evaluate_formula_gold_script_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
