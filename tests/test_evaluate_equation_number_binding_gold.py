import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pymupdf


def test_cli_evaluates_source_bound_equation_number_binding_against_gold(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((60, 120), "x = y", fontsize=12)
    page.insert_text((500, 120), "(1)", fontsize=12)
    document.save(pdf_path)
    document.close()
    source_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    cache_path = tmp_path / "layout.json"
    cache_path.write_text(
        json.dumps(
            {
                "generator": "pp_doclayout",
                "source_pdf_sha256": source_hash,
                "pages": [
                    {
                        "page_num": 1,
                        "page_size_pt": [612, 792],
                        "coordinate_space": "pdf",
                        "regions": [
                            {"cls": "formula", "bbox_pt": [50, 105, 350, 135], "conf": 0.95},
                            {"cls": "formula_number", "bbox_pt": [490, 105, 550, 135], "conf": 0.95},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "item_key": "ITEM0001",
                        "source_pdf_sha256": source_hash,
                        "pages": [
                            {
                                "page_num": 1,
                                "regions": [
                                    {
                                        "cls": "formula",
                                        "bbox_pt": [50, 105, 350, 135],
                                        "equation_number": "(1)",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval/evaluate_equation_number_binding_gold.py",
            "--pdf",
            str(pdf_path),
            "--gold",
            str(gold_path),
            "--cache",
            str(cache_path),
            "--item-key",
            "ITEM0001",
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
    assert report["true_positive"] == 1
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["binding_diagnostics"]["bound_numbers"] == ["(1)"]
