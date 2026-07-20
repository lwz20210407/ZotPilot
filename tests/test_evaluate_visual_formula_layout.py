import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pymupdf


def test_cli_evaluates_cached_visual_formula_layout_without_model_or_store(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    cache_path = tmp_path / "pp_doclayout_layout.json"
    output_path = tmp_path / "report.json"
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((70, 180), "x = y", fontsize=12)
    page.insert_text((500, 180), "(1)", fontsize=12)
    document.save(pdf_path)
    document.close()
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "pp_doclayout",
                "source_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                "pages": [
                    {
                        "page_num": 1,
                        "page_size_pt": [612, 792],
                        "coordinate_space": "pdf",
                        "regions": [
                            {"cls": "formula", "bbox_pt": [60, 160, 350, 190], "conf": 0.95},
                            {"cls": "formula_number", "bbox_pt": [490, 160, 550, 190], "conf": 0.95},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval/evaluate_visual_formula_layout.py",
            "--pdf",
            str(pdf_path),
            "--cache",
            str(cache_path),
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
    assert report["mode"] == "read_only_pp_doclayout_layout_evaluation"
    assert report["candidate_detector_floor"] == 0.5
    assert report["raw_formula_region_count"] == 1
    assert report["strict_formula_region_count"] == 1
    assert report["raw_formula_number_region_count"] == 1
    assert report["display_candidate_count"] == 1
    assert report["bound_equation_number_count"] == 1
    assert report["candidates"][0]["equation_number"] == "(1)"
