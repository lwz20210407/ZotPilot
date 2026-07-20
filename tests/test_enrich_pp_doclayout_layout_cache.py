import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pymupdf


def test_cli_writes_projection_columns_to_a_derived_cache_only(tmp_path):
    pdf_path = tmp_path / "two-column.pdf"
    cache_path = tmp_path / "detector.json"
    output_path = tmp_path / "derived.json"
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    for index in range(16):
        y = 70 + index * 24
        page.insert_text((50, y), f"left prose {index}", fontsize=9)
        page.insert_text((340, y), f"right prose {index}", fontsize=9)
    document.save(pdf_path)
    document.close()
    source_payload = {
        "schema_version": 1,
        "generator": "pp_doclayout",
        "source_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "pages": [{"page_num": 1, "page_size_pt": [612, 792], "coordinate_space": "pdf", "regions": []}],
    }
    cache_path.write_text(json.dumps(source_payload), encoding="utf-8")
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval/enrich_pp_doclayout_layout_cache.py",
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
    assert json.loads(cache_path.read_text(encoding="utf-8")) == source_payload
    derived = json.loads(output_path.read_text(encoding="utf-8"))
    assert derived["layout_enrichment"]["generator"] == "zotpilot_text_projection_columns"
    blocks = derived["pages"][0]["blocks"]
    assert len(blocks) == 2
    assert all(block["cls"] == "column" for block in blocks)
    assert all(block["source"] == "zotpilot_text_projection" for block in blocks)
