import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pymupdf


def test_export_cli_can_write_local_static_image_urls(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    document = pymupdf.open()
    document.new_page(width=612, height=792)
    document.save(pdf_path)
    document.close()
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "generator": "pp_doclayout",
                "source_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                "pages": [{"page_num": 1, "page_size_pt": [612, 792], "regions": []}],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "tasks.json"
    environment = os.environ | {"PYTHONPATH": str(Path(__file__).parents[1] / "src")}

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval/export_formula_labelstudio.py",
            str(pdf_path),
            str(cache_path),
            str(output_path),
            "--images",
            str(tmp_path / "images"),
            "--item-key",
            "ITEM0001",
            "--image-url-root",
            str(tmp_path),
            "--image-url-prefix",
            "http://127.0.0.1:8092",
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))[0]["data"]["image"] == "http://127.0.0.1:8092/images/ITEM0001-0001.png"
