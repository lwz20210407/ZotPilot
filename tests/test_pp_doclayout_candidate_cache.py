import hashlib
import json

from zotpilot.feature_extraction.vision_layout.pp_doclayout_candidate_cache import (
    load_pp_doclayout_formula_regions,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cache(path, pdf_path, regions):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "pp_doclayout",
                "source_pdf_sha256": _sha256(pdf_path),
                "pages": [
                    {
                        "page_num": 1,
                        "page_size_pt": [612, 792],
                        "coordinate_space": "pdf",
                        "regions": regions,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_cache_reader_returns_only_valid_display_formula_regions_for_same_pdf(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"not a real PDF; only the content hash matters here")
    cache_path = tmp_path / "pp_doclayout_layout.json"
    _write_cache(
        cache_path,
        pdf_path,
        [
            {"cls": "formula", "bbox_pt": [50, 100, 300, 140], "conf": 0.9},
            {"cls": "formula_number", "bbox_pt": [500, 100, 530, 140], "conf": 0.9},
            {"cls": "text", "bbox_pt": [50, 200, 300, 230], "conf": 0.99},
            {"cls": "formula", "bbox_pt": [50, 250, 300, 270], "conf": 0.2},
        ],
    )

    regions = load_pp_doclayout_formula_regions(pdf_path, [cache_path], min_confidence=0.6)

    assert len(regions) == 1
    assert regions[0].page_num == 1
    assert regions[0].bbox == (50.0, 100.0, 300.0, 140.0)
    assert regions[0].confidence == 0.9
    assert regions[0].source_artifact_hash == _sha256(cache_path)


def test_cache_reader_rejects_mismatched_or_out_of_page_coordinates(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"pdf A")
    cache_path = tmp_path / "pp_doclayout_layout.json"
    _write_cache(
        cache_path,
        pdf_path,
        [{"cls": "formula", "bbox_pt": [50, 100, 800, 140], "conf": 0.9}],
    )
    mismatched = tmp_path / "other-pp_doclayout_layout.json"
    _write_cache(
        mismatched,
        tmp_path / "paper.pdf",
        [{"cls": "formula", "bbox_pt": [50, 100, 300, 140], "conf": 0.9}],
    )
    payload = json.loads(mismatched.read_text(encoding="utf-8"))
    payload["source_pdf_sha256"] = "0" * 64
    mismatched.write_text(json.dumps(payload), encoding="utf-8")

    assert load_pp_doclayout_formula_regions(pdf_path, [cache_path, mismatched]) == []
