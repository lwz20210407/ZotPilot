"""Write a derived PP-DocLayout cache with deterministic projection columns.

The source detector cache is never modified.  This gives later formula-number
binding runs a source-bound, reviewable page-column artifact even when the
optional PP-DocBlockLayout runtime is not installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from zotpilot.feature_extraction.vision_layout.formula_layout_filter import infer_visual_page_columns
from zotpilot.feature_extraction.vision_layout.pp_doclayout_candidate_cache import (
    COLUMN_ENRICHMENT_GENERATOR,
    COLUMN_ENRICHMENT_VERSION,
)


def main() -> int:
    args = _arguments()
    payload = _json_object(args.cache)
    _validate_source_binding(args.pdf, payload)
    columns = infer_visual_page_columns(args.pdf)
    enriched = _with_projection_columns(payload, columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "page_count": len(enriched.get("pages", [])),
                "column_block_count": sum(
                    1
                    for page in enriched.get("pages", [])
                    if isinstance(page, dict)
                    for block in page.get("blocks", [])
                    if isinstance(block, dict) and block.get("cls") == "column"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _with_projection_columns(
    payload: dict[str, Any],
    columns_by_page: dict[int, tuple[Any, ...]],
) -> dict[str, Any]:
    enriched = deepcopy(payload)
    pages = enriched.get("pages")
    if not isinstance(pages, list):
        raise ValueError("PP-DocLayout cache must contain pages")
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_num = _positive_int(page.get("page_num"))
        page_size = page.get("page_size_pt")
        if page_num is None or not isinstance(page_size, list) or len(page_size) != 2:
            continue
        try:
            height = float(page_size[1])
        except (TypeError, ValueError):
            continue
        if height <= 0:
            continue
        existing = page.get("blocks")
        blocks = list(existing) if isinstance(existing, list) else []
        blocks = [
            block
            for block in blocks
            if not (
                isinstance(block, dict)
                and block.get("cls") == "column"
                and block.get("source") == "zotpilot_text_projection"
            )
        ]
        blocks.extend(
            {
                "cls": "column",
                "bbox_pt": [round(column.x0, 3), 0.0, round(column.x1, 3), round(height, 3)],
                "coordinate_space": "pdf",
                "source": "zotpilot_text_projection",
            }
            for column in columns_by_page.get(page_num, ())
        )
        page["blocks"] = blocks
    enriched["layout_enrichment"] = {
        "generator": COLUMN_ENRICHMENT_GENERATOR,
        "schema_version": COLUMN_ENRICHMENT_VERSION,
        "source_pdf_sha256": str(enriched.get("source_pdf_sha256", "")),
    }
    return enriched


def _validate_source_binding(pdf_path: Path, payload: dict[str, Any]) -> None:
    if str(payload.get("generator", "")).strip().lower() != "pp_doclayout":
        raise ValueError("Expected a PP-DocLayout cache")
    expected = str(payload.get("source_pdf_sha256", "")).strip().lower()
    actual = _sha256(pdf_path)
    if not expected or expected != actual:
        raise ValueError("PP-DocLayout cache does not belong to the supplied PDF")


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("PP-DocLayout cache must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
