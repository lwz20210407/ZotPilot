"""Export a PP-DocLayout visual-region cache for ZotPilot review."""

from __future__ import annotations

import argparse
from pathlib import Path

from zotpilot.feature_extraction.vision_layout.pp_doclayout_runner import export_pp_doclayout_cache


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export PP-DocLayout detections for ZotPilot formula review.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-name", default="PP-DocLayout_plus-L")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--engine", choices=("paddle_static", "onnxruntime"), default="onnxruntime")
    return parser


def main() -> None:
    args = _parser().parse_args()
    export_pp_doclayout_cache(
        args.pdf,
        args.output,
        model_name=args.model_name,
        model_dir=args.model_dir,
        dpi=args.dpi,
        max_pages=args.max_pages,
        confidence_threshold=args.threshold,
        device=args.device,
        engine=args.engine,
    )


if __name__ == "__main__":
    main()
