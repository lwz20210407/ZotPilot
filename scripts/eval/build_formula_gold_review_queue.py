"""Select a compact, source-bound first-pass Label Studio review queue."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse


def main() -> int:
    args = _arguments()
    tasks = _json_list(Path(args.tasks))
    queue = build_review_queue(
        tasks,
        per_document=args.per_document,
        empty_pages_per_document=args.empty_pages_per_document,
    )
    if (args.image_url_root is None) != (args.image_url_prefix is None):
        raise ValueError("image-url-root and image-url-prefix must be provided together")
    if args.image_url_root is not None:
        queue = rewrite_file_image_urls(queue, args.image_url_root, args.image_url_prefix)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "task_count": len(queue)}, ensure_ascii=False))
    return 0


def build_review_queue(
    tasks: list[Mapping[str, Any]],
    *,
    per_document: int,
    empty_pages_per_document: int,
) -> list[Mapping[str, Any]]:
    """Prioritize dense formula pages plus explicit negative pages per PDF."""
    if per_document < 1 or empty_pages_per_document < 0:
        raise ValueError("per-document must be at least 1 and empty-pages-per-document cannot be negative")
    groups: dict[tuple[str, str], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, task in enumerate(tasks):
        if _has_completed_annotation(task):
            continue
        data = _mapping(task.get("data"))
        item_key = str(data.get("item_key", "")).strip()
        source_hash = str(data.get("source_pdf_sha256", "")).strip().lower()
        if item_key and source_hash:
            groups[(item_key, source_hash)].append((index, task))
    selected: list[Mapping[str, Any]] = []
    for _, group in sorted(groups.items()):
        scored = [(*_prediction_counts(task), _page_num(task), index, task) for index, task in group]
        formula_pages = [row for row in scored if row[0] > 0 or row[1] > 0]
        empty_pages = [row for row in scored if row[0] == 0 and row[1] == 0]
        formula_pages.sort(key=lambda row: (-row[1], -row[0], row[2], row[3]))
        empty_pages.sort(key=lambda row: (row[2], row[3]))
        selected.extend(row[4] for row in formula_pages[:per_document])
        selected.extend(row[4] for row in empty_pages[:empty_pages_per_document])
    return selected


def rewrite_file_image_urls(
    tasks: list[Mapping[str, Any]],
    image_url_root: Path | str,
    image_url_prefix: str | None,
) -> list[Mapping[str, Any]]:
    """Replace source-rooted ``file://`` task images with local static URLs."""
    root = Path(image_url_root).resolve()
    prefix = str(image_url_prefix or "").rstrip("/")
    rewritten: list[Mapping[str, Any]] = []
    for task in tasks:
        task_copy = dict(task)
        data = dict(_mapping(task.get("data")))
        image_url = str(data.get("image", "")).strip()
        image_path = _file_uri_path(image_url)
        if image_path is not None:
            try:
                relative_path = image_path.resolve().relative_to(root)
            except ValueError as error:
                raise ValueError("Task image is outside image-url-root") from error
            data["image"] = f"{prefix}/{quote(relative_path.as_posix(), safe='/')}"
        task_copy["data"] = data
        rewritten.append(task_copy)
    return rewritten


def _prediction_counts(task: Mapping[str, Any]) -> tuple[int, int]:
    formula_count = 0
    number_count = 0
    predictions = task.get("predictions")
    for prediction in predictions if isinstance(predictions, list) else []:
        if not isinstance(prediction, Mapping):
            continue
        results = prediction.get("result")
        for result in results if isinstance(results, list) else []:
            if not isinstance(result, Mapping) or str(result.get("from_name", "")) != "formula_region":
                continue
            labels = _mapping(result.get("value")).get("rectanglelabels")
            label = str(labels[0]).strip().lower() if isinstance(labels, list) and labels else ""
            formula_count += label == "formula"
            number_count += label == "formula_number"
    return formula_count, number_count


def _has_completed_annotation(task: Mapping[str, Any]) -> bool:
    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        return False
    return any(
        isinstance(annotation, Mapping)
        and not bool(annotation.get("was_cancelled"))
        and isinstance(annotation.get("result"), list)
        for annotation in annotations
    )


def _page_num(task: Mapping[str, Any]) -> int:
    value = _mapping(task.get("data")).get("page_num")
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _file_uri_path(value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme.lower() != "file":
        return None
    path = unquote(parsed.path)
    if parsed.netloc:
        return Path(f"//{parsed.netloc}{path}")
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Label Studio task-export JSON")
    parser.add_argument("--output", required=True, help="Selected task JSON for a first review pass")
    parser.add_argument("--per-document", type=int, default=3, help="Formula-bearing pages selected per source PDF")
    parser.add_argument(
        "--empty-pages-per-document",
        type=int,
        default=1,
        help="Prediction-empty negative pages selected per source PDF",
    )
    parser.add_argument("--image-url-root", help="Root directory allowed for file:// image URL rewriting")
    parser.add_argument("--image-url-prefix", help="Static URL prefix for rewritten task images")
    return parser.parse_args()


def _json_list(path: Path) -> list[Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{path} must contain a JSON list of Label Studio tasks")
    return list(value)


if __name__ == "__main__":
    raise SystemExit(main())
