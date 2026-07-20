"""Report which Label Studio formula tasks are actually reviewed."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zotpilot.feature_extraction.formula_gold import summarize_label_studio_review


def main() -> int:
    args = _arguments()
    tasks = _json_list(Path(args.tasks))
    report = summarize_label_studio_review(tasks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "reviewed_tasks": report["reviewed_task_count"],
                "unreviewed_tasks": report["unreviewed_task_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Label Studio task-export JSON")
    parser.add_argument("--output", required=True, help="Review-readiness report JSON")
    return parser.parse_args()


def _json_list(path: Path) -> list[Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{path} must contain a JSON list of Label Studio tasks")
    return list(value)


if __name__ == "__main__":
    raise SystemExit(main())
