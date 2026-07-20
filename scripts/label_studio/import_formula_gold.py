"""Convert completed Label Studio formula tasks into ZotPilot Gold JSON."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zotpilot.feature_extraction.formula_gold import import_label_studio_gold


def main() -> int:
    args = _arguments()
    tasks = _json_list(Path(args.tasks))
    gold = import_label_studio_gold(tasks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "documents": len(gold["documents"]),
                "pages": sum(len(document["pages"]) for document in gold["documents"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Completed Label Studio task-export JSON")
    parser.add_argument("--output", required=True, help="ZotPilot reviewed Gold JSON")
    return parser.parse_args()


def _json_list(path: Path) -> list[Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{path} must contain a JSON list of Label Studio tasks")
    return list(value)


if __name__ == "__main__":
    raise SystemExit(main())
