"""Convert completed Label Studio formula labels into ZotPilot gold JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zotpilot.feature_extraction.formula_gold import import_label_studio_gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        parser.error("Label Studio export must be a JSON list of tasks")
    gold = import_label_studio_gold(tasks)
    args.output.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
