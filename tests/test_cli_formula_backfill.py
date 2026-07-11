import json
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_print_json_falls_back_for_narrow_stdout(monkeypatch):
    from zotpilot.cli import _print_json

    class GbkOnlyStdout:
        encoding = "ascii"

        def __init__(self):
            self.text = ""

        def write(self, value):
            value.encode(self.encoding)
            self.text += value
            return len(value)

        def flush(self):
            return None

    fake_stdout = GbkOnlyStdout()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    _print_json({"title": "A® paper"}, ensure_ascii=False, indent=2)

    assert "\\u00ae" in fake_stdout.text
    assert json.loads(fake_stdout.text)["title"] == "A® paper"


def test_print_json_avoids_partial_narrow_stdout(monkeypatch):
    from zotpilot.cli import _print_json

    class PartialFailingStdout:
        encoding = "ascii"

        def __init__(self):
            self.text = ""

        def write(self, value):
            self.text += value
            value.encode(self.encoding)
            return len(value)

        def flush(self):
            return None

    fake_stdout = PartialFailingStdout()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    _print_json({"title": "A® paper"}, ensure_ascii=False, indent=2)

    assert fake_stdout.text.count("{") == 1
    assert json.loads(fake_stdout.text)["title"] == "A® paper"


def test_print_json_escapes_non_utf8_stdout(monkeypatch):
    from zotpilot.cli import _print_json

    class GbkStdout:
        encoding = "gbk"

        def __init__(self):
            self.text = ""

        def write(self, value):
            value.encode(self.encoding)
            self.text += value
            return len(value)

        def flush(self):
            return None

    fake_stdout = GbkStdout()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    _print_json({"title": "EN‐AW 6082‐T6"}, ensure_ascii=False, indent=2)

    assert "\\u2010" in fake_stdout.text
    assert json.loads(fake_stdout.text)["title"] == "EN‐AW 6082‐T6"


def test_formula_pdf_number_options_append_enables_enrichment():
    from zotpilot.cli import _with_formula_pdf_number_options

    config = SimpleNamespace()

    updated = _with_formula_pdf_number_options(
        config,
        cache_pdf_number_enrichment=False,
        append_missing_pdf_number_candidates=True,
    )

    assert updated is config
    assert config.formula_candidate_cache_pdf_number_enrichment is True
    assert config.formula_candidate_pdf_number_append_missing_candidates is True


def test_formula_pdf_number_options_preserves_config_append_for_dataclass():
    from zotpilot.cli import _with_formula_pdf_number_options

    @dataclass(frozen=True)
    class RuntimeConfig:
        formula_candidate_cache_pdf_number_enrichment: bool = False
        formula_candidate_pdf_number_append_missing_candidates: bool = True

    config = RuntimeConfig()

    updated = _with_formula_pdf_number_options(
        config,
        cache_pdf_number_enrichment=True,
        append_missing_pdf_number_candidates=False,
    )

    assert updated.formula_candidate_cache_pdf_number_enrichment is True
    assert updated.formula_candidate_pdf_number_append_missing_candidates is True


def test_read_formula_auto_candidate_item_keys_requires_current_route_schema(tmp_path):
    from zotpilot.cli import _read_formula_auto_candidate_item_keys

    estimate_path = tmp_path / "estimate.json"
    estimate_path.write_text(
        json.dumps(
            {
                "readonly_index_changed": False,
                "request_complete": True,
                "formula_quality_route_counts": {
                    "auto_candidate": 2,
                    "review_queue": 1,
                },
                "formula_quality_route_item_keys": {
                    "auto_candidate": ["DOC1", "DOC2"],
                    "review_queue": ["DOC3"],
                },
                "formula_quality_route_summary": [
                    {"item_key": "DOC1", "quality_route": "auto_candidate"},
                    {"item_key": "DOC2", "quality_route": "auto_candidate"},
                    {"item_key": "DOC3", "quality_route": "review_queue"},
                ],
                "formula_auto_candidate_item_keys": ["DOC1", "DOC2"],
            }
        ),
        encoding="utf-8",
    )

    assert _read_formula_auto_candidate_item_keys(str(estimate_path)) == ["DOC1", "DOC2"]


def test_read_formula_auto_candidate_item_keys_rejects_changed_index(tmp_path):
    from zotpilot.cli import _read_formula_auto_candidate_item_keys

    estimate_path = tmp_path / "estimate.json"
    estimate_path.write_text(
        json.dumps(
            {
                "readonly_index_changed": True,
                "request_complete": True,
                "formula_quality_route_counts": {"auto_candidate": 1},
                "formula_auto_candidate_item_keys": ["DOC1"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="readonly_index_changed"):
        _read_formula_auto_candidate_item_keys(str(estimate_path))


def test_estimate_formula_backfill_cli_ignores_simpletex_auth_for_read_only_estimate(capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    config = MagicMock()
    config.validate.return_value = ["SimpleTex formula OCR requires formula_ocr_simpletex_token"]
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 2,
        "candidate_count": 5,
        "average_candidates_per_paper": 2.5,
        "estimated_provider_calls": 5,
        "estimated_external_calls": 5,
        "estimated_min_duration": "2.5s",
        "daily_call_budget": 2,
        "estimated_runs": 3,
        "data_egress": True,
        "summary": {
            "next_action": "Run index_formulas with the same daily_call_budget.",
            "warnings": ["SimpleTex will send formula crops to the configured HTTPS endpoint."],
        },
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key="DOC1",
                item_keys=None,
                limit=10,
                resume_after="DOC0",
                daily_call_budget=2,
                preview_candidates=1,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=0,
                cache_pdf_number_enrichment=True,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                exclude_item_keys=None,
                fail_on_write_blocked=False,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Formula backfill estimate:" in out
    assert "Candidate provider:        mineru_cache" in out
    assert "Estimated runs:            3" in out
    assert config.formula_candidate_cache_pdf_number_enrichment is True
    indexer.estimate_formula_backfill.assert_called_once_with(
        item_key="DOC1",
        item_keys=None,
        limit=10,
        resume_after="DOC0",
        daily_call_budget=2,
        candidate_preview_limit=1,
        candidate_preview_chars=160,
        pdf_fallback_max_pages=0,
        page_min=None,
        page_max=None,
        sample_size=None,
        sample_seed=0,
        exclude_item_keys=None,
    )


def test_estimate_formula_backfill_cli_forwards_sample_exclude_keys(capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 2,
        "candidate_count": 2,
        "average_candidates_per_paper": 1.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 0,
        "estimated_runs": 1,
        "data_egress": False,
        "summary": {"next_action": "Review estimate.", "warnings": []},
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                limit=None,
                resume_after=None,
                daily_call_budget=0,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=2,
                sample_seed=11,
                exclude_item_keys=["DOC1", "DOC2"],
                exclude_item_keys_file=None,
                fail_on_write_blocked=False,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    assert rc == 0
    capsys.readouterr()
    indexer.estimate_formula_backfill.assert_called_once_with(
        item_key=None,
        item_keys=None,
        limit=None,
        resume_after=None,
        daily_call_budget=0,
        candidate_preview_limit=0,
        candidate_preview_chars=160,
        pdf_fallback_max_pages=None,
        page_min=None,
        page_max=None,
        sample_size=2,
        sample_seed=11,
        exclude_item_keys=["DOC1", "DOC2"],
    )


def test_estimate_formula_backfill_cli_forwards_include_high_density(capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 1,
        "average_candidates_per_paper": 1.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 0,
        "estimated_runs": 1,
        "data_egress": False,
        "summary": {"next_action": "Review estimate.", "warnings": []},
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                limit=None,
                resume_after=None,
                daily_call_budget=0,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=2,
                sample_seed=11,
                exclude_item_keys=None,
                exclude_item_keys_file=None,
                include_high_density=True,
                fail_on_write_blocked=False,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    assert rc == 0
    capsys.readouterr()
    assert indexer.estimate_formula_backfill.call_args.kwargs["include_high_density"] is True


def test_estimate_formula_backfill_cli_applies_candidate_provider_override(capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "local",
        "candidate_provider": "pdf_extract_kit_json",
        "processed": 1,
        "candidate_count": 1,
        "average_candidates_per_paper": 1.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 0,
        "estimated_runs": 1,
        "data_egress": False,
        "summary": {"next_action": "Review estimate.", "warnings": []},
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer) as estimate_factory,
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key="DOC1",
                item_keys=None,
                limit=None,
                resume_after=None,
                daily_call_budget=0,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                append_missing_pdf_number_candidates=False,
                candidate_provider="pdf_extract_kit_json",
                candidate_cache_dirs="F:/parser-cache",
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                exclude_item_keys=None,
                exclude_item_keys_file=None,
                include_high_density=False,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=False,
                fail_on_readonly_index_changed=False,
                json=False,
            )
        )

    assert rc == 0
    capsys.readouterr()
    assert config.formula_candidate_provider == "pdf_extract_kit_json"
    assert config.formula_candidate_cache_dirs == "F:/parser-cache"
    estimate_factory.assert_called_once_with(config)


def test_estimate_formula_backfill_cli_merges_exclude_key_file(tmp_path, capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    exclude_file = tmp_path / "tested-keys.txt"
    exclude_file.write_text("DOC2\nDOC3, DOC4\n# comment\nDOC2\n", encoding="utf-8")
    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 1,
        "average_candidates_per_paper": 1.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 0,
        "estimated_runs": 1,
        "data_egress": False,
        "summary": {"next_action": "Review estimate.", "warnings": []},
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                limit=None,
                resume_after=None,
                daily_call_budget=0,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=2,
                sample_seed=11,
                exclude_item_keys=["DOC1", "DOC3"],
                exclude_item_keys_file=str(exclude_file),
                fail_on_write_blocked=False,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    assert rc == 0
    capsys.readouterr()
    indexer.estimate_formula_backfill.assert_called_once_with(
        item_key=None,
        item_keys=None,
        limit=None,
        resume_after=None,
        daily_call_budget=0,
        candidate_preview_limit=0,
        candidate_preview_chars=160,
        pdf_fallback_max_pages=None,
        page_min=None,
        page_max=None,
        sample_size=2,
        sample_seed=11,
        exclude_item_keys=["DOC1", "DOC3", "DOC2", "DOC4"],
    )


def test_estimate_formula_backfill_cli_reads_json_exclude_key_file(tmp_path, capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    exclude_file = tmp_path / "tested-keys.json"
    exclude_file.write_text('["DOC2", "DOC3", "DOC2"]', encoding="utf-8")
    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 1,
        "average_candidates_per_paper": 1.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 0,
        "estimated_runs": 1,
        "data_egress": False,
        "summary": {"next_action": "Review estimate.", "warnings": []},
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                limit=None,
                resume_after=None,
                daily_call_budget=0,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=2,
                sample_seed=11,
                exclude_item_keys=None,
                exclude_item_keys_file=str(exclude_file),
                fail_on_write_blocked=False,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    assert rc == 0
    capsys.readouterr()
    assert indexer.estimate_formula_backfill.call_args.kwargs["exclude_item_keys"] == [
        "DOC2",
        "DOC3",
    ]


def test_estimate_formula_backfill_json_redirects_third_party_stdout_to_stderr(capfd):
    from zotpilot.cli import cmd_estimate_formula_backfill

    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()

    def noisy_estimate(**_kwargs):
        print("MinerU cache warning")
        os.write(1, b"MuPDF fd warning\n")
        return {
            "provider": "simpletex",
            "candidate_provider": "mineru_cache",
            "processed": 0,
            "candidate_count": 0,
            "summary": {"warnings": [], "next_action": "No matches."},
        }

    indexer.estimate_formula_backfill.side_effect = noisy_estimate

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                limit=None,
                resume_after=None,
                daily_call_budget=None,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=False,
                json=True,
            )
        )

    captured = capfd.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["provider"] == "simpletex"
    assert captured.out.lstrip().startswith("{")
    assert "MinerU cache warning" in captured.err
    assert "MuPDF fd warning" in captured.err


def test_estimate_formula_backfill_cli_can_fail_on_candidate_quality_blocked(capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 2,
        "candidate_count": 20,
        "average_candidates_per_paper": 10.0,
        "estimated_provider_calls": 12,
        "estimated_external_calls": 12,
        "estimated_min_duration": "6s",
        "daily_call_budget": 1800,
        "estimated_runs": 1,
        "data_egress": True,
        "candidate_quality_blocking_paper_count": 1,
        "candidate_quality_blocking_papers": [
            {
                "item_key": "DOC1",
                "candidate_count": 12,
                "review_reasons": ["cached_latex_low_quality"],
            }
        ],
        "summary": {
            "next_action": "Review candidate-stage formula quality warnings before running index_formulas.",
            "warnings": [],
        },
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                limit=None,
                resume_after=None,
                daily_call_budget=1800,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_candidate_quality_blocked=True,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 4
    assert "Candidate quality blocked: 1" in out
    assert "Review candidate-stage formula quality warnings before running index_formulas." in out


def test_estimate_formula_backfill_cli_json_can_fail_on_candidate_quality_blocked(capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 12,
        "candidate_quality_blocking_paper_count": 1,
        "candidate_quality_blocking_papers": [{"item_key": "DOC1"}],
        "summary": {"warnings": [], "next_action": "Review candidates."},
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                limit=None,
                resume_after=None,
                daily_call_budget=1800,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_candidate_quality_blocked=True,
                json=True,
            )
        )

    out = capsys.readouterr().out
    assert rc == 4
    assert json.loads(out)["candidate_quality_blocking_paper_count"] == 1


def test_estimate_formula_backfill_cli_can_fail_on_unmatched_requested_items(capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 2,
        "average_candidates_per_paper": 2.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 1800,
        "estimated_runs": 1,
        "data_egress": False,
        "unmatched_requested_item_keys": ["MISSING1"],
        "summary": {"warnings": [], "next_action": "Resolve unmatched requested item keys before writing."},
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=["DOC1", "MISSING1"],
                limit=None,
                resume_after=None,
                daily_call_budget=1800,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=True,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 5
    assert "Unmatched requested:      1" in out
    assert "Missing item keys:        MISSING1" in out


def test_estimate_formula_backfill_cli_json_can_fail_on_unmatched_requested_items(capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 2,
        "unmatched_requested_item_keys": ["MISSING1"],
        "summary": {"warnings": [], "next_action": "Resolve unmatched requested item keys."},
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=["DOC1", "MISSING1"],
                limit=None,
                resume_after=None,
                daily_call_budget=1800,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=True,
                json=True,
            )
        )

    out = capsys.readouterr().out
    assert rc == 5
    assert json.loads(out)["unmatched_requested_item_keys"] == ["MISSING1"]


def test_estimate_formula_backfill_cli_can_fail_on_readonly_index_change(capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    warning = (
        "The Chroma SQLite index changed during this read-only estimate; discard this "
        "batch as validation evidence and check for concurrent ZotPilot index writers."
    )
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 2,
        "average_candidates_per_paper": 2.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 1800,
        "estimated_runs": 1,
        "data_egress": False,
        "readonly_index_changed": True,
        "summary": {"warnings": [warning], "next_action": "Rerun after the index is stable."},
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=["DOC1"],
                limit=None,
                resume_after=None,
                daily_call_budget=1800,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_readonly_index_changed=True,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 6
    assert "changed during this read-only estimate" in out


def test_index_formulas_cli_passes_budget_resume_and_status_jsonl(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.index_formulas.return_value = {
        "provider": "simpletex",
        "processed": 1,
        "formulas_indexed": 2,
        "provider_calls_used": 2,
        "external_calls_used": 2,
        "daily_call_budget": 2,
        "daily_call_budget_remaining": 0,
        "write_blocked": True,
        "write_ready": False,
        "write_block_reasons": ["candidate_quality_review_required"],
        "formula_write_route_counts": {
            "indexed": 1,
            "review_queue": 1,
            "deferred": 1,
            "skipped": 0,
            "failed": 0,
            "no_formula": 0,
            "unknown": 0,
        },
        "next_action": "Review candidate-stage formula quality warnings before rerunning.",
        "stopped_reason": "daily_call_budget",
        "resume_cursor": "DOC1",
        "next_item_key": "DOC2",
        "state_path": str(tmp_path / "formula_backfill_state.jsonl"),
        "low_confidence_review_count": 1,
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease"),
        patch("zotpilot.index_authority.release_lease"),
        patch("zotpilot.indexer.Indexer", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=["DOC1", "DOC2"],
                limit=2,
                no_refresh_existing=False,
                daily_call_budget=2,
                resume_after="DOC0",
                no_stop_on_quota=False,
                status_jsonl="",
                low_confidence_threshold=0.7,
                include_high_density=True,
                allow_candidate_quality_warnings=True,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=True,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Formula backfill complete:" in out
    assert "Write status:            blocked" in out
    assert "Routes:                  indexed=1, review=1, deferred=1" in out
    assert "Next:                    Review candidate-stage formula quality warnings before rerunning." in out
    assert "Resume after:            DOC1" in out
    assert "Next item:               DOC2" in out
    assert config.formula_candidate_cache_pdf_number_enrichment is True
    indexer.index_formulas.assert_called_once_with(
        item_key=None,
        item_keys=["DOC1", "DOC2"],
        limit=2,
        refresh_existing=True,
        daily_call_budget=2,
        resume_after="DOC0",
        stop_on_quota=True,
        status_jsonl="",
        low_confidence_threshold=0.7,
        include_high_density=True,
        allow_candidate_quality_warnings=True,
        pdf_fallback_max_pages=None,
        page_min=None,
        page_max=None,
    )


def test_index_formulas_dry_run_allows_disabled_formula_ocr_for_readonly_estimate(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = ["SimpleTex formula OCR requires formula_ocr_simpletex_token"]
    config.formula_ocr_enabled = False
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 2,
        "average_candidates_per_paper": 2.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 0,
        "estimated_runs": 1,
        "data_egress": False,
        "summary": {"next_action": "Review candidates.", "warnings": []},
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease") as acquire_lease,
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key="DOC1",
                item_keys=None,
                auto_candidates_from_estimate=None,
                limit=None,
                all_indexed=False,
                dry_run=True,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                daily_call_budget=0,
                resume_after=None,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                append_missing_pdf_number_candidates=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                exclude_item_keys=None,
                exclude_item_keys_file=None,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                fail_on_write_blocked=False,
                fail_on_review_required=False,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=False,
                fail_on_readonly_index_changed=False,
                json=False,
            )
        )

    assert rc == 0
    assert "[dry-run] No formula chunks were written." in capsys.readouterr().out
    acquire_lease.assert_not_called()
    indexer.estimate_formula_backfill.assert_called_once()


def test_index_formulas_write_still_requires_formula_ocr_enabled(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = False
    config.chroma_db_path = tmp_path / "chroma"

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease") as acquire_lease,
        patch("zotpilot.indexer.Indexer") as indexer_cls,
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key="DOC1",
                item_keys=None,
                auto_candidates_from_estimate=None,
                limit=None,
                all_indexed=False,
                dry_run=False,
                no_refresh_existing=False,
                daily_call_budget=0,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=None,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                append_missing_pdf_number_candidates=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=False,
                fail_on_review_required=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    assert rc == 1
    assert "formula_ocr_enabled must be true" in capsys.readouterr().err
    acquire_lease.assert_not_called()
    indexer_cls.assert_not_called()


def test_index_formulas_cli_scopes_write_to_estimate_auto_candidates(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    estimate_path = tmp_path / "estimate.json"
    estimate_path.write_text(
        json.dumps(
            {
                "readonly_index_changed": False,
                "request_complete": True,
                "formula_quality_route_counts": {
                    "auto_candidate": 2,
                    "review_queue": 1,
                },
                "formula_quality_route_item_keys": {
                    "auto_candidate": ["DOC1", "DOC2"],
                    "review_queue": ["DOC3"],
                },
                "formula_quality_route_summary": [
                    {"item_key": "DOC1", "quality_route": "auto_candidate"},
                    {"item_key": "DOC2", "quality_route": "auto_candidate"},
                    {"item_key": "DOC3", "quality_route": "review_queue"},
                ],
                "formula_auto_candidate_item_keys": ["DOC1", "DOC2"],
            }
        ),
        encoding="utf-8",
    )
    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.index_formulas.return_value = {
        "provider": "local",
        "processed": 2,
        "formulas_indexed": 4,
        "provider_calls_used": 0,
        "external_calls_used": 0,
        "write_blocked": False,
        "write_ready": True,
        "write_review_required": False,
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease") as acquire_lease,
        patch("zotpilot.index_authority.release_lease"),
        patch("zotpilot.indexer.Indexer", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                auto_candidates_from_estimate=str(estimate_path),
                limit=None,
                all_indexed=False,
                no_refresh_existing=False,
                daily_call_budget=0,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=None,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                append_missing_pdf_number_candidates=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=False,
                fail_on_review_required=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    assert rc == 0
    assert "Formula backfill complete:" in capsys.readouterr().out
    acquire_lease.assert_called_once()
    indexer.index_formulas.assert_called_once_with(
        item_key=None,
        item_keys=["DOC1", "DOC2"],
        limit=None,
        refresh_existing=True,
        daily_call_budget=0,
        resume_after=None,
        stop_on_quota=True,
        status_jsonl=None,
        low_confidence_threshold=None,
        include_high_density=False,
        allow_candidate_quality_warnings=False,
        pdf_fallback_max_pages=None,
        page_min=None,
        page_max=None,
    )


def test_index_formulas_cli_rejects_unstable_auto_candidate_estimate(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    estimate_path = tmp_path / "estimate.json"
    estimate_path.write_text(
        json.dumps(
            {
                "readonly_index_changed": True,
                "request_complete": True,
                "formula_quality_route_counts": {"auto_candidate": 1},
                "formula_auto_candidate_item_keys": ["DOC1"],
            }
        ),
        encoding="utf-8",
    )
    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease") as acquire_lease,
        patch("zotpilot.indexer.Indexer") as indexer_cls,
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                auto_candidates_from_estimate=str(estimate_path),
                limit=None,
                all_indexed=False,
                no_refresh_existing=False,
                daily_call_budget=0,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=None,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                append_missing_pdf_number_candidates=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=False,
                fail_on_review_required=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    assert rc == 1
    assert "readonly_index_changed=true" in capsys.readouterr().err
    acquire_lease.assert_not_called()
    indexer_cls.assert_not_called()


def test_index_formulas_cli_refuses_unscoped_write_without_all_indexed(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease") as acquire_lease,
        patch("zotpilot.indexer.Indexer") as indexer_cls,
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                limit=None,
                all_indexed=False,
                no_refresh_existing=False,
                daily_call_budget=2,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=None,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                append_missing_pdf_number_candidates=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=False,
                fail_on_review_required=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    err = capsys.readouterr().err
    assert rc == 1
    assert "refusing unscoped formula write" in err
    acquire_lease.assert_not_called()
    indexer_cls.assert_not_called()


def test_index_formulas_cli_all_indexed_allows_intentional_unscoped_write(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.index_formulas.return_value = {
        "provider": "local",
        "processed": 1,
        "formulas_indexed": 1,
        "provider_calls_used": 0,
        "external_calls_used": 0,
        "write_blocked": False,
        "write_ready": True,
        "write_review_required": False,
        "next_action": "Formula chunks were written; verify sampled formula search results before scaling up.",
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease") as acquire_lease,
        patch("zotpilot.index_authority.release_lease"),
        patch("zotpilot.indexer.Indexer", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                limit=None,
                all_indexed=True,
                no_refresh_existing=False,
                daily_call_budget=0,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=None,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                append_missing_pdf_number_candidates=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=False,
                fail_on_review_required=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Formula backfill complete:" in out
    acquire_lease.assert_called_once()
    indexer.index_formulas.assert_called_once_with(
        item_key=None,
        item_keys=None,
        limit=None,
        refresh_existing=True,
        daily_call_budget=0,
        resume_after=None,
        stop_on_quota=True,
        status_jsonl=None,
        low_confidence_threshold=None,
        include_high_density=False,
        allow_candidate_quality_warnings=False,
        pdf_fallback_max_pages=None,
        page_min=None,
        page_max=None,
    )


def test_index_formulas_cli_shows_review_required_writes(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.index_formulas.return_value = {
        "provider": "simpletex",
        "processed": 1,
        "formulas_indexed": 1,
        "provider_calls_used": 1,
        "external_calls_used": 1,
        "write_blocked": False,
        "write_ready": True,
        "write_review_required": True,
        "write_block_reasons": [],
        "next_action": "Review 1 low-confidence formula row(s) before scaling up.",
        "low_confidence_review_count": 1,
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease"),
        patch("zotpilot.index_authority.release_lease"),
        patch("zotpilot.indexer.Indexer", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key="DOC1",
                item_keys=None,
                limit=None,
                no_refresh_existing=False,
                daily_call_budget=2,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=0.7,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=True,
                fail_on_review_required=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Write status:            ready" in out
    assert "Review required:         yes" in out
    assert "Next:                    Review 1 low-confidence formula row(s) before scaling up." in out


def test_index_formulas_cli_can_fail_on_unmatched_requested_items(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.index_formulas.return_value = {
        "provider": "simpletex",
        "processed": 1,
        "formulas_indexed": 1,
        "provider_calls_used": 0,
        "external_calls_used": 0,
        "write_blocked": False,
        "write_ready": True,
        "write_review_required": False,
        "unmatched_requested_item_key_count": 1,
        "unmatched_requested_item_keys": ["MISSING1"],
        "next_action": "Resolve unmatched requested item keys before scaling up.",
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease"),
        patch("zotpilot.index_authority.release_lease"),
        patch("zotpilot.indexer.Indexer", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=["DOC1", "MISSING1"],
                limit=None,
                no_refresh_existing=False,
                daily_call_budget=2,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=None,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=True,
                fail_on_review_required=True,
                fail_on_unmatched=True,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 5
    assert "Unmatched requested:     1" in out
    assert "Missing item keys:       MISSING1" in out


def test_index_formulas_cli_can_fail_when_review_required(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.index_formulas.return_value = {
        "provider": "simpletex",
        "processed": 1,
        "formulas_indexed": 1,
        "provider_calls_used": 1,
        "external_calls_used": 1,
        "write_blocked": False,
        "write_ready": True,
        "write_review_required": True,
        "write_block_reasons": [],
        "next_action": "Review 1 low-confidence formula row(s) before scaling up.",
        "low_confidence_review_count": 1,
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease"),
        patch("zotpilot.index_authority.release_lease"),
        patch("zotpilot.indexer.Indexer", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key="DOC1",
                item_keys=None,
                limit=None,
                no_refresh_existing=False,
                daily_call_budget=2,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=0.7,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=True,
                fail_on_review_required=True,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 3
    assert "Write status:            ready" in out
    assert "Review required:         yes" in out
    assert "Next:                    Review 1 low-confidence formula row(s) before scaling up." in out


def test_index_formulas_cli_can_fail_when_write_blocked(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.index_formulas.return_value = {
        "provider": "simpletex",
        "processed": 1,
        "formulas_indexed": 0,
        "provider_calls_used": 0,
        "external_calls_used": 0,
        "write_blocked": True,
        "write_ready": False,
        "write_block_reasons": ["candidate_quality_review_required"],
        "next_action": "Review candidate-stage formula quality warnings before rerunning.",
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease"),
        patch("zotpilot.index_authority.release_lease"),
        patch("zotpilot.indexer.Indexer", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=["DOC1"],
                limit=None,
                no_refresh_existing=False,
                daily_call_budget=2,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=None,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=True,
                fail_on_review_required=True,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 2
    assert "Write status:            blocked" in out
    assert "Next:                    Review candidate-stage formula quality warnings before rerunning." in out


def test_index_formulas_cli_prioritizes_write_blocked_exit_code(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.index_formulas.return_value = {
        "provider": "simpletex",
        "processed": 1,
        "formulas_indexed": 0,
        "provider_calls_used": 0,
        "external_calls_used": 0,
        "write_blocked": True,
        "write_ready": False,
        "write_review_required": True,
        "write_block_reasons": ["candidate_quality_review_required"],
        "next_action": "Review candidate-stage formula quality warnings before rerunning.",
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease"),
        patch("zotpilot.index_authority.release_lease"),
        patch("zotpilot.indexer.Indexer", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=["DOC1"],
                limit=None,
                no_refresh_existing=False,
                daily_call_budget=2,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=None,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=True,
                fail_on_review_required=True,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 2
    assert "Write status:            blocked" in out
    assert "Review required:         yes" in out


def test_index_formulas_cli_json_can_fail_when_write_blocked(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.index_formulas.return_value = {
        "provider": "simpletex",
        "processed": 1,
        "formulas_indexed": 0,
        "write_blocked": True,
        "write_ready": False,
        "write_block_reasons": ["candidate_quality_review_required"],
        "next_action": "Review candidate-stage formula quality warnings before rerunning.",
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease"),
        patch("zotpilot.index_authority.release_lease"),
        patch("zotpilot.indexer.Indexer", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=["DOC1"],
                limit=None,
                no_refresh_existing=False,
                daily_call_budget=2,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=None,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=True,
                fail_on_review_required=True,
                json=True,
            )
        )

    out = capsys.readouterr().out
    assert rc == 2
    assert '"write_blocked": true' in out
    assert '"write_block_reasons": [' in out


def test_index_formulas_cli_json_can_fail_when_review_required(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = []
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.index_formulas.return_value = {
        "provider": "simpletex",
        "processed": 1,
        "formulas_indexed": 1,
        "write_blocked": False,
        "write_ready": True,
        "write_review_required": True,
        "write_block_reasons": [],
        "next_action": "Review 1 low-confidence formula row(s) before scaling up.",
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease"),
        patch("zotpilot.index_authority.release_lease"),
        patch("zotpilot.indexer.Indexer", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=["DOC1"],
                limit=None,
                no_refresh_existing=False,
                daily_call_budget=2,
                resume_after=None,
                no_stop_on_quota=False,
                status_jsonl=None,
                low_confidence_threshold=None,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                fail_on_write_blocked=True,
                fail_on_review_required=True,
                json=True,
            )
        )

    out = capsys.readouterr().out
    assert rc == 3
    assert '"write_review_required": true' in out
    assert '"write_blocked": false' in out


def test_index_formulas_cli_dry_run_uses_estimate_without_lease(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = ["SimpleTex formula OCR requires formula_ocr_simpletex_token"]
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 1,
        "average_candidates_per_paper": 1.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 1800,
        "estimated_runs": 1,
        "data_egress": True,
        "write_blocked": False,
        "write_review_required": False,
        "summary": {"next_action": "Review candidates.", "warnings": []},
        "results": [
            {
                "item_key": "DOC1",
                "candidate_preview": [
                    {
                        "page_num": 1,
                        "source": "mineru_markdown",
                        "confidence": 0.9,
                        "equation_number": "(1)",
                        "has_latex": True,
                        "needs_ocr": False,
                        "latex_preview": r"E = mc^2",
                    }
                ],
            }
        ],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease") as acquire_lease,
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key="DOC1",
                item_keys=None,
                limit=1,
                dry_run=True,
                preview_candidates=1,
                preview_all_candidates=False,
                preview_chars=160,
                daily_call_budget=1800,
                resume_after=None,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=True,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                include_high_density=True,
                allow_candidate_quality_warnings=True,
                fail_on_write_blocked=True,
                fail_on_review_required=True,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "[dry-run] No formula chunks were written." in out
    assert "Candidate provider:        mineru_cache" in out
    assert "p1 (1) mineru_markdown cached" in out
    assert "E = mc^2" in out
    assert config.formula_candidate_cache_pdf_number_enrichment is True
    acquire_lease.assert_not_called()
    indexer.index_formulas.assert_not_called()
    indexer.estimate_formula_backfill.assert_called_once_with(
        item_key="DOC1",
        item_keys=None,
        limit=1,
        resume_after=None,
        daily_call_budget=1800,
        candidate_preview_limit=1,
        candidate_preview_chars=160,
        pdf_fallback_max_pages=None,
        page_min=None,
        page_max=None,
        sample_size=None,
        sample_seed=0,
        exclude_item_keys=None,
        include_high_density=True,
        allow_candidate_quality_warnings=True,
    )


def test_index_formulas_dry_run_cli_can_fail_on_write_blocked(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = ["SimpleTex formula OCR requires formula_ocr_simpletex_token"]
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 12,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "daily_call_budget": 1800,
        "data_egress": False,
        "write_blocked": True,
        "write_review_required": False,
        "summary": {"next_action": "Review high-density documents first.", "warnings": []},
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key=None,
                item_keys=None,
                limit=None,
                dry_run=True,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                daily_call_budget=1800,
                resume_after=None,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                fail_on_write_blocked=True,
                fail_on_review_required=True,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    assert rc == 2


def test_index_formulas_dry_run_cli_can_fail_on_review_required(tmp_path, capsys):
    from zotpilot.cli import cmd_index_formulas

    config = MagicMock()
    config.validate.return_value = ["SimpleTex formula OCR requires formula_ocr_simpletex_token"]
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 2,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "daily_call_budget": 1800,
        "data_egress": False,
        "write_blocked": False,
        "write_review_required": True,
        "summary": {"next_action": "Review sampled formulas after writing.", "warnings": []},
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_index_formulas(
            SimpleNamespace(
                config="config.json",
                item_key="DOC1",
                item_keys=None,
                limit=None,
                dry_run=True,
                preview_candidates=0,
                preview_all_candidates=False,
                preview_chars=160,
                daily_call_budget=1800,
                resume_after=None,
                pdf_fallback_max_pages=None,
                cache_pdf_number_enrichment=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                include_high_density=False,
                allow_candidate_quality_warnings=False,
                fail_on_write_blocked=True,
                fail_on_review_required=True,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=False,
                json=False,
            )
        )

    assert rc == 3


def test_index_formulas_parser_rejects_item_key_and_item_keys_together():
    from zotpilot.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["index-formulas", "--dry-run", "--item-key", "DOC1", "--item-keys", "DOC2"])

    assert exc.value.code == 2


def test_index_formulas_parser_rejects_item_key_and_estimate_scope_together():
    from zotpilot.cli import main

    with pytest.raises(SystemExit) as exc:
        main([
            "index-formulas",
            "--dry-run",
            "--item-key",
            "DOC1",
            "--auto-candidates-from-estimate",
            "estimate.json",
        ])

    assert exc.value.code == 2


def test_index_formulas_parser_rejects_page_window_without_single_item_scope():
    from zotpilot.cli import main

    with patch("zotpilot.cli.resolve_runtime_config", side_effect=AssertionError("config should not load")):
        with pytest.raises(SystemExit) as exc:
            main(["index-formulas", "--dry-run", "--item-keys", "DOC1", "DOC2", "--page-min", "4"])

    assert exc.value.code == 2


def test_index_formulas_parser_rejects_negative_preview_chars():
    from zotpilot.cli import main

    with patch("zotpilot.cli.resolve_runtime_config", side_effect=AssertionError("config should not load")):
        with pytest.raises(SystemExit) as exc:
            main(["index-formulas", "--dry-run", "--item-key", "DOC1", "--preview-chars", "-1"])

    assert exc.value.code == 2


def test_index_formulas_dry_run_cli_can_fail_on_candidate_quality_blocked(tmp_path, capsys):
    from zotpilot.cli import main

    config = MagicMock()
    config.validate.return_value = ["SimpleTex formula OCR requires formula_ocr_simpletex_token"]
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 12,
        "candidate_quality_blocking_paper_count": 1,
        "average_candidates_per_paper": 12.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 1800,
        "estimated_runs": 1,
        "data_egress": False,
        "summary": {
            "next_action": "Review candidate-stage formula quality warnings before writing formulas.",
            "warnings": [],
        },
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease") as acquire_lease,
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = main(
            [
                "index-formulas",
                "--dry-run",
                "--item-key",
                "DOC1",
                "--fail-on-candidate-quality-blocked",
            ]
        )

    out = capsys.readouterr().out
    assert rc == 4
    assert "[dry-run] No formula chunks were written." in out
    assert "Candidate quality blocked: 1" in out
    assert "Next: Review candidate-stage formula quality warnings before writing formulas." in out
    acquire_lease.assert_not_called()


def test_index_formulas_dry_run_cli_can_fail_on_readonly_index_change(tmp_path, capsys):
    from zotpilot.cli import main

    config = MagicMock()
    config.validate.return_value = ["SimpleTex formula OCR requires formula_ocr_simpletex_token"]
    config.formula_ocr_enabled = True
    config.chroma_db_path = tmp_path / "chroma"
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "mineru_cache",
        "processed": 1,
        "candidate_count": 0,
        "average_candidates_per_paper": 0.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 0,
        "estimated_runs": 1,
        "data_egress": False,
        "readonly_index_changed": True,
        "summary": {
            "next_action": "Rerun after the index is stable.",
            "warnings": ["The Chroma SQLite index changed during this read-only estimate."],
        },
        "results": [],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.index_authority.acquire_lease") as acquire_lease,
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = main(
            [
                "index-formulas",
                "--dry-run",
                "--item-key",
                "DOC1",
                "--fail-on-readonly-index-changed",
            ]
        )

    out = capsys.readouterr().out
    assert rc == 6
    assert "[dry-run] No formula chunks were written." in out
    assert "changed during this read-only estimate" in out
    acquire_lease.assert_not_called()


def test_estimate_formula_backfill_cli_can_export_all_candidate_preview(capsys):
    from zotpilot.cli import cmd_estimate_formula_backfill

    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "auto",
        "processed": 1,
        "candidate_count": 2,
        "average_candidates_per_paper": 2.0,
        "estimated_provider_calls": 0,
        "estimated_external_calls": 0,
        "estimated_min_duration": "0s",
        "daily_call_budget": 1800,
        "estimated_runs": 1,
        "data_egress": True,
        "summary": {"next_action": "Review candidates.", "warnings": []},
        "results": [{"item_key": "DOC1", "candidate_preview": []}],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key="DOC1",
                item_keys=None,
                limit=None,
                resume_after=None,
                daily_call_budget=1800,
                preview_candidates=0,
                preview_all_candidates=True,
                preview_chars=0,
                pdf_fallback_max_pages=0,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                json=True,
            )
        )

    assert rc == 0
    indexer.estimate_formula_backfill.assert_called_once_with(
        item_key="DOC1",
        item_keys=None,
        limit=None,
        resume_after=None,
        daily_call_budget=1800,
        candidate_preview_limit=-1,
        candidate_preview_chars=0,
        pdf_fallback_max_pages=0,
        page_min=None,
        page_max=None,
        sample_size=None,
        sample_seed=0,
        exclude_item_keys=None,
    )


def test_audit_formula_candidates_cli_outputs_provider_cross_review_json(capsys):
    from zotpilot.cli import cmd_audit_formula_candidates

    config = MagicMock()
    config.validate.return_value = ["SimpleTex formula OCR requires formula_ocr_simpletex_token"]
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = {
        "provider": "simpletex",
        "candidate_provider": "auto",
        "processed": 1,
        "candidate_count": 1,
        "estimated_provider_calls": 1,
        "estimated_external_calls": 1,
        "daily_call_budget": 10,
        "data_egress": True,
        "readonly_index_changed": False,
        "summary": {"next_action": "Review provider evidence.", "warnings": []},
        "results": [
            {
                "item_key": "DOC1",
                "title": "Paper",
                "candidate_count": 1,
                "estimated_external_calls": 1,
                "candidate_audit": {
                    "candidate_count": 1,
                    "source_counts": {"mineru_content_list": 1},
                    "ocr_needed_count": 1,
                    "cached_latex_count": 0,
                    "page_min": 1,
                    "page_max": 1,
                    "page_count_with_candidates": 1,
                    "page_tagged_count": 1,
                    "page_missing_count": 0,
                    "bbox_present_count": 1,
                    "bbox_missing_count": 0,
                    "numbered_count": 1,
                    "unnumbered_count": 0,
                    "first_equation_number": "(1)",
                    "last_equation_number": "(1)",
                    "equation_number_warnings": [],
                },
                "candidate_preview": [
                    {
                        "candidate_index": 0,
                        "page_num": 1,
                        "source": "mineru_content_list",
                        "equation_number": "(1)",
                        "bbox": [1, 2, 3, 4],
                        "latex_preview": r"E = mc^2",
                        "has_latex": True,
                        "needs_ocr": False,
                    }
                ],
            }
        ],
        "formula_quality_route_summary": [
            {
                "item_key": "DOC1",
                "quality_route": "auto_candidate",
                "route_reason": "candidate_quality_clear",
            }
        ],
    }

    with (
        patch("zotpilot.cli.resolve_runtime_config", return_value=config),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", return_value=indexer),
    ):
        rc = cmd_audit_formula_candidates(
            SimpleNamespace(
                config="config.json",
                item_key="DOC1",
                item_keys=None,
                limit=10,
                resume_after="DOC0",
                daily_call_budget=10,
                preview_candidates=1,
                preview_all_candidates=False,
                preview_chars=160,
                pdf_fallback_max_pages=0,
                cache_pdf_number_enrichment=True,
                append_missing_pdf_number_candidates=False,
                page_min=None,
                page_max=None,
                sample_size=None,
                sample_seed=0,
                exclude_item_keys=None,
                exclude_item_keys_file=None,
                include_high_density=False,
                fail_on_candidate_quality_blocked=False,
                fail_on_unmatched=False,
                fail_on_readonly_index_changed=False,
                json=True,
            )
        )

    payload = json.loads(capsys.readouterr().out)
    review = payload["provider_cross_review"]
    assert rc == 0
    assert payload["mode"] == "read_only_formula_candidate_audit"
    assert review["simpletex_role"] == "fallback_recognizer_only"
    assert review["provider_group_totals"] == {"mineru_cache": 1}
    assert review["candidate_previewed_paper_count"] == 1
    assert review["candidate_consensus_cluster_count"] == 1
    assert review["rows"][0]["simpletex_external_call_count"] == 1
    assert review["rows"][0]["candidate_preview_coverage"] == "complete"
    assert config.formula_candidate_cache_pdf_number_enrichment is True
    indexer.estimate_formula_backfill.assert_called_once_with(
        item_key="DOC1",
        item_keys=None,
        limit=10,
        resume_after="DOC0",
        daily_call_budget=10,
        candidate_preview_limit=1,
        candidate_preview_chars=160,
        pdf_fallback_max_pages=0,
        page_min=None,
        page_max=None,
        sample_size=None,
        sample_seed=0,
        exclude_item_keys=None,
        include_high_density=False,
    )


def test_audit_formula_candidates_parser_rejects_page_window_without_single_item_scope():
    from zotpilot.cli import main

    with patch("zotpilot.cli.resolve_runtime_config", side_effect=AssertionError("config should not load")):
        with pytest.raises(SystemExit) as exc:
            main(["audit-formula-candidates", "--item-keys", "DOC1", "DOC2", "--page-min", "4"])

    assert exc.value.code == 2


def test_compare_formula_parsers_cli_outputs_json_without_loading_config(tmp_path, capsys):
    from zotpilot.cli import main

    mineru_path = tmp_path / "mineru.json"
    pdf_extract_kit_path = tmp_path / "pdf_extract_kit.json"
    mineru_path.write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "results": [
                    {
                        "item_key": "DOC1",
                        "title": "Paper",
                        "candidate_count": 1,
                        "candidate_audit": {"source_counts": {"mineru_content_list": 1}},
                        "candidate_preview": [
                            {
                                "candidate_index": 0,
                                "page_num": 1,
                                "source": "mineru_content_list",
                                "equation_number": "(1)",
                                "bbox": [10, 20, 300, 48],
                                "latex_preview": r"E = mc^2",
                                "has_latex": True,
                                "needs_ocr": False,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    pdf_extract_kit_path.write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "results": [
                    {
                        "item_key": "DOC1",
                        "title": "Paper",
                        "candidate_count": 1,
                        "candidate_audit": {"source_counts": {"pdf_extract_kit_formula_recognition": 1}},
                        "candidate_preview": [
                            {
                                "candidate_index": 0,
                                "page_num": 1,
                                "source": "pdf_extract_kit_formula_recognition",
                                "equation_number": "(1)",
                                "bbox": [11, 20, 299, 49],
                                "latex_preview": r"E = mc^2",
                                "has_latex": True,
                                "needs_ocr": False,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with patch("zotpilot.cli.resolve_runtime_config", side_effect=AssertionError("config should not load")):
        rc = main(
            [
                "compare-formula-parsers",
                "--estimate",
                f"mineru={mineru_path}",
                "--estimate",
                f"pdf_extract_kit={pdf_extract_kit_path}",
                "--json",
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["mode"] == "read_only_external_parser_comparison"
    assert payload["parser_labels"] == ["mineru", "pdf_extract_kit"]
    assert payload["multi_provider_cluster_count"] == 1
    assert payload["rows"][0]["write_recommendation"] == "candidate_supported_by_cross_parser_review"


def test_compare_formula_parsers_cli_writes_output_file(tmp_path, capsys):
    from zotpilot.cli import main

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    output_path = tmp_path / "reports" / "comparison.json"
    estimate = {
        "candidate_count": 1,
        "results": [
            {
                "item_key": "DOC1",
                "title": "Paper",
                "candidate_count": 1,
                "candidate_audit": {"source_counts": {"mineru_content_list": 1}},
                "candidate_preview": [
                    {
                        "candidate_index": 0,
                        "page_num": 1,
                        "source": "mineru_content_list",
                        "equation_number": "(1)",
                        "bbox": [10, 20, 300, 48],
                        "latex_preview": r"E = mc^2",
                        "has_latex": True,
                        "needs_ocr": False,
                    }
                ],
            }
        ],
    }
    first_path.write_text(json.dumps(estimate), encoding="utf-8")
    second_path.write_text(json.dumps(estimate), encoding="utf-8")

    rc = main(
        [
            "compare-formula-parsers",
            "--estimate",
            f"first={first_path}",
            "--estimate",
            f"second={second_path}",
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "Formula external parser comparison:" in output
    assert "Write recommendations:" in output
    assert "candidate_supported_by_cross_parser_review: 1" in output
    assert "Parser summaries:" in output
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "read_only_external_parser_comparison"
    assert saved["parser_labels"] == ["first", "second"]
    assert saved["write_recommendation_counts"] == {
        "candidate_supported_by_cross_parser_review": 1
    }


def test_compare_formula_parsers_cli_can_fail_on_conflicts(tmp_path, capsys):
    from zotpilot.cli import main

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    base_row = {
        "item_key": "DOC1",
        "title": "Paper",
        "candidate_count": 1,
        "candidate_audit": {"source_counts": {"mineru_content_list": 1}},
    }
    first_path.write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "results": [
                    {
                        **base_row,
                        "candidate_preview": [
                            {
                                "candidate_index": 0,
                                "page_num": 1,
                                "source": "mineru_content_list",
                                "equation_number": "(1)",
                                "bbox": [10, 20, 300, 48],
                                "latex_preview": r"E = mc^2",
                                "has_latex": True,
                                "needs_ocr": False,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "results": [
                    {
                        **base_row,
                        "candidate_preview": [
                            {
                                "candidate_index": 0,
                                "page_num": 1,
                                "source": "pdf_extract_kit_formula_detection",
                                "equation_number": "(2)",
                                "bbox": [11, 20, 299, 49],
                                "raw_text_preview": r"E = mc^2",
                                "has_latex": False,
                                "needs_ocr": True,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "compare-formula-parsers",
            "--estimate",
            f"first={first_path}",
            "--estimate",
            f"second={second_path}",
            "--fail-on-conflicts",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 7
    assert payload["conflict_cluster_count"] == 1


def test_compare_formula_parsers_cli_treats_partial_review_as_manual_failure(tmp_path, capsys):
    from zotpilot.cli import main

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "results": [
                    {
                        "item_key": "DOC1",
                        "title": "Paper",
                        "candidate_count": 1,
                        "candidate_audit": {"source_counts": {"mineru_content_list": 1}},
                        "candidate_preview": [
                            {
                                "candidate_index": 0,
                                "page_num": 1,
                                "source": "mineru_content_list",
                                "equation_number": "(1)",
                                "bbox": [10, 20, 300, 48],
                                "latex_preview": r"\sigma = E\varepsilon",
                                "has_latex": True,
                                "needs_ocr": False,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps(
            {
                "candidate_count": 2,
                "results": [
                    {
                        "item_key": "DOC1",
                        "title": "Paper",
                        "candidate_count": 2,
                        "candidate_audit": {"source_counts": {"text_layer": 2}},
                        "candidate_preview": [
                            {
                                "candidate_index": 0,
                                "page_num": 1,
                                "source": "text_layer",
                                "equation_number": "(1)",
                                "bbox": [12, 20, 299, 49],
                                "raw_text_preview": "σ = E ε",
                                "has_latex": False,
                                "needs_ocr": True,
                            },
                            {
                                "candidate_index": 1,
                                "page_num": 2,
                                "source": "text_layer",
                                "equation_number": "(2)",
                                "bbox": [12, 60, 299, 88],
                                "raw_text_preview": "D = 1 - exp(-a epsilon)",
                                "has_latex": False,
                                "needs_ocr": True,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "compare-formula-parsers",
            "--estimate",
            f"first={first_path}",
            "--estimate",
            f"second={second_path}",
            "--fail-on-manual-review",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 8
    assert payload["manual_review_paper_count"] == 0
    assert payload["partial_review_paper_count"] == 1
    assert payload["review_required_paper_count"] == 1


def test_compare_formula_parsers_cli_can_fail_on_readonly_index_change(tmp_path, capsys):
    from zotpilot.cli import main

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(
        json.dumps({"candidate_count": 0, "readonly_index_changed": False, "results": []}),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps({"candidate_count": 0, "readonly_index_changed": True, "results": []}),
        encoding="utf-8",
    )

    rc = main(
        [
            "compare-formula-parsers",
            "--estimate",
            f"first={first_path}",
            "--estimate",
            f"second={second_path}",
            "--fail-on-readonly-index-changed",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 6
    assert payload["readonly_index_changed"] is True
    assert payload["readonly_index_changed_by_parser"] == {
        "first": False,
        "second": True,
    }


def test_compare_formula_parsers_cli_can_run_candidate_provider_estimates(capsys):
    from zotpilot.cli import main

    configs = []
    indexers = []

    def make_config():
        config = MagicMock()
        config.validate.return_value = []
        config.formula_candidate_provider = "auto"
        config.formula_candidate_cache_dirs = ""
        configs.append(config)
        return config

    def estimate_factory(config):
        indexer = MagicMock()
        provider = config.formula_candidate_provider
        source = (
            "mineru_content_list"
            if provider == "mineru_json"
            else "pdf_extract_kit_formula_recognition"
        )
        indexer.estimate_formula_backfill.return_value = {
            "provider": "local",
            "candidate_provider": provider,
            "candidate_count": 1,
            "results": [
                {
                    "item_key": "DOC1",
                    "title": "Paper",
                    "candidate_count": 1,
                    "candidate_audit": {"source_counts": {source: 1}},
                    "candidate_preview": [
                        {
                            "candidate_index": 0,
                            "page_num": 1,
                            "source": source,
                            "equation_number": "(1)",
                            "bbox": [10, 20, 300, 48],
                            "latex_preview": r"E = mc^2",
                            "has_latex": True,
                            "needs_ocr": False,
                        }
                    ],
                }
            ],
        }
        indexers.append(indexer)
        return indexer

    with (
        patch("zotpilot.cli.resolve_runtime_config", side_effect=[make_config(), make_config()]),
        patch("zotpilot.indexer.Indexer.for_formula_estimate", side_effect=estimate_factory),
    ):
        rc = main(
            [
                "compare-formula-parsers",
                "--candidate-provider",
                "mineru=mineru_json",
                "--candidate-provider",
                "pdfkit=pdf_extract_kit_json",
                "--candidate-cache-dirs",
                "F:/parser-cache",
                "--item-key",
                "DOC1",
                "--json",
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["parser_labels"] == ["mineru", "pdfkit"]
    assert payload["multi_provider_cluster_count"] == 1
    assert set(payload["estimate_runtime_seconds_by_parser"]) == {"mineru", "pdfkit"}
    assert payload["estimate_runtime_seconds_total"] >= 0
    assert configs[0].formula_candidate_provider == "mineru_json"
    assert configs[1].formula_candidate_provider == "pdf_extract_kit_json"
    assert configs[0].formula_candidate_cache_dirs == "F:/parser-cache"
    assert configs[1].formula_candidate_cache_dirs == "F:/parser-cache"
    assert indexers[0].estimate_formula_backfill.call_args.kwargs["candidate_preview_limit"] == -1
    assert indexers[1].estimate_formula_backfill.call_args.kwargs["candidate_preview_limit"] == -1
