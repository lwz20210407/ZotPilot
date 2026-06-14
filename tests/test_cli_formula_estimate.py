from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from zotpilot.cli import cmd_estimate_formula_backfill


def _estimate_result() -> dict:
    return {
        "provider": "simpletex",
        "processed": 2,
        "candidate_count": 5,
        "average_candidates_per_paper": 2.5,
        "estimated_provider_calls": 5,
        "estimated_external_calls": 5,
        "estimated_min_duration": "2.8s",
        "data_egress": True,
        "summary": {
            "next_action": "SimpleTex is configured; review the external-call estimate.",
            "warnings": ["SimpleTex will send formula crops to the configured HTTPS endpoint."],
        },
    }


def test_estimate_formula_backfill_cli_prints_readable_summary(capsys):
    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = _estimate_result()

    with patch("zotpilot.cli.resolve_runtime_config", return_value=config), \
         patch("zotpilot.indexer.Indexer", return_value=indexer):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config="config.json",
                item_key="DOC1",
                item_keys=None,
                limit=2,
                json=False,
            )
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "Formula backfill estimate:" in out
    assert "Provider:                  simpletex" in out
    assert "Estimated external calls:  5" in out
    assert "Data egress:               yes" in out
    assert "SimpleTex will send formula crops" in out
    indexer.estimate_formula_backfill.assert_called_once_with(
        item_key="DOC1",
        item_keys=None,
        limit=2,
    )


def test_estimate_formula_backfill_cli_json_output(capsys):
    config = MagicMock()
    config.validate.return_value = []
    indexer = MagicMock()
    indexer.estimate_formula_backfill.return_value = _estimate_result()

    with patch("zotpilot.cli.resolve_runtime_config", return_value=config), \
         patch("zotpilot.indexer.Indexer", return_value=indexer):
        rc = cmd_estimate_formula_backfill(
            SimpleNamespace(
                config=None,
                item_key=None,
                item_keys=["DOC1", "DOC2"],
                limit=None,
                json=True,
            )
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert '"provider": "simpletex"' in out
    assert '"estimated_external_calls": 5' in out
    indexer.estimate_formula_backfill.assert_called_once_with(
        item_key=None,
        item_keys=["DOC1", "DOC2"],
        limit=None,
    )
