import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from zotpilot.feature_extraction.formula_ocr import FormulaCandidate
from zotpilot.indexer import Indexer
from zotpilot.models import (
    Chunk,
    ExtractedFigure,
    ExtractedFormula,
    ExtractedTable,
    ZoteroItem,
)
from zotpilot.vector_store import VectorStore


def _formula_config(chroma_path):
    return SimpleNamespace(
        chunk_size=400,
        chunk_overlap=100,
        embedding_provider="local",
        dashscope_embedding_endpoint="compatible",
        embedding_dimensions=384,
        embedding_model="local",
        ocr_language="eng",
        vision_enabled=False,
        vision_provider="anthropic",
        vision_model="",
        zotero_data_dir=chroma_path.parent,
        chroma_db_path=chroma_path,
        formula_ocr_enabled=True,
        formula_ocr_provider="local",
        formula_candidate_provider="text_layer",
        formula_candidate_pdf_fallback_max_pages=0,
        formula_ocr_daily_call_budget=0,
        formula_ocr_max_formulas_per_doc=40,
        formula_ocr_max_formulas_per_page=6,
        formula_ocr_min_confidence=0.6,
        formula_ocr_low_confidence_threshold=0.0,
        formula_ocr_high_density_call_threshold=80,
        formula_ocr_high_density_candidate_threshold=160,
    )


def _chunk_ids_by_type(store: VectorStore, doc_id: str, chunk_type: str) -> list[str]:
    result = store.collection.get(
        where={
            "$and": [
                {"doc_id": {"$eq": doc_id}},
                {"chunk_type": {"$eq": chunk_type}},
            ]
        },
        include=[],
    )
    return sorted(result["ids"])


def _indexed_item(tmp_path, doc_id: str = "DOC1") -> ZoteroItem:
    pdf_path = tmp_path / f"{doc_id}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
    return ZoteroItem(
        item_key=doc_id,
        title="Formula write safety paper",
        authors="Author",
        year=2026,
        pdf_path=pdf_path,
        citation_key="author2026",
        publication="Journal of Test Safety",
        doi="10.0000/test",
        tags="formula; safety",
        collections="Regression",
    )


def _doc_meta(item: ZoteroItem) -> dict:
    return {
        "title": item.title,
        "authors": item.authors,
        "year": item.year,
        "citation_key": item.citation_key,
        "publication": item.publication,
        "doi": item.doi,
        "tags": item.tags,
        "collections": item.collections,
        "journal_quartile": "Q1",
        "pdf_hash": "pdf-hash",
        "quality_grade": "A",
    }


def _add_existing_non_formula_chunks(store: VectorStore, doc_id: str, doc_meta: dict, sample_chunks) -> None:
    store.add_chunks(doc_id, doc_meta, sample_chunks)
    store.add_tables(
        doc_id,
        doc_meta,
        [
            ExtractedTable(
                page_num=4,
                table_index=0,
                bbox=(10, 20, 200, 260),
                headers=["variable", "value"],
                rows=[["E", "210 GPa"]],
                caption="Table 1. Material constants.",
            )
        ],
    )
    store.add_figures(
        doc_id,
        doc_meta,
        [
            ExtractedFigure(
                page_num=5,
                figure_index=0,
                bbox=(20, 30, 320, 430),
                caption="Fig. 1. Stress-strain response.",
            )
        ],
    )


def _formula_candidate(index: int, equation_number: str) -> FormulaCandidate:
    return FormulaCandidate(
        page_num=6,
        bbox=(10, 20 + index * 30, 300, 45 + index * 30),
        raw_text=rf"\sigma_{{{index}}}=E\varepsilon {equation_number}",
        confidence=0.95,
        equation_number=equation_number,
        latex=rf"\sigma_{{{index}}}=E\varepsilon",
        source="mineru_content_list",
    )


def _formula(index: int, equation_number: str, latex: str, provider: str) -> ExtractedFormula:
    return ExtractedFormula(
        page_num=6,
        formula_index=index,
        bbox=(10, 20 + index * 30, 300, 45 + index * 30),
        latex=latex,
        confidence=0.95,
        equation_number=equation_number,
        provider=provider,
        source="structured_cache",
    )


def _indexer_for_formula_write(config, store: VectorStore, item: ZoteroItem) -> Indexer:
    indexer = Indexer.__new__(Indexer)
    indexer.config = config
    indexer.store = store
    indexer.zotero = MagicMock()
    indexer.zotero.get_item.return_value = item
    indexer.zotero.get_all_items_with_pdfs.return_value = [item]
    indexer.zotero.resolve_original_pdf_path.return_value = item.pdf_path
    indexer.journal_ranker = MagicMock()
    indexer.journal_ranker.lookup.return_value = "Q1"
    indexer._assert_config_hash_current = MagicMock()
    indexer._ensure_formula_provider_available = MagicMock()
    indexer._pdf_hash = MagicMock(return_value="pdf-hash")
    indexer._formula_candidate_provider = object()
    indexer._formula_provider = None
    return indexer


def _indexer_for_formula_batch(config, store: VectorStore, items: list[ZoteroItem]) -> Indexer:
    items_by_key = {item.item_key: item for item in items}
    indexer = Indexer.__new__(Indexer)
    indexer.config = config
    indexer.store = store
    indexer.zotero = MagicMock()
    indexer.zotero.get_item.side_effect = lambda key: items_by_key.get(key)
    indexer.zotero.get_all_items_with_pdfs.return_value = items
    indexer.zotero.resolve_original_pdf_path.side_effect = (
        lambda _key, title, fallback_path: fallback_path
    )
    indexer.journal_ranker = MagicMock()
    indexer.journal_ranker.lookup.return_value = "Q1"
    indexer._assert_config_hash_current = MagicMock()
    indexer._ensure_formula_provider_available = MagicMock()
    indexer._pdf_hash = MagicMock(return_value="pdf-hash")
    indexer._formula_candidate_provider = object()
    indexer._formula_provider = None
    return indexer


def test_index_formulas_writes_to_isolated_store_without_touching_existing_chunks(
    tmp_path,
    mock_embedder,
    sample_chunks,
):
    doc_id = "DOC1"
    item = _indexed_item(tmp_path, doc_id)
    config = _formula_config(tmp_path / "chroma")
    store = VectorStore(config.chroma_db_path, mock_embedder)
    doc_meta = _doc_meta(item)
    _add_existing_non_formula_chunks(store, doc_id, doc_meta, sample_chunks)
    protected_ids = {
        chunk_type: _chunk_ids_by_type(store, doc_id, chunk_type)
        for chunk_type in ("text", "table", "figure")
    }
    indexer = _indexer_for_formula_write(config, store, item)

    first_candidates = [_formula_candidate(0, "(1)"), _formula_candidate(1, "(2)")]
    indexer._recognize_formulas_for_item = MagicMock(
        return_value=[
            _formula(0, "(1)", r"\sigma = E\varepsilon", "mineru-cache"),
            _formula(1, "(2)", r"\varepsilon_p = \varepsilon-\sigma/E", "mineru-cache"),
        ]
    )
    with patch(
        "zotpilot.feature_extraction.formula_ocr.extract_formula_candidates",
        return_value=first_candidates,
    ):
        first = indexer.index_formulas(item_key=doc_id, refresh_existing=False)

    assert first["processed"] == 1
    assert first["formulas_indexed"] == 2
    assert first["write_ready"] is True
    assert first["results"][0]["status"] == "indexed"
    assert first["formula_scope_doc_ids"] == [doc_id]
    assert first["formula_scope_chunk_type_counts_before"] == {
        "text": 3,
        "table": 1,
        "figure": 1,
        "formula": 0,
    }
    assert first["formula_scope_chunk_type_counts_after"] == {
        "text": 3,
        "table": 1,
        "figure": 1,
        "formula": 2,
    }
    assert first["formula_scope_chunk_type_count_delta"] == {
        "text": 0,
        "table": 0,
        "figure": 0,
        "formula": 2,
    }
    assert first["formula_scope_non_formula_chunk_change"] is False
    assert first["formula_scope_non_formula_chunk_deltas"] == {}
    assert store.count_chunk_types({doc_id}) == {
        "text": 3,
        "table": 1,
        "figure": 1,
        "formula": 2,
    }
    assert _chunk_ids_by_type(store, doc_id, "formula") == [
        "DOC1_formula_0000",
        "DOC1_formula_0001",
    ]
    for chunk_type, expected_ids in protected_ids.items():
        assert _chunk_ids_by_type(store, doc_id, chunk_type) == expected_ids

    second_candidates = [_formula_candidate(0, "(1)")]
    indexer._recognize_formulas_for_item = MagicMock(
        return_value=[
            _formula(0, "(1)", r"\sigma = E(\varepsilon-\varepsilon_p)", "pdf-extract-kit")
        ]
    )
    with patch(
        "zotpilot.feature_extraction.formula_ocr.extract_formula_candidates",
        return_value=second_candidates,
    ):
        second = indexer.index_formulas(item_key=doc_id, refresh_existing=True)

    assert second["processed"] == 1
    assert second["formulas_indexed"] == 1
    assert second["write_ready"] is True
    assert second["results"][0]["status"] == "indexed"
    assert second["formula_scope_doc_ids"] == [doc_id]
    assert second["formula_scope_chunk_type_counts_before"] == {
        "text": 3,
        "table": 1,
        "figure": 1,
        "formula": 2,
    }
    assert second["formula_scope_chunk_type_counts_after"] == {
        "text": 3,
        "table": 1,
        "figure": 1,
        "formula": 1,
    }
    assert second["formula_scope_chunk_type_count_delta"] == {
        "text": 0,
        "table": 0,
        "figure": 0,
        "formula": -1,
    }
    assert second["formula_scope_non_formula_chunk_change"] is False
    assert second["formula_scope_non_formula_chunk_deltas"] == {}
    assert store.count_chunk_types({doc_id}) == {
        "text": 3,
        "table": 1,
        "figure": 1,
        "formula": 1,
    }
    for chunk_type, expected_ids in protected_ids.items():
        assert _chunk_ids_by_type(store, doc_id, chunk_type) == expected_ids

    formula_results = store.collection.get(
        where={
            "$and": [
                {"doc_id": {"$eq": doc_id}},
                {"chunk_type": {"$eq": "formula"}},
            ]
        },
        include=["metadatas"],
    )
    assert formula_results["ids"] == ["DOC1_formula_0000"]
    assert formula_results["metadatas"][0]["formula_latex"] == (
        r"\sigma = E(\varepsilon-\varepsilon_p)"
    )
    assert formula_results["metadatas"][0]["formula_provider"] == "pdf-extract-kit"


def test_index_formulas_isolated_batch_routes_quality_before_writing(
    tmp_path,
    mock_embedder,
    sample_chunks,
):
    config = _formula_config(tmp_path / "chroma")
    store = VectorStore(config.chroma_db_path, mock_embedder)
    good = _indexed_item(tmp_path, "GOOD1")
    gap = _indexed_item(tmp_path, "GAP1")
    semantic = _indexed_item(tmp_path, "SEM1")
    semantic_chunks = [
        Chunk(
            text=r"The constitutive update follows Eq. (3): \sigma = E\varepsilon.",
            chunk_index=0,
            page_num=3,
            char_start=0,
            char_end=65,
            section="methods",
            section_confidence=1.0,
        )
    ]
    for item, chunks in (
        (good, sample_chunks),
        (gap, sample_chunks),
        (semantic, semantic_chunks),
    ):
        _add_existing_non_formula_chunks(store, item.item_key, _doc_meta(item), chunks)
    protected_ids = {
        item.item_key: {
            chunk_type: _chunk_ids_by_type(store, item.item_key, chunk_type)
            for chunk_type in ("text", "table", "figure")
        }
        for item in (good, gap, semantic)
    }
    indexer = _indexer_for_formula_batch(config, store, [good, gap, semantic])
    candidates_by_key = {
        "GOOD1": [_formula_candidate(0, "(1)"), _formula_candidate(1, "(2)")],
        "GAP1": [_formula_candidate(0, "(1)"), _formula_candidate(1, "(3)")],
        "SEM1": [_formula_candidate(0, "(1)"), _formula_candidate(1, "(2)")],
    }
    formulas_by_key = {
        "GOOD1": [
            _formula(0, "(1)", r"\sigma = E\varepsilon", "mineru-cache"),
            _formula(1, "(2)", r"\eta = \sigma_m/\sigma_{eq}", "mineru-cache"),
        ],
    }
    indexer._recognize_formulas_for_item = MagicMock(
        side_effect=lambda item, **_kwargs: formulas_by_key.get(item.item_key, [])
    )

    def extract_for_pdf(pdf_path, **_kwargs):
        return candidates_by_key[pdf_path.stem]

    with patch(
        "zotpilot.feature_extraction.formula_ocr.extract_formula_candidates",
        side_effect=extract_for_pdf,
    ):
        state_path = tmp_path / "formula-status.jsonl"
        result = indexer.index_formulas(
            item_keys=["GOOD1", "GAP1", "SEM1"],
            refresh_existing=False,
            status_jsonl=state_path,
        )

    rows_by_key = {row["item_key"]: row for row in result["results"]}
    assert result["processed"] == 3
    assert result["formulas_indexed"] == 2
    assert result["write_blocked"] is True
    assert result["write_ready"] is False
    assert result["formula_write_status_counts"] == {"indexed": 1, "needs_review": 2}
    assert result["formula_write_route_counts"] == {
        "indexed": 1,
        "review_queue": 2,
        "deferred": 0,
        "skipped": 0,
        "failed": 0,
        "no_formula": 0,
        "unknown": 0,
    }
    report_by_key = {row["item_key"]: row for row in result["formula_write_report"]}
    assert report_by_key["GOOD1"]["route"] == "indexed"
    assert report_by_key["GOOD1"]["status"] == "indexed"
    assert report_by_key["GOOD1"]["n_formulas"] == 2
    assert report_by_key["GAP1"]["route"] == "review_queue"
    assert report_by_key["GAP1"]["recommended_review_mode"] == "candidate_numbering_review"
    assert report_by_key["GAP1"]["review_reasons"] == ["missing_equation_number_gap"]
    assert report_by_key["SEM1"]["route"] == "review_queue"
    assert report_by_key["SEM1"]["recommended_review_mode"] == "semantic_missing_candidate_repair"
    assert report_by_key["SEM1"]["semantic_evidence_count"] >= 1
    assert report_by_key["SEM1"]["semantic_unmatched_reference_count"] == 1
    assert report_by_key["SEM1"]["semantic_unmatched_reference_numbers"] == ["(3)"]
    assert report_by_key["SEM1"]["semantic_reference_match_status"] == "no_match"
    assert set(report_by_key["SEM1"]["semantic_review_flags"]) >= {
        "semantic_unmatched_references",
        "semantic_no_candidate_reference_overlap",
    }
    assert result["candidate_quality_review_count"] == 2
    assert result["semantic_formula_unmatched_reference_paper_count"] == 1
    assert rows_by_key["GOOD1"]["status"] == "indexed"
    assert rows_by_key["GAP1"]["status"] == "needs_review"
    assert rows_by_key["GAP1"]["review_reasons"] == ["missing_equation_number_gap"]
    assert rows_by_key["SEM1"]["status"] == "needs_review"
    assert rows_by_key["SEM1"]["review_reasons"] == [
        "semantic_evidence_unmatched_equation_references"
    ]
    assert rows_by_key["SEM1"]["semantic_formula_unmatched_reference_numbers"] == ["(3)"]
    assert [call.args[0].item_key for call in indexer._recognize_formulas_for_item.call_args_list] == [
        "GOOD1"
    ]
    assert store.count_chunk_types({"GOOD1"})["formula"] == 2
    assert store.count_chunk_types({"GAP1"})["formula"] == 0
    assert store.count_chunk_types({"SEM1"})["formula"] == 0
    for item in (good, gap, semantic):
        for chunk_type, expected_ids in protected_ids[item.item_key].items():
            assert _chunk_ids_by_type(store, item.item_key, chunk_type) == expected_ids

    events = [json.loads(line) for line in state_path.read_text().splitlines()]
    assert events[-1]["event"] == "formula_backfill_run_finished"
    assert events[-1]["formula_write_status_counts"] == result["formula_write_status_counts"]
    assert events[-1]["formula_write_route_counts"] == result["formula_write_route_counts"]
    assert events[-1]["formula_write_report"] == result["formula_write_report"]
    assert events[-1]["formula_scope_doc_ids"] == result["formula_scope_doc_ids"]
    assert events[-1]["formula_scope_chunk_type_counts_before"] == result[
        "formula_scope_chunk_type_counts_before"
    ]
    assert events[-1]["formula_scope_chunk_type_counts_after"] == result[
        "formula_scope_chunk_type_counts_after"
    ]
    assert events[-1]["formula_scope_chunk_type_count_delta"] == result[
        "formula_scope_chunk_type_count_delta"
    ]
    assert events[-1]["formula_scope_non_formula_chunk_change"] == result[
        "formula_scope_non_formula_chunk_change"
    ]
    assert events[-1]["formula_scope_non_formula_chunk_deltas"] == result[
        "formula_scope_non_formula_chunk_deltas"
    ]


def test_index_formulas_blocks_scaling_when_non_formula_scope_counts_change(
    tmp_path,
    mock_embedder,
    sample_chunks,
):
    doc_id = "DOC2"
    item = _indexed_item(tmp_path, doc_id)
    config = _formula_config(tmp_path / "chroma")
    store = VectorStore(config.chroma_db_path, mock_embedder)
    _add_existing_non_formula_chunks(store, doc_id, _doc_meta(item), sample_chunks)
    indexer = _indexer_for_formula_write(config, store, item)
    indexer._recognize_formulas_for_item = MagicMock(
        return_value=[_formula(0, "(1)", r"\sigma = E\varepsilon", "mineru-cache")]
    )
    count_before = {"text": 3, "table": 1, "figure": 1, "formula": 0}
    count_after = {"text": 2, "table": 1, "figure": 1, "formula": 1}

    with (
        patch.object(store, "count_chunk_types", side_effect=[count_before, count_after]),
        patch(
            "zotpilot.feature_extraction.formula_ocr.extract_formula_candidates",
            return_value=[_formula_candidate(0, "(1)")],
        ),
    ):
        result = indexer.index_formulas(item_key=doc_id, refresh_existing=False)

    assert result["processed"] == 1
    assert result["formulas_indexed"] == 1
    assert result["formula_scope_chunk_type_count_delta"] == {
        "text": -1,
        "table": 0,
        "figure": 0,
        "formula": 1,
    }
    assert result["formula_scope_non_formula_chunk_change"] is True
    assert result["formula_scope_non_formula_chunk_deltas"] == {"text": -1}
    assert result["write_blocked"] is True
    assert result["write_ready"] is False
    assert result["write_block_reasons"] == ["formula_scope_non_formula_chunk_changed"]
    assert "text/table/figure chunk counts changed" in result["next_action"]
