from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from zotpilot.feature_extraction.formula_ocr import FormulaCandidate
from zotpilot.indexer import Indexer
from zotpilot.models import (
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


def test_index_formulas_writes_to_isolated_store_without_touching_existing_chunks(
    tmp_path,
    mock_embedder,
    sample_chunks,
):
    doc_id = "DOC1"
    item = _indexed_item(tmp_path, doc_id)
    config = _formula_config(tmp_path / "chroma")
    store = VectorStore(config.chroma_db_path, mock_embedder)
    doc_meta = {
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
