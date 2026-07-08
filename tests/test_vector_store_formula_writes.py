from zotpilot.models import ExtractedFigure, ExtractedFormula, ExtractedTable
from zotpilot.vector_store import VectorStore


def _chunk_ids_by_type(store, doc_id: str, chunk_type: str) -> list[str]:
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


def test_formula_incremental_add_and_replace_preserves_existing_chunks(
    tmp_path,
    mock_embedder,
    sample_chunks,
):
    doc_id = "TEST001"
    store = VectorStore(tmp_path / "chroma", mock_embedder)
    doc_meta = {
        "title": "Test Paper",
        "authors": "Test Author",
        "year": 2020,
        "citation_key": "test2020",
        "publication": "Test Journal",
        "doi": "10.1234/test",
        "tags": "ml; ai",
        "collections": "Test Collection",
        "journal_quartile": "Q1",
        "pdf_hash": "abc123",
        "quality_grade": "A",
    }
    store.add_chunks(doc_id, doc_meta, sample_chunks)
    table = ExtractedTable(
        page_num=4,
        table_index=0,
        bbox=(10, 20, 200, 260),
        headers=["variable", "value"],
        rows=[["alpha", "0.2"]],
        caption="Table 1. Calibration constants.",
    )
    figure = ExtractedFigure(
        page_num=5,
        figure_index=0,
        bbox=(15, 25, 300, 420),
        caption="Fig. 1. Stress-strain curve.",
    )
    store.add_tables(doc_id, doc_meta, [table])
    store.add_figures(doc_id, doc_meta, [figure])

    protected_ids = {
        chunk_type: _chunk_ids_by_type(store, doc_id, chunk_type)
        for chunk_type in ("text", "table", "figure")
    }
    assert store.count_chunk_types({doc_id}) == {
        "text": 3,
        "table": 1,
        "figure": 1,
        "formula": 0,
    }

    added = store.add_new_formulas(
        doc_id,
        doc_meta,
        [
            ExtractedFormula(
                page_num=6,
                formula_index=0,
                bbox=(1, 2, 30, 40),
                latex=r"\sigma = E\varepsilon",
                equation_number="(1)",
                confidence=0.94,
                provider="mineru-cache",
                source="structured_cache",
            ),
            ExtractedFormula(
                page_num=6,
                formula_index=1,
                bbox=(1, 45, 30, 80),
                latex=r"D = \int_0^{\bar{\varepsilon}_p} f(\eta)\,d\bar{\varepsilon}_p",
                equation_number="(2)",
                confidence=0.91,
                provider="mineru-cache",
                source="structured_cache",
            ),
        ],
    )

    assert added == 2
    assert store.count_chunk_types({doc_id}) == {
        "text": 3,
        "table": 1,
        "figure": 1,
        "formula": 2,
    }

    added_again = store.add_new_formulas(
        doc_id,
        doc_meta,
        [
            ExtractedFormula(
                page_num=6,
                formula_index=0,
                bbox=(1, 2, 30, 40),
                latex=r"\sigma = E\varepsilon",
                equation_number="(1)",
                confidence=0.94,
            ),
        ],
    )
    assert added_again == 0

    replaced = store.replace_formulas(
        doc_id,
        doc_meta,
        [
            ExtractedFormula(
                page_num=6,
                formula_index=0,
                bbox=(1, 2, 30, 40),
                latex=r"\sigma = E(\varepsilon-\varepsilon_p)",
                equation_number="(1)",
                confidence=0.96,
                provider="pdf-extract-kit",
                source="recognition_json",
            ),
            ExtractedFormula(
                page_num=7,
                formula_index=2,
                bbox=(2, 4, 36, 44),
                latex=r"\eta = \sigma_m / \sigma_{eq}",
                equation_number="(3)",
                confidence=0.93,
                provider="pdf-extract-kit",
                source="recognition_json",
            ),
        ],
    )

    assert replaced == 2
    assert store.count_chunk_types({doc_id}) == {
        "text": 3,
        "table": 1,
        "figure": 1,
        "formula": 2,
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
    assert sorted(formula_results["ids"]) == [
        "TEST001_formula_0000",
        "TEST001_formula_0002",
    ]
    formulas_by_id = dict(zip(formula_results["ids"], formula_results["metadatas"]))
    assert formulas_by_id["TEST001_formula_0000"]["formula_latex"] == (
        r"\sigma = E(\varepsilon-\varepsilon_p)"
    )
    assert formulas_by_id["TEST001_formula_0000"]["formula_provider"] == "pdf-extract-kit"
    assert formulas_by_id["TEST001_formula_0002"]["formula_equation_number"] == "(3)"
