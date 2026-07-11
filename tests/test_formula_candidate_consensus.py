from zotpilot.feature_extraction.formula_candidate_consensus import build_formula_candidate_consensus


def test_candidate_consensus_clusters_same_formula_across_structured_providers():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 4,
                "source": "mineru_content_list",
                "equation_number": "(1)",
                "bbox": [10, 20, 300, 48],
                "latex_preview": r"\sigma = E\varepsilon",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 4,
                "source": "pdf_extract_kit_formula_recognition",
                "equation_number": "(1)",
                "bbox": [12, 21, 298, 49],
                "latex_preview": r"\sigma = E\varepsilon",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 2,
                "page_num": 5,
                "source": "text_layer",
                "equation_number": "(2)",
                "bbox": [20, 70, 250, 95],
                "raw_text_preview": "eta = sigma_m / sigma_eq",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["mode"] == "candidate_preview_consensus"
    assert consensus["cluster_count"] == 2
    assert consensus["multi_provider_cluster_count"] == 1
    assert consensus["single_provider_cluster_count"] == 1
    assert consensus["supported_cluster_count"] == 1
    assert consensus["review_cluster_count"] == 1
    assert consensus["single_provider_review_cluster_count"] == 1
    assert consensus["conflict_review_cluster_count"] == 0
    assert consensus["ocr_fallback_cluster_count"] == 1
    first_cluster = consensus["clusters"][0]
    assert first_cluster["cluster_route"] == "supported_candidate"
    assert first_cluster["source_group_counts"] == {
        "mineru_cache": 1,
        "pdf_extract_kit": 1,
    }
    assert first_cluster["review_flags"] == ["multi_provider_agreement"]
    assert first_cluster["representative_latex_preview"] == r"\sigma = E\varepsilon"
    assert first_cluster["candidate_details"][0]["source"] == "mineru_content_list"


def test_candidate_consensus_flags_number_conflict_on_same_bbox():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 7,
                "source": "mineru_content_list",
                "equation_number": "(3)",
                "bbox": [10, 20, 300, 48],
                "latex_preview": r"D = 1 - \exp(-a\varepsilon_p)",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 7,
                "source": "pdf_extract_kit_formula_detection",
                "equation_number": "(4)",
                "bbox": [11, 20, 299, 49],
                "raw_text_preview": r"D = 1 - exp(-a epsilon_p)",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["conflict_cluster_count"] == 1
    cluster = consensus["clusters"][0]
    assert cluster["cluster_route"] == "conflict_review"
    assert "equation_number_conflict" in cluster["conflict_flags"]
    assert "ocr_fallback_required" in cluster["review_flags"]
    assert cluster["candidate_details"][1]["raw_text_preview"] == r"D = 1 - exp(-a epsilon_p)"


def test_candidate_consensus_does_not_merge_truncated_conflicting_numbers():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 45,
                "source": "pdf_text_equation_number_truncated",
                "parser_label": "auto",
                "equation_number": "(3-13)",
                "bbox": [10, 20, 300, 80],
                "raw_text_preview": "D_s = phi(theta, eta) ... (3-13) phi(theta, eta) = ...",
                "has_latex": False,
                "needs_ocr": True,
            },
            {
                "candidate_index": 1,
                "page_num": 45,
                "source": "text_layer",
                "parser_label": "text",
                "equation_number": "(3-14)",
                "bbox": [12, 22, 299, 78],
                "raw_text_preview": "(1 - xi^2)(1 - T_k) + T_k eta <= 0 (3-14)",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 2
    assert consensus["conflict_cluster_count"] == 0


def test_candidate_consensus_keeps_same_parser_different_numbers_separate():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 4,
                "source": "pdf_text_equation_number",
                "parser_label": "auto",
                "equation_number": "(7)",
                "bbox": [10, 20, 300, 48],
                "raw_text_preview": r"\rho_i = \sum_j m_j W(x_i-x_j,h) (7)",
                "has_latex": False,
                "needs_ocr": True,
            },
            {
                "candidate_index": 1,
                "page_num": 4,
                "source": "pdf_text_equation_number",
                "parser_label": "auto",
                "equation_number": "(8)",
                "bbox": [11, 20, 299, 49],
                "raw_text_preview": r"\rho_i = \sum_j m_j W(x_i-x_j,h) (8)",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 2
    assert consensus["conflict_cluster_count"] == 0
    assert [cluster["primary_equation_number"] for cluster in consensus["clusters"]] == ["(7)", "(8)"]


def test_candidate_consensus_keeps_same_number_different_page_formulas_separate():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 3,
                "source": "pdf_text_equation_number",
                "parser_label": "auto",
                "equation_number": "(5)",
                "bbox": [10, 20, 300, 48],
                "raw_text_preview": r"\dot{\lambda} = f(\sigma) (5)",
                "has_latex": False,
                "needs_ocr": True,
            },
            {
                "candidate_index": 1,
                "page_num": 6,
                "source": "text_layer",
                "parser_label": "text",
                "equation_number": "(5)",
                "bbox": [40, 60, 220, 90],
                "raw_text_preview": r"b_1 = b_3 + b_5 (5)",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 2
    assert consensus["conflict_cluster_count"] == 0
    assert [cluster["page_nums"] for cluster in consensus["clusters"]] == [[3], [6]]


def test_candidate_consensus_keeps_far_page_repeated_numbers_separate_even_when_similar():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 76,
                "source": "text_layer",
                "parser_label": "auto",
                "equation_number": "(1)",
                "bbox": [10, 20, 300, 48],
                "raw_text_preview": "x = 1/2 (1)",
                "has_latex": False,
                "needs_ocr": True,
            },
            {
                "candidate_index": 1,
                "page_num": 165,
                "source": "text_layer",
                "parser_label": "text",
                "equation_number": "(1)",
                "bbox": [40, 60, 220, 90],
                "raw_text_preview": "x = 1/2 (1)",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 2
    assert consensus["conflict_cluster_count"] == 0


def test_candidate_consensus_keeps_same_source_numbered_and_unnumbered_variants_separate():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 45,
                "source": "text_layer",
                "parser_label": "text",
                "equation_number": "",
                "bbox": [10, 20, 300, 80],
                "raw_text_preview": "phi(theta, eta) = 1 - xi^2 eta > 0",
                "has_latex": False,
                "needs_ocr": True,
            },
            {
                "candidate_index": 1,
                "page_num": 45,
                "source": "text_layer",
                "parser_label": "text",
                "equation_number": "(3-14)",
                "bbox": [12, 22, 299, 78],
                "raw_text_preview": "(1 - xi^2)(1 - T_k) + T_k eta <= 0 (3-14)",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 2
    assert consensus["conflict_cluster_count"] == 0


def test_candidate_consensus_keeps_cross_source_numbered_and_unnumbered_variants_separate():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 7,
                "source": "mineru_content_list",
                "parser_label": "auto",
                "equation_number": "(1)",
                "bbox": [10, 20, 300, 80],
                "latex_preview": r"\sigma=A(\varepsilon_0+\varepsilon_p)^n",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 7,
                "source": "text_layer",
                "parser_label": "text",
                "equation_number": "",
                "bbox": [12, 22, 299, 78],
                "raw_text_preview": "sigma = (1 - alpha) [A (epsilon_0 + epsilon_p)^n] +",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 2
    assert consensus["conflict_cluster_count"] == 0


def test_candidate_consensus_clusters_same_page_latex_and_text_signature():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 3,
                "source": "mineru_content_list",
                "equation_number": "",
                "bbox": [40, 100, 260, 130],
                "latex_preview": r"\sigma = E\varepsilon",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 3,
                "source": "text_layer",
                "equation_number": "",
                "bbox": [300, 500, 520, 530],
                "raw_text_preview": "σ = E ε",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["multi_provider_cluster_count"] == 1
    assert consensus["supported_cluster_count"] == 1
    cluster = consensus["clusters"][0]
    assert cluster["cluster_route"] == "supported_candidate"
    assert "multi_provider_agreement" in cluster["review_flags"]
    assert "missing_equation_number" in cluster["review_flags"]


def test_candidate_consensus_normalizes_math_unicode_text_signatures():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 4,
                "source": "mineru_content_list",
                "equation_number": "(1)",
                "bbox": [40, 100, 260, 130],
                "latex_preview": r"\sigma = E\varepsilon",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 4,
                "source": "text_layer",
                "equation_number": "(1)",
                "bbox": [280, 500, 520, 530],
                "raw_text_preview": "𝜎 = 𝐸𝜀",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["supported_cluster_count"] == 1


def test_candidate_consensus_does_not_replace_le_inside_left_command():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 4,
                "source": "mineru_content_list",
                "equation_number": "",
                "bbox": [40, 100, 260, 130],
                "latex_preview": r"y=\left(\sigma_1+\sigma_2\right)^{p_*}",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 4,
                "source": "text_layer",
                "equation_number": "",
                "bbox": [280, 500, 520, 530],
                "raw_text_preview": "𝑦=(𝜎_1+𝜎_2)^{𝑝_*}",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["supported_cluster_count"] == 1


def test_candidate_consensus_collapses_text_layer_duplicate_math_glyphs():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 9,
                "source": "mineru_content_list",
                "equation_number": "",
                "bbox": [40, 100, 260, 130],
                "latex_preview": r"c = \sqrt{\frac{1}{T}\int_0^T(f-g)^2dt}",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 9,
                "source": "text_layer",
                "equation_number": "",
                "bbox": [280, 500, 520, 530],
                "raw_text_preview": "𝑐𝑐 = √{1/𝑇𝑇 ∫_0^𝑇𝑇(𝑓𝑓−𝑔𝑔)^2𝑑𝑑𝑡𝑡}",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["supported_cluster_count"] == 1


def test_candidate_consensus_normalizes_symbol_font_private_use_greek():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 16,
                "source": "mineru_content_list",
                "equation_number": "(2-2)",
                "bbox": [40, 100, 260, 130],
                "latex_preview": r"\sigma_m=(\sigma_1+\sigma_2+\sigma_3)/3",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 16,
                "source": "text_layer",
                "equation_number": "(2-2)",
                "bbox": [280, 500, 520, 530],
                "raw_text_preview": "\uf073m=(\uf0731+\uf0732+\uf0733)/3 (2-2)",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["supported_cluster_count"] == 1


def test_candidate_consensus_normalizes_cjk_pdf_text_sqrt_and_subscript_markers():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 2,
                "source": "mineru_content_list",
                "parser_label": "auto",
                "equation_number": "",
                "bbox": [40, 100, 260, 130],
                "latex_preview": r"\bar{\varepsilon_f} = \sqrt{3}/2",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 2,
                "source": "text_layer",
                "parser_label": "text",
                "equation_number": "",
                "bbox": [280, 500, 520, 530],
                "raw_text_preview": "ε － f = 槡 3 / 2",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["supported_cluster_count"] == 1


def test_candidate_consensus_normalizes_cjk_pdf_text_subscript_dash_before_equals():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 2,
                "source": "mineru_content_list",
                "parser_label": "auto",
                "equation_number": "",
                "bbox": [40, 100, 260, 130],
                "latex_preview": r"\bar{\varepsilon}_{f} = a e^{b\eta}",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 2,
                "source": "text_layer",
                "parser_label": "text",
                "equation_number": "",
                "bbox": [280, 500, 520, 530],
                "raw_text_preview": "ε f － = a e b η",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["supported_cluster_count"] == 1


def test_candidate_consensus_ignores_equation_tags_and_subscript_separators_in_signatures():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 3,
                "source": "mineru_content_list",
                "parser_label": "auto",
                "equation_number": "(1)",
                "bbox": [40, 100, 260, 130],
                "latex_preview": r"17.5 < i_{\mathrm{cmod}} < 19.9\tag{1}",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 3,
                "source": "text_layer",
                "parser_label": "text",
                "equation_number": "(1)",
                "bbox": [280, 500, 520, 530],
                "raw_text_preview": "17 . 5 < i cmod < 19 . 9 (1)",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["conflict_cluster_count"] == 0
    assert consensus["supported_cluster_count"] == 1


def test_candidate_consensus_normalizes_text_layer_temperature_subscripts():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 10,
                "source": "mineru_content_list",
                "parser_label": "auto",
                "equation_number": "(2)",
                "bbox": [40, 100, 260, 130],
                "latex_preview": (
                    r"T_{\mathrm{norm}}\text{or}T^{*}="
                    r"(T-T_{\mathrm{ref}})/(T_{\mathrm{m}}-T_{\mathrm{ref}})\tag{2}"
                ),
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 10,
                "source": "text_layer",
                "parser_label": "text",
                "equation_number": "(2)",
                "bbox": [280, 500, 520, 530],
                "raw_text_preview": "T norm or T ∗ = ( T − T ref ) / ( T m − T ref ) (2)",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["conflict_cluster_count"] == 0
    assert consensus["supported_cluster_count"] == 1


def test_candidate_consensus_ignores_low_information_text_signature_conflict():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 84,
                "source": "mineru_content_list",
                "equation_number": "(6-1)",
                "bbox": [40, 100, 260, 130],
                "latex_preview": r"f_{0w}=f_{0b}+f_0^*",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 84,
                "source": "text_layer",
                "equation_number": "(6-1)",
                "bbox": [280, 500, 520, 530],
                "raw_text_preview": "* 0 0 0 = + f f f 接头 母材 （ 6-1 ）",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["conflict_cluster_count"] == 0
    assert consensus["supported_cluster_count"] == 1


def test_candidate_consensus_ignores_no_latex_text_layer_signature_conflict():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 12,
                "source": "mineru_content_list",
                "parser_label": "auto",
                "equation_number": "(2.30)",
                "bbox": [40, 100, 360, 130],
                "latex_preview": (
                    r"\varepsilon_{\mathrm{nom}} = \frac{L - L_0}{L_0} = "
                    r"\frac{\Delta L}{L_0}"
                ),
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 12,
                "source": "text_layer",
                "parser_label": "text",
                "equation_number": "(2.30)",
                "bbox": [42, 101, 358, 131],
                "raw_text_preview": "0 0 = ε - Δ = L L L L L （ 2.30 ）",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["conflict_cluster_count"] == 0
    assert consensus["supported_cluster_count"] == 1


def test_candidate_consensus_ignores_operator_heavy_short_text_signature_conflict():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 10,
                "source": "mineru_content_list",
                "equation_number": "",
                "bbox": [40, 100, 260, 130],
                "latex_preview": r"\eta_0 = \frac{\sigma_1+\sigma_2+\sigma_3}{3\bar{\sigma}}",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 10,
                "source": "text_layer",
                "equation_number": "",
                "bbox": [42, 102, 258, 131],
                "raw_text_preview": "= = - - = - ( )( ) ξ σ",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 1
    assert consensus["conflict_cluster_count"] == 0


def test_candidate_consensus_keeps_different_same_page_formulas_separate():
    consensus = build_formula_candidate_consensus(
        [
            {
                "candidate_index": 0,
                "page_num": 3,
                "source": "mineru_content_list",
                "equation_number": "",
                "bbox": [40, 100, 260, 130],
                "latex_preview": r"\sigma = E\varepsilon",
                "has_latex": True,
                "needs_ocr": False,
            },
            {
                "candidate_index": 1,
                "page_num": 3,
                "source": "text_layer",
                "equation_number": "",
                "bbox": [300, 500, 520, 530],
                "raw_text_preview": "D = 1 - exp(-a epsilon)",
                "has_latex": False,
                "needs_ocr": True,
            },
        ]
    )

    assert consensus["cluster_count"] == 2
    assert consensus["multi_provider_cluster_count"] == 0
    assert consensus["supported_cluster_count"] == 0
    assert consensus["single_provider_review_cluster_count"] == 2
