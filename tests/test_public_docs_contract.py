from pathlib import Path
import json


def test_readme_language_switch_links():
    en = Path("README.md").read_text(encoding="utf-8")
    vi = Path("README.vi.md").read_text(encoding="utf-8")
    assert "[Tiếng Việt](README.vi.md)" in en
    assert "[English](README.md)" in vi


def test_kaggle_guide_language_switch_links():
    en = Path("docs/kaggle-public-demo.md").read_text(encoding="utf-8")
    vi = Path("docs/kaggle-public-demo.vi.md").read_text(encoding="utf-8")
    assert "[Tiếng Việt](kaggle-public-demo.vi.md)" in en
    assert "[English](kaggle-public-demo.md)" in vi


def test_public_demo_prompt_presentation_statement_is_unique_per_language():
    en = Path("docs/kaggle-public-demo.md").read_text(encoding="utf-8")
    vi = Path("docs/kaggle-public-demo.vi.md").read_text(encoding="utf-8")

    en_stmt = (
        "The notebook's primary per-image presentation shows every generated PNG "
        "independently and prints the exact deterministic prompt immediately above "
        "its corresponding image"
    )
    vi_stmt = (
        "Phần trình bày chính theo từng ảnh của notebook hiển thị độc lập mọi PNG "
        "đã sinh và in prompt xác định chính xác ngay phía trên ảnh tương ứng"
    )

    assert en.count(en_stmt) == 1
    assert vi.count(vi_stmt) == 1


def test_readme_distinguishes_two_lanes():
    en = Path("README.md").read_text(encoding="utf-8")
    vi = Path("README.vi.md").read_text(encoding="utf-8")
    for source in (en, vi):
        assert "Lane A" in source
        assert "Lane B" in source
    assert "one logical T2I trajectory" in en
    assert "512×512" in en
    assert "512×512" in vi


def test_public_notebook_uses_git_source_bootstrap_only():
    path = Path("notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU.git" in text
    assert 'SOURCE_REF = os.environ.get("SOURCE_REF", "v1.0.0")' in text
    assert "RELEASE_REF" not in text
    assert "EXPECTED_SOURCE_ZIP_BASENAME" not in text
    assert "EXPECTED_IDENTITY_SIDECAR_BASENAME" not in text
    assert "zipfile.ZipFile" not in text
    assert "SOURCE_EXTRACTION_ROOT" not in text


def test_public_notebook_is_two_lane_showcase():
    path = Path("notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Lane A" in text
    assert "Lane B" in text
    assert "canonical qualification" in text
    assert "production showcase" in text


def test_public_notebook_exposes_showcase_verdict_markers_and_category():
    path = Path("notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for marker in (
        "Category",
        "PUBLIC_SHOWCASE_FINAL_VERDICT",
        "PUBLIC_DEMO_FINAL_VERDICT",
        "PUBLIC_NOTEBOOK_FINAL_VERDICT",
    ):
        assert marker in text


def test_public_notebook_has_fail_closed_combined_verdict():
    path = Path(
        "notebooks/"
        "kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))

    final_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "PUBLIC_NOTEBOOK_FINAL_VERDICT" in "".join(cell.get("source", []))
    ]

    assert len(final_cells) == 1
    final = final_cells[0]

    assert "PUBLIC_NOTEBOOK_FINAL_VERDICT=" in final
    assert 'lane_a["status"] == "PASS"' in final
    assert 'lane_b["status"] == "PASS"' in final
    assert "if not combined:" in final
    assert "raise SystemExit" in final


def test_each_code_cell_has_markdown_explanation_and_is_pristine():
    path = Path("notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert index > 0
        assert notebook["cells"][index - 1].get("cell_type") == "markdown"
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []


def test_showcase_runner_is_documented_in_docs():
    docs = (
        Path("docs/kaggle-public-demo.md").read_text(encoding="utf-8")
        + Path("docs/kaggle.md").read_text(encoding="utf-8")
    )
    assert "run_public_showcase" in docs
    assert "artifacts/public-showcase" in docs


# --------------------------------------------------------------------------- #
# V4.4 — notebook evidence/presentation static contract.
# --------------------------------------------------------------------------- #


def _notebook():
    path = Path("notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    return notebook, text


def test_v44_individual_output_presentation_includes_exact_prompt():
    _, text = _notebook()

    assert '**Prompt:** {prompt}' in text
    assert 'row.get("prompt", "")' in text
    assert "missing prompt for" in text
    assert text.index('**Prompt:** {prompt}') < \
        text.index('display(IPImage(filename=str(output_path)))')


def test_v44_lane_a_views_use_persisted_summary():
    _, text = _notebook()
    assert 'lane_a["summary_path"]' in text
    assert "lane_a_summary" in text
    assert 'lane_a.get("placement")' not in text
    assert 'lane_a.get("routing")' not in text


def test_v44_individual_output_presentation_exists():
    _, text = _notebook()
    for marker in (
        "INDIVIDUAL_OUTPUT_IMAGES",
        "case_results",
        "output_rel",
        "IPImage",
    ):
        assert marker in text


def test_v44_contact_sheet_semantics_markers():
    _, text = _notebook()
    assert "CONTACT_SHEET_IS_DERIVED_ARTIFACT" in text
    assert "CONTACT_SHEET_IS_MODEL_OUTPUT" in text
    assert "CONTACT_SHEET_IS_DERIVED_ARTIFACT=True" in text
    assert "CONTACT_SHEET_IS_MODEL_OUTPUT=False" in text
    assert "not a direct model output" in text


def test_v44_individual_presentation_precedes_contact_sheet():
    _, text = _notebook()
    assert text.index("INDIVIDUAL_OUTPUT_IMAGES") < \
        text.index("CONTACT_SHEET_IS_DERIVED_ARTIFACT")


def test_v44_contact_sheet_retained():
    _, text = _notebook()
    assert "gallery" in text
    assert "GALLERY_IMAGES" in text


def test_v44_final_verdict_markers_and_fail_closed_logic():
    notebook, text = _notebook()
    for marker in (
        "PUBLIC_DEMO_FINAL_VERDICT",
        "PUBLIC_SHOWCASE_FINAL_VERDICT",
        "PUBLIC_NOTEBOOK_FINAL_VERDICT",
    ):
        assert marker in text

    final_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "PUBLIC_NOTEBOOK_FINAL_VERDICT" in "".join(cell.get("source", []))
    ]

    assert len(final_cells) == 1
    final = final_cells[0]
    assert "PUBLIC_NOTEBOOK_FINAL_VERDICT=" in final
    assert 'lane_a["status"] == "PASS"' in final
    assert 'lane_b["status"] == "PASS"' in final
    assert "if not combined:" in final
    assert "raise SystemExit" in final