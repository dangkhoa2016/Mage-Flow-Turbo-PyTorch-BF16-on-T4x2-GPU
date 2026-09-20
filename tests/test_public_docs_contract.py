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


def test_public_notebook_uses_git_source_bootstrap_only():
    path = Path("notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU.git" in text
    assert 'RELEASE_REF = "v1.0.0"' in text
    assert "EXPECTED_SOURCE_ZIP_BASENAME" not in text
    assert "EXPECTED_IDENTITY_SIDECAR_BASENAME" not in text
    assert "zipfile.ZipFile" not in text
    assert "SOURCE_EXTRACTION_ROOT" not in text


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
