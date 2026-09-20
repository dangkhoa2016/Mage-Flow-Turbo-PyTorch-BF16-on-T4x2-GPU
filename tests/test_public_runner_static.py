from pathlib import Path


def test_public_runner_uses_git_pypi_bootstrap_not_publication_archives():
    source = Path("public_demo/runner.py").read_text()
    assert "prepare_public_bootstrap" in source
    assert "publication ZIP" not in source
    assert "identity JSON" not in source
    assert "candidate-sdpa-fixed" not in source


def test_public_runner_requires_owner_qualified_model_path():
    source = Path("public_demo/runner.py").read_text()
    assert '"/kaggle/input/models/" + C.MODEL_ID + "/pytorch/default/1"' in source
    assert "C.MODEL_ID.split" not in source


def test_sdpa_guards_surround_model_execution():
    source = Path("public_demo/live_execution.py").read_text()
    load = source.index("load_runtime_state_dual_t4_spread")
    post = source.index("SDPA_POST_LOAD")
    placement = source.index("apply_dual_t4_mixed")
    pre = source.index("SDPA_PRE_INFERENCE")
    run = source.index("gs.run_t2i")
    assert load < post < placement < pre < run


def test_failed_runner_never_leaves_first_failed_gate_empty():
    source = Path("public_demo/runner.py").read_text()
    assert 'summary["first_failed_gate"] = "PUBLIC_BOOTSTRAP"' in source
    assert 'summary["first_failed_gate"] = "UNHANDLED_EXCEPTION"' in source
