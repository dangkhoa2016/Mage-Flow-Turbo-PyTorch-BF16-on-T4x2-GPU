from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from public_demo.contract import (
    CANONICAL,
    MODEL_ID,
    MODEL_REVISION,
    UPSTREAM_MAGE_COMMIT,
    UPSTREAM_MAGE_TREE,
    CANONICAL_PROMPT,
    PublicDemoContract,
    PublicDemoContractError,
    canonical_model_path,
    public_demo_input_mismatches,
    require_canonical_public_demo_inputs,
    require_public_demo_model_path,
    require_public_demo_model_source,
)

NON_CANONICAL_OVERRIDES = {
    "model_id": "some-other-owner/some-other-model",
    "model_revision": "0" * 40,
    "upstream_mage_commit": "0" * 40,
    "upstream_mage_tree": "0" * 40,
    "attention_backend": "math",
    "dtype": "float16",
    "prompt": "a completely different prompt",
    "width": 768,
    "height": 768,
    "steps": 5,
    "cfg_scale": 2.0,
    "seed": 43,
    "num_transformer_blocks": 11,
    "split_block": 2,
}


def test_canonical_contract_is_valid():
    CANONICAL.validate()


def test_canonical_profile_values():
    assert CANONICAL.model_id == "dangkhoa2016/mage-flow-community-mage-flow-turbo"
    assert CANONICAL.model_revision == "65bb3500f0da9df6a41ec6383716fc02cf014773"
    assert CANONICAL.upstream_mage_commit == "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"
    assert CANONICAL.upstream_mage_tree == "946b91bcb2cac75e6cfe8399f0f7f330a2280adf"
    assert CANONICAL.prompt == CANONICAL_PROMPT
    assert CANONICAL.seed == 42
    assert CANONICAL.steps == 4
    assert CANONICAL.cfg_scale == 1.0
    assert CANONICAL.width == 512
    assert CANONICAL.height == 512
    assert CANONICAL.num_transformer_blocks == 12
    assert CANONICAL.split_block == 1
    assert CANONICAL.attention_backend == "sdpa"
    assert CANONICAL.dtype == "bfloat16"


def test_canonical_model_path_is_owner_qualified():
    assert canonical_model_path() == Path(
        "/kaggle/input/models/dangkhoa2016/"
        "mage-flow-community-mage-flow-turbo/pytorch/default/1"
    )
    assert str(canonical_model_path()).endswith("/pytorch/default/1")


def test_expected_block_devices_split_one():
    devices = CANONICAL.expected_block_devices
    assert devices[0] == "cuda:0"
    for index in range(1, 12):
        assert devices[index] == "cuda:1"


def test_no_public_demo_input_mismatches_by_default():
    assert public_demo_input_mismatches() == []


@pytest.mark.parametrize("name", sorted(NON_CANONICAL_OVERRIDES))
def test_non_canonical_public_input_fails_closed(name):
    mismatches = public_demo_input_mismatches(**{name: NON_CANONICAL_OVERRIDES[name]})
    assert any(f"non-canonical {name}" in error for error in mismatches)


def test_require_canonical_inputs_raises_on_seed_override():
    with pytest.raises(PublicDemoContractError, match="NONCANONICAL_PUBLIC_INPUT"):
        require_canonical_public_demo_inputs(seed=43)


def test_require_canonical_inputs_rejects_unknown_field():
    with pytest.raises(PublicDemoContractError, match="unknown public demo input"):
        require_canonical_public_demo_inputs(some_unknown_input=1)


@pytest.mark.parametrize("field,value", list(NON_CANONICAL_OVERRIDES.items()))
def test_non_canonical_contract_instance_fails_closed(field, value):
    mutated = dataclasses.replace(CANONICAL, **{field: value})
    assert mutated.errors()
    with pytest.raises(PublicDemoContractError, match="NONCANONICAL_PUBLIC_CONTRACT"):
        mutated.validate()


def test_model_id_mismatch_is_rejected():
    with pytest.raises(PublicDemoContractError, match="non-canonical model_id"):
        dataclasses.replace(CANONICAL, model_id="wrong/owner-model").validate()


def test_model_revision_mismatch_is_rejected():
    with pytest.raises(PublicDemoContractError, match="non-canonical model_revision"):
        dataclasses.replace(CANONICAL, model_revision="0" * 40).validate()


def test_upstream_commit_mismatch_is_rejected():
    with pytest.raises(PublicDemoContractError, match="non-canonical upstream_mage_commit"):
        dataclasses.replace(CANONICAL, upstream_mage_commit="0" * 40).validate()


def test_upstream_tree_mismatch_is_rejected():
    with pytest.raises(PublicDemoContractError, match="non-canonical upstream_mage_tree"):
        dataclasses.replace(CANONICAL, upstream_mage_tree="0" * 40).validate()


def test_public_demo_model_path_accepts_exact_canonical_path():
    require_public_demo_model_path(str(canonical_model_path()))


def test_public_demo_model_path_rejects_wrong_owner_qualified_model():
    with pytest.raises(PublicDemoContractError, match="model path must be the canonical owner-qualified"):
        require_public_demo_model_path(
            "/kaggle/input/models/dangkhoa2016/some-other-model/pytorch/default/1"
        )


def test_public_demo_model_path_rejects_basename_only_path():
    with pytest.raises(PublicDemoContractError, match="model path must be the canonical owner-qualified"):
        require_public_demo_model_path("/kaggle/input/mage-flow-community-mage-flow-turbo")


def test_public_demo_model_path_rejects_dropped_owner_namespace():
    with pytest.raises(PublicDemoContractError, match="model path must be the canonical owner-qualified"):
        require_public_demo_model_path(
            "/kaggle/input/models/mage-flow-community-mage-flow-turbo/pytorch/default/1"
        )


def test_public_demo_model_source_rejects_remote_fallback():
    with pytest.raises(PublicDemoContractError, match="remote fallback is forbidden"):
        require_public_demo_model_source({"model_source": "REMOTE_FALLBACK"})


def test_public_demo_model_source_accepts_local_attachment():
    require_public_demo_model_source({"model_source": "LOCAL_ATTACHMENT"})


def test_missing_model_source_fails_closed():
    with pytest.raises(PublicDemoContractError, match="remote fallback is forbidden"):
        require_public_demo_model_source({"selected_path": "/kaggle/input/..."})


def test_contract_constants_do_not_drift_from_module_level_authority():
    assert CANONICAL.model_id == MODEL_ID
    assert CANONICAL.model_revision == MODEL_REVISION
    assert CANONICAL.upstream_mage_commit == UPSTREAM_MAGE_COMMIT
    assert CANONICAL.upstream_mage_tree == UPSTREAM_MAGE_TREE


def test_public_contract_matches_runtime_constants():
    from mage_t4x2 import constants as C

    assert CANONICAL.model_id == C.MODEL_ID
    assert CANONICAL.model_revision == C.MODEL_REVISION
    assert CANONICAL.upstream_mage_commit == C.UPSTREAM_MAGE_COMMIT_SHA
    assert CANONICAL.width == C.RESOLUTION_WIDTH
    assert CANONICAL.height == C.RESOLUTION_HEIGHT
    assert CANONICAL.seed == C.SEED
    assert CANONICAL.steps == C.STEPS
    assert CANONICAL.cfg_scale == C.CFG


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_transformer_blocks": 0},
        {"num_transformer_blocks": -1},
        {"split_block": 0},
        {"split_block": -1},
    ],
)
def test_malformed_topology_fails_with_contract_error(kwargs):
    mutated = dataclasses.replace(CANONICAL, **kwargs)
    with pytest.raises(PublicDemoContractError):
        mutated.validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_transformer_blocks": 0},
        {"num_transformer_blocks": -1},
    ],
)
def test_malformed_block_count_reports_structural_error(kwargs):
    mutated = dataclasses.replace(CANONICAL, **kwargs)
    assert any(
        "num_transformer_blocks must be a positive integer" in error
        for error in mutated.errors()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split_block": 0},
        {"split_block": -1},
    ],
)
def test_malformed_split_block_reports_structural_error(kwargs):
    mutated = dataclasses.replace(CANONICAL, **kwargs)
    assert any(
        "split_block must be an integer within the transformer block range" in error
        for error in mutated.errors()
    )