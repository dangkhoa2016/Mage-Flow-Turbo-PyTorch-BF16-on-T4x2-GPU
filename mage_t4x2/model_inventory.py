"""Static model-component inventory for the Mage-Flow-Turbo T2I pipeline.

The inventory documents, for every pipeline component, the class name,
upstream construction/load site, dtype and device behaviour, statically
discoverable characteristics and the forward entry point.

This is a *static* CPU-side inventory: it does not load the 20 GB model.
Upstream file references are anchored to the pinned commit in
``mage_t4x2.source_pin``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from . import constants as C

_UPSTREAM_BASE = "https://github.com/microsoft/Mage/blob/76bec2bb3818863f470de7e867c2dc7f1d0bfd83"


def _component(
    key: str,
    class_name: str,
    category: str,
    construction_site: str,
    load_site: str,
    dtype_behavior: str,
    device_behavior: str,
    n_blocks: int | str | None,
    forward_entry: str,
    output_type: str,
    classification: str,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "key": key,
        "class_name": class_name,
        "category": category,
        "construction_site": construction_site,
        "load_site": load_site,
        "dtype_behavior": dtype_behavior,
        "device_behavior": device_behavior,
        "num_transformer_blocks": n_blocks,
        "forward_entry_point": forward_entry,
        "output_type": output_type,
        "precision_classification": classification,
        "notes": notes,
    }


def static_component_inventory() -> Dict[str, Any]:
    """Return the frozen static inventory document."""
    components: List[Dict[str, Any]] = [
        _component(
            key="tokenizer",
            class_name="Qwen3-VL tokenizer (sentencepiece / AutoTokenizer)",
            category="non_floating",
            construction_site=f"{_UPSTREAM_BASE}/mage_flow/model.py (ModelConfig → text encoder) ",
            load_site=f"{_UPSTREAM_BASE}/mage_flow/pipeline.py (load_from_repo, _text_encoder_path)",
            dtype_behavior="integer token ids; no floating state",
            device_behavior="host-side tokenization; no GPU tensors",
            n_blocks=None,
            forward_entry="text encoder encode pipeline stage",
            output_type="tensor of token ids + attention mask",
            classification="NON_FLOATING",
            notes="Text encoder prompts are screened by a mandatory content gate (screen_text).",
        ),
        _component(
            key="text_encoder",
            class_name="Qwen3-VL vision-language text encoder (MageFlowTxtEnc)",
            category="major_model",
            construction_site=f"{_UPSTREAM_BASE}/mage_flow/model.py (from_model_index → Qwen3-VL)",
            load_site=f"{_UPSTREAM_BASE}/mage_flow/pipeline.py load_from_repo → model.txt_enc.to(torch.bfloat16)",
            dtype_behavior="Mage upstream already casts text encoder to torch.bfloat16 after load (load_from_repo)",
            device_behavior="placed on `device` set at load time; GPU phase will pin to cuda:0",
            n_blocks=None,
            forward_entry="encode / screen_text",
            output_type="prompt embeddings (float tensor)",
            classification="BF16_REQUIRED",
        ),
        _component(
            key="transformer",
            class_name="NR-MMDiT — Native-Resolution Multimodal Diffusion Transformer (MageFlowTransformer)",
            category="major_model",
            construction_site=f"{_UPSTREAM_BASE}/mage_flow/model.py ",
            load_site=f"{_UPSTREAM_BASE}/mage_flow/pipeline.py load_from_repo → diffusion_pytorch_model.safetensors; model.transformer.to(torch.bfloat16)",
            dtype_behavior="Mage upstream materializes transformer bf16 from bf16 safetensors (os.environ param_dtype=bf16)",
            device_behavior="placed on `device` at load; GPU phase will block-partition across cuda:0/cuda:1",
            n_blocks="depth from transformer/config.json (static discovery at GPU load)",
            forward_entry="forward (packed varlen sequence, rectified-flow denoising)",
            output_type="denoised latent tokens (float tensor)",
            classification="BF16_REQUIRED",
            notes=(
                "Config meta keys stripped upstream: _class_name, txt_max_length, packing, "
                "schedule_mode, static_shift, use_time_shift, rope_type, apply_text_rotary_emb, "
                "mlp_ratio, depth_single_blocks, theta, qkv_bias, guidance_embed, vec_in_dim, "
                "vec_type, time_type, double_block_type. Core DiT reads in_channels, out_channels, "
                "context_in_dim, hidden_size, num_heads, depth, axes_dim, checkpoint, patch_size."
            ),
        ),
        _component(
            key="scheduler",
            class_name="diffusers.FlowMatchEulerDiscreteScheduler",
            category="runtime_float",
            construction_site="diffusers scheduler_config.json",
            load_site=f"{_UPSTREAM_BASE}/mage_flow/pipeline.py load_from_repo → FlowMatchEulerDiscreteScheduler.from_pretrained(.../scheduler)",
            dtype_behavior="scheduler coefficients are float runtime tensors; BF16_TRACKED not BF16_REQUIRED",
            device_behavior="host-side scheduler stepping; noise scheduled with torch.manual_seed per sample",
            n_blocks=None,
            forward_entry="step()",
            output_type="time-step scalars / coefficient buffers",
            classification="BF16_TRACKED",
            notes="static_shift default 6.0 from scheduler_config.json; Turbo uses cfg=1.0, steps=4.",
        ),
        _component(
            key="vae",
            class_name="MageVAE — one-step diffusion latent codec (128-channel, 16x downsampled)",
            category="major_model",
            construction_site=f"{_UPSTREAM_BASE}/mage_flow/model.py (MageVAE)",
            load_site=f"{_UPSTREAM_BASE}/mage_flow/pipeline.py load_from_repo → model.vae.to(torch.bfloat16)",
            dtype_behavior="Mage upstream casts VAE to torch.bfloat16 after load",
            device_behavior="placed on `device` at load; GPU phase will pin to cuda:1",
            n_blocks=None,
            forward_entry="decode (one-step pixel diffusion decoder)",
            output_type="RGB image tensor (uint8-able float)",
            classification="BF16_REQUIRED",
        ),
        _component(
            key="helper_tensors",
            class_name="latents / prompt embeddings / position ids / cu_seqlens",
            category="runtime_float",
            construction_site=f"{_UPSTREAM_BASE}/mage_flow/pipeline.py generate_images",
            load_site="created per run",
            dtype_behavior="latents and embeddings materialized torch.bfloat16 by upstream; tracked not required",
            device_behavior="follow the transformer device during a phase",
            n_blocks=None,
            forward_entry="n/a",
            output_type="floating tensors",
            classification="BF16_TRACKED",
            notes="cu_seqlens are integer tensors (NON_FLOATING).",
        ),
    ]
    return {
        "model_identifier": C.MODEL_ID,
        "model_revision": C.MODEL_REVISION,
        "task": C.TASK,
        "major_model_components": [
            c for c in components if c["category"] == "major_model"
        ],
        "runtime_float_components": [
            c for c in components if c["category"] == "runtime_float"
        ],
        "non_floating_components": [
            c for c in components if c["category"] == "non_floating"
        ],
        "components": components,
        "inventory_scope": "static",
        "model_loaded": False,
    }


def write_component_inventory(path: str) -> Dict[str, Any]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = static_component_inventory()
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data