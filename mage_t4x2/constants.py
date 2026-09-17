"""Frozen project-wide constants for the Mage-Flow-Turbo dual-T4 qualification.

Nothing in this module touches CUDA.
"""

REPOSITORY_NAME = "Mage-Flow-Turbo-PyTorch-BF16-T4x2"
PACKAGE_NAME = "mage_t4x2"

MODEL_ID = "dangkhoa2016/mage-flow-community-mage-flow-turbo"
MODEL_REVISION = "65bb3500f0da9df6a41ec6383716fc02cf014773"

UPSTREAM_MAGE_REPO_URL = "https://github.com/microsoft/Mage"
UPSTREAM_MAGE_COMMIT_SHA = "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"

FRAMEWORK = "upstream Mage / PyTorch"
TASK = "text-to-image"

RESOLUTION_HEIGHT = 512
RESOLUTION_WIDTH = 512
RESOLUTION = (RESOLUTION_HEIGHT, RESOLUTION_WIDTH)
SEED = 42
STEPS = 4
CFG = 1.0

ACCELERATOR = "NVIDIA Tesla T4 x2"
T4_VRAM_BYTES = 16 * 1024**3
T4_COMPUTE_CAPABILITY = (7, 5)
T4_USABLE_VRAM_BYTES = 15_636_037_632

RESERVED_RUNTIME_HEADROOM_BYTES = int(1.5 * 1024**3)
MEMORY_PLANNER_NAME = "plan_dual_t4_spread"
L0_PHASE_NAME = "L0"
L0_ACCEPTED_STATE_PREFIX = "L0_SPREAD_LOAD_PASS"
GPU0 = "cuda:0"
GPU1 = "cuda:1"
LOGICAL_GPU0 = "logical_gpu_0"
LOGICAL_GPU1 = "logical_gpu_1"

CPU_FALLBACK = False
STABLE_DIFFUSION_CPP_ALLOWED = False
SD_CLI_ALLOWED = False

BF16_REQUIRED_STR = "BF16_REQUIRED"
BF16_TRACKED_STR = "BF16_TRACKED"
NON_FLOATING_STR = "NON_FLOATING"

# Attribute names probed, in order, when discovering transformer blocks.
BLOCK_ATTRIBUTE_NAMES = (
    "blocks",
    "double_blocks",
    "single_blocks",
    "layers",
    "transformer_blocks",
    "dit_blocks",
)

# Pre-block stateful modules (Corrective A4.3 device plan). Kept on device0.
TRANSFORMER_PRE_MODULES = ("img_in", "txt_norm", "time_text_embed", "txt_in")

# Post-block stateful modules (Corrective A4.3 device plan). Must live with the
# final block device so no extra cross-GPU transfer happens after the loop.
TRANSFORMER_POST_MODULES = ("norm_out", "proj_out")

# Optional RoPE embedding; may be parameterless (then RUNTIME_TENSOR_DEVICE).
TRANSFORMER_POS_EMBED_ATTR = "pos_embed"

# Canonical order of pipeline components when building a route / device map.
CANONICAL_COMPONENTS = (
    "tokenizer",
    "text_encoder",
    "transformer",
    "scheduler",
    "vae",
)

MAJOR_COMPONENTS = ("text_encoder", "transformer", "vae")