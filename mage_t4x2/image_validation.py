"""Output image validation helpers (CPU-safe, PIL + numpy only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import constants as C


def validate_image(
    path: str,
    expected_size: Tuple[int, int] = C.RESOLUTION,
) -> Dict[str, Any]:
    """Validate an output PNG/JPEG image file.

    Returns a JSON-friendly dict with existence, geometry, mode and NaN/Inf
    facts. Absence of the file yields ``valid=False`` (never raises).
    """
    p = Path(path)
    exists = p.exists()
    result: Dict[str, Any] = {
        "path": path,
        "exists": exists,
        "is_512x512": False,
        "rgb_valid": False,
        "mode": None,
        "format": None,
        "has_nan": False,
        "has_inf": False,
        "pixels_min": None,
        "pixels_max": None,
        "valid": False,
        "error": None,
    }
    if not exists:
        result["error"] = "file_not_found"
        return result

    try:
        from PIL import Image

        with Image.open(p) as im:
            result["mode"] = im.mode
            result["format"] = im.format
            result["format_size"] = im.size
            w, h = im.size
            result["width"] = w
            result["height"] = h
            result["is_512x512"] = (w, h) == expected_size
            result["rgb_valid"] = im.mode in ("RGB", "RGBA")
            arr = im.convert("RGB")
            import numpy as np

            np_arr = np.asarray(arr, dtype=np.float32)
            result["pixels_min"] = float(np_arr.min())
            result["pixels_max"] = float(np_arr.max())
            result["has_nan"] = bool(np.isnan(np_arr).any())
            result["has_inf"] = bool(np.isinf(np_arr).any())
    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["valid"] = (
        result["exists"]
        and result["is_512x512"]
        and result["rgb_valid"]
        and not result["has_nan"]
        and not result["has_inf"]
    )
    return result


def write_validation(path: str, out_path: str) -> Dict[str, Any]:
    data = validate_image(path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def make_fixture_png(path: str, width: int = 512, height: int = 512, mode: str = "RGB") -> str:
    """Create a tiny synthetic PNG fixture for tests."""
    from PIL import Image

    im = Image.new(mode, (width, height), color=(128, 60, 200) if mode == "RGB" else 128)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG")
    return path