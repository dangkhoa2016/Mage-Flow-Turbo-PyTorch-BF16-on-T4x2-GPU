from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from public_demo.runner import run_public_demo

    parser = argparse.ArgumentParser(description="Mage-Flow-Turbo PyTorch BF16 dual-T4 public demo")
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    result = run_public_demo(
        project_root=args.project_root,
        model_path=args.model_path,
        output_root=args.output_root,
    )
    for line in result["verdict_lines"]:
        print(line)
    print(f"run_id={result['run_id']}")
    print(f"summary={result['summary_path']}")
    if result["error"]:
        print(f"error={result['error']}")
    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
