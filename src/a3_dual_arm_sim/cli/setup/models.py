"""Download the public Qwen verifier backbone to the server's data disk."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from a3_dual_arm_sim.core.paths import project_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="a3-sim setup models", description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output", type=Path,
                        default=project_root() / "models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args(argv)
    # Set endpoint before huggingface_hub is imported; no global user config edits.
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ.setdefault("HF_HOME", str(project_root() / ".runtime/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        args.model, revision=args.revision, local_dir=args.output, max_workers=2,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"],
    )
    print(json.dumps({"model": args.model, "path": str(path), "endpoint": args.endpoint,
                      "fine_tuned_verifier": False}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
