"""Download pinned A3 training inputs and record their Hub revisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_REPO = "Eter0109/a3-front-close-left-100"
MODEL_REPO = "lerobot/smolvla_base"


def download_training_inputs(
    dataset_root: Path,
    model_root: Path,
    manifest: Path,
    *,
    dataset_revision: str = "main",
    model_revision: str = "main",
    vlm_revision: str = "main",
) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    dataset_sha = api.dataset_info(DATASET_REPO, revision=dataset_revision).sha
    model_sha = api.model_info(MODEL_REPO, revision=model_revision).sha
    dataset_root, model_root = (
        dataset_root.expanduser().resolve(),
        model_root.expanduser().resolve(),
    )
    # Resume only the same pinned repository; never mix unrelated/stale datasets.
    for root, repo, sha in (
        (dataset_root, DATASET_REPO, dataset_sha),
        (model_root, MODEL_REPO, model_sha),
    ):
        identity = {"repo_id": repo, "revision": sha}
        marker = root / ".a3_download_revision.json"
        if (
            root.exists()
            and any(root.iterdir())
            and (not marker.is_file() or json.loads(marker.read_text()) != identity)
        ):
            raise FileExistsError(f"Directory contains unrelated inputs: {root}")
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + "\n")
    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        revision=dataset_sha,
        local_dir=dataset_root,
        max_workers=2,
    )
    snapshot_download(
        MODEL_REPO,
        revision=model_sha,
        local_dir=model_root,
        allow_patterns=["*.json", "*.safetensors", "README.md"],
        max_workers=2,
    )
    config = json.loads((model_root / "config.json").read_text())
    vlm_repo = config["vlm_model_name"]
    vlm_sha = api.model_info(vlm_repo, revision=vlm_revision).sha
    vlm_path = snapshot_download(
        vlm_repo,
        revision=vlm_sha,
        max_workers=2,
        allow_patterns=["*.json", "*.model", "*.txt", "*.jinja"],
    )
    result = {
        "dataset": {"repo_id": DATASET_REPO, "revision": dataset_sha, "root": str(dataset_root)},
        "model": {"repo_id": MODEL_REPO, "revision": model_sha, "root": str(model_root)},
        "vlm": {"repo_id": vlm_repo, "revision": vlm_sha, "root": str(vlm_path)},
    }
    (model_root / "a3_dependencies.json").write_text(json.dumps(result["vlm"], indent=2) + "\n")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(args) -> int:
    result = download_training_inputs(
        args.dataset_root,
        args.model_root,
        args.manifest,
        dataset_revision=args.dataset_revision,
        model_revision=args.model_revision,
        vlm_revision=args.vlm_revision,
    )
    print(json.dumps(result, indent=2))
    return 0
