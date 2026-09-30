"""Make the gated PaliGemma tokenizer name resolve to local files.

Why this exists
---------------
pi0.5's ``policy_preprocessor.json`` names ``google/paligemma-3b-pt-224`` for its
tokenizer.  That repository is gated, so ``AutoTokenizer.from_pretrained`` fails with
``OSError: You are trying to access a gated repo`` on a machine without credentials --
and the failure arrives while building the processor pipeline, which is a long way from
anything that mentions a tokenizer.

For inference the problem is avoidable by passing the local path directly, which is what
``adapters/pi05_a3.py`` does.  **Training is not**: ``lerobot_train.py`` calls
``make_pre_post_processors`` itself and the tokenizer name comes from the checkpoint's
own JSON, with no CLI hook to override it.  So the name itself has to resolve.

What this does
--------------
Populates the Hugging Face cache under ``$HF_HOME/hub`` with an entry for
``models--google--paligemma-3b-pt-224`` whose snapshot contains the same files, taken from
an ungated mirror.  With the entry present, the name resolves offline and nothing else
changes -- no checkpoint is edited and no library is patched.

Why a copy of the files is equivalent rather than approximate
-----------------------------------------------------------
π0.5 uses PaliGemma's tokenizer unmodified, so what matters is the ``tokenizer.model``
(SentencePiece) and its config; the mirror carries both.  The alternative -- obtaining
access to the gated repo -- would produce byte-identical files.

    python scripts/setup_paligemma_tokenizer_cache.py [--source DIR] [--check-only]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

#: The name the checkpoint asks for, and therefore the cache entry that has to exist.
GATED_REPO = "google/paligemma-3b-pt-224"
#: Where the same files were fetched from (ungated mirror), by pi05's setup.
DEFAULT_SOURCE = Path("/root/autodl-tmp/models/paligemma-tokenizer")
#: The files a tokenizer needs.  Listed rather than globbed so a missing piece is named.
FILENAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    # Not part of the tokenizer, and required anyway.  `AutoTokenizer.from_pretrained`
    # given a *repository id* resolves the model config before the tokenizer -- to pick
    # the tokenizer class from the architecture -- so without this the entry looks
    # complete (every tokenizer file resolves) and the name still fails, with a network
    # error that says nothing about a missing config.
    "config.json",
)
#: Required, unlike `added_tokens.json`.
OPTIONAL = ("added_tokens.json",)
#: Not a real upstream commit: this is a local stand-in, and pretending to know the
#: upstream revision would be a lie that a future reader could act on.  It stands in for
#: "the files here", which is exactly what the snapshot directory contains.
LOCAL_REVISION = "local-" + "0" * 36


def hub_root() -> Path:
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cache_entry(hub: Path) -> Path:
    return hub / ("models--" + GATED_REPO.replace("/", "--"))


def build(source: Path, hub: Path) -> int:
    missing = [name for name in FILENAMES if not (source / name).is_file()]
    required = [name for name in missing if name not in OPTIONAL]
    if required:
        print(f"missing from {source}: {required}", file=sys.stderr)
        print("fetch them first, e.g.:", file=sys.stderr)
        print(
            "  HF_ENDPOINT=https://hf-mirror.com python -c \"from huggingface_hub import "
            "snapshot_download; snapshot_download('leo009/paligemma-3b-pt-224', "
            "allow_patterns=['tokenizer*','special_tokens_map.json','added_tokens.json',"
            f"'config.json'], local_dir='{source}')\"",
            file=sys.stderr,
        )
        return 1

    entry = cache_entry(hub)
    blobs = entry / "blobs"
    snapshot = entry / "snapshots" / LOCAL_REVISION
    refs = entry / "refs"
    for directory in (blobs, snapshot, refs):
        directory.mkdir(parents=True, exist_ok=True)

    for name in FILENAMES:
        src = source / name
        if not src.is_file():
            print(f"  skipping {name} (absent, and not required)")
            continue
        # Name the blob after its own content, which is what the real cache does and makes
        # two runs idempotent: re-running with the same files writes the same names.
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        blob = blobs / digest
        if not blob.exists():
            shutil.copy2(src, blob)
        link = snapshot / name
        if link.is_symlink() or link.exists():
            link.unlink()
        # A relative symlink, so the entry stays valid if $HF_HOME is moved -- which is
        # how the real cache stores blobs, and the reason `du` on a snapshot shows nothing.
        link.symlink_to(os.path.relpath(blob, snapshot))
        print(f"  {name} -> blobs/{digest[:12]}...")

    # `refs/main` is what an unqualified `from_pretrained(name)` resolves through.
    (refs / "main").write_text(LOCAL_REVISION, encoding="utf-8")
    print(f"\ncache entry: {entry}")
    return 0


def check() -> int:
    """Resolve the gated name the way the training script will, and say what happened."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"transformers is not importable here ({exc}); run with the pi05 venv")
        return 2

    try:
        tokenizer = AutoTokenizer.from_pretrained(GATED_REPO, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - the message is the point
        print(f"FAILED to resolve {GATED_REPO}: {type(exc).__name__}: {str(exc)[:300]}")
        return 1

    print(f"resolved {GATED_REPO}")
    print(f"  vocab size : {tokenizer.vocab_size}")
    print(f"  pad token  : {tokenizer.pad_token_id}")
    sample = tokenizer("Task: move the blocks, State: 1 2 3;\nAction: ")["input_ids"]
    print(f"  encoding   : {len(sample)} tokens, first ids {sample[:8]}")
    if sample[:8] == [0] * 8:
        print("  WARNING: the encoding is all padding, which is not a valid tokenization")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="do not write anything; only resolve the gated name offline",
    )
    args = parser.parse_args()

    hub = hub_root()
    if args.check_only:
        print(f"hub root: {hub}")
        return check()

    if not args.source.is_dir():
        print(f"source directory does not exist: {args.source}", file=sys.stderr)
        return 1

    print(f"source    : {args.source}")
    print(f"hub root  : {hub}")
    print(f"populating cache entry for {GATED_REPO}\n")
    written = build(args.source, hub)
    if written != 0:
        return written
    print()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
