"""Place the assembled T5-11B weights into the Hugging Face cache, HF's own way.

`get_text_embedding()` constructs ``CosmosT5TextEncoder(device="cuda")`` with no
``cache_dir`` and no local path, so the model is resolved as the Hub id
``google-t5/t5-11b`` against ``$HF_HOME``.  The file therefore has to land in that
cache in the layout the library expects -- a content-addressed blob plus a symlink
from the snapshot directory -- rather than in the directory it was downloaded to.

The script deliberately does not fabricate that layout from scratch.  It checks the
pieces the cache already has (config and tokenizer, already symlinked) and only adds
the missing weight blob, because guessing at a cache layout is how a "present but
unloadable" model happens: everything looks downloaded and `from_pretrained` still
goes to the network.

It also verifies the weight file against the SHA-256 the Hub records for it, so a
truncated or reassembled-wrong download cannot be mistaken for a good one, and it
removes the stale partial blob left by an earlier interrupted attempt.
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path


def sha256_of(path: Path, label: str = "") -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as handle:
        while True:
            block = handle.read(64 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    print(f"  {'hashed ' + label if label else 'hashed'} {size:,} bytes")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--repo", default="google-t5/t5-11b")
    parser.add_argument("--expect-sha256", default="", help="recorded LFS sha, if known")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache = hf_home / "hub" / ("models--" + args.repo.replace("/", "--"))
    print(f"HF_HOME : {hf_home}")
    print(f"cache   : {cache}")

    if not cache.is_dir():
        print(f"ERROR: no cache directory at {cache}")
        return 1

    snapshots = sorted((cache / "snapshots").iterdir())
    if not snapshots:
        print(f"ERROR: no snapshot under {cache / 'snapshots'}")
        return 1
    snapshot = snapshots[-1]
    print(f"snapshot: {snapshot.name}")

    # What the cache already has.  config.json is what from_pretrained reads first,
    # so its absence would mean we are looking at the wrong snapshot.
    present = {}
    for name in ("config.json", "tokenizer.json", "spiece.model", "pytorch_model.bin"):
        entry = snapshot / name
        if entry.exists():
            present[name] = entry.resolve().stat().st_size
    print("  already in the snapshot:")
    for name, size in present.items():
        print(f"    {name:24s} {size:,} bytes")
    if "config.json" not in present:
        print("ERROR: the snapshot has no config.json; refusing to guess")
        return 1

    if not args.weights.is_file():
        print(f"ERROR: {args.weights} is not a file")
        return 1
    weights_size = args.weights.stat().st_size
    print(f"weights : {args.weights} ({weights_size:,} bytes)")

    print("verifying ...")
    digest = sha256_of(args.weights, "weights")
    if args.expect_sha256:
        if digest != args.expect_sha256:
            print(f"ERROR: sha256 mismatch\n  expected {args.expect_sha256}\n  got      {digest}")
            return 1
        print(f"  sha256 matches the Hub record: {digest}")
    else:
        print(f"  sha256 {digest} (nothing to compare against)")

    blob = cache / "blobs" / digest
    link = snapshot / "pytorch_model.bin"

    if args.dry_run:
        print(f"\nDRY RUN -- would place:")
        print(f"  blob {blob}")
        print(f"  link {link} -> ../../blobs/{digest}")
        return 0

    if blob.exists() and blob.stat().st_size == weights_size:
        print(f"  blob already present at {blob}")
    else:
        # Move rather than copy: same filesystem, and 45 GB of duplicate is exactly
        # the kind of thing that fills a disk quietly.
        print(f"  moving weights into {blob}")
        if blob.exists():
            blob.unlink()
        shutil.move(str(args.weights), str(blob))
        os.chmod(blob, 0o644)

    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path("..") / ".." / "blobs" / digest)
    print(f"  linked {link.name} -> ../../blobs/{digest}")

    # The stale partial blob from the earlier aborted attempt: 1.2 GB of nothing,
    # and its presence makes the cache look like it has more of the model than it does.
    for stale in (cache / "blobs").glob("*.incomplete"):
        print(f"  removing stale partial blob {stale.name}")
        stale.unlink()

    print("\n--- resulting cache ---")
    for name in sorted(p.name for p in snapshot.iterdir()):
        entry = snapshot / name
        if entry.is_symlink():
            target = os.readlink(entry)
            size = entry.resolve().stat().st_size if entry.resolve().exists() else -1
            print(f"  {name:24s} -> {target}  ({size:,} bytes)")
        else:
            print(f"  {name:24s} (regular file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
