"""Parallel, resumable fetch of one large file from the HF mirror.

A single curl stream from the Xet bridge runs at about 3.4 MB/s here, which would
make the 45.23 GB T5-11B checkpoint take nearly four hours.  Eight concurrent range
requests reach about 11 MB/s, so the file is pulled as a set of independent chunks.

Chunks are deliberately small (200 MB).  A single 45 GB stream that fails at 90
percent has to start over; a chunk that fails costs at most its own 200 MB, and a
chunk already on disk with the right size is skipped, so re-running the script
resumes rather than restarts.

Nothing here trusts the transfer: the assembled file's size and, when the Hub
exposes one, its SHA-256 are both checked before the file is accepted.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHUNK_BYTES = 200 * 1024 * 1024


def remote_size(url: str) -> int:
    """Total size of the file behind ``url``, following redirects."""

    out = subprocess.run(
        ["curl", "-sIL", url],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lengths = [
        int(line.split(":", 1)[1].strip())
        for line in out.splitlines()
        if line.lower().startswith("content-length:")
    ]
    if not lengths:
        raise RuntimeError(f"no Content-Length for {url}:\n{out[:500]}")
    # The largest is the real file: a redirect chain reports the size at each hop,
    # and intermediate responses can be shorter.
    return max(lengths)


def expected_sha256(repo: str, filename: str) -> str | None:
    """The LFS SHA-256 the Hub records for ``filename``, if it records one."""

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        info = HfApi(endpoint=endpoint).model_info(repo, files_metadata=True)
    except Exception as error:  # noqa: BLE001
        print(f"  could not read file metadata ({type(error).__name__}); skipping sha256")
        return None
    for sibling in info.siblings:
        if sibling.rfilename == filename:
            lfs = getattr(sibling, "lfs", None)
            if isinstance(lfs, dict):
                return lfs.get("sha256")
            if lfs is not None:
                return getattr(lfs, "sha256", None)
    return None


def fetch_chunk(url: str, start: int, end: int, target: Path, attempts: int = 6) -> None:
    """Download one inclusive byte range, verifying the length before keeping it."""

    want = end - start + 1
    if target.exists() and target.stat().st_size == want:
        return
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            [
                "curl", "-sL",
                "--retry", "4",
                "--retry-delay", "3",
                "--connect-timeout", "30",
                "--max-time", "1800",
                "-r", f"{start}-{end}",
                "-o", str(target),
                url,
            ],
            capture_output=True,
        )
        if target.exists() and target.stat().st_size == want:
            return
        got = target.stat().st_size if target.exists() else 0
        if attempt == attempts:
            raise RuntimeError(
                f"chunk {start}-{end} failed after {attempts} attempts "
                f"(got {got} of {want} bytes, curl rc={result.returncode})"
            )
        time.sleep(min(2 ** attempt, 20))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="google-t5/t5-11b")
    parser.add_argument("--file", default="pytorch_model.bin")
    parser.add_argument("--output", required=True, help="Final assembled file path")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mb", type=int, default=200)
    args = parser.parse_args()

    global CHUNK_BYTES
    CHUNK_BYTES = args.chunk_mb * 1024 * 1024

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    url = f"{endpoint}/{args.repo}/resolve/main/{args.file}"
    output = Path(args.output)
    if output.exists() and output.stat().st_size > 0:
        print(f"{output} already exists ({output.stat().st_size} bytes); nothing to do")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output.parent / f".{output.name}.parts"
    parts_dir.mkdir(exist_ok=True)

    total = remote_size(url)
    count = (total + CHUNK_BYTES - 1) // CHUNK_BYTES
    print(f"source : {url}")
    print(f"size   : {total:,} bytes ({total / 1e9:.2f} GB)")
    print(f"chunks : {count} x {args.chunk_mb} MB, {args.workers} in parallel")
    print(f"parts  : {parts_dir}")
    print(flush=True)

    jobs = []
    for index in range(count):
        start = index * CHUNK_BYTES
        end = min(start + CHUNK_BYTES, total) - 1
        jobs.append((index, start, end, parts_dir / f"{index:05d}.part"))

    done = sum(1 for _, start, end, path in jobs if path.exists() and path.stat().st_size == end - start + 1)
    print(f"resuming: {done}/{count} chunks already complete", flush=True)

    started = time.time()
    completed = done
    lock = __import__("threading").Lock()

    def run(job):
        nonlocal completed
        index, start, end, path = job
        fetch_chunk(url, start, end, path)
        with lock:
            completed += 1
            if completed % 10 == 0 or completed == count:
                elapsed = time.time() - started
                got = sum(
                    p.stat().st_size for _, _, _, p in jobs if p.exists()
                )
                rate = got / elapsed if elapsed else 0.0
                remaining = (total - got) / rate if rate else 0.0
                print(
                    f"  {completed}/{count} chunks, {got / 1e9:.2f}/{total / 1e9:.2f} GB, "
                    f"{rate / 1e6:.1f} MB/s, ~{remaining / 60:.0f} min left",
                    flush=True,
                )

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001
                failures.append(str(error))

    if failures:
        print(f"\n{len(failures)} chunk(s) failed; re-run to resume:")
        for message in failures[:5]:
            print(f"  {message}")
        return 1

    print(f"\nassembling {count} chunks into {output} ...", flush=True)
    with open(output, "wb") as sink:
        for index, _, _, path in jobs:
            with open(path, "rb") as source:
                while True:
                    block = source.read(64 * 1024 * 1024)
                    if not block:
                        break
                    sink.write(block)

    size = output.stat().st_size
    print(f"assembled: {size:,} bytes")
    if size != total:
        print(f"MISMATCH: expected {total:,} bytes")
        return 1

    want = expected_sha256(args.repo, args.file)
    if want:
        print(f"hashing {output.name} (~1 min) ...", flush=True)
        digest = hashlib.sha256()
        with open(output, "rb") as handle:
            for block in iter(lambda: handle.read(64 * 1024 * 1024), b""):
                digest.update(block)
        got = digest.hexdigest()
        if got != want:
            print(f"SHA-256 MISMATCH\n  expected {want}\n  got      {got}")
            return 1
        print(f"sha256 OK: {got}")
    else:
        print("no recorded sha256; size check only")

    for _, _, _, path in jobs:
        path.unlink()
    parts_dir.rmdir()
    print(f"done: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
