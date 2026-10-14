"""Run one policy manifest over a list of seeds, several at a time, and summarise it.

Why a driver rather than a shell loop
-------------------------------------
Three things a loop of `render_bridge_policy_video.py` calls cannot do, each of which cost
real time on the sweeps this replaces:

* **Share one model across the seeds.**  A rollout is a CPU-bound simulation that asks the
  model for a chunk every few control steps, so the model process is idle most of the time.
  Measured on this machine: physics is ~80% of a step, the inference round trip ~5%, and the
  GPU is busy under a tenth of the time.  One model process per rollout therefore wastes the
  card, and cannot be scaled past roughly three -- 10 GB of weights each on a 32 GB card --
  however idle that card is.  With ``--serve-checkpoint`` the driver starts one listening
  bridge (``a3_bridge_server.py --listen``) and points every worker at it, so the number of
  parallel rollouts is a CPU decision rather than a GPU-memory one.  Eight was measured to
  scale: eight concurrent rollouts each ran at the same per-step rate as one.

* **Resume.**  A seed is the unit of work, and its result is one line in its log, so a sweep
  that is interrupted (or extended with more seeds) re-runs only what is missing.  The
  sweeps this replaces were re-driven by hand after a disk filled, and the alternative --
  re-running everything -- costs hours.

* **Summarise.**  The number a sweep is read for is the distribution of ``in target X/10``
  across seeds, which is three lines of arithmetic that were being done by grep and eye.

Usage
-----
::

    python scripts/sweep_policy_seeds.py \\
        --manifest configs/policies/pi05_a3_b2_unfreeze_20k.yaml \\
        --scene a3_cookie_batch_second_batch \\
        --seeds 0-9 --workers 8 --steps 1800 \\
        --serve-checkpoint /root/autodl-tmp/pi05-runs/.../checkpoints/020000/pretrained_model \\
        --serve-adapter /root/autodl-tmp/a3-collect/adapters/pi05_a3.py \\
        --output-dir /root/autodl-tmp/videos --log-dir /root/autodl-tmp/logs

Without ``--serve-checkpoint`` the driver still works: each worker then starts its own
bridge from the manifest, which is what a manifest with ``transport.kind: subprocess``
means, and the concurrency is bounded by GPU memory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / "scripts" / "render_bridge_policy_video.py"
BRIDGE_SERVER = ROOT / "scripts" / "a3_bridge_server.py"

#: The line the renderer prints for an episode's outcome, e.g. ``in target 5/10``.
OUTCOME = re.compile(r"^final\s+: in target (\d+)/(\d+)\s+in source (\d+)\s*$", re.MULTILINE)


def parse_seeds(spec: str) -> list[int]:
    """``"0-9"``, ``"0,2,5"`` or a mixture, in the order given, without repeats."""

    seeds: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    seen: set[int] = set()
    ordered = [seed for seed in seeds if not (seed in seen or seen.add(seed))]
    if not ordered:
        raise SystemExit(f"--seeds {spec!r} selected nothing")
    return ordered


def read_result(log: Path, total: int | None = None) -> dict[str, int] | None:
    """The episode outcome from a finished render log, or ``None`` if it did not finish.

    Read from the ``final`` line rather than the per-50-step progress lines: those are a
    progress report, and reading one as the result is how a successful roll-out was once
    recorded as a failure.
    """

    if not log.is_file():
        return None
    match = None
    for match in OUTCOME.finditer(log.read_text(encoding="utf-8", errors="replace")):
        pass
    if match is None:
        return None
    placed, required, source = (int(group) for group in match.groups())
    if total is not None and required != total:
        # A scene whose contract changed under an old log; treat it as absent rather than
        # silently comparing counts that are about different boxes.
        return None
    return {"in_target": placed, "required": required, "in_source": source}


def socket_url_for(path_text: str | None, scratch: Path | None = None) -> tuple[str, Path | None]:
    """A listen URL, and the directory to remove afterwards when one was created here."""

    if path_text:
        return path_text, None
    directory = scratch if scratch is not None else Path(tempfile.mkdtemp(prefix="a3-sweep-"))
    directory.mkdir(parents=True, exist_ok=True)
    return f"unix://{directory / 'bridge.sock'}", directory


def wait_for_listen(url: str, timeout: float = 1800.0) -> None:
    """Block until the bridge is accepting connections.

    The server loads the model *before* it listens, so a successful connection means the
    weights are in memory -- which is the only readiness signal that matters here, since the
    first rollout to arrive would otherwise pay the load inside its own request timeout.
    """

    if url.startswith("unix://"):
        address: Any = url[len("unix://") :]
        family = socket.AF_UNIX
    elif url.startswith("tcp://"):
        host, _, port = url[len("tcp://") :].partition(":")
        address = (host or "127.0.0.1", int(port))
        family = socket.AF_INET
    else:
        raise SystemExit(f"unsupported bridge url {url!r}; use unix:// or tcp://")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = socket.socket(family, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect(address)
            probe.close()
            return
        except OSError:
            probe.close()
            time.sleep(0.25)
    raise SystemExit(f"the shared bridge never started listening on {url}")


def write_socket_manifest(manifest: Path, url: str, into: Path) -> Path:
    """The manifest with its transport replaced by the shared bridge.

    Written into the run's log directory, not beside the original: it is a generated artefact
    of one invocation, and the two differ in exactly one key, which is what makes a run's
    diff readable.
    """

    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    transport = dict(payload.get("transport") or {})
    payload["transport"] = {
        "kind": "socket",
        "url": url,
        # The load already happened, so this only has to cover a queue of concurrent
        # requests behind the model's lock -- generous, because a slow first forward on a
        # busy card is not a failure.
        "startup_timeout_s": 300,
        "request_timeout_s": 900,
        # Kept for provenance: the environment the bridge is *meant* to be run in, even
        # though a socket transport starts nothing itself.
        **({"env": transport["env"]} if "env" in transport else {}),
    }
    target = into / f"{manifest.stem}.shared.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def run_seed(
    *,
    seed: int,
    manifest: Path,
    scene: str,
    steps: int,
    view_size: int,
    output: Path,
    log: Path,
    extra_args: list[str],
) -> int:
    """One rollout, in its own process; returns the exit status."""

    command = [
        sys.executable,
        str(RENDER),
        "--manifest",
        str(manifest),
        "--scene",
        scene,
        "--output",
        str(output),
        "--steps",
        str(steps),
        "--view-size",
        str(view_size),
        "--seed",
        str(seed),
        *extra_args,
    ]
    with log.open("wb") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return completed.returncode


def summarise(results: dict[int, dict[str, int]], total: int) -> dict[str, Any]:
    """The numbers a sweep is read for: the distribution of ``in target`` across seeds."""

    placed = [entry["in_target"] for entry in results.values()]
    if not placed:
        return {"seeds": 0}
    ordered = sorted(placed)
    half = len(ordered) // 2
    median = (
        float(ordered[half])
        if len(ordered) % 2
        else (ordered[half - 1] + ordered[half]) / 2.0
    )
    return {
        "seeds": len(placed),
        "total": total,
        "mean": round(sum(placed) / len(placed), 3),
        "median": median,
        "min": min(placed),
        "max": max(placed),
        # "Placed something" and "placed at least five" are the two thresholds the previous
        # hand-tallied sweeps were read at, so they are what this reports.
        "placed_something": sum(1 for value in placed if value > 0),
        "at_least_five": sum(1 for value in placed if value >= 5),
        "distribution": {str(value): placed.count(value) for value in sorted(set(placed))},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seeds", default="0-9", help="e.g. 0-9, or 0,2,5, or 0-3,7")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--view-size", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--tag", default=None, help="Names the outputs; defaults to the manifest stem")
    parser.add_argument(
        "--total",
        type=int,
        default=10,
        help="Cookies the scene asks for, for cross-checking an old log's contract",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run seeds that already have a result"
    )
    parser.add_argument(
        "--serve-checkpoint",
        type=Path,
        default=None,
        help="Start one shared bridge for this checkpoint and point every worker at it",
    )
    parser.add_argument("--serve-adapter", type=Path, default=None)
    parser.add_argument("--serve-config", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--serve-cwd", type=Path, default=None)
    parser.add_argument(
        "--socket",
        default=None,
        help="Listen URL for the shared bridge (default: a fresh unix socket)",
    )
    parser.add_argument(
        "--extra",
        default="",
        help="Extra arguments passed through to the renderer, shell-quoted",
    )
    args = parser.parse_args(argv)

    extra_args = shlex.split(args.extra)
    tag = args.tag or args.manifest.stem
    seeds = parse_seeds(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest
    server: subprocess.Popen[bytes] | None = None
    server_log: Path | None = None
    scratch: Path | None = None
    try:
        if args.serve_checkpoint is not None:
            if args.serve_adapter is None:
                raise SystemExit("--serve-checkpoint needs --serve-adapter")
            url, scratch = socket_url_for(args.socket, args.log_dir / f"{tag}-socket")
            if url.startswith("unix://"):
                socket_path = Path(url[len("unix://") :])
                socket_path.parent.mkdir(parents=True, exist_ok=True)
                with suppress(FileNotFoundError):
                    socket_path.unlink()
            server_log = args.log_dir / f"{tag}.bridge-server.log"
            command = [
                sys.executable,
                str(BRIDGE_SERVER),
                "--adapter",
                str(args.serve_adapter),
                "--listen",
                url,
                *[item for pair in args.serve_config for item in ("--config", pair)],
            ]
            # The manifest's own `transport.env` is the environment this model needs
            # (`HF_HOME`, `HF_HUB_OFFLINE`, ...).  A socket transport starts nothing itself,
            # so that block is the only place the requirement is written down, and the
            # server has to be given it here -- otherwise it fails to find the weights for a
            # reason that has nothing to do with the weights.
            server_env = {**os.environ}
            declared = (yaml.safe_load(args.manifest.read_text(encoding="utf-8")).get("transport") or {}).get("env")
            server_env.update({str(key): str(value) for key, value in (declared or {}).items()})
            print(f"bridge     : {' '.join(command)}", flush=True)
            with server_log.open("wb") as handle:
                server = subprocess.Popen(
                    command,
                    cwd=str(args.serve_cwd) if args.serve_cwd else None,
                    env=server_env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            try:
                wait_for_listen(url, timeout=1800.0)
            except SystemExit:
                print(f"the bridge did not come up; see {server_log}", file=sys.stderr)
                raise
            print(f"bridge     : listening on {url} (model loaded once)", flush=True)
            manifest = write_socket_manifest(args.manifest, url, args.log_dir)
            print(f"manifest   : {manifest}", flush=True)

        done: dict[int, dict[str, int]] = {}
        if not args.force:
            for seed in seeds:
                existing = read_result(args.log_dir / f"{tag}_seed{seed}.log", total=args.total)
                if existing is not None:
                    done[seed] = existing
        pending = [seed for seed in seeds if seed not in done]

        print(f"sweep      : {len(seeds)} seeds, {len(done)} already have results, {len(pending)} to run")
        print(f"scene      : {args.scene}   steps {args.steps}   workers {min(args.workers, max(len(pending), 1))}")
        print(flush=True)

        started = time.perf_counter()
        failures: dict[int, int] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    run_seed,
                    seed=seed,
                    manifest=manifest,
                    scene=args.scene,
                    steps=args.steps,
                    view_size=args.view_size,
                    output=args.output_dir / f"{tag}_seed{seed}.mp4",
                    log=args.log_dir / f"{tag}_seed{seed}.log",
                    extra_args=extra_args,
                ): seed
                for seed in pending
            }
            # Not `as_completed`: the seeds are reported in the order they finish, which for a
            # sweep of tens of minutes is the difference between watching it and waiting for it.
            for future in concurrent.futures.as_completed(futures):
                seed = futures[future]
                status = future.result()
                if status != 0:
                    failures[seed] = status
                    print(f"seed {seed:>3}  FAILED (exit {status})", flush=True)
                    continue
                result = read_result(args.log_dir / f"{tag}_seed{seed}.log", total=args.total)
                if result is None:
                    failures[seed] = 0
                    print(f"seed {seed:>3}  no outcome in the log", flush=True)
                    continue
                done[seed] = result
                elapsed = time.perf_counter() - started
                print(
                    f"seed {seed:>3}  in target {result['in_target']}/{result['required']}"
                    f"   in source {result['in_source']}"
                    f"   [{len(done)}/{len(seeds)}  {elapsed / 60:.1f} min]",
                    flush=True,
                )
        wall = time.perf_counter() - started
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        elif args.socket and args.socket.startswith("unix://"):
            socket_path = Path(args.socket[len("unix://") :])
            with suppress(FileNotFoundError):
                socket_path.unlink()

    print()
    stats = summarise(done, args.total)
    width = max((len(str(seed)) for seed in done), default=1)
    for seed in sorted(done):
        entry = done[seed]
        print(f"  seed {seed:>{width}}   {entry['in_target']:>2}/{entry['required']}")
    if stats.get("seeds"):
        print()
        print(
            f"  seeds {stats['seeds']}   mean {stats['mean']}   median {stats['median']}   "
            f"range {stats['min']}-{stats['max']}"
        )
        print(
            f"  placed something {stats['placed_something']}/{stats['seeds']}   "
            f"at least five {stats['at_least_five']}/{stats['seeds']}"
        )
        print(f"  distribution {stats['distribution']}")
    if failures:
        print(f"\n  failed seeds: {sorted(failures)}")
    print(f"\n  wall {wall / 60:.1f} min   logs {args.log_dir}")

    summary_path = args.log_dir / f"{tag}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "scene": args.scene,
                "steps": args.steps,
                "workers": args.workers,
                "shared_bridge": str(args.serve_checkpoint) if args.serve_checkpoint else None,
                "results": {str(seed): done[seed] for seed in sorted(done)},
                "stats": stats,
                "failures": {str(seed): status for seed, status in sorted(failures.items())},
                "wall_minutes": round(wall / 60, 2),
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  summary {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
