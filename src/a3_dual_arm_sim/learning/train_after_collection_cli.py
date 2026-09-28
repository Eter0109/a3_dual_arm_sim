"""Wait for a collector to exit, validate finalized data, then train once."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from a3_dual_arm_sim.data.audit import audit_training_dataset
from a3_dual_arm_sim.paths import project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-pid", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    project = project_root()
    dataset, output = (project / args.root).resolve(), (project / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(stage, **kwargs):
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(dict(stage=stage, **kwargs), indent=2))
        temporary.replace(output / "status.json")

    try:
        if (output / "model").exists():
            raise FileExistsError("Existing model output; refusing duplicate training")
        status("waiting_for_collection", collector_pid=args.collector_pid)
        while True:
            try:
                os.kill(args.collector_pid, 0)
            except ProcessLookupError:
                break
            stat = Path(f"/proc/{args.collector_pid}/stat").read_text()
            if stat.rsplit(")", 1)[1].split()[0] == "Z":
                break
            time.sleep(30)
        status("auditing")
        audit = audit_training_dataset(dataset, repo_id=args.repo_id)
        if audit["episodes"] != 100:
            raise RuntimeError(f"Expected 100 successful episodes, found {audit['episodes']}")
        episodes = [
            json.loads(s)
            for s in (dataset / "a3_episode_metadata.jsonl").read_text().splitlines()
            if s
        ]
        if sorted(r["episode_index"] for r in episodes) != list(range(100)):
            raise RuntimeError("Episode IDs are not unique and contiguous")
        if len({r["seed"] for r in episodes}) != 100:
            raise RuntimeError("Duplicate scene seeds")
        tables = [pq.read_table(p) for p in (dataset / "meta/episodes").rglob("*.parquet")]
        if sum(len(t) for t in tables) != 100:
            raise RuntimeError("Episode parquet metadata is incomplete")
        if sum(r["frames"] for r in episodes) != audit["frames"]:
            raise RuntimeError("Frame counts differ")
        (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2))
        command = [
            sys.executable,
            "-u",
            "-m",
            "a3_dual_arm_sim.cli",
            "train-smolvla",
            "--root",
            str(dataset),
            "--repo-id",
            args.repo_id,
            "--output",
            str(output / "model"),
            "--steps",
            "20000",
            "--batch-size",
            "64",
            "--lr",
            "0.00005",
            "--device",
            "cuda",
        ]
        (output / "train_command.json").write_text(json.dumps(command, indent=2))
        status("training", steps=20000, batch_size=64, dataset=str(dataset))
        with (output / "train.log").open("x") as log:
            subprocess.run(
                command,
                cwd=project,
                env=dict(os.environ, HF_HUB_OFFLINE="1", MUJOCO_GL="egl"),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        status("training_complete", checkpoint=str(output / "model/checkpoints/020000"))
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
