"""Independent CPU environments sharing one SmolVLA model and isolated inference state."""

from __future__ import annotations

import contextlib
import copy
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from a3_dual_arm_sim.contracts import EpisodeContext


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class InferenceStream:
    """Share parameter storage, while keeping action queues and random streams private."""

    def __init__(self, base, seed, task):
        import torch

        self.torch = torch
        self.devices = []
        if base._device.type == "cuda":
            self.devices = [base._device.index or torch.cuda.current_device()]
        self.policy = copy.copy(base)
        self.policy._policy = copy.copy(base._policy)
        self.policy.env = None
        self.policy.inference_seed = seed
        with torch.random.fork_rng(devices=self.devices):
            self.policy.reset(EpisodeContext(seed=seed, task=task, action_mode="joint_position"))
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(self.devices[0]) if self.devices else None
        if self.policy._policy._queues is base._policy._queues:
            raise RuntimeError("SmolVLA reset did not allocate independent action queues")

    def act(self, observation, task):
        with self.torch.random.fork_rng(devices=self.devices):
            self.torch.set_rng_state(self.cpu_rng)
            if self.devices:
                self.torch.cuda.set_rng_state(self.cuda_rng, self.devices[0])
            action = self.policy.act(observation, task)
            self.cpu_rng = self.torch.get_rng_state()
            if self.devices:
                self.cuda_rng = self.torch.cuda.get_rng_state(self.devices[0])
        return action


def _inference_server(options, output, requests, replies, events):
    from a3_dual_arm_sim.workflows.policy_adapters import SmolVLAPolicyAdapter

    adapter = None
    with (
        (Path(output) / "inference.log").open("a", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            import torch

            torch.set_num_threads(1)
            adapter = SmolVLAPolicyAdapter(**options)
            clients = {}
            events.put({"kind": "inference_ready", "pid": os.getpid()})
            while True:
                request = requests.get()
                if request is None:
                    break
                worker = request["worker"]
                if request["operation"] == "reset":
                    clients[worker] = InferenceStream(
                        adapter.plugin, request["seed"], request["task"]
                    )
                    result = {"ok": True}
                else:
                    result = {
                        "ok": True,
                        "action": clients[worker].act(request["observation"], request["task"]),
                    }
                replies[worker].put(result)
        except BaseException:
            error = traceback.format_exc()
            events.put({"kind": "error", "error": error})
            for reply in replies:
                reply.put({"ok": False, "error": error})
            raise
        finally:
            if adapter is not None:
                adapter.close()


class RemotePolicy:
    """Send only checkpoint inputs to the model; track privileged counts separately."""

    action_mode = "joint_position"
    name = "smolvla"

    def __init__(self, worker, requests, reply, input_keys, progress, trace):
        self.worker, self.requests, self.reply = worker, requests, reply
        self.input_keys, self.progress, self.trace = input_keys, progress, trace

    def _request(self, message):
        self.requests.put({"worker": self.worker, **message})
        result = self.reply.get(timeout=300)
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result

    def bind_env(self, env):
        original_step = env.step

        def step(action):
            result = original_step(action)
            info = result[4]
            self.current_count = int(info["cookies_in_target"])
            self.peak_count = max(self.peak_count, self.current_count)
            self.source_count = int(info["cookies_in_source"])
            # These diagnostics never enter policy observations or affect its actions.
            if self.calls == 1 or self.calls % 10 == 0:
                self.trace.write(
                    json.dumps(
                        {
                            "step": self.calls,
                            "cookies_in_target": self.current_count,
                            "peak_cookies_in_target": self.peak_count,
                            "cookies_in_source": self.source_count,
                            "state": result[0]["observation.state"].tolist(),
                            "action": np.asarray(action).tolist(),
                        }
                    )
                    + "\n"
                )
                self.trace.flush()
            return result

        env.step = step

    def reset(self, context):
        self.calls = self.peak_count = self.current_count = 0
        self.source_count = 80
        self.started, self.last_report = time.monotonic(), 0.0
        self._request({"operation": "reset", "seed": context.seed, "task": context.task})

    def act(self, observation, task=""):
        self.calls += 1
        now = time.monotonic()
        if self.calls == 1 or now - self.last_report >= 30:
            _write_json(
                self.progress,
                {
                    "pid": os.getpid(),
                    "stage": "running",
                    "step": self.calls,
                    "wall_seconds": round(now - self.started, 1),
                    "cookies_in_target": self.current_count,
                    "peak_cookies_in_target": self.peak_count,
                    "cookies_in_source": self.source_count,
                    "task_instruction": task,
                },
            )
            self.last_report = now
        inputs = {name: observation[name] for name in self.input_keys}
        return self._request({"operation": "act", "observation": inputs, "task": task})["action"]


def _simulation_worker(worker, jobs, events, requests, reply, input_keys, output):
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    output = Path(output)
    with (
        (output / f"worker_{worker}.log").open("a", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            while True:
                job = jobs.get()
                if job is None:
                    break
                stem = f"{job['benchmark']}_{job['profile']}_{job['seed']}"
                _write_json(output / f"worker_{worker}_current.json", job)
                with (output / "traces" / f"{stem}.jsonl").open("w") as trace:
                    policy = RemotePolicy(
                        worker, requests, reply, input_keys, output / f"worker_{worker}.json", trace
                    )
                    benchmark = make_cookie_benchmark(job["benchmark"], **job["options"])
                    result = benchmark.evaluate(
                        policy, num_episodes=1, seed_start=job["seed"], workers=1, verbose=False
                    )
                    row = {
                        **asdict(result.episodes[0]),
                        "episode": job["episode"],
                        "benchmark": job["benchmark"],
                        "profile": job["profile"],
                        "peak_cookies_in_target": policy.peak_count,
                    }
                events.put({"kind": "result", "row": row})
            _write_json(output / f"worker_{worker}.json", {"stage": "completed"})
        except BaseException:
            events.put({"kind": "error", "error": traceback.format_exc()})
            raise


def run_parallel_grid(options, jobs, workers, output, on_episode_end):
    """Run a finite job list. Completed rows are persisted by the caller callback."""
    if not jobs:
        return
    if workers < 1:
        raise ValueError("workers must be positive")
    workers = min(workers, len(jobs))
    output = Path(output)
    (output / "traces").mkdir(exist_ok=True)
    input_keys = tuple(
        json.loads((Path(options["checkpoint"]) / "config.json").read_text())["input_features"]
    )
    context = mp.get_context("spawn")
    tasks, events, requests = context.Queue(), context.Queue(), context.Queue()
    replies = [context.Queue() for _ in range(workers)]
    for job in jobs:
        tasks.put(job)
    for _ in range(workers):
        tasks.put(None)
    inference = context.Process(
        target=_inference_server, args=(options, str(output), requests, replies, events)
    )
    simulations = []
    completed = 0
    try:
        inference.start()
        event = events.get(timeout=300)
        if event["kind"] != "inference_ready":
            raise RuntimeError(event.get("error", "Shared model failed to initialize"))
        _write_json(output / "inference_pid.json", {"pid": event["pid"], "model_instances": 1})
        for worker in range(workers):
            process = context.Process(
                target=_simulation_worker,
                args=(worker, tasks, events, requests, replies[worker], input_keys, str(output)),
            )
            process.start()
            simulations.append(process)
        while completed < len(jobs):
            try:
                event = events.get(timeout=5)
            except queue.Empty:
                event = None
            if event is not None:
                if event["kind"] == "error":
                    raise RuntimeError(event["error"])
                on_episode_end(event["row"])
                completed += 1
            if inference.exitcode is not None or any(
                process.exitcode not in (None, 0) for process in simulations
            ):
                raise RuntimeError("A grid subprocess exited unexpectedly; inspect worker logs")
        requests.put(None)
    finally:
        for process in [*simulations, inference]:
            if completed < len(jobs) and process.is_alive():
                process.terminate()
        for process in [*simulations, inference]:
            if process.pid is not None:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
        for channel in [tasks, events, requests, *replies]:
            channel.cancel_join_thread()
            channel.close()
