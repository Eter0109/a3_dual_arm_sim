"""One compact proposal per round via a compatible API or optional Codex CLI."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .config import BOUNDS, read_json, write_json
from .process import run_job


def api_settings():
    base = os.environ.get("AUTOVLA_API_BASE_URL", "").rstrip("/")
    model = os.environ.get("AUTOVLA_API_MODEL", "")
    key = os.environ.get("AUTOVLA_API_KEY", "")
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in ("https", "http") or not parsed.netloc or parsed.query:
        raise ValueError("Set AUTOVLA_API_BASE_URL to the provider's API base URL (usually .../v1)")
    if parsed.username or parsed.password:
        raise ValueError("Do not put API credentials in the URL; use AUTOVLA_API_KEY")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Remote API endpoints require HTTPS")
    if not model or not key:
        raise ValueError("Set AUTOVLA_API_MODEL and AUTOVLA_API_KEY in the environment")
    mode = os.environ.get("AUTOVLA_API_JSON_MODE", "json_object")
    if mode not in ("json_object", "none"):
        raise ValueError("AUTOVLA_API_JSON_MODE must be json_object or none")
    return base, model, key, mode


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A redirect must not forward the provider credential to a different host.
        return None


def api_proposal(prompt, output, timeout):
    base, model, key, mode = api_settings()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return exactly one JSON experiment proposal. No tools."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1600,
    }
    if mode == "json_object":
        body["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            payload = json.loads(response.read(1_000_000))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Agent API returned HTTP {exc.code}; check provider settings") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Agent API connection failed or timed out") from None
    try:
        content = payload["choices"][0]["message"]["content"]
        # Some compatible providers wrap JSON in one Markdown fence.
        if content.startswith("```json\n") and content.rstrip().endswith("```"):
            content = content.strip()[8:-3].strip()
        proposal = json.loads(content)
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        raise ValueError("Agent API did not return a JSON proposal") from None
    write_json(output, proposal)
    write_json(
        output.with_suffix(".usage.json"),
        {
            "provider_host": urllib.parse.urlsplit(base).hostname,
            "model": model,
            "usage": payload.get("usage"),
        },
    )
    return proposal


def propose(kind, *, prompt, output, repo, schema, timeout, tested):
    if kind == "api":
        return api_proposal(prompt, output, timeout)
    if kind == "codex":
        run_job(
            [
                "codex",
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(schema),
                "--output-last-message",
                str(output),
                "-",
            ],
            cwd=repo,
            log=output.with_suffix(".log"),
            timeout=timeout,
            prompt=prompt,
        )
        return read_json(output)
    # Deterministic proposals solely for synthetic end-to-end tests.
    options = [("n_action_steps", 4), ("n_action_steps", 16), ("lr", 0.0001)]
    index = min(len(tested) - 1, len(options) - 1)
    parameter, value = options[index]
    proposal = {
        "hypothesis": "SYNTHETIC demonstration of keep/discard bookkeeping",
        "change": {"parameter": parameter, "value": value},
        "expected_effect": "Exercise the controller; not a robotics finding",
        "finding": "Synthetic results must never be interpreted as model performance",
        "stop": len(tested) > len(options),
    }
    write_json(output, proposal)
    return proposal


def make_prompt(program, state, schema):
    # Paths, API keys, videos and full training logs are not sent to the API.
    def public_metrics(metrics):
        return {k: v for k, v in metrics.items() if k != "representative_failures"}

    best = state["best_public"]
    best = {**best, "metrics": public_metrics(best["metrics"])} if best else None
    recent = [
        {
            k: (public_metrics(v) if k == "metrics" and v else v)
            for k, v in row.items()
            if k != "error"
        }
        for row in state["recent"]
    ]
    public_state = {
        "training_steps": state["training_steps"],
        "best": best,
        "recent_experiments": recent,
        "findings": state["findings"],
        "failure_report": state["failure_public"],
        "tested_configs": state["tested"],
    }
    return (
        program
        + "\nParameter bounds: "
        + json.dumps({k: v[:2] for k, v in BOUNDS.items()})
        + "\nResearch state:\n"
        + json.dumps(public_state, ensure_ascii=False)
        + "\nReturn JSON matching this schema:\n"
        + json.dumps(schema)
    )
