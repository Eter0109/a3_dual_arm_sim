"""Does pi0.5 keep a decodable text pathway, and did fine-tuning change it?

Two questions, answered from the weights first because that part is decisive and
cheap:

1. *Is a language pathway retained?*  The released pi0.5 code has no `generate`:
   `PI05Pytorch` exposes `sample_actions` (flow matching) and nothing else.  But
   the vision-language backbone is a complete PaliGemma with an `lm_head`, so the
   capacity to emit tokens is present even though no shipped code path uses it.
   That distinction matters -- "no text output" could mean the ability was
   removed during adaptation, or only that nobody calls it.

2. *Did our fine-tune disturb it?*  Our run trains with `train_expert_only=True`,
   which puts `self.paligemma` in eval mode and clears `requires_grad` on every
   one of its parameters, so the action expert learns while the backbone stays
   fixed.  If that is what happened, the backbone tensors must be bit-identical
   between `lerobot/pi05_base` and the fine-tuned checkpoint, and any text the
   backbone produces must be identical too.

   That is a falsifiable prediction rather than a description: a single differing
   tensor, or one differing token, would mean gradients reached the backbone after
   all and the fine-tune was not the run it was documented as.

The comparison is over the `paligemma` subtree, reported per group of layers so a
partial update localises instead of just failing.

    python probe_pi05_text.py --finetuned <checkpoint dir> [--generate]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

BASE = "lerobot/pi05_base"

#: Tensor-name prefixes.  These have to be exact: both subtrees live under
#: `paligemma_with_expert`, and a loose substring test for "paligemma" matches the
#: action expert's tensors as well, which would compare the trained part against
#: itself and miss the question entirely.
BACKBONE = "paligemma_with_expert.paligemma."
EXPERT = "paligemma_with_expert.gemma_expert."


def normalise(name: str) -> str:
    """Strip the wrapper prefix that only some checkpoints carry.

    `lerobot/pi05_base` ships bare names (`paligemma_with_expert.paligemma...`)
    while a saved training checkpoint nests them under the policy module
    (`model.paligemma_with_expert.paligemma...`).  Both describe the same weights,
    so matching without stripping reports every tensor as present only on one
    side -- which is what a first pass at this comparison did.
    """
    return name[len("model."):] if name.startswith("model.") else name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--generate", action="store_true", help="Also greedily decode text from the prefix")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/autodl-tmp/datasets/a3_b1/shard_00"))
    parser.add_argument("--step", type=int, default=0, help="Dataset frame to feed the backbone")
    parser.add_argument("--task", default="grasp five yellow blocks together from the source bin")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def shard_files(path: Path) -> list[Path]:
    if path.suffix == ".safetensors":
        return [path]
    single = path / "model.safetensors"
    if single.is_file():
        return [single]
    return sorted(path.glob("*.safetensors"))


def load_state(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    files = shard_files(path)
    if not files:
        raise SystemExit(f"no safetensors found under {path}")
    state: dict[str, torch.Tensor] = {}
    for f in files:
        state.update({normalise(k): v for k, v in load_file(str(f)).items()})
    return state


def group_of(name: str) -> str:
    """Coarse layer group, so a difference localises rather than just failing."""
    if "vision_tower" in name:
        return "vision_tower (SigLIP)"
    if "multi_modal_projector" in name:
        return "multi_modal_projector"
    if "language_model" in name:
        if "embed_tokens" in name or "lm_head" in name:
            return "language_model embeddings/head"
        return "language_model transformer"
    return "other"


def tied_alias(name: str) -> str | None:
    """The other name of a tied weight, or None.

    PaliGemma ties the input embedding to the output head, so the two are the same
    tensor.  Whether a checkpoint stores both, one, or the other depends on the
    saving code, and a plain key comparison then reports a difference that is not
    one -- which is exactly what happened here: base saves only `lm_head.weight`,
    the training checkpoint saves only `embed_tokens.weight`, and both are the same
    matrix.

    The two live at different depths -- `paligemma.lm_head` sits on the
    conditioning model, while the embedding table sits under
    `paligemma.model.language_model` -- so the swap is not a plain rename.
    """
    head = ".lm_head.weight"
    table = ".model.language_model.embed_tokens.weight"
    if name.endswith(table):
        return name[: -len(table)] + head
    if name.endswith(head):
        return name[: -len(head)] + table
    return None


def resolve_missing(names: list[str], other: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    """Split missing keys into genuinely missing and tied aliases of a stored one."""
    genuinely: list[str] = []
    tied: list[str] = []
    for name in names:
        alias = tied_alias(name)
        if alias and alias in other:
            tied.append(name)
        else:
            genuinely.append(name)
    return genuinely, tied


def compare(base: dict[str, torch.Tensor], tuned: dict[str, torch.Tensor]) -> dict:
    base_pali = {k: v for k, v in base.items() if k.startswith(BACKBONE)}
    tuned_pali = {k: v for k, v in tuned.items() if k.startswith(BACKBONE)}
    only_base_raw = sorted(set(base_pali) - set(tuned_pali))
    only_tuned_raw = sorted(set(tuned_pali) - set(base_pali))
    only_base, tied_base = resolve_missing(only_base_raw, tuned_pali)
    only_tuned, tied_tuned = resolve_missing(only_tuned_raw, base_pali)
    tied_names = sorted(set(tied_base) | set(tied_tuned))

    # A tied weight stored under one name on each side still has to hold the same
    # numbers; compare the pairs rather than trusting the names.
    tied_mismatch: list[str] = []
    for name in tied_names:
        alias = tied_alias(name)
        this = base_pali.get(name, tuned_pali.get(name))
        that = tuned_pali.get(alias) if name in base_pali else base_pali.get(alias)
        if this is None or that is None or this.shape != that.shape:
            tied_mismatch.append(name)
        elif (this.to(torch.float64) - that.to(torch.float64)).abs().max().item() > 0:
            tied_mismatch.append(name)

    per_group: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for name in sorted(set(base_pali) & set(tuned_pali)):
        a, b = base_pali[name], tuned_pali[name]
        if a.shape != b.shape:
            per_group[group_of(name)].append((name, float("inf")))
            continue
        # Compare in float64: the tensors are bf16, and a bf16 subtraction of two
        # equal values is exactly zero, so this cannot miss an update by rounding.
        diff = (a.to(torch.float64) - b.to(torch.float64)).abs().max().item()
        per_group[group_of(name)].append((name, diff))
    return {
        "per_group": per_group,
        "only_base": only_base,
        "only_tuned": only_tuned,
        "tied": tied_names,
        "tied_mismatch": tied_mismatch,
    }


def report(comparison: dict) -> tuple[bool, dict]:
    per_group = comparison["per_group"]
    print(f"{'group':<34} {'tensors':>8} {'max |diff|':>12}  status")
    print("-" * 74)
    identical_everywhere = True
    summary = {}
    for group in sorted(per_group):
        entries = per_group[group]
        worst = max(value for _, value in entries)
        moved = [name for name, value in entries if value > 0]
        status = "identical" if not moved else f"{len(moved)} tensor(s) differ"
        identical_everywhere &= not moved
        print(f"{group:<34} {len(entries):>8} {worst:>12.3e}  {status}")
        summary[group] = {"tensors": len(entries), "max_abs_diff": worst, "differing": len(moved)}
        for name in moved[:5]:
            print(f"      {name}")
    if comparison["only_base"]:
        identical_everywhere = False
        print(f"\n{len(comparison['only_base'])} tensor(s) only in the base checkpoint, e.g. "
              f"{comparison['only_base'][:3]}")
    if comparison["only_tuned"]:
        identical_everywhere = False
        print(f"\n{len(comparison['only_tuned'])} tensor(s) only in the fine-tuned checkpoint, e.g. "
              f"{comparison['only_tuned'][:3]}")
    if comparison["tied"]:
        mismatch = comparison["tied_mismatch"]
        note = "values match" if not mismatch else f"VALUES DIFFER: {mismatch}"
        print(f"\n{len(comparison['tied'])} tied lm_head/embed_tokens weight stored under one name on"
              f" each side ({note})")
        identical_everywhere &= not mismatch
    return identical_everywhere, summary


def greedy_from_prefix(policy, batch: dict, max_tokens: int, device: str) -> tuple[str, list[int]]:
    """Decode text autoregressively from the policy's own prefix.

    The shipped code never runs the backbone for generation, so this drives it by
    hand: embed the prefix exactly as training does (images through SigLIP, text
    tokens through the embedding table with the same sqrt(d) scaling), then take
    the argmax of the next-token distribution and feed it back in.

    Greedy rather than sampled on purpose.  The question is what this backbone
    produces for our scene, and sampling would replace an answer with a
    distribution over answers.
    """
    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    tokens = batch["observation.language.tokens"]
    masks = batch["observation.language.attention_mask"]

    language_model = model.paligemma_with_expert.paligemma.model.language_model
    head = model.paligemma_with_expert.paligemma.lm_head

    with torch.inference_mode():
        embs, pad_masks, _ = model.embed_prefix(images, img_masks, tokens, masks)
        generated: list[int] = []
        for _ in range(max_tokens):
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            outputs = language_model(
                inputs_embeds=embs,
                attention_mask=pad_masks,
                position_ids=position_ids,
                use_cache=False,
            )
            next_id = int(head(outputs.last_hidden_state)[0, -1].argmax())
            generated.append(next_id)
            next_emb = language_model.embed_tokens(torch.tensor([[next_id]], device=device))
            next_emb = next_emb * (next_emb.shape[-1] ** 0.5)
            embs = torch.cat([embs, next_emb], dim=1)
            pad_masks = torch.cat(
                [pad_masks, torch.ones((1, 1), dtype=pad_masks.dtype, device=device)], dim=1
            )
    return generated


class _Request:
    """The subset of the bridge's Request the adapter reads.

    A local stand-in for the same reason `eval_pi05_openloop.py` uses one: the
    bridge's own dataclass lives on the A3 side, and the adapter only ever calls
    `.task`, `.chunk`, `.image()` and `.vector()`.  The camera names are the
    adapter's own keys -- `front_view`, not `observation.images.front` -- because
    `CAMERAS` in the adapter maps the LeRobot feature names onto bridge names.
    """

    def __init__(self, arrays: dict, task: str, chunk: int) -> None:
        self._arrays = arrays
        self.task = task
        self.chunk = chunk

    def image(self, name: str) -> "np.ndarray":
        return self._arrays[name]

    def vector(self, name: str) -> "np.ndarray":
        return self._arrays[name]


def build_batch(dataset_root: Path, step: int, task: str, device: str, checkpoint: Path) -> dict:
    """Reproduce the adapter's preprocessing so the probe sees the model's real input.

    The batch has to come from the adapter rather than being assembled here: the
    prompt includes the state discretized into 256 bins, and the normalizer has to
    be the one carrying this branch's dataset statistics.  Building it by hand
    would feed the backbone a different prefix than inference does, and the text it
    produced would describe that prefix instead of the real one.
    """
    import sys

    sys.path.insert(0, "/root/autodl-tmp/a3-collect")
    import numpy as np
    from adapters import pi05_a3  # type: ignore[import-not-found]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/a3", root=dataset_root, video_backend="pyav")
    frame = dict(dataset[step])
    arrays = {
        f"{name}_view": np.transpose(frame[f"observation.images.{name}"].numpy(), (1, 2, 0)) * 255.0
        for name in ("front", "left_wrist", "right_wrist")
    }
    arrays = {k: v.clip(0, 255).astype(np.uint8) for k, v in arrays.items()}
    arrays["observation.state"] = frame["observation.state"].numpy().astype(np.float64)
    request = _Request(arrays, task, 50)

    model = pi05_a3.load({"ckpt": str(checkpoint), "device": device})
    batch = pi05_a3._images(request, model)
    batch["observation.state"] = torch.as_tensor(
        pi05_a3._state_vector(request, model)[None, :], dtype=torch.float32
    )
    batch["task"] = [task]
    return {**model["preprocess"](batch), "_policy": model["policy"]}


def decode_tokens(ids: list[int]) -> str:
    """Turn generated ids into text with pi0.5's own tokenizer.

    Loaded from the local snapshot rather than by name.  `google/paligemma-3b-pt-224`
    is gated, so fetching it by name fails on an unauthenticated machine even though
    the files are already on disk -- the same problem
    `scripts/setup_paligemma_tokenizer_cache.py` exists to solve, and the snapshot it
    prepares is what this reads.  `local_files_only` keeps it from trying the
    network first.
    """
    from transformers import AutoTokenizer

    snapshot = Path("/root/autodl-tmp/hf/hub/models--google--paligemma-3b-pt-224/snapshots")
    candidates = sorted(p for p in snapshot.glob("*") if (p / "tokenizer.json").is_file())
    if not candidates:
        raise SystemExit(
            f"no paligemma tokenizer snapshot under {snapshot}; run "
            "scripts/setup_paligemma_tokenizer_cache.py first"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(candidates[0]), local_files_only=True)
    return tokenizer.decode(ids, skip_special_tokens=True)


def main() -> int:
    args = parse_args()
    ckpt: Path = args.finetuned.expanduser().resolve()
    print(f"base      : {args.base}")
    print(f"fine-tuned: {ckpt}\n")

    base_state = load_state(Path(args.base))
    tuned_state = load_state(ckpt)
    print("loading weights ...", flush=True)

    comparison = compare(base_state, tuned_state)
    identical, summary = report(comparison)

    print()
    if identical:
        print("VERDICT: the vision-language backbone is bit-identical between base and fine-tuned.")
        print("         `train_expert_only=True` froze it as documented; whatever text this")
        print("         backbone can produce, our fine-tune did not change it. Comparing")
        print("         generations across the two checkpoints would therefore show nothing new.")
    else:
        print("VERDICT: the backbone differs, so gradients reached it despite")
        print("         `train_expert_only=True`; the fine-tune was not the documented run.")

    tuned_pali = sum(1 for k in tuned_state if k.startswith(BACKBONE))
    tuned_expert = sum(1 for k in tuned_state if k.startswith(EXPERT))
    print(f"\ntensor counts in the fine-tuned checkpoint: backbone {tuned_pali}, "
          f"action expert {tuned_expert}")

    if args.generate:
        print("\ndriving the backbone by hand to see what it produces ...")
        print("(no shipped pi0.5 code path generates text; this is the probe's own loop)\n")
        device = "cuda"
        batch = build_batch(args.dataset_root, args.step, args.task, device, ckpt)
        policy = batch.pop("_policy")
        for label, source in (("fine-tuned", policy), ("base", None)):
            if source is None:
                import sys

                sys.path.insert(0, "/root/autodl-tmp/a3-collect")
                from adapters import pi05_a3  # type: ignore[import-not-found]

                source = pi05_a3.load({"ckpt": str(Path(args.base).parent), "device": device})["policy"]
            ids = greedy_from_prefix(source, batch, args.tokens, device)
            text = decode_tokens(ids)
            print(f"--- {label} ---")
            print(text)
            print(f"    token ids: {ids[:16]}{' ...' if len(ids) > 16 else ''}\n")
        del policy

    result = {
        "base": args.base,
        "finetuned": str(ckpt),
        "backbone_identical": identical,
        "groups": summary,
        "only_in_base": comparison["only_base"][:20],
        "only_in_finetuned": comparison["only_tuned"][:20],
        "tied_weights": comparison["tied"],
        "tied_mismatch": comparison["tied_mismatch"],
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
