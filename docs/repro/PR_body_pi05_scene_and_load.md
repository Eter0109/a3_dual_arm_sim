# pi0.5: `--pi05-scene-mode full` is required, and a failed weight load must not score

Reproducing the released π0.₅ checkpoint (`yunsx114/a3-dual-arm-cookie-pi05-aligned`) through
`--policy pi05:<ckpt>` on current `main` gives **0/10** while reporting no error. Two independent
causes, both silent. This PR fixes both, and states plainly that **`--pi05-scene ... --pi05-scene-mode
full` is required** — it is not an optional comparison mode.

---

## 1. A failed load returns a random model, and the benchmark scores it

`PI05Policy.from_pretrained` wraps the whole load in `try/except`: an unexpected key, a missing key or
a version mismatch all end in

```
Warning: Could not load state dict: ...
```

followed by `return model` — a **freshly initialised** network. It runs happily for the full budget
and produces a real-looking score.

The released checkpoint triggers exactly that. It stores a 16→2048 state projection under the `model.`
prefix, which π₀.₅ has no module for. lerobot's converter does drop that pair — but only in its
unprefixed form:

```python
if key.startswith("state_proj."):        # never matches "model.state_proj."
    logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
    continue
```

so `model.state_proj.{weight,bias}` survive, `strict=True` raises, and the exception is swallowed.

**Fix.** `_skip_prefixed_state_proj()` filters the pair inside the converter, keeping `strict=True` so
lerobot itself confirms the load (`All keys loaded successfully!`). `_assert_weights_loaded()`
additionally compares a sample of tensors against the safetensors file: a loader that catches
exceptions is not evidence that a load happened.

The `_convert_openpi_state_dict` fast path remains primary. The fallback only engages on builds
without it — at least one `lerobot==0.5.1` wheel on PyPI lacks the method while reporting the same
version, which is how this surfaced.

## 2. `--pi05-scene-mode full` is required; `cameras` does not help

Same checkpoint, same seed, task 2x5, on this code:

| flags | score |
| --- | --- |
| `--pi05-scene <fp> --pi05-scene-mode full` | **5/10** (repeat run: **10/10**, see below) |
| no `--pi05-scene` (main's default scene) | **0/10** |
| `--pi05-scene <fp> --pi05-scene-mode cameras` | **0/10** |

`cameras` cannot help because the static geometry already agrees — measured directly:

| quantity | difference |
| --- | --- |
| `source_bin` position | 0.0 mm |
| `target_bin` position | 0.0 mm (3.5 mm z = body vs mocap reference point) |
| all 80 Cookie positions | mean 0.22 mm, max 0.38 mm |
| front / left-wrist / right-wrist intrinsics (pos + fovy) | identical |
| 16-D reset state | identical |

What differs is what `full` re-derives from the reconstructed joints — the **wrist-camera aim**:

| camera | pixels differing | MAE |
| --- | --- | --- |
| front | 12.3 % | 3.6 / 255 |
| **left_wrist** | **31.4 %** | **18.6 / 255** |
| right_wrist | 0 % | 0 |

π₀.₅ has `n_obs_steps = 1` and no history, so every frame must lie in the distribution the training
scene produces. Given a wrist view it has never seen, the policy does not fail — it parks, and the
harness reports 0/10.

**Fix.** The requirement is stated in the module docstring with the numbers above, and the CLI prints
a warning naming `--pi05-scene × full`. It is deliberately **not** `parser.error`: loading on another
scene is a legitimate comparison, it simply is not comparable with the published number.

## Verification

```
--pi05-scene … --pi05-scene-mode full
  All keys loaded successfully!
  [01/01] Seed 0 | Score: 10/10 | Steps: 867 | Status: SUCCESS
```

and without the flags the same command now prints the scene warning before scoring.

## A note on 5/10 vs 10/10

The `full` configuration scored 5/10 twice and 10/10 once, all at `--max-steps 900`, same seed, same
`--inference-seed`. That is the known bf16 nondeterminism this adapter already documents
(`A3_PI05_DETERMINISTIC` / `A3_PI05_FP32`): bf16 GPU kernels are not bit-deterministic, and on a
contact-rich task a last-bit difference decides whether one batch lands. Anyone quoting a single
number here should quote the seed set and the step budget with it.

## Files

- `src/a3_dual_arm_sim/policies/pi05.py` — module docstring (the `full` requirement and its
  measurements), `_skip_prefixed_state_proj`, `_assert_weights_loaded`, dual-path loading.
- `src/a3_dual_arm_sim/workflows/benchmark_cookie_batch_cli.py` — the scene warning for `pi05:`.
