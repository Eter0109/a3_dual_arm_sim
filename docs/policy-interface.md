# Unified policy interface

Status: proposal, 2026-09-12 (VLA workstream)

## Goal

Run any LeRobot policy against the A3 environment through one interface, so that
switching models is a configuration change rather than a new plugin file.

## Where the interface sits

`EpisodeRunner` is the only thing that drives a policy. Scripted experts, keyboard
teleoperation and learned policies already share it, and its contract is the
`policy.Policy` protocol in `src/a3_dual_arm_sim/policy.py`:

```python
class Policy(Protocol):
    action_mode: ActionMode                      # "joint_position" for learned policies
    def reset(self, context: EpisodeContext) -> None: ...
    def act(self, observation: dict, task: str) -> np.ndarray: ...
    def close(self) -> None: ...
```

The runner additionally reads four optional duck-typed attributes
(`recording`, `stop_requested`, `discard_requested`, `emergency_requested`), all
of which default to "keep going" when absent.

That protocol is the right seam: it is already what `evaluation.py`, `collection.py`
and the CLI consume. Nothing above it needs to change.

```
        a3-sim CLI / evaluation / collection          (unchanged)
                          │
                          ▼
                 EpisodeRunner                        (unchanged)
                          │  Policy protocol: action_mode / reset / act / close
                          ▼
        ┌─────────────────────────────────────┐
        │  LeRobotPolicyAdapter   (new, one)  │
        │    observation projection            │
        │    lerobot policy + pre/post proc.   │
        │    action adaptation (pad trim, …)   │
        └─────────────────────────────────────┘
                          │  lerobot Policy protocol: select_action(batch) / reset()
                          ▼
   make_policy + make_pre_post_processors   (lerobot's own factory)
                          │
      ┌────────┬──────────┼──────────┬──────────┐
      ▼        ▼          ▼          ▼          ▼
   smolvla   act     diffusion     pi05      pi0 / vqbet / …
```

## Why the current plugin does not generalise

`smolvla_policy.py` is the only learned policy today. It is one file per model,
and it hardcodes things that are properties of the *checkpoint*, not of A3:

| hardcoded in `smolvla_policy.py` | should be |
| --- | --- |
| SmolVLA-specific `prepare_observation_for_inference` import | lerobot's generic path |
| `action...reshape(16)` | action width read from the checkpoint |
| `return_uint8=True` | **not a parameter in lerobot 0.4.4** — raises `TypeError` today |
| camera keys taken from `config.input_features` | explicit, overridable mapping |
| device via `A3_SMOLVLA_*` env vars | shared configuration |

Adding ACT, Diffusion or pi05 should not mean copying this file.

## Verified LeRobot behaviour

Measured against the pinned `lerobot 0.4.4` in `envs/a3_sim` on 2026-09-12.

1. **One factory covers every policy.** `make_policy(cfg, ds_meta)` and
   `make_pre_post_processors(...)` work for all 16 registered types
   (`act`, `diffusion`, `groot`, `pi0`, `pi05`, `pi0_fast`, `sac`, `sarm`,
   `smolvla`, `tdmpc`, `vqbet`, `wall_x`, `xvla`, …). Verified importable:
   `smolvla`, `pi05`, `act`, `diffusion`, `pi0`, `vqbet`.

2. **`select_action` already keeps an action queue.** It calls
   `predict_action_chunk` only when the queue is empty and returns one action per
   call, transposing `n_action_steps` rows in. So A3's per-step `act()` maps 1:1,
   and the re-planning cadence is `n_action_steps` — no A3-side chunking needed.

3. **Key renaming is a first-class pipeline step.** Every preprocessor pipeline
   contains `rename_observations_processor` with a `rename_map`. Unmapped keys
   pass through unchanged. Mapping A3 camera names onto a checkpoint's expected
   names is therefore *configuration*, injected at build time.

4. **Inference does not need the dataset.** `make_pre_post_processors(...,
   dataset_stats=None)` succeeds: the checkpoint carries its own normalisation
   statistics (`policy_preprocessor_step_*_normalizer_processor.safetensors`).
   Only `make_policy` still wants a dataset metadata object, and only to derive
   features.

5. **`make_policy` takes features from the dataset, not the config.** It always
   overwrites `cfg.output_features` from `ds_meta.features` and only fills
   `cfg.input_features` when the config has none. **Consequence: the dataset
   passed here defines the action width the policy will emit**, so it must be the
   A3 dataset (16-D), not an unrelated one.

6. **Dimensionality is absorbed by padding.** Policy configs carry
   `max_state_dim` / `max_action_dim` (32 for SmolVLA and pi05). A3's 16-D state
   and 16-D action fit, so no per-model reshaping is required on the way in.

7. **`dtype` is model-specific.** `pi05`/`pi0` accept only `bfloat16` or
   `float32`; `smolvla`/`act`/`diffusion` have no `dtype` field at all. This is a
   checkpoint property and must not be forced by the adapter.

8. **Observation keys are tolerated silently.** Feeding the pipeline extra keys
   (`observation.velocity`, `observation.eef_pose`, `observation.force`, `time`,
   `safety_stop`) is harmless. Feeding it *fewer* keys than the checkpoint
   declares is also accepted: dropping a declared camera still returned a
   `(1, 16)` action, emitted **no warning**, and produced a different action than
   the complete observation. A mis-wired camera would therefore degrade a rollout
   quietly. **Validating the declared keys is the adapter's job, because LeRobot
   will not do it.**

9. **A checkpoint that already contains the VLM must not reload it.**
   `lerobot/smolvla_base` stores the full 450 M parameters, yet its config still
   sets `load_vlm_weights=True`, which tries to fetch SmolVLM2's separate 2 GB of
   weights. The adapter has to clear that flag for such checkpoints; loading
   fails outright on an offline worker otherwise.

10. **`prepare_observation_for_inference` accepts only environment frames.**
   It is not a generic formatter: it expects uint8 HWC, divides by 255 itself,
   permutes to CHW, adds the batch axis, and injects top-level `task` and
   `robot_type`. `LeRobotDataset` frames are *already* float32 CHW in `[0, 1]`, so
   feeding them through it divides by 255 a second time and swaps the spatial axes,
   and the tokenizer processor then raises `KeyError: 'task'` unless `task` is set.
   Offline scoring must assemble its batch by hand; the adapter's use of this
   function is correct only because the live environment really does emit uint8 HWC.

11. **Inference is stochastic, so a rollout is not reproducible.**
   Every `select_action` restarts the flow-matching integration from fresh Gaussian
   noise. Two calls on a byte-identical observation differ by 0.0034 rad per
   dimension on the cookie 20000-step checkpoint, which is 41% of the expert's
   mean per-step motion (0.00824 rad). Averaging eight samples did **not** reduce
   the error against the expert (-2%), so this is bias rather than variance to
   average away, and it is re-injected at every re-plan. Consequence for
   evaluation: identical seeds give materially different episodes, and a single
   rollout carries no information.

12. **An open-loop chunk can be better than a per-step re-plan.**
   With `chunk_size=50` and `n_action_steps=50`, executing the whole predicted
   chunk costs one forward pass per 50 steps. Measured error against the
   demonstrated actions stays nearly flat across the chunk (0.0055 -> 0.0106 rad)
   while the expert moves 0.233 rad, so the chunk tracks well and the policy's
   error at step 0 already aligns with `expert[t]` (no time lag). Reducing
   `n_action_steps` to re-observe more often buys little and costs 50x the compute,
   so it is not the lever it looks like on a CPU-only host.

## Design

### Adapter responsibilities

Exactly four things, all of them mechanical:

1. **Load** — resolve the checkpoint, build `PreTrainedConfig`, apply the
   caller's overrides (`device`, `dtype`, `n_action_steps`, `rename_map`,
   `load_vlm_weights`), then `make_policy` + `make_pre_post_processors`.
2. **Project the observation** — check that every key the checkpoint declares is
   present, select those keys, apply the rename map, and hand lerobot a batch.
   LeRobot accepts a missing camera silently (finding 8), so this check cannot be
   delegated; it fails loudly here instead.
3. **Adapt the action** — trim padding to the A3 action width, assert finiteness,
   and validate against `ActionSpec`. No semantic rescaling: the policy is
   trained on A3 actions, in A3 units.
4. **Own the lifecycle** — `reset()` clears the policy's internal queue;
   `close()` releases the model.

### What is data and what is code

Almost everything that differs between models is data. This is the property that
makes the design extensible: a new model should be a config file, not a module.

| Concern | Kind | Source |
| --- | --- | --- |
| which policy class | data | `policy_type` (16 options) |
| weights | data | `checkpoint` path |
| camera key names | data | `rename_map` |
| which cameras, state width | data | checkpoint `input_features` |
| action width | data | A3 dataset used for training |
| re-plan cadence | data | `n_action_steps` |
| image resize | data | checkpoint preprocessor |
| normalisation | data | checkpoint safetensors |
| dtype, device | data | checkpoint + caller override |
| padded-vs-real action width | **code** | adapter trims to `ActionSpec` |
| missing-key validation | **code** | lerobot is silent about it (finding 8) |
| gripper units | **code** | already A3-native; assert, do not convert |

### Interface sketch

```python
@dataclass(frozen=True)
class PolicyRuntimeConfig:
    """Everything needed to instantiate a policy, and nothing about A3."""

    checkpoint: Path                 # local dir or hub id
    dataset_root: Path               # defines the action-space contract
    repo_id: str
    policy_type: str | None = None   # omit when the checkpoint's config.json has it
    device: str = "cuda"
    dtype: str | None = None         # only for checkpoints that expose one
    n_action_steps: int | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class LeRobotPolicyAdapter:
    """One adapter for every LeRobot policy."""

    action_mode: ActionMode = "joint_position"

    def __init__(self, config: PolicyRuntimeConfig) -> None:
        self._cfg = config
        self._action_dim = 0          # resolved from the dataset at load time
        self._input_keys: tuple[str, ...] = ()
        self._load()
        self._require_declared_keys()  # lerobot would accept a missing camera

    # --- Policy protocol -------------------------------------------------
    def reset(self, context: EpisodeContext) -> None:
        self._policy.reset()

    def act(self, observation: dict[str, Any], task: str) -> np.ndarray:
        batch = self._project(observation, task)
        with torch.inference_mode():
            action = self._postprocessor(self._policy.select_action(self._preprocessor(batch)))
        return self._adapt(action)

    def close(self) -> None:
        self._policy = None
```

`_require_declared_keys` compares the checkpoint's `input_features` against the keys
an A3 observation actually provides and raises with both lists on a mismatch.
`_project` selects the declared keys and renames; `_adapt` trims to
`self._action_dim`, checks finiteness and returns float64 in the A3 action space.

### Configuration surface

Environment variables keep `a3-sim run --policy ...` working without new CLI
flags, and a shared prefix avoids one set of names per model:

```
A3_POLICY_CHECKPOINT     # path or hub id
A3_POLICY_DATASET_ROOT   # A3 dataset whose action contract the policy obeys
A3_POLICY_REPO_ID
A3_POLICY_TYPE           # optional; read from config.json when omitted
A3_POLICY_DEVICE         # default cuda
A3_POLICY_DTYPE          # optional
A3_POLICY_N_ACTION_STEPS # optional
A3_POLICY_RENAME_MAP     # optional JSON, e.g. {"observation.images.front": "observation.images.image"}
```

A CLI flag (`--policy-config path.toml`) is worth adding once a second model is
actually in use.

## Migration

1. Add `lerobot_policy.py` with the adapter above.
2. Re-point `examples/` and the README at
   `a3_dual_arm_sim.lerobot_policy:make_policy`.
3. Keep `smolvla_policy.py` until the adapter is verified against the SmolVLA
   checkpoint, then delete it — it is the only per-model plugin, and the adapter
   subsumes it.
4. Add a test that builds the adapter from a tiny synthetic checkpoint and asserts
   the action contract, so a new model cannot silently change the interface.

## Open questions

1. **Do we want explicit chunk aggregation on the A3 side?** LeRobot's queue
   already gives receding-horizon control. Temporal ensembling across overlapping
   chunks (as `vla_ur5e_sim` does with `TemporalEnsemble`) would need
   `predict_action_chunk` instead of `select_action`. Recommendation: start with
   `select_action`; revisit only if rollout quality demands it.
2. **Where should `n_action_steps` be tuned?** It trades inference cost against
   reactivity. Should be an evaluation sweep, not a guess.
3. **Is a dataset always available at deployment?** Today `make_policy` needs
   `ds_meta`. A minimal stored feature contract next to the checkpoint would
   remove the dependency, but that is a change we should only make once
   deployment matters.
4. **Camera count mismatch.** A 2-camera checkpoint against A3's 3 cameras works
   by dropping one, but which one is a research decision (probably the wrist
   view, keeping both fixed views). Needs an explicit mapping, not a default.
