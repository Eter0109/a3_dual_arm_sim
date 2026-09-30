# Policy manifests

One file per external model. Adding a model to this repository means adding a
manifest and, on the model's side, a small adapter — never a new Python module in
`src/a3_dual_arm_sim`.

| File | Kind | Purpose |
| --- | --- | --- |
| `replay_hold.yaml` | replay | Drives a real episode from a recording; proves the wiring with nothing installed |
| `cosmos_policy_aloha.yaml` | subprocess | NVIDIA Cosmos Policy ALOHA 2B — joint-space, 14 model dims onto 16 native |
| `gr00t_bimanual.yaml` | subprocess | GR00T-N1-2B (ACG's DexMimicGen tune) — EEF deltas, A3 solves the IK |
| `subprocess_template.yaml` | subprocess | Copy this for a model in its own conda environment |
| `openpi_socket_template.yaml` | socket | Copy this for a model already serving (openpi family) |

```bash
# Audit every manifest without loading anything
a3-sim policy check configs/policies/*.yaml

# Run one
a3-sim run --scene sandbox --policy-manifest configs/policies/replay_hold.yaml --steps 100
```

`replay_hold.json` is generated, not committed as a source of truth:

```bash
python scripts/make_replay_recording.py --source hold --steps 200 \
    --output configs/policies/replay_hold.json
```

## What a manifest can and cannot express

Expressible, and worth expressing:

* which interpreter runs the model (`transport.runtime`), because the GPU sizing
  guide's matrix spans torch 2.1.0 to 2.9.1 and Python 3.10 to 3.13;
* which cameras the model sees, and at what resolution — the dropped views are
  recorded, so a comparison cannot mistake a wiring difference for a capability
  difference;
* the action space, width, chunk length and normalisation, which are properties of
  the checkpoint rather than of A3.

Not expressible on purpose:

* **a different joint topology.** `action.joint_map` rearranges *the same*
  semantics; it cannot turn six joints into seven. A model whose joint count
  differs from A3's needs an adapter on its own side, or the `cartesian_delta`
  space, where the shared contract is Carteisan and the IK belongs to A3.
* **a shell line.** `transport.command` is a list, so a manifest cannot quietly
  become a script.
* **secrets.** Everything here is committed; credentials belong in the
  environment the model is started in.

## Known limits of this machine

The GPU sizing guide assumes an H100 80GB with a 1TB disk; this container reports
an RTX 5090 (sm_120, 32GB, single card) with about 79GB total of disk. That does
not change the manifest format, but it does change which models can run here —
DreamZero needs two cards, and Fast-WAM is expected to exceed 32GB. Check
`nvidia-smi` before copying a template that the guide sized for H100.
