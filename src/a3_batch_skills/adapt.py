"""Wrap the existing pi05 trainer with validated left8 clips and padded-loss masking.

Pass the established trainer arguments after --. The wrapper does not choose a
learning rate, optimizer, unfreezing range, or training budget for the user.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from .markers import MARKER_VERSION
from .training import install_pi05_skill_loss


def argument_value(args, name):
    for i, value in enumerate(args):
        if value.startswith(name+"="):
            return value.split("=",1)[1]
        if value == name and i+1 < len(args):
            return args[i+1]
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launcher", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("trainer_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    rest = a.trainer_args[1:] if a.trainer_args[:1] == ["--"] else a.trainer_args
    if a.output.exists():
        p.error("Use a new independent adaptation output; preserve existing checkpoints")
    root = a.dataset.resolve()
    descriptor = json.loads((root/"goal_conditioning.json").read_text())
    info = json.loads((root/"meta/info.json").read_text())
    if descriptor.get("goal_conditioning") != MARKER_VERSION:
        p.error("Expected the explicit projected-goal skill dataset")
    for key in ("action","observation.state"):
        if info["features"][key]["shape"] != [8]:
            p.error("Skill adaptation requires an 8D state/action dataset")
    supplied = argument_value(rest,"--dataset.root")
    if supplied and Path(supplied).resolve() != root:
        p.error("Trainer dataset path differs from the audited skill dataset")
    supplied_output = argument_value(rest,"--output_dir")
    if supplied_output and Path(supplied_output).resolve() != a.output.resolve():
        p.error("Trainer output differs from the independent skill output")
    if argument_value(rest,"--resume") in ("true","True","1"):
        p.error("This entry creates a new adaptation run, not an in-place resume")
    # Preserve the trained state projector when continuing the current left8
    # checkpoint. Its weights must be instantiated before loading/optimization.
    checkpoint = argument_value(rest,"--policy.path")
    if checkpoint is None:
        p.error("Specify --policy.path=<existing left8 checkpoint> for continuation training")
    if checkpoint:
        cfg = json.loads((Path(checkpoint)/"config.json").read_text())
        if (cfg["output_features"]["action"]["shape"] != [8]
                or cfg["input_features"]["observation.state"]["shape"] != [8]):
            p.error("The continuation checkpoint is not a left8 policy")
        from safetensors import safe_open
        with safe_open(Path(checkpoint)/"model.safetensors",framework="pt",device="cpu") as weights:
            if "model.state_proj.weight" in weights.keys():
                os.environ["A3_STATE_PROJECTOR"] = "1"
    # Retain the final retreat frames; their synthetic chunk tail is masked.
    os.environ["A3_DROP_N_LAST_FRAMES"] = "0"
    install_pi05_skill_loss()
    spec = importlib.util.spec_from_file_location("a3_existing_skill_trainer",a.launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    overrides = []
    if not supplied:
        overrides.append("--dataset.root="+str(root))
    if not supplied_output:
        overrides.append("--output_dir="+str(a.output.resolve()))
    if argument_value(rest,"--resume") is None:
        overrides.append("--resume=false")
    sys.argv = [str(a.launcher),*rest,*overrides]
    result = module.main()
    if result not in (0,None):
        raise RuntimeError("The existing trainer did not finish successfully")
    descriptor.update(training_completed=True,dataset=str(root),time_padding_masked=True,active_action_dimensions=8)
    for cfg in a.output.glob("checkpoints/*/pretrained_model/config.json"):
        if not (cfg.parent/"model.safetensors").is_file():
            continue
        (cfg.parent/"goal_conditioning.json").write_text(json.dumps(descriptor,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
