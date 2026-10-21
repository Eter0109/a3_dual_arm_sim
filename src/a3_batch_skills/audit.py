"""Audit actual LeRobot episode boundaries and optional existing pi05 preprocessing."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np


def audit(root,repo_id,checkpoint=None,job=None,policy_module_dir=None):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from .training import masked_flow_loss
    root = Path(root)
    info = json.loads((root/"meta/info.json").read_text())
    provenance = json.loads((root/"a3_skill_metadata.json").read_text())
    for key in ("action","observation.state"):
        if info["features"][key]["shape"] != [8]:
            raise ValueError("Dataset is not left8")
    dataset = LeRobotDataset(repo_id=repo_id,root=root,video_backend="pyav",
                            delta_timestamps={"action":[i/info["fps"] for i in range(50)]})
    offset, examples, rows = 0, [], []
    for meta in provenance:
        last = offset+meta["frames"]-1
        item = dataset[last]
        mask = item["action_is_pad"]
        if item["action"].shape != (50,8) or mask.shape != (50,) or mask[0] or not mask[1:].all():
            raise ValueError("A chunk crossed a batch boundary or has an invalid padding mask")
        raw = np.load(Path(meta["source_clip"])/"trajectory.npz",allow_pickle=False)
        np.testing.assert_allclose(item["action"][0].numpy(),raw["action"][-1],atol=1e-6)
        np.testing.assert_allclose(item["observation.state"].numpy(),raw["state"][-1],atol=1e-6)
        if meta.get('per_frame_geometry'):
            from .geometry_context import GEOMETRY_KEY,GEOMETRY_DIM
            if item[GEOMETRY_KEY].shape!=(GEOMETRY_DIM,):raise ValueError('Geometry shape changed in dataset reader')
            np.testing.assert_allclose(item[GEOMETRY_KEY].numpy(),raw['geometry'][-1],atol=1e-6)
        raw.close()
        examples.append(item)
        rows.append(dict(batch_index=meta["batch_index"],entry_kind=meta["entry_kind"],
                         final_dataset_row=last,real_action_rows_in_tail_chunk=int((~mask).sum())))
        offset += meta["frames"]
    report = dict(episodes=len(rows),frames=offset,actual_dataset_tail_checks=rows,
                  action_dim=8,state_dim=8,model_weights_loaded=False)
    if checkpoint is not None:
        from torch.utils.data import default_collate
        sys.path.insert(0,str(policy_module_dir))
        adapter = importlib.import_module("pi05_adapter")
        adapter.install_state_projector(job)
        config,pre,_ = adapter.load_processors(checkpoint,job,"cpu")
        from .geometry_context import GEOMETRY_KEY
        collated=default_collate(examples)
        batch = pre({key:value for key,value in collated.items() if key!=GEOMETRY_KEY})
        if "action_is_pad" not in batch:
            raise ValueError("The pi05 preprocessing pipeline discarded action_is_pad")
        losses = torch.ones(len(examples),50,config.max_action_dim,requires_grad=True)
        reduced = masked_flow_loss(losses,batch["action_is_pad"])
        reduced.mean().backward()
        if losses.grad[:,:,8:].abs().sum() or losses.grad[:,1:,:].abs().sum():
            raise ValueError("Padding contributed a loss gradient")
        report.update(preprocessing_preserves_action_is_pad=True,padding_and_unused_gradients_zero=True,
                      checkpoint_state_dim=list(config.input_features["observation.state"].shape),
                      checkpoint_action_dim=list(config.output_features["action"].shape))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--repo-id",default="local/a3_goal_batch_left8")
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--job",type=Path)
    p.add_argument("--policy-module-dir",type=Path)
    a = p.parse_args()
    if a.checkpoint and not (a.job and a.policy_module_dir):
        p.error("Processor audit requires job and policy-module-dir")
    report = audit(a.root,a.repo_id,a.checkpoint,a.job,a.policy_module_dir)
    (a.root/"audit_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
