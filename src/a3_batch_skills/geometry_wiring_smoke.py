"""Actual pretrained prefix/head wiring; not a bulk flow-training test."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True);p.add_argument('--policy-module-dir',type=Path,required=True)
    p.add_argument('--clip',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a=p.parse_args()
    if a.output.exists():p.error('Never overwrite a smoke result')
    a.output.mkdir(parents=True)
    stat=Path('/sys/fs/cgroup/memory.stat');limit=Path('/sys/fs/cgroup/memory.max')
    if a.device=='cpu' and stat.exists() and limit.exists() and limit.read_text().strip()!='max':
        values=dict(line.split() for line in stat.read_text().splitlines())
        available=int(limit.read_text())-sum(int(values.get(k,0)) for k in ('anon','kernel','unevictable'))
        if available<28*1024**3:raise RuntimeError('Too little CPU memory including model-loading transients')
    import torch
    if a.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<14*1024**3):
        raise RuntimeError('Wiring check needs 14 GiB free; it will not displace another run')
    torch.set_num_threads(4)
    sys.path.insert(0,str(a.policy_module_dir))
    from pi05_adapter import Pi05BenchmarkPolicy,observation_batch
    from .geometry_context import GEOMETRY_KEY
    from .geometry_model import attach_geometry_head,install_geometry_model
    from .export import rgb_frames
    from .markers import MARKED_INSTRUCTION
    with np.load(a.clip/'trajectory.npz',allow_pickle=False) as rows:
        valid=np.flatnonzero((rows['geometry'][:,28:]>0).all(1))
        if not len(valid):raise RuntimeError('No fully observed object/target row')
        index=int(valid[len(valid)//2]);geometry=rows['geometry'][index].copy();state=rows['state'][index].copy()
    observation={'observation.state':state}
    for name in ('front','left_wrist','right_wrist'):
        frames=rgb_frames(a.clip/(name+'.mkv'))
        for _ in range(index+1):image=next(frames)
        frames.close();observation['observation.images.'+name]=image
    started=time.monotonic();adapter=Pi05BenchmarkPolicy(a.checkpoint,a.job,device=a.device)
    model=adapter.policy.model
    try:
        attach_geometry_head(model);install_geometry_model()
        for param in model.parameters():param.requires_grad_(False)
        model.a3_geometry_proj.requires_grad_(True)
        batch=adapter.pre(observation_batch(observation,MARKED_INSTRUCTION))
        images,image_masks=adapter.policy._preprocess_images(batch)
        from lerobot.policies.pi05.modeling_pi05 import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK
        args=(images,image_masks,batch[OBS_LANGUAGE_TOKENS],batch[OBS_LANGUAGE_ATTENTION_MASK])
        model._a3_state=batch['observation.state'];model._a3_geometry=None
        with torch.no_grad():baseline,pad0,attention0=model.embed_prefix(*args)
        model._a3_state=batch['observation.state'];model._a3_geometry=torch.tensor(geometry)[None].to(a.device)
        result,pad1,attention1=model.embed_prefix(*args)
        if not torch.equal(baseline,result):raise RuntimeError('Zero head changed the pretrained prefix')
        if not torch.equal(pad0,pad1) or not torch.equal(attention0,attention1):raise RuntimeError('Token masks changed')
        result[:,-1].float().square().mean().backward()
        gradient=model.a3_geometry_proj[-1].weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or not gradient.count_nonzero():
            raise RuntimeError('Real pretrained prefix did not deliver a geometry gradient')
        report=dict(status='real_prefix_wiring_passed',device=a.device,frame=index,
            zero_initial_prefix_exact=True,token_masks_unchanged=True,
            geometry_gradient_norm=float(gradient.float().norm()),prefix_tokens=result.shape[1],
            original_weights_updated=False,full_flow_training_smoke_passed=False,
            wall_seconds=time.monotonic()-started)
        (a.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    finally:adapter.close()


if __name__=='__main__':main()
