"""Bounded RGB-D goal adaptation with exact 50/50 original-demonstration replay.

Continue existing left8 weights/processors; do not initialize a new state head.
Effective batch six, micro batch one, original 18-layer/vision unfreezing scope.
Only each 5000th update (and the final update) saves one resumable checkpoint.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
from .visual_goals import VISUAL_MARKER_VERSION
from .rgbd_localization import LOCALIZER_VERSION
from .training import install_pi05_skill_loss


def atomic_json(path,value):
    path=Path(path);pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(value,indent=2));pending.replace(path)


class PolicyRows:
    """Keep only deployment observations, action labels, task and padding mask."""
    def __init__(self,root,horizon,*,geometry=False,require_geometry=False):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        info=json.loads((Path(root)/'meta/info.json').read_text())
        self.dataset=LeRobotDataset(info.get('repo_id','local/a3_left8'),root=Path(root),
            delta_timestamps={'action':[i/info['fps'] for i in range(horizon)]},video_backend='pyav')
        self.geometry,self.require_geometry=geometry,require_geometry

    def __len__(self):return len(self.dataset)

    def __getitem__(self,index):
        raw=self.dataset[index]
        from .export import DATASET_CAMERAS
        rename={'observation.images.'+old:'observation.images.'+new for old,new in DATASET_CAMERAS.items()}
        row={rename.get(k,k):v for k,v in raw.items() if k.startswith('observation.images.')
             or k in ('observation.state','action','action_is_pad','task')}
        if row['observation.state'].shape[-1]!=8 or row['action'].shape[-1]!=8:
            raise ValueError('Replay and skill data must both be genuinely left8')
        if 'action_is_pad' not in row or not row.get('task'):raise ValueError('Missing padding or instruction')
        if getattr(self,'geometry',False):
            import torch
            from .geometry_context import GEOMETRY_KEY,GEOMETRY_DIM
            if self.require_geometry and GEOMETRY_KEY not in raw:raise ValueError('Missing measured training geometry')
            value=raw.get(GEOMETRY_KEY,torch.zeros(GEOMETRY_DIM))
            if value.shape!=(GEOMETRY_DIM,) or not torch.isfinite(value).all():raise ValueError('Invalid geometry row')
            row[GEOMETRY_KEY]=value
        return row


def save_checkpoint(adapter,base,output,step,optimizer,manifest,descriptor):
    import torch
    root=Path(output)/'checkpoints';root.mkdir(exist_ok=True)
    destination=root/f'{step:06d}'
    pending=root/f'.pending_{step:06d}'
    if destination.exists() or pending.exists():raise FileExistsError(destination)
    model_bytes=sum(v.numel()*v.element_size() for v in adapter.policy.state_dict().values())
    optimizer_bytes=sum(p.numel()*p.element_size()*2 for p in adapter.policy.parameters() if p.requires_grad)
    if shutil.disk_usage(root).free<model_bytes+optimizer_bytes+4*1024**3:
        raise RuntimeError('Insufficient data-disk space for one resumable checkpoint')
    pending.mkdir()
    model=pending/'pretrained_model'
    adapter.policy.save_pretrained(model)
    for item in Path(base).iterdir():
        if item.is_file() and item.name not in ('model.safetensors','config.json','train_config.json','goal_conditioning.json','geometry_conditioning.json'):
            shutil.copy2(item,model/item.name)
    atomic_json(model/'goal_conditioning.json',dict(descriptor,training_completed=True,
        completed_updates=step,source_checkpoint=str(base),time_padding_masked=True,active_action_dimensions=8))
    if manifest.get('learned_geometry'):
        from .geometry_model import geometry_descriptor
        atomic_json(model/'geometry_conditioning.json',geometry_descriptor(training_completed=True,
            completed_updates=step,source_checkpoint=str(base),external_action_takeover_required=False))
    atomic_json(pending/'adaptation_manifest.json',dict(manifest,completed_updates=step))
    torch.save(dict(step=step,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),
        pending/'training_state.pt')
    pending.rename(destination)
    atomic_json(Path(output)/'latest_checkpoint.json',dict(step=step,path=str(destination)))
    # Only remove optimizer states created by THIS adaptation; keep all models.
    for old in root.iterdir():
        state=old/'training_state.pt'
        if old!=destination and old.name.isdigit() and state.exists() and (old/'adaptation_manifest.json').exists():
            if not state.resolve().is_relative_to(root.resolve()):raise RuntimeError('Unsafe retention path')
            state.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--policy-module-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--seed',type=int,default=26400)
    p.add_argument('--learning-rate',type=float,default=2.5e-6)
    p.add_argument('--smoke',action='store_true',help='Two real gradient updates; save no weights or optimizer')
    p.add_argument('--geometry-conditioning',action='store_true',help='Learn a geometry residual; original replay uses zero/invalid geometry')
    a=p.parse_args()
    if a.output.exists():p.error('Use an independent output; preserve existing runs')
    if a.steps<1 or a.learning_rate<=0:p.error('Invalid adaptation budget')
    if a.smoke:a.steps=2
    descriptor=json.loads((a.dataset/'goal_conditioning.json').read_text())
    if descriptor.get('goal_conditioning')!=VISUAL_MARKER_VERSION or descriptor.get('localization')!=LOCALIZER_VERSION:
        p.error('Expected independently measured RGB-D goal clips')
    if a.geometry_conditioning:
        from .geometry_context import GEOMETRY_VERSION
        if descriptor.get('geometry_conditioning')!=GEOMETRY_VERSION:p.error('Missing per-frame geometry training data')
    else:descriptor.pop('geometry_conditioning',None)
    cfg=json.loads((a.checkpoint/'config.json').read_text())
    if cfg['output_features']['action']['shape']!=[8] or cfg['input_features']['observation.state']['shape']!=[8]:
        p.error('Continuation checkpoint must be left8')
    a.output.mkdir(parents=True)
    import torch
    from torch.utils.data import DataLoader,RandomSampler
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<26*1024**3:
        raise RuntimeError('At least 26 GiB free GPU memory required before this bounded adaptation')
    torch.set_num_threads(4)
    os.environ['A3_STATE_PROJECTOR']='1'
    os.environ['A3_UNFREEZE_LLM_LAYERS']='18';os.environ['A3_UNFREEZE_VISION']='1'
    install_pi05_skill_loss()  # before the legacy state-projector forward wrapper
    sys.path.insert(0,str(a.policy_module_dir))
    from pi05_adapter import Pi05BenchmarkPolicy
    if a.geometry_conditioning and (a.checkpoint/'geometry_conditioning.json').exists():
        from .geometry_adapter import GeometryPi05Adapter
        adapter=GeometryPi05Adapter(a.checkpoint,a.job)
    else:adapter=Pi05BenchmarkPolicy(a.checkpoint,a.job)
    if a.geometry_conditioning:
        from .geometry_model import attach_geometry_head,install_geometry_model
        torch.manual_seed(a.seed)
        if not hasattr(adapter.policy.model,'a3_geometry_proj'):attach_geometry_head(adapter.policy.model)
        install_geometry_model()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    spec=importlib.util.spec_from_file_location('visual_training_scope',a.job/'repo/scripts/train_pi05_launcher.py')
    scope=importlib.util.module_from_spec(spec);spec.loader.exec_module(scope)
    scope._patch_partial_unfreeze()
    policy=adapter.policy
    policy.requires_grad_(True)
    policy.model.paligemma_with_expert._set_requires_grad()
    scope._apply_unfreeze(policy.model.paligemma_with_expert,True)
    policy.train()
    old=PolicyRows(a.job/'datasets/cookie_enhanced_1k_left8',policy.config.chunk_size,geometry=a.geometry_conditioning)
    new=PolicyRows(a.dataset,policy.config.chunk_size,geometry=a.geometry_conditioning,require_geometry=a.geometry_conditioning)
    # Independent worker pools keep video decoding off the training thread.
    loaders=[]
    for offset,data in enumerate((old,new)):
        sampler=RandomSampler(data,replacement=True,num_samples=a.steps*3,
            generator=torch.Generator().manual_seed(a.seed+offset))
        loaders.append(DataLoader(data,batch_size=3,sampler=sampler,num_workers=2,
            persistent_workers=True,prefetch_factor=2))
    streams=[iter(loader) for loader in loaders]
    optimizer=torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad],lr=a.learning_rate,
        betas=(.9,.95),eps=1e-8,weight_decay=.01,fused=True)
    manifest=dict(method='single_batch_rgbd_goal_imitation_with_original_replay',
        base_checkpoint=str(a.checkpoint),dataset=str(a.dataset),requested_updates=a.steps,seed=a.seed,
        effective_batch_size=6,micro_batch_size=1,original_replay_fraction=.5,
        learning_rate=a.learning_rate,learning_rate_basis='continue from original 30k terminal LR',
        unfreeze_llm_layers=18,unfreeze_vision=True,state_dim=8,action_dim=8,
        trainable_parameters=sum(x.numel() for x in policy.parameters() if x.requires_grad),
        normalization='unchanged trained checkpoint processors',action_chunk=policy.config.chunk_size,
        n_action_steps=policy.config.n_action_steps,save_every_updates=5000,
        loss='flow matching; time padding and action dimensions >=8 masked',
        deploy_controller='RGB-D supervisor; same single VLA executes complete batch',
        teacher='privileged expert actions; independent RGB-D cues; no privileged model observations',
        training_goal_frames=len(new),original_frames=len(old),
        claimed_task_improvement=False,smoke_only=a.smoke)
    manifest['training_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['learned_geometry']=bool(a.geometry_conditioning)
    from collections import Counter
    clip_metadata=json.loads((a.dataset/'a3_skill_metadata.json').read_text())
    manifest['data_clip_roles']=dict(Counter(m.get('clip_role','legacy_teacher') for m in clip_metadata))
    manifest['teacher']='verified action provenance in a3_skill_metadata; sensor-only deployed observation fields'
    if a.geometry_conditioning:
        from .geometry_model import geometry_descriptor
        manifest['geometry']=geometry_descriptor(training_completed=False)
        manifest['method']='geometry_conditioned_flow_imitation_with_original_replay'
        manifest['deploy_controller']='RGB-D observations and batch verification; learned VLA emits all movement actions'
    atomic_json(a.output/'manifest.json',manifest)
    gradient_checked=False;geometry_gradient_checked=False
    def check_gradient(gradient):
        nonlocal gradient_checked
        if not gradient_checked:
            if not torch.isfinite(gradient).all() or torch.count_nonzero(gradient[8:]):
                raise RuntimeError('Padded action dimensions affected the left8 gradient')
            atomic_json(a.output/'gradient_check.json',dict(passed=True,unused_action_gradient_max=0.))
            gradient_checked=True
        return gradient
    policy.model.action_out_proj.weight.register_hook(check_gradient)
    def check_geometry_gradient(gradient):
        nonlocal geometry_gradient_checked
        if not torch.isfinite(gradient).all():raise RuntimeError('Nonfinite geometry gradient')
        if torch.count_nonzero(gradient):
            geometry_gradient_checked=True
            atomic_json(a.output/'geometry_gradient_check.json',dict(passed=True,
                gradient_norm=float(gradient.float().norm()),geometry_is_trainable=True))
        return gradient
    if a.geometry_conditioning:policy.model.a3_geometry_proj[-1].weight.register_hook(check_geometry_gradient)
    started=time.monotonic()
    for step in range(1,a.steps+1):
        before=time.monotonic();optimizer.zero_grad(set_to_none=True)
        batches=[next(stream) for stream in streams]
        raw={key:batches[0][key]+batches[1][key] if isinstance(batches[0][key],list)
             else torch.cat([batches[0][key],batches[1][key]],dim=0) for key in batches[0]}
        data_seconds=time.monotonic()-before
        losses=[]
        for i in range(6):
            micro={key:value[i:i+1] for key,value in raw.items()}
            from .geometry_context import GEOMETRY_KEY
            batch=adapter.pre({key:value for key,value in micro.items() if key!=GEOMETRY_KEY})
            if a.geometry_conditioning:
                batch[GEOMETRY_KEY]=micro[GEOMETRY_KEY].to(policy.model.state_proj.weight.device)
            loss,_=policy(batch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite flow-matching loss')
            (loss/6).backward();losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_([x for x in policy.parameters() if x.requires_grad],1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
        lr=a.learning_rate*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/max(a.steps,1))))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step()
        if step==1 or step%20==0 or step==a.steps:
            report=dict(step=step,loss=float(np.mean(losses)),old_replay_loss=float(np.mean(losses[:3])),
                skill_loss=float(np.mean(losses[3:])),grad_norm=float(norm),learning_rate=lr,
                data_seconds=data_seconds,step_seconds=time.monotonic()-before,
                elapsed_seconds=time.monotonic()-started,gpu_allocated_bytes=torch.cuda.memory_allocated(),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            atomic_json(a.output/'status.json',report);print(json.dumps(report),flush=True)
        if not a.smoke and (step%5000==0 or step==a.steps):
            if a.geometry_conditioning and not geometry_gradient_checked:raise RuntimeError('Geometry head received no learning signal')
            save_checkpoint(adapter,a.checkpoint,a.output,step,optimizer,manifest,descriptor)
    if a.geometry_conditioning and not geometry_gradient_checked:raise RuntimeError('Geometry gradient smoke failed')
    atomic_json(a.output/'result.json',dict(status='smoke_complete' if a.smoke else 'training_complete',updates=a.steps,
        wall_seconds=time.monotonic()-started,task_improvement_requires_separate_evaluation=True))


if __name__=='__main__':main()
