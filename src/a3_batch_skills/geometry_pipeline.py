"""Finite measured-geometry post-training, matched image control, pure-VLA eval.

Wait for owned collections and free GPU memory; never terminate another run.
Only verified full sensor/action clips train the model. No privileged pose input.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from .geometry_context import GEOMETRY_VERSION
from .train_visual import atomic_json


def inspect_clip(path):
    meta=json.loads((path/'metadata.json').read_text())
    if not meta.get('clip_success') or not meta.get('visual_boundary_verified'):return None
    if meta.get('geometry_conditioning')!=GEOMETRY_VERSION or not meta.get('per_frame_geometry'):
        raise ValueError('Training clip has no measured frame-aligned geometry')
    if meta.get('geometry_source')!='calibrated_RGBD_and_robot_FK_only':raise ValueError('Unapproved geometry provenance')
    with np.load(path/'trajectory.npz',allow_pickle=False) as rows:
        n=meta['frames'];geometry=rows['geometry']
        if (rows['state'].shape!=(n,8) or rows['action'].shape!=(n,8)
                or geometry.shape!=(n,30) or not np.all(np.diff(rows['global_step'])==1)):
            raise ValueError('Training clip timing or left8 representation failed')
        if not all(np.isfinite(rows[key]).all() for key in ('state','action','geometry')):
            raise ValueError('Nonfinite training values')
        flags=geometry[:,27:]
        if not np.all((flags==0)|(flags==1)):raise ValueError('Invalid geometry visibility bits')
        if np.any(geometry[:,:27].reshape(-1,3,9)[np.repeat(~flags.astype(bool),9,axis=1).reshape(-1,3,9)]!=0):
            raise ValueError('Unavailable geometry contains labels')
        if flags[:,1].mean()<.7:return None
        if meta.get('clip_role')=='hybrid_correction':
            servo=rows['action_source']=='visual_servo'
            if not servo.any() or flags[servo,2].mean()<.5:return None
    if meta.get('clip_role')=='hybrid_correction':
        if meta.get('visual_servo_frames',0)<50:return None
        if meta.get('native_count_at_accept',0)!=5*(meta['batch_index']+1):return None
    return dict(path=str(path.resolve()),metadata=meta)


def data_gate(clips):
    corrections=[c for c in clips if c['metadata'].get('clip_role')=='hybrid_correction']
    later=sum(c['metadata']['entry_kind']=='after_previous_release' for c in clips)
    frames=sum(c['metadata']['frames'] for c in clips)
    columns={c['metadata'].get('source_column',c['metadata'].get('visual_goal',{}).get('source_column')) for c in clips}
    columns.discard(None)
    return dict(passed=len(clips)>=12 and frames>=4000 and len(corrections)>=2 and later>=2 and len(columns)>=2,
        clips=len(clips),frames=frames,correction_clips=len(corrections),later_batch_clips=later,
        source_columns=sorted(columns),
        correction_servo_frames=sum(c['metadata'].get('visual_servo_frames',0) for c in corrections))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--teacher-root',type=Path,required=True);p.add_argument('--correction-root',type=Path,required=True)
    p.add_argument('--collection-snapshot-root',type=Path,action='append',default=[],
        help='Reuse finalized clips from an interrupted collection without waiting for its vanished process')
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--verified-wiring',type=Path)
    p.add_argument('--matched-control',action='store_true')
    a=p.parse_args()
    if a.output.exists() or a.steps<1:p.error('Use a new output and positive finite training budget')
    a.output.mkdir(parents=True)
    started=time.monotonic();job=a.job;camera=job/'artifacts/training_camera_profile.json'
    policy_dir=job/'benchmark_research_v3';waiting_roots=[a.teacher_root,a.correction_root]
    roots=[*waiting_roots,*a.collection_snapshot_root]
    def phase(stage,**fields):
        atomic_json(a.output/'status.json',dict(stage=stage,elapsed_seconds=time.monotonic()-started,**fields))
        print(json.dumps(dict(stage=stage,**fields)),flush=True)
    def run(module,extra,log,timeout):
        with Path(log).open('w') as stream:
            process=subprocess.Popen([sys.executable,'-u','-m',module,*map(str,extra)],
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:return process.wait(timeout=timeout)
            finally:
                if process.poll() is None:
                    # Only the group created by this invocation, including its
                    # rendering/model workers. Never target another experiment.
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    def clips_from_roots():
        # A completed clip is reusable even if reboot interrupted the enclosing
        # suite. Partial trajectories have no finalized metadata and are ignored.
        # Resolve duplicates when a snapshot and a resumed suite share clips.
        paths=sorted({p.parent.resolve() for root in roots
            for p in root.glob('cases/*/skill_clips/batch_*/metadata.json')})
        return [c for p in paths if (c:=inspect_clip(p)) is not None]
    def wait_for_gpu(required_gib=26):
        deadline=time.monotonic()+4*3600
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=required_gib*1024:return
            phase('waiting_for_free_gpu',free_mib=free,required_mib=required_gib*1024,other_runs_preserved=True)
            if time.monotonic()>deadline:raise RuntimeError('GPU remained occupied; no other process was stopped')
            time.sleep(30)
    try:
        if a.verified_wiring is not None:
            wiring=json.loads((a.verified_wiring/'result.json').read_text())
            if wiring.get('status')!='real_prefix_wiring_passed':raise RuntimeError('Real pretrained wiring did not pass')
            code_root=Path(json.loads(a.verified_wiring.with_suffix('.launch.json').read_text())['code_root'])
            for name in ('geometry_context.py','geometry_model.py'):
                current=Path(__file__).with_name(name);checked=code_root/'src/a3_batch_skills'/name
                if hashlib.sha256(current.read_bytes()).digest()!=hashlib.sha256(checked.read_bytes()).digest():
                    raise RuntimeError('Geometry wiring source changed after verification')
        phase('waiting_for_full_sensor_collections')
        deadline=time.monotonic()+3*3600
        while not all(root.with_suffix('.exit.json').exists() for root in waiting_roots):
            if time.monotonic()>deadline:raise RuntimeError('Collection wait exceeded finite budget')
            time.sleep(15)
        clips=clips_from_roots();gate=data_gate(clips)
        atomic_json(a.output/'initial_data_gate.json',gate)
        if not gate['passed']:
            # One bounded supplement per data source, rather than relaxing gates.
            if gate['clips']<12 or gate['later_batch_clips']<2 or gate['frames']<4000 or len(gate['source_columns'])<2:
                target=a.output/'supplement_teacher';phase('supplementing_teacher_clips',**gate)
                if run('a3_batch_skills.geometry_collect',['--output',target,'--camera-reference',camera,
                    '--workers','3','--seeds','26442'],target.with_suffix('.log'),4*3600):
                    raise RuntimeError('Supplement collection runtime failed')
                roots.append(target)
            if gate['correction_clips']<2:
                target=a.output/'supplement_corrections';phase('supplementing_correction_clips')
                if run('a3_batch_skills.visual_suite',['--output',target,'--checkpoint',a.checkpoint,'--job',job,
                    '--policy-module-dir',policy_dir,'--camera-reference',camera,'--workers','2',
                    '--seeds','26468','26470','--profiles','basic','medium','--task','2x5',
                    '--max-batches','1','--record-training-clips'],target.with_suffix('.log'),2*3600):
                    raise RuntimeError('Supplement correction runtime failed')
                roots.append(target)
            clips=clips_from_roots();gate=data_gate(clips)
        atomic_json(a.output/'data_gate.json',gate)
        if not gate['passed']:raise RuntimeError('Insufficient verified correction/later-batch data; training was not started')
        selected=a.output/'selected_clips';selected.mkdir()
        for index,clip in enumerate(clips):(selected/f'batch_{index:04d}').symlink_to(Path(clip['path']),target_is_directory=True)
        atomic_json(a.output/'selected_manifest.json',clips)
        training_seeds={c['metadata'].get('seed',c['metadata'].get('provenance',{}).get('seed')) for c in clips}
        if training_seeds&{26490,26491}:raise RuntimeError('Development evaluation seeds leaked into training')
        dataset=a.output/'dataset';phase('exporting_and_auditing',**gate)
        repo_id='local/a3_geometry30_corrections_left8'
        if run('a3_batch_skills.export',['--inputs',selected,'--output',dataset,'--repo-id',repo_id],a.output/'export.log',2400):
            raise RuntimeError('Geometry dataset export failed')
        if run('a3_batch_skills.audit',['--root',dataset,'--repo-id',repo_id,'--checkpoint',a.checkpoint,
            '--job',job,'--policy-module-dir',policy_dir],a.output/'audit.log',1200):raise RuntimeError('Dataset reader/timing audit failed')
        if a.verified_wiring is None:
            wait_for_gpu(14);phase('real_pretrained_prefix_wiring')
            clip=next(c for c in clips if c['metadata'].get('clip_role')=='hybrid_correction')
            wiring=a.output/'wiring'
            if run('a3_batch_skills.geometry_wiring_smoke',['--output',wiring,'--checkpoint',a.checkpoint,
                '--job',job,'--policy-module-dir',policy_dir,'--clip',clip['path'],'--device','cuda'],a.output/'wiring.log',1800):
                raise RuntimeError('Real prefix wiring check failed; bulk training blocked')
        wait_for_gpu();phase('real_geometry_gradient_smoke')
        common=['--dataset',dataset,'--checkpoint',a.checkpoint,'--job',job,'--policy-module-dir',policy_dir]
        smoke=a.output/'smoke'
        if run('a3_batch_skills.train_visual',['--output',smoke,*common,'--geometry-conditioning','--smoke'],a.output/'smoke.log',1800):
            raise RuntimeError('Real-model geometry gradient smoke failed; bulk training blocked')
        gradient=json.loads((smoke/'geometry_gradient_check.json').read_text())
        if not gradient.get('passed'):raise RuntimeError('No geometry gradient')
        variants=['geometry','image_only'] if a.matched_control else ['geometry']
        comparisons=[]
        for variant in variants:
            wait_for_gpu();training=a.output/('training_'+variant);phase('training_'+variant,updates=a.steps,replay_fraction=.5)
            flags=['--geometry-conditioning'] if variant=='geometry' else []
            if run('a3_batch_skills.train_visual',['--output',training,*common,'--steps',a.steps,*flags],
                a.output/(variant+'_training.log'),18*3600):raise RuntimeError('Neural post-training failed: '+variant)
            checkpoint=training/'checkpoints'/f'{a.steps:06d}'/'pretrained_model'
            for task in ('2x5','2x10'):
                output=a.output/(variant+'_evaluation_'+task);phase('evaluating_'+variant,task=task,external_action_takeover=False)
                code=run('a3_batch_skills.visual_suite',['--output',output,'--checkpoint',checkpoint,'--job',job,
                    '--policy-module-dir',policy_dir,'--camera-reference',camera,'--workers','2',
                    '--seeds','26490','26491','--profiles','basic','--task',task,'--source-column','auto',
                    '--placement-control','vla'],output.with_suffix('.log'),2*3600)
                if code:raise RuntimeError('Pure-VLA evaluation runtime failed: '+variant+'/'+task)
                results=json.loads((output/'report.json').read_text());comparisons.append(dict(variant=variant,task=task,results=results))
        atomic_json(a.output/'result.json',dict(status='complete',updates_per_variant=a.steps,comparisons=comparisons,
            data_gate=gate,claimed_improvement=False,note='Development examples; assess strict task outcomes and matched control before claiming benefits.'))
        phase('complete')
    except Exception as exc:
        atomic_json(a.output/'error.json',dict(type=type(exc).__name__,message=str(exc)));phase('failed',reason=str(exc));raise


if __name__=='__main__':main()
