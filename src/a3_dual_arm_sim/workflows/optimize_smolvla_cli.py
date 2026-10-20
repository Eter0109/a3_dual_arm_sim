"""Fixed-checkpoint horizon development experiment, with conditional confirmation."""
import argparse
import hashlib
import json
import os
import signal
from dataclasses import asdict
from pathlib import Path

from a3_dual_arm_sim.paths import project_root, resource_root
from a3_dual_arm_sim.sim.randomization import load_randomization_config
from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_cookie_2x10_settings
from .benchmark import make_cookie_benchmark
from .policy_adapters import SmolVLAPolicyAdapter
from .experiment_store import ExperimentStore, atomic_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def totals(rows):
    return {'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
            'cookies': sum(r['cookies_in_target'] for r in rows)}


def validate_trace(path, steps):
    import cv2
    trace = path / 'trace.jsonl'
    with trace.open() as f:
        count = 0
        for count, line in enumerate(f, 1):
            if json.loads(line)['step'] != count - 1:
                raise RuntimeError('Trace step mismatch')
    if count != steps:
        raise RuntimeError('Trace length mismatch')
    for name in ('front', 'left_wrist', 'right_wrist'):
        cap = cv2.VideoCapture(str(path / f'{name}.mp4'))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        if n != steps:
            raise RuntimeError(f'Video length mismatch: {name}: {n} vs {steps}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=project_root() / 'outputs/cookie_multi_left8_40k_v1/model/checkpoints/040000/pretrained_model')
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not silently change runtime')
    from a3_dual_arm_sim.learning.checkpoint_bundle import inference_fingerprints
    settings5 = load_randomization_config(resource_root() / 'configs/randomization.yaml')
    settings10, first = load_cookie_2x10_settings(resource_root() / 'configs/randomization_2x10.yaml')
    protocol = {'checkpoint': str(args.checkpoint.resolve()), 'weights_sha256': digest(args.checkpoint / 'model.safetensors'),
                'processors': inference_fingerprints(args.checkpoint, None),
                'config': json.loads((args.checkpoint / 'config.json').read_text()),
                'inference_seed': 13000, 'flow_steps': 25, 'horizons': [8,4,2],
                'development_seeds': list(range(13000,13005)), 'confirmation_seeds': list(range(13100,13105)),
                'max_steps': {'2x5':500,'2x10':3200}, 'settings5': settings5, 'settings10': settings10,
                'rules': 'all disabled; 8D left arm, right arm held by standard adapter',
                'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                'source_sha256': {str(p.relative_to(Path(__file__).parent.parent)):digest(p) for p in [
                    Path(__file__), Path(__file__).with_name('experiment_store.py'), Path(__file__).with_name('diagnostics.py'),
                    Path(__file__).with_name('episode_execution.py'), Path(__file__).with_name('cookie_2x10_execution.py'),
                    Path(__file__).with_name('benchmark.py'), Path(__file__).parent.parent/'policies/smolvla.py']}}
    def stop(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    store = ExperimentStore(args.output, protocol)
    policy = None
    try:
        atomic_json(args.output/'status.json', {'status':'running','pid':os.getpid()})
        def run(stage, horizon, task, profile, seed):
            nonlocal policy
            key=f'{stage}_h{horizon}_{task}_{profile}_s{seed}'
            if key in store.records:
                return
            if policy is None or policy.plugin._policy.config.n_action_steps != horizon:
                if policy is not None:
                    policy.plugin.close()
                    import gc
                    gc.collect(); torch.cuda.empty_cache()
                policy = SmolVLAPolicyAdapter(args.checkpoint, device='cuda', n_action_steps=horizon,
                    num_steps=25, inference_seed=13000, ema_alpha=0, gripper_sharpening=False,
                    align_vertical=False, clamp_z=False, anchor_right_arm=False)
            # Every retry has a new directory; incomplete diagnostics are retained.
            parent=args.output/'diagnostics'/key
            parent.mkdir(parents=True,exist_ok=True)
            attempt=0
            while (parent/f'attempt_{attempt}').exists(): attempt+=1
            diagnostic_dir=parent/f'attempt_{attempt}'
            diagnostic_dir.mkdir()
            settings=dict(settings5 if task=='2x5' else settings10, profile=profile, source_column='1')
            bench=make_cookie_benchmark(task=task, prompt_style='training' if task=='2x5' else 'task',
                profile=profile, source_column='1', randomization_settings=settings,
                max_steps=500 if task=='2x5' else 3200, **({'first_grasp':first} if task=='2x10' else {}))
            bench.diagnostic_dir=diagnostic_dir
            atomic_json(args.output/'current.json',{'key':key,'attempt':attempt})
            print('START',key,flush=True)
            result=bench.evaluate(policy,num_episodes=1,seed_start=seed,workers=1,verbose=False).episodes[0]
            trace=diagnostic_dir/f'seed_{seed}'
            validate_trace(trace,result.steps)
            row=dict(asdict(result),stage=stage,horizon=horizon,task=task,profile=profile,diagnostic_dir=str(trace.resolve()))
            store.commit(key,row)
            atomic_json(args.output/'progress.json',{'completed':len(store.records),'last_key':key})
            print('DONE',key,result.cookies_in_target,result.success,flush=True)
        for horizon in (8,4,2):
            for seed in range(13000,13005): run('development',horizon,'2x5','basic',seed)
        development={h:totals([r for r in store.records.values() if r['stage']=='development' and r['horizon']==h]) for h in (8,4,2)}
        candidate=max(development,key=lambda h:(development[h]['successes'],development[h]['cookies'],h))
        atomic_json(args.output/'selection.json',{'development':development,'candidate':candidate})
        recommend=False
        confirmation={}
        if candidate!=8:
            for horizon in (8,candidate):
                for task in ('2x5','2x10'):
                    for profile in ('basic','medium','advanced'):
                        for seed in range(13100,13105): run('confirmation',horizon,task,profile,seed)
            for h in (8,candidate):
                rows=[r for r in store.records.values() if r['stage']=='confirmation' and r['horizon']==h]
                confirmation[h]={'total':totals(rows),'cells':{f'{t}_{p}':totals([r for r in rows if r['task']==t and r['profile']==p]) for t in ('2x5','2x10') for p in ('basic','medium','advanced')}}
            a,b=confirmation[8],confirmation[candidate]
            nondecreasing=lambda x,y: y['successes']>=x['successes'] and y['cookies']>=x['cookies']
            recommend=nondecreasing(a['total'],b['total']) and any(b['total'][k]>a['total'][k] for k in ('successes','cookies')) and all(nondecreasing(a['cells'][c],b['cells'][c]) for c in a['cells'])
        atomic_json(args.output/'result.json',{'development':development,'candidate':candidate,'confirmation':confirmation,'recommend_replace':recommend,'evidence_level':'small-sample development validation'})
        atomic_json(args.output/'status.json',{'status':'completed','episodes':len(store.records)})
    except BaseException as exc:
        atomic_json(args.output/'status.json',{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'error','error':repr(exc),'episodes':len(store.records)})
        raise
    finally:
        if policy is not None: policy.plugin.close()
        store.close()

if __name__=='__main__':
    main()
