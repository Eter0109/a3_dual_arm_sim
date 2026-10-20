"""Physically executed second-round suffix demonstrations, one atomic shard per episode."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from a3_dual_arm_sim.controllers.same_column_batch_expert import A3VariedColumnBatchExpert
from a3_dual_arm_sim.controllers.expert import CookiePhase
from a3_dual_arm_sim.paths import project_root

QUOTAS = {'normal': 40, 'offset': 30, 'empty_recovery': 20, 'first_stack_variation': 10}


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


class TargetedExpert(A3VariedColumnBatchExpert):
    def __init__(self, env, category, seed):
        self.category = category
        rng = np.random.default_rng(seed)
        angle = rng.uniform(0, 2*np.pi)
        self.offset = rng.uniform(.005, .012)*np.array([np.cos(angle), np.sin(angle), 0.])
        self.yaw = rng.choice([-1, 1])*rng.uniform(.05, .12)
        self.first_offset = rng.choice([-1, 1])*.001
        super().__init__(env)

    def reset(self, context=None):
        super().reset(context)
        self.record_enabled = False
        self.intervention_done = False
        self.stage = None
        self.stage_steps = 0
        self.events = []
        self.tick = -1

    def _place_pose(self, clearance=0., column_index=None):
        pos, rot = super()._place_pose(clearance, column_index)
        if self.category == 'first_stack_variation' and self.batch_index == 0 and column_index is None:
            pos = pos + self.data.xmat[self._tb_id].reshape(3,3)[:, 0]*self.first_offset
        return pos, rot

    def _act_step(self, observation=None, task=''):
        self.tick += 1
        if self.batch_index == 1 and not self.intervention_done:
            if self.category in ('normal', 'first_stack_variation'):
                self.record_enabled = self.intervention_done = True
                self.events.append({'step': self.tick, 'event': 'suffix_start'})
            elif self.phase is CookiePhase.PRE_CLOSE and self.stage is None:
                self.stage = 'perturb'
                self.saved_opening = self._opening
                self.stage_steps = 0
                self.anchor = self.data.site_xpos[self._l_site].copy()
                self.rotation = self.data.site_xmat[self._l_site].reshape(3,3).copy()
                self.cookie_before = self.data.xpos[list(self.env._cookie_bodies)].copy()
                self.events.append({'step': self.tick, 'event': 'perturb_start'})
            if self.stage:
                self.stage_steps += 1
                if self.stage_steps > 180:
                    self._fail('targeted intervention timed out')
                    return self._hold_command(self._opening)
                if self.stage == 'perturb':
                    if self.category == 'offset':
                        c, s = np.cos(self.yaw), np.sin(self.yaw)
                        rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]]) @ self.rotation
                        action, reached = self._servo(self.anchor+self.offset, rotation, 1., speed=.003)
                        ready = reached
                    else:
                        # Close above the source; no object teleport or artificial attachment.
                        action, reached = self._servo(self.anchor, self.rotation, .20, speed=.003)
                        ready = abs(float(self.env.current_joint_action[7])-.20)<.015
                    if ready and self.stage_steps >= 12:
                        self.stage = 'recover'
                        self.stage_steps = 0
                    return action
                if self.stage == 'recover':
                    if not self.record_enabled:
                        displacement = np.max(np.linalg.norm(self.data.xpos[list(self.env._cookie_bodies)]-self.cookie_before, axis=1))
                        if self.category == 'empty_recovery' and displacement > .004:
                            self._fail('empty recovery invalid: cookies moved during empty close')
                            return self._hold_command(self._opening)
                        self.events.append({'step': self.tick, 'event': 'suffix_start', 'anchor_eef': self.anchor.tolist(), 'cookie_max_displacement_m': float(displacement), 'eef_position': self.data.site_xpos[self._l_site].tolist(), 'gripper': float(self.env.current_joint_action[7])})
                        self.record_enabled = True
                    if self.category == 'empty_recovery':
                        # Open at the current pose before retreating/reapproaching.
                        action, _ = self._servo(self.anchor, self.rotation, 1., speed=.003)
                        if float(self.env.current_joint_action[7]) < .98:
                            return action
                        action, reached = self._servo(self.anchor + [0., 0., .02], self.rotation, 1., speed=.003)
                        if not reached:
                            return action
                    self._opening = self.saved_opening
                    self.intervention_done = True
                    self.stage = None
                    self._advance(CookiePhase.APPROACH)
        return super()._act_step(observation, task)


class SuffixRecorder:
    def __init__(self, recorder, policy):
        self.recorder, self.policy = recorder, policy
        self.started = False
        self.frames = 0
        self.metadata = {}
        self.total = 0
        self.start_step = None
    def start_episode(self, context, name):
        self.context, self.name = context, name
    def set_episode_metadata(self, value):
        self.metadata = value
    def add_frame(self, observation, action):
        if self.policy.expert.record_enabled:
            if not self.started:
                self.start_step = self.total
                self.recorder.start_episode(self.context, self.name)
                self.started = True
            self.recorder.add_frame(observation, action)
            self.frames += 1
        self.total += 1
    def finish_episode(self, success):
        if not self.started or not self.policy.expert.intervention_done:
            raise RuntimeError('No valid targeted suffix')
        self.recorder.set_episode_metadata({**self.metadata, 'targeted_category': self.policy.expert.category, 'recording_start_step': self.start_step, 'prefix_source': 'physical_expert', 'intervention_events': self.policy.expert.events})
        self.recorder.finish_episode(success)
    def discard_episode(self):
        if self.started:
            self.recorder.discard_episode()


def worker(args):
    from a3_dual_arm_sim.data.recording import LeRobotV3Recorder
    from a3_dual_arm_sim.data.audit import audit_training_dataset
    from a3_dual_arm_sim.data.prepare_left_arm_dataset_cli import convert
    from a3_dual_arm_sim.sim.randomization import load_randomization_config
    from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark
    from a3_dual_arm_sim.workflows.policy_adapters import ExpertPolicyAdapter
    from a3_dual_arm_sim.workflows.diagnostics import EpisodeDiagnostic
    import pyarrow.parquet as pq
    import cv2
    root = args.root
    # Keep the collection profile in the protocol while using the tracked
    # randomization configuration instead of a local audit run directory.
    settings = load_randomization_config(
        config={**load_randomization_config(project_root()/'configs/randomization.yaml'),
                'profile': 'medium'}
    )
    bench = CookieBatchBenchmark(project_root()/'configs/cookie_batch.yaml', max_steps=1600, complete_demonstrations=True, success_hold_steps=20, profile='medium', source_column=1, randomization_settings=settings)
    policy = ExpertPolicyAdapter(lambda env: TargetedExpert(env, args.category, args.seed), name='second_round_'+args.category)
    raw = root/'raw'
    rec = LeRobotV3Recorder(raw, repo_id='local/second-round', fps=bench.config.control_hz, image_height=256, image_width=256, use_videos=True)
    gated = SuffixRecorder(rec, policy)
    diag = EpisodeDiagnostic(root/'diagnostic', fps=bench.config.control_hz)
    try:
        score = bench.run_episode(policy, args.seed, recorder=gated, diagnostic=diag)
    finally:
        diag.close()
        rec.close()
    result = {'score': asdict(score), 'category': args.category, 'seed': args.seed, 'recorded_frames': gated.frames, 'recording_start_step': gated.start_step, 'events': policy.expert.events, 'offset_m': policy.expert.offset.tolist(), 'yaw_rad': float(policy.expert.yaw), 'first_stack_offset_m': float(policy.expert.first_offset)}
    save(root/'result.json', result)
    if not score.success:
        return 2
    summary = {'schema_version':2, 'task':'cookie_transfer', 'task_instruction':'transfer 10 cookies into target box', 'stored_action_mode':'joint_position', 'source_action_mode':'cartesian_delta', 'complete_demonstrations':True, 'placement_order':'front_camera_bottom_first', 'randomization_profile':'medium', 'source_column':1, 'use_videos':True, 'config':asdict(bench.config), 'sample_scope':'second_round_suffix', 'prefix_source':'physical_expert', 'category':args.category, 'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'randomization_settings':settings}
    save(raw/'collection_summary.json', summary)
    report = audit_training_dataset(raw, repo_id='local/second-round')
    rows = [json.loads(line) for line in (root/'diagnostic/trace.jsonl').read_text().splitlines()]
    assert len(rows) == score.steps == gated.total
    assert len(rows)-gated.start_step == gated.frames == report['frames']
    saved = np.concatenate([np.asarray(pq.read_table(p, columns=['action'])['action'].to_pylist()) for p in sorted((raw/'data').rglob('*.parquet'))])
    expected = np.asarray([r['applied_action'] for r in rows[gated.start_step:]])
    np.testing.assert_allclose(saved, expected, atol=1e-6)
    for p in (root/'diagnostic').glob('*.mp4'):
        video = cv2.VideoCapture(str(p)); count=0
        while video.read()[0]: count+=1
        video.release()
        assert count == score.steps, (p, count, score.steps)
    assert len(list((root/'diagnostic').glob('*.mp4'))) == 3
    reports = score.randomization['grasp_reports']
    assert len(reports) == 2 and all(r.get('lifted') == 5 and r.get('released') for r in reports)
    assert rows[gated.start_step-1]['after']['cookies_in_target'] == 5
    if args.category in ('offset', 'empty_recovery'):
        event = policy.expert.events[-1]
        assert event['event'] == 'suffix_start'
        if args.category == 'offset':
            assert np.linalg.norm(np.asarray(event['eef_position'])-event['anchor_eef']) >= .002
        else:
            assert event['gripper'] < .25 and event['cookie_max_displacement_m'] < .004
    assert rows[-1]['after']['cookies_in_target'] == 10
    assert rows[-1]['after']['success_hold_count'] >= 20
    convert(raw, root/'left11', include_left_eef_position=True)
    result['audit'] = audit_training_dataset(root/'left11', repo_id='local/second-round')
    result['committed'] = True
    save(root/'COMMITTED.json', result)
    print(json.dumps({'committed':str(root), 'frames':gated.frames}), flush=True)
    return 0


def merge_shards(root):
    """Build a separate ready-to-train dataset only after all committed shards pass."""
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.compute_stats import aggregate_stats
    from a3_dual_arm_sim.data.audit import audit_training_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq
    paths = []
    for category, quota in QUOTAS.items():
        group = sorted(root.glob(category+'/*/COMMITTED.json'))
        if len(group) != quota:
            raise RuntimeError('Incomplete category: '+category)
        paths.extend(p.parent/'left11' for p in group)
    destination = root/'training_left11'
    if destination.exists():
        return audit_training_dataset(destination, repo_id='local/cookie-second-round-medium100-left11')
    temporary = root/'training_left11.incomplete'
    if temporary.exists():
        raise RuntimeError('Interrupted merge retained at '+str(temporary)+'; inspect before retry')
    reports = [audit_training_dataset(p, repo_id='local/second-round') for p in paths]
    merge_datasets([LeRobotDataset('local/second-round', root=p, video_backend='pyav') for p in paths], 'local/cookie-second-round-medium100-left11', output_dir=temporary)
    metadata_files = sorted((temporary/'meta/episodes').rglob('*.parquet'))
    per_episode_stats = []
    for file in metadata_files:
        table = pq.read_table(file)
        for key, value in [('meta/episodes/file_index', int(file.stem.split('-')[-1])), ('meta/episodes/chunk_index', int(file.parent.name.split('-')[-1]))]:
            if key in table.column_names:
                table = table.set_column(table.schema.get_field_index(key), key, pa.array([value]*len(table), type=table.schema.field(key).type))
        pq.write_table(table, file)
        for row in table.to_pylist():
            stats = {}
            for k, v in row.items():
                if k.startswith('stats/'):
                    _, feature, stat = k.split('/', 2)
                    stats.setdefault(feature, {})[stat] = np.asarray(v)
            per_episode_stats.append(stats)
    save(temporary/'meta/stats.json', json.loads(json.dumps(aggregate_stats(per_episode_stats), default=lambda x:x.tolist())))
    sidecars = []
    for index, path in enumerate(paths):
        rows = [json.loads(x) for x in (path/'a3_episode_metadata.jsonl').read_text().splitlines()]
        if len(rows) != 1: raise RuntimeError('Expected one-episode shard')
        rows[0].update(episode_index=index, source_shard=str(path), source_episode_index=0)
        sidecars.append(rows[0])
    (temporary/'a3_episode_metadata.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in sidecars))
    summary = json.loads((paths[0]/'collection_summary.json').read_text())
    summary.pop('category', None)
    summary.update(category_counts=QUOTAS, components=[str(p) for p in paths], sample_scope='second_round_suffix', provenance='controlled_expert_demonstrations_not_model_takeover')
    save(temporary/'collection_summary.json', summary)
    report = audit_training_dataset(temporary, repo_id='local/cookie-second-round-medium100-left11')
    if report['episodes'] != 100 or report['frames'] != sum(r['frames'] for r in reports):
        raise RuntimeError('Merged counts mismatch')
    temporary.rename(destination)
    report['root'] = str(destination)
    save(root/'TRAINING_DATA_READY.json', report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--category', choices=QUOTAS)
    parser.add_argument('--seed', type=int, default=25000)
    args=parser.parse_args()
    args.root=args.root.resolve()
    if args.worker:
        return worker(args)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/'writer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest={'version':1, 'quotas':QUOTAS, 'seed_start':args.seed, 'profile':'medium', 'source_column':1, 'prefix':'physical_expert', 'recovery_scope':'controlled_empty_grasp_only', 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if (args.root/'plan.json').exists():
            if json.loads((args.root/'plan.json').read_text()) != manifest:
                raise RuntimeError('Collection protocol changed; use a new root')
        else: save(args.root/'plan.json', manifest)
        counts={key: 0 for key in QUOTAS}
        for ci,(category, quota) in enumerate(QUOTAS.items()):
            wanted=1 if args.pilot else quota
            paths=sorted(args.root.glob(category+'/*/COMMITTED.json'))
            from a3_dual_arm_sim.data.audit import audit_training_dataset
            for committed in paths:
                previous = json.loads(committed.read_text())
                if previous['category'] != category or not previous['committed']:
                    raise RuntimeError('Invalid committed episode')
                checked = audit_training_dataset(committed.parent/'left11', repo_id='local/second-round')
                if checked['frames'] != previous['recorded_frames']:
                    raise RuntimeError('Committed episode data changed')
            seeds=[json.loads(p.read_text())['seed'] for p in paths]
            if len(seeds)!=len(set(seeds)): raise RuntimeError('Duplicate episode keys')
            counts[category]=len(paths)
            for attempt in range(quota*12):
                if counts[category]>=wanted: break
                seed=args.seed+ci*1000+attempt
                destination=args.root/category/f'seed_{seed}'
                if destination.exists(): continue
                destination.mkdir(parents=True)
                with (destination/'worker.log').open('w') as log:
                    run=subprocess.run([sys.executable,'-m','a3_dual_arm_sim.data.collect_second_round_cli','--worker','--root',str(destination),'--category',category,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
                if (destination/'COMMITTED.json').exists(): counts[category]+=1
                save(args.root/'status.json', {'counts':counts, 'last_seed':seed, 'last_returncode':run.returncode, 'pilot':args.pilot})
                print(json.dumps({'category':category,'count':counts[category],'seed':seed,'returncode':run.returncode}),flush=True)
                if run.returncode not in (0,2): return run.returncode
            if counts[category]<wanted: raise RuntimeError('Attempt budget exhausted: '+category)
        save(args.root/('PILOT_COMPLETE.json' if args.pilot else 'COMPLETE.json'), {'counts':counts})
        if not args.pilot:
            merge_shards(args.root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
