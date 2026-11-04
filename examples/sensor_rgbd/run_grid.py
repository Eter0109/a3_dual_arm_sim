"""Run the original frozen V7 or V9 grid with portable explicit paths.

The controller and benchmark remain unchanged. V7 is the default; V9 is an
explicit opt-in to the measured-gap release sequence evaluated in the report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def build_command(args):
    bundle = ROOT / 'bundles' / args.variant
    manifest = json.loads((bundle/'manifest.json').read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen release checksum mismatch: '+name)
    job = args.job.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if job == output or job.is_relative_to(output):
        raise ValueError('The output directory cannot contain the workspace')
    for relative in ('benchmark_main_29fe90e/SOURCE_COMMIT.txt',
                     'benchmark_research_v3/determinism.py'):
        if not (job/relative).is_file():
            raise ValueError('Prepare this workspace first; missing '+relative)
    commit = (job/'benchmark_main_29fe90e/SOURCE_COMMIT.txt').read_text().strip()
    if commit != '29fe90e423259f388ebc061926c3ec860b9edc74':
        raise ValueError('The workspace does not use the pinned benchmark')
    simulator = job/'benchmark_main_29fe90e'
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=simulator,text=True).strip()
    if head != commit:
        raise ValueError('The actual simulator Git HEAD differs from the pinned benchmark')
    if subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=simulator,text=True).strip():
        raise ValueError('The simulator has tracked modifications; use a clean pinned checkout')
    command = [sys.executable, '-u', str(bundle/'code/run_main_sensor_grid.py'),
               '--job',str(job), '--skill-code',str(bundle/'code/vendor'),
               '--output',str(output), '--workers',str(args.workers),
               '--episodes',str(args.episodes), '--seed-start',str(args.seed_start),
               '--case-timeout',str(args.case_timeout),
               '--tasks',*args.tasks, '--profiles',*args.profiles,
               '--handeye','--table-height','.75','--refine',
               '--scene-memory','--placement-planning']
    if args.variant == 'v9':
        command.append('--guarded-release')
    if args.resume:
        command.append('--resume')
    env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join([str(bundle/'code'), str(bundle/'code/vendor'),
        *([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])])
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('v7','v9'),default='v7')
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tasks',nargs='+',choices=('2x5','2x10'),default=['2x5','2x10'])
    parser.add_argument('--profiles',nargs='+',choices=('basic','medium','advanced'),
                        default=['basic','medium','advanced'])
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--seed-start',type=int,default=28000)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--case-timeout',type=int,default=1800)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Verify hashes and print the command only')
    args = parser.parse_args()
    if min(args.episodes,args.workers,args.case_timeout)<1:
        parser.error('episodes, workers and case-timeout must be positive')
    command, env = build_command(args)
    print(json.dumps(dict(variant=args.variant,command=shlex.join(command),
        original_controller_unchanged=True,budgets={'2x5':1500,'2x10':3000}),ensure_ascii=False),flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(command,env=env))


if __name__ == '__main__':
    main()
