"""Call main's actual benchmark CLI with an external RGB-D policy factory."""
import argparse
import json
import os
from pathlib import Path
import sys
from a3_continuability.runtime import setup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job',type=Path,default=Path('/root/autodl-tmp/pi05_repro_20261015'))
    p.add_argument('--skill-code',type=Path,default=Path('/root/autodl-tmp/pi05_repro_20261015/batch_skill_code/47402a61dc8fe29c/src'))
    p.add_argument('--task',choices=('2x5','2x10'),default='2x5')
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--seed-start',type=int,default=26972)
    p.add_argument('--max-steps',type=int)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--require-release',action='store_true')
    p.add_argument('--right-observe',action='store_true')
    p.add_argument('--right-observe-each-batch',action='store_true')
    p.add_argument('--handeye',action='store_true',help='Physical multi-pose observed-rim calibration before manipulation')
    p.add_argument('--refine',action='store_true',help='Measured view completion and visually validated insert release')
    p.add_argument('--profile',choices=('basic','medium','advanced'),default='basic')
    p.add_argument('--table-height',type=float,help='Measured static tabletop height; enables observed-plane calibration')
    a=p.parse_args()
    if a.output.exists():p.error('Never overwrite a benchmark run')
    sys.path.insert(0,str(a.skill_code));setup(a.job)
    a.output.mkdir(parents=True)
    os.environ['A3_SENSOR_NATIVE_OUTPUT']=str(a.output/'policy')
    os.environ['A3_SENSOR_REQUIRE_RELEASE']='1' if a.require_release else '0'
    os.environ['A3_SENSOR_ACTIVE_RIGHT']='1' if a.right_observe or a.right_observe_each_batch else '0'
    os.environ['A3_SENSOR_RIGHT_EACH_BATCH']='1' if a.right_observe_each_batch else '0'
    os.environ['A3_SENSOR_SOURCE_COLUMN']='1'
    os.environ['A3_SENSOR_HANDEYE']='1' if a.handeye else '0'
    os.environ['A3_SENSOR_REFINEMENT']='1' if a.refine else '0'
    if a.handeye and a.table_height is None:p.error('--handeye requires a measured --table-height')
    os.environ['A3_SENSOR_SNAPSHOTS']='failure'
    if a.table_height is not None:os.environ['A3_SENSOR_TABLE_HEIGHT_M']=str(a.table_height)
    else:os.environ.pop('A3_SENSOR_TABLE_HEIGHT_M',None)
    from a3_dual_arm_sim.workflows import benchmark_cookie_batch_cli
    cli=['benchmark_cookie_batch','--policy','a3_continuability.native_sensor_policy:make_policy',
        '--task',a.task,'--profile',a.profile,'--source-column','1',
        '--episodes',str(a.episodes),'--seed-start',str(a.seed_start),
        '--max-steps',str(a.max_steps or (1500 if a.task=='2x5' else 3000)),
        '--workers',str(a.workers),'--output',str(a.output/'native_benchmark.json')]
    if a.task=='2x10':cli+=['--first-grasp','5']
    (a.output/'protocol.json').write_text(json.dumps(dict(entry='main benchmark_cookie_batch_cli.main',
        argv=cli,source_commit=(a.job/'benchmark_main_29fe90e/SOURCE_COMMIT.txt').read_text().strip(),
        main_sources_edited=False,native_camera_configuration=True,native_rendering=True,
        native_success_predicate=True,added_modality='768px metric RGB-D on the same three cameras, independent rasterizer',
        action_source='classic_rgbd_visual_expert',object_truth_for_control=False,
        optional_right_observation=a.right_observe or a.right_observe_each_batch,
        right_observation_each_batch=a.right_observe_each_batch,profile=a.profile,
        observed_handeye_calibration=a.handeye,
        task_conditioned_completion=a.refine,
        observed_plane_calibration=a.table_height is not None,table_height_m=a.table_height,
        require_release_before_episode_end=a.require_release,
        note='Explicit 1500/3000 step budgets per project preference; not default 1000/3200 budgets'),indent=2))
    sys.argv=cli
    benchmark_cookie_batch_cli.main()
    report=json.loads((a.output/'native_benchmark.json').read_text())
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    (a.output/'status.json').write_text(json.dumps(dict(status='completed',episodes=len(report['episodes']))))


if __name__=='__main__':main()
