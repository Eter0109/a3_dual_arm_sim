"""Evaluation-only copies of native info; never a policy input or new predicate.

Install before the native CLI and close in a finally block after it returns.
The wrappers return the exact original step tuple and forward the exact action.
They inspect only the info already produced by the environment. In particular,
they do not query object poses, contacts, or alter observations/task configuration.
"""
from collections import deque
from functools import wraps
import json
import math
from pathlib import Path
import threading


INFO_FIELDS=(
    'success','cookies_in_target','cookies_in_source','required_cookies',
    'target_slot_occupancy','target_touches_all_walls','count_fill',
    'success_criterion','success_hold_count','required_success_hold_steps',
    'exact_2x5_fill','exact_2x10_fill','target_layout','target_column_counts',
    'safety_reason',
)


def _copy_json(value):
    if hasattr(value,'tolist'):return _copy_json(value.tolist())
    if isinstance(value,dict):return {str(k):_copy_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_copy_json(v) for v in value]
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float):return value if math.isfinite(value) else None
    if hasattr(value,'item'):return _copy_json(value.item())
    raise TypeError('Unsupported native info value: '+type(value).__name__)


def native_failure_components(info):
    """Explain native fields without turning the explanation into a policy gate."""
    if not info:return ['no_native_steps_observed']
    if info.get('success'):return []
    reasons=[]
    required=info.get('required_cookies')
    if required is not None and info.get('cookies_in_target')!=required:
        reasons.append('target_count')
    if info.get('count_fill') is False:reasons.append('target_or_source_count')
    if info.get('success_criterion')=='exact_slots':
        if any(slot<0 for slot in info.get('target_slot_occupancy',[])):
            reasons.append('unoccupied_target_slot')
        if info.get('target_touches_all_walls') is False:
            reasons.append('target_wall_coverage')
    if info.get('target_layout')=='2x10' and info.get('target_column_counts')!=[10,10]:
        reasons.append('target_column_counts')
    hold=info.get('success_hold_count');needed=info.get('required_success_hold_steps')
    if hold is not None and needed is not None and hold<needed:
        reasons.append('native_success_hold_not_completed')
    if info.get('safety_reason'):reasons.append('safety_reason')
    return reasons or ['native_success_false_unexplained_by_recorded_fields']


class NativeBenchmarkDiagnostics:
    """Per-process recorder for one benchmark case; original methods restored."""
    def __init__(self,output,env_classes):
        self.output=Path(output)
        self.steps=0;self.final_info=None;self.recent=deque(maxlen=12)
        self.max_success_hold_count=0;self.diagnostic_errors=[]
        self._local=threading.local();self._installed=[];self._closed=False
        for cls in env_classes:
            # Only a class's own implementation is wrapped. Inherited methods
            # are already covered by the base wrapper and must not be restored
            # as a new override when this diagnostic is uninstalled.
            original=cls.__dict__.get('step')
            if original is None:continue
            wrapper=self._wrap(original)
            setattr(cls,'step',wrapper)
            self._installed.append((cls,original,wrapper))

    def _wrap(self,original):
        @wraps(original)
        def step(env,action,*args,**kwargs):
            depth=getattr(self._local,'depth',0)
            self._local.depth=depth+1
            try:
                result=original(env,action,*args,**kwargs)
            finally:self._local.depth=depth
            if depth==0:
                try:self._observe(result[4])
                except Exception as exc:
                    # A malformed diagnostic value cannot change simulation or
                    # cause a policy decision. It is exposed in the receipt.
                    self.diagnostic_errors.append(type(exc).__name__+': '+str(exc))
            return result
        return step

    def _observe(self,info):
        record={name:_copy_json(info[name]) for name in INFO_FIELDS if name in info}
        self.steps+=1;self.final_info=record
        self.max_success_hold_count=max(self.max_success_hold_count,
            int(record.get('success_hold_count',0)))
        self.recent.append(dict(step=self.steps,**record))

    def close(self):
        if self._closed:return
        self._closed=True
        for cls,original,wrapper in reversed(self._installed):
            if cls.__dict__.get('step') is wrapper:setattr(cls,'step',original)
        record=dict(evaluation_only=True,policy_receives_diagnostics=False,
            original_step_returned_unchanged=True,steps_observed=self.steps,
            final_info=self.final_info,max_success_hold_count=self.max_success_hold_count,
            final_failure_components=native_failure_components(self.final_info),
            recent=list(self.recent),diagnostic_errors=self.diagnostic_errors)
        self.output.mkdir(parents=True,exist_ok=True)
        path=self.output/'final_benchmark_info.json'
        temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temporary.replace(path)


def install_benchmark_diagnostics(output,*,env_classes=None):
    """Return a recorder whose close() saves final_benchmark_info.json."""
    if env_classes is None:
        from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
        from a3_dual_arm_sim.tasks.cookie_2x10 import A3Cookie2x10Env
        env_classes=(A3CookieTransferEnv,A3Cookie2x10Env)
    return NativeBenchmarkDiagnostics(output,env_classes)
