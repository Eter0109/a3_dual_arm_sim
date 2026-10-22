"""Paired grid planning and honest partial summaries, independent of rendering."""
from collections import Counter

TASKS=('2x5','2x10')
PROFILES=('basic','medium','advanced')


def cases(seed_start=13000,episodes=20,*,tasks=TASKS,profiles=PROFILES):
    if not tasks or not profiles or set(tasks)-set(TASKS) or set(profiles)-set(PROFILES):
        raise ValueError('Invalid task/profile selection')
    if len(set(tasks))!=len(tasks) or len(set(profiles))!=len(profiles):
        raise ValueError('Duplicate grid cells would run conflicting case directories')
    return [dict(task=t,profile=p,seed=s) for s in range(seed_start,seed_start+episodes)
            for t in tasks for p in profiles]


def key(case):return f"{case['task']}_{case['profile']}_{case['seed']}"


def summarize(rows,planned_per_cell,*,tasks=TASKS,profiles=PROFILES):
    result=[]
    for task in tasks:
        for profile in profiles:
            group=[r for r in rows if r['task']==task and r['profile']==profile]
            valid=[r for r in group if not r.get('infrastructure_error')]
            counts=[r['cookies_in_target'] for r in valid]
            successes=sum(bool(r['success']) for r in valid)
            verified=sum(bool(r['success']) and r.get('controller_phase')=='DONE' for r in valid)
            reasons=Counter(r.get('controller_failure') or r.get('failure_reason') or 'unspecified'
                for r in valid if not r['success'] or r.get('controller_phase')!='DONE')
            result.append(dict(task=task,profile=profile,completed=len(group),planned=planned_per_cell,
                valid_episodes=len(valid),infrastructure_errors=len(group)-len(valid),
                native_successes=successes,native_success_rate=successes/len(valid) if valid else None,
                release_verified_successes=verified,release_verified_success_rate=verified/len(valid) if valid else None,
                mean_final_count=sum(counts)/len(counts) if counts else None,
                count_distribution=dict(Counter(counts)),failure_reasons=dict(reasons)))
    return result
