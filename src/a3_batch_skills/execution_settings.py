"""Execution-only overrides, including the shared-model process boundary."""
import numbers


def checked_action_steps(value, chunk_size):
    if not isinstance(value, numbers.Integral) or not 1 <= value <= chunk_size:
        raise ValueError('Action execution horizon must be within the trained chunk')
    return int(value)


def set_action_steps(policy, value):
    value = checked_action_steps(value, policy.config.chunk_size)
    if hasattr(policy, 'configure_execution'):
        policy.configure_execution(value)
    else:
        policy.config.n_action_steps = value
        policy.policy.config.n_action_steps = value
    return value


def target_planning_settings(spec):
    fraction = float(spec.get('target_floor_fraction', .90))
    tolerance = float(spec.get('target_overlap_tolerance_m', .001))
    if not .80 <= fraction <= 1. or not 0. <= tolerance <= .012:
        raise ValueError('Unsupported target planning tolerance')
    return fraction, tolerance
