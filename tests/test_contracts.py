from __future__ import annotations

import numpy as np
import pytest

from a3_dual_arm_sim.contracts import ContractError, validate_action


def test_action_contract_shapes_and_cartesian_range() -> None:
    assert validate_action(np.zeros(16), "joint_position").shape == (16,)
    assert validate_action(np.zeros(14), "cartesian_delta").shape == (14,)
    with pytest.raises(ContractError, match="shape"):
        validate_action(np.zeros(15), "joint_position")
    with pytest.raises(ContractError, match="NaN"):
        action = np.zeros(14)
        action[0] = np.nan
        validate_action(action, "cartesian_delta")
    with pytest.raises(ContractError, match=r"\[-1, 1\]"):
        validate_action(np.full(14, 1.1), "cartesian_delta")

