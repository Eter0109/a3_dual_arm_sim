"""Bridge an external VLA or world-model project onto the A3 policy protocol.

Importing this package is cheap and pulls in no simulator: MuJoCo is reached only
when a manifest actually asks for a cross-space translation.  That keeps the
protocol, manifest and bridge-server halves usable on a host that cannot import
MuJoCo at all, which is the same constraint ``a3_dual_arm_sim/__init__.py``
already works around.

Start with ``docs/external-policy-bridge.md``.
"""

from __future__ import annotations

__all__: list[str] = []
