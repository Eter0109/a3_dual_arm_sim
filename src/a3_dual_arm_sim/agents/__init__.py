"""Semantic planning, verification and execution orchestration.

Use ``planning`` for subgoals, ``verification`` for completion/progress verdicts,
``executors`` for policy/expert adapters, and ``controller`` for the closed loop.
Visual-language loading lives in ``backends``; training and inference share the
exact templates in ``prompts``. Shared skill types belong to ``core.skills``.

Importing this package never initializes an environment or a model backend.
"""
