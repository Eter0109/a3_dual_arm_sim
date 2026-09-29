"""Regression checks for the Agent package's dependency direction."""

from __future__ import annotations

import ast
from pathlib import Path

from a3_dual_arm_sim.agents import backends, controller, executors, planning, prompts, verification
from a3_dual_arm_sim.core import skills as core_skills
from a3_dual_arm_sim.core.skills import CookieSkill, SkillRequest


def test_agents_share_the_core_skill_contracts():
    assert planning.CookieSkill is CookieSkill
    assert planning.SkillRequest is SkillRequest
    assert executors.CookieSkill is CookieSkill
    assert executors.SkillRequest is SkillRequest
    assert verification.CookieSkill is CookieSkill
    assert verification.SkillRequest is SkillRequest
    assert prompts.CookieSkill is CookieSkill
    assert controller.CookieSkill is CookieSkill


def test_agents_do_not_depend_on_datasets_or_training():
    package = Path(controller.__file__).parent
    for path in package.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith((
                    "a3_dual_arm_sim.data",
                    "a3_dual_arm_sim.training",
                    "a3_dual_arm_sim.simulation",
                )), (path.name, module)
            elif isinstance(node, ast.Import):
                for item in node.names:
                    assert not item.name.startswith((
                        "a3_dual_arm_sim.data",
                        "a3_dual_arm_sim.training",
                        "a3_dual_arm_sim.simulation",
                    )), (path.name, item.name)


def _eager_imports(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from _eager_imports(child)


def test_heavy_backends_and_experts_have_no_eager_imports():
    package = Path(controller.__file__).parent
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Check class/global branches too; only function bodies are lazy.
        for node in _eager_imports(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith((
                    "torch", "transformers", "peft", "mujoco", "a3_dual_arm_sim.experts",
                )), (path.name, module)
            elif isinstance(node, ast.Import):
                for item in node.names:
                    assert not item.name.startswith((
                        "torch", "transformers", "peft", "mujoco", "a3_dual_arm_sim.experts",
                    )), (path.name, item.name)


def test_core_skill_contracts_have_no_reverse_package_dependencies():
    tree = ast.parse(Path(core_skills.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("a3_dual_arm_sim")
        elif isinstance(node, ast.Import):
            assert all(not item.name.startswith("a3_dual_arm_sim") for item in node.names)


def test_bundled_checkpoint_paths_are_independent_of_working_directory(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(backends, "project_root", lambda: checkout)
    monkeypatch.chdir(elsewhere)
    assert backends._project_checkpoint_path("models/Qwen3.5-4B") == str(
        checkout / "models" / "Qwen3.5-4B"
    )
    assert backends._project_checkpoint_path("outputs/train/verifier/adapter") == str(
        checkout / "outputs" / "train" / "verifier" / "adapter"
    )
    absolute = str(tmp_path / "explicit-checkpoint")
    assert backends._project_checkpoint_path(absolute) == absolute
    assert backends._project_checkpoint_path("Qwen/Qwen3.5-4B") == "Qwen/Qwen3.5-4B"
