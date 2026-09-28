from __future__ import annotations

from xml.etree import ElementTree as ET

from a3_dual_arm_sim.contracts import ARM_JOINTS
from a3_dual_arm_sim.paths import asset_root

from .model_types import JointSource


def _floats(text: str | None, count: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in (text or "").split())
    if len(values) != count:
        raise ValueError(f"expected {count} numbers, got {text!r}")
    return values


def _source() -> tuple[ET.Element, dict[str, JointSource]]:
    root = ET.parse(asset_root() / "source" / "A3_mujoco.urdf").getroot()
    joints: dict[str, JointSource] = {}
    for node in root.findall("joint"):
        name = node.attrib["name"]
        if node.attrib.get("type") != "revolute":
            continue
        limit = node.find("limit")
        if limit is None:
            raise ValueError(f"joint {name} has no limit")
        joints[name] = JointSource(
            name=name,
            parent=node.find("parent").attrib["link"],  # type: ignore[union-attr]
            child=node.find("child").attrib["link"],  # type: ignore[union-attr]
            origin=_floats(node.find("origin").attrib.get("xyz"), 3),  # type: ignore[union-attr]
            axis=_floats(node.find("axis").attrib.get("xyz"), 3),  # type: ignore[union-attr]
            limits=(float(limit.attrib["lower"]), float(limit.attrib["upper"])),
            effort=float(limit.attrib["effort"]),
            velocity=float(limit.attrib["velocity"]),
        )
    if tuple(joints) != ARM_JOINTS:
        raise ValueError(f"unexpected A3 joint order: {tuple(joints)}")
    return root, joints


def _vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)
