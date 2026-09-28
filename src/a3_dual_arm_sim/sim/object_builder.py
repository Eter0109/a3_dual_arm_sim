from __future__ import annotations

from xml.etree import ElementTree as ET

from a3_dual_arm_sim.config import SimConfig

from .urdf import _vec

BIN_RGBA = "0.70 0.73 0.77 1"


def _add_open_bin(
    parent: ET.Element,
    *,
    name: str,
    center: tuple[float, float],
    half_size: tuple[float, float],
    height: float,
    rgba: str,
    thickness: float,
    friction: tuple[float, ...],
    floor_z: float = 0.753,
    wall_z: float | None = None,
) -> None:
    """Create a collision-enabled, open-topped box on the work surface."""
    x, y = center
    hx, hy = half_size
    if wall_z is None:
        wall_z = floor_z + height / 2
    collision_common = {
        "rgba": "0 0 0 0",
        "contype": "1",
        "conaffinity": "3",
        "friction": _vec(friction),
        "group": "3",
    }
    visual_common = {
        "rgba": rgba,
        "contype": "0",
        "conaffinity": "0",
        "group": "2",
    }
    ET.SubElement(
        parent,
        "geom",
        name=f"{name}_floor",
        type="box",
        pos=f"{x} {y} {floor_z}",
        size=f"{hx} {hy} {thickness / 2}",
        **collision_common,
    )
    ET.SubElement(
        parent,
        "geom",
        name=f"{name}_floor_visual",
        type="box",
        pos=f"{x} {y} {floor_z}",
        size=f"{hx} {hy} {thickness / 2}",
        **visual_common,
    )
    collision_walls = (
        # Extend adjacent walls by one thickness.  This makes the four corners
        # overlap instead of leaving diagonal gaps a small block can escape through.
        ("front", x + hx, y, thickness, hy + thickness),
        ("back", x - hx, y, thickness, hy + thickness),
        ("left", x, y + hy, hx + thickness, thickness),
        ("right", x, y - hy, hx + thickness, thickness),
    )
    for suffix, px, py, sx, sy in collision_walls:
        ET.SubElement(
            parent,
            "geom",
            name=f"{name}_{suffix}_wall",
            type="box",
            pos=f"{px} {py} {wall_z}",
            size=f"{sx} {sy} {height / 2}",
            **collision_common,
        )

    # Visible walls tile the outline without overlapping. Collision walls above
    # retain overlap at the corners, following robosuite's visual/collision split.
    visual_walls = (
        ("front", x + hx, y, thickness, hy - thickness),
        ("back", x - hx, y, thickness, hy - thickness),
        ("left", x, y + hy, hx + thickness, thickness),
        ("right", x, y - hy, hx + thickness, thickness),
    )
    for suffix, px, py, sx, sy in visual_walls:
        ET.SubElement(
            parent,
            "geom",
            name=f"{name}_{suffix}_wall_visual",
            type="box",
            pos=f"{px} {py} {wall_z}",
            size=f"{sx} {sy} {height / 2}",
            **visual_common,
        )


def _add_beveled_cookie_mesh(
    assets: ET.Element,
    name: str,
    half_size: tuple[float, ...],
    bevel: float,
    bottom_bevel: float | None = None,
) -> None:
    """Extrude an octagonal y-z profile along x for real sloped contacts."""
    if len(half_size) != 3 or any(value <= 0 for value in half_size):
        raise ValueError("cookie half-size must contain three positive values")
    hx, hy, hz = half_size
    if bottom_bevel is None:
        bottom_bevel = bevel
    if not 0 < bevel < min(hy, hz):
        raise ValueError("cookie edge bevel must be positive and smaller than y/z half-size")
    if not 0 < bottom_bevel < min(hy, hz):
        raise ValueError("cookie bottom bevel must be positive and smaller than y/z half-size")

    # Keep the overall x width unchanged. Four long x-direction edges are
    # cut away so a descending finger meets a ramp rather than a square corner.
    profile = (
        (-hy + bevel, hz),
        (hy - bevel, hz),
        (hy, hz - bevel),
        (hy, -hz + bottom_bevel),
        (hy - bottom_bevel, -hz),
        (-hy + bottom_bevel, -hz),
        (-hy, -hz + bottom_bevel),
        (-hy, hz - bevel),
    )
    vertices = tuple((x, y, z) for x in (-hx, hx) for y, z in profile)
    faces: list[tuple[int, int, int]] = []
    for index in range(1, 7):
        faces.extend(((0, index, index + 1), (8, 8 + index + 1, 8 + index)))
    for index in range(8):
        next_index = (index + 1) % 8
        faces.extend(
            (
                (index, 8 + index, 8 + next_index),
                (index, 8 + next_index, next_index),
            )
        )
    ET.SubElement(
        assets,
        "mesh",
        name=name,
        vertex=_vec(tuple(value for vertex in vertices for value in vertex)),
        face=" ".join(str(value) for triangle in faces for value in triangle),
    )


def _add_cookie_scene(root: ET.Element, world: ET.Element, config: SimConfig) -> None:
    scene = config.cookie_transfer
    assets = root.find("asset")
    if assets is None:
        raise RuntimeError("model assets must exist before adding the Cookie scene")
    # Keep visual and collision envelopes coincident.  A smaller visual mesh
    # made a correctly contacting Cookie look detached from the finger pads.
    visual_half_size = scene.cookie_half_size_m
    bottom_bevel = (
        scene.cookie_edge_bevel_m
        if scene.cookie_bottom_edge_bevel_m is None
        else scene.cookie_bottom_edge_bevel_m
    )
    _add_beveled_cookie_mesh(
        assets,
        "cookie_collision_mesh",
        scene.cookie_half_size_m,
        scene.cookie_edge_bevel_m,
        bottom_bevel,
    )
    _add_beveled_cookie_mesh(
        assets,
        "cookie_visual_mesh",
        visual_half_size,
        scene.cookie_edge_bevel_m,
        bottom_bevel,
    )
    if scene.cookie_collision_mode not in ("mesh", "compound"):
        raise ValueError("cookie_collision_mode must be mesh or compound")
    if scene.cookie_collision_mode == "compound":
        hx, hy, hz = scene.cookie_half_size_m
        bevel = scene.cookie_edge_bevel_m
        # Exact non-overlapping decomposition of the same octagonal solid.
        # Flat side contacts use box-box collision; ramps remain real meshes.
        for label, sign, edge in (("top", 1, bevel), ("bottom", -1, bottom_bevel)):
            vertices = [
                (x, y, sign * z)
                for x in (-hx, hx)
                for z, width in ((hz - edge, hy), (hz, hy - edge))
                for y in (-width, width)
            ]
            ET.SubElement(
                assets,
                "mesh",
                name=f"cookie_{label}_mesh",
                vertex=_vec(tuple(v for vertex in vertices for v in vertex)),
            )
    # Source bin: on table
    source_bin = ET.SubElement(
        world,
        "body",
        name="source_bin",
        pos=_vec((scene.source_bin_center_m[0], scene.source_bin_center_m[1], 0.0)),
    )
    _add_open_bin(
        source_bin,
        name="source_bin",
        center=(0.0, 0.0),
        half_size=scene.source_bin_half_size_m,
        height=scene.source_bin_wall_height_m,
        rgba=BIN_RGBA,
        thickness=scene.bin_wall_thickness_m,
        friction=scene.bin_friction,
        floor_z=scene.source_floor_z_m,
        wall_z=scene.source_wall_base_z_m + scene.source_bin_wall_height_m / 2,
    )

    # A free rigid box can only follow the gripper through physical contact.
    target_bin = ET.SubElement(
        world,
        "body",
        name="target_bin",
        pos=_vec(scene.target_bin_world_position_m),
    )
    ET.SubElement(target_bin, "freejoint", name="target_bin_free")
    ET.SubElement(
        target_bin, "inertial", pos="0 0 0.01", mass="0.10", diaginertia="0.0002 0.0002 0.0003"
    )
    _add_open_bin(
        target_bin,
        name="target_bin",
        center=(0.0, 0.0),
        half_size=scene.target_bin_half_size_m,
        height=scene.target_bin_wall_height_m,
        rgba=BIN_RGBA,
        thickness=scene.bin_wall_thickness_m,
        friction=scene.bin_friction,
        floor_z=scene.target_floor_z_m,
        wall_z=scene.target_bin_wall_height_m / 2,
    )
    if scene.spare_target_bin_world_position_m is not None:
        spare_bin = ET.SubElement(
            world,
            "body",
            name="spare_target_bin",
            pos=_vec(scene.spare_target_bin_world_position_m),
        )
        ET.SubElement(spare_bin, "freejoint", name="spare_target_bin_free")
        ET.SubElement(
            spare_bin,
            "inertial",
            pos="0 0 0.01",
            mass="0.10",
            diaginertia="0.0002 0.0002 0.0003",
        )
        _add_open_bin(
            spare_bin,
            name="spare_target_bin",
            center=(0.0, 0.0),
            half_size=scene.target_bin_half_size_m,
            height=scene.target_bin_wall_height_m,
            rgba=BIN_RGBA,
            thickness=scene.bin_wall_thickness_m,
            friction=scene.bin_friction,
            floor_z=scene.target_floor_z_m,
            wall_z=scene.target_bin_wall_height_m / 2,
        )

    columns = sorted({p[0] for p in scene.cookie_source_positions_m})
    rows = sorted({p[1] for p in scene.cookie_source_positions_m})
    for index, (x, y) in enumerate(scene.cookie_source_positions_m):
        cookie = ET.SubElement(
            world,
            "body",
            name=f"cookie_{index}",
            pos=f"{x} {y} {scene.cookie_model_z_m}",
        )
        ET.SubElement(cookie, "freejoint", name=f"cookie_{index}_free")
        collision = ET.SubElement(
            cookie,
            "geom",
            name=f"cookie_{index}_geom",
            type="mesh",
            mesh="cookie_collision_mesh",
            mass=f"{scene.cookie_mass_kg:.10g}",
            rgba="0 0 0 0",
            contype="1",
            conaffinity="3",
            friction=_vec(scene.cookie_friction),
            group="3",
            # Override the scene's softer default floor contact; otherwise the
            # mixed time constant still lets the narrow beveled base sink.
            priority="1",
            # The beveled underside has a narrower floor contact patch than the
            # old box. Its former 0.04 s soft contact let a light Cookie sink
            # deep into the bin floor, so use a firmer contact response.
            solref=_vec(scene.cookie_solref),
            solimp=_vec(scene.cookie_solimp),
            # condim=6 enables full tangential + torsional friction so the gripper
            # keeps lateral purchase while extracting a cookie from the stack.
            condim="6",
        )
        if scene.cookie_collision_mode == "compound":
            hx, hy, hz = scene.cookie_half_size_m
            top_bevel = scene.cookie_edge_bevel_m
            core_half_height = hz - (top_bevel + bottom_bevel) / 2
            core_center_z = (bottom_bevel - top_bevel) / 2
            core_volume = 8 * hx * hy * core_half_height
            cap_volumes = {
                "top": 2 * hx * top_bevel * (2 * hy - top_bevel),
                "bottom": 2 * hx * bottom_bevel * (2 * hy - bottom_bevel),
            }
            volume = core_volume + sum(cap_volumes.values())
            collision.attrib.pop("mesh")
            collision.set("type", "box")
            collision.set("pos", _vec((0, 0, core_center_z)))
            collision.set("size", _vec((hx, hy, core_half_height)))
            collision.set("mass", str(scene.cookie_mass_kg * core_volume / volume))
            for label in ("top", "bottom"):
                attributes = dict(collision.attrib)
                attributes.pop("size")
                attributes.pop("pos")
                attributes.update(
                    name=f"cookie_{index}_{label}_geom",
                    type="mesh",
                    mesh=f"cookie_{label}_mesh",
                    mass=str(scene.cookie_mass_kg * cap_volumes[label] / volume),
                )
                ET.SubElement(cookie, "geom", **attributes)
        column, row = columns.index(x), rows.index(y)
        cookie_rgba = "1.0 0.78 0.20 1" if (column + row) % 2 == 0 else "0.88 0.50 0.04 1"
        ET.SubElement(
            cookie,
            "geom",
            name=f"cookie_{index}_visual",
            type="mesh",
            mesh="cookie_visual_mesh",
            mass="0.0001",
            rgba=cookie_rgba,
            contype="0",
            conaffinity="0",
            group="2",
        )
