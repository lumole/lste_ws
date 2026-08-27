#!/usr/bin/env python3
"""Generate the deterministic office-building benchmark world.

The generated world intentionally does not contain Pro3.  The lifecycle
launcher spawns the robot after Gazebo is ready, so all benchmark runs use the
same robot model, sensors, and initial-pose contract.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Sequence


WORLD_NAME = "office_building_v1"
WALL_HEIGHT = 2.8
WALL_THICKNESS = 0.18


def element(parent: ET.Element, tag: str, text: str | None = None, **attrs: str) -> ET.Element:
    node = ET.SubElement(parent, tag, attrs)
    if text is not None:
        node.text = str(text)
    return node


def pose_text(x: float, y: float, z: float = 0.0, yaw: float = 0.0) -> str:
    return f"{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.6f}"


def material(parent: ET.Element, rgba: Sequence[float]) -> None:
    mat = element(parent, "material")
    element(mat, "ambient", "%.3f %.3f %.3f %.3f" % tuple(rgba))
    element(mat, "diffuse", "%.3f %.3f %.3f %.3f" % tuple(rgba))
    element(mat, "specular", "0.15 0.15 0.15 1")


def box_geometry(parent: ET.Element, size: Sequence[float]) -> None:
    geometry = element(parent, "geometry")
    box = element(geometry, "box")
    element(box, "size", "%.3f %.3f %.3f" % tuple(size))


def add_box_link(
    structure: ET.Element,
    name: str,
    x: float,
    y: float,
    size: Sequence[float],
    *,
    z: float | None = None,
    rgba: Sequence[float] = (0.72, 0.74, 0.78, 1.0),
    collision: bool = True,
) -> None:
    """Add one axis-aligned box link to a static structure model."""
    height = float(size[2])
    link = element(structure, "link", name=name)
    element(link, "pose", pose_text(x, y, height / 2.0 if z is None else z))
    if collision:
        collision_node = element(link, "collision", name=f"{name}_collision")
        box_geometry(collision_node, size)
    visual = element(link, "visual", name=f"{name}_visual")
    box_geometry(visual, size)
    material(visual, rgba)


def add_floor_zone(
    structure: ET.Element,
    name: str,
    x: float,
    y: float,
    sx: float,
    sy: float,
    rgba: Sequence[float],
) -> None:
    add_box_link(
        structure,
        name,
        x,
        y,
        (sx, sy, 0.012),
        z=0.006,
        rgba=rgba,
        collision=False,
    )


def add_include(
    world: ET.Element,
    uri: str,
    name: str,
    x: float,
    y: float,
    z: float = 0.0,
    yaw: float = 0.0,
    scale: Sequence[float] | None = None,
) -> None:
    include = element(world, "include")
    element(include, "uri", f"model://{uri}")
    element(include, "name", name)
    element(include, "pose", pose_text(x, y, z, yaw))
    if scale is not None:
        element(include, "scale", "%.3f %.3f %.3f" % tuple(scale))
    element(include, "static", "true")


def add_world_header(world: ET.Element) -> None:
    element(world, "gravity", "0 0 -9.8")
    element(world, "magnetic_field", "6e-06 2.3e-05 -4.2e-05")
    physics = element(world, "physics", type="ode")
    element(physics, "max_step_size", "0.001")
    element(physics, "real_time_factor", "1")
    element(physics, "real_time_update_rate", "1000")
    scene = element(world, "scene")
    element(scene, "ambient", "0.45 0.45 0.45 1")
    element(scene, "background", "0.72 0.75 0.80 1")
    element(scene, "shadows", "1")

    ground = element(world, "model", name="ground_plane")
    element(ground, "static", "true")
    ground_link = element(ground, "link", name="link")
    collision = element(ground_link, "collision", name="collision")
    geometry = element(collision, "geometry")
    plane = element(geometry, "plane")
    element(plane, "normal", "0 0 1")
    element(plane, "size", "100 100")
    visual = element(ground_link, "visual", name="visual")
    visual_geometry = element(visual, "geometry")
    visual_plane = element(visual_geometry, "plane")
    element(visual_plane, "normal", "0 0 1")
    element(visual_plane, "size", "100 100")
    material(visual, (0.50, 0.52, 0.55, 1.0))

    light = element(world, "light", name="sun", type="directional")
    element(light, "pose", pose_text(12.5, 11.0, 25.0, 0.0))
    element(light, "cast_shadows", "1")
    element(light, "diffuse", "0.85 0.85 0.85 1")
    element(light, "specular", "0.20 0.20 0.20 1")
    element(light, "direction", "-0.35 0.20 -0.90")
    attenuation = element(light, "attenuation")
    element(attenuation, "range", "80")
    element(attenuation, "constant", "0.8")
    element(attenuation, "linear", "0.01")
    element(attenuation, "quadratic", "0.001")


def build_structure(world: ET.Element, level: str) -> None:
    structure = element(world, "model", name="office_building_structure")
    element(structure, "static", "true")

    # Floor zones make the semantic spaces visible in Gazebo without changing
    # the lidar map. They are intentionally collision-free.
    zones = [
        ("floor_lobby", 2.5, 4.5, 4.8, 8.8, (0.62, 0.69, 0.78, 1)),
        ("floor_open_office", 9.0, 4.5, 7.8, 8.8, (0.72, 0.76, 0.66, 1)),
        ("floor_printer", 15.0, 4.5, 3.7, 8.8, (0.78, 0.70, 0.56, 1)),
        ("floor_storage", 19.5, 4.5, 4.7, 8.8, (0.63, 0.63, 0.67, 1)),
        ("floor_lounge", 23.5, 4.5, 2.8, 8.8, (0.60, 0.73, 0.73, 1)),
        ("floor_conference", 4.5, 17.4, 8.7, 9.0, (0.74, 0.67, 0.59, 1)),
        ("floor_manager", 10.0, 17.4, 3.0, 9.0, (0.67, 0.73, 0.78, 1)),
        ("floor_kitchen", 16.0, 17.4, 5.8, 9.0, (0.78, 0.72, 0.58, 1)),
        ("floor_target_room", 22.0, 17.4, 5.8, 9.0, (0.72, 0.66, 0.76, 1)),
    ]
    for zone in zones:
        add_floor_zone(structure, *zone)

    wall = WALL_THICKNESS
    h = WALL_HEIGHT
    gray = (0.68, 0.70, 0.74, 1)
    dark = (0.43, 0.46, 0.50, 1)

    # Outer shell. The south wall has a 3.2 m entrance into the lobby.
    add_box_link(structure, "outer_west", 0.0, 11.0, (wall, 22.0, h), rgba=gray)
    add_box_link(structure, "outer_east", 25.0, 11.0, (wall, 22.0, h), rgba=gray)
    add_box_link(structure, "outer_north", 12.5, 22.0, (25.0, wall, h), rgba=gray)
    add_box_link(structure, "outer_south_west", 0.75, 0.0, (1.5, wall, h), rgba=gray)
    add_box_link(structure, "outer_south_east", 14.5, 0.0, (20.0, wall, h), rgba=gray)

    # The robot starts inside the building. Keep the entrance visible as an
    # architectural feature, but close it for the autonomous benchmark so
    # frontier exploration cannot escape onto the unbounded ground plane and
    # turn the experiment into an outdoor search.
    add_box_link(
        structure,
        "south_entry_closed_door",
        3.4,
        0.0,
        (3.2, wall, 2.25),
        z=1.125,
        rgba=(0.30, 0.38, 0.45, 1),
    )

    # Main corridor boundaries. Gaps are doorways, not random holes.
    for index, (x0, x1) in enumerate(
        [(5.0, 6.1), (7.2, 13.0), (14.0, 16.9), (17.9, 21.9), (22.9, 25.0)]
    ):
        add_box_link(
            structure,
            f"lower_partition_{index}",
            (x0 + x1) / 2.0,
            9.3,
            (x1 - x0, wall, h),
            rgba=dark,
        )
    for index, (x0, x1) in enumerate(
        [(0.0, 2.2), (3.8, 6.9), (8.5, 12.8), (14.4, 15.9), (17.5, 18.9), (20.5, 25.0)]
    ):
        add_box_link(
            structure,
            f"upper_partition_{index}",
            (x0 + x1) / 2.0,
            12.7,
            (x1 - x0, wall, h),
            rgba=dark,
        )

    # Room separators. Each opening is wide enough for Pro3 but forces a
    # deliberate heading change, making it useful for TEB continuity tests.
    add_box_link(structure, "lobby_office_separator", 5.0, 4.3, (wall, 7.4, h), rgba=dark)
    add_box_link(structure, "office_printer_separator_a", 13.0, 1.5, (wall, 3.0, h), rgba=dark)
    add_box_link(structure, "office_printer_separator_b", 13.0, 7.2, (wall, 2.0, h), rgba=dark)
    add_box_link(structure, "printer_storage_separator_a", 17.0, 1.4, (wall, 2.8, h), rgba=dark)
    add_box_link(structure, "printer_storage_separator_b", 17.0, 7.5, (wall, 2.4, h), rgba=dark)
    # Storage has a single corridor door and is intentionally a dead end.
    add_box_link(structure, "storage_lounge_separator", 22.0, 4.65, (wall, 9.3, h), rgba=dark)

    add_box_link(structure, "conference_manager_separator_a", 7.0, 13.9, (wall, 2.4, h), rgba=dark)
    add_box_link(structure, "conference_manager_separator_b", 7.0, 20.6, (wall, 2.0, h), rgba=dark)
    add_box_link(structure, "manager_kitchen_separator_a", 13.0, 13.9, (wall, 4.0, h), rgba=dark)
    add_box_link(structure, "manager_kitchen_separator_b", 13.0, 20.9, (wall, 1.4, h), rgba=dark)
    add_box_link(structure, "kitchen_target_separator_a", 19.0, 13.9, (wall, 2.4, h), rgba=dark)
    add_box_link(structure, "kitchen_target_separator_b", 19.0, 20.7, (wall, 2.2, h), rgba=dark)

    # A short inner circulation loop connects the open office to the lobby and
    # the corridor through two separate openings. It is a real alternate route,
    # not merely a visual marking.
    add_box_link(structure, "office_loop_block_north", 8.8, 7.9, (6.0, wall, h), rgba=dark)
    add_box_link(structure, "office_loop_block_south", 8.8, 1.1, (6.0, wall, h), rgba=dark)
    add_box_link(structure, "office_loop_block_west", 6.0, 4.5, (wall, 6.8, h), rgba=dark)

    # Door lintels and frames are visual only; the actual doorway remains open.
    door_color = (0.30, 0.38, 0.45, 1)
    for index, x in enumerate((5.55, 14.45, 18.45, 23.45)):
        add_box_link(structure, f"lower_door_lintel_{index}", x, 9.3, (1.05, wall, 0.10), z=2.45, rgba=door_color, collision=False)
    for index, x in enumerate((3.0, 8.0, 15.15, 19.7)):
        add_box_link(structure, f"upper_door_lintel_{index}", x, 12.7, (1.05, wall, 0.10), z=2.45, rgba=door_color, collision=False)

    # Furniture collision blockers are enabled by difficulty. Level 1 keeps
    # the building shell and target desk; higher levels add semantic room
    # clutter, occlusion and the storage dead end.
    if level in ("level_2", "level_3", "level_4"):
        add_box_link(structure, "printer_body", 15.0, 4.8, (1.0, 0.85, 1.05), z=0.525, rgba=(0.25, 0.27, 0.30, 1))
        add_box_link(structure, "printer_paper_stack", 15.8, 6.2, (0.55, 0.45, 0.9), z=0.45, rgba=(0.88, 0.88, 0.82, 1))
        add_box_link(structure, "kitchen_counter", 16.2, 18.8, (3.6, 0.65, 0.95), z=0.475, rgba=(0.55, 0.42, 0.28, 1))
        add_box_link(structure, "target_room_occluder", 23.0, 15.0, (1.35, 0.75, 1.65), z=0.825, rgba=(0.45, 0.28, 0.18, 1))
    if level in ("level_3", "level_4"):
        add_box_link(structure, "storage_shelf_block", 19.0, 4.0, (1.5, 0.65, 2.0), z=1.0, rgba=(0.48, 0.32, 0.20, 1))


def add_furniture(world: ET.Element, level: str) -> None:
    include_full_semantics = level in ("level_2", "level_3", "level_4")
    include_storage = level in ("level_3", "level_4")

    # Lobby / reception.
    if include_full_semantics:
        add_include(world, "sofa_set_1", "lobby_sofa", 2.4, 4.0, 0.0, math.pi / 2)
        add_include(world, "cafe_table", "lobby_table", 3.2, 6.6)
        add_include(world, "bookshelf", "lobby_notice_shelf", 1.2, 8.0, 0.0, math.pi / 2)

    # Open-plan office: desks, screens, chairs, and two visual routes around
    # the furniture island.
    if include_full_semantics:
        for index, (x, y, yaw) in enumerate(((7.4, 3.0, 0.0), (10.2, 3.0, 0.0), (7.4, 6.3, math.pi), (10.2, 6.3, math.pi))):
            add_include(world, "desk_brown", f"open_desk_{index}", x, y, 0.0, yaw)
            add_include(world, "monitor_1" if index % 2 == 0 else "monitor_2", f"open_monitor_{index}", x, y, 0.72, yaw)
            add_include(world, "office_chair", f"open_chair_{index}", x + (0.0 if yaw == 0.0 else 0.55), y - (0.85 if yaw == 0.0 else -0.85), 0.0, yaw)
        add_include(world, "file_cabinet_large", "open_file_cabinet", 11.8, 7.3, 0.0, math.pi / 2)

    # Conference room.
    if include_full_semantics:
        add_include(world, "table_conference_2", "conference_table", 4.5, 17.2, 0.0, 0.0)
        for index, (x, y, yaw) in enumerate(((2.8, 15.2, 0.0), (6.2, 15.2, math.pi), (2.8, 19.1, 0.0), (6.2, 19.1, math.pi))):
            add_include(world, "office_chair", f"conference_chair_{index}", x, y, 0.0, yaw)
        add_include(world, "monitor_3", "conference_wall_screen", 3.0, 21.2, 0.75, 0.0)

    # Manager office.
    if include_full_semantics:
        add_include(world, "desk_brown", "manager_desk", 10.1, 17.0)
        add_include(world, "monitor_2", "manager_monitor", 10.1, 17.0, 0.72)
        add_include(world, "office_chair", "manager_chair", 10.1, 15.7, 0.0, math.pi)
        add_include(world, "bookshelf", "manager_bookshelf", 11.8, 20.8, 0.0, math.pi / 2)

    # Kitchen / break room. Counter geometry is guaranteed by the structure;
    # these models provide recognizable visual semantics.
    if include_full_semantics:
        add_include(world, "cafe_table", "kitchen_table", 16.0, 16.0)
        add_include(world, "kitchen_chair", "kitchen_chair_0", 15.1, 16.0, 0.0, math.pi / 2)
        add_include(world, "kitchen_chair", "kitchen_chair_1", 16.9, 16.0, 0.0, -math.pi / 2)
        add_include(world, "cup_blue", "kitchen_blue_cup", 15.8, 18.55, 1.02)
        add_include(world, "cup_green", "kitchen_green_cup", 17.0, 18.55, 1.02)

    # Printer area and storage dead end.
    if include_full_semantics:
        add_include(world, "file_cabinet_large", "printer_file_cabinet", 14.1, 7.5, 0.0, math.pi / 2)
    if include_storage:
        add_include(world, "bookshelf", "storage_shelf_0", 18.5, 2.2, 0.0, 0.0)
        add_include(world, "bookshelf", "storage_shelf_1", 20.2, 2.2, 0.0, 0.0)
        add_include(world, "cardboard_box", "storage_box", 19.4, 6.8, 0.0, 0.2)

    # Lounge at the east end gives the main corridor a visually distinct
    # destination and provides a second route around the central branch.
    if include_full_semantics:
        add_include(world, "sofa_set_2", "lounge_sofa", 23.3, 5.3, 0.0, math.pi / 2)
        add_include(world, "coffee_table_1", "lounge_table", 23.5, 7.2)

    # Target room. Place the workstation just beyond the west-side doorway so
    # an indoor exploration route gets a legitimate camera exposure while it
    # enters the room. Keep the central/east furniture and the room walls as
    # occlusion and rerouting challenges; do not require one lucky heading at
    # the far end of the room to discover the task object.
    add_include(world, "desk_brown", "target_desk", 21.6, 17.6)
    add_include(world, "monitor_1", "target_monitor_left", 21.0, 17.6, 0.72)
    add_include(world, "monitor_2", "target_monitor_right", 22.2, 17.6, 0.72)
    if include_full_semantics:
        add_include(world, "office_chair", "target_chair", 21.6, 15.8, 0.0, math.pi)
    # The source cup is about 8 cm wide and becomes only a few pixels at the
    # far end of this 25 m building.  Scale only the benchmark target include
    # so Level 1 can test exploration and target handoff without changing the
    # shared Gazebo model or the production environments.  The evaluator's
    # matching cuboid is updated in office_building_config.yaml accordingly.
    add_include(
        world,
        "cup_yellow",
        "cup_yellow",
        21.6,
        17.6,
        1.02,
        scale=(1.8, 1.8, 1.8),
    )
    if include_full_semantics:
        add_include(world, "cup_blue", "cup_blue_target_distractor", 22.9, 19.35, 1.02)
        add_include(world, "cup_blue", "cup_blue_target_distractor_2", 24.2, 19.35, 1.02)
        add_include(world, "bookshelf", "target_room_shelf", 24.2, 20.5, 0.0, math.pi / 2)


def add_dynamic_obstacle(world: ET.Element) -> None:
    """Add a static-in-physics box that the Level 4 node moves explicitly."""
    obstacle = element(world, "model", name="benchmark_dynamic_obstacle")
    element(obstacle, "static", "true")
    element(obstacle, "pose", pose_text(10.0, 11.0, 0.6))
    link = element(obstacle, "link", name="link")
    element(link, "pose", pose_text(0.0, 0.0, 0.0))
    collision = element(link, "collision", name="collision")
    box_geometry(collision, (0.65, 0.65, 1.2))
    visual = element(link, "visual", name="visual")
    box_geometry(visual, (0.65, 0.65, 1.2))
    material(visual, (0.88, 0.37, 0.12, 1.0))


def add_gui(world: ET.Element) -> None:
    gui = element(world, "gui", fullscreen="0")
    camera = element(gui, "camera", name="user_camera")
    # Near-top-down view; keep a small pitch margin from the orbit singularity.
    element(camera, "pose", "12.500 11.000 34.000 0 1.490 0")
    element(camera, "view_controller", "orbit")
    element(camera, "projection_type", "perspective")


def build_world(level: str) -> ET.ElementTree:
    sdf = ET.Element("sdf", {"version": "1.7"})
    world = element(sdf, "world", name=WORLD_NAME)
    add_world_header(world)
    build_structure(world, level)
    add_furniture(world, level)
    if level == "level_4":
        add_dynamic_obstacle(world)
    add_gui(world)
    return ET.ElementTree(sdf)


def write_world(path: Path, level: str) -> None:
    tree = build_world(level)
    # Python 3.8 is still used by the ROS Noetic environment, where
    # ElementTree.indent is unavailable.
    def indent(node: ET.Element, level: int = 0) -> None:
        padding = "\n" + "  " * level
        if len(node):
            if not node.text or not node.text.strip():
                node.text = padding + "  "
            for child in node:
                indent(child, level + 1)
            if not node[-1].tail or not node[-1].tail.strip():
                node[-1].tail = padding
        if level and (not node.tail or not node.tail.strip()):
            node.tail = padding

    indent(tree.getroot())
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("worlds/benchmark/office_building_v1_level_2_no_pro3.world"),
    )
    parser.add_argument(
        "--level",
        choices=("level_1", "level_2", "level_3", "level_4"),
        default="level_2",
        help="Difficulty profile used to include furniture and dynamic obstacles",
    )
    args = parser.parse_args()
    write_world(args.output, args.level)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
