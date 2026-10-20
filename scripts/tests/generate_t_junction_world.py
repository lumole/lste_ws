#!/usr/bin/env python3
"""
Generate the deterministic T-Junction 2-Room Office Benchmark World (Scheme A).
Outputs:
  - worlds/t_junction_office_no_pro3.world
  - worlds/topo3.0_catch_mode/session_test/t_junction_office_no_pro3.world
  - worlds/t_junction_office_manifest.yaml
  - High-res layout plots
"""

from __future__ import annotations

import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

WORLD_NAME = "t_junction_office"
WALL_HEIGHT = 2.5
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
    rgba: Sequence[float] = (0.68, 0.70, 0.74, 1.0),
    collision: bool = True,
) -> None:
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
    width: float,
    depth: float,
    rgba: Sequence[float],
) -> None:
    add_box_link(
        structure,
        name,
        x,
        y,
        (width, depth, 0.005),
        z=0.0025,
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

def build_world() -> ET.ElementTree:
    world_root = ET.Element("sdf", version="1.7")
    world = element(world_root, "world", name=WORLD_NAME)

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

    # Ground plane
    ground = element(world, "model", name="ground_plane")
    element(ground, "static", "true")
    ground_link = element(ground, "link", name="link")
    collision = element(ground_link, "collision", name="collision")
    box_geometry(collision, (60, 60, 0.01))
    visual = element(ground_link, "visual", name="visual")
    box_geometry(visual, (60, 60, 0.01))
    material(visual, (0.50, 0.52, 0.55, 1.0))

    # Directional Light
    light = element(world, "light", name="sun", type="directional")
    element(light, "pose", pose_text(6.0, 5.0, 20.0, 0.0))
    element(light, "cast_shadows", "1")
    element(light, "diffuse", "0.85 0.85 0.85 1")
    element(light, "specular", "0.20 0.20 0.20 1")
    element(light, "direction", "-0.35 0.20 -0.90")
    attenuation = element(light, "attenuation")
    element(attenuation, "range", "50")
    element(attenuation, "constant", "0.8")
    element(attenuation, "linear", "0.01")
    element(attenuation, "quadratic", "0.001")

    # Structure Model
    structure = element(world, "model", name="t_junction_structure")
    element(structure, "static", "true")

    # Floor Zones
    add_floor_zone(structure, "floor_foyer", 6.0, 1.75, 2.5, 3.5, (0.80, 0.82, 0.85, 1))
    add_floor_zone(structure, "floor_cross_hall", 6.0, 4.5, 12.0, 2.0, (0.75, 0.77, 0.80, 1))
    add_floor_zone(structure, "floor_room_a", 3.0, 8.0, 6.0, 5.0, (0.70, 0.76, 0.82, 1))
    add_floor_zone(structure, "floor_room_b", 9.0, 8.0, 6.0, 5.0, (0.82, 0.72, 0.74, 1))

    # Walls Definition
    wall = WALL_THICKNESS
    h = WALL_HEIGHT
    gray = (0.68, 0.70, 0.74, 1)
    dark = (0.42, 0.45, 0.50, 1)
    door_lintel_color = (0.30, 0.38, 0.45, 1)

    # 1. Outer North wall: y = 10.5, x: 0 to 12
    add_box_link(structure, "wall_outer_north", 6.0, 10.5, (12.0, wall, h), rgba=gray)

    # 2. Outer West wall: x = 0, y: 3.5 to 10.5
    add_box_link(structure, "wall_outer_west", 0.0, 7.0, (wall, 7.0, h), rgba=gray)

    # 3. Outer East wall: x = 12, y: 3.5 to 10.5
    add_box_link(structure, "wall_outer_east", 12.0, 7.0, (wall, 7.0, h), rgba=gray)

    # 4. Transverse corridor south walls: y = 3.5
    # Left segment: x from 0 to 4.75
    add_box_link(structure, "wall_hall_south_left", 2.375, 3.5, (4.75, wall, h), rgba=gray)
    # Right segment: x from 7.25 to 12.0
    add_box_link(structure, "wall_hall_south_right", 9.625, 3.5, (4.75, wall, h), rgba=gray)

    # 5. Entrance corridor (Stem of T):
    # West wall: x = 4.75, y from 0 to 3.5
    add_box_link(structure, "wall_stem_west", 4.75, 1.75, (wall, 3.5, h), rgba=gray)
    # East wall: x = 7.25, y from 0 to 3.5
    add_box_link(structure, "wall_stem_east", 7.25, 1.75, (wall, 3.5, h), rgba=gray)
    # South wall: y = 0, x from 4.75 to 7.25
    add_box_link(structure, "wall_stem_south", 6.0, 0.0, (2.5 + wall, wall, h), rgba=gray)

    # 6. Central partition wall between Room A and Room B: x = 6.0, y: 5.5 to 10.5
    add_box_link(structure, "wall_partition_ab", 6.0, 8.0, (wall, 5.0, h), rgba=dark)

    # 7. Room A South wall (y = 5.5, door A at x=3.0, width 1.2m -> opening [2.4, 3.6])
    add_box_link(structure, "wall_room_a_south_w", 1.2, 5.5, (2.4, wall, h), rgba=dark)
    add_box_link(structure, "wall_room_a_south_e", 4.8, 5.5, (2.4, wall, h), rgba=dark)
    # Door A Lintel
    add_box_link(structure, "lintel_door_a", 3.0, 5.5, (1.2, wall, 0.25), z=2.375, rgba=door_lintel_color, collision=False)

    # 8. Room B South wall (y = 5.5, door B at x=9.0, width 1.2m -> opening [8.4, 9.6])
    add_box_link(structure, "wall_room_b_south_w", 7.2, 5.5, (2.4, wall, h), rgba=dark)
    add_box_link(structure, "wall_room_b_south_e", 10.8, 5.5, (2.4, wall, h), rgba=dark)
    # Door B Lintel
    add_box_link(structure, "lintel_door_b", 9.0, 5.5, (1.2, wall, 0.25), z=2.375, rgba=door_lintel_color, collision=False)

    # 9. Room B Occluder (屏风隔断: x=7.8, y=7.0, size: 1.6 x 0.25 x 1.6)
    # Blocks direct line of sight from Door B to Target Desk on the left wall
    add_box_link(structure, "target_room_occluder", 7.8, 7.0, (1.6, 0.25, 1.6), z=0.8, rgba=(0.45, 0.28, 0.18, 1))

    # Furniture & Semantics Models
    # --- Room A (Discussion / Left Room) ---
    add_include(world, "table_conference_2", "room_a_conference_table", 3.0, 8.2, 0.0, 0.0)
    add_include(world, "bookshelf", "room_a_bookshelf", 0.5, 7.8, 0.0, 0.0)
    # Fire hydrants in Room A tucked into true wall corners (墙脚)
    add_include(world, "fire_hydrant", "fire_hydrant_1", 0.35, 5.85, 0.0, 0.0)
    add_include(world, "fire_hydrant", "fire_hydrant_2", 0.35, 10.15, 0.0, 0.0)
    add_include(world, "fire_hydrant", "fire_hydrant_3", 5.65, 10.15, 0.0, 0.0)

    # --- Room B (Target Office / Right Room) ---
    # Target Desk against the left partition wall (x=6.0)
    add_include(world, "desk_brown", "target_desk", 6.6, 8.5, 0.0, math.pi / 2)
    # Two monitors flanking the yellow cup on the target desk
    add_include(world, "monitor_1", "target_monitor_left", 6.6, 8.0, 0.72, math.pi / 2)
    add_include(world, "monitor_2", "target_monitor_right", 6.6, 9.0, 0.72, math.pi / 2)
    add_include(world, "file_cabinet_large", "room_b_cabinet", 11.2, 8.5, 0.0, math.pi / 2)
    add_include(world, "bookshelf", "room_b_bookshelf", 11.2, 6.5, 0.0, math.pi / 2)

    # ★ PRIMARY TARGET: Yellow Cup on target desk flanked by monitors
    add_include(
        world,
        "cup_yellow",
        "cup_yellow",
        6.6,
        8.5,
        0.75,
        scale=(1.5, 1.5, 1.5),
    )

    # Near-top-down GUI camera
    gui = element(world, "gui", fullscreen="0")
    camera = element(gui, "camera", name="user_camera")
    element(camera, "pose", "6.000 5.250 18.000 0 1.500 0")

    return world_root

def generate_manifest() -> dict:
    return {
        "schema_version": 1,
        "scenario_id": "t_junction_office",
        "display_name": "T-Junction Two-Room Office Benchmark",
        "world": "worlds/t_junction_office_no_pro3.world",
        "building": {
            "bounds_m": [0.0, 0.0, 12.0, 10.5],
            "wall_height_m": 2.5,
            "wall_thickness_m": 0.18,
            "door_width_m": 1.20,
            "corridor_width_m": 2.00,
        },
        "robot_profiles": {
            "primary": {
                "pose": [6.0, 1.0, 0.05, 1.570796],
                "semantic": "Entrance stem facing T-junction north",
            },
            "junction_center": {
                "pose": [6.0, 4.5, 0.05, 1.570796],
                "semantic": "Center of T-junction crossbar",
            }
        },
        "rooms": [
            {
                "id": "corridor_foyer",
                "semantic": "T-junction entrance and transverse corridor",
                "bounds_m": [0.0, 0.0, 12.0, 5.5],
                "doors": ["door_a", "door_b"],
            },
            {
                "id": "room_a",
                "semantic": "Left meeting and discussion room with fire hydrants",
                "bounds_m": [0.0, 5.5, 6.0, 10.5],
                "doors": ["door_a"],
                "models": ["room_a_conference_table", "room_a_bookshelf", "fire_hydrant_1", "fire_hydrant_2", "fire_hydrant_3"],
            },
            {
                "id": "room_b",
                "semantic": "Right target office with workstation against left wall",
                "bounds_m": [6.0, 5.5, 12.0, 10.5],
                "doors": ["door_b"],
                "models": ["target_desk", "target_monitor_left", "target_monitor_right", "cup_yellow", "target_room_occluder", "room_b_cabinet", "room_b_bookshelf"],
            }
        ],
        "doors": [
            {
                "id": "door_a",
                "center_m": [3.0, 5.5],
                "width_m": 1.20,
                "connects": ["corridor_foyer", "room_a"],
            },
            {
                "id": "door_b",
                "center_m": [9.0, 5.5],
                "width_m": 1.20,
                "connects": ["corridor_foyer", "room_b"],
            }
        ],
        "targets": {
            "primary": {
                "model": "cup_yellow",
                "label": "yellow cup",
                "pose_xy": [6.6, 8.5],
                "room": "room_b",
                "occluder": "target_room_occluder",
                "distractors": [],
            }
        }
    }

def plot_layout(output_png: str) -> None:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    ax.set_facecolor('#F8F9FA')

    # Floor Zones
    ax.add_patch(patches.Rectangle((4.75, 0.0), 2.5, 3.5, facecolor='#ECEFF1', alpha=0.8, zorder=1))
    ax.add_patch(patches.Rectangle((0.0, 3.5), 12.0, 2.0, facecolor='#ECEFF1', alpha=0.8, zorder=1))
    ax.add_patch(patches.Rectangle((0.0, 5.5), 6.0, 5.0, facecolor='#E3F2FD', alpha=0.75, zorder=1))
    ax.add_patch(patches.Rectangle((6.0, 5.5), 6.0, 5.0, facecolor='#FCE4EC', alpha=0.75, zorder=1))

    # Room Labels
    ax.text(6.0, 2.2, "入口走廊 (Foyer)", ha='center', va='center', fontsize=11, color='#455A64', fontweight='bold')
    ax.text(6.0, 4.5, "横向分叉主道 (T-Junction)", ha='center', va='center', fontsize=11, color='#455A64', fontweight='bold')
    ax.text(3.0, 9.8, "研讨室 A (Room A)\n[会议桌 & 消防栓]", ha='center', va='center', fontsize=12, color='#1565C0', fontweight='bold')
    ax.text(9.0, 9.8, "目标办公室 B (Room B) ★\n[目标桌靠左墙摆放]", ha='center', va='center', fontsize=12, color='#C2185B', fontweight='bold')

    # Outer & Partition Walls
    wall_kw = dict(facecolor='#263238', edgecolor='none', zorder=5)
    wall_th = 0.18

    # North
    ax.add_patch(patches.Rectangle((0.0, 10.5 - wall_th/2), 12.0, wall_th, **wall_kw))
    # West
    ax.add_patch(patches.Rectangle((-wall_th/2, 3.5), wall_th, 7.0, **wall_kw))
    # East
    ax.add_patch(patches.Rectangle((12.0 - wall_th/2, 3.5), wall_th, 7.0, **wall_kw))
    # South transverse
    ax.add_patch(patches.Rectangle((0.0, 3.5 - wall_th/2), 4.75, wall_th, **wall_kw))
    ax.add_patch(patches.Rectangle((7.25, 3.5 - wall_th/2), 4.75, wall_th, **wall_kw))
    # Stem
    ax.add_patch(patches.Rectangle((4.75 - wall_th/2, 0.0), wall_th, 3.5, **wall_kw))
    ax.add_patch(patches.Rectangle((7.25 - wall_th/2, 0.0), wall_th, 3.5, **wall_kw))
    ax.add_patch(patches.Rectangle((4.75, -wall_th/2), 2.5, wall_th, **wall_kw))

    # Partition wall AB
    ax.add_patch(patches.Rectangle((6.0 - wall_th/2, 5.5), wall_th, 5.0, **wall_kw))

    # Room A South Wall (door at 3.0, w=1.2)
    ax.add_patch(patches.Rectangle((0.0, 5.5 - wall_th/2), 2.4, wall_th, **wall_kw))
    ax.add_patch(patches.Rectangle((3.6, 5.5 - wall_th/2), 2.4, wall_th, **wall_kw))
    ax.plot([2.4, 3.6], [5.5, 5.5], color='#00E676', linewidth=5, zorder=7)
    ax.text(3.0, 5.15, "门 A (Door A, 1.2m)", ha='center', va='top', fontsize=9, color='#00796B', fontweight='bold', zorder=10)

    # Room B South Wall (door at 9.0, w=1.2)
    ax.add_patch(patches.Rectangle((6.0, 5.5 - wall_th/2), 2.4, wall_th, **wall_kw))
    ax.add_patch(patches.Rectangle((9.6, 5.5 - wall_th/2), 2.4, wall_th, **wall_kw))
    ax.plot([8.4, 9.6], [5.5, 5.5], color='#00E676', linewidth=5, zorder=7)
    ax.text(9.0, 5.15, "门 B (Door B, 1.2m)", ha='center', va='top', fontsize=9, color='#00796B', fontweight='bold', zorder=10)

    # Occluder in Room B (blocks sightline from door to left-wall desk)
    ax.add_patch(patches.Rectangle((7.0, 6.875), 1.6, 0.25, facecolor='#E65100', edgecolor='#BF360C', hatch='//', zorder=8))
    ax.text(7.8, 7.0, "Occluder 屏风隔断\n(视线遮挡)", ha='center', va='center', fontsize=7.5, color='white', fontweight='bold', zorder=9)

    # Furniture
    furn_kw = dict(facecolor='#8D6E63', edgecolor='#5D4037', linewidth=1, zorder=6)

    # Room A furniture
    ax.add_patch(patches.Rectangle((1.8, 7.6), 2.4, 1.2, **furn_kw))
    ax.text(3.0, 8.2, "研讨会议桌", ha='center', va='center', fontsize=8, color='white', zorder=7)
    ax.add_patch(patches.Rectangle((0.2, 7.2), 0.6, 1.2, facecolor='#A1887F', edgecolor='#4E342E', zorder=6)) # shelf
    # Fire hydrants in Room A tucked into true wall corners (墙脚，不占通道)
    hydrants = [(0.35, 5.85), (0.35, 10.15), (5.65, 10.15)]
    for hx, hy in hydrants:
        ax.plot(hx, hy, marker='o', markersize=9, color='#D32F2F', markeredgecolor='#B71C1C', markeredgewidth=1.5, zorder=10)
        ax.text(hx, hy - 0.35 if hy > 8 else hy + 0.35, "墙脚消防栓", ha='center', va='center', fontsize=7, color='#B71C1C', fontweight='bold', zorder=10)

    # Room B furniture: Target desk placed against the left partition wall (x=6.0)
    ax.add_patch(patches.Rectangle((6.2, 7.7), 0.8, 1.6, facecolor='#4E342E', edgecolor='#27130F', zorder=6))
    ax.text(6.8, 8.5, "目标办公桌\n(靠左墙)", ha='center', va='center', rotation=90, fontsize=7.5, color='white', zorder=7)
    # Flanking monitors on target desk
    ax.add_patch(patches.Rectangle((6.45, 7.82), 0.12, 0.36, facecolor='#263238', edgecolor='#37474F', zorder=12))
    ax.text(6.35, 8.00, "显示器1", ha='right', va='center', fontsize=7, color='#263238', fontweight='bold', zorder=12)
    ax.add_patch(patches.Rectangle((6.45, 8.82), 0.12, 0.36, facecolor='#263238', edgecolor='#37474F', zorder=12))
    ax.text(6.35, 9.00, "显示器2", ha='right', va='center', fontsize=7, color='#263238', fontweight='bold', zorder=12)

    ax.add_patch(patches.Rectangle((10.9, 7.9), 0.6, 1.2, facecolor='#78909C', edgecolor='#37474F', zorder=6)) # cabinet
    ax.add_patch(patches.Rectangle((10.9, 5.9), 0.6, 1.2, facecolor='#A1887F', edgecolor='#4E342E', zorder=6)) # shelf

    # ★ Target Yellow Cup on desk flanked by monitors
    ax.plot(6.6, 8.5, marker='*', markersize=20, markerfacecolor='#FFD600', markeredgecolor='#E65100', markeredgewidth=1.8, zorder=15)
    ax.text(6.6, 7.2, "★ Yellow Cup (目标黄色水杯)\n[x=6.6, y=8.5 (双显示器夹放)]", ha='center', va='top', fontsize=9, color='#C2185B', fontweight='bold', zorder=15,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9C4', edgecolor='#FBC02D', alpha=0.9))

    # Robot Start Pose: (6.0, 1.0, yaw=pi/2)
    ax.plot(6.0, 1.0, marker='o', markersize=14, markerfacecolor='#2979FF', markeredgecolor='#0D47A1', markeredgewidth=2, zorder=15)
    ax.annotate('', xy=(6.0, 1.8), xytext=(6.0, 1.0),
                arrowprops=dict(arrowstyle="->", color='#0D47A1', lw=2.5), zorder=15)
    ax.text(6.0, 0.4, "Robot Spawn (主起点)\n[x=6.0, y=1.0, yaw=90°]", ha='center', va='top', fontsize=9, color='#0D47A1', fontweight='bold', zorder=15,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD', edgecolor='#2196F3', alpha=0.9))

    # Exploration Paths
    # Left Branch Exploration (to Room A)
    path_a_x = [6.0, 6.0, 3.0, 3.0, 2.5]
    path_a_y = [1.0, 4.5, 4.5, 6.5, 7.5]
    ax.plot(path_a_x, path_a_y, linestyle=':', color='#7B1FA2', linewidth=2.5, alpha=0.8, zorder=9, label='分支A探索 (探查Room A会议区与消防栓)')

    # Recovery & Right Branch (to Room B Target)
    path_b_x = [2.5, 3.0, 3.0, 6.0, 9.0, 9.0, 9.2, 7.8, 6.8]
    path_b_y = [7.5, 6.5, 4.5, 4.5, 4.5, 6.2, 7.5, 8.5, 8.5]
    ax.plot(path_b_x, path_b_y, linestyle='--', color='#2E7D32', linewidth=2.5, alpha=0.85, zorder=9, label='回退至走廊并进Room B绕过屏风锁定左墙目标')

    # Coordinates & Legend
    ax.set_xlim(-1.0, 13.0)
    ax.set_ylim(-1.0, 11.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', color='#B0BEC5', alpha=0.6, zorder=0)

    ax.set_xticks(range(0, 13, 2))
    ax.set_yticks(range(0, 12, 2))
    ax.set_xlabel('Gazebo X 坐标 (米 / meters)', fontsize=11, labelpad=6, fontweight='bold')
    ax.set_ylabel('Gazebo Y 坐标 (米 / meters)', fontsize=11, labelpad=6, fontweight='bold')
    ax.set_title('方案 A：T字形分叉双室办公基准环境 (12m × 10.5m, T-Junction Office Benchmark)', fontsize=13.5, pad=12, fontweight='bold', color='#1A237E')

    legend_elements = [
        patches.Patch(facecolor='#263238', label='承重外墙与分室隔断 (Wall 0.18m)'),
        Line2D([0], [0], color='#00E676', lw=5, label='标准门洞 / Directed Portal (1.2m)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2979FF', markeredgecolor='#0D47A1', markersize=10, label='机器人出生点 (Spawn: 6.0, 1.0)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD600', markeredgecolor='#E65100', markersize=14, label='全局目标: 黄色水杯 (Yellow Cup)'),
        patches.Patch(facecolor='#263238', label='显示器 1 & 2 (Monitors)'),
        patches.Patch(facecolor='#E65100', hatch='//', label='视线遮挡屏风 (Occluder)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D32F2F', markeredgecolor='#B71C1C', markersize=8, label='消防栓 (Fire Hydrants)'),
        Line2D([0], [0], color='#7B1FA2', lw=2.5, linestyle=':', label='左侧Room A探索路径 (无目标)'),
        Line2D([0], [0], color='#2E7D32', lw=2.5, linestyle='--', label='回退至走廊并进Room B锁定目标'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8.5, frameon=True, facecolor='#FFFFFF', edgecolor='#CFD8DC')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_png}")

def main():
    ws = Path(__file__).resolve().parent.parent.parent
    
    # 1. World file paths
    target_world_1 = ws / "worlds" / "t_junction_office_no_pro3.world"
    target_world_2 = ws / "worlds" / "topo3.0_catch_mode" / "session_test" / "t_junction_office_no_pro3.world"
    target_manifest = ws / "worlds" / "t_junction_office_manifest.yaml"

    tree = build_world()
    
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

    indent(tree)
    target_world_1.parent.mkdir(parents=True, exist_ok=True)
    target_world_2.parent.mkdir(parents=True, exist_ok=True)
    et = ET.ElementTree(tree)
    et.write(target_world_1, encoding="utf-8", xml_declaration=True)
    et.write(target_world_2, encoding="utf-8", xml_declaration=True)
    print(f"Generated: {target_world_1}")
    print(f"Generated: {target_world_2}")

    # Write Manifest
    manifest_data = generate_manifest()
    with open(target_manifest, "w", encoding="utf-8") as f:
        yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Generated: {target_manifest}")

    # Generate Layout Plot
    plot_path_1 = ws / "docs" / "testing" / "assets" / "t_junction_office_layout.png"
    brain_plot = Path("/home/dh/.gemini/antigravity/brain/6f323971-c1bc-4e5e-8f8c-990df59e121c/t_junction_office_layout.png")
    
    plot_layout(str(plot_path_1))
    plot_layout(str(brain_plot))

if __name__ == "__main__":
    main()
