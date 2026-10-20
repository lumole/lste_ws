#!/usr/bin/env python3
"""
Pop up the T-Junction Office Layout directly using Matplotlib GUI window on DISPLAY.
"""

import os
import matplotlib
# Ensure interactive GUI backend is used
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Setup fonts
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 10))
fig.canvas.manager.set_window_title('方案 A：T字形分叉双室办公场景平面图 (T-Junction Office Benchmark)')
ax.set_facecolor('#F8F9FA')

# Floor Zones
ax.add_patch(patches.Rectangle((4.75, 0.0), 2.5, 3.5, facecolor='#ECEFF1', alpha=0.8, zorder=1))
ax.add_patch(patches.Rectangle((0.0, 3.5), 12.0, 2.0, facecolor='#ECEFF1', alpha=0.8, zorder=1))
ax.add_patch(patches.Rectangle((0.0, 5.5), 6.0, 5.0, facecolor='#E3F2FD', alpha=0.75, zorder=1))
ax.add_patch(patches.Rectangle((6.0, 5.5), 6.0, 5.0, facecolor='#FCE4EC', alpha=0.75, zorder=1))

# Room Labels
ax.text(6.0, 2.2, "入口走廊 (Foyer)", ha='center', va='center', fontsize=11, color='#455A64', fontweight='bold')
ax.text(6.0, 4.5, "横向分叉主道 (T-Junction)", ha='center', va='center', fontsize=11, color='#455A64', fontweight='bold')
ax.text(3.0, 9.8, "研讨室 A (Room A)\n[空房/会议室]", ha='center', va='center', fontsize=12, color='#1565C0', fontweight='bold')
ax.text(9.0, 9.8, "目标办公室 B (Room B) ★\n[含视线遮挡与目标杯]", ha='center', va='center', fontsize=12, color='#C2185B', fontweight='bold')

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

# ★ Target Yellow Cup on desk against left wall
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
path_a_x = [6.0, 6.0, 3.0, 3.0, 2.5]
path_a_y = [1.0, 4.5, 4.5, 6.5, 7.5]
ax.plot(path_a_x, path_a_y, linestyle=':', color='#7B1FA2', linewidth=2.5, alpha=0.8, zorder=9, label='分支A探索 (探查Room A会议区与消防栓)')

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
ax.set_title('方案 A：T字形分叉双室办公场景 (12m × 10.5m)', fontsize=13.5, pad=12, fontweight='bold', color='#1A237E')

legend_elements = [
    patches.Patch(facecolor='#263238', label='承重外墙与分室隔断 (Wall 0.18m)'),
    Line2D([0], [0], color='#00E676', lw=5, label='标准门洞 / Directed Portal (1.2m)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2979FF', markeredgecolor='#0D47A1', markersize=10, label='机器人出生点 (Spawn: 6.0, 1.0)'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD600', markeredgecolor='#E65100', markersize=14, label='全局目标: 黄色水杯 (Yellow Cup)'),
    patches.Patch(facecolor='#263238', label='显示器 1 & 2 (Monitors)'),
    patches.Patch(facecolor='#E65100', hatch='//', label='视线遮挡屏风 (Occluder)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#D32F2F', markeredgecolor='#B71C1C', markersize=8, label='消防栓 (Fire Hydrants)'),
    Line2D([0], [0], color='#7B1FA2', lw=2.5, linestyle=':', label='左侧Room A探索路径 (无目标)'),
    Line2D([0], [0], color='#2E7D32', lw=2.5, linestyle='--', label='回退并前往Room B锁定目标'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8.5, frameon=True, facecolor='#FFFFFF', edgecolor='#CFD8DC')

plt.tight_layout()
print("Popping up Matplotlib window...")
plt.show()
