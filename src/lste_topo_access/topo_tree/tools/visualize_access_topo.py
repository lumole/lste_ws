#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Access-Topo JSON dumps (Updated Color Scheme).
"""

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection

# --- Color Palette Configuration ---
# 1. Anchor: 黑点
COLOR_ANCHOR_POINTS = "#000000"
# 2. Branch Search: 水鸭色线 (Teal)
COLOR_BRANCH_SEARCH = "#00897B"  # Deep Teal
# 3. Branch Trackback: 琥珀色线 (Amber)
COLOR_BRANCH_TRACKBACK = "#FFB300" # Vivid Amber
# 4. Interest: 红色星
COLOR_INTEREST = "#D32F2F"       # Red
# 5. Backtrack Start: 紫色星
COLOR_BACKTRACK_START = "#7B1FA2" # Purple
# 6. Robot Pose: 黄色星
COLOR_ROBOT = "#FFEB3B"          # Bright Yellow

# 辅助颜色 (Anchor路径连线，保持柔和以免干扰)
COLOR_ANCHOR_PATH = "#B0BEC5"    # Blue Grey

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_latest_json(tree_dir: str) -> Optional[str]:
    pattern = os.path.join(tree_dir, "**", "access_topo_*.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def build_anchor_map(nodes: List[dict]) -> Dict[int, dict]:
    return {int(n["id"]): n for n in nodes if "id" in n and "x" in n and "y" in n}

def plot_path(ax, anchors: Dict[int, dict]):
    segments = []
    for node in anchors.values():
        prev = node.get("prev")
        if prev is None:
            continue
        pnode = anchors.get(int(prev))
        if pnode is None:
            continue
        segments.append([(pnode["x"], pnode["y"]), (node["x"], node["y"])])
    if segments:
        # Anchor连线不作为主要图例展示，仅作为背景
        lc = LineCollection(segments, colors=COLOR_ANCHOR_PATH, linewidths=2.0, alpha=0.6, zorder=1)
        ax.add_collection(lc)
    
    # Draw Anchors
    xs = [n["x"] for n in anchors.values()]
    ys = [n["y"] for n in anchors.values()]
    # zorder稍微高一点盖住线
    ax.scatter(xs, ys, s=20, c=COLOR_ANCHOR_POINTS, alpha=1.0, zorder=2)

def plot_branches(ax, anchors: Dict[int, dict]):
    ray_len = 1.2
    search_segments = []
    trackback_segments = []
    
    for node in anchors.values():
        x0, y0 = node["x"], node["y"]
        for br in node.get("branches", []):
            status = br.get("status", "PENDING")
            heading = float(br.get("heading_world", 0.0))
            x1 = x0 + ray_len * math.cos(heading)
            y1 = y0 + ray_len * math.sin(heading)
            
            # 收集线段
            if status == "TRACKBACK":
                trackback_segments.append([(x0, y0), (x1, y1)])
            else:
                search_segments.append([(x0, y0), (x1, y1)])
            
            # 绘制 Interest (Red Star)
            ax.scatter([x1], [y1], s=80, marker="*", color=COLOR_INTEREST, edgecolors="white", linewidths=0.5, zorder=4)

    # 批量绘制线段
    if search_segments:
        lc_search = LineCollection(search_segments, colors=COLOR_BRANCH_SEARCH, linewidths=2.0, alpha=0.9, zorder=3)
        ax.add_collection(lc_search)
    if trackback_segments:
        lc_tb = LineCollection(trackback_segments, colors=COLOR_BRANCH_TRACKBACK, linewidths=2.0, alpha=0.9, zorder=3)
        ax.add_collection(lc_tb)

def plot_pose(ax, pose: dict):
    x, y = pose.get("x"), pose.get("y")
    if x is None or y is None:
        return
    # Robot Pose: 黄色星，加个黑边增加对比度
    ax.scatter([x], [y], s=120, marker="*", color=COLOR_ROBOT, edgecolors="black", linewidths=0.8, zorder=10)

def plot_backtrack(ax, data: dict, anchors: Dict[int, dict]):
    """仅绘制回退起点 (紫色星) 和 高亮路径 (使用琥珀色表示回退流)"""
    start_pose = data.get("backtrack_start_pose") or {}
    if start_pose.get("x") is not None and start_pose.get("y") is not None:
        x0, y0 = start_pose["x"], start_pose["y"]
        # Backtrack Start: 紫色星
        ax.scatter([x0], [y0], s=120, marker="*", color=COLOR_BACKTRACK_START, edgecolors="white", linewidths=0.5, zorder=6)

    path_ids = data.get("backtrack_path") or []
    pts = []
    for aid in path_ids:
        node = anchors.get(int(aid))
        if node:
            pts.append((node["x"], node["y"]))
    
    # 如果有回退路径，用琥珀色高亮显示，表示这是 Trackback 属性的路径
    if len(pts) >= 2:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=COLOR_BRANCH_TRACKBACK, linewidth=3.0, alpha=0.5, zorder=1.5)

def create_custom_legend(ax):
    """
    强制生成符合要求的6个图例，无论当前数据中是否存在这些元素。
    """
    legend_elements = [
        # 1. Anchor: 黑点
        mlines.Line2D([], [], color='white', marker='o', markerfacecolor=COLOR_ANCHOR_POINTS, 
                      markersize=8, label='anchor'),
        
        # 2. Branch Search: 水鸭色线
        mlines.Line2D([], [], color=COLOR_BRANCH_SEARCH, linewidth=2, label='branch search'),
        
        # 3. Branch Trackback: 琥珀色线
        mlines.Line2D([], [], color=COLOR_BRANCH_TRACKBACK, linewidth=2, label='branch trackback'),
        
        # 4. Interest: 红色星
        mlines.Line2D([], [], color='white', marker='*', markerfacecolor=COLOR_INTEREST, 
                      markersize=10, label='interest'),
        
        # 5. Backtrack Start: 紫色星
        mlines.Line2D([], [], color='white', marker='*', markerfacecolor=COLOR_BACKTRACK_START, 
                      markersize=10, label='backtrack start'),
        
        # 6. Robot Pose: 黄色星
        mlines.Line2D([], [], color='white', marker='*', markerfacecolor=COLOR_ROBOT, markeredgecolor='black',
                      markersize=10, label='robot pose'),
    ]
    
    ax.legend(handles=legend_elements, loc="lower left", frameon=True, framealpha=0.9, fontsize=10)

def main():
    parser = argparse.ArgumentParser(description="Visualize Access-Topo JSON (Custom Colors).")
    parser.add_argument("--json", help="Path to access_topo_*.json")
    parser.add_argument("--tree-dir", default="/home/zrz/lste_ws/src/lste_topo_access/topo_tree/tree",
                        help="Root directory to search for latest json")
    parser.add_argument("--save", help="Save figure to file.")
    args = parser.parse_args()

    if args.json:
        json_path = args.json
    else:
        json_path = find_latest_json(args.tree_dir)
        if json_path is None:
            print(f"No access_topo_*.json found under {args.tree_dir}")
            return

    data = load_json(json_path)
    anchors = build_anchor_map(data.get("nodes", []))

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f5f7fa")
    
    # 绘图流程
    plot_path(ax, anchors)
    plot_branches(ax, anchors)
    plot_backtrack(ax, data, anchors)
    plot_pose(ax, data.get("pose", {}))

    # 生成固定图例
    create_custom_legend(ax)

    ax.set_title(f"Access-Topo: {os.path.basename(json_path)}", fontsize=12)
    ax.grid(True, linestyle="--", color="#d0d7de", alpha=0.7)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved to {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()