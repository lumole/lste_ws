#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Access-Topo JSON dumps.
- Reads a JSON file produced by gp_subgoals_sim_topo.py.
- Shows anchor chain (traveled path) and interest branches with status colors.
"""

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# color palette
COLOR_PENDING = "#d1495b"     # red
COLOR_TRACKBACK = "#edae49"   # amber
COLOR_BACKTRACK_START = "#7a5195"  # purple
COLOR_BACKTRACK_LINE = "#edae49"   # amber
COLOR_EXPLORE_LINE = "#2f9c95"     # teal
COLOR_ANCHOR_PATH = "#1b9aaa"      # deep teal
COLOR_ANCHOR_POINTS = "#0b132b"    # dark
COLOR_ROBOT = "#ffb703"            # yellow


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
        lc = LineCollection(segments, colors=COLOR_ANCHOR_PATH, linewidths=2.5, alpha=0.9)
        ax.add_collection(lc)
    # anchors as points
    xs = [n["x"] for n in anchors.values()]
    ys = [n["y"] for n in anchors.values()]
    ax.scatter(xs, ys, s=18, c=COLOR_ANCHOR_POINTS, alpha=0.9, label="anchor")


def plot_branches(ax, anchors: Dict[int, dict]) -> Dict[Tuple[int, int], Tuple[float, float, str, str]]:
    """Draw interest points as stars; return endpoints indexed by (node_id, branch_id)."""
    colors = {
        "PENDING": COLOR_PENDING,
        "TRACKBACK": COLOR_TRACKBACK,
    }
    ray_len = 1.2
    endpoints: Dict[Tuple[int, int], Tuple[float, float, str, str]] = {}
    for node in anchors.values():
        x0, y0 = node["x"], node["y"]
        for br in node.get("branches", []):
            status = br.get("status", "PENDING")
            color = colors.get(status, "#888888")
            heading = float(br.get("heading_world", 0.0))
            x1 = x0 + ray_len * math.cos(heading)
            y1 = y0 + ray_len * math.sin(heading)
            endpoints[(int(node["id"]), int(br.get("id", 0)))] = (x1, y1, color, status)
            # 只画星标，不画射线，突出兴趣点本身
            ax.scatter([x1], [y1], s=90, marker="*", color=color, edgecolors="k", linewidths=0.6, alpha=0.95)
    # Legend proxies
    for name, color in colors.items():
        ax.plot([], [], color=color, marker="*", linewidth=0, markersize=10, label=f"branch {name.lower()}")
    return endpoints


def plot_pose(ax, pose: dict):
    x, y = pose.get("x"), pose.get("y")
    if x is None or y is None:
        return
    ax.scatter([x], [y], s=70, marker="*", color="#ffb703", edgecolors="k", linewidths=0.5, zorder=5, label="robot pose")


def plot_backtrack(ax, data: dict, anchors: Dict[int, dict]):
    """画回退起点（紫星）以及回退路径（琥珀线）"""
    start_pose = data.get("backtrack_start_pose") or {}
    if start_pose.get("x") is None or start_pose.get("y") is None:
        return
    x0, y0 = start_pose["x"], start_pose["y"]
    ax.scatter([x0], [y0], s=110, marker="*", color=COLOR_BACKTRACK_START, edgecolors="k", linewidths=0.7, zorder=6, label="backtrack start")

    path_ids = data.get("backtrack_path") or []
    pts = []
    for aid in path_ids:
        node = anchors.get(int(aid))
        if node:
            pts.append((node["x"], node["y"]))
    if len(pts) >= 2:
        xs, ys = zip(*pts)
        # 探索阶段：目标兴趣点到回退起点，用水鸭色（路径反向）
        ax.plot(list(xs[::-1]), list(ys[::-1]), color=COLOR_EXPLORE_LINE, linewidth=2.8, alpha=0.9, zorder=3, label="explore path")
        # 回退阶段：起点到目标，用琥珀色
        ax.plot(xs, ys, color=COLOR_BACKTRACK_LINE, linewidth=2.8, alpha=0.95, zorder=4, label="backtrack path")


def main():
    parser = argparse.ArgumentParser(description="Visualize Access-Topo JSON.")
    parser.add_argument("--json", help="Path to access_topo_*.json")
    parser.add_argument("--tree-dir", default="/home/zrz/lste_ws/src/lste_topo_access/topo_tree/tree",
                        help="Root directory to search for latest json when --json is not provided.")
    parser.add_argument("--save", help="Save figure to file (e.g., topo.png). If not set, show interactively.")
    args = parser.parse_args()

    if args.json:
        json_path = args.json
    else:
        json_path = find_latest_json(args.tree_dir)
        if json_path is None:
            raise SystemExit(f"No access_topo_*.json found under {args.tree_dir}")

    data = load_json(json_path)
    anchors = build_anchor_map(data.get("nodes", []))

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f5f7fa")
    plot_path(ax, anchors)
    endpoints = plot_branches(ax, anchors)
    plot_backtrack(ax, data, endpoints)
    plot_pose(ax, data.get("pose", {}))

    ax.set_title(f"Access-Topo: {os.path.basename(json_path)}", fontsize=12, color="#222222")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle="--", color="#d0d7de", alpha=0.7)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
