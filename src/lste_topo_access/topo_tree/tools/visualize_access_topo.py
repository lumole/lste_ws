#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Access-Topo JSON dumps (Updated Color Scheme).

Update (2025-12-26):
- Only visualize *completed* backtrack sessions from backtrack_history (must have "finished").
- Ignore current/in-progress backtrack fields: current_backtrack, backtrack_start_pose, backtrack_path.
- Draw full completed backtrack segments in Amber by reconstructing path on the anchor "prev" tree
  using (start_anchor -> end_anchor). Fallback to sess["path"] if needed.
"""

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

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


# ---------------------------
# Anchor-tree path utilities
# ---------------------------

def _get_prev_id(anchors: Dict[int, dict], aid: int) -> Optional[int]:
    node = anchors.get(aid)
    if not node:
        return None
    prev = node.get("prev")
    if prev is None:
        return None
    try:
        return int(prev)
    except Exception:
        return None


def _trace_to_root(anchors: Dict[int, dict], start_id: int, max_hops: int = 200000) -> List[int]:
    """Return chain [start, parent, parent, ...] up to root (inclusive)."""
    chain = []
    cur = start_id
    hops = 0
    while cur is not None and cur in anchors and hops < max_hops:
        chain.append(cur)
        cur = _get_prev_id(anchors, cur)
        hops += 1
    return chain


def path_between_anchors_tree(anchors: Dict[int, dict], a: int, b: int) -> List[int]:
    """
    Get path from anchor a to anchor b in a rooted tree defined by "prev" pointers.
    Returns list of anchor ids in order [a ... b]. Empty if cannot build.
    """
    if a not in anchors or b not in anchors:
        return []

    chain_a = _trace_to_root(anchors, a)
    if not chain_a:
        return []
    ancestors_a = {nid: idx for idx, nid in enumerate(chain_a)}

    chain_b = _trace_to_root(anchors, b)
    if not chain_b:
        return []

    lca = None
    lca_idx_b = None
    for j, nid in enumerate(chain_b):
        if nid in ancestors_a:
            lca = nid
            lca_idx_b = j
            break
    if lca is None:
        return []

    idx_lca_in_a = ancestors_a[lca]
    part_a = chain_a[: idx_lca_in_a + 1]          # [a ... lca]
    part_b = list(reversed(chain_b[: lca_idx_b + 1]))  # [lca ... b]
    if part_b and part_a and part_b[0] == part_a[-1]:
        part_b = part_b[1:]  # remove duplicated lca
    return part_a + part_b


def anchor_ids_to_xy(anchors: Dict[int, dict], ids: List[int]) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for aid in ids:
        node = anchors.get(int(aid))
        if node:
            pts.append((float(node["x"]), float(node["y"])))
    return pts


# ---------------------------
# Plotting functions
# ---------------------------

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
        lc = LineCollection(segments, colors=COLOR_ANCHOR_PATH, linewidths=2.0, alpha=0.6, zorder=1)
        ax.add_collection(lc)

    xs = [n["x"] for n in anchors.values()]
    ys = [n["y"] for n in anchors.values()]
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

            if status == "TRACKBACK":
                trackback_segments.append([(x0, y0), (x1, y1)])
            else:
                search_segments.append([(x0, y0), (x1, y1)])

            ax.scatter([x1], [y1], s=80, marker="*", color=COLOR_INTEREST,
                       edgecolors="white", linewidths=0.5, zorder=4)

    if search_segments:
        lc_search = LineCollection(search_segments, colors=COLOR_BRANCH_SEARCH,
                                   linewidths=2.0, alpha=0.9, zorder=3)
        ax.add_collection(lc_search)
    if trackback_segments:
        lc_tb = LineCollection(trackback_segments, colors=COLOR_BRANCH_TRACKBACK,
                               linewidths=2.0, alpha=0.9, zorder=3)
        ax.add_collection(lc_tb)


def plot_pose(ax, pose: dict):
    x, y = pose.get("x"), pose.get("y")
    if x is None or y is None:
        return
    ax.scatter([x], [y], s=120, marker="*", color=COLOR_ROBOT,
               edgecolors="black", linewidths=0.8, zorder=10)


def plot_backtrack(ax, data: dict, anchors: Dict[int, dict]):
    """
    ONLY draw completed backtrack sessions:
    - Source: backtrack_history (dict sessions)
    - Condition: sess has "finished" (not None)
    - Draw:
      * Backtrack start pose (purple star)
      * Full backtrack path in amber (reconstructed by start_anchor -> end_anchor)
    """
    history = data.get("backtrack_history") or []
    if not (history and isinstance(history, list) and isinstance(history[0], dict)):
        return

    amber_line_segments = []

    for sess in history:
        # Only completed sessions
        if sess.get("finished") is None:
            continue

        # Purple star: session start_pose (if available)
        sp = sess.get("start_pose") or {}
        if sp.get("x") is not None and sp.get("y") is not None:
            ax.scatter([float(sp["x"])], [float(sp["y"])], s=120, marker="*",
                       color=COLOR_BACKTRACK_START, edgecolors="white", linewidths=0.5, zorder=6)

        # Prefer reconstruct full path from start_anchor -> end_anchor
        sa = sess.get("start_anchor")
        ea = sess.get("end_anchor")
        sa_i, ea_i = None, None
        try:
            sa_i = int(sa) if sa is not None else None
            ea_i = int(ea) if ea is not None else None
        except Exception:
            sa_i, ea_i = None, None

        pts: List[Tuple[float, float]] = []
        if sa_i is not None and ea_i is not None:
            full_ids = path_between_anchors_tree(anchors, sa_i, ea_i)
            pts = anchor_ids_to_xy(anchors, full_ids)

        # Fallback: use sess["path"] if reconstruction failed
        if len(pts) < 2:
            path_ids = sess.get("path") or []
            pts = anchor_ids_to_xy(anchors, path_ids)

        # Add segments for LineCollection
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                amber_line_segments.append([pts[i], pts[i + 1]])

    if amber_line_segments:
        lc = LineCollection(
            amber_line_segments,
            colors=COLOR_BRANCH_TRACKBACK,
            linewidths=3.0,
            alpha=0.55,
            zorder=1.6
        )
        ax.add_collection(lc)


def create_custom_legend(ax):
    legend_elements = [
        mlines.Line2D([], [], color='white', marker='o', markerfacecolor=COLOR_ANCHOR_POINTS,
                      markersize=8, label='anchor'),

        mlines.Line2D([], [], color=COLOR_BRANCH_SEARCH, linewidth=2, label='branch search'),

        mlines.Line2D([], [], color=COLOR_BRANCH_TRACKBACK, linewidth=2, label='branch trackback'),

        mlines.Line2D([], [], color='white', marker='*', markerfacecolor=COLOR_INTEREST,
                      markersize=10, label='interest'),

        mlines.Line2D([], [], color='white', marker='*', markerfacecolor=COLOR_BACKTRACK_START,
                      markersize=10, label='backtrack start'),

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

    plot_path(ax, anchors)
    plot_branches(ax, anchors)
    plot_backtrack(ax, data, anchors)  # <-- only completed sessions
    plot_pose(ax, data.get("pose", {}))

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
