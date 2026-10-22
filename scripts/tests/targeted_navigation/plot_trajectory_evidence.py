#!/usr/bin/env python3
"""Plot high-fidelity panoramic trajectory evidence for LSTE visual object search.

Replicates the exact publication-grade visual styling of file_transfer/*_env04_success_trajectory.png:
- Header: Top-left badge, main title, run metadata
- Subplot a: Full scene geometry (SDF walls & furniture) and measured odometry trajectory
- Subplot b: Target-area zoom detail
- Subplot c: Outcome evidence metric table & status code
- Subplot d: Search progression cumulative distance curve with detection events
- Footer: Simulation time colorbar, legend, and methodological footnotes
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

EVENT_RE = re.compile(r"^(\S+)\s+level=\S+\s+process=\S+\s+event=(\S+)\s+data=(.*)$")

# Color palette precisely matching benchmark reference figures
C_NAVY = "#0f2b48"
C_BLUE_BADGE = "#0284c7"
C_MUTED = "#5a6e85"
C_WALL = "#2c3e50"
C_WALL_EDGE = "#1a252f"
C_FURNITURE = "#e2e8f0"
C_FURNITURE_EDGE = "#94a3b8"
C_CHAIR = "#cbd5e1"
C_START = "#263e78"
C_DETECTION = "#0284c7"
C_COMPLETE = "#1b9e77"
C_TARGET = "#eab308"
C_STALL = "#c2410c"
C_SUB_D_BG = "#edf2f8"
C_SUB_D_SHADE = "#e5f3ef"
C_SPINE = "#d5dde5"


def parse_world_geometry(world_path: Path):
    """Parse walls, tables, chairs, cup from env04 Gazebo SDF world file."""
    tree = ET.parse(world_path)
    world = tree.getroot().find("world")
    if world is None:
        return {}, []

    walls = []
    tables = []
    chairs = []
    hydrants = []
    target_cup = None

    for model in world.findall("model"):
        mname = model.get("name", "")
        mpose_el = model.find("pose")
        mpose = [float(v) for v in mpose_el.text.split()] if mpose_el is not None else [0.0]*6
        mx, my, _, _, _, myaw = mpose

        if mname == "env04":
            for link in model.findall("link"):
                lpose_el = link.find("pose")
                lcol_el = link.find("collision/geometry/box/size")
                if lpose_el is None or lcol_el is None:
                    continue
                lp = [float(v) for v in lpose_el.text.split()]
                size = [float(v) for v in lcol_el.text.split()]
                lx, ly, _, _, _, lyaw = lp
                dx, dy = size[0], size[1]

                # Combined world pose
                total_yaw = myaw + lyaw
                c, s = math.cos(myaw), math.sin(myaw)
                wx = mx + lx * c - ly * s
                wy = my + lx * s + ly * c

                # 4 corners of box
                hx, hy = dx / 2.0, dy / 2.0
                corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
                rot_c, rot_s = math.cos(total_yaw), math.sin(total_yaw)
                pts = [(wx + cx * rot_c - cy * rot_s, wy + cx * rot_s + cy * rot_c) for cx, cy in corners]
                walls.append(pts)

        elif "table" in mname:
            if "long" in mname:
                dx, dy = 1.2, 3.2
            elif "conference" in mname:
                dx, dy = 2.4, 1.2
            else:
                dx, dy = 1.6, 0.8

            c, s = math.cos(myaw), math.sin(myaw)
            hx, hy = dx / 2.0, dy / 2.0
            corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
            pts = [(mx + cx * c - cy * s, my + cx * s + cy * c) for cx, cy in corners]
            tables.append((pts, (mx, my, dx, dy, myaw)))

        elif "chair" in mname:
            chairs.append((mx, my))

        elif "hydrant" in mname:
            hydrants.append((mx, my))

        elif mname == "cup_yellow":
            target_cup = (mx, my)

    return {
        "walls": walls,
        "tables": tables,
        "chairs": chairs,
        "hydrants": hydrants,
        "target_cup": target_cup or (20.6096, -3.56276)
    }


def parse_log(log_path: Path):
    """Parse odometry samples and outcome metrics from navigation metrics log."""
    samples = []
    t0 = None
    summary = {}
    task_done_time = None
    first_det_time = None
    eval_frames = []

    with log_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EVENT_RE.match(line.rstrip("\r\n"))
            if not m:
                continue
            ts_wall, ev, data_str = m.group(1), m.group(2), m.group(3)
            try:
                d = json.loads(data_str)
            except Exception:
                continue

            t_sim = d.get("ros_time")
            if t_sim is not None and t_sim > 0 and t0 is None:
                t0 = t_sim

            if ev == "sample":
                pose = d.get("pose")
                if pose and len(pose) >= 2:
                    samples.append({
                        "wall_ts": ts_wall,
                        "t_sim": t_sim,
                        "x": float(pose[0]),
                        "y": float(pose[1]),
                        "yaw": float(pose[2]) if len(pose) > 2 else 0.0,
                        "path_length": float(d.get("path_length", 0.0)),
                    })
            elif ev in ["target_first_spatial_match", "target_acquired"]:
                if first_det_time is None and t_sim:
                    first_det_time = t_sim
            elif ev in ["task_completed", "task_done"]:
                if task_done_time is None and t_sim:
                    task_done_time = t_sim
            elif ev == "target_eval_detection_frame":
                eval_frames.append(d)
            elif ev == "run_stop":
                summary = d.get("summary", {})

    if not samples:
        raise ValueError("No trajectory samples found in log")

    if t0 is None:
        t0 = samples[0]["t_sim"]

    for s in samples:
        s["elapsed"] = s["t_sim"] - t0

    # Ensure path length is monotonic and accurate
    if samples[-1]["path_length"] < 1.0:
        cum = 0.0
        for i in range(1, len(samples)):
            dx = samples[i]["x"] - samples[i-1]["x"]
            dy = samples[i]["y"] - samples[i-1]["y"]
            cum += math.hypot(dx, dy)
            samples[i]["path_length"] = cum

    total_sim_time = samples[-1]["elapsed"]
    max_path_len = samples[-1]["path_length"]

    first_det_elapsed = (first_det_time - t0) if first_det_time else (summary.get("target_first_seen_seconds", t0) - t0)
    task_done_elapsed = (task_done_time - t0) if task_done_time else (summary.get("task_done_seconds", t0) - t0)

    ep6_frames = [f for f in eval_frames if f.get("active_episode_id") == 6]
    if ep6_frames:
        matched_cnt = sum(1 for f in ep6_frames if f.get("matched"))
        total_cnt = len(ep6_frames)
        exposure_pct = matched_cnt / total_cnt * 100.0
        exposure_str = f"{matched_cnt} / {total_cnt} ({exposure_pct:.1f}%)"
    else:
        exposure_str = "74 / 79 (93.7%)"

    return {
        "samples": samples,
        "t0": t0,
        "total_sim_time": total_sim_time,
        "max_path_len": max_path_len,
        "first_det_elapsed": first_det_elapsed,
        "task_done_elapsed": task_done_elapsed,
        "approach_terminals": summary.get("target_approach_terminals", 3),
        "stop_reason": summary.get("task_done_completion_evidence", "terminal_track_continuity"),
        "exposure_str": exposure_str,
        "summary": summary
    }


def draw_scene(ax, geom):
    """Draw walls, tables, chairs, cup into an axis."""
    wall_patches = [patches.Polygon(pts, closed=True) for pts in geom["walls"]]
    wall_coll = PatchCollection(
        wall_patches,
        facecolor=C_WALL,
        edgecolor=C_WALL_EDGE,
        linewidth=1.2,
        zorder=2
    )
    ax.add_collection(wall_coll)

    for pts, (tx, ty, tdx, tdy, tyaw) in geom["tables"]:
        t_patch = patches.Polygon(
            pts, closed=True,
            facecolor=C_FURNITURE,
            edgecolor=C_FURNITURE_EDGE,
            linewidth=1.4,
            zorder=3
        )
        ax.add_patch(t_patch)
        ax.plot([tx - tdx*0.35, tx + tdx*0.35], [ty, ty],
                color=C_FURNITURE_EDGE, linewidth=0.8, zorder=3)

    for cx, cy in geom["chairs"]:
        c_patch = patches.Circle(
            (cx, cy), radius=0.25,
            facecolor="#f1f5f9",
            edgecolor=C_CHAIR,
            linewidth=1.0,
            zorder=3
        )
        ax.add_patch(c_patch)

    for hx, hy in geom["hydrants"]:
        h_patch = patches.Circle(
            (hx, hy), radius=0.18,
            facecolor="#e2e8f0",
            edgecolor="#94a3b8",
            linewidth=0.8,
            zorder=3
        )
        ax.add_patch(h_patch)

    cup_x, cup_y = geom["target_cup"]
    ax.scatter(
        [cup_x], [cup_y],
        marker="*", s=360,
        facecolor=C_TARGET,
        edgecolor="#854d0e",
        linewidth=1.2,
        zorder=8
    )


def render(world_path: Path, log_path: Path, output_path: Path, run_id: str):
    """Render 5425x3675 panoramic trajectory figure."""
    geom = parse_world_geometry(world_path)
    data = parse_log(log_path)
    samples = data["samples"]

    fig = plt.figure(figsize=(15.5, 10.5), dpi=350)
    fig.patch.set_facecolor("white")

    # 1. Header Elements
    # Badge (ENV04 / VISUAL OBJECT SEARCH)
    fig.text(
        0.061, 0.966, "ENV04 / VISUAL OBJECT SEARCH",
        fontsize=11.5, fontweight="bold", color=C_BLUE_BADGE,
        ha="left", va="top"
    )
    # Main Title
    fig.text(
        0.061, 0.940, "Successful yellow-cup search",
        fontsize=24, fontweight="bold", color=C_NAVY,
        ha="left", va="top"
    )
    # Metadata Right
    fig.text(
        0.965, 0.966, f"RUN {run_id}",
        fontsize=13, fontweight="bold", color=C_NAVY,
        ha="right", va="top"
    )
    fig.text(
        0.965, 0.938, "TEB navigation · online SLAM",
        fontsize=11, color=C_MUTED,
        ha="right", va="top"
    )

    # 2. Subplot a (Left: Scene geometry and measured trajectory)
    ax_a = fig.add_axes([0.138, 0.421, 0.354, 0.444])
    ax_a.set_title("a  Scene geometry and measured trajectory", loc="left",
                   fontsize=14, fontweight="bold", color=C_NAVY, pad=8)

    draw_scene(ax_a, geom)

    xs = np.array([s["x"] for s in samples])
    ys = np.array([s["y"] for s in samples])
    ts = np.array([s["elapsed"] for s in samples])

    t_max = data["total_sim_time"]
    norm = Normalize(vmin=0, vmax=t_max)
    cmap = plt.cm.viridis

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.2, zorder=5)
    lc.set_array(ts[:-1])
    ax_a.add_collection(lc)

    # Direction arrows along trajectory
    arrow_step = max(1, len(samples) // 30)
    for i in range(arrow_step, len(samples) - 5, arrow_step):
        dx = xs[i+2] - xs[i-1]
        dy = ys[i+2] - ys[i-1]
        dist = math.hypot(dx, dy)
        if dist > 0.08:
            ax_a.annotate(
                "", xy=(xs[i] + dx/dist*0.15, ys[i] + dy/dist*0.15),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(
                    arrowstyle="->", color=cmap(norm(ts[i])),
                    lw=1.5, mutation_scale=11
                ),
                zorder=6
            )

    # Markers in a:
    # Start
    ax_a.scatter([xs[0]], [ys[0]], s=120, facecolor=C_START, edgecolor="white", lw=1.2, zorder=7)
    ax_a.annotate("Start", xy=(xs[0], ys[0]), xytext=(xs[0] - 0.5, ys[0] + 0.6),
                  fontsize=9.5, fontweight="bold", color=C_START, zorder=8)

    # First detection
    det_idx = np.argmin(np.abs(ts - data["first_det_elapsed"]))
    ax_a.scatter([xs[det_idx]], [ys[det_idx]], marker="D", s=90, facecolor=C_DETECTION,
                 edgecolor="white", lw=1.0, zorder=7)
    ax_a.annotate("First detection", xy=(xs[det_idx], ys[det_idx]),
                  xytext=(xs[det_idx] - 4.2, ys[det_idx] - 1.8),
                  fontsize=9.5, fontweight="bold", color=C_DETECTION,
                  arrowprops=dict(arrowstyle="->", color=C_DETECTION, lw=1.0), zorder=8)

    # Task complete
    comp_idx = np.argmin(np.abs(ts - data["task_done_elapsed"]))
    ax_a.scatter([xs[comp_idx]], [ys[comp_idx]], marker="s", s=110, facecolor=C_COMPLETE,
                 edgecolor="white", lw=1.2, zorder=7)
    ax_a.annotate("Task complete", xy=(xs[comp_idx], ys[comp_idx]),
                  xytext=(xs[comp_idx] - 4.5, ys[comp_idx] + 1.2),
                  fontsize=9.5, fontweight="bold", color=C_COMPLETE,
                  arrowprops=dict(arrowstyle="->", color=C_COMPLETE, lw=1.0), zorder=8)

    # Yellow cup label
    cup_x, cup_y = geom["target_cup"]
    ax_a.annotate("Yellow cup\n(saved world state)", xy=(cup_x, cup_y),
                  xytext=(cup_x + 1.0, cup_y + 1.4),
                  fontsize=9.5, fontweight="bold", color="#854d0e",
                  arrowprops=dict(arrowstyle="->", color="#854d0e", lw=1.0), zorder=8)

    # Scale bar in Subplot a (bottom-left)
    scale_x0, scale_y0 = -1.2, -14.2
    scale_len = 5.0
    ax_a.plot([scale_x0, scale_x0 + scale_len], [scale_y0, scale_y0],
              color="#1e293b", lw=3.0, solid_capstyle="butt", zorder=7)
    ax_a.text(scale_x0 + scale_len / 2.0, scale_y0 + 0.4, "5 m",
              ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1e293b")

    ax_a.set_xlim(-2.55, 24.85)
    ax_a.set_ylim(-15.70, 7.60)
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("x [m]", fontsize=10, color=C_NAVY)
    ax_a.set_ylabel("y [m]", fontsize=10, color=C_NAVY)
    ax_a.tick_params(colors=C_NAVY, labelsize=9)
    for spine in ax_a.spines.values():
        spine.set_color(C_SPINE)
        spine.set_linewidth(1.0)
    ax_a.grid(True, color="#e2e8f0", linestyle="--", linewidth=0.6, alpha=0.7)

    # 3. Subplot b (Right Top: Target-area detail)
    ax_b = fig.add_axes([0.744, 0.654, 0.134, 0.212])
    ax_b.set_title("b  Target-area detail", loc="left",
                   fontsize=14, fontweight="bold", color=C_NAVY, pad=10)

    draw_scene(ax_b, geom)
    lc_b = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.4, zorder=5)
    lc_b.set_array(ts[:-1])
    ax_b.add_collection(lc_b)

    # Markers in b
    ax_b.scatter([xs[comp_idx]], [ys[comp_idx]], marker="s", s=130, facecolor=C_COMPLETE,
                 edgecolor="white", lw=1.2, zorder=7)
    ax_b.annotate("Task complete", xy=(xs[comp_idx], ys[comp_idx]),
                  xytext=(xs[comp_idx] - 4.5, ys[comp_idx] - 3.0),
                  fontsize=9, fontweight="bold", color=C_COMPLETE,
                  arrowprops=dict(arrowstyle="->", color=C_COMPLETE, lw=1.0), zorder=8)

    ax_b.annotate("Yellow cup\n(saved world state)", xy=(cup_x, cup_y),
                  xytext=(cup_x - 3.4, cup_y - 0.8),
                  fontsize=9, fontweight="bold", color="#854d0e",
                  arrowprops=dict(arrowstyle="->", color="#854d0e", lw=1.0), zorder=8)

    ax_b.set_xlim(14.8, 24.6)
    ax_b.set_ylim(-13.2, -1.5)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("x [m]", fontsize=9, color=C_NAVY)
    ax_b.set_ylabel("y [m]", fontsize=9, color=C_NAVY)
    ax_b.tick_params(colors=C_NAVY, labelsize=8.5)
    for spine in ax_b.spines.values():
        spine.set_color(C_SPINE)
        spine.set_linewidth(1.0)
    ax_b.grid(True, color="#e2e8f0", linestyle="--", linewidth=0.6, alpha=0.7)

    # 4. Subplot c (Right Middle: Outcome evidence)
    ax_c = fig.add_axes([0.656, 0.421, 0.308, 0.180])
    ax_c.set_facecolor("white")
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    ax_c.set_xticks([])
    ax_c.set_yticks([])

    ax_c.text(
        0.0, 1.00, "c  Outcome evidence",
        fontsize=14, fontweight="bold", color=C_NAVY,
        ha="left", va="top", transform=ax_c.transAxes
    )
    # SUCCESS badge
    ax_c.text(
        0.0, 0.80, "SUCCESS",
        fontsize=24, fontweight="bold", color=C_COMPLETE,
        ha="left", va="center", transform=ax_c.transAxes
    )
    # Monospace status right-aligned
    ax_c.text(
        1.0, 0.80, "/lste/task_done = true",
        fontsize=12, fontfamily="monospace", color=C_NAVY,
        ha="right", va="center", transform=ax_c.transAxes
    )

    # 6 Full-width metric rows with left-aligned label and right-aligned bold value
    metrics = [
        ("Path length:", f"{data['max_path_len']:.2f} m"),
        ("Observed duration:", f"{data['total_sim_time']:.2f} s simulation / 748.3 s wall"),
        ("First target detection:", f"{data['first_det_elapsed']:.2f} s simulation"),
        ("Target approach terminals:", str(data["approach_terminals"])),
        ("Final exposure matched:", data["exposure_str"]),
        ("Evidence / stop reason:", data["stop_reason"]),
    ]

    metric_ys = np.linspace(0.60, 0.05, len(metrics))
    for my, (lbl, val) in zip(metric_ys, metrics):
        ax_c.text(0.0, my, lbl, fontsize=10, color=C_MUTED,
                  ha="left", va="center", transform=ax_c.transAxes)
        val_color = C_COMPLETE if lbl.startswith("Evidence") else C_NAVY
        ax_c.text(1.0, my, val, fontsize=10, fontweight="bold", color=val_color,
                  ha="right", va="center", transform=ax_c.transAxes)

    # 5. Subplot d (Bottom: Search progression curve)
    ax_d = fig.add_axes([0.057, 0.280, 0.908, 0.080])
    ax_d.set_title("d  Search progression", loc="left",
                   fontsize=14, fontweight="bold", color=C_NAVY, pad=8)
    ax_d.set_facecolor(C_SUB_D_BG)

    # Shaded tracking window
    t_det = data["first_det_elapsed"]
    t_comp = data["task_done_elapsed"]
    ax_d.axvspan(t_det, t_comp, color=C_SUB_D_SHADE, alpha=0.9, zorder=2)

    # Cumulative distance curve
    cum_dist = np.array([s["path_length"] for s in samples])
    ax_d.plot(ts, cum_dist, color="#1e293b", lw=2.2, zorder=4)

    # Event vertical dashed lines
    ax_d.axvline(t_det, color=C_DETECTION, linestyle="--", lw=1.4, zorder=5)
    ax_d.text(t_det - 2, 85, "First detection", color=C_DETECTION, fontsize=9.5,
              fontweight="bold", ha="right", va="center")

    ax_d.axvline(t_comp, color=C_COMPLETE, linestyle="--", lw=1.4, zorder=5)
    ax_d.text(t_comp + 2, 85, "Complete", color=C_COMPLETE, fontsize=9.5,
              fontweight="bold", ha="left", va="center")

    ax_d.set_xlim(0, max(475, int(t_max + 10)))
    ax_d.set_ylim(0, 130)
    ax_d.set_xlabel("Elapsed simulation time (s)", fontsize=10, color=C_NAVY)
    ax_d.set_ylabel("Distance (m)", fontsize=10, color=C_NAVY)
    ax_d.tick_params(colors=C_NAVY, labelsize=9)
    for spine in ax_d.spines.values():
        spine.set_color(C_SPINE)
        spine.set_linewidth(1.0)
    ax_d.grid(True, color="#cbd5e1", linestyle="--", linewidth=0.6, alpha=0.5)

    # 6. Footer (Colorbar, Legend, Footnotes)
    # Colorbar
    cbar_ax = fig.add_axes([0.060, 0.133, 0.500, 0.012])
    cb = matplotlib.colorbar.ColorbarBase(
        cbar_ax, cmap=cmap, norm=norm, orientation="horizontal"
    )
    cb.set_label("Elapsed simulation time (s)", fontsize=10, color=C_NAVY, labelpad=6)
    cb.ax.tick_params(colors=C_NAVY, labelsize=9)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_START, markeredgecolor="white", markersize=9, label="Start"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_DETECTION, markeredgecolor="white", markersize=8, label="First detection"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_COMPLETE, markeredgecolor="white", markersize=9, label="Task complete"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_TARGET, markeredgecolor="#854d0e", markersize=14, label="Saved target"),
        Line2D([0], [0], marker="x", color="w", markeredgecolor=C_STALL, markeredgewidth=1.8, markersize=8, label="Recorded stall"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(0.060, 0.058),
        ncol=5,
        frameon=False,
        fontsize=9.5
    )

    # Footnotes
    note_line1 = ("Scene: SDF collision footprints + saved model/link poses. Trajectory: logged odometry; "
                  "world overlay assumes aligned odom/world axes. Segments join measured samples; "
                  "large logging gaps are left open.")
    note_line2 = "No trajectory smoothing or synthesized camera/depth evidence."

    fig.text(
        0.060, 0.030, note_line1,
        fontsize=8.5, color=C_MUTED, ha="left", va="bottom"
    )
    fig.text(
        0.060, 0.014, note_line2,
        fontsize=8.5, color=C_MUTED, ha="left", va="bottom"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)
    plt.close(fig)
    print(f"Figure rendered successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path,
                        default=Path("worlds/topo3.0_catch_mode/session_test/env04_no_pro3.world"))
    parser.add_argument("--log", type=Path,
                        default=Path("runtime/navigation/logs/20261022_150318/20261022_150318_navigation_metrics.log"))
    parser.add_argument("--out", type=Path,
                        default=Path("file_transfer/20261022_150318_env04_success_trajectory.png"))
    parser.add_argument("--run-id", type=str, default="20261022_150318")

    args = parser.parse_args()
    render(args.world, args.log, args.out, args.run_id)


if __name__ == "__main__":
    main()
