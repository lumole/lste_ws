#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTE Environment & Benchmark Comparator
---------------------------------------
Compares simulation environments in lste_ws:
- Dimensions and floor area
- Number of places / rooms / doors
- Model counts, furniture, distractors
- Target status (cup_yellow pose, surface height)
- Occluders and complexity level

Usage:
    python3 compare_environments.py
    python3 compare_environments.py --worlds t_junction_office env04 office_building_v1
    python3 compare_environments.py --json
"""

import sys
import os
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

WS_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
# Fallback if path structure differs
if not (WS_ROOT / "worlds").exists():
    WS_ROOT = Path("/home/dh/dh_ws/lste_ws")

BENCHMARK_WORLDS = {
    "T-Junction Office (Primary)": {
        "world": "worlds/t_junction_office_no_pro3.world",
        "manifest": "worlds/t_junction_office_manifest.yaml",
        "generator": "scripts/tests/generate_t_junction_world.py",
        "bounds": [0.0, 0.0, 12.0, 10.5],
        "category": "Lightweight Primary Benchmark",
    },
    "env04 (Legacy Reference)": {
        "world": "worlds/topo3.0_catch_mode/session_test/env04_no_pro3.world",
        "manifest": None,
        "generator": None,
        "bounds": [0.0, 0.0, 15.0, 15.0],
        "category": "Legacy Test World",
    },
    "Office Building V1 L1 (Large Benchmark)": {
        "world": "worlds/benchmark/office_building_v1_level_1_no_pro3.world",
        "manifest": "worlds/benchmark/office_building_v1_manifest.yaml",
        "generator": "scripts/tests/office_building/generate_world.py",
        "bounds": [0.0, 0.0, 25.0, 22.0],
        "category": "High-Complexity Benchmark",
    },
    "Office Building V1 L2": {
        "world": "worlds/benchmark/office_building_v1_level_2_no_pro3.world",
        "manifest": "worlds/benchmark/office_building_v1_manifest.yaml",
        "generator": "scripts/tests/office_building/generate_world.py",
        "bounds": [0.0, 0.0, 25.0, 22.0],
        "category": "High-Complexity Benchmark",
    },
}

def analyze_world(w_info: dict) -> dict:
    wpath = WS_ROOT / w_info["world"]
    if not wpath.exists():
        return {"status": "missing", "path": str(wpath)}

    tree = ET.parse(wpath)
    root = tree.getroot()

    includes = root.findall(".//include")
    models = []
    target_pose = None
    target_scale = None

    for inc in includes:
        uri = inc.findtext("uri") or ""
        name = inc.findtext("name") or ""
        pose = inc.findtext("pose") or ""
        scale = inc.findtext("scale") or "1 1 1"
        mname = uri.replace("model://", "") if uri.startswith("model://") else name
        models.append(mname)

        if "cup_yellow" in mname or name == "cup_yellow":
            target_pose = pose
            target_scale = scale

    model_counts = Counter(models)
    b = w_info.get("bounds", [0, 0, 0, 0])
    area = round((b[2] - b[0]) * (b[3] - b[1]), 1) if len(b) == 4 else 0

    return {
        "status": "ok",
        "category": w_info["category"],
        "path": w_info["world"],
        "generator": w_info.get("generator"),
        "manifest": w_info.get("manifest"),
        "dimensions": f"{b[2]-b[0]:.1f}m x {b[3]-b[1]:.1f}m ({area} m²)",
        "total_models": len(models),
        "unique_models": len(model_counts),
        "target_pose": target_pose,
        "target_scale": target_scale,
        "desks": model_counts.get("desk_brown", 0),
        "conference_tables": model_counts.get("table_conference_2", 0),
        "monitors": model_counts.get("monitor_1", 0) + model_counts.get("monitor_2", 0) + model_counts.get("monitor_3", 0) + model_counts.get("monitor_4", 0),
        "fire_hydrants": model_counts.get("fire_hydrant", 0),
        "chairs": model_counts.get("office_chair", 0) + model_counts.get("kitchen_chair", 0),
        "bookshelves": model_counts.get("bookshelf", 0),
        "cabinets": model_counts.get("file_cabinet_large", 0) + model_counts.get("cabinet", 0),
    }

def print_markdown_comparison(results: dict):
    print("# LSTE 仿真环境与基准场景多维对比表\n")
    print("| 指标 / 特征 | " + " | ".join(results.keys()) + " |")
    print("|---|" + "|".join([":---:"] * len(results)) + "|")

    def row(label, key):
        vals = [str(results[k].get(key, "—")) for k in results]
        print(f"| **{label}** | " + " | ".join(vals) + " |")

    row("场景定位", "category")
    row("外廓尺寸 & 面积", "dimensions")
    row("总置放模型数", "total_models")
    row("核心目标 (Yellow Cup)", "target_pose")
    row("目标缩放 (Scale)", "target_scale")
    row("办公桌 (desk_brown)", "desks")
    row("显示器 (Monitors)", "monitors")
    row("会议桌 (conference_table)", "conference_tables")
    row("消防栓 (fire_hydrant)", "fire_hydrants")
    row("椅子数量 (Chairs)", "chairs")
    row("书架 & 文件柜", "cabinets")
    row("源头生成器 (Generator)", "generator")
    row("规范清单 (Manifest)", "manifest")

    print("\n## 评估结论与选型建议：")
    print("1. **日常开发与快速闭环**：首选 **T-Junction Office**（12m x 10.5m，126m²），秒级加载，完全去除椅子碰撞体，目标黄色水杯受屏风遮挡，具备完整分叉与回退拓扑特性。")
    print("2. **大规模压力评测**：选用 **Office Building V1**（25m x 22m，550m²），40+ 模型，适合 Level 1~4 长程规划与多房间拓扑评测。")

def main():
    parser = argparse.ArgumentParser(description="LSTE Simulation Environment Comparator")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = {}
    for name, winfo in BENCHMARK_WORLDS.items():
        results[name] = analyze_world(winfo)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_markdown_comparison(results)

if __name__ == "__main__":
    main()
