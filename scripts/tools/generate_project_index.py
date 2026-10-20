#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTE Workspace Project Index Generator
--------------------------------------
Automatically scans:
1. Gazebo model cache (~/.gazebo/models) and workspace model usages.
2. World files (worlds/**/*.world) and their generator scripts (scripts/**/*.py).
3. ROS packages in src/, including launch files, nodes, and AST-extracted topics.
4. Runtime launch conventions (run_demo_teb.sh, demo.sh, controller modes).

Outputs:
- docs/YYYYMMDD_project_fast_index.md (Full reference manual)
- Updates docs/introduction.md index link
- Updates AGENTS.md fast reference block (between FAST_INDEX markers)

Usage:
    python3 scripts/tools/generate_project_index.py
    python3 scripts/tools/generate_project_index.py --dry-run
"""

import os
import sys
import re
import ast
import json
import argparse
import datetime
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

WS_ROOT = Path(__file__).resolve().parent.parent.parent
GAZEBO_MODELS_DIR = Path(os.path.expanduser("~/.gazebo/models"))
TODAY = datetime.date.today().strftime("%Y%m%d")

# Semantic category mapping for common models
CATEGORY_KEYWORDS = {
    "Target / Cup": ["cup", "drink", "target"],
    "Desks & Tables": ["desk", "table"],
    "Chairs & Seating": ["chair", "sofa", "stool"],
    "Storage & Cabinets": ["cabinet", "bookshelf", "box", "bin", "shelf", "pallet"],
    "Electronics & Office": ["monitor", "computer", "keyboard", "mouse", "file"],
    "Fixtures & Safety": ["fire_hydrant", "door", "light", "partition", "occluder"],
    "Structures & Outdoor": ["building", "foyer", "campus", "garage", "house", "indoor", "tunnel", "forest"],
}

def categorize_model(name: str) -> str:
    n_lower = name.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in n_lower for kw in kws):
            return cat
    return "Miscellaneous"

def scan_model_dae_bbox(model_dir: Path):
    """Attempt to parse mesh bounding box from DAE files."""
    dae_files = list(model_dir.glob("**/*.dae"))
    if not dae_files:
        return None
    try:
        tree = ET.parse(dae_files[0])
        root = tree.getroot()
        pts = []
        for elem in root.iter():
            if elem.tag.endswith("float_array") and elem.text:
                vals = [float(x) for x in elem.text.split()]
                if len(vals) >= 12 and len(vals) % 3 == 0:
                    for i in range(0, len(vals) - 2, 3):
                        pts.append((vals[i], vals[i + 1], vals[i + 2]))
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        unit_scale = 1.0
        for elem in root.iter():
            if elem.tag.endswith("unit"):
                m = elem.attrib.get("meter")
                if m:
                    unit_scale = float(m)

        dx = round((max(xs) - min(xs)) * unit_scale, 2)
        dy = round((max(ys) - min(ys)) * unit_scale, 2)
        dz = round((max(zs) - min(zs)) * unit_scale, 2)
        min_z = round(min(zs) * unit_scale, 2)
        max_z = round(max(zs) * unit_scale, 2)
        return {"size": (dx, dy, dz), "z_range": (min_z, max_z), "mesh": dae_files[0].name}
    except Exception:
        return None

def scan_gazebo_models():
    """Scan ~/.gazebo/models and extract references in workspace."""
    model_ref_pattern = re.compile(r"model://([a-zA-Z0-9_]+)")
    used_models = set()

    for root, dirs, files in os.walk(WS_ROOT):
        if ".git" in dirs:
            dirs.remove(".git")
        if "runtime" in dirs:
            dirs.remove("runtime")
        for f in files:
            if f.endswith((".world", ".sdf", ".xml", ".launch", ".py", ".yaml")):
                try:
                    content = (Path(root) / f).read_text(errors="ignore")
                    for m in model_ref_pattern.findall(content):
                        used_models.add(m)
                except Exception:
                    pass

    # Gather empirical poses from all .world files
    empirical_usage = defaultdict(list)
    for wf in WS_ROOT.glob("worlds/**/*.world"):
        try:
            tree = ET.parse(wf)
            root = tree.getroot()
            for inc in root.iter("include"):
                uri = inc.findtext("uri")
                pose = inc.findtext("pose")
                scale = inc.findtext("scale")
                name = inc.findtext("name")
                if uri and uri.startswith("model://"):
                    mname = uri.replace("model://", "")
                    empirical_usage[mname].append({
                        "world": wf.relative_to(WS_ROOT).as_posix(),
                        "name": name,
                        "pose": pose,
                        "scale": scale,
                    })
        except Exception:
            pass

    catalog = {}
    for mname in sorted(used_models):
        mdir = GAZEBO_MODELS_DIR / mname
        exists_in_cache = mdir.exists()
        static = True
        bbox = None
        if exists_in_cache:
            bbox = scan_model_dae_bbox(mdir)
            sdf_file = mdir / "model.sdf"
            if sdf_file.exists():
                try:
                    tree = ET.parse(sdf_file)
                    st = tree.getroot().findtext(".//static")
                    if st is not None:
                        static = st.strip() in ["1", "true"]
                except Exception:
                    pass

        # Empirical stats
        instances = empirical_usage.get(mname, [])
        z_vals = []
        scales = []
        for inst in instances:
            if inst["pose"]:
                parts = inst["pose"].split()
                if len(parts) >= 3:
                    z_vals.append(float(parts[2]))
            if inst["scale"]:
                scales.append(inst["scale"])
        unique_z = sorted(list(set(z_vals)))
        unique_scales = sorted(list(set(scales)))

        catalog[mname] = {
            "name": mname,
            "category": categorize_model(mname),
            "cached": exists_in_cache,
            "static": static,
            "bbox": bbox,
            "instances_count": len(instances),
            "unique_z": unique_z,
            "unique_scales": unique_scales,
            "sample_world": instances[0]["world"] if instances else None,
        }

    return catalog

def scan_world_files():
    """Map world files to generator scripts and manifest configs."""
    worlds = {}
    for wf in WS_ROOT.glob("worlds/**/*.world"):
        rel_path = wf.relative_to(WS_ROOT).as_posix()
        worlds[rel_path] = {
            "path": rel_path,
            "generator": None,
            "manifest": None,
            "model_count": 0,
            "includes": [],
        }
        try:
            tree = ET.parse(wf)
            root = tree.getroot()
            includes = [inc.findtext("name") for inc in root.iter("include") if inc.findtext("name")]
            worlds[rel_path]["model_count"] = len(includes)
            worlds[rel_path]["includes"] = includes[:10]
        except Exception:
            pass

    # Find generator scripts
    for py_path in WS_ROOT.glob("scripts/**/*.py"):
        if py_path.name == "generate_project_index.py":
            continue
        try:
            content = py_path.read_text(errors="ignore")
            for wpath, winfo in worlds.items():
                wname = Path(wpath).name
                if wname in content and ("ElementTree" in content or "ET." in content or "build_world" in content or "write" in content):
                    winfo["generator"] = py_path.relative_to(WS_ROOT).as_posix()
        except Exception:
            pass

    # Find manifests
    for yml in WS_ROOT.glob("worlds/**/*.yaml"):
        y_name = yml.stem.replace("_manifest", "")
        for wpath, winfo in worlds.items():
            if y_name in wpath:
                winfo["manifest"] = yml.relative_to(WS_ROOT).as_posix()

    return worlds

def scan_ros_packages():
    """Scan package.xml files across src/."""
    src = WS_ROOT / "src"
    pkgs = {}
    for pxml in src.glob("**/package.xml"):
        try:
            tree = ET.parse(pxml)
            root = tree.getroot()
            pname = root.findtext("name")
            desc = (root.findtext("description") or "").strip().replace("\n", " ")
            pkg_dir = pxml.parent
            launch_files = [f.name for f in pkg_dir.glob("**/*.launch")]
            scripts = [f.name for f in pkg_dir.glob("scripts/*.py")] + [f.name for f in pkg_dir.glob("nodes/*.py")]
            pkgs[pname] = {
                "name": pname,
                "dir": pkg_dir.relative_to(WS_ROOT).as_posix(),
                "description": desc,
                "launch_files": launch_files,
                "scripts": scripts,
            }
        except Exception:
            pass
    return pkgs

def scan_ros_topics():
    """Parse Python AST in src/ to extract topics and services."""
    src = WS_ROOT / "src"
    pubs = defaultdict(list)
    subs = defaultdict(list)

    for py_file in src.glob("**/*.py"):
        try:
            tree = ast.parse(py_file.read_text(errors="ignore"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "Publisher" and node.args:
                        arg0 = node.args[0]
                        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                            pubs[arg0.value].append(py_file.name)
                    elif node.func.attr == "Subscriber" and node.args:
                        arg0 = node.args[0]
                        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                            subs[arg0.value].append(py_file.name)
        except Exception:
            pass

    return {"pubs": pubs, "subs": subs}

def generate_markdown(models_catalog, worlds_info, pkgs_info, topics_info) -> str:
    """Render full markdown index document."""
    lines = []
    lines.append(f"# LSTE 工程极速全景索引（自动扫描生成）")
    lines.append(f"> 自动生成日期: {datetime.date.today().strftime('%Y-%m-%d')} | 生成工具: `scripts/tools/generate_project_index.py`\n")
    lines.append("本文档由工程扫描程序全自动生成，汇聚了 Gazebo 模型库、仿真环境生成映射、ROS 核心包结构、数据契约与常用操作流程。Agent 与开发者可直接据此进行精准开发，**无需盲目搜索文件**。\n")

    lines.append("## 目录")
    lines.append("1. [快速操作指南 (SOP)](#1-快速操作指南-sop)")
    lines.append("2. [仿真环境与生成器映射](#2-仿真环境与生成器映射)")
    lines.append("3. [Gazebo 核心模型资产大字典](#3-gazebo-核心模型资产大字典)")
    lines.append("4. [ROS 核心工作区与包架构](#4-ros-核心工作区与包架构)")
    lines.append("5. [核心 ROS 话题契约 (AST 提取)](#5-核心-ros-话题契约-ast-提取)")
    lines.append("\n---\n")

    # 1. SOP
    lines.append("## 1. 快速操作指南 (SOP)\n")
    lines.append("### 1.1 仿真与导航栈启动")
    lines.append("```bash")
    lines.append("# 一键启动/重启 TEB 自主导航全栈 (默认 T-junction 世界)")
    lines.append("./run_demo_teb.sh restart\n")
    lines.append("# 指定自定义世界与出生点启动")
    lines.append('WORLD="/home/dh/dh_ws/lste_ws/worlds/t_junction_office_no_pro3.world" \\')
    lines.append("SPAWN_X=6.0 SPAWN_Y=1.0 SPAWN_YAW=1.5708 DISPLAY=:0 ./run_demo_teb.sh restart\n")
    lines.append("# 停止全栈")
    lines.append("./run_demo_teb.sh stop")
    lines.append("```\n")

    lines.append("### 1.2 控制器模式切换")
    lines.append("```bash")
    lines.append("# 切换为 TEB 拓扑自主规划驱动")
    lines.append("rostopic pub -1 /lste/controller_mode std_msgs/String \"data: 'teb'\"\n")
    lines.append("# 切换为键盘/遥控 Teleop 手动驱动")
    lines.append("rostopic pub -1 /lste/controller_mode std_msgs/String \"data: 'teleop'\"")
    lines.append("```\n")

    lines.append("### 1.3 实时状态诊断 (免 Grep 专用工具)")
    lines.append("```bash")
    lines.append("# 查看当前活跃导航目标 (WorkItem ID、坐标)、Place 进度与租约")
    lines.append("python3 .agents/skills/lste-status-debug/scripts/inspect_demo_status.py")
    lines.append("```\n")

    lines.append("### 1.4 SDF 世界文件语法校验")
    lines.append("```bash")
    lines.append("gz sdf -k <path_to_world_file>")
    lines.append("```\n")

    lines.append("---\n")

    # 2. Worlds & Generators
    lines.append("## 2. 仿真环境与生成器映射\n")
    lines.append("| 仿真世界 (.world) | 源头生成脚本 (Generator) | 规范清单 (Manifest) | 包含模型数 |")
    lines.append("|---|---|---|:---:|")
    for wpath, winfo in sorted(worlds_info.items()):
        gen = f"`{winfo['generator']}`" if winfo["generator"] else "*(静态手写 / 外部导入)*"
        mf = f"`{winfo['manifest']}`" if winfo["manifest"] else "—"
        lines.append(f"| [`{wpath}`]({wpath}) | {gen} | {mf} | {winfo['model_count']} |")
    lines.append("\n> **修改世界黄金准则**：若该世界存在对应的源头生成脚本（如 `generate_t_junction_world.py`），**务必修改生成脚本并重新运行**，严禁直接手改生成产物，避免后续被覆盖！\n")

    lines.append("---\n")

    # 3. Gazebo Models
    lines.append("## 3. Gazebo 核心模型资产大字典\n")
    lines.append("以下为工作区代码与世界文件中实际引用的 Gazebo 模型资产实测参数：\n")
    lines.append("| 模型名称 (URI) | 类别 | 静态 | 典型地面/桌面高度 (Z) | 默认缩放 (Scale) | 出现频次 | 典型用途 |")
    lines.append("|---|---|:---:|---|---|:---:|---|")

    # Group by category
    cats = defaultdict(list)
    for mname, minfo in sorted(models_catalog.items()):
        cats[minfo["category"]].append(minfo)

    for cat_name, mlist in sorted(cats.items()):
        for minfo in mlist:
            z_str = ", ".join(f"{z:.2f}m" for z in minfo["unique_z"]) if minfo["unique_z"] else "0.00m"
            scale_str = ", ".join(minfo["unique_scales"]) if minfo["unique_scales"] else "1 1 1"
            lines.append(
                f"| `model://{minfo['name']}` | {cat_name} | {'是' if minfo['static'] else '否'} | "
                f"`{z_str}` | `{scale_str}` | {minfo['instances_count']} | {minfo['name']} |"
            )
    lines.append("\n> **高度经验值备忘**：")
    lines.append("> - `desk_brown`（办公桌）表面高度约 **0.72m**，放置在其上的物品（如 `monitor_1`, `monitor_2`）设为 `z=0.72`；黄色目标杯 `cup_yellow` 设为 `z=0.75`。")
    lines.append("> - `table_conference_2`（研讨长桌）表面高度约 **0.75m**。")
    lines.append("> - 地面摆放物品（`fire_hydrant`, `bookshelf`, `file_cabinet_large` 等）一律设为 `z=0.00`。\n")

    lines.append("---\n")

    # 4. ROS Packages
    lines.append("## 4. ROS 核心工作区与包架构\n")
    lines.append("| Package 名称 | 相对路径 | 启动文件 (Launch) | 脚本节点 (Nodes) | 功能描述 |")
    lines.append("|---|---|:---:|:---:|---|")
    for pname, pinfo in sorted(pkgs_info.items()):
        launches = f"{len(pinfo['launch_files'])} 个" if pinfo["launch_files"] else "—"
        scripts = f"{len(pinfo['scripts'])} 个" if pinfo["scripts"] else "—"
        desc = pinfo["description"][:50] + "..." if len(pinfo["description"]) > 50 else pinfo["description"]
        lines.append(f"| `{pname}` | `{pinfo['dir']}` | {launches} | {scripts} | {desc} |")
    lines.append("\n")

    # 5. Topics
    lines.append("## 5. 核心 ROS 话题契约 (AST 提取)\n")
    lines.append("### 5.1 发布话题 (Publishers)")
    lines.append("| 话题名称 | 发布节点源码文件 |")
    lines.append("|---|---|")
    for topic, files in sorted(topics_info["pubs"].items()):
        f_list = ", ".join(f"`{f}`" for f in sorted(set(files))[:4])
        lines.append(f"| `{topic}` | {f_list} |")
    lines.append("\n### 5.2 订阅话题 (Subscribers)")
    lines.append("| 话题名称 | 订阅节点源码文件 |")
    lines.append("|---|---|")
    for topic, files in sorted(topics_info["subs"].items()):
        f_list = ", ".join(f"`{f}`" for f in sorted(set(files))[:4])
        lines.append(f"| `{topic}` | {f_list} |")
    lines.append("\n")

    return "\n".join(lines)

def build_compact_agent_table(models_catalog, worlds_info) -> str:
    """Build concise table to inject directly into AGENTS.md."""
    lines = []
    lines.append("<!-- FAST_INDEX_START -->")
    lines.append("## Fast Navigation & Asset Directory")
    lines.append("- **Canonical Worlds & Generators**:")
    lines.append("  - T-Junction Office (Primary Benchmark): Source `scripts/tests/generate_t_junction_world.py` -> outputs `worlds/t_junction_office_no_pro3.world`, manifest `worlds/t_junction_office_manifest.yaml`.")
    lines.append("  - Complex Office (Large Benchmark): Source `scripts/tests/office_building/generate_world.py` -> `worlds/benchmark/office_building_v1_level_*.world`.")
    lines.append("  - Always modify the generator script first, regenerate, and run `gz sdf -k <world_file>` to validate.")
    lines.append("- **Gazebo Model Placement Rules**:")
    lines.append("  - `cup_yellow`: Target yellow cup, desk $Z=0.75$, scale=(1.5, 1.5, 1.5).")
    lines.append("  - `desk_brown`: Office desk, top surface $Z=0.72$, length 1.6m, width 0.8m.")
    lines.append("  - `monitor_1`, `monitor_2`: Computer monitors, base $Z=0.72$, screen flanking offset $\\pm 0.5\\text{m}$.")
    lines.append("  - `table_conference_2`: Large conference table, top surface $Z=0.75$.")
    lines.append("  - `fire_hydrant`: Ground safety item $Z=0.0$, must tuck into wall corners (墙脚).")
    lines.append("  - `target_room_occluder`: Screen divider box (1.6 x 0.25 x 1.6m), $Z=0.8$.")
    lines.append("- **Core Runtime Commands**:")
    lines.append("  - Stack restart: `./run_demo_teb.sh restart` (autonomous TEB) or `CONTROLLER_MODE=teleop ./run_demo_teb.sh restart`.")
    lines.append("  - Switch controller: `rostopic pub -1 /lste/controller_mode std_msgs/String \"data: 'teb'\"` (or `'teleop'`).")
    lines.append("  - Diagnostics: `python3 .agents/skills/lste-status-debug/scripts/inspect_demo_status.py`.")
    lines.append("- **Detailed Full Index**: Refer to `docs/" + TODAY + "_project_fast_index.md`.")
    lines.append("<!-- FAST_INDEX_END -->")
    return "\n".join(lines)

def update_agents_md(compact_table: str):
    """Update AGENTS.md with the compact table."""
    agents_md = WS_ROOT / "AGENTS.md"
    if not agents_md.exists():
        return False
    content = agents_md.read_text(encoding="utf-8")
    start_tag = "<!-- FAST_INDEX_START -->"
    end_tag = "<!-- FAST_INDEX_END -->"

    if start_tag in content and end_tag in content:
        pattern = re.compile(rf"{re.escape(start_tag)}.*?{re.escape(end_tag)}", re.DOTALL)
        new_content = pattern.sub(lambda _: compact_table, content)
    else:
        new_content = content.rstrip() + "\n\n" + compact_table + "\n"

    agents_md.write_text(new_content, encoding="utf-8")
    return True

def update_docs_introduction(doc_filename: str):
    """Add link to docs/introduction.md under operations."""
    intro_file = WS_ROOT / "docs" / "introduction.md"
    if not intro_file.exists():
        return False
    content = intro_file.read_text(encoding="utf-8")
    entry = f"| [{doc_filename}]({doc_filename}) | 全局极速索引：Gazebo模型字典、世界生成器与ROS数据契约 |"
    if doc_filename in content:
        return True
    
    # insert under operations table
    op_header = "## operations"
    if op_header in content:
        idx = content.find(op_header)
        table_end = content.find("\n## ", idx + len(op_header))
        if table_end != -1:
            new_content = content[:table_end].rstrip() + f"\n{entry}\n\n" + content[table_end:]
        else:
            new_content = content.rstrip() + f"\n{entry}\n"
        intro_file.write_text(new_content, encoding="utf-8")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="LSTE Project Fast Index Scanner")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing files")
    parser.add_argument("--no-agents-md", action="store_true", help="Skip updating AGENTS.md")
    args = parser.parse_args()

    print("[1/5] Scanning Gazebo models & empirical usages...")
    models = scan_gazebo_models()
    print(f"      Found {len(models)} workspace-referenced models.")

    print("[2/5] Scanning simulation worlds & generator scripts...")
    worlds = scan_world_files()
    print(f"      Found {len(worlds)} world files.")

    print("[3/5] Scanning ROS packages & AST topics...")
    pkgs = scan_ros_packages()
    topics = scan_ros_topics()
    print(f"      Found {len(pkgs)} packages, {len(topics['pubs'])} published topics, {len(topics['subs'])} subscribed topics.")

    doc_filename = f"{TODAY}_project_fast_index.md"
    doc_path = WS_ROOT / "docs" / doc_filename
    md_content = generate_markdown(models, worlds, pkgs, topics)

    if args.dry_run:
        print(f"\n[Dry Run] Would write {len(md_content.splitlines())} lines to {doc_path}")
        return

    print(f"[4/5] Writing full index to {doc_path}...")
    doc_path.write_text(md_content, encoding="utf-8")
    update_docs_introduction(doc_filename)
    print(f"      Updated docs/introduction.md with index entry.")

    if not args.no_agents_md:
        print("[5/5] Updating AGENTS.md with Fast Navigation table...")
        compact = build_compact_agent_table(models, worlds)
        update_agents_md(compact)
        print("      AGENTS.md successfully updated.")

    print(f"\n[Success] Project index generation completed in < 1s!")
    print(f"Index document: {doc_path}")

if __name__ == "__main__":
    main()
