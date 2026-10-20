---
name: lste-project-index
description: >-
  Fast project navigation, Gazebo model & asset lookup, simulation environment & benchmark comparison,
  and automated index synchronization for the lste_ws workspace. Use this skill whenever
  the user or agent needs to compare simulation environments (e.g. T-junction office benchmark vs env04 vs office_building_v1),
  look up Gazebo model dimensions/heights/scales, resolve world generator sources of truth,
  or re-scan and synchronize the project index without performing arbitrary codebase searches.
---

# LSTE Project Index & Environment Comparison Skill

This skill standardizes rapid project navigation, automated index synchronization, and multi-environment benchmarking across the `lste_ws` workspace.

---

## 1. Fast Diagnostic & Comparison Commands

Run these specialized scripts instead of manually searching files:

### 1.1 Compare Simulation Environments
To compare active benchmark worlds (dimensions, model counts, yellow cup poses, occluders, generators):
```bash
python3 .agents/skills/lste-project-index/scripts/compare_environments.py
```
*(Pass `--json` for machine-readable JSON output)*

### 1.2 Re-scan and Sync Full Project Index
Whenever models, ROS packages, topics, or world files are added or modified:
```bash
python3 scripts/tools/generate_project_index.py
```
- Scans `~/.gazebo/models` and AST topics in `< 1s`.
- Updates `docs/YYYYMMDD_project_fast_index.md`.
- Synchronizes the `AGENTS.md` fast reference block.

---

## 2. Environment Comparison Matrix

| Environment | Role & Scope | Bounds / Area | Target (cup_yellow) | Key Characteristics & Generators |
|---|---|---|---|---|
| **T-Junction Office** (`worlds/t_junction_office_no_pro3.world`) | **Primary Active Benchmark** | $12.0\text{m} \times 10.5\text{m}$ ($126\text{m}^2$) | `(6.60, 8.50, z=0.75)` flanked by 2 monitors | Generator: `scripts/tests/generate_t_junction_world.py`. Manifest: `worlds/t_junction_office_manifest.yaml`. 2 rooms, occluder screen, wall-corner hydrants, **zero chairs** (clean mobility). |
| **env04** (`worlds/topo3.0_catch_mode/session_test/env04_no_pro3.world`) | Legacy Simplicity Baseline | $15.0\text{m} \times 15.0\text{m}$ ($225\text{m}^2$) | `(20.0, -3.0)` | Legacy session test environment; static world. |
| **Office Building V1 L1** (`worlds/benchmark/office_building_v1_level_1_no_pro3.world`) | Multi-Room Benchmark (Level 1) | $25.0\text{m} \times 22.0\text{m}$ ($550\text{m}^2$) | `(21.6, 17.6, z=1.02)` | Generator: `scripts/tests/office_building/generate_world.py`. Manifest: `office_building_v1_manifest.yaml`. 4 target desks. |
| **Office Building V1 L2-L4** (`worlds/benchmark/office_building_v1_level_2..4_no_pro3.world`) | High-Clutter Benchmark | $25.0\text{m} \times 22.0\text{m}$ ($550\text{m}^2$) | `(21.6, 17.6, z=1.02)` | 40+ furniture models, chairs, cabinets, occluders. Used for stress-testing topological exploration. |

---

## 3. Gazebo Model Catalog & Empirical Poses

Always reference these standard coordinates and surface heights:

| Model URI | Typical $Z$ | Scale | Canonical Purpose / Notes |
|---|---|---|---|
| `model://cup_yellow` | `0.75` (or `1.02`) | `(1.5, 1.5, 1.5)` | **Primary Task Target**. Must be placed on tabletop ($Z=0.75$). |
| `model://desk_brown` | `0.00` | `1 1 1` | Office desk. **Top surface is at $Z=0.72\text{m}$**, length $1.6\text{m}$, width $0.8\text{m}$. |
| `model://monitor_1` / `monitor_2` | `0.72` | `1 1 1` | Desktop computer monitors. Base at $Z=0.72\text{m}$, flanking cup $\pm 0.5\text{m}$. |
| `model://table_conference_2` | `0.00` | `1 1 1` | Conference table. **Top surface is at $Z=0.75\text{m}$**, length $2.4\text{m}$. |
| `model://fire_hydrant` | `0.00` | `1 1 1` | Floor fixture. **Must be placed in corners (墙脚)**, never blocking aisles. |
| `model://bookshelf` | `0.00` | `1 1 1` | Wall-adjacent storage. Height $2.0\text{m}$. |
| `model://file_cabinet_large` | `0.00` | `1 1 1` | Wall-adjacent filing cabinet. |
| `target_room_occluder` | `0.80` | `(1.6, 0.25, 1.6)` | Screen divider box ($Z_{\text{center}}=0.8\text{m}$ for bottom at floor level). |

---

## 4. World Modification Golden Rules

1. **Never edit generated `.world` files directly.**
   - If modifying T-Junction Office: edit `scripts/tests/generate_t_junction_world.py`, run `python3 scripts/tests/generate_t_junction_world.py`.
   - If modifying Office Building: edit `scripts/tests/office_building/generate_world.py`.
2. **Always validate SDF syntax after modifying**:
   ```bash
   gz sdf -k <path_to_world_file>
   ```
3. **Restart the stack with correct spawn arguments**:
   ```bash
   WORLD="/home/dh/dh_ws/lste_ws/worlds/t_junction_office_no_pro3.world" \
   SPAWN_X=6.0 SPAWN_Y=1.0 SPAWN_YAW=1.5708 DISPLAY=:0 ./run_demo_teb.sh restart
   ```

---

## 5. Live Runtime & Status Telemetry

- **Switch controller mode**:
  - Autonomous TEB: `rostopic pub -1 /lste/controller_mode std_msgs/String "data: 'teb'"`
  - Manual Teleop: `rostopic pub -1 /lste/controller_mode std_msgs/String "data: 'teleop'"`
- **Inspect live demo execution**:
  - `python3 .agents/skills/lste-status-debug/scripts/inspect_demo_status.py`
