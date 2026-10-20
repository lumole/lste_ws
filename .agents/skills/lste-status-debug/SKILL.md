---
name: lste-status-debug
description: >-
  Diagnose, monitor, and debug running LSTE navigation and topological demo status in the lste_ws workspace.
  Use this skill whenever the user or agent needs to inspect the current navigation target (WorkItem ID and Goal coordinates),
  controller execution mode (TEB autonomous vs teleop manual), topological Place progress, controller lease locks, or failure episodes
  without performing arbitrary codebase searches.
---

# LSTE Demo Status Diagnostics & Debugging Guide

This skill standardizes the path, metrics, and methodology for inspecting and debugging the LSTE topological navigation stack (specifically the two-room `no_pro3` demo and benchmark navigation runs).

---

## 1. Quick Diagnostic Execution

Instead of grepping or manual JSON parsing, run the dedicated inspection script:

```bash
python3 .agents/skills/lste-status-debug/scripts/inspect_demo_status.py
```

*To inspect a specific historical run directory:*
```bash
python3 .agents/skills/lste-status-debug/scripts/inspect_demo_status.py --run-dir runtime/manual_teleop/<YYYYMMDD_HHMMSS>
```

This script automatically outputs:
- **Active Target**: Target WorkItem (e.g. `W7`), map goal coordinates `[x, y]`, and attempt status.
- **Controller Mode**: `TEB` (autonomous driving) vs `TELEOP` (manual).
- **Place Progress**: Current Place (e.g. `P1`), phase (`observe` vs `cross`), and unresolved work items remaining.
- **WorkItem Ledger**: Full list of resolved vs unresolved WorkItem IDs.
- **Anomaly Detection**: Flags repeating `stale_lease_release_ack` (lease deadlocks) or `planner_recovery_observed` (obstacle blockages).

---

## 2. Status Data Sources & Locations

All runtime telemetry is recorded in dedicated timestamped directories under `runtime/manual_teleop/` (or `runtime/rl_fixed_goal_test/logs/`):

| File / Topic | Format | Primary Responsibility |
|---|---|---|
| `runtime/manual_teleop/<TIMESTAMP>/status.jsonl` | JSON Lines | Frontier executive status: WorkItems, Places, Portals, goals, events |
| `runtime/manual_teleop/<TIMESTAMP>/bridge_status.jsonl` | JSON Lines | TEB goal bridge: controller mode (`teb`/`teleop`), leases, watchdogs |
| `/lste/global_frontier/status` | ROS JSON string | Live stream identical to `status.jsonl` payload |
| `/lste/teb_goal_bridge/status` | ROS JSON string | Live stream identical to `bridge_status.jsonl` payload |
| `/lste/controller_mode` | `std_msgs/String` | Active controller owner: `'teb'` or `'teleop'` |

---

## 3. Core Status Schema & Interpretation

Every record in `status.jsonl` contains:
```json
{
  "wall": "2026-10-19T17:05:10",
  "sim": 2220.425,
  "event": "work_item_dispatched",
  "payload": { ... }
}
```

### Key Payload Fields to Check:

1. **Who is the current target?**
   - `payload.work_item_id`: Integer ID of the active WorkItem (e.g., `7` -> `W7`).
   - `payload.goal`: `[x, y]` navigation waypoint sent to the planner.
   - `payload.work_items[]`: Array of WorkItems. Find item with matching `id`:
     - `anchor_xy`: Physical centroid/boundary of the observation target.
     - `state`: `"unresolved"` -> `"resolved"`.
     - `attempts[]`: History of viewpoints tried, including `state`: `"active"`, `"succeeded"`, or `"failed"`.

2. **Is TEB driving or waiting?**
   - Check `bridge_status.jsonl`'s `payload.mode`:
     - `"teb"`: Executive owns `/cmd_vel` via TEB planner.
     - `"teleop"`: Mux is ignoring TEB commands; keyboard teleop owns base.
   - Switch live controller mode:
     ```bash
     rostopic pub -1 /lste/controller_mode std_msgs/String "data: 'teb'"
     ```

3. **Topological Space & Progress**:
   - `payload.place_progress`:
     - `place_id`: Current place (e.g. `1` -> `P1`).
     - `phase`: `"observe"` (clearing local WorkItems) or `"cross"` (traversing DirectedPortal).
     - `unresolved_work_items`: Count of items remaining in this place.

---

## 4. Debugging Common Failure Modes

### Case A: Repeating `stale_lease_release_ack` (Controller Stall)
- **Symptom**: `status.jsonl` is flooded with `stale_lease_release_ack` events, robot stops moving, no new goals dispatched.
- **Cause**: The frontier executive and TEB goal bridge had a desynchronized lease handshake (e.g. action terminal was received but release was not acknowledged).
- **Resolution**:
  - Check `bridge_status.jsonl` for `pending_controller_lease_release_watchdog_armed`.
  - Trigger controller mode toggle (`teleop` -> `teb`) or restart stack with `./run_demo_teb.sh`.

### Case B: Repeating `planner_recovery_observed`
- **Symptom**: Viewpoint attempts fail with reason `endpoint_reached_support_unobserved` or Navfn path fails.
- **Cause**: Goal location is inside local costmap inflation or doorway clearance is too narrow.
- **Resolution**:
  - Check clearance parameter (default `0.52`, test rooms often require `clearance:=0.35`).

---

## 5. Reference Scripts in Workspace

- [inspect_demo_status.py](./scripts/inspect_demo_status.py): Primary diagnostic command.
- [watch_topology.py](file:///home/dh/dh_ws/lste_ws/temp/manual_teleop/watch_topology.py): Terminal text watcher for live topological graph.
- [run_demo_teb.sh](file:///home/dh/dh_ws/lste_ws/run_demo_teb.sh): One-click TEB launcher.
