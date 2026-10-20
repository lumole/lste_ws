#!/usr/bin/env python3
"""Diagnostic script for inspecting the live or latest LSTE demo navigation status.

Parses status.jsonl and bridge_status.jsonl to extract:
- Active target (WorkItem ID, goal coordinates, attempt state)
- Controller mode (TEB vs teleop)
- Place & phase progress
- WorkItem resolution summary
- Anomalies (stale leases, planner recovery loops, stuck dispatches)
"""

import argparse
import glob
import json
import os
import sys


def find_latest_run_dir(workspace_dir):
    manual_root = os.path.join(workspace_dir, "runtime", "manual_teleop")
    dirs = [d for d in glob.glob(os.path.join(manual_root, "20*")) if os.path.isdir(d)]
    if not dirs:
        return None
    # Sort by modification time of directory or status.jsonl if present
    def run_time(d):
        sf = os.path.join(d, "status.jsonl")
        if os.path.isfile(sf) and os.path.getsize(sf) > 0:
            return os.path.getmtime(sf)
        return os.path.getmtime(d)

    dirs.sort(key=run_time)
    return dirs[-1]


def read_last_jsonl_records(file_path, n=50):
    if not os.path.isfile(file_path):
        return []
    records = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[-n:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    except Exception as e:
        print(f"[Warning] Failed to read {file_path}: {e}", file=sys.stderr)
    return records


def main():
    parser = argparse.ArgumentParser(description="Inspect LSTE Demo Status")
    parser.add_argument("--run-dir", default="", help="Path to run directory")
    args = parser.parse_args()

    ws = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    run_dir = args.run_dir or find_latest_run_dir(ws)

    if not run_dir or not os.path.isdir(run_dir):
        print(f"[Error] No active run directory found in {ws}/runtime/manual_teleop", file=sys.stderr)
        return 1

    status_file = os.path.join(run_dir, "status.jsonl")
    bridge_file = os.path.join(run_dir, "bridge_status.jsonl")

    status_records = read_last_jsonl_records(status_file, n=40)
    bridge_records = read_last_jsonl_records(bridge_file, n=20)

    print(f"=== LSTE Demo Diagnostic Report ===")
    print(f"Run Directory: {os.path.relpath(run_dir, ws)}")

    # 1. Controller & Bridge status
    bridge_mode = "unknown"
    active_bridge_goal = None
    turn_sup_state = None
    if bridge_records:
        last_b = bridge_records[-1]
        bp = last_b.get("payload", {})
        bridge_mode = bp.get("mode", "unknown")
        active_bridge_goal = bp.get("active_goal")
        turn_sup_state = bp.get("turn_supervisor_state")
        print(f"Controller Mode: {bridge_mode.upper()} (Bridge Event: {last_b.get('event')})")
        if turn_sup_state:
            print(f"Turn Supervisor State: {turn_sup_state}")
    else:
        print("Bridge Status: No records found")

    # 2. Frontier status & Target
    if not status_records:
        print("Frontier Status: No records found")
        return 0

    last_s = status_records[-1]
    sp = last_s.get("payload", {})
    sim_time = last_s.get("sim")
    latest_event = last_s.get("event")

    print(f"Sim Time: {sim_time} | Latest Frontier Event: {latest_event}")

    # Active target
    work_item_id = sp.get("work_item_id")
    goal = sp.get("goal") or active_bridge_goal
    route_id = sp.get("route_id")
    route_kind = sp.get("route_kind")

    print("\n--- Current Target & Navigation Goal ---")
    if work_item_id is not None:
        print(f"Target WorkItem: W{work_item_id}")
    else:
        print("Target WorkItem: None")

    if goal:
        print(f"Goal Coordinates [x, y]: {goal[:2]}")
    else:
        print("Goal Coordinates: None")
    print(f"Route: ID={route_id}, Kind={route_kind}")

    # Place progress
    pp = sp.get("place_progress", {})
    if pp:
        print(f"\n--- Place & Topological Progress ---")
        print(f"Place ID: P{pp.get('place_id')} | Phase: {pp.get('phase')} | Reason: {pp.get('reason')}")
        print(f"Remaining WorkItems in Place: {pp.get('unresolved_work_items')}")

    # Work items breakdown
    wis = sp.get("work_items", [])
    if wis:
        resolved = [w["id"] for w in wis if w.get("state") == "resolved"]
        unresolved = [w["id"] for w in wis if w.get("state") == "unresolved"]
        print(f"\n--- WorkItems Ledger (Total: {len(wis)}) ---")
        print(f"Resolved ({len(resolved)}): {resolved}")
        print(f"Unresolved ({len(unresolved)}): {unresolved}")

        # Details on active target work item
        if work_item_id is not None:
            active_wi = next((w for w in wis if w.get("id") == work_item_id), None)
            if active_wi:
                print(f"Active W{work_item_id} Anchor: {active_wi.get('anchor_xy')}")
                print(f"Active W{work_item_id} Normal: {active_wi.get('normal_xy')}")
                active_attempts = [a for a in active_wi.get("attempts", []) if a.get("state") == "active"]
                if active_attempts:
                    print(f"Active Attempt: ID={active_attempts[0].get('id')}, Goal={active_attempts[0].get('map_goal')}")

    # 3. Anomaly detection (check trailing events)
    recent_events = [r.get("event") for r in status_records if r.get("event")]
    stale_lease_count = recent_events[-20:].count("stale_lease_release_ack")
    recovery_count = recent_events[-20:].count("planner_recovery_observed")

    anomalies = []
    if stale_lease_count >= 5:
        anomalies.append(f"CRITICAL: Detected repeating stale_lease_release_ack ({stale_lease_count}/20 recent events) - Controller lease may be stalled!")
    if recovery_count >= 5:
        anomalies.append(f"WARNING: High frequency of planner_recovery_observed ({recovery_count}/20) - Navigation may be obstructed!")

    if anomalies:
        print("\n--- Diagnostics / Warnings ---")
        for a in anomalies:
            print(f"! {a}")
    else:
        print("\nHealth Status: Normal (no lease deadlock or recovery loops detected)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
