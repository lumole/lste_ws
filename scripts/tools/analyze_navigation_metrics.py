#!/usr/bin/env python3
"""Summarize one or more structured navigation metric logs.

The live metrics node writes one JSON object after ``data=`` on every log
line.  This tool deliberately consumes those logs instead of scraping tmux or
ROS console output, so comparisons remain reproducible after a run ends.
"""

import argparse
import datetime
import json
import math
from pathlib import Path


def latest_log(root):
    candidates = sorted(root.glob("*/*_navigation_metrics.log"))
    return candidates[-1] if candidates else None


def load_records(path):
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            marker = " data="
            if marker not in line:
                continue
            try:
                record = json.loads(line.split(marker, 1)[1])
            except (TypeError, ValueError):
                continue
            event = line.split(" event=", 1)[1].split(" data=", 1)[0] if " event=" in line else ""
            record["_event"] = event
            record["_line"] = line.rstrip()
            records.append(record)
    return records


def finite_min(values):
    values = [float(value) for value in values if value is not None]
    values = [value for value in values if math.isfinite(value)]
    return min(values) if values else None


def summarize(path):
    records = load_records(path)
    events = {}
    samples = []
    goal_changes = []
    bridge_events = []
    for record in records:
        event = record.get("_event", "")
        events[event] = events.get(event, 0) + 1
        if event == "sample":
            samples.append(record)
        elif event == "goal_change":
            goal_changes.append(record)
        elif event == "teb_bridge_event":
            bridge_events.append(record)

    last = samples[-1] if samples else {}
    first = samples[0] if samples else {}
    clearances = [sample.get("min_scan_clearance") for sample in samples]
    minimum_clearance = finite_min(clearances)
    duration = None
    if first.get("ros_time") is not None and last.get("ros_time") is not None:
        duration = max(0.0, float(last["ros_time"]) - float(first["ros_time"]))
    source_counts = {}
    for record in goal_changes:
        source = str(record.get("goal_source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1

    replacement_counts = {}
    for record in bridge_events:
        if not record.get("replacement", False):
            continue
        kind = str(record.get("replacement_kind", "unknown"))
        replacement_counts[kind] = replacement_counts.get(kind, 0) + 1

    dispatch_reasons = {}
    for record in bridge_events:
        if record.get("bridge_event") == "dispatch":
            reason = str(record.get("reason", "unknown"))
            dispatch_reasons[reason] = dispatch_reasons.get(reason, 0) + 1
    retry_dispatches = int(dispatch_reasons.get("retry_move_base_goal", 0))
    path_length = last.get("path_length") or 0.0
    forward_distance = last.get("forward_distance_m") or 0.0
    preemptions = last.get("move_base_preemptions") or 0
    dispatches = last.get("move_base_dispatches") or 0
    brakes = last.get("linear_brake_events") or 0
    brake_clear = last.get("brake_events_clear") or 0
    brake_near = last.get("brake_events_near") or 0

    result = {
        "log": str(path),
        "run_id": path.parent.name,
        "records": len(records),
        "samples": len(samples),
        "duration_seconds": None if duration is None else round(duration, 3),
        "goal_first_seen_seconds": last.get("target_first_seen_seconds"),
        "target_lock_seconds": last.get("target_lock_seconds"),
        "task_done_seconds": last.get("task_done_seconds"),
        "goal_changes": len(goal_changes),
        "goal_delta_max_m": last.get("goal_delta_max"),
        "goal_delta_mean_m": last.get("goal_delta_mean"),
        "goal_change_sources": source_counts,
        "move_base_dispatches": last.get("move_base_dispatches"),
        "move_base_preemptions": last.get("move_base_preemptions"),
        "move_base_successes": last.get("move_base_successes"),
        "move_base_aborts": last.get("move_base_aborts"),
        "bridge_dispatches": last.get("teb_bridge_dispatches"),
        "bridge_terminals": last.get("teb_bridge_terminal_events"),
        "bridge_deferred_updates": last.get("teb_bridge_deferred_goal_updates"),
        "bridge_goal_replacements": last.get("teb_bridge_goal_replacements"),
        "bridge_priority_goal_replacements": last.get(
            "teb_bridge_priority_goal_replacements"
        ),
        "bridge_target_goal_replacements": last.get(
            "teb_bridge_target_goal_replacements"
        ),
        "bridge_frontier_goal_replacements": last.get(
            "teb_bridge_frontier_segment_handoffs"
        ),
        "bridge_frontier_sharp_replacements": last.get(
            "teb_bridge_frontier_sharp_replacements"
        ),
        "bridge_replacement_kinds": replacement_counts,
        "target_route_accepts": last.get("target_route_accepts"),
        "target_route_rejections": last.get("target_route_rejections"),
        "target_route_deferrals": last.get("target_route_deferrals"),
        "target_route_holds": last.get("target_route_holds"),
        "target_route_failures": last.get("target_route_failures"),
        "target_route_releases": last.get("target_route_releases"),
        "linear_brake_events": last.get("linear_brake_events"),
        "linear_brake_rate_per_minute": last.get("linear_brake_rate_per_minute"),
        "retry_dispatches": retry_dispatches,
        "dispatch_reasons": dispatch_reasons,
        "preemption_ratio": round(preemptions / max(1, dispatches), 3),
        "forward_distance_m": last.get("forward_distance_m"),
        "forward_angular_energy_per_m": last.get("forward_angular_energy_per_m"),
        "brake_events_clear": brake_clear,
        "brake_events_near": brake_near,
        "brake_events_clear_fraction": (
            None if (brake_clear + brake_near) == 0
            else round(brake_clear / (brake_clear + brake_near), 3)
        ),
        "brakes_per_m": round(brakes / max(0.01, path_length), 4),
        "brakes_per_m_forward": round(brakes / max(0.01, forward_distance), 4),
        "teb_linear_brake_events": last.get("teb_linear_brake_events"),
        "stop_events": last.get("stop_events"),
        "stop_rate_per_minute": last.get("stop_rate_per_minute"),
        "average_stop_duration_seconds": last.get("average_stop_duration_seconds"),
        "max_stop_duration_seconds": last.get("max_stop_duration_seconds"),
        "goal_change_rate_per_minute": last.get("goal_change_rate_per_minute"),
        "angular_sign_flips": last.get("angular_sign_flips"),
        "strong_angular_sign_flips": last.get("strong_angular_sign_flips"),
        "teb_angular_sign_flips": last.get("teb_angular_sign_flips"),
        "teb_strong_angular_sign_flips": last.get("teb_strong_angular_sign_flips"),
        "min_scan_clearance_m": None if minimum_clearance is None else round(minimum_clearance, 4),
        "path_length_m": last.get("path_length"),
        "controller_mode": last.get("controller_mode"),
        "goal_source": last.get("goal_source"),
        "task_done": last.get("task_done"),
        "bridge_event_counts": {
            str(record.get("bridge_event", "unknown")): sum(
                1 for item in bridge_events
                if item.get("bridge_event", "unknown") == record.get("bridge_event", "unknown")
            )
            for record in bridge_events
        },
        "log_event_counts": events,
    }
    return result


def print_human(result):
    print("Navigation run: %s" % result["run_id"])
    print("  log: %s" % result["log"])
    print("  duration: %s s, samples: %s" % (result["duration_seconds"], result["samples"]))
    print("  target: first_seen=%s s lock=%s s done=%s s" % (
        result["goal_first_seen_seconds"],
        result["target_lock_seconds"],
        result["task_done_seconds"],
    ))
    print("  goals: changes=%s max_delta=%s m mean_delta=%s m" % (
        result["goal_changes"], result["goal_delta_max_m"], result["goal_delta_mean_m"],
    ))
    print("  actions: dispatch=%s success=%s preempt=%s abort=%s bridge_deferred=%s" % (
        result["move_base_dispatches"], result["move_base_successes"],
        result["move_base_preemptions"], result["move_base_aborts"],
        result["bridge_deferred_updates"],
    ))
    print("  replacements: total=%s priority=%s frontier_segment=%s target_segment=%s kinds=%s" % (
        result["bridge_goal_replacements"],
        result["bridge_priority_goal_replacements"],
        result["bridge_frontier_goal_replacements"],
        result["bridge_target_goal_replacements"],
        json.dumps(result["bridge_replacement_kinds"], sort_keys=True),
    ))
    print("  target route: accepted=%s rejected=%s deferred=%s held=%s failures=%s releases=%s" % (
        result["target_route_accepts"],
        result["target_route_rejections"],
        result["target_route_deferrals"],
        result["target_route_holds"],
        result["target_route_failures"],
        result["target_route_releases"],
    ))
    print("  control: brakes=%s (%.2f/min) teb_brakes=%s stops=%s (%.2f/min) "
          "avg_stop=%.3fs max_stop=%.3fs angular_flips=%s strong_flips=%s" % (
        result["linear_brake_events"],
        result["linear_brake_rate_per_minute"] or 0.0,
        result["teb_linear_brake_events"],
        result["stop_events"], result["stop_rate_per_minute"] or 0.0,
        result["average_stop_duration_seconds"] or 0.0,
        result["max_stop_duration_seconds"] or 0.0,
        result["angular_sign_flips"],
        result["strong_angular_sign_flips"],
    ))
    print("  smoothness: forward=%.1fm steer_energy=%.3f rad/m flips/m=%.3f "
          "brakes/m=%.3f clear_brake_frac=%.3f" % (
        result["forward_distance_m"] or 0.0,
        result["forward_angular_energy_per_m"] or 0.0,
        (result["angular_sign_flips"] or 0) / max(0.01, result["forward_distance_m"] or 0.0),
        result["brakes_per_m"] or 0.0,
        result["brake_events_clear_fraction"] or 0.0,
    ))
    print("  goal churn: retries=%s preemption_ratio=%.2f reasons=%s" % (
        result["retry_dispatches"],
        result["preemption_ratio"] or 0.0,
        json.dumps(result["dispatch_reasons"], sort_keys=True),
    ))
    print("  safety: min_clearance=%s m path=%s m mode=%s source=%s done=%s" % (
        result["min_scan_clearance_m"], result["path_length_m"],
        result["controller_mode"], result["goal_source"], result["task_done"],
    ))
    print("  bridge events: %s" % json.dumps(result["bridge_event_counts"], sort_keys=True))


COMPARE_FIELDS = [
    ("duration", "duration_seconds"),
    ("goal changes", "goal_changes"),
    ("goal churn/min", "goal_change_rate_per_minute"),
    ("dispatches", "move_base_dispatches"),
    ("preemptions", "move_base_preemptions"),
    ("preemption ratio", "preemption_ratio"),
    ("retries", "retry_dispatches"),
    ("successes", "move_base_successes"),
    ("brakes", "linear_brake_events"),
    ("brakes/m", "brakes_per_m"),
    ("clear-brake frac", "brake_events_clear_fraction"),
    ("stops", "stop_events"),
    ("avg stop (s)", "average_stop_duration_seconds"),
    ("max stop (s)", "max_stop_duration_seconds"),
    ("angular flips", "angular_sign_flips"),
    ("steer energy rad/m", "forward_angular_energy_per_m"),
    ("min clearance (m)", "min_scan_clearance_m"),
    ("path (m)", "path_length_m"),
    ("task done (s)", "task_done_seconds"),
]


def _resolve_log(path):
    path = Path(path)
    if path.is_dir():
        matches = sorted(path.glob("*_navigation_metrics.log"))
        return matches[-1] if matches else None
    return path if path.is_file() else None


def compare_runs(base_a, base_b):
    base_a = _resolve_log(base_a)
    base_b = _resolve_log(base_b)
    if base_a is None or base_b is None:
        raise SystemExit("compare: could not find a navigation metrics log in both paths")
    result_a = summarize(base_a)
    result_b = summarize(base_b)
    print("A/B navigation smoothness comparison")
    print("  A: %s (id=%s)" % (result_a["log"], result_a["run_id"]))
    print("  B: %s (id=%s)" % (result_b["log"], result_b["run_id"]))
    print("%-20s %12s %12s %10s" % ("metric", "A", "B", "delta"))
    print("-" * 58)
    for label, key in COMPARE_FIELDS:
        value_a = result_a.get(key)
        value_b = result_b.get(key)
        text_a = "-" if value_a is None else str(value_a)
        text_b = "-" if value_b is None else str(value_b)
        delta = ""
        try:
            delta = "%.3g" % (float(value_b) - float(value_a))
        except (TypeError, ValueError):
            delta = ""
        print("%-20s %12s %12s %10s" % (label, text_a, text_b, delta))


def diagnose(path):
    """Print the concrete event sequences behind the smoothness metrics.

    Rather than dumping every record, extract the three problem patterns the
    navigation task cares about: goal churn, straight-line steering wobble, and
    obstacle/safety braking. Each printed line keeps enough context (clearance,
    speed, goal distance) to distinguish a legitimate obstacle response from a
    system-side jitter stop or a goal that moved on its own.
    """
    records = load_records(path)
    print("Diagnostic trace: %s (%d records)" % (path, len(records)))
    print("-" * 78)
    for record in records:
        event = record.get("_event", "")
        if event == "goal_change":
            print("GOAL   t=%.1f delta=%.2fm goal=%s prev=%s source=%s cmd=(%.2f,%.2f)" % (
                record.get("ros_time") or 0.0,
                record.get("delta_m") or 0.0,
                record.get("goal"),
                record.get("previous_goal"),
                record.get("goal_source"),
                (record.get("command") or [0, 0])[0],
                (record.get("command") or [0, 0])[1],
            ))
        elif event == "linear_brake":
            clear = record.get("scan_forward_min")
            print("BRAKE  t=%.1f %.2f->%.2f m/s w=%.2f fwd_clear=%s goal_dist=%s src=%s/%s" % (
                record.get("ros_time") or 0.0,
                record.get("previous_linear") or 0.0,
                record.get("current_linear") or 0.0,
                record.get("angular") or 0.0,
                clear,
                record.get("robot_goal_distance") if "robot_goal_distance" in record else record.get("goal"),
                record.get("controller_source"),
                record.get("controller_reason"),
            ))
        elif event == "command_stop":
            print("STOP   t=%.1f mode=%s src=%s teb_status=%s mb=%s" % (
                record.get("ros_time") or 0.0,
                record.get("controller_mode"),
                record.get("controller_source"),
                record.get("teb_status"),
                record.get("move_base_status"),
            ))
        elif event == "teb_raw_angular_sign_flip":
            print("FLIP   t=%.1f %.3f->%.3f rad/s clear=%.2f status=%s" % (
                record.get("ros_time") or 0.0,
                record.get("previous_angular") or 0.0,
                record.get("angular") or 0.0,
                record.get("scan_forward_min") or float("nan"),
                record.get("teb_status"),
            ))
        elif event == "teb_bridge_event" and record.get("bridge_event") == "dispatch":
            print("DISP   t=%.1f %s goal=%s kind=%s reason=%s" % (
                record.get("ros_time") or 0.0,
                record.get("active_intent_source"),
                record.get("dispatched_goal"),
                record.get("route_kind"),
                record.get("reason"),
            ))
        elif event == "teb_bridge_event" and record.get("bridge_event") == "handoff_requested":
            print("HANDOFF t=%.1f reason=%s pending_delta=%s" % (
                record.get("ros_time") or 0.0,
                record.get("reason"),
                record.get("pending_delta"),
            ))
        elif event == "move_base_recovery":
            print("RECOVER t=%.1f %s" % (record.get("ros_time") or 0.0, record.get("behavior")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, help="metrics log or run directory")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runtime/navigation/logs"),
        help="navigation log root when no log path is supplied",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="print the raw goal/brake/flip/dispatch event trace for a log",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("A", "B"),
        help="print an A/B comparison of two metrics log paths/directories",
    )
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
        return

    path = args.log
    if path is None:
        path = latest_log(args.root)
    elif path.is_dir():
        matches = sorted(path.glob("*_navigation_metrics.log"))
        path = matches[-1] if matches else None
    if path is None or not path.is_file():
        parser.error("navigation metrics log not found")
    if args.diagnose:
        diagnose(path)
        return
    result = summarize(path)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)


if __name__ == "__main__":
    main()
