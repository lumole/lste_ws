#!/usr/bin/env python3
"""Summarize one LSTE navigation metrics log for the office benchmark."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LOG_ROOTS = (
    ROOT / "runtime/office_building_benchmark/logs",
    ROOT / "runtime/navigation/logs",
)
EVENT_RE = re.compile(r"event=(\S+) data=(\{.*\})$")


def latest_log() -> Path:
    logs = []
    for root in LOG_ROOTS:
        logs.extend(root.glob("*/*_navigation_metrics.log"))
    if not logs:
        raise SystemExit(
            "no navigation metrics logs found under %s"
            % ", ".join(str(root) for root in LOG_ROOTS)
        )
    return max(logs, key=lambda path: path.stat().st_mtime)


def summarize(path: Path) -> dict:
    events = Counter()
    samples = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = EVENT_RE.search(line.rstrip())
            if not match:
                continue
            event, payload = match.groups()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            events[event] += 1
            if event == "sample":
                samples.append(data)
    if not samples:
        raise SystemExit("metrics log has no sample event: %s" % path)
    final = samples[-1]
    completion = next((sample for sample in samples if sample.get("task_done")), None)
    reference = completion or final
    task_done_events = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = EVENT_RE.search(line.rstrip())
            if not match or match.group(1) != "task_done":
                continue
            try:
                task_done_events.append(json.loads(match.group(2)))
            except json.JSONDecodeError:
                continue
    completion_event = task_done_events[-1] if task_done_events else None
    return {
        "log": str(path),
        "run_timestamp": path.parent.name,
        "task_done": bool(completion or final.get("task_done")),
        "task_done_seconds": (completion_event or {}).get("ros_time") if completion else None,
        "ros_time": reference.get("ros_time"),
        "pose": reference.get("pose"),
        "goal": reference.get("goal"),
        "goal_source": reference.get("goal_source"),
        "path_length_m": reference.get("path_length"),
        "distance_to_goal_m": reference.get("distance_to_goal"),
        "map_coverage": reference.get("map_coverage"),
        "move_base_aborts": reference.get("move_base_aborts"),
        "move_base_preemptions": reference.get("move_base_preemptions"),
        "move_base_unexpected_preemptions": reference.get("move_base_unexpected_preemptions"),
        "turn_only_duration_seconds": reference.get("turn_only_duration_seconds"),
        "turn_only_events": reference.get("turn_only_events"),
        "stop_events": reference.get("stop_events"),
        "min_scan_clearance": reference.get("min_scan_clearance"),
        "goal_changes": reference.get("goal_changes"),
        "target_segments_committed": reference.get("target_segments_committed"),
        "target_route_rejections": reference.get("target_route_rejections"),
        "target_route_deferrals": reference.get("target_route_deferrals"),
        "completion_snapshot": bool(completion),
        "post_completion_samples": len(samples) - (samples.index(completion) + 1) if completion else 0,
        "target_event_counts": {
            name: count
            for name, count in sorted(events.items())
            if name.startswith("target_")
        },
        "event_counts": dict(sorted(events.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the JSON summary to this path",
    )
    args = parser.parse_args()
    path = args.log or latest_log()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise SystemExit("metrics log not found: %s" % path)
    rendered = json.dumps(summarize(path), ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
