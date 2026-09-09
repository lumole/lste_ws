#!/usr/bin/env python3
"""Run or plan independently restarted office-building method comparisons.

The default is deliberately read-only. ``--execute`` is the explicit opt-in
because a cold trial starts Gazebo and takes exclusive ownership of the LSTE
tmux sessions. Diagnostic warm mode keeps one compatible runtime alive across
bounded slices so startup is paid once rather than once per slice.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from experiment_protocol import ROOT, build_trials, load_matrix
from experiment_video import finish_capture, start_capture


TOOLS = ROOT / "scripts/tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from navigation_failure_report import build_report as build_failure_report  # noqa: E402
from navigation_failure_report import render_markdown as render_failure_report  # noqa: E402


RUNNER = ROOT / "scripts/tests/office_building/run_office_building.sh"
SUMMARY_SCRIPT = ROOT / "scripts/tests/office_building/summarize_run.py"
VERIFY_SCRIPT = ROOT / "scripts/tests/office_building/verify_topology_exploration.py"
CURRENT = ROOT / "runtime/office_building_benchmark/current"

_STARTUP_TAIL_BYTES = 131072
_STARTUP_ROUTE_EVENTS = frozenset((
    "route_selected",
    "route_command",
    "work_item_dispatched",
))
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_METRICS_EVENT_RE = re.compile(
    r"event=(?P<event>[A-Za-z0-9_]+) data=(?P<data>\{.*\})\s*$"
)
_DEFAULT_STARTUP_READINESS_TIMEOUT_SECONDS = 45.0
_TRIAL_END_SAMPLE_WINDOW = 16
_MAX_WARM_SLICE_SECONDS = 60.0


@dataclass
class WarmRuntime:
    """One live ROS/Gazebo runtime shared by bounded diagnostic slices."""

    run_dir: Path
    metrics_path: Path
    environment: dict
    startup_elapsed_wall_seconds: float
    startup_readiness: dict
    started_here: bool
    stop_at_end: bool


def trial_as_dict(trial):
    return {
        "study_id": trial.study_id,
        "phase": trial.phase,
        "method": trial.method,
        "level": trial.level,
        "trial_id": trial.trial_id,
        "profile": trial.profile,
        "controller": trial.controller,
        "startup_timeout_seconds": trial.startup_timeout_seconds,
        "startup_readiness_timeout_seconds": trial_startup_readiness_timeout(
            trial
        ),
        "timeout_seconds": trial.timeout_seconds,
        "seed": trial.seed,
        "terminal_event": trial.terminal_event,
        "task_json": trial.task_json,
        "task_id": trial.task_id,
        "target_eval_enabled": trial.target_eval_enabled,
        "video_required": trial.video_required,
        "video_window": trial.video_window,
        "video_fps": trial.video_fps,
        "video_override": getattr(trial, "video_override", "") or None,
    }


def current_metrics_path():
    run_dir = current_run_directory()
    if run_dir is None:
        return None
    candidates = sorted(run_dir.glob("*_navigation_metrics.log"))
    return candidates[0] if candidates else None


def trial_startup_readiness_timeout(trial):
    """Resolve the gate bound while accepting pre-gate trial-like objects."""
    value = getattr(
        trial,
        "startup_readiness_timeout_seconds",
        _DEFAULT_STARTUP_READINESS_TIMEOUT_SECONDS,
    )
    value = float(value)
    if value <= 0.0:
        raise ValueError("startup readiness timeout must be positive")
    return value


def apply_diagnostic_video_override(trials, disable_video=False):
    """Apply the explicit local no-video override without mutating the matrix."""
    trials = list(trials)
    if not disable_video:
        return trials
    return [
        replace(
            trial,
            video_required=False,
            video_override="disabled_by_cli",
        )
        for trial in trials
    ]


def current_run_directory():
    """Return the current run directory, or ``None`` for a broken link."""
    if not CURRENT.exists():
        return None
    try:
        run_dir = CURRENT.resolve(strict=True)
    except OSError:
        return None
    return run_dir if run_dir.is_dir() else None


def metrics_path_for_run(run_dir: Path):
    """Resolve the metrics file by run identity, never by global recency."""
    path = run_dir / (run_dir.name + "_navigation_metrics.log")
    return path if path.is_file() else None


def wait_for_metrics_log(run_dir: Path, timeout_seconds: float = 30.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        path = metrics_path_for_run(run_dir)
        if path is not None:
            return path
        time.sleep(0.25)
    return metrics_path_for_run(run_dir)


def _bounded_metrics_text(
    metrics_path: Path | None,
    start_offset: int = 0,
    limit: int = 262144,
):
    """Read a bounded log segment without mixing an earlier warm slice."""
    if metrics_path is None:
        return ""
    path = Path(metrics_path)
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            start = max(int(start_offset or 0), size - limit)
            stream.seek(start, os.SEEK_SET)
            return stream.read(limit).decode("utf-8", errors="replace")
    except (OSError, TypeError, ValueError):
        return ""


def terminal_event_observed(
    metrics_path: Path,
    terminal_event: str,
    start_offset: int = 0,
) -> bool:
    """Read only the current bounded slice of the metrics log."""
    tail = _bounded_metrics_text(metrics_path, start_offset=start_offset)
    if terminal_event == "task_done":
        return 'event=task_done ' in tail
    if terminal_event == "frontier_exhausted":
        return 'frontier_event":"frontier_exhausted"' in tail
    raise ValueError("unsupported terminal event: %s" % terminal_event)


def latest_failure_snapshot(metrics_path: Path, exclude_failure_ids=()):
    """Return the newest closed failure episode from the metrics tail.

    ``failure_snapshot_ready`` is emitted only after the configured pre/post
    evidence window has been serialized.  Reading the bounded tail keeps the
    diagnostic stop path cheap even when a metrics log is large.
    """
    excluded = {str(value) for value in (exclude_failure_ids or ())}
    tail = _tail_file(metrics_path, limit=262144)
    matches = re.findall(r"event=failure_snapshot_ready data=(\{.*\})$", tail, re.MULTILINE)
    for payload in reversed(matches):
        try:
            value = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            isinstance(value, dict)
            and str(value.get("failure_id", "")).strip()
            and str(value.get("failure_id")) not in excluded
        ):
            return value
    return None


def failure_snapshot_ids(metrics_path: Path | None):
    """Return closed failure IDs already present before a warm slice."""
    tail = _tail_file(metrics_path, limit=262144)
    result = set()
    matches = re.findall(
        r"event=failure_snapshot_ready data=(\{.*\})$",
        tail,
        re.MULTILINE,
    )
    for payload in matches:
        try:
            value = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        failure_id = value.get("failure_id") if isinstance(value, dict) else None
        if str(failure_id or "").strip():
            result.add(str(failure_id))
    return result


_TRIAL_END_SAMPLE_KEYS = (
    "ros_time", "wall_elapsed_seconds", "sample", "pose", "pose_frame",
    "goal", "goal_frame", "distance_to_goal", "goal_source",
    "goal_transaction_id", "teb_xy_goal_tolerance", "cmd", "cmd_vel",
    "teb_cmd", "teb_planner_cmd",
    "teb_status", "teb_feedback", "teb_turn_supervisor", "move_base_status",
    "move_base_feedback", "navfn_plan", "global_planner_plan", "teb_global_plan",
    "teb_local_plan", "scan_forward_min", "scan_min", "scan_left_min",
    "scan_right_min", "global_costmap", "local_costmap", "route_identity",
    "teb_bridge_active", "teb_bridge_last_event", "teb_bridge_latest_intent_source",
    "navigation_hold", "recovery", "state", "task_done",
)


def _compact_trial_end_sample(sample):
    """Keep one bounded, directly readable sample for a trial-end artifact."""
    if not isinstance(sample, dict):
        return None
    return {
        key: sample[key]
        for key in _TRIAL_END_SAMPLE_KEYS
        if key in sample
    }


def _trial_end_route_identity(sample):
    """Return a stable, small route identity from one metrics sample."""
    if not isinstance(sample, dict):
        return {}
    identity = sample.get("route_identity")
    identity = identity if isinstance(identity, dict) else {}
    route_id = (
        identity.get("active_route_id")
        or identity.get("route_id")
        or identity.get("released_route_id")
    )
    route_kind = (
        identity.get("active_route_kind")
        or identity.get("route_kind")
        or identity.get("released_route_kind")
    )
    result = {}
    if route_id not in (None, "", 0, "0"):
        result["route_id"] = route_id
    if route_kind not in (None, ""):
        result["route_kind"] = route_kind
    for key in ("goal", "first_portal_id", "obligation_id", "work_item_id"):
        if identity.get(key) is not None:
            result[key] = identity[key]
    transaction_id = sample.get("goal_transaction_id")
    if transaction_id is not None:
        result["goal_transaction_id"] = transaction_id
    return result


def _trial_end_progress_summary(samples):
    """Summarize progress and route churn in the bounded final sample window.

    This is a diagnostic reducer, not a navigation threshold.  It intentionally
    reports the raw first/last values so an investigator can distinguish a
    moving route from a stationary route without replaying a full ROS run.
    """
    samples = [item for item in samples if isinstance(item, dict)]
    if not samples:
        return {
            "sample_count": 0,
            "distance_to_goal_m": {},
            "pose_displacement_m": None,
            "active_route_sample_count": 0,
            "zero_command_sample_count": 0,
            "route_transition_count": 0,
            "route_transitions": [],
            "route_segments": [],
            "active_route": None,
            "interpretation": "insufficient_samples",
        }

    def segment_summary(segment):
        """Reduce one stable route/transaction segment independently."""
        distances = []
        poses = []
        zero_commands = 0
        for item in segment:
            try:
                distance = float(item.get("distance_to_goal"))
                if distance >= 0.0:
                    distances.append(distance)
            except (TypeError, ValueError):
                pass
            pose = item.get("pose")
            if isinstance(pose, (list, tuple)) and len(pose) >= 2:
                try:
                    poses.append((float(pose[0]), float(pose[1])))
                except (TypeError, ValueError):
                    pass
            command = item.get("cmd")
            if command is None:
                command = item.get("cmd_vel")
            try:
                if abs(float(command[0])) <= 0.05 and abs(float(command[1])) <= 0.03:
                    zero_commands += 1
            except (TypeError, ValueError, IndexError):
                pass
        distance_summary = {}
        if distances:
            first_distance = distances[0]
            last_distance = distances[-1]
            distance_summary = {
                "first": round(first_distance, 4),
                "last": round(last_distance, 4),
                "minimum": round(min(distances), 4),
                "maximum": round(max(distances), 4),
                "progress_toward_goal": round(first_distance - last_distance, 4),
            }
        displacement = None
        if len(poses) >= 2:
            displacement = round(
                ((poses[-1][0] - poses[0][0]) ** 2
                 + (poses[-1][1] - poses[0][1]) ** 2) ** 0.5,
                4,
            )
        progress = distance_summary.get("progress_toward_goal")
        if progress is not None and progress >= 0.05:
            interpretation = "moving_toward_goal"
        elif displacement is not None and displacement >= 0.05:
            interpretation = "moving_without_goal_progress"
        elif zero_commands == len(segment):
            interpretation = "stationary_active_route"
        else:
            interpretation = "insufficient_progress_evidence"
        identity = _trial_end_route_identity(segment[-1])
        return {
            "first_sample": segment[0].get("sample"),
            "last_sample": segment[-1].get("sample"),
            "sample_count": len(segment),
            "route_id": identity.get("route_id"),
            "route_kind": identity.get("route_kind"),
            "goal_transaction_id": identity.get("goal_transaction_id"),
            "goal": identity.get("goal") or segment[-1].get("goal"),
            "distance_to_goal_m": distance_summary,
            "pose_displacement_m": displacement,
            "zero_command_sample_count": zero_commands,
            "interpretation": interpretation,
        }

    distances = []
    poses = []
    zero_commands = 0
    active_route_samples = 0
    route_transitions = []
    previous_route_key = None
    route_segments = []
    current_segment = []
    for sample in samples:
        try:
            distance = float(sample.get("distance_to_goal"))
            if distance >= 0.0:
                distances.append(distance)
        except (TypeError, ValueError):
            pass
        pose = sample.get("pose")
        if isinstance(pose, (list, tuple)) and len(pose) >= 2:
            try:
                poses.append((float(pose[0]), float(pose[1])))
            except (TypeError, ValueError):
                pass
        command = sample.get("cmd")
        if command is None:
            command = sample.get("cmd_vel")
        try:
            if abs(float(command[0])) <= 0.05 and abs(float(command[1])) <= 0.03:
                zero_commands += 1
        except (TypeError, ValueError, IndexError):
            pass
        route = _trial_end_route_identity(sample)
        if route.get("route_id") not in (None, "", 0, "0"):
            active_route_samples += 1
        route_key = (
            route.get("route_id"),
            route.get("route_kind"),
            route.get("goal_transaction_id"),
        )
        if route and route_key != previous_route_key:
            if current_segment:
                route_segments.append(segment_summary(current_segment))
            current_segment = []
            wall_elapsed = sample.get("wall_elapsed_seconds")
            if wall_elapsed is None:
                # Older metrics samples expose ROS time but not the runner's
                # monotonic wall elapsed field. It is still a useful ordered
                # timestamp for the final bounded window.
                wall_elapsed = sample.get("ros_time")
            route_transitions.append({
                "sample": sample.get("sample"),
                "wall_elapsed_seconds": wall_elapsed,
                **route,
            })
            previous_route_key = route_key
        current_segment.append(sample)
    if current_segment:
        route_segments.append(segment_summary(current_segment))

    distance_summary = {}
    if distances:
        first_distance = distances[0]
        last_distance = distances[-1]
        distance_summary = {
            "first": round(first_distance, 4),
            "last": round(last_distance, 4),
            "minimum": round(min(distances), 4),
            "maximum": round(max(distances), 4),
            "progress_toward_goal": round(first_distance - last_distance, 4),
        }
    displacement = None
    if len(poses) >= 2:
        displacement = round(
            ((poses[-1][0] - poses[0][0]) ** 2
             + (poses[-1][1] - poses[0][1]) ** 2) ** 0.5,
            4,
        )
    progress = distance_summary.get("progress_toward_goal")
    if progress is not None and progress >= 0.05:
        interpretation = "moving_toward_goal"
    elif displacement is not None and displacement >= 0.05:
        interpretation = "moving_without_goal_progress"
    elif active_route_samples and zero_commands == len(samples):
        interpretation = "stationary_active_route"
    else:
        interpretation = "insufficient_progress_evidence"
    active_route = route_segments[-1] if route_segments else None
    return {
        "sample_count": len(samples),
        "first_sample": samples[0].get("sample"),
        "last_sample": samples[-1].get("sample"),
        "distance_to_goal_m": distance_summary,
        "pose_displacement_m": displacement,
        "active_route_sample_count": active_route_samples,
        "zero_command_sample_count": zero_commands,
        "route_transition_count": len(route_transitions),
        "route_transitions": route_transitions,
        # The aggregate values above are retained for compatibility.  These
        # route-local values are the authoritative view when a successor goal
        # is installed near the trial boundary; comparing the first sample of
        # route N with the last sample of route N+1 is not physical progress.
        "route_segments": route_segments,
        "active_route": active_route,
        "active_route_interpretation": (
            None if active_route is None else active_route.get("interpretation")
        ),
        "interpretation": interpretation,
    }


def _goal_tolerance_from_records(records, sample, metrics_path=None):
    """Resolve the actual TEB XY envelope captured by this run."""
    candidates = []
    if isinstance(sample, dict):
        candidates.append(sample.get("teb_xy_goal_tolerance"))
    for event, payload in records:
        if event != "run_start" or not isinstance(payload, dict):
            continue
        params = payload.get("resolved_params")
        if isinstance(params, dict):
            candidates.append(params.get("teb_xy_goal_tolerance"))
        break
    # Metrics can start before move_base uploads its private parameters. The
    # launcher captures the later roslaunch parameter dump in the sibling TEB
    # log, so use that immutable evidence before falling back to 0.50 m.
    if metrics_path is not None:
        try:
            run_dir = Path(metrics_path).parent
            for path in run_dir.glob("*_teb_navigation.log"):
                text = path.read_text(encoding="utf-8", errors="replace")[-512 * 1024 :]
                for match in re.finditer(
                    r"/(?:move_base/TebLocalPlannerROS|lste_global_frontier)/"
                    r"(?:xy_goal_tolerance|teb_xy_goal_tolerance)\s*:\s*([^\s]+)",
                    text,
                ):
                    candidates.append(match.group(1))
        except OSError:
            pass
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value >= 0.05:
            return value
    return 0.50


def _command_output_gap_evidence(records, goal_tolerance):
    """Summarize explicit forward-only command suppression near trial end."""
    candidates = []
    for event, payload in records:
        if event != "sample" or not isinstance(payload, dict):
            continue
        mux = payload.get("cmd_vel_mux")
        mux = mux if isinstance(mux, dict) else {}
        if str(mux.get("filter_reason") or "") != "forward_only_reverse_clamp":
            continue
        try:
            distance = float(payload.get("distance_to_goal"))
            planner = payload.get("teb_planner_cmd") or [0.0, 0.0]
            output = (mux.get("output") or {}).get("linear_x", 0.0)
            planner_linear = float(planner[0])
            output_linear = float(output)
        except (TypeError, ValueError, IndexError):
            continue
        if distance <= goal_tolerance + 0.02:
            continue
        turn = payload.get("teb_turn_supervisor")
        turn = turn if isinstance(turn, dict) else {}
        if planner_linear < -0.002 and abs(output_linear) <= 0.02 \
                and str(turn.get("state") or "").upper() != "TURNING":
            candidates.append(payload)
    recent = candidates[-16:]
    return {
        "sample_count": len(candidates),
        "recent_sample_count": len(recent),
        "last_gap": bool(recent),
        "last_distance_to_goal": (
            None if not recent else recent[-1].get("distance_to_goal")
        ),
    }


def _terminal_boundary_evidence(records):
    """Return explicit terminal evidence that can follow the last sample."""
    terminal_events = {
        "persistent_execution_terminal",
        "task_completed",
        "task_done",
    }
    frontier_terminal_reasons = {
        "source_viewpoint_arrived",
        "endpoint_coverage_recorded",
        "frontier_observed_at_standoff",
        "portal_place_entered",
        "portal_crossing_verified",
    }
    bridge_terminal_tokens = (
        "terminal",
        "endpoint_reached",
        "frontier_endpoint_terminal",
    )
    for event, payload in reversed(records[-128:]):
        if event in terminal_events:
            return True, event
        if event == "move_base_status":
            status = str(payload.get("status_name") or "").upper()
            if status == "SUCCEEDED":
                return True, "move_base_succeeded"
        if event == "teb_bridge_event":
            bridge_event = str(payload.get("bridge_event") or "").lower()
            if any(token in bridge_event for token in bridge_terminal_tokens):
                return True, bridge_event
        if event == "global_frontier_event":
            reason = str(payload.get("reason") or "").strip().lower()
            frontier_event = str(payload.get("frontier_event") or "").strip().lower()
            if reason in frontier_terminal_reasons or frontier_event in frontier_terminal_reasons:
                return True, reason or frontier_event
    return False, None


def build_trial_end_diagnostic(metrics_path: Path, start_offset: int = 0):
    """Reduce the last metrics sample to a non-failure termination state.

    A trial timeout is an experiment boundary, not proof of a navigation
    failure.  It can nevertheless leave an active route stalled or waiting
    for materialization.  Keep this reducer deliberately separate from the
    metrics failure-episode classifier: the runner cannot create a
    metrics-owned ``<run>-F####`` episode after the node has stopped.
    """
    # Global-frontier and run-stop events can be tens of kilobytes each.  The
    # small startup tail is intentionally cheap, but it may contain no sample
    # when several large events are the final records.  Use a still-bounded
    # larger tail for the one-time trial-end artifact.
    records = _startup_metrics_records(
        metrics_path,
        limit=max(_STARTUP_TAIL_BYTES, 1024 * 1024),
        start_offset=start_offset,
    )
    sample_window = [
        _compact_trial_end_sample(payload)
        for event, payload in records
        if event == "sample" and isinstance(payload, dict)
    ][-_TRIAL_END_SAMPLE_WINDOW:]
    latest_sample = sample_window[-1] if sample_window else None
    termination_id = None
    try:
        termination_id = Path(metrics_path).parent.name + "-T0001"
    except (AttributeError, TypeError):
        pass
    if latest_sample is None:
        return {
            "schema_version": 1,
            "termination_id": termination_id,
            "termination_kind": "trial_boundary",
            "classification": "missing_metrics_sample",
            "state": "unknown",
            "reason": "metrics_sample_missing",
            "latest_sample": None,
            "recent_samples": [],
            "progress": _trial_end_progress_summary([]),
            "recent_events": [],
            "event_counts": {},
        }

    feedback = latest_sample.get("move_base_feedback")
    feedback = feedback if isinstance(feedback, dict) else {}
    status = str(
        latest_sample.get("move_base_status")
        or feedback.get("status_name")
        or ""
    ).strip().upper()
    command = latest_sample.get("cmd")
    if command is None:
        command = latest_sample.get("cmd_vel")
    try:
        command_active = (
            abs(float(command[0])) > 0.05
            or abs(float(command[1])) > 0.03
        )
    except (TypeError, ValueError, IndexError):
        command_active = False
    navfn = latest_sample.get("navfn_plan")
    navfn = navfn if isinstance(navfn, dict) else {}
    teb_status = str(latest_sample.get("teb_status") or "").strip().lower()
    plan_available = bool(navfn.get("poses")) or teb_status == "trajectory_valid"
    bridge_active = bool(
        latest_sample.get("teb_bridge_active")
        or latest_sample.get("bridge_active")
    )
    goal_tolerance = _goal_tolerance_from_records(
        records, latest_sample, metrics_path=metrics_path
    )
    command_gap = _command_output_gap_evidence(records, goal_tolerance)
    terminal_boundary_observed, terminal_boundary_event = _terminal_boundary_evidence(
        records
    )
    try:
        goal_distance = float(latest_sample.get("distance_to_goal"))
    except (TypeError, ValueError):
        goal_distance = None
    within_goal_tolerance = (
        goal_distance is not None
        and goal_distance >= 0.0
        and goal_distance <= goal_tolerance
    )
    if bool(latest_sample.get("task_done")):
        state, reason = "completed", "task_done"
    elif status in {"ABORTED", "REJECTED", "LOST"}:
        state, reason = "failure", "move_base_terminal_failure"
    elif bridge_active and within_goal_tolerance:
        # A persistent bridge can remain ACTIVE while the terminal contract
        # propagates. Zero velocity in this envelope is expected settling,
        # not evidence of a controller stall.
        state, reason = "terminal_settle", "within_goal_tolerance_waiting_for_terminal"
    elif bridge_active and terminal_boundary_observed:
        # A terminal callback can arrive after the final periodic sample. The
        # bridge may therefore still look active with a zero command even
        # though the route has already crossed its physical endpoint.
        state, reason = "terminal_settle", "terminal_boundary_observed_after_last_sample"
    elif bridge_active and command_gap["last_gap"]:
        state, reason = "controller_output_gap_candidate", "forward_only_reverse_clamp_at_trial_end"
    elif bridge_active and not command_active and plan_available:
        state, reason = "stall_candidate", "active_route_without_effective_command"
    elif bridge_active and not plan_available:
        state, reason = "active_without_plan", "active_route_without_executable_plan"
    elif bridge_active:
        state, reason = "in_progress", "active_route_at_trial_end"
    else:
        state, reason = "idle", "no_active_controller_route"

    classification = {
        "terminal_settle": "terminal_settle",
        "completed": "completed",
        "idle": "idle_at_trial_boundary",
        "controller_output_gap_candidate": "controller_output_gap",
        "stall_candidate": "active_route_stall",
        "active_without_plan": "planner_no_path",
        "in_progress": "active_route_timeout",
    }.get(state, "unknown_trial_boundary")
    progress = _trial_end_progress_summary(sample_window)
    evidence_missing = [
        key for key in ("pose", "goal", "route_identity", "cmd")
        if key not in latest_sample
    ]

    event_counts = {}
    for event, _payload in records:
        event_counts[event] = event_counts.get(event, 0) + 1
    recent_events = []
    for event, payload in records:
        if event == "sample":
            continue
        compact = _compact_startup_event(event, payload)
        if compact is not None:
            recent_events.append(compact)

    route_identity = latest_sample.get("route_identity")
    route_identity = route_identity if isinstance(route_identity, dict) else {}
    return {
        "schema_version": 1,
        "termination_id": termination_id,
        "termination_kind": "trial_boundary",
        "classification": classification,
        "state": state,
        "reason": reason,
        "ros_time": latest_sample.get("ros_time"),
        "status": status or None,
        "bridge_active": bridge_active,
        "command_active": command_active,
        "plan_available": plan_available,
        "goal_tolerance_m": round(goal_tolerance, 4),
        "within_goal_tolerance": within_goal_tolerance,
        "controller_output_gap": command_gap,
        "terminal_boundary_observed": terminal_boundary_observed,
        "terminal_boundary_event": terminal_boundary_event,
        "active_route_id": (
            route_identity.get("active_route_id")
            or route_identity.get("route_id")
        ),
        "active_route_kind": (
            route_identity.get("active_route_kind")
            or route_identity.get("route_kind")
        ),
        "latest_sample": _compact_trial_end_sample(latest_sample),
        "recent_samples": sample_window,
        "progress": progress,
        "evidence": {
            "complete": not evidence_missing,
            "missing": evidence_missing,
            "sample_window_size": len(sample_window),
        },
        "recent_events": recent_events[-16:],
        "event_counts": event_counts,
    }


def write_trial_end_diagnostic_artifact(
    run_dir: Path,
    trial,
    diagnostic: dict,
    metrics_path: Path | None = None,
    outcome: str = "timeout",
    artifact_name: str | None = None,
    slice_id: str | None = None,
):
    """Atomically persist one timeout/termination evidence pointer.

    This artifact is intentionally not a failure episode.  Failure IDs are
    owned by ``lste_navigation_metrics`` and are only assigned after a
    sustained trigger.  The runner still needs a small, self-contained record
    when the experiment deadline arrives before such a trigger closes.
    """
    run_dir = Path(run_dir)
    diagnostic = diagnostic if isinstance(diagnostic, dict) else {}
    run_timestamp = run_dir.name
    artifact_path = run_dir / (
        artifact_name or (run_timestamp + "_trial_end_diagnostic.json")
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "trial_end_diagnostic",
        "created_at": _startup_failure_timestamp(),
        "run_timestamp": run_timestamp,
        "outcome": str(outcome),
        "slice_id": slice_id,
        "trial": trial_as_dict(trial),
        "metrics_log": None if metrics_path is None else str(metrics_path),
        "termination_id": diagnostic.get("termination_id"),
        "classification": diagnostic.get("classification"),
        "diagnostic": diagnostic if isinstance(diagnostic, dict) else {},
        "artifact_path": str(artifact_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = artifact_path.with_name(artifact_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, artifact_path)
    _append_lifecycle_event(
        run_dir,
        "trial_end_diagnostic",
        level="WARN" if diagnostic.get("state") not in {"completed", "idle"} else "INFO",
        outcome=str(outcome),
        state=diagnostic.get("state"),
        reason=diagnostic.get("reason"),
        slice_id=slice_id,
        artifact_path=str(artifact_path),
    )
    return str(artifact_path)


def _tail_file(path: Path | None, limit: int = _STARTUP_TAIL_BYTES):
    """Read a bounded tail without loading a potentially huge run log."""
    if path is None:
        return ""
    path = Path(path)
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit), os.SEEK_SET)
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _startup_metrics_records(
    metrics_path: Path | None,
    limit: int = _STARTUP_TAIL_BYTES,
    start_offset: int = 0,
):
    """Decode a bounded metrics tail for a compact evidence view."""
    records = []
    for line in _bounded_metrics_text(
        metrics_path,
        start_offset=start_offset,
        limit=limit,
    ).splitlines():
        match = _METRICS_EVENT_RE.search(line)
        if match is None:
            continue
        try:
            payload = json.loads(match.group("data"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append((match.group("event"), payload))
    return records


def _compact_startup_sample(sample):
    """Keep the fields needed to localize a startup failure without a full ring."""
    if not isinstance(sample, dict):
        return None
    keys = (
        "ros_time", "wall_elapsed_seconds", "pose", "pose_frame",
        "pose_goal_frame", "pose_transform_available", "goal", "goal_frame",
        "distance_to_goal", "goal_source", "goal_transaction_id", "cmd_vel",
        "teb_cmd", "teb_planner_cmd", "teb_status", "teb_feedback",
        "navfn_plan", "navfn_path_remaining_m", "navfn_path_endpoint", "scan",
        "global_costmap", "local_costmap", "move_base_status", "move_base_feedback",
        "cmd_vel_mux", "bridge_active", "navigation_hold", "controller",
        "execution_architecture", "state", "frontier_route_unavailable_count",
    )
    return {key: sample[key] for key in keys if key in sample}


def _compact_startup_event(event, payload):
    """Retain identity and ownership fields from a recent lifecycle event."""
    if not isinstance(payload, dict):
        return None
    keys = (
        "ros_time", "route_id", "route_kind", "mission_route_kind", "goal",
        "frame_id", "reason", "bridge_event", "source", "transaction_id",
        "active", "active_route_id", "active_route_kind", "active_goal",
        "active_goal_context", "active_mission_route_kind", "result_status",
        "status_name", "persistent_execution", "controller_lease",
    )
    result = {"event": str(event)}
    result.update({key: payload[key] for key in keys if key in payload})
    return result


def _startup_file_metadata(path: Path | None):
    """Return small, serializable metadata for a diagnostic log path."""
    if path is None:
        return {"path": None, "exists": False, "bytes": 0}
    path = Path(path)
    try:
        return {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
        }
    except OSError:
        return {"path": str(path), "exists": False, "bytes": 0}


def startup_readiness_snapshot(
    metrics_path: Path | None,
    global_frontier_path: Path | None = None,
    require_frontier: bool = True,
):
    """Summarize whether a run has entered executable navigation.

    This deliberately reads artifacts rather than ROS.  ``navigation_readiness``
    is the primary contract.  A real ``route_selected``/``route_command`` is a
    valid fallback because it proves that the frontier node reached execution
    even if the status event was lost by the metrics subscriber.
    """
    records = _startup_metrics_records(metrics_path)
    metric_events = [
        payload for event, payload in records if event == "global_frontier_event"
    ]
    frontier_text = _tail_file(global_frontier_path)
    frontier_lines = frontier_text.splitlines()
    clean_frontier_lines = [
        _ANSI_ESCAPE_RE.sub("", line) for line in frontier_lines
    ]

    readiness_events = [
        event for event in metric_events
        if event.get("frontier_event") == "navigation_readiness"
    ]
    route_events = [
        event for event in metric_events
        if event.get("frontier_event") in _STARTUP_ROUTE_EVENTS
        and _positive_route_id(event.get("route_id")) is not None
    ]
    readiness_ready = any(
        event.get("ready") is True or event.get("state") == "ready"
        for event in readiness_events
    )
    frontier_gate_passed = any(
        "Global frontier startup gate passed" in line
        or "Global frontier startup gate: ready for persistent_stream" in line
        for line in clean_frontier_lines
    )
    frontier_route_lines = [
        line for line in clean_frontier_lines
        if "Global frontier route command " in line
    ]
    route_seen = bool(route_events or frontier_route_lines)

    gate_states = []
    for line in clean_frontier_lines:
        match = re.search(
            r"Global frontier startup gate:\s*([A-Za-z0-9_]+)", line
        )
        if match:
            gate_states.append(match.group(1))
    latest_readiness = readiness_events[-1] if readiness_events else None
    latest_route = route_events[-1] if route_events else None
    latest_event = metric_events[-1] if metric_events else None
    latest_sample = next(
        (payload for event, payload in reversed(records) if event == "sample"),
        None,
    )
    latest_bridge_event = next(
        (
            _compact_startup_event(event, payload)
            for event, payload in reversed(records)
            if event == "teb_bridge_event"
        ),
        None,
    )
    latest_mission_goal = next(
        (
            _compact_startup_event(event, payload)
            for event, payload in reversed(records)
            if event == "mission_goal_transaction"
        ),
        None,
    )
    event_counts = {}
    for event, _payload in records:
        event_counts[event] = event_counts.get(event, 0) + 1

    # Target-entry is an explicit local isolation profile.  Its contract is
    # SLAM + perception + controller, so a missing frontier node is expected;
    # the first bounded metrics sample with a pose proves that the runtime is
    # executable.  Formal exploration trials retain the stricter frontier
    # readiness gate above.
    isolated_sample_ready = (
        not require_frontier
        and isinstance(latest_sample, dict)
        and latest_sample.get("pose") is not None
    )
    ready = bool(
        readiness_ready
        or frontier_gate_passed
        or route_seen
        or isolated_sample_ready
    )
    if ready:
        if isolated_sample_ready and not (
            readiness_ready or frontier_gate_passed or route_seen
        ):
            reason = "navigation_readiness_without_frontier"
        elif readiness_ready or frontier_gate_passed:
            reason = "navigation_readiness_announced"
        else:
            reason = "route_observed_without_readiness_event"
        status = "ready"
    elif gate_states:
        state = gate_states[-1]
        if "navfn" in state:
            reason = "navfn_startup_probe_not_ready"
        elif "costmap" in state:
            reason = "global_costmap_not_ready"
        else:
            reason = "startup_gate_%s" % state
        status = "waiting"
    elif metric_events:
        reason = "global_frontier_no_route_or_readiness"
        status = "waiting"
    else:
        reason = "navigation_metrics_not_ready"
        status = "waiting"

    return {
        "status": status,
        "ready": ready,
        "route_seen": route_seen,
        "reason": reason,
        "metrics": _startup_file_metadata(metrics_path),
        "global_frontier": _startup_file_metadata(global_frontier_path),
        "metrics_global_frontier_event_count": len(metric_events),
        "metrics_last_frontier_event": (
            None if latest_event is None else latest_event.get("frontier_event")
        ),
        "metrics_last_ros_time": (
            None if latest_event is None else latest_event.get("ros_time")
        ),
        "metrics_event_counts": event_counts,
        "latest_sample": _compact_startup_sample(latest_sample),
        "latest_bridge_event": latest_bridge_event,
        "latest_mission_goal": latest_mission_goal,
        "latest_frontier_event": latest_event,
        "readiness_event_count": len(readiness_events),
        "readiness_last_state": (
            None if latest_readiness is None else latest_readiness.get("state")
        ),
        "readiness_last_ready": (
            None if latest_readiness is None else latest_readiness.get("ready")
        ),
        "route_event_count": len(route_events),
        "route_last_id": (
            None if latest_route is None else latest_route.get("route_id")
        ),
        "global_frontier_gate_passed": frontier_gate_passed,
        "global_frontier_last_gate_state": (
            None if not gate_states else gate_states[-1]
        ),
        "global_frontier_last_evidence": [
            line[-512:] for line in clean_frontier_lines
            if (
                "Global frontier startup gate" in line
                or "Global frontier route command " in line
            )
        ][-8:],
    }


def _positive_route_id(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def wait_for_startup_readiness(
    run_dir: Path,
    metrics_path: Path | None,
    timeout_seconds: float,
    poll_interval: float = 0.25,
    require_frontier: bool = True,
):
    """Wait briefly for readiness, returning a diagnostic timeout snapshot."""
    timeout_seconds = float(timeout_seconds)
    poll_interval = float(poll_interval)
    if timeout_seconds <= 0.0:
        raise ValueError("startup readiness timeout must be positive")
    if poll_interval <= 0.0:
        raise ValueError("startup readiness poll interval must be positive")
    run_dir = Path(run_dir)
    global_frontier_path = (
        run_dir / (run_dir.name + "_global_frontier.log")
    )
    started = time.monotonic()
    while True:
        snapshot = startup_readiness_snapshot(
            metrics_path, global_frontier_path,
            require_frontier=require_frontier,
        )
        elapsed = time.monotonic() - started
        snapshot["elapsed_wall_seconds"] = round(elapsed, 3)
        snapshot["timeout_seconds"] = timeout_seconds
        if snapshot["ready"]:
            return snapshot
        if elapsed >= timeout_seconds:
            snapshot["status"] = "timeout"
            snapshot["failure_reason"] = snapshot["reason"]
            return snapshot
        time.sleep(min(poll_interval, timeout_seconds - elapsed))


def _startup_failure_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_lifecycle_event(run_dir: Path, event: str, level="INFO", **fields):
    """Append one runner-owned lifecycle event to the run's log."""
    run_dir = Path(run_dir)
    path = run_dir / (run_dir.name + "_lifecycle.log")
    payload = json.dumps(fields, ensure_ascii=False, sort_keys=True)
    line = (
        "%s [%s] process=office_experiment_runner event=%s "
        "run_timestamp=%s data=%s\n"
        % (
            _startup_failure_timestamp(),
            str(level).upper(),
            str(event),
            run_dir.name,
            payload,
        )
    )
    try:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
    except OSError:
        # The artifact itself remains authoritative if lifecycle logging races
        # a failed filesystem or an already-closed run directory.
        pass


def write_navigation_failure_report(run_dir: Path):
    """Write the single investigator-facing report for one finished run.

    Report generation is deliberately in-process.  It must not add another
    subprocess to the stop transaction, because a mocked or interrupted stop
    still needs to be represented by the same experiment record.
    """
    run_dir = Path(run_dir)
    json_path = run_dir / (run_dir.name + "_failure_report.json")
    markdown_path = run_dir / (run_dir.name + "_failure_report.md")
    try:
        report = build_failure_report(run_dir)
        rendered_json = json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        json_path.write_text(rendered_json, encoding="utf-8")
        markdown_path.write_text(
            render_failure_report(report), encoding="utf-8"
        )
        _append_lifecycle_event(
            run_dir,
            "failure_report_ready",
            json_path=str(json_path),
            markdown_path=str(markdown_path),
            outcome=report.get("outcome"),
        )
        return {
            "status": "written",
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
            "outcome": report.get("outcome"),
        }
    except (OSError, TypeError, ValueError, KeyError) as exc:
        _append_lifecycle_event(
            run_dir,
            "failure_report_failed",
            level="WARN",
            error_type=type(exc).__name__,
        )
        return {
            "status": "failed",
            "json_path": None,
            "markdown_path": None,
            "error": type(exc).__name__,
        }


def write_startup_failure_artifact(
    run_dir: Path,
    trial,
    readiness_snapshot: dict,
    reason: str | None = None,
):
    """Persist one self-contained startup failure pointer and event log."""
    run_dir = Path(run_dir)
    run_timestamp = run_dir.name
    artifact_path = run_dir / (run_timestamp + "_startup_failure.json")
    log_path = run_dir / (run_timestamp + "_startup_failure.log")
    lifecycle_path = run_dir / (run_timestamp + "_lifecycle.log")
    failure_id = run_timestamp + "-S0001"
    failure_reason = str(reason or readiness_snapshot.get(
        "failure_reason", readiness_snapshot.get("reason", "unknown")
    ))
    payload = {
        "schema_version": 1,
        "failure_kind": "startup_failure",
        "failure_id": failure_id,
        "created_at": _startup_failure_timestamp(),
        "run_timestamp": run_timestamp,
        "reason": failure_reason,
        "trial": trial_as_dict(trial),
        "readiness": readiness_snapshot,
        "artifact_path": str(artifact_path),
        "log_path": str(log_path),
        "lifecycle_log": str(lifecycle_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = artifact_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, artifact_path)
    event_data = {
        "failure_id": failure_id,
        "reason": failure_reason,
        "artifact_path": str(artifact_path),
        "readiness": readiness_snapshot,
    }
    log_line = (
        "%s [ERROR] process=office_experiment_runner "
        "event=startup_failure run_timestamp=%s data=%s\n"
        % (
            _startup_failure_timestamp(),
            run_timestamp,
            json.dumps(event_data, ensure_ascii=False, sort_keys=True),
        )
    )
    log_path.write_text(log_line, encoding="utf-8")
    _append_lifecycle_event(
        run_dir,
        "startup_failure",
        level="ERROR",
        failure_id=failure_id,
        reason=failure_reason,
        artifact_path=str(artifact_path),
    )
    return {
        "failure_id": failure_id,
        "reason": failure_reason,
        "artifact_path": str(artifact_path),
        "log_path": str(log_path),
        "lifecycle_log": str(lifecycle_path),
    }


def trial_environment(
    trial,
    base_environment=None,
    failure_post_window=None,
    failure_pre_window=None,
):
    """Build the isolated environment for one matrix trial."""
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update({
        "GLOBAL_FRONTIER_METHOD": trial.method,
        "OFFICE_BUILDING_PROFILE": trial.profile,
        "OFFICE_BUILDING_TRIAL_ID": str(trial.trial_id),
        "GAZEBO_RANDOM_SEED": str(trial.seed),
        "OFFICE_BUILDING_TASK_JSON": trial.task_json,
        "OFFICE_BUILDING_TASK_ID": trial.task_id,
        "TARGET_EVAL_ENABLED": "true" if trial.target_eval_enabled else "false",
        # The matrix contract must win over an inherited interactive-shell
        # controller. Otherwise the record can claim TEB while SAPPO is live.
        "LSTE_CONTROLLER": trial.controller,
    })
    if failure_post_window is not None:
        value = float(failure_post_window)
        if value <= 0.0:
            raise ValueError("failure_post_window must be positive")
        environment["NAVIGATION_FAILURE_EVIDENCE_POST_WINDOW"] = str(value)
    if failure_pre_window is not None:
        value = float(failure_pre_window)
        if value <= 0.0:
            raise ValueError("failure_pre_window must be positive")
        environment["NAVIGATION_FAILURE_EVIDENCE_PRE_WINDOW"] = str(value)
    return environment


def _terminate_process_group(process):
    """Stop a timed-out launcher and descendants before cleanup starts."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # ``Popen.kill`` still handles a launcher that detached its process
        # group before the group-wide signal was delivered.
        process.kill()
        process.wait(timeout=5)


@contextmanager
def _interrupt_safe_cleanup():
    """Keep the cleanup transaction intact after the operator presses Ctrl-C.

    ``run_trial`` catches the first ``KeyboardInterrupt`` so it can persist an
    interrupted record.  A second interrupt must not terminate the stop
    command halfway through, otherwise Gazebo/tmux children can survive and
    contaminate the next targeted reproduction.  Signal handlers can only be
    changed from the main thread, so a non-main-thread caller simply uses the
    normal subprocess behavior.
    """
    previous = None
    try:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            try:
                signal.signal(signal.SIGINT, previous)
            except (AttributeError, OSError, ValueError):
                pass


def _run_benchmark_stop(environment, timeout_seconds=30.0):
    """Run the unified stop path with a bounded, interrupt-safe wait."""
    command = [str(RUNNER), "stop"]
    with _interrupt_safe_cleanup():
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
            timeout=float(timeout_seconds),
        )
    return getattr(completed, "returncode", None)


def _run_benchmark_start(trial, environment):
    """Run one benchmark start with a hard, process-group timeout."""
    command = [str(RUNNER), "start", trial.level]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=trial.startup_timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        raise
    except KeyboardInterrupt:
        # The launcher owns a process group.  Do not leave a half-started
        # Gazebo/ROS tree behind when an operator aborts during startup.
        _terminate_process_group(process)
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _launcher_start_metadata(run_dir: Path):
    """Read the small immutable identity written by the office launcher."""
    lifecycle = Path(run_dir) / (Path(run_dir).name + "_lifecycle.log")
    try:
        text = lifecycle.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    line = next(
        (item for item in text.splitlines() if "event=run_start" in item),
        "",
    )
    metadata = {}
    for key in (
        "level", "profile", "gazebo_seed", "exploration_method",
        "controller", "task_id", "world",
    ):
        match = re.search(r"(?:^| )%s=([^ ]+)" % re.escape(key), line)
        if match:
            metadata[key] = match.group(1)
    return metadata


def _assert_reusable_runtime(run_dir: Path, trial):
    """Reject reuse when the live process identity differs from the trial."""
    metadata = _launcher_start_metadata(run_dir)
    expected = {
        "level": trial.level,
        "profile": trial.profile,
        "gazebo_seed": str(trial.seed),
        "exploration_method": trial.method,
        "controller": trial.controller,
        "task_id": trial.task_id,
    }
    expected_world = ROOT / (
        "worlds/benchmark/office_building_v1_%s_no_pro3.world" % trial.level
    )
    mismatches = []
    for key, value in expected.items():
        actual = metadata.get(key)
        if actual != str(value):
            mismatches.append("%s=%s (runtime=%s)" % (key, value, actual or "missing"))
    if metadata.get("world") != str(expected_world):
        mismatches.append(
            "world=%s (runtime=%s)"
            % (expected_world, metadata.get("world") or "missing")
        )
    if mismatches:
        raise ValueError(
            "existing runtime is not compatible with this diagnostic slice: "
            + ", ".join(mismatches)
        )


def _runtime_nodes_alive():
    """Require the benchmark nodes, not merely a stale ``current`` link."""
    try:
        completed = subprocess.run(
            ["rosnode", "list"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if completed.returncode != 0:
        return False
    nodes = set((completed.stdout or "").splitlines())
    return {"/lste_navigation_metrics", "/move_base"}.issubset(nodes)


def _prepare_warm_runtime(
    trial,
    reuse_existing=False,
    failure_post_window=None,
    failure_pre_window=None,
):
    """Start once, or attach to a compatible live runtime without waiting."""
    environment = trial_environment(
        trial,
        failure_post_window=failure_post_window,
        failure_pre_window=failure_pre_window,
    )
    if reuse_existing:
        run_dir = current_run_directory()
        if run_dir is None:
            raise RuntimeError("--reuse-runtime requires an active benchmark run")
        if not _runtime_nodes_alive():
            raise RuntimeError(
                "--reuse-runtime found a run directory but benchmark nodes are not alive"
            )
        _assert_reusable_runtime(run_dir, trial)
        metrics_path = wait_for_metrics_log(run_dir, timeout_seconds=2.0)
        if metrics_path is None:
            raise RuntimeError(
                "--reuse-runtime found no navigation metrics log in %s" % run_dir
            )
        return WarmRuntime(
            run_dir=run_dir,
            metrics_path=metrics_path,
            environment=environment,
            startup_elapsed_wall_seconds=0.0,
            startup_readiness={
                "status": "reused",
                "ready": True,
                "reason": "existing_runtime_reused",
                "elapsed_wall_seconds": 0.0,
            },
            started_here=False,
            stop_at_end=False,
        )

    previous_run_dir = current_run_directory()
    startup_started = time.monotonic()
    try:
        _run_benchmark_start(trial, environment)
    except (OSError, subprocess.SubprocessError) as exc:
        elapsed = round(time.monotonic() - startup_started, 3)
        run_dir = current_run_directory()
        if run_dir is not None and run_dir != previous_run_dir:
            _append_lifecycle_event(
                run_dir,
                "warm_session_start_failed",
                level="ERROR",
                reason=type(exc).__name__,
                elapsed_wall_seconds=elapsed,
            )
        raise RuntimeError("warm session startup failed: %s" % type(exc).__name__) from exc

    run_dir = current_run_directory()
    if run_dir is None or run_dir == previous_run_dir:
        raise RuntimeError("warm session did not create a new benchmark run")
    metrics_path = wait_for_metrics_log(
        run_dir,
        timeout_seconds=min(30.0, trial_startup_readiness_timeout(trial)),
    )
    if metrics_path is None:
        raise RuntimeError("warm session navigation metrics log did not appear")
    readiness = wait_for_startup_readiness(
        run_dir,
        metrics_path,
        trial_startup_readiness_timeout(trial),
        require_frontier=(trial.profile != "target_entry"),
    )
    if not readiness.get("ready"):
        write_startup_failure_artifact(
            run_dir,
            trial,
            readiness,
            reason=readiness.get("failure_reason", readiness.get("reason")),
        )
        raise RuntimeError(
            "warm session readiness failed: %s"
            % readiness.get("failure_reason", readiness.get("reason"))
        )
    startup_elapsed = round(time.monotonic() - startup_started, 3)
    _append_lifecycle_event(
        run_dir,
        "warm_session_ready",
        startup_elapsed_wall_seconds=startup_elapsed,
        readiness=readiness,
    )
    return WarmRuntime(
        run_dir=run_dir,
        metrics_path=metrics_path,
        environment=environment,
        startup_elapsed_wall_seconds=startup_elapsed,
        startup_readiness=readiness,
        started_here=True,
        stop_at_end=True,
    )


def run_trial(
    trial,
    stop_on_failure=False,
    failure_post_window=None,
    failure_pre_window=None,
):
    """Execute one run, then persist its summary beside its timestamped logs."""
    environment = trial_environment(
        trial,
        failure_post_window=failure_post_window,
        failure_pre_window=failure_pre_window,
    )
    previous_run_dir = current_run_directory()
    startup_started = time.monotonic()
    run_dir = None
    capture = None
    metrics_path = None
    startup_elapsed_wall_seconds = None
    trial_started_at = None
    trial_elapsed_wall_seconds = None
    failure_stop_observed_wall_seconds = None
    startup_readiness = None
    startup_failure = None
    outcome = "startup_failed"
    infrastructure_failure = None
    failure_stop = None
    trial_end_diagnosis = None
    trial_end_diagnostic_path = None
    failure_report = None
    video = {"video_status": "missing"}
    cleanup_elapsed_wall_seconds = None
    cleanup_return_code = None
    cleanup_failure = None
    summary_elapsed_wall_seconds = None
    startup_readiness_timeout_seconds = trial_startup_readiness_timeout(trial)
    try:
        try:
            _run_benchmark_start(trial, environment)
            startup_elapsed_wall_seconds = round(
                time.monotonic() - startup_started, 3
            )
        except subprocess.TimeoutExpired:
            startup_elapsed_wall_seconds = round(
                time.monotonic() - startup_started, 3
            )
            infrastructure_failure = "benchmark_start_timeout"
        except (OSError, subprocess.CalledProcessError) as exc:
            startup_elapsed_wall_seconds = round(
                time.monotonic() - startup_started, 3
            )
            infrastructure_failure = "benchmark_start_failed:%s" % type(exc).__name__

        run_dir = current_run_directory()
        if run_dir is not None and run_dir == previous_run_dir:
            infrastructure_failure = "benchmark_run_directory_not_replaced"
            run_dir = None
        if infrastructure_failure is None:
            if run_dir is None:
                infrastructure_failure = "benchmark_run_directory_missing"
            else:
                # The startup barrier is outside the trial timeout.  This
                # keeps slow Gazebo/model initialization from shortening the
                # navigation episode.
                startup_gate_started = time.monotonic()
                metrics_path = wait_for_metrics_log(
                    run_dir,
                    timeout_seconds=min(
                        30.0,
                        startup_readiness_timeout_seconds,
                    ),
                )
                if metrics_path is None:
                    startup_readiness = startup_readiness_snapshot(
                        None,
                        run_dir / (run_dir.name + "_global_frontier.log"),
                    )
                    startup_readiness["status"] = "timeout"
                    startup_readiness["elapsed_wall_seconds"] = round(
                        time.monotonic() - startup_gate_started, 3
                    )
                    startup_readiness["timeout_seconds"] = float(
                        startup_readiness_timeout_seconds
                    )
                    startup_readiness["failure_reason"] = (
                        "navigation_metrics_log_missing"
                    )
                    startup_failure = write_startup_failure_artifact(
                        run_dir,
                        trial,
                        startup_readiness,
                        reason="navigation_metrics_log_missing",
                    )
                    outcome = "startup_failure"
                    infrastructure_failure = "navigation_metrics_log_missing"
                else:
                    _append_lifecycle_event(
                        run_dir,
                        "readiness_wait_started",
                        timeout_seconds=startup_readiness_timeout_seconds,
                    )
                    startup_gate_elapsed = time.monotonic() - startup_gate_started
                    startup_gate_remaining = (
                        startup_readiness_timeout_seconds
                        - startup_gate_elapsed
                    )
                    if startup_gate_remaining <= 0.0:
                        startup_readiness = startup_readiness_snapshot(
                            metrics_path,
                            run_dir / (run_dir.name + "_global_frontier.log"),
                        )
                        startup_readiness["status"] = "timeout"
                        startup_readiness["elapsed_wall_seconds"] = round(
                            startup_gate_elapsed, 3
                        )
                        startup_readiness["timeout_seconds"] = float(
                            startup_readiness_timeout_seconds
                        )
                        startup_readiness["failure_reason"] = (
                            startup_readiness.get("reason")
                        )
                    else:
                        startup_readiness = wait_for_startup_readiness(
                            run_dir,
                            metrics_path,
                            startup_gate_remaining,
                            require_frontier=(trial.profile != "target_entry"),
                        )
                    if not startup_readiness["ready"]:
                        startup_failure = write_startup_failure_artifact(
                            run_dir,
                            trial,
                            startup_readiness,
                            reason=startup_readiness.get(
                                "failure_reason",
                                startup_readiness.get("reason"),
                            ),
                        )
                        outcome = "startup_failure"
                        infrastructure_failure = (
                            "navigation_startup_readiness_timeout"
                        )
                    else:
                        _append_lifecycle_event(
                            run_dir,
                            "navigation_ready",
                            readiness=startup_readiness,
                        )
                        trial_started_at = time.monotonic()
                        if trial.video_required:
                            capture = start_capture(run_dir, trial, environment)
                        outcome = "timeout"
                        deadline = trial_started_at + trial.timeout_seconds
                        poll_interval = 0.25 if stop_on_failure else 2.0
                        while time.monotonic() < deadline:
                            if stop_on_failure:
                                failure_stop = latest_failure_snapshot(metrics_path)
                                if failure_stop is not None:
                                    failure_stop_observed_wall_seconds = round(
                                        time.monotonic() - trial_started_at, 3
                                    )
                                    _append_lifecycle_event(
                                        run_dir,
                                        "failure_stop_requested",
                                        level="WARN",
                                        failure_id=failure_stop.get("failure_id"),
                                        classification=failure_stop.get(
                                            "classification"
                                        ),
                                        observed_wall_seconds=(
                                            failure_stop_observed_wall_seconds
                                        ),
                                    )
                                    outcome = "failure_snapshot"
                                    break
                            if terminal_event_observed(
                                metrics_path, trial.terminal_event,
                            ):
                                outcome = trial.terminal_event
                                break
                            time.sleep(poll_interval)
                        if outcome == "timeout" and stop_on_failure:
                            # A snapshot can be written in the same scheduler tick
                            # as the deadline.  Take one final bounded read before
                            # classifying the trial as a timeout.
                            failure_stop = latest_failure_snapshot(metrics_path)
                            if failure_stop is not None:
                                failure_stop_observed_wall_seconds = round(
                                    time.monotonic() - trial_started_at, 3
                                )
                                _append_lifecycle_event(
                                    run_dir,
                                    "failure_stop_requested",
                                    level="WARN",
                                    failure_id=failure_stop.get("failure_id"),
                                    classification=failure_stop.get("classification"),
                                    observed_wall_seconds=(
                                        failure_stop_observed_wall_seconds
                                    ),
                                )
                                outcome = "failure_snapshot"
                        trial_elapsed_wall_seconds = round(
                            time.monotonic() - trial_started_at, 3
                        )
        elif run_dir is not None:
            # A launcher can time out after creating the run directory but
            # before the metrics node or all tmux panes become ready. Preserve
            # that partial startup scene in the same self-contained artifact.
            metrics_path = metrics_path_for_run(run_dir)
            startup_readiness = startup_readiness_snapshot(
                metrics_path,
                run_dir / (run_dir.name + "_global_frontier.log"),
            )
            startup_readiness["status"] = "launcher_failed"
            startup_readiness["failure_reason"] = infrastructure_failure
            startup_failure = write_startup_failure_artifact(
                run_dir,
                trial,
                startup_readiness,
                reason=infrastructure_failure,
            )
            outcome = "startup_failure"
        elif run_dir is None:
            # A failed launcher can still have left the old link in place;
            # never attach that old log to this trial.
            metrics_path = None
    except KeyboardInterrupt:
        # Preserve an auditable record and let the finally block perform the
        # same complete stop path as a timeout or terminal outcome.
        infrastructure_failure = infrastructure_failure or "operator_interrupt"
        outcome = "interrupted"
        if run_dir is None:
            candidate = current_run_directory()
            if candidate is not None and candidate != previous_run_dir:
                run_dir = candidate
        if run_dir is not None and metrics_path is None:
            metrics_path = metrics_path_for_run(run_dir)
        if trial_started_at is not None and trial_elapsed_wall_seconds is None:
            trial_elapsed_wall_seconds = round(
                time.monotonic() - trial_started_at, 3
            )
        if run_dir is not None:
            _append_lifecycle_event(
                run_dir,
                "operator_interrupt",
                level="WARN",
                reason="operator_interrupt",
            )
    except Exception as exc:
        infrastructure_failure = infrastructure_failure or (
            "trial_runtime_failed:%s" % type(exc).__name__
        )
    finally:
        # Always clean up after invoking the runner, including startup errors.
        if outcome == "timeout":
            environment["OFFICE_BUILDING_STOP_REASON"] = "trial_timeout"
        elif outcome == "failure_snapshot":
            environment["OFFICE_BUILDING_STOP_REASON"] = (
                "failure_snapshot:%s"
                % str((failure_stop or {}).get("failure_id", "unknown"))
            )
        elif outcome == "startup_failure":
            environment["OFFICE_BUILDING_STOP_REASON"] = (
                "startup_failure:%s"
                % str((startup_failure or {}).get("failure_id", "unknown"))
            )
        elif outcome == "interrupted":
            environment["OFFICE_BUILDING_STOP_REASON"] = "operator_interrupt"
        elif infrastructure_failure is not None:
            environment["OFFICE_BUILDING_STOP_REASON"] = infrastructure_failure
        else:
            environment["OFFICE_BUILDING_STOP_REASON"] = "terminal:%s" % outcome
        cleanup_started = time.monotonic()
        if run_dir is not None:
            _append_lifecycle_event(
                run_dir,
                "runner_stop_requested",
                level="WARN" if outcome in {"interrupted", "failure_snapshot"} else "INFO",
                reason=environment["OFFICE_BUILDING_STOP_REASON"],
            )
        try:
            cleanup_return_code = _run_benchmark_stop(environment)
            if cleanup_return_code not in (None, 0):
                cleanup_failure = "benchmark_stop_failed:exit_%s" % (
                    cleanup_return_code
                )
        except subprocess.TimeoutExpired:
            cleanup_failure = "benchmark_stop_timeout"
        except (OSError, subprocess.SubprocessError) as exc:
            cleanup_failure = "benchmark_stop_failed:%s" % type(exc).__name__
        except KeyboardInterrupt:
            # This is only reachable when the runner is not executing in the
            # main thread and signal shielding is unavailable.  Do not lose
            # the record; the next invocation's stopall remains idempotent.
            cleanup_failure = "benchmark_stop_interrupted"
        cleanup_elapsed_wall_seconds = round(
            time.monotonic() - cleanup_started, 3
        )
        if run_dir is not None:
            _append_lifecycle_event(
                run_dir,
                "runner_stop_completed",
                level="ERROR" if cleanup_failure else "INFO",
                reason=environment["OFFICE_BUILDING_STOP_REASON"],
                return_code=cleanup_return_code,
                cleanup_failure=cleanup_failure,
                elapsed_wall_seconds=cleanup_elapsed_wall_seconds,
            )
        if cleanup_failure is not None:
            infrastructure_failure = infrastructure_failure or cleanup_failure
        # A normal timeout or terminal run may close an active failure episode
        # during the metrics node's graceful shutdown. Read once more after
        # cleanup so the experiment record still points at the final evidence
        # artifact even when --stop-on-failure was not requested.
        if failure_stop is None and metrics_path is not None:
            failure_stop = latest_failure_snapshot(metrics_path)
        if (
            failure_stop is not None
            and outcome == "timeout"
        ):
            # The metrics process can finish its post-failure evidence window
            # only after the runner has requested the normal timeout cleanup.
            # Keep the experiment record consistent with the artifact that was
            # actually written; otherwise one run is reported as ``timeout``
            # in its record and ``failure`` in its failure report.
            outcome = "failure_snapshot"
            failure_stop_observed_wall_seconds = (
                failure_stop_observed_wall_seconds
                if failure_stop_observed_wall_seconds is not None
                else trial_elapsed_wall_seconds
            )
            if run_dir is not None:
                _append_lifecycle_event(
                    run_dir,
                    "failure_snapshot_detected_after_cleanup",
                    level="WARN",
                    failure_id=failure_stop.get("failure_id"),
                    classification=failure_stop.get("classification"),
                )
        # A timeout can end before the metrics watchdog promotes a sustained
        # condition to a closed failure episode.  Preserve the final physical
        # state in a separate diagnostic artifact instead of leaving only the
        # ambiguous string ``outcome=timeout`` in the experiment record.
        if (
            run_dir is not None
            and metrics_path is not None
            and outcome in {"timeout", "interrupted"}
            and failure_stop is None
        ):
            try:
                trial_end_diagnosis = build_trial_end_diagnostic(metrics_path)
                trial_end_diagnostic_path = write_trial_end_diagnostic_artifact(
                    run_dir,
                    trial,
                    trial_end_diagnosis,
                    metrics_path=metrics_path,
                    outcome=outcome,
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                infrastructure_failure = infrastructure_failure or (
                    "trial_end_diagnostic_failed:%s" % type(exc).__name__
                )
        if run_dir is not None:
            # Keep report creation after the trial-end artifact so the single
            # report includes either a closed failure episode or the final
            # bounded timeout state.  This is read-only post-processing and
            # does not extend the navigation window.
            failure_report = write_navigation_failure_report(run_dir)
        try:
            with _interrupt_safe_cleanup():
                video = finish_capture(capture)
        except KeyboardInterrupt:
            infrastructure_failure = infrastructure_failure or (
                "video_finalize_interrupted"
            )
            video = {"video_status": "failed"}
        except Exception as exc:
            infrastructure_failure = infrastructure_failure or (
                "video_finalize_failed:%s" % type(exc).__name__
            )
            video = {"video_status": "failed"}

    record = trial_as_dict(trial)
    record["outcome"] = outcome
    record["startup_elapsed_wall_seconds"] = startup_elapsed_wall_seconds
    record["elapsed_wall_seconds"] = round(
        time.monotonic() - startup_started, 3
    )
    record["trial_elapsed_wall_seconds"] = trial_elapsed_wall_seconds
    record["failure_stop_observed_wall_seconds"] = (
        failure_stop_observed_wall_seconds
    )
    record["stop_on_failure"] = bool(stop_on_failure)
    record["failure_stop_id"] = (
        None if failure_stop is None else failure_stop.get("failure_id")
    )
    record["failure_stop_classification"] = (
        None if failure_stop is None else failure_stop.get("classification")
    )
    record["failure_stop_diagnosis"] = (
        None if failure_stop is None else failure_stop.get("diagnosis")
    )
    record["failure_stop_artifact_path"] = (
        None if failure_stop is None else failure_stop.get("artifact_path")
    )
    record["failure_stop_route_id"] = (
        None if failure_stop is None else failure_stop.get("route_id")
    )
    record["failure_stop_route_kind"] = (
        None if failure_stop is None else failure_stop.get("route_kind")
    )
    record["failure_stop_trigger_pose"] = (
        None if failure_stop is None else failure_stop.get("trigger_pose")
    )
    record["failure_stop_trigger_goal"] = (
        None if failure_stop is None else failure_stop.get("trigger_goal")
    )
    record["failure_stop_command_chain"] = (
        None if failure_stop is None else failure_stop.get("command_chain")
    )
    record["failure_stop_started_metrics_seconds"] = (
        None
        if failure_stop is None
        else failure_stop.get("started_wall_elapsed_seconds")
    )
    record["failure_stop_completed_metrics_seconds"] = (
        None
        if failure_stop is None
        else failure_stop.get("ended_wall_elapsed_seconds")
    )
    record["failure_stop_evidence_duration_seconds"] = (
        None if failure_stop is None else failure_stop.get("duration_seconds")
    )
    record["trial_end_diagnostic_path"] = (
        None
        if trial_end_diagnostic_path is None
        else str(trial_end_diagnostic_path)
    )
    record["trial_end_diagnostic_state"] = (
        None
        if trial_end_diagnosis is None
        else trial_end_diagnosis.get("state")
    )
    record["trial_end_diagnostic_reason"] = (
        None
        if trial_end_diagnosis is None
        else trial_end_diagnosis.get("reason")
    )
    record["trial_end_diagnostic_termination_id"] = (
        None
        if trial_end_diagnosis is None
        else trial_end_diagnosis.get("termination_id")
    )
    record["trial_end_diagnostic_classification"] = (
        None
        if trial_end_diagnosis is None
        else trial_end_diagnosis.get("classification")
    )
    record["trial_end_diagnostic_progress"] = (
        None
        if trial_end_diagnosis is None
        else trial_end_diagnosis.get("progress")
    )
    record["failure_report_status"] = (
        None if failure_report is None else failure_report.get("status")
    )
    record["failure_report_json"] = (
        None if failure_report is None else failure_report.get("json_path")
    )
    record["failure_report_markdown"] = (
        None if failure_report is None else failure_report.get("markdown_path")
    )
    record["interrupted"] = outcome == "interrupted"
    record["cleanup_elapsed_wall_seconds"] = cleanup_elapsed_wall_seconds
    record["cleanup_return_code"] = cleanup_return_code
    record["cleanup_failure"] = cleanup_failure
    record["startup_readiness"] = startup_readiness
    record["startup_readiness_elapsed_wall_seconds"] = (
        None
        if startup_readiness is None
        else startup_readiness.get("elapsed_wall_seconds")
    )
    record["startup_failure_id"] = (
        None if startup_failure is None else startup_failure.get("failure_id")
    )
    record["startup_failure_reason"] = (
        None if startup_failure is None else startup_failure.get("reason")
    )
    record["startup_failure_artifact_path"] = (
        None
        if startup_failure is None
        else startup_failure.get("artifact_path")
    )
    record["startup_failure_log_path"] = (
        None if startup_failure is None else startup_failure.get("log_path")
    )
    record.update(video)
    if infrastructure_failure is not None:
        record["infrastructure_failure"] = infrastructure_failure
    if metrics_path is None:
        if run_dir is not None:
            record["run_directory"] = str(run_dir)
            record["phase_timings"] = {
                "launcher_start_seconds": startup_elapsed_wall_seconds,
                "readiness_wait_seconds": (
                    None
                    if startup_readiness is None
                    else startup_readiness.get("elapsed_wall_seconds")
                ),
                "navigation_trial_seconds": trial_elapsed_wall_seconds,
                "failure_evidence_window_seconds": (
                    None
                    if failure_stop is None
                    else failure_stop.get("duration_seconds")
                ),
                "cleanup_seconds": cleanup_elapsed_wall_seconds,
                "summary_seconds": None,
                "total_seconds": record.get("elapsed_wall_seconds"),
            }
            (run_dir / (run_dir.name + "_experiment_record.json")).write_text(
                json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        return record
    run_dir = metrics_path.parent
    summary_path = run_dir / (run_dir.name + "_summary.json")
    verification_path = run_dir / (run_dir.name + "_topology_verification.json")
    summary_ok = False
    summary_started = time.monotonic()
    try:
        with summary_path.open("w", encoding="utf-8") as stream:
            with _interrupt_safe_cleanup():
                subprocess.run(
                    [sys.executable, str(SUMMARY_SCRIPT), str(metrics_path)],
                    cwd=ROOT, stdout=stream, check=True,
                )
        summary_ok = True
    except KeyboardInterrupt:
        infrastructure_failure = infrastructure_failure or (
            "summary_generation_interrupted"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        infrastructure_failure = infrastructure_failure or (
            "summary_generation_failed:%s" % type(exc).__name__
        )
    summary_elapsed_wall_seconds = round(time.monotonic() - summary_started, 3)
    verification = None
    try:
        with verification_path.open("w", encoding="utf-8") as stream:
            with _interrupt_safe_cleanup():
                verification = subprocess.run(
                    [sys.executable, str(VERIFY_SCRIPT), str(metrics_path)],
                    cwd=ROOT, stdout=stream, check=False,
                )
    except KeyboardInterrupt:
        infrastructure_failure = infrastructure_failure or (
            "topology_verification_interrupted"
        )
    except OSError as exc:
        infrastructure_failure = infrastructure_failure or (
            "topology_verification_failed:%s" % type(exc).__name__
        )
    record.update({
        "run_directory": str(run_dir),
        "metrics_log": str(metrics_path),
        "summary": str(summary_path),
        "topology_verification": str(verification_path),
        "summary_status": "written" if summary_ok else "failed",
        "topology_verification_exit_code": (
            None if verification is None else verification.returncode
        ),
        "summary_elapsed_wall_seconds": summary_elapsed_wall_seconds,
        "phase_timings": {
            "launcher_start_seconds": startup_elapsed_wall_seconds,
            "readiness_wait_seconds": (
                None
                if startup_readiness is None
                else startup_readiness.get("elapsed_wall_seconds")
            ),
            "navigation_trial_seconds": trial_elapsed_wall_seconds,
            "failure_evidence_window_seconds": (
                None if failure_stop is None else failure_stop.get("duration_seconds")
            ),
            "cleanup_seconds": cleanup_elapsed_wall_seconds,
            "summary_seconds": summary_elapsed_wall_seconds,
            "total_seconds": record.get("elapsed_wall_seconds"),
        },
    })
    if infrastructure_failure is not None:
        record["infrastructure_failure"] = infrastructure_failure
    (run_dir / (run_dir.name + "_experiment_record.json")).write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def _metrics_file_offset(path: Path):
    try:
        return int(Path(path).stat().st_size)
    except OSError:
        return 0


def _next_warm_slice_index(run_dir: Path):
    """Allocate a monotonic slice number across separate reuse invocations."""
    pattern = re.compile(
        r"^%s_slice_(\d{4})_experiment_record\.json$"
        % re.escape(Path(run_dir).name)
    )
    indices = []
    for path in Path(run_dir).glob(
        Path(run_dir).name + "_slice_*_experiment_record.json"
    ):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def _next_warm_session_index(run_dir: Path):
    """Allocate a monotonic session number without touching earlier evidence."""
    run_name = Path(run_dir).name
    pattern = re.compile(
        r"^%s_warm_session_(\d{4})\.json$" % re.escape(run_name)
    )
    indices = []
    for path in Path(run_dir).glob(run_name + "_warm_session_*.json"):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    # The first implementation used one fixed name. Treat it as session 1 so
    # a later reuse invocation never overwrites it again.
    if (Path(run_dir) / (run_name + "_warm_session_record.json")).is_file():
        indices.append(1)
    return max(indices, default=0) + 1


def _warm_slice_record(
    runtime: WarmRuntime,
    trial,
    slice_index: int,
    session_id: str,
    stop_on_failure=False,
):
    """Run one bounded continuation window without restarting ROS/Gazebo."""
    slice_id = "%s-D%04d" % (runtime.run_dir.name, slice_index)
    slice_started = time.monotonic()
    metrics_offset = _metrics_file_offset(runtime.metrics_path)
    known_failure_ids = failure_snapshot_ids(runtime.metrics_path)
    timeout_seconds = min(float(trial.timeout_seconds), _MAX_WARM_SLICE_SECONDS)
    outcome = "timeout"
    failure_stop = None
    interrupted = False
    _append_lifecycle_event(
        runtime.run_dir,
        "diagnostic_slice_started",
        session_id=session_id,
        slice_id=slice_id,
        slice_index=slice_index,
        timeout_seconds=timeout_seconds,
        metrics_offset=metrics_offset,
        startup_reused=True,
    )
    deadline = slice_started + timeout_seconds
    try:
        while time.monotonic() < deadline:
            if stop_on_failure:
                failure_stop = latest_failure_snapshot(
                    runtime.metrics_path,
                    exclude_failure_ids=known_failure_ids,
                )
                if failure_stop is not None:
                    outcome = "failure_snapshot"
                    _append_lifecycle_event(
                        runtime.run_dir,
                        "diagnostic_slice_failure_observed",
                        level="WARN",
                        session_id=session_id,
                        slice_id=slice_id,
                        failure_id=failure_stop.get("failure_id"),
                        classification=failure_stop.get("classification"),
                    )
                    break
            if terminal_event_observed(
                runtime.metrics_path,
                trial.terminal_event,
                start_offset=metrics_offset,
            ):
                outcome = trial.terminal_event
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        outcome = "interrupted"
        interrupted = True
        _append_lifecycle_event(
            runtime.run_dir,
            "diagnostic_slice_interrupted",
            level="WARN",
            session_id=session_id,
            slice_id=slice_id,
        )

    elapsed = round(time.monotonic() - slice_started, 3)
    diagnostic = None
    diagnostic_path = None
    if outcome in {"timeout", "interrupted"}:
        try:
            diagnostic = build_trial_end_diagnostic(
                runtime.metrics_path,
                start_offset=metrics_offset,
            )
            diagnostic_path = write_trial_end_diagnostic_artifact(
                runtime.run_dir,
                trial,
                diagnostic,
                metrics_path=runtime.metrics_path,
                outcome=outcome,
                artifact_name=(
                    "%s_slice_%04d_trial_end_diagnostic.json"
                    % (runtime.run_dir.name, slice_index)
                ),
                slice_id=slice_id,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            diagnostic = None
    _append_lifecycle_event(
        runtime.run_dir,
        "diagnostic_slice_finished",
        level="WARN" if outcome in {"failure_snapshot", "interrupted"} else "INFO",
        session_id=session_id,
        slice_id=slice_id,
        outcome=outcome,
        elapsed_wall_seconds=elapsed,
        failure_id=(None if failure_stop is None else failure_stop.get("failure_id")),
        diagnostic_path=diagnostic_path,
    )
    record = trial_as_dict(trial)
    record.update({
        "outcome": outcome,
        "session_id": session_id,
        "slice_id": slice_id,
        "slice_index": slice_index,
        "runtime_reused": True,
        "runtime_run_directory": str(runtime.run_dir),
        "run_directory": str(runtime.run_dir),
        "metrics_log": str(runtime.metrics_path),
        "startup_elapsed_wall_seconds": 0.0,
        "startup_reused_from_session": True,
        "startup_readiness": runtime.startup_readiness,
        "startup_readiness_elapsed_wall_seconds": 0.0,
        "elapsed_wall_seconds": elapsed,
        "trial_elapsed_wall_seconds": elapsed,
        "stop_on_failure": bool(stop_on_failure),
        "failure_stop_id": (
            None if failure_stop is None else failure_stop.get("failure_id")
        ),
        "failure_stop_classification": (
            None if failure_stop is None else failure_stop.get("classification")
        ),
        "failure_stop_artifact_path": (
            None if failure_stop is None else failure_stop.get("artifact_path")
        ),
        "failure_stop_diagnosis": (
            None if failure_stop is None else failure_stop.get("diagnosis")
        ),
        "trial_end_diagnostic_path": diagnostic_path,
        "trial_end_diagnostic_state": (
            None if diagnostic is None else diagnostic.get("state")
        ),
        "trial_end_diagnostic_reason": (
            None if diagnostic is None else diagnostic.get("reason")
        ),
        "phase_timings": {
            "launcher_start_seconds": 0.0,
            "readiness_wait_seconds": 0.0,
            "navigation_trial_seconds": elapsed,
            "failure_evidence_window_seconds": (
                None
                if failure_stop is None
                else failure_stop.get("duration_seconds")
            ),
            "cleanup_seconds": 0.0,
            "summary_seconds": None,
            "total_seconds": elapsed,
        },
        "video_status": "disabled_warm_session",
        "video_override": "warm_session_no_video",
        "interrupted": interrupted,
    })
    artifact_path = runtime.run_dir / (
        "%s_slice_%04d_experiment_record.json"
        % (runtime.run_dir.name, slice_index)
    )
    record["experiment_record"] = str(artifact_path)
    artifact_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def _write_warm_session_summary(
    runtime: WarmRuntime,
    slice_records,
    cleanup,
    session_id: str,
):
    """Persist the session-level pointer; detailed evidence stays per slice."""
    session_path = runtime.run_dir / (
        runtime.run_dir.name + "_warm_session_%s.json" % session_id[-4:]
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "warm_diagnostic_session",
        "run_timestamp": runtime.run_dir.name,
        "session_id": session_id,
        "runtime_reused": not runtime.started_here,
        "startup_elapsed_wall_seconds": runtime.startup_elapsed_wall_seconds,
        "startup_readiness": runtime.startup_readiness,
        "slice_count": len(slice_records),
        "slices": slice_records,
        "cleanup": cleanup,
        "metrics_log": str(runtime.metrics_path),
        "run_directory": str(runtime.run_dir),
        "artifact_path": str(session_path),
    }
    temporary_path = session_path.with_name(session_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, session_path)
    return session_path


def run_warm_session(
    trial,
    slice_count: int,
    stop_on_failure=False,
    reuse_existing=False,
    failure_post_window=None,
    failure_pre_window=None,
):
    """Run short continuation slices against one live ROS/Gazebo runtime."""
    if slice_count <= 0:
        raise ValueError("warm slice count must be positive")
    if float(trial.timeout_seconds) > _MAX_WARM_SLICE_SECONDS:
        raise ValueError(
            "warm diagnostic slices must be <= %.0f seconds; select a short "
            "phase or pass --max-trial-seconds" % _MAX_WARM_SLICE_SECONDS
        )
    previous_run_dir = current_run_directory()
    runtime = None
    slice_records = []
    session_id = None
    slice_start_index = 1
    session_failure = None
    cleanup = {
        "performed": False,
        "return_code": None,
        "elapsed_wall_seconds": 0.0,
        "failure": None,
    }
    try:
        runtime = _prepare_warm_runtime(
            trial,
            reuse_existing=reuse_existing,
            failure_post_window=failure_post_window,
            failure_pre_window=failure_pre_window,
        )
        _append_lifecycle_event(
            runtime.run_dir,
            "diagnostic_warm_session_started",
            session_id=(
                "%s-W%04d" % (runtime.run_dir.name, _next_warm_session_index(runtime.run_dir))
            ),
            slice_count=slice_count,
            startup_reused=not runtime.started_here,
            startup_elapsed_wall_seconds=runtime.startup_elapsed_wall_seconds,
        )
        session_id = "%s-W%04d" % (
            runtime.run_dir.name,
            _next_warm_session_index(runtime.run_dir),
        )
        slice_start_index = _next_warm_slice_index(runtime.run_dir)
        for slice_index in range(slice_start_index, slice_start_index + slice_count):
            record = _warm_slice_record(
                runtime,
                trial,
                slice_index,
                session_id,
                stop_on_failure=stop_on_failure,
            )
            slice_records.append(record)
            if record.get("outcome") in {"failure_snapshot", "interrupted"}:
                break
    except KeyboardInterrupt:
        session_failure = "operator_interrupt"
    except (OSError, RuntimeError, ValueError) as exc:
        session_failure = "%s:%s" % (type(exc).__name__, exc)
    finally:
        candidate_run_dir = (
            None if runtime is not None else current_run_directory()
        )
        if runtime is not None and runtime.stop_at_end:
            stop_reason = "warm_session_complete"
            if slice_records:
                last = slice_records[-1]
                if last.get("outcome") == "failure_snapshot":
                    stop_reason = "failure_snapshot:%s" % (
                        last.get("failure_stop_id") or "unknown"
                    )
                elif last.get("outcome") == "interrupted":
                    stop_reason = "operator_interrupt"
            if session_failure is not None:
                stop_reason = session_failure
            runtime.environment["OFFICE_BUILDING_STOP_REASON"] = stop_reason
            _append_lifecycle_event(
                runtime.run_dir,
                "warm_session_stop_requested",
                level="WARN" if session_failure else "INFO",
                reason=stop_reason,
            )
            started = time.monotonic()
            try:
                cleanup["return_code"] = _run_benchmark_stop(runtime.environment)
                if cleanup["return_code"] not in (None, 0):
                    cleanup["failure"] = "benchmark_stop_failed:exit_%s" % (
                        cleanup["return_code"]
                    )
            except (OSError, subprocess.SubprocessError) as exc:
                cleanup["failure"] = "benchmark_stop_failed:%s" % type(exc).__name__
            cleanup["elapsed_wall_seconds"] = round(time.monotonic() - started, 3)
            cleanup["performed"] = True
            _append_lifecycle_event(
                runtime.run_dir,
                "warm_session_stop_completed",
                level="ERROR" if cleanup["failure"] else "INFO",
                reason=stop_reason,
                return_code=cleanup["return_code"],
                elapsed_wall_seconds=cleanup["elapsed_wall_seconds"],
                cleanup_failure=cleanup["failure"],
            )
        elif runtime is not None:
            _append_lifecycle_event(
                runtime.run_dir,
                "diagnostic_warm_session_detached",
                reason="runtime_left_running_by_request",
            )
        elif not reuse_existing and candidate_run_dir is not None \
                and candidate_run_dir != previous_run_dir:
            # Startup can fail after creating the run directory but before the
            # WarmRuntime object is returned. Clean only that newly created run.
            environment = trial_environment(
                trial,
                failure_post_window=failure_post_window,
                failure_pre_window=failure_pre_window,
            )
            environment["OFFICE_BUILDING_STOP_REASON"] = session_failure or (
                "warm_session_start_failed"
            )
            try:
                _run_benchmark_stop(environment)
            except (OSError, subprocess.SubprocessError):
                pass

    if runtime is None:
        return {
            "session": {
                "outcome": "startup_failure",
                "infrastructure_failure": session_failure,
                "slice_count": 0,
            },
            "records": slice_records,
        }
    failure_report = write_navigation_failure_report(runtime.run_dir)
    session_path = _write_warm_session_summary(
        runtime,
        slice_records,
        cleanup,
        session_id or "%s-W%04d" % (
            runtime.run_dir.name,
            _next_warm_session_index(runtime.run_dir),
        ),
    )
    session = {
        "outcome": (
            "failure_snapshot"
            if any(item.get("outcome") == "failure_snapshot" for item in slice_records)
            else ("interrupted" if session_failure == "operator_interrupt" else "completed")
        ),
        "run_directory": str(runtime.run_dir),
        "metrics_log": str(runtime.metrics_path),
        "warm_session_record": str(session_path),
        "failure_report_json": failure_report.get("json_path"),
        "failure_report_markdown": failure_report.get("markdown_path"),
        "slice_count": len(slice_records),
        "startup_reused": not runtime.started_here,
        "startup_elapsed_wall_seconds": runtime.startup_elapsed_wall_seconds,
        "cleanup": cleanup,
    }
    if session_failure is not None:
        session["infrastructure_failure"] = session_failure
    if cleanup.get("failure"):
        session["infrastructure_failure"] = session.get(
            "infrastructure_failure"
        ) or cleanup["failure"]
    return {"session": session, "records": slice_records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", type=Path,
        default=ROOT / "scripts/tests/office_building/experiment_matrix.yaml",
    )
    parser.add_argument("--phase", default="topology_smoke")
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="run only this declared method; repeat to select a small subset",
    )
    parser.add_argument(
        "--level",
        action="append",
        dest="levels",
        help="run only this level; repeat to select a small subset",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        type=int,
        help="run only this seed; repeat to select a small subset",
    )
    parser.add_argument(
        "--trial-id",
        action="append",
        dest="trial_ids",
        type=int,
        help="run only this trial id; repeat to select a small subset",
    )
    parser.add_argument(
        "--profile",
        help=(
            "override the robot profile for a diagnostic run (for example "
            "office_entry or target_entry); formal matrix runs keep their "
            "declared profile"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help=(
            "stop a running trial after its first closed failure snapshot; "
            "useful for short diagnostic experiments"
        ),
    )
    parser.add_argument(
        "--failure-post-window",
        type=float,
        help=(
            "override the diagnostic failure post window in seconds; only "
            "use with --stop-on-failure"
        ),
    )
    parser.add_argument(
        "--failure-pre-window",
        type=float,
        help=(
            "override the diagnostic failure pre window in seconds; only "
            "use with --stop-on-failure"
        ),
    )
    parser.add_argument(
        "--max-trial-seconds",
        type=float,
        help=(
            "diagnostic-only upper bound for the navigation episode; use with "
            "--stop-on-failure, or with --warm-slices, to reproduce a local "
            "condition without waiting for the formal matrix timeout"
        ),
    )
    parser.add_argument(
        "--warm-slices",
        type=int,
        metavar="N",
        help=(
            "run N bounded diagnostic slices in one ROS/Gazebo session; "
            "requires exactly one selected trial and keeps every slice <= 60s"
        ),
    )
    parser.add_argument(
        "--reuse-runtime",
        action="store_true",
        help=(
            "reuse a compatible already-running benchmark runtime; skips "
            "launcher startup, readiness wait, and automatic stop"
        ),
    )
    parser.add_argument(
        "--no-video", "--skip-video",
        dest="no_video",
        action="store_true",
        help=(
            "diagnostic-only: skip screen recording to shorten local failure "
            "reproduction; requires --stop-on-failure unless using warm mode"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    trials = build_trials(matrix, args.phase)
    if args.methods:
        requested = set(args.methods)
        declared = set(matrix["methods"])
        unknown = sorted(requested - declared)
        if unknown:
            raise SystemExit(
                "unknown matrix method(s): %s" % ", ".join(unknown)
            )
        trials = [trial for trial in trials if trial.method in requested]
    if args.levels:
        requested_levels = set(args.levels)
        declared_levels = {
            str(level) for trial in trials for level in (trial.level,)
        }
        unknown = sorted(requested_levels - declared_levels)
        if unknown:
            raise SystemExit("unknown matrix level(s): %s" % ", ".join(unknown))
        trials = [trial for trial in trials if trial.level in requested_levels]
    if args.seeds:
        requested_seeds = set(args.seeds)
        trials = [trial for trial in trials if trial.seed in requested_seeds]
    if args.trial_ids:
        requested_trial_ids = set(args.trial_ids)
        trials = [trial for trial in trials if trial.trial_id in requested_trial_ids]
    if args.profile:
        profile = str(args.profile).strip()
        if not profile:
            raise SystemExit("profile must not be empty")
        trials = [replace(trial, profile=profile) for trial in trials]
    if not trials:
        raise SystemExit("selection filters produced no trials")
    warm_mode = args.warm_slices is not None or args.reuse_runtime
    if warm_mode and len(trials) != 1:
        raise SystemExit(
            "warm diagnostic mode requires exactly one selected trial; "
            "use --method/--level/--seed to select it"
        )
    warm_slices = args.warm_slices if args.warm_slices is not None else 1
    if warm_mode and warm_slices <= 0:
        raise SystemExit("warm-slices must be positive")
    if (
        args.failure_post_window is not None
        or args.failure_pre_window is not None
        or args.profile
        or args.max_trial_seconds is not None
        or args.no_video
    ) and not args.stop_on_failure and not warm_mode:
        raise SystemExit(
            "diagnostic profile/window/timeout/video overrides require "
            "--stop-on-failure"
        )
    if args.max_trial_seconds is not None:
        if args.max_trial_seconds <= 0.0:
            raise SystemExit("max-trial-seconds must be positive")
        trials = [
            replace(
                trial,
                timeout_seconds=min(
                    float(trial.timeout_seconds),
                    float(args.max_trial_seconds),
                ),
            )
            for trial in trials
        ]
    if warm_mode and float(trials[0].timeout_seconds) > _MAX_WARM_SLICE_SECONDS:
        raise SystemExit(
            "warm diagnostic slices must be <= %.0f seconds; pass "
            "--max-trial-seconds <= %.0f or select diagnostic_short"
            % (_MAX_WARM_SLICE_SECONDS, _MAX_WARM_SLICE_SECONDS)
        )
    if warm_mode:
        trials = [
            replace(
                trials[0],
                video_required=False,
                video_override=(
                    "disabled_by_cli" if args.no_video else "warm_session_no_video"
                ),
            )
        ]
    if args.no_video:
        # The matrix remains the source of truth for formal runs. This
        # replacement is allowed only in the explicitly diagnostic mode above
        # and is carried into every trial artifact via video_override.
        trials = apply_diagnostic_video_override(trials, disable_video=True)
    plan = {
        "execute": bool(args.execute),
        "stop_on_failure": bool(args.stop_on_failure),
        "failure_post_window": args.failure_post_window,
        "failure_pre_window": args.failure_pre_window,
        "max_trial_seconds": args.max_trial_seconds,
        "video_override": (
            "disabled_by_cli" if args.no_video
            else ("warm_session_no_video" if warm_mode else None)
        ),
        "warm_slices": warm_slices if warm_mode else None,
        "reuse_runtime": bool(args.reuse_runtime),
        "trials": [trial_as_dict(t) for t in trials],
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    records = []
    session = None
    if warm_mode:
        result = run_warm_session(
            trials[0],
            warm_slices,
            stop_on_failure=args.stop_on_failure,
            reuse_existing=args.reuse_runtime,
            failure_post_window=args.failure_post_window,
            failure_pre_window=args.failure_pre_window,
        )
        records = result["records"]
        session = result["session"]
    else:
        for trial in trials:
            record = run_trial(
                trial,
                stop_on_failure=args.stop_on_failure,
                failure_post_window=args.failure_post_window,
                failure_pre_window=args.failure_pre_window,
            )
            records.append(record)
            # An operator interrupt is a deliberate end to the diagnostic batch;
            # never launch the next matrix trial after its cleanup completes.
            if record.get("outcome") == "interrupted":
                break
    rendered = json.dumps(
        {"plan": plan, "session": session, "records": records},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if warm_mode:
        return 1 if session is None or session.get("infrastructure_failure") else 0
    incomplete = [
        record for record, trial in zip(records, trials)
        if record.get("infrastructure_failure")
        or (
            record.get("outcome") != trial.terminal_event
            and not (
                args.stop_on_failure
                and record.get("outcome") == "failure_snapshot"
                and record.get("failure_stop_id")
            )
        )
        or (
            trial.video_required
            and record.get("video_status") != "recorded"
        )
    ]
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
