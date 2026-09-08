"""External execution-event telemetry for the navigation observer.

This module translates read-only scan, controller, TEB bridge, and frontier
status topics into structured evidence. It has no publisher and never changes
controller state.
"""

import json
import math
import re
import time


def route_recovery_preemption_reason(event, payload, status_code):
    """Return the explicit reason for an intentional frontier cancellation."""
    if int(status_code) != 2 or not isinstance(payload, dict):
        return None
    event = str(event or "").strip()
    if event == "controller_lease_released":
        return str(
            payload.get("reason")
            or payload.get("frontier_lease_released_reason")
            or "controller_lease_released"
        )
    # The bridge reports the cancellation first and the frontier node reports
    # the matching invalidation immediately afterwards. Count only the
    # cancellation to avoid double-accounting one action boundary.
    if event == "cancel":
        reason = str(payload.get("reason", ""))
        if "frontier_route_invalidated" in reason:
            return reason
    return None


def route_invalidation_failure_trigger(event, payload):
    """Map an explicit frontier route invalidation to a failure trigger.

    ``frontier_observed_at_standoff`` is a normal observation boundary: the
    robot has learned enough at the endpoint and the old route is intentionally
    released. Other invalidations are terminal evidence from the exploration
    executive and must remain visible even when the bridge subsequently emits
    an intentional ``PREEMPTED`` during recovery.
    """
    if str(event or "").strip() != "route_invalidated":
        return None
    if not isinstance(payload, dict):
        return None
    reason = str(payload.get("reason", "")).strip().lower()
    if not reason or reason == "frontier_observed_at_standoff":
        return None
    if (
        reason in ("stall", "post_turn_no_progress")
        or "stall" in reason
        or "no_progress" in reason
    ):
        return "route_invalidated_stall"
    if reason in ("active_timeout", "portal_edge_deadline") or "timeout" in reason:
        return "route_invalidated_timeout"
    if reason == "disconnected" or any(
        token in reason for token in ("no_path", "unreachable", "disconnected")
    ):
        return "route_invalidated_disconnected"
    if any(token in reason for token in ("recovery", "abort", "controller")):
        return "route_invalidated_controller_failure"
    return "route_invalidated_failure"


class NavigationMetricsExecutionEventsMixin:
    def on_scan(self, message):
        all_ranges = list(message.ranges)
        forward, left, right = [], [], []
        for index, value in enumerate(all_ranges):
            angle = message.angle_min + index * message.angle_increment
            if not math.isfinite(value) or value <= 0.01:
                continue
            if abs(angle) <= math.radians(20.0):
                forward.append(value)
            elif 0.0 < angle <= math.radians(90.0):
                left.append(value)
            elif -math.radians(90.0) <= angle < 0.0:
                right.append(value)
        with self.lock:
            self.last_scan_wall = time.monotonic()
            self.scan_minimum = self._finite_min(all_ranges)
            self.scan_forward_minimum = self._finite_min(forward)
            self.scan_left_minimum = self._finite_min(left)
            self.scan_right_minimum = self._finite_min(right)
            if math.isfinite(self.scan_minimum):
                self.min_clearance = min(self.min_clearance, self.scan_minimum)

    def on_controller_mode(self, message):
        with self.lock:
            value = message.data.strip().lower()
            if value and value != self.controller_mode:
                self.controller_mode = value
                if value != "sappo":
                    # The SA-PPO status topic remains alive while TEB is
                    # selected. Never expose that stale policy decision as a
                    # TEB diagnosis in the run summary.
                    self.controller_status = "not_applicable"
                self._write("INFO", "controller_mode", mode=value)

    def on_bridge_status(self, message):
        """Record action-level goal queueing separately from move_base status.

        A queued update is expected during a healthy TEB action; a move_base
        PREEMPTED status is not. Keeping both counters makes that distinction
        explicit in the run summary.
        """
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"event": "invalid", "raw": message.data}
        if not isinstance(payload, dict):
            payload = {"event": "invalid", "raw": message.data}
        with self.lock:
            event = str(payload.get("event", "unknown"))
            context_payload = dict(payload)
            self._failure_record_context_locked("bridge", event, context_payload)
            # Bridge status calls this field ``result_status`` while
            # actionlib uses ``status``. Accept both so an intentional cancel
            # is correlated before /move_base/status arrives.
            status_code = int(
                payload.get("status", payload.get("result_status", -1)) or -1
            )
            route_recovery_reason = route_recovery_preemption_reason(
                event, payload, status_code
            )
            if route_recovery_reason is not None:
                self.pending_route_recovery_preemptions += 1
                self._increment_reason(
                    self.route_recovery_preemption_reasons,
                    route_recovery_reason,
                )
            if event in (
                "target_route_failed",
                "persistent_target_plan_failed",
                "target_plan_failed",
            ) or (event == "terminal" and status_code in (4, 5, 8, 9)):
                self._begin_failure_episode_locked(
                    "target_route_failed" if "target" in event else "bridge_terminal",
                    "teb_goal_bridge",
                    context_payload,
                )
            self.lifecycle_event_wall[event] = time.monotonic()
            self.bridge_events += 1
            self.bridge_last_event = event
            self.bridge_active = bool(payload.get("active", False))
            if "persistent_execution" in payload:
                self.persistent_execution = self._as_bool(
                    payload.get("persistent_execution")
                )
                self.execution_architecture = (
                    "persistent_stream" if self.persistent_execution
                    else "endpoint_action"
                )
            self.bridge_active_intent_source = str(
                payload.get("active_intent_source", self.bridge_active_intent_source)
            )
            self.bridge_latest_intent_source = str(
                payload.get("latest_intent_source", self.bridge_latest_intent_source)
            )
            if event == "goal_deferred":
                self.bridge_deferred_goal_updates += 1
            elif event == "dispatch":
                self.bridge_dispatches += 1
            elif event == "terminal":
                self.bridge_terminal_events += 1
                self._mark_action_terminal_locked(time.monotonic(), "bridge_terminal")
                self._reclassify_recent_discontinuities_locked(
                    time.monotonic(), "bridge_terminal"
                )
            elif event == "frontier_observation_completion_requested":
                self.bridge_frontier_observation_completions += 1
                self.pending_frontier_observation_preemptions += 1
            elif event == "frontier_terminal_settle_completion_requested":
                self.bridge_frontier_terminal_settle_completions += 1
                self.pending_frontier_terminal_settle_preemptions += 1
            elif event == "frontier_continuous_prefetch_handoff_requested":
                # Native `send_goal` replacement still appears as PREEMPTED
                # in /move_base/status. It is an intentional continuous route
                # transition, not an action failure or goal churn defect.
                self.pending_frontier_continuous_prefetch_preemptions += 1
            elif event == "frontier_continuous_prefetch_handoff_completed":
                self.bridge_frontier_continuous_prefetch_handoffs += 1
            elif event == "frontier_continuous_prefetch_handoff_fallback":
                self.bridge_frontier_continuous_prefetch_fallbacks += 1
            elif event in (
                "persistent_frontier_lookahead_handoff_promoted",
                "persistent_frontier_curve_handoff_promoted",
            ):
                self.bridge_persistent_lookahead_handoffs += 1
                if event == "persistent_frontier_curve_handoff_promoted":
                    self.bridge_persistent_curve_handoffs += 1
            elif event == "persistent_frontier_prefetch_admission_deferred":
                self.bridge_persistent_lookahead_admission_deferred += 1
                deferred_reason = str(payload.get("reason", "unknown"))
                self._increment_reason(
                    self.bridge_persistent_lookahead_admission_reasons,
                    deferred_reason,
                )
                if deferred_reason in (
                    "entry_tangent_too_sharp",
                    "navfn_entry_tangent_too_sharp",
                ):
                    active_goal = payload.get("active_goal")
                    if isinstance(active_goal, (list, tuple)) and len(active_goal) >= 2:
                        try:
                            self.pending_terminal_native_reorientation = {
                                "armed_wall": time.monotonic(),
                                "goal": (
                                    float(active_goal[0]),
                                    float(active_goal[1]),
                                ),
                                "route_id": int(payload.get("route_id", 0) or 0),
                                "entry_heading_delta_deg": float(
                                    payload.get("entry_heading_delta_deg", 0.0)
                                ),
                            }
                        except (TypeError, ValueError):
                            self.pending_terminal_native_reorientation = None
            elif event == "frontier_prefetch_requires_turn":
                self.bridge_frontier_prefetch_requires_turn += 1
            elif event == "priority_handoff_requested":
                self.bridge_priority_handoffs += 1
                self.pending_priority_preemptions += 1
            elif (
                event == "handoff_requested"
                and str(payload.get("reason", "")) == "higher_priority_intent"
            ):
                # With in-place replacement disabled, the bridge uses the
                # explicit cancel -> terminal callback -> dispatch lifecycle.
                # It has the same mission semantics as a priority-intent
                # replacement, but carries ``handoff_requested`` rather than
                # ``replacement_kind=priority_intent`` on the wire.
                self.bridge_priority_handoffs += 1
                self.pending_priority_preemptions += 1
            elif event == "target_retry_requested":
                self.bridge_target_retries += 1
            elif event == "target_route_failed":
                self.target_route_failures += 1
            elif event == "target_segment_handoff_requested":
                self.bridge_target_segment_handoffs += 1
            elif event == "cancel" and str(payload.get("reason", "")) == "task_done":
                # Only task completion is a deliberate cancel. Other bridge
                # cancel reasons remain observable as unexpected unless their
                # own explicit lifecycle classification handles them.
                self.pending_task_done_preemptions += 1
            replacement = bool(payload.get("replacement", False))
            replacement_kind = str(payload.get("replacement_kind", "none"))
            if replacement:
                self.bridge_goal_replacements += 1
                if replacement_kind == "priority_intent":
                    self.bridge_priority_goal_replacements += 1
                    # New bridge versions report the handoff on the dispatch
                    # itself; keep the legacy counter meaningful as well.
                    if event == "dispatch":
                        self.bridge_priority_handoffs += 1
                        # Native actionlib replacement reports the superseded
                        # move_base goal as PREEMPTED. It is an intentional
                        # mission-priority transition, not goal churn.
                        self.pending_priority_preemptions += 1
                elif replacement_kind == "target_segment":
                    self.bridge_target_goal_replacements += 1
                    if event == "dispatch":
                        self.bridge_target_segment_handoffs += 1
                        # Native actionlib replacement appears as PREEMPTED on
                        # /move_base/status. This is an expected same-track
                        # continuation, not unexpected goal churn.
                        self.pending_target_segment_preemptions += 1
                elif replacement_kind in (
                    "frontier_segment",
                    "frontier_route_connector",
                    "frontier_route_endpoint",
                    "frontier_sharp_branch",
                ):
                    self.bridge_frontier_segment_handoffs += 1
                    if replacement_kind == "frontier_sharp_branch":
                        self.bridge_frontier_sharp_replacements += 1
                    if event == "dispatch":
                        # Legacy rolling connector experiments still use a
                        # native action replacement. It is intentionally
                        # observable in the smoothness metrics, but must not
                        # be reported as an unexplained move_base failure.
                        self.pending_frontier_segment_preemptions += 1
            # ``_write`` already has an ``event`` positional argument; keep
            # the bridge's event name as data instead of passing it twice.
            bridge_event = payload.pop("event", event)
            self._write(
                "INFO",
                "teb_bridge_event",
                bridge_event=bridge_event,
                **payload,
            )

    def on_persistent_execution_terminal(self, message):
        """Record a bridge endpoint-release signal with its real architecture."""
        with self.lock:
            endpoint = (
                round(float(message.pose.position.x), 3),
                round(float(message.pose.position.y), 3),
            )
            now = time.monotonic()
            lifecycle_event = (
                "persistent_execution_terminal"
                if self.persistent_execution else "endpoint_action_terminal"
            )
            replacement_reason = (
                "persistent_endpoint_terminal"
                if self.persistent_execution else "endpoint_action_terminal"
            )
            self.lifecycle_event_wall[lifecycle_event] = now
            self._reclassify_recent_discontinuities_locked(
                now,
                lifecycle_event,
                replacement_reason=replacement_reason,
                window=1.0,
            )
            self._write(
                "INFO",
                lifecycle_event,
                execution_architecture=self.execution_architecture,
                endpoint=endpoint,
                frame=(message.header.frame_id or "").strip().lstrip("/"),
                current_goal=(
                    None
                    if self.goal is None
                    else [round(float(self.goal[0]), 3), round(float(self.goal[1]), 3)]
                ),
                teb_command=[
                    round(float(self.teb_command.linear.x), 4),
                    round(float(self.teb_command.angular.z), 4),
                ],
            )

    def on_global_frontier_status(self, message):
        """Preserve frontier route ownership changes beside controller events."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self.lock:
            event = str(payload.pop("event", "unknown")).strip() or "unknown"
            now = time.monotonic()
            self._failure_record_context_locked("frontier", event, payload)
            route_failure_trigger = route_invalidation_failure_trigger(
                event, payload
            )
            if event in (
                "execution_terminal_failure",
                "frontier_route_failed",
                "portal_hypothesis_failed",
                "portal_execution_failure_observed",
            ):
                self._begin_failure_episode_locked(
                    "frontier_route_failure", "global_frontier", payload
                )
            elif route_failure_trigger is not None:
                # The following bridge PREEMPTED is an intentional recovery
                # action, but the invalidation itself is already a failure
                # boundary. Keep both facts with one failure ID.
                failure_details = dict(payload)
                failure_details["route_invalidation_reason"] = str(
                    payload.get("reason", "unknown")
                )
                self._begin_failure_episode_locked(
                    route_failure_trigger,
                    "global_frontier",
                    failure_details,
                )
            self.lifecycle_event_wall["global_frontier_" + event] = now
            # The terminal pose topic is intentionally non-latched. A metrics
            # node can attach after a rapid terminal callback, so the frontier
            # node's direct successor-promotion record is the durable evidence
            # needed to classify the preceding zero command correctly.
            if event == "terminal_prefetch_promoted":
                lifecycle_event = (
                    "persistent_execution_terminal"
                    if self.persistent_execution else "endpoint_action_terminal"
                )
                replacement_reason = (
                    "persistent_endpoint_terminal"
                    if self.persistent_execution else "endpoint_action_terminal"
                )
                self.lifecycle_event_wall[lifecycle_event] = now
                self._mark_action_terminal_locked(
                    now, "global_frontier_terminal_prefetch_promoted"
                )
                self._reclassify_recent_discontinuities_locked(
                    now,
                    "global_frontier_terminal_prefetch_promoted",
                    replacement_reason=replacement_reason,
                    window=1.5,
                )
            elif event in {
                "persistent_frontier_endpoint_ignored",
                "persistent_frontier_goal_replaced",
            }:
                # A persistent bridge can reject a stale endpoint while the
                # graph layer installs the next route identity. The resulting
                # zero command is a goal handoff, not a clear-path safety
                # brake; preserve the raw bridge event and repair telemetry
                # causality for the preceding command sample.
                self._reclassify_recent_discontinuities_locked(
                    now,
                    "global_frontier_%s" % event,
                    replacement_reason="goal_transition",
                    window=1.5,
                )
            self._write(
                "INFO",
                "global_frontier_event",
                frontier_event=event,
                execution_architecture=self.execution_architecture,
                **payload,
            )

    def on_controller_status(self, message):
        with self.lock:
            if self.controller_mode == "sappo":
                text = message.data.strip()
                self.controller_status = text
                self.last_status_text = text
                match = re.search(
                    r"source=(\S+)\s+reason=(.*?)\s+requested=\(([-+0-9.eE]+),([-+0-9.eE]+)\)\s+"
                    r"action=\(([-+0-9.eE]+),([-+0-9.eE]+)\)\s+clearance=([-+0-9.eE]+|nan|inf)",
                    text,
                )
                if match:
                    self.controller_source = match.group(1)
                    self.controller_reason = match.group(2).strip()
                    self.controller_requested = (float(match.group(3)), float(match.group(4)))
                    self.controller_action = (float(match.group(5)), float(match.group(6)))
                    self.controller_action_delta = math.hypot(
                        self.controller_requested[0] - self.controller_action[0],
                        self.controller_requested[1] - self.controller_action[1],
                    )
                    safety_source = self.controller_source in (
                        "grid_guard", "dwa_guard", "mppi_guard", "turn_recovery"
                    )
                    safety_reason = any(
                        token in self.controller_reason
                        for token in ("collision", "emergency_stop", "no_safe", "blocked")
                    )
                    if safety_source or safety_reason:
                        self.safety_intervention_samples += 1
                    if any(
                        token in self.controller_reason
                        for token in ("emergency_stop", "no_safe")
                    ):
                        self.hard_stop_events += 1
                    try:
                        self.controller_predicted_clearance = float(match.group(7))
                    except ValueError:
                        self.controller_predicted_clearance = float("nan")
                else:
                    self.controller_source = "unknown"
                    self.controller_reason = text
                # The status contains rolling diagnostics (point counts,
                # waypoint coordinates, lock timers) that change every control
                # cycle. Treat only source plus the stable reason prefix as a
                # controller-state transition; the latest numeric fields are
                # still retained in every periodic sample.
                reason_signature = self.controller_reason.split(" grid_points=", 1)[0]
                reason_signature = re.sub(
                    r"grid_waypoint=\([^)]*\)", "grid_waypoint", reason_signature
                )
                signature = (self.controller_source, reason_signature)
                if signature == self.last_status_signature:
                    return
                self.last_status_signature = signature
                self.controller_status_changes += 1
                if self.controller_source not in ("policy", "unknown", "stop"):
                    self.safety_override_events += 1
                self._write(
                    "INFO" if self.controller_source in ("policy", "unknown") else "WARN",
                    "controller_status_change",
                    count=self.controller_status_changes,
                    source=self.controller_source,
                    reason=reason_signature,
                    requested=list(self.controller_requested),
                    action=list(self.controller_action),
                    action_delta=round(self.controller_action_delta, 4),
                    predicted_clearance=None if not math.isfinite(self.controller_predicted_clearance) else round(self.controller_predicted_clearance, 4),
                    safety_override_events=self.safety_override_events,
                    safety_intervention_samples=self.safety_intervention_samples,
                    hard_stop_events=self.hard_stop_events,
                    pose=None if self.pose is None else [round(value, 3) for value in self.pose],
                    goal=None if self.goal is None else [round(value, 3) for value in self.goal],
                    scan_forward_min=None if not math.isfinite(self.scan_forward_minimum) else round(self.scan_forward_minimum, 4),
                    scan_min=None if not math.isfinite(self.scan_minimum) else round(self.scan_minimum, 4),
                )
