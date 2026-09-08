"""Stall ownership policy for an active TEB move_base action."""

import math
import time

import rospy


class TebGoalBridgeHandoffStallMixin:
    def maybe_handoff_locked(self):
        """Handle a real no-progress failure without stealing healthy routes."""
        now = time.monotonic()
        if not self._handoff_is_eligible_locked(now):
            return
        pending_delta = self._pending_goal_delta_locked()
        if self._try_frontier_progress_transition_locked(pending_delta):
            return
        stalled = self._action_is_stalled_locked(now)
        target_progress_basis, target_stalled = self._target_stall_state_locked(
            now, pending_delta
        )
        if pending_delta < self.handoff_distance and not target_stalled:
            return
        # Endpoint proximity alone is not a failure: move_base may publish
        # SUCCEEDED on this callback turn. Only a real no-progress window may
        # transfer an action away from its current owner.
        if not stalled and not target_stalled:
            return
        if self._defer_native_teb_reorientation_locked(now, pending_delta):
            return
        if self.active_intent_priority < 2:
            self._report_stalled_frontier_locked(now, pending_delta)
            return
        self.frontier_stale_wait_started_monotonic = 0.0
        if target_stalled:
            self._latch_target_failure_locked(
                "target_%s_stalled" % target_progress_basis,
                status_text="NO_PROGRESS",
                cancel_action=True,
            )
            return
        self._cancel_stalled_action_locked(now, pending_delta)

    def _handoff_is_eligible_locked(self, now):
        """Return whether the watchdog may inspect the active action."""
        if self.frontier_continuous_prefetch_handoff_pending is not None:
            self._expire_frontier_continuous_prefetch_handoff_locked(now)
            return False
        if (
            not self.action_active
            or self.handoff_requested
            or self.latest_goal is None
            or self.last_dispatched_goal is None
        ):
            return False
        if (
            self.turn_supervisor_state == "TURNING"
            and self.active_intent_priority == 0
            and self.latest_intent_priority == 0
        ):
            # The supervisor owns this short atomic turn. A progress watchdog
            # must not reintroduce a stop/restart race while it closes yaw.
            return False
        return now - self.last_dispatch_monotonic >= self.handoff_min_interval

    def _try_frontier_progress_transition_locked(self, pending_delta):
        """Let healthy frontier endpoint and prefetch transitions complete first."""
        # An endpoint inside its observation region is not a failed point goal.
        # These checks intentionally precede the generic no-progress watchdog.
        return bool(
            self._start_frontier_continuous_prefetch_handoff_locked(pending_delta)
            or self._complete_frontier_observation_locked(pending_delta)
        )

    def _action_is_stalled_locked(self, now):
        return bool(
            self.active_progress_monotonic > 0.0
            and now - self.active_progress_monotonic >= self.progress_timeout
        )

    def _target_stall_state_locked(self, now, pending_delta):
        """Return the target-progress evidence for mission-level recovery."""
        progress_basis, progress_age = self._target_progress_state_locked(now)
        target_stalled = bool(
            self.active_intent_priority >= 2
            and progress_age >= self.progress_timeout
            and pending_delta < self.handoff_distance
        )
        return progress_basis, target_stalled

    def _defer_native_teb_reorientation_locked(self, now, pending_delta):
        """Keep one healthy native TEB rotation under its bounded yaw budget."""
        if not self._teb_reorientation_progressing_locked(now):
            return False
        self.teb_reorientation_deferrals += 1
        if now - self.teb_reorientation_last_status_monotonic >= 1.0:
            self.teb_reorientation_last_status_monotonic = now
            duration = now - self.teb_reorientation_started_monotonic
            progress_age = now - self.teb_reorientation_last_yaw_progress_monotonic
            self.publish_bridge_status(
                "teb_in_place_reorientation",
                reason="native_teb_yaw_progress",
                duration=round(duration, 3),
                yaw_progress=round(self.teb_reorientation_total_yaw, 4),
                yaw_progress_age=round(progress_age, 3),
                extension_budget=round(self.teb_reorientation_max_extension, 3),
                pending_delta=round(pending_delta, 3),
                feedback_distance=(
                    None
                    if self.active_feedback_distance is None
                    else round(self.active_feedback_distance, 3)
                ),
            )
            rospy.loginfo(
                "TEB goal bridge defers stale handoff during native "
                "reorientation: yaw_progress=%.1fdeg duration=%.1fs",
                math.degrees(self.teb_reorientation_total_yaw),
                duration,
            )
        return True

    def _report_stalled_frontier_locked(self, now, pending_delta):
        """Record frontier stall evidence, leaving recovery to move_base."""
        # A frontier route belongs to move_base after dispatch. TEB can make no
        # XY progress while resolving a corner or its own recovery. Cancelling
        # it here would turn that local recovery into PREEMPTED and an avoidable
        # zero-velocity gap. GlobalFrontier observes conclusive terminal state.
        if now - self.frontier_stale_wait_started_monotonic < 1.0:
            return
        self.frontier_stale_wait_started_monotonic = now
        self.publish_bridge_status(
            "frontier_stall_observed",
            pending_delta=round(pending_delta, 3),
            feedback_distance=(
                None
                if self.active_feedback_distance is None
                else round(self.active_feedback_distance, 3)
            ),
            progress_age=round(now - self.active_progress_monotonic, 3)
            if self.active_progress_monotonic > 0.0
            else None,
            action_owner="move_base",
        )
        rospy.loginfo(
            "TEB goal bridge leaves stalled frontier action to move_base: "
            "pending_delta=%.2fm feedback_distance=%s progress_age=%.1fs",
            pending_delta,
            "n/a"
            if self.active_feedback_distance is None
            else "%.2f" % self.active_feedback_distance,
            now - self.active_progress_monotonic
            if self.active_progress_monotonic > 0.0
            else 0.0,
        )

    def _cancel_stalled_action_locked(self, now, pending_delta):
        """Cancel an eligible non-frontier stale action and report its evidence."""
        reason = "no_feedback_progress"
        self.handoff_requested = True
        self.handoff_count += 1
        self.action_client.cancel_goal()
        self.publish_bridge_status(
            "handoff_requested",
            reason=reason,
            pending_delta=round(pending_delta, 3),
            feedback_distance=(
                None
                if self.active_feedback_distance is None
                else round(self.active_feedback_distance, 3)
            ),
            progress_age=round(now - self.active_progress_monotonic, 3)
            if self.active_progress_monotonic > 0.0
            else None,
            handoffs=int(self.handoff_count),
        )
        rospy.logwarn(
            "TEB goal bridge handing off stale action: reason=%s "
            "pending_delta=%.2fm feedback_distance=%s progress_age=%.1fs",
            reason,
            pending_delta,
            "n/a"
            if self.active_feedback_distance is None
            else "%.2f" % self.active_feedback_distance,
            now - self.active_progress_monotonic
            if self.active_progress_monotonic > 0.0
            else 0.0,
        )
