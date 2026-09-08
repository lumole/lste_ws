"""MoveBase terminal lifecycle for the TEB goal bridge.

This module publishes outcomes upward but never selects or sends a successor.
That ownership belongs to Global Frontier and the dispatch policy respectively.
"""

import copy
import time

import rospy
from actionlib_msgs.msg import GoalStatus
from lste_topo_access.msg import FrontierExecutionTerminal


TURN_ROUTE_KIND = "frontier_turn_connector"


class TebGoalBridgeActionTerminalMixin:
    @staticmethod
    def _action_contract_status_fields(action_contract):
        """Expose the dispatch identity and target metadata in diagnostics."""
        if action_contract is None:
            return {}
        return {
            "action_generation": int(action_contract.get("generation", 0) or 0),
            "transaction_id": int(action_contract.get("transaction_id", 0) or 0),
            "route_id": int(action_contract.get("route_id", 0) or 0),
            "route_kind": str(action_contract.get("route_kind", "") or ""),
            "mission_route_kind": str(
                action_contract.get("mission_route_kind", "") or ""
            ),
            "intent_source": str(
                action_contract.get("source", "unknown") or "unknown"
            ),
            "intent_priority": int(action_contract.get("priority", 0) or 0),
            "target_epoch": int(action_contract.get("target_epoch", 0) or 0),
            "target_track_id": str(
                action_contract.get("target_track_id", "") or ""
            ),
            "target_viewpoint_candidate_id": str(
                action_contract.get("target_viewpoint_candidate_id", "") or ""
            ),
            "target_viewpoint_attempt_id": str(
                action_contract.get("target_viewpoint_attempt_id", "") or ""
            ),
        }

    def _publish_execution_terminal_locked(
        self, source_goal, action_contract=None
    ):
        """Publish a terminal using an explicit action or semantic contract.

        Persistent endpoint reports intentionally use the current semantic
        route when no contract is supplied. MoveBase callbacks pass the
        dispatch-time contract so a later persistent mission update cannot
        relabel the transport result as a successor route.
        """
        if action_contract is not None:
            contract_goal = action_contract.get("source_goal")
            if contract_goal is not None:
                source_goal = contract_goal
        if source_goal is None:
            return None
        if action_contract is not None:
            route_id = int(action_contract.get("route_id", 0) or 0)
            route_kind = str(action_contract.get("route_kind", "") or "")
            intent_source = str(
                action_contract.get("source", "unknown") or "unknown"
            )
            intent_priority = int(action_contract.get("priority", 0) or 0)
            action_generation = int(
                action_contract.get("generation", self.action_generation)
            )
        else:
            route_id = int(self.active_route_id)
            route_kind = str(self.active_route_kind or "")
            intent_source = str(self.active_intent_source or "unknown")
            intent_priority = int(self.active_intent_priority)
            action_generation = int(self.action_generation)
        terminal_goal = copy.deepcopy(source_goal)
        terminal_goal.header.stamp = rospy.Time.now()
        self.last_terminal_goal = copy.deepcopy(terminal_goal)
        self.terminal_pub.publish(terminal_goal)

        terminal = FrontierExecutionTerminal()
        is_frontier = (
            intent_source == "global_slam_frontier"
            and intent_priority == 0
            and route_id > 0
        )
        terminal.route_id = route_id if is_frontier else 0
        terminal.action_generation = action_generation
        terminal.route_kind = (
            route_kind or "frontier_endpoint"
            if is_frontier else "non_frontier"
        )
        terminal.goal = terminal_goal
        self.terminal_contract_pub.publish(terminal)
        return terminal_goal

    def _close_observation_region_terminal_locked(self, generation, status):
        completion = self.frontier_observation_completion_pending
        if (
            completion is None
            or int(completion.get("generation", -1)) != int(generation)
        ):
            return False
        # The action was intentionally cancelled after the endpoint's
        # observation region was completed.  Its PREEMPTED result is a
        # successful mission-level terminal, not an unreachable route.
        self.frontier_observation_completion_pending = None
        self.action_active = False
        self.last_result_status = int(status)
        self.last_result_monotonic = time.monotonic()
        self.frontier_observation_completion_count += 1
        self.terminal_count += 1
        self.publish_bridge_status(
            "terminal",
            status=int(GoalStatus.SUCCEEDED),
            status_text="FRONTIER_OBSERVATION_COMPLETE",
            move_base_status=int(status),
            route_id=int(completion["route_id"]),
            feedback_distance=round(float(completion["feedback_distance"]), 3),
            completion_radius=round(
                float(completion.get(
                    "completion_radius",
                    self.frontier_observation_completion_radius,
                )), 3
            ),
            pending_delta=round(float(completion["pending_delta"]), 3),
            lifecycle="observation_region_terminal",
        )
        self._clear_action_health_locked()
        self._clear_failed_route_lease_locked()
        self.schedule_terminal_dispatch_locked()
        rospy.loginfo(
            "TEB goal bridge closed frontier observation terminal: "
            "move_base_status=%s route_id=%d",
            GoalStatus.to_string(status),
            int(completion["route_id"]),
        )
        return True

    def _retain_turn_terminal_until_yaw_complete_locked(
        self, status, action_contract=None
    ):
        route_kind = (
            str(action_contract.get("route_kind", "") or "")
            if action_contract is not None
            else self.active_route_kind
        )
        if not (
            status == GoalStatus.SUCCEEDED
            and route_kind == TURN_ROUTE_KIND
            and self.turn_supervisor_state != "PASS_THROUGH"
        ):
            return False
        self.last_result_status = int(status)
        self.last_result_monotonic = time.monotonic()
        self.move_base_terminal_pending = True
        self.publish_bridge_status(
            "turn_execution_terminal",
            status=int(status),
            status_text="SUCCEEDED_XY_WAITING_YAW",
            **self._action_contract_status_fields(action_contract),
        )
        rospy.loginfo(
            "TEB connector reached XY terminal; retaining logical action "
            "until turn supervisor completes yaw"
        )
        return True

    def _close_persistent_task_terminal_locked(
        self, status, action_contract=None
    ):
        if not (
            self.persistent_execution
            and status == GoalStatus.SUCCEEDED
            and self.task_done
        ):
            return False
        self.publish_bridge_status(
            "persistent_task_complete_terminal",
            status=int(status),
            status_text="TASK_DONE",
            **self._action_contract_status_fields(action_contract),
        )
        self._clear_action_health_locked()
        self._clear_failed_route_lease_locked()
        return True

    def _record_target_terminal_state_locked(self, status, action_contract=None):
        active_priority = (
            int(action_contract.get("priority", 0) or 0)
            if action_contract is not None
            else int(self.active_intent_priority)
        )
        if status == GoalStatus.SUCCEEDED and active_priority < 2:
            if self.target_failure_latched:
                self._clear_target_failure_locked("frontier_progress")
            return
        if status != GoalStatus.SUCCEEDED and active_priority >= 2:
            self._latch_target_failure_locked(
                "move_base_%s" % GoalStatus.to_string(status).lower(),
                status_text=GoalStatus.to_string(status),
                cancel_action=False,
                action_contract=action_contract,
            )

    def _publish_successful_terminal_locked(
        self, source_goal, status, action_contract=None
    ):
        if status != GoalStatus.SUCCEEDED or source_goal is None:
            return False
        terminal_goal = self._publish_execution_terminal_locked(
            source_goal, action_contract=action_contract
        )
        self.terminal_count += 1
        self.publish_bridge_status(
            "terminal",
            status=int(status),
            status_text="SUCCEEDED",
            **self._action_contract_status_fields(action_contract),
            terminal_goal=[
                round(terminal_goal.pose.position.x, 3),
                round(terminal_goal.pose.position.y, 3),
            ],
        )
        rospy.loginfo(
            "TEB goal bridge successful terminal event: target=(%.2f,%.2f)",
            terminal_goal.pose.position.x,
            terminal_goal.pose.position.y,
        )
        self._clear_action_health_locked()
        self._clear_failed_route_lease_locked()
        # Actionlib invokes done_cb before its own client handle reaches DONE.
        # The timer is the only authorized post-success dispatch boundary.
        self.schedule_terminal_dispatch_locked()
        return True

    def _publish_failed_terminal_locked(
        self, status, handoff_requested, action_contract=None
    ):
        """Report a failure once, then leave route replacement to Frontier."""
        self.publish_bridge_status(
            "terminal",
            status=int(status),
            status_text=GoalStatus.to_string(status),
            **self._action_contract_status_fields(action_contract),
        )
        rospy.logwarn(
            "TEB move_base action finished without success: status=%s "
            "handoff=%s route_id=%s generation=%s",
            GoalStatus.to_string(status),
            handoff_requested,
            None
            if action_contract is None
            else action_contract.get("route_id"),
            None
            if action_contract is None
            else action_contract.get("generation"),
        )
        self._remember_failed_route_lease_locked(action_contract)
        self._clear_action_health_locked()
        # Only an explicit higher-priority cancellation owns a pending
        # successor.  A failed frontier route remains idle until Global
        # Frontier publishes a replacement route ID.
        if handoff_requested and self._is_active_mode() and self.latest_goal is not None:
            self.schedule_terminal_dispatch_locked()

    def on_done(self, generation, status, _result):
        """Close one action; do not send actions from the actionlib callback."""
        with self.lock:
            if generation != self.action_generation:
                return
            action_contract = self._action_contract_for_generation_locked(generation)
            if action_contract is None:
                rospy.logwarn_throttle(
                    3.0,
                    "TEB goal bridge ignored action terminal without dispatch "
                    "contract: generation=%s",
                    generation,
                )
                return
            if self._close_observation_region_terminal_locked(generation, status):
                return
            if self._retain_turn_terminal_until_yaw_complete_locked(
                status, action_contract
            ):
                return

            self.action_active = False
            self.last_result_status = int(status)
            self.last_result_monotonic = time.monotonic()
            if self._close_persistent_task_terminal_locked(
                status, action_contract
            ):
                return
            self._record_target_terminal_state_locked(status, action_contract)
            handoff_requested = self.handoff_requested
            self.handoff_requested = False
            source_goal = copy.deepcopy(action_contract.get("source_goal"))
            if self._publish_successful_terminal_locked(
                source_goal, status, action_contract
            ):
                return
            self._publish_failed_terminal_locked(
                status, handoff_requested, action_contract
            )
