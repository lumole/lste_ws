"""Persistent target-plan transaction policy for the TEB goal bridge.

The composition root owns ROS publishers, subscribers, and the MoveBase action
lease.  This mixin owns only the typed transaction exchanged with the optional
persistent Navfn executor: request a target plan, accept the matching plan, or
clear the request.  Keeping it separate makes the ordinary endpoint-action
path readable without changing its public ROS interface.
"""

import copy

import rospy
from geometry_msgs.msg import PoseStamped
from lste_topo_access.msg import PersistentGoalCommand


class TebGoalBridgePersistentTargetMixin:
    def _install_persistent_target_locked(self, transaction_id, message):
        """Commit a Navfn acknowledgement for exactly one target request."""
        if not self.persistent_execution:
            return
        if self.latest_goal is None:
            self.publish_bridge_status(
                "persistent_target_install_ignored",
                reason="no_current_mission",
                received_transaction_id=transaction_id,
                latest_transaction_id=int(self.latest_goal_transaction_id),
            )
            return
        latest_global = self._goal_in_global_frame(self.latest_goal)
        if (
            self.latest_intent_priority < 2
            or transaction_id <= 0
            or transaction_id != int(self.latest_goal_transaction_id)
            or transaction_id != int(self.persistent_target_pending_transaction)
            or self.target_failure_latched
            or latest_global is None
            or not self._same_goal(latest_global, message)
        ):
            self.publish_bridge_status(
                "persistent_target_install_ignored",
                received_transaction_id=transaction_id,
                latest_transaction_id=int(self.latest_goal_transaction_id),
            )
            return
        self.persistent_installed_target_goal = copy.deepcopy(message)
        self.persistent_installed_target_transaction = int(transaction_id)
        self.persistent_target_pending_transaction = 0
        self.persistent_target_pending_goal = None
        # Publish the approved mission before clearing the speculative request,
        # so the persistent C++ planner can atomically promote the same route.
        self._publish_persistent_mission_goal_locked("target_plan_installed")
        self._clear_persistent_target_request_locked("target_plan_installed")
        if self.action_active:
            self._adopt_persistent_mission_goal_locked()
        self.publish_bridge_status(
            "persistent_target_plan_installed",
            transaction_id=transaction_id,
            target_goal=[
                round(float(message.pose.position.x), 3),
                round(float(message.pose.position.y), 3),
            ],
        )

    @staticmethod
    def _persistent_goal_command(kind, transaction_id, goal):
        """Build the typed cross-node ownership command."""
        command = PersistentGoalCommand()
        command.kind = int(kind)
        command.transaction_id = max(0, int(transaction_id))
        command.goal = copy.deepcopy(goal)
        return command

    def _request_persistent_target_locked(self, reason):
        """Request Navfn validation without promoting an unverified target."""
        transaction_id = int(self.latest_goal_transaction_id)
        if self.latest_goal is None or transaction_id <= 0:
            return False
        target_request = copy.deepcopy(self.latest_goal)
        target_request.header.stamp = rospy.Time.now()
        self.persistent_target_pending_transaction = transaction_id
        self.persistent_target_pending_goal = copy.deepcopy(target_request)
        self.persistent_target_request_transaction = transaction_id
        self.persistent_target_goal_pub.publish(target_request)
        self.persistent_target_command_pub.publish(
            self._persistent_goal_command(
                PersistentGoalCommand.KIND_TARGET_REQUEST,
                transaction_id,
                target_request,
            )
        )
        self.publish_bridge_status(
            "persistent_target_plan_requested",
            reason=str(reason),
            transaction_id=transaction_id,
            target_goal=[
                round(float(target_request.pose.position.x), 3),
                round(float(target_request.pose.position.y), 3),
            ],
        )
        return True

    def _clear_persistent_target_request_locked(self, reason, force=False):
        """Overwrite the latched speculative target with an explicit tombstone."""
        if not self.persistent_execution:
            return False
        if (
            not force
            and self.persistent_target_request_transaction <= 0
            and self.persistent_target_pending_transaction <= 0
        ):
            return False
        tombstone = PoseStamped()
        tombstone.header.stamp = rospy.Time.now()
        tombstone.header.frame_id = self.global_frame
        self.persistent_target_goal_pub.publish(tombstone)
        # The request transaction is zero after Navfn has installed a target,
        # but the installed target still owns the persistent planner.  Use the
        # newest known target identity for the tombstone so both the global and
        # local persistent plugins clear an approved route as well as a pending
        # speculative request.
        cleared_transaction = max(
            int(getattr(self, "persistent_target_request_transaction", 0) or 0),
            int(getattr(self, "persistent_target_pending_transaction", 0) or 0),
            int(getattr(self, "persistent_installed_target_transaction", 0) or 0),
            int(getattr(self, "latest_goal_transaction_id", 0) or 0)
            if str(getattr(self, "latest_intent_source", "") or "") in (
                "target_parallax",
                "target_follow",
                "target_terminal_advance",
                "target_terminal_observation",
            )
            else 0,
        )
        self.persistent_target_command_pub.publish(
            self._persistent_goal_command(
                PersistentGoalCommand.KIND_CLEAR,
                cleared_transaction,
                tombstone,
            )
        )
        self.persistent_target_request_transaction = 0
        self.publish_bridge_status(
            "persistent_target_request_cleared",
            reason=str(reason),
            cleared_transaction_id=cleared_transaction,
        )
        return True

    def _publish_persistent_mission_goal_locked(self, reason):
        """Publish one bridge-approved route to the persistent Navfn planner."""
        if not self.persistent_execution or self.latest_goal is None:
            return False
        approved = copy.deepcopy(self.latest_goal)
        approved.header.stamp = rospy.Time.now()
        self.persistent_mission_goal_pub.publish(approved)
        self.persistent_mission_command_pub.publish(
            self._persistent_goal_command(
                PersistentGoalCommand.KIND_MISSION,
                int(self.latest_goal_transaction_id),
                approved,
            )
        )
        self.publish_bridge_status(
            "persistent_mission_goal_approved",
            reason=str(reason),
            transaction_id=int(self.latest_goal_transaction_id),
            priority=int(self.latest_intent_priority),
            goal=[
                round(float(approved.pose.position.x), 3),
                round(float(approved.pose.position.y), 3),
            ],
        )
        return True

    def _republish_persistent_target_request_locked(self, reason):
        """Request installation of the current target path once per transaction."""
        transaction_id = int(self.latest_goal_transaction_id)
        if (
            self.latest_goal is None
            or self.latest_intent_priority < 2
            or transaction_id <= 0
            or self.persistent_target_republish_transaction == transaction_id
        ):
            return False
        self._request_persistent_target_locked(reason)
        self.persistent_target_republish_transaction = transaction_id
        self.publish_bridge_status(
            "persistent_target_reinstall_requested",
            reason=reason,
            transaction_id=transaction_id,
            target_goal=[
                round(self.latest_goal.pose.position.x, 3),
                round(self.latest_goal.pose.position.y, 3),
            ],
        )
        return True

    def _goal_from_persistent_result(self, payload):
        """Decode the pose embedded in a C++ planner result event."""
        raw_goal = payload.get("goal")
        if not isinstance(raw_goal, (list, tuple)) or len(raw_goal) < 2:
            return None
        try:
            x, y = float(raw_goal[0]), float(raw_goal[1])
        except (TypeError, ValueError):
            return None
        frame = str(payload.get("frame_id", self.global_frame)).strip().lstrip("/")
        if not frame:
            frame = self.global_frame
        result = PoseStamped()
        result.header.stamp = rospy.Time.now()
        result.header.frame_id = frame
        result.pose.position.x = x
        result.pose.position.y = y
        result.pose.orientation.w = 1.0
        return result
