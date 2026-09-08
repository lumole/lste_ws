"""Validation of terminal messages received from the TEB goal bridge."""

import math

import rospy


class GlobalFrontierTerminalValidationMixin:
    """Accept only the terminal notification for the active route contract."""

    def matching_execution_terminal(self, message):
        """Return a completion only for this exact route transaction.

        Adjacent frontier endpoints can be closer than any useful geometric
        acceptance radius. A pose-only terminal can therefore close the next
        action when ROS delivers the previous callback late. The bridge sends
        an immutable ``route_id`` contract; coordinate checks remain a
        diagnostic guard, never the primary identity proof.
        """
        if self.task_done or self.active_last_waypoint_map is None:
            return None
        try:
            terminal_route_id = int(message.route_id)
        except (AttributeError, TypeError, ValueError):
            return None
        if terminal_route_id != int(self.active_route_id):
            rospy.loginfo_throttle(
                2.0,
                "Global frontier ignored stale execution terminal route_id=%d "
                "active_route_id=%d",
                terminal_route_id,
                self.active_route_id,
            )
            return None
        terminal_route_kind = str(
            getattr(message, "route_kind", "") or ""
        ).strip()
        if (
            terminal_route_kind
            and terminal_route_kind != str(self.active_route_kind or "")
        ):
            rospy.loginfo_throttle(
                2.0,
                "Global frontier ignored execution terminal route-kind mismatch "
                "route_id=%d terminal=%s active=%s",
                terminal_route_id,
                terminal_route_kind,
                self.active_route_kind,
            )
            return None
        terminal_goal = getattr(message, "goal", None)
        if terminal_goal is None:
            return None
        frame = (terminal_goal.header.frame_id or "").strip().lstrip("/")
        expected_frame = (
            "" if self.map_msg is None
            else (self.map_msg.header.frame_id or "map").strip().lstrip("/")
        )
        expected_xy = self.active_last_waypoint_map
        frozen_portal_odom = getattr(
            self, "active_portal_destination_odom_xy", None,
        )
        if (
            self.active_route_kind == "portal_transition"
            and frozen_portal_odom is not None
        ):
            # A portal is a physical graph edge. Its command is intentionally
            # frozen in odom, so the matching terminal must be checked against
            # that same endpoint rather than a later SLAM-corrected map goal.
            expected_frame = "odom"
            expected_xy = frozen_portal_odom
        if frame and expected_frame and frame != expected_frame:
            rospy.logwarn_throttle(
                3.0,
                "Global frontier ignored execution terminal frame mismatch "
                "route_id=%d terminal=%s expected=%s",
                terminal_route_id,
                frame,
                expected_frame,
            )
            return None
        terminal_delta = math.hypot(
            float(terminal_goal.pose.position.x) - float(expected_xy[0]),
            float(terminal_goal.pose.position.y) - float(expected_xy[1]),
        )
        if terminal_delta > 0.25:
            rospy.logwarn_throttle(
                3.0,
                "Global frontier ignored unmatched execution terminal "
                "delta=%.2fm frame=%s active=(%.2f,%.2f)",
                terminal_delta,
                expected_frame,
                expected_xy[0],
                expected_xy[1],
            )
            return None
        return self.active_frontier, terminal_delta
