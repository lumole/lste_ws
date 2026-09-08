"""Top-level action dispatch composition for the TEB goal bridge.

The façade keeps the historical import stable.  Focused siblings contain the
active-action policy, idle retry policy, and concrete action-client calls.
"""

import rospy

from teb_goal_bridge_action_active_dispatch import (
    TebGoalBridgeActionActiveDispatchMixin,
)
from teb_goal_bridge_action_client import TebGoalBridgeActionClientMixin
from teb_goal_bridge_action_retry_policy import TebGoalBridgeActionRetryPolicyMixin


class TebGoalBridgeActionDispatchMixin(
    TebGoalBridgeActionActiveDispatchMixin,
    TebGoalBridgeActionRetryPolicyMixin,
    TebGoalBridgeActionClientMixin,
):
    def dispatch_locked(self, force, reason):
        """Dispatch the latest intent after common lifecycle admission checks."""
        if self.latest_goal is None or not self._is_active_mode():
            return
        if (
            not force
            and self.frontier_portal_wait
            and self.latest_intent_source == "global_slam_frontier"
            and self.latest_intent_priority == 0
            and self.latest_route_id <= self.frontier_portal_wait_route_id
        ):
            rospy.loginfo_throttle(
                3.0,
                "TEB goal bridge waits for a newer portal transaction after "
                "frontier route_id=%d",
                self.frontier_portal_wait_route_id,
            )
            return
        # A goal-manager subscriber can run before SimpleActionClient has
        # reached DONE.  The one-shot timer is a synchronization barrier, not
        # a planning delay.  Explicit controller transitions retain force.
        if (
            not force
            and self.terminal_dispatch_timer is not None
            and reason != "terminal_followup"
        ):
            return
        if not self._action_server_ready():
            return
        if not force and self._target_failure_blocks_latest_locked():
            rospy.loginfo_throttle(
                3.0,
                "TEB goal bridge holds failed target transaction until "
                "frontier progress or a new target epoch: epoch=%d",
                self.target_failure_epoch,
            )
            return

        if self.action_active and not force:
            self._dispatch_active_action_locked(reason)
            return
        self._dispatch_inactive_action_locked(force, reason)
