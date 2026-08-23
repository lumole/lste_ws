#!/usr/bin/env python3
"""Own the move_base action lifecycle for the TEB controller.

``/lste/final_goal`` is intentionally a topic because the existing LSTE goal
manager and RViz/Gazebo tools use it.  It is not a velocity setpoint, however:
one ``PoseStamped`` starts (and normally replaces) a ``move_base`` action.
This bridge turns the topic into an explicit action contract:

* one goal remains active until ``move_base`` returns a result;
* newer route-planning goals are coalesced while that action is healthy;
* a higher-priority mission intent (a confirmed visual target) may take over a
  lower-priority frontier action by replacing the action goal in-place;
* feedback-based health checks may hand off a stale frontier action when the
  vehicle is no longer making progress toward it;
* controller/task lifecycle events are the only normal reasons to cancel;
* the action server, rather than a hand-written ``/move_base/status`` parser,
  owns goal state and result delivery.

The result is the same public LSTE interface with a smaller, well-defined
boundary between mission decisions and trajectory execution.  The bridge
publishes machine-readable lifecycle events for navigation metrics and emits a
successful terminal source goal for the frontier manager.
"""

import copy
import json
import math
import threading
import time

import actionlib
import rospy
import tf
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Bool, String


TURN_ROUTE_KIND = "frontier_turn_connector"


class TebGoalBridge:
    def __init__(self):
        rospy.init_node("lste_teb_goal_bridge")
        gp = rospy.get_param

        self.goal_topic = gp("~goal_topic", "/lste/final_goal")
        # Kept as a compatibility parameter for old launch files.  New goals
        # are sent through the typed MoveBaseAction client below, not through
        # the simple-goal replacement topic.
        self.simple_goal_topic = gp("~simple_goal_topic", "/move_base_simple/goal")
        self.cancel_topic = gp("~cancel_topic", "/move_base/cancel")
        self.terminal_topic = gp("~terminal_topic", "/lste/teb_goal_terminal")
        # A failed visual target is a mission event, not a request to retry the
        # same move_base transaction.  Goal Manager consumes this event and
        # releases ownership back to the map frontier.
        self.target_failure_topic = gp(
            "~target_failure_topic", "/lste/teb_goal_failure"
        )
        self.intent_topic = gp("~intent_topic", "/lste/goal_intent")
        # In the production pipeline every pose is paired with a JSON intent.
        # Waiting for that pair prevents a latched pose from being dispatched
        # with ``unknown`` route semantics when ROS delivers the two latches
        # in the opposite order. Plain external pose publishers can opt out.
        self.require_intent = self._as_bool(gp("~require_intent", True))
        self.frontier_status_topic = gp(
            "~frontier_status_topic", "/lste/global_frontier/status"
        )
        self.controller_topic = gp("~controller_topic", "/lste/controller_mode")
        self.task_done_topic = gp("~task_done_topic", "/lste/task_done")
        self.bridge_status_topic = gp(
            "~bridge_status_topic", "/lste/teb_goal_bridge/status"
        )
        self.turn_supervisor_status_topic = gp(
            "~turn_supervisor_status_topic", "/lste/teb_turn_supervisor/status"
        )
        self.global_frame = str(gp("~global_frame", "map")).strip().lstrip("/") or "map"
        self.active_mode = str(gp("~active_mode", "teb")).strip().lower()
        self.mode = str(gp("~initial_mode", "teb")).strip().lower()
        # A MoveBaseAction is the execution transaction.  In-place
        # ``send_goal`` replacement looks attractive for smoothness, but it
        # makes the client stop tracking the old goal while move_base is still
        # publishing feedback/result for it.  That leaves the bridge, TEB
        # global plan and mission layer with different goal identities.  Keep
        # replacement disabled in the production contract: queue the newest
        # mission goal and hand it over only after an explicit terminal result.
        self.allow_in_place_replacement = self._as_bool(
            gp("~allow_in_place_replacement", False)
        )
        # Route continuation is a narrower contract than generic in-place
        # replacement.  It is safe only for adjacent, already validated
        # frontier segments; mission target takeovers and branch changes must
        # still wait for an action terminal so ownership remains unambiguous.
        self.allow_route_continuation_replacement = self._as_bool(
            gp("~allow_route_continuation_replacement", True)
        )
        self.position_epsilon = max(0.001, float(gp("~position_epsilon", 0.05)))
        self.yaw_epsilon = max(0.001, float(gp("~yaw_epsilon", 0.08)))
        compare_goal_yaw = gp("~compare_goal_yaw", False)
        self.compare_goal_yaw = str(compare_goal_yaw).strip().lower() in (
            "1", "true", "yes", "on",
        )
        self.min_update_interval = max(0.0, float(gp("~min_update_interval", 1.5)))
        self.goal_retry_interval = max(
            self.min_update_interval, float(gp("~goal_retry_interval", 2.0))
        )
        # A frontier is a mission-level exploration intent, not a promise to
        # stop forever at one map cell. If the map layer publishes a materially
        # newer intent after move_base has made no physical progress for a
        # while, hand the action over in a controlled recovery cycle. This
        # keeps action ownership in this bridge without racing a normal
        # near-endpoint SUCCEEDED callback. Visual target updates use the
        # separate in-place replacement path below.
        self.handoff_distance = max(
            self.position_epsilon * 2.0,
            float(gp("~handoff_distance", 1.0)),
        )
        # Accepted for compatibility with older launch files. Near-endpoint
        # handoff is intentionally disabled; only progress_timeout can cancel
        # an active action.
        self.handoff_radius = max(
            self.position_epsilon * 2.0,
            float(gp("~handoff_radius", 0.85)),
        )
        self.progress_timeout = max(
            2.0, float(gp("~progress_timeout", 12.0))
        )
        self.progress_epsilon = max(
            0.01, float(gp("~progress_epsilon", 0.12))
        )
        self.handoff_min_interval = max(
            self.min_update_interval,
            float(gp("~handoff_min_interval", 4.0)),
        )
        # Frontier and confirmed visual targets are streamed as bounded route
        # segments. When the robot is close enough to the active segment and a
        # safe next segment is waiting, replace the action goal in-place before
        # move_base reaches its terminal tolerance. The move_base action server
        # accepts a newer goal without the explicit-cancel path, so TEB can
        # continue its command stream instead of inserting a zero-velocity gap.
        self.target_early_handoff_distance = max(
            self.position_epsilon * 2.0,
            float(gp("~target_early_handoff_distance", 0.70)),
        )
        self.target_early_handoff_min_delta = max(
            self.position_epsilon * 2.0,
            float(gp("~target_early_handoff_min_delta", 0.40)),
        )
        # TEB resets its timed elastic band when a new goal jumps farther than
        # this distance. Replacing an action during that reset can expose a
        # zero TimeDiff to getVelocityCommand(); let the current action finish
        # for those branch-sized jumps and reserve in-place replacement for
        # hot-startable route segments.
        self.in_place_replacement_max_delta = max(
            self.target_early_handoff_min_delta,
            float(rospy.get_param(
                "/move_base/TebLocalPlannerROS/force_reinit_new_goal_dist", 4.0
            )),
        )
        teb_xy_goal_tolerance = max(
            self.position_epsilon,
            float(rospy.get_param(
                "/move_base/TebLocalPlannerROS/xy_goal_tolerance", 0.35
            )),
        )
        # Inside this band TEB is already shrinking the trajectory to satisfy
        # its terminal tolerance; replacing the action there can expose a
        # zero first TimeDiff for one controller cycle.
        self.in_place_replacement_min_distance = max(
            self.position_epsilon * 2.0,
            teb_xy_goal_tolerance * 1.8,
        )
        # Frontier updates are already filtered by Goal Manager and represent
        # the next map segment, so a small but real shift is useful. Keep the
        # visual target threshold stricter because its bearing is detector
        # driven and should not chase bbox noise.
        self.frontier_replacement_min_delta = max(
            self.position_epsilon * 2.0, 0.10
        )
        # A frontier branch is a new exploration direction, not a short visual
        # servo segment. Replacing it while the old endpoint is only a few
        # decimetres away makes TEB throw away its nearly finished band and
        # brake sharply. The frontier explorer now exposes a validated branch
        # early; the derived distance/delta window below accepts that handoff
        # while there is still room to turn, and leaves oversized jumps queued.
        self.frontier_replacement_max_delta = max(
            self.frontier_replacement_min_delta,
            float(gp("~frontier_replacement_max_delta", 1.0)),
        )
        # The frontier explorer now exposes receding-horizon route points.
        # Their separation may be larger than the old terminal frontier
        # epsilon, but it is still bounded by TEB's own warm-start distance.
        # The heading-continuity gate below distinguishes this route
        # continuation from a genuinely new branch topology.
        self.frontier_early_handoff_max_delta = max(
            self.frontier_replacement_max_delta,
            self.in_place_replacement_max_delta,
        )
        self.frontier_early_handoff_max_heading_delta = math.radians(max(
            1.0, float(gp("~frontier_early_handoff_max_heading_deg", 75.0))
        ))
        self.in_place_replacement_max_distance = max(
            self.target_early_handoff_distance,
            teb_xy_goal_tolerance * 3.0,
        )
        # A prefetched frontier branch is handed over while the old endpoint
        # is still about a metre away, but only when it is a short, directionally
        # continuous continuation.  Large or sharp branch changes wait for the
        # current action result, so TEB does not throw away a usable band while
        # the car is still approaching the old endpoint.
        self.frontier_early_handoff_min_distance = max(
            self.in_place_replacement_min_distance,
            0.85,
        )
        self.frontier_early_handoff_max_distance = max(
            self.in_place_replacement_max_distance,
            self.frontier_early_handoff_min_distance + 0.45,
        )
        # Do not replace on the exact terminal boundary: move_base may report
        # SUCCEEDED on the same callback turn. A small margin outside TEB's
        # xy tolerance is enough to let a sharp branch take over without
        # waiting for the old endpoint to settle at zero velocity.
        self.frontier_sharp_replacement_min_distance = max(
            self.position_epsilon * 2.0,
            teb_xy_goal_tolerance * 1.10,
        )
        # A sharp frontier branch can become visible while the current action
        # is already at its endpoint. Cancelling that action immediately
        # publishes a zero command before the next goal is accepted. Give
        # move_base a short chance to report success, then use actionlib's
        # native replacement path so the local planner can continue its
        # command stream. Distant stale actions still use explicit cancel.
        self.frontier_stale_recovery_grace = max(
            0.0, float(gp("~frontier_stale_recovery_grace", 4.0))
        )
        self.frontier_stale_recovery_max_distance = max(
            self.frontier_early_handoff_max_distance,
            float(gp("~frontier_stale_recovery_max_distance", 1.50)),
        )
        rospy.set_param(
            "~in_place_replacement_max_delta", self.in_place_replacement_max_delta
        )
        rospy.set_param(
            "~frontier_replacement_max_delta", self.frontier_replacement_max_delta
        )
        rospy.set_param(
            "~in_place_replacement_min_distance", self.in_place_replacement_min_distance
        )
        rospy.set_param(
            "~in_place_replacement_max_distance", self.in_place_replacement_max_distance
        )
        rospy.set_param(
            "~frontier_replacement_min_delta", self.frontier_replacement_min_delta
        )
        rospy.set_param(
            "~frontier_early_handoff_max_delta", self.frontier_early_handoff_max_delta
        )
        rospy.set_param(
            "~frontier_early_handoff_max_heading_deg",
            math.degrees(self.frontier_early_handoff_max_heading_delta),
        )
        rospy.set_param(
            "~frontier_stale_recovery_grace", self.frontier_stale_recovery_grace
        )
        rospy.set_param(
            "~frontier_stale_recovery_max_distance",
            self.frontier_stale_recovery_max_distance,
        )
        rospy.set_param(
            "~frontier_sharp_replacement_min_distance",
            self.frontier_sharp_replacement_min_distance,
        )

        self.lock = threading.RLock()
        self.latest_goal = None
        self.last_dispatched_goal = None
        self.last_terminal_goal = None
        self.last_dispatch_monotonic = 0.0
        self.last_result_monotonic = 0.0
        self.last_result_status = None
        self.task_done = False
        self.action_active = False
        self.action_server_seen = False
        self.deferred_goal_updates = 0
        self.deferred_goal_log_wall = 0.0
        # The mission topic is latched and the timer also reevaluates it. Keep
        # the last queued transaction identity so an unchanged goal is not
        # counted as a fresh route update on every health tick.
        self.deferred_signature = None
        self.bridge_events = 0
        self.dispatch_count = 0
        self.terminal_count = 0
        # A generation makes late callbacks from a cancelled action harmless.
        self.action_generation = 0
        self.active_goal_global = None
        self.active_feedback_distance = None
        self.active_feedback_pose = None
        self.active_feedback_frame = ""
        # Retain the most recent execution pose across an action terminal.
        # A queued turn connector is a pure in-place action; its mission-layer
        # coordinates may be stale by the time the previous route terminates.
        self.last_feedback_pose_global = None
        self.feedback_transform_failures = 0
        self.active_best_distance = None
        self.active_progress_monotonic = 0.0
        self.last_feedback_monotonic = 0.0
        self.handoff_requested = False
        self.handoff_count = 0
        self.priority_handoff_count = 0
        self.target_segment_handoff_count = 0
        self.frontier_segment_handoff_count = 0
        self.frontier_stale_recovery_count = 0
        # Explicit target lifecycle.  A latched failure blocks only the exact
        # target transaction that failed; a lower-priority frontier intent or
        # a newer target observation clears it.  This is the bridge-side guard
        # that makes infinite cancel/retry impossible.
        self.target_failure_latched = False
        self.target_failure_goal = None
        self.target_failure_epoch = 0
        self.target_failure_track_id = ""
        self.target_failure_generation = 0
        self.target_failure_count = 0
        self.frontier_stale_wait_started_monotonic = 0.0
        self.handoff_log_monotonic = 0.0
        self.latest_intent_source = "unknown"
        self.latest_intent_priority = 0
        self.latest_route_kind = ""
        self.latest_target_epoch = 0
        self.latest_target_track_id = ""
        # Goal Manager publishes intent metadata immediately before the pose,
        # but ROS callback scheduling can still deliver the two messages in
        # the opposite order. Keep the intent coordinates as a transaction
        # key so a PoseStamped is never dispatched with stale route semantics.
        self.latest_intent_goal = None
        self.intent_seen = False
        self.active_intent_source = "unknown"
        self.active_intent_priority = 0
        self.active_route_kind = ""
        self.active_target_epoch = 0
        self.active_target_track_id = ""
        self.turn_supervisor_state = "UNKNOWN"
        self.turn_supervisor_last_event = "unknown"
        # A turn connector has two completion boundaries.  move_base may
        # report XY success as soon as the connector position is reached,
        # while the turn supervisor still owns the yaw contract.  Keep the
        # bridge transaction logically active during that interval so the
        # supervisor is not released by an intermediate action callback.
        self.move_base_terminal_pending = False
        # A completed atomic turn authorizes one phase transition from the
        # execution-local connector to its already validated frontier
        # endpoint. It is consumed by the next endpoint dispatch.
        self.turn_transition_ready = False
        self.tf_listener = tf.TransformListener()
        self.action_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        # A done callback runs before actionlib finishes its own client-side
        # state transition. Schedule the next mission goal one controller
        # cycle later instead of waiting for the 0.2 s health timer; this
        # removes an avoidable terminal-to-next-goal zero-velocity gap without
        # reintroducing the callback race that motivated the bridge.
        self.terminal_dispatch_timer = None

        self.terminal_pub = rospy.Publisher(
            self.terminal_topic, PoseStamped, queue_size=1
        )
        self.target_failure_pub = rospy.Publisher(
            self.target_failure_topic, String, queue_size=10
        )
        self.bridge_status_pub = rospy.Publisher(
            self.bridge_status_topic, String, queue_size=10, latch=True
        )
        rospy.Subscriber(self.goal_topic, PoseStamped, self.on_goal, queue_size=1)
        rospy.Subscriber(self.intent_topic, String, self.on_intent, queue_size=1)
        rospy.Subscriber(
            self.frontier_status_topic,
            String,
            self.on_frontier_status,
            queue_size=10,
        )
        rospy.Subscriber(
            self.turn_supervisor_status_topic,
            String,
            self.on_turn_supervisor_status,
            queue_size=10,
        )
        rospy.Subscriber(self.controller_topic, String, self.on_mode, queue_size=1)
        rospy.Subscriber(self.task_done_topic, Bool, self.on_task_done, queue_size=1)
        rospy.Timer(rospy.Duration(0.2), self.on_timer)

        rospy.loginfo(
            "TEB goal bridge ready: action=move_base mode=%s active_mode=%s "
            "goal=%s global_frame=%s",
            self.mode,
            self.active_mode,
            self.goal_topic,
            self.global_frame,
        )

    def publish_bridge_status(self, event, **fields):
        payload = {
            "event": str(event),
            "mode": self.mode,
            "active": bool(self.action_active),
            "allow_in_place_replacement": bool(self.allow_in_place_replacement),
            "allow_route_continuation_replacement": bool(
                self.allow_route_continuation_replacement
            ),
            "result_status": self.last_result_status,
            "deferred_goal_updates": int(self.deferred_goal_updates),
            "dispatches": int(self.dispatch_count),
            "terminals": int(self.terminal_count),
            "active_intent_source": self.active_intent_source,
            "active_intent_priority": int(self.active_intent_priority),
            "latest_intent_source": self.latest_intent_source,
            "latest_intent_priority": int(self.latest_intent_priority),
            "require_intent": bool(self.require_intent),
            "intent_seen": bool(self.intent_seen),
            "latest_intent_goal": (
                None
                if self.latest_intent_goal is None
                else [
                    round(float(self.latest_intent_goal[0]), 3),
                    round(float(self.latest_intent_goal[1]), 3),
                ]
            ),
            "active_route_kind": self.active_route_kind,
            "latest_route_kind": self.latest_route_kind,
            "active_goal": (
                None
                if self.active_goal_global is None
                else [
                    round(float(self.active_goal_global.pose.position.x), 3),
                    round(float(self.active_goal_global.pose.position.y), 3),
                    round(float(self._yaw(self.active_goal_global)), 4),
                ]
            ),
            "active_goal_frame": (
                None
                if self.active_goal_global is None
                else self.active_goal_global.header.frame_id
            ),
            "active_source_goal": (
                None
                if self.last_dispatched_goal is None
                else [
                    round(float(self.last_dispatched_goal.pose.position.x), 3),
                    round(float(self.last_dispatched_goal.pose.position.y), 3),
                    round(float(self._yaw(self.last_dispatched_goal)), 4),
                ]
            ),
            "active_source_goal_frame": (
                None
                if self.last_dispatched_goal is None
                else self.last_dispatched_goal.header.frame_id
            ),
            "turn_supervisor_state": self.turn_supervisor_state,
            "turn_supervisor_last_event": self.turn_supervisor_last_event,
            "turn_transition_ready": bool(self.turn_transition_ready),
            "move_base_terminal_pending": bool(self.move_base_terminal_pending),
            "active_feedback_frame": self.active_feedback_frame,
            "feedback_transform_failures": int(self.feedback_transform_failures),
            "priority_handoffs": int(self.priority_handoff_count),
            "target_segment_handoffs": int(self.target_segment_handoff_count),
            "frontier_segment_handoffs": int(self.frontier_segment_handoff_count),
            "active_target_epoch": int(self.active_target_epoch),
            "latest_target_epoch": int(self.latest_target_epoch),
            "active_target_track_id": self.active_target_track_id,
            "latest_target_track_id": self.latest_target_track_id,
            "target_failure_latched": bool(self.target_failure_latched),
            "target_failure_count": int(self.target_failure_count),
            "target_failure_epoch": int(self.target_failure_epoch),
            "target_failure_track_id": self.target_failure_track_id,
        }
        payload.update(fields)
        self.bridge_events += 1
        try:
            self.bridge_status_pub.publish(
                String(data=json.dumps(payload, sort_keys=True))
            )
        except (TypeError, ValueError):
            rospy.logwarn_throttle(5.0, "TEB goal bridge status serialization failed")

    @staticmethod
    def _as_bool(value):
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _yaw(message):
        q = message.pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def _angle_delta(first, second):
        return math.atan2(math.sin(first - second), math.cos(first - second))

    def _same_goal(self, first, second):
        if first is None or second is None:
            return False
        if (first.header.frame_id or "") != (second.header.frame_id or ""):
            return False
        dx = first.pose.position.x - second.pose.position.x
        dy = first.pose.position.y - second.pose.position.y
        return (
            math.hypot(dx, dy) <= self.position_epsilon
            and (
                not self.compare_goal_yaw
                or abs(self._angle_delta(self._yaw(first), self._yaw(second)))
                <= self.yaw_epsilon
            )
        )

    def _intent_signature_locked(self):
        """Return the pending mission transaction identity.

        Position equality alone is insufficient for a route connector: the
        same map cell can carry a different execution contract. Conversely,
        the exact same latched goal and intent must be a no-op while its action
        is active. This signature is diagnostic/lifecycle state, not a motion
        threshold.
        """
        goal = self.latest_goal
        if goal is None:
            return None
        return (
            (goal.header.frame_id or "").strip().lstrip("/"),
            round(float(goal.pose.position.x), 3),
            round(float(goal.pose.position.y), 3),
            self.latest_intent_source,
            int(self.latest_intent_priority),
            self.latest_route_kind,
            int(self.latest_target_epoch),
        )

    def _clear_target_failure_locked(self, reason):
        """Release the failed-target latch after a real mission transition."""
        if not self.target_failure_latched:
            return
        previous_goal = self.target_failure_goal
        previous_epoch = self.target_failure_epoch
        previous_track_id = self.target_failure_track_id
        self.target_failure_latched = False
        self.target_failure_goal = None
        self.target_failure_epoch = 0
        self.target_failure_track_id = ""
        self.target_failure_generation = 0
        self.publish_bridge_status(
            "target_failure_cleared",
            reason=str(reason),
            previous_target=(
                None
                if previous_goal is None
                else [
                    round(float(previous_goal.pose.position.x), 3),
                    round(float(previous_goal.pose.position.y), 3),
                ]
            ),
            previous_target_epoch=int(previous_epoch),
            previous_target_track_id=previous_track_id,
        )

    def _target_failure_blocks_latest_locked(self):
        """Guard re-dispatch of a target that has already failed.

        The latch is intentionally cleared by a new visual track or by a real
        successful frontier action, never by an arbitrary detector frame or a
        short target-reacquisition segment. Neither path changes a TEB
        optimizer parameter.
        """
        if not self.target_failure_latched:
            return False
        if self.latest_intent_priority < 2:
            # Frontier may take ownership and recover the robot, but this
            # failed visual track remains blocked until that recovery has
            # demonstrably completed.
            return False
        if (
            self.latest_target_track_id
            and self.latest_target_track_id != self.target_failure_track_id
        ):
            self._clear_target_failure_locked("new_target_track")
            return False
        return True

    @staticmethod
    def _normalize_goal(message):
        goal = copy.deepcopy(message)
        if not goal.header.frame_id:
            goal.header.frame_id = "odom"
        return goal

    def _goal_in_global_frame(self, source_goal):
        """Transform a source goal once at action dispatch time."""
        goal = copy.deepcopy(source_goal)
        source_frame = (goal.header.frame_id or "").strip().lstrip("/") or "odom"
        goal.header.frame_id = source_frame
        if source_frame == self.global_frame:
            goal.header.stamp = rospy.Time.now()
            return goal
        goal.header.stamp = rospy.Time(0)
        try:
            self.tf_listener.waitForTransform(
                self.global_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(0.5),
            )
            transformed = self.tf_listener.transformPose(self.global_frame, goal)
        except (
            tf.Exception,
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                3.0,
                "TEB goal bridge waiting for %s <- %s transform: %s",
                self.global_frame,
                source_frame,
                exc,
            )
            return None
        transformed.header.frame_id = self.global_frame
        transformed.header.stamp = rospy.Time.now()
        return transformed

    def _feedback_in_global_frame(self, feedback_pose):
        """Convert move_base feedback pose to the action goal frame.

        ``MoveBaseFeedback.base_position`` is normally published in ``odom``
        while online SLAM goals are in ``map``.  Comparing the two coordinate
        pairs directly produces a false near-goal distance and can trigger an
        early handoff.  Frame conversion belongs at this action boundary so
        every health decision uses one coordinate system.
        """
        pose = copy.deepcopy(feedback_pose)
        source_frame = (pose.header.frame_id or "").strip().lstrip("/") or "odom"
        pose.header.frame_id = source_frame
        if source_frame == self.global_frame:
            return pose
        pose.header.stamp = rospy.Time(0)
        try:
            self.tf_listener.waitForTransform(
                self.global_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(0.05),
            )
            transformed = self.tf_listener.transformPose(self.global_frame, pose)
        except (
            tf.Exception,
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as exc:
            self.feedback_transform_failures += 1
            rospy.logwarn_throttle(
                3.0,
                "TEB goal bridge cannot transform feedback %s -> %s: %s",
                source_frame,
                self.global_frame,
                exc,
            )
            return None
        transformed.header.frame_id = self.global_frame
        return transformed

    def _is_active_mode(self):
        return self.mode == self.active_mode and not self.task_done

    def on_intent(self, message):
        """Record mission ownership for the next PoseStamped goal.

        JSON is used so the source remains visible in rosbag/logs while a plain
        source string remains a valid fallback for simple external publishers.
        """
        source = "unknown"
        priority = 0
        route_kind = ""
        intent_goal = None
        target_epoch = 0
        target_track_id = ""
        try:
            payload = json.loads(message.data)
            if isinstance(payload, dict):
                source = str(payload.get("source", source)).strip().lower() or source
                priority = int(payload.get("priority", priority))
                route_kind = str(payload.get("route_kind", "")).strip().lower()
                target_epoch = max(0, int(payload.get("target_epoch", 0)))
                target_track_id = str(payload.get("target_track_id", "")).strip()
                raw_goal = payload.get("goal")
                if isinstance(raw_goal, (list, tuple)) and len(raw_goal) >= 2:
                    intent_goal = (float(raw_goal[0]), float(raw_goal[1]))
            else:
                source = str(message.data).strip().lower() or source
        except (TypeError, ValueError, json.JSONDecodeError):
            source = str(message.data).strip().lower() or source
        with self.lock:
            self.intent_seen = True
            self.latest_intent_source = source
            self.latest_intent_priority = max(0, min(3, priority))
            self.latest_route_kind = route_kind
            self.latest_intent_goal = intent_goal
            self.latest_target_epoch = target_epoch
            self.latest_target_track_id = target_track_id
            if self._is_active_mode() and self.latest_goal is not None:
                # Goal Manager publishes intent before the matching pose. Do
                # not clear a failed-target latch or dispatch the previous
                # latched pose during that transaction gap.
                if not self._intent_matches_goal_locked(self.latest_goal):
                    return
                self.dispatch_locked(force=False, reason="intent_received")

    def _intent_matches_goal_locked(self, goal):
        """Return whether the current intent describes ``goal``."""
        if goal is None:
            return False
        if self.require_intent and not self.intent_seen:
            return False
        if self.latest_intent_goal is None:
            # Plain-string intents from external callers have no transaction
            # key and retain the historical immediate-dispatch behavior.
            return True
        return math.hypot(
            float(goal.pose.position.x) - self.latest_intent_goal[0],
            float(goal.pose.position.y) - self.latest_intent_goal[1],
        ) <= max(self.position_epsilon, 0.05)

    def on_frontier_status(self, message):
        """Release an action whose exploration route was invalidated upstream.

        The frontier explorer owns route validity.  Keeping a stale action
        alive after it has abandoned the route makes move_base repeat recovery
        behaviors forever and prevents a new branch from becoming active.
        """
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("event") != "route_invalidated":
            return
        reason = str(payload.get("reason", "unknown"))
        with self.lock:
            frontier_owned = (
                self.active_intent_source == "global_slam_frontier"
                or self.latest_intent_source == "global_slam_frontier"
            )
            if not frontier_owned or not self._is_active_mode():
                return
            stale_goal = self.last_dispatched_goal
            self.cancel_locked("frontier_route_invalidated_%s" % reason)
            # Wait for the next explicit frontier publication.  Do not retry
            # the same pose from the bridge timer while the explorer is
            # selecting and validating a replacement branch.
            self.latest_goal = None
            self.last_dispatched_goal = None
            self.last_terminal_goal = None
            self.latest_intent_source = "waiting_global_slam_frontier"
            self.latest_intent_priority = 0
            self.latest_route_kind = ""
            self.latest_intent_goal = None
            self.publish_bridge_status(
                "frontier_route_invalidated",
                reason=reason,
                stale_goal=(
                    None
                    if stale_goal is None
                    else [
                        round(float(stale_goal.pose.position.x), 3),
                        round(float(stale_goal.pose.position.y), 3),
                    ]
                ),
                source_status=payload,
            )
            rospy.logwarn(
                "TEB goal bridge released stale frontier action: reason=%s",
                reason,
            )

    def on_turn_supervisor_status(self, message):
        """Hold same-priority frontier replacements during an atomic turn."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        state = str(payload.get("state", "UNKNOWN")).strip().upper() or "UNKNOWN"
        event = str(payload.get("event", "unknown"))
        with self.lock:
            self.turn_supervisor_state = state
            self.turn_supervisor_last_event = event
            if (
                event in ("turn_completed", "turn_released")
                and self.action_active
                and self.active_route_kind == "frontier_turn_connector"
            ):
                # Completion of an atomic yaw action is execution progress
                # even when xy feedback stayed constant.  Restart the action
                # health epoch so the route planner can publish the following
                # translational endpoint before stale-action recovery runs.
                self.active_progress_monotonic = time.monotonic()
                if self.active_feedback_distance is not None:
                    self.active_best_distance = self.active_feedback_distance
            if (
                event == "turn_completed"
                and self.action_active
                and self.active_route_kind == "frontier_turn_connector"
            ):
                # The connector's yaw contract is complete. If the frontier
                # endpoint is already pending, dispatch_locked performs the
                # one authorized phase handoff without waiting for a second
                # move_base terminal/zero-command cycle.
                self.turn_transition_ready = True
                self.dispatch_locked(
                    force=False, reason="turn_completed_route_release"
                )
        if state == "TURNING":
            rospy.loginfo_throttle(
                3.0,
                "TEB goal bridge holds frontier action while turn supervisor is TURNING",
            )

    def on_goal(self, message):
        with self.lock:
            self.latest_goal = self._normalize_goal(message)
            if self._is_active_mode():
                if not self._intent_matches_goal_locked(self.latest_goal):
                    intent_x = (
                        float(self.latest_intent_goal[0])
                        if self.latest_intent_goal is not None else float("nan")
                    )
                    intent_y = (
                        float(self.latest_intent_goal[1])
                        if self.latest_intent_goal is not None else float("nan")
                    )
                    rospy.loginfo_throttle(
                        2.0,
                        "TEB goal bridge waiting for matching goal intent before "
                        "dispatch: pose=(%.2f,%.2f) intent=(%.2f,%.2f)",
                        self.latest_goal.pose.position.x,
                        self.latest_goal.pose.position.y,
                        intent_x,
                        intent_y,
                    )
                    return
                self.dispatch_locked(force=False, reason="global_goal_changed")

    def on_timer(self, _event):
        with self.lock:
            if self._is_active_mode():
                self.maybe_handoff_locked()
                self.dispatch_locked(force=False, reason="coalesced_global_goal")

    def on_terminal_timer(self, _event):
        with self.lock:
            self.terminal_dispatch_timer = None
            if self._is_active_mode():
                self.dispatch_locked(force=False, reason="terminal_followup")

    def schedule_terminal_dispatch_locked(self):
        if self.terminal_dispatch_timer is None and not rospy.is_shutdown():
            self.terminal_dispatch_timer = rospy.Timer(
                rospy.Duration(0.05), self.on_terminal_timer, oneshot=True
            )

    def on_mode(self, message):
        mode = message.data.strip().lower()
        if not mode:
            return
        with self.lock:
            previous = self.mode
            self.mode = mode
            if previous == self.active_mode and mode != self.active_mode:
                self.cancel_locked("controller_switched_to_%s" % mode)
            elif mode == self.active_mode and previous != self.active_mode:
                self.dispatch_locked(force=True, reason="controller_switched_to_teb")

    def on_task_done(self, message):
        with self.lock:
            previous = self.task_done
            self.task_done = bool(message.data)
            if self.task_done:
                self.cancel_locked("task_done")
            elif previous and self._is_active_mode():
                self.dispatch_locked(force=True, reason="new_task")

    def cancel_locked(self, reason):
        self.action_generation += 1
        if self.action_active:
            self.action_client.cancel_goal()
        self.action_active = False
        self.handoff_requested = False
        self._clear_target_failure_locked("cancel:%s" % reason)
        self._clear_action_health_locked()
        self.last_result_status = GoalStatus.PREEMPTED
        self.last_result_monotonic = time.monotonic()
        self.publish_bridge_status("cancel", reason=reason)
        rospy.loginfo("TEB goal bridge cancelled move_base action: reason=%s", reason)

    def _clear_action_health_locked(self):
        self.active_goal_global = None
        self.active_feedback_distance = None
        self.active_feedback_pose = None
        self.active_feedback_frame = ""
        self.active_best_distance = None
        self.active_progress_monotonic = 0.0
        self.last_feedback_monotonic = 0.0
        self.frontier_stale_wait_started_monotonic = 0.0
        self.turn_transition_ready = False
        self.move_base_terminal_pending = False
        self.active_intent_source = "unknown"
        self.active_intent_priority = 0
        self.active_route_kind = ""
        self.active_target_epoch = 0
        self.active_target_track_id = ""

    def _pending_goal_delta_locked(self):
        """Return the pending-vs-active distance in the source goal frame.

        Normal LSTE goals are odom-frame poses, so this is a cheap Euclidean
        comparison.  For a caller using another frame, transform the pending
        goal once and compare it to the already transformed action goal.
        """
        if self.latest_goal is None or self.last_dispatched_goal is None:
            return 0.0
        if (
            (self.latest_goal.header.frame_id or "")
            == (self.last_dispatched_goal.header.frame_id or "")
        ):
            return math.hypot(
                self.latest_goal.pose.position.x
                - self.last_dispatched_goal.pose.position.x,
                self.latest_goal.pose.position.y
                - self.last_dispatched_goal.pose.position.y,
            )
        pending_global = self._goal_in_global_frame(self.latest_goal)
        if pending_global is None or self.active_goal_global is None:
            return 0.0
        return math.hypot(
            pending_global.pose.position.x - self.active_goal_global.pose.position.x,
            pending_global.pose.position.y - self.active_goal_global.pose.position.y,
        )

    def _pending_heading_delta_locked(self):
        """Return the bearing change from the active to pending goal.

        This is deliberately measured at the latest move_base feedback pose,
        rather than at the robot's initial pose.  A branch can be far away but
        still be a smooth continuation; only the local turn required at the
        handoff should block an in-place replacement.
        """
        if self.latest_goal is None or self.active_goal_global is None:
            return None
        feedback = self.active_feedback_pose
        if feedback is None:
            return None
        pending_global = self._goal_in_global_frame(self.latest_goal)
        if pending_global is None:
            return None
        base_x, base_y, base_yaw = feedback
        active_bearing = math.atan2(
            self.active_goal_global.pose.position.y - base_y,
            self.active_goal_global.pose.position.x - base_x,
        )
        pending_bearing = math.atan2(
            pending_global.pose.position.y - base_y,
            pending_global.pose.position.x - base_x,
        )
        return abs(self._angle_delta(pending_bearing, active_bearing))

    def _sharp_frontier_recovery_heading_locked(self, pending_delta):
        """Return the pending branch turn if stale recovery is near its end.

        A frontier branch is allowed to replace an active action in-place only
        when the replacement is a short continuation (the segment handoff
        path above). A large or sharp branch still has to change direction,
        but when the old endpoint is already close, an explicit cancel creates
        an avoidable zero-velocity pulse. This helper identifies exactly that
        narrow recovery case; it never relaxes collision or progress checks.
        """
        if (
            self.active_intent_priority != 0
            or self.latest_intent_priority != 0
            or self.latest_intent_source != "global_slam_frontier"
            or self.active_feedback_distance is None
            or self.active_feedback_distance > self.frontier_stale_recovery_max_distance
        ):
            return None
        heading_delta = self._pending_heading_delta_locked()
        if (
            pending_delta <= self.frontier_early_handoff_max_delta
            and (
                heading_delta is None
                or heading_delta <= self.frontier_early_handoff_max_heading_delta
            )
        ):
            return None
        # A missing feedback pose should not disable the distance gate for a
        # clearly oversized branch; zero here means "heading unavailable" in
        # the diagnostic payload, while the branch-size condition still holds.
        return heading_delta if heading_delta is not None else 0.0

    def _latch_target_failure_locked(
        self, reason, status_text="NO_PROGRESS", cancel_action=True
    ):
        """End a target transaction and notify mission arbitration exactly once.

        A target is an observation-backed hypothesis, not an action that can be
        retried forever. Once TEB has no physical progress, this method emits
        one failure event, records the failed transaction identity, and lets
        Goal Manager select a map route. The bridge-side latch prevents the
        terminal callback/timer race from reissuing the same target.
        """
        if self.active_intent_priority < 2:
            return False
        if (
            self.target_failure_latched
            and self.target_failure_generation == self.action_generation
        ):
            return False
        source_goal = copy.deepcopy(self.last_dispatched_goal)
        now = time.monotonic()
        self.target_failure_latched = True
        self.target_failure_goal = source_goal
        self.target_failure_epoch = int(self.active_target_epoch)
        self.target_failure_track_id = self.active_target_track_id
        self.target_failure_generation = int(self.action_generation)
        self.target_failure_count += 1
        self.handoff_requested = True
        payload = {
            "event": "target_route_failed",
            "reason": str(reason),
            "status": str(status_text),
            "goal": (
                None
                if source_goal is None
                else [
                    round(float(source_goal.pose.position.x), 4),
                    round(float(source_goal.pose.position.y), 4),
                ]
            ),
            "goal_frame": (
                None if source_goal is None else source_goal.header.frame_id
            ),
            "active_goal": (
                None
                if self.active_goal_global is None
                else [
                    round(float(self.active_goal_global.pose.position.x), 4),
                    round(float(self.active_goal_global.pose.position.y), 4),
                ]
            ),
            "active_goal_frame": (
                None
                if self.active_goal_global is None
                else self.active_goal_global.header.frame_id
            ),
            "target_epoch": int(self.active_target_epoch),
            "target_track_id": self.active_target_track_id,
            "failure_count": int(self.target_failure_count),
            "feedback_distance": (
                None
                if self.active_feedback_distance is None
                else round(float(self.active_feedback_distance), 4)
            ),
            "progress_age": (
                None
                if self.active_progress_monotonic <= 0.0
                else round(float(now - self.active_progress_monotonic), 4)
            ),
        }
        self.target_failure_pub.publish(
            String(data=json.dumps(payload, sort_keys=True))
        )
        self.publish_bridge_status(
            "target_route_failed",
            reason=str(reason),
            status_text=str(status_text),
            target_epoch=int(self.active_target_epoch),
            target_track_id=self.active_target_track_id,
            failure_count=int(self.target_failure_count),
            feedback_distance=payload["feedback_distance"],
            progress_age=payload["progress_age"],
            lifecycle="release_to_frontier",
        )
        if cancel_action and self.action_active:
            self.action_client.cancel_goal()
        rospy.logwarn(
            "TEB goal bridge released failed target to mission layer: "
            "reason=%s status=%s goal=%s epoch=%d",
            reason,
            status_text,
            payload["goal"],
            self.active_target_epoch,
        )
        return True

    def maybe_handoff_locked(self):
        """Cancel a stale action only when a newer intent and health evidence exist."""
        if (
            not self.action_active
            or self.handoff_requested
            or self.latest_goal is None
            or self.last_dispatched_goal is None
        ):
            return
        if (
            self.turn_supervisor_state == "TURNING"
            and self.active_intent_priority == 0
            and self.latest_intent_priority == 0
        ):
            # The turn supervisor owns this short atomic action.  A normal
            # progress timeout must not cancel it and reintroduce the exact
            # stop/restart race that the supervisor removes.
            return
        now = time.monotonic()
        if now - self.last_dispatch_monotonic < self.handoff_min_interval:
            return
        pending_delta = self._pending_goal_delta_locked()
        stalled = (
            self.active_progress_monotonic > 0.0
            and now - self.active_progress_monotonic >= self.progress_timeout
        )
        # Frontier actions are allowed to finish unless a newer map waypoint
        # is waiting. A visual target is different: once its committed route
        # makes no physical progress, it becomes a mission-level failure and
        # ownership must return to frontier exploration. Retrying the same ray
        # here was the source of the observed 100+ cancel/retry loop.
        target_stalled = (
            self.active_intent_priority >= 2
            and stalled
            and pending_delta < self.handoff_distance
        )
        if pending_delta < self.handoff_distance and not target_stalled:
            return
        # Reaching the xy tolerance is not a reason to cancel an action.  The
        # move_base action can report SUCCEEDED on the same callback turn; a
        # timer-side near-endpoint cancel races that result and turns healthy
        # frontier completions into PREEMPTED.  Only a real no-progress window
        # is allowed to hand the action over. The old handoff_radius parameter
        # remains accepted for launch compatibility but is intentionally not a
        # lifecycle decision anymore.
        if not stalled:
            return

        # A sharp frontier branch is a new route topology, not a continuation
        # of the current local trajectory.  Never replace it in-place merely
        # because the old endpoint is close: that resets TEB's band and causes
        # the exact brake/turn pulse this bridge is meant to prevent.  A
        # genuinely stalled action still reaches the explicit recovery path
        # below, where move_base owns cancellation and recovery semantics.
        self.frontier_stale_wait_started_monotonic = 0.0
        if target_stalled:
            self._latch_target_failure_locked(
                "target_no_feedback_progress",
                status_text="NO_PROGRESS",
                cancel_action=True,
            )
            return
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
            "TEB goal bridge %s stale action: reason=%s "
            "pending_delta=%.2fm feedback_distance=%s progress_age=%.1fs",
            "handing off",
            reason,
            pending_delta,
            "n/a"
            if self.active_feedback_distance is None
            else "%.2f" % self.active_feedback_distance,
            now - self.active_progress_monotonic
            if self.active_progress_monotonic > 0.0
            else 0.0,
        )

    def maybe_segment_handoff_locked(self):
        """Replace a nearly reached route segment without a cancel/stop gap.

        Goal Manager publishes bounded segments for both online frontiers and
        confirmed visual targets. A same-priority update is safe to replace
        once the current action is close to its endpoint; far-away updates are
        still coalesced until the action finishes. The action server accepts
        this new goal in its existing execute loop, so TEB keeps publishing a
        trajectory instead of stopping between two adjacent segments.
        """
        if (
            not self.action_active
            or self.handoff_requested
            or self.latest_goal is None
            or self.last_dispatched_goal is None
            or self.active_feedback_distance is None
            or self.latest_intent_priority != self.active_intent_priority
            or self.active_intent_priority not in (0, 2)
        ):
            return False
        # A route turn is a semantic action, not a replaceable waypoint.  The
        # supervisor owns its angle closure; replacing the move_base goal here
        # would retarget that action before the actuator reaches its contract.
        # Higher-priority target intents are handled by dispatch_locked below,
        # so only the same-priority frontier stream is frozen.
        if (
            self.turn_supervisor_state == "TURNING"
            and self.active_intent_priority == 0
            and self.latest_intent_priority == 0
        ):
            return False
        now = time.monotonic()
        if now - self.last_dispatch_monotonic < self.min_update_interval:
            return False
        pending_delta = self._pending_goal_delta_locked()
        frontier_branch = (
            self.active_intent_priority == 0
            and pending_delta > self.frontier_replacement_max_delta
        )
        # The online frontier planner now labels each command as a point on a
        # validated connected route.  A route connector may turn sharply at a
        # doorway, but that turn is part of the same map path and must not be
        # mistaken for an unrelated branch replacement.  Keep the normal
        # distance/hot-start gates; only remove the old bearing gate for this
        # explicit route contract.
        route_continuation = (
            self.active_intent_priority == 0
            and self.latest_intent_source == "global_slam_frontier"
            and self.latest_route_kind in (
                "frontier_connector",
                "frontier_turn_connector",
                "frontier_endpoint",
            )
            and self.active_route_kind in (
                "frontier_connector",
                "frontier_turn_connector",
                "frontier_endpoint",
            )
        )
        minimum_distance = (
            self.frontier_early_handoff_min_distance
            if frontier_branch
            else self.in_place_replacement_min_distance
        )
        maximum_distance = (
            self.frontier_early_handoff_max_distance
            if frontier_branch
            else self.in_place_replacement_max_distance
        )
        if route_continuation:
            # A validated route connector is allowed to replace the active
            # action until just outside TEB's terminal tolerance.  Waiting for
            # the older 0.85 m handoff band made a connector lose a race to
            # SUCCEEDED at about 0.7 m and inserted a needless stop/restart.
            minimum_distance = self.frontier_sharp_replacement_min_distance
            maximum_distance = self.frontier_early_handoff_max_distance
        maximum_delta = (
            self.frontier_early_handoff_max_delta
            if frontier_branch
            else (
                self.in_place_replacement_max_delta
                if self.active_intent_priority >= 2
                else self.frontier_replacement_max_delta
            )
        )
        heading_delta = None
        sharp_frontier_branch = False
        if frontier_branch:
            heading_delta = self._pending_heading_delta_locked()
            sharp_frontier_branch = (
                heading_delta is not None
                and heading_delta > self.frontier_early_handoff_max_heading_delta
            )
            if sharp_frontier_branch and not route_continuation:
                # A sharp branch is normally held until the current action
                # succeeds.  Once feedback has brought the old endpoint into
                # the terminal neighbourhood, however, waiting for the
                # result inserts the exact zero-velocity gap this bridge is
                # intended to remove.  The action server can accept the new
                # branch in-place while TEB keeps its command loop alive.
                if not (
                    self.frontier_sharp_replacement_min_distance
                    <= self.active_feedback_distance
                    <= self.frontier_stale_recovery_max_distance
                    and pending_delta <= self.frontier_early_handoff_max_delta
                ):
                    rospy.loginfo_throttle(
                        3.0,
                        "TEB goal bridge defers sharp frontier replacement: "
                        "heading_delta=%.1fdeg feedback_distance=%.2fm",
                        math.degrees(heading_delta),
                        self.active_feedback_distance,
                    )
                    return False
                minimum_distance = self.frontier_sharp_replacement_min_distance
                maximum_distance = self.frontier_stale_recovery_max_distance
        if self.active_feedback_distance > maximum_distance:
            return False
        if self.active_feedback_distance < minimum_distance:
            rospy.loginfo_throttle(
                3.0,
                "TEB goal bridge waits for terminal action near goal: "
                "feedback_distance=%.2fm min_replacement_distance=%.2fm",
                self.active_feedback_distance,
                minimum_distance,
            )
            return False
        minimum_delta = (
            self.target_early_handoff_min_delta
            if self.active_intent_priority >= 2
            else self.frontier_replacement_min_delta
        )
        if pending_delta < minimum_delta:
            return False
        if pending_delta > maximum_delta:
            rospy.loginfo_throttle(
                3.0,
                "TEB goal bridge defers oversized replacement: "
                "pending_delta=%.2fm max_hot_start_delta=%.2fm "
                "feedback_distance=%.2fm intent=%s",
                pending_delta, maximum_delta,
                self.active_feedback_distance,
                self.active_intent_source,
            )
            return False

        feedback_distance = self.active_feedback_distance
        replacement_kind = (
            "target_segment"
            if self.active_intent_priority >= 2
            else (
                "frontier_route_connector"
                if route_continuation and self.latest_route_kind == "frontier_connector"
                else (
                    "frontier_route_turn_connector"
                    if route_continuation
                    and self.latest_route_kind == "frontier_turn_connector"
                    else (
                    "frontier_route_endpoint"
                    if route_continuation
                    else ("frontier_sharp_branch" if sharp_frontier_branch else "frontier_segment")
                    )
                )
            )
        )
        if replacement_kind == "target_segment":
            self.target_segment_handoff_count += 1
            handoffs = self.target_segment_handoff_count
        else:
            self.frontier_segment_handoff_count += 1
            handoffs = self.frontier_segment_handoff_count
        if not self._send_goal_locked(
            self.latest_goal,
            reason=("sharp_frontier_transition" if sharp_frontier_branch else "near_segment_end"),
            replacement=True,
            replacement_kind=replacement_kind,
            pending_delta=round(pending_delta, 3),
                feedback_distance=round(feedback_distance, 3),
                heading_delta_deg=(
                    None if heading_delta is None else round(math.degrees(heading_delta), 1)
                ),
                handoffs=int(handoffs),
            early_branch=bool(frontier_branch),
            topology_transition=bool(sharp_frontier_branch and not route_continuation),
            route_continuation=bool(route_continuation),
        ):
            if replacement_kind == "target_segment":
                self.target_segment_handoff_count -= 1
            else:
                self.frontier_segment_handoff_count -= 1
            return False
        rospy.loginfo(
            "TEB goal bridge replaced %s segment in-place: "
            "feedback_distance=%.2fm pending_delta=%.2fm handoffs=%d",
            replacement_kind,
            feedback_distance,
            pending_delta,
            handoffs,
        )
        return True

    def _safe_route_continuation_pending_locked(self):
        """Return whether the queued goal is the same validated route.

        This gate is intentionally semantic.  It does not infer safety from a
        distance threshold alone: both the active and pending intents must be
        frontier-owned route segments.  A different branch or a visual target
        therefore remains a normal action transaction.
        """
        route_kinds = {
            "frontier_connector",
            "frontier_turn_connector",
            "frontier_endpoint",
        }
        return (
            self.active_intent_priority == 0
            and self.latest_intent_priority == 0
            and self.active_intent_source == "global_slam_frontier"
            and self.latest_intent_source == "global_slam_frontier"
            and self.active_route_kind in route_kinds
            and self.latest_route_kind in route_kinds
            and self.latest_route_kind != "frontier_turn_connector"
        )

    def _action_server_ready(self):
        if self.action_server_seen:
            return True
        if not self.action_client.wait_for_server(rospy.Duration(0.0)):
            rospy.loginfo_throttle(3.0, "TEB goal bridge waiting for move_base action server")
            return False
        self.action_server_seen = True
        rospy.loginfo("TEB goal bridge connected to move_base action server")
        return True

    def dispatch_locked(self, force, reason):
        if self.latest_goal is None or not self._is_active_mode():
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

        # The mission layer may publish a new observation or frontier at any
        # time, but an action is an atomic navigation intent. Coalesce updates
        # until the current intent has a result. Only lifecycle transitions
        # use force=True and cancel the active action.
        if self.action_active and not force:
            # A latched /lste/final_goal and the bridge health timer can both
            # present the currently active transaction. It is already owned
            # by move_base; do not turn that observation into a deferred
            # update. A route-kind/source change at the same position remains
            # a distinct semantic intent and is handled below.
            if (
                self._same_goal(self.latest_goal, self.last_dispatched_goal)
                and self.latest_intent_source == self.active_intent_source
                and self.latest_intent_priority == self.active_intent_priority
                and self.latest_route_kind == self.active_route_kind
            ):
                return
            if (
                self.turn_transition_ready
                and self.active_intent_priority == 0
                and self.latest_intent_priority == 0
                and self.active_route_kind == "frontier_turn_connector"
                and self.latest_intent_source == "global_slam_frontier"
                and self.latest_route_kind == "frontier_endpoint"
                and not self._same_goal(self.latest_goal, self.last_dispatched_goal)
            ):
                if self._send_goal_locked(
                    self.latest_goal,
                    reason="turn_completed_route_release",
                    replacement=True,
                    replacement_kind="turn_phase_transition",
                    from_route_kind=self.active_route_kind,
                    to_route_kind=self.latest_route_kind,
                ):
                    self.turn_transition_ready = False
                return
            if (
                self.turn_supervisor_state == "TURNING"
                and self.active_intent_priority == 0
                and self.latest_intent_priority == 0
            ):
                pending_signature = self._intent_signature_locked()
                if pending_signature == self.deferred_signature:
                    return
                self.deferred_signature = pending_signature
                self.deferred_goal_updates += 1
                rospy.loginfo_throttle(
                    3.0,
                    "TEB goal bridge queues frontier update during atomic turn: "
                    "deferred=%d",
                    self.deferred_goal_updates,
                )
                return
            # Mission ownership is separate from route freshness.  A newly
            # confirmed target should not wait up to the frontier stall timeout
            # before the car can react, but a normal map refresh must not cancel
            # a healthy TEB trajectory.  Replace a higher-priority goal through
            # actionlib's native new-goal preemption; do not send an explicit
            # cancel because that makes move_base publish a stop first.
            if (
                self.latest_intent_priority > self.active_intent_priority
                and not self._same_goal(self.latest_goal, self.last_dispatched_goal)
            ):
                pending_delta = self._pending_goal_delta_locked()
                self.priority_handoff_count += 1
                if self.allow_in_place_replacement:
                    if not self._send_goal_locked(
                        self.latest_goal,
                        reason="higher_priority_intent",
                        replacement=True,
                        replacement_kind="priority_intent",
                        pending_delta=round(pending_delta, 3),
                        from_source=self.active_intent_source,
                        to_source=self.latest_intent_source,
                        handoffs=int(self.priority_handoff_count),
                    ):
                        self.priority_handoff_count -= 1
                else:
                    self._request_cancel_for_pending_goal_locked(
                        reason="higher_priority_intent",
                        pending_delta=pending_delta,
                    )
                return
            allow_route_handoff = (
                self.allow_route_continuation_replacement
                and self._safe_route_continuation_pending_locked()
            )
            if (
                (self.allow_in_place_replacement or allow_route_handoff)
                and self.maybe_segment_handoff_locked()
            ):
                # The action server accepts this newer goal while its execute
                # loop remains alive; no cancel transition is needed here.
                return
            pending_signature = self._intent_signature_locked()
            if pending_signature == self.deferred_signature:
                return
            self.deferred_signature = pending_signature
            self.deferred_goal_updates += 1
            now = time.monotonic()
            if now - self.deferred_goal_log_wall >= 2.0:
                self.deferred_goal_log_wall = now
                latest = self.latest_goal
                self.publish_bridge_status(
                    "goal_deferred",
                    reason=reason,
                    latest_goal=[
                        round(latest.pose.position.x, 3),
                        round(latest.pose.position.y, 3),
                    ],
                )
                rospy.loginfo(
                    "TEB goal bridge queued latest goal while action is active: "
                    "deferred=%d reason=%s",
                    self.deferred_goal_updates,
                    reason,
                )
            return

        same_goal = self._same_goal(self.latest_goal, self.last_dispatched_goal)
        if not force and same_goal:
            if self.action_active or self.last_result_status == GoalStatus.SUCCEEDED:
                return
            if (
                time.monotonic() - self.last_result_monotonic < self.goal_retry_interval
            ):
                return
            reason = "retry_move_base_goal"

        if (
            not force
            and self.last_dispatched_goal is not None
            and time.monotonic() - self.last_dispatch_monotonic
            < self.min_update_interval
        ):
            return

        if force and self.action_active:
            self.action_generation += 1
            self.action_client.cancel_goal()
            self.action_active = False

        self._send_goal_locked(self.latest_goal, reason=reason, replacement=False)

    def _request_cancel_for_pending_goal_locked(self, reason, pending_delta):
        """Cancel once and let the action result authorize the next dispatch."""
        if not self.action_active or self.handoff_requested:
            return
        self.handoff_requested = True
        self.action_client.cancel_goal()
        self.publish_bridge_status(
            "handoff_requested",
            reason=reason,
            pending_delta=round(float(pending_delta), 3),
            from_source=self.active_intent_source,
            to_source=self.latest_intent_source,
            lifecycle="explicit_cancel_then_terminal_dispatch",
        )
        rospy.loginfo(
            "TEB goal bridge queued higher-priority goal behind action terminal: "
            "reason=%s pending_delta=%.2fm",
            reason,
            pending_delta,
        )

    def _send_goal_locked(self, source_message, reason, replacement=False,
                          replacement_kind=None, **event_fields):
        """Send a goal, optionally replacing the active action in-place.

        ``SimpleActionClient.send_goal`` stops tracking the old client-side
        handle but does not send a cancel request.  ``move_base`` then accepts
        the new goal in its existing execute loop and keeps the local planner
        alive.  This is the important distinction from ``cancel_goal`` for
        short visual-servo segments.
        """
        if source_message is None or not self._is_active_mode():
            return False
        source_goal = self._normalize_goal(source_message)
        goal = self._goal_in_global_frame(source_goal)
        if goal is None:
            return False

        # Keep the mission/source pose for lifecycle accounting, but execute a
        # turn connector at the robot's terminal pose. The connector's
        # orientation remains the route tangent; its XY is an execution fact,
        # not a second navigation waypoint.
        execution_goal = copy.deepcopy(goal)
        if (
            self.latest_route_kind == "frontier_turn_connector"
            and self.last_feedback_pose_global is not None
        ):
            execution_goal.pose.position.x = self.last_feedback_pose_global.pose.position.x
            execution_goal.pose.position.y = self.last_feedback_pose_global.pose.position.y
            execution_goal.header.stamp = rospy.Time.now()

        action_goal = MoveBaseGoal()
        action_goal.target_pose = execution_goal
        self.action_generation += 1
        generation = self.action_generation
        self.last_dispatched_goal = copy.deepcopy(source_goal)
        self.last_terminal_goal = None
        self.last_dispatch_monotonic = time.monotonic()
        self.last_result_status = None
        self.action_active = True
        self.active_intent_source = self.latest_intent_source
        self.active_intent_priority = self.latest_intent_priority
        self.active_route_kind = self.latest_route_kind
        self.active_target_epoch = int(self.latest_target_epoch)
        self.active_target_track_id = self.latest_target_track_id
        self.handoff_requested = False
        self.active_goal_global = copy.deepcopy(execution_goal)
        self.active_feedback_distance = None
        self.active_best_distance = None
        self.active_progress_monotonic = time.monotonic()
        self.last_feedback_monotonic = 0.0
        self.move_base_terminal_pending = False
        # The goal currently being executed is no longer pending. A later
        # semantic update at the same coordinates can be recognized once,
        # while repeated timer evaluations remain no-ops.
        self.deferred_signature = None
        self.dispatch_count += 1
        self.action_client.send_goal(
            action_goal,
            done_cb=lambda status, result: self.on_done(generation, status, result),
            active_cb=lambda: self.on_active(generation),
            feedback_cb=lambda feedback: self.on_feedback(generation, feedback),
        )
        dispatch_fields = {
            "reason": reason,
            "source_goal": [
                round(source_goal.pose.position.x, 3),
                round(source_goal.pose.position.y, 3),
            ],
            "dispatched_goal": [
                round(execution_goal.pose.position.x, 3),
                round(execution_goal.pose.position.y, 3),
            ],
            "handoffs": int(self.handoff_count),
            "intent_source": self.active_intent_source,
            "intent_priority": int(self.active_intent_priority),
            "route_kind": self.active_route_kind,
            "execution_goal": [
                round(execution_goal.pose.position.x, 3),
                round(execution_goal.pose.position.y, 3),
                round(self._yaw(execution_goal), 4),
            ],
            "replacement": bool(replacement),
            "replacement_kind": (replacement_kind or "none"),
        }
        # A replacement may carry a more specific handoff count or source
        # transition. Merge it after the common fields so a field is emitted
        # exactly once in the JSON status payload.
        dispatch_fields.update(event_fields)
        self.publish_bridge_status("dispatch", **dispatch_fields)
        rospy.loginfo(
            "TEB goal bridge %s move_base action: reason=%s frame=%s "
            "target=(%.2f,%.2f) source_frame=%s source=(%.2f,%.2f)",
            "replaced" if replacement else "dispatched",
            reason,
            goal.header.frame_id,
            execution_goal.pose.position.x,
            execution_goal.pose.position.y,
            source_goal.header.frame_id,
            source_goal.pose.position.x,
            source_goal.pose.position.y,
        )
        return True

    def on_active(self, generation):
        with self.lock:
            if generation != self.action_generation:
                return
            self.action_active = True
            self.publish_bridge_status("active")

    def on_feedback(self, generation, feedback):
        with self.lock:
            if generation != self.action_generation or not self.action_active:
                return
            now = time.monotonic()
            self.last_feedback_monotonic = now
            if self.active_goal_global is None:
                return
            base_pose = self._feedback_in_global_frame(feedback.base_position)
            if base_pose is None:
                return
            base = base_pose.pose.position
            distance = math.hypot(
                self.active_goal_global.pose.position.x - base.x,
                self.active_goal_global.pose.position.y - base.y,
            )
            self.active_feedback_distance = distance
            self.active_feedback_pose = (
                float(base.x),
                float(base.y),
                self._yaw(base_pose),
            )
            self.active_feedback_frame = base_pose.header.frame_id
            self.last_feedback_pose_global = copy.deepcopy(base_pose)
            if (
                self.active_best_distance is None
                or distance < self.active_best_distance - self.progress_epsilon
            ):
                self.active_best_distance = distance
                self.active_progress_monotonic = now

    def on_done(self, generation, status, _result):
        with self.lock:
            if generation != self.action_generation:
                return
            # TEB's XY terminal condition is intentionally independent from
            # the explicit turn supervisor's yaw condition.  Do not close the
            # bridge transaction (or publish a frontier terminal) while a
            # connector is still TURNING: doing so makes the supervisor see an
            # inactive bridge and release the turn as incomplete.  The action
            # client has finished its XY goal, but the logical route phase
            # remains owned by the supervisor until ``turn_completed``.
            if (
                status == GoalStatus.SUCCEEDED
                and self.active_route_kind == TURN_ROUTE_KIND
                and self.turn_supervisor_state != "PASS_THROUGH"
            ):
                self.last_result_status = int(status)
                self.last_result_monotonic = time.monotonic()
                self.move_base_terminal_pending = True
                self.publish_bridge_status(
                    "turn_execution_terminal",
                    status=int(status),
                    status_text="SUCCEEDED_XY_WAITING_YAW",
                )
                rospy.loginfo(
                    "TEB connector reached XY terminal; retaining logical action "
                    "until turn supervisor completes yaw"
                )
                return
            self.action_active = False
            self.last_result_status = int(status)
            self.last_result_monotonic = time.monotonic()
            active_priority = int(self.active_intent_priority)
            if (
                status == GoalStatus.SUCCEEDED
                and active_priority < 2
                and self.target_failure_latched
            ):
                # A frontier action is the recovery boundary for a blocked
                # visual target. Do not release it merely because a detector
                # publishes another frame.
                self._clear_target_failure_locked("frontier_progress")
            if status != GoalStatus.SUCCEEDED and active_priority >= 2:
                # A planner abort/reject is a target-route failure even when
                # the progress timer did not fire first. Never let the generic
                # same-goal retry path turn that terminal result into a loop.
                self._latch_target_failure_locked(
                    "move_base_%s" % GoalStatus.to_string(status).lower(),
                    status_text=GoalStatus.to_string(status),
                    cancel_action=False,
                )
            handoff_requested = self.handoff_requested
            self.handoff_requested = False
            source_goal = copy.deepcopy(self.last_dispatched_goal)
            if status == GoalStatus.SUCCEEDED and source_goal is not None:
                source_goal.header.stamp = rospy.Time.now()
                self.last_terminal_goal = copy.deepcopy(source_goal)
                self.terminal_pub.publish(source_goal)
                self.terminal_count += 1
                self.publish_bridge_status(
                    "terminal",
                    status=int(status),
                    status_text="SUCCEEDED",
                    terminal_goal=[
                        round(source_goal.pose.position.x, 3),
                        round(source_goal.pose.position.y, 3),
                    ],
                )
                rospy.loginfo(
                    "TEB goal bridge successful terminal event: target=(%.2f,%.2f)",
                    source_goal.pose.position.x,
                    source_goal.pose.position.y,
                )
                self._clear_action_health_locked()
                # Do not call send_goal from inside SimpleActionClient's
                # done_cb. actionlib invokes the user callback before it has
                # finished transitioning its own state to DONE; sending the
                # next goal here races that transition and produces
                # "ACTIVE when ... DONE" errors. The regular timer performs
                # this handoff on the next callback turn.
                self.schedule_terminal_dispatch_locked()
                return

            self.publish_bridge_status(
                "terminal",
                status=int(status),
                status_text=GoalStatus.to_string(status),
            )
            rospy.logwarn(
                "TEB move_base action finished without success: status=%s handoff=%s",
                GoalStatus.to_string(status),
                handoff_requested,
            )
            self._clear_action_health_locked()
            if handoff_requested and self._is_active_mode() and self.latest_goal is not None:
                self.schedule_terminal_dispatch_locked()
            # An explicit cancel requested for a pending higher-priority goal
            # still has a well-defined handoff: wait for this result callback,
            # then the timer dispatches the latest coalesced goal.  A normal
            # terminal follows the same path without a special replacement.
            if handoff_requested and self._is_active_mode() and self.latest_goal is not None:
                self.schedule_terminal_dispatch_locked()


if __name__ == "__main__":
    TebGoalBridge()
    rospy.spin()
