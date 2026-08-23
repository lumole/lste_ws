#!/usr/bin/env python3
"""Adapt TEB execution phases without creating another navigation action.

TEB is the production local planner.  A forward-only base may nevertheless
start a valid Navfn route with its first tangent behind the current heading.
In that case TEB can alternate between rotation and a blocked forward command
near a wall.  This node makes the short *execution phase* explicit: it rotates
to the first Navfn path tangent, then releases the same TEB command stream.
The MoveBaseAction identity and endpoint never change.

The control boundary is deliberately narrow:

* legacy ``frontier_turn_connector`` actions remain supported;
* online-SLAM ``frontier_endpoint`` actions get a one-time pre-route alignment
  only when their Navfn path starts more than 90 degrees behind the base;
* visual targets and all other route kinds pass through unchanged;
* the turn velocity is an acceleration-limited angle-closed-loop command,
  derived from TEB's own ``max_vel_theta`` and ``acc_lim_theta`` parameters;
* after the route yaw tolerance is reached, the latest TEB command is released
  on the same output cycle.

The node sits between move_base and the command mux:

    move_base/TEB -> /lste/cmd_vel/teb_planner
                    -> turn supervisor -> /lste/cmd_vel/teb -> mux

It is therefore an execution-state adapter, not a second path planner and not
another PID tuning layer.
"""

import copy
import json
import math
import threading
import time

import rospy
import tf
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Bool, String


TURN_ROUTE_KIND = "frontier_turn_connector"
FRONTIER_ENDPOINT_KIND = "frontier_endpoint"
FRONTIER_SOURCE = "global_slam_frontier"
STATE_PASS_THROUGH = "PASS_THROUGH"
STATE_TURNING = "TURNING"


def normalize_angle(angle):
    """Return ``angle`` in [-pi, pi]."""
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def angle_from_pose(message):
    """Extract planar yaw from a PoseStamped quaternion."""
    q = message.pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class TebTurnSupervisor:
    """Small state machine which owns only explicit in-place turn actions."""

    def __init__(self):
        rospy.init_node("lste_teb_turn_supervisor")
        gp = rospy.get_param

        self.planner_cmd_topic = gp(
            "~planner_cmd_topic", "/lste/cmd_vel/teb_planner"
        )
        self.navfn_plan_topic = gp(
            "~navfn_plan_topic", "/move_base/NavfnROS/plan"
        )
        self.output_cmd_topic = gp("~output_cmd_topic", "/lste/cmd_vel/teb")
        self.goal_topic = gp("~goal_topic", "/lste/final_goal")
        self.intent_topic = gp("~intent_topic", "/lste/goal_intent")
        self.bridge_status_topic = gp(
            "~bridge_status_topic", "/lste/teb_goal_bridge/status"
        )
        self.pose_topic = gp("~pose_topic", "/rbt_pose")
        self.pose_frame = str(gp("~pose_frame", "odom")).strip().lstrip("/") or "odom"
        self.mode_topic = gp("~mode_topic", "/lste/controller_mode")
        self.task_done_topic = gp("~task_done_topic", "/lste/task_done")
        self.navigation_hold_topic = gp(
            "~navigation_hold_topic", "/lste/navigation_hold"
        )
        self.status_topic = gp(
            "~status_topic", "/lste/teb_turn_supervisor/status"
        )
        self.active_mode = str(gp("~active_mode", "teb")).strip().lower()
        self.mode = self.active_mode
        self.command_frequency = max(
            5.0,
            float(
                gp(
                    "~command_frequency",
                    rospy.get_param("/move_base/controller_frequency", 20.0),
                )
            ),
        )
        # These are execution limits already owned by TEB. Reading them keeps
        # the action envelope coherent without introducing a second set of
        # user-tuned velocity parameters.
        self.max_vel_theta = max(
            0.05,
            float(
                gp(
                    "~max_vel_theta",
                    rospy.get_param(
                        "/move_base/TebLocalPlannerROS/max_vel_theta", 0.65
                    ),
                )
            ),
        )
        self.acc_lim_theta = max(
            0.05,
            float(
                gp(
                    "~acc_lim_theta",
                    rospy.get_param(
                        "/move_base/TebLocalPlannerROS/acc_lim_theta", 0.90
                    ),
                )
            ),
        )
        self.yaw_goal_tolerance = max(
            0.02,
            float(
                gp(
                    "~yaw_goal_tolerance",
                    rospy.get_param(
                        "/move_base/TebLocalPlannerROS/yaw_goal_tolerance", 0.35
                    ),
                )
            ),
        )
        self.planner_command_timeout = max(
            0.10, float(gp("~planner_command_timeout", 0.30))
        )
        # A frontier endpoint action can be replaced in-place by the bridge
        # while the map is still settling.  Starting a pre-route turn the
        # instant a managed action appears makes the supervisor chase every
        # transient goal change and spin in place repeatedly.  Require the
        # same action identity to persist for this window before the turn
        # begins, so the robot only rotates once the route is committed.
        self.turn_start_confirm_duration = max(
            0.0, float(gp("~turn_start_confirm_duration", 0.50))
        )
        # Hard safety net: a single alignment turn may legitimately overshoot
        # by a small margin, but it must never wind up a full circle or
        # re-start on the same action.  If the cumulative absolute rotation
        # during one turn exceeds this many times the initial heading error
        # (or a full revolution), force-release the turn.
        self.turn_rotation_cap_factor = max(
            1.1, float(gp("~turn_rotation_cap_factor", 1.6))
        )
        # After a completed pre-route turn, hold the achieved heading for this
        # long instead of immediately passing TEB's command through.  TEB's
        # timed elastic band is still anchored to the previous goal and can
        # command a rotation back the other way; a short settle lets it
        # re-optimise against the robot's new heading.
        self.turn_settle_duration = max(
            0.0, float(gp("~turn_settle_duration", 0.35))
        )
        # A frontier_endpoint pre-route alignment turn is started when the
        # Navfn path tangent is more than this angle behind the base heading.
        # The original 90 deg left 60-90 deg heading changes to TEB's
        # while-moving reorientation, and after a goal change TEB's hot-started
        # band briefly keeps the OLD heading -- so the robot drifted toward the
        # wall it had already mapped before turning.  Lowering the threshold to
        # 75 deg makes those medium turns explicit pre-route rotations instead.
        # The rotation cap and post-turn settle keep them single, clean turns.
        self.pre_route_turn_threshold = math.radians(max(
            30.0, min(90.0, float(gp("~pre_route_turn_threshold_deg", 75.0)))
        ))

        self.lock = threading.RLock()
        self.state = STATE_PASS_THROUGH
        self.task_done = False
        self.navigation_hold = False
        self.pose = None
        self.latest_goal = None
        # ``latest_goal`` is mission-layer intent and may be queued behind a
        # running move_base action.  Turns are allowed only for this
        # bridge-owned active action goal.
        self.active_action_goal = None
        # A turn connector may execute at the terminal robot pose while its
        # mission identity remains the original frontier source pose. Keep
        # both records: identity gates the transaction, execution pose drives
        # the actuator and turn key.
        self.active_action_source_goal = None
        self.active_action_route_kind = ""
        self.active_action_source = "unknown"
        self.active_action_identity = None
        self.active_action = False
        self.latest_navfn_plan = None
        self.pre_turn_checked_identity = None
        self.latest_planner_command = Twist()
        self.latest_planner_command_wall = 0.0
        self.latest_route_kind = ""
        self.latest_intent_source = "unknown"
        self.latest_intent_priority = 0
        self.latest_intent_goal = None
        self.turn_target_yaw = None
        self.turn_goal_xy = None
        self.turn_key = None
        self.completed_turn_key = None
        self.turn_velocity = 0.0
        self.turn_started_wall = 0.0
        # Turn-start confirmation: the managed action identity must stay
        # unchanged for ``turn_start_confirm_duration`` before a pre-route
        # turn may begin.
        self.turn_pending_identity = None
        self.turn_pending_since_wall = 0.0
        # Rotation budget: cumulative absolute rotation during the active turn
        # and the cap derived from the initial heading error.
        self.turn_initial_abs_error = 0.0
        self.turn_abs_rotation = 0.0
        self.turn_rotation_cap = 0.0
        # Post-turn settle: the achieved heading to hold briefly after release.
        self.turn_settle_until_wall = 0.0
        self.turn_settle_yaw = None
        self.turn_capped_releases = 0
        self.turn_count = 0
        self.turn_completed_count = 0
        self.turn_released_count = 0
        self.last_status_wall = 0.0
        self.tf_listener = tf.TransformListener()

        self.output_pub = rospy.Publisher(
            self.output_cmd_topic, Twist, queue_size=10
        )
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=10, latch=True
        )
        rospy.Subscriber(
            self.planner_cmd_topic, Twist, self.on_planner_command, queue_size=1
        )
        rospy.Subscriber(
            self.navfn_plan_topic, Path, self.on_navfn_plan, queue_size=1
        )
        rospy.Subscriber(self.goal_topic, PoseStamped, self.on_goal, queue_size=1)
        rospy.Subscriber(self.intent_topic, String, self.on_intent, queue_size=1)
        rospy.Subscriber(
            self.bridge_status_topic,
            String,
            self.on_bridge_status,
            queue_size=10,
        )
        rospy.Subscriber(self.pose_topic, Pose2D, self.on_pose, queue_size=1)
        rospy.Subscriber(self.mode_topic, String, self.on_mode, queue_size=1)
        rospy.Subscriber(self.task_done_topic, Bool, self.on_task_done, queue_size=1)
        rospy.Subscriber(
            self.navigation_hold_topic,
            Bool,
            self.on_navigation_hold,
            queue_size=1,
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.command_frequency), self.on_timer
        )
        rospy.on_shutdown(self.on_shutdown)

        self.publish_status_locked("startup")
        rospy.loginfo(
            "TEB turn supervisor ready: planner=%s navfn_plan=%s output=%s frequency=%.1fHz "
            "max_theta=%.3f acc_theta=%.3f yaw_tolerance=%.3f",
            self.planner_cmd_topic,
            self.navfn_plan_topic,
            self.output_cmd_topic,
            self.command_frequency,
            self.max_vel_theta,
            self.acc_lim_theta,
            self.yaw_goal_tolerance,
        )

    @staticmethod
    def _goal_xy(message):
        if message is None:
            return None
        return (
            float(message.pose.position.x),
            float(message.pose.position.y),
        )

    def _goal_yaw_in_pose_frame_locked(self, goal):
        """Transform a route tangent once into the odometry execution frame."""
        if goal is None:
            return None
        source_frame = (goal.header.frame_id or self.pose_frame).strip().lstrip("/")
        if source_frame == self.pose_frame:
            return normalize_angle(angle_from_pose(goal))
        transformed = copy.deepcopy(goal)
        transformed.header.stamp = rospy.Time(0)
        try:
            self.tf_listener.waitForTransform(
                self.pose_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(0.20),
            )
            transformed = self.tf_listener.transformPose(
                self.pose_frame, transformed
            )
        except (
            tf.Exception,
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                3.0,
                "TEB turn supervisor waiting for %s <- %s yaw transform: %s",
                self.pose_frame,
                source_frame,
                exc,
            )
            return None
        return normalize_angle(angle_from_pose(transformed))

    def _intent_matches_goal_locked(self, goal):
        if goal is None or self.latest_intent_goal is None:
            return True
        xy = self._goal_xy(goal)
        return math.hypot(
            xy[0] - self.latest_intent_goal[0],
            xy[1] - self.latest_intent_goal[1],
        ) <= 0.08

    def _is_frontier_endpoint_action_locked(self):
        return (
            self.active_action
            and self.active_action_route_kind == FRONTIER_ENDPOINT_KIND
            and self.active_action_source == FRONTIER_SOURCE
        )

    def _is_managed_action_locked(self):
        """Return whether this execution adapter owns the active action phase."""
        return (
            self.active_action
            and self.active_action_route_kind == TURN_ROUTE_KIND
        ) or self._is_frontier_endpoint_action_locked()

    def _active_action_key_locked(self):
        goal = self.active_action_goal
        if goal is None:
            return None
        return (
            self.active_action_route_kind,
            self.active_action_source,
            round(float(goal.pose.position.x), 3),
            round(float(goal.pose.position.y), 3),
            round(normalize_angle(angle_from_pose(goal)), 3),
        )

    def _initial_navfn_yaw_in_pose_frame_locked(self):
        """Return the first non-zero Navfn path tangent in the odom frame."""
        plan = self.latest_navfn_plan
        if plan is None or len(plan.poses) < 2:
            return None
        source_frame = (plan.header.frame_id or self.pose_frame).strip().lstrip(
            "/"
        ) or self.pose_frame
        transformed = []
        # The first pose can be duplicated by Navfn at the robot location. The
        # first pair separated by a physical map step is the route tangent.
        for pose in plan.poses[: min(len(plan.poses), 24)]:
            candidate = copy.deepcopy(pose)
            candidate.header.frame_id = source_frame
            candidate.header.stamp = rospy.Time(0)
            if source_frame != self.pose_frame:
                try:
                    self.tf_listener.waitForTransform(
                        self.pose_frame,
                        source_frame,
                        rospy.Time(0),
                        rospy.Duration(0.05),
                    )
                    candidate = self.tf_listener.transformPose(
                        self.pose_frame, candidate
                    )
                except (
                    tf.Exception,
                    tf.LookupException,
                    tf.ConnectivityException,
                    tf.ExtrapolationException,
                ) as exc:
                    rospy.logwarn_throttle(
                        3.0,
                        "TEB turn supervisor waiting for %s <- %s plan transform: %s",
                        self.pose_frame,
                        source_frame,
                        exc,
                    )
                    return None
            transformed.append(candidate)
        if len(transformed) < 2:
            return None
        first = transformed[0].pose.position
        for candidate in transformed[1:]:
            point = candidate.pose.position
            distance = math.hypot(point.x - first.x, point.y - first.y)
            if distance >= 0.12:
                return normalize_angle(math.atan2(point.y - first.y, point.x - first.x))
        return None

    def _turn_key_for_goal_locked(self, goal, target_yaw=None):
        if goal is None:
            return None
        xy = self._goal_xy(goal)
        if target_yaw is None:
            target_yaw = angle_from_pose(goal)
        return (
            round(xy[0], 3),
            round(xy[1], 3),
            round(normalize_angle(target_yaw), 3),
        )

    def publish_status_locked(self, event, **fields):
        pose = self.pose
        error = None
        if pose is not None and self.turn_target_yaw is not None:
            error = normalize_angle(self.turn_target_yaw - pose.theta)
        payload = {
            "event": str(event),
            "state": self.state,
            "mode": self.mode,
            "route_kind": self.latest_route_kind,
            "active_action": bool(self.active_action),
            "active_route_kind": self.active_action_route_kind,
            "intent_source": self.latest_intent_source,
            "intent_priority": int(self.latest_intent_priority),
            "goal": None
            if self.turn_goal_xy is None
            else [round(value, 3) for value in self.turn_goal_xy],
            "target_yaw": None
            if self.turn_target_yaw is None
            else round(float(self.turn_target_yaw), 4),
            "pose": None
            if pose is None
            else [round(float(pose.x), 3), round(float(pose.y), 3), round(float(pose.theta), 4)],
            "yaw_error": None if error is None else round(float(error), 4),
            "turn_velocity": round(float(self.turn_velocity), 4),
            "turns": int(self.turn_count),
            "completed_turns": int(self.turn_completed_count),
            "released_turns": int(self.turn_released_count),
        }
        payload.update(fields)
        try:
            self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        except (TypeError, ValueError):
            rospy.logwarn_throttle(5.0, "TEB turn supervisor status serialization failed")

    def _release_turn_locked(self, reason, completed=False):
        if self.state != STATE_TURNING:
            self.turn_target_yaw = None
            self.turn_goal_xy = None
            self.turn_velocity = 0.0
            return
        previous_key = self.turn_key
        previous_error = None
        if self.pose is not None and self.turn_target_yaw is not None:
            previous_error = normalize_angle(self.turn_target_yaw - self.pose.theta)
        if completed:
            self.completed_turn_key = previous_key
            self.turn_completed_count += 1
            # After a successfully completed pre-route turn, hold the achieved
            # heading briefly so TEB's band can re-anchor to the new pose
            # instead of immediately commanding a rotation back the other way.
            self.turn_settle_until_wall = (
                time.monotonic() + self.turn_settle_duration
                if self.turn_settle_duration > 0.0 else 0.0
            )
            self.turn_settle_yaw = (
                self.pose.theta
                if self.pose is not None
                else self.turn_target_yaw
            )
        else:
            self.turn_settle_until_wall = 0.0
            self.turn_settle_yaw = None
        self.turn_released_count += 1
        self.state = STATE_PASS_THROUGH
        self.turn_target_yaw = None
        self.turn_goal_xy = None
        self.turn_key = None
        self.turn_velocity = 0.0
        self.publish_status_locked(
            "turn_completed" if completed else "turn_released",
            reason=str(reason),
            previous_key=previous_key,
            previous_yaw_error=(
                None if previous_error is None else round(float(previous_error), 4)
            ),
        )
        rospy.loginfo(
            "TEB turn supervisor released turn: reason=%s completed=%s",
            reason,
            completed,
        )

    def _activate_turn_locked(self):
        if (
            self.mode != self.active_mode
            or self.task_done
            or self.navigation_hold
            or not self._is_managed_action_locked()
            or self.active_action_goal is None
            or not self._intent_matches_goal_locked(
                self.active_action_source_goal or self.active_action_goal
            )
        ):
            return False
        endpoint_alignment = self._is_frontier_endpoint_action_locked()
        action_identity = self.active_action_identity or self._active_action_key_locked()
        if endpoint_alignment:
            # A route endpoint is still one action.  Only evaluate its initial
            # tangent once, after Navfn has published the plan, so a later SLAM
            # update cannot restart a turn while the vehicle is moving.
            if self.pre_turn_checked_identity == action_identity:
                return False
            target_yaw = self._initial_navfn_yaw_in_pose_frame_locked()
            if target_yaw is None or self.pose is None:
                return False
            heading_error = normalize_angle(target_yaw - self.pose.theta)
            self.pre_turn_checked_identity = action_identity
            # This is a route-topology decision, not a velocity threshold: the
            # base must first face a path that starts behind it.  Bends within
            # ``pre_route_turn_threshold`` remain under TEB's normal optimizer;
            # larger ones get an explicit pre-route rotation so the robot does
            # not drift toward a mapped wall while TEB re-orients.
            if abs(heading_error) <= self.pre_route_turn_threshold:
                return False
        else:
            target_yaw = self._goal_yaw_in_pose_frame_locked(self.active_action_goal)
            if target_yaw is None:
                return False
        # Keep the action key in the goal's native map coordinates.  The
        # map->odom transform can legitimately change while SLAM integrates a
        # scan; using the transformed odom yaw as the key would make one
        # physical turn look like a new action and restart it repeatedly.
        key = self._turn_key_for_goal_locked(self.active_action_goal, target_yaw)
        if key is None or key == self.completed_turn_key:
            return False
        if self.state == STATE_TURNING and key == self.turn_key:
            return True
        if self.state == STATE_TURNING:
            # A connector is an atomic execution action.  Map refreshes may
            # publish the next route pose while this turn is still running,
            # but they must not retarget the actuator mid-rotation.  The
            # latest goal remains cached and is considered after completion.
            return False
        # Do not start a pre-route turn the instant a managed action appears:
        # the bridge may replace it in-place again as the online map settles,
        # and turning for every transient goal produces the repeated spinning
        # the operator rejected.  Wait until the same turn key has been the
        # candidate for ``turn_start_confirm_duration``.
        now = time.monotonic()
        if self.turn_pending_identity != key:
            self.turn_pending_identity = key
            self.turn_pending_since_wall = now
            return False
        if now - self.turn_pending_since_wall < self.turn_start_confirm_duration:
            return False
        self.state = STATE_TURNING
        self.turn_key = key
        self.turn_target_yaw = target_yaw
        self.turn_goal_xy = self._goal_xy(self.active_action_goal)
        self.turn_velocity = 0.0
        self.turn_started_wall = now
        if endpoint_alignment:
            initial_abs_error = abs(heading_error)
        else:
            initial_abs_error = (
                abs(normalize_angle(target_yaw - self.pose.theta))
                if self.pose is not None else math.pi
            )
        self.turn_initial_abs_error = initial_abs_error
        self.turn_abs_rotation = 0.0
        self.turn_rotation_cap = max(
            initial_abs_error * self.turn_rotation_cap_factor,
            math.radians(120.0),
        )
        self.turn_count += 1
        self.publish_status_locked(
            "turn_started",
            source_goal=[round(value, 3) for value in self.turn_goal_xy],
            turn_phase=("pre_route_alignment" if endpoint_alignment else "legacy_connector"),
            plan_heading=round(float(target_yaw), 4),
        )
        rospy.loginfo(
            "TEB turn supervisor started %s: target=(%.2f,%.2f) yaw=%.1fdeg "
            "source=%s",
            "pre-route alignment" if endpoint_alignment else "legacy connector turn",
            self.turn_goal_xy[0],
            self.turn_goal_xy[1],
            math.degrees(target_yaw),
            self.latest_intent_source,
        )
        return True

    def on_intent(self, message):
        source = "unknown"
        priority = 0
        route_kind = ""
        intent_goal = None
        try:
            payload = json.loads(message.data)
            if isinstance(payload, dict):
                source = str(payload.get("source", source)).strip().lower() or source
                priority = int(payload.get("priority", priority))
                route_kind = str(payload.get("route_kind", "")).strip().lower()
                raw_goal = payload.get("goal")
                if isinstance(raw_goal, (list, tuple)) and len(raw_goal) >= 2:
                    intent_goal = (float(raw_goal[0]), float(raw_goal[1]))
            else:
                source = str(message.data).strip().lower() or source
        except (TypeError, ValueError, json.JSONDecodeError):
            source = str(message.data).strip().lower() or source
        with self.lock:
            self.latest_intent_source = source
            self.latest_intent_priority = max(0, min(3, priority))
            self.latest_route_kind = route_kind
            self.latest_intent_goal = intent_goal
            endpoint_intent = (
                route_kind == FRONTIER_ENDPOINT_KIND
                and source == FRONTIER_SOURCE
            )
            if route_kind != TURN_ROUTE_KIND and not endpoint_intent:
                self._release_turn_locked("route_contract_released", completed=False)
            else:
                # The intent only describes a pending mission decision.  The
                # bridge status callback below is the authority that makes it
                # executable.
                self._activate_turn_locked()

    def on_bridge_status(self, message):
        """Synchronize execution state with the bridge's active action.

        A final goal topic is intentionally allowed to be ahead of move_base;
        using it directly here made the supervisor rotate for a goal that was
        still queued.  The bridge publishes the dispatched pose and route kind
        as one lifecycle record, so this node can execute only that action.
        """
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self.lock:
            previous_identity = self.active_action_identity
            self.active_action = bool(payload.get("active", False))
            self.active_action_route_kind = str(
                payload.get("active_route_kind", "")
            ).strip().lower()
            self.active_action_source = str(
                payload.get("active_intent_source", "unknown")
            ).strip().lower() or "unknown"
            raw_goal = payload.get("active_goal")
            frame = str(payload.get("active_goal_frame", "") or "map").strip()
            if (
                isinstance(raw_goal, (list, tuple))
                and len(raw_goal) >= 2
                and self.active_action
            ):
                active_goal = PoseStamped()
                active_goal.header.stamp = rospy.Time.now()
                active_goal.header.frame_id = frame.lstrip("/") or "map"
                active_goal.pose.position.x = float(raw_goal[0])
                active_goal.pose.position.y = float(raw_goal[1])
                yaw = float(raw_goal[2]) if len(raw_goal) >= 3 else 0.0
                active_goal.pose.orientation.z = math.sin(0.5 * yaw)
                active_goal.pose.orientation.w = math.cos(0.5 * yaw)
                self.active_action_goal = active_goal
                raw_source_goal = payload.get("active_source_goal")
                source_frame = str(
                    payload.get("active_source_goal_frame", frame) or frame
                ).strip().lstrip("/") or "map"
                if (
                    isinstance(raw_source_goal, (list, tuple))
                    and len(raw_source_goal) >= 2
                ):
                    source_goal = PoseStamped()
                    source_goal.header.stamp = rospy.Time.now()
                    source_goal.header.frame_id = source_frame
                    source_goal.pose.position.x = float(raw_source_goal[0])
                    source_goal.pose.position.y = float(raw_source_goal[1])
                    source_yaw = (
                        float(raw_source_goal[2])
                        if len(raw_source_goal) >= 3 else 0.0
                    )
                    source_goal.pose.orientation.z = math.sin(0.5 * source_yaw)
                    source_goal.pose.orientation.w = math.cos(0.5 * source_yaw)
                    self.active_action_source_goal = source_goal
                else:
                    self.active_action_source_goal = None
            else:
                self.active_action_goal = None
                self.active_action_source_goal = None
            self.active_action_identity = self._active_action_key_locked()
            if self.active_action_identity != previous_identity:
                self.pre_turn_checked_identity = None
                self.completed_turn_key = None
                self.latest_navfn_plan = None
            if not self._is_managed_action_locked():
                if self.state == STATE_TURNING:
                    self._release_turn_locked("bridge_action_changed", completed=False)
            else:
                self._activate_turn_locked()

    def on_goal(self, message):
        goal = copy.deepcopy(message)
        if not goal.header.frame_id:
            goal.header.frame_id = "odom"
        with self.lock:
            self.latest_goal = goal
            if self._is_managed_action_locked():
                activated = self._activate_turn_locked()
                if (
                    not activated
                    and self.state == STATE_TURNING
                    and not self._intent_matches_goal_locked(goal)
                ):
                    # A different frontier pose is a queued next segment, not
                    # permission to interrupt the current turn.  The bridge
                    # applies the same atomic-action rule below.
                    pass
            elif self.state == STATE_TURNING:
                self._release_turn_locked("goal_route_changed", completed=False)

    def on_navfn_plan(self, message):
        """Cache the active global path for one-time endpoint alignment."""
        with self.lock:
            self.latest_navfn_plan = copy.deepcopy(message)
            if self._is_frontier_endpoint_action_locked():
                self._activate_turn_locked()

    def on_pose(self, message):
        with self.lock:
            self.pose = copy.deepcopy(message)

    def on_planner_command(self, message):
        with self.lock:
            self.latest_planner_command = copy.deepcopy(message)
            self.latest_planner_command_wall = time.monotonic()

    def on_mode(self, message):
        mode = message.data.strip().lower()
        if not mode:
            return
        with self.lock:
            previous = self.mode
            self.mode = mode
            if mode != self.active_mode and self.state == STATE_TURNING:
                self._release_turn_locked("controller_switched_to_%s" % mode, completed=False)
            elif previous != self.active_mode and mode == self.active_mode:
                self._activate_turn_locked()

    def on_task_done(self, message):
        with self.lock:
            self.task_done = bool(message.data)
            if self.task_done:
                self._release_turn_locked("task_done", completed=False)
            else:
                # A new mission may legitimately revisit the same map pose.
                self.completed_turn_key = None
                self._activate_turn_locked()

    def on_navigation_hold(self, message):
        with self.lock:
            self.navigation_hold = bool(message.data)
            if self.navigation_hold:
                self._release_turn_locked("navigation_hold", completed=False)
            else:
                self._activate_turn_locked()

    def _turn_command_locked(self, dt):
        """Return one acceleration-limited turn command.

        The speed envelope is derived from the remaining angle and TEB's
        acceleration limit: ``v = sqrt(2*a*remaining_angle)``.  This is the
        standard braking envelope for a bounded angular actuator and avoids a
        separately tuned proportional gain.
        """
        if self.pose is None or self.turn_target_yaw is None:
            self.turn_velocity = 0.0
            return Twist()
        error = normalize_angle(self.turn_target_yaw - self.pose.theta)
        if abs(error) <= self.yaw_goal_tolerance:
            self._release_turn_locked("yaw_tolerance_reached", completed=True)
            return copy.deepcopy(self.latest_planner_command)

        remaining = max(0.0, abs(error) - self.yaw_goal_tolerance)
        desired = min(
            self.max_vel_theta,
            math.sqrt(max(0.0, 2.0 * self.acc_lim_theta * remaining)),
        )
        requested_sign = 1.0 if error > 0.0 else -1.0
        previous = float(self.turn_velocity)
        step = self.acc_lim_theta * max(0.001, min(0.2, float(dt)))
        if previous != 0.0 and math.copysign(1.0, previous) != requested_sign:
            magnitude = max(0.0, abs(previous) - step)
            self.turn_velocity = math.copysign(magnitude, previous) if magnitude else 0.0
        else:
            previous_magnitude = abs(previous)
            if desired >= previous_magnitude:
                magnitude = min(desired, previous_magnitude + step)
            else:
                magnitude = max(desired, previous_magnitude - step)
            self.turn_velocity = requested_sign * magnitude

        # Rotation budget: if this single turn has already swept more than the
        # cap (1.6x the initial heading error, floor 120 deg), something is
        # preventing convergence (stale pose, repeated re-targeting).  Release
        # instead of letting the robot wind up a full circle.
        self.turn_abs_rotation += abs(self.turn_velocity) * max(0.001, float(dt))
        if self.turn_abs_rotation > self.turn_rotation_cap:
            self.turn_capped_releases += 1
            rospy.logwarn(
                "TEB turn supervisor forced turn release: abs_rotation=%.0fdeg "
                "cap=%.0fdeg initial_error=%.0fdeg",
                math.degrees(self.turn_abs_rotation),
                math.degrees(self.turn_rotation_cap),
                math.degrees(self.turn_initial_abs_error),
            )
            self._release_turn_locked("rotation_budget_exceeded", completed=False)
            return copy.deepcopy(self.latest_planner_command)

        command = Twist()
        command.angular.z = self.turn_velocity
        return command

    def on_timer(self, _event):
        with self.lock:
            if (
                self.mode == self.active_mode
                and not self.task_done
                and not self.navigation_hold
                and self.state == STATE_TURNING
            ):
                command = self._turn_command_locked(
                    1.0 / self.command_frequency
                )
                state = self.state
            elif (
                self.mode == self.active_mode
                and not self.task_done
                and not self.navigation_hold
            ):
                if (
                    time.monotonic() - self.latest_planner_command_wall
                    <= self.planner_command_timeout
                ):
                    command = copy.deepcopy(self.latest_planner_command)
                else:
                    command = Twist()
                # Immediately after a completed pre-route turn, hold the
                # achieved heading (gentle correction) for the settle window.
                # TEB's band is still anchored to the old goal and would
                # otherwise command a rotation back the other way, which looks
                # like the robot spinning in place.
                if (
                    self.turn_settle_until_wall > 0.0
                    and self.pose is not None
                    and self.turn_settle_yaw is not None
                ):
                    if time.monotonic() < self.turn_settle_until_wall:
                        yaw_err = normalize_angle(
                            self.turn_settle_yaw - self.pose.theta
                        )
                        command = copy.deepcopy(command)
                        command.angular.z = max(
                            -0.20, min(0.20, yaw_err * 0.6)
                        )
                    else:
                        self.turn_settle_until_wall = 0.0
                        self.turn_settle_yaw = None
                state = self.state
            else:
                command = Twist()
                state = self.state
            if (
                state == STATE_PASS_THROUGH
                and self._is_managed_action_locked()
                and not self.task_done
                and not self.navigation_hold
            ):
                # A new endpoint plan may arrive after the bridge status.  The
                # identity gate in _activate_turn_locked makes this a one-time
                # pre-route phase rather than a repeated turn loop.
                self._activate_turn_locked()
        self.output_pub.publish(command)
        now = time.monotonic()
        with self.lock:
            if now - self.last_status_wall >= 1.0:
                self.last_status_wall = now
                self.publish_status_locked("heartbeat", command=[
                    round(float(command.linear.x), 4),
                    round(float(command.angular.z), 4),
                ])

    def on_shutdown(self):
        try:
            self.output_pub.publish(Twist())
        except Exception:
            pass


if __name__ == "__main__":
    TebTurnSupervisor()
    rospy.spin()
