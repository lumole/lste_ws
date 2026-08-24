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
* legacy ``frontier_turn_connector`` actions get an explicit in-place turn;
* online-SLAM ``frontier_endpoint`` alignment is an opt-in comparison mode,
  not a second production local controller;
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
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from teb_local_planner.msg import FeedbackMsg


TURN_ROUTE_KIND = "frontier_turn_connector"
FRONTIER_ENDPOINT_KIND = "frontier_endpoint"
FRONTIER_SOURCE = "global_slam_frontier"
STATE_PASS_THROUGH = "PASS_THROUGH"
STATE_TURNING = "TURNING"
# This is a route-geometry invariant, rather than a controller tuning knob.
# When Navfn starts more than a quarter turn behind the base, forwarding an
# old forward TEB sample is never a continuous trajectory; the new route must
# first acquire its own in-place turn command.
SHARP_ENTRY_CONTINUITY_BLOCK_RAD = math.pi / 2.0


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
        self.scan_topic = gp("~scan_topic", "/pro3/rlscan")
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
        # A valid TEB trajectory can briefly disagree with its raw cmd_vel
        # callback: either an all-zero sample or a reverse-only sample appears
        # before the selected trajectory resumes forward motion. A forward-only
        # base cannot execute the latter, and a mux clamp would turn it into an
        # unnecessary brake pulse. Use only TEB's currently selected forward
        # trajectory for this bounded adapter gap; never synthesize a command.
        self.trajectory_continuity_enabled = str(
            gp("~trajectory_continuity_enabled", True)
        ).strip().lower() in ("1", "true", "yes", "on")
        # Feedback arrives on a separate callback stream from cmd_vel.  In a
        # loaded Gazebo/SLAM run it can be slower than the controller timer;
        # treating a fixed 200 ms age as stale then creates an artificial
        # zero-command pulse between otherwise valid TEB trajectories.  Keep
        # a configured lower bound, adapt to the measured feedback cadence,
        # and cap it so a genuinely lost feedback stream is never trusted.
        self.trajectory_feedback_timeout = max(
            0.02, float(gp("~trajectory_feedback_timeout", 0.35))
        )
        self.trajectory_feedback_timeout_cap = max(
            self.trajectory_feedback_timeout,
            float(gp("~trajectory_feedback_timeout_cap", 0.60)),
        )
        self.trajectory_feedback_period_scale = max(
            1.0, float(gp("~trajectory_feedback_period_scale", 1.50))
        )
        self.trajectory_continuity_min_forward = max(
            0.01, float(gp("~trajectory_continuity_min_forward", 0.05))
        )
        self.trajectory_continuity_max_hold = max(
            0.02, float(gp("~trajectory_continuity_max_hold", 0.10))
        )
        # TEB feedback can lag one callback behind a terminal action. A
        # feedback velocity is therefore never used near the dispatched goal:
        # the raw zero remains authoritative in that arrival envelope.
        self.trajectory_continuity_terminal_radius = max(
            0.05, float(gp("~trajectory_continuity_terminal_radius", 0.75))
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
        # Production has exactly one local trajectory controller: TEB. An
        # external pre-route rotation can fight its obstacle optimizer after a
        # changing online-SLAM plan, producing the observed rotate/stop loop.
        # Keep the old endpoint-alignment policy available for an explicit
        # comparison, while legacy connector actions remain semantic turns.
        self.endpoint_alignment_enabled = str(
            gp("~endpoint_alignment_enabled", False)
        ).strip().lower() in ("1", "true", "yes", "on")
        # A forward-only base occasionally reaches a valid Navfn route whose
        # first tangent is moderately behind it.  This is below the proactive
        # alignment threshold, yet TEB can still settle on a near-zero band
        # instead of choosing a forward arc.  Treat that as an execution
        # deadlock only after live TEB feedback has proved it is persistent;
        # then rotate once toward Navfn's already validated initial tangent.
        # This is intentionally a route-health recovery, not another local
        # controller and it never synthesises a translational command.
        self.stalled_route_reorientation_enabled = str(
            gp("~stalled_route_reorientation_enabled", True)
        ).strip().lower() in ("1", "true", "yes", "on")
        self.stalled_route_reorientation_delay = max(
            0.10, float(gp("~stalled_route_reorientation_delay", 0.80))
        )
        self.stalled_route_reorientation_heading = math.radians(max(
            20.0, min(90.0, float(gp(
                "~stalled_route_reorientation_heading_deg", 45.0
            )))
        ))
        self.stalled_route_reorientation_linear = max(
            0.001, float(gp("~stalled_route_reorientation_linear", 0.03))
        )
        self.stalled_route_reorientation_angular = max(
            0.01, float(gp("~stalled_route_reorientation_angular", 0.15))
        )
        self.stalled_route_reorientation_min_distance = max(
            0.20, float(gp("~stalled_route_reorientation_min_distance", 1.00))
        )
        # The robot footprint is circular, therefore turning cannot make a
        # collision safer. Never let this execution adapter override TEB's
        # own feasibility/recovery decision while a lidar return is already
        # inside the physical footprint. Derive the limit from TEB's model
        # instead of introducing another user-facing clearance parameter.
        self.turn_min_clearance = max(
            0.05,
            float(rospy.get_param(
                "/move_base/TebLocalPlannerROS/footprint_model/radius", 0.30
            )) + 0.02,
        )
        self.turn_scan_timeout = 0.50

        self.lock = threading.RLock()
        self.state = STATE_PASS_THROUGH
        self.task_done = False
        self.navigation_hold = False
        self.pose = None
        self.scan_minimum = float("inf")
        self.scan_monotonic = 0.0
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
        self.active_action_priority = 0
        # Target continuity is allowed only within a bridge-confirmed detector
        # track.  A new visual target or frontier->target ownership change must
        # establish its own trajectory from rest.
        self.active_action_target_track_id = ""
        self.active_action_identity = None
        self.active_action = False
        self.latest_navfn_plan = None
        self.pre_turn_checked_identity = None
        self.latest_planner_command = Twist()
        self.latest_planner_command_wall = 0.0
        self.latest_trajectory_command = Twist()
        self.latest_trajectory_command_wall = 0.0
        self.trajectory_feedback_period_ema = None
        self.trajectory_zero_started_wall = 0.0
        self.trajectory_continuity_events = 0
        self.trajectory_continuity_goal_distance = None
        self.trajectory_continuity_sharp_entry_identity = None
        self.trajectory_continuity_sharp_entry_suppressions = 0
        self.latest_route_kind = ""
        self.latest_intent_source = "unknown"
        self.latest_intent_priority = 0
        self.latest_intent_goal = None
        self.latest_target_track_id = ""
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
        self.stalled_route_candidate_identity = None
        self.stalled_route_candidate_since_wall = 0.0
        # Readiness and completion are distinct.  The route must remain
        # eligible through the normal turn-start confirmation window before
        # this action is considered recovered.
        self.stalled_route_ready_identity = None
        self.stalled_route_completed_identity = None
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
            "/move_base/TebLocalPlannerROS/teb_feedback",
            FeedbackMsg,
            self.on_teb_feedback,
            queue_size=1,
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
        rospy.Subscriber(self.scan_topic, LaserScan, self.on_scan, queue_size=1)
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
            "max_theta=%.3f acc_theta=%.3f yaw_tolerance=%.3f continuity=%s "
            "stalled_route_reorientation=%s",
            self.planner_cmd_topic,
            self.navfn_plan_topic,
            self.output_cmd_topic,
            self.command_frequency,
            self.max_vel_theta,
            self.acc_lim_theta,
            self.yaw_goal_tolerance,
            self.trajectory_continuity_enabled,
            self.stalled_route_reorientation_enabled,
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

    def _goal_distance_in_pose_frame_locked(self, goal):
        """Return planar distance to ``goal`` in the execution pose frame.

        The action goal is normally in ``map`` while the base pose is in
        ``odom``. Comparing their raw coordinates would make the terminal
        safety gate meaningless once online SLAM updates map->odom, so this
        helper always transforms the action goal first.
        """
        if goal is None or self.pose is None:
            return None
        source_frame = (goal.header.frame_id or self.pose_frame).strip().lstrip("/")
        transformed = copy.deepcopy(goal)
        transformed.header.stamp = rospy.Time(0)
        if source_frame != self.pose_frame:
            try:
                self.tf_listener.waitForTransform(
                    self.pose_frame,
                    source_frame,
                    rospy.Time(0),
                    rospy.Duration(0.05),
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
                    "TEB continuity gate waiting for %s <- %s goal transform: %s",
                    self.pose_frame,
                    source_frame,
                    exc,
                )
                return None
        return math.hypot(
            float(transformed.pose.position.x) - float(self.pose.x),
            float(transformed.pose.position.y) - float(self.pose.y),
        )

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

    def _is_same_target_segment_continuation_locked(self):
        """Whether a target replacement keeps one continuous target identity.

        Detector source names alone are insufficient: a newly acquired target
        may need an immediate stop.  Only the bridge-propagated, non-empty
        track id proves both visual segments belong to the same object.
        """
        return (
            self.active_action
            and self.active_action_priority == 2
            and self.active_action_source.startswith("target_")
            and bool(self.active_action_target_track_id)
            and self.latest_intent_priority == 2
            and self.latest_intent_source.startswith("target_")
            and self.latest_target_track_id == self.active_action_target_track_id
        )

    def _is_continuity_eligible_action_locked(self):
        """The only action classes permitted to bridge a transient TEB gap."""
        return (
            self._is_managed_action_locked()
            or self._is_same_target_segment_continuation_locked()
        )

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

    def _sharp_navfn_entry_heading_error_locked(self):
        """Return a sharp active-route entry error, otherwise ``None``.

        ``latest_navfn_plan`` is reset whenever the bridge installs a new
        action identity, so a cached tangent here belongs to the action whose
        raw command is being considered.  A large initial mismatch means the
        old TEB feedback was selected for the preceding route and cannot be
        used as a continuity command for this one.
        """
        if self.pose is None:
            return None
        target_yaw = self._initial_navfn_yaw_in_pose_frame_locked()
        if target_yaw is None:
            return None
        error = normalize_angle(target_yaw - self.pose.theta)
        if abs(error) < SHARP_ENTRY_CONTINUITY_BLOCK_RAD:
            return None
        return error

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
            "scan_minimum": (
                None if not math.isfinite(self.scan_minimum)
                else round(float(self.scan_minimum), 4)
            ),
            "turn_min_clearance": round(float(self.turn_min_clearance), 4),
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

    def _turn_clearance_reason_locked(self, now):
        """Return why the adapter must not command an in-place rotation."""
        if self.scan_monotonic <= 0.0:
            return "scan_unavailable"
        if now - self.scan_monotonic > self.turn_scan_timeout:
            return "scan_stale"
        if self.scan_minimum < self.turn_min_clearance:
            return "obstacle_inside_circular_footprint"
        return ""

    def _stalled_route_reorientation_locked(self, action_identity):
        """Return a Navfn heading only for a persistent clear-path TEB stall.

        A raw zero alone is not evidence of a failed route: it can be a
        terminal, a planner scheduling gap, or an ordinary obstacle turn.
        This guard requires fresh *selected* TEB feedback, a distant endpoint,
        and a meaningful Navfn initial-heading mismatch before it changes the
        execution phase.  The returned heading remains TEB/Navfn-owned data;
        this supervisor supplies rotation only and releases the original action
        to TEB immediately afterwards.
        """
        if (
            not self.stalled_route_reorientation_enabled
            or not self._is_frontier_endpoint_action_locked()
            or action_identity is None
            or action_identity == self.stalled_route_completed_identity
            or self.pose is None
        ):
            self.stalled_route_candidate_identity = None
            self.stalled_route_candidate_since_wall = 0.0
            self.stalled_route_ready_identity = None
            return None
        goal_distance = self._goal_distance_in_pose_frame_locked(
            self.active_action_goal
        )
        feedback_age = max(0.0, time.monotonic() - self.latest_trajectory_command_wall)
        feedback_timeout = self._trajectory_feedback_timeout_locked()
        linear = float(self.latest_trajectory_command.linear.x)
        angular = float(self.latest_trajectory_command.angular.z)
        target_yaw = self._initial_navfn_yaw_in_pose_frame_locked()
        heading_error = (
            None if target_yaw is None
            else normalize_angle(target_yaw - self.pose.theta)
        )
        is_near_zero = (
            abs(linear) <= self.stalled_route_reorientation_linear
            and abs(angular) <= self.stalled_route_reorientation_angular
        )
        eligible = (
            goal_distance is not None
            and goal_distance >= self.stalled_route_reorientation_min_distance
            and feedback_age <= feedback_timeout
            and is_near_zero
            and heading_error is not None
            and abs(heading_error) >= self.stalled_route_reorientation_heading
        )
        if not eligible:
            self.stalled_route_candidate_identity = None
            self.stalled_route_candidate_since_wall = 0.0
            self.stalled_route_ready_identity = None
            return None
        now = time.monotonic()
        if self.stalled_route_candidate_identity != action_identity:
            self.stalled_route_candidate_identity = action_identity
            self.stalled_route_candidate_since_wall = now
            return None
        if now - self.stalled_route_candidate_since_wall < self.stalled_route_reorientation_delay:
            return None
        if self.stalled_route_ready_identity != action_identity:
            self.stalled_route_ready_identity = action_identity
            self.publish_status_locked(
                "stalled_route_reorientation_ready",
                action_goal_distance=round(float(goal_distance), 4),
                feedback_age_seconds=round(float(feedback_age), 4),
                selected_command=[round(linear, 4), round(angular, 4)],
                heading_error_deg=round(math.degrees(heading_error), 2),
            )
            rospy.logwarn(
                "TEB selected a persistent near-zero route command; rotating once "
                "toward Navfn tangent: endpoint_distance=%.2fm heading_error=%.1fdeg "
                "selected=(%.3f,%.3f)",
                goal_distance,
                math.degrees(heading_error),
                linear,
                angular,
            )
        return target_yaw, heading_error

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
        endpoint_alignment = (
            self.endpoint_alignment_enabled
            and self._is_frontier_endpoint_action_locked()
        )
        action_identity = self.active_action_identity or self._active_action_key_locked()
        target_yaw = None
        heading_error = None
        turn_phase = None
        if endpoint_alignment:
            # A route endpoint is still one action.  Only evaluate its initial
            # tangent once, after Navfn has published the plan, so a later SLAM
            # update cannot restart a turn while the vehicle is moving.
            if self.pre_turn_checked_identity != action_identity:
                target_yaw = self._initial_navfn_yaw_in_pose_frame_locked()
                if target_yaw is None or self.pose is None:
                    return False
                heading_error = normalize_angle(target_yaw - self.pose.theta)
                # This is a route-topology decision, not a velocity threshold: the
                # base must first face a path that starts behind it.  Bends within
                # ``pre_route_turn_threshold`` remain under TEB's normal optimizer;
                # larger ones get an explicit pre-route rotation so the robot does
                # not drift toward a mapped wall while TEB re-orients.
                if abs(heading_error) <= self.pre_route_turn_threshold:
                    # This action does not need an immediate alignment. A later
                    # evidence-based stalled-route check may still recover it.
                    self.pre_turn_checked_identity = action_identity
                    target_yaw = None
                else:
                    turn_phase = "pre_route_alignment"
        if target_yaw is None and self.active_action_route_kind == TURN_ROUTE_KIND:
            # A semantic connector is explicitly a rotate-in-place phase.
            # Its orientation belongs to the route contract, so the
            # supervisor (rather than the ordinary local planner) owns this
            # one atomic action.
            target_yaw = self._goal_yaw_in_pose_frame_locked(self.active_action_goal)
            if target_yaw is None:
                return False
            turn_phase = "legacy_connector"
        if target_yaw is None and self._is_frontier_endpoint_action_locked():
            stalled_alignment = self._stalled_route_reorientation_locked(
                action_identity
            )
            if stalled_alignment is not None:
                target_yaw, heading_error = stalled_alignment
                turn_phase = "stalled_route_reorientation"
        if target_yaw is None:
            # A normal frontier endpoint must be passed straight through to
            # TEB when endpoint alignment is disabled.  Falling through to
            # the old connector branch here made every endpoint look like a
            # yaw-only action, repeatedly stopping and turning toward the
            # PoseStamped orientation even on a continuous forward path.
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
        # A connector's only purpose is an atomic yaw phase. Unlike a normal
        # endpoint-alignment candidate, it already passed the frontier route
        # transaction and must seize the command before TEB can translate
        # toward its short keep-alive waypoint.
        confirm_duration = (
            0.0
            if turn_phase == "legacy_connector"
            else self.turn_start_confirm_duration
        )
        if now - self.turn_pending_since_wall < confirm_duration:
            return False
        clearance_reason = self._turn_clearance_reason_locked(now)
        if clearance_reason:
            # No turn is safer than a forced circular-footprint rotation in a
            # known collision envelope. Mark this action as evaluated so a
            # 20 Hz callback cannot repeatedly seize/release the command;
            # TEB recovery and the frontier lifecycle retain ownership.
            if endpoint_alignment:
                self.pre_turn_checked_identity = action_identity
            self.completed_turn_key = key
            self.turn_pending_identity = None
            self.turn_pending_since_wall = 0.0
            self.publish_status_locked(
                "turn_skipped",
                reason=clearance_reason,
                scan_minimum=(
                    None if not math.isfinite(self.scan_minimum)
                    else round(float(self.scan_minimum), 4)
                ),
            )
            rospy.logwarn(
                "TEB turn supervisor skipped unsafe in-place turn: reason=%s "
                "scan_min=%s required=%.2fm",
                clearance_reason,
                "n/a" if not math.isfinite(self.scan_minimum)
                else "%.2f" % self.scan_minimum,
                self.turn_min_clearance,
            )
            return False
        # Do not mark a frontier action as pre-turn-checked until the
        # confirmation window has elapsed and the turn is actually committed.
        # Marking it before the early return above made every large initial
        # Navfn heading change a no-op: later callbacks saw the identity as
        # checked and TEB was left to oscillate against the nearby wall.
        if endpoint_alignment and turn_phase == "pre_route_alignment":
            self.pre_turn_checked_identity = action_identity
        if turn_phase == "stalled_route_reorientation":
            # Do this only once the confirmation window has elapsed and this
            # supervisor has actually seized the command.  Marking it in the
            # readiness check made every recovery turn a no-op.
            self.stalled_route_completed_identity = action_identity
            self.stalled_route_candidate_identity = None
            self.stalled_route_candidate_since_wall = 0.0
            self.stalled_route_ready_identity = None
        self.state = STATE_TURNING
        self.turn_key = key
        self.turn_target_yaw = target_yaw
        self.turn_goal_xy = self._goal_xy(self.active_action_goal)
        self.turn_velocity = 0.0
        self.turn_started_wall = now
        if heading_error is not None:
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
            turn_phase=turn_phase,
            plan_heading=round(float(target_yaw), 4),
        )
        rospy.loginfo(
            "TEB turn supervisor started %s: target=(%.2f,%.2f) yaw=%.1fdeg "
            "source=%s",
            turn_phase.replace("_", " "),
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
        target_track_id = ""
        try:
            payload = json.loads(message.data)
            if isinstance(payload, dict):
                source = str(payload.get("source", source)).strip().lower() or source
                priority = int(payload.get("priority", priority))
                route_kind = str(payload.get("route_kind", "")).strip().lower()
                target_track_id = str(
                    payload.get("target_track_id", "")
                ).strip()
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
            self.latest_target_track_id = target_track_id
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
            self.active_action_priority = max(
                0, min(3, int(payload.get("active_intent_priority", 0) or 0))
            )
            self.active_action_target_track_id = str(
                payload.get("active_target_track_id", "")
            ).strip()
            self.latest_target_track_id = str(
                payload.get("latest_target_track_id", self.latest_target_track_id)
            ).strip()
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
                self.trajectory_continuity_sharp_entry_identity = None
                self.stalled_route_candidate_identity = None
                self.stalled_route_candidate_since_wall = 0.0
                self.stalled_route_ready_identity = None
                self.stalled_route_completed_identity = None
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

    def on_scan(self, message):
        minimum = float("inf")
        for value in message.ranges:
            if math.isfinite(value) and value > 0.01:
                minimum = min(minimum, float(value))
        with self.lock:
            self.scan_minimum = minimum
            self.scan_monotonic = time.monotonic()

    def on_planner_command(self, message):
        with self.lock:
            self.latest_planner_command = copy.deepcopy(message)
            self.latest_planner_command_wall = time.monotonic()

    def on_teb_feedback(self, message):
        """Cache the selected TEB velocity for a bounded raw-command gap.

        This is deliberately not an alternate local controller: the cached
        velocity is copied from TEB's selected trajectory and expires quickly
        when feedback stops. The normal raw command remains authoritative for
        every non-zero command and every explicit TEB stop/turn.
        """
        selected_index = int(message.selected_trajectory_idx)
        trajectories = list(message.trajectories)
        if selected_index < 0 or selected_index >= len(trajectories):
            return
        points = list(trajectories[selected_index].trajectory)
        if not points:
            return
        # The first feedback point is the command TEB selected for the current
        # control cycle. Prefer it over a later look-ahead point so this node
        # cannot jump ahead in the trajectory.
        velocity = points[0].velocity
        command = Twist()
        command.linear.x = float(velocity.linear.x)
        command.angular.z = float(velocity.angular.z)
        with self.lock:
            now = time.monotonic()
            if self.latest_trajectory_command_wall > 0.0:
                period = now - self.latest_trajectory_command_wall
                # Ignore startup and post-pause gaps. They must not enlarge
                # the short continuity window for the next normal callback.
                if 0.005 <= period <= self.trajectory_feedback_timeout_cap:
                    if self.trajectory_feedback_period_ema is None:
                        self.trajectory_feedback_period_ema = period
                    else:
                        self.trajectory_feedback_period_ema = (
                            0.75 * self.trajectory_feedback_period_ema
                            + 0.25 * period
                        )
            self.latest_trajectory_command = command
            self.latest_trajectory_command_wall = now

    def _trajectory_feedback_timeout_locked(self):
        """Return bounded freshness based on the observed feedback cadence."""
        if self.trajectory_feedback_period_ema is None:
            return self.trajectory_feedback_timeout
        return min(
            self.trajectory_feedback_timeout_cap,
            max(
                self.trajectory_feedback_timeout,
                self.trajectory_feedback_period_scale
                * self.trajectory_feedback_period_ema,
            ),
        )

    def _trajectory_continuity_command_locked(self, now, planner_command_stale=False):
        """Return a TEB-selected forward command for one transient gap.

        A zero/reverse raw sample, or a brief gap in raw ``cmd_vel`` publishing,
        is respected unless the live action and fresh TEB feedback itself say
        that the selected trajectory still moves forward.  The latter happens
        when move_base momentarily misses a controller publication while TEB's
        feedback stream continues at the controller rate.  This adapter never
        invents a velocity: it forwards only TEB's selected command, for a
        bounded time, outside terminal and turn-only envelopes.
        """
        raw = self.latest_planner_command
        raw_linear_is_zero = abs(float(raw.linear.x)) <= 0.01
        raw_is_zero = raw_linear_is_zero and abs(float(raw.angular.z)) <= 0.01
        # Negative command with no requested rotation is not executable on the
        # forward-only base. It is safe to bridge only when TEB feedback has
        # already selected a forward continuation; a negative command with an
        # angular request remains a deliberate turn and passes through.
        raw_is_reverse_only = (
            float(raw.linear.x) < -0.01
            and abs(float(raw.angular.z)) <= 0.01
        )
        transient_gap = (
            raw_is_zero
            or raw_is_reverse_only
            or planner_command_stale
        )
        # A successor whose Navfn entry begins behind the base must turn using
        # its own fresh TEB band.  Forwarding selected feedback from the old
        # route during a raw-zero handoff produces a visible, brief turn in
        # the opposite direction before TEB corrects itself.  This leaves the
        # normal zero/reverse/publish-gap bridge intact for every non-sharp
        # route, but prevents it from crossing this geometric discontinuity.
        sharp_entry_error = (
            self._sharp_navfn_entry_heading_error_locked()
            if transient_gap else None
        )
        if sharp_entry_error is not None:
            action_identity = self.active_action_identity
            if action_identity != self.trajectory_continuity_sharp_entry_identity:
                self.trajectory_continuity_sharp_entry_identity = action_identity
                self.trajectory_continuity_sharp_entry_suppressions += 1
                self.publish_status_locked(
                    "trajectory_continuity_suppressed",
                    reason="sharp_navfn_entry",
                    entry_heading_error_deg=round(
                        math.degrees(sharp_entry_error), 2
                    ),
                    threshold_deg=90.0,
                    raw_command=[
                        round(float(raw.linear.x), 4),
                        round(float(raw.angular.z), 4),
                    ],
                    count=int(self.trajectory_continuity_sharp_entry_suppressions),
                )
            self.trajectory_zero_started_wall = 0.0
            return None
        goal_distance = self._goal_distance_in_pose_frame_locked(
            self.active_action_goal
        )
        feedback_age = max(0.0, now - self.latest_trajectory_command_wall)
        feedback_timeout = self._trajectory_feedback_timeout_locked()
        self.trajectory_continuity_goal_distance = goal_distance
        # The frontier node may prefetch the next same-priority exploration
        # endpoint while the bridge deliberately keeps the current action
        # active.  Requiring the newest intent coordinates to equal the
        # current action then turns a one-tick raw TEB zero into a visible
        # brake, despite TEB feedback selecting a valid forward trajectory.
        # The bridge action identity is authoritative.  Permit this bounded
        # continuity bridge only when the pending intent has the same source
        # and priority; a target acquisition or any priority escalation still
        # takes the normal immediate-stop path.
        action_or_same_tier_intent = (
            self._intent_matches_goal_locked(
                self.active_action_source_goal or self.active_action_goal
            )
            or (
                self.latest_intent_source == self.active_action_source
                and self.latest_intent_priority == self.active_action_priority
            )
        )
        if not (
            self.trajectory_continuity_enabled
            and self.state == STATE_PASS_THROUGH
            and self._is_continuity_eligible_action_locked()
            # A newer mission intent means the bridge is deliberately between
            # actions, even if its status callback has not yet cleared the
            # old action. Never carry velocity across that transaction edge.
            and action_or_same_tier_intent
            and not self.task_done
            and not self.navigation_hold
            and transient_gap
            and feedback_age <= feedback_timeout
            and float(self.latest_trajectory_command.linear.x)
            >= self.trajectory_continuity_min_forward
            and goal_distance is not None
            and goal_distance > self.trajectory_continuity_terminal_radius
        ):
            self.trajectory_zero_started_wall = 0.0
            return None
        if self.trajectory_zero_started_wall <= 0.0:
            self.trajectory_zero_started_wall = now
        if now - self.trajectory_zero_started_wall > self.trajectory_continuity_max_hold:
            return None
        self.trajectory_continuity_events += 1
        self.publish_status_locked(
            "trajectory_continuity",
            continuity_scope=(
                "target_same_track"
                if self._is_same_target_segment_continuation_locked()
                else "frontier_or_connector"
            ),
            gap_kind=(
                "planner_publish_gap"
                if planner_command_stale
                else ("reverse_only" if raw_is_reverse_only else "zero")
            ),
            raw_command=[
                round(float(raw.linear.x), 4),
                round(float(raw.angular.z), 4),
            ],
            selected_command=[
                round(float(self.latest_trajectory_command.linear.x), 4),
                round(float(self.latest_trajectory_command.angular.z), 4),
            ],
            feedback_age_seconds=round(feedback_age, 4),
            feedback_timeout_seconds=round(feedback_timeout, 4),
            feedback_period_seconds=(
                None
                if self.trajectory_feedback_period_ema is None
                else round(float(self.trajectory_feedback_period_ema), 4)
            ),
            action_goal_distance=round(float(goal_distance), 4),
            count=int(self.trajectory_continuity_events),
        )
        return copy.deepcopy(self.latest_trajectory_command)

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
                clearance_reason = self._turn_clearance_reason_locked(
                    time.monotonic()
                )
                if clearance_reason:
                    self._release_turn_locked(
                        "turn_clearance_lost_%s" % clearance_reason,
                        completed=False,
                    )
                    # Resume the planner stream on this same cycle. The mux
                    # remains the sole safety authority for any forward
                    # component, while this adapter no longer forces rotation.
                    command = copy.deepcopy(self.latest_planner_command)
                else:
                    command = self._turn_command_locked(
                        1.0 / self.command_frequency
                    )
                state = self.state
            elif (
                self.mode == self.active_mode
                and not self.task_done
                and not self.navigation_hold
            ):
                planner_command_stale = (
                    time.monotonic() - self.latest_planner_command_wall
                    > self.planner_command_timeout
                )
                if not planner_command_stale:
                    command = copy.deepcopy(self.latest_planner_command)
                    continuity = self._trajectory_continuity_command_locked(
                        time.monotonic()
                    )
                    if continuity is not None:
                        command = continuity
                else:
                    command = Twist()
                    # The raw planner publisher can briefly stall while its
                    # selected TEB trajectory and feedback remain current.
                    # Preserve that controller-owned command for the same
                    # strictly bounded continuity window used for raw zeros.
                    continuity = self._trajectory_continuity_command_locked(
                        time.monotonic(), planner_command_stale=True
                    )
                    if continuity is not None:
                        command = continuity
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
                self.trajectory_zero_started_wall = 0.0
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
