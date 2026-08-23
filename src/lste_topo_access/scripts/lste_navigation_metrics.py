#!/usr/bin/env python3
"""Write one structured navigation telemetry log for each live LSTE run.

The node intentionally observes the complete goal/controller/safety chain. It
does not publish commands or alter navigation decisions, so it can be left in
the normal pipeline while comparing detector, RL, and TEB behavior.
"""

import datetime
import json
import math
import os
import re
import shutil
import threading
import time
from pathlib import Path

import rospy
import tf
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from lste_msgs.msg import LsteDetections, LsteScores, LsteState
from move_base_msgs.msg import MoveBaseActionFeedback, MoveBaseActionGoal, RecoveryStatus
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from teb_local_planner.msg import FeedbackMsg
from tf.transformations import euler_from_quaternion


STATUS_NAMES = {
    0: "PENDING",
    1: "ACTIVE",
    2: "PREEMPTED",
    3: "SUCCEEDED",
    4: "ABORTED",
    5: "REJECTED",
    8: "PREEMPTING",
    9: "RECALLING",
}


class NavigationMetrics:
    def __init__(self):
        rospy.init_node("lste_navigation_metrics")
        self.lock = threading.RLock()
        self.process_name = "lste_navigation_metrics"
        self.log_root = self._resolve_path(
            rospy.get_param("~log_dir", "runtime/navigation/logs")
        )
        self.retention_days = max(
            1, int(rospy.get_param("~retention_days", 15))
        )
        self.run_timestamp, self.log_dir = self._create_run_dir()
        self.log_path = self.log_dir / (self.run_timestamp + "_navigation_metrics.log")
        self.stream = self.log_path.open("w", encoding="utf-8", buffering=1)

        self.start_wall = time.monotonic()
        self.start_ros = rospy.Time.now().to_sec()
        self.pose = None
        self.goal = None
        self.goal_frame = "odom"
        self.last_goal_frame = "odom"
        self.goal_message = None
        self.tf_listener = tf.TransformListener()
        self.distance_transform_failures = 0
        self.subgoal = None
        self.command = Twist()
        self.teb_command = Twist()
        self.teb_planner_command = Twist()
        self.teb_turn_supervisor_status = None
        self.teb_turn_supervisor_events = 0
        self.teb_turn_supervisor_last_event = "unknown"
        self.controller_mode = "unknown"
        self.controller_status = "not_available"
        self.controller_source = "unknown"
        self.controller_reason = "not_available"
        self.controller_requested = (float("nan"), float("nan"))
        self.controller_action = (float("nan"), float("nan"))
        self.controller_predicted_clearance = float("nan")
        self.controller_status_changes = 0
        self.bridge_events = 0
        self.bridge_deferred_goal_updates = 0
        self.bridge_dispatches = 0
        self.bridge_terminal_events = 0
        self.bridge_priority_handoffs = 0
        self.bridge_target_retries = 0
        self.bridge_target_segment_handoffs = 0
        self.bridge_frontier_segment_handoffs = 0
        self.bridge_frontier_sharp_replacements = 0
        self.bridge_goal_replacements = 0
        self.bridge_priority_goal_replacements = 0
        self.bridge_target_goal_replacements = 0
        self.bridge_active = False
        self.bridge_last_event = "unknown"
        self.bridge_active_intent_source = "unknown"
        self.bridge_latest_intent_source = "unknown"
        self.safety_override_events = 0
        # ``safety_override_events`` is a state-transition count kept for
        # backwards-compatible dashboards. These counters describe the actual
        # control stream more precisely: an intervention sample is one where
        # the guard source or requested/applied action differs materially.
        self.safety_intervention_samples = 0
        self.controller_action_delta = 0.0
        self.hard_stop_events = 0
        self.linear_brake_events = 0
        self.turn_only_events = 0
        self.turn_only_start_wall = None
        self.turn_only_duration_total = 0.0
        self.last_brake_wall = 0.0
        # Raw velocity changes have no inherent cause.  Retain recent action
        # lifecycle events so post-run analysis can separate a legitimate
        # terminal/turn boundary from an unexplained clear-path brake.
        self.lifecycle_event_wall = {}
        self.last_move_base_status = "UNKNOWN"
        self.brake_reason_counts = {}
        self.stop_reason_counts = {}
        self.last_status_text = ""
        self.last_status_signature = None
        self.teb_status = "not_available"
        self.teb_feedback_state = None
        self.move_base_feedback_state = None
        self.recovery_state = None
        self.global_costmap_stats = None
        self.local_costmap_stats = None
        self.navfn_plan_stats = None
        self.global_planner_plan_stats = None
        self.teb_global_plan_stats = None
        self.teb_local_plan_stats = None
        self.last_teb_feedback_log_wall = 0.0
        self.last_plan_log_wall = {}
        self.state = "unknown"
        self.goal_diagnostic = None
        self.task_done = False
        self.navigation_hold = False
        self.navigation_hold_events = 0
        self.navigation_hold_start_wall = None
        self.navigation_hold_duration_total = 0.0
        self.scan_minimum = float("nan")
        self.scan_forward_minimum = float("nan")
        self.scan_left_minimum = float("nan")
        self.scan_right_minimum = float("nan")
        # A forward laser arc alone is not a collision certificate for a
        # circular base that may be turning. Use the physical footprint plus
        # TEB's hard obstacle clearance when classifying a command brake.
        self.discontinuity_obstacle_clearance = max(
            0.05,
            float(rospy.get_param("/move_base/local_costmap/robot_radius", 0.30))
            + float(rospy.get_param("/move_base/TebLocalPlannerROS/min_obstacle_dist", 0.22))
            + 0.05,
        )
        self.target = None
        self.scores = None
        self.map_stats = None

        self.path_length = 0.0
        self.last_pose_xy = None
        self.goal_messages = 0
        self.goal_changes = 0
        self.goal_delta_sum = 0.0
        self.goal_delta_max = 0.0
        self.goal_last_change_ros = None
        self.goal_last_change_wall = None
        self.last_goal_xy = None
        self.dispatch_count = 0
        self.dispatch_last_xy = None
        self.status_seen = set()
        self.status_counts = {}
        self.preemptions = 0
        self.aborts = 0
        self.successes = 0
        self.cmd_messages = 0
        self.teb_cmd_messages = 0
        self.angular_sign_flips = 0
        self.last_nonzero_angular_sign = 0
        self.strong_angular_sign_flips = 0
        self.teb_angular_sign_flips = 0
        self.teb_last_nonzero_angular_sign = 0
        self.teb_strong_angular_sign_flips = 0
        self.teb_strong_angular_threshold = max(
            0.0, float(rospy.get_param("~teb_strong_angular_threshold", 0.12))
        )
        self.teb_linear_brake_events = 0
        self.stop_events = 0
        self.zero_start_wall = None
        self.zero_duration_total = 0.0
        self.stop_duration_count = 0
        self.max_stop_duration = 0.0
        self.last_stop_duration = None
        self.detector_messages = 0
        self.target_messages = 0
        self.target_first_seen_ros = None
        self.target_lock_ros = None
        self.task_done_ros = None
        self.goal_source = "unknown"
        self.last_goal_publish_wall = None
        self.target_goal_changes = 0
        self.target_route_accepts = 0
        self.target_route_rejections = 0
        self.target_route_deferrals = 0
        self.target_route_holds = 0
        self.target_route_failures = 0
        self.target_route_releases = 0
        self.target_approach_terminals = 0
        self.last_detection_stamp = None
        self.min_clearance = float("inf")
        self.sample_count = 0
        # Smoothness diagnostics.  ``forward_angular_energy`` accumulates how
        # much the robot steers while it is travelling at a meaningful forward
        # speed, normalised by forward distance so runs of different length are
        # comparable.  A straight-line wobble shows up as a high value per
        # metre even when the mean angular velocity is near zero.  Brake events
        # are split by whether a real obstacle occupied the forward lidar arc
        # at the moment of the brake: a brake with plenty of clearance is a
        # system-side jitter stop, not an obstacle avoidance response.
        self.forward_distance = 0.0
        self.forward_angular_energy = 0.0
        self.last_cmd_wall = None
        self.brake_events_clear = 0
        self.brake_events_near = 0
        self.brake_events_unknown_clearance = 0
        # Forward speed threshold (m/s) above which steering counts as
        # "straight-line" steering energy.  Derived from the TEB speed limit
        # so a slower comparison controller is not unfairly penalised.
        try:
            self.forward_speed_threshold = 0.30 * float(
                self._resolved_startup_params().get("teb_max_vel_x") or 0.50
            )
        except Exception:
            self.forward_speed_threshold = 0.15
        self.forward_speed_threshold = max(0.08, self.forward_speed_threshold)

        self._write(
            "INFO",
            "run_start",
            run_timestamp=self.run_timestamp,
            log_path=str(self.log_path),
            retention_days=self.retention_days,
            ros_time=self.start_ros,
            topics={
                "pose": "/pro3/wheel_odom",
                "goal": "/lste/final_goal",
                "dispatch": "/move_base_simple/goal",
                "cmd_vel": "/cmd_vel",
                "scan": "/pro3/rlscan",
                "status": "/move_base/status",
                "teb_feedback": "/move_base/TebLocalPlannerROS/teb_feedback",
                "teb_planner_cmd": "/lste/cmd_vel/teb_planner",
                "teb_turn_supervisor_status": "/lste/teb_turn_supervisor/status",
                "teb_bridge_status": "/lste/teb_goal_bridge/status",
                "teb_goal_failure": "/lste/teb_goal_failure",
                "goal_arbitration": "/lste/goal_arbitration",
                "navigation_hold": "/lste/navigation_hold",
                "global_costmap": "/move_base/global_costmap/costmap",
                "local_costmap": "/move_base/local_costmap/costmap",
            },
            resolved_params=self._resolved_startup_params(),
        )
        rospy.loginfo("Navigation metrics log: %s", self.log_path)

        rospy.Subscriber("/pro3/wheel_odom", Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber("/rbt_pose", Pose2D, self.on_pose2d, queue_size=1)
        rospy.Subscriber("/lste/final_goal", PoseStamped, self.on_goal, queue_size=1)
        rospy.Subscriber("/lste/goal_diagnostic", String, self.on_goal_diagnostic, queue_size=10)
        rospy.Subscriber("/lste/goal_arbitration", String, self.on_goal_arbitration, queue_size=10)
        # The TEB bridge uses the typed move_base action. Keep the legacy
        # simple-goal observer for older comparison launches, but count either
        # transport through the same dispatch recorder.
        rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, self.on_action_dispatch, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.on_dispatch, queue_size=1)
        rospy.Subscriber("/move_base/status", GoalStatusArray, self.on_status, queue_size=1)
        rospy.Subscriber("/cmd_vel", Twist, self.on_cmd, queue_size=1)
        rospy.Subscriber("/lste/cmd_vel/teb", Twist, self.on_teb_cmd, queue_size=1)
        rospy.Subscriber(
            "/lste/cmd_vel/teb_planner", Twist, self.on_teb_planner_cmd, queue_size=1
        )
        rospy.Subscriber("/pro3/rlscan", LaserScan, self.on_scan, queue_size=1)
        rospy.Subscriber("/lste/controller_mode", String, self.on_controller_mode, queue_size=1)
        rospy.Subscriber(
            "/lste/teb_goal_bridge/status", String, self.on_bridge_status, queue_size=10
        )
        rospy.Subscriber(
            "/lste/teb_turn_supervisor/status",
            String,
            self.on_turn_supervisor_status,
            queue_size=10,
        )
        rospy.Subscriber("/lste/sappo_controller_status", String, self.on_controller_status, queue_size=1)
        rospy.Subscriber(
            "/move_base/TebLocalPlannerROS/teb_feedback",
            FeedbackMsg,
            self.on_teb_feedback,
            queue_size=1,
        )
        rospy.Subscriber("/move_base/feedback", MoveBaseActionFeedback, self.on_move_base_feedback, queue_size=1)
        rospy.Subscriber("/move_base/recovery_status", RecoveryStatus, self.on_recovery, queue_size=1)
        rospy.Subscriber("/lste/state", LsteState, self.on_state, queue_size=1)
        rospy.Subscriber("/lste/detections", LsteDetections, self.on_detections, queue_size=1)
        rospy.Subscriber("/lste/scores", LsteScores, self.on_scores, queue_size=1)
        rospy.Subscriber("/lste/task_done", Bool, self.on_task_done, queue_size=1)
        rospy.Subscriber("/lste/navigation_hold", Bool, self.on_navigation_hold, queue_size=1)
        rospy.Subscriber("/map", OccupancyGrid, self.on_map, queue_size=1)
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.on_global_costmap, queue_size=1)
        rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.on_local_costmap, queue_size=1)
        rospy.Subscriber("/move_base/NavfnROS/plan", NavPath, self.on_navfn_plan, queue_size=1)
        rospy.Subscriber("/move_base/GlobalPlanner/plan", NavPath, self.on_global_planner_plan, queue_size=1)
        rospy.Subscriber("/move_base/TebLocalPlannerROS/global_plan", NavPath, self.on_teb_global_plan, queue_size=1)
        rospy.Subscriber("/move_base/TebLocalPlannerROS/local_plan", NavPath, self.on_teb_local_plan, queue_size=1)
        rospy.Timer(rospy.Duration(0.5), self.on_sample)
        rospy.on_shutdown(self.close)

    @staticmethod
    def _resolve_path(value):
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        return Path(os.environ.get("LSTE_WS", os.getcwd())) / path

    @staticmethod
    def _resolved_startup_params():
        """Read launch parameters after concurrently started nodes register them.

        ``run_nodes_tmux.sh`` starts metrics beside move_base.  A single early
        ``get_param(..., None)`` therefore recorded null TEB values even though
        the controller was configured correctly a moment later.  Poll the
        parameter server with wall time for a short bounded window; missing
        compatibility parameters still remain null.
        """
        names = {
            "controller_mode": "/lste_cmd_vel_mux/initial_mode",
            "teb_max_vel_x": "/move_base/TebLocalPlannerROS/max_vel_x",
            "teb_max_vel_x_backwards": "/move_base/TebLocalPlannerROS/max_vel_x_backwards",
            "teb_max_vel_theta": "/move_base/TebLocalPlannerROS/max_vel_theta",
            "teb_min_obstacle_dist": "/move_base/TebLocalPlannerROS/min_obstacle_dist",
            "teb_inflation_dist": "/move_base/TebLocalPlannerROS/inflation_dist",
            "teb_homotopy_class_planning": "/move_base/TebLocalPlannerROS/enable_homotopy_class_planning",
            "teb_homotopy_simple_exploration": "/move_base/TebLocalPlannerROS/simple_exploration",
            "teb_homotopy_max_number_classes": "/move_base/TebLocalPlannerROS/max_number_classes",
            "teb_homotopy_viapoints_all_candidates": "/move_base/TebLocalPlannerROS/viapoints_all_candidates",
            "teb_controller_frequency": "/move_base/controller_frequency",
            "teb_turn_supervisor_frequency": "/lste_teb_turn_supervisor/command_frequency",
            "teb_turn_supervisor_max_vel_theta": "/lste_teb_turn_supervisor/max_vel_theta",
            "teb_turn_supervisor_acc_lim_theta": "/lste_teb_turn_supervisor/acc_lim_theta",
            "teb_turn_supervisor_yaw_goal_tolerance": "/lste_teb_turn_supervisor/yaw_goal_tolerance",
            "teb_angular_switch_threshold": "/lste_cmd_vel_mux/teb_angular_sign_switch_threshold",
            "teb_angular_deadband": "/lste_cmd_vel_mux/teb_angular_deadband",
            "teb_target_early_handoff_distance": "/lste_teb_goal_bridge/target_early_handoff_distance",
            "teb_target_early_handoff_min_delta": "/lste_teb_goal_bridge/target_early_handoff_min_delta",
            "teb_in_place_replacement_max_delta": "/lste_teb_goal_bridge/in_place_replacement_max_delta",
            "teb_in_place_replacement_min_distance": "/lste_teb_goal_bridge/in_place_replacement_min_distance",
            "teb_in_place_replacement_max_distance": "/lste_teb_goal_bridge/in_place_replacement_max_distance",
            "teb_frontier_replacement_min_delta": "/lste_teb_goal_bridge/frontier_replacement_min_delta",
            "teb_allow_in_place_replacement": "/lste_teb_goal_bridge/allow_in_place_replacement",
            "teb_require_intent": "/lste_teb_goal_bridge/require_intent",
            "frontier_mission_endpoint_only": "/lste_global_frontier/mission_endpoint_only",
            "target_route_validation": "/lste_goal_manager/target_route_validation",
            "target_route_validation_service": "/lste_goal_manager/target_route_validation_service",
        }
        values = {key: None for key in names}
        pending = set(names)
        deadline = time.monotonic() + 3.0
        while pending and time.monotonic() < deadline:
            for key in tuple(pending):
                name = names[key]
                if rospy.has_param(name):
                    values[key] = rospy.get_param(name)
                    pending.remove(key)
            if pending:
                time.sleep(0.05)
        return values

    def _create_run_dir(self):
        self.log_root.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - self.retention_days * 86400.0
        for child in self.log_root.iterdir():
            try:
                if child.is_dir() and not child.is_symlink() and child.stat().st_mtime < cutoff:
                    shutil.rmtree(str(child))
            except OSError as exc:
                rospy.logwarn("Navigation metrics retention cleanup failed for %s: %s", child, exc)
        while True:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = self.log_root / timestamp
            try:
                directory.mkdir()
                return timestamp, directory
            except FileExistsError:
                time.sleep(1.0)

    @staticmethod
    def _finite_min(values):
        finite = [float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.01]
        return min(finite) if finite else float("nan")

    def _pose_xy_in_frame_locked(self, frame):
        """Return the latest odom pose expressed in ``frame``."""
        if self.pose is None:
            return None
        target_frame = (frame or "odom").strip().lstrip("/") or "odom"
        if target_frame == "odom":
            return self.pose
        stamped = PoseStamped()
        stamped.header.stamp = rospy.Time(0)
        stamped.header.frame_id = "odom"
        stamped.pose.position.x = float(self.pose[0])
        stamped.pose.position.y = float(self.pose[1])
        stamped.pose.orientation.z = math.sin(0.5 * float(self.pose[2]))
        stamped.pose.orientation.w = math.cos(0.5 * float(self.pose[2]))
        try:
            self.tf_listener.waitForTransform(
                target_frame,
                "odom",
                rospy.Time(0),
                rospy.Duration(0.02),
            )
            transformed = self.tf_listener.transformPose(target_frame, stamped)
        except (
            tf.Exception,
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as exc:
            self.distance_transform_failures += 1
            rospy.logwarn_throttle(
                3.0,
                "Navigation metrics cannot transform pose odom -> %s: %s",
                target_frame,
                exc,
            )
            return None
        yaw = euler_from_quaternion(
            [
                transformed.pose.orientation.x,
                transformed.pose.orientation.y,
                transformed.pose.orientation.z,
                transformed.pose.orientation.w,
            ]
        )[2]
        return (
            float(transformed.pose.position.x),
            float(transformed.pose.position.y),
            float(yaw),
        )

    def _write(self, level, event, **fields):
        # Earlier event records omitted simulated time, making diagnostic
        # traces appear at t=0 despite their wall-clock timestamps.  Preserve
        # an explicitly supplied value and stamp every other event here.
        fields.setdefault("ros_time", round(rospy.Time.now().to_sec(), 3))
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
        line = "%s level=%s process=%s event=%s data=%s\n" % (
            datetime.datetime.now().isoformat(timespec="milliseconds"),
            level,
            self.process_name,
            event,
            payload,
        )
        try:
            self.stream.write(line)
        except (AttributeError, ValueError):
            pass

    def on_odom(self, message):
        orientation = message.pose.pose.orientation
        yaw = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )[2]
        position = message.pose.pose.position
        with self.lock:
            self.pose = (position.x, position.y, yaw)
            xy = (position.x, position.y)
            if self.last_pose_xy is not None:
                self.path_length += math.hypot(xy[0] - self.last_pose_xy[0], xy[1] - self.last_pose_xy[1])
            self.last_pose_xy = xy

    def on_pose2d(self, message):
        with self.lock:
            if self.pose is None:
                self.pose = (message.x, message.y, message.theta)

    def on_goal(self, message):
        xy = (float(message.pose.position.x), float(message.pose.position.y))
        frame = (message.header.frame_id or "odom").strip().lstrip("/") or "odom"
        with self.lock:
            self.goal_messages += 1
            if self.last_goal_xy is not None:
                delta = (
                    math.hypot(xy[0] - self.last_goal_xy[0], xy[1] - self.last_goal_xy[1])
                    if frame == self.last_goal_frame
                    else float("nan")
                )
                if math.isfinite(delta):
                    self.goal_delta_sum += delta
                if math.isfinite(delta):
                    self.goal_delta_max = max(self.goal_delta_max, delta)
                if math.isfinite(delta) and delta > 0.03:
                    self.goal_changes += 1
                    self.goal_last_change_ros = rospy.Time.now().to_sec()
                    self.goal_last_change_wall = time.monotonic()
                    previous_goal = self.last_goal_xy
                    pose = self._pose_xy_in_frame_locked(frame)
                    previous_distance = (
                        float("nan") if pose is None else
                        math.hypot(previous_goal[0] - pose[0], previous_goal[1] - pose[1])
                    )
                    new_distance = (
                        float("nan") if pose is None else
                        math.hypot(xy[0] - pose[0], xy[1] - pose[1])
                    )
                    old_bearing = (
                        float("nan") if pose is None else
                        math.atan2(previous_goal[1] - pose[1], previous_goal[0] - pose[0])
                    )
                    new_bearing = (
                        float("nan") if pose is None else
                        math.atan2(xy[1] - pose[1], xy[0] - pose[0])
                    )
                    self._write(
                        "INFO",
                        "goal_change",
                        delta_m=round(delta, 4),
                        goal=[round(xy[0], 3), round(xy[1], 3)],
                        previous_goal=[round(previous_goal[0], 3), round(previous_goal[1], 3)],
                        pose=None if pose is None else [round(value, 3) for value in pose],
                        previous_distance_m=None if not math.isfinite(previous_distance) else round(previous_distance, 4),
                        new_distance_m=None if not math.isfinite(new_distance) else round(new_distance, 4),
                        old_bearing_rad=None if not math.isfinite(old_bearing) else round(old_bearing, 4),
                        new_bearing_rad=None if not math.isfinite(new_bearing) else round(new_bearing, 4),
                        controller_source=self.controller_source,
                        controller_reason=self.controller_reason,
                        goal_source=self.goal_source,
                        goal_hold_seconds=(
                            None if self.last_goal_publish_wall is None else
                            round(time.monotonic() - self.last_goal_publish_wall, 3)
                        ),
                        command=[round(float(self.command.linear.x), 4), round(float(self.command.angular.z), 4)],
                        scan_forward_min=None if not math.isfinite(self.scan_forward_minimum) else round(self.scan_forward_minimum, 4),
                        scan_min=None if not math.isfinite(self.scan_minimum) else round(self.scan_minimum, 4),
                        source_stamp=message.header.stamp.to_sec(),
                        pose_frame=frame,
                        goal_changes=self.goal_changes,
                    )
            self.last_goal_xy = xy
            self.goal = xy
            self.goal_frame = frame
            self.last_goal_frame = frame
            self.goal_message = message
            self.last_goal_publish_wall = time.monotonic()

    def on_teb_cmd(self, message):
        """Track planner output before mux/task completion changes it."""
        with self.lock:
            previous = self.teb_command
            self.teb_command = message
            self.teb_cmd_messages += 1
            angular = float(message.angular.z)
            previous_angular = float(previous.angular.z)
            current_linear = float(message.linear.x)
            previous_linear = float(previous.linear.x)
            sign = 1 if angular > 0.05 else -1 if angular < -0.05 else 0
            if sign and self.teb_last_nonzero_angular_sign and sign != self.teb_last_nonzero_angular_sign:
                self.teb_angular_sign_flips += 1
                if (
                    abs(angular) >= self.teb_strong_angular_threshold
                    and abs(previous_angular) >= self.teb_strong_angular_threshold
                ):
                    self.teb_strong_angular_sign_flips += 1
                self._write(
                    "WARN",
                    "teb_raw_angular_sign_flip",
                    angular=round(angular, 4),
                    previous_angular=round(previous_angular, 4),
                    count=self.teb_angular_sign_flips,
                    strong_count=self.teb_strong_angular_sign_flips,
                    goal=None if self.goal is None else [round(value, 3) for value in self.goal],
                    teb_status=self.teb_status,
                )
            if sign:
                self.teb_last_nonzero_angular_sign = sign
            if previous_linear > 0.05 and current_linear < previous_linear - 0.08:
                self.teb_linear_brake_events += 1
                self._write(
                    "WARN" if current_linear <= 0.01 else "INFO",
                    "teb_raw_linear_brake",
                    count=self.teb_linear_brake_events,
                    previous_linear=round(previous_linear, 4),
                    current_linear=round(current_linear, 4),
                    angular=round(angular, 4),
                    scan_forward_min=None if not math.isfinite(self.scan_forward_minimum) else round(self.scan_forward_minimum, 4),
                    goal=None if self.goal is None else [round(value, 3) for value in self.goal],
                )

    def on_teb_planner_cmd(self, message):
        """Record TEB's raw command separately from the supervisor output."""
        with self.lock:
            self.teb_planner_command = message

    def on_turn_supervisor_status(self, message):
        """Persist the explicit turn action state in the formal run log."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"event": "invalid", "raw": message.data}
        if not isinstance(payload, dict):
            payload = {"event": "invalid", "raw": message.data}
        with self.lock:
            event = str(payload.get("event", "unknown"))
            self.lifecycle_event_wall["turn_%s" % event] = time.monotonic()
            self.teb_turn_supervisor_status = payload
            if event != self.teb_turn_supervisor_last_event:
                self.teb_turn_supervisor_events += 1
                self.teb_turn_supervisor_last_event = event
                turn_event = payload.pop("event", event)
                self._write(
                    "INFO" if event not in ("turn_released",) else "WARN",
                    "teb_turn_supervisor_event",
                    turn_event=turn_event,
                    **payload,
                )

    def _record_dispatch(self, xy, transport):
        with self.lock:
            self.dispatch_count += 1
            delta = float("nan")
            if self.dispatch_last_xy is not None:
                delta = math.hypot(xy[0] - self.dispatch_last_xy[0], xy[1] - self.dispatch_last_xy[1])
            self.dispatch_last_xy = xy
            self.subgoal = xy
            self._write(
                "INFO",
                "move_base_dispatch",
                count=self.dispatch_count,
                transport=transport,
                delta_m=None if not math.isfinite(delta) else round(delta, 4),
                goal=[round(xy[0], 3), round(xy[1], 3)],
            )

    def on_action_dispatch(self, message):
        target = message.goal.target_pose
        self._record_dispatch(
            (float(target.pose.position.x), float(target.pose.position.y)),
            "move_base_action",
        )

    def on_dispatch(self, message):
        self._record_dispatch(
            (float(message.pose.position.x), float(message.pose.position.y)),
            "move_base_simple_goal",
        )

    def on_status(self, message):
        with self.lock:
            for status in message.status_list:
                key = (status.goal_id.id, int(status.status))
                if key in self.status_seen:
                    continue
                self.status_seen.add(key)
                code = int(status.status)
                name = STATUS_NAMES.get(code, "STATUS_%d" % code)
                self.last_move_base_status = name
                self.lifecycle_event_wall["move_base_%s" % name.lower()] = time.monotonic()
                self.status_counts[name] = self.status_counts.get(name, 0) + 1
                if code == 2:
                    self.preemptions += 1
                elif code == 3:
                    self.successes += 1
                elif code in (4, 5, 8, 9):
                    self.aborts += 1
                self._write(
                    "INFO" if code in (0, 1, 3) else "WARN",
                    "move_base_status",
                    goal_id=status.goal_id.id,
                    status=code,
                    status_name=name,
                    text=status.text or "-",
                    preemptions=self.preemptions,
                    aborts=self.aborts,
                )

    @staticmethod
    def _recent_lifecycle_event(events, names, now, window=0.8):
        """Return the newest named lifecycle event inside ``window`` seconds."""
        newest_name = None
        newest_age = None
        for name in names:
            stamp = events.get(name)
            if stamp is None:
                continue
            age = max(0.0, now - stamp)
            if age <= window and (newest_age is None or age < newest_age):
                newest_name = name
                newest_age = age
        return newest_name, newest_age

    def _command_discontinuity_reason_locked(self, now):
        """Classify a command gap without influencing navigation control.

        This is an observability boundary: its purpose is to prove whether a
        visible stop belongs to a real topology/action transition, a deliberate
        in-place turn, an obstacle response, or an unexplained clear-space
        interruption before changing the execution architecture.
        """
        if self.task_done:
            return "task_complete", None
        event, age = self._recent_lifecycle_event(
            self.lifecycle_event_wall,
            ("turn_turn_started", "turn_turning", "turn_turn_completed"),
            now,
        )
        if event is not None:
            return "explicit_turn", age
        event, age = self._recent_lifecycle_event(
            self.lifecycle_event_wall,
            ("terminal", "move_base_succeeded"),
            now,
        )
        if event is not None:
            return "action_terminal", age
        event, age = self._recent_lifecycle_event(
            self.lifecycle_event_wall,
            ("cancel", "frontier_route_invalidated", "handoff_requested"),
            now,
        )
        if event is not None:
            return "route_recovery", age
        event, age = self._recent_lifecycle_event(
            self.lifecycle_event_wall,
            ("dispatch",),
            now,
            window=0.45,
        )
        if event is not None:
            return "action_dispatch", age
        if (
            self.goal_last_change_wall is not None
            and now - self.goal_last_change_wall <= 0.8
        ):
            return "goal_transition", now - self.goal_last_change_wall
        if (
            math.isfinite(self.scan_minimum)
            and self.scan_minimum <= self.discontinuity_obstacle_clearance
        ):
            return "near_obstacle", None
        if math.isfinite(self.scan_forward_minimum):
            return "unexplained_clear_path", None
        return "unknown_clearance", None

    @staticmethod
    def _increment_reason(counter, reason):
        counter[reason] = int(counter.get(reason, 0)) + 1

    def on_cmd(self, message):
        with self.lock:
            previous = self.command
            self.command = message
            self.cmd_messages += 1
            angular = float(message.angular.z)
            previous_angular = float(previous.angular.z)
            current_linear = float(message.linear.x)
            previous_linear = float(previous.linear.x)
            # Forward steering energy: integrate |angular| only while the robot
            # is actually travelling forward, so pure in-place turns at walls
            # do not pollute the straight-line wobble measurement.
            now_wall = time.monotonic()
            if self.last_cmd_wall is not None:
                dt = max(0.0, now_wall - self.last_cmd_wall)
                if current_linear > self.forward_speed_threshold:
                    self.forward_distance += current_linear * dt
                    self.forward_angular_energy += abs(angular) * dt
            self.last_cmd_wall = now_wall
            sign = 1 if angular > 0.05 else -1 if angular < -0.05 else 0
            previous_sign = 1 if previous_angular > 0.05 else -1 if previous_angular < -0.05 else 0
            if sign and self.last_nonzero_angular_sign and sign != self.last_nonzero_angular_sign:
                self.angular_sign_flips += 1
                if (
                    abs(angular) >= self.teb_strong_angular_threshold
                    and abs(previous_angular) >= self.teb_strong_angular_threshold
                ):
                    self.strong_angular_sign_flips += 1
                self._write(
                    "WARN",
                    "angular_sign_flip",
                    angular=round(angular, 4),
                    previous_angular=round(previous_angular, 4),
                    count=self.angular_sign_flips,
                    strong_count=self.strong_angular_sign_flips,
                )
            if sign:
                self.last_nonzero_angular_sign = sign
            if previous_linear > 0.05 and current_linear < previous_linear - 0.08:
                self.linear_brake_events += 1
                forward_clear = self.scan_forward_minimum
                if not math.isfinite(forward_clear):
                    self.brake_events_unknown_clearance += 1
                elif forward_clear >= 0.60:
                    self.brake_events_clear += 1
                else:
                    self.brake_events_near += 1
                now = time.monotonic()
                brake_reason, lifecycle_age = self._command_discontinuity_reason_locked(now)
                self._increment_reason(self.brake_reason_counts, brake_reason)
                if now - self.last_brake_wall >= 0.15:
                    self.last_brake_wall = now
                    self._write(
                        "WARN" if current_linear <= 0.01 else "INFO",
                        "linear_brake",
                        count=self.linear_brake_events,
                        previous_linear=round(previous_linear, 4),
                        current_linear=round(current_linear, 4),
                        angular=round(angular, 4),
                        delta=round(current_linear - previous_linear, 4),
                        controller_source=self.controller_source,
                        controller_reason=self.controller_reason,
                        scan_forward_min=None if not math.isfinite(self.scan_forward_minimum) else round(self.scan_forward_minimum, 4),
                        scan_min=None if not math.isfinite(self.scan_minimum) else round(self.scan_minimum, 4),
                        obstacle_clearance_threshold=round(
                            self.discontinuity_obstacle_clearance, 4
                        ),
                        pose=None if self.pose is None else [round(value, 3) for value in self.pose],
                        goal=None if self.goal is None else [round(value, 3) for value in self.goal],
                        reason=brake_reason,
                        lifecycle_age_seconds=(
                            None if lifecycle_age is None else round(lifecycle_age, 4)
                        ),
                        move_base_status=self.last_move_base_status,
                        bridge_event=self.bridge_last_event,
                    )
            was_turn_only = abs(previous_linear) <= 0.01 and abs(previous_angular) > 0.05
            is_turn_only = abs(current_linear) <= 0.01 and abs(angular) > 0.05
            if is_turn_only and not was_turn_only:
                self.turn_only_events += 1
                self.turn_only_start_wall = time.monotonic()
                self._write(
                    "INFO",
                    "turn_only_start",
                    count=self.turn_only_events,
                    angular=round(angular, 4),
                    controller_source=self.controller_source,
                    controller_reason=self.controller_reason,
                    scan_forward_min=None if not math.isfinite(self.scan_forward_minimum) else round(self.scan_forward_minimum, 4),
                    scan_min=None if not math.isfinite(self.scan_minimum) else round(self.scan_minimum, 4),
                )
            elif not is_turn_only and was_turn_only and self.turn_only_start_wall is not None:
                duration = time.monotonic() - self.turn_only_start_wall
                self.turn_only_duration_total += duration
                self._write(
                    "INFO",
                    "turn_only_end",
                    duration_seconds=round(duration, 3),
                    total_duration_seconds=round(self.turn_only_duration_total, 3),
                )
                self.turn_only_start_wall = None
            was_moving = abs(float(previous.linear.x)) > 0.05 or abs(previous_angular) > 0.05
            is_zero = abs(float(message.linear.x)) <= 0.01 and abs(angular) <= 0.01
            if was_moving and is_zero:
                self.stop_events += 1
                self.zero_start_wall = time.monotonic()
                stop_reason, lifecycle_age = self._command_discontinuity_reason_locked(
                    self.zero_start_wall
                )
                self._increment_reason(self.stop_reason_counts, stop_reason)
                self._write(
                    "WARN",
                    "command_stop",
                    count=self.stop_events,
                    controller_mode=self.controller_mode,
                    controller_source=self.controller_source,
                    controller_reason=self.controller_reason,
                    teb_status=self.teb_status,
                    bridge_event=self.bridge_last_event,
                    reason=stop_reason,
                    lifecycle_age_seconds=(
                        None if lifecycle_age is None else round(lifecycle_age, 4)
                    ),
                    move_base_status=(
                        None
                        if self.move_base_feedback_state is None
                        else self.move_base_feedback_state.get("status_name")
                    ),
                )
            elif not is_zero and self.zero_start_wall is not None:
                duration = time.monotonic() - self.zero_start_wall
                self.zero_duration_total += duration
                self.stop_duration_count += 1
                self.last_stop_duration = duration
                self.max_stop_duration = max(self.max_stop_duration, duration)
                self._write(
                    "INFO",
                    "stop_end",
                    duration_seconds=round(duration, 3),
                    total_duration_seconds=round(self.zero_duration_total, 3),
                    max_duration_seconds=round(self.max_stop_duration, 3),
                )
                self.zero_start_wall = None

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
        with self.lock:
            event = str(payload.get("event", "unknown"))
            self.lifecycle_event_wall[event] = time.monotonic()
            self.bridge_events += 1
            self.bridge_last_event = event
            self.bridge_active = bool(payload.get("active", False))
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
            elif event == "priority_handoff_requested":
                self.bridge_priority_handoffs += 1
            elif event == "target_retry_requested":
                self.bridge_target_retries += 1
            elif event == "target_route_failed":
                self.target_route_failures += 1
            elif event == "target_segment_handoff_requested":
                self.bridge_target_segment_handoffs += 1
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
                elif replacement_kind == "target_segment":
                    self.bridge_target_goal_replacements += 1
                    if event == "dispatch":
                        self.bridge_target_segment_handoffs += 1
                elif replacement_kind == "frontier_segment":
                    self.bridge_frontier_segment_handoffs += 1
                elif replacement_kind == "frontier_sharp_branch":
                    self.bridge_frontier_sharp_replacements += 1
            # ``_write`` already has an ``event`` positional argument; keep
            # the bridge's event name as data instead of passing it twice.
            bridge_event = payload.pop("event", event)
            self._write(
                "INFO",
                "teb_bridge_event",
                bridge_event=bridge_event,
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

    @staticmethod
    def _grid_stats(message):
        values = list(message.data)
        return {
            "width": int(message.info.width),
            "height": int(message.info.height),
            "resolution": float(message.info.resolution),
            "free": values.count(0),
            "occupied": sum(1 for value in values if value >= 50),
            "unknown": values.count(-1),
        }

    @staticmethod
    def _path_stats(message):
        poses = message.poses
        length = 0.0
        for previous, current in zip(poses, poses[1:]):
            length += math.hypot(
                current.pose.position.x - previous.pose.position.x,
                current.pose.position.y - previous.pose.position.y,
            )
        endpoint = None
        if poses:
            endpoint = [
                round(float(poses[-1].pose.position.x), 3),
                round(float(poses[-1].pose.position.y), 3),
            ]
        return {"poses": len(poses), "length": round(length, 3), "endpoint": endpoint}

    def on_teb_feedback(self, message):
        with self.lock:
            trajectories = list(message.trajectories)
            selected_index = int(message.selected_trajectory_idx)
            selected = None
            if 0 <= selected_index < len(trajectories):
                selected = trajectories[selected_index]
            first = selected.trajectory[0] if selected is not None and selected.trajectory else None
            selected_velocity = None if first is None else {
                "linear_x": round(float(first.velocity.linear.x), 4),
                "angular_z": round(float(first.velocity.angular.z), 4),
            }
            obstacle_count = len(message.obstacles_msg.obstacles)
            self.teb_feedback_state = {
                "trajectories": len(trajectories),
                "selected_index": selected_index,
                "selected_points": 0 if selected is None else len(selected.trajectory),
                "selected_velocity": selected_velocity,
                "obstacles": obstacle_count,
            }
            if selected is None:
                self.teb_status = "no_selected_trajectory"
            elif first is None:
                self.teb_status = "selected_trajectory_empty"
            elif abs(float(first.velocity.linear.x)) <= 0.002 and abs(float(first.velocity.angular.z)) <= 0.01:
                self.teb_status = "selected_command_near_zero"
            else:
                self.teb_status = "trajectory_valid"
            now = time.monotonic()
            if selected is None or now - self.last_teb_feedback_log_wall >= 1.0:
                self.last_teb_feedback_log_wall = now
                self._write(
                    "WARN" if selected is None else "INFO",
                    "teb_feedback",
                    status=self.teb_status,
                    **self.teb_feedback_state,
                )

    def on_move_base_feedback(self, message):
        with self.lock:
            status = message.status
            self.move_base_feedback_state = {
                "goal_id": status.goal_id.id,
                "status": int(status.status),
                "status_name": STATUS_NAMES.get(int(status.status), "STATUS_%d" % int(status.status)),
                "base": [
                    round(float(message.feedback.base_position.pose.position.x), 3),
                    round(float(message.feedback.base_position.pose.position.y), 3),
                ],
                "frame": message.feedback.base_position.header.frame_id or "odom",
            }

    def on_recovery(self, message):
        with self.lock:
            current = {
                "current": int(message.current_recovery_number),
                "total": int(message.total_number_of_recoveries),
                "behavior": message.recovery_behavior_name,
            }
            if current != self.recovery_state:
                self.recovery_state = current
                self._write("WARN", "move_base_recovery", **current)

    def _on_path(self, source, message):
        with self.lock:
            stats = self._path_stats(message)
            setattr(self, source, stats)
            now = time.monotonic()
            previous = self.last_plan_log_wall.get(source, 0.0)
            if now - previous >= 1.0:
                self.last_plan_log_wall[source] = now
                self._write("INFO", "planner_path", planner=source, **stats)

    def on_navfn_plan(self, message):
        self._on_path("navfn_plan_stats", message)

    def on_global_planner_plan(self, message):
        self._on_path("global_planner_plan_stats", message)

    def on_teb_global_plan(self, message):
        self._on_path("teb_global_plan_stats", message)

    def on_teb_local_plan(self, message):
        self._on_path("teb_local_plan_stats", message)

    def on_state(self, message):
        with self.lock:
            new_state = "%d:%s" % (int(message.state), message.subtype or "-")
            if new_state != self.state:
                previous = self.state
                self.state = new_state
                self._write(
                    "INFO",
                    "state_change",
                    previous=previous,
                    current=new_state,
                    state=int(message.state),
                    subtype=message.subtype or "",
                    goal=None if self.goal is None else [round(value, 3) for value in self.goal],
                    pose=None if self.pose is None else [round(value, 3) for value in self.pose],
                    scores=self.scores,
                )
                if int(message.state) == 2 and self.target_lock_ros is None:
                    self.target_lock_ros = rospy.Time.now().to_sec()
                    self._write(
                        "INFO",
                        "target_lock",
                        latency_seconds=round(self.target_lock_ros - self.start_ros, 3),
                        state=new_state,
                    )

    def on_goal_diagnostic(self, message):
        with self.lock:
            try:
                diagnostic = json.loads(message.data)
            except (TypeError, ValueError):
                diagnostic = {"raw": message.data}
            self.goal_diagnostic = diagnostic
            self.goal_source = str(diagnostic.get("source", self.goal_source))
            if self.goal_source.startswith("target_"):
                self.target_goal_changes += 1
            self._write("INFO", "goal_diagnostic", **diagnostic)

    def on_goal_arbitration(self, message):
        """Record mission/execution ownership decisions as first-class events."""
        with self.lock:
            try:
                payload = json.loads(message.data)
            except (TypeError, ValueError):
                payload = {"raw": message.data}
            if not isinstance(payload, dict):
                payload = {"raw": message.data}
            event = str(payload.get("event", "unknown"))
            if event == "target_route_accepted":
                self.target_route_accepts += 1
            elif event == "target_route_rejected":
                self.target_route_rejections += 1
            elif event == "target_route_deferred":
                self.target_route_deferrals += 1
            elif event == "target_route_held":
                self.target_route_holds += 1
            elif event == "target_route_released":
                self.target_route_releases += 1
            elif event == "target_approach_terminal":
                self.target_approach_terminals += 1
            record = dict(payload)
            record.pop("event", None)
            self._write("INFO" if event != "target_route_rejected" else "WARN",
                        "goal_arbitration", goal_event=event, **record)

    def on_detections(self, message):
        target = None
        if message.target_dets:
            target = max(message.target_dets, key=lambda item: float(item.score))
        with self.lock:
            self.detector_messages += 1
            if target is not None:
                self.target_messages += 1
                stamp = message.header.stamp.to_sec()
                if stamp != self.last_detection_stamp:
                    self.last_detection_stamp = stamp
                    if self.target_first_seen_ros is None:
                        self.target_first_seen_ros = rospy.Time.now().to_sec()
                        self._write(
                            "INFO",
                            "target_acquired",
                            latency_seconds=round(self.target_first_seen_ros - self.start_ros, 3),
                            score=round(float(target.score), 4),
                            center=[round(float(target.cx), 4), round(float(target.cy), 4)],
                        )
                    self._write(
                        "INFO",
                        "target_observation",
                        score=round(float(target.score), 4),
                        center=[round(float(target.cx), 4), round(float(target.cy), 4)],
                        box=[round(float(target.w), 4), round(float(target.h), 4)],
                        detector_stamp=stamp,
                    )
                self.target = {
                    "label": target.label,
                    "score": float(target.score),
                    "cx": float(target.cx),
                    "cy": float(target.cy),
                    "w": float(target.w),
                    "h": float(target.h),
                }
            else:
                self.target = None

    def on_scores(self, message):
        with self.lock:
            self.scores = {
                "total": float(message.s_total),
                "target": float(message.s_target),
                "env": float(message.s_env),
                "ctx": float(message.s_ctx),
                "detected": bool(message.detected),
            }

    def on_task_done(self, message):
        done = bool(message.data)
        with self.lock:
            if done != self.task_done:
                self.task_done = done
                if done:
                    self.task_done_ros = rospy.Time.now().to_sec()
                    self._write(
                        "INFO",
                        "task_completed",
                        latency_seconds=round(self.task_done_ros - self.start_ros, 3),
                        target_acquired_latency=(
                            None if self.target_first_seen_ros is None else
                            round(self.target_first_seen_ros - self.start_ros, 3)
                        ),
                        target_lock_latency=(
                            None if self.target_lock_ros is None else
                            round(self.target_lock_ros - self.start_ros, 3)
                        ),
                    )
                self._write("INFO", "task_done", value=done)

    def on_navigation_hold(self, message):
        with self.lock:
            active = bool(message.data)
            if active == self.navigation_hold:
                return
            now = time.monotonic()
            if active:
                self.navigation_hold_events += 1
                self.navigation_hold_start_wall = now
                self.navigation_hold = True
                self._write(
                    "INFO",
                    "navigation_hold_start",
                    count=self.navigation_hold_events,
                    goal=None if self.goal is None else [round(value, 3) for value in self.goal],
                    pose=None if self.pose is None else [round(value, 3) for value in self.pose],
                )
            else:
                duration = (
                    0.0 if self.navigation_hold_start_wall is None
                    else max(0.0, now - self.navigation_hold_start_wall)
                )
                self.navigation_hold_duration_total += duration
                self.navigation_hold_start_wall = None
                self.navigation_hold = False
                self._write(
                    "INFO",
                    "navigation_hold_end",
                    duration_seconds=round(duration, 3),
                    total_duration_seconds=round(self.navigation_hold_duration_total, 3),
                )

    def on_map(self, message):
        with self.lock:
            self.map_stats = self._grid_stats(message)

    def on_global_costmap(self, message):
        with self.lock:
            self.global_costmap_stats = self._grid_stats(message)

    def on_local_costmap(self, message):
        with self.lock:
            self.local_costmap_stats = self._grid_stats(message)

    def _snapshot(self):
        pose = self.pose
        goal = self.goal
        distance = float("nan")
        pose_for_goal = self._pose_xy_in_frame_locked(self.goal_frame)
        if pose_for_goal is not None and goal is not None:
            distance = math.hypot(goal[0] - pose_for_goal[0], goal[1] - pose_for_goal[1])
        average_goal_delta = (
            self.goal_delta_sum / max(1, self.goal_messages - 1)
        )
        elapsed_wall = max(0.001, time.monotonic() - self.start_wall)
        stop_rate_per_minute = self.stop_events * 60.0 / elapsed_wall
        goal_change_rate_per_minute = self.goal_changes * 60.0 / elapsed_wall
        brake_rate_per_minute = self.linear_brake_events * 60.0 / elapsed_wall
        average_stop_duration = (
            self.zero_duration_total / self.stop_duration_count
            if self.stop_duration_count else 0.0
        )
        return {
            "ros_time": round(rospy.Time.now().to_sec(), 3),
            "pose": None if pose is None else [round(float(value), 4) for value in pose],
            "goal": None if goal is None else [round(float(value), 4) for value in goal],
            "distance_to_goal": None if not math.isfinite(distance) else round(distance, 4),
            "pose_frame": "odom",
            "goal_frame": self.goal_frame,
            "distance_transform_failures": self.distance_transform_failures,
            "path_length": round(self.path_length, 4),
            "cmd": [round(float(self.command.linear.x), 4), round(float(self.command.angular.z), 4)],
            "teb_cmd": [round(float(self.teb_command.linear.x), 4), round(float(self.teb_command.angular.z), 4)],
            "teb_planner_cmd": [
                round(float(self.teb_planner_command.linear.x), 4),
                round(float(self.teb_planner_command.angular.z), 4),
            ],
            "teb_turn_supervisor": self.teb_turn_supervisor_status,
            "teb_turn_supervisor_events": self.teb_turn_supervisor_events,
            "teb_turn_supervisor_last_event": self.teb_turn_supervisor_last_event,
            "scan_min": None if not math.isfinite(self.scan_minimum) else round(self.scan_minimum, 4),
            "scan_forward_min": None if not math.isfinite(self.scan_forward_minimum) else round(self.scan_forward_minimum, 4),
            "scan_left_min": None if not math.isfinite(self.scan_left_minimum) else round(self.scan_left_minimum, 4),
            "scan_right_min": None if not math.isfinite(self.scan_right_minimum) else round(self.scan_right_minimum, 4),
            "controller_mode": self.controller_mode,
            "controller_status": self.controller_status,
            "controller_source": self.controller_source,
            "controller_reason": self.controller_reason,
            "controller_requested": [round(value, 4) if math.isfinite(value) else None for value in self.controller_requested],
            "controller_action": [round(value, 4) if math.isfinite(value) else None for value in self.controller_action],
            "controller_predicted_clearance": None if not math.isfinite(self.controller_predicted_clearance) else round(self.controller_predicted_clearance, 4),
            "teb_status": self.teb_status if self.controller_mode == "teb" else "not_applicable",
            "teb_feedback": self.teb_feedback_state if self.controller_mode == "teb" else None,
            "move_base_feedback": self.move_base_feedback_state,
            "recovery": self.recovery_state,
            "global_costmap": self.global_costmap_stats,
            "local_costmap": self.local_costmap_stats,
            "navfn_plan": self.navfn_plan_stats,
            "global_planner_plan": self.global_planner_plan_stats,
            "teb_global_plan": self.teb_global_plan_stats,
            "teb_local_plan": self.teb_local_plan_stats,
            "state": self.state,
            "goal_source": self.goal_source,
            "goal_diagnostic": self.goal_diagnostic,
            "task_done": self.task_done,
            "navigation_hold": self.navigation_hold,
            "navigation_hold_events": self.navigation_hold_events,
            "navigation_hold_duration_seconds": round(
                self.navigation_hold_duration_total
                + (
                    0.0 if self.navigation_hold_start_wall is None
                    else max(0.0, time.monotonic() - self.navigation_hold_start_wall)
                ),
                3,
            ),
            "goal_messages": self.goal_messages,
            "goal_changes": self.goal_changes,
            "goal_delta_mean": round(average_goal_delta, 4),
            "goal_delta_max": round(self.goal_delta_max, 4),
            "move_base_dispatches": self.dispatch_count,
            "move_base_preemptions": self.preemptions,
            "move_base_aborts": self.aborts,
            "move_base_successes": self.successes,
            "angular_sign_flips": self.angular_sign_flips,
            "strong_angular_sign_flips": self.strong_angular_sign_flips,
            "teb_angular_sign_flips": self.teb_angular_sign_flips,
            "teb_strong_angular_sign_flips": self.teb_strong_angular_sign_flips,
            "teb_linear_brake_events": self.teb_linear_brake_events,
            "forward_distance_m": round(self.forward_distance, 3),
            "forward_angular_energy_rad": round(self.forward_angular_energy, 4),
            "forward_angular_energy_per_m": round(
                self.forward_angular_energy / max(0.01, self.forward_distance), 4
            ),
            "forward_speed_threshold": round(self.forward_speed_threshold, 4),
            "brake_events_clear": self.brake_events_clear,
            "brake_events_near": self.brake_events_near,
            "brake_events_unknown_clearance": self.brake_events_unknown_clearance,
            "stop_events": self.stop_events,
            "stop_rate_per_minute": round(stop_rate_per_minute, 3),
            "stop_duration_count": self.stop_duration_count,
            "average_stop_duration_seconds": round(average_stop_duration, 3),
            "max_stop_duration_seconds": round(self.max_stop_duration, 3),
            "last_stop_duration_seconds": (
                None if self.last_stop_duration is None
                else round(self.last_stop_duration, 3)
            ),
            "linear_brake_events": self.linear_brake_events,
            "linear_brake_rate_per_minute": round(brake_rate_per_minute, 3),
            "brake_reason_counts": dict(self.brake_reason_counts),
            "stop_reason_counts": dict(self.stop_reason_counts),
            "goal_change_rate_per_minute": round(goal_change_rate_per_minute, 3),
            "turn_only_events": self.turn_only_events,
            "turn_only_duration_seconds": round(self.turn_only_duration_total, 3),
            "safety_override_events": self.safety_override_events,
            "safety_intervention_samples": self.safety_intervention_samples,
            "hard_stop_events": self.hard_stop_events,
            "controller_action_delta": round(self.controller_action_delta, 4),
            "controller_status_changes": self.controller_status_changes,
            "teb_bridge_events": self.bridge_events,
            "teb_bridge_deferred_goal_updates": self.bridge_deferred_goal_updates,
            "teb_bridge_dispatches": self.bridge_dispatches,
            "teb_bridge_terminal_events": self.bridge_terminal_events,
            "teb_bridge_priority_handoffs": self.bridge_priority_handoffs,
            "teb_bridge_target_retries": self.bridge_target_retries,
            "teb_bridge_target_segment_handoffs": self.bridge_target_segment_handoffs,
            "teb_bridge_frontier_segment_handoffs": self.bridge_frontier_segment_handoffs,
            "teb_bridge_frontier_sharp_replacements": self.bridge_frontier_sharp_replacements,
            "teb_bridge_goal_replacements": self.bridge_goal_replacements,
            "teb_bridge_priority_goal_replacements": self.bridge_priority_goal_replacements,
            "teb_bridge_target_goal_replacements": self.bridge_target_goal_replacements,
            "teb_bridge_active": self.bridge_active,
            "teb_bridge_last_event": self.bridge_last_event,
            "teb_bridge_active_intent_source": self.bridge_active_intent_source,
            "teb_bridge_latest_intent_source": self.bridge_latest_intent_source,
            "zero_duration_seconds": round(self.zero_duration_total, 3),
            "detector_messages": self.detector_messages,
            "target_messages": self.target_messages,
            "target_first_seen_seconds": None if self.target_first_seen_ros is None else round(self.target_first_seen_ros - self.start_ros, 3),
            "target_lock_seconds": None if self.target_lock_ros is None else round(self.target_lock_ros - self.start_ros, 3),
            "task_done_seconds": None if self.task_done_ros is None else round(self.task_done_ros - self.start_ros, 3),
            "target_goal_changes": self.target_goal_changes,
            "target_route_accepts": self.target_route_accepts,
            "target_route_rejections": self.target_route_rejections,
            "target_route_deferrals": self.target_route_deferrals,
            "target_route_holds": self.target_route_holds,
            "target_route_failures": self.target_route_failures,
            "target_route_releases": self.target_route_releases,
            "target_approach_terminals": self.target_approach_terminals,
            "min_scan_clearance": None if not math.isfinite(self.min_clearance) else round(self.min_clearance, 4),
            "discontinuity_obstacle_clearance": round(
                self.discontinuity_obstacle_clearance, 4
            ),
            "map": self.map_stats,
        }

    def on_sample(self, _event):
        with self.lock:
            self.sample_count += 1
            self._write("INFO", "sample", sample=self.sample_count, **self._snapshot())

    def close(self):
        with self.lock:
            if self.navigation_hold_start_wall is not None:
                self.navigation_hold_duration_total += max(
                    0.0, time.monotonic() - self.navigation_hold_start_wall
                )
                self.navigation_hold_start_wall = None
            if self.zero_start_wall is not None:
                duration = time.monotonic() - self.zero_start_wall
                self.zero_duration_total += duration
                self.stop_duration_count += 1
                self.last_stop_duration = duration
                self.max_stop_duration = max(self.max_stop_duration, duration)
                self.zero_start_wall = None
            if self.turn_only_start_wall is not None:
                self.turn_only_duration_total += time.monotonic() - self.turn_only_start_wall
                self.turn_only_start_wall = None
            self._write(
                "INFO",
                "run_stop",
                wall_duration_seconds=round(time.monotonic() - self.start_wall, 3),
                summary=self._snapshot(),
                status_counts=self.status_counts,
            )
            try:
                self.stream.close()
            except (AttributeError, ValueError):
                pass


if __name__ == "__main__":
    NavigationMetrics()
    rospy.spin()
