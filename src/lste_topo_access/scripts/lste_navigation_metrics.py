#!/usr/bin/env python3
"""Write one structured navigation telemetry log for each live LSTE run.

The node intentionally observes the complete goal/controller/safety chain. It
does not publish commands or alter navigation decisions, so it can be left in
the normal pipeline while comparing detector, RL, and TEB behavior.
"""

import collections
import datetime
import json
import math
import os
import re
import shutil
import struct
import threading
import time
from collections import deque
from pathlib import Path

import rospy
import tf
from actionlib_msgs.msg import GoalStatusArray
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from lste_msgs.msg import LsteDetections, LsteScores, LsteState
from move_base_msgs.msg import MoveBaseActionFeedback, MoveBaseActionGoal, RecoveryStatus
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan
from std_msgs.msg import Bool, String
from teb_local_planner.msg import FeedbackMsg
from tf.transformations import euler_from_quaternion

try:
    # ``compressedDepth`` uses PNG payloads.  Keep these optional because this
    # observer must still start on a robot installation without OpenCV/NumPy;
    # it will report depth validation as unavailable rather than fail the
    # navigation run.
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


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
        self.task_id = str(rospy.get_param("~task_id", "")).strip()
        self.execution_architecture = str(
            rospy.get_param("~execution_architecture", "unknown")
        ).strip().lower() or "unknown"
        self.persistent_execution = self.execution_architecture == "persistent_stream"
        self.pose = None
        self.goal = None
        self.goal_frame = "odom"
        self.last_goal_frame = "odom"
        self.goal_message = None
        self.tf_listener = tf.TransformListener()
        self.distance_transform_failures = 0
        # This is intentionally an evaluator, never a source of navigation
        # inputs. Gazebo has no world -> odom TF in this workspace, so each
        # ModelStates sample records the simultaneous world/odom base poses.
        # It lets logs distinguish "the target was not in the camera frustum"
        # from "the target was in view but the detector did not match it".
        self.target_eval_enabled = self._as_bool(
            rospy.get_param("~target_eval_enabled", False)
        )
        self.target_eval_target_model = str(
            rospy.get_param("~target_eval_gazebo_target_model", "")
        ).strip()
        self.target_eval_robot_model = str(
            rospy.get_param("~target_eval_gazebo_robot_model", "pro3")
        ).strip()
        self.target_eval_world_frame = str(
            rospy.get_param("~target_eval_world_frame", "world")
        ).strip().lstrip("/") or "world"
        self.target_eval_odom_frame = str(
            rospy.get_param("~target_eval_odom_frame", "odom")
        ).strip().lstrip("/") or "odom"
        self.target_eval_camera_frame = str(
            rospy.get_param("~target_eval_camera_frame", "")
        ).strip().lstrip("/")
        self.target_eval_camera_info_topic = str(
            rospy.get_param("~target_eval_camera_info_topic", "/kinect/hd/camera_info")
        ).strip()
        # This is the depth-compressed transport advertised by the Gazebo
        # Kinect in this workspace.  It is sampled solely for evaluation: a
        # Gazebo target projected into the RGB frustum counts as *visible*
        # only when a time-matched depth image contains target-range support
        # at that projection.  None of this data reaches GoalManager, Navfn,
        # TEB, or the actuator mux.
        self.target_eval_depth_enabled = self._as_bool(
            rospy.get_param("~target_eval_depth_enabled", True)
        )
        self.target_eval_depth_topic = str(
            rospy.get_param(
                "~target_eval_depth_topic",
                "/kinect/hd/image_color_rect/compressedDepth",
            )
        ).strip()
        self.target_eval_depth_max_source_age = max(
            0.01,
            float(rospy.get_param("~target_eval_depth_max_source_age_s", 0.25)),
        )
        self.target_eval_depth_sample_grid = min(9, max(
            2, int(rospy.get_param("~target_eval_depth_sample_grid", 5))
        ))
        self.target_eval_depth_min_valid_samples = max(
            1, int(rospy.get_param("~target_eval_depth_min_valid_samples", 3))
        )
        self.target_eval_depth_min_matching_samples = max(
            1, int(rospy.get_param("~target_eval_depth_min_matching_samples", 2))
        )
        self.target_eval_depth_abs_tolerance = max(
            0.0,
            float(rospy.get_param("~target_eval_depth_abs_tolerance_m", 0.12)),
        )
        self.target_eval_depth_relative_tolerance = max(
            0.0,
            float(rospy.get_param("~target_eval_depth_relative_tolerance", 0.05)),
        )
        self.target_eval_detections_topic = str(
            rospy.get_param("~target_eval_detections_topic", "/lste/detections")
        ).strip()
        # Retain configured aliases as run metadata only.  A detection node
        # already separates the current prompt's candidates into
        # ``target_dets`` and ``env_dets``. Re-filtering ``target_dets`` here
        # with a static alias list can silently discard valid detections when
        # a task prompt changes or a shell splits an alias containing spaces.
        self.target_eval_labels = {
            self._normalize_label(value)
            for value in str(
                rospy.get_param("~target_eval_target_labels", "")
            ).split(",")
            if self._normalize_label(value)
        }
        self.target_eval_center = self._csv_floats(
            rospy.get_param("~target_eval_target_center_m", "0,0,0"), 3
        )
        self.target_eval_size = self._csv_floats(
            rospy.get_param("~target_eval_target_size_m", "0.1,0.1,0.1"), 3
        )
        self.target_eval_min_depth = max(
            0.01, float(rospy.get_param("~target_eval_min_depth_m", 0.20))
        )
        self.target_eval_max_depth = max(
            self.target_eval_min_depth,
            float(rospy.get_param("~target_eval_max_depth_m", 12.0)),
        )
        self.target_eval_edge_margin = max(
            0.0, float(rospy.get_param("~target_eval_edge_margin_px", 8.0))
        )
        self.target_eval_max_state_age = max(
            0.05, float(rospy.get_param("~target_eval_max_gazebo_state_age_s", 0.50))
        )
        self.target_eval_exposure_hold = max(
            0.0, float(rospy.get_param("~target_eval_exposure_hold_s", 0.25))
        )
        self.target_eval_episode_gap = max(
            0.05, float(rospy.get_param("~target_eval_episode_gap_s", 0.75))
        )
        self.target_eval_min_match_iou = min(1.0, max(
            0.0, float(rospy.get_param("~target_eval_min_match_iou", 0.10))
        ))
        self.target_eval_min_score = min(1.0, max(
            0.0, float(rospy.get_param("~target_eval_min_score", 0.20))
        ))
        self.target_eval_camera = None
        self.target_eval_depth_frames = deque(maxlen=10)
        self.target_eval_depth_frames_received = 0
        self.target_eval_depth_frames_decoded = 0
        self.target_eval_depth_decode_failures = 0
        self.target_eval_depth_last_decode_reason = ""
        self.target_eval_odom_pose = None
        self.target_eval_states = deque(maxlen=80)
        self.target_eval_last_model_wall = None
        self.target_eval_last_source_stamp = None
        self.target_eval_last_exposed_stamp = None
        self.target_eval_candidate_started_stamp = None
        self.target_eval_active_episode = None
        self.target_eval_completed_episodes = []
        self.target_eval_episode_sequence = 0
        self.target_eval_exposed_frames = 0
        self.target_eval_matched_frames = 0
        # These are deliberately distinct from the established geometric
        # counters above.  A frustum projection only proves line-of-sight in
        # ideal geometry; depth support proves that the rendered camera has a
        # surface at the target's expected range.
        self.target_eval_depth_evaluated_frames = 0
        self.target_eval_depth_visible_frames = 0
        self.target_eval_depth_visible_matched_frames = 0
        self.target_eval_depth_occluded_frames = 0
        self.target_eval_depth_inconsistent_frames = 0
        self.target_eval_depth_unavailable_frames = 0
        self.target_eval_depth_last_visible_stamp = None
        self.target_eval_depth_candidate_started_stamp = None
        self.target_eval_depth_active_episode = None
        self.target_eval_depth_completed_episodes = []
        self.target_eval_depth_episode_sequence = 0
        self.target_eval_evaluated_frames = 0
        # A target-stream candidate without a spatial match is not necessarily
        # a false positive: geometric frustum exposure does not prove that the
        # target is unoccluded in RGB. Keep the literal metric name.
        self.target_eval_unmatched_target_candidate_frames = 0
        self.target_eval_outside_exposure_candidates = 0
        self.target_eval_unavailable_frames = 0
        self.target_eval_tf_failures = 0
        self.target_eval_delivery_latency_total = 0.0
        self.target_eval_delivery_latency_max = 0.0
        self.target_eval_delivery_latency_count = 0
        self.target_eval_first_exposure_stamp = None
        self.target_eval_first_match_stamp = None
        self.target_eval_first_depth_visible_stamp = None
        self.target_eval_first_depth_visible_match_stamp = None
        self.target_eval_last_unavailable_reason = ""
        self.target_eval_last_unavailable_log_wall = 0.0
        self.target_eval_last_frame_log_wall = 0.0
        self.subgoal = None
        self.command = Twist()
        self.teb_command = Twist()
        self.teb_planner_command = Twist()
        self.teb_turn_supervisor_status = None
        self.teb_turn_supervisor_events = 0
        self.teb_turn_supervisor_last_event = "unknown"
        # A bounded pass-through event where TEB feedback remains forward but
        # its raw cmd_vel has one zero scheduler tick. Keep this separate from
        # safety brakes and route terminal stops so a smoothness result can be
        # audited instead of inferred from the visual behavior.
        self.teb_trajectory_continuity_events = 0
        self.controller_mode = "unknown"
        self.controller_status = "not_available"
        self.controller_source = "unknown"
        self.controller_reason = "not_available"
        self.controller_requested = (float("nan"), float("nan"))
        self.controller_action = (float("nan"), float("nan"))
        self.controller_predicted_clearance = float("nan")
        self.controller_status_changes = 0
        # The mux is the actuator boundary. Its structured status is the
        # authoritative cause for a governor cap or a forced zero; scan-based
        # classification below remains a compatibility fallback only.
        self.cmd_vel_mux_status = None
        self.cmd_vel_mux_status_wall = None
        self.mux_governor_limited_events = 0
        self.mux_forced_zero_events = 0
        self.mux_status_reason_counts = {}
        self.bridge_events = 0
        self.bridge_deferred_goal_updates = 0
        self.bridge_dispatches = 0
        self.bridge_terminal_events = 0
        self.bridge_priority_handoffs = 0
        self.bridge_target_retries = 0
        self.bridge_target_segment_handoffs = 0
        self.bridge_frontier_segment_handoffs = 0
        self.bridge_frontier_observation_completions = 0
        self.bridge_frontier_terminal_settle_completions = 0
        self.bridge_frontier_continuous_prefetch_handoffs = 0
        self.bridge_frontier_continuous_prefetch_fallbacks = 0
        self.bridge_frontier_prefetch_requires_turn = 0
        # Persistent RouteCorridor handoffs keep the same move_base lease.
        # Track their local-costmap/Navfn admission separately from legacy
        # actionlib handoffs, otherwise a zero-stop improvement would have no
        # evidence that the successor route was actually checked.
        self.bridge_persistent_lookahead_handoffs = 0
        self.bridge_persistent_curve_handoffs = 0
        self.bridge_persistent_lookahead_admission_deferred = 0
        self.bridge_persistent_lookahead_admission_reasons = {}
        self.bridge_frontier_sharp_replacements = 0
        self.bridge_goal_replacements = 0
        self.bridge_priority_goal_replacements = 0
        self.bridge_target_goal_replacements = 0
        self.bridge_active = False
        self.bridge_last_event = "unknown"
        self.bridge_active_intent_source = "unknown"
        self.bridge_latest_intent_source = "unknown"
        # A persistent prefetch may be rejected because the successor begins
        # with a proven sharp BFS tangent.  The current endpoint is then
        # intentionally allowed to complete before TEB rotates for the next
        # branch. Retain that narrow route contract so its terminal stop is
        # not misreported as an unexplained clear-space brake.
        self.pending_terminal_native_reorientation = None
        # Endpoint stops and local-planner recoveries are both visible as a
        # zero command, but demand very different fixes. Keep their lifecycle
        # evidence separate and measure how quickly a successful action is
        # replaced by its successor.
        self.move_base_recovery_events = 0
        self.pending_action_terminal_wall = None
        self.pending_action_terminal_source = ""
        self.terminal_to_dispatch_count = 0
        self.terminal_to_dispatch_total = 0.0
        self.terminal_to_dispatch_max = 0.0
        self.terminal_to_dispatch_last = None
        self.safety_override_events = 0
        # ``safety_override_events`` is a state-transition count kept for
        # backwards-compatible dashboards. These counters describe the actual
        # control stream more precisely: an intervention sample is one where
        # the guard source or requested/applied action differs materially.
        self.safety_intervention_samples = 0
        self.controller_action_delta = 0.0
        self.hard_stop_events = 0
        self.linear_brake_events = 0
        # A local planner deliberately modulates speed for curvature and
        # endpoint approach.  Keep those non-zero changes separate from an
        # actual motion interruption, otherwise a normal 0.50 -> 0.37 m/s
        # turn is reported as an "emergency brake".
        self.speed_modulation_events = 0
        self.teb_speed_modulation_events = 0
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
        # A zero command usually arrives one scheduler tick before move_base
        # publishes its SUCCEEDED status. Keep those provisional records long
        # enough to correct their cause when the terminal status follows.
        self.recent_discontinuities = deque()
        self.discontinuity_sequence = 0
        self.last_status_text = ""
        self.last_status_signature = None
        self.teb_status = "not_available"
        self.teb_feedback_state = None
        # TEB feedback and raw cmd_vel use separate ROS callback paths. Keep
        # the receive time so a one-tick raw-zero can be distinguished from a
        # real planner stop when the selected trajectory is still forward.
        self.teb_feedback_wall = None
        self.teb_control_cycle_gap_max_feedback_age = max(
            0.05,
            float(rospy.get_param("~teb_control_cycle_gap_max_feedback_age", 0.60)),
        )
        self.teb_control_cycle_gap_min_goal_distance = max(
            0.05,
            float(rospy.get_param("~teb_control_cycle_gap_min_goal_distance", 0.90)),
        )
        self.move_base_feedback_state = None
        self.recovery_state = None
        self.global_costmap_stats = None
        self.local_costmap_stats = None
        self.navfn_plan_stats = None
        self.global_planner_plan_stats = None
        self.teb_global_plan_stats = None
        self.teb_local_plan_stats = None
        # Keep the active TEB global path geometry only for evaluation. The
        # metric uses it to separate intentional route bends from left/right
        # corrections while the planned route is locally straight.
        self.teb_global_plan_geometry = None
        self.teb_local_plan_geometry = None
        self.last_teb_feedback_log_wall = 0.0
        # A valid TEB trajectory whose first velocity remains nearly zero is
        # materially different from an invalid trajectory or a safety-layer
        # stop. Capture its geometry once per plateau so an offline diagnosis
        # can distinguish a blocked endpoint from optimizer degeneration.
        self.teb_zero_velocity_start_wall = None
        self.teb_zero_velocity_snapshot_wall = 0.0
        self.teb_zero_velocity_snapshot_min_duration = 0.70
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
        self.move_base_goal_ids = set()
        self.status_counts = {}
        self.preemptions = 0
        self.frontier_observation_preemptions = 0
        self.frontier_terminal_settle_preemptions = 0
        self.frontier_continuous_prefetch_preemptions = 0
        self.frontier_segment_preemptions = 0
        self.target_segment_preemptions = 0
        self.priority_preemptions = 0
        self.task_done_preemptions = 0
        self.unexpected_preemptions = 0
        self.pending_frontier_observation_preemptions = 0
        self.pending_frontier_terminal_settle_preemptions = 0
        self.pending_frontier_continuous_prefetch_preemptions = 0
        self.pending_frontier_segment_preemptions = 0
        self.pending_target_segment_preemptions = 0
        self.pending_priority_preemptions = 0
        self.pending_task_done_preemptions = 0
        self.aborts = 0
        self.successes = 0
        self.cmd_messages = 0
        self.teb_cmd_messages = 0
        self.angular_sign_flips = 0
        self.last_nonzero_angular_sign = 0
        self.strong_angular_sign_flips = 0
        # Unlike the legacy count above, this counts only direct left/right
        # changes while both adjacent commands are forward motion. It is the
        # metric for actual corridor wobble, excluding route-boundary turns.
        self.forward_steering_sign_flips = 0
        self.teb_angular_sign_flips = 0
        self.teb_last_nonzero_angular_sign = 0
        self.teb_strong_angular_sign_flips = 0
        self.teb_forward_steering_sign_flips = 0
        self.teb_strong_angular_threshold = max(
            0.0, float(rospy.get_param("~teb_strong_angular_threshold", 0.12))
        )
        self.teb_linear_brake_events = 0
        self.teb_planner_linear_brake_events = 0
        self.teb_planner_speed_modulation_events = 0
        self.stop_events = 0
        self.zero_start_wall = None
        self.zero_duration_total = 0.0
        self.stop_duration_count = 0
        self.max_stop_duration = 0.0
        self.last_stop_duration = None
        self.detector_messages = 0
        self.target_messages = 0
        self.target_first_seen_ros = None
        self.target_follow_confirmed_ros = None
        self.target_close_confirmation_started_ros = None
        self.target_close_confirmed_ros = None
        self.legacy_state_locked_ros = None
        self.target_lock_ros = None
        self.task_done_ros = None
        # Keep target evidence by its mission identity. A detector may lose a
        # weak track and later confirm another one during the same task; the
        # timestamps must never be combined across those tracks.
        self.target_lifecycle_sessions = {}
        self.target_lifecycle_last_completed_key = None
        self.goal_source = "unknown"
        self.goal_transaction_id = 0
        self.mission_goal_messages = 0
        self.goal_transition_kind = "unknown"
        self.goal_predecessor_route_id = 0
        self.goal_transition_distance = None
        self.last_goal_transition_wall = None
        self.persistent_plan_received = 0
        self.persistent_plan_equivalent_retained = 0
        self.persistent_plan_installed = 0
        self.persistent_plan_last_event = "unknown"
        self.persistent_plan_route_version = 0
        self.persistent_plan_geometry_hash = None
        self.continuous_goal_transitions = 0
        self.divergent_goal_transitions = 0
        self.terminal_goal_transitions = 0
        self.unknown_goal_transitions = 0
        self.transition_brake_events = collections.Counter()
        self.last_goal_publish_wall = None
        self.target_goal_changes = 0
        self.target_route_accepts = 0
        self.target_route_rejections = 0
        self.target_route_deferrals = 0
        self.target_route_holds = 0
        self.target_route_semantic_replans = 0
        self.target_route_failures = 0
        self.target_route_releases = 0
        self.target_approach_terminals = 0
        self.target_segments_committed = 0
        self.target_continuous_handoffs_prepared = 0
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
        self.straight_path_distance = 0.0
        self.straight_path_angular_energy = 0.0
        self.straight_path_steering_sign_flips = 0
        self.straight_path_last_nonzero_sign = 0
        self.straight_path_last_nonzero_sign_wall = 0.0
        self.straight_path_sign_flip_window = 1.0
        self.straight_path_samples = 0
        self.straight_path_state = None
        self.straight_path_last_eval_wall = 0.0
        self.straight_path_eval_period = max(
            0.05, float(rospy.get_param("~straight_path_eval_period", 0.20))
        )
        self.straight_path_lookahead = max(
            0.30, float(rospy.get_param("~straight_path_lookahead", 1.00))
        )
        self.straight_path_max_curvature = max(
            0.01, float(rospy.get_param("~straight_path_max_curvature", 0.14))
        )
        self.straight_path_max_heading_error = max(
            0.01, float(rospy.get_param("~straight_path_max_heading_error", 0.20))
        )
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

        run_context = self._run_start_context()
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
                "persistent_terminal": "/lste/teb_goal_terminal",
                "global_frontier_status": "/lste/global_frontier/status",
                "teb_goal_failure": "/lste/teb_goal_failure",
                "persistent_plan_event": "/lste/persistent_execution/plan_event",
                "goal_arbitration": "/lste/goal_arbitration",
                "perception_decision": "/lste/perception_decision",
                "navigation_hold": "/lste/navigation_hold",
                "global_costmap": "/move_base/global_costmap/costmap",
                "local_costmap": "/move_base/local_costmap/costmap",
                "gazebo_model_states": "/gazebo/model_states" if self.target_eval_enabled else None,
                "camera_info": self.target_eval_camera_info_topic if self.target_eval_enabled else None,
                "depth": (
                    self.target_eval_depth_topic
                    if self.target_eval_enabled and self.target_eval_depth_enabled
                    else None
                ),
            },
            resolved_params=self._resolved_startup_params(),
            experiment=run_context["experiment"],
            task_definition=run_context["task_definition"],
            detector=run_context["detector"],
            target_thresholds=run_context["target_thresholds"],
            target_geometric_evaluation=self._target_eval_configuration(),
        )
        rospy.loginfo("Navigation metrics log: %s", self.log_path)

        rospy.Subscriber("/pro3/wheel_odom", Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber("/rbt_pose", Pose2D, self.on_pose2d, queue_size=1)
        rospy.Subscriber("/lste/final_goal", PoseStamped, self.on_goal, queue_size=1)
        # The action bridge executes this atomic GoalManager contract. Observe
        # it directly so a goal-change record has the correct source/route id
        # even before the human-readable diagnostic callback arrives.
        rospy.Subscriber("/lste/mission_goal", String, self.on_mission_goal, queue_size=10)
        rospy.Subscriber("/lste/goal_diagnostic", String, self.on_goal_diagnostic, queue_size=10)
        rospy.Subscriber("/lste/goal_arbitration", String, self.on_goal_arbitration, queue_size=10)
        # The TEB bridge uses the typed move_base action. Keep the legacy
        # simple-goal observer for older comparison launches, but count either
        # transport through the same dispatch recorder.
        rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, self.on_action_dispatch, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.on_dispatch, queue_size=1)
        rospy.Subscriber("/move_base/status", GoalStatusArray, self.on_status, queue_size=1)
        rospy.Subscriber("/cmd_vel", Twist, self.on_cmd, queue_size=1)
        rospy.Subscriber(
            "/lste/cmd_vel_mux/status", String, self.on_cmd_vel_mux_status,
            queue_size=50,
        )
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
            "/lste/teb_goal_terminal",
            PoseStamped,
            self.on_persistent_execution_terminal,
            queue_size=10,
        )
        rospy.Subscriber(
            "/lste/global_frontier/status",
            String,
            self.on_global_frontier_status,
            queue_size=20,
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
        rospy.Subscriber(
            "/lste/persistent_execution/plan_event",
            String,
            self.on_persistent_plan_event,
            queue_size=20,
        )
        rospy.Subscriber("/move_base/feedback", MoveBaseActionFeedback, self.on_move_base_feedback, queue_size=1)
        rospy.Subscriber("/move_base/recovery_status", RecoveryStatus, self.on_recovery, queue_size=1)
        rospy.Subscriber("/lste/state", LsteState, self.on_state, queue_size=1)
        rospy.Subscriber(
            self.target_eval_detections_topic,
            LsteDetections,
            self.on_detections,
            queue_size=1,
        )
        if self.target_eval_enabled:
            rospy.Subscriber(
                "/gazebo/model_states", ModelStates, self.on_gazebo_model_states, queue_size=5
            )
            rospy.Subscriber(
                self.target_eval_camera_info_topic,
                CameraInfo,
                self.on_target_eval_camera_info,
                queue_size=1,
            )
            if self.target_eval_depth_enabled and self.target_eval_depth_topic:
                rospy.Subscriber(
                    self.target_eval_depth_topic,
                    CompressedImage,
                    self.on_target_eval_depth,
                    queue_size=2,
                    buff_size=8 * 1024 * 1024,
                )
        rospy.Subscriber(
            "/lste/perception_decision", String, self.on_perception_decision, queue_size=20
        )
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
            "persistent_execution": "/move_base/persistent_execution",
            "persistent_frontier_lookahead_enabled": "/lste_teb_goal_bridge/persistent_frontier_lookahead_handoff_enabled",
            "persistent_frontier_lookahead_trigger_distance": "/lste_teb_goal_bridge/persistent_frontier_lookahead_trigger_distance",
            "persistent_frontier_admission_horizon": "/lste_teb_goal_bridge/persistent_frontier_admission_horizon",
            "persistent_frontier_admission_max_costmap_age": "/lste_teb_goal_bridge/persistent_frontier_admission_max_costmap_age",
            "persistent_frontier_curve_handoff_max_heading_deg": "/lste_teb_goal_bridge/persistent_frontier_curve_handoff_max_heading_deg",
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
            "frontier_turn_execution_mode": "/lste_global_frontier/turn_execution_mode",
            "frontier_legacy_turn_connector_threshold_deg": "/lste_global_frontier/explicit_turn_connector_threshold_deg",
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

    def _run_start_context(self):
        """Capture immutable task and perception settings once per run."""
        task_json = str(rospy.get_param("~task_json", "")).strip()
        task_definition = {
            "task_id": str(rospy.get_param("~task_id", "")).strip(),
            "json_path": task_json or None,
        }
        if task_json:
            try:
                with open(task_json, "r", encoding="utf-8") as stream:
                    parsed = json.load(stream)
                definition = parsed.get("task_parsed", parsed)
                if isinstance(definition, dict):
                    task_definition["definition"] = definition
                else:
                    task_definition["definition_error"] = "task JSON is not an object"
            except (OSError, ValueError, TypeError) as exc:
                task_definition["definition_error"] = str(exc)

        def param(name, default=None):
            return rospy.get_param("~" + name, default)

        return {
            "experiment": {
                "pipeline_config": param("pipeline_config"),
                "pipeline_config_sha256": param("pipeline_config_sha256"),
                "world": param("world"),
                "initial_pose": {
                    "x": param("pro3_spawn_x"),
                    "y": param("pro3_spawn_y"),
                    "z": param("pro3_spawn_z"),
                    "yaw": param("pro3_spawn_yaw"),
                },
                "online_slam_enabled": self._as_bool(
                    param("online_slam_enabled", False)
                ),
                "global_frontier_enabled": self._as_bool(
                    param("global_frontier_enabled", False)
                ),
                "startup_forward_enabled": self._as_bool(
                    param("startup_forward_enabled", False)
                ),
                "global_goal_source": param("global_goal_source"),
                "git_revision": param("git_revision"),
                "git_dirty": self._as_bool(param("git_dirty", False)),
            },
            "task_definition": task_definition,
            "detector": {
                "name": param("detector"),
                "backend": param("detector_backend"),
                "variant": param("detector_variant"),
                "runtime": param("detector_runtime"),
                "score_threshold": param("detector_score_threshold"),
                "nms_iou": param("detector_nms_iou"),
                "min_inference_interval": param("detector_min_inference_interval"),
            },
            "target_thresholds": {
                "follow_min_score": param("target_follow_min_score"),
                "follow_min_box_size": param("target_follow_min_box_size"),
                "follow_confirm_hits": param("target_follow_confirm_hits"),
                "follow_confirm_window": param("target_follow_confirm_window"),
                "done_min_score": param("target_done_min_score"),
                "done_min_box_width": param("target_done_min_box_width"),
                "done_min_box_height": param("target_done_min_box_height"),
                "done_min_fresh_hits": param("target_done_min_fresh_hits"),
                "done_min_hold_time": param("target_done_min_hold_time"),
                "done_require_approach_terminal": param(
                    "target_done_require_approach_terminal"
                ),
            },
        }

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

    @staticmethod
    def _as_bool(value):
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _normalize_label(value):
        return " ".join(str(value).strip().lower().replace("_", " ").split())

    @staticmethod
    def _csv_floats(value, count):
        try:
            values = [float(item.strip()) for item in str(value).split(",")]
        except (TypeError, ValueError):
            values = []
        if len(values) != count or not all(math.isfinite(item) for item in values):
            return tuple(0.0 for _ in range(count))
        return tuple(values)

    @staticmethod
    def _quaternion_conjugate(quaternion):
        x, y, z, w = (float(value) for value in quaternion)
        norm = x * x + y * y + z * z + w * w
        if norm <= 1e-12:
            return (0.0, 0.0, 0.0, 1.0)
        return (-x / norm, -y / norm, -z / norm, w / norm)

    @staticmethod
    def _rotate_point(quaternion, point):
        """Apply a quaternion rotation without adding a NumPy dependency."""
        qx, qy, qz, qw = (float(value) for value in quaternion)
        px, py, pz = (float(value) for value in point)
        norm = qx * qx + qy * qy + qz * qz + qw * qw
        if norm <= 1e-12:
            return (px, py, pz)
        qx, qy, qz, qw = qx / math.sqrt(norm), qy / math.sqrt(norm), qz / math.sqrt(norm), qw / math.sqrt(norm)
        tx = 2.0 * (qy * pz - qz * py)
        ty = 2.0 * (qz * px - qx * pz)
        tz = 2.0 * (qx * py - qy * px)
        return (
            px + qw * tx + (qy * tz - qz * ty),
            py + qw * ty + (qz * tx - qx * tz),
            pz + qw * tz + (qx * ty - qy * tx),
        )

    def _target_eval_configuration(self):
        """Return the simulator-only evaluator contract recorded at run start."""
        return {
            "enabled": self.target_eval_enabled,
            "truth_source": "gazebo_evaluation_only",
            "target_model": self.target_eval_target_model,
            "task_id": self.task_id or None,
            "robot_model": self.target_eval_robot_model,
            "world_frame": self.target_eval_world_frame,
            "odom_frame": self.target_eval_odom_frame,
            "camera_frame": self.target_eval_camera_frame or "from_camera_info",
            "camera_info_topic": self.target_eval_camera_info_topic,
            "depth_validation": {
                "enabled": self.target_eval_depth_enabled,
                "topic": self.target_eval_depth_topic or None,
                "transport": "sensor_msgs/CompressedImage compressedDepth",
                "alignment_contract": (
                    "depth pixels are normalized against the CameraInfo image; "
                    "the configured depth topic must be registered/aligned to it"
                ),
                "max_source_age_s": self.target_eval_depth_max_source_age,
                "sample_grid": self.target_eval_depth_sample_grid,
                "min_valid_samples": self.target_eval_depth_min_valid_samples,
                "min_matching_samples": self.target_eval_depth_min_matching_samples,
                "absolute_tolerance_m": self.target_eval_depth_abs_tolerance,
                "relative_tolerance": self.target_eval_depth_relative_tolerance,
                "decoder_available": cv2 is not None and np is not None,
            },
            "detections_topic": self.target_eval_detections_topic,
            "configured_target_labels_metadata": sorted(self.target_eval_labels),
            "candidate_contract": (
                "LsteDetections.target_dets for the matching task_id; "
                "boxes and projected truth are normalized xyxy"
            ),
            "target_center_m": list(self.target_eval_center),
            "target_size_m": list(self.target_eval_size),
            "min_depth_m": self.target_eval_min_depth,
            "max_depth_m": self.target_eval_max_depth,
            "edge_margin_px": self.target_eval_edge_margin,
            "max_gazebo_state_age_s": self.target_eval_max_state_age,
            "exposure_hold_s": self.target_eval_exposure_hold,
            "episode_gap_s": self.target_eval_episode_gap,
            "min_match_iou": self.target_eval_min_match_iou,
            "min_score": self.target_eval_min_score,
            "control_contract": "observer_only_no_publishers_or_control_params",
        }

    def _target_eval_note_unavailable_locked(self, reason, **fields):
        """Count unavailable detector frames without creating a log flood."""
        self.target_eval_unavailable_frames += 1
        now = time.monotonic()
        if (
            reason != self.target_eval_last_unavailable_reason
            or now - self.target_eval_last_unavailable_log_wall >= 2.0
        ):
            self._write(
                "WARN",
                "target_eval_unavailable",
                truth_source="gazebo_evaluation_only",
                reason=reason,
                **fields
            )
            self.target_eval_last_unavailable_reason = reason
            self.target_eval_last_unavailable_log_wall = now

    @staticmethod
    def _target_eval_bbox_iou(first, second):
        left = max(float(first[0]), float(second[0]))
        top = max(float(first[1]), float(second[1]))
        right = min(float(first[2]), float(second[2]))
        bottom = min(float(first[3]), float(second[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, float(first[2]) - float(first[0])) * max(
            0.0, float(first[3]) - float(first[1])
        )
        second_area = max(0.0, float(second[2]) - float(second[0])) * max(
            0.0, float(second[3]) - float(second[1])
        )
        union = first_area + second_area - intersection
        return 0.0 if union <= 1e-9 else intersection / union

    def _target_eval_world_to_odom_point(self, state, world_point):
        """Map a Gazebo world point through the sampled world/odom base relation."""
        robot_world = state["robot_world"]
        odom = state["odom"]
        delta_world = (
            float(world_point[0]) - robot_world["position"][0],
            float(world_point[1]) - robot_world["position"][1],
            float(world_point[2]) - robot_world["position"][2],
        )
        in_base = self._rotate_point(
            self._quaternion_conjugate(robot_world["orientation"]), delta_world
        )
        in_odom = self._rotate_point(odom["orientation"], in_base)
        return (
            odom["position"][0] + in_odom[0],
            odom["position"][1] + in_odom[1],
            odom["position"][2] + in_odom[2],
        )

    def _target_eval_projected_box_locked(self, source_stamp, state):
        """Project the configured Gazebo target bounds into the RGB camera."""
        if self.target_eval_camera is None:
            return None, "camera_info_unavailable"
        camera = self.target_eval_camera
        try:
            translation, rotation = self.tf_listener.lookupTransform(
                camera["frame"],
                self.target_eval_odom_frame,
                rospy.Time.from_sec(source_stamp),
            )
        except (
            tf.Exception,
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as exc:
            self.target_eval_tf_failures += 1
            return None, "camera_tf_unavailable:%s" % type(exc).__name__

        target = state["target_world"]
        local_points = [self.target_eval_center]
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                for sz in (-0.5, 0.5):
                    local_points.append((
                        self.target_eval_center[0] + sx * self.target_eval_size[0],
                        self.target_eval_center[1] + sy * self.target_eval_size[1],
                        self.target_eval_center[2] + sz * self.target_eval_size[2],
                    ))

        camera_points = []
        for local_point in local_points:
            rotated = self._rotate_point(target["orientation"], local_point)
            world_point = (
                target["position"][0] + rotated[0],
                target["position"][1] + rotated[1],
                target["position"][2] + rotated[2],
            )
            odom_point = self._target_eval_world_to_odom_point(state, world_point)
            rotated_camera = self._rotate_point(rotation, odom_point)
            camera_points.append((
                rotated_camera[0] + translation[0],
                rotated_camera[1] + translation[1],
                rotated_camera[2] + translation[2],
            ))

        center = camera_points[0]
        if not self.target_eval_min_depth <= center[2] <= self.target_eval_max_depth:
            return {
                "exposed": False,
                "reason": "center_depth_out_of_range",
                "center_depth_m": center[2],
                "target_depth_min_m": None,
                "target_depth_max_m": None,
                "predicted_box_px": None,
                "predicted_box_normalized": None,
            }, None
        projected = []
        for point in camera_points[1:]:
            if point[2] <= self.target_eval_min_depth:
                continue
            projected.append((
                camera["fx"] * point[0] / point[2] + camera["cx"],
                camera["fy"] * point[1] / point[2] + camera["cy"],
            ))
        if len(projected) < 2:
            return {
                "exposed": False,
                "reason": "target_bounds_behind_camera",
                "center_depth_m": center[2],
                "target_depth_min_m": None,
                "target_depth_max_m": None,
                "predicted_box_px": None,
                "predicted_box_normalized": None,
            }, None
        predicted_box = (
            min(point[0] for point in projected),
            min(point[1] for point in projected),
            max(point[0] for point in projected),
            max(point[1] for point in projected),
        )
        center_px = (
            camera["fx"] * center[0] / center[2] + camera["cx"],
            camera["fy"] * center[1] / center[2] + camera["cy"],
        )
        intersection_width = max(
            0.0, min(float(camera["width"]), predicted_box[2]) - max(0.0, predicted_box[0])
        )
        intersection_height = max(
            0.0, min(float(camera["height"]), predicted_box[3]) - max(0.0, predicted_box[1])
        )
        in_margin = (
            self.target_eval_edge_margin <= center_px[0] <= camera["width"] - self.target_eval_edge_margin
            and self.target_eval_edge_margin <= center_px[1] <= camera["height"] - self.target_eval_edge_margin
        )
        exposed = intersection_width > 0.0 and intersection_height > 0.0 and in_margin
        # Detector messages carry normalized cx/cy/w/h.  Keep both forms in
        # the evidence log, but calculate IoU in this common normalized xyxy
        # coordinate system.  Clip the truth to the visible image just as the
        # detector backend clips its output at the image boundary.
        visible_box_px = (
            max(0.0, min(float(camera["width"]), predicted_box[0])),
            max(0.0, min(float(camera["height"]), predicted_box[1])),
            max(0.0, min(float(camera["width"]), predicted_box[2])),
            max(0.0, min(float(camera["height"]), predicted_box[3])),
        )
        predicted_box_normalized = (
            visible_box_px[0] / float(camera["width"]),
            visible_box_px[1] / float(camera["height"]),
            visible_box_px[2] / float(camera["width"]),
            visible_box_px[3] / float(camera["height"]),
        )
        # The transformed cuboid corners bound the camera-axis depth of any
        # physical surface represented by the configured Gazebo target.  The
        # depth evaluator below uses this interval with an explicit tolerance
        # instead of requiring a single, brittle centre-pixel depth.
        target_depths = [
            point[2] for point in camera_points
            if point[2] > self.target_eval_min_depth
        ]
        return {
            "exposed": exposed,
            "reason": "frustum_exposed" if exposed else "outside_image_or_margin",
            "center_depth_m": center[2],
            "target_depth_min_m": min(target_depths) if target_depths else None,
            "target_depth_max_m": max(target_depths) if target_depths else None,
            "center_px": center_px,
            "predicted_box_px": predicted_box,
            "predicted_box_normalized": predicted_box_normalized,
        }, None

    def _target_eval_decode_depth_message(self, message):
        """Decode a compressedDepth message into a depth image plus metre scale.

        The ROS ``compressed_depth_image_transport`` convention writes 16UC1
        images as a PNG containing millimetres.  Its 32FC1 form prepends a
        twelve-byte ``ConfigHeader`` and stores inverse depth in PNG.  Gazebo
        normally publishes the former, but accepting both avoids silently
        weakening the experiment when the sensor encoding changes.
        """
        if cv2 is None or np is None:
            return None, None, "depth_decoder_unavailable"
        format_text = str(message.format or "").strip()
        lower_format = format_text.lower()
        if "compresseddepth" not in lower_format:
            return None, None, "unsupported_depth_transport"
        payload = bytes(message.data)
        if not payload:
            return None, None, "empty_depth_payload"
        encoding = lower_format.split(";", 1)[0].strip()
        png_offset = 0
        depth_quant_a = None
        depth_quant_b = None
        if "32fc1" in encoding:
            # ``ConfigHeader`` is C++ ``int + float + float``. It has no
            # padding on the ROS platforms used here (twelve bytes).
            if len(payload) <= 12:
                return None, None, "truncated_32fc1_depth_header"
            try:
                _format, depth_quant_a, depth_quant_b = struct.unpack(
                    "<iff", payload[:12]
                )
            except struct.error:
                return None, None, "invalid_32fc1_depth_header"
            if not math.isfinite(depth_quant_a) or depth_quant_a <= 0.0:
                return None, None, "invalid_32fc1_depth_quantization"
            png_offset = 12
        encoded = np.frombuffer(payload[png_offset:], dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2 or image.size == 0:
            return None, None, "depth_png_decode_failed"
        if "16uc1" in encoding:
            if image.dtype != np.uint16:
                return None, None, "unexpected_16uc1_depth_dtype"
            return image, 0.001, None
        if "32fc1" in encoding:
            if image.dtype != np.uint16:
                return None, None, "unexpected_32fc1_depth_dtype"
            denominator = image.astype(np.float32) - float(depth_quant_b)
            with np.errstate(divide="ignore", invalid="ignore"):
                depth = float(depth_quant_a) / denominator
            depth[image == 0] = np.nan
            return depth, 1.0, None
        # Some image_transport versions omit the encoding prefix but still
        # produce a valid 16-bit PNG. Treat only that unambiguous case as mm.
        if image.dtype == np.uint16:
            return image, 0.001, None
        return None, None, "unsupported_depth_encoding"

    def on_target_eval_depth(self, message):
        """Keep a small timestamp-indexed depth history for detector evidence."""
        if not self.target_eval_enabled or not self.target_eval_depth_enabled:
            return
        source_stamp = float(message.header.stamp.to_sec())
        with self.lock:
            self.target_eval_depth_frames_received += 1
        if source_stamp <= 0.0:
            decoded, scale, reason = None, None, "depth_stamp_unavailable"
        else:
            decoded, scale, reason = self._target_eval_decode_depth_message(message)
        with self.lock:
            if reason is not None:
                self.target_eval_depth_decode_failures += 1
                if reason != self.target_eval_depth_last_decode_reason:
                    self._write(
                        "WARN",
                        "target_eval_depth_decode_unavailable",
                        truth_source="gazebo_depth_evaluation_only",
                        depth_topic=self.target_eval_depth_topic,
                        reason=reason,
                        format=str(message.format or ""),
                    )
                    self.target_eval_depth_last_decode_reason = reason
                return
            self.target_eval_depth_frames.append({
                "ros_time": source_stamp,
                "wall_time": time.monotonic(),
                "image": decoded,
                "scale": float(scale),
                "encoding": str(message.format or ""),
                "width": int(decoded.shape[1]),
                "height": int(decoded.shape[0]),
            })
            self.target_eval_depth_frames_decoded += 1
            self.target_eval_depth_last_decode_reason = ""

    def _target_eval_depth_visibility_locked(self, source_stamp, projection):
        """Validate frustum exposure against aligned, time-matched depth.

        The result is intentionally tri-state.  ``unavailable`` means the
        experiment has insufficient evidence and must not be counted as an
        occlusion; ``occluded`` means a majority of usable support pixels are
        materially closer than the target's rendered depth interval.
        """
        result = {
            "state": "not_in_frustum",
            "visible": False,
            "available": False,
            "reason": "target_not_frustum_exposed",
            "source_age_seconds": None,
            "valid_samples": 0,
            "matching_samples": 0,
            "foreground_samples": 0,
            "background_samples": 0,
            "sample_pixels": 0,
            "observed_depth_median_m": None,
            "expected_depth_min_m": projection.get("target_depth_min_m"),
            "expected_depth_max_m": projection.get("target_depth_max_m"),
            "tolerance_m": None,
        }
        if not projection.get("exposed"):
            return result
        if not self.target_eval_depth_enabled:
            result.update(
                state="disabled",
                reason="depth_validation_disabled",
            )
            return result
        if cv2 is None or np is None:
            result.update(state="unavailable", reason="depth_decoder_unavailable")
            return result
        if not self.target_eval_depth_frames:
            result.update(state="unavailable", reason="depth_frame_unavailable")
            return result
        depth_frame = min(
            self.target_eval_depth_frames,
            key=lambda item: abs(float(item["ros_time"]) - source_stamp),
        )
        source_age = abs(float(depth_frame["ros_time"]) - source_stamp)
        result["source_age_seconds"] = source_age
        if source_age > self.target_eval_depth_max_source_age:
            result.update(
                state="unavailable",
                reason="depth_timestamp_mismatch",
            )
            return result
        expected_min = projection.get("target_depth_min_m")
        expected_max = projection.get("target_depth_max_m")
        predicted_box = projection.get("predicted_box_px")
        camera = self.target_eval_camera
        if (
            expected_min is None or expected_max is None or predicted_box is None
            or camera is None or expected_min <= 0.0 or expected_max < expected_min
        ):
            result.update(state="unavailable", reason="target_depth_projection_invalid")
            return result
        depth_width = int(depth_frame["width"])
        depth_height = int(depth_frame["height"])
        if depth_width <= 0 or depth_height <= 0:
            result.update(state="unavailable", reason="depth_dimensions_invalid")
            return result
        left = max(0.0, min(float(camera["width"]), float(predicted_box[0])))
        top = max(0.0, min(float(camera["height"]), float(predicted_box[1])))
        right = max(0.0, min(float(camera["width"]), float(predicted_box[2])))
        bottom = max(0.0, min(float(camera["height"]), float(predicted_box[3])))
        if right <= left or bottom <= top:
            result.update(state="unavailable", reason="depth_support_outside_image")
            return result

        # Sample the interior rather than the hard bbox edge: projection of a
        # rotated cuboid contains background at the corners, whereas interior
        # points retain a useful amount of rendered-target support. Normalised
        # coordinates preserve alignment when a registered depth image is a
        # different resolution from the RGB CameraInfo image.
        sample_pixels = set()
        grid = self.target_eval_depth_sample_grid
        for row in range(grid):
            rgb_y = top + (bottom - top) * (0.20 + 0.60 * row / (grid - 1))
            depth_y = int((rgb_y / float(camera["height"])) * depth_height)
            depth_y = min(depth_height - 1, max(0, depth_y))
            for column in range(grid):
                rgb_x = left + (right - left) * (0.20 + 0.60 * column / (grid - 1))
                depth_x = int((rgb_x / float(camera["width"])) * depth_width)
                depth_x = min(depth_width - 1, max(0, depth_x))
                sample_pixels.add((depth_x, depth_y))
        values = []
        image = depth_frame["image"]
        scale = float(depth_frame["scale"])
        for depth_x, depth_y in sample_pixels:
            value = float(image[depth_y, depth_x]) * scale
            if (
                math.isfinite(value)
                and self.target_eval_min_depth <= value <= self.target_eval_max_depth
            ):
                values.append(value)
        result["sample_pixels"] = len(sample_pixels)
        result["valid_samples"] = len(values)
        if len(values) < self.target_eval_depth_min_valid_samples:
            result.update(state="unavailable", reason="insufficient_valid_depth_samples")
            return result
        values.sort()
        middle = len(values) // 2
        result["observed_depth_median_m"] = (
            values[middle]
            if len(values) % 2
            else 0.5 * (values[middle - 1] + values[middle])
        )
        tolerance = max(
            self.target_eval_depth_abs_tolerance,
            self.target_eval_depth_relative_tolerance * max(expected_max, 0.01),
        )
        lower_bound = expected_min - tolerance
        upper_bound = expected_max + tolerance
        matching = sum(lower_bound <= value <= upper_bound for value in values)
        foreground = sum(value < lower_bound for value in values)
        background = sum(value > upper_bound for value in values)
        result.update(
            available=True,
            matching_samples=matching,
            foreground_samples=foreground,
            background_samples=background,
            tolerance_m=tolerance,
        )
        if matching >= self.target_eval_depth_min_matching_samples:
            result.update(state="visible", visible=True, reason="depth_support_matches_target")
            return result
        occlusion_quorum = max(
            self.target_eval_depth_min_valid_samples,
            int(math.ceil(0.70 * len(values))),
        )
        if foreground >= occlusion_quorum:
            result.update(state="occluded", reason="foreground_depth_occludes_target")
            return result
        result.update(state="inconsistent", reason="depth_support_outside_target_range")
        return result

    def _target_eval_end_episode_locked(self, reason):
        episode = self.target_eval_active_episode
        if episode is None:
            return
        episode["end_source_stamp"] = self.target_eval_last_exposed_stamp
        episode["end_reason"] = reason
        self.target_eval_completed_episodes.append(episode)
        self._write(
            "INFO",
            "target_exposure_ended",
            truth_source="gazebo_evaluation_only",
            episode_id=episode["id"],
            start_source_stamp=round(episode["start_source_stamp"], 4),
            end_source_stamp=(
                None if episode["end_source_stamp"] is None
                else round(episode["end_source_stamp"], 4)
            ),
            exposed_detector_frames=episode["frames"],
            matched_detector_frames=episode["matches"],
            matched=episode["matches"] > 0,
            end_reason=reason,
        )
        self.target_eval_active_episode = None
        self.target_eval_candidate_started_stamp = None

    def _target_eval_update_exposure_locked(self, source_stamp, exposed):
        """Maintain held frustum-exposure episodes on detector source frames."""
        started = False
        if exposed:
            self.target_eval_last_exposed_stamp = source_stamp
            if self.target_eval_active_episode is None:
                if self.target_eval_candidate_started_stamp is None:
                    self.target_eval_candidate_started_stamp = source_stamp
                elif source_stamp - self.target_eval_candidate_started_stamp >= self.target_eval_exposure_hold:
                    self.target_eval_episode_sequence += 1
                    self.target_eval_active_episode = {
                        "id": self.target_eval_episode_sequence,
                        "start_source_stamp": self.target_eval_candidate_started_stamp,
                        "frames": 0,
                        "matches": 0,
                        "first_match_source_stamp": None,
                    }
                    started = True
                    if self.target_eval_first_exposure_stamp is None:
                        self.target_eval_first_exposure_stamp = self.target_eval_candidate_started_stamp
                    self._write(
                        "INFO",
                        "target_exposure_started",
                        truth_source="gazebo_evaluation_only",
                        episode_id=self.target_eval_episode_sequence,
                        source_stamp=round(self.target_eval_candidate_started_stamp, 4),
                        hold_seconds=self.target_eval_exposure_hold,
                    )
        elif self.target_eval_active_episode is not None:
            last_exposed = self.target_eval_last_exposed_stamp
            if last_exposed is not None and source_stamp - last_exposed >= self.target_eval_episode_gap:
                self._target_eval_end_episode_locked("frustum_gap")
        else:
            self.target_eval_candidate_started_stamp = None
        return started

    def _target_eval_end_depth_episode_locked(self, reason):
        """Close one depth-confirmed visible episode without touching legacy data."""
        episode = self.target_eval_depth_active_episode
        if episode is None:
            return
        episode["end_source_stamp"] = self.target_eval_depth_last_visible_stamp
        episode["end_reason"] = reason
        self.target_eval_depth_completed_episodes.append(episode)
        self._write(
            "INFO",
            "target_depth_visible_exposure_ended",
            truth_source="gazebo_depth_evaluation_only",
            episode_id=episode["id"],
            start_source_stamp=round(episode["start_source_stamp"], 4),
            end_source_stamp=(
                None if episode["end_source_stamp"] is None
                else round(episode["end_source_stamp"], 4)
            ),
            depth_visible_detector_frames=episode["frames"],
            depth_visible_matched_detector_frames=episode["matches"],
            matched=episode["matches"] > 0,
            end_reason=reason,
        )
        self.target_eval_depth_active_episode = None
        self.target_eval_depth_candidate_started_stamp = None

    def _target_eval_update_depth_visibility_locked(self, source_stamp, visible):
        """Maintain held *depth-confirmed* visibility episodes separately.

        Frustum episodes above are part of the old metric contract and remain
        unchanged.  This parallel sequence makes an occluded-in-frustum target
        measurable without redefining historical recall values.
        """
        started = False
        if visible:
            self.target_eval_depth_last_visible_stamp = source_stamp
            if self.target_eval_depth_active_episode is None:
                if self.target_eval_depth_candidate_started_stamp is None:
                    self.target_eval_depth_candidate_started_stamp = source_stamp
                elif (
                    source_stamp - self.target_eval_depth_candidate_started_stamp
                    >= self.target_eval_exposure_hold
                ):
                    self.target_eval_depth_episode_sequence += 1
                    self.target_eval_depth_active_episode = {
                        "id": self.target_eval_depth_episode_sequence,
                        "start_source_stamp": self.target_eval_depth_candidate_started_stamp,
                        "frames": 0,
                        "matches": 0,
                        "first_match_source_stamp": None,
                    }
                    started = True
                    if self.target_eval_first_depth_visible_stamp is None:
                        self.target_eval_first_depth_visible_stamp = (
                            self.target_eval_depth_candidate_started_stamp
                        )
                    self._write(
                        "INFO",
                        "target_depth_visible_exposure_started",
                        truth_source="gazebo_depth_evaluation_only",
                        episode_id=self.target_eval_depth_episode_sequence,
                        source_stamp=round(
                            self.target_eval_depth_candidate_started_stamp, 4
                        ),
                        hold_seconds=self.target_eval_exposure_hold,
                    )
        elif self.target_eval_depth_active_episode is not None:
            last_visible = self.target_eval_depth_last_visible_stamp
            if (
                last_visible is not None
                and source_stamp - last_visible >= self.target_eval_episode_gap
            ):
                self._target_eval_end_depth_episode_locked("depth_visibility_gap")
        else:
            self.target_eval_depth_candidate_started_stamp = None
        return started

    def _target_eval_snapshot_locked(self):
        episodes = list(self.target_eval_completed_episodes)
        if self.target_eval_active_episode is not None:
            episodes.append(self.target_eval_active_episode)
        episode_matches = sum(1 for episode in episodes if episode["matches"] > 0)
        depth_episodes = list(self.target_eval_depth_completed_episodes)
        if self.target_eval_depth_active_episode is not None:
            depth_episodes.append(self.target_eval_depth_active_episode)
        depth_episode_matches = sum(
            1 for episode in depth_episodes if episode["matches"] > 0
        )
        first_exposure = self.target_eval_first_exposure_stamp
        first_match = self.target_eval_first_match_stamp
        first_depth_visible = self.target_eval_first_depth_visible_stamp
        first_depth_visible_match = self.target_eval_first_depth_visible_match_stamp
        return {
            "enabled": self.target_eval_enabled,
            "truth_source": "gazebo_evaluation_only",
            "target_model": self.target_eval_target_model,
            "camera_ready": self.target_eval_camera is not None,
            "model_state_ready": bool(self.target_eval_states),
            "exposure_episodes": len(episodes),
            "matched_exposure_episodes": episode_matches,
            "exposed_detector_frames": self.target_eval_exposed_frames,
            "matched_detector_frames": self.target_eval_matched_frames,
            "frame_geometric_recall": (
                None if self.target_eval_exposed_frames == 0 else round(
                    self.target_eval_matched_frames / self.target_eval_exposed_frames, 4
                )
            ),
            "episode_geometric_recall": (
                None if not episodes else round(episode_matches / len(episodes), 4)
            ),
            "depth_validation_enabled": self.target_eval_depth_enabled,
            "depth_topic": self.target_eval_depth_topic or None,
            "depth_frame_ready": bool(self.target_eval_depth_frames),
            "depth_frames_received": self.target_eval_depth_frames_received,
            "depth_frames_decoded": self.target_eval_depth_frames_decoded,
            "depth_decode_failures": self.target_eval_depth_decode_failures,
            "depth_evaluated_frustum_detector_frames": (
                self.target_eval_depth_evaluated_frames
            ),
            "depth_visible_exposure_episodes": len(depth_episodes),
            "depth_matched_visible_exposure_episodes": depth_episode_matches,
            "depth_visible_detector_frames": self.target_eval_depth_visible_frames,
            "depth_visible_matched_detector_frames": (
                self.target_eval_depth_visible_matched_frames
            ),
            "frame_depth_visible_recall": (
                None if self.target_eval_depth_visible_frames == 0 else round(
                    self.target_eval_depth_visible_matched_frames
                    / self.target_eval_depth_visible_frames,
                    4,
                )
            ),
            "episode_depth_visible_recall": (
                None if not depth_episodes else round(
                    depth_episode_matches / len(depth_episodes), 4
                )
            ),
            "depth_occluded_detector_frames": self.target_eval_depth_occluded_frames,
            "depth_inconsistent_detector_frames": (
                self.target_eval_depth_inconsistent_frames
            ),
            "depth_unavailable_detector_frames": (
                self.target_eval_depth_unavailable_frames
            ),
            "unmatched_target_candidate_frames": (
                self.target_eval_unmatched_target_candidate_frames
            ),
            "target_candidates_outside_exposure": self.target_eval_outside_exposure_candidates,
            "unavailable_detector_frames": self.target_eval_unavailable_frames,
            "tf_failures": self.target_eval_tf_failures,
            "evaluated_detector_frames": self.target_eval_evaluated_frames,
            "detector_delivery_latency_mean_seconds": (
                None if self.target_eval_delivery_latency_count == 0 else round(
                    self.target_eval_delivery_latency_total
                    / self.target_eval_delivery_latency_count,
                    4,
                )
            ),
            "detector_delivery_latency_max_seconds": (
                None if self.target_eval_delivery_latency_count == 0
                else round(self.target_eval_delivery_latency_max, 4)
            ),
            "first_exposure_source_stamp": first_exposure,
            "first_match_source_stamp": first_match,
            "first_depth_visible_source_stamp": first_depth_visible,
            "first_depth_visible_match_source_stamp": first_depth_visible_match,
            "first_match_after_exposure_seconds": (
                None if first_exposure is None or first_match is None
                else round(max(0.0, first_match - first_exposure), 4)
            ),
            "first_match_after_depth_visible_seconds": (
                None
                if first_depth_visible is None or first_depth_visible_match is None
                else round(
                    max(0.0, first_depth_visible_match - first_depth_visible), 4
                )
            ),
            "active_episode_id": (
                None if self.target_eval_active_episode is None
                else self.target_eval_active_episode["id"]
            ),
            "active_depth_visible_episode_id": (
                None if self.target_eval_depth_active_episode is None
                else self.target_eval_depth_active_episode["id"]
            ),
        }

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

    @staticmethod
    def _target_session_key(payload):
        """Return the only identity allowed for target lifecycle latency."""
        task_id = str(payload.get("task_id", "")).strip()
        target_track_id = str(payload.get("target_track_id", "")).strip()
        if not task_id or not target_track_id:
            return None
        return (task_id, target_track_id)

    def _record_target_lifecycle_locked(self, event, payload):
        """Persist identity-bound target lifecycle evidence.

        ``/lste/task_done`` is a Bool by design and cannot identify which
        target completed. GoalManager's arbitration stream supplies the task
        and target-track pair, so only these records may feed end-to-end
        target metrics.
        """
        field_by_event = {
            "target_track_started": "first_seen_ros",
            "target_follow_confirmed": "follow_confirmed_ros",
            "target_approach_terminal": "approach_terminal_ros",
            "target_close_confirmation_started": "close_started_ros",
            "target_close_confirmed": "close_confirmed_ros",
            "target_task_completed": "task_completed_ros",
        }
        timestamp_field = field_by_event.get(event)
        if timestamp_field is None:
            return
        key = self._target_session_key(payload)
        if key is None:
            self._write(
                "WARN",
                "target_lifecycle_unattributed",
                lifecycle_event=event,
                task_id=payload.get("task_id"),
                target_track_id=payload.get("target_track_id"),
            )
            return
        now = rospy.Time.now().to_sec()
        session = self.target_lifecycle_sessions.setdefault(
            key,
            {
                "task_id": key[0],
                "target_track_id": key[1],
                "first_seen_ros": None,
                "follow_confirmed_ros": None,
                "approach_terminal_ros": None,
                "close_started_ros": None,
                "close_confirmed_ros": None,
                "task_completed_ros": None,
            },
        )
        if session[timestamp_field] is None:
            session[timestamp_field] = now
        session["last_event_ros"] = now
        if event == "target_task_completed":
            self.target_lifecycle_last_completed_key = key
        record = dict(payload)
        record.pop("event", None)
        record.pop("task_id", None)
        record.pop("target_track_id", None)
        self._write(
            "INFO",
            "target_lifecycle",
            lifecycle_event=event,
            task_id=key[0],
            target_track_id=key[1],
            session_started_seconds=(
                None
                if session["first_seen_ros"] is None
                else round(session["first_seen_ros"] - self.start_ros, 3)
            ),
            event_seconds=round(now - self.start_ros, 3),
            **record
        )

    def on_odom(self, message):
        orientation = message.pose.pose.orientation
        yaw = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )[2]
        position = message.pose.pose.position
        with self.lock:
            self.pose = (position.x, position.y, yaw)
            self.target_eval_odom_pose = {
                "position": (float(position.x), float(position.y), float(position.z)),
                "orientation": (
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                ),
            }
            xy = (position.x, position.y)
            if self.last_pose_xy is not None:
                self.path_length += math.hypot(xy[0] - self.last_pose_xy[0], xy[1] - self.last_pose_xy[1])
            self.last_pose_xy = xy

    def on_pose2d(self, message):
        with self.lock:
            if self.pose is None:
                self.pose = (message.x, message.y, message.theta)

    def on_target_eval_camera_info(self, message):
        """Capture only camera calibration required by the observer."""
        if not self.target_eval_enabled:
            return
        fx, fy = float(message.K[0]), float(message.K[4])
        if message.width <= 0 or message.height <= 0 or fx <= 0.0 or fy <= 0.0:
            return
        frame = self.target_eval_camera_frame or (
            message.header.frame_id or ""
        ).strip().lstrip("/")
        if not frame:
            return
        with self.lock:
            self.target_eval_camera = {
                "frame": frame,
                "width": int(message.width),
                "height": int(message.height),
                "fx": fx,
                "fy": fy,
                "cx": float(message.K[2]),
                "cy": float(message.K[5]),
            }

    def on_gazebo_model_states(self, message):
        """Record a short world/odom calibration history for evaluation only."""
        if not self.target_eval_enabled:
            return
        with self.lock:
            if self.target_eval_odom_pose is None:
                return
            try:
                target_index = message.name.index(self.target_eval_target_model)
                robot_index = message.name.index(self.target_eval_robot_model)
            except ValueError:
                return
            target_pose = message.pose[target_index]
            robot_pose = message.pose[robot_index]
            self.target_eval_states.append({
                "ros_time": rospy.Time.now().to_sec(),
                "wall_time": time.monotonic(),
                "target_world": {
                    "position": (
                        float(target_pose.position.x),
                        float(target_pose.position.y),
                        float(target_pose.position.z),
                    ),
                    "orientation": (
                        float(target_pose.orientation.x),
                        float(target_pose.orientation.y),
                        float(target_pose.orientation.z),
                        float(target_pose.orientation.w),
                    ),
                },
                "robot_world": {
                    "position": (
                        float(robot_pose.position.x),
                        float(robot_pose.position.y),
                        float(robot_pose.position.z),
                    ),
                    "orientation": (
                        float(robot_pose.orientation.x),
                        float(robot_pose.orientation.y),
                        float(robot_pose.orientation.z),
                        float(robot_pose.orientation.w),
                    ),
                },
                "odom": dict(self.target_eval_odom_pose),
            })
            self.target_eval_last_model_wall = self.target_eval_states[-1]["wall_time"]

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
                        transition_kind=self.goal_transition_kind,
                        predecessor_route_id=self.goal_predecessor_route_id,
                        transition_distance_to_previous_endpoint=self.goal_transition_distance,
                    )
                    if self.goal_source == "global_slam_frontier":
                        if self.goal_transition_kind == "prefix_continuation":
                            self.continuous_goal_transitions += 1
                        elif self.goal_transition_kind == "endpoint_divergence":
                            self.divergent_goal_transitions += 1
                        elif self.goal_transition_kind == "terminal_prefetched_successor":
                            self.terminal_goal_transitions += 1
                        else:
                            self.unknown_goal_transitions += 1
                        self.last_goal_transition_wall = time.monotonic()
            self.last_goal_xy = xy
            self.goal = xy
            self.goal_frame = frame
            self.last_goal_frame = frame
            self.goal_message = message
            self.last_goal_publish_wall = time.monotonic()

    def on_mission_goal(self, message):
        """Record the atomic mission-to-execution transaction."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("event") != "mission_goal":
            return
        with self.lock:
            self.mission_goal_messages += 1
            self.goal_source = str(payload.get("source", self.goal_source))
            self.goal_transaction_id = max(
                self.goal_transaction_id,
                int(payload.get("transaction_id", 0) or 0),
            )
            self.goal_transition_kind = str(
                payload.get("transition_kind", "unknown")
            ).strip().lower() or "unknown"
            try:
                self.goal_predecessor_route_id = max(
                    0, int(payload.get("predecessor_route_id", 0) or 0)
                )
            except (TypeError, ValueError):
                self.goal_predecessor_route_id = 0
            try:
                transition_distance = payload.get(
                    "transition_distance_to_previous_endpoint"
                )
                self.goal_transition_distance = (
                    None if transition_distance is None else float(transition_distance)
                )
            except (TypeError, ValueError):
                self.goal_transition_distance = None
            self._write(
                "INFO",
                "mission_goal_transaction",
                transaction_id=self.goal_transaction_id,
                source=self.goal_source,
                priority=int(payload.get("priority", 0) or 0),
                route_id=int(payload.get("route_id", 0) or 0),
                route_kind=str(payload.get("route_kind", "")),
                transition_kind=self.goal_transition_kind,
                predecessor_route_id=self.goal_predecessor_route_id,
                transition_distance_to_previous_endpoint=self.goal_transition_distance,
                frame_id=str(payload.get("frame_id", "")),
                goal=payload.get("goal"),
            )

    def on_teb_cmd(self, message):
        """Track turn-supervisor output before the final safety mux."""
        with self.lock:
            previous = self.teb_command
            self.teb_command = message
            self.teb_cmd_messages += 1
            angular = float(message.angular.z)
            previous_angular = float(previous.angular.z)
            current_linear = float(message.linear.x)
            previous_linear = float(previous.linear.x)
            sign = 1 if angular > 0.05 else -1 if angular < -0.05 else 0
            previous_sign = (
                1 if previous_angular > 0.05
                else -1 if previous_angular < -0.05
                else 0
            )
            if sign and self.teb_last_nonzero_angular_sign and sign != self.teb_last_nonzero_angular_sign:
                self.teb_angular_sign_flips += 1
                if (
                    abs(angular) >= self.teb_strong_angular_threshold
                    and abs(previous_angular) >= self.teb_strong_angular_threshold
                ):
                    self.teb_strong_angular_sign_flips += 1
                self._write(
                    "WARN",
                    "teb_supervisor_angular_sign_flip",
                    angular=round(angular, 4),
                    previous_angular=round(previous_angular, 4),
                    count=self.teb_angular_sign_flips,
                    strong_count=self.teb_strong_angular_sign_flips,
                    goal=None if self.goal is None else [round(value, 3) for value in self.goal],
                    teb_status=self.teb_status,
                )
            if (
                sign
                and previous_sign
                and sign != previous_sign
                and current_linear > self.forward_speed_threshold
                and previous_linear > self.forward_speed_threshold
            ):
                self.teb_forward_steering_sign_flips += 1
                self._write(
                    "WARN",
                    "teb_supervisor_forward_steering_sign_flip",
                    angular=round(angular, 4),
                    previous_angular=round(previous_angular, 4),
                    count=self.teb_forward_steering_sign_flips,
                    current_linear=round(current_linear, 4),
                    previous_linear=round(previous_linear, 4),
                    goal=None if self.goal is None else [round(value, 3) for value in self.goal],
                )
            if sign:
                self.teb_last_nonzero_angular_sign = sign
            if previous_linear > 0.05 and current_linear < previous_linear - 0.08:
                is_stop_brake = current_linear <= 0.05
                if is_stop_brake:
                    self.teb_linear_brake_events += 1
                else:
                    self.teb_speed_modulation_events += 1
                self._write(
                    "WARN" if is_stop_brake else "INFO",
                    (
                        "teb_supervisor_linear_brake"
                        if is_stop_brake else "teb_supervisor_speed_modulation"
                    ),
                    count=(
                        self.teb_linear_brake_events
                        if is_stop_brake
                        else self.teb_speed_modulation_events
                    ),
                    previous_linear=round(previous_linear, 4),
                    current_linear=round(current_linear, 4),
                    angular=round(angular, 4),
                    scan_forward_min=None if not math.isfinite(self.scan_forward_minimum) else round(self.scan_forward_minimum, 4),
                    goal=None if self.goal is None else [round(value, 3) for value in self.goal],
                )

    def on_teb_planner_cmd(self, message):
        """Record TEB's raw command separately from the supervisor output."""
        with self.lock:
            previous = self.teb_planner_command
            self.teb_planner_command = message
            previous_linear = float(previous.linear.x)
            current_linear = float(message.linear.x)
            if previous_linear > 0.05 and current_linear < previous_linear - 0.08:
                is_stop_brake = current_linear <= 0.05
                if is_stop_brake:
                    self.teb_planner_linear_brake_events += 1
                else:
                    self.teb_planner_speed_modulation_events += 1
                self._write(
                    "WARN" if is_stop_brake else "INFO",
                    (
                        "teb_planner_linear_brake"
                        if is_stop_brake else "teb_planner_speed_modulation"
                    ),
                    count=(
                        self.teb_planner_linear_brake_events
                        if is_stop_brake
                        else self.teb_planner_speed_modulation_events
                    ),
                    previous_linear=round(previous_linear, 4),
                    current_linear=round(current_linear, 4),
                    angular=round(float(message.angular.z), 4),
                    scan_forward_min=(
                        None if not math.isfinite(self.scan_forward_minimum)
                        else round(self.scan_forward_minimum, 4)
                    ),
                    goal=(
                        None if self.goal is None
                        else [round(value, 3) for value in self.goal]
                    ),
                )

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
            if event == "trajectory_continuity":
                self.teb_trajectory_continuity_events = max(
                    self.teb_trajectory_continuity_events,
                    int(payload.get("count", 0)),
                )
                # This is a scheduler-quality counter.  A status callback
                # can arrive after an unrelated brake or turn, so it must not
                # rewrite command discontinuity causality by timestamp alone.
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
        with self.lock:
            target = message.goal.target_pose
            self._record_dispatch(
                (float(target.pose.position.x), float(target.pose.position.y)),
                "move_base_action",
            )
            self._record_terminal_to_dispatch_locked(
                time.monotonic(), "move_base_action"
            )

    def on_dispatch(self, message):
        with self.lock:
            self._record_dispatch(
                (float(message.pose.position.x), float(message.pose.position.y)),
                "move_base_simple_goal",
            )
            self._record_terminal_to_dispatch_locked(
                time.monotonic(), "move_base_simple_goal"
            )

    def on_status(self, message):
        with self.lock:
            for status in message.status_list:
                key = (status.goal_id.id, int(status.status))
                if key in self.status_seen:
                    continue
                self.status_seen.add(key)
                if status.goal_id.id:
                    self.move_base_goal_ids.add(status.goal_id.id)
                code = int(status.status)
                name = STATUS_NAMES.get(code, "STATUS_%d" % code)
                self.last_move_base_status = name
                self.lifecycle_event_wall["move_base_%s" % name.lower()] = time.monotonic()
                self.status_counts[name] = self.status_counts.get(name, 0) + 1
                if code == 2:
                    self.preemptions += 1
                    if self.pending_task_done_preemptions > 0:
                        self.pending_task_done_preemptions -= 1
                        self.task_done_preemptions += 1
                    elif self.pending_priority_preemptions > 0:
                        self.pending_priority_preemptions -= 1
                        self.priority_preemptions += 1
                    elif self.pending_target_segment_preemptions > 0:
                        self.pending_target_segment_preemptions -= 1
                        self.target_segment_preemptions += 1
                    elif self.pending_frontier_continuous_prefetch_preemptions > 0:
                        self.pending_frontier_continuous_prefetch_preemptions -= 1
                        self.frontier_continuous_prefetch_preemptions += 1
                    elif self.pending_frontier_segment_preemptions > 0:
                        self.pending_frontier_segment_preemptions -= 1
                        self.frontier_segment_preemptions += 1
                    elif self.pending_frontier_terminal_settle_preemptions > 0:
                        self.pending_frontier_terminal_settle_preemptions -= 1
                        self.frontier_terminal_settle_preemptions += 1
                    elif self.pending_frontier_observation_preemptions > 0:
                        self.pending_frontier_observation_preemptions -= 1
                        self.frontier_observation_preemptions += 1
                    else:
                        self.unexpected_preemptions += 1
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
                    frontier_observation_preemptions=(
                        self.frontier_observation_preemptions
                    ),
                    frontier_continuous_prefetch_preemptions=(
                        self.frontier_continuous_prefetch_preemptions
                    ),
                    frontier_segment_preemptions=self.frontier_segment_preemptions,
                    priority_preemptions=self.priority_preemptions,
                    task_done_preemptions=self.task_done_preemptions,
                    unexpected_preemptions=self.unexpected_preemptions,
                    aborts=self.aborts,
                )
                if code == 3:
                    self._mark_action_terminal_locked(
                        time.monotonic(), "move_base_succeeded"
                    )
                    self._reclassify_recent_discontinuities_locked(
                        time.monotonic(),
                        "move_base_succeeded",
                    )

    def _remember_discontinuity_locked(self, kind, reason, now):
        """Retain a provisional brake/stop cause for terminal correlation."""
        self.discontinuity_sequence += 1
        record = {
            "id": int(self.discontinuity_sequence),
            "kind": str(kind),
            "reason": str(reason),
            "wall": float(now),
        }
        self.recent_discontinuities.append(record)
        cutoff = now - 2.0
        while (
            self.recent_discontinuities
            and self.recent_discontinuities[0]["wall"] < cutoff
        ):
            self.recent_discontinuities.popleft()
        return record

    def _mark_action_terminal_locked(self, now, source):
        """Start one terminal-to-successor timing interval.

        The bridge terminal callback and move_base SUCCEEDED status normally
        arrive within one scheduler tick of each other. Preserve the first
        timestamp, rather than treating their duplicate reports as two action
        boundaries.
        """
        if (
            self.pending_action_terminal_wall is None
            or now - self.pending_action_terminal_wall > 0.75
        ):
            self.pending_action_terminal_wall = now
            self.pending_action_terminal_source = str(source)

    def _record_terminal_to_dispatch_locked(self, now, transport):
        """Record one successful-action handoff, excluding unrelated goals."""
        if self.pending_action_terminal_wall is None:
            return
        delay = now - self.pending_action_terminal_wall
        # A later manual/new mission dispatch is not the successor of this
        # terminal action. Five seconds already exceeds the expected online
        # frontier planning cycle by a wide margin.
        if delay < 0.0 or delay > 5.0:
            self.pending_action_terminal_wall = None
            self.pending_action_terminal_source = ""
            return
        self.terminal_to_dispatch_count += 1
        self.terminal_to_dispatch_total += delay
        self.terminal_to_dispatch_max = max(self.terminal_to_dispatch_max, delay)
        self.terminal_to_dispatch_last = delay
        self._write(
            "INFO",
            "terminal_to_dispatch",
            count=self.terminal_to_dispatch_count,
            delay_seconds=round(delay, 4),
            terminal_source=self.pending_action_terminal_source,
            transport=str(transport),
        )
        self.pending_action_terminal_wall = None
        self.pending_action_terminal_source = ""

    def _reclassify_recent_discontinuities_locked(
        self, now, lifecycle_event, replacement_reason="action_terminal", window=0.75
    ):
        """Correct command-first observations once a terminal status arrives.

        ROS publishes the final zero command and action status from different
        callbacks. Treating their delivery order as physical causality made a
        normal endpoint deceleration look like an unexplained safety brake.
        This only changes telemetry counters; it never affects navigation.
        """
        for record in self.recent_discontinuities:
            delay = now - record["wall"]
            if delay < 0.0 or delay > window:
                continue
            previous = record["reason"]
            if previous == replacement_reason:
                continue
            # A late supervisor feedback message can follow the successful
            # move_base terminal for the *same* command sample.  Arrival is a
            # stronger lifecycle fact than the optional continuity adapter;
            # never rewrite a confirmed endpoint stop into a continuity gap.
            if (
                replacement_reason == "trajectory_continuity"
                and previous == "action_terminal"
            ):
                continue
            counts = (
                self.brake_reason_counts
                if record["kind"] == "linear_brake"
                else self.stop_reason_counts
            )
            if counts.get(previous, 0) > 0:
                counts[previous] -= 1
                if counts[previous] == 0:
                    del counts[previous]
            self._increment_reason(counts, replacement_reason)
            record["reason"] = replacement_reason
            self._write(
                "INFO",
                "discontinuity_reclassified",
                discontinuity_id=int(record["id"]),
                kind=record["kind"],
                previous_reason=previous,
                reason=replacement_reason,
                lifecycle_event=lifecycle_event,
                delay_seconds=round(delay, 4),
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
        if self.navigation_hold:
            return "navigation_hold", None
        mux_status = self.cmd_vel_mux_status
        if isinstance(mux_status, dict):
            block_reason = str(mux_status.get("block_reason", "none"))
            if block_reason == "navigation_hold":
                return "navigation_hold", None
            if block_reason == "task_complete":
                return "task_complete", None
            output = mux_status.get("output")
            input_command = mux_status.get("input")
            if isinstance(output, dict) and isinstance(input_command, dict):
                try:
                    output_linear = abs(float(output.get("linear_x", 0.0)))
                    input_linear = float(input_command.get("linear_x", 0.0))
                except (TypeError, ValueError):
                    output_linear = float("inf")
                    input_linear = 0.0
                if (
                    str(mux_status.get("governor_reason", ""))
                    == "governor_limited"
                    and input_linear > 0.05
                    and output_linear <= 0.05
                ):
                    return "mux_governor_stop", None
        event, age = self._recent_lifecycle_event(
            self.lifecycle_event_wall,
            ("turn_turn_started", "turn_turning", "turn_turn_completed"),
            now,
        )
        if event is not None:
            return "explicit_turn", age
        # TEB may deliberately rotate in place to acquire a newly exposed
        # Navfn route tangent even when the turn supervisor did not create a
        # dedicated connector phase.  The raw mux command is then a zero
        # linear velocity, but its selected trajectory explicitly requests
        # angular motion.  This is a route-geometry reorientation, not a
        # clear-path safety brake.
        selected_velocity = None
        if isinstance(self.teb_feedback_state, dict):
            selected_velocity = self.teb_feedback_state.get("selected_velocity")
        if isinstance(selected_velocity, dict):
            try:
                selected_linear = abs(float(selected_velocity.get("linear_x", 0.0)))
                selected_angular = abs(float(selected_velocity.get("angular_z", 0.0)))
            except (TypeError, ValueError):
                selected_linear = float("inf")
                selected_angular = 0.0
            if selected_linear <= 0.05 and selected_angular >= 0.10:
                return "teb_reorientation", None
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
            ("move_base_recovery",),
            now,
            window=2.0,
        )
        if event is not None:
            return "planner_recovery", age
        event, age = self._recent_lifecycle_event(
            self.lifecycle_event_wall,
            ("dispatch",),
            now,
            window=0.45,
        )
        if event is not None:
            return "action_dispatch", age
        terminal_reorientation = self.pending_terminal_native_reorientation
        if (
            isinstance(terminal_reorientation, dict)
            and self.goal_source == "global_slam_frontier"
            and self.goal is not None
            and terminal_reorientation.get("goal") is not None
            and now - float(terminal_reorientation.get("armed_wall", now)) <= 30.0
        ):
            terminal_goal = terminal_reorientation["goal"]
            if math.hypot(
                float(self.goal[0]) - float(terminal_goal[0]),
                float(self.goal[1]) - float(terminal_goal[1]),
            ) <= 0.15:
                pose_for_goal = self._pose_xy_in_frame_locked(self.goal_frame)
                if pose_for_goal is not None:
                    remaining = math.hypot(
                        float(self.goal[0]) - float(pose_for_goal[0]),
                        float(self.goal[1]) - float(pose_for_goal[1]),
                    )
                    if remaining <= 1.20:
                        return "terminal_native_teb_reorientation", remaining
        # Gmapping/costmap updates can make the raw TEB command publisher
        # emit one zero sample while the selected TEB trajectory remains
        # forward. This is a sub-control-cycle synchronization gap, not an
        # obstacle brake. Require a fresh selected trajectory, an active
        # action, and sufficient distance from the terminal envelope so a
        # late feedback sample can never hide a genuine endpoint stop.
        selected_velocity = None
        if isinstance(self.teb_feedback_state, dict):
            selected_velocity = self.teb_feedback_state.get("selected_velocity")
        feedback_age = (
            None
            if self.teb_feedback_wall is None
            else max(0.0, now - self.teb_feedback_wall)
        )
        pose_for_goal = self._pose_xy_in_frame_locked(self.goal_frame)
        goal_distance = None
        if pose_for_goal is not None and self.goal is not None:
            goal_distance = math.hypot(
                float(self.goal[0]) - float(pose_for_goal[0]),
                float(self.goal[1]) - float(pose_for_goal[1]),
            )
        if isinstance(selected_velocity, dict):
            try:
                selected_forward = float(selected_velocity.get("linear_x", 0.0))
            except (TypeError, ValueError):
                selected_forward = 0.0
            if (
                self.bridge_active
                and self.teb_status == "trajectory_valid"
                and feedback_age is not None
                and feedback_age <= self.teb_control_cycle_gap_max_feedback_age
                and selected_forward >= self.forward_speed_threshold
                and goal_distance is not None
                and goal_distance >= self.teb_control_cycle_gap_min_goal_distance
            ):
                return "teb_control_cycle_gap", feedback_age
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
            straight_path_state = self._straight_path_state_locked(now_wall)
            if self.last_cmd_wall is not None:
                dt = max(0.0, now_wall - self.last_cmd_wall)
                if current_linear > self.forward_speed_threshold:
                    self.forward_distance += current_linear * dt
                    self.forward_angular_energy += abs(angular) * dt
                    if straight_path_state["is_straight"]:
                        self.straight_path_distance += current_linear * dt
                        self.straight_path_angular_energy += abs(angular) * dt
                        self.straight_path_samples += 1
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
            if (
                sign
                and previous_sign
                and sign != previous_sign
                and current_linear > self.forward_speed_threshold
                and previous_linear > self.forward_speed_threshold
            ):
                self.forward_steering_sign_flips += 1
                self._write(
                    "WARN",
                    "forward_steering_sign_flip",
                    angular=round(angular, 4),
                    previous_angular=round(previous_angular, 4),
                    count=self.forward_steering_sign_flips,
                    current_linear=round(current_linear, 4),
                    previous_linear=round(previous_linear, 4),
                    goal=None if self.goal is None else [round(value, 3) for value in self.goal],
                )
            if not straight_path_state["is_straight"]:
                # A direction change across a bend is intentional route
                # tracking, never evidence of corridor wobble.
                self.straight_path_last_nonzero_sign = 0
                self.straight_path_last_nonzero_sign_wall = 0.0
            elif sign and current_linear > self.forward_speed_threshold:
                prior_sign = self.straight_path_last_nonzero_sign
                prior_age = now_wall - self.straight_path_last_nonzero_sign_wall
                if (
                    prior_sign
                    and prior_sign != sign
                    and prior_age <= self.straight_path_sign_flip_window
                ):
                    self.straight_path_steering_sign_flips += 1
                    self._write(
                        "WARN",
                        "straight_path_wobble_sign_flip",
                        angular=round(angular, 4),
                        previous_angular=round(previous_angular, 4),
                        count=self.straight_path_steering_sign_flips,
                        previous_straight_sign=prior_sign,
                        sign_gap_seconds=round(prior_age, 4),
                        current_linear=round(current_linear, 4),
                        plan_curvature_rad=straight_path_state["plan_curvature_rad"],
                        heading_error_rad=straight_path_state["heading_error_rad"],
                        local_plan_curvature_rad=straight_path_state[
                            "local_plan_curvature_rad"
                        ],
                        local_heading_error_rad=straight_path_state[
                            "local_heading_error_rad"
                        ],
                        goal=None if self.goal is None else [
                            round(value, 3) for value in self.goal
                        ],
                    )
                self.straight_path_last_nonzero_sign = sign
                self.straight_path_last_nonzero_sign_wall = now_wall
            if sign:
                self.last_nonzero_angular_sign = sign
            if previous_linear > 0.05 and current_linear < previous_linear - 0.08:
                is_stop_brake = current_linear <= 0.05
                if not is_stop_brake:
                    self.speed_modulation_events += 1
                    self._write(
                        "INFO",
                        "speed_modulation",
                        count=self.speed_modulation_events,
                        previous_linear=round(previous_linear, 4),
                        current_linear=round(current_linear, 4),
                        angular=round(angular, 4),
                        delta=round(current_linear - previous_linear, 4),
                        scan_forward_min=(
                            None
                            if not math.isfinite(self.scan_forward_minimum)
                            else round(self.scan_forward_minimum, 4)
                        ),
                        pose=(
                            None
                            if self.pose is None
                            else [round(value, 3) for value in self.pose]
                        ),
                        goal=(
                            None
                            if self.goal is None
                            else [round(value, 3) for value in self.goal]
                        ),
                    )
                else:
                    self.linear_brake_events += 1
                    self.hard_stop_events += 1
                if is_stop_brake:
                    forward_clear = self.scan_forward_minimum
                    if not math.isfinite(forward_clear):
                        self.brake_events_unknown_clearance += 1
                    elif forward_clear >= 0.60:
                        self.brake_events_clear += 1
                    else:
                        self.brake_events_near += 1
                    now = time.monotonic()
                    transition_age = (
                        None if self.last_goal_transition_wall is None
                        else max(0.0, now - self.last_goal_transition_wall)
                    )
                    transition_kind = (
                        None
                        if transition_age is None or transition_age > 0.50
                        else self.goal_transition_kind
                    )
                    if transition_kind is not None:
                        self.transition_brake_events[transition_kind] += 1
                    brake_reason, lifecycle_age = self._command_discontinuity_reason_locked(now)
                    self._increment_reason(self.brake_reason_counts, brake_reason)
                    discontinuity = self._remember_discontinuity_locked(
                        "linear_brake", brake_reason, now
                    )
                    if now - self.last_brake_wall >= 0.15:
                        self.last_brake_wall = now
                        self._write(
                            "WARN",
                            "linear_brake",
                            count=self.linear_brake_events,
                            discontinuity_id=int(discontinuity["id"]),
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
                            route_transition_kind=transition_kind,
                            route_transition_age_seconds=(
                                None if transition_kind is None else round(transition_age, 4)
                            ),
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
                discontinuity = self._remember_discontinuity_locked(
                    "command_stop", stop_reason, self.zero_start_wall
                )
                self._write(
                    "WARN",
                    "command_stop",
                    count=self.stop_events,
                    discontinuity_id=int(discontinuity["id"]),
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

    def on_cmd_vel_mux_status(self, message):
        """Record the controller/safety decision that produced /cmd_vel.

        This callback never changes a command. It only gives the metrics
        layer direct causality for final actuator limits and can revise the
        preceding /cmd_vel discontinuity when ROS delivers the two topics in
        opposite callback order.
        """
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self.lock:
            now = time.monotonic()
            self.cmd_vel_mux_status = payload
            self.cmd_vel_mux_status_wall = now
            block_reason = str(payload.get("block_reason", "none"))
            governor_reason = str(payload.get("governor_reason", "none"))
            output = payload.get("output")
            input_command = payload.get("input")
            output_linear = None
            input_linear = None
            if isinstance(output, dict):
                try:
                    output_linear = float(output.get("linear_x", 0.0))
                except (TypeError, ValueError):
                    pass
            if isinstance(input_command, dict):
                try:
                    input_linear = float(input_command.get("linear_x", 0.0))
                except (TypeError, ValueError):
                    pass
            reason = "none"
            if block_reason in ("navigation_hold", "task_complete"):
                reason = block_reason
                self.mux_forced_zero_events += 1
                self.lifecycle_event_wall["mux_" + reason] = now
            elif (
                governor_reason == "governor_limited"
                and input_linear is not None
                and input_linear > 0.05
            ):
                reason = "mux_governor_stop" if (
                    output_linear is not None and abs(output_linear) <= 0.05
                ) else "mux_governor_limited"
                self.mux_governor_limited_events += 1
                self.lifecycle_event_wall[reason] = now
            self._increment_reason(self.mux_status_reason_counts, reason)
            if reason != "none":
                self._reclassify_recent_discontinuities_locked(
                    now,
                    "cmd_vel_mux_status",
                    replacement_reason=reason,
                    window=0.35,
                )
                self._write(
                    "WARN" if reason != "mux_governor_limited" else "INFO",
                    "cmd_vel_mux_intervention",
                    reason=reason,
                    source=payload.get("source"),
                    selected_mode=payload.get("selected_mode"),
                    input=input_command,
                    output=output,
                    governor_linear_limit=payload.get("governor_linear_limit"),
                    scan_forward_min=payload.get("scan_forward_min"),
                    scan_age_seconds=payload.get("scan_age_seconds"),
                    filter_reason=payload.get("filter_reason"),
                )

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

    @staticmethod
    def _point_segment_distance(px, py, ax, ay, bx, by):
        """Return Euclidean distance from one 2-D point to a finite segment."""
        dx = bx - ax
        dy = by - ay
        denominator = dx * dx + dy * dy
        if denominator <= 1e-12:
            return math.hypot(px - ax, py - ay)
        ratio = ((px - ax) * dx + (py - ay) * dy) / denominator
        ratio = max(0.0, min(1.0, ratio))
        return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))

    def _teb_obstacle_snapshot_locked(self, message):
        """Summarize nearby converter obstacles in the feedback frame.

        Feedback obstacle polygons use the same frame as the trajectory. The
        wheel odometry pose is normally ``odom`` too; retain the frame in the
        event so a future frame mismatch is explicit instead of silently
        becoming a misleading distance.
        """
        pose = self.pose
        if pose is None:
            return {"frame": message.header.frame_id or None, "nearest": None}
        px, py = float(pose[0]), float(pose[1])
        nearest = None
        summaries = []
        for obstacle in list(message.obstacles_msg.obstacles):
            points = list(obstacle.polygon.points)
            boundary_distance = float("inf")
            if len(points) == 1:
                boundary_distance = math.hypot(
                    px - float(points[0].x), py - float(points[0].y)
                )
            elif len(points) >= 2:
                for first, second in zip(points, points[1:] + points[:1]):
                    boundary_distance = min(
                        boundary_distance,
                        self._point_segment_distance(
                            px,
                            py,
                            float(first.x),
                            float(first.y),
                            float(second.x),
                            float(second.y),
                        ),
                    )
            effective_distance = max(0.0, boundary_distance - float(obstacle.radius))
            xs = [float(point.x) for point in points]
            ys = [float(point.y) for point in points]
            summary = {
                "id": int(obstacle.id),
                "points": len(points),
                "radius": round(float(obstacle.radius), 4),
                "boundary_distance": (
                    None if not math.isfinite(boundary_distance)
                    else round(boundary_distance, 4)
                ),
                "effective_distance": (
                    None if not math.isfinite(effective_distance)
                    else round(effective_distance, 4)
                ),
                "bounds": (
                    None if not xs else [
                        round(min(xs), 3), round(min(ys), 3),
                        round(max(xs), 3), round(max(ys), 3),
                    ]
                ),
            }
            summaries.append(summary)
            if math.isfinite(effective_distance) and (
                nearest is None
                or effective_distance < nearest["effective_distance"]
            ):
                nearest = summary
        summaries.sort(
            key=lambda item: float("inf")
            if item["effective_distance"] is None else item["effective_distance"]
        )
        return {
            "frame": message.header.frame_id or None,
            "nearest": nearest,
            "nearest_three": summaries[:3],
        }

    def on_teb_feedback(self, message):
        with self.lock:
            now = time.monotonic()
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
            self.teb_feedback_wall = now
            if selected is None:
                self.teb_status = "no_selected_trajectory"
            elif first is None:
                self.teb_status = "selected_trajectory_empty"
            elif abs(float(first.velocity.linear.x)) <= 0.002 and abs(float(first.velocity.angular.z)) <= 0.01:
                self.teb_status = "selected_command_near_zero"
            else:
                self.teb_status = "trajectory_valid"
            near_zero_linear = (
                first is not None
                and abs(float(first.velocity.linear.x)) <= 0.01
            )
            turn_in_progress = (
                isinstance(self.teb_turn_supervisor_status, dict)
                and str(
                    self.teb_turn_supervisor_status.get("state", "")
                ).strip().upper() == "TURNING"
            )
            clear_forward = (
                math.isfinite(self.scan_forward_minimum)
                and self.scan_forward_minimum > self.discontinuity_obstacle_clearance
            )
            if (
                self.bridge_active
                and not turn_in_progress
                and near_zero_linear
                and clear_forward
            ):
                if self.teb_zero_velocity_start_wall is None:
                    self.teb_zero_velocity_start_wall = now
                plateau_seconds = now - self.teb_zero_velocity_start_wall
                if (
                    plateau_seconds >= self.teb_zero_velocity_snapshot_min_duration
                    and self.teb_zero_velocity_snapshot_wall
                    < self.teb_zero_velocity_start_wall
                ):
                    self.teb_zero_velocity_snapshot_wall = now
                    selected_start = None
                    selected_endpoint = None
                    if selected is not None and selected.trajectory:
                        selected_start = selected.trajectory[0].pose.position
                        selected_endpoint = selected.trajectory[-1].pose.position
                    self._write(
                        "WARN",
                        "teb_zero_velocity_snapshot",
                        plateau_seconds=round(plateau_seconds, 3),
                        pose=None if self.pose is None else [
                            round(float(value), 4) for value in self.pose
                        ],
                        goal=None if self.goal is None else [
                            round(float(value), 4) for value in self.goal
                        ],
                        goal_frame=self.goal_frame,
                        scan_forward_min=round(
                            float(self.scan_forward_minimum), 4
                        ),
                        selected_velocity=selected_velocity,
                        selected_start=(
                            None if selected_start is None else [
                                round(float(selected_start.x), 4),
                                round(float(selected_start.y), 4),
                            ]
                        ),
                        selected_endpoint=(
                            None if selected_endpoint is None else [
                                round(float(selected_endpoint.x), 4),
                                round(float(selected_endpoint.y), 4),
                            ]
                        ),
                        teb_local_plan=self.teb_local_plan_stats,
                        teb_global_plan=self.teb_global_plan_stats,
                        obstacles=self._teb_obstacle_snapshot_locked(message),
                    )
            else:
                self.teb_zero_velocity_start_wall = None
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
                now = time.monotonic()
                self.move_base_recovery_events += 1
                self.lifecycle_event_wall["move_base_recovery"] = now
                # Recovery status often arrives just after TEB has emitted its
                # zero command. Reclassify that command as a planner recovery
                # rather than falsely attributing it to a clear-path brake.
                self._reclassify_recent_discontinuities_locked(
                    now,
                    "move_base_recovery",
                    replacement_reason="planner_recovery",
                    window=2.0,
                )
                self._write("WARN", "move_base_recovery", **current)

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(float(angle)), math.cos(float(angle)))

    @staticmethod
    def _path_geometry(message):
        """Return the latest path in its native frame for observer-only math."""
        return {
            "frame": (message.header.frame_id or "odom").strip().lstrip("/") or "odom",
            "points": [
                (float(pose.pose.position.x), float(pose.pose.position.y))
                for pose in message.poses
            ],
        }

    def _path_straightness_locked(self, geometry, label):
        """Evaluate one TEB path without using it to control the vehicle."""
        state = {
            "is_straight": False,
            "reason": "no_%s" % label,
            "plan_curvature_rad": None,
            "heading_error_rad": None,
            "nearest_path_distance_m": None,
            "frame": None,
        }
        if not geometry or len(geometry.get("points", ())) < 2:
            return state
        frame = geometry["frame"]
        pose = self._pose_xy_in_frame_locked(frame)
        if pose is None:
            state.update(reason="pose_transform_unavailable", frame=frame)
            return state
        points = geometry["points"]
        nearest_index = min(
            range(len(points)),
            key=lambda index: (points[index][0] - pose[0]) ** 2
            + (points[index][1] - pose[1]) ** 2,
        )
        nearest_distance = math.hypot(
            points[nearest_index][0] - pose[0],
            points[nearest_index][1] - pose[1],
        )
        start_index = nearest_index
        while start_index + 1 < len(points) and math.hypot(
            points[start_index + 1][0] - points[start_index][0],
            points[start_index + 1][1] - points[start_index][1],
        ) <= 1e-4:
            start_index += 1
        if start_index + 1 >= len(points):
            state.update(
                reason="path_terminal",
                nearest_path_distance_m=round(nearest_distance, 4),
                frame=frame,
            )
            return state
        first = points[start_index]
        second = points[start_index + 1]
        initial_heading = math.atan2(second[1] - first[1], second[0] - first[0])
        travelled = math.hypot(second[0] - first[0], second[1] - first[1])
        horizon_index = start_index + 1
        while (
            horizon_index + 1 < len(points)
            and travelled < self.straight_path_lookahead
        ):
            previous = points[horizon_index]
            horizon_index += 1
            current = points[horizon_index]
            travelled += math.hypot(current[0] - previous[0], current[1] - previous[1])
        horizon_previous = points[max(start_index, horizon_index - 1)]
        horizon = points[horizon_index]
        horizon_heading = math.atan2(
            horizon[1] - horizon_previous[1],
            horizon[0] - horizon_previous[0],
        )
        curvature = abs(self._normalize_angle(horizon_heading - initial_heading))
        heading_error = abs(self._normalize_angle(initial_heading - pose[2]))
        aligned = (
            curvature <= self.straight_path_max_curvature
            and heading_error <= self.straight_path_max_heading_error
        )
        state.update(
            is_straight=aligned,
            reason=(
                "straight_aligned"
                if aligned
                else "planned_bend"
                if curvature > self.straight_path_max_curvature
                else "heading_alignment"
            ),
            plan_curvature_rad=round(curvature, 4),
            heading_error_rad=round(heading_error, 4),
            nearest_path_distance_m=round(nearest_distance, 4),
            frame=frame,
        )
        return state

    def _straight_path_state_locked(self, now):
        """Classify a settled straight segment in both TEB path horizons.

        Navfn's global route can become straight one cycle before TEB has
        finished a turn in its selected local trajectory. Counting that tail
        as corridor wobble creates a false regression. A command therefore
        qualifies only when the global route *and* the current local TEB path
        are straight and aligned with the base.
        """
        if (
            self.straight_path_state is not None
            and now - self.straight_path_last_eval_wall < self.straight_path_eval_period
        ):
            return self.straight_path_state
        self.straight_path_last_eval_wall = now
        global_state = self._path_straightness_locked(
            self.teb_global_plan_geometry, "teb_global_plan"
        )
        local_state = self._path_straightness_locked(
            self.teb_local_plan_geometry, "teb_local_plan"
        )
        is_straight = (
            bool(global_state["is_straight"])
            and bool(local_state["is_straight"])
        )
        state = {
            "is_straight": is_straight,
            "reason": (
                "straight_aligned"
                if is_straight
                else "global_%s" % global_state["reason"]
                if not global_state["is_straight"]
                else "local_%s" % local_state["reason"]
            ),
            # Preserve the historical global keys for existing log readers.
            "plan_curvature_rad": global_state["plan_curvature_rad"],
            "heading_error_rad": global_state["heading_error_rad"],
            "nearest_path_distance_m": global_state["nearest_path_distance_m"],
            "frame": global_state["frame"],
            "local_plan_curvature_rad": local_state["plan_curvature_rad"],
            "local_heading_error_rad": local_state["heading_error_rad"],
            "local_nearest_path_distance_m": local_state["nearest_path_distance_m"],
            "local_frame": local_state["frame"],
        }
        self.straight_path_state = state
        return state

    def _on_path(self, source, message):
        with self.lock:
            stats = self._path_stats(message)
            setattr(self, source, stats)
            if source == "teb_global_plan_stats":
                self.teb_global_plan_geometry = self._path_geometry(message)
            elif source == "teb_local_plan_stats":
                self.teb_local_plan_geometry = self._path_geometry(message)
            now = time.monotonic()
            previous = self.last_plan_log_wall.get(source, 0.0)
            if now - previous >= 1.0:
                self.last_plan_log_wall[source] = now
                self._write("INFO", "planner_path", planner=source, **stats)

    def on_navfn_plan(self, message):
        self._on_path("navfn_plan_stats", message)

    def on_persistent_plan_event(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self.lock:
            self.persistent_plan_received = max(
                self.persistent_plan_received,
                int(payload.get("received", 0) or 0),
            )
            self.persistent_plan_equivalent_retained = max(
                self.persistent_plan_equivalent_retained,
                int(payload.get("equivalent_retained", 0) or 0),
            )
            self.persistent_plan_installed = max(
                self.persistent_plan_installed,
                int(payload.get("installed", 0) or 0),
            )
            self.persistent_plan_last_event = (
                str(payload.get("event", "unknown")).strip() or "unknown"
            )
            try:
                self.persistent_plan_route_version = max(
                    self.persistent_plan_route_version,
                    int(payload.get("route_version", 0) or 0),
                )
            except (TypeError, ValueError):
                pass
            geometry_hash = str(payload.get("geometry_hash", "")).strip()
            if geometry_hash:
                self.persistent_plan_geometry_hash = geometry_hash
            self._write(
                "INFO",
                "persistent_plan_event",
                event_name=self.persistent_plan_last_event,
                received=self.persistent_plan_received,
                equivalent_retained=self.persistent_plan_equivalent_retained,
                installed=self.persistent_plan_installed,
                route_version=self.persistent_plan_route_version,
                geometry_hash=self.persistent_plan_geometry_hash,
                poses=int(payload.get("poses", 0) or 0),
                goal=payload.get("goal"),
            )

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
                if int(message.state) == 2 and self.legacy_state_locked_ros is None:
                    self.legacy_state_locked_ros = rospy.Time.now().to_sec()
                    self._write(
                        "INFO",
                        "legacy_state_locked",
                        latency_seconds=round(
                            self.legacy_state_locked_ros - self.start_ros, 3
                        ),
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
            self._record_target_lifecycle_locked(event, payload)
            if event == "target_follow_confirmed":
                if self.target_follow_confirmed_ros is None:
                    self.target_follow_confirmed_ros = rospy.Time.now().to_sec()
                    # ``target_lock`` was previously inferred from the legacy
                    # LSTE state value 2. TEB target following has its own
                    # evidence contract, so retain this field for consumers
                    # while giving it the correct mission-level meaning.
                    self.target_lock_ros = self.target_follow_confirmed_ros
                    self._write(
                        "INFO",
                        "target_follow_confirmed",
                        latency_seconds=round(
                            self.target_follow_confirmed_ros - self.start_ros, 3
                        ),
                        target_track_id=payload.get("target_track_id"),
                        hits=payload.get("hits"),
                        average_score=payload.get("average_score"),
                    )
            elif event == "target_close_confirmation_started":
                if self.target_close_confirmation_started_ros is None:
                    self.target_close_confirmation_started_ros = rospy.Time.now().to_sec()
            elif event == "target_close_confirmed":
                if self.target_close_confirmed_ros is None:
                    self.target_close_confirmed_ros = rospy.Time.now().to_sec()
                    self._write(
                        "INFO",
                        "target_close_confirmed",
                        latency_seconds=round(
                            self.target_close_confirmed_ros - self.start_ros, 3
                        ),
                        target_track_id=payload.get("target_track_id"),
                        hits=payload.get("hits"),
                        hold_seconds=payload.get("hold_seconds"),
                    )
            elif event == "target_segment_committed":
                self.target_segments_committed += 1
            elif event == "target_continuous_handoff_prepared":
                self.target_continuous_handoffs_prepared += 1
            elif event == "target_route_accepted":
                self.target_route_accepts += 1
            elif event == "target_route_rejected":
                self.target_route_rejections += 1
            elif event == "target_route_deferred":
                self.target_route_deferrals += 1
            elif event == "target_route_held":
                self.target_route_holds += 1
            elif event == "target_route_semantic_replan":
                self.target_route_semantic_replans += 1
            elif event == "target_route_released":
                self.target_route_releases += 1
            elif event == "target_approach_terminal":
                self.target_approach_terminals += 1
            record = dict(payload)
            record.pop("event", None)
            self._write("INFO" if event != "target_route_rejected" else "WARN",
                        "goal_arbitration", goal_event=event, **record)

    def _evaluate_target_detection_locked(self, message):
        """Measure geometric detector recall without influencing any ROS control path."""
        if not self.target_eval_enabled:
            return
        source_stamp = float(message.header.stamp.to_sec())
        if source_stamp <= 0.0:
            self._target_eval_note_unavailable_locked("detection_stamp_unavailable")
            return
        if (
            self.target_eval_last_source_stamp is not None
            and source_stamp <= self.target_eval_last_source_stamp
        ):
            return
        self.target_eval_last_source_stamp = source_stamp
        if not self.target_eval_target_model:
            self._target_eval_note_unavailable_locked("target_eval_configuration_incomplete")
            return
        if not self.target_eval_states:
            self._target_eval_note_unavailable_locked("gazebo_model_state_unavailable")
            return
        now_wall = time.monotonic()
        newest_age = now_wall - self.target_eval_states[-1]["wall_time"]
        if newest_age > self.target_eval_max_state_age:
            self._target_eval_note_unavailable_locked(
                "gazebo_model_state_stale", model_state_rx_age_seconds=round(newest_age, 4)
            )
            return
        state = min(
            self.target_eval_states,
            key=lambda item: abs(float(item["ros_time"]) - source_stamp),
        )
        source_state_age = abs(float(state["ros_time"]) - source_stamp)
        if source_state_age > self.target_eval_max_state_age:
            self._target_eval_note_unavailable_locked(
                "gazebo_model_state_timestamp_mismatch",
                source_to_state_age_seconds=round(source_state_age, 4),
            )
            return
        projection, unavailable_reason = self._target_eval_projected_box_locked(
            source_stamp, state
        )
        if unavailable_reason is not None:
            self._target_eval_note_unavailable_locked(unavailable_reason)
            return

        self.target_eval_evaluated_frames += 1
        delivery_latency = max(0.0, rospy.Time.now().to_sec() - source_stamp)
        self.target_eval_delivery_latency_total += delivery_latency
        self.target_eval_delivery_latency_max = max(
            self.target_eval_delivery_latency_max, delivery_latency
        )
        self.target_eval_delivery_latency_count += 1
        exposed = bool(projection["exposed"])
        started_episode = self._target_eval_update_exposure_locked(source_stamp, exposed)
        depth_visibility = self._target_eval_depth_visibility_locked(
            source_stamp, projection
        )
        depth_visible = bool(depth_visibility["visible"])
        started_depth_episode = self._target_eval_update_depth_visibility_locked(
            source_stamp, depth_visible
        )
        if exposed:
            if depth_visibility["available"]:
                self.target_eval_depth_evaluated_frames += 1
                if depth_visibility["state"] == "occluded":
                    self.target_eval_depth_occluded_frames += 1
                elif depth_visibility["state"] == "inconsistent":
                    self.target_eval_depth_inconsistent_frames += 1
            elif depth_visibility["state"] == "unavailable":
                self.target_eval_depth_unavailable_frames += 1

        task_id = str(message.task_id).strip()
        task_matches = not self.task_id or not task_id or task_id == self.task_id
        target_candidates = []
        for detection in message.target_dets:
            label = self._normalize_label(detection.label)
            score = float(detection.score)
            if not task_matches or score < self.target_eval_min_score:
                continue
            target_candidates.append({
                "label": label,
                "score": score,
                "box": (
                    float(detection.cx) - 0.5 * float(detection.w),
                    float(detection.cy) - 0.5 * float(detection.h),
                    float(detection.cx) + 0.5 * float(detection.w),
                    float(detection.cy) + 0.5 * float(detection.h),
                ),
            })
        predicted_box_px = projection["predicted_box_px"]
        predicted_box_normalized = projection["predicted_box_normalized"]
        best_candidate = None
        best_iou = 0.0
        if predicted_box_normalized is not None:
            for candidate in target_candidates:
                iou = self._target_eval_bbox_iou(
                    predicted_box_normalized, candidate["box"]
                )
                if best_candidate is None or iou > best_iou:
                    best_candidate = candidate
                    best_iou = iou
        matched = bool(
            exposed
            and best_candidate is not None
            and best_iou >= self.target_eval_min_match_iou
        )
        depth_confirmed_matched = bool(depth_visible and matched)
        episode = self.target_eval_active_episode
        if exposed and episode is not None:
            episode["frames"] += 1
            self.target_eval_exposed_frames += 1
            if matched:
                episode["matches"] += 1
                self.target_eval_matched_frames += 1
                if episode["first_match_source_stamp"] is None:
                    episode["first_match_source_stamp"] = source_stamp
                if self.target_eval_first_match_stamp is None:
                    self.target_eval_first_match_stamp = source_stamp
                    self._write(
                        "INFO",
                        "target_first_spatial_match",
                        truth_source="gazebo_evaluation_only",
                        episode_id=episode["id"],
                        source_stamp=round(source_stamp, 4),
                        iou=round(best_iou, 4),
                        score=round(best_candidate["score"], 4),
                        label=best_candidate["label"],
                    )
        depth_episode = self.target_eval_depth_active_episode
        if depth_visible and depth_episode is not None:
            depth_episode["frames"] += 1
            self.target_eval_depth_visible_frames += 1
            if depth_confirmed_matched:
                depth_episode["matches"] += 1
                self.target_eval_depth_visible_matched_frames += 1
                if depth_episode["first_match_source_stamp"] is None:
                    depth_episode["first_match_source_stamp"] = source_stamp
                if self.target_eval_first_depth_visible_match_stamp is None:
                    self.target_eval_first_depth_visible_match_stamp = source_stamp
                    self._write(
                        "INFO",
                        "target_first_depth_visible_spatial_match",
                        truth_source="gazebo_depth_evaluation_only",
                        episode_id=depth_episode["id"],
                        source_stamp=round(source_stamp, 4),
                        iou=round(best_iou, 4),
                        score=round(best_candidate["score"], 4),
                        label=best_candidate["label"],
                    )
        if target_candidates and not exposed:
            self.target_eval_outside_exposure_candidates += 1
        if target_candidates and not matched:
            self.target_eval_unmatched_target_candidate_frames += 1

        # Periodic frame evidence keeps logs bounded at long-running detector
        # rates, while every match/false positive and exposure transition stays
        # individually auditable.
        interesting = (
            started_episode
            or started_depth_episode
            or matched
            or depth_confirmed_matched
            or bool(target_candidates and not matched)
        )
        if interesting or now_wall - self.target_eval_last_frame_log_wall >= 1.0:
            self._write(
                "INFO",
                "target_eval_detection_frame",
                truth_source="gazebo_evaluation_only",
                task_id=task_id or None,
                task_matches=task_matches,
                prompt_a=str(message.prompt_a),
                source_stamp=round(source_stamp, 4),
                detector_delivery_latency_seconds=round(delivery_latency, 4),
                model_state_source_age_seconds=round(source_state_age, 4),
                model_state_rx_age_seconds=round(newest_age, 4),
                frustum_exposed=exposed,
                exposure_reason=projection["reason"],
                active_episode_id=None if episode is None else episode["id"],
                depth_visibility_state=depth_visibility["state"],
                depth_visibility_reason=depth_visibility["reason"],
                depth_visible=depth_visible,
                depth_available=depth_visibility["available"],
                depth_source_age_seconds=(
                    None
                    if depth_visibility["source_age_seconds"] is None
                    else round(depth_visibility["source_age_seconds"], 4)
                ),
                depth_valid_samples=depth_visibility["valid_samples"],
                depth_matching_samples=depth_visibility["matching_samples"],
                depth_foreground_samples=depth_visibility["foreground_samples"],
                depth_background_samples=depth_visibility["background_samples"],
                depth_sample_pixels=depth_visibility["sample_pixels"],
                observed_depth_median_m=(
                    None
                    if depth_visibility["observed_depth_median_m"] is None
                    else round(depth_visibility["observed_depth_median_m"], 4)
                ),
                expected_depth_min_m=(
                    None
                    if depth_visibility["expected_depth_min_m"] is None
                    else round(depth_visibility["expected_depth_min_m"], 4)
                ),
                expected_depth_max_m=(
                    None
                    if depth_visibility["expected_depth_max_m"] is None
                    else round(depth_visibility["expected_depth_max_m"], 4)
                ),
                depth_tolerance_m=(
                    None
                    if depth_visibility["tolerance_m"] is None
                    else round(depth_visibility["tolerance_m"], 4)
                ),
                active_depth_visible_episode_id=(
                    None if depth_episode is None else depth_episode["id"]
                ),
                center_depth_m=round(projection["center_depth_m"], 4),
                center_px=(
                    None if projection.get("center_px") is None
                    else [round(value, 2) for value in projection["center_px"]]
                ),
                predicted_box_px=(
                    None if predicted_box_px is None
                    else [round(value, 2) for value in predicted_box_px]
                ),
                predicted_box_normalized=(
                    None if predicted_box_normalized is None
                    else [round(value, 5) for value in predicted_box_normalized]
                ),
                target_candidate_count=len(target_candidates),
                best_candidate=(
                    None if best_candidate is None else {
                        "label": best_candidate["label"],
                        "score": round(best_candidate["score"], 4),
                        "box_normalized": [
                            round(value, 5) for value in best_candidate["box"]
                        ],
                    }
                ),
                best_iou=round(best_iou, 4),
                matched=matched,
                depth_confirmed_matched=depth_confirmed_matched,
            )
            self.target_eval_last_frame_log_wall = now_wall

    def on_detections(self, message):
        target = None
        if message.target_dets:
            target = max(message.target_dets, key=lambda item: float(item.score))
        with self.lock:
            self.detector_messages += 1
            self._evaluate_target_detection_locked(message)
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

    def on_perception_decision(self, message):
        """Persist detector-side acceptance/rejection evidence in this run log."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            payload = {"event": "malformed_perception_decision", "raw": message.data}
        if not isinstance(payload, dict):
            payload = {"event": "malformed_perception_decision", "raw": message.data}
        event = str(payload.pop("event", "unknown"))
        with self.lock:
            if self.pose is not None:
                payload["metrics_pose"] = [round(value, 4) for value in self.pose]
            payload["perception_event"] = event
            level = "WARN" if event in (
                "empty_image", "empty_prompt", "inference_exception", "stale_source_image"
            ) else "INFO"
            self._write(level, "perception_decision", **payload)

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
                    # Runtime logging deliberately continues after the task
                    # reaches its terminal state. Preserve that raw evidence,
                    # while writing one immutable benchmark boundary so idle
                    # time cannot dilute rates in offline comparisons.
                    completion_snapshot = {
                        "schema_version": 1,
                        "boundary": "task_done_callback",
                        "boundary_ros_time": round(self.task_done_ros, 3),
                        "boundary_wall_elapsed_seconds": round(
                            time.monotonic() - self.start_wall, 3
                        ),
                        "metrics": self._snapshot(),
                    }
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
                        completion_snapshot=completion_snapshot,
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
            "execution_architecture": self.execution_architecture,
            "persistent_execution": self.persistent_execution,
            "ros_time": round(rospy.Time.now().to_sec(), 3),
            "pose": None if pose is None else [round(float(value), 4) for value in pose],
            "goal": None if goal is None else [round(float(value), 4) for value in goal],
            "distance_to_goal": None if not math.isfinite(distance) else round(distance, 4),
            "pose_frame": "odom",
            "goal_frame": self.goal_frame,
            "distance_transform_failures": self.distance_transform_failures,
            "path_length": round(self.path_length, 4),
            "cmd": [round(float(self.command.linear.x), 4), round(float(self.command.angular.z), 4)],
            "cmd_vel_mux": self.cmd_vel_mux_status,
            "mux_governor_limited_events": self.mux_governor_limited_events,
            "mux_forced_zero_events": self.mux_forced_zero_events,
            "mux_status_reason_counts": self.mux_status_reason_counts,
            "teb_cmd": [round(float(self.teb_command.linear.x), 4), round(float(self.teb_command.angular.z), 4)],
            "teb_planner_cmd": [
                round(float(self.teb_planner_command.linear.x), 4),
                round(float(self.teb_planner_command.angular.z), 4),
            ],
            "teb_turn_supervisor": self.teb_turn_supervisor_status,
            "teb_turn_supervisor_events": self.teb_turn_supervisor_events,
            "teb_turn_supervisor_last_event": self.teb_turn_supervisor_last_event,
            "teb_trajectory_continuity_events": self.teb_trajectory_continuity_events,
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
            "move_base_recovery_events": self.move_base_recovery_events,
            "terminal_to_dispatch_count": self.terminal_to_dispatch_count,
            "terminal_to_dispatch_last_seconds": (
                None
                if self.terminal_to_dispatch_last is None
                else round(self.terminal_to_dispatch_last, 4)
            ),
            "terminal_to_dispatch_mean_seconds": round(
                self.terminal_to_dispatch_total
                / max(1, self.terminal_to_dispatch_count),
                4,
            ),
            "terminal_to_dispatch_max_seconds": round(
                self.terminal_to_dispatch_max, 4
            ),
            "global_costmap": self.global_costmap_stats,
            "local_costmap": self.local_costmap_stats,
            "navfn_plan": self.navfn_plan_stats,
            "global_planner_plan": self.global_planner_plan_stats,
            "teb_global_plan": self.teb_global_plan_stats,
            "teb_local_plan": self.teb_local_plan_stats,
            "state": self.state,
            "goal_source": self.goal_source,
            "goal_transaction_id": self.goal_transaction_id,
            "mission_goal_messages": self.mission_goal_messages,
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
            "move_base_unique_goal_ids": len(self.move_base_goal_ids),
            "move_base_preemptions": self.preemptions,
            "move_base_frontier_observation_preemptions": (
                self.frontier_observation_preemptions
            ),
            "move_base_frontier_terminal_settle_preemptions": (
                self.frontier_terminal_settle_preemptions
            ),
            "move_base_frontier_continuous_prefetch_preemptions": (
                self.frontier_continuous_prefetch_preemptions
            ),
            "move_base_frontier_segment_preemptions": self.frontier_segment_preemptions,
            "move_base_target_segment_preemptions": self.target_segment_preemptions,
            "move_base_priority_preemptions": self.priority_preemptions,
            "move_base_task_done_preemptions": self.task_done_preemptions,
            "move_base_unexpected_preemptions": self.unexpected_preemptions,
            "move_base_aborts": self.aborts,
            "move_base_successes": self.successes,
            "angular_sign_flips": self.angular_sign_flips,
            "strong_angular_sign_flips": self.strong_angular_sign_flips,
            "forward_steering_sign_flips": self.forward_steering_sign_flips,
            "teb_angular_sign_flips": self.teb_angular_sign_flips,
            "teb_strong_angular_sign_flips": self.teb_strong_angular_sign_flips,
            "teb_forward_steering_sign_flips": self.teb_forward_steering_sign_flips,
            # ``teb_cmd`` is after the turn supervisor; ``teb_planner_cmd``
            # is the unmodified move_base/TEB stream. `/cmd_vel` remains the
            # actual actuator request after the safety mux.
            "teb_supervisor_linear_brake_events": self.teb_linear_brake_events,
            "teb_supervisor_speed_modulation_events": self.teb_speed_modulation_events,
            "teb_planner_linear_brake_events": self.teb_planner_linear_brake_events,
            "teb_planner_speed_modulation_events": (
                self.teb_planner_speed_modulation_events
            ),
            # Compatibility aliases for logs/analyzers written before the
            # three command channels were explicitly named.
            "teb_linear_brake_events": self.teb_linear_brake_events,
            "teb_speed_modulation_events": self.teb_speed_modulation_events,
            "forward_distance_m": round(self.forward_distance, 3),
            "forward_angular_energy_rad": round(self.forward_angular_energy, 4),
            "forward_angular_energy_per_m": round(
                self.forward_angular_energy / max(0.01, self.forward_distance), 4
            ),
            # Unlike forward_angular_energy_per_m, these values exclude both
            # planned bends and heading-acquisition turns. They are the
            # primary evidence for actual left/right wobble on a straight
            # corridor or open path.
            "straight_path_distance_m": round(self.straight_path_distance, 3),
            "straight_path_angular_energy_rad": round(
                self.straight_path_angular_energy, 4
            ),
            "straight_path_angular_energy_per_m": round(
                self.straight_path_angular_energy
                / max(0.01, self.straight_path_distance),
                4,
            ),
            "straight_path_steering_sign_flips": (
                self.straight_path_steering_sign_flips
            ),
            "straight_path_samples": self.straight_path_samples,
            "straight_path_tracking": self.straight_path_state,
            "straight_path_lookahead_m": round(self.straight_path_lookahead, 3),
            "straight_path_max_curvature_rad": round(
                self.straight_path_max_curvature, 4
            ),
            "straight_path_max_heading_error_rad": round(
                self.straight_path_max_heading_error, 4
            ),
            "forward_speed_threshold": round(self.forward_speed_threshold, 4),
            "brake_events_clear": self.brake_events_clear,
            "brake_events_near": self.brake_events_near,
            "brake_events_unknown_clearance": self.brake_events_unknown_clearance,
            # Raw clearance describes sensor state at a speed drop. Final
            # lifecycle attribution distinguishes an expected action endpoint
            # from an actual clear-path interruption.
            "endpoint_terminal_brake_events": (
                self.brake_reason_counts.get("action_terminal", 0)
                + self.brake_reason_counts.get("endpoint_action_terminal", 0)
                + self.brake_reason_counts.get("persistent_endpoint_terminal", 0)
            ),
            "action_terminal_brake_events": self.brake_reason_counts.get("action_terminal", 0),
            "unexplained_clear_path_brake_events": self.brake_reason_counts.get(
                "unexplained_clear_path", 0
            ),
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
            "speed_modulation_events": self.speed_modulation_events,
            "brake_reason_counts": dict(self.brake_reason_counts),
            "stop_reason_counts": dict(self.stop_reason_counts),
            "goal_change_rate_per_minute": round(goal_change_rate_per_minute, 3),
            "goal_transition_counts": {
                "prefix_continuation": self.continuous_goal_transitions,
                "endpoint_divergence": self.divergent_goal_transitions,
                "terminal_prefetched_successor": self.terminal_goal_transitions,
                "unknown": self.unknown_goal_transitions,
            },
            "goal_transition_brake_events": dict(self.transition_brake_events),
            "persistent_plan_received": self.persistent_plan_received,
            "persistent_plan_equivalent_retained": (
                self.persistent_plan_equivalent_retained
            ),
            "persistent_plan_installed": self.persistent_plan_installed,
            "persistent_plan_last_event": self.persistent_plan_last_event,
            "persistent_plan_route_version": self.persistent_plan_route_version,
            "persistent_plan_geometry_hash": self.persistent_plan_geometry_hash,
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
            "teb_bridge_frontier_observation_completions": self.bridge_frontier_observation_completions,
            "teb_bridge_frontier_terminal_settle_completions": (
                self.bridge_frontier_terminal_settle_completions
            ),
            "teb_bridge_frontier_continuous_prefetch_handoffs": self.bridge_frontier_continuous_prefetch_handoffs,
            "teb_bridge_frontier_continuous_prefetch_fallbacks": self.bridge_frontier_continuous_prefetch_fallbacks,
            "teb_bridge_frontier_prefetch_requires_turn": self.bridge_frontier_prefetch_requires_turn,
            "teb_bridge_persistent_lookahead_handoffs": self.bridge_persistent_lookahead_handoffs,
            "teb_bridge_persistent_curve_handoffs": self.bridge_persistent_curve_handoffs,
            "teb_bridge_persistent_lookahead_admission_deferred": (
                self.bridge_persistent_lookahead_admission_deferred
            ),
            "teb_bridge_persistent_lookahead_admission_reasons": dict(
                self.bridge_persistent_lookahead_admission_reasons
            ),
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
            "target_follow_confirmed_seconds": None if self.target_follow_confirmed_ros is None else round(self.target_follow_confirmed_ros - self.start_ros, 3),
            "target_close_confirmation_started_seconds": None if self.target_close_confirmation_started_ros is None else round(self.target_close_confirmation_started_ros - self.start_ros, 3),
            "target_close_confirmed_seconds": None if self.target_close_confirmed_ros is None else round(self.target_close_confirmed_ros - self.start_ros, 3),
            "legacy_state_locked_seconds": None if self.legacy_state_locked_ros is None else round(self.legacy_state_locked_ros - self.start_ros, 3),
            "target_lock_seconds": None if self.target_lock_ros is None else round(self.target_lock_ros - self.start_ros, 3),
            "task_done_seconds": None if self.task_done_ros is None else round(self.task_done_ros - self.start_ros, 3),
            "target_lifecycle_sessions_observed": len(self.target_lifecycle_sessions),
            "target_lifecycle_completed_session": (
                None
                if self.target_lifecycle_last_completed_key is None
                else list(self.target_lifecycle_last_completed_key)
            ),
            "target_segments_committed": self.target_segments_committed,
            "target_continuous_handoffs_prepared": self.target_continuous_handoffs_prepared,
            "target_goal_changes": self.target_goal_changes,
            "target_route_accepts": self.target_route_accepts,
            "target_route_rejections": self.target_route_rejections,
            "target_route_deferrals": self.target_route_deferrals,
            "target_route_holds": self.target_route_holds,
            "target_route_semantic_replans": self.target_route_semantic_replans,
            "target_route_failures": self.target_route_failures,
            "target_route_releases": self.target_route_releases,
            "target_approach_terminals": self.target_approach_terminals,
            "min_scan_clearance": None if not math.isfinite(self.min_clearance) else round(self.min_clearance, 4),
            "discontinuity_obstacle_clearance": round(
                self.discontinuity_obstacle_clearance, 4
            ),
            "target_geometric_evaluation": self._target_eval_snapshot_locked(),
            "map": self.map_stats,
        }

    def on_sample(self, _event):
        with self.lock:
            self.sample_count += 1
            self._write("INFO", "sample", sample=self.sample_count, **self._snapshot())

    def close(self):
        with self.lock:
            self._target_eval_end_episode_locked("run_stop")
            self._target_eval_end_depth_episode_locked("run_stop")
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
