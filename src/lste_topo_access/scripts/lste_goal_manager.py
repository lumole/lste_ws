#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goal Manager
- 唯一对外发布 /lste/final_goal (PoseStamped; normal local goals use odom,
  online SLAM frontier goals retain the map frame)
- 依据 state/subtype + frontier/检测/激光裁剪生成全局目标
- 各状态有独立更新周期；state 变化可立即打断更新
"""

import copy
import json
import math
from typing import Optional, Tuple, List

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose2D, PoseStamped, PointStamped, Vector3Stamped, TransformStamped, Twist
from nav_msgs.srv import GetPlan, GetPlanRequest
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from std_msgs.msg import UInt8, Bool, String
from image_geometry import PinholeCameraModel
import tf2_ros
from tf.transformations import quaternion_matrix, quaternion_from_euler

from lste_msgs.msg import LsteState, LsteDetections, LsteDetection, LsteTask, LsteFrontiers, LsteScores


STATE_PASS = 0
STATE_SUSPICIOUS = 1
STATE_LOCKED = 2
# STATE_EXHAUSTED = 3  # 当前忽略

# 统一的 goal 生成模式命名（对应你约定的 explore/catch 两大类）
EXPLORE_PASS_MODE = "explore_pass_mode"          # 原 PASS coarse
EXPLORE_SUS_C_MODE = "explore_sus_c_mode"        # 原 Sus-C fine
CATCH_TARGET_MODE = "catch_target_mode"          # 原 follow target
CATCH_CTX_MODE = "catch_ctx_mode"                # 原 follow ctx


class GoalManager:
    def __init__(self):
        rospy.init_node("lste_goal_manager")

        # --- 参数 ---
        gp = rospy.get_param
        self.pass_period = float(gp("~pass_period", 5.0))
        self.sus_c_period = float(gp("~sus_c_period", 3.0))
        self.locked_period = float(gp("~locked_period", 2.0))
        self.sus_a_period = float(gp("~sus_a_period", 2.0))
        self.sus_b_period = float(gp("~sus_b_period", 2.0))
        self.forward_dist = float(gp("~forward_dist", 10.0))
        self.goal_dist_det = float(gp("~goal_dist_det", 5.0))  # 无深度：检测方向前推距离
        # RGB-only target following has no metric range.  Advance in short
        # segments and re-observe instead of sending the robot a fixed 5 m
        # beyond a small object.  Context following keeps goal_dist_det.
        self.follow_target_step_distance = max(
            0.2, float(gp("~follow_target_step_distance", 1.5))
        )
        # The production rl_grid_guard already plans a collision-free local
        # route from lidar.  Clipping a target ray to the first desk/chair can
        # instead create a sub-goal inside the controller's arrival radius,
        # causing visual pursuit to stop before it has approached the object.
        # Keep clipping available for a policy-only comparison, where no local
        # safety planner is active.
        target_scan_clip = gp("~follow_target_use_scan_clip", False)
        self.follow_target_use_scan_clip = str(target_scan_clip).strip().lower() in (
            "1", "true", "yes", "on",
        )
        # Context is evidence for where to look, not a metric target. Inspect
        # a confirmed context pair with one short visual-servo step, then return
        # to global coverage unless the actual target becomes visible.
        self.follow_context_step_distance = max(
            0.2, float(gp("~follow_context_step_distance", 1.5))
        )
        self.frontier_topic = gp("~frontier_topic", "/lste/gp_frontier_dir")
        self.frontiers_topic = gp("~frontiers_topic", "/lste/gp_frontiers")
        self.depth_topic = gp("~depth_topic", "/kinect/hd/image_depth_rect")
        self.image_topic = gp("~image_topic", "/kinect/hd/image_color_rect")
        self.use_depth = bool(gp("~use_depth", False))  # 默认关闭深度，当前环境无 depth
        # follow 模式鲁棒性参数
        self.follow_target_lost_timeout = float(gp("~follow_target_lost_timeout", 10.0))
        # After a genuine target sighting, a forward-facing camera can lose the
        # object when the local controller turns around a desk. Perform one
        # bounded turn-and-look before surrendering that nearby evidence to
        # global exploration. This is a goal for the normal controller, never
        # a direct velocity command.
        self.target_reacquire_duration = max(
            0.0, float(gp("~target_reacquire_duration", 6.0))
        )
        self.target_reacquire_distance = max(
            0.2, float(gp("~target_reacquire_distance", 1.0))
        )
        self.target_reacquire_max_attempts = max(
            0, int(gp("~target_reacquire_max_attempts", 1))
        )
        self.follow_ctx_lost_timeout = float(gp("~follow_ctx_lost_timeout", 5.0))
        # Sus-B is often a one-cycle classification caused by context-box
        # flicker. Keep the context mode for a short dwell before allowing a
        # PASS/Sus-C update to replace its route. A confirmed target still
        # preempts this hold immediately.
        self.context_state_hold_time = max(
            0.0, float(gp("~context_state_hold_time", 4.0))
        )
        # Context labels identify a place worth looking at once; continuously
        # chasing the same monitor/desk pair can otherwise monopolize search
        # while the actual target remains absent.  After one observation
        # window, return to global coverage for this bounded cooldown.
        self.context_follow_cooldown = max(
            0.0, float(gp("~context_follow_cooldown", 30.0))
        )
        self.follow_min_update_period = float(gp("~follow_min_update_period", 0.3))
        self.follow_heading_alpha = float(gp("~follow_heading_alpha", 0.5))
        # Target-follow measurements arrive asynchronously from the detector.
        # Filter the image ray once per *new* detector frame, rather than
        # recomputing it from the same box on every Goal Manager timer tick.
        # The latter made a moving robot generate a new one-metre goal every
        # 300 ms and forced TEB to preempt its active trajectory.
        self.target_bbox_alpha = min(
            1.0, max(0.05, float(gp("~target_bbox_alpha", 0.35)))
        )
        self.target_heading_alpha = min(
            1.0, max(0.05, float(gp("~target_heading_alpha", 0.30)))
        )
        self.target_max_heading_step = math.radians(max(
            1.0, float(gp("~target_max_heading_step_deg", 25.0))
        ))
        self.target_goal_reached_radius = max(
            0.20, float(gp("~target_goal_reached_radius", 0.75))
        )
        self.target_goal_heading_update_threshold = math.radians(max(
            1.0, float(gp("~target_goal_heading_update_threshold_deg", 35.0))
        ))
        # A detector ray is an observation, not proof that the corresponding
        # point is a navigable destination.  Before a confirmed visual target
        # can take ownership from the map route, ask the same Navfn instance
        # used by move_base for a connected plan.  This is a mission/execution
        # contract: an unreachable visual ray remains evidence while the
        # current frontier action continues.
        target_route_validation = gp("~target_route_validation", True)
        self.target_route_validation = str(target_route_validation).strip().lower() in (
            "1", "true", "yes", "on",
        )
        self.target_route_validation_service = gp(
            "~target_route_validation_service", "/move_base/NavfnROS/make_plan"
        )
        self.target_route_validation_timeout = max(
            0.01, float(gp("~target_route_validation_timeout", 0.05))
        )
        self.target_route_validation_tolerance = max(
            0.0, float(gp("~target_route_validation_tolerance", 0.20))
        )
        self.target_route_validation_period = max(
            1.0, float(gp("~target_route_validation_period", 1.0))
        )
        self.target_cache_max_advances = max(
            0, int(gp("~target_cache_max_advances", 2))
        )
        self.follow_locked_done_time = float(gp("~follow_locked_done_time", 10.0))
        self.follow_goal_publish_period = max(
            0.05, float(gp("~follow_goal_publish_period", 0.3))
        )
        # A task is complete only after the target occupies a close-range
        # portion of the camera image in several fresh observations. The mux
        # stops the vehicle immediately after this confirmation; time-only
        # LOCKED completion could report success from a distant view.
        self.target_done_min_box_width = float(gp("~target_done_min_box_width", 0.06))
        self.target_done_min_box_height = float(gp("~target_done_min_box_height", 0.06))
        self.target_done_min_score = float(gp("~target_done_min_score", 0.40))
        self.target_done_min_hold_time = max(
            0.0, float(gp("~target_done_min_hold_time", 2.0))
        )
        self.target_done_min_fresh_hits = max(
            1, int(gp("~target_done_min_fresh_hits", 3))
        )
        self.target_done_max_detection_age = max(
            0.0, float(gp("~target_done_max_detection_age", 0.75))
        )
        # Detector inference is slower than the Goal Manager timer. Keep a
        # qualifying target frame briefly even if the next detector message is
        # empty, then discard an unconfirmed candidate instead of chasing a
        # one-frame false positive indefinitely.
        self.target_follow_min_score = max(
            0.0, float(gp("~target_follow_min_score", 0.20))
        )
        self.target_follow_min_box_size = max(
            0.0, float(gp("~target_follow_min_box_size", 0.01))
        )
        self.target_follow_candidate_timeout = max(
            0.5, float(gp("~target_follow_candidate_timeout", 8.0))
        )
        self.target_follow_confirm_hits = max(
            1, int(gp("~target_follow_confirm_hits", 2))
        )
        self.target_follow_confirm_window = max(
            self.target_follow_candidate_timeout,
            float(gp("~target_follow_confirm_window", 20.0)),
        )
        # A weak WeDetect box is not a reliable target identity by itself.
        # Require a longer, spatially consistent run before allowing a tiny
        # low-score box to take over a map-connected frontier route. Strong
        # boxes still use the shorter confirmation window above.
        self.target_follow_weak_confirm_hits = max(
            self.target_follow_confirm_hits,
            int(gp("~target_follow_weak_confirm_hits", 5)),
        )
        self.target_follow_weak_min_average_score = max(
            self.target_follow_min_score,
            float(gp("~target_follow_weak_min_average_score", 0.24)),
        )
        self.target_follow_spatial_tolerance = max(
            0.02, float(gp("~target_follow_spatial_tolerance", 0.10))
        )
        # WeDetect-Large may take several seconds between independent frames.
        # Freeze the vehicle at the first (or close-range) visual hit long
        # enough to obtain another frame before advancing the visual ray.
        self.target_observation_hold = max(
            0.0, float(gp("~target_observation_hold", 4.5))
        )
        # A close target can disappear before the exploratory state machine
        # accumulates every LOCKED hit. Keep strict LOCKED-only completion
        # available for comparison runs, but do not require it by default.
        require_locked = gp("~target_done_require_locked", False)
        self.target_done_require_locked = str(require_locked).strip().lower() in (
            "1", "true", "yes", "on",
        )
        # The normal LSTE path lets state/detections/frontiers choose each
        # global goal.  ``fixed`` deliberately bypasses that decision layer but
        # keeps the exact same /lste/final_goal contract for controllers.
        self.global_goal_source = str(gp("~global_goal_source", "brain")).strip().lower()
        if self.global_goal_source not in ("brain", "fixed"):
            raise ValueError(
                "~global_goal_source must be 'brain' or 'fixed' "
                "(got %r)" % self.global_goal_source
            )
        self.fixed_goal = (
            float(gp("~fixed_goal_x", 0.0)),
            float(gp("~fixed_goal_y", 0.0)),
            float(gp("~fixed_goal_yaw", 0.0)),
        )
        self.fixed_goal_publish_period = max(
            0.05, float(gp("~fixed_goal_publish_period", 1.0))
        )
        # During search, an online SLAM frontier may provide a globally
        # connected map waypoint. It is optional and never overrides a visual
        # target/context follow mode, fixed-goal tests, or the final-goal API.
        global_frontier_enabled = gp("~global_frontier_enabled", False)
        self.global_frontier_enabled = str(global_frontier_enabled).strip().lower() in (
            "1", "true", "yes", "on",
        )
        self.global_frontier_topic = gp("~global_frontier_topic", "/lste/global_frontier_goal")
        self.global_frontier_status_topic = gp(
            "~global_frontier_status_topic", "/lste/global_frontier/status"
        )
        # A target segment and an exploration route are separate mission
        # transactions.  When target ownership ends, the previous frontier
        # endpoint can be far behind the robot.  Request a new branch from the
        # current map pose instead of treating that cached endpoint as a safe
        # fallback.
        self.global_frontier_replan_request_topic = gp(
            "~global_frontier_replan_request_topic",
            "/lste/global_frontier/replan_request",
        )
        self.frontier_replan_request_id = 0
        self.frontier_replan_pending_id = 0
        self.frontier_replan_ready_id = 0
        self.teb_goal_terminal_topic = gp(
            "~teb_goal_terminal_topic", "/lste/teb_goal_terminal"
        )
        self.teb_goal_failure_topic = gp(
            "~teb_goal_failure_topic", "/lste/teb_goal_failure"
        )
        self.global_frontier_max_age = max(
            0.0, float(gp("~global_frontier_max_age", 3.0))
        )
        # The global frontier node recomputes a connected map path at 1 Hz.
        # Forward its latest short waypoint at a similar cadence: the legacy
        # 5 s PASS period makes a local RL controller overshoot corridor
        # bends before it receives the next path segment.
        self.global_frontier_period = max(
            0.1, float(gp("~global_frontier_period", 1.0))
        )
        # The frontier explorer publishes a short map-frame waypoint that moves as the
        # robot advances along a still-valid map path. Do not turn every such
        # one-metre update into a new move_base action: hold the current
        # waypoint until it is nearly reached, while still accepting a large
        # branch change immediately.
        self.global_frontier_update_radius = max(
            0.2, float(gp("~global_frontier_update_radius", 0.75))
        )
        # Permit a continuous map route to replace its short waypoint before
        # move_base reports SUCCEEDED.  The bridge can then preempt the old
        # simple goal with the next path segment instead of waiting for the
        # planner to reach a terminal tolerance and brake to zero.
        self.global_frontier_early_handoff_radius = max(
            self.global_frontier_update_radius,
            float(gp("~global_frontier_early_handoff_radius", 0.95)),
        )
        self.global_frontier_jump_distance = max(
            self.global_frontier_update_radius,
            float(gp("~global_frontier_jump_distance", 2.0)),
        )
        # A large frontier branch change is only useful after the current short
        # map waypoint has been reached. Previously the jump-distance rule
        # bypassed this condition, so a map branch change could turn an RL car
        # around while it was still 1--2 m from the old waypoint.
        self.global_frontier_jump_release_radius = max(
            self.global_frontier_update_radius,
            float(gp("~global_frontier_jump_release_radius", 0.90)),
        )
        self.global_frontier_terminal_hold_timeout = max(
            0.0, float(gp("~global_frontier_terminal_hold_timeout", 20.0))
        )
        # A frontier publisher runs at 1 Hz while SLAM and the local controller
        # are both moving.  Do not turn every small waypoint refresh into a new
        # final goal immediately after the previous one was accepted.  This is
        # a temporal hysteresis only; a large branch jump and a completed TEB
        # action are still allowed through.
        self.global_frontier_min_hold_time = max(
            0.0, float(gp("~global_frontier_min_hold_time", 2.5))
        )
        # A context pair is useful evidence, but switching from a map route to
        # a context midpoint while the robot is still driving causes a large
        # heading jump.  Keep the active frontier commitment until its short
        # waypoint is reached unless a confirmed target ray is available.
        preempt_context = gp("~global_frontier_preempt_context", False)
        self.global_frontier_preempt_context = str(preempt_context).strip().lower() in (
            "1", "true", "yes", "on",
        )
        # In TEB mode a frontier waypoint is an action, not a continuously
        # moving setpoint. Hold ordinary map updates until the bridge reports
        # SUCCEEDED; RL keeps the legacy distance-based handoff. The mode is
        # initialized from the launcher and also follows hot controller
        # switches through /lste/controller_mode.
        self.controller_mode = str(gp("~controller_mode", "teb")).strip().lower()
        if self.controller_mode not in ("sappo", "teleop", "teb"):
            self.controller_mode = "teb"
        self.controller_mode_topic = gp(
            "~controller_mode_topic", "/lste/controller_mode"
        )
        # Goal Manager owns mission arbitration.  The intent topic makes that
        # decision explicit to the TEB action bridge: a confirmed visual target
        # may take over an exploration action, while ordinary frontier/context
        # refreshes remain queued until the current action finishes.
        self.goal_intent_topic = gp("~goal_intent_topic", "/lste/goal_intent")
        # Observation is a mission state, not a new navigation destination.
        # The mux consumes this latch and temporarily gates the selected
        # controller while the detector collects fresh frames.
        self.navigation_hold_topic = gp(
            "~navigation_hold_topic", "/lste/navigation_hold"
        )
        allow_fixed_click_override = gp("~fixed_goal_allow_click_override", False)
        self.fixed_goal_allow_click_override = str(allow_fixed_click_override).strip().lower() in (
            "1", "true", "yes", "on",
        )
        self.fixed_goal_click_topic = gp("~fixed_goal_click_topic", "/move_base/current_goal")
        # 激光裁剪相关
        self.scan_topic = gp("~scan_topic", "/pro3/rl_scan")
        self.scan_frame = gp("~scan_frame", "")  # 留空则使用 scan.header.frame_id
        self.safety_margin = float(gp("~safety_margin", 0.6))
        self.scan_window_bins = int(gp("~scan_window_bins", 2))
        # 前进优先/投票相关参数
        self.front_sigma = math.radians(float(gp("~front_sigma_deg", 25.0)))
        self.area_power = float(gp("~area_power", 1.0))
        self.v_min = float(gp("~v_min", 0.05))
        self.v_scale = float(gp("~v_scale", 0.3))
        self.front_deg = math.radians(float(gp("~front_deg", 60.0)))
        self.back_deg = math.radians(float(gp("~back_deg", 60.0)))
        self.w_front_max = float(gp("~w_front_max", 1.5))
        self.w_back_min = float(gp("~w_back_min", 0.2))
        self.forward_only = bool(gp("~forward_only", True))
        self.v_gate = float(gp("~v_gate", 0.08))
        self.min_allowed_score = float(gp("~min_allowed_score", 0.05))
        self.switch_margin = float(gp("~switch_margin", 0.15))
        # Diagnostic-only: emits one compact line per final-goal publication.
        debug_goal_log = gp("~debug_goal_log", False)
        self.debug_goal_log = str(debug_goal_log).strip().lower() in ("1", "true", "yes", "on")

        # --- 状态缓存 ---
        self.current_state = STATE_PASS
        self.current_subtype = ""
        self.last_state = None
        self.last_goal: Optional[PoseStamped] = None
        self.next_update_time = 0.0
        self.start_pose: Optional[Pose2D] = None

        self.latest_pose: Optional[Pose2D] = None
        self.latest_state_msg: Optional[LsteState] = None
        self.latest_task: Optional[LsteTask] = None
        self.latest_dets: Optional[LsteDetections] = None
        self.latest_scores: Optional[LsteScores] = None
        self.latest_frontiers: Optional[LsteFrontiers] = None
        self.latest_frontier: Optional[Vector3Stamped] = None
        self.latest_scan: Optional[LaserScan] = None
        self.latest_cmd_vel: Optional[Twist] = None
        self.latest_global_frontier_goal: Optional[PoseStamped] = None
        # The frontier planner labels whether the current goal is a validated
        # route connector or the final approach point.  This semantic is
        # forwarded with the goal intent so the TEB bridge can perform an
        # atomic handoff only for a continuous map route.
        self.global_frontier_route_kind = "frontier_endpoint"
        # Keep the last map-connected waypoint separate from the currently
        # published visual/context segment. This lets a stale target segment
        # release cleanly after handoff instead of leaving the controller
        # stopped on an obsolete target pose.
        self.last_frontier_goal: Optional[PoseStamped] = None
        # Set only when TEB reports a terminal event for the current frontier
        # pose. It is consumed by the next successful publish, so one finished
        # action can release exactly one held waypoint.
        self.teb_terminal_goal: Optional[PoseStamped] = None
        # The bridge may coalesce several frontier publications while one
        # action is executing.  Keep the small committed history so a terminal
        # result is matched to the action that actually finished, rather than
        # to whichever frontier happened to be published most recently.
        self.teb_frontier_goal_history: List[PoseStamped] = []
        self.headings: Optional[List[float]] = None
        self.last_dir_idx: Optional[int] = None
        self.goal_source = "uninitialized"
        self.last_goal_source = ""
        self.frontier_debug = "unavailable"
        # Access-Topo 覆盖
        self.access_mode = 0
        self.access_backtrack_goal: Optional[PoseStamped] = None
        # follow 模式内部计时与缓存
        self.target_last_seen: Optional[float] = None
        self.target_last_goal: Optional[PoseStamped] = None
        self.target_last_update: float = 0.0
        self.target_last_heading: Optional[float] = None
        self.target_filtered_cx: Optional[float] = None
        self.target_filtered_cy: Optional[float] = None
        self.target_filtered_heading: Optional[float] = None
        self.target_last_detection_stamp = None
        self.target_goal_detection_stamp = None
        # Detection evidence and navigation actions have different lifecycles.
        # Keep the current visual-servo segment committed until its move_base
        # action reaches a terminal result; detector frames only update this
        # track state and cannot replace an active TEB action mid-segment.
        self.target_segment_terminal_ready = False
        self.target_track_label = ""
        # A detector epoch changes for every inference frame. This id remains
        # stable for one continuous visual target track.
        self.target_track_sequence = 0
        self.target_track_id = ""
        self.target_cache_advances = 0
        self.frontier_goal_sent_at: Optional[float] = None
        self.target_observation_hold_goal: Optional[PoseStamped] = None
        self.target_candidate: Optional[LsteDetection] = None
        self.target_candidate_last_seen: Optional[float] = None
        self.target_candidate_source_stamp = None
        self.target_candidate_hits = 0
        self.target_candidate_anchor_cx: Optional[float] = None
        self.target_candidate_anchor_cy: Optional[float] = None
        self.target_candidate_score_sum = 0.0
        self.target_follow_confirmed = False
        # Target navigation is an explicit mission state, separate from raw
        # detector evidence.  A failed visual route becomes BLOCKED and can be
        # reconsidered only after frontier progress; this prevents perception
        # from reclaiming the controller on every detector frame.
        self.target_execution_state = "TARGET_CANDIDATE"
        self.target_observation_epoch = 0
        self.target_blocked = False
        self.target_blocked_goal: Optional[PoseStamped] = None
        self.target_blocked_since: Optional[float] = None
        self.target_blocked_reason = ""
        self.target_failure_count = 0
        self.target_observation_hold_until = 0.0
        self.target_route_validation_next_time = 0.0
        self.target_route_validation_failures = 0
        self.target_route_validation_last_result = "not_checked"
        self.target_route_validation_last_goal = None
        self.navigation_hold_active = False
        self.target_reacquire_goal: Optional[PoseStamped] = None
        self.target_reacquire_started: Optional[float] = None
        self.target_reacquire_attempts = 0
        self.locked_enter_time: Optional[float] = None
        self.target_close_since: Optional[float] = None
        self.target_close_hits = 0
        self.target_close_last_stamp = None
        # Wall time of the most recent close-box detector frame.  A small
        # object at the frame edge can flicker below the box threshold for one
        # detector cycle without leaving the scene; the completion counter must
        # survive that dip instead of resetting to zero.
        self.target_close_last_seen: Optional[float] = None
        self.task_done_published = False
        self.current_task_id = ""
        self.ctx_start_time: Optional[float] = None
        self.ctx_state_hold_until = 0.0
        self.ctx_cooldown_until = 0.0
        self.ctx_last_goal: Optional[PoseStamped] = None
        self.ctx_last_update: float = 0.0
        # Context detections (for example monitor + monitor) are a visual
        # direction cue, not a continuously moving setpoint.  Keep one short
        # inspection segment until the robot reaches it or the bearing has
        # changed materially.
        self.ctx_goal_reached_radius = max(
            0.20, float(gp("~context_goal_reached_radius", 0.55))
        )
        self.ctx_goal_heading_update_threshold = math.radians(max(
            1.0, float(gp("~context_goal_heading_update_threshold_deg", 35.0))
        ))

        # 模式：base_mode 直接由 state/subtype 映射得到；后续可在 effective_mode 上做 override
        self.base_mode = EXPLORE_PASS_MODE
        self.effective_mode = self.base_mode

        # 深度/相机
        self.bridge = CvBridge()
        self.depth_image: Optional[Image] = None
        self.depth_stamp: Optional[rospy.Time] = None
        self.camera_info: Optional[CameraInfo] = None
        self.camera_frame: Optional[str] = None
        self.camera_model = PinholeCameraModel()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.target_route_validation_client = rospy.ServiceProxy(
            self.target_route_validation_service, GetPlan
        )

        # 发布
        # Controllers can start after the first frontier decision.  Latching
        # the current goal makes startup order irrelevant and prevents a
        # wait_for_goal controller from remaining stopped until the next map
        # update happens to produce a different coordinate.
        self.pub_goal = rospy.Publisher(
            "/lste/final_goal", PoseStamped, queue_size=1, latch=True
        )
        # Machine-readable decision trace consumed by navigation metrics. The
        # human-readable GOAL_DIAG line remains for rosout/tmux inspection.
        self.pub_goal_diagnostic = rospy.Publisher(
            "/lste/goal_diagnostic", String, queue_size=10
        )
        self.pub_goal_intent = rospy.Publisher(
            self.goal_intent_topic, String, queue_size=1, latch=True
        )
        self.goal_arbitration_topic = gp(
            "~goal_arbitration_topic", "/lste/goal_arbitration"
        )
        self.pub_goal_arbitration = rospy.Publisher(
            self.goal_arbitration_topic, String, queue_size=10
        )
        self.pub_global_frontier_replan = rospy.Publisher(
            self.global_frontier_replan_request_topic, String, queue_size=10
        )
        self.pub_access_mode = rospy.Publisher("/lste/access_topo/active_mode", String, queue_size=1, latch=True)
        self.pub_task_done = rospy.Publisher("/lste/task_done", Bool, queue_size=1, latch=True)
        self.pub_navigation_hold = rospy.Publisher(
            self.navigation_hold_topic, Bool, queue_size=1, latch=True
        )
        self.pub_navigation_hold.publish(Bool(data=False))

        # 订阅
        self.sub_state = rospy.Subscriber("/lste/state", LsteState, self.on_state, queue_size=1)
        self.sub_dets = rospy.Subscriber("/lste/detections", LsteDetections, self.on_dets, queue_size=1)
        self.sub_scores = rospy.Subscriber("/lste/scores", LsteScores, self.on_scores, queue_size=1)
        self.sub_task = rospy.Subscriber("/lste/task", LsteTask, self.on_task, queue_size=1)
        self.sub_pose = rospy.Subscriber("/rbt_pose", Pose2D, self.on_pose, queue_size=1)
        self.sub_cam_info = rospy.Subscriber("/kinect/hd/camera_info", CameraInfo, self.on_cam_info, queue_size=1)
        if self.use_depth:
            self.sub_depth = rospy.Subscriber(self.depth_topic, Image, self.on_depth, queue_size=1)
        self.sub_frontier = rospy.Subscriber(self.frontier_topic, Vector3Stamped, self.on_frontier, queue_size=1)
        self.sub_frontiers = rospy.Subscriber(self.frontiers_topic, LsteFrontiers, self.on_frontiers, queue_size=1)
        if self.global_frontier_enabled:
            self.sub_global_frontier = rospy.Subscriber(
                self.global_frontier_topic, PoseStamped, self.on_global_frontier, queue_size=1
            )
            self.sub_global_frontier_status = rospy.Subscriber(
                self.global_frontier_status_topic,
                String,
                self.on_global_frontier_status,
                queue_size=10,
            )
        self.sub_teb_goal_terminal = rospy.Subscriber(
            self.teb_goal_terminal_topic, PoseStamped,
            self.on_teb_goal_terminal, queue_size=1,
        )
        self.sub_teb_goal_failure = rospy.Subscriber(
            self.teb_goal_failure_topic,
            String,
            self.on_teb_goal_failure,
            queue_size=10,
        )
        self.sub_controller_mode = rospy.Subscriber(
            self.controller_mode_topic, String, self.on_controller_mode,
            queue_size=1,
        )
        self.sub_scan = rospy.Subscriber(self.scan_topic, LaserScan, self.on_scan, queue_size=1)
        self.sub_cmd_vel = rospy.Subscriber("/cmd_vel", Twist, self.on_cmd_vel, queue_size=1)
        self.sub_access_mode = rospy.Subscriber("/lste/access_topo/mode", UInt8, self.on_access_mode, queue_size=1)
        self.sub_access_goal = rospy.Subscriber("/lste/access_topo/backtrack_goal", PoseStamped, self.on_access_goal, queue_size=1)
        if self.global_goal_source == "fixed" and self.fixed_goal_allow_click_override:
            self.sub_fixed_goal_click = rospy.Subscriber(
                self.fixed_goal_click_topic, PoseStamped, self.on_fixed_goal_click, queue_size=1,
            )

        self.timer = rospy.Timer(rospy.Duration(0.2), self.on_timer)  # 5Hz
        rospy.loginfo(
            "Goal Manager started: source=%s publishes /lste/final_goal "
            "target_route_validation=%s service=%s",
            self.global_goal_source,
            self.target_route_validation,
            self.target_route_validation_service,
        )
        if self.global_goal_source == "fixed":
            rospy.loginfo(
                "Fixed global goal configured: (%.2f, %.2f, %.2f), publish_period=%.2fs",
                *self.fixed_goal, self.fixed_goal_publish_period,
            )

    # -------------------- Callbacks --------------------
    def set_navigation_hold(self, active: bool, reason: str):
        """Publish mission-level observation hold only on state changes."""
        active = bool(active)
        if active == self.navigation_hold_active:
            return
        self.navigation_hold_active = active
        self.pub_navigation_hold.publish(Bool(data=active))
        rospy.loginfo(
            "GoalManager: navigation_hold=%s reason=%s remaining=%.2fs",
            active,
            reason,
            max(0.0, self.target_observation_hold_until - rospy.Time.now().to_sec()),
        )

    def on_state(self, msg: LsteState):
        prev_state = self.current_state
        prev_subtype = self.current_subtype
        self.latest_state_msg = msg
        self.current_state = msg.state
        self.current_subtype = msg.subtype or ""
        now = rospy.Time.now().to_sec()
        # 记录 LOCKED 持续时间
        if self.current_state == STATE_LOCKED:
            if self.locked_enter_time is None:
                self.locked_enter_time = now
        else:
            self.locked_enter_time = None
            if self.target_done_require_locked:
                self.reset_target_close_confirmation()

        # 基于 state/subtype 冻结基础模式；后续 override（犹豫/降级）统一在 effective_mode 上做
        new_base = self.mode_from_state(self.current_state, self.current_subtype)
        # The state node may briefly classify a low-score but still visible
        # target as Sus-B because its context boxes remain strong.  Preserve a
        # recently confirmed target ray through those one-frame dips instead
        # of switching the vehicle to an unrelated context midpoint.
        target_recent = self.target_tracking_active(now)
        if new_base == CATCH_CTX_MODE and target_recent:
            new_base = CATCH_TARGET_MODE
        elif new_base == CATCH_CTX_MODE:
            self.ctx_state_hold_until = max(
                self.ctx_state_hold_until,
                now + self.context_state_hold_time,
            )
        elif (
            not target_recent
            and now < self.ctx_state_hold_until
            and self.base_mode == CATCH_CTX_MODE
        ):
            rospy.loginfo_throttle(
                2.0,
                "GoalManager: hold context mode through transient state=%s "
                "subtype=%s for %.1fs",
                str(self.current_state),
                self.current_subtype or "-",
                max(0.0, self.ctx_state_hold_until - now),
            )
            new_base = CATCH_CTX_MODE
        if new_base != self.base_mode:
            rospy.loginfo("GoalManager: base_mode %s -> %s (state=%s subtype=%s)",
                          self.base_mode, new_base, str(self.current_state), self.current_subtype)
        self.base_mode = new_base
        self.effective_mode = new_base

        if self.last_state is None or prev_state != msg.state or prev_subtype != self.current_subtype:
            # 立即打断更新
            self.next_update_time = 0.0
        self.last_state = msg.state

    def on_dets(self, msg: LsteDetections):
        self.latest_dets = msg
        # Once the task has reached its terminal state, keep recording the
        # latest detector message for diagnostics but do not refresh target
        # follow/observation-hold state.  Otherwise a still-visible object
        # continually extends the hold timer and produces misleading
        # post-completion activity even though on_timer has stopped publishing
        # goals.
        if self.task_done_published:
            return
        det = self.target_detection_for_track(msg)
        if det is None:
            return
        score = float(det.score)
        box_size = max(float(det.w), float(det.h))
        if score < self.target_follow_min_score or box_size < self.target_follow_min_box_size:
            return

        now = rospy.Time.now().to_sec()
        source_stamp = (
            msg.header.stamp.secs,
            msg.header.stamp.nsecs,
            int(getattr(msg.header, "seq", 0)),
        )
        # A detector result may be observed by the Goal Manager several times
        # before the next inference completes.  It is evidence only once;
        # reusing it must not move the target goal again.
        if self.target_last_detection_stamp == source_stamp:
            return
        self.target_last_detection_stamp = source_stamp
        self.target_observation_epoch += 1
        if not self.target_blocked:
            self.target_execution_state = "TARGET_CANDIDATE"
        within_window = (
            self.target_candidate_last_seen is not None
            and now - self.target_candidate_last_seen <= self.target_follow_confirm_window
        )
        if not within_window:
            self.target_candidate_hits = 0
            self.target_candidate_score_sum = 0.0
            self.target_candidate_anchor_cx = None
            self.target_candidate_anchor_cy = None
        # A high-score / large box is trustworthy even when it drifts across
        # the image as the robot moves past it.  Only a weak speck has to stay
        # spatially stable to keep accumulating votes, otherwise a passing
        # robot would reset the candidate counter every frame and never commit
        # to approaching the target it can already see.
        strong_detection = (
            score >= max(0.40, self.target_done_min_score)
            or max(float(det.w), float(det.h)) >= 0.05
        )
        center_delta = float("inf")
        if (
            within_window
            and self.target_candidate_anchor_cx is not None
            and self.target_candidate_anchor_cy is not None
        ):
            center_delta = math.hypot(
                float(det.cx) - self.target_candidate_anchor_cx,
                float(det.cy) - self.target_candidate_anchor_cy,
            )
            if (
                center_delta > self.target_follow_spatial_tolerance
                and not strong_detection
            ):
                # A new image ray is a new candidate, not another vote for the
                # previous object. This blocks one-frame associations that
                # alternate between a mug-sized speck and a real target.
                self.target_candidate_hits = 0
                self.target_candidate_score_sum = 0.0
        if self.target_candidate_source_stamp != source_stamp:
            self.target_candidate_hits += 1
            self.target_candidate_score_sum += score
        self.target_candidate = self.clone_detection(det)
        if not self.target_track_label:
            self.target_track_label = (det.label or "").strip().lower()
            self.target_track_sequence += 1
            self.target_track_id = "%s:%s:%d" % (
                self.current_task_id or "task",
                self.target_track_label or "target",
                self.target_track_sequence,
            )
        self.target_candidate_last_seen = now
        self.target_candidate_source_stamp = source_stamp
        self.target_candidate_anchor_cx = float(det.cx)
        self.target_candidate_anchor_cy = float(det.cy)
        candidate_average_score = self.target_candidate_score_sum / max(
            1, self.target_candidate_hits
        )

        # Smooth the normalized image centre and then transform that single
        # filtered ray into odom.  Limit the per-frame heading change as a
        # second guard against a one-frame false association while allowing a
        # genuine turn to converge over a few detector frames.
        if self.target_filtered_cx is None:
            self.target_filtered_cx = float(det.cx)
            self.target_filtered_cy = float(det.cy)
        else:
            alpha = self.target_bbox_alpha
            self.target_filtered_cx += alpha * (float(det.cx) - self.target_filtered_cx)
            self.target_filtered_cy += alpha * (float(det.cy) - self.target_filtered_cy)
        filtered_det = self.clone_detection(det)
        filtered_det.cx = self.target_filtered_cx
        filtered_det.cy = self.target_filtered_cy
        raw_heading = self.det_heading_world(det)
        filtered_heading = self.det_heading_world(filtered_det)
        if filtered_heading is not None:
            if self.target_filtered_heading is None:
                self.target_filtered_heading = filtered_heading
            else:
                delta = wrap_angle(filtered_heading - self.target_filtered_heading)
                delta = max(
                    -self.target_max_heading_step,
                    min(self.target_max_heading_step, delta),
                )
                self.target_filtered_heading = wrap_angle(
                    self.target_filtered_heading + self.target_heading_alpha * delta
                )
            self.target_last_heading = self.target_filtered_heading
        elif raw_heading is not None and self.target_filtered_heading is None:
            self.target_filtered_heading = raw_heading
            self.target_last_heading = raw_heading

        # Do not build a new navigation goal here.  This callback receives
        # asynchronous detector frames while TEB is optimizing its current
        # band.  Re-projecting the image ray from the robot pose at this point
        # changes the action goal during motion and creates an avoidable
        # brake/turn pulse.  ``goal_from_target_follow`` commits the next
        # segment only after the current action is terminal.

        high_quality = score >= max(0.40, self.target_done_min_score) or box_size >= 0.05
        weak_quality_confirmed = (
            self.target_candidate_hits >= self.target_follow_weak_confirm_hits
            and candidate_average_score >= self.target_follow_weak_min_average_score
        )
        if (
            high_quality and self.target_candidate_hits >= self.target_follow_confirm_hits
        ) or weak_quality_confirmed:
            if not self.target_follow_confirmed:
                rospy.loginfo(
                    "GoalManager: target follow confirmed hits=%d avg_score=%.3f "
                    "score=%.3f box=(%.3f,%.3f) center_delta=%.3f",
                    self.target_candidate_hits,
                    candidate_average_score,
                    score,
                    float(det.w),
                    float(det.h),
                    center_delta if math.isfinite(center_delta) else float("nan"),
                )
            self.target_follow_confirmed = True
            self.target_last_seen = now
        else:
            rospy.loginfo_throttle(
                2.0,
                "GoalManager: target candidate hits=%d/%d avg_score=%.3f "
                "score=%.3f box=(%.3f,%.3f) center_delta=%.3f",
                self.target_candidate_hits,
                self.target_follow_weak_confirm_hits,
                candidate_average_score,
                score,
                float(det.w),
                float(det.h),
                center_delta if math.isfinite(center_delta) else float("nan"),
            )

        # Observation is deliberately decoupled from motion.  The detector
        # keeps contributing fresh evidence while the committed TEB segment is
        # running; only maybe_publish_task_done() can stop the vehicle after
        # the independent close-target completion contract is satisfied.

    def on_scores(self, msg: LsteScores):
        self.latest_scores = msg

    def on_task(self, msg: LsteTask):
        task_id = (msg.task_id or "").strip()
        if task_id != self.current_task_id:
            previous = self.current_task_id or "<none>"
            self.current_task_id = task_id
            self.task_done_published = False
            self.reset_target_close_confirmation()
            self.clear_target_memory()
            self.set_navigation_hold(False, "new_task")
            # task_done is latched so consumers such as the velocity mux must
            # see an explicit false when a new task starts.
            self.pub_task_done.publish(Bool(data=False))
            rospy.loginfo("GoalManager: task changed %s -> %s; clear task_done", previous, task_id)
        self.latest_task = msg

    def on_pose(self, msg: Pose2D):
        self.latest_pose = msg
        if self.start_pose is None:
            self.start_pose = msg
            self.init_headings(msg)
        elif self.headings is None:
            self.init_headings(msg)

    def on_cam_info(self, msg: CameraInfo):
        try:
            self.camera_model.fromCameraInfo(msg)
            self.camera_info = msg
            self.camera_frame = msg.header.frame_id or self.camera_frame
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Failed to load camera info: %s", exc)

    def on_depth(self, msg: Image):
        self.depth_image = msg
        self.depth_stamp = msg.header.stamp

    def on_frontier(self, msg: Vector3Stamped):
        self.latest_frontier = msg

    def on_frontiers(self, msg: LsteFrontiers):
        self.latest_frontiers = msg

    def on_global_frontier(self, msg: PoseStamped):
        frame = (msg.header.frame_id or "odom").strip().lstrip("/")
        if frame not in ("odom", "map"):
            rospy.logwarn_throttle(
                3.0, "Ignoring global frontier in unsupported frame: %s", msg.header.frame_id
            )
            return
        self.latest_global_frontier_goal = msg
        self.last_frontier_goal = msg

    def on_global_frontier_status(self, message: String):
        """Synchronize mission ownership when the explorer abandons a route."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        route_kind = str(payload.get("route_kind", "")).strip().lower()
        if route_kind in (
            "frontier_connector",
            "frontier_turn_connector",
            "frontier_endpoint",
        ):
            self.global_frontier_route_kind = route_kind
        if payload.get("event") == "replan_ready":
            request_id = int(payload.get("replan_request_id", 0) or 0)
            if (
                self.frontier_replan_pending_id > 0
                and request_id == self.frontier_replan_pending_id
            ):
                self.frontier_replan_ready_id = request_id
                self.next_update_time = 0.0
                self.publish_goal_arbitration(
                    "frontier_replan_ready",
                    replan_request_id=request_id,
                    reason=str(payload.get("reason", "unknown")),
                    goal=payload.get("goal"),
                )
                rospy.loginfo(
                    "GoalManager: received fresh frontier replan id=%d goal=%s",
                    request_id,
                    payload.get("goal"),
                )
            return
        if payload.get("event") != "route_invalidated":
            return
        with_status_goal = payload.get("goal")
        rospy.logwarn(
            "GoalManager: invalidating frontier route reason=%s goal=%s",
            payload.get("reason", "unknown"),
            with_status_goal,
        )
        # The explorer will publish a replacement only after it validates a
        # new connected branch.  Clear the cached mission goal now so the
        # manager cannot re-latch the abandoned pose while waiting.
        self.latest_global_frontier_goal = None
        self.last_frontier_goal = None
        self.global_frontier_route_kind = "frontier_endpoint"
        self.teb_terminal_goal = None
        self.teb_frontier_goal_history = []
        self.frontier_goal_sent_at = None
        if (
            self.last_goal_source == "global_slam_frontier"
            and not self.target_tracking_active(rospy.Time.now().to_sec())
        ):
            self.last_goal = None
            self.last_goal_source = "waiting_global_slam_frontier"
        self.goal_source = "waiting_global_slam_frontier"
        self.next_update_time = 0.0

    def request_global_frontier_replan(self, reason: str, **fields):
        """Require a frontier branch recomputed from the current robot pose.

        This is an ownership boundary, rather than a timing heuristic.  A
        target-route terminal must never resume the endpoint that was cached
        before the target diverted the robot.  The explorer replies with a
        matching ``replan_ready`` status before this manager accepts another
        frontier pose.
        """
        if not self.global_frontier_enabled:
            return 0
        if (
            self.frontier_replan_pending_id > 0
            and self.frontier_replan_ready_id != self.frontier_replan_pending_id
        ):
            return self.frontier_replan_pending_id
        self.frontier_replan_request_id += 1
        request_id = self.frontier_replan_request_id
        self.frontier_replan_pending_id = request_id
        self.frontier_replan_ready_id = 0
        # Do not accidentally re-publish a latched goal while the frontier
        # explorer is rebuilding its connected route from the latest map.
        self.latest_global_frontier_goal = None
        self.last_frontier_goal = None
        payload = {
            "event": "replan_request",
            "request_id": request_id,
            "reason": str(reason),
        }
        payload.update(fields)
        self.pub_global_frontier_replan.publish(
            String(data=json.dumps(payload, sort_keys=True))
        )
        self.publish_goal_arbitration(
            "frontier_replan_requested",
            replan_request_id=request_id,
            reason=str(reason),
            **fields
        )
        rospy.loginfo(
            "GoalManager: requested fresh global frontier replan id=%d reason=%s",
            request_id,
            reason,
        )
        return request_id

    def on_teb_goal_terminal(self, msg: PoseStamped):
        """Release exactly one committed segment after a TEB terminal result.

        Frontier and visual-target segments share the same move_base action,
        but they must not share a rolling setpoint lifecycle.  A target
        detector frame is allowed to update the bearing while its action is
        active; this callback is the only event that authorizes the next
        target segment.
        """
        # TEB stays alive on its own mux input for hot switching.  Its terminal
        # topic must not release an SA-PPO frontier commitment, otherwise a
        # background TEB action can make the RL goal jump every second.
        if self.controller_mode != "teb":
            return
        if self.last_goal is None:
            return
        if self.last_goal_source.startswith("target_"):
            if msg.header.frame_id and msg.header.frame_id != self.last_goal.header.frame_id:
                return
            distance = math.hypot(
                msg.pose.position.x - self.last_goal.pose.position.x,
                msg.pose.position.y - self.last_goal.pose.position.y,
            )
            if distance > max(self.global_frontier_update_radius, 0.30):
                rospy.logwarn_throttle(
                    3.0,
                    "GoalManager: ignoring TEB target terminal for stale goal delta=%.2fm",
                    distance,
                )
                return
            target_distance = float("inf")
            if self.target_last_goal is not None:
                target_distance = math.hypot(
                    msg.pose.position.x - self.target_last_goal.pose.position.x,
                    msg.pose.position.y - self.target_last_goal.pose.position.y,
                )
            if target_distance <= max(self.target_goal_reached_radius, 0.30):
                self.target_segment_terminal_ready = True
                self.target_goal_detection_stamp = None
                self.target_execution_state = "TARGET_CANDIDATE"
                self.next_update_time = 0.0
                rospy.loginfo(
                    "GoalManager: TEB terminal committed target segment "
                    "(%.2f,%.2f); next segment waits for timer boundary",
                    msg.pose.position.x,
                    msg.pose.position.y,
                )
            return
        if self.last_goal_source != "global_slam_frontier":
            return
        # Match against the committed frontier action history.  ``last_goal``
        # is the newest mission intent and may already point at the next route
        # segment when move_base reports completion of the previous one.
        candidates = [
            candidate for candidate in self.teb_frontier_goal_history
            if not msg.header.frame_id
            or candidate.header.frame_id == msg.header.frame_id
        ]
        if not candidates:
            candidates = [self.last_goal]
        matched = min(
            candidates,
            key=lambda candidate: math.hypot(
                msg.pose.position.x - candidate.pose.position.x,
                msg.pose.position.y - candidate.pose.position.y,
            ),
        )
        distance = math.hypot(
            msg.pose.position.x - matched.pose.position.x,
            msg.pose.position.y - matched.pose.position.y,
        )
        if distance > max(self.global_frontier_update_radius, 0.30):
            rospy.logwarn_throttle(
                3.0,
                "GoalManager: ignoring TEB frontier terminal for unknown action "
                "goal delta=%.2fm latest_delta=%.2fm",
                distance,
                math.hypot(
                    msg.pose.position.x - self.last_goal.pose.position.x,
                    msg.pose.position.y - self.last_goal.pose.position.y,
                ),
            )
            return
        self.teb_terminal_goal = msg
        if self.target_blocked:
            # A successful frontier action is the explicit map-progress event
            # that reopens a previously failed visual hypothesis.  A detector
            # frame alone is not enough: it contains no evidence that the
            # blocked corridor has become navigable.
            blocked_goal = self.target_blocked_goal
            blocked_reason = self.target_blocked_reason
            self.target_blocked = False
            self.target_blocked_goal = None
            self.target_blocked_since = None
            self.target_blocked_reason = ""
            self.target_execution_state = "TARGET_CANDIDATE"
            self.publish_goal_arbitration(
                "target_route_released",
                reason="frontier_progress",
                blocked_reason=blocked_reason,
                blocked_goal=(
                    None
                    if blocked_goal is None
                    else [
                        round(float(blocked_goal.pose.position.x), 3),
                        round(float(blocked_goal.pose.position.y), 3),
                    ]
                ),
                frontier_goal=[
                    round(float(msg.pose.position.x), 3),
                    round(float(msg.pose.position.y), 3),
                ],
            )
            rospy.loginfo(
                "GoalManager: frontier progress reopens blocked target route "
                "reason=%s frontier=(%.2f,%.2f)",
                blocked_reason,
                msg.pose.position.x,
                msg.pose.position.y,
            )
        rospy.loginfo(
            "GoalManager: TEB terminal matched frontier goal (%.2f,%.2f); "
            "next frontier update may replace it",
            msg.pose.position.x,
            msg.pose.position.y,
        )

    def on_teb_goal_failure(self, message: String):
        """Release a failed visual action to the map planner.

        This is deliberately an event-driven state transition.  No retry
        timer, speed parameter, or detector frame may re-submit the same
        target until a frontier action has made map progress.
        """
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or self.controller_mode != "teb":
            return
        if self.last_goal is None:
            return
        raw_goal = payload.get("goal")
        failure_goal = None
        if isinstance(raw_goal, (list, tuple)) and len(raw_goal) >= 2:
            failure_goal = self.make_goal_pose(
                (float(raw_goal[0]), float(raw_goal[1]), 0.0), 0.0
            )
            failure_goal.header.frame_id = str(
                payload.get("goal_frame", "map") or "map"
            ).strip().lstrip("/") or "map"
        failure_track_id = str(payload.get("target_track_id", "")).strip()
        if (
            failure_track_id
            and self.target_track_id
            and failure_track_id != self.target_track_id
        ):
            rospy.loginfo(
                "GoalManager: ignore target failure from obsolete track=%s current=%s",
                failure_track_id,
                self.target_track_id,
            )
            return
        # The mission layer can queue a short reacquisition pose while the
        # original target action is still executing. Match a failure against
        # the committed target segment, not that pending pose.
        associated_goal = self.target_last_goal
        if associated_goal is None and self.last_goal_source.startswith("target_"):
            associated_goal = self.last_goal
        if associated_goal is None:
            return
        target_delta = None
        if failure_goal is not None:
            target_delta = self.pose_distance(failure_goal, associated_goal)
        if (
            target_delta is not None
            and target_delta > max(self.target_goal_reached_radius, 0.75)
            and not failure_track_id
        ):
            rospy.logwarn(
                "GoalManager: ignore stale target failure delta=%.2fm current=(%.2f,%.2f)",
                target_delta,
                associated_goal.pose.position.x,
                associated_goal.pose.position.y,
            )
            return

        now = rospy.Time.now().to_sec()
        blocked_goal = copy.deepcopy(associated_goal)
        self.target_blocked = True
        self.target_blocked_goal = blocked_goal
        self.target_blocked_since = now
        self.target_blocked_reason = str(payload.get("reason", "unknown"))
        self.target_failure_count += 1
        self.target_execution_state = "TARGET_BLOCKED"
        # Preserve the visual evidence/heading for a later re-plan, but remove
        # the failed segment itself so the next target commit is generated from
        # the robot's then-current pose after frontier progress.
        self.target_last_goal = None
        self.target_goal_detection_stamp = None
        self.target_segment_terminal_ready = False
        self.target_cache_advances = 0
        self.effective_mode = EXPLORE_SUS_C_MODE
        self.pub_access_mode.publish(String(data=self.effective_mode))
        self.goal_source = "target_route_failed"
        self.next_update_time = 0.0
        self.publish_goal_arbitration(
            "target_route_failed",
            reason=self.target_blocked_reason,
            status=str(payload.get("status", "unknown")),
            target_epoch=int(payload.get("target_epoch", self.target_observation_epoch)),
            failure_count=int(self.target_failure_count),
            blocked_goal=[
                round(float(blocked_goal.pose.position.x), 3),
                round(float(blocked_goal.pose.position.y), 3),
            ],
            blocked_goal_frame=blocked_goal.header.frame_id,
        )
        self.request_global_frontier_replan(
            "target_route_failed",
            target_track_id=failure_track_id or self.target_track_id,
            blocked_goal=[
                round(float(blocked_goal.pose.position.x), 3),
                round(float(blocked_goal.pose.position.y), 3),
            ],
        )
        rospy.logwarn(
            "GoalManager: target route failed; wait for a frontier branch "
            "recomputed from the current pose"
        )

    def on_controller_mode(self, msg: String):
        mode = (msg.data or "").strip().lower()
        if mode in ("sappo", "teleop", "teb"):
            if mode != self.controller_mode:
                rospy.loginfo(
                    "GoalManager: controller mode %s -> %s", self.controller_mode, mode
                )
            self.controller_mode = mode

    def on_scan(self, msg: LaserScan):
        self.latest_scan = msg

    def on_cmd_vel(self, msg: Twist):
        self.latest_cmd_vel = msg

    def on_access_mode(self, msg: UInt8):
        try:
            self.access_mode = int(msg.data)
        except Exception:
            self.access_mode = 0

    def on_access_goal(self, msg: PoseStamped):
        self.access_backtrack_goal = msg

    def on_fixed_goal_click(self, msg: PoseStamped):
        """Replace the fixed target from the Gazebo Shift-click UI.

        Gazebo Classic reports world coordinates.  In this simulator wheel
        odometry uses that same world origin, so the coordinates are valid for
        the odom-framed global-goal contract.  Real-robot callers should leave
        this override disabled unless they provide odom-frame poses.
        """
        previous = self.fixed_goal
        yaw = self.yaw_from_pose(msg)
        self.fixed_goal = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            previous[2] if yaw is None else yaw,
        )
        self.next_update_time = 0.0
        rospy.logwarn(
            "Fixed global goal replaced by click: (%.2f, %.2f) -> (%.2f, %.2f)",
            previous[0], previous[1], self.fixed_goal[0], self.fixed_goal[1],
        )

    # -------------------- Timer --------------------
    def on_timer(self, _event):
        now = rospy.Time.now().to_sec()
        if self.navigation_hold_active and (
            now >= self.target_observation_hold_until
            or self.effective_mode != CATCH_TARGET_MODE
        ):
            self.set_navigation_hold(False, "observation_window_complete")
        if self.global_goal_source == "fixed":
            # A fixed target must not wait for state, detections, frontier
            # output, or /rbt_pose.  Publishing it at a bounded rate lets late
            # controller subscribers recover while never changing its value.
            if self.last_goal is None or now >= self.next_update_time:
                self.goal_source = "fixed_config"
                self.effective_mode = "fixed_goal_mode"
                self.publish_goal(
                    self.make_goal_pose(self.fixed_goal, self.fixed_goal[2]),
                    force_republish=True,
                )
                self.next_update_time = now + self.fixed_goal_publish_period
            return
        self.maybe_publish_task_done(now)
        if self.task_done_published:
            # Keep the final goal stable after visual completion. The mux has
            # already been told to stop, so recomputing visual rays here could
            # only create confusing post-completion motion.
            return
        if self.latest_pose is None:
            return

        # With online global coverage enabled, wait for the first map-connected
        # frontier rather than moving an arbitrary 10 m from the spawn pose.
        # That prevents the controller from entering a corridor before SLAM has
        # enough scans to judge whether it is useful or safe to explore.
        if self.last_goal is None and self.start_pose is not None:
            if self.global_frontier_enabled:
                goal = self.compute_goal()
                if goal is not None:
                    self.publish_goal(goal)
                    self.next_update_time = now + self.pass_period
                return
            goal = self.build_goal_from_pose(self.start_pose, self.forward_dist)
            if goal:
                self.goal_source = "startup_forward"
                self.publish_goal(goal)
                self.next_update_time = now + self.pass_period
            return

        period = self.state_period()
        if now < self.next_update_time and not self.force_update_due_state():
            return

        goal = self.compute_goal()
        if goal is None:
            return  # 保持原 goal，不发布

        self.publish_goal(goal)
        self.next_update_time = now + period

    def force_update_due_state(self) -> bool:
        if self.latest_state_msg is None:
            return False
        if self.last_goal is None:
            return True
        # 如果 state/subtype 发生变化则强制更新（在 on_state 已经 next_update_time=0）
        return False

    def init_headings(self, pose: Pose2D):
        base = wrap_angle(pose.theta)
        self.headings = [
            base,
            wrap_angle(base + math.pi * 0.5),
            wrap_angle(base + math.pi),
            wrap_angle(base + math.pi * 1.5),
        ]
        self.last_dir_idx = 0

    # -------------------- Goal computation --------------------
    def state_period(self) -> float:
        # Vision-following goals must track at the same cadence as confirmed
        # detections.  The older 2 s state cadence made a moving robot chase a
        # stale image ray.
        if self.effective_mode in (CATCH_TARGET_MODE, CATCH_CTX_MODE):
            return self.follow_goal_publish_period
        if self.global_frontier_enabled:
            return self.global_frontier_period
        if self.current_state == STATE_LOCKED:
            return self.locked_period
        if self.current_state == STATE_SUSPICIOUS:
            if self.current_subtype == "Sus-C":
                return self.sus_c_period
            if self.current_subtype == "Sus-A":
                return self.sus_a_period
            if self.current_subtype == "Sus-B":
                return self.sus_b_period
        return self.pass_period

    def compute_goal(self) -> Optional[PoseStamped]:
        # A global SLAM frontier is a map-connected exploration waypoint. It
        # dominates the old local GP/access recovery until a visual target ray
        # is available; visual target following remains higher priority.
        now = rospy.Time.now().to_sec()
        # A fresh target ray remains the only reason to suppress global map
        # coverage. Context-only states previously let a chair/monitor pair
        # keep the robot in one corridor indefinitely, even though the target
        # was not visible. Global coverage is therefore the fallback for every
        # non-target state; as soon as target detections resume, the existing
        # target-follow path takes priority again.
        target_recent = self.target_tracking_active(now)
        if target_recent:
            # Never let legacy access-topology recovery preempt a current (or
            # just-lost) visual target ray.  goal_from_target_follow preserves
            # the last short visual-servo goal through a brief detector gap.
            target_goal = self.goal_from_target_follow(now)
            if target_goal is not None:
                return target_goal
        if not target_recent:
            # With online SLAM enabled, context boxes are inspection evidence,
            # not a second global planner. Keep the map-connected waypoint as
            # the sole exploration commitment unless an actual target ray is
            # confirmed or the caller explicitly opts into context preemption.
            if self.global_frontier_enabled and not self.global_frontier_preempt_context:
                reacquire_goal = self.target_reacquisition_goal(now)
                if reacquire_goal is not None:
                    return reacquire_goal
                global_frontier = self.fresh_global_frontier_goal(now)
                if global_frontier is not None:
                    self.goal_source = "global_slam_frontier"
                    return global_frontier
            # Context detections are deliberately lower priority than an
            # active map-connected route.  The state node can briefly enter
            # Sus-B when a monitor/desk pair is visible; replacing the
            # frontier waypoint at that instant made the controller turn back
            # into the already explored corridor.  Release the commitment only
            # after the robot is close to the current waypoint.  A confirmed
            # target above still preempts this guard immediately.
            # ``last_goal`` is the newest mission intent and may already be a
            # context observation while TEB is still executing a committed
            # frontier action. Use the frontier publication history as the
            # execution boundary so context evidence cannot overwrite a live
            # map route before its terminal result.
            active_frontier_goal = None
            if self.last_goal_source == "global_slam_frontier":
                active_frontier_goal = self.last_goal
            elif (
                self.teb_terminal_goal is None
                and self.teb_frontier_goal_history
            ):
                active_frontier_goal = self.teb_frontier_goal_history[-1]
            if (
                self.global_frontier_enabled
                and not self.global_frontier_preempt_context
                and active_frontier_goal is not None
                and self.effective_mode == CATCH_CTX_MODE
                and self.latest_pose is not None
            ):
                distance_to_frontier = self.goal_robot_distance(active_frontier_goal)
                if distance_to_frontier is None:
                    distance_to_frontier = float("inf")
                release_radius = min(
                    self.global_frontier_update_radius,
                    self.global_frontier_jump_release_radius,
                )
                if distance_to_frontier > release_radius:
                    self.goal_source = "global_slam_frontier"
                    rospy.loginfo_throttle(
                        3.0,
                        "GoalManager: hold frontier during context state "
                        "distance=%.2fm release_radius=%.2fm",
                        distance_to_frontier,
                        release_radius,
                    )
                    return active_frontier_goal
            reacquire_goal = self.target_reacquisition_goal(now)
            if reacquire_goal is not None:
                return reacquire_goal
            # A task-specific pair, for example monitor + monitor for the
            # yellow-cup task, is stronger evidence than an arbitrary map
            # frontier. Do not react to a lone chair, door, or fire hydrant:
            # pick_ctx_pair requires both configured context detections.
            if (
                self.effective_mode == CATCH_CTX_MODE
                and now >= self.ctx_cooldown_until
            ):
                left_ctx, right_ctx = self.pick_ctx_pair()
                if left_ctx is not None and right_ctx is not None:
                    return self.goal_from_ctx_follow(now)
            global_frontier = self.fresh_global_frontier_goal(now)
            if global_frontier is not None:
                self.goal_source = "global_slam_frontier"
                return global_frontier
            if self.global_frontier_enabled:
                # Do not fall back to an arbitrary local ray while online SLAM
                # is starting or unavailable. A new controller then waits for
                # a connected map route instead of moving blindly.
                self.goal_source = "waiting_global_slam_frontier"
                return None

        # Access-topology recovery is valid only while searching.  A recovery
        # waypoint must never replace a visually confirmed target-follow goal.
        if (
            not self.global_frontier_enabled
            and
            self.effective_mode in (EXPLORE_PASS_MODE, EXPLORE_SUS_C_MODE)
            and self.access_mode in (1, 2)
            and self.access_backtrack_goal is not None
        ):
            self.goal_source = "access_backtrack"
            return self.access_backtrack_goal

        # 基于 effective_mode 分发（后续犹豫/降级都在 effective_mode 上动手）
        if self.effective_mode == EXPLORE_PASS_MODE:
            return self.goal_from_frontiers_prior()
        if self.effective_mode == EXPLORE_SUS_C_MODE:
            return self.goal_from_frontiers_prior()
        if self.effective_mode == CATCH_TARGET_MODE:
            return self.goal_from_target_follow(now)
        if self.effective_mode == CATCH_CTX_MODE:
            return self.goal_from_ctx_follow(now)
        # 未知模式：保持现状
        rospy.logwarn_throttle(5.0, "GoalManager: unknown mode=%s", str(self.effective_mode))
        return None

    def mode_from_state(self, state: int, subtype: str) -> str:
        """冻结映射表：state/subtype -> 基础 goal 模式。"""
        if state == STATE_LOCKED:
            return CATCH_TARGET_MODE
        if state == STATE_SUSPICIOUS:
            if subtype == "Sus-A":
                return CATCH_TARGET_MODE
            if subtype == "Sus-B":
                return CATCH_CTX_MODE
            if subtype == "Sus-C":
                return EXPLORE_SUS_C_MODE
        # 默认 PASS
        return EXPLORE_PASS_MODE

    # -------------------- Follow helpers --------------------
    def commit_target_segment(self, now: float, source: str, advance: bool = False) -> Optional[PoseStamped]:
        """Commit one visual-servo horizon as an atomic navigation intent.

        The segment is deliberately created from the latest robot pose only at
        a lifecycle boundary (initial target lock or a terminal action result),
        never from an arbitrary detector callback.  This keeps the world-frame
        goal stable while TEB optimizes and executes the current route.
        """
        if self.latest_pose is None or self.target_last_heading is None:
            return None
        distance = self.target_segment_distance()
        x = self.latest_pose.x + distance * math.cos(self.target_last_heading)
        y = self.latest_pose.y + distance * math.sin(self.target_last_heading)
        candidate_goal_odom = self.make_goal_pose(
            (x, y, 0.0), self.target_last_heading
        )
        # Visual rays are measured in odom, but a committed TEB action must
        # have one stable world-frame identity. Transform once at the mission
        # boundary; do not let subsequent SLAM map->odom updates reinterpret
        # an active target segment.
        candidate_goal = self._pose_in_frame(candidate_goal_odom, "map")
        if candidate_goal is None:
            self.target_execution_state = "TARGET_ROUTE_PENDING"
            route_status = None
        else:
            route_status = self.validate_target_route(candidate_goal, now)
        if route_status is not True:
            self.goal_source = (
                "global_slam_frontier"
                if self.last_goal is not None
                and self.last_goal_source == "global_slam_frontier"
                else "target_waiting_navfn_route"
            )
            self.publish_goal_arbitration(
                "target_route_held",
                reason=(
                    "navfn_empty_plan"
                    if route_status is False
                    else "route_validation_unavailable"
                ),
                goal=[round(float(x), 3), round(float(y), 3)],
                fallback_goal=(
                    None
                    if self.last_goal is None
                    else [
                        round(float(self.last_goal.pose.position.x), 3),
                        round(float(self.last_goal.pose.position.y), 3),
                    ]
                ),
            )
            return None
        self.target_last_goal = candidate_goal
        self.target_execution_state = "TARGET_ROUTE_VALIDATED"
        if advance:
            self.target_cache_advances += 1
        else:
            self.target_cache_advances = 0
        self.target_last_update = now
        self.target_goal_detection_stamp = self.target_last_detection_stamp
        self.target_segment_terminal_ready = False
        self.goal_source = source
        rospy.loginfo(
            "GoalManager: committed target segment source=%s advance=%d/%d "
            "heading=%.3f goal=(%.2f,%.2f)",
            source,
            self.target_cache_advances,
            self.target_cache_max_advances,
            self.target_last_heading,
            x,
            y,
        )
        return self.target_last_goal

    def goal_from_target_follow(self, now: float) -> Optional[PoseStamped]:
        if self.target_blocked:
            self.target_execution_state = "TARGET_BLOCKED"
            self.goal_source = "target_route_blocked"
            if self.global_frontier_enabled:
                frontier = self.fresh_global_frontier_goal(now)
                if frontier is not None:
                    self.goal_source = "global_slam_frontier"
                    return frontier
                self.request_global_frontier_replan("target_route_blocked")
            # The target action has failed and was cancelled by the bridge.
            # Re-publishing its pose under another source would quietly turn a
            # recovery wait into a retry. Keep the executor idle until the
            # matching frontier replan becomes ready.
            return None
        # The candidate must be confirmed before it can steer the vehicle.
        # A first weak box is retained for diagnostics, but it must not replace
        # a valid frontier route until an independent frame confirms it.
        if not self.target_follow_confirmed:
            # A weak single-frame candidate is evidence to observe, not a
            # reason to replace a map-connected route with a new GP ray. Keep
            # the active frontier goal byte-for-byte stable until the detector
            # confirms the target on an independent frame.
            if self.last_goal is not None and self.last_goal_source == "global_slam_frontier":
                self.goal_source = self.last_goal_source
                return self.last_goal
            self.goal_source = "target_candidate_pending"
            return self.goal_from_frontiers_prior()

        if self.target_last_goal is None:
            candidate = self.commit_target_segment(now, "target_follow")
            if candidate is not None:
                return candidate
            # A rejected visual ray must not interrupt the map-connected
            # exploration transaction.  Keep the detector track alive so the
            # same target can be reconsidered after SLAM reveals a route.
            if (
                self.last_goal is not None
                and self.last_goal_source == "global_slam_frontier"
            ):
                self.goal_source = "global_slam_frontier"
                return self.last_goal
            return None

        if self.target_last_goal is not None:
            self.target_reacquire_goal = None
            self.target_reacquire_started = None
            self.target_reacquire_attempts = 0
            current_distance = float("inf")
            if self.latest_pose is not None:
                measured_distance = self.goal_robot_distance(self.target_last_goal)
                if measured_distance is not None:
                    current_distance = measured_distance
            if current_distance <= self.target_goal_reached_radius:
                # TEB owns an atomic move_base action.  Physical proximity is
                # not enough to mutate its goal: wait for the matching
                # SUCCEEDED terminal event so a detector frame cannot race the
                # action callback and create a stop/restart pulse.
                if self.controller_mode == "teb" and not self.target_segment_terminal_ready:
                    self.goal_source = "target_waiting_terminal"
                    return self.target_last_goal

                # A detector gap is common with WeDetect-Large.  Permit only a
                # bounded continuation on the last observed bearing; then
                # release the stale visual lock to the map-connected route.
                if (
                    self.target_last_heading is not None
                    and self.target_cache_advances < self.target_cache_max_advances
                ):
                    candidate = self.commit_target_segment(
                        now, "target_terminal_advance", advance=True
                    )
                    if candidate is not None:
                        return candidate
                    # The previous visual segment reached its terminal
                    # boundary, but its next ray is not map-connected. Drop
                    # the stale visual action and resume exploration instead
                    # of asking the bridge to retry the same blocked pose.
                    frontier = self.release_target_follow_to_frontier(now)
                    if frontier is not None:
                        return frontier
                    self.goal_source = "target_waiting_navfn_route"
                    return self.target_last_goal
                # SA-PPO treats an online visual/frontier segment as complete
                # at its intermediate handoff radius. Once the bounded blind
                # continuation budget is exhausted, returning the same target
                # pose makes the controller wait forever. Release the stale
                # visual lock and resume the last map-connected route.
                frontier = self.release_target_follow_to_frontier(now)
                if frontier is not None:
                    return frontier
                self.goal_source = "target_waiting_fresh_observation"
                return self.target_last_goal

            if self.target_goal_detection_stamp == self.target_last_detection_stamp:
                self.goal_source = "target_follow"
            else:
                self.goal_source = "target_cached"
            return self.target_last_goal

        if self.target_last_seen is not None and (
            now - self.target_last_seen
        ) < self.follow_target_lost_timeout:
            self.goal_source = "target_cached"
            return self.last_goal

        if self.effective_mode == CATCH_TARGET_MODE:
            rospy.loginfo_throttle(
                2.0,
                "GoalManager: target lost >= %.1fs, fallback to explore_sus_c_mode",
                self.follow_target_lost_timeout,
            )
            self.effective_mode = EXPLORE_SUS_C_MODE
            self.pub_access_mode.publish(String(data=self.effective_mode))
            self.clear_target_memory()
        return self.goal_from_frontiers_prior()

    def release_target_follow_to_frontier(self, now: float) -> Optional[PoseStamped]:
        """Release a completed visual segment without leaving a stale goal.

        A frontier saved before the visual approach may now be far behind the
        robot.  Request a new map-connected branch instead of resuming that
        stale endpoint.  The short replan wait is intentional and observable;
        a long unvalidated reverse route is not.
        """
        frontier = self.fresh_global_frontier_goal(now)
        previous_target = self.target_last_goal
        self.clear_target_memory()
        if frontier is None:
            self.request_global_frontier_replan(
                "target_segment_complete",
                previous_target=(
                    None
                    if previous_target is None
                    else [
                        round(float(previous_target.pose.position.x), 3),
                        round(float(previous_target.pose.position.y), 3),
                    ]
                ),
            )
            rospy.logwarn(
                "GoalManager: visual segment reached; waiting for fresh frontier replan"
            )
            return None
        self.goal_source = "global_slam_frontier"
        rospy.loginfo(
            "GoalManager: release completed target segment to fresh frontier "
            "goal=(%.2f,%.2f) previous_target=(%.2f,%.2f)",
            frontier.pose.position.x,
            frontier.pose.position.y,
            previous_target.pose.position.x if previous_target is not None else float("nan"),
            previous_target.pose.position.y if previous_target is not None else float("nan"),
        )
        return frontier

    def publish_goal_arbitration(self, event: str, **fields):
        """Publish mission/execution arbitration decisions for run logs.

        The final-goal topic intentionally carries only executable poses.  A
        separate event stream records why a perception intent was accepted,
        deferred, or rejected, so a stopped vehicle can be diagnosed without
        inferring policy decisions from velocity samples alone.
        """
        payload = {
            "event": str(event),
            "source": str(self.goal_source),
            "mode": str(self.effective_mode),
            "controller_mode": str(self.controller_mode),
            "target_state": str(self.target_execution_state),
            "target_epoch": int(self.target_observation_epoch),
            "target_blocked": bool(self.target_blocked),
            "target_failure_count": int(self.target_failure_count),
        }
        payload.update(fields)
        try:
            self.pub_goal_arbitration.publish(
                String(data=json.dumps(payload, sort_keys=True))
            )
        except (TypeError, ValueError):
            rospy.logwarn_throttle(5.0, "GoalManager: arbitration event serialization failed")

    def _pose_in_frame(self, pose: PoseStamped, target_frame: str) -> Optional[PoseStamped]:
        """Transform a planar pose without introducing a tf message helper."""
        if pose is None:
            return None
        source_frame = (pose.header.frame_id or "odom").strip().lstrip("/") or "odom"
        target_frame = (target_frame or "map").strip().lstrip("/") or "map"
        if source_frame == target_frame:
            return pose
        transform = self.lookup_transform(target_frame, source_frame, rospy.Time(0))
        if transform is None:
            return None
        rotation = transform.transform.rotation
        matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        point = matrix[:3, :3].dot(
            np.array([
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            ], dtype=float)
        )
        translation = transform.transform.translation
        out = PoseStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = target_frame
        out.pose.position.x = float(point[0] + translation.x)
        out.pose.position.y = float(point[1] + translation.y)
        out.pose.position.z = float(point[2] + translation.z)
        source_yaw = self.yaw_from_pose(pose) or 0.0
        transform_yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        qx, qy, qz, qw = quaternion_from_euler(
            0.0, 0.0, wrap_angle(source_yaw + transform_yaw)
        )
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        return out

    def pose_distance(self, first: PoseStamped, second: PoseStamped) -> Optional[float]:
        """Measure two mission poses in one frame for lifecycle matching."""
        if first is None or second is None:
            return None
        first_frame = self._frame_name(first.header.frame_id)
        second_frame = self._frame_name(second.header.frame_id)
        comparable = first
        if first_frame != second_frame:
            comparable = self._pose_in_frame(first, second_frame)
            if comparable is None:
                return None
        return math.hypot(
            float(comparable.pose.position.x) - float(second.pose.position.x),
            float(comparable.pose.position.y) - float(second.pose.position.y),
        )

    def validate_target_route(self, goal: PoseStamped, now: float) -> Optional[bool]:
        """Ask Navfn whether a visual approach point is currently connected.

        Return values are deliberately tri-state: ``True`` is a live route,
        ``False`` is a live empty plan (the target ray is blocked), and
        ``None`` means the planner/TF is not ready.  Both non-true states keep
        the current map route in charge; neither is converted into a blind
        action retry.
        """
        if self.controller_mode != "teb" or not self.target_route_validation:
            return True
        if goal is None:
            return False
        goal_frame = (goal.header.frame_id or "odom").strip().lstrip("/") or "odom"
        goal_xy = (
            round(float(goal.pose.position.x), 3),
            round(float(goal.pose.position.y), 3),
        )
        if now < self.target_route_validation_next_time:
            if self.target_route_validation_last_result == "reachable":
                return True
            if self.target_route_validation_last_result == "blocked":
                return False
            return None
        if self.latest_pose is None:
            return None
        start_odom = self.make_goal_pose(
            (self.latest_pose.x, self.latest_pose.y, 0.0), self.latest_pose.theta
        )
        start_map = self._pose_in_frame(start_odom, "map")
        goal_map = self._pose_in_frame(goal, "map")
        self.target_route_validation_next_time = now + self.target_route_validation_period
        self.target_route_validation_last_goal = goal_xy
        if start_map is None or goal_map is None:
            self.target_route_validation_last_result = "unavailable"
            self.publish_goal_arbitration(
                "target_route_deferred",
                reason="tf_unavailable",
                goal_frame=goal_frame,
                goal=list(goal_xy),
            )
            return None
        try:
            rospy.wait_for_service(
                self.target_route_validation_service,
                timeout=self.target_route_validation_timeout,
            )
        except (rospy.ROSException, rospy.ROSInterruptException):
            self.target_route_validation_last_result = "unavailable"
            self.publish_goal_arbitration(
                "target_route_deferred",
                reason="navfn_service_unavailable",
                service=self.target_route_validation_service,
                goal=list(goal_xy),
            )
            return None
        request = GetPlanRequest()
        request.start = start_map
        request.goal = goal_map
        request.tolerance = self.target_route_validation_tolerance
        try:
            response = self.target_route_validation_client(request)
        except (rospy.ServiceException, rospy.ROSException) as exc:
            self.target_route_validation_last_result = "unavailable"
            self.publish_goal_arbitration(
                "target_route_deferred",
                reason="navfn_service_error",
                service=self.target_route_validation_service,
                error=str(exc),
                goal=list(goal_xy),
            )
            return None
        reachable = bool(response.plan.poses)
        self.target_route_validation_last_result = "reachable" if reachable else "blocked"
        if reachable:
            self.target_route_validation_failures = 0
            self.publish_goal_arbitration(
                "target_route_accepted",
                goal=list(goal_xy),
                plan_poses=len(response.plan.poses),
                plan_frame=response.plan.header.frame_id or "map",
            )
            return True
        self.target_route_validation_failures += 1
        self.publish_goal_arbitration(
            "target_route_rejected",
            reason="navfn_empty_plan",
            goal=list(goal_xy),
            robot=[
                round(float(start_map.pose.position.x), 3),
                round(float(start_map.pose.position.y), 3),
            ],
            failures=self.target_route_validation_failures,
        )
        rospy.logwarn(
            "GoalManager: visual target route rejected by Navfn goal=(%.2f,%.2f) "
            "robot=(%.2f,%.2f) failures=%d; retaining map route",
            goal_map.pose.position.x,
            goal_map.pose.position.y,
            start_map.pose.position.x,
            start_map.pose.position.y,
            self.target_route_validation_failures,
        )
        return False

    def target_reacquisition_goal(self, now: float) -> Optional[PoseStamped]:
        """Make one bounded forward continuation after a target is lost.

        The camera saw the target along ``target_last_heading``.  Continuing
        along that bearing is the only evidence-backed way to bring a small,
        distant object back into view.  The previous reverse-heading sweep
        pointed the vehicle away from the detected object and made the normal
        pipeline abandon valid sightings near desks.
        """
        if (
            self.target_reacquire_duration <= 0.0
            or self.target_reacquire_attempts >= self.target_reacquire_max_attempts
            or self.target_last_seen is None
            or self.target_last_heading is None
            or self.latest_pose is None
        ):
            return None
        if now - self.target_last_seen < self.follow_target_lost_timeout:
            return None
        if self.target_reacquire_started is None:
            heading = self.target_last_heading
            x = self.latest_pose.x + self.target_reacquire_distance * math.cos(heading)
            y = self.latest_pose.y + self.target_reacquire_distance * math.sin(heading)
            self.target_reacquire_goal = self.make_goal_pose((x, y, 0.0), heading)
            self.target_reacquire_started = now
            self.target_reacquire_attempts += 1
            rospy.loginfo(
                "GoalManager: target lost; continue toward last target bearing %d/%d",
                self.target_reacquire_attempts, self.target_reacquire_max_attempts,
            )
        if now - self.target_reacquire_started < self.target_reacquire_duration:
            self.goal_source = "target_reacquisition_sweep"
            return self.target_reacquire_goal
        rospy.loginfo("GoalManager: target reacquisition sweep expired; resume global exploration")
        self.target_reacquire_goal = None
        self.target_reacquire_started = None
        return None

    def fresh_global_frontier_goal(self, now: float) -> Optional[PoseStamped]:
        if (
            self.frontier_replan_pending_id > 0
            and self.frontier_replan_ready_id != self.frontier_replan_pending_id
        ):
            return None
        if not self.global_frontier_enabled or self.latest_global_frontier_goal is None:
            return None
        stamp = self.latest_global_frontier_goal.header.stamp
        if self.global_frontier_max_age > 0.0 and stamp:
            age = now - stamp.to_sec()
            if age > self.global_frontier_max_age:
                return None
        return self.latest_global_frontier_goal

    def maybe_publish_task_done(self, now: float):
        """Close a task from independent fresh visual observations.

        Completion must come from a close target in several fresh detector
        messages, not merely from elapsed state time. LOCKED-only completion
        remains available as an explicit strict mode. Counting distinct
        detector messages prevents the 5 Hz Goal Manager timer from treating
        one stale box as several observations. Once confirmed the velocity
        mux stops all commands atomically, preserving the view instead of
        turning away from a target that has already been reached.
        """
        if self.task_done_published:
            return
        if self.target_done_require_locked and self.current_state != STATE_LOCKED:
            self.reset_target_close_confirmation()
            return
        det = self.target_detection_for_track(self.latest_dets)
        if self.latest_dets is None:
            self.reset_target_close_confirmation()
            return
        stamp = self.latest_dets.header.stamp
        stamp_sec = stamp.to_sec()
        if self.target_done_max_detection_age > 0.0 and now - stamp_sec > self.target_done_max_detection_age:
            self.reset_target_close_confirmation()
            return
        width_close = (
            self.target_done_min_box_width > 0.0
            and det is not None
            and float(det.w) >= self.target_done_min_box_width
        )
        height_close = (
            self.target_done_min_box_height > 0.0
            and det is not None
            and float(det.h) >= self.target_done_min_box_height
        )
        close_enough = (
            det is not None
            and float(det.score) >= self.target_done_min_score
            and (width_close or height_close)
        )
        if not close_enough:
            # A small object at the frame edge flickers below the box
            # threshold for a single detector cycle.  Do not reset the
            # close-target counter on that one dip: only a sustained absence
            # of close evidence (the target left the scene) should restart the
            # confirmation window.  The detector can take a few seconds per
            # frame, so the dip window is the freshness bound plus a frame.
            dip_window = max(self.target_done_max_detection_age, 2.0)
            if (
                self.target_close_last_seen is None
                or now - self.target_close_last_seen > dip_window
            ):
                self.reset_target_close_confirmation()
            return
        stamp_key = (stamp.secs, stamp.nsecs)
        if self.target_close_last_stamp != stamp_key:
            self.target_close_last_stamp = stamp_key
            self.target_close_hits += 1
        self.target_close_last_seen = now
        if self.target_close_since is None:
            self.target_close_since = now
            rospy.loginfo(
                "GoalManager: close target confirmation started score=%.3f box=(%.3f,%.3f) hits=%d/%d",
                float(det.score), float(det.w), float(det.h),
                self.target_close_hits, self.target_done_min_fresh_hits,
            )
            return
        if (
            self.target_close_hits >= self.target_done_min_fresh_hits
            and (now - self.target_close_since) >= self.target_done_min_hold_time
        ):
            rospy.loginfo(
                "GoalManager: close target confirmed hits=%d hold=%.1fs, publish task_done",
                self.target_close_hits, now - self.target_close_since,
            )
            self.pub_task_done.publish(Bool(data=True))
            self.task_done_published = True

    def reset_target_close_confirmation(self):
        self.target_close_since = None
        self.target_close_hits = 0
        self.target_close_last_stamp = None
        self.target_close_last_seen = None

    def clear_target_memory(self):
        self.target_last_seen = None
        self.target_last_goal = None
        self.target_last_update = 0.0
        self.target_last_heading = None
        self.target_filtered_cx = None
        self.target_filtered_cy = None
        self.target_filtered_heading = None
        self.target_last_detection_stamp = None
        self.target_goal_detection_stamp = None
        self.target_segment_terminal_ready = False
        self.target_track_label = ""
        self.target_track_id = ""
        self.target_cache_advances = 0
        self.target_observation_hold_goal = None
        self.target_candidate = None
        self.target_candidate_last_seen = None
        self.target_candidate_source_stamp = None
        self.target_candidate_hits = 0
        self.target_candidate_anchor_cx = None
        self.target_candidate_anchor_cy = None
        self.target_candidate_score_sum = 0.0
        self.target_follow_confirmed = False
        self.target_execution_state = "TARGET_CANDIDATE"
        self.target_observation_epoch = 0
        self.target_blocked = False
        self.target_blocked_goal = None
        self.target_blocked_since = None
        self.target_blocked_reason = ""
        self.target_failure_count = 0
        self.target_observation_hold_until = 0.0
        self.target_route_validation_next_time = 0.0
        self.target_route_validation_failures = 0
        self.target_route_validation_last_result = "not_checked"
        self.target_route_validation_last_goal = None

    def target_segment_distance(self) -> float:
        """Return a controller-appropriate visual pursuit horizon.

        A short visual waypoint is useful for a direct policy, but it is a
        poor contract for TEB: reaching every 1.5 m waypoint completes the
        ``move_base`` action and inserts a zero-velocity interval before the
        detector has produced its next frame. TEB already performs local
        obstacle avoidance, so give it a horizon of at least four arrival
        radii (3 m with the production 0.75 m radius). This is derived from
        the existing arrival contract rather than exposed as another tuning
        parameter; SAPPO and teleop comparison runs retain the old segment
        length.
        """
        if self.controller_mode != "teb":
            return self.follow_target_step_distance
        return max(
            self.follow_target_step_distance,
            4.0 * self.target_goal_reached_radius,
        )

    def goal_from_ctx_follow(self, now: float) -> Optional[PoseStamped]:
        if self.ctx_start_time is None:
            self.ctx_start_time = now

        # ctx 期间检测到 target：直接切到 target 模式
        if self.pick_best_target() is not None:
            self.effective_mode = CATCH_TARGET_MODE
            self.pub_access_mode.publish(String(data=self.effective_mode))
            self.ctx_start_time = None
            return self.goal_from_target_follow(now)

        goal = None
        left_det, right_det = self.pick_ctx_pair()
        if left_det is not None and right_det is not None:
            # 用两框中心的中点方向
            mid_det = LsteDetection()
            mid_det.cx = 0.5 * (float(left_det.cx) + float(right_det.cx))
            mid_det.cy = 0.5 * (float(left_det.cy) + float(right_det.cy))
            mid_det.w = mid_det.h = 0.0
            heading_world = self.det_heading_world(mid_det)
            if heading_world is not None:
                dist = self.clip_distance(heading_world, self.follow_context_step_distance)
                if dist is not None:
                    x = self.latest_pose.x + dist * math.cos(heading_world)
                    y = self.latest_pose.y + dist * math.sin(heading_world)
                    old_distance = float("inf")
                    old_heading = None
                    if self.ctx_last_goal is not None and self.latest_pose is not None:
                        old_distance = math.hypot(
                            self.ctx_last_goal.pose.position.x - self.latest_pose.x,
                            self.ctx_last_goal.pose.position.y - self.latest_pose.y,
                        )
                        old_heading = self.yaw_from_pose(self.ctx_last_goal)
                    heading_delta = (
                        abs(wrap_angle(heading_world - old_heading))
                        if old_heading is not None else float("inf")
                    )
                    # Keep a context inspection segment stable while the robot
                    # is approaching it.  Replacing it on every detector/timer
                    # tick makes the target move with the robot and produces
                    # the same steering chase that previously affected TEB.
                    can_refresh = (
                        self.ctx_last_goal is None
                        or old_distance <= self.ctx_goal_reached_radius
                        or heading_delta >= self.ctx_goal_heading_update_threshold
                    )
                    if can_refresh and (now - self.ctx_last_update) >= self.follow_min_update_period:
                        if self.follow_heading_alpha and old_heading is not None:
                            heading_world = self._slerp_yaw(
                                old_heading, heading_world, self.follow_heading_alpha
                            )
                            x = self.latest_pose.x + dist * math.cos(heading_world)
                            y = self.latest_pose.y + dist * math.sin(heading_world)
                        goal = self.make_goal_pose((x, y, 0.0), heading_world)
                        self.goal_source = "ctx_pair_follow"
                        self.ctx_last_goal = goal
                        self.ctx_last_update = now
                        rospy.loginfo(
                            "GoalManager: context segment refreshed old_dist=%.2f "
                            "heading_delta=%.1fdeg goal=(%.2f,%.2f)",
                            old_distance,
                            math.degrees(heading_delta) if math.isfinite(heading_delta) else float("inf"),
                            x,
                            y,
                        )
                    elif self.ctx_last_goal is not None:
                        goal = self.ctx_last_goal
                        self.goal_source = "ctx_pair_cached"

        # ctx 窗口过期：回到探索
        if (now - self.ctx_start_time) >= self.follow_ctx_lost_timeout:
            if goal is None:
                goal = self.ctx_last_goal
            rospy.loginfo_throttle(2.0, "GoalManager: ctx window %.1fs expired, switch to explore_sus_c_mode",
                                   self.follow_ctx_lost_timeout)
            self.effective_mode = EXPLORE_SUS_C_MODE
            self.pub_access_mode.publish(String(data=self.effective_mode))
            self.ctx_start_time = None
            self.ctx_cooldown_until = now + self.context_follow_cooldown
            rospy.loginfo(
                "GoalManager: context observation complete; global coverage cooldown %.1fs",
                self.context_follow_cooldown,
            )
            return goal if goal is not None else self.goal_from_frontiers_prior()

        if goal is None and self.ctx_last_goal is not None:
            self.goal_source = "ctx_cached"
            return self.ctx_last_goal
        if goal is None:
            return self.goal_from_frontiers_prior()
        return goal

    def goal_from_frontier(self, period_mode: str) -> Optional[PoseStamped]:
        if self.latest_frontier is None:
            rospy.logwarn_throttle(5.0, "GoalManager: frontier info not available")
            return None
        theta_rel = float(self.latest_frontier.vector.x)
        heading_world = float(self.latest_pose.theta) + theta_rel
        dist = self.clip_distance(heading_world, self.forward_dist)
        if dist is None:
            return None
        x = self.latest_pose.x + dist * math.cos(heading_world)
        y = self.latest_pose.y + dist * math.sin(heading_world)
        return self.make_goal_pose((x, y, 0.0), heading_world)

    def goal_from_frontiers_prior(self) -> Optional[PoseStamped]:
        """PASS/Sus-C：使用全量 frontier + 运动先验投影到 4 固定方向后选取目标。"""
        if self.headings is None:
            if self.start_pose:
                self.init_headings(self.start_pose)
            elif self.latest_pose:
                self.init_headings(self.latest_pose)
        if self.headings is None:
            rospy.logwarn_throttle(5.0, "GoalManager: headings not initialized")
            return None
        if self.latest_pose is None:
            return None
        msg = self.latest_frontiers
        if msg is None or len(msg.theta_rel) == 0:
            rospy.logwarn_throttle(5.0, "GoalManager: gp_frontiers not available")
            return None

        thetas_rel = list(msg.theta_rel)
        areas_raw = list(msg.area) if msg.area else [1.0] * len(thetas_rel)
        if len(areas_raw) < len(thetas_rel):
            areas_raw += [1.0] * (len(thetas_rel) - len(areas_raw))
        areas = [max(0.0, float(a)) for a in areas_raw[: len(thetas_rel)]]

        total_area = sum(areas)
        if total_area <= 0:
            return None
        # 运动方向（世界系）
        dir_move = None
        prior_strength = 0.0
        if self.latest_cmd_vel is not None:
            vx = float(self.latest_cmd_vel.linear.x)
            speed = abs(vx)
            if speed >= self.v_min and self.latest_pose is not None:
                dir_move = wrap_angle(self.latest_pose.theta if vx >= 0 else self.latest_pose.theta + math.pi)
                if self.v_scale > 1e-3:
                    prior_strength = min(1.0, max(0.0, (speed - self.v_min) / self.v_scale))
                else:
                    prior_strength = 1.0

        scores = [0.0, 0.0, 0.0, 0.0]
        sigma = self.front_sigma if self.front_sigma > 1e-3 else 0.35
        front_thr = self.front_deg
        back_thr = math.pi - self.back_deg
        base_heading = float(self.latest_pose.theta)

        for theta_rel, area in zip(thetas_rel, areas):
            heading_world = wrap_angle(base_heading + float(theta_rel))
            a_norm = area / total_area
            weight = math.pow(a_norm, self.area_power)
            # 运动先验权重（投影前）
            if dir_move is not None:
                delta = abs(wrap_angle(heading_world - dir_move))
                if delta <= front_thr:
                    w_move = 1.0 + prior_strength * (self.w_front_max - 1.0)
                elif delta >= back_thr:
                    w_move = 1.0 - prior_strength * (1.0 - self.w_back_min)
                else:
                    w_move = 1.0
            else:
                w_move = 1.0
            vote = weight * w_move
            for idx, Hk in enumerate(self.headings):
                diff = abs(wrap_angle(heading_world - Hk))
                proj = math.exp(-0.5 * (diff / sigma) ** 2)
                scores[idx] += vote * proj

        # 前进时限制后半平面（可选）
        allow_mask = [True] * 4
        if self.forward_only and dir_move is not None and abs(float(self.latest_cmd_vel.linear.x)) > self.v_gate:
            for idx, Hk in enumerate(self.headings):
                if math.cos(wrap_angle(Hk - base_heading)) < 0:
                    allow_mask[idx] = False
            if not any(allow_mask) and any(s > self.min_allowed_score for s in scores):
                allow_mask = [True if s > self.min_allowed_score else False for s in scores]

        best_idx = None
        best_val = -1.0
        second_val = -1.0
        for idx, s in enumerate(scores):
            if not allow_mask[idx]:
                continue
            if s > best_val:
                second_val = best_val
                best_val = s
                best_idx = idx
            elif s > second_val:
                second_val = s
        if best_idx is None:
            return None

        # 滞回：优势不够则保持上次方向
        selected_idx = best_idx
        if self.last_dir_idx is not None and best_idx != self.last_dir_idx:
            margin = second_val * (1.0 + self.switch_margin)
            if best_val < margin:
                best_idx = self.last_dir_idx
                best_val = scores[best_idx]
        self.last_dir_idx = best_idx
        self.goal_source = "frontier_vote"
        self.frontier_debug = "n=%d selected=%d raw_best=%d scores=[%s]" % (
            len(thetas_rel), best_idx, selected_idx,
            ",".join("%.3f" % score for score in scores),
        )

        heading_world = self.headings[best_idx]
        dist = self.clip_distance(heading_world, self.forward_dist)
        if dist is None:
            return None
        x = self.latest_pose.x + dist * math.cos(heading_world)
        y = self.latest_pose.y + dist * math.sin(heading_world)
        return self.make_goal_pose((x, y, 0.0), heading_world)

    # -------------------- Helpers --------------------
    @staticmethod
    def _frame_name(frame: str) -> str:
        return (frame or "odom").strip().lstrip("/") or "odom"

    def robot_xy_in_frame(self, frame: str) -> Optional[Tuple[float, float]]:
        """Return the current robot position in a goal's frame.

        ``/rbt_pose`` is expressed in odom, while an online frontier is a
        stable map-frame point.  Comparing those coordinates directly creates
        a false distance whenever gmapping updates map->odom.  Transforming
        the robot origin for the comparison keeps the mission layer frame
        agnostic without rewriting the goal itself.
        """
        if self.latest_pose is None:
            return None
        target = self._frame_name(frame)
        if target == "odom":
            return float(self.latest_pose.x), float(self.latest_pose.y)
        tfm = self.lookup_transform(target, "odom", rospy.Time(0))
        if tfm is None:
            return None
        rot = tfm.transform.rotation
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        point = mat[:3, :3].dot(
            np.array([self.latest_pose.x, self.latest_pose.y, 0.0], dtype=float)
        )
        trans = tfm.transform.translation
        return float(point[0] + trans.x), float(point[1] + trans.y)

    def goal_robot_distance(self, goal: Optional[PoseStamped]) -> Optional[float]:
        if goal is None:
            return None
        robot = self.robot_xy_in_frame(goal.header.frame_id)
        if robot is None:
            return None
        return math.hypot(
            float(goal.pose.position.x) - robot[0],
            float(goal.pose.position.y) - robot[1],
        )

    def make_goal_pose(self, xyz: Tuple[float, float, float], yaw: Optional[float] = None) -> PoseStamped:
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "odom"
        goal.pose.position.x = float(xyz[0])
        goal.pose.position.y = float(xyz[1])
        goal.pose.position.z = float(xyz[2])
        # 始终填充合法四元数，避免下游解析 yaw 失败
        if yaw is None:
            goal.pose.orientation.w = 1.0
        else:
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(yaw))
            goal.pose.orientation.x = qx
            goal.pose.orientation.y = qy
            goal.pose.orientation.z = qz
            goal.pose.orientation.w = qw
        return goal

    def publish_goal(self, goal: PoseStamped, force_republish: bool = False):
        previous_goal = self.last_goal
        terminal_override = False
        if (
            self.controller_mode == "teb"
            and
            previous_goal is not None
            and self.teb_terminal_goal is not None
            and self.last_goal_source == "global_slam_frontier"
        ):
            terminal_delta = math.hypot(
                self.teb_terminal_goal.pose.position.x - previous_goal.pose.position.x,
                self.teb_terminal_goal.pose.position.y - previous_goal.pose.position.y,
            )
            terminal_override = terminal_delta <= max(
                self.global_frontier_update_radius, 0.30
            )
            if terminal_override and self.latest_pose is not None:
                # A stale/mis-associated action status can arrive while the
                # robot is still far from the source waypoint.  Never let
                # that status authorize a large branch jump and an immediate
                # in-place stop; require physical proximity as well.
                terminal_distance = self.goal_robot_distance(previous_goal)
                if terminal_distance is None:
                    terminal_distance = float("inf")
                terminal_override = (
                    terminal_distance <= self.global_frontier_early_handoff_radius
                )
        if (
            previous_goal is not None
            and self.goal_source == "global_slam_frontier"
            and self.last_goal_source == "global_slam_frontier"
            and self.latest_pose is not None
        ):
            distance_to_previous = self.goal_robot_distance(previous_goal)
            if distance_to_previous is None:
                distance_to_previous = float("inf")
            goal_delta = math.hypot(
                goal.pose.position.x - previous_goal.pose.position.x,
                goal.pose.position.y - previous_goal.pose.position.y,
            )
            if self.controller_mode == "teb" and not terminal_override:
                # TEB owns one atomic move_base action.  Do not cancel it from
                # this mission-layer callback based on a second wall clock:
                # the bridge coalesces map updates and uses action feedback to
                # hand off only near the endpoint or after real no-progress.
                # Keeping both timeout policies active caused a healthy action
                # at 0.24 m from its goal to be preempted at exactly 15 s.
                if goal_delta > 0.03:
                    rospy.loginfo_throttle(
                        3.0,
                        "GoalManager: forward frontier update to TEB bridge "
                        "distance=%.2fm update_delta=%.2fm",
                        distance_to_previous,
                        goal_delta,
                    )
            else:
                elapsed = float("inf")
                if self.frontier_goal_sent_at is not None:
                    elapsed = max(0.0, rospy.Time.now().to_sec() - self.frontier_goal_sent_at)
                if (
                    not terminal_override
                    and goal_delta >= self.global_frontier_jump_distance
                    and distance_to_previous > self.global_frontier_jump_release_radius
                ):
                    rospy.loginfo_throttle(
                        3.0,
                        "GoalManager: hold distant frontier branch jump "
                        "distance=%.2fm update_delta=%.2fm release_radius=%.2fm",
                        distance_to_previous,
                        goal_delta,
                        self.global_frontier_jump_release_radius,
                    )
                    return False
                if (
                    not terminal_override
                    and elapsed < self.global_frontier_min_hold_time
                    and goal_delta < self.global_frontier_jump_distance
                ):
                    rospy.loginfo_throttle(
                        3.0,
                        "GoalManager: hold frontier minimum dwell elapsed=%.1fs/%.1fs "
                        "distance=%.2fm update_delta=%.2fm",
                        elapsed,
                        self.global_frontier_min_hold_time,
                        distance_to_previous,
                        goal_delta,
                    )
                    return False
                if (
                    not terminal_override
                    and distance_to_previous > self.global_frontier_update_radius
                    and goal_delta < self.global_frontier_jump_distance
                ):
                    rospy.loginfo_throttle(
                        3.0,
                        "GoalManager: hold frontier goal distance=%.2fm "
                        "update_delta=%.2fm",
                        distance_to_previous,
                        goal_delta,
                    )
                    return False
        if terminal_override:
            rospy.loginfo(
                "GoalManager: replacing completed frontier action despite "
                "small update delta"
            )
            self.teb_terminal_goal = None
        # Do not rebroadcast an unchanged goal on every 5 Hz timer tick.  Apart
        # from wasting bandwidth, those messages used to look like target
        # movement to downstream consumers and made diagnosis impossible.
        # Fixed-goal mode is the exception: its latched periodic republish is
        # intentional for late controller subscribers.
        if (
            previous_goal is not None
            and not force_republish
            and math.hypot(
                goal.pose.position.x - previous_goal.pose.position.x,
                goal.pose.position.y - previous_goal.pose.position.y,
            ) <= 0.03
            and (
                self.last_goal_source == self.goal_source
                or (
                    self.last_goal_source.startswith("target_")
                    and self.goal_source.startswith("target_")
                )
            )
        ):
            return False
        self.last_goal = goal
        self.last_goal_source = self.goal_source
        if self.goal_source == "global_slam_frontier":
            self.teb_frontier_goal_history.append(copy.deepcopy(goal))
            # Keep enough history for coalesced route updates and native
            # in-place segment replacements without retaining a full mission.
            del self.teb_frontier_goal_history[:-12]
        if self.goal_source.startswith("target_"):
            self.target_execution_state = "TARGET_EXECUTING"
        if self.goal_source == "global_slam_frontier":
            self.frontier_goal_sent_at = rospy.Time.now().to_sec()
        # Publish the mission decision before the pose.  The bridge can then
        # classify the following PoseStamped before it considers dispatching an
        # action, avoiding a race between a target takeover and a frontier
        # update.  Priority is deliberately coarse: it describes ownership,
        # not a controller tuning value.
        intent = {
            "source": self.goal_source,
            "priority": self.goal_intent_priority(self.goal_source),
            "goal": [
                round(float(goal.pose.position.x), 4),
                round(float(goal.pose.position.y), 4),
            ],
        }
        if self.goal_source == "global_slam_frontier":
            intent["route_kind"] = self.global_frontier_route_kind
        if self.goal_source.startswith("target_"):
            intent["target_epoch"] = int(self.target_observation_epoch)
            intent["target_track_id"] = self.target_track_id
            intent["target_state"] = self.target_execution_state
        self.pub_goal_intent.publish(String(data=json.dumps(intent, sort_keys=True)))
        self.pub_goal.publish(goal)
        rospy.loginfo_throttle(2.0, "Publish /lste/final_goal: x=%.2f y=%.2f state=%s",
                               goal.pose.position.x, goal.pose.position.y, self.current_state)
        if self.debug_goal_log:
            self.log_goal_diagnostic(goal, previous_goal)
        return True

    @staticmethod
    def goal_intent_priority(source: str) -> int:
        """Return mission ownership priority for the TEB action bridge.

        A target is a time-sensitive observation and may interrupt exploration.
        Frontier and context updates are route-planning refreshes and must not
        interrupt one another.  The values are an internal ordering, not user
        configuration.
        """
        source = str(source or "").strip().lower()
        if source.startswith("target_"):
            return 2
        if source == "fixed_config":
            return 3
        return 0

    def log_goal_diagnostic(self, goal: PoseStamped, previous_goal: Optional[PoseStamped]):
        """Record inputs and movement behind one final-goal decision."""
        pose = self.latest_pose
        if pose is None:
            return
        goal_yaw = self.yaw_from_pose(goal) or 0.0
        robot_goal_dist = self.goal_robot_distance(goal)
        if robot_goal_dist is None:
            robot_goal_dist = float("nan")
        goal_delta = 0.0
        if previous_goal is not None:
            goal_delta = math.hypot(
                goal.pose.position.x - previous_goal.pose.position.x,
                goal.pose.position.y - previous_goal.pose.position.y,
            )
        cmd_v = float(self.latest_cmd_vel.linear.x) if self.latest_cmd_vel is not None else 0.0
        cmd_w = float(self.latest_cmd_vel.angular.z) if self.latest_cmd_vel is not None else 0.0
        if self.latest_scores is None:
            score_info = "unavailable"
        else:
            score_info = "total=%.3f,target=%.3f,env=%.3f,ctx=%.3f,detected=%s" % (
                self.latest_scores.s_total, self.latest_scores.s_target,
                self.latest_scores.s_env, self.latest_scores.s_ctx,
                self.latest_scores.detected,
            )
        det_info = self.diagnostic_detection_summary()
        rospy.loginfo(
            "GOAL_DIAG source=%s state=%d subtype=%s mode=%s access=%d "
            "robot=(%.2f,%.2f,%.2f) cmd=(%.2f,%.2f) "
            "goal=(%.2f,%.2f,%.2f) robot_goal_dist=%.2f goal_delta=%.2f "
            "scores={%s} detections={%s} frontiers={%s}",
            self.goal_source, self.current_state, self.current_subtype or "-",
            self.effective_mode, self.access_mode, pose.x, pose.y, pose.theta,
            cmd_v, cmd_w, goal.pose.position.x, goal.pose.position.y, goal_yaw,
            robot_goal_dist, goal_delta, score_info, det_info, self.frontier_debug,
        )
        self.pub_goal_diagnostic.publish(String(data=json.dumps({
            "source": self.goal_source,
            "target_state": self.target_execution_state,
            "target_epoch": int(self.target_observation_epoch),
            "target_blocked": bool(self.target_blocked),
            "target_failure_count": int(self.target_failure_count),
            "navigation_hold": bool(self.navigation_hold_active),
            "navigation_hold_remaining": round(
                max(0.0, self.target_observation_hold_until - rospy.Time.now().to_sec()),
                3,
            ),
            "state": int(self.current_state),
            "subtype": self.current_subtype or "",
            "mode": self.effective_mode,
            "access_mode": int(self.access_mode),
            "robot": [round(float(pose.x), 4), round(float(pose.y), 4), round(float(pose.theta), 4)],
            "cmd": [round(cmd_v, 4), round(cmd_w, 4)],
            "goal": [round(float(goal.pose.position.x), 4), round(float(goal.pose.position.y), 4), round(float(goal_yaw), 4)],
            "robot_goal_distance": round(float(robot_goal_dist), 4),
            "goal_delta": round(float(goal_delta), 4),
            "scores": None if self.latest_scores is None else {
                "total": round(float(self.latest_scores.s_total), 4),
                "target": round(float(self.latest_scores.s_target), 4),
                "env": round(float(self.latest_scores.s_env), 4),
                "ctx": round(float(self.latest_scores.s_ctx), 4),
                "detected": bool(self.latest_scores.detected),
            },
            "detections": det_info,
            "frontier_debug": self.frontier_debug,
        }, sort_keys=True)))

    def diagnostic_detection_summary(self) -> str:
        if self.latest_dets is None:
            return "unavailable"
        target_count = len(self.latest_dets.target_dets)
        env_count = len(self.latest_dets.env_dets)
        best = self.pick_best_target()
        if best is None:
            return "target=0,env=%d" % env_count
        return "target=%d,env=%d,best=%s:%.3f@%.3f,%.3f box=%.3fx%.3f" % (
            target_count, env_count, best.label, best.score, best.cx, best.cy,
            best.w, best.h,
        )

    def build_goal_from_pose(self, pose: Pose2D, dist: float) -> PoseStamped:
        x = pose.x + dist * math.cos(pose.theta)
        y = pose.y + dist * math.sin(pose.theta)
        return self.make_goal_pose((x, y, 0.0), pose.theta)

    def pick_best_target(self) -> Optional[LsteDetection]:
        return self.best_target_from(self.latest_dets)

    def target_detection_for_track(
        self, message: Optional[LsteDetections]
    ) -> Optional[LsteDetection]:
        """Select the active target identity before comparing confidence.

        ``best_target_from`` is appropriate for an initial discovery, but
        choosing the highest-scoring box on every frame lets a nearby distractor
        replace the locked object and changes the projected world bearing. A
        locked label therefore owns the track until the target memory is
        explicitly released.
        """
        if message is None or not message.target_dets:
            return None
        if self.target_track_label:
            same_label = [
                detection for detection in message.target_dets
                if (detection.label or "").strip().lower() == self.target_track_label
            ]
            if same_label:
                return max(same_label, key=lambda detection: float(detection.score))
        return self.best_target_from(message)

    @staticmethod
    def best_target_from(msg: Optional[LsteDetections]) -> Optional[LsteDetection]:
        if msg is None or len(msg.target_dets) == 0:
            return None
        dets = sorted(msg.target_dets, key=lambda d: float(d.score), reverse=True)
        return dets[0]

    @staticmethod
    def clone_detection(det: LsteDetection) -> LsteDetection:
        cloned = LsteDetection()
        cloned.label = det.label
        cloned.score = float(det.score)
        cloned.cx = float(det.cx)
        cloned.cy = float(det.cy)
        cloned.w = float(det.w)
        cloned.h = float(det.h)
        return cloned

    def target_tracking_active(self, now: float) -> bool:
        if self.target_blocked:
            return False
        if self.target_last_seen is None:
            return False
        # A weak, one-frame candidate is active only during its explicit
        # observation hold.  Once that hold ends, state/frontier exploration
        # remains in charge until the candidate is independently confirmed.
        if not self.target_follow_confirmed and now >= self.target_observation_hold_until:
            return False
        timeout = (
            self.follow_target_lost_timeout
            if self.target_follow_confirmed
            else self.target_follow_candidate_timeout
        )
        return (now - self.target_last_seen) < timeout

    def pick_ctx_pair(self) -> Tuple[Optional[LsteDetection], Optional[LsteDetection]]:
        """
        返回左右各一个 ctx 检测（label 分别包含 ctx_left / ctx_right），选 2D 中心距离最近的一对。
        若左右缺任意一侧则返回 (None, None)。
        """
        if self.latest_dets is None or self.latest_task is None:
            return None, None
        left_term = (self.latest_task.ctx_left or "").strip().lower()
        right_term = (self.latest_task.ctx_right or "").strip().lower()
        if not left_term or left_term == "none" or not right_term or right_term == "none":
            return None, None
        all_dets: List[LsteDetection] = list(self.latest_dets.env_dets) + list(self.latest_dets.target_dets)
        left_hits = [d for d in all_dets if left_term in (d.label or "").lower()]
        right_hits = [d for d in all_dets if right_term in (d.label or "").lower()]
        if not left_hits or not right_hits:
            return None, None
        best_pair: Tuple[Optional[LsteDetection], Optional[LsteDetection]] = (None, None)
        best_dist = float("inf")
        for l in left_hits:
            for r in right_hits:
                if l is r:
                    continue  # 左右不能是同一个框
                try:
                    dx = float(l.cx) - float(r.cx)
                    dy = float(l.cy) - float(r.cy)
                except Exception:
                    continue
                d = math.hypot(dx, dy)
                if d < best_dist:
                    best_dist = d
                    best_pair = (l, r)
        if best_pair[0] is None or best_pair[1] is None:
            return None, None
        return best_pair

    def yaw_from_pose(self, pose: PoseStamped) -> Optional[float]:
        try:
            q = pose.pose.orientation
            mat = quaternion_matrix([q.x, q.y, q.z, q.w])
            return math.atan2(mat[1, 0], mat[0, 0])
        except Exception:
            return None

    def _slerp_yaw(self, prev: float, new: float, alpha: float) -> float:
        """简单对 yaw 做插值，避免大跳变。"""
        delta = wrap_angle(new - prev)
        blended = prev + alpha * delta
        return wrap_angle(blended)

    # -------------------- Projection --------------------
    def det_heading_world(self, det: LsteDetection) -> Optional[float]:
        """无深度：返回检测射线在 odom 下的朝向（弧度）。"""
        if self.camera_info is None or self.camera_frame is None:
            rospy.logwarn_throttle(5.0, "GoalManager: camera info not ready")
            return None
        w = int(self.camera_info.width) if self.camera_info.width else 0
        h = int(self.camera_info.height) if self.camera_info.height else 0
        if w <= 0 or h <= 0:
            return None
        u = int(float(det.cx) * w)
        v = int(float(det.cy) * h)
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))
        ray = self.camera_model.projectPixelTo3dRay((u, v))  # optical frame direction
        tfm = self.lookup_transform("odom", self.camera_frame, rospy.Time(0))
        if tfm is None:
            return None
        rot = tfm.transform.rotation
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        dir_cam = np.array([ray[0], ray[1], ray[2]], dtype=float)
        dir_odom = mat[:3, :3].dot(dir_cam)
        heading_world = math.atan2(dir_odom[1], dir_odom[0])
        return heading_world

    def clip_distance(self, heading_world: float, desired: float) -> Optional[float]:
        """
        用 LaserScan 沿给定方向裁剪距离，返回裁剪后的距离。
        - heading_world: 方向（odom 下弧度）
        - desired: 希望的前推距离
        """
        if self.latest_scan is None:
            return desired

        scan = self.latest_scan
        scan_frame = self.scan_frame or scan.header.frame_id
        if not scan_frame:
            return desired

        yaw_scan = self.frame_yaw(scan_frame)
        if yaw_scan is None:
            return desired

        theta_scan = wrap_angle(heading_world - yaw_scan)
        ang_min = scan.angle_min
        ang_inc = scan.angle_increment if scan.angle_increment != 0 else 1e-6
        idx = int(round((theta_scan - ang_min) / ang_inc))
        idx = max(0, min(len(scan.ranges) - 1, idx))

        win = self.scan_window_bins
        idx_min = max(0, idx - win)
        idx_max = min(len(scan.ranges) - 1, idx + win)
        r_min = None
        for i in range(idx_min, idx_max + 1):
            r = scan.ranges[i]
            if math.isfinite(r) and r > 0.05:
                r_min = r if r_min is None else min(r_min, r)
        if r_min is None:
            return desired

        clipped = min(desired, r_min - self.safety_margin)
        if clipped <= 0.2:
            rospy.logwarn_throttle(2.0, "GoalManager: clipped dist too small (%.2f)", clipped)
            return None
        return clipped

    def transform_point(self, pt: PointStamped, target_frame: str) -> Optional[PointStamped]:
        try:
            tfm = self.tf_buffer.lookup_transform(
                target_frame,
                pt.header.frame_id,
                pt.header.stamp,
                rospy.Duration(0.05),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(5.0, "GoalManager: TF lookup failed: %s", exc)
            return None

        trans = tfm.transform.translation
        rot = tfm.transform.rotation
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        mat[0, 3] = trans.x
        mat[1, 3] = trans.y
        mat[2, 3] = trans.z
        vec = np.array([pt.point.x, pt.point.y, pt.point.z, 1.0], dtype=float)
        out = mat.dot(vec)

        out_pt = PointStamped()
        out_pt.header.frame_id = target_frame
        out_pt.header.stamp = tfm.header.stamp if tfm.header.stamp else rospy.Time.now()
        out_pt.point.x = float(out[0])
        out_pt.point.y = float(out[1])
        out_pt.point.z = float(out[2])
        return out_pt

    def lookup_transform(self, target: str, source: str, stamp: rospy.Time) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                target,
                source,
                stamp,
                rospy.Duration(0.05),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(5.0, "GoalManager: TF lookup failed: %s", exc)
            return None

    def frame_yaw(self, frame: str) -> Optional[float]:
        """返回 frame 在 odom 下的 yaw。"""
        tfm = self.lookup_transform("odom", frame, rospy.Time(0))
        if tfm is None:
            return None
        rot = tfm.transform.rotation
        mat = quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        yaw = math.atan2(mat[1, 0], mat[0, 0])
        return yaw


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def main():
    GoalManager()
    rospy.spin()


if __name__ == "__main__":
    main()
