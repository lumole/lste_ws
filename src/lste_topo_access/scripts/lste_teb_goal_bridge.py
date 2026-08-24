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
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from lste_topo_access.msg import PersistentGoalCommand
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetPlan, GetPlanRequest
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Bool, String
from teb_local_planner.msg import FeedbackMsg


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
        # GoalManager now publishes the position and its ownership metadata
        # as one transaction. Legacy topics remain available to RViz and older
        # external callers, but executing from two separate latches permits a
        # stale route to be replayed after a hot restart.
        self.goal_command_topic = gp("~goal_command_topic", "/lste/mission_goal")
        self.use_goal_command = self._as_bool(gp("~use_goal_command", True))
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
        self.persistent_target_goal_topic = gp(
            "~persistent_target_goal_topic",
            "/lste/persistent_execution/target_goal",
        )
        # The streaming Navfn plugin consumes only this bridge-owned topic.
        # /lste/final_goal remains GoalManager's public observer API, while a
        # priority-2 target reaches this topic only after Navfn has confirmed
        # the matching provisional target transaction.
        self.persistent_mission_goal_topic = gp(
            "~persistent_mission_goal_topic",
            "/lste/persistent_execution/mission_goal",
        )
        # PersistentGoalCommand is the execution contract. PoseStamped topics
        # below remain latched observer APIs for RViz and legacy tools, but
        # never carry transaction identity: ROS owns Header.seq.
        self.persistent_target_command_topic = gp(
            "~persistent_target_command_topic",
            "/lste/persistent_execution/target_command",
        )
        self.persistent_mission_command_topic = gp(
            "~persistent_mission_command_topic",
            "/lste/persistent_execution/mission_command",
        )
        self.persistent_installed_target_goal_topic = gp(
            "~persistent_installed_target_goal_topic",
            "/lste/persistent_execution/installed_target_goal",
        )
        self.persistent_installed_target_command_topic = gp(
            "~persistent_installed_target_command_topic",
            "/lste/persistent_execution/installed_target_command",
        )
        self.persistent_target_approach_topic = gp(
            "~persistent_target_approach_topic",
            "/lste/persistent_execution/target_approach",
        )
        # PersistentTebLocalPlanner keeps the MoveBase action alive as a
        # controller lease. It therefore reports a reached frontier endpoint
        # separately; this bridge validates it against the active route before
        # turning it into the logical terminal that advances GlobalFrontier.
        self.persistent_frontier_endpoint_topic = gp(
            "~persistent_frontier_endpoint_topic",
            "/lste/persistent_execution/frontier_endpoint_reached",
        )
        self.persistent_target_plan_result_topic = gp(
            "~persistent_target_plan_result_topic",
            "/lste/persistent_execution/target_plan_result",
        )
        self.bridge_status_topic = gp(
            "~bridge_status_topic", "/lste/teb_goal_bridge/status"
        )
        self.turn_supervisor_status_topic = gp(
            "~turn_supervisor_status_topic", "/lste/teb_turn_supervisor/status"
        )
        # ``move_base`` feedback reports XY progress only.  TEB can also make
        # legitimate progress by rotating in place around a constrained local
        # endpoint.  Keep that execution signal separate from the explicit
        # pre-route turn supervisor: this bridge observes TEB's selected
        # trajectory and the odometry yaw, but never publishes a velocity.
        self.teb_feedback_topic = gp(
            "~teb_feedback_topic", "/move_base/TebLocalPlannerROS/teb_feedback"
        )
        # TEB stops publishing a selected elastic-band trajectory once it has
        # reached a point goal.  Its raw command topic is therefore the
        # authoritative observation for the final zero-velocity envelope used
        # to release a prefetched frontier successor.
        self.teb_planner_cmd_topic = gp(
            "~teb_planner_cmd_topic", "/lste/cmd_vel/teb_planner"
        )
        # Navfn is the global-route authority.  A visual target can require a
        # legitimate detour that initially increases its Euclidean distance,
        # so action health must use this path rather than a straight-line ray.
        self.navfn_plan_topic = gp("~navfn_plan_topic", "/move_base/NavfnROS/plan")
        # A prefetched frontier is selected from the global SLAM map, but the
        # first metres of its route must also be executable in the rolling
        # lidar costmap before a persistent executor hands it to TEB.  This is
        # deliberately an admission check, not a second velocity controller:
        # it either releases one already safe route transaction or keeps the
        # current one intact.
        self.local_costmap_topic = gp(
            "~local_costmap_topic", "/move_base/local_costmap/costmap"
        )
        self.navfn_make_plan_service = gp(
            "~navfn_make_plan_service", "/move_base/NavfnROS/make_plan"
        )
        self.persistent_frontier_lookahead_handoff_enabled = self._as_bool(
            gp("~persistent_frontier_lookahead_handoff_enabled", False)
        )
        # This cannot reuse frontier_observation_completion_radius: the normal
        # production value is zero because an observation region must not
        # cancel a healthy endpoint. A lookahead route splice, however, must
        # be admitted while TEB is still outside its point-goal terminal
        # envelope.
        self.persistent_frontier_lookahead_trigger_distance = max(
            0.20,
            float(gp("~persistent_frontier_lookahead_trigger_distance", 1.0)),
        )
        self.persistent_frontier_admission_horizon = max(
            0.20,
            float(gp("~persistent_frontier_admission_horizon", 2.0)),
        )
        self.persistent_frontier_admission_max_costmap_age = max(
            0.05,
            float(gp("~persistent_frontier_admission_max_costmap_age", 0.50)),
        )
        self.persistent_frontier_admission_blocked_cost = min(
            100,
            max(1, int(gp("~persistent_frontier_admission_blocked_cost", 100))),
        )
        self.persistent_frontier_admission_service_timeout = max(
            0.01,
            float(gp("~persistent_frontier_admission_service_timeout", 0.10)),
        )
        self.persistent_frontier_admission_endpoint_epsilon = max(
            0.05,
            float(gp("~persistent_frontier_admission_endpoint_epsilon", 0.20)),
        )
        self.pose_topic = gp("~pose_topic", "/rbt_pose")
        self.global_frame = str(gp("~global_frame", "map")).strip().lstrip("/") or "map"
        self.active_mode = str(gp("~active_mode", "teb")).strip().lower()
        self.mode = str(gp("~initial_mode", "teb")).strip().lower()
        # Experimental execution architecture: retain one MoveBase action as
        # a controller lease while pluginized Navfn/TEB consume fresh mission
        # paths. Ordinary endpoint-action mode remains the production default.
        self.persistent_execution = self._as_bool(
            gp("~persistent_execution", False)
        )
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
        # A frontier endpoint denotes a safe *observation region*, unlike a
        # visual target which remains a precise approach point.  Once the
        # base is inside this region and the explorer has already validated a
        # successor, forcing TEB into its smaller point-goal tolerance wastes
        # time rotating beside a wall.  Zero explicitly disables this semantic
        # completion contract for A/B experiments.
        self.frontier_observation_completion_radius = max(
            0.0, float(gp("~frontier_observation_completion_radius", 0.90))
        )
        # Do not turn an observation-region boundary into an emergency brake.
        # TEB owns the deceleration profile; this lifecycle layer may only
        # close or replace an action after fresh TEB feedback has already
        # selected a low forward speed. A missing/stale feedback sample falls
        # back to move_base's ordinary point-goal terminal.
        self.frontier_observation_completion_max_linear_speed = max(
            0.0,
            float(gp("~frontier_observation_completion_max_linear_speed", 0.15)),
        )
        # Require a short sequence of actual stationary TEB commands. This
        # rejects a single planner-cycle gap while avoiding the 0.5-1.5 s wait
        # caused by stale selected-trajectory feedback at an endpoint.
        self.frontier_observation_stationary_hold = max(
            0.0,
            float(gp("~frontier_observation_stationary_hold", 0.10)),
        )
        self.teb_planner_command_timeout = max(
            0.05,
            float(gp("~teb_planner_command_timeout", 0.30)),
        )
        # A prefetched successor has already passed the frontier mask,
        # inflated-costmap and Navfn checks.  Prefer its native actionlib
        # replacement over cancel->PREEMPTED->dispatch: the latter inserts a
        # visible zero-velocity boundary at every explored observation region.
        # This remains deliberately narrower than generic goal replacement.
        self.frontier_continuous_prefetch_handoff_enabled = self._as_bool(
            gp("~frontier_continuous_prefetch_handoff_enabled", False)
        )
        self.frontier_continuous_prefetch_handoff_timeout = max(
            0.20,
            float(gp("~frontier_continuous_prefetch_handoff_timeout", 1.50)),
        )
        # Bounded native-TEB reorientation exemption.  A fresh selected
        # near-zero linear command with meaningful angular velocity is not by
        # itself proof of progress: the wheel-odometry yaw must also advance.
        # This prevents a legitimate obstacle/corner reorientation from being
        # cancelled as ``no_feedback_progress`` while retaining a finite
        # escape hatch for a real optimizer deadlock.
        self.teb_reorientation_enabled = self._as_bool(
            gp("~teb_reorientation_enabled", True)
        )
        self.teb_reorientation_linear_threshold = max(
            0.001, float(gp("~teb_reorientation_linear_threshold", 0.03))
        )
        self.teb_reorientation_angular_threshold = max(
            0.01, float(gp("~teb_reorientation_angular_threshold", 0.15))
        )
        self.teb_reorientation_feedback_timeout = max(
            0.05, float(gp("~teb_reorientation_feedback_timeout", 0.50))
        )
        self.teb_reorientation_yaw_progress = max(
            0.01, float(gp("~teb_reorientation_yaw_progress", 0.10))
        )
        self.teb_reorientation_stagnation_timeout = max(
            0.1, float(gp("~teb_reorientation_stagnation_timeout", 1.50))
        )
        self.teb_reorientation_max_extension = max(
            self.teb_reorientation_stagnation_timeout,
            float(gp("~teb_reorientation_max_extension", 8.0)),
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
            # A 52.9 degree handoff was observed to reset the active TEB band
            # to a zero-speed turn despite a short 1.8 m endpoint delta.  This
            # contract is therefore a local curvature bound, not merely a
            # route-validity check: only a modest tangent change can preserve
            # forward motion through a native action replacement.
            1.0, float(gp("~frontier_early_handoff_max_heading_deg", 45.0))
        ))
        # A persistent stream has no actionlib replacement at the route
        # boundary, so it can safely absorb a modestly larger tangent change
        # as a forward curve.  This is deliberately a distinct transition
        # class: a branch beyond this envelope waits for endpoint completion
        # and native TEB reorientation, never an optimistic velocity carry-
        # through.
        self.persistent_frontier_curve_handoff_max_heading_delta = math.radians(
            min(
                89.0,
                max(
                    math.degrees(self.frontier_early_handoff_max_heading_delta),
                    float(gp("~persistent_frontier_curve_handoff_max_heading_deg", 65.0)),
                ),
            )
        )
        # Classify a persistent frontier handoff from the Navfn route that
        # will actually be installed, rather than from an earlier frontier
        # BFS approximation.  A half-metre chord is long enough to ignore a
        # costmap-cell snap at the feedback pose while still describing the
        # route entry the local planner must execute.
        self.persistent_frontier_entry_tangent_distance = max(
            0.20,
            float(gp("~persistent_frontier_entry_tangent_distance", 0.50)),
        )
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
            "~persistent_frontier_curve_handoff_max_heading_deg",
            math.degrees(self.persistent_frontier_curve_handoff_max_heading_delta),
        )
        rospy.set_param(
            "~persistent_frontier_entry_tangent_distance",
            self.persistent_frontier_entry_tangent_distance,
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
        # Keep independent physical and Navfn-route progress records.  The
        # old ``active_progress_monotonic`` remains the direct goal-distance
        # measure required by frontier endpoint logic; target health selects
        # one of the two records below.
        self.active_motion_reference = None
        self.active_motion_progress_monotonic = 0.0
        self.active_navfn_plan_points = []
        self.active_navfn_plan_endpoint = None
        self.active_navfn_remaining = None
        self.active_navfn_best_remaining = None
        self.active_navfn_progress_monotonic = 0.0
        self.last_feedback_monotonic = 0.0
        self.handoff_requested = False
        self.handoff_count = 0
        self.priority_handoff_count = 0
        self.target_segment_handoff_count = 0
        self.frontier_segment_handoff_count = 0
        self.frontier_observation_completion_count = 0
        self.frontier_continuous_prefetch_handoff_count = 0
        self.frontier_continuous_prefetch_handoff_fallback_count = 0
        # The handoff timer may revisit a route pair many times while the
        # robot approaches it. Keep the diagnostic idempotent per route pair.
        self.frontier_prefetch_requires_turn_pairs = set()
        # Persistent execution has one physical action lease but many semantic
        # frontier endpoints. Record the logical promotions separately so the
        # timer cannot release one prefetch more than once.
        self.persistent_frontier_prefetch_promoted_pairs = set()
        # A local TEB endpoint report can arrive on several control cycles
        # before StreamingNavfnPlanner consumes the promoted successor. Keep
        # the report idempotent per frontier route transaction.
        self.persistent_frontier_endpoint_terminal_routes = set()
        # Set only between an intentional observation-region cancel and its
        # matching actionlib PREEMPTED callback.  It turns that transport
        # result into one logical frontier terminal without changing the
        # semantics of true target or route failures.
        self.frontier_observation_completion_pending = None
        # A logical terminal is sent to the frontier planner before the old
        # action finishes.  The following route command must carry exactly
        # this successor route id before it is allowed to replace the action.
        self.frontier_continuous_prefetch_handoff_pending = None
        # The frontier planner validates one successor internally before it
        # publishes a new mission command. Retaining that lifecycle-only
        # prefetch lets this bridge complete an observation region without
        # promoting arbitrary map refreshes into controller inputs.
        self.prefetched_frontier_goal = None
        self.prefetched_frontier_route_id = 0
        # GlobalFrontier publishes the entry tangent of its already validated
        # BFS route with every prefetch. This prevents the bridge from treating
        # a straight line to a remote endpoint as the local path direction.
        self.prefetched_frontier_entry_yaw = None
        self.prefetched_frontier_entry_yaw_basis = ""
        self.frontier_stale_recovery_count = 0
        # Explicit target lifecycle. A latched failure blocks only the exact
        # target transaction that failed; a new visual track or a completed
        # map-frontier observation clears it. This prevents infinite retries.
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
        self.latest_route_id = 0
        self.latest_target_epoch = 0
        self.latest_target_track_id = ""
        self.latest_goal_transaction_id = 0
        # The streaming global planner publishes this only after Navfn has
        # installed a path whose endpoint is a confirmed visual target. It is
        # deliberately different from the desired target pose: local-plan
        # completion is valid only after this acknowledgement exists.
        self.persistent_installed_target_goal = None
        self.persistent_installed_target_transaction = 0
        self.persistent_target_pending_transaction = 0
        self.persistent_target_pending_goal = None
        # This tracks the message currently retained by the latched target
        # request topic. A completed request must actively overwrite that
        # latch, otherwise a restarted move_base receives an obsolete visual
        # ray before it sees the current approved mission.
        self.persistent_target_request_transaction = 0
        self.persistent_target_approach_reported_transaction = 0
        # A delayed local-planner terminal can refer to the previously
        # installed target endpoint after the mission stream has advanced.
        # Retry installation once for the newer transaction without changing
        # the single move_base action lease.
        self.persistent_target_republish_transaction = 0
        # Goal Manager publishes intent metadata immediately before the pose,
        # but ROS callback scheduling can still deliver the two messages in
        # the opposite order. Keep the intent coordinates as a transaction
        # key so a PoseStamped is never dispatched with stale route semantics.
        self.latest_intent_goal = None
        self.intent_seen = False
        self.active_intent_source = "unknown"
        self.active_intent_priority = 0
        self.active_route_kind = ""
        self.active_route_id = 0
        self.active_target_epoch = 0
        self.active_target_track_id = ""
        self.turn_supervisor_state = "UNKNOWN"
        self.turn_supervisor_last_event = "unknown"
        self.latest_teb_selected_linear = None
        self.latest_teb_selected_angular = None
        self.latest_teb_feedback_monotonic = 0.0
        self.latest_teb_planner_linear = None
        self.latest_teb_planner_angular = None
        self.latest_teb_planner_command_monotonic = 0.0
        self.teb_planner_stationary_since = 0.0
        self.local_costmap = None
        self.local_costmap_received_monotonic = 0.0
        self.persistent_prefetch_admission_cache = None
        self.persistent_prefetch_admission_last_report_monotonic = 0.0
        self.odom_yaw = None
        self.odom_pose_monotonic = 0.0
        self.teb_reorientation_started_monotonic = 0.0
        self.teb_reorientation_reference_yaw = None
        self.teb_reorientation_last_yaw_progress_monotonic = 0.0
        self.teb_reorientation_total_yaw = 0.0
        self.teb_reorientation_deferrals = 0
        self.teb_reorientation_last_status_monotonic = 0.0
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
        self.persistent_target_goal_pub = rospy.Publisher(
            self.persistent_target_goal_topic, PoseStamped, queue_size=1, latch=True
        )
        self.persistent_target_command_pub = rospy.Publisher(
            self.persistent_target_command_topic,
            PersistentGoalCommand,
            queue_size=1,
            latch=True,
        )
        self.persistent_mission_goal_pub = rospy.Publisher(
            self.persistent_mission_goal_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )
        self.persistent_mission_command_pub = rospy.Publisher(
            self.persistent_mission_command_topic,
            PersistentGoalCommand,
            queue_size=1,
            latch=True,
        )
        self.bridge_status_pub = rospy.Publisher(
            self.bridge_status_topic, String, queue_size=10, latch=True
        )
        # Make a hot bridge/move_base restart start with no speculative target
        # request. GoalManager's latched mission command will immediately
        # reissue the current request when there actually is one.
        self._clear_persistent_target_request_locked("bridge_startup", force=True)
        rospy.Subscriber(self.goal_topic, PoseStamped, self.on_goal, queue_size=1)
        rospy.Subscriber(self.intent_topic, String, self.on_intent, queue_size=1)
        if self.use_goal_command:
            rospy.Subscriber(
                self.goal_command_topic, String, self.on_goal_command, queue_size=1
            )
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
        rospy.Subscriber(
            self.teb_feedback_topic,
            FeedbackMsg,
            self.on_teb_feedback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.teb_planner_cmd_topic,
            Twist,
            self.on_teb_planner_command,
            queue_size=20,
        )
        rospy.Subscriber(
            self.navfn_plan_topic, Path, self.on_navfn_plan, queue_size=2
        )
        rospy.Subscriber(
            self.local_costmap_topic,
            OccupancyGrid,
            self.on_local_costmap,
            queue_size=1,
        )
        rospy.Subscriber(self.pose_topic, Pose2D, self.on_pose, queue_size=10)
        rospy.Subscriber(self.controller_topic, String, self.on_mode, queue_size=1)
        rospy.Subscriber(self.task_done_topic, Bool, self.on_task_done, queue_size=1)
        rospy.Subscriber(
            self.persistent_target_plan_result_topic,
            String,
            self.on_persistent_target_plan_result,
            queue_size=10,
        )
        rospy.Subscriber(
            self.persistent_frontier_endpoint_topic,
            PoseStamped,
            self.on_persistent_frontier_endpoint_reached,
            queue_size=10,
        )
        rospy.Timer(rospy.Duration(0.2), self.on_timer)

        rospy.loginfo(
            "TEB goal bridge ready: action=move_base mode=%s active_mode=%s "
            "goal=%s command=%s atomic=%s global_frame=%s persistent_mission=%s",
            self.mode,
            self.active_mode,
            self.goal_topic,
            self.goal_command_topic,
            self.use_goal_command,
            self.global_frame,
            self.persistent_mission_goal_topic,
        )

    def publish_bridge_status(self, event, **fields):
        payload = {
            "event": str(event),
            "mode": self.mode,
            "persistent_execution": bool(self.persistent_execution),
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
            "use_goal_command": bool(self.use_goal_command),
            "goal_command_topic": self.goal_command_topic,
            "latest_goal_transaction_id": int(self.latest_goal_transaction_id),
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
            "active_route_id": int(self.active_route_id),
            "latest_route_id": int(self.latest_route_id),
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
            "navfn_plan_topic": self.navfn_plan_topic,
            "navfn_path_remaining": (
                None
                if self.active_navfn_remaining is None
                else round(float(self.active_navfn_remaining), 4)
            ),
            "navfn_path_endpoint": self.active_navfn_plan_endpoint,
            "navfn_path_progress_age": (
                None
                if self.active_navfn_progress_monotonic <= 0.0
                else round(
                    max(0.0, time.monotonic() - self.active_navfn_progress_monotonic),
                    4,
                )
            ),
            "physical_motion_progress_age": (
                None
                if self.active_motion_progress_monotonic <= 0.0
                else round(
                    max(0.0, time.monotonic() - self.active_motion_progress_monotonic),
                    4,
                )
            ),
            "teb_selected_velocity": (
                None
                if self.latest_teb_selected_linear is None
                else {
                    "linear_x": round(float(self.latest_teb_selected_linear), 4),
                    "angular_z": round(float(self.latest_teb_selected_angular), 4),
                    "age": round(
                        max(0.0, time.monotonic() - self.latest_teb_feedback_monotonic),
                        3,
                    ),
                }
            ),
            "teb_planner_command": (
                None
                if self.latest_teb_planner_linear is None
                else {
                    "linear_x": round(float(self.latest_teb_planner_linear), 4),
                    "angular_z": round(float(self.latest_teb_planner_angular), 4),
                    "age": round(
                        max(
                            0.0,
                            time.monotonic()
                            - self.latest_teb_planner_command_monotonic,
                        ),
                        3,
                    ),
                    "stationary_for": round(
                        max(0.0, time.monotonic() - self.teb_planner_stationary_since),
                        3,
                    ) if self.teb_planner_stationary_since > 0.0 else 0.0,
                }
            ),
            "teb_reorientation": {
                "active": bool(self.teb_reorientation_started_monotonic > 0.0),
                "duration": (
                    0.0
                    if self.teb_reorientation_started_monotonic <= 0.0
                    else round(
                        max(0.0, time.monotonic() - self.teb_reorientation_started_monotonic),
                        3,
                    )
                ),
                "yaw_progress": round(float(self.teb_reorientation_total_yaw), 4),
                "deferrals": int(self.teb_reorientation_deferrals),
            },
            "priority_handoffs": int(self.priority_handoff_count),
            "target_segment_handoffs": int(self.target_segment_handoff_count),
            "frontier_segment_handoffs": int(self.frontier_segment_handoff_count),
            "frontier_observation_completions": int(
                self.frontier_observation_completion_count
            ),
            "frontier_continuous_prefetch_handoffs": int(
                self.frontier_continuous_prefetch_handoff_count
            ),
            "frontier_continuous_prefetch_fallbacks": int(
                self.frontier_continuous_prefetch_handoff_fallback_count
            ),
            "frontier_continuous_prefetch_pending": (
                None
                if self.frontier_continuous_prefetch_handoff_pending is None
                else {
                    "route_id": int(
                        self.frontier_continuous_prefetch_handoff_pending["route_id"]
                    ),
                    "successor_route_id": int(
                        self.frontier_continuous_prefetch_handoff_pending[
                            "successor_route_id"
                        ]
                    ),
                }
            ),
            "prefetched_frontier": (
                None
                if self.prefetched_frontier_goal is None
                else [
                    round(float(self.prefetched_frontier_goal[0]), 3),
                    round(float(self.prefetched_frontier_goal[1]), 3),
                ]
            ),
            "prefetched_frontier_route_id": int(self.prefetched_frontier_route_id),
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
            int(self.latest_route_id),
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

        The latch is intentionally cleared by a new visual track or by a
        completed map-frontier observation, never by an arbitrary detector
        frame or a short target-reacquisition segment. A completed observation
        may use actionlib SUCCEEDED or the bridge's explicit terminal contract.
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
        if self.use_goal_command:
            return
        source = "unknown"
        priority = 0
        route_kind = ""
        route_id = 0
        intent_goal = None
        target_epoch = 0
        target_track_id = ""
        try:
            payload = json.loads(message.data)
            if isinstance(payload, dict):
                source = str(payload.get("source", source)).strip().lower() or source
                priority = int(payload.get("priority", priority))
                route_kind = str(payload.get("route_kind", "")).strip().lower()
                route_id = max(0, int(payload.get("route_id", 0) or 0))
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
            self.latest_route_id = route_id
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
        """Consume frontier lifecycle state without treating it as a goal."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        event = str(payload.get("event", "")).strip()
        with self.lock:
            if event == "frontier_prefetched":
                raw_goal = payload.get("pending_goal")
                try:
                    pending_route_id = max(
                        0, int(payload.get("pending_route_id", 0) or 0)
                    )
                    active_route_id = max(
                        0, int(payload.get("route_id", 0) or 0)
                    )
                    x, y = float(raw_goal[0]), float(raw_goal[1])
                except (TypeError, ValueError, IndexError):
                    return
                if (
                    pending_route_id == active_route_id + 1
                    and math.isfinite(x)
                    and math.isfinite(y)
                ):
                    self.prefetched_frontier_goal = (x, y)
                    self.prefetched_frontier_route_id = pending_route_id
                    raw_entry_yaw = payload.get("pending_entry_yaw")
                    try:
                        entry_yaw = float(raw_entry_yaw)
                        self.prefetched_frontier_entry_yaw = (
                            entry_yaw if math.isfinite(entry_yaw) else None
                        )
                    except (TypeError, ValueError):
                        self.prefetched_frontier_entry_yaw = None
                    self.prefetched_frontier_entry_yaw_basis = str(
                        payload.get("pending_entry_yaw_basis", "")
                    ).strip()
                    self.publish_bridge_status(
                        "frontier_prefetch_available",
                        active_route_id=active_route_id,
                        pending_route_id=pending_route_id,
                        pending_goal=[round(x, 3), round(y, 3)],
                        pending_entry_yaw=(
                            None
                            if self.prefetched_frontier_entry_yaw is None
                            else round(self.prefetched_frontier_entry_yaw, 4)
                        ),
                        pending_entry_yaw_basis=(
                            self.prefetched_frontier_entry_yaw_basis or None
                        ),
                    )
                return
            # A promotion, invalidation, or fresh non-pending route makes any
            # cached successor ineligible for the old endpoint.
            if event in (
                "terminal_prefetch_promoted",
                "route_invalidated",
                "frontier_exhausted",
                "terminal_replan_requested",
                "replan_acknowledged",
            ):
                self.prefetched_frontier_goal = None
                self.prefetched_frontier_route_id = 0
                self.prefetched_frontier_entry_yaw = None
                self.prefetched_frontier_entry_yaw_basis = ""
            if event not in ("route_invalidated", "frontier_exhausted"):
                return
            reason = str(payload.get("reason", "unknown"))
            # The persistent move_base action is a controller lease. A frontier
            # invalidation may describe the old route while a visual target is
            # waiting for Navfn validation or has already taken ownership.
            # Cancelling that lease here would erase the only planning cycle
            # which can validate the target and then blank its current mission.
            if self.persistent_execution and (
                self.latest_intent_priority >= 2
                or self.active_intent_priority >= 2
                or self.persistent_target_pending_transaction > 0
            ):
                self.publish_bridge_status(
                    "frontier_invalidation_ignored",
                    reason=reason,
                    latest_transaction_id=int(self.latest_goal_transaction_id),
                    latest_priority=int(self.latest_intent_priority),
                    active_priority=int(self.active_intent_priority),
                    pending_target_transaction=int(
                        self.persistent_target_pending_transaction
                    ),
                )
                return
            frontier_owned = (
                self.active_intent_source == "global_slam_frontier"
                or self.latest_intent_source == "global_slam_frontier"
            )
            if not frontier_owned or not self._is_active_mode():
                return
            stale_goal = self.last_dispatched_goal
            self.cancel_locked("frontier_%s_%s" % (event, reason))
            # Wait for the next explicit frontier publication.  Do not retry
            # the same pose from the bridge timer while the explorer is
            # selecting and validating a replacement branch.
            self.latest_goal = None
            self.last_dispatched_goal = None
            self.last_terminal_goal = None
            self.latest_intent_source = "waiting_global_slam_frontier"
            self.latest_intent_priority = 0
            self.latest_route_kind = ""
            self.latest_route_id = 0
            self.latest_intent_goal = None
            self.publish_bridge_status(
                "frontier_%s" % event,
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

    def _reset_teb_reorientation_locked(self):
        """Forget a native TEB turn when its command/action boundary changes."""
        self.teb_reorientation_started_monotonic = 0.0
        self.teb_reorientation_reference_yaw = None
        self.teb_reorientation_last_yaw_progress_monotonic = 0.0
        self.teb_reorientation_total_yaw = 0.0

    def _teb_reorientation_command_active_locked(self, now):
        """Return true only for a fresh selected TEB in-place command."""
        return bool(
            self.teb_reorientation_enabled
            and self.latest_teb_selected_linear is not None
            and self.latest_teb_selected_angular is not None
            and now - self.latest_teb_feedback_monotonic
            <= self.teb_reorientation_feedback_timeout
            and abs(float(self.latest_teb_selected_linear))
            <= self.teb_reorientation_linear_threshold
            and abs(float(self.latest_teb_selected_angular))
            >= self.teb_reorientation_angular_threshold
        )

    def on_teb_feedback(self, message):
        """Track TEB's selected command without becoming a local controller."""
        trajectories = list(message.trajectories)
        selected_index = int(message.selected_trajectory_idx)
        selected = (
            trajectories[selected_index]
            if 0 <= selected_index < len(trajectories)
            else None
        )
        first = (
            selected.trajectory[0]
            if selected is not None and selected.trajectory
            else None
        )
        now = time.monotonic()
        with self.lock:
            if first is None:
                self.latest_teb_selected_linear = None
                self.latest_teb_selected_angular = None
                self.latest_teb_feedback_monotonic = 0.0
                self._reset_teb_reorientation_locked()
                return
            self.latest_teb_selected_linear = float(first.velocity.linear.x)
            self.latest_teb_selected_angular = float(first.velocity.angular.z)
            self.latest_teb_feedback_monotonic = now
            if not self._teb_reorientation_command_active_locked(now):
                self._reset_teb_reorientation_locked()
            elif (
                self.action_active
                and self.odom_yaw is not None
                and self.teb_reorientation_started_monotonic <= 0.0
            ):
                self.teb_reorientation_started_monotonic = now
                self.teb_reorientation_reference_yaw = float(self.odom_yaw)
                self.teb_reorientation_last_yaw_progress_monotonic = now
                self.teb_reorientation_total_yaw = 0.0

    def on_teb_planner_command(self, message):
        """Track the raw TEB output used for endpoint lifecycle release."""
        now = time.monotonic()
        linear = float(message.linear.x)
        angular = float(message.angular.z)
        with self.lock:
            self.latest_teb_planner_linear = linear
            self.latest_teb_planner_angular = angular
            self.latest_teb_planner_command_monotonic = now
            if abs(linear) <= self.frontier_observation_completion_max_linear_speed:
                if self.teb_planner_stationary_since <= 0.0:
                    self.teb_planner_stationary_since = now
            else:
                self.teb_planner_stationary_since = 0.0

    def on_pose(self, message):
        """Measure real turn progress in odometry, independent of SLAM drift."""
        now = time.monotonic()
        yaw = float(message.theta)
        with self.lock:
            self.odom_yaw = yaw
            self.odom_pose_monotonic = now
            if (
                self.teb_reorientation_started_monotonic <= 0.0
                or self.teb_reorientation_reference_yaw is None
            ):
                return
            yaw_delta = abs(
                self._angle_delta(yaw, self.teb_reorientation_reference_yaw)
            )
            if yaw_delta >= self.teb_reorientation_yaw_progress:
                self.teb_reorientation_total_yaw += yaw_delta
                self.teb_reorientation_reference_yaw = yaw
                self.teb_reorientation_last_yaw_progress_monotonic = now

    def _teb_reorientation_progressing_locked(self, now):
        """Check the finite action-health exemption for a native TEB turn."""
        if (
            not self.action_active
            or self.turn_supervisor_state == "TURNING"
            or not self._teb_reorientation_command_active_locked(now)
            or self.odom_yaw is None
            or now - self.odom_pose_monotonic
            > self.teb_reorientation_feedback_timeout
        ):
            self._reset_teb_reorientation_locked()
            return False
        if self.teb_reorientation_started_monotonic <= 0.0:
            self.teb_reorientation_started_monotonic = now
            self.teb_reorientation_reference_yaw = float(self.odom_yaw)
            self.teb_reorientation_last_yaw_progress_monotonic = now
            self.teb_reorientation_total_yaw = 0.0
            return False
        duration = now - self.teb_reorientation_started_monotonic
        yaw_progress_age = now - self.teb_reorientation_last_yaw_progress_monotonic
        if (
            duration > self.teb_reorientation_max_extension
            or yaw_progress_age > self.teb_reorientation_stagnation_timeout
            or self.teb_reorientation_total_yaw < self.teb_reorientation_yaw_progress
        ):
            return False
        return True

    def on_goal(self, message):
        if self.use_goal_command:
            return
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

    def on_goal_command(self, message):
        """Accept an atomic GoalManager pose-plus-intent transaction."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            rospy.logwarn_throttle(
                3.0, "TEB goal bridge ignored malformed mission_goal transaction"
            )
            return
        if not isinstance(payload, dict) or payload.get("event") != "mission_goal":
            return
        raw_goal = payload.get("goal")
        if not isinstance(raw_goal, (list, tuple)) or len(raw_goal) < 2:
            return
        try:
            x, y = float(raw_goal[0]), float(raw_goal[1])
            yaw = float(payload.get("yaw", 0.0))
            priority = max(0, min(3, int(payload.get("priority", 0))))
            route_id = max(0, int(payload.get("route_id", 0) or 0))
            target_epoch = max(0, int(payload.get("target_epoch", 0) or 0))
            transaction_id = max(0, int(payload.get("transaction_id", 0) or 0))
        except (TypeError, ValueError):
            return
        frame = str(payload.get("frame_id", self.global_frame)).strip().lstrip("/")
        if not frame:
            frame = self.global_frame
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = frame
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = math.sin(0.5 * yaw)
        goal.pose.orientation.w = math.cos(0.5 * yaw)
        with self.lock:
            # Topic queues and latched reconnects can deliver an older command
            # after its successor.  The transaction number is the ownership
            # boundary; accepting the stale payload would reissue an obsolete
            # target request and make its installation acknowledgement race a
            # newer GoalManager decision.
            if (
                transaction_id > 0
                and self.latest_goal_transaction_id > 0
                and transaction_id < self.latest_goal_transaction_id
            ):
                self.publish_bridge_status(
                    "mission_goal_ignored",
                    reason="stale_transaction",
                    received_transaction_id=transaction_id,
                    latest_transaction_id=int(self.latest_goal_transaction_id),
                )
                return
            self.intent_seen = True
            self.latest_intent_source = (
                str(payload.get("source", "unknown")).strip().lower() or "unknown"
            )
            self.latest_intent_priority = priority
            self.latest_route_kind = str(payload.get("route_kind", "")).strip().lower()
            self.latest_route_id = route_id
            self.latest_target_epoch = target_epoch
            self.latest_target_track_id = str(
                payload.get("target_track_id", "")
            ).strip()
            self.latest_intent_goal = (x, y)
            self.latest_goal_transaction_id = transaction_id
            self.latest_goal = self._normalize_goal(goal)
            self.publish_bridge_status(
                "mission_goal_received",
                transaction_id=transaction_id,
                priority=priority,
                source=self.latest_intent_source,
                goal=[round(x, 3), round(y, 3)],
            )
            if priority >= 2:
                # A visual ray is only a requested target.  Do not switch the
                # persistent action's semantic ownership until Navfn has
                # installed a route for this exact transaction.
                self.persistent_installed_target_goal = None
                self.persistent_installed_target_transaction = 0
                self.persistent_target_approach_reported_transaction = 0
                self._request_persistent_target_locked("mission_goal_transaction")
            elif self.persistent_execution:
                # A frontier route is already map validated by GlobalFrontier
                # and may become the approved stream immediately.  It also
                # supersedes any older speculative target request in the C++
                # planner through its larger transaction ID.
                self.persistent_target_pending_transaction = 0
                self.persistent_target_pending_goal = None
                self._clear_persistent_target_request_locked(
                    "superseded_by_non_target_mission"
                )
                self._publish_persistent_mission_goal_locked(
                    "non_target_mission_goal"
                )
                if self.action_active:
                    self._adopt_persistent_mission_goal_locked()
            if self._is_active_mode():
                self.dispatch_locked(force=False, reason="mission_goal_transaction")

    def on_timer(self, _event):
        with self.lock:
            if self._is_active_mode():
                if self.persistent_execution:
                    self._promote_persistent_frontier_prefetch_locked()
                else:
                    self.maybe_handoff_locked()
                self.dispatch_locked(force=False, reason="coalesced_global_goal")

    def _adopt_persistent_mission_goal_locked(self):
        """Advance the semantic route while retaining the actionlib lease."""
        if self.latest_goal is None or not self.action_active:
            return
        semantic_goal = self._goal_in_global_frame(self.latest_goal)
        if semantic_goal is None:
            return
        self.active_goal_global = copy.deepcopy(semantic_goal)
        self.last_dispatched_goal = copy.deepcopy(semantic_goal)
        self.active_intent_source = self.latest_intent_source
        self.active_intent_priority = int(self.latest_intent_priority)
        self.active_route_kind = self.latest_route_kind
        self.active_route_id = int(self.latest_route_id)
        self.active_target_epoch = int(self.latest_target_epoch)
        self.active_target_track_id = self.latest_target_track_id
        self.active_best_distance = None
        self.active_progress_monotonic = time.monotonic()
        self.active_motion_reference = None
        self.active_motion_progress_monotonic = time.monotonic()
        self.active_navfn_plan_points = []
        self.active_navfn_plan_endpoint = None
        self.active_navfn_remaining = None
        self.active_navfn_best_remaining = None
        self.active_navfn_progress_monotonic = 0.0
        self.publish_bridge_status(
            "persistent_mission_path_adopted",
            route_id=int(self.active_route_id),
            priority=int(self.active_intent_priority),
            source=self.active_intent_source,
            transaction_id=int(self.latest_goal_transaction_id),
            target_track_id=self.active_target_track_id,
            goal=[
                round(float(semantic_goal.pose.position.x), 3),
                round(float(semantic_goal.pose.position.y), 3),
            ],
        )

    def _promote_persistent_frontier_prefetch_locked(self):
        """Release one pre-admitted successor before a route-end stop.

        The fallback remains the conservative terminal handoff: a missing
        successor, unavailable Navfn service, or a blocked local horizon must
        never be converted into an optimistic continuous command.  When the
        optional lookahead transaction succeeds, the existing persistent
        action lease receives the successor path while TEB is still outside
        its point-goal tolerance, avoiding the terminal zero pulse.
        """
        if (
            not self.persistent_execution
            or not self.action_active
            or self.task_done
            or self.active_intent_priority != 0
            or self.latest_intent_priority != 0
            or self.active_intent_source != "global_slam_frontier"
            or self.latest_intent_source != "global_slam_frontier"
            or self.active_route_kind != "frontier_endpoint"
            or self.active_route_id <= 0
            or self.active_feedback_distance is None
            or self.active_feedback_distance
            > (
                self.persistent_frontier_lookahead_trigger_distance
                if self.persistent_frontier_lookahead_handoff_enabled
                else self.frontier_observation_completion_radius
            )
            or self.last_dispatched_goal is None
            or self.prefetched_frontier_goal is None
            or self.prefetched_frontier_route_id != self.active_route_id + 1
        ):
            return False
        route_pair = (int(self.active_route_id), int(self.prefetched_frontier_route_id))
        if route_pair in self.persistent_frontier_prefetch_promoted_pairs:
            return False
        admission = None
        if self.persistent_frontier_lookahead_handoff_enabled:
            admitted, admission = self._admit_persistent_frontier_prefetch_locked()
            if not admitted:
                now = time.monotonic()
                if now - self.persistent_prefetch_admission_last_report_monotonic >= 1.0:
                    self.persistent_prefetch_admission_last_report_monotonic = now
                    self.publish_bridge_status(
                        "persistent_frontier_prefetch_admission_deferred",
                        **admission,
                    )
                return False
        elif not self._frontier_observation_speed_ready_locked():
            # Legacy persistent behavior deliberately waits for TEB's own
            # terminal speed envelope. It remains available as the A/B
            # baseline for the pre-admitted RouteCorridor handoff above.
            return False
        source_goal = copy.deepcopy(self.last_dispatched_goal)
        feedback_distance = float(self.active_feedback_distance)
        self.persistent_frontier_prefetch_promoted_pairs.add(route_pair)
        source_goal.header.stamp = rospy.Time.now()
        self.last_terminal_goal = copy.deepcopy(source_goal)
        self._clear_target_failure_locked("persistent_frontier_prefetch_progress")
        # Both GoalManager and GlobalFrontier already treat this topic as the
        # authoritative route-release event. Unlike endpoint mode, no cancel
        # or replacement follows: StreamingNavfnPlanner installs the promoted
        # route under the existing move_base action.
        self.terminal_pub.publish(source_goal)
        self.terminal_count += 1
        transition_kind = (
            str(admission.get("transition_kind", "smooth_handoff"))
            if isinstance(admission, dict)
            else "smooth_handoff"
        )
        self.publish_bridge_status(
            (
                "persistent_frontier_curve_handoff_promoted"
                if transition_kind == "curve_handoff"
                else (
                    "persistent_frontier_lookahead_handoff_promoted"
                    if self.persistent_frontier_lookahead_handoff_enabled
                    else "persistent_frontier_prefetch_promoted"
                )
            ),
            route_id=route_pair[0],
            successor_route_id=route_pair[1],
            feedback_distance=round(feedback_distance, 3),
            completion_radius=round(self.frontier_observation_completion_radius, 3),
            lookahead_trigger_distance=round(
                self.persistent_frontier_lookahead_trigger_distance, 3
            ),
            selected_linear_speed=round(
                float(self.latest_teb_selected_linear), 3
            ),
            completion_max_linear_speed=round(
                float(self.frontier_observation_completion_max_linear_speed), 3
            ),
            lifecycle=(
                "pre_admitted_%s_then_streaming_plan_update" % transition_kind
                if self.persistent_frontier_lookahead_handoff_enabled
                else "logical_terminal_then_streaming_plan_update"
            ),
            transition_kind=transition_kind,
            admission=admission,
        )
        rospy.loginfo(
            "TEB goal bridge promoted persistent frontier successor: "
            "route_id=%d successor_route_id=%d distance=%.2fm lookahead=%s",
            route_pair[0],
            route_pair[1],
            feedback_distance,
            self.persistent_frontier_lookahead_handoff_enabled,
        )
        return True

    def on_terminal_timer(self, _event):
        with self.lock:
            self.terminal_dispatch_timer = None
            if self._is_active_mode():
                self.dispatch_locked(force=False, reason="terminal_followup")

    def schedule_terminal_dispatch_locked(self):
        if self.terminal_dispatch_timer is None and not rospy.is_shutdown():
            self.terminal_dispatch_timer = rospy.Timer(
                # This is a lifecycle synchronization barrier, not a planning
                # delay. A short ROS-time interval lets actionlib finish its
                # client-side DONE transition before a callback from Goal
                # Manager can submit the successor action.
                rospy.Duration(0.10), self.on_terminal_timer, oneshot=True
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
                if self.persistent_execution:
                    # PersistentTebLocalPlanner observes this same latch and
                    # returns the one action lease naturally. Cancelling here
                    # would create the avoidable stop/preempt boundary.
                    self.publish_bridge_status(
                        "persistent_task_done_waiting_for_action_terminal"
                    )
                else:
                    self.cancel_locked("task_done")
            elif previous and self._is_active_mode():
                self.persistent_installed_target_goal = None
                self.persistent_installed_target_transaction = 0
                self.persistent_target_approach_reported_transaction = 0
                self.dispatch_locked(force=True, reason="new_task")

    def _install_persistent_target_locked(self, transaction_id, message):
        """Commit a Navfn acknowledgement for exactly one target request.

        This runs from the structured result stream.  The legacy installed
        PoseStamped topic is intentionally observer-only because its header
        sequence is rewritten by ROS publishers and cannot identify a mission.
        """
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
        # The approved mission command must be visible before the target
        # request is cleared. The C++ planner can then atomically promote the
        # exact Navfn-validated path instead of falling back to a stale route.
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
        """Build the typed cross-node ownership command.

        ``Header.seq`` is deliberately left to ROS. Transaction identity must
        remain in this message's explicit field through every relay/restart.
        """
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
        cleared_transaction = int(self.persistent_target_request_transaction)
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
        """Publish a bridge-approved path input for StreamingNavfnPlanner.

        This is the second phase of a priority-2 target transaction.  The
        provisional target has already been checked by Navfn when this method
        is called; frontiers use the same stream directly because their exact
        endpoint was validated by GlobalFrontier before GoalManager emitted it.
        """
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

    def on_persistent_target_plan_result(self, message):
        """Consume typed target lifecycle events from persistent Navfn/TEB."""
        try:
            payload = json.loads(message.data)
            transaction_id = int(payload.get("transaction_id", 0) or 0)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        event = str(payload.get("event", ""))
        result_goal = self._goal_from_persistent_result(payload)
        with self.lock:
            if event == "target_plan_installed":
                if result_goal is None:
                    self.publish_bridge_status(
                        "persistent_target_install_ignored",
                        reason="malformed_result_goal",
                        received_transaction_id=transaction_id,
                    )
                    return
                self._install_persistent_target_locked(transaction_id, result_goal)
                return
            if event == "target_approach":
                if result_goal is None:
                    return
                self._handle_persistent_target_approach_locked(
                    transaction_id, result_goal
                )
                return
            if event != "target_plan_failed":
                return
            # A target request is pending only until its first successful
            # Navfn installation. Later map updates can invalidate that same
            # approved target during a normal replan, after the pending slot
            # has intentionally been cleared. Both phases describe one exact
            # target transaction and must therefore reach GoalManager.
            pending_transaction = int(self.persistent_target_pending_transaction)
            installed_transaction = int(self.persistent_installed_target_transaction)
            matching_target_transaction = (
                transaction_id == pending_transaction
                or transaction_id == installed_transaction
            )
            if (
                not self.persistent_execution
                or self.task_done
                or self.latest_goal is None
                or self.latest_intent_priority < 2
                or transaction_id <= 0
                or transaction_id != int(self.latest_goal_transaction_id)
                or not matching_target_transaction
            ):
                self.publish_bridge_status(
                    "persistent_target_plan_failure_ignored",
                    received_transaction_id=transaction_id,
                    latest_transaction_id=int(self.latest_goal_transaction_id),
                    pending_transaction_id=pending_transaction,
                    installed_transaction_id=installed_transaction,
                )
                return
            if self.target_failure_latched:
                return
            source_goal = copy.deepcopy(self.latest_goal)
            self.target_failure_latched = True
            self.target_failure_goal = copy.deepcopy(source_goal)
            self.target_failure_epoch = int(self.latest_target_epoch)
            self.target_failure_track_id = self.latest_target_track_id
            self.target_failure_generation = int(self.action_generation)
            self.target_failure_count += 1
            self.persistent_target_pending_transaction = 0
            self.persistent_target_pending_goal = None
            self.persistent_installed_target_goal = None
            self.persistent_installed_target_transaction = 0
            self._clear_persistent_target_request_locked("target_plan_failed")
            failure = {
                "event": "target_route_failed",
                "reason": "persistent_navfn_target_unreachable",
                "status": "NAVFN_NO_PATH",
                "goal": [
                    round(float(source_goal.pose.position.x), 4),
                    round(float(source_goal.pose.position.y), 4),
                ],
                "goal_frame": source_goal.header.frame_id,
                "target_epoch": int(self.latest_target_epoch),
                "target_track_id": self.latest_target_track_id,
                "transaction_id": transaction_id,
                "failure_count": int(self.target_failure_count),
            }
            self.target_failure_pub.publish(String(data=json.dumps(failure, sort_keys=True)))
            self.publish_bridge_status(
                "persistent_target_plan_failed",
                transaction_id=transaction_id,
                reason=failure["reason"],
                target_track_id=self.latest_target_track_id,
                lifecycle="release_to_frontier_without_action_cancel",
            )

    def on_persistent_frontier_endpoint_reached(self, message):
        """Advance one persistent frontier after TEB reaches its endpoint.

        PersistentTebLocalPlanner deliberately keeps MoveBase's one action
        alive, so its native ``isGoalReached`` cannot be the frontier
        lifecycle terminal. The plugin emits this endpoint report instead.
        Treat it as an untrusted local observation: only the bridge knows the
        active route transaction and only the bridge may publish the logical
        terminal consumed by GlobalFrontier.
        """
        with self.lock:
            if (
                not self.persistent_execution
                or self.task_done
                or not self.action_active
                or self.active_intent_source != "global_slam_frontier"
                or self.active_intent_priority != 0
                or self.active_route_kind != "frontier_endpoint"
                or self.active_route_id <= 0
            ):
                return
            # A target request can arrive after the active frontier has
            # reached its local endpoint but before its target plan installs.
            # Never use that stale frontier report to overwrite the
            # higher-priority ownership decision.
            if (
                self.latest_intent_source != "global_slam_frontier"
                or self.latest_intent_priority != 0
                or self.latest_route_kind != "frontier_endpoint"
                or self.latest_route_id != self.active_route_id
            ):
                self.publish_bridge_status(
                    "persistent_frontier_endpoint_ignored",
                    reason="newer_non_frontier_or_route_intent",
                    route_id=int(self.active_route_id),
                    latest_route_id=int(self.latest_route_id),
                    latest_source=self.latest_intent_source,
                    latest_priority=int(self.latest_intent_priority),
                )
                return
            route_id = int(self.active_route_id)
            if route_id in self.persistent_frontier_endpoint_terminal_routes:
                return
            if self.last_dispatched_goal is None:
                self.publish_bridge_status(
                    "persistent_frontier_endpoint_ignored",
                    reason="missing_active_endpoint",
                    route_id=route_id,
                )
                return
            reported_goal = self._goal_in_global_frame(message)
            canonical_goal = self._goal_in_global_frame(self.last_dispatched_goal)
            active_goal = self._goal_in_global_frame(self.active_goal_global)
            if (
                reported_goal is None
                or canonical_goal is None
                or active_goal is None
            ):
                self.publish_bridge_status(
                    "persistent_frontier_endpoint_ignored",
                    reason="endpoint_transform_failed",
                    route_id=route_id,
                )
                return
            canonical_delta = math.hypot(
                float(reported_goal.pose.position.x)
                - float(canonical_goal.pose.position.x),
                float(reported_goal.pose.position.y)
                - float(canonical_goal.pose.position.y),
            )
            action_delta = math.hypot(
                float(reported_goal.pose.position.x)
                - float(active_goal.pose.position.x),
                float(reported_goal.pose.position.y)
                - float(active_goal.pose.position.y),
            )
            endpoint_epsilon = max(
                self.position_epsilon,
                self.persistent_frontier_admission_endpoint_epsilon,
            )
            if (
                canonical_delta > endpoint_epsilon
                or action_delta > endpoint_epsilon
            ):
                self.publish_bridge_status(
                    "persistent_frontier_endpoint_ignored",
                    reason="endpoint_mismatch",
                    route_id=route_id,
                    canonical_delta=round(canonical_delta, 4),
                    action_delta=round(action_delta, 4),
                    endpoint_epsilon=round(endpoint_epsilon, 4),
                    reported_goal=[
                        round(float(reported_goal.pose.position.x), 3),
                        round(float(reported_goal.pose.position.y), 3),
                    ],
                    canonical_goal=[
                        round(float(canonical_goal.pose.position.x), 3),
                        round(float(canonical_goal.pose.position.y), 3),
                    ],
                    action_goal=[
                        round(float(active_goal.pose.position.x), 3),
                        round(float(active_goal.pose.position.y), 3),
                    ],
                )
                return
            terminal_goal = copy.deepcopy(self.last_dispatched_goal)
            terminal_goal.header.stamp = rospy.Time.now()
            self.persistent_frontier_endpoint_terminal_routes.add(route_id)
            self.last_terminal_goal = copy.deepcopy(terminal_goal)
            self._clear_target_failure_locked("persistent_frontier_endpoint")
            self.terminal_pub.publish(terminal_goal)
            self.terminal_count += 1
            self.publish_bridge_status(
                "persistent_frontier_endpoint_terminal",
                route_id=route_id,
                successor_route_id=int(self.prefetched_frontier_route_id),
                canonical_delta=round(canonical_delta, 4),
                action_delta=round(action_delta, 4),
                endpoint_epsilon=round(endpoint_epsilon, 4),
                lifecycle=(
                    "local_teb_endpoint_reached_then_prefetched_successor"
                    if self.prefetched_frontier_goal is not None
                    else "local_teb_endpoint_reached_then_fresh_frontier"
                ),
            )
            rospy.loginfo(
                "TEB goal bridge accepted persistent frontier endpoint: "
                "route_id=%d successor_route_id=%d canonical_delta=%.3fm "
                "action_delta=%.3fm",
                route_id,
                int(self.prefetched_frontier_route_id),
                canonical_delta,
                action_delta,
            )

    def _handle_persistent_target_approach_locked(self, transaction_id, message):
        """Forward an identity-bound persistent target arrival to GoalManager."""
        if not self.persistent_execution or self.task_done:
            return
        if (
            self.latest_goal is None
            or self.latest_intent_priority < 2
            or self.persistent_target_approach_reported_transaction
            == int(self.latest_goal_transaction_id)
        ):
            self.publish_bridge_status(
                "persistent_target_approach_ignored",
                received_transaction_id=transaction_id,
                latest_transaction_id=int(self.latest_goal_transaction_id),
            )
            return
        latest_global = self._goal_in_global_frame(self.latest_goal)
        if latest_global is None or not self._same_goal(latest_global, message):
            self.publish_bridge_status(
                "persistent_target_approach_ignored",
                reason="endpoint_mismatch",
                received_transaction_id=transaction_id,
                latest_transaction_id=int(self.latest_goal_transaction_id),
                received_goal=[
                    round(message.pose.position.x, 3),
                    round(message.pose.position.y, 3),
                ],
                latest_goal=(
                    None
                    if latest_global is None
                    else [
                        round(latest_global.pose.position.x, 3),
                        round(latest_global.pose.position.y, 3),
                    ]
                ),
            )
            self._republish_persistent_target_request_locked(
                "stale_target_approach_endpoint"
            )
            return
        strict_transaction_match = (
            transaction_id > 0
            and transaction_id == int(self.latest_goal_transaction_id)
            and self.persistent_installed_target_goal is not None
            and transaction_id == int(self.persistent_installed_target_transaction)
            and self._same_goal(self.persistent_installed_target_goal, message)
        )
        if not strict_transaction_match:
            self.publish_bridge_status(
                "persistent_target_approach_ignored",
                reason="missing_matching_installation",
                received_transaction_id=transaction_id,
                latest_transaction_id=int(self.latest_goal_transaction_id),
            )
            return
        terminal_goal = copy.deepcopy(self.latest_goal)
        terminal_goal.header.stamp = rospy.Time.now()
        self.last_terminal_goal = copy.deepcopy(terminal_goal)
        self.persistent_target_approach_reported_transaction = int(
            self.latest_goal_transaction_id
        )
        self.terminal_pub.publish(terminal_goal)
        self.terminal_count += 1
        self.publish_bridge_status(
            "persistent_target_approach_terminal",
            transaction_id=int(self.latest_goal_transaction_id),
            received_transaction_id=transaction_id,
            identity_mode="transaction",
            target_epoch=int(self.latest_target_epoch),
            target_track_id=self.latest_target_track_id,
            terminal_goal=[
                round(terminal_goal.pose.position.x, 3),
                round(terminal_goal.pose.position.y, 3),
            ],
            lifecycle="single_action_target_approach",
        )
        rospy.loginfo(
            "TEB goal bridge forwarded persistent target approach: "
            "transaction=%d received=%d track=%s",
            self.latest_goal_transaction_id,
            transaction_id,
            self.latest_target_track_id,
        )

    def cancel_locked(self, reason):
        self.action_generation += 1
        if self.action_active:
            self.action_client.cancel_goal()
        self.action_active = False
        self.handoff_requested = False
        self.frontier_observation_completion_pending = None
        self.frontier_continuous_prefetch_handoff_pending = None
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
        self.active_motion_reference = None
        self.active_motion_progress_monotonic = 0.0
        self.active_navfn_plan_points = []
        self.active_navfn_plan_endpoint = None
        self.active_navfn_remaining = None
        self.active_navfn_best_remaining = None
        self.active_navfn_progress_monotonic = 0.0
        self.last_feedback_monotonic = 0.0
        self._reset_teb_reorientation_locked()
        self.frontier_stale_wait_started_monotonic = 0.0
        self.turn_transition_ready = False
        self.move_base_terminal_pending = False
        self.active_intent_source = "unknown"
        self.active_intent_priority = 0
        self.active_route_kind = ""
        self.active_route_id = 0
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

    def _frontier_prefetch_heading_delta_locked(self, source_goal, successor):
        """Return the local turn needed to enter a prefetched frontier branch.

        A prefetched point is intentionally not promoted to ``latest_goal``
        until its source observation region is terminal.  Consequently the
        generic ``_pending_heading_delta_locked`` cannot evaluate it.  Compare
        both bearings at the latest action feedback pose instead: this is the
        turn TEB would need to absorb if the action were replaced in-place.
        """
        feedback = self.active_feedback_pose
        if feedback is None:
            return None, "feedback_unavailable"
        # This route tangent is calculated by GlobalFrontier from the actual
        # connected BFS path, with the same map/costmap validation that admits
        # the successor. Compare it directly to the current base yaw at the
        # action boundary. The old endpoint-bearing calculation below is only
        # a compatibility fallback for an external/older frontier publisher.
        if self.prefetched_frontier_entry_yaw is not None:
            _base_x, _base_y, base_yaw = feedback
            return (
                abs(self._angle_delta(self.prefetched_frontier_entry_yaw, base_yaw)),
                self.prefetched_frontier_entry_yaw_basis or "bfs_initial_tangent",
            )
        if source_goal is None or successor is None:
            return None, "endpoint_bearing_unavailable"
        source_global = self._goal_in_global_frame(source_goal)
        if source_global is None:
            return None, "endpoint_bearing_unavailable"
        try:
            successor_x, successor_y = float(successor[0]), float(successor[1])
        except (TypeError, ValueError, IndexError):
            return None, "endpoint_bearing_unavailable"
        if not (math.isfinite(successor_x) and math.isfinite(successor_y)):
            return None, "endpoint_bearing_unavailable"
        base_x, base_y, _base_yaw = feedback
        active_bearing = math.atan2(
            source_global.pose.position.y - base_y,
            source_global.pose.position.x - base_x,
        )
        successor_bearing = math.atan2(
            successor_y - base_y,
            successor_x - base_x,
        )
        return abs(self._angle_delta(successor_bearing, active_bearing)), "legacy_endpoint_bearing"

    def _navfn_entry_tangent_locked(self, path, feedback_xy):
        """Return the actual Navfn entry heading sampled from live feedback.

        Frontier prefetch metadata is useful for choosing a candidate, but it
        can be stale by the time the bridge receives a new Navfn plan.  This
        helper samples the plan returned by the admission request itself, so
        the handoff state describes the route passed to PersistentTEB.
        """
        if path is None or len(path.poses) < 2 or feedback_xy is None:
            return None
        start_x, start_y = float(feedback_xy[0]), float(feedback_xy[1])
        previous_x, previous_y = start_x, start_y
        travelled = 0.0
        sample_distance = self.persistent_frontier_entry_tangent_distance
        for pose in path.poses:
            point = pose.pose.position
            point_x, point_y = float(point.x), float(point.y)
            segment = math.hypot(point_x - previous_x, point_y - previous_y)
            if segment <= 1e-6:
                continue
            if travelled + segment >= sample_distance:
                fraction = (sample_distance - travelled) / segment
                sample_x = previous_x + fraction * (point_x - previous_x)
                sample_y = previous_y + fraction * (point_y - previous_y)
                if math.hypot(sample_x - start_x, sample_y - start_y) > 1e-4:
                    return math.atan2(sample_y - start_y, sample_x - start_x)
            travelled += segment
            previous_x, previous_y = point_x, point_y
        # Short admissible routes still need a deterministic classification.
        # Their final point is the furthest real heading evidence available.
        if math.hypot(previous_x - start_x, previous_y - start_y) > 1e-4:
            return math.atan2(previous_y - start_y, previous_x - start_x)
        return None

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
        retried forever. Once its Navfn route has stopped making route progress
        (or, before a plan is available, the base has stopped moving), this
        method emits one failure event and lets Goal Manager select a map route.
        The bridge-side latch prevents the terminal callback/timer race from
        reissuing the same target.
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
        progress_basis, target_progress_age = self._target_progress_state_locked(now)
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
            "target_progress_basis": progress_basis,
            "target_progress_age": (
                None
                if not math.isfinite(target_progress_age)
                else round(float(target_progress_age), 4)
            ),
            "navfn_path_remaining": (
                None
                if self.active_navfn_remaining is None
                else round(float(self.active_navfn_remaining), 4)
            ),
            "motion_progress_age": (
                None
                if self.active_motion_progress_monotonic <= 0.0
                else round(float(now - self.active_motion_progress_monotonic), 4)
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
            target_progress_basis=progress_basis,
            target_progress_age=payload["target_progress_age"],
            navfn_path_remaining=payload["navfn_path_remaining"],
            motion_progress_age=payload["motion_progress_age"],
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

    def _start_frontier_continuous_prefetch_handoff_locked(self, pending_delta):
        """Promote and replace one validated frontier successor without a stop.

        The lifecycle-only prefetch is intentionally the authority boundary:
        no detector update, map refresh, or arbitrary new frontier is allowed
        through this path.  GlobalFrontier has already checked this exact
        successor against its clearance mask, the live inflated costmap, and
        Navfn.  Once the base enters the frontier's observation envelope, we
        publish a logical terminal and atomically replace the MoveBase goal.
        A successor endpoint is an information boundary rather than a local
        trajectory, so endpoint distance is deliberately not a qualification
        gate. Its validated BFS entry tangent is different: replacing a
        MoveBase action across a large tangent discontinuity necessarily
        creates a zero-linear-velocity turn. Only tangent-continuous branches
        use this path; a sharp branch completes through the normal terminal
        lifecycle and is dispatched afterward.
        """
        if (
            not self.frontier_continuous_prefetch_handoff_enabled
            or self.frontier_continuous_prefetch_handoff_pending is not None
            or self.frontier_observation_completion_pending is not None
            or not self.action_active
            or self.active_intent_priority != 0
            or self.latest_intent_priority != 0
            or self.active_intent_source != "global_slam_frontier"
            or self.latest_intent_source != "global_slam_frontier"
            or self.active_route_kind != "frontier_endpoint"
            or self.latest_route_kind != "frontier_endpoint"
            or self.active_route_id <= 0
            or self.active_feedback_distance is None
            or self.active_feedback_distance
            > self.frontier_observation_completion_radius
            or self.last_dispatched_goal is None
            or self.prefetched_frontier_goal is None
            or self.prefetched_frontier_route_id != self.active_route_id + 1
        ):
            return False

        source_goal = copy.deepcopy(self.last_dispatched_goal)
        feedback_distance = float(self.active_feedback_distance)
        pending_delta = math.hypot(
            self.prefetched_frontier_goal[0] - source_goal.pose.position.x,
            self.prefetched_frontier_goal[1] - source_goal.pose.position.y,
        )
        prefetch_heading_delta, prefetch_heading_basis = self._frontier_prefetch_heading_delta_locked(
            source_goal, self.prefetched_frontier_goal
        )
        if (
            prefetch_heading_delta is not None
            and prefetch_heading_delta > self.frontier_early_handoff_max_heading_delta
        ):
            route_pair = (
                int(self.active_route_id),
                int(self.prefetched_frontier_route_id),
            )
            if route_pair not in self.frontier_prefetch_requires_turn_pairs:
                self.frontier_prefetch_requires_turn_pairs.add(route_pair)
                self.publish_bridge_status(
                    "frontier_prefetch_requires_turn",
                    route_id=route_pair[0],
                    successor_route_id=route_pair[1],
                    heading_delta_deg=round(math.degrees(prefetch_heading_delta), 3),
                    max_heading_delta_deg=round(
                        math.degrees(self.frontier_early_handoff_max_heading_delta), 3
                    ),
                    heading_basis=prefetch_heading_basis,
                    lifecycle="terminal_then_successor_dispatch",
                )
                rospy.loginfo(
                    "TEB goal bridge deferred continuous frontier handoff: "
                    "route_id=%d successor_route_id=%d heading=%.1fdeg limit=%.1fdeg",
                    route_pair[0],
                    route_pair[1],
                    math.degrees(prefetch_heading_delta),
                    math.degrees(self.frontier_early_handoff_max_heading_delta),
                )
            return False
        pending = {
            "generation": int(self.action_generation),
            "route_id": int(self.active_route_id),
            "successor_route_id": int(self.prefetched_frontier_route_id),
            "source_goal": source_goal,
            "feedback_distance": feedback_distance,
            "pending_delta": float(pending_delta),
            "heading_delta": float(prefetch_heading_delta),
            "heading_basis": prefetch_heading_basis,
            "started_monotonic": time.monotonic(),
        }
        self.frontier_continuous_prefetch_handoff_pending = pending
        # This is a logical terminal only. The old action deliberately keeps
        # running until the successor transaction reaches this bridge.
        source_goal.header.stamp = rospy.Time.now()
        self.last_terminal_goal = copy.deepcopy(source_goal)
        self._clear_target_failure_locked("frontier_continuous_prefetch_progress")
        self.terminal_pub.publish(source_goal)
        self.publish_bridge_status(
            "frontier_continuous_prefetch_handoff_requested",
            route_id=int(pending["route_id"]),
            successor_route_id=int(pending["successor_route_id"]),
            completion_radius=round(self.frontier_observation_completion_radius, 3),
            feedback_distance=round(feedback_distance, 3),
            pending_delta=round(float(pending_delta), 3),
            heading_delta_deg=round(math.degrees(prefetch_heading_delta), 3),
            heading_basis=prefetch_heading_basis,
            timeout_seconds=round(
                self.frontier_continuous_prefetch_handoff_timeout, 3
            ),
            lifecycle="logical_terminal_then_native_action_replacement",
        )
        rospy.loginfo(
            "TEB goal bridge requested continuous frontier handoff: "
            "route_id=%d successor_route_id=%d distance=%.2fm delta=%.2fm "
            "heading=%.1fdeg",
            pending["route_id"],
            pending["successor_route_id"],
            feedback_distance,
            pending_delta,
            math.degrees(prefetch_heading_delta),
        )
        return True

    def _expire_frontier_continuous_prefetch_handoff_locked(self, now):
        """Fall back to the audited terminal path if promotion never arrives."""
        pending = self.frontier_continuous_prefetch_handoff_pending
        if pending is None:
            return False
        if now - float(pending["started_monotonic"]) < (
            self.frontier_continuous_prefetch_handoff_timeout
        ):
            return True
        self.frontier_continuous_prefetch_handoff_pending = None
        self.frontier_continuous_prefetch_handoff_fallback_count += 1
        # The source terminal was already delivered to the mission planner.
        # Reuse the existing intentional-observation completion handling so a
        # transport PREEMPTED cannot become an unexpected action failure.
        self.frontier_observation_completion_pending = pending
        self.handoff_requested = True
        if self.action_active:
            self.action_client.cancel_goal()
        self.publish_bridge_status(
            "frontier_continuous_prefetch_handoff_fallback",
            route_id=int(pending["route_id"]),
            successor_route_id=int(pending["successor_route_id"]),
            waited_seconds=round(
                now - float(pending["started_monotonic"]), 3
            ),
            lifecycle="prefetch_promotion_timeout_then_terminal_handoff",
        )
        rospy.logwarn(
            "TEB goal bridge continuous frontier handoff timed out; "
            "falling back to terminal dispatch route_id=%d",
            pending["route_id"],
        )
        return True

    def _complete_frontier_observation_locked(self, pending_delta):
        """Finish a frontier observation region and hand its successor over.

        ``move_base`` has no notion of an exploration observation region: it
        keeps optimizing to a point even after the route has supplied all the
        information the frontier manager needs. Do this only for a committed
        endpoint with either a newly published route command or a validated
        lifecycle-only prefetch. The terminal pose remains the original source
        goal so the frontier manager can atomically promote that exact route.
        """
        completion_radius = self.frontier_observation_completion_radius
        if (
            completion_radius <= 0.0
            or self.frontier_observation_completion_pending is not None
            or not self.action_active
            or self.active_intent_priority != 0
            or self.latest_intent_priority != 0
            or self.active_intent_source != "global_slam_frontier"
            or self.latest_intent_source != "global_slam_frontier"
            or self.active_route_kind != "frontier_endpoint"
            or self.latest_route_kind != "frontier_endpoint"
            or self.active_route_id <= 0
            or self.active_feedback_distance is None
            or self.active_feedback_distance
            > completion_radius
            or not self._frontier_observation_speed_ready_locked()
            or self.last_dispatched_goal is None
        ):
            return False
        published_successor = self.latest_route_id > self.active_route_id
        prefetched_successor = (
            self.prefetched_frontier_goal is not None
            and self.prefetched_frontier_route_id == self.active_route_id + 1
        )
        if not published_successor and not prefetched_successor:
            return False
        if prefetched_successor and not published_successor:
            pending_delta = math.hypot(
                self.prefetched_frontier_goal[0]
                - self.last_dispatched_goal.pose.position.x,
                self.prefetched_frontier_goal[1]
                - self.last_dispatched_goal.pose.position.y,
            )

        source_goal = copy.deepcopy(self.last_dispatched_goal)
        feedback_distance = float(self.active_feedback_distance)
        self.frontier_observation_completion_pending = {
            "generation": int(self.action_generation),
            "route_id": int(self.active_route_id),
            "source_goal": source_goal,
            "feedback_distance": feedback_distance,
            "pending_delta": float(pending_delta),
            "completion_radius": float(completion_radius),
            "successor_route_id": int(
                self.latest_route_id
                if published_successor
                else self.prefetched_frontier_route_id
            ),
        }
        self.handoff_requested = True
        # Publish before canceling so the frontier manager can validate and
        # promote its prefetched successor while actionlib is closing this
        # transport transaction. dispatch_locked keeps that successor behind
        # the result callback and its synchronization timer.
        source_goal.header.stamp = rospy.Time.now()
        self.last_terminal_goal = copy.deepcopy(source_goal)
        self._clear_target_failure_locked("frontier_observation_progress")
        self.terminal_pub.publish(source_goal)
        self.action_client.cancel_goal()
        self.publish_bridge_status(
            "frontier_observation_completion_requested",
            route_id=int(self.active_route_id),
            completion_radius=round(completion_radius, 3),
            feedback_distance=round(feedback_distance, 3),
            pending_delta=round(float(pending_delta), 3),
            successor_route_id=int(
                self.latest_route_id
                if published_successor
                else self.prefetched_frontier_route_id
            ),
            lifecycle="observation_region_then_terminal_handoff",
        )
        rospy.loginfo(
            "TEB goal bridge completed frontier %s region: "
            "route_id=%d distance=%.2fm radius=%.2fm successor_delta=%.2fm",
            "observation",
            self.active_route_id,
            feedback_distance,
            completion_radius,
            pending_delta,
        )
        return True

    def _frontier_observation_speed_ready_locked(self):
        """Return true after TEB has actually held its endpoint speed envelope.

        TEB feedback contains a predicted selected trajectory. At a point-goal
        terminal it can remain at the last cruise value or stop publishing
        entirely, while the planner command has already been zero for several
        control cycles. Prefer the latter when it is fresh; retain the
        feedback fallback for launch configurations without the raw topic.
        """
        now = time.monotonic()
        raw_fresh = (
            self.latest_teb_planner_linear is not None
            and now - self.latest_teb_planner_command_monotonic
            <= self.teb_planner_command_timeout
        )
        if raw_fresh:
            return (
                abs(float(self.latest_teb_planner_linear))
                <= self.frontier_observation_completion_max_linear_speed
                and self.teb_planner_stationary_since > 0.0
                and now - self.teb_planner_stationary_since
                >= self.frontier_observation_stationary_hold
            )
        if self.latest_teb_selected_linear is None:
            return False
        age = now - self.latest_teb_feedback_monotonic
        return (
            age <= self.teb_reorientation_feedback_timeout
            and abs(float(self.latest_teb_selected_linear))
            <= self.frontier_observation_completion_max_linear_speed
        )

    def _target_progress_state_locked(self, now):
        """Return the target-route health clock and the evidence behind it."""
        if (
            self.active_navfn_plan_points
            and self.active_navfn_progress_monotonic > 0.0
        ):
            return (
                "navfn_path_remaining",
                now - self.active_navfn_progress_monotonic,
            )
        if self.active_motion_progress_monotonic > 0.0:
            return (
                "physical_motion",
                now - self.active_motion_progress_monotonic,
            )
        return ("no_feedback", float("inf"))

    def maybe_handoff_locked(self):
        """Cancel a stale action only when a newer intent and health evidence exist."""
        now = time.monotonic()
        if self.frontier_continuous_prefetch_handoff_pending is not None:
            self._expire_frontier_continuous_prefetch_handoff_locked(now)
            return
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
        if now - self.last_dispatch_monotonic < self.handoff_min_interval:
            return
        pending_delta = self._pending_goal_delta_locked()
        # This precedes the generic no-progress watchdog. A route endpoint
        # inside its observation region is not a failed point-goal action, so
        # waiting twelve seconds for ``no_feedback_progress`` is both noisy
        # and semantically wrong.
        if self._start_frontier_continuous_prefetch_handoff_locked(pending_delta):
            return
        if self._complete_frontier_observation_locked(pending_delta):
            return
        stalled = (
            self.active_progress_monotonic > 0.0
            and now - self.active_progress_monotonic >= self.progress_timeout
        )
        # Frontier actions are allowed to finish unless a newer map waypoint
        # is waiting. A visual target is different: once its committed route
        # makes no physical progress, it becomes a mission-level failure and
        # ownership must return to frontier exploration. Retrying the same ray
        # here was the source of the observed 100+ cancel/retry loop.
        target_progress_basis, target_progress_age = self._target_progress_state_locked(now)
        target_stalled = (
            self.active_intent_priority >= 2
            and target_progress_age >= self.progress_timeout
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
        if not stalled and not target_stalled:
            return

        # TEB may correctly rotate in place at a constrained obstacle/corner
        # endpoint. MoveBase feedback then has no XY improvement, but wheel
        # odometry has yaw progress and TEB continues selecting the turn. Do
        # not cancel that healthy action for one bounded reorientation budget.
        # A stale feedback stream, stopped yaw, or exhausted budget falls
        # through to the regular handoff/recovery below.
        if self._teb_reorientation_progressing_locked(now):
            self.teb_reorientation_deferrals += 1
            if (
                now - self.teb_reorientation_last_status_monotonic >= 1.0
            ):
                self.teb_reorientation_last_status_monotonic = now
                duration = now - self.teb_reorientation_started_monotonic
                progress_age = (
                    now - self.teb_reorientation_last_yaw_progress_monotonic
                )
                self.publish_bridge_status(
                    "teb_in_place_reorientation",
                    reason="native_teb_yaw_progress",
                    duration=round(duration, 3),
                    yaw_progress=round(self.teb_reorientation_total_yaw, 4),
                    yaw_progress_age=round(progress_age, 3),
                    extension_budget=round(self.teb_reorientation_max_extension, 3),
                    pending_delta=round(pending_delta, 3),
                    feedback_distance=(
                        None
                        if self.active_feedback_distance is None
                        else round(self.active_feedback_distance, 3)
                    ),
                )
                rospy.loginfo(
                    "TEB goal bridge defers stale handoff during native "
                    "reorientation: yaw_progress=%.1fdeg duration=%.1fs",
                    math.degrees(self.teb_reorientation_total_yaw),
                    duration,
                )
            return

        # A sharp frontier branch is a new route topology, not a continuation
        # of the current local trajectory.  More importantly, a frontier
        # action belongs to move_base once it has been dispatched. TEB can
        # legitimately make no XY progress while it resolves a corner or its
        # own oscillation recovery. Cancelling it here used to turn that local
        # recovery into PREEMPTED -> zero command -> successor action, despite
        # ample lidar clearance. The global-frontier node already observes
        # move_base recovery and terminal failure and will select another
        # route when the execution stack has conclusive evidence.
        if self.active_intent_priority < 2:
            if now - self.frontier_stale_wait_started_monotonic >= 1.0:
                self.frontier_stale_wait_started_monotonic = now
                self.publish_bridge_status(
                    "frontier_stall_observed",
                    pending_delta=round(pending_delta, 3),
                    feedback_distance=(
                        None
                        if self.active_feedback_distance is None
                        else round(self.active_feedback_distance, 3)
                    ),
                    progress_age=round(now - self.active_progress_monotonic, 3)
                    if self.active_progress_monotonic > 0.0
                    else None,
                    action_owner="move_base",
                )
                rospy.loginfo(
                    "TEB goal bridge leaves stalled frontier action to move_base: "
                    "pending_delta=%.2fm feedback_distance=%s progress_age=%.1fs",
                    pending_delta,
                    "n/a"
                    if self.active_feedback_distance is None
                    else "%.2f" % self.active_feedback_distance,
                    now - self.active_progress_monotonic
                    if self.active_progress_monotonic > 0.0
                    else 0.0,
                )
            return

        self.frontier_stale_wait_started_monotonic = 0.0
        if target_stalled:
            self._latch_target_failure_locked(
                "target_%s_stalled" % target_progress_basis,
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
        # Route kind is insufficient: two unrelated branches are both usually
        # labelled ``frontier_endpoint``. Global Frontier preserves route_id
        # only after its discrete BFS path-prefix test proves continuation.
        route_continuation = (
            self.active_intent_priority == 0
            and self.latest_intent_priority == 0
            and self.active_intent_source == "global_slam_frontier"
            and self.latest_intent_source == "global_slam_frontier"
            and self.active_route_id > 0
            and self.active_route_id == self.latest_route_id
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
        # A new branch is an action boundary. Do not let a generic launch
        # compatibility flag bypass this topology contract near an endpoint.
        if self.active_intent_priority == 0 and not route_continuation:
            return False
        # Do not let a compatibility flag turn arbitrary visual updates into
        # rolling targets. The dispatcher also applies this predicate; keeping
        # it here makes the segment routine safe for future call sites.
        if (
            self.active_intent_priority >= 2
            and not self._safe_target_segment_pending_locked()
        ):
            return False
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
            and self.active_route_id > 0
            and self.active_route_id == self.latest_route_id
            and self.latest_route_kind != "frontier_turn_connector"
        )

    def _safe_target_segment_pending_locked(self):
        """Return whether the queued goal advances the active visual track.

        GoalManager supplies this successor only after new detector evidence
        and Navfn validation. This gate further requires the same stable track
        identity; distance, delta, and TEB warm-start limits are enforced by
        ``maybe_segment_handoff_locked``.
        """
        return (
            self.active_intent_priority == 2
            and self.latest_intent_priority == 2
            and self.active_intent_source.startswith("target_")
            and self.latest_intent_source.startswith("target_")
            and bool(self.active_target_track_id)
            and self.active_target_track_id == self.latest_target_track_id
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
        # ``terminal_pub`` wakes Goal Manager immediately, which in turn
        # publishes the next mission transaction before SimpleActionClient has
        # finished the current done callback. Do not let that subscriber path
        # bypass the one-shot terminal timer: it otherwise creates an actionlib
        # "goal handle not tracking" race. Explicit controller transitions
        # retain ``force`` authority to override this short barrier.
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

        # The mission layer may publish a new observation or frontier at any
        # time, but an action is an atomic navigation intent. Coalesce updates
        # until the current intent has a result. Only lifecycle transitions
        # use force=True and cancel the active action.
        if self.action_active and not force:
            # In persistent execution the one active move_base action is a
            # lease for the controller loop, not ownership of a specific
            # frontier. StreamingNavfnPlanner receives every latest mission
            # goal directly and TEB receives its new plan without an action
            # terminal. Never replace this lease for an ordinary goal update.
            if self.persistent_execution:
                return
            continuous_handoff = self.frontier_continuous_prefetch_handoff_pending
            if continuous_handoff is not None:
                # Accept only the successor that GlobalFrontier emitted after
                # revalidating the exact prefetched branch. This is not a
                # generic same-priority replacement gate.
                if (
                    self.latest_intent_source == "global_slam_frontier"
                    and self.latest_intent_priority == 0
                    and self.latest_route_kind == "frontier_endpoint"
                    and self.latest_route_id
                    == int(continuous_handoff["successor_route_id"])
                    and not self._same_goal(
                        self.latest_goal, self.last_dispatched_goal
                    )
                ):
                    fields = {
                        "route_id": int(continuous_handoff["route_id"]),
                        "successor_route_id": int(
                            continuous_handoff["successor_route_id"]
                        ),
                        "feedback_distance": round(
                            float(continuous_handoff["feedback_distance"]), 3
                        ),
                        "pending_delta": round(
                            float(continuous_handoff["pending_delta"]), 3
                        ),
                        "handoff_wait_seconds": round(
                            time.monotonic()
                            - float(continuous_handoff["started_monotonic"]),
                            4,
                        ),
                    }
                    if self._send_goal_locked(
                        self.latest_goal,
                        reason="frontier_continuous_prefetch_handoff",
                        replacement=True,
                        replacement_kind="frontier_prefetched_successor",
                        **fields
                    ):
                        self.frontier_continuous_prefetch_handoff_pending = None
                        self.frontier_continuous_prefetch_handoff_count += 1
                        self.terminal_count += 1
                        self.publish_bridge_status(
                            "frontier_continuous_prefetch_handoff_completed",
                            **fields
                        )
                        rospy.loginfo(
                            "TEB goal bridge replaced validated frontier "
                            "successor without stop: route_id=%d -> %d",
                            fields["route_id"], fields["successor_route_id"],
                        )
                    return
                # The old action remains valid during the bounded promotion
                # wait. `maybe_handoff_locked` owns timeout and fallback.
                return
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
            allow_target_handoff = self._safe_target_segment_pending_locked()
            if (
                (
                    self.allow_in_place_replacement
                    or allow_route_handoff
                    or allow_target_handoff
                )
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
        self.active_route_id = int(self.latest_route_id)
        self.active_target_epoch = int(self.latest_target_epoch)
        self.active_target_track_id = self.latest_target_track_id
        self.handoff_requested = False
        self.frontier_observation_completion_pending = None
        self.active_goal_global = copy.deepcopy(execution_goal)
        self.active_feedback_distance = None
        self.active_best_distance = None
        self.active_progress_monotonic = time.monotonic()
        self.active_motion_reference = None
        self.active_motion_progress_monotonic = time.monotonic()
        self.active_navfn_plan_points = []
        self.active_navfn_plan_endpoint = None
        self.active_navfn_remaining = None
        self.active_navfn_best_remaining = None
        self.active_navfn_progress_monotonic = 0.0
        self.last_feedback_monotonic = 0.0
        self._reset_teb_reorientation_locked()
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

    def _path_remaining_distance(self, points, x, y):
        """Return remaining arc length after projecting ``(x, y)`` onto a path."""
        if not points:
            return None
        if len(points) == 1:
            return math.hypot(points[0][0] - x, points[0][1] - y)
        suffix = [0.0] * len(points)
        for index in range(len(points) - 2, -1, -1):
            suffix[index] = suffix[index + 1] + math.hypot(
                points[index + 1][0] - points[index][0],
                points[index + 1][1] - points[index][1],
            )
        nearest_distance_sq = float("inf")
        remaining = None
        for index in range(len(points) - 1):
            ax, ay = points[index]
            bx, by = points[index + 1]
            dx = bx - ax
            dy = by - ay
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                continue
            projection = ((x - ax) * dx + (y - ay) * dy) / (length * length)
            projection = min(1.0, max(0.0, projection))
            px = ax + projection * dx
            py = ay + projection * dy
            distance_sq = (x - px) * (x - px) + (y - py) * (y - py)
            if distance_sq < nearest_distance_sq:
                nearest_distance_sq = distance_sq
                remaining = (1.0 - projection) * length + suffix[index + 1]
        return remaining

    def _update_navfn_path_progress_locked(self, now):
        if (
            self.active_feedback_pose is None
            or not self.active_navfn_plan_points
        ):
            return
        x, y, _ = self.active_feedback_pose
        remaining = self._path_remaining_distance(
            self.active_navfn_plan_points, x, y
        )
        if remaining is None:
            return
        self.active_navfn_remaining = remaining
        if (
            self.active_navfn_best_remaining is None
            or remaining < self.active_navfn_best_remaining - self.progress_epsilon
        ):
            self.active_navfn_best_remaining = remaining
            self.active_navfn_progress_monotonic = now

    def on_navfn_plan(self, message):
        """Bind the active target health check to Navfn's actual route.

        The endpoint match is deliberate: ``NavfnROS/plan`` is a shared topic,
        so a plan for a superseded action must never refresh a newer target's
        watchdog.  Navfn may snap its final pose to a nearby costmap cell.
        """
        with self.lock:
            if (
                not self.action_active
                or self.active_goal_global is None
                or len(message.poses) < 2
            ):
                return
            frame = (message.header.frame_id or message.poses[-1].header.frame_id)
            frame = (frame or "").strip().lstrip("/")
            if frame != self.global_frame:
                return
            points = [
                (float(pose.pose.position.x), float(pose.pose.position.y))
                for pose in message.poses
            ]
            endpoint = points[-1]
            goal = self.active_goal_global.pose.position
            if math.hypot(endpoint[0] - goal.x, endpoint[1] - goal.y) > 0.50:
                return
            self.active_navfn_plan_points = points
            self.active_navfn_plan_endpoint = [
                round(endpoint[0], 3), round(endpoint[1], 3)
            ]
            self.active_navfn_remaining = None
            self.active_navfn_best_remaining = None
            self.active_navfn_progress_monotonic = 0.0
            self._update_navfn_path_progress_locked(time.monotonic())

    def on_local_costmap(self, message):
        """Keep the most recent rolling lidar map for route admission only."""
        with self.lock:
            self.local_costmap = message
            self.local_costmap_received_monotonic = time.monotonic()

    @staticmethod
    def _local_costmap_cell(message, x, y):
        """Return the row-major cell index for one local-costmap point."""
        resolution = float(message.info.resolution)
        if resolution <= 1e-9:
            return None
        col = int(math.floor((float(x) - message.info.origin.position.x) / resolution))
        row = int(math.floor((float(y) - message.info.origin.position.y) / resolution))
        if row < 0 or col < 0 or row >= int(message.info.height) or col >= int(message.info.width):
            return None
        return row * int(message.info.width) + col

    def _local_costmap_prefix_admission_locked(self, path):
        """Check a Navfn route's near prefix against the current lidar map.

        Navfn validates the global SLAM costmap, which can legitimately lag a
        newly seen wall or doorway.  Before a running TEB band is redirected
        to a prefetched branch, sample the first local horizon in the exact
        rolling costmap that TEB will use.  This is intentionally conservative
        at lethal/unknown cells; inflated but traversable cells remain TEB's
        optimization problem rather than a duplicated obstacle policy here.
        """
        message = self.local_costmap
        now = time.monotonic()
        if message is None:
            return False, {"reason": "local_costmap_unavailable"}
        age = now - self.local_costmap_received_monotonic
        if age > self.persistent_frontier_admission_max_costmap_age:
            return False, {
                "reason": "local_costmap_stale",
                "costmap_age_seconds": round(age, 3),
            }
        if len(path.poses) < 2:
            return False, {"reason": "navfn_plan_too_short"}
        local_frame = (message.header.frame_id or "").strip().lstrip("/")
        plan_frame = (
            path.header.frame_id
            or path.poses[0].header.frame_id
            or self.global_frame
        ).strip().lstrip("/")
        if not local_frame or not plan_frame:
            return False, {"reason": "local_costmap_frame_unavailable"}
        try:
            if local_frame == plan_frame:
                translation = (0.0, 0.0, 0.0)
                rotation = (0.0, 0.0, 0.0, 1.0)
            else:
                translation, rotation = self.tf_listener.lookupTransform(
                    local_frame, plan_frame, rospy.Time(0)
                )
        except (
            tf.Exception,
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as exc:
            return False, {
                "reason": "local_costmap_transform_unavailable",
                "error": type(exc).__name__,
                "local_frame": local_frame,
                "plan_frame": plan_frame,
            }
        yaw = tf.transformations.euler_from_quaternion(rotation)[2]
        cosine, sine = math.cos(yaw), math.sin(yaw)
        previous = None
        traversed = 0.0
        checked_cells = set()
        max_cost = 0
        for pose in path.poses:
            px = float(pose.pose.position.x)
            py = float(pose.pose.position.y)
            if previous is not None:
                traversed += math.hypot(px - previous[0], py - previous[1])
            previous = (px, py)
            if traversed > self.persistent_frontier_admission_horizon:
                break
            local_x = cosine * px - sine * py + translation[0]
            local_y = sine * px + cosine * py + translation[1]
            index = self._local_costmap_cell(message, local_x, local_y)
            if index is None:
                return False, {
                    "reason": "local_horizon_outside_costmap",
                    "checked_distance_m": round(traversed, 3),
                    "local_frame": local_frame,
                }
            if index in checked_cells:
                continue
            checked_cells.add(index)
            cost = int(message.data[index])
            max_cost = max(max_cost, cost)
            if cost < 0 or cost >= self.persistent_frontier_admission_blocked_cost:
                return False, {
                    "reason": "local_prefix_blocked",
                    "checked_distance_m": round(traversed, 3),
                    "cell_cost": cost,
                    "max_cost": max_cost,
                    "blocked_cost": self.persistent_frontier_admission_blocked_cost,
                    "local_frame": local_frame,
                }
        if not checked_cells:
            return False, {"reason": "local_prefix_empty"}
        return True, {
            "reason": "accepted",
            "checked_distance_m": round(
                min(traversed, self.persistent_frontier_admission_horizon), 3
            ),
            "checked_cells": len(checked_cells),
            "max_cost": max_cost,
            "local_frame": local_frame,
            "costmap_age_seconds": round(age, 3),
        }

    def _admit_persistent_frontier_prefetch_locked(self):
        """Return whether the pending branch is safe to stream before stop.

        The cached frontier choice alone is not sufficient: it was selected
        before the latest lidar scan and from the global map.  This transaction
        replans from the live feedback pose through Navfn and requires the
        resulting local prefix to be admissible.  A failure never cancels the
        active route; the normal safe-terminal path remains the fallback.
        """
        if self.active_feedback_pose is None or self.prefetched_frontier_goal is None:
            return False, {"reason": "feedback_or_prefetch_unavailable"}
        route_pair = (int(self.active_route_id), int(self.prefetched_frontier_route_id))
        feedback_xy = (float(self.active_feedback_pose[0]), float(self.active_feedback_pose[1]))
        now = time.monotonic()
        cached = self.persistent_prefetch_admission_cache
        if (
            cached is not None
            and cached.get("route_pair") == route_pair
            and now - cached.get("monotonic", 0.0) <= 0.30
            and math.hypot(
                feedback_xy[0] - cached.get("feedback_xy", feedback_xy)[0],
                feedback_xy[1] - cached.get("feedback_xy", feedback_xy)[1],
            ) <= 0.10
        ):
            return bool(cached["accepted"]), dict(cached["details"])

        details = {
            "route_id": route_pair[0],
            "successor_route_id": route_pair[1],
            "feedback": [round(feedback_xy[0], 3), round(feedback_xy[1], 3)],
            "pending_goal": [
                round(float(self.prefetched_frontier_goal[0]), 3),
                round(float(self.prefetched_frontier_goal[1]), 3),
            ],
        }
        # Keep the frontier's BFS tangent as diagnostic evidence only.  The
        # actual transition class is assigned below from the fresh Navfn plan
        # requested from the live feedback pose.
        prefetch_heading_delta, prefetch_heading_basis = (
            self._frontier_prefetch_heading_delta_locked(
                self.last_dispatched_goal,
                self.prefetched_frontier_goal,
            )
        )
        details["prefetch_entry_heading_basis"] = prefetch_heading_basis
        details["prefetch_entry_heading_delta_deg"] = (
            None
            if prefetch_heading_delta is None
            else round(math.degrees(prefetch_heading_delta), 3)
        )
        details["max_entry_heading_delta_deg"] = round(
            math.degrees(self.frontier_early_handoff_max_heading_delta), 3
        )
        details["max_curve_heading_delta_deg"] = round(
            math.degrees(self.persistent_frontier_curve_handoff_max_heading_delta),
            3,
        )
        details["navfn_entry_tangent_sample_distance_m"] = round(
            self.persistent_frontier_entry_tangent_distance, 3
        )
        request = GetPlanRequest()
        request.start.header.frame_id = self.global_frame
        request.start.header.stamp = rospy.Time.now()
        request.start.pose.position.x = feedback_xy[0]
        request.start.pose.position.y = feedback_xy[1]
        request.start.pose.orientation.w = 1.0
        request.goal.header.frame_id = self.global_frame
        request.goal.header.stamp = request.start.header.stamp
        request.goal.pose.position.x = float(self.prefetched_frontier_goal[0])
        request.goal.pose.position.y = float(self.prefetched_frontier_goal[1])
        request.goal.pose.orientation.w = 1.0
        request.tolerance = 0.0
        try:
            rospy.wait_for_service(
                self.navfn_make_plan_service,
                timeout=self.persistent_frontier_admission_service_timeout,
            )
            response = rospy.ServiceProxy(
                self.navfn_make_plan_service, GetPlan
            )(request)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            details.update({
                "reason": "navfn_admission_unavailable",
                "error": type(exc).__name__,
            })
            accepted = False
        else:
            path = response.plan
            if not path.poses:
                details["reason"] = "navfn_admission_empty"
                accepted = False
            else:
                endpoint = path.poses[-1].pose.position
                endpoint_error = math.hypot(
                    float(endpoint.x) - request.goal.pose.position.x,
                    float(endpoint.y) - request.goal.pose.position.y,
                )
                if endpoint_error > self.persistent_frontier_admission_endpoint_epsilon:
                    details.update({
                        "reason": "navfn_admission_endpoint_offset",
                        "endpoint_error_m": round(endpoint_error, 3),
                    })
                    accepted = False
                else:
                    details["navfn_poses"] = len(path.poses)
                    details["endpoint_error_m"] = round(endpoint_error, 3)
                    entry_heading = self._navfn_entry_tangent_locked(
                        path, feedback_xy
                    )
                    if entry_heading is None:
                        details.update({
                            "reason": "navfn_entry_tangent_unavailable",
                            "transition_kind": "terminal_then_native_teb_reorientation",
                        })
                        accepted = False
                    else:
                        entry_delta = abs(self._angle_delta(
                            entry_heading, self.active_feedback_pose[2]
                        ))
                        details["entry_heading_basis"] = "navfn_live_feedback_tangent"
                        details["entry_heading_delta_deg"] = round(
                            math.degrees(entry_delta), 3
                        )
                        if (
                            entry_delta
                            > self.persistent_frontier_curve_handoff_max_heading_delta
                        ):
                            # This is deliberately a terminal fallback, not a
                            # claimed explicit-turn state: production lets the
                            # native TEB controller reorient after the active
                            # endpoint completes.
                            details.update({
                                "reason": "navfn_entry_tangent_too_sharp",
                                "transition_kind": "terminal_then_native_teb_reorientation",
                            })
                            accepted = False
                        else:
                            details["transition_kind"] = (
                                "curve_handoff"
                                if entry_delta
                                > self.frontier_early_handoff_max_heading_delta
                                else "smooth_handoff"
                            )
                            accepted, local = self._local_costmap_prefix_admission_locked(path)
                            details.update(local)
        self.persistent_prefetch_admission_cache = {
            "route_pair": route_pair,
            "feedback_xy": feedback_xy,
            "monotonic": now,
            "accepted": bool(accepted),
            "details": dict(details),
        }
        return bool(accepted), details

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
            if self.active_motion_reference is None:
                self.active_motion_reference = (float(base.x), float(base.y))
                self.active_motion_progress_monotonic = now
            elif math.hypot(
                base.x - self.active_motion_reference[0],
                base.y - self.active_motion_reference[1],
            ) >= self.progress_epsilon:
                self.active_motion_reference = (float(base.x), float(base.y))
                self.active_motion_progress_monotonic = now
            self._update_navfn_path_progress_locked(now)
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
            completion = self.frontier_observation_completion_pending
            if (
                completion is not None
                and int(completion.get("generation", -1)) == int(generation)
            ):
                # The cancel was intentional and its source endpoint was
                # already published to the frontier manager.  Actionlib quite
                # correctly reports PREEMPTED, but the mission-level result is
                # a completed observation region, not a navigation failure.
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
                self.schedule_terminal_dispatch_locked()
                rospy.loginfo(
                    "TEB goal bridge closed frontier observation terminal: "
                    "move_base_status=%s route_id=%d",
                    GoalStatus.to_string(status),
                    int(completion["route_id"]),
                )
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
            if (
                self.persistent_execution
                and status == GoalStatus.SUCCEEDED
                and self.task_done
            ):
                # The persistent local planner returns success only after the
                # independently verified task completion latch. The one
                # action lease has served its purpose; never schedule a new
                # action from this expected terminal callback.
                self.publish_bridge_status(
                    "persistent_task_complete_terminal",
                    status=int(status),
                    status_text="TASK_DONE",
                    action_generation=int(self.action_generation),
                )
                self._clear_action_health_locked()
                return
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
