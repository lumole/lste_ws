#!/usr/bin/env python3
"""Online-map frontier explorer for the normal LSTE search pipeline.

The node never knows an object's coordinates and never publishes
``/lste/final_goal``.  It builds an online SLAM map, finds an unknown-space
boundary reachable through known free cells, and publishes stable route points
in the ``map`` frame.  Goal Manager remains the sole final-goal owner. Keeping
the route in the SLAM frame is important: converting each point to ``odom``
would make an otherwise fixed map point move whenever gmapping updates the
``map -> odom`` transform. Navfn/TEB then follows the complete known-free path
in one action instead of stopping at a sequence of drifting setpoints.
"""

import collections
import copy
import json
import math
import threading
import time
import traceback

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Pose2D, PoseStamped
from map_msgs.msg import OccupancyGridUpdate
from move_base_msgs.msg import RecoveryStatus
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetPlan, GetPlanRequest
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf.transformations import quaternion_matrix


class GlobalFrontierExplorer:
    def __init__(self):
        rospy.init_node("lste_global_frontier")
        gp = rospy.get_param
        self.map_topic = gp("~map_topic", "/map")
        # Navfn plans on the inflated global costmap, not directly on the raw
        # SLAM occupancy grid.  Use it as a candidate validator whenever it is
        # available so a frontier that is connected in /map but lethal after
        # costmap inflation is never handed to move_base.
        self.costmap_topic = gp(
            "~costmap_topic", "/move_base/global_costmap/costmap"
        )
        self.costmap_updates_topic = gp(
            "~costmap_updates_topic", "/move_base/global_costmap/costmap_updates"
        )
        self.costmap_max_age = max(0.5, float(gp("~costmap_max_age", 3.0)))
        self.pose_topic = gp("~pose_topic", "/rbt_pose")
        self.goal_topic = gp("~goal_topic", "/lste/global_frontier_goal")
        # A PoseStamped cannot carry route kind and rospy owns Header.seq.
        # Publish the complete route transaction on one companion topic so
        # consumers never have to correlate independent pose/status messages.
        self.command_topic = gp(
            "~command_topic", "/lste/global_frontier/route_command"
        )
        self.task_done_topic = gp("~task_done_topic", "/lste/task_done")
        self.status_topic = gp(
            "~status_topic", "/lste/global_frontier/status"
        )
        self.replan_request_topic = gp(
            "~replan_request_topic", "/lste/global_frontier/replan_request"
        )
        self.turn_status_topic = gp(
            "~turn_status_topic", "/lste/teb_turn_supervisor/status"
        )
        # A successful execution terminal is a route-lifecycle event. It must
        # wake the frontier selector immediately instead of waiting for the
        # next 1 Hz timer.
        self.terminal_topic = gp("~terminal_topic", "/lste/teb_goal_terminal")
        # Local recovery and action failure are different lifecycle states.
        # A recovery is deliberately local: TEB/move_base may still clear a
        # transient lidar obstacle or rotate into an open doorway. Only a
        # terminal *failed* action proves that the globally validated route
        # must be replaced.
        self.recovery_topic = gp("~recovery_topic", "/move_base/recovery_status")
        self.bridge_status_topic = gp(
            "~bridge_status_topic", "/lste/teb_goal_bridge/status"
        )
        self.scan_topic = gp("~scan_topic", "/pro3/rlscan")
        # ``clearance`` is the route clearance, not merely a preference.  It
        # must cover the robot footprint and TEB's minimum obstacle distance;
        # otherwise this node can find a connected BFS route that Navfn/TEB
        # correctly rejects.  The production launch passes 0.52 m
        # (0.30 m footprint radius + 0.22 m TEB clearance).
        self.clearance = max(0.05, float(gp("~clearance", 0.52)))
        # ``fallback_clearance`` identifies known-free observation topology at
        # an unknown boundary. It can connect a recovery route through a
        # narrow passage, but never authorizes the endpoint itself: every
        # published endpoint remains in ``strict_free`` below.
        self.frontier_clearance = max(
            0.15,
            min(
                self.clearance,
                float(gp("~fallback_clearance", max(0.20, self.clearance - 0.10))),
            ),
        )
        # The goal is an approach point in the safe route mask.  It may sit
        # short of the actual unknown-boundary cell; the lidar then reveals the
        # next part of the map without driving the footprint into a doorway.
        self.frontier_approach_distance = max(
            0.20, float(gp("~frontier_approach_distance", 1.0))
        )
        self.min_path_distance = max(0.2, float(gp("~min_path_distance", 1.2)))
        observation_recovery = gp("~navfn_observation_recovery_enabled", True)
        self.navfn_observation_recovery_enabled = str(
            observation_recovery
        ).strip().lower() in ("1", "true", "yes", "on")
        # Selection mode is attached to the next route transaction and status
        # record. It is diagnostic only; it never changes TEB's constraints.
        self.last_frontier_selection_mode = "strict_clearance"
        # A bounded Navfn validation pass may reject a batch while other
        # candidates remain. This is distinct from genuine map exhaustion.
        self.frontier_validation_budget_exhausted = False
        # A live service can still be initializing its costmap. This is also
        # distinct from exhaustion and must remain retryable on the next map
        # cycle.
        self.frontier_validation_pending = False
        # The frontier is a mission endpoint.  ``lookahead_distance`` and the
        # rolling segment distance remain compatibility values for the legacy
        # experiment, but the production endpoint contract lets Navfn/TEB own
        # the complete known-free path in one action.
        self.lookahead_distance = max(0.2, float(gp("~lookahead_distance", 2.0)))
        self.waypoint_release_radius = max(
            0.2, float(gp("~waypoint_release_radius", 0.40))
        )
        # Leave room for the vehicle's current command-goal tolerance when a
        # route segment is handed off in-place.  The resulting horizon is
        # derived from the existing route/action contract: with the production
        # 4 m TEB reinitialization distance and a 1.2 m release radius it is
        # about 2.8 m, rather than a separately tuned controller constant.
        self.route_segment_distance = max(
            0.8,
            self.lookahead_distance - self.waypoint_release_radius,
        )
        # Exploration owns a frontier mission, while Navfn/TEB own the live
        # path to that mission.  Publishing a short rolling waypoint here
        # creates an action boundary in the middle of a perfectly valid path
        # and forces the executor to brake, replace its band, and start again.
        # The production contract therefore publishes the selected frontier
        # endpoint.  The legacy rolling-horizon behavior remains available
        # only for explicitly isolated experiments.
        endpoint_only = gp("~mission_endpoint_only", True)
        self.mission_endpoint_only = str(endpoint_only).strip().lower() in (
            "1", "true", "yes", "on",
        )
        # The persistent executor keeps one MoveBase action alive while its
        # pluginized Navfn/TEB pair consumes new plans. In that architecture
        # the frontier node, rather than an action terminal callback, advances
        # a validated prefetched successor. It is opt-in until its complete
        # search behavior has been compared with endpoint-action execution.
        self.persistent_execution = str(
            gp("~persistent_execution", False)
        ).strip().lower() in ("1", "true", "yes", "on")
        # Compute the next exploration branch before the current endpoint is
        # reached.  It remains an internal cache until the validated early
        # handoff window below; this prevents map refreshes from replacing a
        # healthy route before a safe continuation is known.
        self.prefetch_distance = max(
            self.waypoint_release_radius * 2.0,
            float(gp("~prefetch_distance", 1.8)),
        )
        # Promote a validated pending branch before the current endpoint is
        # reached.  The old endpoint remains a safe approach point; waiting
        # for move_base's SUCCEEDED callback first inserts a zero-velocity gap
        # between otherwise connected exploration routes.  Keep this window
        # below the prefetch radius so a branch is selected and validated for
        # at least one map cycle before it can become the active goal.
        self.early_handoff_distance = max(
            self.waypoint_release_radius,
            min(
                self.prefetch_distance,
                float(gp("~early_handoff_distance", 1.35)),
            ),
        )
        self.active_timeout = max(2.0, float(gp("~active_timeout", 18.0)))
        self.stall_timeout = max(2.0, float(gp("~stall_timeout", 8.0)))
        # Once a committed pre-route turn has completed, a route that never
        # physically launches is usually an unusable local plan. This is a
        # *launch* watchdog, not a shorter version of the general progress
        # watchdog: after measurable odom translation, the normal watchdog
        # remains responsible for deciding whether a moving route is healthy.
        # Set <= 0 to disable this specialized lifecycle rule for comparisons.
        raw_post_turn_stall_timeout = float(
            gp("~post_turn_stall_timeout", 6.0)
        )
        self.post_turn_stall_timeout = (
            None if raw_post_turn_stall_timeout <= 0.0 else max(
                1.0, min(self.stall_timeout, raw_post_turn_stall_timeout)
            )
        )
        # SLAM can temporarily disconnect an active frontier while integrating
        # a scan. Keep the last safe endpoint briefly instead of switching
        # branches on every map update.
        self.unreachable_grace = max(
            0.0, float(gp("~unreachable_grace", 20.0))
        )
        self.progress_epsilon = max(0.02, float(gp("~progress_epsilon", 0.12)))
        # The frontier node owns map-level route identity while move_base owns
        # the physical terminal.  Do not use a second, unrelated 0.8 m magic
        # number to retire that identity.  Online SLAM can shift map->odom by
        # several cells during arrival, so the exploration layer waits for the
        # matching move_base terminal throughout a small transform envelope
        # around TEB's real XY tolerance.  This is deliberately a *hold*, not
        # a new completion condition: only the bridge terminal advances the
        # frontier route.
        teb_xy_goal_tolerance = max(
            0.05,
            float(rospy.get_param(
                "/move_base/TebLocalPlannerROS/xy_goal_tolerance", 0.35
            )),
        )
        self.endpoint_terminal_wait_radius = max(
            0.8, teb_xy_goal_tolerance + 0.50
        )
        # Route/goal distance is measured in the online-SLAM map and can grow
        # during a legitimate detour. Use small odom-space coverage cells as
        # a second, map-correction-independent progress proof: entering a new
        # cell renews the watchdog, while a local loop eventually revisits its
        # cells and still times out.
        self.odom_novel_cell_size = max(
            0.20, float(gp("~odom_novel_cell_size", 0.40))
        )
        # A frontier can be several metres away while the collision-aware
        # SA-PPO controller is deliberately slow in a narrow corridor. Keep a
        # failed route out of candidate selection long enough to explore a
        # different branch instead of alternating between the same two
        # unreachable boundaries every 30 seconds.
        self.rejected_timeout = max(5.0, float(gp("~rejected_timeout", 180.0)))
        # A frontier reached by the robot has already contributed its camera
        # observation and local lidar scan.  Keep that coverage memory for the
        # whole task, rather than forgetting it after the short retry timeout
        # used for a temporarily blocked route.  As SLAM reveals more space,
        # the true unknown boundary moves beyond this radius and remains a
        # valid candidate.
        self.completed_radius = max(0.2, float(gp("~completed_radius", 1.25)))
        self.completed_limit = max(16, int(gp("~completed_limit", 256)))
        self.candidate_limit = max(32, int(gp("~candidate_limit", 512)))
        self.active_reassociation_radius = max(
            0.4, float(gp("~active_reassociation_radius", 1.0))
        )
        self.info_radius = max(1, int(gp("~info_radius_cells", 8)))
        # A lidar max-range frontier in an empty open area is often just the
        # edge of a scan, not an entrance worth searching. Doorways, corridor
        # branches and room boundaries have occupied cells nearby. Rewarding
        # that local structure makes coverage spend time in discoverable indoor
        # space before drifting into unconstrained open floor.
        self.structure_radius = max(1, int(gp("~structure_radius_cells", 10)))
        self.structure_weight = max(0.0, float(gp("~structure_weight", 0.16)))
        self.min_structure_cells = max(0, int(gp("~min_structure_cells", 3)))
        # Dead-end detection: when the robot is within this distance of the
        # active frontier and no unknown remains within ``dead_end_unknown_cells``
        # of the frontier cell, the route ends at a resolved wall with no
        # opening.  Mark it inspected and choose a real passage instead of
        # driving into the known wall.
        self.dead_end_check_distance = max(
            0.0, float(gp("~dead_end_check_distance", 4.0))
        )
        self.dead_end_unknown_cells = max(
            2, int(gp("~dead_end_unknown_cells", 15))
        )
        # Exploration still rewards information and doorway-like structure,
        # but a branch whose *route's first tangent* is behind the robot
        # should not win by a small score margin and force TEB to brake and
        # make an abrupt U-turn.  Endpoint bearing is not sufficient here: a
        # reachable endpoint in front of the robot may require a doorway
        # detour that starts behind it. This is a soft continuity cost rather
        # than a hard direction filter, so a necessary turn remains valid.
        self.heading_weight = max(0.0, float(gp("~heading_weight", 2.5)))
        self.heading_hard_limit = math.radians(max(
            0.0, float(gp("~heading_hard_limit_deg", 115.0))
        ))
        # Successor selection is a route-transition decision, not merely a
        # weighted frontier score. Reuse the bridge's three execution
        # envelopes: first seek a smooth continuation, then a curve that TEB
        # may stream, and only then accept a terminal reorientation branch.
        # The final hard limit still permits necessary indoor turns.
        self.successor_smooth_heading_limit = math.radians(max(
            0.0, float(gp("~successor_smooth_heading_limit_deg", 45.0))
        ))
        self.successor_curve_heading_limit = math.radians(min(
            math.degrees(self.heading_hard_limit),
            max(
                math.degrees(self.successor_smooth_heading_limit),
                float(gp("~successor_curve_heading_limit_deg", 65.0)),
            ),
        ))
        # TEB normally owns route turns itself. The retired connector path is
        # retained solely for controlled comparisons: it inserts a separate
        # MoveBase action and an external velocity owner, which creates an
        # action boundary and can make the frontier watchdog see no
        # translational progress during a valid rotation.
        self.turn_execution_mode = str(
            gp("~turn_execution_mode", "native_teb")
        ).strip().lower()
        if self.turn_execution_mode not in ("native_teb", "legacy_connector"):
            rospy.logwarn(
                "Global frontier invalid turn_execution_mode=%r; using native_teb",
                self.turn_execution_mode,
            )
            self.turn_execution_mode = "native_teb"
        # Used only by the legacy connector comparison mode.
        self.explicit_turn_connector_threshold = math.radians(max(
            90.0,
            min(
                180.0,
                float(gp("~explicit_turn_connector_threshold_deg", 135.0)),
            ),
        ))
        # This must exceed move_base's 0.35 m XY terminal tolerance. The
        # connector is sampled from the already validated BFS/Navfn route, so
        # it exists solely to keep the action active for the supervised yaw
        # phase and is never an arbitrary free-space waypoint.
        self.turn_connector_distance = max(
            0.45, float(gp("~turn_connector_distance", 0.50))
        )
        # A confirmed target can be visible beyond a mapped wall.  Its visual
        # ray is not an executable goal in that case, but it is useful evidence
        # for choosing the next *reachable* information boundary.  This is a
        # soft score term applied only to the one replan requested by Goal
        # Manager after Navfn rejects that ray; ordinary coverage is unchanged.
        self.semantic_hint_weight = max(
            0.0, float(gp("~semantic_hint_weight", 0.35))
        )
        self.semantic_hint_max_distance = max(
            0.5, float(gp("~semantic_hint_max_distance", 12.0))
        )
        # A turn connector is complete when its route tangent is inside the
        # same orientation tolerance used by TEB's action goal. This is a
        # route-state contract, not a velocity/controller tuning knob.
        self.turn_yaw_tolerance = 0.35
        # Full map/costmap BFS is useful near a frontier, but it is not needed
        # while the active endpoint is metres away and its route is healthy.
        # Throttling that work leaves more CPU headroom for Gazebo, SLAM and
        # move_base without changing the committed goal.
        self.planning_period = max(0.2, float(gp("~planning_period", 1.5)))
        self.map_msg = None
        self.costmap_msg = None
        self.costmap_message_count = 0
        self.costmap_last_receive_wall = 0.0
        # A full costmap BFS is useful for candidate validation, but it is not
        # a control-cycle operation.  Reuse the last connected mask briefly;
        # the local costmap/TEB remains responsible for immediate obstacles.
        self.costmap_validation_period = max(
            0.2, float(gp("~costmap_validation_period", 2.0))
        )
        self.cached_costmap_validation = None
        self.cached_costmap_validation_wall = 0.0
        navfn_validation = gp("~navfn_plan_validation", True)
        self.navfn_plan_validation = str(navfn_validation).strip().lower() in (
            "1", "true", "yes", "on",
        )
        self.navfn_make_plan_service = gp(
            "~navfn_make_plan_service", "/move_base/NavfnROS/make_plan"
        )
        self.navfn_make_plan_timeout = max(
            0.01, float(gp("~navfn_make_plan_timeout", 0.05))
        )
        # Kept as a visible launch parameter for compatibility. The execution
        # contract deliberately validates exact endpoints, so its value must
        # never relax the GetPlan request.
        self.navfn_make_plan_tolerance = 0.0
        # A ROS service can appear before move_base has populated its global
        # costmap. During that short bootstrap window Navfn legally replies
        # with an empty plan for every goal. Track the first live response so
        # this lifecycle state is not mistaken for a real unreachable branch.
        self.navfn_empty_warmup_seconds = max(
            0.0, float(gp("~navfn_empty_warmup_seconds", 5.0))
        )
        # Do not start selecting frontiers merely because the Navfn service
        # has registered.  During online-SLAM startup that service can exist
        # before GMapping owns map->odom and before the global costmap has a
        # connected free cell around the base.  A short, costmap-derived
        # nearby plan proves the complete map/TF/costmap/Navfn chain before
        # any real frontier is allowed to become a mission transaction.
        self.navfn_startup_probe_distance = max(
            0.10, float(gp("~navfn_startup_probe_distance", 0.40))
        )
        self.navigation_stack_ready = not self.navfn_plan_validation
        self.navigation_readiness_state = (
            "navfn_validation_disabled"
            if self.navigation_stack_ready
            else "waiting_for_map_tf_costmap_navfn"
        )
        self.navfn_first_response_wall = None
        # A live make_plan service can still have no usable global route while
        # online SLAM is constructing its initial costmap. An empty response
        # cannot prove a candidate unreachable until Navfn has first produced
        # at least one non-empty plan in this process lifetime.
        self.navfn_first_nonempty_response_wall = None
        # The selector needs to distinguish a live-but-cold Navfn response
        # from a service/TF failure. Only the former may use the conservative
        # costmap-connected bootstrap route below.
        self.navfn_last_validation_state = "unavailable"
        self.navfn_service = rospy.ServiceProxy(
            self.navfn_make_plan_service, GetPlan
        )
        self.pose_odom = None
        # Recovery arbitration uses the forward lidar arc, not the global
        # minimum: a corridor wall close to either side is normal, while an
        # obstacle directly in front means a new global endpoint cannot make
        # the base move until local recovery has created some clearance.
        self.scan_forward_minimum = float("nan")
        self.task_done = False
        self.active_frontier = None
        self.active_since = 0.0
        # Keep the physical mission endpoint distinct from a grid cell that is
        # re-associated after each online-SLAM update.  The latter is useful
        # for route validation, but it is not a stable identity for measuring
        # whether the robot is approaching the command sent to move_base.
        self.active_best_distance = None
        self.active_best_goal_distance = None
        self.active_best_path_distance = None
        self.active_progress_time = 0.0
        self.active_last_progress_signal = "none"
        self.active_last_robot_xy = None
        # Map-space distances are authoritative for the endpoint, but online
        # SLAM can legitimately require an initial detour that increases them.
        # Keep an odom-relative start only as secondary evidence of one-way
        # physical transit through that detour; it is never used as a goal.
        self.active_start_odom_xy = None
        self.active_best_detour_odom_distance = 0.0
        self.active_visited_odom_cells = set()
        self.active_unreachable_since = None
        # Route points stay in the SLAM frame.  The old ``*_odom`` contract
        # made progress checks disagree with move_base after a SLAM correction.
        self.active_last_waypoint_map = None
        # The last command may be a semantic turn connector.  Keep its
        # orientation with the position so a map refresh cannot replace a
        # turn-in-place goal with a pose whose yaw silently resets to zero.
        self.active_last_waypoint_yaw = None
        self.active_route_kind = "frontier_endpoint"
        # Stable identity for an exploration route transaction. It changes
        # only when a new BFS branch is selected; an early handoff preserves
        # it only after the discrete path-prefix test proves continuity.
        self.active_route_id = 0
        # Route id, rather than a bare Boolean, makes a late recovery message
        # harmless after the selector has already promoted a new branch.
        self.recovery_pending_route_id = 0
        self.recovery_pending_behavior = ""
        self.recovery_pending_reason = ""
        self.recovery_pending_wall = 0.0
        # In production each frontier endpoint is an atomic move_base action.
        # Physical proximity is useful for prefetching, but it must not retire
        # the active route before the matching action reports success.
        self.active_terminal_received = False
        # A turn connector is a committed execution phase. Keep its pose and
        # tangent frozen until the supervisor reports completion.
        self.turn_connector_released = True
        self.turn_supervisor_state = "UNKNOWN"
        # The turn supervisor does not own frontier route ids, so retain both
        # route identity and endpoint. A late completion from a superseded
        # action must never affect a newer route.
        self.active_turn_completed_route_id = 0
        self.active_turn_completed_goal = None
        self.active_turn_completed_wall = 0.0
        self.active_turn_completed_odom_xy = None
        self.active_turn_completed_translation = 0.0
        self.active_turn_completed_launched = False
        self.last_status_command_map = None
        self.last_status_command_yaw = None
        self.last_status_mission_map = None
        self.prefetched_frontier = None
        self.prefetched_goal_map = None
        # Reaching the last safe unknown-space boundary is not a local-planner
        # failure.  Remember this state so the explorer emits one lifecycle
        # event instead of letting the persistent move_base lease recover
        # forever against an endpoint for which no successor exists.
        self.frontier_exhausted = False
        # Every published endpoint is a new destination, but not every update
        # is a disruptive path change. Keep the BFS verdict with the route
        # transaction so metrics can distinguish a prefix extension from a
        # necessary branch selected after inspecting the old endpoint.
        self.active_transition_kind = "initial"
        self.active_predecessor_route_id = 0
        self.active_transition_distance = None
        # A replan request is issued when target execution relinquishes
        # exploration ownership. The following endpoint must be selected from
        # the robot's then-current map pose, not recovered from this cache.
        self.pending_replan_request_id = 0
        self.pending_replan_reason = ""
        self.pending_semantic_hint_map = None
        self.last_planning_wall = 0.0
        self.planning_lock = threading.RLock()
        self.immediate_plan_timer = None
        self.rejected_frontiers = collections.deque(maxlen=24)
        self.completed_frontiers = collections.deque(maxlen=self.completed_limit)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.publisher = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1, latch=True)
        self.command_publisher = rospy.Publisher(
            self.command_topic, String, queue_size=10, latch=True
        )
        self.status_publisher = rospy.Publisher(
            self.status_topic, String, queue_size=10, latch=True
        )
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.on_map, queue_size=1)
        rospy.Subscriber(self.costmap_topic, OccupancyGrid, self.on_costmap, queue_size=1)
        rospy.Subscriber(
            self.costmap_updates_topic,
            OccupancyGridUpdate,
            self.on_costmap_update,
            queue_size=10,
        )
        rospy.Subscriber(self.pose_topic, Pose2D, self.on_pose, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.on_scan, queue_size=1)
        rospy.Subscriber(
            self.replan_request_topic, String, self.on_replan_request, queue_size=10
        )
        rospy.Subscriber(self.task_done_topic, Bool, self.on_task_done, queue_size=1)
        rospy.Subscriber(
            self.recovery_topic,
            RecoveryStatus,
            self.on_move_base_recovery,
            queue_size=10,
        )
        rospy.Subscriber(
            self.bridge_status_topic,
            String,
            self.on_bridge_status,
            queue_size=20,
        )
        rospy.Subscriber(
            self.turn_status_topic,
            String,
            self.on_turn_status,
            queue_size=10,
        )
        rospy.Subscriber(
            self.terminal_topic,
            PoseStamped,
            self.on_execution_terminal,
            queue_size=10,
        )
        rospy.Timer(rospy.Duration(1.0), self.on_timer)
        rospy.loginfo(
            "Global frontier explorer started: map=%s costmap=%s scan=%s recovery=%s bridge=%s goal=%s "
            "route_tangent_weight=%.2f heading_hard_limit=%.1fdeg "
            "successor_envelopes=[%.1f,%.1f]deg "
            "semantic_hint_weight=%.2f route_segment=%.2fm "
            "mission_endpoint_only=%s persistent_execution=%s turn_execution=%s planning_period=%.2fs "
            "navfn_startup_probe=%.2fm",
            self.map_topic,
            self.costmap_topic,
            self.scan_topic,
            self.recovery_topic,
            self.bridge_status_topic,
            self.goal_topic,
            self.heading_weight,
            math.degrees(self.heading_hard_limit),
            math.degrees(self.successor_smooth_heading_limit),
            math.degrees(self.successor_curve_heading_limit),
            self.semantic_hint_weight,
            self.route_segment_distance,
            self.mission_endpoint_only,
            self.persistent_execution,
            self.turn_execution_mode,
            self.planning_period,
            self.navfn_startup_probe_distance,
        )

    def publish_status(self, event, **fields):
        """Publish the frontier planner's route lifecycle to its consumers."""
        pending_goal = (
            None
            if self.prefetched_frontier is None
            else [
                round(float(self.prefetched_frontier[2]), 3),
                round(float(self.prefetched_frontier[3]), 3),
            ]
        )
        payload = {
            "event": str(event),
            "active": self.active_frontier is not None,
            "pending": self.prefetched_frontier is not None,
            # This is lifecycle evidence only. The pending pose is not a
            # command: Goal Manager receives it only after a matching terminal
            # promotes the validated branch.
            "pending_goal": pending_goal,
            "pending_route_id": (
                int(self.active_route_id + 1)
                if pending_goal is not None and self.active_route_id > 0
                else 0
            ),
            "turn_supervisor_state": self.turn_supervisor_state,
            "route_id": int(self.active_route_id),
        }
        payload.update(fields)
        try:
            self.status_publisher.publish(
                String(data=json.dumps(payload, sort_keys=True))
            )
        except (TypeError, ValueError):
            rospy.logwarn_throttle(
                5.0, "Global frontier status serialization failed"
            )

    def publish_route_command(self, frame_id, x, y, yaw, route_kind):
        """Publish one self-contained frontier command transaction."""
        payload = {
            "event": "route_command",
            "frame_id": str(frame_id or "map").strip().lstrip("/") or "map",
            "goal": [round(float(x), 4), round(float(y), 4)],
            "yaw": None if yaw is None else round(float(yaw), 4),
            "route_id": int(self.active_route_id),
            "route_kind": str(route_kind or "frontier_endpoint"),
            "transition_kind": str(self.active_transition_kind),
            "predecessor_route_id": int(self.active_predecessor_route_id),
            "transition_distance_to_previous_endpoint": (
                None
                if self.active_transition_distance is None
                else round(float(self.active_transition_distance), 4)
            ),
            "stamp": rospy.Time.now().to_sec(),
        }
        self.command_publisher.publish(String(data=json.dumps(payload, sort_keys=True)))

    def on_map(self, message):
        self.map_msg = message

    def on_costmap(self, message):
        self.costmap_msg = message
        self.costmap_message_count += 1
        self.costmap_last_receive_wall = time.monotonic()
        rospy.loginfo_once(
            "Global frontier received costmap frame=%s size=%dx%d resolution=%.3f "
            "stamp=%.3f cells=%d",
            message.header.frame_id,
            message.info.width,
            message.info.height,
            message.info.resolution,
            message.header.stamp.to_sec(),
            len(message.data),
        )

    def on_costmap_update(self, update):
        """Apply costmap_2d's incremental update to the cached full grid."""
        base = self.costmap_msg
        if base is None:
            return
        width = int(base.info.width)
        height = int(base.info.height)
        x, y = int(update.x), int(update.y)
        update_width, update_height = int(update.width), int(update.height)
        # When gmapping expands the map, costmap_2d can emit a full-grid
        # OccupancyGridUpdate whose dimensions no longer fit the old cached
        # OccupancyGrid. This is a replacement, not a malformed increment.
        # Dropping it leaves the frontier validator permanently stale until a
        # future full /costmap publication happens to arrive.
        if (
            x == 0
            and y == 0
            and update_width > 0
            and update_height > 0
            and len(update.data) == update_width * update_height
            and (update_width != width or update_height != height)
        ):
            updated = copy.deepcopy(base)
            updated.header = copy.deepcopy(update.header)
            updated.info.width = update_width
            updated.info.height = update_height
            updated.data = list(update.data)
            self.costmap_msg = updated
            self.costmap_message_count += 1
            self.costmap_last_receive_wall = time.monotonic()
            self.cached_costmap_validation = None
            self.cached_costmap_validation_wall = 0.0
            rospy.loginfo(
                "Global frontier accepted resized full costmap update "
                "old=%dx%d new=%dx%d",
                width, height, update_width, update_height,
            )
            return
        if (
            width <= 0
            or height <= 0
            or x < 0
            or y < 0
            or x + update_width > width
            or y + update_height > height
            or len(update.data) != update_width * update_height
        ):
            rospy.logwarn_throttle(
                5.0,
                "Global frontier ignored malformed costmap update "
                "origin=(%d,%d) size=%dx%d base=%dx%d cells=%d",
                x,
                y,
                update_width,
                update_height,
                width,
                height,
                len(update.data),
            )
            return
        # Do not mutate a message currently being read by the timer thread.
        updated = copy.deepcopy(base)
        data = list(updated.data)
        for row in range(update_height):
            start = (y + row) * width + x
            end = start + update_width
            source_start = row * update_width
            data[start:end] = update.data[source_start:source_start + update_width]
        updated.data = data
        self.costmap_msg = updated
        self.costmap_last_receive_wall = time.monotonic()
        self.cached_costmap_validation = None
        self.cached_costmap_validation_wall = 0.0

    def on_pose(self, message):
        self.pose_odom = message

    def _odom_coverage_cell(self, pose):
        """Return a stable, coarse odom cell for one active route transaction."""
        if pose is None:
            return None
        scale = self.odom_novel_cell_size
        return (
            int(math.floor(float(pose.x) / scale)),
            int(math.floor(float(pose.y) / scale)),
        )

    def _reset_active_odom_coverage(self):
        """Start route-local coverage accounting at the current base pose."""
        self.active_visited_odom_cells.clear()
        cell = self._odom_coverage_cell(self.pose_odom)
        if cell is not None:
            self.active_visited_odom_cells.add(cell)

    def _entered_novel_active_odom_cell(self):
        """Return true exactly once for each newly entered active-route cell."""
        cell = self._odom_coverage_cell(self.pose_odom)
        if cell is None or cell in self.active_visited_odom_cells:
            return False
        self.active_visited_odom_cells.add(cell)
        return True

    def _clear_active_post_turn_watchdog(self):
        """Discard turn-completion evidence when frontier ownership changes."""
        self.active_turn_completed_route_id = 0
        self.active_turn_completed_goal = None
        self.active_turn_completed_wall = 0.0
        self.active_turn_completed_odom_xy = None
        self.active_turn_completed_translation = 0.0
        self.active_turn_completed_launched = False

    def on_task_done(self, message):
        done = bool(message.data)
        if done == self.task_done:
            return
        self.task_done = done
        if done:
            # The TEB bridge and mux stop the vehicle when the target is
            # confirmed. Clear the exploration commitment as well so this node
            # cannot publish another endpoint after task completion.
            self.active_frontier = None
            self.active_last_robot_xy = None
            self.active_last_waypoint_map = None
            self.active_last_waypoint_yaw = None
            self.active_route_kind = "frontier_endpoint"
            self.active_terminal_received = False
            self.recovery_pending_route_id = 0
            self.recovery_pending_behavior = ""
            self.recovery_pending_reason = ""
            self.turn_connector_released = True
            self.last_status_command_map = None
            self.last_status_command_yaw = None
            self.last_status_mission_map = None
            self.active_best_distance = None
            self.active_best_goal_distance = None
            self.active_best_path_distance = None
            self.active_last_progress_signal = "none"
            self.active_visited_odom_cells.clear()
            self.active_unreachable_since = None
            self.prefetched_frontier = None
            self.prefetched_goal_map = None
            rospy.loginfo("Global frontier paused: task_done=true")
        else:
            rospy.loginfo("Global frontier resumed: task_done=false")

    def on_move_base_recovery(self, message):
        """Record a local recovery without preempting the global action.

        ``RecoveryStatus`` is emitted when a recovery *starts*, including the
        first behaviour in a multi-step sequence. Treating that first event as
        a global failure used to cancel the action before TEB could complete
        its own recovery, then publish a distant frontier every few seconds.
        The action bridge reports the authoritative terminal outcome through
        :meth:`on_bridge_status`; this callback only keeps the no-progress
        watchdog from racing the local recovery.
        """
        if self.task_done or self.active_frontier is None or self.active_route_id <= 0:
            return
        behavior = str(message.recovery_behavior_name or "unknown")
        route_id = int(self.active_route_id)
        forward_clearance = self.scan_forward_minimum
        self.active_progress_time = time.monotonic()
        self.active_last_progress_signal = "local_recovery:%s" % behavior
        self.publish_status(
            "planner_recovery_observed",
            route_id=route_id,
            recovery_behavior=behavior,
            recovery_index=int(message.current_recovery_number),
            recovery_total=int(message.total_number_of_recoveries),
            forward_clearance=(
                None if not math.isfinite(forward_clearance)
                else round(float(forward_clearance), 3)
            ),
            required_clearance=round(float(self.clearance), 3),
        )
        rospy.logwarn(
            "Global frontier retains route_id=%d during local recovery "
            "behavior=%s (%d/%d); waiting for action outcome",
            route_id,
            behavior,
            int(message.current_recovery_number) + 1,
            int(message.total_number_of_recoveries),
        )

    def on_bridge_status(self, message):
        """Promote only a matching failed frontier action to a route change."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("event") != "terminal":
            return
        if self.task_done or self.active_frontier is None:
            return
        route_id = max(0, int(payload.get("active_route_id", 0) or 0))
        if route_id <= 0 or route_id != self.active_route_id:
            return
        if str(payload.get("active_intent_source", "")) != "global_slam_frontier":
            return
        if str(payload.get("active_route_kind", "")) != "frontier_endpoint":
            return
        status = int(payload.get("status", -1) or -1)
        # PREEMPTED is an intentional lifecycle handoff/cancel and therefore
        # cannot be used as evidence that the frontier is unreachable.
        if status not in (4, 5, 8, 9):  # ABORTED, REJECTED, RECALLED, LOST
            return
        status_name = str(payload.get("status_text", "FAILED")).upper()
        self.recovery_pending_route_id = route_id
        self.recovery_pending_behavior = "move_base_terminal"
        self.recovery_pending_reason = "move_base_%s" % status_name.lower()
        self.recovery_pending_wall = time.monotonic()
        # An action terminal is conclusive. Wake the selector immediately;
        # it will still perform map, costmap, and Navfn validation before
        # publishing a replacement endpoint.
        self.last_planning_wall = 0.0
        self.publish_status(
            "execution_terminal_failure",
            route_id=route_id,
            status=status,
            status_text=status_name,
            reason=self.recovery_pending_reason,
        )
        rospy.logwarn(
            "Global frontier will replace route_id=%d after terminal action failure: %s",
            route_id,
            status_name,
        )

    def on_scan(self, message):
        """Keep the current front-sector physical clearance for recovery gating."""
        ranges = []
        for index, value in enumerate(message.ranges):
            angle = message.angle_min + index * message.angle_increment
            if (
                abs(angle) <= math.radians(30.0)
                and math.isfinite(value)
                and value > 0.01
            ):
                ranges.append(float(value))
        self.scan_forward_minimum = min(ranges) if ranges else float("nan")

    def on_replan_request(self, message):
        """Discard cached route ownership and rebuild from the current pose."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("event") != "replan_request":
            return
        request_id = max(0, int(payload.get("request_id", 0) or 0))
        if request_id <= 0 or self.task_done:
            return
        old_goal = self.active_frontier
        hint = payload.get("semantic_hint_map")
        semantic_hint = None
        if isinstance(hint, (list, tuple)) and len(hint) >= 2:
            try:
                x, y = float(hint[0]), float(hint[1])
                if math.isfinite(x) and math.isfinite(y):
                    semantic_hint = (x, y)
            except (TypeError, ValueError):
                pass
        self.pending_replan_request_id = request_id
        self.pending_replan_reason = str(payload.get("reason", "unknown"))
        self.pending_semantic_hint_map = semantic_hint
        self.active_frontier = None
        self.active_since = 0.0
        self.active_best_distance = None
        self.active_best_goal_distance = None
        self.active_best_path_distance = None
        self.active_progress_time = 0.0
        self.active_last_progress_signal = "none"
        self.active_last_robot_xy = None
        self.active_visited_odom_cells.clear()
        self.active_unreachable_since = None
        self.active_last_waypoint_map = None
        self.active_last_waypoint_yaw = None
        self.active_route_kind = "frontier_endpoint"
        self.active_terminal_received = False
        self.recovery_pending_route_id = 0
        self.recovery_pending_behavior = ""
        self.recovery_pending_reason = ""
        self.turn_connector_released = True
        self._clear_active_post_turn_watchdog()
        self.last_status_command_map = None
        self.last_status_command_yaw = None
        self.last_status_mission_map = None
        self.prefetched_frontier = None
        self.prefetched_goal_map = None
        self.last_planning_wall = 0.0
        self.publish_status(
            "replan_acknowledged",
            replan_request_id=request_id,
            reason=self.pending_replan_reason,
            previous_goal=(
                None
                if old_goal is None
                else [round(float(old_goal[2]), 3), round(float(old_goal[3]), 3)]
            ),
            semantic_hint_map=(
                None if semantic_hint is None
                else [round(semantic_hint[0], 3), round(semantic_hint[1], 3)]
            ),
        )
        rospy.loginfo(
            "Global frontier accepted replan request id=%d reason=%s old=%s hint=%s",
            request_id, self.pending_replan_reason, old_goal, semantic_hint,
        )

    def on_turn_status(self, message):
        """Receive the execution adapter's atomic-turn state."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self.planning_lock:
            self.turn_supervisor_state = (
                str(payload.get("state", "UNKNOWN")).strip().upper() or "UNKNOWN"
            )
            turn_event = str(
                payload.get("turn_event", payload.get("event", ""))
            ).strip().lower()
            if turn_event == "turn_started":
                self.turn_connector_released = False
            elif turn_event == "turn_completed":
                self.turn_connector_released = True
                previous_key = payload.get("previous_key")
                if (
                    self.post_turn_stall_timeout is None
                    or self.active_frontier is None
                    or self.active_route_id <= 0
                    or not isinstance(previous_key, (list, tuple))
                    or len(previous_key) < 2
                ):
                    return
                try:
                    turn_goal_x = float(previous_key[0])
                    turn_goal_y = float(previous_key[1])
                except (TypeError, ValueError):
                    return
                endpoint_x = float(self.active_frontier[2])
                endpoint_y = float(self.active_frontier[3])
                if math.hypot(turn_goal_x - endpoint_x, turn_goal_y - endpoint_y) > 0.10:
                    return
                self.active_turn_completed_route_id = int(self.active_route_id)
                self.active_turn_completed_goal = (endpoint_x, endpoint_y)
                self.active_turn_completed_wall = time.monotonic()
                self.active_turn_completed_odom_xy = (
                    None if self.pose_odom is None else (
                        float(self.pose_odom.x), float(self.pose_odom.y)
                    )
                )
                self.active_turn_completed_translation = 0.0
                self.active_turn_completed_launched = False
                # The normal progress clock may predate a multi-second atomic
                # turn. Restart it at the execution boundary so a successful
                # turn cannot immediately inherit an old stall deadline.
                self.active_progress_time = self.active_turn_completed_wall
                self.active_last_progress_signal = "post_turn_watchdog_armed"
                self.publish_status(
                    "post_turn_progress_watchdog_armed",
                    route_id=int(self.active_route_id),
                    goal=[round(endpoint_x, 3), round(endpoint_y, 3)],
                    timeout=round(float(self.post_turn_stall_timeout), 3),
                    odom_start=(
                        None if self.active_turn_completed_odom_xy is None
                        else [
                            round(self.active_turn_completed_odom_xy[0], 3),
                            round(self.active_turn_completed_odom_xy[1], 3),
                        ]
                    ),
                    launch_distance=round(float(self.progress_epsilon), 3),
                )

    def on_execution_terminal(self, message):
        """Wake planning as soon as the active frontier action succeeds.

        The bridge publishes this only for a successful move_base action.
        Match it to the currently commanded route point before scheduling work;
        a late terminal from a replaced action must not advance a new branch.
        The actual selection remains in the normal planner, preserving the
        existing SLAM, costmap, and Navfn validation path.
        """
        with self.planning_lock:
            if self.task_done or self.active_last_waypoint_map is None:
                return
            frame = (message.header.frame_id or "").strip().lstrip("/")
            map_frame = (
                "" if self.map_msg is None
                else (self.map_msg.header.frame_id or "map").strip().lstrip("/")
            )
            if frame and map_frame and frame != map_frame:
                return
            terminal_delta = math.hypot(
                float(message.pose.position.x) - self.active_last_waypoint_map[0],
                float(message.pose.position.y) - self.active_last_waypoint_map[1],
            )
            if terminal_delta > max(self.waypoint_release_radius, 0.30):
                rospy.logwarn_throttle(
                    3.0,
                    "Global frontier ignored unmatched execution terminal "
                    "delta=%.2fm active=(%.2f,%.2f)",
                    terminal_delta,
                    self.active_last_waypoint_map[0],
                    self.active_last_waypoint_map[1],
                )
                return
            self.active_terminal_received = True
            if self.promote_prefetched_terminal(message):
                return
            # The matching move_base action has already completed. Do not let
            # the next SLAM timer re-project this old mission endpoint onto a
            # nearby grid cell and publish a synthetic 10 cm follow-up action.
            # Release the completed transaction before the immediate planner
            # selects its real successor.
            completed = self.active_frontier
            if completed is not None:
                self.mark_frontier_completed(completed[2], completed[3])
            self.active_frontier = None
            self.active_since = 0.0
            self.active_best_distance = None
            self.active_best_goal_distance = None
            self.active_best_path_distance = None
            self.active_progress_time = 0.0
            self.active_last_progress_signal = "terminal_replan"
            self.active_last_robot_xy = None
            self.active_start_odom_xy = None
            self.active_best_detour_odom_distance = 0.0
            self.active_visited_odom_cells.clear()
            self.active_unreachable_since = None
            self.active_last_waypoint_map = None
            self.active_last_waypoint_yaw = None
            self.active_route_kind = "frontier_endpoint"
            self.active_terminal_received = False
            self.recovery_pending_route_id = 0
            self.recovery_pending_behavior = ""
            self.recovery_pending_reason = ""
            self.turn_connector_released = True
            self.last_status_command_map = None
            self.last_status_command_yaw = None
            self.last_status_mission_map = None
            self.publish_status(
                "terminal_replan_requested",
                completed_goal=[
                    round(float(completed[2]), 3),
                    round(float(completed[3]), 3),
                ] if completed is not None else None,
            )
            # Bypass only the compute-throttle, never route validation. A
            # one-shot timer keeps work outside the action callback, and the
            # shared lock prevents it racing the regular 1 Hz timer.
            self.last_planning_wall = 0.0
            if self.immediate_plan_timer is None and not rospy.is_shutdown():
                self.immediate_plan_timer = rospy.Timer(
                    rospy.Duration(0.01), self.on_immediate_plan, oneshot=True
                )
            rospy.loginfo(
                "Global frontier schedules immediate successor selection "
                "after terminal delta=%.3fm route_id=%d pending=%s",
                terminal_delta,
                self.active_route_id,
                self.prefetched_frontier is not None,
            )

    def promote_prefetched_terminal(self, terminal):
        """Commit a prevalidated successor without rebuilding the full BFS.

        A prefetch has already passed frontier-mask, inflated-costmap, and
        Navfn checks before the robot entered the endpoint. At its matching
        action terminal, validate it once more with Navfn from the terminal
        pose, then publish it directly. This removes the avoidable full-grid
        planning delay from the zero-velocity action boundary. Missing or
        invalid prefetches intentionally fall back to the normal planner.
        """
        if self.prefetched_frontier is None or self.map_msg is None:
            return False
        row, col, x, y = self.prefetched_frontier
        frame = self.map_msg.header.frame_id or "map"
        # A native frontier handoff deliberately happens *inside* the
        # observation envelope, before move_base reaches the old point-goal.
        # Its logical terminal still carries the old endpoint for route
        # identity, but Navfn validation and progress accounting must start at
        # the physical base pose. Otherwise the successor is validated from a
        # point the robot never occupied and its watchdog can later classify a
        # legitimate route as a stall. A normal SUCCEEDED terminal is covered
        # too: its current pose is simply within the normal goal tolerance.
        execution_start = (
            float(terminal.pose.position.x),
            float(terminal.pose.position.y),
        )
        if self.pose_odom is not None:
            current_map = self.transform_xy(
                frame,
                "odom",
                float(self.pose_odom.x),
                float(self.pose_odom.y),
            )
            if current_map is not None:
                execution_start = (float(current_map[0]), float(current_map[1]))
        reachable = self.navfn_goal_reachable(
            execution_start,
            (float(x), float(y)),
            frame,
        )
        if reachable is not True:
            rospy.loginfo(
                "Global frontier defers cached successor after terminal: "
                "Navfn state=%s goal=(%.2f,%.2f)",
                self.navfn_last_validation_state,
                x,
                y,
            )
            return False
        previous = self.active_frontier
        predecessor_route_id = int(self.active_route_id)
        if previous is not None:
            self.mark_frontier_completed(previous[2], previous[3])
        now = time.monotonic()
        self.active_frontier = (int(row), int(col), float(x), float(y))
        self.active_route_id += 1
        self.active_transition_kind = "terminal_prefetched_successor"
        self.active_predecessor_route_id = predecessor_route_id
        self.active_transition_distance = (
            None if previous is None else math.hypot(
                float(previous[2]) - execution_start[0],
                float(previous[3]) - execution_start[1],
            )
        )
        self.active_since = now
        self.active_best_distance = math.hypot(
            float(x) - execution_start[0],
            float(y) - execution_start[1],
        )
        self.active_best_goal_distance = self.active_best_distance
        self.active_best_path_distance = None
        self.active_progress_time = now
        self.active_last_progress_signal = "terminal_prefetch_promoted"
        self.active_last_robot_xy = execution_start
        self.active_start_odom_xy = (
            None if self.pose_odom is None
            else (float(self.pose_odom.x), float(self.pose_odom.y))
        )
        self.active_best_detour_odom_distance = 0.0
        self._reset_active_odom_coverage()
        self.active_unreachable_since = None
        self.active_last_waypoint_map = (float(x), float(y))
        # The endpoint yaw is a soft terminal preference (TEB permits any yaw
        # within pi here). Point it along the newly validated branch until the
        # next normal planning cycle supplies the route tangent.
        command_yaw = math.atan2(
            float(y) - execution_start[1],
            float(x) - execution_start[0],
        )
        self.active_last_waypoint_yaw = command_yaw
        self.active_route_kind = "frontier_endpoint"
        self.active_terminal_received = False
        self.turn_connector_released = True
        self._clear_active_post_turn_watchdog()
        self.last_status_command_map = (float(x), float(y))
        self.last_status_command_yaw = command_yaw
        self.last_status_mission_map = (float(x), float(y))
        self.prefetched_frontier = None
        self.prefetched_goal_map = None
        self.last_planning_wall = now
        self.publish_status(
            "terminal_prefetch_promoted",
            route_kind=self.active_route_kind,
            route_id=int(self.active_route_id),
            predecessor_route_id=int(predecessor_route_id),
            transition_kind=self.active_transition_kind,
            transition_distance_to_previous_endpoint=(
                None
                if self.active_transition_distance is None
                else round(float(self.active_transition_distance), 3)
            ),
            terminal_received=True,
            command_goal=[round(float(x), 3), round(float(y), 3)],
            mission_goal=[round(float(x), 3), round(float(y), 3)],
            execution_start=[
                round(float(execution_start[0]), 3),
                round(float(execution_start[1]), 3),
            ],
            terminal_identity=[
                round(float(terminal.pose.position.x), 3),
                round(float(terminal.pose.position.y), 3),
            ],
        )
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = frame
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.orientation.z = math.sin(0.5 * command_yaw)
        goal.pose.orientation.w = math.cos(0.5 * command_yaw)
        self.publish_route_command(
            goal.header.frame_id,
            x,
            y,
            command_yaw,
            self.active_route_kind,
        )
        self.publisher.publish(goal)
        rospy.loginfo(
            "Global frontier directly promoted prevalidated successor "
            "old=%s new=(%.2f,%.2f) route_id=%d",
            previous,
            x,
            y,
            self.active_route_id,
        )
        return True

    def on_immediate_plan(self, event):
        with self.planning_lock:
            self.immediate_plan_timer = None
            self.on_timer(event)

    def transform_xy(self, target_frame, source_frame, x, y):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.15))
        except Exception as exc:
            rospy.logwarn_throttle(3.0, "Global frontier waiting for %s <- %s transform: %s", target_frame, source_frame, exc)
            return None
        rotation = transform.transform.rotation
        matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        translation = transform.transform.translation
        point = matrix[:3, :3].dot(np.array([x, y, 0.0], dtype=float))
        return point[0] + translation.x, point[1] + translation.y

    def transform_yaw(self, target_frame, source_frame, yaw):
        """Transform a planar yaw using the same TF lookup as ``transform_xy``."""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(0.15),
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                3.0,
                "Global frontier waiting for %s <- %s yaw transform: %s",
                target_frame,
                source_frame,
                exc,
            )
            return None
        rotation = transform.transform.rotation
        tf_yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        return math.atan2(
            math.sin(float(yaw) + tf_yaw),
            math.cos(float(yaw) + tf_yaw),
        )

    @staticmethod
    def heading_delta(x, y, robot_xy, robot_yaw):
        """Return the absolute bearing change needed to reach a candidate."""
        if robot_xy is None or robot_yaw is None:
            return None
        bearing = math.atan2(y - robot_xy[1], x - robot_xy[0])
        return abs(math.atan2(
            math.sin(bearing - robot_yaw),
            math.cos(bearing - robot_yaw),
        ))

    @staticmethod
    def _angle_delta(first, second):
        return math.atan2(math.sin(first - second), math.cos(first - second))

    @staticmethod
    def inflate(occupied, cells):
        inflated = occupied.copy()
        rows, cols = occupied.shape
        for dr in range(-cells, cells + 1):
            for dc in range(-cells, cells + 1):
                if dr * dr + dc * dc > cells * cells:
                    continue
                source_r0, source_r1 = max(0, -dr), min(rows, rows - dr)
                source_c0, source_c1 = max(0, -dc), min(cols, cols - dc)
                target_r0, target_r1 = max(0, dr), min(rows, rows + dr)
                target_c0, target_c1 = max(0, dc), min(cols, cols + dc)
                inflated[target_r0:target_r1, target_c0:target_c1] |= occupied[source_r0:source_r1, source_c0:source_c1]
        return inflated

    @staticmethod
    def nearest_seed(free, row, col, limit):
        rows, cols = free.shape
        if 0 <= row < rows and 0 <= col < cols and free[row, col]:
            return row, col
        for radius in range(1, limit + 1):
            r0, r1 = max(0, row - radius), min(rows, row + radius + 1)
            c0, c1 = max(0, col - radius), min(cols, col + radius + 1)
            candidates = np.argwhere(free[r0:r1, c0:c1])
            if candidates.size:
                candidates[:, 0] += r0
                candidates[:, 1] += c0
                distance = (candidates[:, 0] - row) ** 2 + (candidates[:, 1] - col) ** 2
                result = candidates[np.argmin(distance)]
                return int(result[0]), int(result[1])
        return None

    @staticmethod
    def bfs(free, seed):
        steps = np.full(free.shape, -1, dtype=np.int32)
        queue = collections.deque([seed])
        steps[seed] = 0
        rows, cols = free.shape
        while queue:
            row, col = queue.popleft()
            next_step = steps[row, col] + 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols and free[nr, nc] and steps[nr, nc] < 0:
                    steps[nr, nc] = next_step
                    queue.append((nr, nc))
        return steps

    @staticmethod
    def is_frontier(free, unknown):
        adjacent = np.zeros_like(unknown, dtype=bool)
        adjacent[1:] |= unknown[:-1]
        adjacent[:-1] |= unknown[1:]
        adjacent[:, 1:] |= unknown[:, :-1]
        adjacent[:, :-1] |= unknown[:, 1:]
        return free & adjacent

    def cell_xy(self, message, row, col):
        return (
            message.info.origin.position.x + (col + 0.5) * message.info.resolution,
            message.info.origin.position.y + (row + 0.5) * message.info.resolution,
        )

    def frontier_information(self, unknown, row, col):
        radius = self.info_radius
        r0, r1 = max(0, row - radius), min(unknown.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(unknown.shape[1], col + radius + 1)
        return float(np.count_nonzero(unknown[r0:r1, c0:c1]))

    def _frontier_has_unknown(self, unknown, row, col):
        """Return whether unknown space remains within the dead-end margin.

        A genuine passage (doorway / room entrance) has unknown space extending
        beyond it, so the margin around the frontier cell still contains
        unknown.  A resolved dead-end wall has no unknown anywhere near the
        cell -- only scanned free space and the wall itself.
        """
        radius = self.dead_end_unknown_cells
        r0, r1 = max(0, row - radius), min(unknown.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(unknown.shape[1], col + radius + 1)
        return bool(np.any(unknown[r0:r1, c0:c1]))

    def frontier_structure(self, occupied, row, col):
        radius = self.structure_radius
        r0, r1 = max(0, row - radius), min(occupied.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(occupied.shape[1], col + radius + 1)
        return float(np.count_nonzero(occupied[r0:r1, c0:c1]))

    @staticmethod
    def nearest_safe_approach(
        steps, row, col, max_cells, preferred_steps=None, preferred_mask=None,
        selection_tier="strict_clearance",
    ):
        """Find an approach point with an explicit endpoint/path contract.

        ``steps`` is the loose known-free observation topology. A normal
        frontier route must also be connected through ``preferred_steps``. A
        recovery route may cross a narrow map region only when its *endpoint*
        remains inside ``preferred_mask``; Navfn then validates that exact
        route. It must never publish a loose, wall-adjacent endpoint merely
        because Navfn found a path close to it.
        """
        rows, cols = steps.shape
        r0 = max(0, row - max_cells)
        r1 = min(rows, row + max_cells + 1)
        c0 = max(0, col - max_cells)
        c1 = min(cols, col + max_cells + 1)
        candidates = np.argwhere(steps[r0:r1, c0:c1] >= 0)
        if candidates.size == 0:
            return None
        candidates[:, 0] += r0
        candidates[:, 1] += c0
        if selection_tier == "strict_clearance":
            if preferred_steps is None:
                return None
            strict = preferred_steps[candidates[:, 0], candidates[:, 1]] >= 0
            candidates = candidates[strict]
        elif selection_tier == "navfn_observation_recovery":
            if preferred_mask is None:
                return None
            recovery = preferred_mask[candidates[:, 0], candidates[:, 1]]
            # A recovery is meaningful only when the endpoint cannot already
            # be reached through the strict topology. Otherwise the strict
            # first pass owns this candidate and its audit identity.
            if preferred_steps is not None:
                recovery &= preferred_steps[candidates[:, 0], candidates[:, 1]] < 0
            candidates = candidates[recovery]
        else:
            return None
        if candidates.size == 0:
            return None
        distance = (candidates[:, 0] - row) ** 2 + (candidates[:, 1] - col) ** 2
        # Prefer the closest safe cell, then the shortest route from the
        # robot.  The latter keeps ties from selecting a remote branch.
        order = np.lexsort((steps[candidates[:, 0], candidates[:, 1]], distance))
        selected = candidates[int(order[0])]
        return int(selected[0]), int(selected[1])

    def frontier_is_rejected(self, x, y, now):
        while (
            self.rejected_frontiers
            and now - self.rejected_frontiers[0][0] > self.rejected_timeout
        ):
            self.rejected_frontiers.popleft()
        return any(math.hypot(x - old_x, y - old_y) < 1.5 for _, old_x, old_y in self.rejected_frontiers)

    def frontier_is_completed(self, x, y):
        return any(
            math.hypot(x - old_x, y - old_y) < self.completed_radius
            for old_x, old_y in self.completed_frontiers
        )

    def mark_frontier_completed(self, x, y):
        if not self.frontier_is_completed(x, y):
            self.completed_frontiers.append((x, y))
            rospy.loginfo(
                "Global frontier coverage complete at map=(%.2f,%.2f); remembered=%d",
                x, y, len(self.completed_frontiers),
            )

    @staticmethod
    def world_cell(message, x, y):
        resolution = float(message.info.resolution)
        if resolution <= 0.0:
            return None
        col = int(math.floor((x - message.info.origin.position.x) / resolution))
        row = int(math.floor((y - message.info.origin.position.y) / resolution))
        if not (0 <= row < message.info.height and 0 <= col < message.info.width):
            return None
        return row, col

    def fresh_costmap(self):
        """Return a recent costmap snapshot, or None while it is unavailable."""
        message = self.costmap_msg
        if message is None or message.info.resolution <= 0.0:
            return None
        # Full grids are normally latched and remain unchanged while
        # costmap_2d publishes only incremental updates. Use wall-clock age of
        # the full-grid/update stream rather than the static map stamp.
        age = (
            float("inf")
            if self.costmap_last_receive_wall <= 0.0
            else time.monotonic() - self.costmap_last_receive_wall
        )
        if age > self.costmap_max_age:
            rospy.logwarn_throttle(
                5.0,
                "Global frontier costmap stale age=%.2fs limit=%.2fs",
                age,
                self.costmap_max_age,
            )
            return None
        expected = int(message.info.height) * int(message.info.width)
        if expected <= 0 or len(message.data) != expected:
            rospy.logwarn_throttle(
                5.0,
                "Global frontier costmap shape invalid size=%dx%d cells=%d expected=%d",
                message.info.width,
                message.info.height,
                len(message.data),
                expected,
            )
            return None
        return message

    def costmap_steps(self, robot_xy):
        """Build the Navfn-compatible connected mask for the latest costmap.

        Values >= 253 are lethal in costmap_2d. Unknown cells remain
        traversable here because the production NavfnROS configuration allows
        unknown space; the frontier goal itself is still selected from the
        known SLAM-free mask.
        """
        message = self.fresh_costmap()
        if message is None:
            return None
        frame = (message.header.frame_id or "map").strip().lstrip("/") or "map"
        map_frame = (self.map_msg.header.frame_id or "map") if self.map_msg else "map"
        costmap_robot = robot_xy
        if frame != map_frame:
            costmap_robot = self.transform_xy(frame, map_frame, robot_xy[0], robot_xy[1])
            if costmap_robot is None:
                return None
        try:
            data = np.asarray(message.data, dtype=np.int16).reshape(
                int(message.info.height), int(message.info.width)
            )
        except (TypeError, ValueError):
            return None
        free = data < 253
        cell = self.world_cell(message, costmap_robot[0], costmap_robot[1])
        if cell is None:
            return None
        seed = self.nearest_seed(
            free,
            cell[0],
            cell[1],
            max(1, int(math.ceil(0.8 / message.info.resolution)),),
        )
        if seed is None:
            return None
        return message, data, self.bfs(free, seed)

    def cached_costmap_steps(self, robot_xy, now):
        """Return a throttled costmap connectivity snapshot.

        The global costmap publishes incremental updates while the map node is
        also running a Python BFS. Rebuilding that BFS on every one-second
        timer tick competes with Gazebo and TEB for CPU. A short-lived snapshot
        is sufficient for rejecting an obviously disconnected frontier; TEB's
        rolling local costmap still checks every command against current lidar.
        """
        if (
            self.cached_costmap_validation is not None
            and now - self.cached_costmap_validation_wall
            < self.costmap_validation_period
        ):
            return self.cached_costmap_validation
        validation = self.costmap_steps(robot_xy)
        if validation is not None:
            self.cached_costmap_validation = validation
            self.cached_costmap_validation_wall = now
            return validation
        # Keep a recent valid snapshot through a transient TF/costmap update;
        # ``fresh_costmap`` still expires it through the normal max-age guard.
        if (
            self.cached_costmap_validation is not None
            and now - self.cached_costmap_validation_wall
            <= self.costmap_max_age
        ):
            return self.cached_costmap_validation
        return None

    def candidate_costmap_distance(self, validation, x, y):
        """Return a candidate's costmap path distance, or None if invalid."""
        if validation is None:
            return None
        message, data, steps = validation
        frame = (message.header.frame_id or "map").strip().lstrip("/") or "map"
        map_frame = (self.map_msg.header.frame_id or "map") if self.map_msg else "map"
        target = (x, y)
        if frame != map_frame:
            target = self.transform_xy(frame, map_frame, x, y)
            if target is None:
                return None
        cell = self.world_cell(message, target[0], target[1])
        if cell is None or data[cell] >= 253 or steps[cell] < 0:
            return None
        return float(steps[cell]) * float(message.info.resolution)

    def startup_probe_goal(self, validation, map_frame):
        """Return one nearby connected costmap cell in the SLAM map frame.

        This is intentionally not a frontier candidate.  It is only a
        startup proof that Navfn can plan from the current robot cell through
        the same global costmap used for mission endpoints.  Keeping the
        probe separate prevents a temporarily cold planner from blacklisting
        the first real frontier forever.
        """
        if validation is None:
            return None
        message, data, steps = validation
        resolution = float(message.info.resolution)
        if resolution <= 0.0:
            return None
        desired_steps = max(
            1, int(math.ceil(self.navfn_startup_probe_distance / resolution))
        )
        # Pick a nonzero path-distance cell so Navfn has to construct an
        # actual short path rather than accepting an identical start/goal.
        candidates = np.argwhere((steps >= 1) & (data < 253))
        if candidates.size == 0:
            return None
        distances = np.abs(steps[candidates[:, 0], candidates[:, 1]] - desired_steps)
        row, col = candidates[int(np.argmin(distances))]
        goal = self.cell_xy(message, int(row), int(col))
        costmap_frame = (
            (message.header.frame_id or "map").strip().lstrip("/") or "map"
        )
        map_frame = (map_frame or "map").strip().lstrip("/") or "map"
        if costmap_frame != map_frame:
            goal = self.transform_xy(map_frame, costmap_frame, goal[0], goal[1])
        return goal

    def navigation_stack_is_ready(self, robot_map, map_frame, now):
        """Gate mission selection on a usable online-SLAM navigation chain.

        A latched /map is insufficient: GMapping may not have published
        map->odom yet and move_base may still be building its global costmap.
        The gate remains closed until a costmap-connected local probe receives
        a non-empty exact Navfn plan.  Once established, normal route checks
        continue to handle transient map updates without pausing a vehicle
        that is already executing a valid frontier.
        """
        if self.navigation_stack_ready:
            return True
        validation = self.cached_costmap_steps(robot_map, now)
        if validation is None:
            state = "waiting_for_global_costmap_connectivity"
            if state != self.navigation_readiness_state:
                self.navigation_readiness_state = state
                rospy.loginfo("Global frontier startup gate: %s", state)
                self.publish_status("navigation_readiness", ready=False, state=state)
            return False
        probe = self.startup_probe_goal(validation, map_frame)
        if probe is None:
            state = "waiting_for_costmap_probe_cell"
            if state != self.navigation_readiness_state:
                self.navigation_readiness_state = state
                rospy.loginfo("Global frontier startup gate: %s", state)
                self.publish_status("navigation_readiness", ready=False, state=state)
            return False
        reachable = self.navfn_goal_reachable(robot_map, probe, map_frame)
        if reachable is not True:
            state = "waiting_for_navfn_probe"
            if reachable is False:
                state = "navfn_probe_empty_after_warmup"
            if state != self.navigation_readiness_state:
                self.navigation_readiness_state = state
                rospy.loginfo(
                    "Global frontier startup gate: %s probe=(%.2f,%.2f)",
                    state,
                    probe[0],
                    probe[1],
                )
                self.publish_status(
                    "navigation_readiness",
                    ready=False,
                    state=state,
                    probe=[round(float(probe[0]), 3), round(float(probe[1]), 3)],
                )
            return False
        self.navigation_stack_ready = True
        self.navigation_readiness_state = "ready"
        rospy.loginfo(
            "Global frontier startup gate passed: Navfn probe=(%.2f,%.2f)",
            probe[0],
            probe[1],
        )
        self.publish_status(
            "navigation_readiness",
            ready=True,
            state="ready",
            probe=[round(float(probe[0]), 3), round(float(probe[1]), 3)],
        )
        return True

    def navfn_goal_reachable(self, robot_map, goal_xy, frame_id):
        """Validate one candidate against move_base's actual Navfn plugin.

        The Python occupancy/costmap masks are deliberately conservative, but
        they cannot reproduce every Navfn detail (unknown handling, planner
        tolerance, and the live costmap reset state).  Validate only the few
        candidates that would become mission goals, never every frontier cell.

        ``None`` means the planner is unavailable or has not yet demonstrated
        one usable route; callers wait for the next planning cycle without
        rejecting the candidate. ``False`` is reserved for an empty response
        after Navfn is operational, i.e. a confirmed non-reachable branch.
        """
        if not self.navfn_plan_validation:
            self.navfn_last_validation_state = "disabled"
            return True
        try:
            rospy.wait_for_service(
                self.navfn_make_plan_service,
                timeout=self.navfn_make_plan_timeout,
            )
        except (rospy.ROSException, rospy.ROSInterruptException):
            rospy.logwarn_throttle(
                5.0,
                "Global frontier Navfn validation unavailable: %s",
                self.navfn_make_plan_service,
            )
            self.navfn_last_validation_state = "unavailable"
            return None
        request = GetPlanRequest()
        request.start.header.frame_id = (frame_id or "map").strip().lstrip("/") or "map"
        request.start.header.stamp = rospy.Time.now()
        request.start.pose.position.x = float(robot_map[0])
        request.start.pose.position.y = float(robot_map[1])
        request.start.pose.orientation.w = 1.0
        request.goal.header.frame_id = request.start.header.frame_id
        request.goal.header.stamp = request.start.header.stamp
        request.goal.pose.position.x = float(goal_xy[0])
        request.goal.pose.position.y = float(goal_xy[1])
        request.goal.pose.orientation.w = 1.0
        # This service validates the exact action endpoint, not a nearby pose
        # that happens to be navigable. A nonzero tolerance can return a plan
        # ending beside a wall-adjacent frontier cell; move_base then receives
        # the original point and TEB correctly stalls.
        request.tolerance = 0.0
        try:
            response = self.navfn_service(request)
        except (rospy.ServiceException, rospy.ROSException) as exc:
            rospy.logwarn_throttle(
                5.0,
                "Global frontier Navfn validation failed for goal=(%.2f,%.2f): %s",
                goal_xy[0], goal_xy[1], exc,
            )
            self.navfn_last_validation_state = "service_error"
            return None
        now_wall = time.monotonic()
        if self.navfn_first_response_wall is None:
            self.navfn_first_response_wall = now_wall
        reachable = bool(response.plan.poses)
        if reachable:
            endpoint = response.plan.poses[-1].pose.position
            endpoint_error = math.hypot(
                float(endpoint.x) - float(goal_xy[0]),
                float(endpoint.y) - float(goal_xy[1]),
            )
            resolution = (
                float(self.map_msg.info.resolution)
                if self.map_msg is not None and self.map_msg.info.resolution > 0.0
                else 0.10
            )
            # The startup probe is selected from a costmap cell centre while
            # Navfn returns its own grid-cell centre. A one-cell endpoint
            # difference is therefore normal during costmap bootstrap. This
            # function validates frontier reachability and startup readiness;
            # strict visual-target identity remains enforced separately by
            # StreamingNavfnPlanner's streaming_goal_epsilon.
            if endpoint_error > max(0.01, 1.01 * resolution):
                self.navfn_last_validation_state = "endpoint_offset"
                rospy.logwarn(
                    "Global frontier rejected Navfn offset endpoint goal=(%.2f,%.2f) "
                    "plan_end=(%.2f,%.2f) error=%.3fm",
                    goal_xy[0],
                    goal_xy[1],
                    endpoint.x,
                    endpoint.y,
                    endpoint_error,
                )
                return False
            self.navfn_last_validation_state = "reachable"
            if self.navfn_first_nonempty_response_wall is None:
                self.navfn_first_nonempty_response_wall = now_wall
                rospy.loginfo(
                    "Global frontier Navfn validation is operational: first "
                    "non-empty plan goal=(%.2f,%.2f)",
                    goal_xy[0], goal_xy[1],
                )
            return True
        if not reachable:
            if self.navfn_first_nonempty_response_wall is None:
                warmup_age = now_wall - self.navfn_first_response_wall
                if warmup_age < self.navfn_empty_warmup_seconds:
                    self.navfn_last_validation_state = "bootstrap_empty"
                    rospy.loginfo_throttle(
                        1.0,
                        "Global frontier waits for Navfn's first usable plan: "
                        "empty response age=%.2fs/%.2fs goal=(%.2f,%.2f)",
                        warmup_age,
                        self.navfn_empty_warmup_seconds,
                        goal_xy[0], goal_xy[1],
                    )
                    return None
                # The startup gate consumes this result and continues to wait
                # on its local probe. Returning False here is deliberate: it
                # avoids an infinite bootstrap_empty state while keeping real
                # frontier candidates out of this path until Navfn succeeds.
                self.navfn_last_validation_state = "bootstrap_timeout"
                rospy.logwarn_throttle(
                    3.0,
                    "Global frontier Navfn startup probe still empty after %.2fs "
                    "goal=(%.2f,%.2f)",
                    warmup_age,
                    goal_xy[0], goal_xy[1],
                )
                return False
            warmup_age = now_wall - self.navfn_first_response_wall
            if warmup_age < self.navfn_empty_warmup_seconds:
                self.navfn_last_validation_state = "warmup_empty"
                rospy.loginfo_throttle(
                    1.0,
                    "Global frontier waits for Navfn costmap warmup: "
                    "empty plan age=%.2fs/%.2fs goal=(%.2f,%.2f)",
                    warmup_age,
                    self.navfn_empty_warmup_seconds,
                    goal_xy[0],
                    goal_xy[1],
                )
                return None
            self.navfn_last_validation_state = "unreachable"
            rospy.logwarn(
                "Global frontier rejected Navfn-empty candidate goal=(%.2f,%.2f) "
                "robot=(%.2f,%.2f)",
                goal_xy[0], goal_xy[1], robot_map[0], robot_map[1],
            )
        return False

    def choose_valid_frontier(
        self, message, steps, frontier, unknown, occupied, now,
        robot_map, validation=None, excluded=None, heading_reference=None,
        max_heading_delta=None, route_seed=None, semantic_hint=None,
        preferred_steps=None, preferred_mask=None,
        allow_observation_recovery=False, route_anchor_xy=None,
        score_path_from_steps=False, allow_heading_fallback=True,
    ):
        """Choose a score-best candidate with an exact executable Navfn route."""
        # A failed service validation is remembered as a temporary rejection,
        # so the next score-best branch is tried instead of repeating the same
        # empty Navfn route on every timer tick.
        self.frontier_validation_budget_exhausted = False
        self.frontier_validation_pending = False
        tiers = ["strict_clearance"]
        if allow_observation_recovery:
            tiers.append("navfn_observation_recovery")
        for selection_tier in tiers:
            heading_limit = max_heading_delta
            for _ in range(8):
                candidate = self.choose_frontier(
                    message,
                    steps,
                    frontier,
                    unknown,
                    occupied,
                    now,
                    excluded=excluded,
                    validation=validation,
                    robot_map=robot_map,
                    heading_reference=heading_reference,
                    max_heading_delta=heading_limit,
                    route_seed=route_seed,
                    route_anchor_xy=route_anchor_xy,
                    score_path_from_steps=score_path_from_steps,
                    semantic_hint=semantic_hint,
                    preferred_steps=preferred_steps,
                    preferred_mask=preferred_mask,
                    selection_tier=selection_tier,
                )
                if (
                    candidate is None
                    and heading_limit is not None
                    and allow_heading_fallback
                ):
                    # Direction continuity is a preference for prefetching,
                    # not a deadlock condition. Retry this safety tier once
                    # without the heading bound before considering recovery.
                    rospy.loginfo(
                        "Global frontier has no %s candidate within heading "
                        "limit %.1fdeg; allowing a necessary turn",
                        selection_tier,
                        math.degrees(heading_limit),
                    )
                    heading_limit = None
                    continue
                if candidate is None:
                    break
                reachable = self.navfn_goal_reachable(
                    robot_map,
                    (candidate[2], candidate[3]),
                    message.header.frame_id or "map",
                )
                if reachable is True:
                    self.last_frontier_selection_mode = selection_tier
                    return candidate
                if reachable is None:
                    # Do not turn a planner startup race into either an unsafe
                    # route or a false exhausted map conclusion.
                    self.frontier_validation_pending = True
                    return None
                self.rejected_frontiers.append((now, candidate[2], candidate[3]))
                rospy.logwarn(
                    "Global frontier deferred %s Navfn-invalid candidate "
                    "map=(%.2f,%.2f) for %.0fs",
                    selection_tier,
                    candidate[2],
                    candidate[3],
                    self.rejected_timeout,
                )
            else:
                # The bounded validation budget protects the 1 Hz planner
                # loop. More untried candidates may remain, so wait rather
                # than incorrectly declaring coverage complete.
                self.frontier_validation_budget_exhausted = True
                return None
        return None

    def choose_frontier(
        self, message, steps, frontier, unknown, occupied, now, excluded=None,
        validation=None, robot_map=None, heading_reference=None,
        max_heading_delta=None, route_seed=None, semantic_hint=None,
        preferred_steps=None, preferred_mask=None,
        selection_tier="strict_clearance", route_anchor_xy=None,
        score_path_from_steps=False,
    ):
        route_anchor_xy = robot_map if route_anchor_xy is None else route_anchor_xy
        minimum_steps = int(math.ceil(self.min_path_distance / message.info.resolution))
        approach_cells = max(
            1, int(math.ceil(self.frontier_approach_distance / message.info.resolution))
        )
        # The frontier mask may use a looser information-boundary clearance;
        # the strict path-distance test is applied after selecting a safe
        # approach cell below.
        rows, cols = np.nonzero(frontier)
        if rows.size == 0:
            return None
        if rows.size > self.candidate_limit:
            indices = np.linspace(0, rows.size - 1, self.candidate_limit, dtype=np.int64)
            rows, cols = rows[indices], cols[indices]
        structured_best = None
        structured_score = -float("inf")
        fallback_best = None
        fallback_score = -float("inf")
        for row, col in zip(rows.tolist(), cols.tolist()):
            approach = self.nearest_safe_approach(
                steps,
                row,
                col,
                approach_cells,
                preferred_steps=preferred_steps,
                preferred_mask=preferred_mask,
                selection_tier=selection_tier,
            )
            if approach is None:
                continue
            target_row, target_col = approach
            x, y = self.cell_xy(message, target_row, target_col)
            if excluded and any(
                math.hypot(x - old_x, y - old_y) < self.completed_radius
                for old_x, old_y in excluded
            ):
                continue
            if self.frontier_is_completed(x, y):
                continue
            if self.frontier_is_rejected(x, y, now):
                continue
            costmap_distance = self.candidate_costmap_distance(validation, x, y)
            if validation is not None and costmap_distance is None:
                continue
            path_distance = (
                costmap_distance
                if costmap_distance is not None
                else steps[target_row, target_col] * message.info.resolution
            )
            # For an ordinary selection the live-robot costmap route is both
            # the executable distance and the score basis. A prefetched
            # successor is different: its lifecycle begins at the active
            # observation endpoint, so rank it by that endpoint-rooted BFS
            # distance while retaining the live costmap result as a required
            # connectivity gate above.
            score_path_distance = (
                float(steps[target_row, target_col]) * message.info.resolution
                if score_path_from_steps
                else path_distance
            )
            information = self.frontier_information(unknown, row, col)
            structure = self.frontier_structure(occupied, row, col)
            endpoint_heading_delta = self.heading_delta(
                x,
                y,
                route_anchor_xy,
                heading_reference,
            )
            # The BFS field is the exact route representation used by this
            # frontier node. Score the first half-metre route tangent, not
            # the direct line to the endpoint. A direct endpoint bearing can
            # conceal an initial U-turn around an observed wall.
            route_initial_heading = None
            if route_seed is not None:
                route_initial_heading, _ = self.route_headings(
                    message,
                    steps,
                    route_seed,
                    (target_row, target_col),
                    route_anchor_xy,
                )
            route_heading_delta = (
                endpoint_heading_delta
                if route_initial_heading is None or heading_reference is None
                else abs(self._angle_delta(route_initial_heading, heading_reference))
            )
            heading_penalty = (
                0.0
                if route_heading_delta is None
                else self.heading_weight * (1.0 - math.cos(route_heading_delta))
            )
            if (
                max_heading_delta is not None
                and route_heading_delta is not None
                and route_heading_delta > max_heading_delta
            ):
                continue
            # Prefer a large unseen boundary, but do not repeatedly choose a
            # remote branch while closer useful coverage remains. Structure is
            # deliberately a bonus rather than a hard constraint: an open map
            # still remains explorable when no doorway-like frontier exists.
            score = (
                information * 0.035
                + structure * self.structure_weight
                - score_path_distance * 0.12
                - heading_penalty
            )
            if semantic_hint is not None and self.semantic_hint_weight > 0.0:
                # The hint is not a geometric shortcut through an obstacle.
                # It ranks only this already BFS/costmap/Navfn-validated
                # frontier by how much it advances the mapped search boundary
                # toward the last reliable visual target direction.
                hint_distance = min(
                    self.semantic_hint_max_distance,
                    math.hypot(x - semantic_hint[0], y - semantic_hint[1]),
                )
                score -= self.semantic_hint_weight * hint_distance
            candidate = (
                target_row,
                target_col,
                x,
                y,
                path_distance,
                information,
                structure,
                score,
            )
            if steps[target_row, target_col] < minimum_steps:
                # No remote frontier remains after a strict search. A nearby,
                # Navfn-reachable observation point can still reveal a door or
                # room entrance. Keep it as a last resort rather than calling
                # the map exhausted merely because it is less than 1.2 m away.
                if selection_tier == "navfn_observation_recovery" and score > fallback_score:
                    fallback_best = candidate
                    fallback_score = score
                continue
            if score > fallback_score:
                fallback_score = score
                fallback_best = candidate
            if structure >= self.min_structure_cells and score > structured_score:
                structured_score = score
                structured_best = candidate
        if structured_best is not None:
            return structured_best
        if fallback_best is not None:
            return fallback_best
        return None

    def prefetch_next_frontier(
        self, message, steps, frontier, unknown, occupied, now, active_xy,
        robot_map, validation=None, heading_reference=None, route_seed=None,
        preferred_steps=None, preferred_mask=None, route_anchor_xy=None,
        transition_basis="robot_bfs_tangent", score_path_from_steps=False,
        max_heading_delta=None, allow_heading_fallback=True,
        transition_preference="unclassified",
    ):
        """Select a successor from the active route's endpoint transition."""
        if self.prefetched_frontier is not None or self.active_frontier is None:
            return
        route_anchor_xy = active_xy if route_anchor_xy is None else route_anchor_xy
        candidate = self.choose_valid_frontier(
            message,
            steps,
            frontier,
            unknown,
            occupied,
            now,
            robot_map,
            excluded=[active_xy],
            validation=validation,
            heading_reference=heading_reference,
            max_heading_delta=(
                self.heading_hard_limit
                if max_heading_delta is None else max_heading_delta
            ),
            route_seed=route_seed,
            route_anchor_xy=route_anchor_xy,
            score_path_from_steps=score_path_from_steps,
            allow_heading_fallback=allow_heading_fallback,
            preferred_steps=preferred_steps,
            preferred_mask=preferred_mask,
        )
        if candidate is None:
            return False
        row, col, x, y, path_distance, information, structure, score = candidate
        # The bridge must decide whether it can replace the current action
        # without a stop. Endpoint-to-endpoint bearings are not sufficient in
        # an indoor map: a doorway route can initially turn away from the
        # endpoint. Publish the same BFS entry tangent used by candidate
        # selection as an explicit route contract for that decision.
        pending_entry_heading = None
        if route_seed is not None:
            pending_entry_heading, _ = self.route_headings(
                message, steps, route_seed, (row, col), route_anchor_xy
            )
        transition_distance = (
            None
            if steps is None or steps[row, col] < 0
            else float(steps[row, col]) * float(message.info.resolution)
        )
        self.prefetched_frontier = (row, col, x, y)
        self.prefetched_goal_map = (float(x), float(y))
        self.publish_status(
            "frontier_prefetched",
            active_goal=[round(float(active_xy[0]), 3), round(float(active_xy[1]), 3)],
            pending_goal=[round(float(x), 3), round(float(y), 3)],
            pending_route_id=int(self.active_route_id + 1),
            pending_entry_yaw=(
                None
                if pending_entry_heading is None
                else round(float(pending_entry_heading), 4)
            ),
            pending_entry_yaw_basis=(
                transition_basis if pending_entry_heading is not None else None
            ),
            transition_path_distance=(
                None if transition_distance is None
                else round(float(transition_distance), 3)
            ),
            transition_preference=transition_preference,
        )
        rospy.loginfo(
            "Global frontier prefetched next branch active=(%.2f,%.2f) "
            "pending=(%.2f,%.2f) robot_path=%.2fm transition_path=%.2fm "
            "information=%.0f structure=%.0f score=%.2f transition_delta=%.1fdeg",
            active_xy[0],
            active_xy[1],
            x,
            y,
            path_distance,
            float("nan") if transition_distance is None else transition_distance,
            information,
            structure,
            score,
            math.degrees(
                self._candidate_route_heading_delta(
                    message, steps, route_seed, row, col, route_anchor_xy,
                    heading_reference,
                )
            )
            if self._candidate_route_heading_delta(
                message, steps, route_seed, row, col, route_anchor_xy,
                heading_reference,
            ) is not None
            else float("nan"),
        )
        return True

    def promote_prefetched_frontier(
        self, message, steps, robot_map, now, validation=None,
        preserve_route_id=False,
    ):
        """Promote a pending branch after the previous frontier is inspected."""
        if self.prefetched_frontier is None:
            return None
        _, _, x, y = self.prefetched_frontier
        if validation is not None and self.candidate_costmap_distance(
            validation, x, y
        ) is None:
            rospy.logwarn(
                "Global frontier discarded prefetched branch map=(%.2f,%.2f): "
                "global costmap has no connected route",
                x,
                y,
            )
            self.prefetched_frontier = None
            self.prefetched_goal_map = None
            return None
        reachable = self.navfn_goal_reachable(
            robot_map, (x, y), message.header.frame_id or "map"
        )
        if reachable is False:
            rospy.logwarn(
                "Global frontier discarded prefetched Navfn-empty branch "
                "map=(%.2f,%.2f)", x, y,
            )
            self.prefetched_frontier = None
            self.prefetched_goal_map = None
            return None
        reassociated = self.nearest_reachable_cell(message, steps, x, y)
        if reassociated is None:
            rospy.logwarn(
                "Global frontier discarded prefetched branch map=(%.2f,%.2f): "
                "no safe connected cell",
                x,
                y,
            )
            self.prefetched_frontier = None
            self.prefetched_goal_map = None
            return None
        row, col = reassociated
        predecessor_route_id = int(self.active_route_id)
        previous_frontier = self.active_frontier
        self.active_frontier = (row, col, x, y)
        if not preserve_route_id:
            self.active_route_id += 1
        self.active_transition_kind = (
            "prefix_continuation" if preserve_route_id else "endpoint_divergence"
        )
        self.active_predecessor_route_id = predecessor_route_id
        self.active_transition_distance = (
            None if previous_frontier is None else math.hypot(
                float(previous_frontier[2]) - robot_map[0],
                float(previous_frontier[3]) - robot_map[1],
            )
        )
        self.active_since = now
        self.active_best_distance = math.hypot(x - robot_map[0], y - robot_map[1])
        self.active_best_goal_distance = self.active_best_distance
        self.active_best_path_distance = (
            float(steps[row, col]) * float(message.info.resolution)
            if steps is not None and steps[row, col] >= 0 else None
        )
        self.active_progress_time = now
        self.active_last_progress_signal = "route_and_goal_initialized"
        self.active_last_robot_xy = (robot_map[0], robot_map[1])
        self.active_start_odom_xy = (
            None if self.pose_odom is None
            else (float(self.pose_odom.x), float(self.pose_odom.y))
        )
        self.active_best_detour_odom_distance = 0.0
        self._reset_active_odom_coverage()
        self.active_unreachable_since = None
        # The promoted endpoint is a mission commitment.  The command sent to
        # move_base is selected below from the current connected route; reset
        # the command cache so a distant branch first receives a connector
        # point instead of being published as one large jump.
        self.active_last_waypoint_map = None
        self.active_last_waypoint_yaw = None
        self.active_route_kind = "frontier_endpoint"
        self.active_terminal_received = False
        self.turn_connector_released = True
        self._clear_active_post_turn_watchdog()
        self.last_status_command_map = None
        self.last_status_command_yaw = None
        self.last_status_mission_map = None
        self.prefetched_frontier = None
        self.prefetched_goal_map = None
        rospy.loginfo(
            "Global frontier promoted prefetched branch map=(%.2f,%.2f) "
            "distance=%.2fm",
            x,
            y,
            self.active_best_distance,
        )
        return row, col, x, y

    @staticmethod
    def waypoint_on_path(steps, row, col, lookahead_steps):
        current = (row, col)
        while steps[current] > lookahead_steps:
            candidates = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = current[0] + dr, current[1] + dc
                if 0 <= neighbour[0] < steps.shape[0] and 0 <= neighbour[1] < steps.shape[1]:
                    if 0 <= steps[neighbour] < steps[current]:
                        candidates.append(neighbour)
            if not candidates:
                return None
            current = min(candidates, key=lambda item: steps[item])
        return current

    @staticmethod
    def route_path(steps, seed, target):
        """Recover one deterministic shortest-cell route from ``seed``.

        ``steps`` is the BFS distance field already used for frontier
        validation.  Recovering the route from that same field keeps the
        direction contract consistent with the path that Navfn/TEB will see;
        this is not a second planner or a guessed steering direction.
        """
        if steps is None or seed is None or target is None:
            return []
        if not (
            0 <= seed[0] < steps.shape[0]
            and 0 <= seed[1] < steps.shape[1]
            and 0 <= target[0] < steps.shape[0]
            and 0 <= target[1] < steps.shape[1]
        ):
            return []
        if steps[seed] < 0 or steps[target] < 0:
            return []
        current = (int(target[0]), int(target[1]))
        reverse_path = [current]
        while current != (int(seed[0]), int(seed[1])):
            current_step = int(steps[current])
            candidates = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = current[0] + dr, current[1] + dc
                if (
                    0 <= neighbour[0] < steps.shape[0]
                    and 0 <= neighbour[1] < steps.shape[1]
                    and 0 <= steps[neighbour] < current_step
                ):
                    candidates.append(neighbour)
            if not candidates:
                return []
            # Match ``waypoint_on_path``'s deterministic lower-step policy;
            # row/column order only breaks equal-length BFS ties.
            current = min(candidates, key=lambda item: (int(steps[item]), item[0], item[1]))
            reverse_path.append(current)
        return list(reversed(reverse_path))

    def route_headings(self, message, steps, seed, target, robot_xy):
        """Return initial and terminal tangent bearings for one BFS route."""
        path = self.route_path(steps, seed, target)
        if len(path) < 2:
            return None, None
        resolution = float(message.info.resolution)
        anchor_index = min(
            len(path) - 1,
            max(1, int(math.ceil(0.5 / max(resolution, 1e-6)))),
        )
        anchor_x, anchor_y = self.cell_xy(message, path[anchor_index][0], path[anchor_index][1])
        if robot_xy is None or math.hypot(anchor_x - robot_xy[0], anchor_y - robot_xy[1]) < 1e-3:
            first_x, first_y = self.cell_xy(message, path[1][0], path[1][1])
            initial = math.atan2(first_y - robot_xy[1], first_x - robot_xy[0]) if robot_xy else None
        else:
            initial = math.atan2(anchor_y - robot_xy[1], anchor_x - robot_xy[0])
        previous_x, previous_y = self.cell_xy(message, path[-2][0], path[-2][1])
        final_x, final_y = self.cell_xy(message, path[-1][0], path[-1][1])
        terminal = math.atan2(final_y - previous_y, final_x - previous_x)
        return initial, terminal

    def _candidate_route_heading_delta(
        self, message, steps, seed, row, col, robot_xy, robot_yaw,
    ):
        """Return the first-route-tangent turn required for one candidate.

        This intentionally falls back to endpoint bearing only when no BFS
        seed is available.  The normal online-SLAM path always supplies a
        seed, so the selection decision is based on an actual connected route
        rather than a line-of-sight assumption.
        """
        endpoint_x, endpoint_y = self.cell_xy(message, row, col)
        fallback = self.heading_delta(endpoint_x, endpoint_y, robot_xy, robot_yaw)
        if seed is None or robot_yaw is None:
            return fallback
        initial, _ = self.route_headings(
            message, steps, seed, (row, col), robot_xy
        )
        if initial is None:
            return fallback
        return abs(self._angle_delta(initial, robot_yaw))

    def prefetched_route_continues_active(
        self, steps, seed, active_cell, pending_cell
    ):
        """Return whether a pending endpoint genuinely extends this route.

        An endpoint that happens to be near the robot is not necessarily the
        continuation of the action currently owned by move_base.  Replacing
        the action before its endpoint is reached is safe only when the new
        deterministic map route contains the full active route as a prefix.
        This is a topological contract, not a heading or distance tuning gate.
        """
        active_path = self.route_path(steps, seed, active_cell)
        pending_path = self.route_path(steps, seed, pending_cell)
        if len(active_path) < 2 or len(pending_path) <= len(active_path):
            return False
        return pending_path[:len(active_path)] == active_path

    def nearest_reachable_cell(self, message, steps, x, y):
        """Reassociate a remembered world-space frontier with a new map grid.

        Gmapping can turn the exact frontier cell into known free space, or
        shift the local cell boundary by one cell, while the route remains
        connected. Reusing the nearest reachable cell avoids treating that
        normal map update as a branch failure.
        """
        if steps is None:
            return None
        resolution = float(message.info.resolution)
        target_col = int((x - message.info.origin.position.x) / resolution)
        target_row = int((y - message.info.origin.position.y) / resolution)
        radius = max(1, int(math.ceil(self.active_reassociation_radius / resolution)))
        r0 = max(0, target_row - radius)
        r1 = min(steps.shape[0], target_row + radius + 1)
        c0 = max(0, target_col - radius)
        c1 = min(steps.shape[1], target_col + radius + 1)
        reachable = np.argwhere(steps[r0:r1, c0:c1] >= 0)
        if reachable.size == 0:
            return None
        reachable[:, 0] += r0
        reachable[:, 1] += c0
        distance = (reachable[:, 0] - target_row) ** 2 + (reachable[:, 1] - target_col) ** 2
        selected = reachable[int(np.argmin(distance))]
        return int(selected[0]), int(selected[1])

    def on_timer(self, _event):
        with self.planning_lock:
            started = time.monotonic()
            try:
                self._on_timer(_event)
            except Exception:
                rospy.logerr(
                    "Global frontier timer failed; retaining the last goal: %s",
                    traceback.format_exc().strip(),
                )
            finally:
                elapsed = time.monotonic() - started
                rospy.loginfo_throttle(
                    5.0,
                    "Global frontier heartbeat active=%s pending=%s map_ready=%s "
                    "costmap_ready=%s cycle=%.3fs",
                    self.active_frontier is not None,
                    self.prefetched_frontier is not None,
                    self.map_msg is not None,
                    self.fresh_costmap() is not None,
                    elapsed,
                )
                if elapsed > 1.5:
                    rospy.logwarn(
                        "Global frontier cycle slow: %.3fs active=%s pending=%s",
                        elapsed,
                        self.active_frontier is not None,
                        self.prefetched_frontier is not None,
                    )

    def _on_timer(self, _event):
        if self.task_done:
            return
        if self.map_msg is None or self.pose_odom is None:
            return
        message = self.map_msg
        robot_map = self.transform_xy(message.header.frame_id or "map", "odom", self.pose_odom.x, self.pose_odom.y)
        if robot_map is None or message.info.resolution <= 0.0:
            return
        now = time.monotonic()
        if not self.navigation_stack_is_ready(
            robot_map, message.header.frame_id or "map", now
        ):
            return
        robot_yaw_map = self.transform_yaw(
            message.header.frame_id or "map",
            "odom",
            self.pose_odom.theta,
        )
        # The active endpoint is intentionally stable while the vehicle is
        # travelling toward it.  Do not rebuild two full occupancy/costmap
        # BFS graphs on every one-second tick in that interval: the local
        # costmap and TEB already perform the high-rate obstacle checks.  Wake
        # the planner immediately when a branch is pending, the endpoint is
        # close enough to prefetch, or the progress watchdog needs recovery.
        if self.active_frontier is not None:
            active_distance = math.hypot(
                self.active_frontier[2] - robot_map[0],
                self.active_frontier[3] - robot_map[1],
            )
            command_distance = (
                float("inf")
                if self.active_last_waypoint_map is None
                else math.hypot(
                    self.active_last_waypoint_map[0] - robot_map[0],
                    self.active_last_waypoint_map[1] - robot_map[1],
                )
            )
            needs_planning = (
                self.recovery_pending_route_id == self.active_route_id
                or
                self.prefetched_frontier is not None
                or active_distance <= self.prefetch_distance
                or command_distance <= self.waypoint_release_radius
                or (
                    self.active_progress_time > 0.0
                    and now - self.active_progress_time >= self.stall_timeout
                )
            )
            if (
                not needs_planning
                and now - self.last_planning_wall < self.planning_period
            ):
                return
        self.last_planning_wall = now
        data = np.asarray(message.data, dtype=np.int8).reshape(message.info.height, message.info.width)
        unknown = data == -1
        known_free = data == 0
        occupied = data >= 50
        clearance_cells = max(1, int(math.ceil(self.clearance / message.info.resolution)) - 1)
        strict_free = known_free & ~self.inflate(occupied, clearance_cells)
        frontier_cells = max(
            1,
            int(math.ceil(self.frontier_clearance / message.info.resolution)) - 1,
        )
        frontier_free = known_free & ~self.inflate(occupied, frontier_cells)
        robot_col = int((robot_map[0] - message.info.origin.position.x) / message.info.resolution)
        robot_row = int((robot_map[1] - message.info.origin.position.y) / message.info.resolution)
        strict_seed = self.nearest_seed(
            strict_free,
            robot_row,
            robot_col,
            max(1, int(0.8 / message.info.resolution)),
        )
        strict_steps = (
            None if strict_seed is None else self.bfs(strict_free, strict_seed)
        )
        # The raw occupancy clearance is an advisory preference. Costmap
        # connectivity plus Navfn are the executable route contract; using a
        # second stricter BFS as a hard gate can falsely declare a narrow
        # doorway unexplorable even while move_base has a valid route to an
        # observation point before it.
        seed = self.nearest_seed(
            frontier_free,
            robot_row,
            robot_col,
            max(1, int(0.8 / message.info.resolution)),
        )
        if seed is None:
            rospy.logwarn_throttle(
                3.0,
                "Global frontier has no known-free observation seed at odom=(%.2f,%.2f)",
                self.pose_odom.x,
                self.pose_odom.y,
            )
            return
        steps = self.bfs(frontier_free, seed)
        # Detect and connect known-free observation boundaries through the
        # same loose topology. Every selected endpoint still passes the live
        # global-costmap and Navfn gates below; strict_steps only prefers a
        # larger raw-map margin when such a route is available.
        frontier = self.is_frontier(frontier_free, unknown)
        active_cell = None
        held_waypoint_map = None
        early_promoted = None
        # The raw map is the source of frontier information.  A second BFS on
        # the latest global costmap is used only to reject candidates that
        # Navfn would regard as lethal or disconnected after inflation.
        validation = self.cached_costmap_steps(robot_map, now)
        route_steps = steps

        if self.active_frontier is not None:
            row, col, x, y = self.active_frontier
            # A max-range lidar update can turn the selected unknown-boundary
            # cell into known free space before the vehicle reaches it. That is
            # useful map information, not a reason to abandon a safe route and
            # choose an unrelated branch. Keep the commitment while the cell is
            # still connected through clearance-safe free space; replace it
            # only after arrival, a genuine stall/timeout, or loss of
            # reachability.
            reassociated = self.nearest_reachable_cell(message, steps, x, y)
            if reassociated is not None and validation is not None:
                if self.candidate_costmap_distance(validation, x, y) is None:
                    reassociated = None
            if reassociated is not None:
                row, col = reassociated
                self.active_unreachable_since = None
                distance = math.hypot(x - robot_map[0], y - robot_map[1])
                route_distance = float(steps[row, col]) * float(message.info.resolution)
                if validation is not None:
                    costmap_distance = self.candidate_costmap_distance(
                        validation, x, y
                    )
                    if costmap_distance is not None:
                        route_distance = costmap_distance
                route_progress = (
                    self.active_best_path_distance is None
                    or route_distance
                    < self.active_best_path_distance - self.progress_epsilon
                )
                goal_progress = (
                    self.active_best_goal_distance is None
                    or distance
                    < self.active_best_goal_distance - self.progress_epsilon
                )
                if route_progress:
                    self.active_best_path_distance = route_distance
                if goal_progress:
                    self.active_best_goal_distance = distance
                    # Kept for old status readers; this is now explicitly the
                    # physical distance to the fixed mission endpoint.
                    self.active_best_distance = distance
                # A collision-free Navfn route can initially lead away from a
                # frontier to reach a doorway.  During online SLAM, both the
                # route length and endpoint distance can rise while that is
                # happening.  Accept only *new maximum* odom displacement from
                # the action start as secondary progress: this permits a real
                # one-way detour, while a wall-follow loop cannot renew the
                # watchdog once it returns inside its previous excursion.
                odom_detour_progress = False
                odom_detour_distance = None
                if self.active_start_odom_xy is not None and self.pose_odom is not None:
                    odom_detour_distance = math.hypot(
                        float(self.pose_odom.x) - self.active_start_odom_xy[0],
                        float(self.pose_odom.y) - self.active_start_odom_xy[1],
                    )
                    if (
                        odom_detour_distance
                        > self.active_best_detour_odom_distance + self.progress_epsilon
                    ):
                        self.active_best_detour_odom_distance = odom_detour_distance
                        odom_detour_progress = True
                # A route can form an S-shaped detour: after the first bend
                # the base may return closer to its action start while still
                # entering new, physically useful space.  Coverage is local
                # to this action, so repeated wall-follow loops stop renewing
                # it once they revisit the same odom cells.
                odom_coverage_progress = self._entered_novel_active_odom_cell()
                if (
                    route_progress
                    or goal_progress
                    or odom_detour_progress
                    or odom_coverage_progress
                ):
                    self.active_progress_time = now
                    if route_progress and goal_progress:
                        self.active_last_progress_signal = "route_and_goal"
                    elif route_progress:
                        self.active_last_progress_signal = "route"
                    elif goal_progress:
                        self.active_last_progress_signal = "mission_goal"
                    elif odom_detour_progress:
                        self.active_last_progress_signal = "odom_detour"
                    else:
                        self.active_last_progress_signal = "odom_novel_coverage"
                # Map-aware path progress remains the primary evidence. The
                # odom detour signal above exists only for the bounded initial
                # excursion that an online-SLAM route may require.
                if self.active_last_robot_xy is None:
                    self.active_last_robot_xy = (robot_map[0], robot_map[1])
                elif math.hypot(
                        robot_map[0] - self.active_last_robot_xy[0],
                        robot_map[1] - self.active_last_robot_xy[1]) >= self.progress_epsilon:
                    self.active_last_robot_xy = (robot_map[0], robot_map[1])
                # A turn connector is a distinct execution phase.  During it
                # the correct progress variable is yaw error, not translational
                # path distance.  Do not abandon a valid frontier merely
                # because the robot spent the watchdog window rotating in
                # place at a doorway.
                turn_phase_active = (
                    self.active_route_kind == "frontier_turn_connector"
                    and not self.turn_connector_released
                )
                recovery_pending = (
                    self.recovery_pending_route_id == self.active_route_id
                )
                if turn_phase_active:
                    self.active_progress_time = now
                post_turn_goal_matches = (
                    self.active_turn_completed_route_id == self.active_route_id
                    and self.active_turn_completed_goal is not None
                    and math.hypot(
                        self.active_turn_completed_goal[0] - x,
                        self.active_turn_completed_goal[1] - y,
                    ) <= 0.10
                )
                post_turn_elapsed = (
                    0.0 if not post_turn_goal_matches else max(
                        0.0, now - self.active_turn_completed_wall
                    )
                )
                post_turn_translation = None
                if (
                    post_turn_goal_matches
                    and self.active_turn_completed_odom_xy is not None
                    and self.pose_odom is not None
                ):
                    post_turn_translation = math.hypot(
                        float(self.pose_odom.x)
                        - self.active_turn_completed_odom_xy[0],
                        float(self.pose_odom.y)
                        - self.active_turn_completed_odom_xy[1],
                    )
                    self.active_turn_completed_translation = max(
                        self.active_turn_completed_translation,
                        post_turn_translation,
                    )
                    if (
                        not self.active_turn_completed_launched
                        and self.active_turn_completed_translation
                        >= self.progress_epsilon
                    ):
                        # A map->odom correction can make the map-space goal
                        # distance increase while the base is correctly
                        # following a doorway detour. Odom launch is the
                        # invariant proof that the post-turn route is alive.
                        self.active_turn_completed_launched = True
                        self.active_progress_time = now
                        self.active_last_progress_signal = "post_turn_odom_launch"
                        self.publish_status(
                            "post_turn_progress_watchdog_released",
                            route_id=int(self.active_route_id),
                            goal=[round(float(x), 3), round(float(y), 3)],
                            odom_translation=round(
                                float(self.active_turn_completed_translation), 3
                            ),
                        )
                post_turn_stalled = (
                    not turn_phase_active
                    and self.post_turn_stall_timeout is not None
                    and post_turn_goal_matches
                    and not self.active_turn_completed_launched
                    and post_turn_elapsed >= self.post_turn_stall_timeout
                    and now - self.active_progress_time >= self.post_turn_stall_timeout
                )
                stalled = (
                    recovery_pending
                    or post_turn_stalled
                    or (
                        not turn_phase_active
                        and now - self.active_progress_time > self.stall_timeout
                    )
                )
                waypoint_distance = (
                    float("inf")
                    if self.active_last_waypoint_map is None
                    else math.hypot(
                        self.active_last_waypoint_map[0] - robot_map[0],
                        self.active_last_waypoint_map[1] - robot_map[1],
                    )
                )
                waypoint_reached = (
                    self.active_last_waypoint_map is None
                    or waypoint_distance <= self.waypoint_release_radius
                )
                # ``active_timeout`` is a guard against a route that never
                # makes progress; it must not cancel a long but healthy route.
                # The old unconditional wall-clock timeout discarded a
                # frontier after 120 s even when the robot was still moving
                # through a doorway toward it.  Treat the route as expired
                # only when its total budget is exceeded *and* it has also
                # been stalled for the normal stall window.
                expired = (
                    now - self.active_since > self.active_timeout
                    and stalled
                )
                # Prefetch the next branch before the current endpoint is
                # reached, but keep it private while a production endpoint
                # action is active.  Promoting it early used to overwrite this
                # node's active route while move_base still executed the old
                # action. A later recovery from the old endpoint then retired
                # the unrelated new route, causing goal churn and zero-speed
                # oscillation. The terminal callback promotes this validated
                # cache directly, so strict ownership costs only one callback
                # turn instead of another full frontier search.
                if distance <= self.prefetch_distance:
                    # Select a successor as a real route transition:
                    # ``robot -> active endpoint -> pending endpoint``.  The
                    # previous selector scored the pending route from the
                    # current robot pose, then discovered its sharp departure
                    # only in the bridge.  Rooting this BFS at the active
                    # endpoint makes its first tangent and distance represent
                    # the next branch itself.  Exact Navfn validation still
                    # starts at ``robot_map`` inside choose_valid_frontier.
                    transition_seed = self.nearest_seed(
                        frontier_free,
                        row,
                        col,
                        max(1, int(0.8 / message.info.resolution)),
                    )
                    transition_steps = (
                        None
                        if transition_seed is None
                        else self.bfs(frontier_free, transition_seed)
                    )
                    _, active_terminal_heading = self.route_headings(
                        message,
                        route_steps,
                        seed,
                        (row, col),
                        robot_map,
                    )
                    if (
                        transition_steps is not None
                        and active_terminal_heading is not None
                    ):
                        prefetch_steps = transition_steps
                        prefetch_seed = transition_seed
                        prefetch_heading = active_terminal_heading
                        prefetch_anchor = (x, y)
                        transition_basis = "active_endpoint_bfs_tangent"
                        score_from_transition = True
                    else:
                        # SLAM can temporarily reassociate an endpoint into a
                        # just-updated occupied cell. Preserve a valid
                        # robot-rooted selection rather than stalling map
                        # coverage; the status makes this degraded fallback
                        # visible in the experiment log.
                        prefetch_steps = route_steps
                        prefetch_seed = seed
                        prefetch_heading = robot_yaw_map
                        prefetch_anchor = robot_map
                        transition_basis = "robot_bfs_tangent_fallback"
                        score_from_transition = False
                        rospy.logwarn_throttle(
                            3.0,
                            "Global frontier successor transition topology unavailable; "
                            "using robot-rooted fallback route_id=%d",
                            self.active_route_id,
                        )
                    # This is lexicographic route-transition selection, not a
                    # larger heading penalty. Information gain decides only
                    # between candidates in the same executable envelope, so
                    # a high-gain right-angle branch cannot displace an
                    # available smooth or curve continuation.
                    transition_envelopes = (
                        [
                            (
                                self.successor_smooth_heading_limit,
                                "smooth_preferred",
                            ),
                            (
                                self.successor_curve_heading_limit,
                                "curve_preferred",
                            ),
                            (
                                self.heading_hard_limit,
                                "terminal_reorientation_preferred",
                            ),
                            (math.pi, "terminal_reorientation_unbounded"),
                        ]
                        if score_from_transition
                        else [
                            (
                                self.heading_hard_limit,
                                "robot_fallback_preferred",
                            ),
                            (math.pi, "robot_fallback_unbounded"),
                        ]
                    )
                    attempted_limits = set()
                    for heading_limit, transition_preference in transition_envelopes:
                        limit_key = round(float(heading_limit), 6)
                        if limit_key in attempted_limits:
                            continue
                        attempted_limits.add(limit_key)
                        selected = self.prefetch_next_frontier(
                            message,
                            prefetch_steps,
                            frontier,
                            unknown,
                            occupied,
                            now,
                            (x, y),
                            robot_map,
                            validation=validation,
                            heading_reference=prefetch_heading,
                            route_seed=prefetch_seed,
                            preferred_steps=strict_steps,
                            preferred_mask=strict_free,
                            route_anchor_xy=prefetch_anchor,
                            transition_basis=transition_basis,
                            score_path_from_steps=score_from_transition,
                            max_heading_delta=heading_limit,
                            allow_heading_fallback=False,
                            transition_preference=transition_preference,
                        )
                        if selected or self.frontier_validation_pending:
                            break
                # Early promotion is retained only for the legacy segmented
                # waypoint experiment. Production publishes one endpoint per
                # move_base transaction, so its active route must remain the
                # exact action owned by the bridge until the matching terminal
                # callback commits the prefetched successor.
                if (
                    (not self.mission_endpoint_only or self.persistent_execution)
                    and self.prefetched_frontier is not None
                    and distance <= self.early_handoff_distance
                ):
                    pending_row, pending_col, pending_x, pending_y = (
                        self.prefetched_frontier
                    )
                    route_continues = self.prefetched_route_continues_active(
                        route_steps,
                        seed,
                        (row, col),
                        (pending_row, pending_col),
                    )
                    # The persistent lease removes an actionlib boundary, but
                    # it does not make two divergent BFS routes one continuous
                    # geometric route. Replacing the input plan here would
                    # make Navfn start from the current base pose and cut away
                    # the final part of the active endpoint approach. Only a
                    # verified prefix continuation may update in place; every
                    # real branch waits for the endpoint lifecycle release.
                    if route_continues:
                        previous_x, previous_y = x, y
                        early_promoted = self.promote_prefetched_frontier(
                            message, route_steps, robot_map, now,
                            validation=validation,
                            preserve_route_id=route_continues,
                        )
                        if early_promoted is not None:
                            self.mark_frontier_completed(previous_x, previous_y)
                            rospy.loginfo(
                                "Global frontier streamed verified prefix route "
                                "old=(%.2f,%.2f) new=(%.2f,%.2f) distance=%.2fm",
                                previous_x,
                                previous_y,
                                early_promoted[2],
                                early_promoted[3],
                                distance,
                            )
                    else:
                        rospy.loginfo_throttle(
                            3.0,
                            "Global frontier defers prefetched branch until terminal: "
                            "active=(%.2f,%.2f) pending=(%.2f,%.2f) "
                            "because the pending route diverges before the endpoint",
                            x,
                            y,
                            pending_x,
                            pending_y,
                        )
                if early_promoted is not None:
                    promoted_row, promoted_col, promoted_x, promoted_y = early_promoted
                    x, y = promoted_x, promoted_y
                    promoted_cost_distance = self.candidate_costmap_distance(
                        validation, promoted_x, promoted_y
                    )
                    active_cell = (
                        promoted_row,
                        promoted_col,
                        promoted_x,
                        promoted_y,
                        promoted_cost_distance
                        if promoted_cost_distance is not None
                        else route_steps[promoted_row, promoted_col]
                        * message.info.resolution,
                        0.0,
                        0.0,
                        0.0,
                    )
                elif not stalled and not expired and (
                    distance > self.endpoint_terminal_wait_radius
                    or (
                        distance <= self.endpoint_terminal_wait_radius
                        and not waypoint_reached
                    )
                ):
                    # Dead-end detection: only when the robot is close enough
                    # that its lidar has genuinely resolved the boundary, and
                    # no unknown remains within a clear margin around the
                    # frontier cell, is the route a dead-end wall rather than
                    # a passage.  The margin is deliberately generous so a
                    # doorway or room entrance (unknown extends beyond it) is
                    # never mistaken for a dead-end, which would make the
                    # robot bounce between frontiers.
                    if (
                        distance < self.dead_end_check_distance
                        and not self._frontier_has_unknown(unknown, row, col)
                    ):
                        if (
                            (self.mission_endpoint_only or self.persistent_execution)
                            and not self.active_terminal_received
                        ):
                            # The lidar has resolved this endpoint as a wall,
                            # but move_base still owns the matching command.
                            # Preserve its identity until the matching terminal
                            # callback can atomically advance to the prefetched
                            # branch. A persistent action lease changes neither
                            # the physical endpoint nor this topology rule.
                            active_cell = (
                                row,
                                col,
                                x,
                                y,
                                route_steps[row, col] * message.info.resolution,
                                0.0,
                            )
                            rospy.loginfo_throttle(
                                2.0,
                                "Global frontier resolved endpoint geometry; "
                                "waiting for TEB terminal before retirement "
                                "route_id=%d",
                                self.active_route_id,
                            )
                        else:
                            self.mark_frontier_completed(x, y)
                            self.publish_status(
                            "route_invalidated",
                            reason="dead_end_resolved",
                            goal=[round(float(x), 3), round(float(y), 3)],
                            distance=round(float(distance), 3),
                            path_distance=round(float(route_distance), 3),
                            best_path_distance=(
                                None if self.active_best_path_distance is None
                                else round(float(self.active_best_path_distance), 3)
                            ),
                            best_goal_distance=(
                                None if self.active_best_goal_distance is None
                                else round(float(self.active_best_goal_distance), 3)
                            ),
                            last_progress_signal=self.active_last_progress_signal,
                            )
                            rospy.loginfo(
                            "Global frontier resolved dead-end map=(%.2f,%.2f) "
                            "at distance=%.2fm; abandoning wall approach",
                            x, y, distance,
                            )
                            self.active_frontier = None
                            self.active_best_distance = None
                            self.active_best_goal_distance = None
                            self.active_last_robot_xy = None
                            self.active_last_waypoint_map = None
                            self.active_route_kind = "frontier_endpoint"
                            self.active_terminal_received = False
                            self.turn_connector_released = True
                            self.last_status_command_map = None
                            self.last_status_command_yaw = None
                            self.last_status_mission_map = None
                            self.active_best_path_distance = None
                            self.active_last_progress_signal = "none"
                            self.active_unreachable_since = None
                            if self.prefetched_frontier is None:
                                self.prefetched_goal_map = None
                    else:
                        active_cell = (row, col, x, y, route_steps[row, col] * message.info.resolution, 0.0)
                        if (
                            distance <= self.endpoint_terminal_wait_radius
                            and not waypoint_reached
                        ):
                            rospy.loginfo_throttle(
                                3.0,
                                "Global frontier holding completed frontier until short "
                                "waypoint is reached distance=%.2fm waypoint_distance=%.2fm "
                                "release_radius=%.2fm",
                                distance,
                                waypoint_distance,
                                self.waypoint_release_radius,
                            )
                else:
                    # Reaching the map-space endpoint is not enough to retire
                    # a persistent endpoint route.  The bridge still owns the
                    # physical MoveBase action and must emit its identity-
                    # matched terminal before this node promotes the cached
                    # successor.  Keep the release decision explicit: the
                    # previous unconditional cleanup below erased the active
                    # route immediately after this hold branch, so the next
                    # SLAM cycle selected a new (often reverse) frontier while
                    # TEB was still decelerating for the old endpoint.
                    release_active_route = False
                    if (
                        distance <= self.endpoint_terminal_wait_radius
                        and waypoint_reached
                        and (
                            not (
                                self.mission_endpoint_only
                                or self.persistent_execution
                            )
                            or self.active_terminal_received
                        )
                    ):
                        self.mark_frontier_completed(x, y)
                        release_active_route = True
                    elif (
                        (self.mission_endpoint_only or self.persistent_execution)
                        and distance <= self.endpoint_terminal_wait_radius
                        and waypoint_reached
                    ):
                        # Map coverage and distance are not execution success.
                        # Keep the endpoint while TEB reaches its own terminal
                        # speed envelope; the bridge then emits an identity-
                        # matched terminal that permits successor selection.
                        active_cell = (
                            row,
                            col,
                            x,
                            y,
                            route_steps[row, col] * message.info.resolution,
                            0.0,
                        )
                        rospy.loginfo_throttle(
                            2.0,
                            "Global frontier reached endpoint geometry; "
                            "waiting for matching TEB terminal route_id=%d "
                            "distance=%.2fm terminal_wait_radius=%.2fm",
                            self.active_route_id,
                            distance,
                            self.endpoint_terminal_wait_radius,
                        )
                    else:
                        # A policy stall or a timeout is not proof that this
                        # frontier was inspected. Defer it briefly while other
                        # map branches are explored, then allow a retry.
                        if recovery_pending:
                            reason = (
                                self.recovery_pending_reason
                                or "move_base_terminal_failure"
                            )
                        elif post_turn_stalled:
                            reason = "post_turn_no_progress"
                        else:
                            reason = "stall" if stalled else "active_timeout"
                        self.rejected_frontiers.append((now, x, y))
                        rospy.logwarn(
                            "Global frontier deferred map=(%.2f,%.2f): %s "
                            "distance=%.2fm elapsed=%.1fs; suppress for %.0fs",
                            x, y, reason, distance, now - self.active_since,
                            self.rejected_timeout,
                        )
                        # A route invalidation is a mission lifecycle event,
                        # not just an internal frontier bookkeeping update.
                        # Consumers must release the old move_base action;
                        # otherwise it keeps retrying recovery behaviors even
                        # though this explorer has already moved on.
                        self.publish_status(
                            "route_invalidated",
                            reason=reason,
                            goal=[round(float(x), 3), round(float(y), 3)],
                            distance=round(float(distance), 3),
                            elapsed=round(float(now - self.active_since), 3),
                            path_distance=round(float(route_distance), 3),
                            best_path_distance=(
                                None
                                if self.active_best_path_distance is None
                                else round(float(self.active_best_path_distance), 3)
                            ),
                            best_goal_distance=(
                                None
                                if self.active_best_goal_distance is None
                                else round(float(self.active_best_goal_distance), 3)
                            ),
                            odom_detour_distance=round(
                                float(self.active_best_detour_odom_distance), 3
                            ),
                            odom_novel_cells=len(self.active_visited_odom_cells),
                            last_progress_signal=self.active_last_progress_signal,
                            post_turn_elapsed=(
                                None if not post_turn_goal_matches
                                else round(float(post_turn_elapsed), 3)
                            ),
                            post_turn_odom_translation=(
                                None if post_turn_translation is None
                                else round(
                                    float(self.active_turn_completed_translation), 3
                                )
                            ),
                            post_turn_launched=bool(
                                self.active_turn_completed_launched
                            ),
                            recovery_behavior=(
                                self.recovery_pending_behavior
                                if recovery_pending else None
                            ),
                            recovery_reason=(
                                self.recovery_pending_reason
                                if recovery_pending else None
                            ),
                        )
                        release_active_route = True
                    if release_active_route:
                        self.active_frontier = None
                        self.active_best_distance = None
                        self.active_best_goal_distance = None
                        self.active_last_robot_xy = None
                        self.active_start_odom_xy = None
                        self.active_best_detour_odom_distance = 0.0
                        self.active_visited_odom_cells.clear()
                        self.active_last_waypoint_map = None
                        self.active_route_kind = "frontier_endpoint"
                        self.active_terminal_received = False
                        self.recovery_pending_route_id = 0
                        self.recovery_pending_behavior = ""
                        self.recovery_pending_reason = ""
                        self.turn_connector_released = True
                        self.last_status_command_map = None
                        self.last_status_command_yaw = None
                        self.last_status_mission_map = None
                        self.active_best_path_distance = None
                        self.active_last_progress_signal = "none"
                        self.active_unreachable_since = None
                        if self.prefetched_frontier is None:
                            self.prefetched_goal_map = None
            else:
                if self.active_unreachable_since is None:
                    self.active_unreachable_since = now
                held_for = now - self.active_unreachable_since
                waypoint_distance = (
                    float("inf")
                    if self.active_last_waypoint_map is None
                    else math.hypot(
                        self.active_last_waypoint_map[0] - robot_map[0],
                        self.active_last_waypoint_map[1] - robot_map[1],
                    )
                )
                if (
                    self.active_last_waypoint_map is not None
                    and held_for <= self.unreachable_grace
                    and waypoint_distance > 0.8
                ):
                    held_waypoint_map = self.active_last_waypoint_map
                    active_cell = (row, col, x, y, 0.0, 0.0)
                    rospy.logwarn_throttle(
                        3.0,
                        "Global frontier temporarily disconnected map=(%.2f,%.2f); "
                        "holding last safe waypoint for %.1fs/%.1fs",
                        x, y, held_for, self.unreachable_grace,
                    )
                else:
                    rospy.logwarn(
                        "Global frontier abandoned map=(%.2f,%.2f): disconnected for %.1fs",
                        x, y, held_for,
                    )
                    self.publish_status(
                        "route_invalidated",
                        reason="disconnected",
                        goal=[round(float(x), 3), round(float(y), 3)],
                        distance=round(
                            float(math.hypot(x - robot_map[0], y - robot_map[1])),
                            3,
                        ),
                        elapsed=round(float(held_for), 3),
                        path_distance=None,
                        best_path_distance=(
                            None
                            if self.active_best_path_distance is None
                            else round(float(self.active_best_path_distance), 3)
                        ),
                        best_goal_distance=(
                            None
                            if self.active_best_goal_distance is None
                            else round(float(self.active_best_goal_distance), 3)
                        ),
                        last_progress_signal=self.active_last_progress_signal,
                    )
                    self.active_frontier = None
                    self.active_best_distance = None
                    self.active_best_goal_distance = None
                    self.active_last_robot_xy = None
                    self.active_start_odom_xy = None
                    self.active_best_detour_odom_distance = 0.0
                    self.active_visited_odom_cells.clear()
                    self.active_last_waypoint_map = None
                    self.active_route_kind = "frontier_endpoint"
                    self.active_terminal_received = False
                    self.turn_connector_released = True
                    self.last_status_command_map = None
                    self.last_status_command_yaw = None
                    self.last_status_mission_map = None
                    self.active_best_path_distance = None
                    self.active_last_progress_signal = "none"
                    self.active_unreachable_since = None
                    self.prefetched_frontier = None
                    self.prefetched_goal_map = None
        selection_mode = "prefetched_successor"
        if active_cell is None:
            promoted = self.promote_prefetched_frontier(
                message, route_steps, robot_map, now, validation=validation
            )
            if promoted is not None:
                promoted_row, promoted_col, promoted_x, promoted_y = promoted
                promoted_cost_distance = self.candidate_costmap_distance(
                    validation, promoted_x, promoted_y
                )
                active_cell = (
                    promoted_row,
                    promoted_col,
                    promoted_x,
                    promoted_y,
                    promoted_cost_distance
                    if promoted_cost_distance is not None
                    else route_steps[promoted_row, promoted_col]
                    * message.info.resolution,
                    0.0,
                    0.0,
                    0.0,
                )
            else:
                active_cell = self.choose_valid_frontier(
                    message,
                    route_steps,
                    frontier,
                    unknown,
                    occupied,
                    now,
                    robot_map,
                    validation=validation,
                    heading_reference=robot_yaw_map,
                    # First-route selection has the same continuity contract
                    # as prefetch. If a forward connected branch exists, do
                    # not let raw information score choose an immediate
                    # U-turn; choose_valid_frontier automatically retries
                    # without this bound when a turn is truly unavoidable.
                    max_heading_delta=self.heading_hard_limit,
                    route_seed=seed,
                    preferred_steps=strict_steps,
                    preferred_mask=strict_free,
                    allow_observation_recovery=(
                        self.navfn_observation_recovery_enabled
                    ),
                    semantic_hint=(
                        self.pending_semantic_hint_map
                        if self.pending_replan_request_id > 0
                        else None
                    ),
                )
                selection_mode = self.last_frontier_selection_mode
            if active_cell is None:
                if (
                    self.frontier_validation_budget_exhausted
                    or self.frontier_validation_pending
                ):
                    self.publish_status(
                        "frontier_validation_pending",
                        reason=(
                            "navfn_validation_budget_exhausted"
                            if self.frontier_validation_budget_exhausted
                            else "navfn_validation_unavailable"
                        ),
                        validation_state=self.navfn_last_validation_state,
                    )
                    rospy.logwarn_throttle(
                        3.0,
                        "Global frontier deferred exhaustion: Navfn validation "
                        "is pending state=%s budget_exhausted=%s",
                        self.navfn_last_validation_state,
                        self.frontier_validation_budget_exhausted,
                    )
                    return
                if not self.frontier_exhausted:
                    self.frontier_exhausted = True
                    self.publish_status(
                        "frontier_exhausted",
                        reason="no_safe_reachable_frontier",
                        route_clearance=round(float(self.clearance), 3),
                        frontier_clearance=round(float(self.frontier_clearance), 3),
                    )
                    rospy.logwarn(
                        "Global frontier exhausted: no safe reachable boundary; "
                        "releasing the current execution lease"
                    )
                rospy.logwarn_throttle(
                    3.0,
                    "Global frontier found no unknown boundary with a safe "
                    "approach (route_clearance=%.2fm frontier_clearance=%.2fm)",
                    self.clearance,
                    self.frontier_clearance,
                )
                return
            row, col, x, y, path_distance, information, structure, score = active_cell
            if self.active_frontier is None:
                self.frontier_exhausted = False
                self.active_frontier = (row, col, x, y)
                self.active_route_id += 1
                self.active_transition_kind = "initial"
                self.active_predecessor_route_id = 0
                self.active_transition_distance = None
                self.recovery_pending_route_id = 0
                self.recovery_pending_behavior = ""
                self.recovery_pending_reason = ""
                self.active_since = now
                self.active_best_distance = math.hypot(
                    x - robot_map[0], y - robot_map[1]
                )
                self.active_best_goal_distance = self.active_best_distance
                self.active_best_path_distance = float(path_distance)
                self.active_progress_time = now
                self.active_last_progress_signal = "route_and_goal_initialized"
                self.active_last_robot_xy = (robot_map[0], robot_map[1])
                self.active_start_odom_xy = (
                    float(self.pose_odom.x), float(self.pose_odom.y)
                )
                self.active_best_detour_odom_distance = 0.0
                self._reset_active_odom_coverage()
                self.active_unreachable_since = None
                self.active_last_waypoint_map = None
                self.active_route_kind = "frontier_endpoint"
                self.active_terminal_received = False
                self.turn_connector_released = True
                self.last_status_command_map = None
                self.last_status_command_yaw = None
                self.last_status_mission_map = None
                rospy.loginfo(
                    "Global frontier selected mode=%s map=(%.2f,%.2f) path=%.2fm "
                    "information=%.0f structure=%.0f score=%.2f "
                    "route_heading_delta=%.1fdeg",
                    selection_mode,
                    x,
                    y,
                    path_distance,
                    information,
                    structure,
                    score,
                    math.degrees(self._candidate_route_heading_delta(
                        message, route_steps, seed, row, col, robot_map, robot_yaw_map
                    ))
                    if self._candidate_route_heading_delta(
                        message, route_steps, seed, row, col, robot_map, robot_yaw_map
                    ) is not None
                    else float("nan"),
                )
                self.publish_status(
                    "route_selected",
                    selection_mode=selection_mode,
                    goal=[round(float(x), 3), round(float(y), 3)],
                    path_distance=round(float(path_distance), 3),
                    goal_distance=round(float(self.active_best_goal_distance), 3),
                    information=round(float(information), 3),
                    structure=round(float(structure), 3),
                    route_heading_delta_deg=(
                        None
                        if self._candidate_route_heading_delta(
                            message, route_steps, seed, row, col, robot_map,
                            robot_yaw_map,
                        ) is None
                        else round(math.degrees(self._candidate_route_heading_delta(
                            message, route_steps, seed, row, col, robot_map,
                            robot_yaw_map,
                        )), 2)
                    ),
                    semantic_hint_map=(
                        None if self.pending_semantic_hint_map is None
                        else [
                            round(float(self.pending_semantic_hint_map[0]), 3),
                            round(float(self.pending_semantic_hint_map[1]), 3),
                        ]
                    ),
                )
                if selection_mode == "navfn_observation_recovery":
                    self.publish_status(
                        "navfn_observation_recovery_selected",
                        goal=[round(float(x), 3), round(float(y), 3)],
                        path_distance=round(float(path_distance), 3),
                        strict_clearance=round(float(self.clearance), 3),
                        observation_clearance=round(
                            float(self.frontier_clearance), 3
                        ),
                        validation_state=self.navfn_last_validation_state,
                    )
                if self.pending_replan_request_id > 0:
                    self.publish_status(
                        "replan_ready",
                        replan_request_id=self.pending_replan_request_id,
                        reason=self.pending_replan_reason,
                        goal=[round(float(x), 3), round(float(y), 3)],
                        path_distance=round(float(path_distance), 3),
                        semantic_hint_map=(
                            None if self.pending_semantic_hint_map is None
                            else [
                                round(float(self.pending_semantic_hint_map[0]), 3),
                                round(float(self.pending_semantic_hint_map[1]), 3),
                            ]
                        ),
                    )
                    rospy.loginfo(
                        "Global frontier replan ready id=%d goal=(%.2f,%.2f)",
                        self.pending_replan_request_id,
                        x,
                        y,
                    )
                    self.pending_replan_request_id = 0
                    self.pending_replan_reason = ""
                    self.pending_semantic_hint_map = None
        row, col = active_cell[0], active_cell[1]
        # After an early promotion this is the new active branch. Otherwise
        # the committed endpoint remains stable until a normal terminal or a
        # validated recovery path changes it.
        route_kind = self.active_route_kind
        command_yaw = self.active_last_waypoint_yaw
        if held_waypoint_map is not None:
            map_xy = held_waypoint_map
        else:
            # Keep the selected mission endpoint stable while its action is
            # active.  A same-position turn connector is the one exception:
            # it is an explicit motion-geometry state used to establish the
            # route tangent before the forward-only base enters a branch.
            previous_waypoint_distance = (
                float("inf") if self.active_last_waypoint_map is None else math.hypot(
                    self.active_last_waypoint_map[0] - robot_map[0],
                    self.active_last_waypoint_map[1] - robot_map[1],
                )
            )
            # The supervisor owns the yaw lock. Until its completed event,
            # a changing map->odom transform cannot resurrect a new connector.
            turn_still_pending = (
                self.active_route_kind == "frontier_turn_connector"
                and not self.turn_connector_released
            )
            if (
                self.active_last_waypoint_map is not None
                and (
                    # The production endpoint contract is one atomic
                    # move_base action. SLAM may reassociate a nearby safe
                    # grid cell by 10-20 cm as scans arrive; changing the
                    # command to that cell near arrival creates a second
                    # artificial terminal before the real successor can run.
                    # Keep the execution pose frozen until its matching
                    # terminal event. The fresh map is still used to validate
                    # and choose the following branch.
                    (
                        self.mission_endpoint_only
                        and not (
                            self.active_route_kind == "frontier_turn_connector"
                            and self.turn_connector_released
                        )
                    )
                    or
                    previous_waypoint_distance > self.waypoint_release_radius
                    or turn_still_pending
                )
            ):
                map_xy = self.active_last_waypoint_map
                command_yaw = self.active_last_waypoint_yaw
                rospy.loginfo_throttle(
                    3.0,
                    "Global frontier holding route command kind=%s "
                    "(%.2f,%.2f) distance=%.2fm release_radius=%.2fm "
                    "turn_pending=%s",
                    self.active_route_kind,
                    map_xy[0], map_xy[1], previous_waypoint_distance,
                    self.waypoint_release_radius,
                    turn_still_pending,
                )
            else:
                remaining_path = float(route_steps[row, col]) * float(message.info.resolution)
                command_row, command_col = row, col
                route_kind = "frontier_endpoint"
                if not self.mission_endpoint_only and remaining_path > self.route_segment_distance:
                    horizon_cells = max(
                        1,
                        int(math.ceil(
                            self.route_segment_distance / message.info.resolution
                        )),
                    )
                    command_cell = self.waypoint_on_path(
                        route_steps, row, col, horizon_cells
                    )
                    if command_cell is not None:
                        command_row, command_col = command_cell
                        route_kind = "frontier_connector"
                initial_heading, terminal_heading = self.route_headings(
                    message,
                    route_steps,
                    seed,
                    (command_row, command_col),
                    robot_map,
                )
                turn_delta = (
                    None
                    if initial_heading is None or robot_yaw_map is None
                    else abs(self._angle_delta(initial_heading, robot_yaw_map))
                )
                # A sharp first tangent is a property of the global route, not
                # a reason to manufacture another action.  In production the
                # frontier is sent as one endpoint transaction; Navfn exposes
                # the complete path and TEB's orientation-aware optimizer
                # turns onto it while preserving the same timed elastic band.
                # Creating an intermediate "transition" goal here would make
                # move_base report SUCCEEDED halfway through a valid route and
                # necessarily insert a stop before the endpoint action.  The
                # old rolling-horizon experiment may still use a short
                # frontier_connector, but it is explicitly outside the
                # production mission_endpoint_only contract.
                command_yaw = terminal_heading
                map_x, map_y = self.cell_xy(message, command_row, command_col)
                map_xy = (float(map_x), float(map_y))
                # ``x, y`` are the physical mission endpoint selected when
                # this route transaction began. The active grid row/column may
                # be reassociated by online SLAM, but a production endpoint
                # action must never follow that cell by a few centimetres.
                if self.mission_endpoint_only:
                    map_xy = (float(x), float(y))
                if (
                    self.turn_execution_mode == "legacy_connector"
                    and
                    self.mission_endpoint_only
                    and turn_delta is not None
                    and turn_delta >= self.explicit_turn_connector_threshold
                    and remaining_path > self.waypoint_release_radius
                ):
                    # Preserve the endpoint transaction identity, but give
                    # the execution adapter one atomic yaw phase before TEB
                    # receives a long route beginning behind the vehicle.
                    # This is intentionally based on Navfn's first tangent,
                    # not the endpoint bearing, so doorway detours are safe.
                    route_kind = "frontier_turn_connector"
                    connector_steps = max(
                        1,
                        int(math.ceil(
                            self.turn_connector_distance
                            / max(float(message.info.resolution), 1e-6)
                        )),
                    )
                    connector_cell = self.waypoint_on_path(
                        route_steps, row, col, connector_steps
                    )
                    if connector_cell is not None:
                        connector_x, connector_y = self.cell_xy(
                            message, connector_cell[0], connector_cell[1]
                        )
                        map_xy = (float(connector_x), float(connector_y))
                    else:
                        # A path shorter than the connector horizon is still
                        # valid; retain the endpoint rather than inventing a
                        # point outside the route-safe mask.
                        map_xy = (float(x), float(y))
                    command_yaw = initial_heading
                    self.turn_connector_released = False
                    rospy.loginfo(
                        "Global frontier inserted explicit turn connector "
                        "before reverse-facing branch: endpoint=(%.2f,%.2f) "
                        "initial_heading=%.1fdeg terminal_heading=%.1fdeg "
                        "turn_delta=%.1fdeg path=%.2fm",
                        map_xy[0],
                        map_xy[1],
                        math.degrees(initial_heading),
                        math.degrees(terminal_heading)
                        if terminal_heading is not None else float("nan"),
                        math.degrees(turn_delta),
                        remaining_path,
                    )
                self.active_last_waypoint_map = map_xy
                self.active_last_waypoint_yaw = command_yaw
                self.active_route_kind = route_kind
                if route_kind != "frontier_turn_connector":
                    self.turn_connector_released = True
                rospy.loginfo(
                    "Global frontier route command kind=%s command=(%.2f,%.2f) "
                    "mission=(%.2f,%.2f) remaining=%.2fm horizon=%.2fm "
                    "yaw=%s",
                    route_kind,
                    map_xy[0],
                    map_xy[1],
                    x,
                    y,
                    remaining_path,
                    self.route_segment_distance,
                    "none" if command_yaw is None else "%.1fdeg" % math.degrees(command_yaw),
                )
        command_changed = (
            self.last_status_command_map is None
            or math.hypot(
                map_xy[0] - self.last_status_command_map[0],
                map_xy[1] - self.last_status_command_map[1],
            ) > 0.05
            or self.active_route_kind != route_kind
            or (
                command_yaw is not None
                and (
                    self.last_status_command_yaw is None
                    or abs(self._angle_delta(command_yaw, self.last_status_command_yaw))
                    > 0.08
                )
            )
            or self.last_status_mission_map is None
            or math.hypot(
                x - self.last_status_mission_map[0],
                y - self.last_status_mission_map[1],
            ) > 0.20
        )
        if command_changed:
            self.publish_status(
                "route_command",
                route_kind=self.active_route_kind,
                route_id=int(self.active_route_id),
                command_goal=[round(float(map_xy[0]), 3), round(float(map_xy[1]), 3)],
                command_yaw=(
                    None if command_yaw is None else round(float(command_yaw), 3)
                ),
                mission_goal=[round(float(x), 3), round(float(y), 3)],
                path_remaining=round(
                    float(route_steps[row, col]) * float(message.info.resolution), 3
                ),
            )
            self.last_status_command_map = (float(map_xy[0]), float(map_xy[1]))
            self.last_status_command_yaw = command_yaw
            self.last_status_mission_map = (float(x), float(y))
        # Both position and tangent stay in the SLAM frame.  move_base's
        # global frame is ``map`` and the turn supervisor converts this yaw to
        # odom once when it starts an explicit in-place turn.
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = message.header.frame_id or "map"
        goal.pose.position.x, goal.pose.position.y = map_xy
        if command_yaw is None:
            goal.pose.orientation.w = 1.0
        else:
            goal.pose.orientation.z = math.sin(0.5 * command_yaw)
            goal.pose.orientation.w = math.cos(0.5 * command_yaw)
        self.publish_route_command(
            goal.header.frame_id,
            map_xy[0],
            map_xy[1],
            command_yaw,
            self.active_route_kind,
        )
        self.publisher.publish(goal)


if __name__ == "__main__":
    GlobalFrontierExplorer()
    rospy.spin()
