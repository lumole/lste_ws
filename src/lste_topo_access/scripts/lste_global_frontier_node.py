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
import time
import traceback

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Pose2D, PoseStamped
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetPlan, GetPlanRequest
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
        # ``clearance`` is the route clearance, not merely a preference.  It
        # must cover the robot footprint and TEB's minimum obstacle distance;
        # otherwise this node can find a connected BFS route that Navfn/TEB
        # correctly rejects.  The production launch passes 0.52 m
        # (0.30 m footprint radius + 0.22 m TEB clearance).
        self.clearance = max(0.05, float(gp("~clearance", 0.52)))
        # ``fallback_clearance`` is retained as a launch/API name for
        # compatibility, but it is no longer a route relaxation.  It defines
        # how close a known-free cell may be to an unknown boundary for that
        # boundary to count as a frontier candidate.  The published goal is
        # always selected from the stricter route-safe mask above.
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
        # SLAM can temporarily disconnect an active frontier while integrating
        # a scan. Keep the last safe endpoint briefly instead of switching
        # branches on every map update.
        self.unreachable_grace = max(
            0.0, float(gp("~unreachable_grace", 20.0))
        )
        self.progress_epsilon = max(0.02, float(gp("~progress_epsilon", 0.12)))
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
        # but a branch that is directly behind the robot should not win by a
        # small score margin and force TEB to brake and make an abrupt U-turn.
        # This is a soft continuity cost rather than a hard direction filter:
        # when a turn is the only connected route, the frontier remains valid.
        self.heading_weight = max(0.0, float(gp("~heading_weight", 2.5)))
        self.heading_hard_limit = math.radians(max(
            0.0, float(gp("~heading_hard_limit_deg", 115.0))
        ))
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
        self.navfn_make_plan_tolerance = max(
            0.0, float(gp("~navfn_make_plan_tolerance", 0.20))
        )
        self.navfn_service = rospy.ServiceProxy(
            self.navfn_make_plan_service, GetPlan
        )
        self.pose_odom = None
        self.task_done = False
        self.active_frontier = None
        self.active_since = 0.0
        self.active_best_distance = None
        # Progress is measured on the currently validated route, not by the
        # straight-line distance to the information boundary.  A doorway or
        # wall-follow detour can increase Euclidean distance while making real
        # progress toward the frontier.
        self.active_best_path_distance = None
        self.active_progress_time = 0.0
        self.active_last_robot_xy = None
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
        # A turn connector is a committed execution phase. Keep its pose and
        # tangent frozen until the supervisor reports completion.
        self.turn_connector_released = True
        self.turn_supervisor_state = "UNKNOWN"
        self.last_status_command_map = None
        self.last_status_command_yaw = None
        self.last_status_mission_map = None
        self.prefetched_frontier = None
        self.prefetched_goal_map = None
        # A replan request is issued when target execution relinquishes
        # exploration ownership. The following endpoint must be selected from
        # the robot's then-current map pose, not recovered from this cache.
        self.pending_replan_request_id = 0
        self.pending_replan_reason = ""
        self.last_planning_wall = 0.0
        self.rejected_frontiers = collections.deque(maxlen=24)
        self.completed_frontiers = collections.deque(maxlen=self.completed_limit)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.publisher = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1, latch=True)
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
        rospy.Subscriber(
            self.replan_request_topic, String, self.on_replan_request, queue_size=10
        )
        rospy.Subscriber(self.task_done_topic, Bool, self.on_task_done, queue_size=1)
        rospy.Subscriber(
            self.turn_status_topic,
            String,
            self.on_turn_status,
            queue_size=10,
        )
        rospy.Timer(rospy.Duration(1.0), self.on_timer)
        rospy.loginfo(
            "Global frontier explorer started: map=%s costmap=%s goal=%s "
            "heading_weight=%.2f heading_hard_limit=%.1fdeg "
            "route_segment=%.2fm mission_endpoint_only=%s planning_period=%.2fs",
            self.map_topic,
            self.costmap_topic,
            self.goal_topic,
            self.heading_weight,
            math.degrees(self.heading_hard_limit),
            self.route_segment_distance,
            self.mission_endpoint_only,
            self.planning_period,
        )

    def publish_status(self, event, **fields):
        """Publish the frontier planner's route lifecycle to its consumers."""
        payload = {
            "event": str(event),
            "active": self.active_frontier is not None,
            "pending": self.prefetched_frontier is not None,
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

    def on_pose(self, message):
        self.pose_odom = message

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
            self.turn_connector_released = True
            self.last_status_command_map = None
            self.last_status_command_yaw = None
            self.last_status_mission_map = None
            self.active_best_path_distance = None
            self.active_unreachable_since = None
            self.prefetched_frontier = None
            self.prefetched_goal_map = None
            rospy.loginfo("Global frontier paused: task_done=true")
        else:
            rospy.loginfo("Global frontier resumed: task_done=false")

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
        self.pending_replan_request_id = request_id
        self.pending_replan_reason = str(payload.get("reason", "unknown"))
        self.active_frontier = None
        self.active_since = 0.0
        self.active_best_distance = None
        self.active_best_path_distance = None
        self.active_progress_time = 0.0
        self.active_last_robot_xy = None
        self.active_unreachable_since = None
        self.active_last_waypoint_map = None
        self.active_last_waypoint_yaw = None
        self.active_route_kind = "frontier_endpoint"
        self.turn_connector_released = True
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
        )
        rospy.loginfo(
            "Global frontier accepted replan request id=%d reason=%s old=%s",
            request_id,
            self.pending_replan_reason,
            old_goal,
        )

    def on_turn_status(self, message):
        """Receive the execution adapter's atomic-turn state."""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if isinstance(payload, dict):
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
    def nearest_safe_approach(steps, row, col, max_cells):
        """Find a route-safe cell close to a looser frontier cell.

        Frontier cells are intentionally detected with a smaller clearance so
        that a narrow doorway is still visible as an exploration opportunity.
        The action target must nevertheless come from ``steps`` (the strict
        route mask).  This is the same separation used by frontier explorers
        that approach an information boundary instead of commanding the
        boundary cell itself.
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

    def navfn_goal_reachable(self, robot_map, goal_xy, frame_id):
        """Validate one candidate against move_base's actual Navfn plugin.

        The Python occupancy/costmap masks are deliberately conservative, but
        they cannot reproduce every Navfn detail (unknown handling, planner
        tolerance, and the live costmap reset state).  Validate only the few
        candidates that would become mission goals, never every frontier cell.

        ``None`` means the service is not available yet; startup must continue
        in that case and the normal costmap checks remain in force. ``False``
        is reserved for a live service response with an empty plan.
        """
        if not self.navfn_plan_validation:
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
        request.tolerance = self.navfn_make_plan_tolerance
        try:
            response = self.navfn_service(request)
        except (rospy.ServiceException, rospy.ROSException) as exc:
            rospy.logwarn_throttle(
                5.0,
                "Global frontier Navfn validation failed for goal=(%.2f,%.2f): %s",
                goal_xy[0], goal_xy[1], exc,
            )
            return None
        reachable = bool(response.plan.poses)
        if not reachable:
            rospy.logwarn(
                "Global frontier rejected Navfn-empty candidate goal=(%.2f,%.2f) "
                "robot=(%.2f,%.2f)",
                goal_xy[0], goal_xy[1], robot_map[0], robot_map[1],
            )
        return reachable

    def choose_valid_frontier(
        self, message, steps, frontier, unknown, occupied, now,
        robot_map, validation=None, excluded=None, heading_reference=None,
        max_heading_delta=None,
    ):
        """Choose a score-best candidate whose live Navfn route is non-empty."""
        # A failed service validation is remembered as a temporary rejection,
        # so the next score-best branch is tried instead of repeating the same
        # empty Navfn route on every timer tick.
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
                max_heading_delta=max_heading_delta,
            )
            if candidate is None and max_heading_delta is not None:
                # Direction continuity is a preference for prefetching, not a
                # deadlock condition. If every remaining frontier is behind
                # the robot, retry once without the hard bound.
                rospy.loginfo(
                    "Global frontier has no candidate within heading limit "
                    "%.1fdeg; allowing a necessary turn",
                    math.degrees(max_heading_delta),
                )
                max_heading_delta = None
                continue
            if candidate is None:
                return None
            reachable = self.navfn_goal_reachable(
                robot_map,
                (candidate[2], candidate[3]),
                message.header.frame_id or "map",
            )
            if reachable is not False:
                return candidate
            self.rejected_frontiers.append((now, candidate[2], candidate[3]))
            rospy.logwarn(
                "Global frontier deferred Navfn-empty candidate map=(%.2f,%.2f) "
                "for %.0fs",
                candidate[2], candidate[3], self.rejected_timeout,
            )
        return None

    def choose_frontier(
        self, message, steps, frontier, unknown, occupied, now, excluded=None,
        validation=None, robot_map=None, heading_reference=None,
        max_heading_delta=None,
    ):
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
            approach = self.nearest_safe_approach(steps, row, col, approach_cells)
            if approach is None:
                continue
            target_row, target_col = approach
            if steps[target_row, target_col] < minimum_steps:
                continue
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
            information = self.frontier_information(unknown, row, col)
            structure = self.frontier_structure(occupied, row, col)
            heading_delta = self.heading_delta(
                x,
                y,
                robot_map,
                heading_reference,
            )
            heading_penalty = (
                0.0
                if heading_delta is None
                else self.heading_weight * (1.0 - math.cos(heading_delta))
            )
            if (
                max_heading_delta is not None
                and heading_delta is not None
                and heading_delta > max_heading_delta
            ):
                continue
            # Prefer a large unseen boundary, but do not repeatedly choose a
            # remote branch while closer useful coverage remains. Structure is
            # deliberately a bonus rather than a hard constraint: an open map
            # still remains explorable when no doorway-like frontier exists.
            score = (
                information * 0.035
                + structure * self.structure_weight
                - path_distance * 0.12
                - heading_penalty
            )
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
            if score > fallback_score:
                fallback_score = score
                fallback_best = candidate
            if structure >= self.min_structure_cells and score > structured_score:
                structured_score = score
                structured_best = candidate
        return structured_best or fallback_best

    def prefetch_next_frontier(
        self, message, steps, frontier, unknown, occupied, now, active_xy,
        robot_map, validation=None, heading_reference=None,
    ):
        """Select one pending branch while the current frontier is still active."""
        if self.prefetched_frontier is not None or self.active_frontier is None:
            return
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
            max_heading_delta=self.heading_hard_limit,
        )
        if candidate is None:
            return
        row, col, x, y, path_distance, information, structure, score = candidate
        self.prefetched_frontier = (row, col, x, y)
        self.prefetched_goal_map = (float(x), float(y))
        rospy.loginfo(
            "Global frontier prefetched next branch active=(%.2f,%.2f) "
            "pending=(%.2f,%.2f) path=%.2fm information=%.0f structure=%.0f "
            "score=%.2f heading_delta=%.1fdeg",
            active_xy[0],
            active_xy[1],
            x,
            y,
            path_distance,
            information,
            structure,
            score,
            math.degrees(self.heading_delta(x, y, robot_map, heading_reference))
            if self.heading_delta(x, y, robot_map, heading_reference) is not None
            else float("nan"),
        )

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
        self.active_frontier = (row, col, x, y)
        if not preserve_route_id:
            self.active_route_id += 1
        self.active_since = now
        self.active_best_distance = math.hypot(x - robot_map[0], y - robot_map[1])
        self.active_best_path_distance = (
            float(steps[row, col]) * float(message.info.resolution)
            if steps is not None and steps[row, col] >= 0 else None
        )
        self.active_progress_time = now
        self.active_last_robot_xy = (robot_map[0], robot_map[1])
        self.active_unreachable_since = None
        # The promoted endpoint is a mission commitment.  The command sent to
        # move_base is selected below from the current connected route; reset
        # the command cache so a distant branch first receives a connector
        # point instead of being published as one large jump.
        self.active_last_waypoint_map = None
        self.active_last_waypoint_yaw = None
        self.active_route_kind = "frontier_endpoint"
        self.turn_connector_released = True
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
        free = known_free & ~self.inflate(occupied, clearance_cells)
        frontier_cells = max(
            1,
            int(math.ceil(self.frontier_clearance / message.info.resolution)) - 1,
        )
        frontier_free = known_free & ~self.inflate(occupied, frontier_cells)
        robot_col = int((robot_map[0] - message.info.origin.position.x) / message.info.resolution)
        robot_row = int((robot_map[1] - message.info.origin.position.y) / message.info.resolution)
        seed = self.nearest_seed(free, robot_row, robot_col, max(1, int(0.8 / message.info.resolution)))
        if seed is None:
            rospy.logwarn_throttle(3.0, "Global frontier has no safe map seed at odom=(%.2f,%.2f)", self.pose_odom.x, self.pose_odom.y)
            return
        steps = self.bfs(free, seed)
        # Detect information boundaries with the looser mask, but route to a
        # nearby safe approach cell using ``steps``.  The target therefore
        # remains valid for the same Navfn/TEB obstacle contract as the rest of
        # the map.
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
                if (
                    self.active_best_path_distance is None
                    or route_distance
                    < self.active_best_path_distance - self.progress_epsilon
                ):
                    self.active_best_path_distance = route_distance
                    # Keep the legacy field populated for status/debug users;
                    # its lifecycle now follows the route metric as well.
                    self.active_best_distance = route_distance
                    self.active_progress_time = now
                # A route can require a turn or a lateral detour before its
                # Euclidean distance to the frontier decreases, but raw
                # displacement is not progress: a wall-follow loop can move
                # metres while never getting closer to the selected branch.
                # Keep the last robot position for diagnostics, but let the
                # distance-to-frontier watchdog decide whether the route is
                # healthy. This prevents one bad local side choice from
                # holding the same frontier indefinitely.
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
                if turn_phase_active:
                    self.active_progress_time = now
                stalled = (
                    not turn_phase_active
                    and now - self.active_progress_time > self.stall_timeout
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
                # reached.  ``early_handoff_distance`` is still the gate for
                # promoting it, but the selection itself runs once the robot is
                # inside ``prefetch_distance`` so a validated continuation is
                # ready before TEB enters its handoff window.  This is what
                # lets the endpoint model hand the route over in-place instead
                # of stopping at every frontier boundary.
                if distance <= self.prefetch_distance:
                    self.prefetch_next_frontier(
                        message,
                        route_steps,
                        frontier,
                        unknown,
                        occupied,
                        now,
                        (x, y),
                        robot_map,
                        validation=validation,
                        heading_reference=robot_yaw_map,
                    )
                # Once the next branch has passed the same map/costmap/Navfn
                # checks, hand it to move_base while the current route still
                # has room for TEB to turn.  This is intentionally separate
                # from the normal terminal-release path: the bridge decides
                # whether the action can be replaced in-place, while this
                # node makes the pending branch visible early enough.
                if (
                    self.prefetched_frontier is not None
                    and distance <= self.early_handoff_distance
                ):
                    pending_row, pending_col, pending_x, pending_y = (
                        self.prefetched_frontier
                    )
                    if self.prefetched_route_continues_active(
                        route_steps,
                        seed,
                        (row, col),
                        (pending_row, pending_col),
                    ):
                        previous_x, previous_y = x, y
                        early_promoted = self.promote_prefetched_frontier(
                            message, route_steps, robot_map, now,
                            validation=validation,
                            preserve_route_id=True,
                        )
                        if early_promoted is not None:
                            self.mark_frontier_completed(previous_x, previous_y)
                            rospy.loginfo(
                                "Global frontier early handoff along continuous route "
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
                    distance > 0.8 or (distance <= 0.8 and not waypoint_reached)
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
                        self.mark_frontier_completed(x, y)
                        self.publish_status(
                            "route_invalidated",
                            reason="dead_end_resolved",
                            goal=[round(float(x), 3), round(float(y), 3)],
                            distance=round(float(distance), 3),
                        )
                        rospy.loginfo(
                            "Global frontier resolved dead-end map=(%.2f,%.2f) "
                            "at distance=%.2fm; abandoning wall approach",
                            x, y, distance,
                        )
                        self.active_frontier = None
                        self.active_last_robot_xy = None
                        self.active_last_waypoint_map = None
                        self.active_route_kind = "frontier_endpoint"
                        self.turn_connector_released = True
                        self.last_status_command_map = None
                        self.last_status_command_yaw = None
                        self.last_status_mission_map = None
                        self.active_best_path_distance = None
                        self.active_unreachable_since = None
                        if self.prefetched_frontier is None:
                            self.prefetched_goal_map = None
                    else:
                        active_cell = (row, col, x, y, route_steps[row, col] * message.info.resolution, 0.0)
                        if distance <= 0.8 and not waypoint_reached:
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
                    if distance <= 0.8 and waypoint_reached:
                        self.mark_frontier_completed(x, y)
                    else:
                        # A policy stall or a timeout is not proof that this
                        # frontier was inspected. Defer it briefly while other
                        # map branches are explored, then allow a retry.
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
                            path_distance=(
                                None
                                if self.active_best_path_distance is None
                                else round(float(self.active_best_path_distance), 3)
                            ),
                        )
                    self.active_frontier = None
                    self.active_last_robot_xy = None
                    self.active_last_waypoint_map = None
                    self.active_route_kind = "frontier_endpoint"
                    self.turn_connector_released = True
                    self.last_status_command_map = None
                    self.last_status_command_yaw = None
                    self.last_status_mission_map = None
                    self.active_best_path_distance = None
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
                    )
                    self.active_frontier = None
                    self.active_last_robot_xy = None
                    self.active_last_waypoint_map = None
                    self.active_route_kind = "frontier_endpoint"
                    self.turn_connector_released = True
                    self.last_status_command_map = None
                    self.last_status_command_yaw = None
                    self.last_status_mission_map = None
                    self.active_best_path_distance = None
                    self.active_unreachable_since = None
                    self.prefetched_frontier = None
                    self.prefetched_goal_map = None
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
                )
            if active_cell is None:
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
                self.active_frontier = (row, col, x, y)
                self.active_route_id += 1
                self.active_since = now
                self.active_best_distance = float(path_distance)
                self.active_best_path_distance = float(path_distance)
                self.active_progress_time = now
                self.active_last_robot_xy = (robot_map[0], robot_map[1])
                self.active_unreachable_since = None
                self.active_last_waypoint_map = None
                self.active_route_kind = "frontier_endpoint"
                self.turn_connector_released = True
                self.last_status_command_map = None
                self.last_status_command_yaw = None
                self.last_status_mission_map = None
                rospy.loginfo(
                    "Global frontier selected map=(%.2f,%.2f) path=%.2fm "
                    "information=%.0f structure=%.0f score=%.2f heading_delta=%.1fdeg",
                    x,
                    y,
                    path_distance,
                    information,
                    structure,
                    score,
                    math.degrees(self.heading_delta(x, y, robot_map, robot_yaw_map))
                    if self.heading_delta(x, y, robot_map, robot_yaw_map) is not None
                    else float("nan"),
                )
                self.publish_status(
                    "route_selected",
                    goal=[round(float(x), 3), round(float(y), 3)],
                    path_distance=round(float(path_distance), 3),
                    information=round(float(information), 3),
                    structure=round(float(structure), 3),
                )
                if self.pending_replan_request_id > 0:
                    self.publish_status(
                        "replan_ready",
                        replan_request_id=self.pending_replan_request_id,
                        reason=self.pending_replan_reason,
                        goal=[round(float(x), 3), round(float(y), 3)],
                        path_distance=round(float(path_distance), 3),
                    )
                    rospy.loginfo(
                        "Global frontier replan ready id=%d goal=(%.2f,%.2f)",
                        self.pending_replan_request_id,
                        x,
                        y,
                    )
                    self.pending_replan_request_id = 0
                    self.pending_replan_reason = ""
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
                if (
                    self.mission_endpoint_only
                    and turn_delta is not None
                    and turn_delta > (0.5 * math.pi)
                    and remaining_path > self.waypoint_release_radius
                ):
                    rospy.loginfo(
                        "Global frontier retained single endpoint across sharp "
                        "route transition: endpoint=(%.2f,%.2f) "
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
        self.publisher.publish(goal)


if __name__ == "__main__":
    GlobalFrontierExplorer()
    rospy.spin()
