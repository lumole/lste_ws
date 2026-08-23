#!/usr/bin/env python3
"""SA-PPO runner with optional controller-side lidar safety recovery.

It uses the production checkpoint and StageWorld lidar/goal encoder.
``policy_only`` is the original behaviour; guarded modes preserve the same
global goal and add only controller-local obstacle memory/recovery.
"""

import math
import os
import sys
import time
import heapq
from collections import deque
from pathlib import Path

from mpi4py import MPI
import numpy as np
import rospy
import torch
from std_msgs.msg import String

# This test runner lives outside rl_navigation, unlike sappo_pure.py.
RL_NAVIGATION_DIR = Path(__file__).resolve().parents[3] / "rl_navigation"
sys.path.insert(0, str(RL_NAVIGATION_DIR))

from model.ppo import generate_action_no_sampling
from model.sa_pe import MLPPolicy
from rl_env import StageWorld


OBS_SIZE = 512
LASER_HISTORY = 3
ACTION_BOUND = [[0.0, -0.5], [0.8, 0.5]]


def wrap_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class ActionShaper:
    """Bound command changes without delaying a genuine safety stop.

    The policy and the lidar guard both run at 10 Hz, while their decisions can
    change by a full action bound between two scans.  A small action slew limit
    removes the resulting left/right and accelerate/stop chatter.  A hard stop
    is deliberately passed through immediately for a completed goal or an
    unavailable safe rollout.
    """

    def __init__(self):
        self.last = np.zeros(2, dtype=np.float32)
        self.max_linear_step = max(
            0.02, float(rospy.get_param("~max_linear_action_step", 0.12))
        )
        self.max_linear_decel_step = max(
            0.02, float(rospy.get_param("~max_linear_action_decel", 0.06))
        )
        self.max_angular_step = max(
            0.03, float(rospy.get_param("~max_angular_action_step", 0.16))
        )
        self.angular_deadband = max(
            0.0, float(rospy.get_param("~angular_action_deadband", 0.035))
        )
        # Small opposite-sign corrections are usually grid quantization or a
        # noisy wall-distance estimate.  Let the command decay through zero;
        # require a meaningful opposite request before reversing the turn.
        self.angular_sign_switch_threshold = max(
            self.angular_deadband,
            float(rospy.get_param("~angular_sign_switch_threshold", 0.18)),
        )
        self.in_place_turn_threshold = max(
            0.05, float(rospy.get_param("~in_place_turn_angular_threshold", 0.12))
        )

    def apply(self, action, hard_stop=False, immediate_turn_stop=False):
        target = np.asarray(action, dtype=np.float32).reshape(-1)
        if target.size < 2:
            target = np.zeros(2, dtype=np.float32)
        target = np.array([
            float(np.clip(target[0], 0.0, 0.8)),
            float(np.clip(target[1], -0.5, 0.5)),
        ], dtype=np.float32)
        # A boundary turn asks for zero forward speed, but it is not by itself
        # an emergency.  Decelerate through the normal slew limiter so a
        # controller-mode change cannot create a repeated 0.17 -> 0.00 m/s
        # brake pulse.  ``hard_stop`` remains the only instantaneous stop.
        if abs(float(target[1])) < self.angular_deadband:
            target[1] = 0.0
        if (
            abs(float(self.last[1])) > self.angular_deadband
            and float(self.last[1]) * float(target[1]) < 0.0
            and abs(float(target[1])) < self.angular_sign_switch_threshold
        ):
            target[1] = 0.0
        if hard_stop:
            self.last[:] = 0.0
            return [0.0, 0.0]
        delta = target - self.last
        linear_step = self.max_linear_step
        if delta[0] < 0.0:
            linear_step = self.max_linear_decel_step
        delta[0] = float(np.clip(delta[0], -linear_step, self.max_linear_step))
        delta[1] = float(np.clip(delta[1], -self.max_angular_step, self.max_angular_step))
        self.last += delta
        self.last[0] = float(np.clip(self.last[0], 0.0, 0.8))
        self.last[1] = float(np.clip(self.last[1], -0.5, 0.5))
        if abs(float(self.last[1])) < self.angular_deadband:
            self.last[1] = 0.0
        return [float(self.last[0]), float(self.last[1])]


class StabilizedRecovery:
    """Turn once toward a behind-the-robot goal, with hysteresis and dwell."""

    def __init__(self):
        self.active = False
        self.exit_candidate_time = None
        self.enter_error = math.radians(120.0)
        self.exit_error = math.radians(25.0)
        self.exit_dwell = 0.3

    def action(self, local_goal):
        heading_error = math.atan2(float(local_goal[1]), float(local_goal[0]))
        magnitude = abs(heading_error)
        now = rospy.Time.now().to_sec()

        if not self.active and magnitude > self.enter_error:
            self.active = True
            self.exit_candidate_time = None
            rospy.loginfo("RL test recovery entered: heading_error=%.1f deg", math.degrees(heading_error))

        if not self.active:
            return None

        if magnitude < self.exit_error:
            if self.exit_candidate_time is None:
                self.exit_candidate_time = now
            elif now - self.exit_candidate_time >= self.exit_dwell:
                self.active = False
                self.exit_candidate_time = None
                rospy.loginfo("RL test recovery exited: heading_error=%.1f deg", math.degrees(heading_error))
                return None
        else:
            self.exit_candidate_time = None

        # Positive heading error is left; negative is right.
        return [0.0, 0.5 if heading_error > 0.0 else -0.5]


class RollingObstacleMap:
    """Short-lived odom-frame obstacle map built from consecutive 2D scans.

    This is deliberately a controller-local rolling obstacle cache, not a new
    source of navigation goals. It preserves wall geometry while the robot
    turns, so a door opening is not forgotten as soon as it leaves one laser
    frame.
    """

    def __init__(self, retention_seconds=6.0, cell_size=0.12, max_range=8.0):
        self.retention_seconds = retention_seconds
        self.cell_size = cell_size
        self.max_range = max_range
        self.frames = deque()

    @staticmethod
    def _scan_points(scan, scan_param, max_range):
        if scan is None or scan_param is None:
            return np.empty((0, 2), dtype=np.float32)
        values = np.asarray(scan, dtype=np.float32)
        if values.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        angles = float(scan_param[0]) + np.arange(values.size, dtype=np.float32) * float(scan_param[2])
        valid = np.isfinite(values) & (values >= 0.05) & (values <= max_range)
        ranges = values[valid]
        angles = angles[valid]
        # A 512-beam scan has much finer angular resolution than this local
        # controller requires. Downsampling keeps planning deterministic.
        return np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))[::4]

    def update(self, scan, scan_param, pose):
        points = self._scan_points(scan, scan_param, self.max_range)
        if points.size == 0 or pose is None:
            return
        x, y, yaw = (float(value) for value in pose)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        world_points = np.empty_like(points)
        world_points[:, 0] = x + cos_yaw * points[:, 0] - sin_yaw * points[:, 1]
        world_points[:, 1] = y + sin_yaw * points[:, 0] + cos_yaw * points[:, 1]
        now = time.monotonic()
        self.frames.append((now, world_points))
        while self.frames and now - self.frames[0][0] > self.retention_seconds:
            self.frames.popleft()

    def local_points(self, pose):
        if pose is None or not self.frames:
            return np.empty((0, 2), dtype=np.float32)
        world_points = np.concatenate([frame[1] for frame in self.frames], axis=0)
        x, y, yaw = (float(value) for value in pose)
        dx = world_points[:, 0] - x
        dy = world_points[:, 1] - y
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        points = np.column_stack((cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy))
        nearby = np.hypot(points[:, 0], points[:, 1]) <= self.max_range
        points = points[nearby]
        if points.size == 0:
            return points

        # Occupancy-cell consolidation turns many overlapping scan endpoints
        # into a compact rolling costmap representation.
        cells = np.floor(points / self.cell_size).astype(np.int32)
        _, indices = np.unique(cells, axis=0, return_index=True)
        return points[np.sort(indices)]


class LocalGridPlanner:
    """Small lidar-only occupancy grid with an inflated-obstacle A* search."""

    def __init__(self, half_extent=6.0, cell_size=0.15, obstacle_radius=0.56):
        self.half_extent = half_extent
        self.cell_size = cell_size
        self.obstacle_radius = obstacle_radius
        self.size = int(round(2.0 * half_extent / cell_size)) + 1
        self.center = self.size // 2

    def _cell(self, point):
        return (
            int(round(float(point[0]) / self.cell_size)) + self.center,
            int(round(float(point[1]) / self.cell_size)) + self.center,
        )

    def _point(self, cell):
        return np.array([
            (cell[0] - self.center) * self.cell_size,
            (cell[1] - self.center) * self.cell_size,
        ], dtype=np.float32)

    @staticmethod
    def _edge_is_clear(start, end, points, clearance=0.42):
        """Use the same circular footprint clearance as the velocity guard."""
        if points.size == 0:
            return True
        delta = end - start
        distance = float(np.hypot(delta[0], delta[1]))
        samples = max(1, int(math.ceil(distance / 0.05)))
        for ratio in np.linspace(0.0, 1.0, samples + 1):
            location = start + float(ratio) * delta
            if float(np.min(np.hypot(points[:, 0] - location[0], points[:, 1] - location[1]))) < clearance:
                return False
        return True

    def waypoint(self, points, local_goal):
        """Return a collision-free local waypoint, or None when none is known."""
        if points.size == 0:
            return None
        occupied = set()
        radius = int(math.ceil(self.obstacle_radius / self.cell_size))
        for point in points:
            cx, cy = self._cell(point)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        x, y = cx + dx, cy + dy
                        if 0 <= x < self.size and 0 <= y < self.size:
                            occupied.add((x, y))

        start = (self.center, self.center)
        # The grid is local, so pursue a reachable segment when the final goal
        # lies outside it. Replanning on every scan rolls that segment forward.
        goal = np.asarray(local_goal, dtype=np.float32)
        length = float(np.hypot(goal[0], goal[1]))
        if length > self.half_extent - 0.5:
            goal *= (self.half_extent - 0.5) / length
        target = self._cell(goal)
        if start in occupied or target in occupied:
            return None

        frontier = [(0.0, start)]
        cost = {start: 0.0}
        parent = {}
        current = None
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == target:
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                neighbour = (current[0] + dx, current[1] + dy)
                if (not (0 <= neighbour[0] < self.size and 0 <= neighbour[1] < self.size)
                        or neighbour in occupied
                        or not self._edge_is_clear(
                            self._point(current), self._point(neighbour), points
                        )):
                    continue
                step = math.hypot(dx, dy)
                candidate = cost[current] + step
                if candidate >= cost.get(neighbour, float("inf")):
                    continue
                cost[neighbour] = candidate
                parent[neighbour] = current
                heuristic = math.hypot(target[0] - neighbour[0], target[1] - neighbour[1])
                heapq.heappush(frontier, (candidate + heuristic, neighbour))
        if current != target:
            return None

        path = [target]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        lookahead = min(len(path) - 1, max(1, int(1.5 / self.cell_size)))
        cell = path[lookahead]
        return np.array([
            (cell[0] - self.center) * self.cell_size,
            (cell[1] - self.center) * self.cell_size,
        ], dtype=np.float32)

    @staticmethod
    def action_to_waypoint(waypoint):
        distance = float(np.hypot(waypoint[0], waypoint[1]))
        heading = math.atan2(float(waypoint[1]), float(waypoint[0]))
        heading_abs = abs(heading)
        # Turn in place only when the local route is genuinely side-on.  The
        # previous 18-degree hard boundary converted small A* cell changes
        # into alternating stop/turn commands in a straight corridor.
        if heading_abs > math.radians(32.0):
            return [0.0, float(np.clip(0.9 * heading, -0.5, 0.5))]
        # Use a continuous proportional steering law with a small deadband.
        # A +/-0.15 m lateral waypoint correction now produces a small angular
        # command instead of jumping between zero and +/-0.27 action.
        # Keep a small steering deadband around the corridor centreline.  The
        # 15 cm grid can quantize the next cell from left to right while the
        # physical path is still straight; feeding those tiny signs directly
        # into the policy creates visible yaw chatter.
        angular = 0.0 if heading_abs < math.radians(5.0) else float(
            np.clip(0.85 * heading, -0.38, 0.38)
        )
        speed = min(0.55, max(0.10, 0.42 * distance))
        if heading_abs > math.radians(12.0):
            speed *= max(0.45, math.cos(heading_abs))
        return [speed, angular]


class DwaSafetyGuard:
    """Test-only local DWA guard around a proposed SA-PPO action.

    It never changes the final goal. When the policy proposes reverse/zero
    forward motion, has a large heading error, or predicts a collision, this
    guard chooses a collision-free short-horizon (linear, angular) command
    from the same action space. The raw policy command passes through when it
    is usable and safe.
    """

    def __init__(self):
        self.dt = 0.10
        self.steps = 24
        # Actions are scaled by StageWorld before they are sent to Gazebo. The
        # local rollout must use that same physical speed; otherwise it judges
        # a 3 s horizon as twice as long as the robot actually travels.
        self.linear_speed_scale = float(rospy.get_param("~linear_speed_scale", 0.5))
        self.robot_radius = 0.30
        self.clearance_margin = 0.12
        self.min_safe_clearance = self.robot_radius + self.clearance_margin
        self.preferred_clearance = 0.70
        self.heading_recovery = math.radians(20.0)
        self.last_reason = "initializing"
        self.obstacle_map = RollingObstacleMap()

        # Commands are policy-space values; StageWorld scales angular velocity
        # by 1.2 before publishing /cmd_vel.
        self.linear_samples = np.array([0.0, 0.08, 0.16, 0.24, 0.32], dtype=np.float32)
        self.angular_samples = np.linspace(-0.5, 0.5, 25, dtype=np.float32)

    def obstacle_points(self, scan, scan_param, pose):
        self.obstacle_map.update(scan, scan_param, pose)
        return self.obstacle_map.local_points(pose)

    def rollout(self, action, local_goal, points):
        linear = self.linear_speed_scale * float(action[0])
        angular = 1.2 * float(action[1])
        x = y = yaw = 0.0
        min_clearance = float("inf")
        clearance_samples = []

        for _ in range(self.steps):
            x += linear * math.cos(yaw) * self.dt
            y += linear * math.sin(yaw) * self.dt
            yaw = wrap_angle(yaw + angular * self.dt)
            if points.size:
                clearance = float(np.min(np.hypot(points[:, 0] - x, points[:, 1] - y)))
                min_clearance = min(min_clearance, clearance)
                clearance_samples.append(clearance)
                # Rotation in place is allowed to escape a tight position, but
                # forward travel must preserve the configured safety margin.
                if linear > 0.02 and clearance < self.min_safe_clearance:
                    return None

        goal_x, goal_y = float(local_goal[0]), float(local_goal[1])
        goal_distance = math.hypot(goal_x - x, goal_y - y)
        final_heading_error = abs(wrap_angle(math.atan2(goal_y - y, goal_x - x) - yaw))
        if not math.isfinite(min_clearance):
            min_clearance = 5.0
            clearance_samples.append(min_clearance)
        return (
            goal_distance,
            final_heading_error,
            min_clearance,
            clearance_samples[-1],
            float(np.mean(clearance_samples)),
        )

    def choose(self, raw_action, policy_action, local_goal, scan, scan_param, pose):
        points = self.obstacle_points(scan, scan_param, pose)
        heading_error = wrap_angle(math.atan2(float(local_goal[1]), float(local_goal[0])))
        policy_rollout = self.rollout(policy_action, local_goal, points)

        if float(raw_action[0]) > 0.01 and abs(heading_error) <= self.heading_recovery and policy_rollout:
            self.last_reason = "policy_safe"
            return policy_action, False, self.last_reason, policy_rollout[2]

        if float(raw_action[0]) <= 0.01:
            self.last_reason = "policy_reverse_or_stop"
        elif abs(heading_error) > self.heading_recovery:
            self.last_reason = "heading_not_aligned"
        else:
            self.last_reason = "policy_collision_risk"

        best_action = None
        best_score = float("inf")
        best_clearance = 0.0
        for linear in self.linear_samples:
            for angular in self.angular_samples:
                candidate = [float(linear), float(angular)]
                rollout = self.rollout(candidate, local_goal, points)
                if rollout is None:
                    continue
                goal_distance, final_heading_error, clearance, final_clearance, mean_clearance = rollout
                # Prefer progress toward the unchanged goal, then a small final
                # heading error and generous obstacle clearance. This is a local
                # control score, not a planner or a replacement goal generator.
                score = (
                    2.0 * goal_distance
                    + 0.65 * final_heading_error
                    + 0.25 / max(clearance, 0.05)
                    + 0.20 / max(final_clearance, 0.05)
                    + 0.10 / max(mean_clearance, 0.05)
                    + 0.03 * abs(angular)
                )
                if score < best_score:
                    best_score = score
                    best_action = candidate
                    best_clearance = clearance

        if best_action is None:
            self.last_reason = "emergency_stop_no_safe_rollout"
            return [0.0, 0.0], True, self.last_reason, 0.0
        return best_action, True, self.last_reason, best_clearance


class MppiSafetyGuard(DwaSafetyGuard):
    """Test-only sampling-MPC guard that can turn first and then move.

    DWA evaluates a constant command and can settle at a wall because turning
    initially increases goal-heading error. This guard evaluates complete
    command sequences over the same local laser snapshot, including explicit
    turn-then-drive maneuvers. It remains a controller-side safety override:
    the global goal and all LSTE brain topics stay untouched.
    """

    def __init__(self):
        super().__init__()
        self.steps = 30
        self.sample_count = 120
        self.rng = np.random.default_rng(20260722)
        self.nominal = np.zeros((self.steps, 2), dtype=np.float32)
        # Sampling a fresh sequence every cycle makes the controller oscillate
        # once a wall is close. This state holds one outward heading until the
        # robot has genuinely created space from that wall.
        self.escape_active = False
        self.escape_heading_world = 0.0
        self.escape_start = None
        self.escape_started_at = 0.0
        self.escape_trigger_range = 0.72
        self.escape_release_range = 0.82
        self.escape_distance = 0.70
        self.escape_timeout = 12.0

    @staticmethod
    def _current_scan_points(scan, scan_param):
        """Return current-frame obstacle endpoints, without rolling-map history."""
        return RollingObstacleMap._scan_points(scan, scan_param, max_range=1.5)

    def _begin_escape(self, scan, scan_param, pose, local_goal):
        """Lock a goal-directed wall-following heading from current lidar data."""
        points = self._current_scan_points(scan, scan_param)
        if pose is None or points.size == 0:
            return False

        ranges = np.hypot(points[:, 0], points[:, 1])
        nearby = ranges < 1.2
        if not np.any(nearby):
            return False
        points = points[nearby]
        ranges = ranges[nearby]

        # Endpoint vectors point from robot to obstacles. Their inverse-square
        # sum points toward occupied space, so its negative is an outward
        # normal. It remains the fallback for non-wall-shaped obstacles.
        weights = 1.0 / np.maximum(ranges, 0.15) ** 2
        outward = -np.sum(points * weights[:, None], axis=0)
        outward_norm = float(np.hypot(outward[0], outward[1]))
        if outward_norm < 1e-5:
            return False
        outward /= outward_norm

        x, y, yaw = (float(value) for value in pose)
        direction = outward
        if points.shape[0] >= 8:
            # A wall produces a strongly elongated endpoint cloud. Follow its
            # principal tangent in whichever direction has more goal progress,
            # then retain a small outward component as a clearance buffer.
            centered = points - np.mean(points, axis=0)
            covariance = centered.T @ centered / float(points.shape[0])
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if float(eigenvalues[1]) > 4.0 * max(float(eigenvalues[0]), 1e-6):
                tangent = eigenvectors[:, 1]
                goal_world = np.array([
                    math.cos(yaw) * float(local_goal[0]) - math.sin(yaw) * float(local_goal[1]),
                    math.sin(yaw) * float(local_goal[0]) + math.cos(yaw) * float(local_goal[1]),
                ], dtype=np.float32)
                if float(np.dot(tangent, goal_world)) < 0.0:
                    tangent = -tangent
                direction = tangent + 0.28 * outward
                direction /= max(float(np.hypot(direction[0], direction[1])), 1e-6)

        self.escape_heading_world = wrap_angle(yaw + math.atan2(direction[1], direction[0]))
        self.escape_start = (x, y)
        self.escape_started_at = rospy.Time.now().to_sec()
        self.escape_active = True
        rospy.logwarn(
            "RL test wall-follow entered: heading=%.1f deg nearest=%.2f m",
            math.degrees(self.escape_heading_world), float(ranges.min()),
        )
        return True

    def _escape_action(self, scan, scan_param, pose):
        """Rotate to the locked safe heading, then drive outward without reversing."""
        if not self.escape_active or pose is None or self.escape_start is None:
            return None

        x, y, yaw = (float(value) for value in pose)
        elapsed = rospy.Time.now().to_sec() - self.escape_started_at
        travelled = math.hypot(x - self.escape_start[0], y - self.escape_start[1])
        current = self._current_scan_points(scan, scan_param)
        current_min = float(np.min(np.hypot(current[:, 0], current[:, 1]))) if current.size else 5.0

        if ((travelled >= self.escape_distance and current_min >= self.escape_release_range)
                or elapsed >= self.escape_timeout):
            self.escape_active = False
            self.escape_start = None
            rospy.loginfo(
                "RL test wall-follow exited: travelled=%.2f m nearest=%.2f m elapsed=%.1f s",
                travelled, current_min, elapsed,
            )
            return None

        heading_error = wrap_angle(self.escape_heading_world - yaw)
        if abs(heading_error) > math.radians(14.0):
            return [0.0, 0.5 if heading_error > 0.0 else -0.5]

        # Continue to favour heading correction while moving. The command stays
        # in the original SA-PPO action domain and never commands reverse.
        return [0.32, float(np.clip(0.9 * heading_error, -0.5, 0.5))]

    def rollout_sequence(self, sequence, local_goal, points):
        x = y = yaw = 0.0
        min_clearance = float("inf")
        clearance_samples = []
        for linear, angular_action in sequence:
            linear_velocity = self.linear_speed_scale * float(linear)
            x += linear_velocity * math.cos(yaw) * self.dt
            y += linear_velocity * math.sin(yaw) * self.dt
            yaw = wrap_angle(yaw + 1.2 * float(angular_action) * self.dt)
            if points.size:
                clearance = float(np.min(np.hypot(points[:, 0] - x, points[:, 1] - y)))
                min_clearance = min(min_clearance, clearance)
                clearance_samples.append(clearance)
                # Never advance into the safety buffer. A zero-linear step is
                # still valid so the guard can rotate away from a nearby wall.
                if linear_velocity > 0.02 and clearance < self.min_safe_clearance:
                    return None

        goal_x, goal_y = float(local_goal[0]), float(local_goal[1])
        goal_distance = math.hypot(goal_x - x, goal_y - y)
        heading_error = abs(wrap_angle(math.atan2(goal_y - y, goal_x - x) - yaw))
        if not math.isfinite(min_clearance):
            min_clearance = 5.0
            clearance_samples.append(min_clearance)
        return (
            goal_distance,
            heading_error,
            min_clearance,
            clearance_samples[-1],
            float(np.mean(clearance_samples)),
        )

    def maneuver_sequences(self):
        """Generate deterministic escape maneuvers plus stochastic MPPI samples."""
        sequences = []
        for direction in (-1.0, 1.0):
            for turn_steps in (0, 4, 8, 12, 16):
                sequence = np.zeros((self.steps, 2), dtype=np.float32)
                sequence[:turn_steps, 1] = direction * 0.5
                # A gentle arc after the turn gives the optimizer trajectories
                # that can leave a wall instead of scoring only spin-in-place.
                sequence[turn_steps:, 0] = 0.20
                sequence[turn_steps:, 1] = direction * 0.12
                sequences.append(sequence)

        # Sample smooth perturbations around the best sequence from the prior
        # control cycle. This is the receding-horizon MPPI exploration term.
        noise = self.rng.normal(
            loc=0.0, scale=np.array([0.10, 0.23], dtype=np.float32),
            size=(self.sample_count, self.steps, 2),
        ).astype(np.float32)
        noise[:, 1:, :] = 0.65 * noise[:, 1:, :] + 0.35 * noise[:, :-1, :]
        samples = np.clip(self.nominal[None, :, :] + noise, ACTION_BOUND[0], ACTION_BOUND[1])
        sequences.extend(samples)
        return sequences

    def choose(self, raw_action, policy_action, local_goal, scan, scan_param, pose):
        escape_action = self._escape_action(scan, scan_param, pose)
        if escape_action is not None:
            self.last_reason = "committed_wall_escape"
            return escape_action, True, self.last_reason, float("nan")

        current_points = self._current_scan_points(scan, scan_param)
        current_min = (float(np.min(np.hypot(current_points[:, 0], current_points[:, 1])))
                       if current_points.size else float("inf"))
        if current_min < self.escape_trigger_range and self._begin_escape(scan, scan_param, pose, local_goal):
            escape_action = self._escape_action(scan, scan_param, pose)
            if escape_action is not None:
                self.last_reason = "committed_wall_escape"
                return escape_action, True, self.last_reason, current_min

        points = self.obstacle_points(scan, scan_param, pose)
        heading_error = wrap_angle(math.atan2(float(local_goal[1]), float(local_goal[0])))
        policy_rollout = self.rollout(policy_action, local_goal, points)
        if float(raw_action[0]) > 0.01 and abs(heading_error) <= self.heading_recovery and policy_rollout:
            self.last_reason = "policy_safe"
            return policy_action, False, self.last_reason, policy_rollout[2]

        if float(raw_action[0]) <= 0.01:
            self.last_reason = "policy_reverse_or_stop"
        elif abs(heading_error) > self.heading_recovery:
            self.last_reason = "heading_not_aligned"
        else:
            self.last_reason = "policy_collision_risk"

        best_sequence = None
        best_score = float("inf")
        best_clearance = 0.0
        for sequence in self.maneuver_sequences():
            rollout = self.rollout_sequence(sequence, local_goal, points)
            if rollout is None:
                continue
            goal_distance, final_heading_error, clearance, final_clearance, mean_clearance = rollout
            # Rotating in place can retain a close clearance, but all moving
            # steps have already passed the hard safety-margin constraint.
            clearance_cost = 0.40 / max(clearance, 0.05)
            # The hard constraint uses the worst clearance. For route choice,
            # however, use final and average clearance too: after the robot
            # has reached a wall, every candidate shares the close starting
            # point, so a minimum-only objective cannot reward escaping it.
            preferred_clearance_cost = (
                4.0 * max(0.0, (self.preferred_clearance - final_clearance) / self.preferred_clearance) ** 2
                + 1.5 * max(0.0, (self.preferred_clearance - mean_clearance) / self.preferred_clearance) ** 2
            )
            smoothness = float(np.mean(np.abs(np.diff(sequence[:, 1]))))
            score = (
                2.0 * goal_distance
                + 0.50 * final_heading_error
                + clearance_cost
                + preferred_clearance_cost
                + 0.04 * smoothness
            )
            if score < best_score:
                best_score = score
                best_sequence = sequence
                best_clearance = clearance

        if best_sequence is None:
            self.last_reason = "emergency_stop_no_safe_sequence"
            return [0.0, 0.0], True, self.last_reason, 0.0

        # Receding horizon: retain the selected plan for the next optimization
        # cycle, shifted by one command. The first command alone is published.
        self.nominal[:-1] = best_sequence[1:]
        self.nominal[-1] = best_sequence[-1]
        return best_sequence[0].tolist(), True, self.last_reason, best_clearance


class GridSafetyGuard(DwaSafetyGuard):
    """Lidar-grid local planner used only for the fixed-goal controller test."""

    def __init__(self):
        super().__init__()
        # The fixed-goal experiment needs to retain the obstacle contour that
        # led to a wall endpoint.  A six-second, six-metre window forgets the
        # U-shaped barrier exactly when the robot reaches its exit, which makes
        # local A* repeatedly choose a different side of the same wall.
        self.obstacle_map = RollingObstacleMap(retention_seconds=90.0, max_range=10.0)
        self.grid_obstacle_radius = max(
            0.30, float(rospy.get_param("~grid_obstacle_radius", 0.48))
        )
        self.grid_edge_clearance = max(
            0.30, float(rospy.get_param("~grid_edge_clearance", 0.42))
        )
        self.grid_planner = LocalGridPlanner(
            half_extent=10.0, obstacle_radius=self.grid_obstacle_radius
        )
        self.boundary_active = False
        self.boundary_heading_world = 0.0
        self.boundary_hit_distance = float("inf")
        self.boundary_start = None
        self.goal_visibility_frames = 0
        # Once a boundary end is detected, keep the selected lateral side for
        # long enough to rotate and obtain a new scan. Re-selecting a side on
        # every control cycle can invert the command while the robot is still
        # turning, which produces an in-place left/right oscillation.
        self.boundary_turn_lock_until = 0.0
        self.boundary_turn_side = 0
        self.boundary_turn_lock_time = max(
            0.5, float(rospy.get_param("~boundary_turn_lock_time", 3.0))
        )
        # A blocked scan is not evidence that another full 90-degree turn is
        # needed.  The lidar can remain blocked while the robot is already
        # looking through the corner, so bound each turn by time/angle and
        # then let the wall-follow controller advance the contour.
        self.boundary_turn_max_duration = max(
            self.boundary_turn_lock_time,
            float(rospy.get_param("~boundary_turn_max_duration", 2.8)),
        )
        self.boundary_turn_max_angle = math.radians(max(
            70.0, float(rospy.get_param("~boundary_turn_max_angle_deg", 112.0))
        ))
        self.boundary_turn_max_attempts = max(
            1, int(rospy.get_param("~boundary_turn_max_attempts", 2))
        )
        self.boundary_turn_active = False
        self.boundary_turn_started_at = 0.0
        self.boundary_turn_start_yaw = 0.0
        self.boundary_turn_requested_angle = 0.0
        self.boundary_turn_attempts = 0
        self.boundary_turn_cooldown_until = 0.0
        self.boundary_reentry_cooldown = max(
            0.0, float(rospy.get_param("~boundary_reentry_cooldown", 1.5))
        )
        self.boundary_goal_cancel_angle = math.radians(max(
            120.0,
            float(rospy.get_param("~boundary_goal_cancel_angle_deg", 165.0)),
        ))
        self.boundary_goal_cancel_distance = max(
            0.45, float(rospy.get_param("~boundary_goal_cancel_distance", 0.75))
        )
        self.boundary_entry_blocked_until = 0.0
        # World-frame target at the moment Bug2 takes ownership.  A target
        # moving behind the robot is expected while tracing a wall; only a
        # genuinely different branch should invalidate this commitment.
        self.boundary_goal_world = None
        self.last_boundary_goal_delta = float("nan")
        self.boundary_entries = 0
        self.boundary_exits = 0
        self.boundary_turns = 0
        # A wall-follow commitment must have a bounded escape plan.  If the
        # distance to the same waypoint has not improved for a while and the
        # robot has already travelled away from the hit point, try the other
        # tangent once, then release ownership so A* can re-evaluate the map.
        self.boundary_progress_timeout = max(
            5.0, float(rospy.get_param("~boundary_progress_timeout", 18.0))
        )
        self.boundary_progress_margin = max(
            0.30, float(rospy.get_param("~boundary_progress_margin", 0.80))
        )
        self.boundary_best_goal_distance = float("inf")
        self.boundary_progress_last_time = 0.0
        self.boundary_side_switches = 0
        self.boundary_stuck_releases = 0
        # Wall-follow feedback.  ``boundary_turn_side`` is the direction of
        # the next turn; this separate sign identifies which side of the car
        # should remain beside the obstacle while driving along its contour.
        self.boundary_wall_side = 0
        self.boundary_wall_distance = max(
            0.45, float(rospy.get_param("~boundary_wall_distance", 0.72))
        )
        self.boundary_wall_max_range = max(
            self.boundary_wall_distance + 0.30,
            float(rospy.get_param("~boundary_wall_max_range", 1.80)),
        )
        self.boundary_wall_kp = max(
            0.05, float(rospy.get_param("~boundary_wall_kp", 0.45))
        )
        self.boundary_wall_heading_kp = max(
            0.0, float(rospy.get_param("~boundary_wall_heading_kp", 0.35))
        )
        self.boundary_wall_max_linear = float(np.clip(
            rospy.get_param("~boundary_wall_max_linear", 0.28), 0.10, 0.55
        ))
        self.boundary_wall_last_distance = float("nan")
        self.boundary_wall_last_front = float("nan")
        self.boundary_wall_last_rear = float("nan")
        self.boundary_wall_last_error = float("nan")
        self.boundary_wall_last_angular = 0.0
        self.boundary_wall_lost_frames = 0
        self.boundary_wall_filter_alpha = min(
            1.0, max(0.05, float(rospy.get_param("~boundary_wall_filter_alpha", 0.28)))
        )
        self.boundary_wall_angular_step = max(
            0.02, float(rospy.get_param("~boundary_wall_angular_step", 0.08))
        )
        self.boundary_wall_angular_deadband = max(
            0.0, float(rospy.get_param("~boundary_wall_angular_deadband", 0.035))
        )
        self.boundary_wall_filtered_distance = float("nan")
        self.boundary_wall_filtered_front = float("nan")
        self.boundary_wall_filtered_rear = float("nan")
        self.boundary_wall_front_missing_frames = 0
        self.boundary_wall_rear_missing_frames = 0
        self.boundary_forward_blocked_frames = 0
        self.boundary_turn_heading_tolerance = math.radians(10.0)
        # Goal-facing rotation uses hysteresis: visual and frontier waypoints
        # can hover near a wall-following turn threshold while the vehicle is
        # turning, so a single threshold would repeatedly hand control back to
        # Bug2 before the vehicle is actually facing the new route.
        self.reorienting_goal = False
        self.reorienting_turn_sign = 0
        self.reorienting_aligned_since = None
        self.reorient_entries = 0
        self.reorient_exits = 0
        # A boundary tangent and a goal-facing turn are mutually exclusive.
        # Keep a small hysteresis band and dwell at the aligned heading so a
        # 10 Hz controller cannot alternate between the two modes when a
        # waypoint sits near the side of the robot.
        self.reorient_enter = math.radians(55.0)
        self.reorient_exit = math.radians(25.0)
        self.reorient_exit_dwell = 0.8
        self.reorient_obstacle_clearance = max(
            0.45, float(rospy.get_param("~reorient_obstacle_clearance", 0.72))
        )
        self.last_point_count = 0
        self.last_forward_clearance = float("nan")
        self.last_waypoint = None
        self.stable_waypoint_world = None
        self.waypoint_hold_radius = max(
            0.20, float(rospy.get_param("~waypoint_hold_radius", 0.40))
        )
        self.waypoint_switch_distance = max(
            self.waypoint_hold_radius,
            float(rospy.get_param("~waypoint_switch_distance", 0.75)),
        )
        self.waypoint_hold_count = 0
        self.waypoint_switch_count = 0
        # A rolling grid can report no path for one or two frames while a scan
        # is integrated.  Keep a deterministic recovery action through that
        # interval instead of invoking the policy/DWA fallback, whose best
        # candidate can alternate between stop and turn commands.
        self.no_path_active = False
        self.no_path_started_at = 0.0
        self.no_path_turn_sign = 0
        self.no_path_entries = 0
        self.no_path_holds = 0
        # Require a valid A* route for a few consecutive frames before leaving
        # a wall contour.  This prevents a single optimistic scan from
        # handing control back to A* and immediately re-entering Bug2.
        self.boundary_astar_clear_frames = 0
        self.boundary_astar_rejoins = 0
        self.last_goal_world = None
        self.forward_clearance_filtered = float("nan")

    @staticmethod
    def _local_to_world(point, pose):
        """Transform a controller-local point into the odom/world frame."""
        x, y, yaw = (float(value) for value in pose)
        local_x, local_y = float(point[0]), float(point[1])
        return np.array([
            x + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
            y + math.sin(yaw) * local_x + math.cos(yaw) * local_y,
        ], dtype=np.float32)

    @staticmethod
    def _world_to_local(point, pose):
        """Transform an odom/world point into the current robot frame."""
        x, y, yaw = (float(value) for value in pose)
        dx, dy = float(point[0]) - x, float(point[1]) - y
        return np.array([
            math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
        ], dtype=np.float32)

    @classmethod
    def _goal_world(cls, local_goal, pose):
        if pose is None:
            return None
        return cls._local_to_world(local_goal, pose)

    def _select_stable_waypoint(self, candidate, pose, points):
        """Keep a safe A* waypoint stable across lidar/grid quantization.

        The rolling grid is rebuilt at 10 Hz.  Without a world-frame hold, a
        straight corridor can alternate between neighbouring y-cells even
        though the physical route has not changed.  Release the held waypoint
        only after it is close, its edge is blocked, or it has fallen behind.
        """
        if candidate is None or pose is None:
            return candidate
        candidate_world = self._local_to_world(candidate, pose)
        if self.stable_waypoint_world is None:
            self.stable_waypoint_world = candidate_world
            self.waypoint_switch_count += 1
            return candidate

        held_local = self._world_to_local(self.stable_waypoint_world, pose)
        held_distance = float(np.hypot(held_local[0], held_local[1]))
        candidate_delta = float(np.hypot(
            candidate_world[0] - self.stable_waypoint_world[0],
            candidate_world[1] - self.stable_waypoint_world[1],
        ))
        held_heading = abs(math.atan2(float(held_local[1]), float(held_local[0])))
        held_clear = self.grid_planner._edge_is_clear(
            np.zeros(2, dtype=np.float32), held_local, points,
            clearance=self.grid_edge_clearance
        )
        release = (
            held_distance <= self.waypoint_hold_radius
            or not held_clear
            or held_local[0] < -0.20
            or held_heading > math.radians(105.0)
        )
        if release:
            self.stable_waypoint_world = candidate_world
            self.waypoint_switch_count += 1
            return candidate

        # A large candidate displacement is logged but does not by itself
        # cancel a still-clear route. This is the same commitment rule used by
        # the global frontier manager and prevents branch flicker at a corner.
        if candidate_delta > self.waypoint_switch_distance:
            self.waypoint_hold_count += 1
        else:
            self.waypoint_hold_count += 1
        return held_local

    def diagnostic(self):
        waypoint = "none" if self.last_waypoint is None else "(%.2f,%.2f)" % (
            self.last_waypoint[0], self.last_waypoint[1]
        )
        clearance = self.last_forward_clearance
        now = time.monotonic()
        lock_remaining = max(0.0, self.boundary_turn_lock_until - now)
        cooldown_remaining = max(0.0, self.boundary_entry_blocked_until - now)
        boundary_goal_delta = float("nan")
        if self.boundary_goal_world is not None:
            # The current pose is not needed here; this field is updated in
            # choose() and is retained for a compact status message.
            boundary_goal_delta = self.last_boundary_goal_delta
        return "grid_points=%d forward_clearance=%.3f waypoint=%s boundary=%s " \
               "side=%d lock=%.2f cooldown=%.2f reorient=%s " \
               "entries=%d exits=%d turns=%d turn_active=%s turn_attempts=%d " \
               "turn_cooldown=%.2f side_switches=%d stuck_releases=%d " \
               "reorient_entries=%d reorient_exits=%d " \
               "wp_holds=%d wp_switches=%d boundary_goal_delta=%.2f " \
               "no_path=%s no_path_entries=%d no_path_holds=%d " \
               "astar_clear=%d astar_rejoins=%d wall_side=%d wall_dist=%.3f " \
               "wall_front=%.3f wall_rear=%.3f wall_error=%.3f wall_angular=%.3f " \
               "wall_filtered=(%.3f,%.3f,%.3f) blocked_frames=%d" % (
            self.last_point_count,
            clearance if math.isfinite(clearance) else float("nan"),
            waypoint,
            self.boundary_active,
            self.boundary_turn_side,
            lock_remaining,
            cooldown_remaining,
            self.reorienting_goal,
            self.boundary_entries,
            self.boundary_exits,
            self.boundary_turns,
            self.boundary_turn_active,
            self.boundary_turn_attempts,
            max(0.0, self.boundary_turn_cooldown_until - now),
            self.boundary_side_switches,
            self.boundary_stuck_releases,
            self.reorient_entries,
            self.reorient_exits,
            self.waypoint_hold_count,
            self.waypoint_switch_count,
            boundary_goal_delta,
            self.no_path_active,
            self.no_path_entries,
            self.no_path_holds,
            self.boundary_astar_clear_frames,
            self.boundary_astar_rejoins,
            self.boundary_wall_side,
            self.boundary_wall_last_distance,
            self.boundary_wall_last_front,
            self.boundary_wall_last_rear,
            self.boundary_wall_last_error,
            self.boundary_wall_last_angular,
            self.boundary_wall_filtered_distance,
            self.boundary_wall_filtered_front,
            self.boundary_wall_filtered_rear,
            self.boundary_forward_blocked_frames,
        )

    @staticmethod
    def _turn_action(error, max_action=0.40):
        """Return a bounded proportional turn in the SA-PPO action domain."""
        return float(np.clip(0.90 * float(error), -max_action, max_action))

    def _reset_no_path_state(self):
        self.no_path_active = False
        self.no_path_started_at = 0.0
        self.no_path_turn_sign = 0

    def _held_waypoint_if_safe(self, pose, points):
        """Reuse the previous world waypoint through a transient A* failure."""
        if self.stable_waypoint_world is None or pose is None:
            return None
        held_local = self._world_to_local(self.stable_waypoint_world, pose)
        distance = float(np.hypot(held_local[0], held_local[1]))
        heading = abs(math.atan2(float(held_local[1]), float(held_local[0])))
        if (
            distance <= self.waypoint_hold_radius
            or held_local[0] < 0.0
            or heading > math.radians(80.0)
            or not self.grid_planner._edge_is_clear(
                np.zeros(2, dtype=np.float32), held_local, points,
                clearance=self.grid_edge_clearance
            )
        ):
            return None
        return held_local

    def _no_path_action(self, local_goal, pose, points, visibility_points):
        """Hold a stable local recovery action while A* catches up.

        The rolling occupancy grid is intentionally conservative.  A scan/map
        update can therefore make a valid corridor disappear for a single
        control cycle.  This state uses the current forward clearance and a
        locked turn sign; it never commands reverse and it only calls Bug2
        when a real near obstacle is present.
        """
        now = time.monotonic()
        if not self.no_path_active:
            self.no_path_active = True
            self.no_path_started_at = now
            self.no_path_entries += 1
            bearing = math.atan2(float(local_goal[1]), float(local_goal[0]))
            self.no_path_turn_sign = 1 if bearing >= 0.0 else -1
        self.no_path_holds += 1

        bearing = math.atan2(float(local_goal[1]), float(local_goal[0]))
        bearing_abs = abs(bearing)
        near_clearance = (
            float(np.min(np.hypot(points[:, 0], points[:, 1])))
            if points.size else float("inf")
        )
        forward_clearance = self.last_forward_clearance

        # A large bearing error is handled as one locked, proportional turn.
        # Unlike the old DWA fallback this cannot change sign because a single
        # noisy grid frame changed the preferred rollout.
        if bearing_abs > math.radians(48.0):
            self.last_reason = "grid_no_path_reorient"
            sign = self.no_path_turn_sign or (1 if bearing >= 0.0 else -1)
            return [0.0, self._turn_action(sign * min(bearing_abs, math.pi / 2.0), 0.34)]

        # If the current scan still shows a close obstacle, let the boundary
        # state choose a verified tangent.  It owns side selection and its
        # lock prevents left/right chatter.
        if near_clearance <= self.reorient_obstacle_clearance:
            boundary_action = self._boundary_action(
                local_goal, pose, points, visibility_points
            )
            if boundary_action is not None:
                return boundary_action

        # Forward clearance is noisy near furniture.  Use a broad hysteresis
        # band and a low-speed continuation instead of dropping to zero at one
        # threshold.  The rollout safety check still rejects a genuinely
        # colliding command before it is published.
        if math.isfinite(forward_clearance) and forward_clearance < 0.62:
            self.last_reason = "grid_no_path_scan_turn"
            sign = self.no_path_turn_sign or 1
            return [0.0, self._turn_action(0.35 * sign, 0.28)]

        speed = 0.24
        if math.isfinite(forward_clearance):
            speed = float(np.clip(
                0.16 + 0.10 * (forward_clearance - 0.75), 0.16, 0.34
            ))
        if bearing_abs > math.radians(12.0):
            speed *= max(0.55, math.cos(bearing_abs))
        angular = 0.0 if bearing_abs < math.radians(4.0) else self._turn_action(
            bearing, 0.22
        )
        self.last_reason = "grid_no_path_hold_forward"
        return [speed, angular]

    def _reset_boundary_state(self, cooldown=True):
        """Release Bug2 ownership and clear all route-specific memory."""
        self.boundary_active = False
        self.boundary_start = None
        self.boundary_goal_world = None
        self.last_boundary_goal_delta = float("nan")
        self.boundary_best_goal_distance = float("inf")
        self.boundary_progress_last_time = 0.0
        self.boundary_side_switches = 0
        self.goal_visibility_frames = 0
        self.boundary_turn_lock_until = 0.0
        self.boundary_turn_side = 0
        self.boundary_turn_active = False
        self.boundary_turn_started_at = 0.0
        self.boundary_turn_start_yaw = 0.0
        self.boundary_turn_requested_angle = 0.0
        self.boundary_turn_attempts = 0
        self.boundary_turn_cooldown_until = 0.0
        self.boundary_wall_side = 0
        self.boundary_wall_last_distance = float("nan")
        self.boundary_wall_last_front = float("nan")
        self.boundary_wall_last_rear = float("nan")
        self.boundary_wall_last_error = float("nan")
        self.boundary_wall_last_angular = 0.0
        self.boundary_wall_filtered_distance = float("nan")
        self.boundary_wall_filtered_front = float("nan")
        self.boundary_wall_filtered_rear = float("nan")
        self.boundary_wall_front_missing_frames = 0
        self.boundary_wall_rear_missing_frames = 0
        self.boundary_wall_lost_frames = 0
        self.boundary_forward_blocked_frames = 0
        if cooldown:
            self.boundary_entry_blocked_until = (
                time.monotonic() + self.boundary_reentry_cooldown
            )
        self.reorienting_goal = False
        self.reorienting_turn_sign = 0
        self.reorienting_aligned_since = None
        self.stable_waypoint_world = None
        self.boundary_astar_clear_frames = 0

    def _begin_boundary_turn(self, pose, angle, reason="bug2_boundary_turn"):
        """Start one bounded in-place scan turn.

        ``boundary_turn_lock_until`` is retained for diagnostics and legacy
        callers, but ``boundary_turn_active`` is the authoritative state. A
        turn ends when its requested heading is reached, not merely when a
        wall scan happens to look clear for one frame.
        """
        if pose is None or self.boundary_turn_attempts >= self.boundary_turn_max_attempts:
            return False
        _, _, yaw = (float(value) for value in pose)
        sign = 1.0 if float(angle) >= 0.0 else -1.0
        requested = min(abs(float(angle)), self.boundary_turn_max_angle)
        if requested < math.radians(35.0):
            requested = math.radians(35.0)
        self.boundary_turn_side = 1 if sign > 0.0 else -1
        self.boundary_heading_world = wrap_angle(yaw + sign * requested)
        now = time.monotonic()
        self.boundary_turn_active = True
        self.boundary_turn_started_at = now
        self.boundary_turn_start_yaw = yaw
        self.boundary_turn_requested_angle = requested
        self.boundary_turn_attempts += 1
        self.boundary_turn_lock_until = now + self.boundary_turn_max_duration
        self.boundary_turns += 1
        self.last_reason = reason
        return True

    def _complete_boundary_turn(self, reason):
        """Release the turn lock and give wall following a chance to advance."""
        self.boundary_turn_active = False
        self.boundary_turn_lock_until = 0.0
        self.boundary_turn_cooldown_until = time.monotonic() + 0.9
        self.boundary_forward_blocked_frames = 0
        self.boundary_wall_last_angular = 0.0
        self.last_reason = reason

    def _turn_around_boundary_end(self, local_goal, pose, points):
        """Choose a verified side around a newly encountered boundary end.

        The chosen directions are expressed in the current robot frame.  They
        deliberately exclude a U-turn: boundary following should look through
        an opening beside the wall, not retreat from the unexplored route.
        """
        if pose is None:
            return False
        now = time.monotonic()
        if self.boundary_turn_active or self.boundary_turn_attempts >= self.boundary_turn_max_attempts:
            self.last_reason = "bug2_turn_budget_exhausted"
            return False
        _, _, yaw = (float(value) for value in pose)
        goal_bearing = math.atan2(float(local_goal[1]), float(local_goal[0]))
        # Once Bug2 has selected a boundary side, keep it at corners. Choosing
        # the other side from a transient scan is what creates +/- turn flips.
        if self.boundary_turn_side > 0:
            candidates = (math.pi / 2.0,)
        elif self.boundary_turn_side < 0:
            candidates = (-math.pi / 2.0,)
        else:
            candidates = (-math.pi / 2.0, math.pi / 2.0)
        viable = []
        for angle in candidates:
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            if self.grid_planner._edge_is_clear(
                    np.zeros(2, dtype=np.float32), direction * 0.90, points,
                    clearance=self.grid_edge_clearance,
            ):
                score = abs(wrap_angle(goal_bearing - angle))
                viable.append((score, angle))
        if not viable:
            return False
        _, angle = min(viable)
        return self._begin_boundary_turn(
            pose, angle, reason="bug2_turn_around_boundary_end"
        )

    def _boundary_progress_watchdog(self, local_goal, pose):
        """Bound Bug2 wall-following when a selected tangent makes no progress."""
        if not self.boundary_active or pose is None or self.boundary_start is None:
            return None
        now = time.monotonic()
        x, y, yaw = (float(value) for value in pose)
        distance = float(np.hypot(local_goal[0], local_goal[1]))
        if not math.isfinite(self.boundary_best_goal_distance):
            self.boundary_best_goal_distance = distance
            self.boundary_progress_last_time = now
            return None
        if distance < self.boundary_best_goal_distance - 0.12:
            self.boundary_best_goal_distance = distance
            self.boundary_progress_last_time = now
            return None

        travelled = math.hypot(
            x - self.boundary_start[0], y - self.boundary_start[1]
        )
        stalled_for = now - self.boundary_progress_last_time
        far_from_best = distance > self.boundary_best_goal_distance + self.boundary_progress_margin
        if stalled_for < self.boundary_progress_timeout or not far_from_best or travelled < 0.80:
            return None

        if self.boundary_side_switches < 1:
            best_before_switch = self.boundary_best_goal_distance
            self.boundary_side_switches += 1
            self.boundary_turn_side = -self.boundary_turn_side or 1
            self.boundary_wall_side = -self.boundary_wall_side or -self.boundary_turn_side
            self.boundary_start = (x, y)
            self.boundary_best_goal_distance = distance
            self.boundary_progress_last_time = now
            self.last_reason = "bug2_switch_side_no_progress"
            rospy.logwarn(
                "RL test Bug2 switched wall side: best_goal=%.2f current=%.2f "
                "travelled=%.2f stalled=%.1fs side=%d",
                best_before_switch,
                distance,
                travelled,
                stalled_for,
                self.boundary_turn_side,
            )
            if self._begin_boundary_turn(
                    pose,
                    self.boundary_turn_side * math.pi / 2.0,
                    reason="bug2_switch_side_no_progress",
            ):
                return [0.0, self._turn_action(
                    self.boundary_turn_side * math.pi / 2.0, 0.40
                )]
            self.last_reason = "bug2_release_no_progress_turn_budget"
            self._reset_boundary_state(cooldown=True)
            return None

        self.boundary_stuck_releases += 1
        self.last_reason = "bug2_release_no_progress"
        rospy.logwarn(
            "RL test Bug2 released after no progress: best_goal=%.2f current=%.2f "
            "travelled=%.2f stalled=%.1fs",
            self.boundary_best_goal_distance,
            distance,
            travelled,
            stalled_for,
        )
        self._reset_boundary_state(cooldown=True)
        return None

    @staticmethod
    def _sector_stat(points, center_angle, half_width, percentile=20.0, max_range=2.5):
        """Return a robust range statistic for one laser angular sector."""
        if points is None or points.size == 0:
            return float("nan")
        angles = np.arctan2(points[:, 1], points[:, 0])
        delta = np.arctan2(
            np.sin(angles - float(center_angle)),
            np.cos(angles - float(center_angle)),
        )
        mask = (np.abs(delta) <= float(half_width))
        if not np.any(mask):
            return float("nan")
        ranges = np.hypot(points[mask, 0], points[mask, 1])
        ranges = ranges[np.isfinite(ranges)]
        # A boundary side is meaningful only while an obstacle is within the
        # local contour window.  Including 6-10 m endpoints in the percentile
        # makes a missing wall look like a large positive distance error and
        # steers the robot toward empty space.
        ranges = ranges[ranges <= float(max_range)]
        if ranges.size == 0:
            return float("nan")
        return float(np.percentile(ranges, percentile))

    def _wall_follow_action(self, pose, visibility_points):
        """Track a selected wall with distance/heading feedback.

        The previous boundary controller kept the heading captured at Bug2
        entry and returned ``[0.35, 0]`` forever.  A small heading error then
        became metres of lateral drift.  This controller uses the current scan
        to regulate the chosen wall side and its tangent, so a doorway/corner
        can be followed without repeatedly reselecting a new goal.
        """
        if pose is None or self.boundary_wall_side == 0:
            return None
        wall_side = 1 if self.boundary_wall_side > 0 else -1
        side_angle = wall_side * math.pi / 2.0
        side_distance = self._sector_stat(
            visibility_points,
            side_angle,
            math.radians(52.0),
            percentile=25.0,
            max_range=self.boundary_wall_max_range,
        )
        front_distance = self._sector_stat(
            visibility_points, wall_side * math.pi / 4.0,
            math.radians(30.0), percentile=35.0,
            max_range=self.boundary_wall_max_range,
        )
        rear_distance = self._sector_stat(
            visibility_points, wall_side * 3.0 * math.pi / 4.0,
            math.radians(30.0), percentile=35.0,
            max_range=self.boundary_wall_max_range,
        )
        self.boundary_wall_last_distance = side_distance
        self.boundary_wall_last_front = front_distance
        self.boundary_wall_last_rear = rear_distance

        def smooth(previous, current):
            if not math.isfinite(current):
                return previous
            if not math.isfinite(previous):
                return current
            alpha = self.boundary_wall_filter_alpha
            return previous + alpha * (current - previous)

        self.boundary_wall_filtered_distance = smooth(
            self.boundary_wall_filtered_distance, side_distance
        )
        if math.isfinite(front_distance):
            self.boundary_wall_front_missing_frames = 0
            self.boundary_wall_filtered_front = smooth(
                self.boundary_wall_filtered_front, front_distance
            )
        else:
            self.boundary_wall_front_missing_frames += 1
            if self.boundary_wall_front_missing_frames >= 2:
                self.boundary_wall_filtered_front = float("nan")
        if math.isfinite(rear_distance):
            self.boundary_wall_rear_missing_frames = 0
            self.boundary_wall_filtered_rear = smooth(
                self.boundary_wall_filtered_rear, rear_distance
            )
        else:
            self.boundary_wall_rear_missing_frames += 1
            if self.boundary_wall_rear_missing_frames >= 2:
                self.boundary_wall_filtered_rear = float("nan")

        # If the wall is briefly out of view at a convex corner, continue in
        # the locked tangent for one scan interval rather than changing sides.
        if not math.isfinite(side_distance):
            self.boundary_wall_lost_frames += 1
            # Do not let a last-seen side range keep steering the car after the
            # contour has disappeared. It is valid only for the current scan.
            self.boundary_wall_filtered_distance = float("nan")
            # A convex corner can hide the wall for a few scans. If it stays
            # absent while the forward sector is open, this is no longer a
            # wall-follow situation; release to A* instead of drifting on a
            # stale tangent.
            if self.boundary_wall_lost_frames >= 3 and (
                    not math.isfinite(self.last_forward_clearance)
                    or self.last_forward_clearance > 0.80
            ):
                self._reset_boundary_state(cooldown=True)
                self.last_reason = "bug2_release_wall_lost"
                return None
            _, _, yaw = (float(value) for value in pose)
            error = wrap_angle(self.boundary_heading_world - yaw)
            angular = self._turn_action(error, 0.30)
            self.boundary_wall_last_error = float("nan")
            self.boundary_wall_last_angular = angular
            return [0.20, angular]

        self.boundary_wall_lost_frames = 0
        side_distance = self.boundary_wall_filtered_distance
        front_distance = self.boundary_wall_filtered_front
        rear_distance = self.boundary_wall_filtered_rear
        if not math.isfinite(side_distance):
            return None
        distance_error = side_distance - self.boundary_wall_distance
        tangent_error = 0.0
        if math.isfinite(front_distance) and math.isfinite(rear_distance):
            # For a left wall, front<rear means the nose points toward the
            # wall and requires a right correction.  Multiplying by the wall
            # sign gives the same convention for a right wall.
            tangent_error = front_distance - rear_distance
        steering = wall_side * (
            self.boundary_wall_kp * distance_error
            + self.boundary_wall_heading_kp * tangent_error
        )
        target_angular = float(np.clip(steering, -0.34, 0.34))
        # A small opposite correction at a corner is usually a percentile/grid
        # fluctuation. Let the previous correction decay through zero; reserve
        # an immediate sign change for a genuinely strong new tangent.
        if (
            self.boundary_wall_last_angular * target_angular < 0.0
            and abs(target_angular) < 0.20
        ):
            target_angular = 0.0
        # Limit wall-feedback changes independently of the policy shaper. This
        # prevents a single percentile jump at a doorway from commanding an
        # immediate opposite turn while still allowing a genuine corner turn.
        angular = float(np.clip(
            target_angular,
            self.boundary_wall_last_angular - self.boundary_wall_angular_step,
            self.boundary_wall_last_angular + self.boundary_wall_angular_step,
        ))
        if abs(angular) < self.boundary_wall_angular_deadband:
            angular = 0.0
        self.boundary_wall_last_error = distance_error
        self.boundary_wall_last_angular = angular

        forward_clearance = self.last_forward_clearance
        if math.isfinite(forward_clearance):
            # Slow down before the footprint buffer, but let the wall feedback
            # continue steering instead of replacing the command with a hard
            # stop on every noisy range sample.
            speed = float(np.clip(
                0.16 + 0.18 * (forward_clearance - 0.55),
                0.12, self.boundary_wall_max_linear,
            ))
        else:
            speed = min(0.25, self.boundary_wall_max_linear)
        if abs(angular) > 0.22:
            speed *= 0.72
        return [speed, angular]

    def _boundary_action(self, local_goal, pose, points, visibility_points=None):
        if pose is None:
            return None
        x, y, yaw = (float(value) for value in pose)
        goal_distance = float(np.hypot(local_goal[0], local_goal[1]))
        direct_goal = np.asarray(local_goal, dtype=np.float32)
        if goal_distance > self.grid_planner.half_extent:
            direct_goal *= self.grid_planner.half_extent / goal_distance

        if self.boundary_active:
            # A goal behind the robot is normal while tracing the obstacle.
            # Compare the world-frame target with the one captured when Bug2
            # entered instead of cancelling on local bearing alone.  The old
            # angle-only rule cancelled a valid wall route at every half-turn,
            # then the goal reorientation state started the same turn again.
            current_goal_world = self._goal_world(local_goal, pose)
            if self.boundary_goal_world is None:
                self.boundary_goal_world = current_goal_world
            if current_goal_world is not None and self.boundary_goal_world is not None:
                self.last_boundary_goal_delta = float(np.hypot(
                    current_goal_world[0] - self.boundary_goal_world[0],
                    current_goal_world[1] - self.boundary_goal_world[1],
                ))
            else:
                self.last_boundary_goal_delta = float("nan")
            goal_bearing = math.atan2(float(local_goal[1]), float(local_goal[0]))
            if (
                math.isfinite(self.last_boundary_goal_delta)
                and self.last_boundary_goal_delta >= self.boundary_goal_cancel_distance
            ):
                # Keep the selected wall side, but re-anchor Bug2 to the new
                # world target and the current hit point. This handles a
                # normal frontier waypoint advance without following the old
                # segment for several metres; it also avoids the old
                # cancel-then-reorient oscillation at the same wall.
                self.boundary_goal_world = current_goal_world
                self.boundary_start = (x, y)
                self.boundary_hit_distance = goal_distance
                self.goal_visibility_frames = 0
                self.last_boundary_goal_delta = 0.0
                self.boundary_best_goal_distance = goal_distance
                self.boundary_progress_last_time = time.monotonic()
                self.boundary_side_switches = 0
                self.last_reason = "bug2_update_boundary_goal"

            watchdog_action = self._boundary_progress_watchdog(local_goal, pose)
            if watchdog_action is not None:
                return watchdog_action
            if not self.boundary_active:
                # The watchdog released an exhausted commitment. Let the
                # caller run A* / no-path selection with the current scan.
                return None

            displacement = (math.hypot(x - self.boundary_start[0], y - self.boundary_start[1])
                            if self.boundary_start is not None else 0.0)
            # Classic Bug2 waits until it is closer to the goal than the hit
            # point.  That condition is unnecessarily conservative for a
            # lidar controller: this U-shaped wall first requires moving away
            # from the goal.  Once the rolling lidar map confirms a direct,
            # footprint-clear segment after genuine exploration, hand control
            # back to A* immediately.
            # Current-frame endpoints are less prone to occlusion ghosts than
            # the rolling map for deciding whether to leave a wall.  Require
            # three consecutive observations before changing modes.
            visibility = points if visibility_points is None else visibility_points
            if self.boundary_wall_side == 0:
                # Recover the wall side if a boundary state was entered by an
                # older checkpoint or a transient no-path branch.
                nearby = visibility[np.hypot(visibility[:, 0], visibility[:, 1]) < 1.6] \
                    if visibility.size else np.empty((0, 2), dtype=np.float32)
                mean_y = float(np.mean(nearby[:, 1])) if nearby.size else 0.0
                self.boundary_wall_side = 1 if mean_y >= 0.0 else -1
            direct_visible = self.grid_planner._edge_is_clear(
                np.zeros(2, dtype=np.float32), direct_goal, visibility,
                clearance=self.grid_edge_clearance,
            )
            self.goal_visibility_frames = self.goal_visibility_frames + 1 if direct_visible else 0
            if displacement >= 0.75 and self.goal_visibility_frames >= 3:
                self.boundary_exits += 1
                self._reset_boundary_state(cooldown=True)
                self.last_reason = "bug2_leave_boundary_direct_route_visible"
                return None
            # A locked outward heading can meet a newly observed obstacle
            # around a corner. Turn for a new scan instead of driving into the
            # safety buffer or handing control back to the reverse-prone policy.
            local_forward = np.array([1.0, 0.0], dtype=np.float32)
            forward_blocked = not self.grid_planner._edge_is_clear(
                np.zeros(2, dtype=np.float32), local_forward * 0.55, visibility,
                clearance=self.grid_edge_clearance,
            )
            if forward_blocked:
                self.boundary_forward_blocked_frames += 1
            else:
                self.boundary_forward_blocked_frames = 0

            # A boundary-end turn is an explicit short maneuver. Finish it by
            # heading or by a bounded time/angle budget, then let the wall
            # controller move. The previous implementation checked only a
            # three-second lock and started another quarter-turn whenever the
            # forward sector was still blocked, which could spin indefinitely.
            now = time.monotonic()
            if self.boundary_turn_active:
                heading_error = wrap_angle(self.boundary_heading_world - yaw)
                turned = abs(wrap_angle(yaw - self.boundary_turn_start_yaw))
                elapsed = now - self.boundary_turn_started_at
                if (
                    abs(heading_error) <= self.boundary_turn_heading_tolerance
                    or turned >= self.boundary_turn_requested_angle - math.radians(4.0)
                ):
                    self._complete_boundary_turn("bug2_boundary_turn_complete")
                elif elapsed >= self.boundary_turn_max_duration or turned >= self.boundary_turn_max_angle:
                    self._complete_boundary_turn("bug2_boundary_turn_budget_exhausted")
                else:
                    self.last_reason = "bug2_boundary_turn_locked"
                    return [0.0, self._turn_action(heading_error, 0.34)]

                # A completed turn gets one scan interval to expose the new
                # tangent. Do not immediately schedule another turn from the
                # same blocked frame.
                wall_action = self._wall_follow_action(pose, visibility)
                if wall_action is not None:
                    return wall_action
                if not self.boundary_active:
                    return None
                self.last_reason = "bug2_boundary_turn_follow_fallback"
                return [0.14, self._turn_action(
                    wrap_angle(self.boundary_heading_world - yaw), 0.24
                )]

            if self.boundary_forward_blocked_frames >= 2:
                if (
                    now >= self.boundary_turn_cooldown_until
                    and self.boundary_turn_attempts < self.boundary_turn_max_attempts
                    and self._turn_around_boundary_end(local_goal, pose, visibility)
                ):
                    error = wrap_angle(self.boundary_heading_world - yaw)
                    return [0.0, self._turn_action(error, 0.40)]
                # The turn budget is exhausted or the last turn just ended.
                # Follow the selected contour at low speed; this keeps the car
                # moving through a narrow doorway instead of issuing another
                # in-place stop/turn pair.
                self.last_reason = "bug2_boundary_blocked_wall_follow"
            wall_action = self._wall_follow_action(pose, visibility)
            if wall_action is not None:
                self.last_reason = "bug2_follow_boundary"
                return wall_action
            if not self.boundary_active:
                # ``_wall_follow_action`` released a stale contour; let the
                # caller immediately try the current A* route.
                return None
            self.last_reason = "bug2_follow_boundary_heading_fallback"
            return [0.24, self._turn_action(
                wrap_angle(self.boundary_heading_world - yaw), 0.28
            )]

        # After leaving a boundary or finishing a goal-facing turn, wait for a
        # fresh scan before re-entering Bug2. This prevents the same obstacle's
        # first partial scan from immediately reversing the previous turn.
        if time.monotonic() < self.boundary_entry_blocked_until:
            self.last_reason = "grid_boundary_entry_cooldown"
            return None
        if points.size == 0:
            return None
        visibility = points if visibility_points is None else visibility_points
        ranges = np.hypot(points[:, 0], points[:, 1])
        # Boundary entry is based on the current scan. The rolling map is used
        # for A*, but old endpoint cells must not make a clear current frame
        # look like a newly blocked wall.
        entry_points = visibility
        entry_ranges = np.hypot(entry_points[:, 0], entry_points[:, 1])
        near = entry_points[entry_ranges < 1.3]
        if near.shape[0] < 4:
            return None
        # In the usual failure case the robot is already moving in a useful
        # free direction, but the direct route to the goal becomes occluded by
        # a wall. Preserve that direction first.  Points are in *robot-local*
        # coordinates, therefore [1, 0] is the current forward edge regardless
        # of the robot's world yaw.  This sends the fixed test south toward the
        # known wall endpoint instead of incorrectly rotating west into it.
        if self.grid_planner._edge_is_clear(
                np.zeros(2, dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32),
                entry_points, clearance=self.grid_edge_clearance,
        ):
            self.boundary_heading_world = yaw
            self.boundary_hit_distance = goal_distance
            self.boundary_start = (x, y)
            self.boundary_goal_world = self._goal_world(local_goal, pose)
            self.last_boundary_goal_delta = 0.0
            self.boundary_active = True
            self.boundary_entries += 1
            self.boundary_best_goal_distance = goal_distance
            self.boundary_progress_last_time = time.monotonic()
            self.boundary_side_switches = 0
            self.goal_visibility_frames = 0
            self.boundary_turn_lock_until = 0.0
            # Deterministic right-hand fallback for a route that was initially
            # clear but later meets a boundary end. Keep this side locked.
            self.boundary_turn_side = -1
            left_distance = self._sector_stat(
                visibility, math.pi / 2.0, math.radians(52.0), percentile=20.0
            )
            right_distance = self._sector_stat(
                visibility, -math.pi / 2.0, math.radians(52.0), percentile=20.0
            )
            if math.isfinite(left_distance) and (
                    not math.isfinite(right_distance)
                    or left_distance + 0.15 < right_distance
            ):
                self.boundary_wall_side = 1
            elif math.isfinite(right_distance) and (
                    not math.isfinite(left_distance)
                    or right_distance + 0.15 < left_distance
            ):
                self.boundary_wall_side = -1
            else:
                nearby = visibility[
                    np.hypot(visibility[:, 0], visibility[:, 1]) < 1.6
                ] if visibility.size else np.empty((0, 2), dtype=np.float32)
                nearby_mean_y = float(np.mean(nearby[:, 1])) if nearby.size else 0.0
                self.boundary_wall_side = 1 if nearby_mean_y >= 0.0 else -1
            self.last_reason = "bug2_enter_boundary_keep_clear_heading"
            return self._boundary_action(local_goal, pose, points, visibility)

        # If forward is genuinely blocked, the weighted normal identifies the
        # wall side. Follow its tangent requiring the least turn from current
        # travel, which is Bug2's deterministic fallback side.
        near_ranges = np.hypot(near[:, 0], near[:, 1])
        normal = -np.sum(near / np.maximum(near_ranges[:, None], 0.15) ** 2, axis=0)
        normal /= max(float(np.hypot(normal[0], normal[1])), 1e-6)
        tangent = np.array([-normal[1], normal[0]], dtype=np.float32)
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        if float(np.dot(tangent, forward)) < 0.0:
            tangent = -tangent
        self.boundary_heading_world = math.atan2(float(tangent[1]), float(tangent[0]))
        self.boundary_hit_distance = goal_distance
        self.boundary_start = (x, y)
        self.boundary_goal_world = self._goal_world(local_goal, pose)
        self.last_boundary_goal_delta = 0.0
        self.boundary_active = True
        self.boundary_entries += 1
        self.boundary_best_goal_distance = goal_distance
        self.boundary_progress_last_time = time.monotonic()
        self.boundary_side_switches = 0
        self.goal_visibility_frames = 0
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        cross = float(forward[0] * tangent[1] - forward[1] * tangent[0])
        self.boundary_turn_side = 1 if cross > 0.0 else -1
        nearby = visibility[
            np.hypot(visibility[:, 0], visibility[:, 1]) < 1.6
        ] if visibility.size else np.empty((0, 2), dtype=np.float32)
        mean_y = float(np.mean(nearby[:, 1])) if nearby.size else 0.0
        if abs(mean_y) >= 0.05:
            self.boundary_wall_side = 1 if mean_y > 0.0 else -1
        else:
            # A head-on wall has no signed side in one scan. Keep the side
            # consistent with the chosen tangent until the next corner.
            self.boundary_wall_side = 1 if cross > 0.0 else -1
        self.boundary_turn_lock_until = 0.0
        self.last_reason = "bug2_enter_boundary"
        return self._boundary_action(local_goal, pose, points)

    def choose(self, raw_action, policy_action, local_goal, scan, scan_param, pose):
        # Update the rolling map before any early turn return.  Previously a
        # goal-facing rotation skipped map updates, so the first post-turn
        # frame used stale obstacle geometry and immediately recreated the
        # previous boundary decision.
        points = self.obstacle_points(scan, scan_param, pose)
        visibility_points = RollingObstacleMap._scan_points(scan, scan_param, max_range=8.0)
        self.last_point_count = int(points.shape[0])
        self.last_waypoint = None
        if pose is not None:
            current_goal_world = self._goal_world(local_goal, pose)
            if self.last_goal_world is not None and current_goal_world is not None:
                goal_delta = float(np.hypot(
                    current_goal_world[0] - self.last_goal_world[0],
                    current_goal_world[1] - self.last_goal_world[1],
                ))
                if goal_delta >= self.boundary_goal_cancel_distance:
                    # A new frontier/visual segment invalidates a held local
                    # waypoint, but not an already selected Bug2 wall side.
                    self.stable_waypoint_world = None
                    self._reset_no_path_state()
            self.last_goal_world = current_goal_world
        forward_points = visibility_points[
            (visibility_points[:, 0] > 0.0)
            & (np.abs(visibility_points[:, 1]) < 0.30)
        ] if visibility_points.size else np.empty((0, 2), dtype=np.float32)
        self.last_forward_clearance = (
            float(np.min(np.hypot(forward_points[:, 0], forward_points[:, 1])))
            if forward_points.size else float("nan")
        )
        if forward_points.size:
            forward_ranges = np.hypot(forward_points[:, 0], forward_points[:, 1])
            raw_min = float(np.min(forward_ranges))
            # One isolated low beam is often a chair leg or scan speckle. Use
            # a robust sector statistic for speed control, while preserving an
            # immediate hard response inside the true footprint buffer.
            robust = float(np.percentile(forward_ranges, 10.0))
            if raw_min <= self.min_safe_clearance:
                robust = raw_min
            if not math.isfinite(self.forward_clearance_filtered):
                self.forward_clearance_filtered = robust
            else:
                self.forward_clearance_filtered += 0.35 * (
                    robust - self.forward_clearance_filtered
                )
            self.last_forward_clearance = self.forward_clearance_filtered
        else:
            self.forward_clearance_filtered = float("nan")

        # Compute the local route before deciding whether to enter a pure
        # goal-facing turn. A valid A* waypoint already encodes a safe bend;
        # using the raw global bearing first was the source of many unnecessary
        # spin-in-place episodes after a frontier handoff.
        waypoint = self.grid_planner.waypoint(points, local_goal)
        if waypoint is not None:
            waypoint = self._select_stable_waypoint(waypoint, pose, points)
            self.last_waypoint = (float(waypoint[0]), float(waypoint[1]))
            if self.reorienting_goal:
                # A newly available collision-free route supersedes a stale
                # pure-turn commitment from a previous no-path scan.
                self.reorienting_goal = False
                self.reorienting_turn_sign = 0
                self.reorienting_aligned_since = None

        # In a clear, nearly straight segment the learned policy should remain
        # the authority. The grid is still used as a collision veto through the
        # rollout check, but rewriting every safe policy action as a waypoint
        # command makes the controller look like a second oscillating planner.
        if (
            waypoint is not None
            and not self.boundary_active
            and not self.reorienting_goal
            and float(raw_action[0]) > 0.08
            and float(policy_action[0]) > 0.04
            and abs(float(policy_action[1])) <= 0.22
            and math.isfinite(self.last_forward_clearance)
            and self.last_forward_clearance >= 0.80
        ):
            # The rolling map is for route planning. Collision prediction must
            # use the current scan; otherwise a wall endpoint observed tens of
            # seconds ago can trigger a false safety intervention after the
            # robot has already passed it.
            policy_rollout = self.rollout(policy_action, local_goal, visibility_points)
            waypoint_heading = math.atan2(float(waypoint[1]), float(waypoint[0]))
            goal_heading = math.atan2(float(local_goal[1]), float(local_goal[0]))
            if (
                policy_rollout is not None
                and abs(waypoint_heading) <= math.radians(28.0)
                and abs(goal_heading) <= math.radians(35.0)
            ):
                self._reset_no_path_state()
                self.last_reason = "policy_safe_open_route"
                return policy_action, False, self.last_reason, policy_rollout[2]

        # A visual-servo or global-frontier update can legitimately put the
        # replacement waypoint behind the robot.  Bug2's wall contour belongs
        # to the local obstacle route, so keep the contour when it is already
        # active.  If a new turn is requested while the robot is close to a
        # wall, enter wall-follow recovery first instead of spinning in place.
        if pose is not None:
            goal_bearing = math.atan2(float(local_goal[1]), float(local_goal[0]))
            bearing_magnitude = abs(goal_bearing)
            now = time.monotonic()
            near_clearance = (
                float(np.min(np.hypot(visibility_points[:, 0], visibility_points[:, 1])))
                if visibility_points.size else float("inf")
            )
            if (
                not self.boundary_active
                and bearing_magnitude >= self.reorient_enter
                and near_clearance <= self.reorient_obstacle_clearance
                and waypoint is None
            ):
                self.reorienting_goal = False
                self.reorienting_turn_sign = 0
                self.reorienting_aligned_since = None
                self.boundary_entry_blocked_until = 0.0
                wall_action = self._boundary_action(
                    local_goal, pose, points, visibility_points
                )
                if wall_action is not None:
                    return wall_action, True, self.last_reason, float("nan")
            # Start the turn before Bug2 claims a wall tangent. This also
            # covers a side-front waypoint when no boundary is active yet.
            if (not self.reorienting_goal and not self.boundary_active
                    and bearing_magnitude >= self.reorient_enter
                    and waypoint is None):
                self.reorienting_goal = True
                self.reorient_entries += 1
                self.reorienting_turn_sign = 1 if goal_bearing > 0.0 else -1
                self.reorienting_aligned_since = None
                self.last_reason = "grid_begin_goal_reorientation"
            if self.reorienting_goal:
                if bearing_magnitude <= self.reorient_exit:
                    if self.reorienting_aligned_since is None:
                        self.reorienting_aligned_since = now
                    elif now - self.reorienting_aligned_since >= self.reorient_exit_dwell:
                        self.reorienting_goal = False
                        self.reorient_exits += 1
                        self.reorienting_turn_sign = 0
                        self.reorienting_aligned_since = None
                        self.boundary_entry_blocked_until = now + self.boundary_reentry_cooldown
                else:
                    self.reorienting_aligned_since = None
            if self.reorienting_goal:
                if near_clearance <= self.reorient_obstacle_clearance:
                    # The route is wall-constrained. Preserve a single
                    # boundary side rather than repeatedly turning toward a
                    # goal that remains occluded from the current pose.
                    self.reorienting_goal = False
                    self.reorienting_turn_sign = 0
                    self.reorienting_aligned_since = None
                    self.boundary_entry_blocked_until = 0.0
                    wall_action = self._boundary_action(
                        local_goal, pose, points, visibility_points
                    )
                    if wall_action is not None:
                        return wall_action, True, self.last_reason, float("nan")
                self.boundary_active = False
                self.boundary_start = None
                self.goal_visibility_frames = 0
                self.boundary_entry_blocked_until = now + self.boundary_reentry_cooldown
                # Slow down near the exit band. A fixed +/-0.5 action can
                # cross both thresholds between observations and immediately
                # hand control to Bug2, recreating the opposite turn.
                turn_magnitude = 0.22 if bearing_magnitude < math.radians(60.0) else 0.5
                self.last_reason = "grid_turn_goal_behind"
                turn_sign = self.reorienting_turn_sign or (1 if goal_bearing > 0.0 else -1)
                return [0.0, self._turn_action(
                    turn_sign * min(bearing_magnitude, math.pi / 2.0),
                    0.34 if turn_magnitude < 0.5 else 0.40,
                )], True, self.last_reason, float("nan")
        # A* is the controller's best local route whenever the rolling lidar
        # map contains a collision-free path. Bug2 is a recovery policy for
        # the genuinely no-path case; letting it run first made it override
        # valid corner/doorway routes and repeatedly turn the camera away from
        # a recently detected target.
        if waypoint is not None:
            action = self.grid_planner.action_to_waypoint(waypoint)
            rollout = self.rollout(action, local_goal, visibility_points)
            if rollout is not None:
                if self.boundary_active:
                    self.boundary_astar_clear_frames += 1
                    if self.boundary_astar_clear_frames < 3:
                        boundary_action = self._boundary_action(
                            local_goal, pose, points, visibility_points
                        )
                        if boundary_action is not None:
                            return boundary_action, True, self.last_reason, float("nan")
                    self.boundary_astar_rejoins += 1
                    self.boundary_exits += 1
                    self._reset_boundary_state(cooldown=True)
                    self.last_reason = "grid_rejoin_astar"
                else:
                    self.boundary_astar_clear_frames = 0
                    self._reset_no_path_state()
                    self.last_reason = "grid_waypoint=(%.2f,%.2f)" % (waypoint[0], waypoint[1])
                return action, True, self.last_reason, rollout[2]

        # If A* briefly loses its target cell, retain the last world waypoint
        # when its edge is still clear.  This is the common scan integration
        # case and is preferable to a full safety re-evaluation.
        held_waypoint = self._held_waypoint_if_safe(pose, points)
        if held_waypoint is not None and not self.boundary_active:
            self.last_waypoint = (float(held_waypoint[0]), float(held_waypoint[1]))
            action = self.grid_planner.action_to_waypoint(held_waypoint)
            rollout = self.rollout(action, local_goal, visibility_points)
            if rollout is not None:
                self.last_reason = "grid_waypoint_hold"
                self._reset_no_path_state()
                return action, True, self.last_reason, rollout[2]

        if waypoint is None:
            # A lidar map cannot represent unobserved space.  While committed
            # to a boundary route, preserve that exploration behavior through
            # temporary no-path intervals; do not surrender to SA-PPO's raw
            # reverse/stop command.
            boundary_action = self._boundary_action(local_goal, pose, points, visibility_points)
            if boundary_action is not None:
                return boundary_action, True, self.last_reason, float("nan")
            action = self._no_path_action(
                local_goal, pose, points, visibility_points
            )
            return action, True, self.last_reason, float("nan")

        boundary_action = self._boundary_action(local_goal, pose, points, visibility_points)
        if boundary_action is not None:
            return boundary_action, True, self.last_reason, float("nan")

        action = self.grid_planner.action_to_waypoint(waypoint)
        rollout = self.rollout(action, local_goal, visibility_points)
        if rollout is None:
            heading = math.atan2(float(waypoint[1]), float(waypoint[0]))
            if abs(heading) > math.radians(3.0):
                # A rotate-only command is safe under rollout(), and gives the
                # next grid update a new viewpoint instead of freezing at the
                # edge of the safety buffer.
                self.last_reason = "grid_turn_before_constrained_edge"
                return [0.0, self._turn_action(
                    math.copysign(min(abs(heading), math.pi / 2.0), heading), 0.34
                )], True, self.last_reason, 0.0
            boundary_action = self._boundary_action(local_goal, pose, points, visibility_points)
            if boundary_action is not None:
                return boundary_action, True, self.last_reason, 0.0
            self.last_reason = "grid_no_safe_forward_edge"
            return [0.0, 0.0], True, self.last_reason, 0.0
        self.last_reason = "grid_waypoint=(%.2f,%.2f)" % (waypoint[0], waypoint[1])
        return action, True, self.last_reason, rollout[2]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    env = StageWorld(OBS_SIZE, index=rank, num_env=1)
    controller_mode = str(rospy.get_param(
        "~controller_mode", rospy.get_param("~recovery_mode", "policy_only")
    )).strip().lower()
    if controller_mode not in ("policy_only", "stabilized_recovery", "rl_dwa_guard", "rl_mppi_guard", "rl_grid_guard"):
        raise ValueError("controller_mode must be policy_only, stabilized_recovery, rl_dwa_guard, rl_mppi_guard, or rl_grid_guard")
    recovery = StabilizedRecovery() if controller_mode == "stabilized_recovery" else None
    dwa_guard = DwaSafetyGuard() if controller_mode == "rl_dwa_guard" else None
    mppi_guard = MppiSafetyGuard() if controller_mode == "rl_mppi_guard" else None
    grid_guard = GridSafetyGuard() if controller_mode == "rl_grid_guard" else None
    controller_status = rospy.Publisher("/lste/sappo_controller_status", String, queue_size=1)
    action_shaper = ActionShaper()
    last_controller_signature = None
    last_applied_action = np.zeros(2, dtype=np.float32)

    if rank == 0:
        policy = MLPPolicy(obs_space=OBS_SIZE, action_space=2).cuda()
        checkpoint = os.path.join("policy", "sa_peppo_1650.pth")
        policy.load_state_dict(torch.load(checkpoint, map_location="cuda"))
        policy.eval()
        rospy.loginfo("SA-PPO controller mode=%s checkpoint=%s", controller_mode, checkpoint)
    else:
        policy = None

    while not env.has_goal() and not rospy.is_shutdown():
        env.control_rl_vel([0.0, 0.0])
        rospy.sleep(0.1)

    observation = env.get_laser_observation()
    observation_stack = deque([observation, observation, observation])
    state = [observation_stack, np.asarray(env.get_local_goal()), np.asarray(env.get_self_speed())]
    diagnostic_last_time = -float("inf")

    while not rospy.is_shutdown():
        state_list = comm.gather(state, root=0)
        raw_mean, scaled_action = generate_action_no_sampling(
            env=env, state_list=state_list, policy=policy, action_bound=ACTION_BOUND
        )
        action = comm.scatter(scaled_action, root=0)
        local_goal = env.get_local_goal()
        policy_action = np.asarray(action, dtype=np.float32).copy()
        controller_source = "policy"
        controller_reason = "policy_only"
        predicted_clearance = float("nan")

        # ``env.terminate`` is raised by the legacy RL environment whenever a
        # short goal-radius is reached.  In the normal LSTE pipeline that is a
        # frontier/visual handoff, not task completion; hold briefly at the
        # waypoint until Goal Manager publishes the next segment.  A fixed-goal
        # test has ``allow_intermediate_goals`` disabled and retains the old
        # terminal behavior.
        intermediate_handoff = (
            env.allow_intermediate_goals
            and env.terminate
            and not env.task_done
        )
        if not env.has_goal():
            action = [0.0, 0.0]
            controller_source = "stop"
            controller_reason = "goal_unavailable"
        elif env.terminate and not intermediate_handoff:
            action = [0.0, 0.0]
            controller_source = "stop"
            controller_reason = "task_done_or_complete"
        elif intermediate_handoff:
            action = [0.0, 0.0]
            controller_source = "handoff"
            controller_reason = "intermediate_waypoint_reached_wait_next_goal"
        elif mppi_guard is not None:
            action, guarded, controller_reason, predicted_clearance = mppi_guard.choose(
                raw_mean[0], policy_action, local_goal, env.scan,
                getattr(env, "scan_param", None), env.get_self_state(),
            )
            controller_source = "mppi_guard" if guarded else "policy"
        elif grid_guard is not None:
            action, guarded, controller_reason, predicted_clearance = grid_guard.choose(
                raw_mean[0], policy_action, local_goal, env.scan,
                getattr(env, "scan_param", None), env.get_self_state(),
            )
            controller_source = "grid_guard" if guarded else "policy"
        elif dwa_guard is not None:
            action, guarded, controller_reason, predicted_clearance = dwa_guard.choose(
                raw_mean[0], policy_action, local_goal, env.scan,
                getattr(env, "scan_param", None), env.get_self_state(),
            )
            controller_source = "dwa_guard" if guarded else "policy"
        elif recovery is not None:
            recovery_action = recovery.action(local_goal)
            if recovery_action is not None:
                action = recovery_action
                controller_source = "turn_recovery"
                controller_reason = "heading_behind_robot"

        requested_action = np.asarray(action, dtype=np.float32).reshape(-1)
        hard_stop = (
            controller_source == "stop"
            or "emergency_stop" in controller_reason
            or "no_safe" in controller_reason
        )
        # Ordinary goal-facing turns are allowed to decelerate through the
        # action slew limit.  Only a boundary-end turn or a constrained edge
        # clears linear velocity immediately; this prevents the repeated
        # full-stop/full-speed pattern seen at corridor corners while keeping
        # the close-obstacle safety behavior intact.
        immediate_turn_stop = (
            controller_source == "grid_guard"
            and (
                "boundary_blocked" in controller_reason
                or "turn_around_boundary" in controller_reason
                or "turn_before_constrained" in controller_reason
                or "no_safe" in controller_reason
            )
        )
        action = action_shaper.apply(
            requested_action,
            hard_stop=hard_stop,
            immediate_turn_stop=immediate_turn_stop,
        )
        status_reason = controller_reason
        if grid_guard is not None and controller_source == "grid_guard":
            status_reason = "%s %s" % (status_reason, grid_guard.diagnostic())
        signature = (controller_source, controller_reason)
        applied_array = np.asarray(action, dtype=np.float32)
        linear_drop = float(last_applied_action[0] - applied_array[0])
        if signature != last_controller_signature:
            rospy.loginfo(
                "RL_SAFETY_STATE source=%s reason=%s requested=(%.3f,%.3f) "
                "applied=(%.3f,%.3f) linear_drop=%.3f hard_stop=%s",
                controller_source,
                controller_reason,
                requested_action[0],
                requested_action[1],
                applied_array[0],
                applied_array[1],
                linear_drop,
                hard_stop,
            )
            last_controller_signature = signature
        if hard_stop or linear_drop >= 0.08:
            finite_scan = (
                np.asarray(env.scan, dtype=np.float32)
                if env.scan is not None else np.array([], dtype=np.float32)
            )
            finite_scan = finite_scan[np.isfinite(finite_scan)]
            min_scan = float(finite_scan.min()) if finite_scan.size else float("nan")
            rospy.logwarn(
                "RL_BRAKE_EVENT source=%s reason=%s requested_v=%.3f applied_v=%.3f "
                "linear_drop=%.3f min_scan=%.3f hard_stop=%s",
                controller_source,
                controller_reason,
                requested_action[0],
                applied_array[0],
                linear_drop,
                min_scan,
                hard_stop,
            )
        last_applied_action = applied_array
        controller_status.publish(String(
            data=("source=%s reason=%s requested=(%.3f,%.3f) action=(%.3f,%.3f) "
                  "clearance=%.3f" % (
                controller_source, status_reason,
                requested_action[0], requested_action[1], action[0], action[1],
                predicted_clearance
            ))
        ))

        now = rospy.Time.now().to_sec()
        if rank == 0 and now - diagnostic_last_time >= 1.0:
            raw_action = np.asarray(raw_mean[0], dtype=np.float32)
            scan = np.asarray(env.scan, dtype=np.float32) if env.scan is not None else np.array([])
            finite_scan = scan[np.isfinite(scan)]
            min_range = float(finite_scan.min()) if finite_scan.size else float("nan")
            rospy.loginfo(
                "RL_TEST_DIAG local_goal=(%.2f,%.2f) raw_mean=(%.3f,%.3f) "
                "clipped=(%.3f,%.3f) requested=(%.3f,%.3f) applied=(%.3f,%.3f) source=%s reason=%s "
                "predicted_clearance=%.3f min_scan=%.3f terminate=%s",
                local_goal[0], local_goal[1], raw_action[0], raw_action[1],
                policy_action[0], policy_action[1], requested_action[0], requested_action[1],
                action[0], action[1], controller_source, status_reason,
                predicted_clearance, min_range, env.terminate,
            )
            diagnostic_last_time = now

        env.control_rl_vel(action)
        rospy.sleep(0.1)
        env.get_reward_and_terminate(0)

        next_observation = env.get_laser_observation()
        observation_stack.popleft()
        observation_stack.append(next_observation)
        state = [
            observation_stack,
            np.asarray(env.get_local_goal()),
            np.asarray(env.get_self_speed()),
        ]


if __name__ == "__main__":
    main()
