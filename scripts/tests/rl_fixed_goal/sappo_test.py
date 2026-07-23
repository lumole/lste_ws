#!/usr/bin/env python3
"""SA-PPO fixed-goal runner with test-only recovery modes.

This intentionally leaves rl_navigation/sappo_pure.py unchanged. Both modes use
the production checkpoint and the production StageWorld lidar/goal encoder.
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
        if abs(heading) > math.radians(18.0):
            return [0.0, 0.5 if heading > 0.0 else -0.5]
        # Near a long wall, a small cell-to-cell lateral fluctuation must not
        # continuously turn the robot into that wall. Larger heading changes
        # still use the rotate-in-place branch above.
        angular = 0.0 if abs(heading) < math.radians(12.0) else float(
            np.clip(1.1 * heading, -0.5, 0.5)
        )
        return [min(0.55, max(0.10, distance)), angular]


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
        self.grid_planner = LocalGridPlanner(half_extent=10.0)
        self.boundary_active = False
        self.boundary_heading_world = 0.0
        self.boundary_hit_distance = float("inf")
        self.boundary_start = None
        self.goal_visibility_frames = 0

    def _turn_around_boundary_end(self, local_goal, pose, points):
        """Choose a verified side around a newly encountered boundary end.

        The chosen directions are expressed in the current robot frame.  They
        deliberately exclude a U-turn: boundary following should look through
        an opening beside the wall, not retreat from the unexplored route.
        """
        if pose is None:
            return False
        _, _, yaw = (float(value) for value in pose)
        goal_bearing = math.atan2(float(local_goal[1]), float(local_goal[0]))
        candidates = (-math.pi / 2.0, math.pi / 2.0)
        viable = []
        for angle in candidates:
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            if self.grid_planner._edge_is_clear(
                    np.zeros(2, dtype=np.float32), direction * 0.90, points
            ):
                # Prefer the route facing the goal, then a deterministic right
                # hand tie-breaker.  This is used only after the current
                # boundary heading becomes blocked.
                score = abs(wrap_angle(goal_bearing - angle)) + 0.01 * (angle > 0.0)
                viable.append((score, angle))
        if not viable:
            return False
        _, angle = min(viable)
        self.boundary_heading_world = wrap_angle(yaw + angle)
        self.last_reason = "bug2_turn_around_boundary_end"
        return True

    def _boundary_action(self, local_goal, pose, points, visibility_points=None):
        if pose is None:
            return None
        x, y, yaw = (float(value) for value in pose)
        goal_distance = float(np.hypot(local_goal[0], local_goal[1]))
        direct_goal = np.asarray(local_goal, dtype=np.float32)
        if goal_distance > self.grid_planner.half_extent:
            direct_goal *= self.grid_planner.half_extent / goal_distance

        if self.boundary_active:
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
            direct_visible = self.grid_planner._edge_is_clear(
                np.zeros(2, dtype=np.float32), direct_goal, visibility
            )
            self.goal_visibility_frames = self.goal_visibility_frames + 1 if direct_visible else 0
            if displacement >= 0.75 and self.goal_visibility_frames >= 3:
                self.boundary_active = False
                self.boundary_start = None
                self.goal_visibility_frames = 0
                self.last_reason = "bug2_leave_boundary_direct_route_visible"
                return None
            error = wrap_angle(self.boundary_heading_world - yaw)
            if abs(error) > math.radians(12.0):
                return [0.0, 0.5 if error > 0.0 else -0.5]
            # A locked outward heading can meet a newly observed obstacle
            # around a corner. Turn for a new scan instead of driving into the
            # safety buffer or handing control back to the reverse-prone policy.
            local_forward = np.array([1.0, 0.0], dtype=np.float32)
            if not self.grid_planner._edge_is_clear(
                    np.zeros(2, dtype=np.float32), local_forward * 0.55, points
            ):
                if self._turn_around_boundary_end(local_goal, pose, points):
                    error = wrap_angle(self.boundary_heading_world - yaw)
                    return [0.0, 0.5 if error > 0.0 else -0.5]
                # No lateral opening is currently observable.  Keep scanning
                # in one direction rather than alternating around the old
                # heading, which had produced an in-place oscillation.
                self.boundary_heading_world = wrap_angle(yaw - math.pi / 2.0)
                self.last_reason = "bug2_boundary_blocked_scan_right"
                return [0.0, -0.5]
            self.last_reason = "bug2_follow_boundary"
            return [0.35, 0.0]

        if points.size == 0:
            return None
        ranges = np.hypot(points[:, 0], points[:, 1])
        near = points[ranges < 1.3]
        if near.shape[0] < 4:
            return None
        # In the usual failure case the robot is already moving in a useful
        # free direction, but the direct route to the goal becomes occluded by
        # a wall. Preserve that direction first.  Points are in *robot-local*
        # coordinates, therefore [1, 0] is the current forward edge regardless
        # of the robot's world yaw.  This sends the fixed test south toward the
        # known wall endpoint instead of incorrectly rotating west into it.
        if self.grid_planner._edge_is_clear(
                np.zeros(2, dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32), points
        ):
            self.boundary_heading_world = yaw
            self.boundary_hit_distance = goal_distance
            self.boundary_start = (x, y)
            self.boundary_active = True
            self.goal_visibility_frames = 0
            self.last_reason = "bug2_enter_boundary_keep_clear_heading"
            return self._boundary_action(local_goal, pose, points)

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
        self.boundary_active = True
        self.goal_visibility_frames = 0
        self.last_reason = "bug2_enter_boundary"
        return self._boundary_action(local_goal, pose, points)

    def choose(self, raw_action, policy_action, local_goal, scan, scan_param, pose):
        points = self.obstacle_points(scan, scan_param, pose)
        visibility_points = RollingObstacleMap._scan_points(scan, scan_param, max_range=8.0)
        waypoint = self.grid_planner.waypoint(points, local_goal)
        if waypoint is None:
            # A lidar map cannot represent unobserved space.  While committed
            # to a boundary route, preserve that exploration behavior through
            # temporary no-path intervals; do not surrender to SA-PPO's raw
            # reverse/stop command.
            boundary_action = self._boundary_action(local_goal, pose, points, visibility_points)
            if boundary_action is not None:
                return boundary_action, True, self.last_reason, float("nan")
            action, _, _, clearance = super().choose(
                raw_action, policy_action, local_goal, scan, scan_param, pose,
            )
            self.last_reason = "grid_no_path_" + self.last_reason
            return action, True, self.last_reason, clearance

        boundary_action = self._boundary_action(local_goal, pose, points, visibility_points)
        if boundary_action is not None:
            return boundary_action, True, self.last_reason, float("nan")

        action = self.grid_planner.action_to_waypoint(waypoint)
        rollout = self.rollout(action, local_goal, points)
        if rollout is None:
            heading = math.atan2(float(waypoint[1]), float(waypoint[0]))
            if abs(heading) > math.radians(3.0):
                # A rotate-only command is safe under rollout(), and gives the
                # next grid update a new viewpoint instead of freezing at the
                # edge of the safety buffer.
                self.last_reason = "grid_turn_before_constrained_edge"
                return [0.0, 0.5 if heading > 0.0 else -0.5], True, self.last_reason, 0.0
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
    controller_status = rospy.Publisher("/rl_fixed_goal_test/controller_status", String, queue_size=1)

    if rank == 0:
        policy = MLPPolicy(obs_space=OBS_SIZE, action_space=2).cuda()
        checkpoint = os.path.join("policy", "sa_peppo_1650.pth")
        policy.load_state_dict(torch.load(checkpoint, map_location="cuda"))
        policy.eval()
        rospy.loginfo("RL fixed-goal test mode=%s checkpoint=%s", controller_mode, checkpoint)
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

        if not env.has_goal() or env.terminate:
            action = [0.0, 0.0]
            controller_source = "stop"
            controller_reason = "goal_unavailable_or_complete"
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

        controller_status.publish(String(
            data=("source=%s reason=%s action=(%.3f,%.3f) clearance=%.3f" % (
                controller_source, controller_reason, action[0], action[1], predicted_clearance
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
                "clipped=(%.3f,%.3f) applied=(%.3f,%.3f) source=%s reason=%s "
                "predicted_clearance=%.3f min_scan=%.3f terminate=%s",
                local_goal[0], local_goal[1], raw_action[0], raw_action[1],
                policy_action[0], policy_action[1], action[0], action[1],
                controller_source, controller_reason, predicted_clearance, min_range, env.terminate,
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
