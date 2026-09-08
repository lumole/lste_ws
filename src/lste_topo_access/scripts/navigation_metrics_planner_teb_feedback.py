"""TEB trajectory and obstacle telemetry for the navigation observer."""

import math
import time


class NavigationMetricsTebFeedbackMixin:
    """Observe selected TEB trajectories without changing controller behavior."""

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
        """Record the selected TEB trajectory and sustained zero-velocity evidence."""
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
