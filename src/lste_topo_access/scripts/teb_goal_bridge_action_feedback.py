"""MoveBase feedback accounting for one active TEB bridge action."""

import copy
import math
import time


class TebGoalBridgeActionFeedbackMixin:
    def on_feedback(self, generation, feedback):
        """Record geometric and Navfn progress without changing route policy."""
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
