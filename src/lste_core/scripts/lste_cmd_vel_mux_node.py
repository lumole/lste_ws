#!/usr/bin/env python3
import copy
import math
import threading

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from teb_local_planner.msg import FeedbackMsg


class CmdVelMuxNode:
    def __init__(self):
        rospy.init_node("lste_cmd_vel_mux")
        self.mode = str(rospy.get_param("~initial_mode", "teb")).strip().lower()
        if self.mode not in ("sappo", "teleop", "teb"):
            self.mode = "teb"
        forward_only = rospy.get_param("~teb_forward_only", True)
        self.teb_forward_only = str(forward_only).strip().lower() in (
            "1", "true", "yes", "on",
        )
        self.teb_angular_sign_switch_threshold = max(
            0.0, float(rospy.get_param("~teb_angular_sign_switch_threshold", 0.12))
        )
        self.teb_angular_deadband = max(
            0.0, float(rospy.get_param("~teb_angular_deadband", 0.02))
        )
        self.teb_last_angular_sign = 0
        self.teb_last_output_angular = 0.0
        self.teb_micro_hold_cycles = 0
        # A micro reversal is normally TEB quantization around a stable
        # trajectory.  Holding the previous output for a few control cycles
        # avoids turning that noise into a zero-velocity pulse at a corner.
        self.teb_micro_hold_limit = 4
        # TEB can encode the first point of an in-place turn as a tiny reverse
        # velocity with near-zero angular velocity. Keep the planner's
        # upcoming turn intent as an interface-level handoff hint so the mux
        # does not emit a reverse -> zero -> rotate transition.
        self.teb_turn_hint = 0.0
        self.teb_turn_hint_wall = 0.0
        self.teb_turn_transition_count = 0
        self.teb_filter_events = 0
        self.teb_reverse_clamp_count = 0
        self.task_done = False
        self.navigation_hold = False
        self.lock = threading.Lock()
        self.output_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        # Forward-collision velocity governor: a hard safety invariant at the
        # sole /cmd_vel publisher.  Whatever TEB, the goal, or the global path
        # command, the commanded forward speed is capped so the robot can
        # always stop within the measured lidar clearance ahead:
        #     v_max = sqrt(2 * a_brake * max(0, clearance - min_gap))
        # This is not a planner tuning knob; it guarantees the vehicle never
        # dives toward a wall it can already see, independent of the goal or
        # the global plan.
        self.scan_topic = rospy.get_param("~scan_topic", "/pro3/rlscan")
        self.governor_decel = max(
            0.05, float(rospy.get_param("~governor_decel", 0.60))
        )
        self.governor_min_gap = max(
            0.05, float(rospy.get_param("~governor_min_gap", 0.45))
        )
        self.governor_max_speed = max(
            0.0, float(rospy.get_param("~governor_max_speed", 0.50))
        )
        self.scan_forward_min = float("inf")
        self.scan_sub = rospy.Subscriber(
            self.scan_topic, LaserScan, self.on_scan, queue_size=1
        )
        self.sappo_sub = rospy.Subscriber(
            "/lste/cmd_vel/sappo", Twist, self.on_sappo_cmd, queue_size=1
        )
        self.teleop_sub = rospy.Subscriber(
            "/lste/cmd_vel/teleop", Twist, self.on_teleop_cmd, queue_size=1
        )
        self.teb_sub = rospy.Subscriber(
            "/lste/cmd_vel/teb", Twist, self.on_teb_cmd, queue_size=1
        )
        self.teb_feedback_sub = rospy.Subscriber(
            "/move_base/TebLocalPlannerROS/teb_feedback",
            FeedbackMsg,
            self.on_teb_feedback,
            queue_size=1,
        )
        self.mode_sub = rospy.Subscriber(
            "/lste/controller_mode", String, self.on_mode, queue_size=1
        )
        self.task_done_sub = rospy.Subscriber(
            "/lste/task_done", Bool, self.on_task_done, queue_size=1
        )
        self.navigation_hold_sub = rospy.Subscriber(
            rospy.get_param("~navigation_hold_topic", "/lste/navigation_hold"),
            Bool,
            self.on_navigation_hold,
            queue_size=1,
        )
        rospy.loginfo(
            "Command velocity mux ready: initial_mode=%s teb_forward_only=%s "
            "teb_angular_switch=%.3f teb_angular_deadband=%.3f",
            self.mode,
            self.teb_forward_only,
            self.teb_angular_sign_switch_threshold,
            self.teb_angular_deadband,
        )

    def reset_teb_filter(self):
        self.teb_last_angular_sign = 0
        self.teb_last_output_angular = 0.0
        self.teb_micro_hold_cycles = 0
        self.teb_turn_hint = 0.0
        self.teb_turn_hint_wall = 0.0

    def filter_teb_angular(self, command):
        """Drop only quantization-scale opposite steering corrections.

        TEB remains responsible for the trajectory and obstacle response. The
        mux only prevents a straight corridor from alternating between tiny
        left/right commands when the map path moves by one cell. A meaningful
        opposite turn still passes as soon as it exceeds the configured
        threshold.
        """
        angular = float(command.angular.z)
        magnitude = abs(angular)
        if self.teb_angular_deadband > 0.0 and magnitude <= self.teb_angular_deadband:
            command.angular.z = 0.0
            self.teb_last_output_angular = 0.0
            self.teb_micro_hold_cycles = 0
            return command
        sign = 1 if angular > 0.0 else -1 if angular < 0.0 else 0
        if (
            sign
            and self.teb_last_angular_sign
            and sign != self.teb_last_angular_sign
            and magnitude < self.teb_angular_sign_switch_threshold
        ):
            self.teb_filter_events += 1
            if (
                self.teb_micro_hold_cycles < self.teb_micro_hold_limit
                and abs(self.teb_last_output_angular) > self.teb_angular_deadband
            ):
                # Preserve the current turn direction briefly.  Returning an
                # immediate zero here makes a turn-only trajectory look like
                # repeated braking, because /cmd_vel is the mux output that
                # the robot actually receives.
                command.angular.z = self.teb_last_output_angular
                self.teb_micro_hold_cycles += 1
            else:
                command.angular.z = 0.0
                self.teb_last_output_angular = 0.0
                self.teb_micro_hold_cycles = 0
            rospy.loginfo_throttle(
                2.0,
                "TEB micro-reversal held/suppressed (count=%d raw=%.3f "
                "output=%.3f threshold=%.3f hold=%d/%d)",
                self.teb_filter_events,
                angular,
                command.angular.z,
                self.teb_angular_sign_switch_threshold,
                self.teb_micro_hold_cycles,
                self.teb_micro_hold_limit,
            )
            return command
        if sign:
            self.teb_last_angular_sign = sign
        self.teb_last_output_angular = float(command.angular.z)
        self.teb_micro_hold_cycles = 0
        return command

    def on_mode(self, message):
        mode = message.data.strip().lower()
        if mode not in ("sappo", "teleop", "teb"):
            return
        with self.lock:
            changed = mode != self.mode
            self.mode = mode
        if changed:
            self.reset_teb_filter()
            self.output_pub.publish(Twist())
            rospy.loginfo("Command velocity source switched to %s", mode)

    def on_scan(self, message):
        """Track the minimum range in the forward sector for the governor."""
        forward = float("inf")
        angle = float(message.angle_min)
        for i, value in enumerate(message.ranges):
            if math.isfinite(value) and value > 0.01 and abs(angle) <= math.radians(25.0):
                forward = min(forward, float(value))
            angle += float(message.angle_increment)
        self.scan_forward_min = forward

    def _governed_linear(self, linear):
        """Cap forward speed by the braking distance available ahead.

        Uses the standard kinematic stopping bound: from speed v the robot
        stops within v^2 / (2*a).  Enforcing ``v <= sqrt(2*a*(c-gap))`` means
        the robot can always halt before the nearest obstacle it can see, no
        matter what upstream layer commanded the motion.
        """
        if linear <= 0.0:
            return linear
        clearance = self.scan_forward_min
        if not math.isfinite(clearance) or clearance <= 0.0:
            return linear
        margin = max(0.0, float(clearance) - self.governor_min_gap)
        max_v = min(
            self.governor_max_speed,
            math.sqrt(2.0 * self.governor_decel * margin),
        )
        return min(linear, max_v) if linear > max_v else linear

    def forward(self, source, message):
        with self.lock:
            active = (
                source == self.mode
                and not self.task_done
                and not self.navigation_hold
            )
        if not active:
            return
        # Apply the forward-collision governor to every selected source (TEB,
        # SA-PPO, teleop) at the actuator boundary.  ``message`` is copied so
        # the source's own command object is never mutated.
        governed = copy.deepcopy(message)
        governed.linear.x = self._governed_linear(float(governed.linear.x))
        self.output_pub.publish(governed)

    def on_task_done(self, message):
        with self.lock:
            changed = bool(message.data) != self.task_done
            self.task_done = bool(message.data)
        if self.task_done:
            # Completion is a safety gate, not a suggestion to the currently
            # selected controller. Stop both autonomous and keyboard commands.
            self.output_pub.publish(Twist())
            self.reset_teb_filter()
            if changed:
                rospy.loginfo("Command velocity output locked at zero: task complete")
        elif changed:
            rospy.loginfo("Command velocity output unlocked: new task")

    def on_navigation_hold(self, message):
        """Gate motion for a mission observation window without changing goal."""
        with self.lock:
            changed = bool(message.data) != self.navigation_hold
            self.navigation_hold = bool(message.data)
        if not changed:
            return
        if self.navigation_hold:
            self.output_pub.publish(Twist())
            self.reset_teb_filter()
        rospy.loginfo(
            "Command velocity observation gate %s",
            "closed" if self.navigation_hold else "opened",
        )

    def on_sappo_cmd(self, message):
        self.forward("sappo", message)

    def on_teleop_cmd(self, message):
        self.forward("teleop", message)

    def on_teb_cmd(self, message):
        command = copy.deepcopy(message)
        if self.teb_forward_only and message.linear.x < 0.0:
            # TEB enforces a small positive reverse lower bound internally
            # (typically 0.01 m/s), even when the launch file requests a
            # near-zero backwards limit. On this differential-drive base that
            # tiny reverse command competes with the turn optimizer and causes
            # the observed left/right oscillation at a goal behind the robot.
            # Clamping only the mux output preserves TEB's angular command and
            # gives the vehicle an in-place turn without changing the planner.
            command.linear.x = 0.0
            # A negative first sample is TEB's way of entering a turn when
            # forward motion is infeasible. The next points already contain
            # the turn, so use that fresh planner intent to avoid one zero
            # cycle before rotation starts.
            with self.lock:
                hint_fresh = (
                    self.teb_turn_hint != 0.0
                    and rospy.get_time() - self.teb_turn_hint_wall <= 0.25
                )
                if hint_fresh and abs(command.angular.z) <= self.teb_angular_deadband:
                    command.angular.z = self.teb_turn_hint
                    self.teb_turn_transition_count += 1
                    rospy.loginfo_throttle(
                        2.0,
                        "TEB turn transition kept continuous (count=%d hint=%.3f)",
                        self.teb_turn_transition_count,
                        self.teb_turn_hint,
                    )
            self.teb_reverse_clamp_count += 1
            rospy.logwarn_throttle(
                2.0,
                "TEB reverse command clamped to zero (count=%d raw=%.3f)",
                self.teb_reverse_clamp_count,
                message.linear.x,
            )
        command = self.filter_teb_angular(command)
        self.forward("teb", command)

    def on_teb_feedback(self, message):
        """Cache the next angular motion in TEB's selected trajectory.

        Feedback is only an adapter input. TEB remains the sole source of the
        turn direction, and the hint expires immediately when feedback stops.
        """
        selected_index = int(message.selected_trajectory_idx)
        trajectories = list(message.trajectories)
        if selected_index < 0 or selected_index >= len(trajectories):
            with self.lock:
                self.teb_turn_hint = 0.0
                self.teb_turn_hint_wall = 0.0
            return
        points = list(trajectories[selected_index].trajectory)
        hint = 0.0
        for point in points[1:]:
            angular = float(point.velocity.angular.z)
            if abs(angular) > 0.05:
                hint = angular
                break
        with self.lock:
            self.teb_turn_hint = hint
            self.teb_turn_hint_wall = rospy.get_time() if hint else 0.0


if __name__ == "__main__":
    CmdVelMuxNode()
    rospy.spin()
