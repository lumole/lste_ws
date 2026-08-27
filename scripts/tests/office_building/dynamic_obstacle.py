#!/usr/bin/env python3
"""Move the Level 4 obstacle along a deterministic corridor crossing."""

from __future__ import annotations

import argparse
import math

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, SetModelStateRequest


def triangular_wave(t: float, period: float, low: float, high: float) -> float:
    phase = (t % period) / period
    if phase < 0.5:
        return low + (high - low) * phase * 2.0
    return high - (high - low) * (phase - 0.5) * 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="benchmark_dynamic_obstacle")
    parser.add_argument("--period", type=float, default=24.0)
    parser.add_argument("--x-low", type=float, default=8.0)
    parser.add_argument("--x-high", type=float, default=15.5)
    parser.add_argument("--y", type=float, default=11.0)
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args()

    rospy.init_node("office_building_dynamic_obstacle")
    rospy.wait_for_service("/gazebo/set_model_state", timeout=30.0)
    set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    state = ModelState()
    state.model_name = args.model
    state.reference_frame = "world"
    start = rospy.Time.now().to_sec()
    rate = rospy.Rate(max(1.0, args.rate))
    update_count = 0
    rospy.loginfo(
        "dynamic obstacle started model=%s period=%.1fs x=[%.2f,%.2f] y=%.2f seed=20260827",
        args.model,
        args.period,
        args.x_low,
        args.x_high,
        args.y,
    )
    while not rospy.is_shutdown():
        elapsed = max(0.0, rospy.Time.now().to_sec() - start)
        state.pose.position.x = triangular_wave(elapsed, max(2.0, args.period), args.x_low, args.x_high)
        state.pose.position.y = args.y
        state.pose.position.z = 0.6
        state.pose.orientation.w = 1.0
        update_count += 1
        try:
            response = set_model_state(SetModelStateRequest(model_state=state))
            rospy.loginfo(
                "dynamic obstacle update count=%d elapsed=%.3f x=%.3f y=%.3f success=%s status=%s",
                update_count,
                elapsed,
                state.pose.position.x,
                state.pose.position.y,
                response.success,
                response.status_message,
            )
            if not response.success:
                rospy.logwarn_throttle(5.0, "dynamic obstacle update failed: %s", response.status_message)
        except rospy.ServiceException as exc:
            rospy.logerr(
                "dynamic obstacle service error count=%d elapsed=%.3f x=%.3f y=%.3f error=%s",
                update_count,
                elapsed,
                state.pose.position.x,
                state.pose.position.y,
                exc,
            )
            rospy.logwarn_throttle(5.0, "dynamic obstacle service error: %s", exc)
        rate.sleep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
