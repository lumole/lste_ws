#!/usr/bin/env python3
import threading

import rospy
from std_msgs.msg import Empty, String


VALID_MODES = ("sappo", "teleop")


class ControllerSwitchNode:
    def __init__(self):
        rospy.init_node("lste_controller_switch")
        initial_mode = rospy.get_param("~initial_mode", "sappo")
        self.mode = initial_mode if initial_mode in VALID_MODES else "sappo"
        self.lock = threading.Lock()
        self.mode_pub = rospy.Publisher("/lste/controller_mode", String, queue_size=1, latch=True)
        self.toggle_sub = rospy.Subscriber(
            "/lste/controller_toggle", Empty, self.on_toggle, queue_size=1
        )
        self.select_sub = rospy.Subscriber(
            "/lste/controller_select", String, self.on_select, queue_size=1
        )
        self.publish_mode()
        rospy.loginfo(
            "Controller switch ready: mode=%s, Gazebo button or Ctrl+Shift+K", self.mode
        )

    def publish_mode(self):
        self.mode_pub.publish(String(data=self.mode))

    def set_mode(self, mode):
        if mode not in VALID_MODES:
            rospy.logwarn("Ignoring invalid controller mode: %s", mode)
            return
        with self.lock:
            if mode == self.mode:
                self.publish_mode()
                return
            previous = self.mode
            self.mode = mode
            self.publish_mode()
        rospy.loginfo("Controller switched: %s -> %s", previous, mode)

    def on_toggle(self, _message):
        target = "teleop" if self.mode == "sappo" else "sappo"
        self.set_mode(target)

    def on_select(self, message):
        self.set_mode(message.data.strip().lower())


if __name__ == "__main__":
    ControllerSwitchNode()
    rospy.spin()
