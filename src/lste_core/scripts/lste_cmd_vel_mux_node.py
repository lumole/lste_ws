#!/usr/bin/env python3
import threading

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class CmdVelMuxNode:
    def __init__(self):
        rospy.init_node("lste_cmd_vel_mux")
        self.mode = "sappo"
        self.lock = threading.Lock()
        self.output_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.sappo_sub = rospy.Subscriber(
            "/lste/cmd_vel/sappo", Twist, self.on_sappo_cmd, queue_size=1
        )
        self.teleop_sub = rospy.Subscriber(
            "/lste/cmd_vel/teleop", Twist, self.on_teleop_cmd, queue_size=1
        )
        self.mode_sub = rospy.Subscriber(
            "/lste/controller_mode", String, self.on_mode, queue_size=1
        )
        rospy.loginfo("Command velocity mux ready")

    def on_mode(self, message):
        mode = message.data.strip().lower()
        if mode not in ("sappo", "teleop"):
            return
        with self.lock:
            changed = mode != self.mode
            self.mode = mode
        if changed:
            self.output_pub.publish(Twist())
            rospy.loginfo("Command velocity source switched to %s", mode)

    def forward(self, source, message):
        with self.lock:
            active = source == self.mode
        if active:
            self.output_pub.publish(message)

    def on_sappo_cmd(self, message):
        self.forward("sappo", message)

    def on_teleop_cmd(self, message):
        self.forward("teleop", message)


if __name__ == "__main__":
    CmdVelMuxNode()
    rospy.spin()
