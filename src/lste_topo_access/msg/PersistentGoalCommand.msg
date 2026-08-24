# Explicit goal ownership for the persistent Navfn/TEB stream.
#
# Do not use PoseStamped.header.seq for this identity: ROS publishers own and
# rewrite that field as a transport sequence counter.

uint8 KIND_CLEAR=0
uint8 KIND_MISSION=1
uint8 KIND_TARGET_REQUEST=2
uint8 KIND_TARGET_INSTALLED=3

uint32 transaction_id
uint8 kind
geometry_msgs/PoseStamped goal
