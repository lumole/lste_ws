# Immutable completion contract for one frontier route action.
#
# A PoseStamped alone identifies an endpoint only geometrically. Adjacent
# frontier actions can be closer than a controller acceptance radius, so the
# route ID is the authoritative identity at the bridge/frontier boundary.

uint32 route_id
uint32 action_generation
string route_kind
geometry_msgs/PoseStamped goal
