"""Runtime state initialization for the TEB turn supervisor."""

from contextlib import nullcontext

import tf
from geometry_msgs.msg import Twist

from teb_turn_supervisor_contract import STATE_PASS_THROUGH


def initialize_turn_supervisor_state(supervisor):
    """Create all mutable state before ROS callbacks can observe it."""
    supervisor.lock = nullcontext()
    supervisor.state = STATE_PASS_THROUGH
    supervisor.task_done = False
    supervisor.navigation_hold = False
    # Latest ROS hold sample is ingress state, not mission ownership. The
    # timer consumes it after lifecycle events so an old transaction event
    # cannot overwrite a newer release.
    supervisor.latest_navigation_hold_sample = None
    supervisor.pose = None
    supervisor.scan_minimum = float("inf")
    supervisor.scan_monotonic = 0.0
    supervisor.latest_goal = None
    supervisor.active_action_goal = None
    supervisor.active_action_source_goal = None
    supervisor.active_action_route_kind = ""
    supervisor.active_action_source = "unknown"
    supervisor.active_action_priority = 0
    supervisor.active_action_goal_transaction_id = 0
    supervisor.active_action_target_track_id = ""
    supervisor.active_action_transaction_id = 0
    supervisor.active_action_route_id = 0
    supervisor.active_action_map_epoch = None
    supervisor.active_action_graph_transaction_id = 0
    supervisor.active_action_generation = 0
    supervisor.active_action_identity = None
    supervisor.active_action = False
    supervisor.latest_navfn_plan = None
    supervisor.pre_turn_checked_identity = None
    supervisor.latest_planner_command = Twist()
    supervisor.latest_planner_command_wall = 0.0
    supervisor.planner_command_sequence = 0
    supervisor.planner_command_transaction_id = 0
    supervisor.planner_command_route_id = 0
    supervisor.planner_command_map_epoch = None
    supervisor.planner_command_graph_transaction_id = 0
    supervisor.planner_command_action_generation = 0
    supervisor.planner_command_action_identity = None
    supervisor.planner_contract_valid = False
    supervisor.planner_contract_state = None
    supervisor.planner_contract_reason = "startup"
    supervisor.planner_contract_identity = None
    supervisor.planner_contract_command = Twist()
    supervisor.planner_contract_sequence = 0
    supervisor.planner_contract_wall = 0.0
    supervisor.planner_contract_awaiting_feedback = True
    supervisor.planner_contract_pending = None
    supervisor.output_sequence = 0
    supervisor.last_output_transaction_id = 0
    supervisor.last_output_decision = "startup"
    supervisor.latest_trajectory_command = Twist()
    supervisor.latest_trajectory_command_wall = 0.0
    # FeedbackMsg has no route identity. It is valid only after a planner
    # command for the current owner has re-armed this stream.
    supervisor.trajectory_feedback_valid = False
    supervisor.trajectory_feedback_invalid_reason = "no_feedback"
    supervisor.trajectory_feedback_requires_fresh_planner_command = True
    supervisor.trajectory_feedback_identity = None
    supervisor.trajectory_feedback_invalidation_count = 0
    supervisor.trajectory_feedback_period_ema = None
    supervisor.trajectory_zero_started_wall = 0.0
    supervisor.trajectory_continuity_events = 0
    supervisor.trajectory_continuity_goal_distance = None
    supervisor.trajectory_continuity_sharp_entry_identity = None
    supervisor.trajectory_continuity_sharp_entry_suppressions = 0
    supervisor.latest_route_kind = ""
    supervisor.latest_route_id = 0
    supervisor.latest_map_epoch = None
    supervisor.latest_graph_transaction_id = 0
    supervisor.latest_intent_source = "unknown"
    supervisor.latest_intent_priority = 0
    supervisor.latest_intent_goal = None
    supervisor.latest_target_track_id = ""
    supervisor.turn_target_yaw = None
    supervisor.turn_goal_xy = None
    supervisor.turn_key = None
    supervisor.turn_action_identity = None
    supervisor.completed_turn_key = None
    supervisor.turn_velocity = 0.0
    supervisor.turn_started_wall = 0.0
    supervisor.turn_pending_identity = None
    supervisor.turn_pending_since_wall = 0.0
    supervisor.turn_initial_abs_error = 0.0
    supervisor.turn_abs_rotation = 0.0
    supervisor.turn_rotation_cap = 0.0
    supervisor.turn_settle_until_wall = 0.0
    supervisor.turn_settle_yaw = None
    supervisor.stalled_route_candidate_identity = None
    supervisor.stalled_route_candidate_since_wall = 0.0
    supervisor.stalled_route_ready_identity = None
    supervisor.stalled_route_completed_identity = None
    supervisor.turn_capped_releases = 0
    supervisor.turn_count = 0
    supervisor.turn_completed_count = 0
    supervisor.turn_released_count = 0
    supervisor.last_status_wall = 0.0
    supervisor.tf_listener = tf.TransformListener()
