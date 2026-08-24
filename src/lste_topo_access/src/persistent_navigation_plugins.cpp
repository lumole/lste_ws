#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

#include <pluginlib/class_list_macros.h>
#include <tf/transform_listener.h>

#include "lste_topo_access/persistent_navigation_plugins.h"

namespace lste_topo_access {

PersistentTebLocalPlanner::PersistentTebLocalPlanner()
    : initialized_(false),
      persistent_execution_(false),
      task_done_(false),
      has_installed_target_goal_(false),
      has_reported_target_goal_(false),
      has_reported_frontier_goal_(false),
      reported_frontier_route_version_(0),
      target_goal_epsilon_(0.05),
      frontier_endpoint_equivalence_distance_(0.05),
      plan_equivalence_distance_(0.25),
      plan_equivalence_prefix_distance_(0.35),
      plan_received_count_(0),
      plan_equivalent_count_(0),
      plan_installed_count_(0),
      route_version_(0),
      route_geometry_hash_(0),
      installed_target_transaction_(0) {}

void PersistentTebLocalPlanner::initialize(
    std::string name, tf2_ros::Buffer* tf,
    costmap_2d::Costmap2DROS* costmap_ros) {
  if (initialized_) {
    ROS_WARN("PersistentTebLocalPlanner is already initialized");
    return;
  }

  ros::NodeHandle private_nh("~/" + name);
  ros::param::param("/move_base/persistent_execution", persistent_execution_,
                    false);
  std::string task_done_topic;
  std::string installed_target_goal_topic;
  std::string installed_target_command_topic;
  std::string target_approach_topic;
  std::string frontier_endpoint_topic;
  std::string plan_event_topic;
  std::string target_plan_result_topic;
  ros::param::param<std::string>("/move_base/persistent_task_done_topic",
                                 task_done_topic, "/lste/task_done");
  ros::param::param<std::string>(
      "/move_base/persistent_installed_target_goal_topic",
      installed_target_goal_topic,
      "/lste/persistent_execution/installed_target_goal");
  ros::param::param<std::string>(
      "/move_base/persistent_installed_target_command_topic",
      installed_target_command_topic,
      "/lste/persistent_execution/installed_target_command");
  ros::param::param<std::string>(
      "/move_base/persistent_target_approach_topic", target_approach_topic,
      "/lste/persistent_execution/target_approach");
  ros::param::param<std::string>(
      "/move_base/persistent_frontier_endpoint_topic", frontier_endpoint_topic,
      "/lste/persistent_execution/frontier_endpoint_reached");
  ros::param::param<std::string>(
      "/move_base/persistent_plan_event_topic", plan_event_topic,
      "/lste/persistent_execution/plan_event");
  ros::param::param<std::string>(
      "/move_base/persistent_target_plan_result_topic", target_plan_result_topic,
      "/lste/persistent_execution/target_plan_result");
  ros::param::param("/move_base/streaming_goal_epsilon", target_goal_epsilon_,
                    0.05);
  target_goal_epsilon_ = std::max(0.001, target_goal_epsilon_);
  ros::param::param("/move_base/persistent_plan_equivalence_distance",
                    plan_equivalence_distance_, 0.25);
  plan_equivalence_distance_ = std::max(0.02, plan_equivalence_distance_);
  // Navfn rounds an endpoint to occupancy-grid cells. On a 0.1 m map the
  // same geometric frontier can therefore alternate between diagonal cells
  // 0.141 m apart as SLAM updates. A visual target remains exact at
  // target_goal_epsilon_, but frontier plan equivalence must absorb that
  // lattice representation error or it will reset TEB's elastic band every
  // time the global planner recomputes the unchanged corridor.
  const double map_cell_diagonal = std::sqrt(2.0) *
      costmap_ros->getCostmap()->getResolution();
  frontier_endpoint_equivalence_distance_ = std::max(
      target_goal_epsilon_, map_cell_diagonal + 0.01);
  // Navfn begins each refreshed plan at the latest robot pose. The first few
  // decimetres therefore move with normal odometry/SLAM correction even when
  // the route corridor ahead is unchanged. Local TEB owns that immediate
  // horizon, so excluding it from global-route equivalence avoids resetting
  // an elastic band for a mere resample while still comparing every later
  // branch and the exact endpoint.
  ros::param::param("/move_base/persistent_plan_equivalence_prefix_distance",
                    plan_equivalence_prefix_distance_, 0.35);
  plan_equivalence_prefix_distance_ =
      std::max(0.0, plan_equivalence_prefix_distance_);

  // Use the original plugin name: all existing TebLocalPlannerROS parameters
  // remain valid, so this wrapper changes lifecycle semantics only.
  teb_.initialize("TebLocalPlannerROS", tf, costmap_ros);
  task_done_subscriber_ = private_nh.subscribe(
      task_done_topic, 1, &PersistentTebLocalPlanner::onTaskDone, this);
  installed_target_command_subscriber_ = private_nh.subscribe(
      installed_target_command_topic, 1,
      &PersistentTebLocalPlanner::onInstalledTargetCommand, this);
  target_approach_publisher_ = private_nh.advertise<geometry_msgs::PoseStamped>(
      target_approach_topic, 1, false);
  target_approach_result_publisher_ = private_nh.advertise<std_msgs::String>(
      target_plan_result_topic, 10, false);
  frontier_endpoint_publisher_ = private_nh.advertise<geometry_msgs::PoseStamped>(
      frontier_endpoint_topic, 1, false);
  plan_event_publisher_ = private_nh.advertise<std_msgs::String>(
      plan_event_topic, 10, false);
  initialized_ = true;
  ROS_INFO_STREAM("PersistentTebLocalPlanner initialized: persistent_execution="
                  << (persistent_execution_ ? "true" : "false"));
}

bool PersistentTebLocalPlanner::setPlan(
    const std::vector<geometry_msgs::PoseStamped>& plan) {
  plan_received_count_.fetch_add(1, std::memory_order_relaxed);
  bool retained_equivalent = false;
  bool emit_equivalent_event = false;
  {
    std::lock_guard<std::mutex> lock(plan_mutex_);
    // move_base invokes the global planner at its configured rate. In the
    // persistent architecture that can produce a freshly allocated Navfn
    // vector for the exact same remaining route every 50 ms. Feeding each
    // duplicate to TEB resets its elastic band and creates visible zero-vel
    // pulses. Preserve TEB's current band only when the new path is wholly
    // geometrically covered by it; a changed endpoint or a real detour still
    // installs immediately. Local costmap collision checking remains in TEB's
    // 20 Hz compute cycle, independently of this global-path gate.
    if (persistent_execution_ && planIsEquivalentLocked(plan)) {
      plan_equivalent_count_.fetch_add(1, std::memory_order_relaxed);
      retained_equivalent = true;
      const ros::WallTime now = ros::WallTime::now();
      if (last_equivalent_plan_event_wall_.isZero() ||
          (now - last_equivalent_plan_event_wall_).toSec() >= 1.0) {
        last_equivalent_plan_event_wall_ = now;
        emit_equivalent_event = true;
      }
    }
  }
  if (retained_equivalent) {
    if (emit_equivalent_event) {
      publishPlanEvent("equivalent_retained", plan);
    }
    return true;
  }
  // Navfn preserves the planning request timestamp on its poses.  In this
  // stack the request can be stamped a millisecond after the latest map->odom
  // transform, making TEB reject an otherwise valid path for one control
  // cycle with a future-extrapolation error.  A global route is geometric
  // data, not a time-indexed trajectory: zero requests the latest transform
  // and keeps the route in the correct current odom frame.
  std::vector<geometry_msgs::PoseStamped> teb_plan = plan;
  for (geometry_msgs::PoseStamped& pose : teb_plan) {
    pose.header.stamp = ros::Time(0);
  }
  const bool accepted = teb_.setPlan(teb_plan);
  if (!accepted) {
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(plan_mutex_);
    applied_plan_ = std::move(teb_plan);
    if (!applied_plan_.empty()) {
      current_plan_goal_ = applied_plan_.back();
    }
    // The endpoint report is scoped to this installed path.  Increment while
    // holding the same mutex used by the reporter, so it cannot associate a
    // new goal with the preceding path version.
    route_version_.fetch_add(1, std::memory_order_relaxed);
  }
  plan_installed_count_.fetch_add(1, std::memory_order_relaxed);
  route_geometry_hash_.store(geometryHash(plan), std::memory_order_relaxed);
  publishPlanEvent("installed", plan);
  return true;
}

void PersistentTebLocalPlanner::publishPlanEvent(
    const char* event,
    const std::vector<geometry_msgs::PoseStamped>& plan) const {
  if (!plan_event_publisher_) {
    return;
  }
  std::ostringstream payload;
  payload << "{\"event\":\"" << event << "\",\"persistent\":"
          << (persistent_execution_ ? "true" : "false")
          << ",\"received\":"
          << plan_received_count_.load(std::memory_order_relaxed)
          << ",\"equivalent_retained\":"
          << plan_equivalent_count_.load(std::memory_order_relaxed)
          << ",\"installed\":"
          << plan_installed_count_.load(std::memory_order_relaxed)
          << ",\"route_version\":"
          << route_version_.load(std::memory_order_relaxed)
          << ",\"geometry_hash\":\""
          << route_geometry_hash_.load(std::memory_order_relaxed) << "\""
          << ",\"poses\":" << plan.size();
  if (!plan.empty()) {
    const geometry_msgs::PoseStamped& goal = plan.back();
    payload << ",\"goal\":[" << goal.pose.position.x << ","
            << goal.pose.position.y << "]";
  }
  payload << "}";
  std_msgs::String message;
  message.data = payload.str();
  plan_event_publisher_.publish(message);
}

uint64_t PersistentTebLocalPlanner::geometryHash(
    const std::vector<geometry_msgs::PoseStamped>& plan) {
  // Quantize to the 0.1 m Navfn grid before hashing. The hash is telemetry,
  // not a safety predicate; its purpose is to reveal meaningful corridor
  // replacements separately from per-message timestamps.
  uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&hash](int64_t value) {
    const uint64_t encoded = static_cast<uint64_t>(value);
    for (size_t byte = 0; byte < sizeof(encoded); ++byte) {
      hash ^= (encoded >> (byte * 8U)) & 0xffU;
      hash *= 1099511628211ULL;
    }
  };
  mix(static_cast<int64_t>(plan.size()));
  for (const geometry_msgs::PoseStamped& pose : plan) {
    mix(static_cast<int64_t>(std::llround(pose.pose.position.x * 10.0)));
    mix(static_cast<int64_t>(std::llround(pose.pose.position.y * 10.0)));
  }
  return hash;
}

bool PersistentTebLocalPlanner::planIsEquivalentLocked(
    const std::vector<geometry_msgs::PoseStamped>& candidate) const {
  if (candidate.empty() || applied_plan_.empty()) {
    return false;
  }
  const geometry_msgs::PoseStamped& candidate_goal = candidate.back();
  const geometry_msgs::PoseStamped& applied_goal = applied_plan_.back();
  if (candidate_goal.header.frame_id != applied_goal.header.frame_id) {
    return false;
  }
  // A target's endpoint is part of its perception-to-navigation identity and
  // must stay exact. Frontier endpoints are grid-derived exploration hints,
  // so use the map-resolution bound above only when the *currently applied*
  // route is not the installed visual target.
  const bool applied_plan_is_target =
      has_installed_target_goal_ &&
      applied_goal.header.frame_id == installed_target_goal_.header.frame_id &&
      std::hypot(
          applied_goal.pose.position.x - installed_target_goal_.pose.position.x,
          applied_goal.pose.position.y - installed_target_goal_.pose.position.y) <=
          target_goal_epsilon_;
  const double endpoint_equivalence_distance = applied_plan_is_target
      ? target_goal_epsilon_
      : frontier_endpoint_equivalence_distance_;
  if (std::hypot(candidate_goal.pose.position.x - applied_goal.pose.position.x,
                 candidate_goal.pose.position.y - applied_goal.pose.position.y) >
      endpoint_equivalence_distance) {
    return false;
  }

  // Navfn paths are normally sampled at the 0.1 m map resolution. Sampling at
  // most 25 deterministic points keeps the comparison cheap at 20 Hz while
  // still rejecting a newly mapped doorway detour instead of treating it as a
  // duplicate of the old corridor route.
  const size_t stride = std::max<size_t>(1, candidate.size() / 24);
  double traversed = 0.0;
  for (size_t i = 0; i < candidate.size(); i += stride) {
    if (i > 0) {
      const geometry_msgs::Point& previous = candidate[i - stride].pose.position;
      const geometry_msgs::Point& current = candidate[i].pose.position;
      traversed += std::hypot(current.x - previous.x, current.y - previous.y);
    }
    if (traversed < plan_equivalence_prefix_distance_) {
      continue;
    }
    const geometry_msgs::Point& point = candidate[i].pose.position;
    double nearest = std::numeric_limits<double>::infinity();
    for (const geometry_msgs::PoseStamped& existing : applied_plan_) {
      nearest = std::min(
          nearest,
          std::hypot(point.x - existing.pose.position.x,
                     point.y - existing.pose.position.y));
    }
    if (nearest > plan_equivalence_distance_) {
      return false;
    }
  }
  return true;
}

bool PersistentTebLocalPlanner::computeVelocityCommands(
    geometry_msgs::Twist& cmd_vel) {
  const bool command_available = teb_.computeVelocityCommands(cmd_vel);
  if (!persistent_execution_ || task_done_.load()) {
    return command_available;
  }
  // TEB returns false at every XY terminal. In the persistent architecture an
  // action is a lease for the whole mission, so both a temporary frontier and
  // a target approach must keep move_base's control loop healthy while the
  // stream supplies the next plan. Returning false here makes move_base start
  // recovery on a perfectly valid reached frontier whenever no successor has
  // been installed in the same control cycle. A target approach additionally
  // emits its one-shot observation event; a frontier simply holds zero.
  if (teb_.isGoalReached()) {
    if (isInstalledTargetPlanLocked()) {
      reportTargetApproachIfReady();
    } else {
      reportFrontierEndpointIfReady();
    }
    cmd_vel = geometry_msgs::Twist();
    return true;
  }
  return command_available;
}

bool PersistentTebLocalPlanner::isGoalReached() {
  if (!persistent_execution_) {
    return teb_.isGoalReached();
  }
  // The action is a controller lease, rather than ownership of one frontier
  // endpoint. Reaching a target must notify GoalManager without ending that
  // lease; otherwise actionlib's terminal callback dispatches a second action
  // and reintroduces the observable stop boundary we are eliminating.
  if (task_done_.load()) {
    return true;
  }
  if (teb_.isGoalReached()) {
    if (isInstalledTargetPlanLocked()) {
      reportTargetApproachIfReady();
    } else {
      reportFrontierEndpointIfReady();
    }
  }
  return false;
}

bool PersistentTebLocalPlanner::reportFrontierEndpointIfReady() {
  if (!frontier_endpoint_publisher_) {
    return false;
  }
  geometry_msgs::PoseStamped endpoint;
  uint64_t route_version = 0;
  {
    std::lock_guard<std::mutex> lock(plan_mutex_);
    if (current_plan_goal_.header.frame_id.empty()) {
      return false;
    }
    route_version = route_version_.load(std::memory_order_relaxed);
    const bool already_reported = has_reported_frontier_goal_ &&
        reported_frontier_route_version_ == route_version;
    if (already_reported) {
      return true;
    }
    reported_frontier_goal_ = current_plan_goal_;
    reported_frontier_route_version_ = route_version;
    has_reported_frontier_goal_ = true;
    endpoint = current_plan_goal_;
    endpoint.header.stamp = ros::Time::now();
  }
  frontier_endpoint_publisher_.publish(endpoint);
  ROS_INFO_STREAM(
      "PersistentTebLocalPlanner reported frontier endpoint: endpoint=("
      << endpoint.pose.position.x << "," << endpoint.pose.position.y
      << ") route_version=" << route_version);
  return true;
}

bool PersistentTebLocalPlanner::reportTargetApproachIfReady() {
  if (!isInstalledTargetPlanLocked()) {
    return false;
  }
  geometry_msgs::PoseStamped approach;
  uint32_t transaction_id = 0;
  {
    std::lock_guard<std::mutex> lock(plan_mutex_);
    // The installed-target callback can race this control cycle. Recheck the
    // endpoint under the same lock before emitting an approach terminal.
    // ``move_base`` may rewrite PoseStamped.header.seq while transforming a
    // Navfn path, so header metadata is not a reliable cross-plugin identity.
    if (!has_installed_target_goal_ ||
        current_plan_goal_.header.frame_id != installed_target_goal_.header.frame_id ||
        std::hypot(current_plan_goal_.pose.position.x -
                       installed_target_goal_.pose.position.x,
                   current_plan_goal_.pose.position.y -
                       installed_target_goal_.pose.position.y) >
            target_goal_epsilon_) {
      return false;
    }
    const bool already_reported = has_reported_target_goal_ &&
        reported_target_goal_.header.frame_id ==
            installed_target_goal_.header.frame_id &&
        std::hypot(reported_target_goal_.pose.position.x -
                       installed_target_goal_.pose.position.x,
                   reported_target_goal_.pose.position.y -
                       installed_target_goal_.pose.position.y) <=
            target_goal_epsilon_;
    if (!already_reported) {
      reported_target_goal_ = installed_target_goal_;
      has_reported_target_goal_ = true;
    }
    approach = installed_target_goal_;
    approach.header.stamp = ros::Time::now();
    transaction_id = installed_target_transaction_;
    if (already_reported) {
      return true;
    }
  }
  target_approach_publisher_.publish(approach);
  if (transaction_id > 0 && target_approach_result_publisher_) {
    std::ostringstream payload;
    payload << "{\"event\":\"target_approach\",\"transaction_id\":"
            << transaction_id << ",\"frame_id\":\""
            << approach.header.frame_id << "\",\"goal\":["
            << approach.pose.position.x << "," << approach.pose.position.y
            << "]}";
    std_msgs::String result;
    result.data = payload.str();
    target_approach_result_publisher_.publish(result);
  }
  ROS_INFO_STREAM("PersistentTebLocalPlanner reported target approach: endpoint=("
                  << approach.pose.position.x << "," << approach.pose.position.y
                  << ") transaction=" << transaction_id);
  return true;
}

void PersistentTebLocalPlanner::onTaskDone(
    const std_msgs::BoolConstPtr& message) {
  task_done_.store(message->data);
}

void PersistentTebLocalPlanner::onInstalledTargetCommand(
    const PersistentGoalCommandConstPtr& message) {
  if (message->kind != PersistentGoalCommand::KIND_TARGET_INSTALLED ||
      message->transaction_id == 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(plan_mutex_);
  const bool changed_target = !has_installed_target_goal_ ||
      installed_target_transaction_ != message->transaction_id ||
      installed_target_goal_.header.frame_id != message->goal.header.frame_id ||
      std::hypot(installed_target_goal_.pose.position.x - message->goal.pose.position.x,
                 installed_target_goal_.pose.position.y - message->goal.pose.position.y) >
          target_goal_epsilon_;
  installed_target_goal_ = message->goal;
  installed_target_transaction_ = message->transaction_id;
  has_installed_target_goal_ = true;
  if (changed_target) {
    has_reported_target_goal_ = false;
  }
}

bool PersistentTebLocalPlanner::isInstalledTargetPlanLocked() {
  std::lock_guard<std::mutex> lock(plan_mutex_);
  if (!has_installed_target_goal_ || current_plan_goal_.header.frame_id.empty()) {
    return false;
  }
  if (current_plan_goal_.header.frame_id != installed_target_goal_.header.frame_id) {
    return false;
  }
  return std::hypot(
             current_plan_goal_.pose.position.x -
                 installed_target_goal_.pose.position.x,
             current_plan_goal_.pose.position.y -
                 installed_target_goal_.pose.position.y) <=
         target_goal_epsilon_;
}

StreamingNavfnPlanner::StreamingNavfnPlanner()
    : initialized_(false),
      has_mission_goal_(false),
      has_target_goal_(false),
      has_validated_target_plan_(false),
      has_active_target_mission_(false),
      plan_dirty_(true),
      goal_epsilon_(0.05),
      replan_distance_(0.50),
      max_replan_interval_(2.0),
      latest_mission_sequence_(0),
      latest_target_sequence_(0),
      active_target_mission_sequence_(0),
      validated_target_sequence_(0),
      failed_target_sequence_(0),
      mission_generation_(0),
      target_failure_reported_generation_(0) {}

void StreamingNavfnPlanner::initialize(
    std::string name, costmap_2d::Costmap2DROS* costmap_ros) {
  if (initialized_) {
    ROS_WARN("StreamingNavfnPlanner is already initialized");
    return;
  }

  std::string mission_command_topic;
  ros::param::param<std::string>(
      "/move_base/streaming_mission_command_topic", mission_command_topic,
      "/lste/persistent_execution/mission_command");
  std::string target_command_topic;
  std::string installed_target_goal_topic;
  std::string installed_target_command_topic;
  std::string target_plan_result_topic;
  ros::param::param<std::string>(
      "/move_base/streaming_target_command_topic", target_command_topic,
      "/lste/persistent_execution/target_command");
  ros::param::param<std::string>(
      "/move_base/persistent_installed_target_goal_topic",
      installed_target_goal_topic,
      "/lste/persistent_execution/installed_target_goal");
  ros::param::param<std::string>(
      "/move_base/persistent_installed_target_command_topic",
      installed_target_command_topic,
      "/lste/persistent_execution/installed_target_command");
  ros::param::param<std::string>(
      "/move_base/persistent_target_plan_result_topic",
      target_plan_result_topic,
      "/lste/persistent_execution/target_plan_result");
  ros::param::param("/move_base/streaming_goal_epsilon", goal_epsilon_, 0.05);
  ros::param::param("/move_base/streaming_replan_distance", replan_distance_,
                    0.50);
  ros::param::param("/move_base/streaming_max_replan_interval",
                    max_replan_interval_, 2.0);
  goal_epsilon_ = std::max(0.001, goal_epsilon_);
  replan_distance_ = std::max(0.05, replan_distance_);
  max_replan_interval_ = std::max(0.10, max_replan_interval_);

  // Keep Navfn's established parameter namespace. The router owns goal
  // streaming, not path-search policy or cost interpretation.
  navfn_.initialize("NavfnROS", costmap_ros);
  global_frame_ = costmap_ros->getGlobalFrameID();
  transform_listener_.reset(new tf::TransformListener());
  ros::NodeHandle nh;
  mission_command_subscriber_ = nh.subscribe(
      mission_command_topic, 1, &StreamingNavfnPlanner::onMissionCommand,
      this);
  target_command_subscriber_ = nh.subscribe(
      target_command_topic, 1, &StreamingNavfnPlanner::onTargetCommand,
      this);
  installed_target_goal_publisher_ = nh.advertise<geometry_msgs::PoseStamped>(
      installed_target_goal_topic, 1, true);
  installed_target_command_publisher_ = nh.advertise<PersistentGoalCommand>(
      installed_target_command_topic, 1, true);
  target_plan_result_publisher_ = nh.advertise<std_msgs::String>(
      target_plan_result_topic, 10, false);
  initialized_ = true;
  ROS_INFO_STREAM("StreamingNavfnPlanner initialized: mission_command="
                  << mission_command_topic << " target_command="
                  << target_command_topic
                  << " result_topic=" << target_plan_result_topic
                  << " global_frame=" << global_frame_
                  << " epsilon=" << goal_epsilon_);
}

void StreamingNavfnPlanner::publishTargetPlanResult(
    const char* event, uint32_t transaction_id,
    const geometry_msgs::PoseStamped& goal) const {
  if (!target_plan_result_publisher_) {
    return;
  }
  std::ostringstream payload;
  payload << "{\"event\":\"" << event << "\",\"transaction_id\":"
          << transaction_id << ",\"frame_id\":\"" << goal.header.frame_id
          << "\",\"goal\":[" << goal.pose.position.x << ","
          << goal.pose.position.y << "]}";
  std_msgs::String message;
  message.data = payload.str();
  target_plan_result_publisher_.publish(message);
}

bool StreamingNavfnPlanner::makePlan(
    const geometry_msgs::PoseStamped& start,
    const geometry_msgs::PoseStamped& action_goal,
    std::vector<geometry_msgs::PoseStamped>& plan) {
  geometry_msgs::PoseStamped selected_goal;
  geometry_msgs::PoseStamped global_goal;
  bool must_replan = false;
  bool selected_is_target = false;
  bool retained_previous_route = false;
  uint32_t selected_target_sequence = 0;
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // A target stream is a two-phase request.  It can temporarily outrank the
    // last bridge-approved mission only while it carries a newer transaction
    // number; once the bridge publishes that same number on the approved
    // mission topic, normal mission ownership resumes.  This prevents a
    // direct /lste/final_goal update from bypassing target-plan confirmation.
    const bool provisional_target = has_target_goal_ &&
        latest_target_sequence_ > 0 &&
        (!has_mission_goal_ || latest_mission_sequence_ == 0 ||
         latest_target_sequence_ > latest_mission_sequence_);
    if (provisional_target) {
      selected_target_sequence = latest_target_sequence_;
      // A rejected transaction stays rejected until GoalManager emits a newer
      // command. Without this guard, each planner tick re-runs Navfn against
      // a target which the bridge has already released back to exploration.
      if (failed_target_sequence_ == selected_target_sequence) {
        if (!cached_plan_.empty()) {
          plan = cached_plan_;
          return true;
        }
        return false;
      }
      // Navfn has already validated this exact target and notified the
      // bridge. Keep executing the last approved route while waiting for the
      // bridge to publish the matching approved mission transaction. This is
      // the actual two-phase boundary: a provisional path never reaches TEB
      // when a previous route is available.
      if (has_validated_target_plan_ &&
          validated_target_sequence_ == selected_target_sequence) {
        if (!cached_plan_.empty()) {
          plan = cached_plan_;
          return true;
        }
        // There is no prior route only during a cold-start visual mission.
        // The candidate was still Navfn-validated exactly before it can be
        // returned, so this is a safe bootstrap exception to the stream lease.
        plan = validated_target_plan_;
        return !plan.empty();
      }
    }
    // After the bridge approves an installed target, its provisional request
    // is deliberately cleared. The approved mission still has to keep the
    // target's exact-endpoint rule on every later Navfn replan; otherwise a
    // map update can silently move the path endpoint to Navfn's tolerance
    // boundary and desynchronize GoalManager, TEB, and Navfn again.
    const bool approved_target_mission = has_active_target_mission_ &&
        has_mission_goal_ &&
        active_target_mission_sequence_ == latest_mission_sequence_;
    if (!provisional_target && approved_target_mission) {
      selected_target_sequence = active_target_mission_sequence_;
    }
    selected_goal = provisional_target ? target_goal_ :
        (has_mission_goal_ ? mission_goal_ : action_goal);
    must_replan = plan_dirty_ || shouldReplanLocked(start, selected_goal);
    if (!must_replan && !cached_plan_.empty()) {
      plan = cached_plan_;
      return true;
    }
    selected_is_target = provisional_target || approved_target_mission;
    generation = mission_generation_;
  }

  if (!transformToGlobalFrame(selected_goal, global_goal)) {
    ROS_WARN_THROTTLE(1.0,
                      "StreamingNavfnPlanner could not transform mission goal to %s",
                      global_frame_.c_str());
    return false;
  }

  std::vector<geometry_msgs::PoseStamped> candidate;
  const bool navfn_success = navfn_.makePlan(start, global_goal, candidate) &&
      !candidate.empty();
  // A target is an exact execution contract.  Navfn can otherwise return a
  // path that terminates at its internal tolerance boundary, which would make
  // the local planner announce arrival at a point GoalManager never approved.
  // Keep the previous frontier path in that case and report a route failure
  // for the original transaction instead.
  const bool target_endpoint_exact = !selected_is_target ||
      (navfn_success && std::hypot(
          candidate.back().pose.position.x - global_goal.pose.position.x,
          candidate.back().pose.position.y - global_goal.pose.position.y) <=
          goal_epsilon_);
  if (!navfn_success || !target_endpoint_exact) {
    bool report_target_failure = false;
    bool retained_previous_plan = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // A target request is speculative until Navfn has found a map-connected
      // route.  Keep a known-good route alive during that decision instead of
      // returning false to move_base: recovery would erase the action lease,
      // while TEB would otherwise sit at the endpoint of the old route with no
      // explanation of why the requested target was not installed.
      if (generation == mission_generation_) {
        if (selected_is_target) {
          has_target_goal_ = false;
          has_validated_target_plan_ = false;
          if (has_active_target_mission_ &&
              active_target_mission_sequence_ == selected_target_sequence) {
            has_active_target_mission_ = false;
          }
          if (failed_target_sequence_ != selected_target_sequence) {
            failed_target_sequence_ = selected_target_sequence;
            target_failure_reported_generation_ = generation;
            report_target_failure = true;
          }
        }
        if (!cached_plan_.empty()) {
          plan = cached_plan_;
          retained_previous_plan = true;
        }
      }
    }
    if (report_target_failure) {
      publishTargetPlanResult("target_plan_failed", selected_target_sequence,
                              global_goal);
    }
    ROS_WARN_THROTTLE(
        1.0,
        "StreamingNavfnPlanner could not install the current mission goal%s%s",
        selected_is_target && navfn_success ? " exactly" : "",
        retained_previous_plan ? "; retaining previous valid path" : "");
    return retained_previous_plan;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generation != mission_generation_) {
      // A detector update can race Navfn by one callback. Do not make
      // move_base enter recovery for a result we know is obsolete; retain the
      // last valid path and let the next planner tick install the newer one.
      ROS_DEBUG("StreamingNavfnPlanner discarded stale planning result");
      plan_dirty_ = true;
      if (!cached_plan_.empty()) {
        plan = cached_plan_;
        return true;
      }
      return false;
    }
    if (selected_is_target) {
      validated_target_plan_ = candidate;
      validated_target_goal_ = global_goal;
      has_validated_target_plan_ = true;
      validated_target_sequence_ = selected_target_sequence;
      plan_dirty_ = false;
      if (!cached_plan_.empty()) {
        // Do not expose the speculative path to move_base. The bridge will
        // publish the matching mission transaction after it receives the
        // installed-target acknowledgement below.
        plan = cached_plan_;
        retained_previous_route = true;
      } else {
        // See the cold-start exception above. Retain the validated candidate
        // as the initial route so the first action lease can stay alive.
        cached_plan_ = candidate;
        cached_goal_ = selected_goal;
        last_plan_start_ = start;
        last_plan_time_ = ros::WallTime::now();
        plan = cached_plan_;
      }
    } else {
      cached_plan_ = candidate;
      cached_goal_ = selected_goal;
      last_plan_start_ = start;
      last_plan_time_ = ros::WallTime::now();
      plan_dirty_ = false;
      plan = cached_plan_;
    }
  }
  if (selected_is_target) {
    geometry_msgs::PoseStamped installed = candidate.back();
    installed.header.stamp = ros::Time::now();
    installed_target_goal_publisher_.publish(installed);
    PersistentGoalCommand installed_command;
    installed_command.transaction_id = selected_target_sequence;
    installed_command.kind = PersistentGoalCommand::KIND_TARGET_INSTALLED;
    installed_command.goal = installed;
    installed_target_command_publisher_.publish(installed_command);
    ROS_INFO_STREAM("StreamingNavfnPlanner validated target transaction="
                    << selected_target_sequence << " endpoint=("
                    << installed.pose.position.x << ","
                    << installed.pose.position.y << ") previous_route="
                    << (retained_previous_route ? "retained" : "bootstrap"));
    publishTargetPlanResult("target_plan_installed", selected_target_sequence,
                            installed);
  }
  return true;
}

void StreamingNavfnPlanner::onMissionCommand(
    const PersistentGoalCommandConstPtr& message) {
  if (message->kind != PersistentGoalCommand::KIND_MISSION ||
      message->transaction_id == 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const uint32_t sequence = message->transaction_id;
  if (sequence > 0 && latest_mission_sequence_ > 0 &&
      sequence < latest_mission_sequence_) {
    ROS_WARN_STREAM("StreamingNavfnPlanner ignored stale approved mission "
                    << sequence << " < " << latest_mission_sequence_);
    return;
  }
  mission_goal_ = message->goal;
  latest_mission_sequence_ = sequence;
  has_mission_goal_ = true;
  // A target is provisional only while it is strictly newer than the approved
  // mission. Once the bridge repeats its transaction here, consume the exact
  // already-validated path and remove the speculative input from selection.
  if (has_target_goal_ && latest_target_sequence_ > 0 &&
      sequence >= latest_target_sequence_) {
    has_target_goal_ = false;
  }
  const bool approving_validated_target =
      sequence > 0 && has_validated_target_plan_ &&
      sequence == validated_target_sequence_;
  if (approving_validated_target) {
    cached_plan_ = validated_target_plan_;
    cached_goal_ = mission_goal_;
    last_plan_time_ = ros::WallTime::now();
    plan_dirty_ = false;
    has_validated_target_plan_ = false;
    has_active_target_mission_ = true;
    active_target_mission_sequence_ = sequence;
  } else {
    plan_dirty_ = true;
    // A different approved mission takes ownership from the prior visual
    // target. The command protocol has no separate target flag here; only a
    // matching previously validated target transaction may retain it.
    has_active_target_mission_ = false;
    active_target_mission_sequence_ = 0;
  }
  ++mission_generation_;
  ROS_INFO_STREAM("StreamingNavfnPlanner approved mission transaction="
                  << sequence << " endpoint=("
                  << message->goal.pose.position.x << ","
                  << message->goal.pose.position.y << ")");
}

void StreamingNavfnPlanner::onTargetCommand(
    const PersistentGoalCommandConstPtr& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  const uint32_t sequence = message->transaction_id;
  if (message->kind == PersistentGoalCommand::KIND_CLEAR) {
    // The bridge publishes this tombstone after a target transaction resolves
    // or is superseded. It clears the latched request for a hot move_base
    // restart and prevents repeated validation of an old visual ray.
    if (sequence > 0 && latest_target_sequence_ > sequence) {
      ROS_DEBUG_STREAM("StreamingNavfnPlanner ignored stale target clear "
                       << sequence << " < " << latest_target_sequence_);
      return;
    }
    has_target_goal_ = false;
    // Keep a just-validated candidate until its matching mission transaction
    // arrives. The two publishers are independent ROS connections, so their
    // delivery order is not guaranteed even though the bridge emits approval
    // before this tombstone.
    plan_dirty_ = true;
    ++mission_generation_;
    ROS_INFO("StreamingNavfnPlanner cleared provisional target request");
    return;
  }
  if (message->kind != PersistentGoalCommand::KIND_TARGET_REQUEST ||
      sequence == 0) {
    return;
  }
  ROS_INFO_STREAM("StreamingNavfnPlanner received target transaction="
                  << sequence << " approved_mission="
                  << latest_mission_sequence_ << " latest_target="
                  << latest_target_sequence_ << " endpoint=("
                  << message->goal.pose.position.x << ","
                  << message->goal.pose.position.y << ")");
  if (latest_mission_sequence_ > 0 && sequence <= latest_mission_sequence_) {
    ROS_WARN_STREAM("StreamingNavfnPlanner ignored stale target request "
                    << sequence << " <= approved mission "
                    << latest_mission_sequence_);
    return;
  }
  if (latest_target_sequence_ > 0 && sequence <= latest_target_sequence_) {
    ROS_WARN_STREAM("StreamingNavfnPlanner ignored stale target request "
                    << sequence << " <= " << latest_target_sequence_);
    return;
  }
  target_goal_ = message->goal;
  latest_target_sequence_ = sequence;
  has_target_goal_ = true;
  has_validated_target_plan_ = false;
  failed_target_sequence_ = 0;
  plan_dirty_ = true;
  ++mission_generation_;
}

bool StreamingNavfnPlanner::shouldReplanLocked(
    const geometry_msgs::PoseStamped& start,
    const geometry_msgs::PoseStamped& selected) const {
  if (cached_plan_.empty()) {
    return true;
  }
  if (selected.header.frame_id != cached_goal_.header.frame_id) {
    return true;
  }
  if (std::hypot(selected.pose.position.x - cached_goal_.pose.position.x,
                 selected.pose.position.y - cached_goal_.pose.position.y) >
      goal_epsilon_) {
    return true;
  }
  if (std::hypot(last_plan_start_.pose.position.x - start.pose.position.x,
                 last_plan_start_.pose.position.y - start.pose.position.y) >
      replan_distance_) {
    return true;
  }
  return !last_plan_time_.isZero() &&
         (ros::WallTime::now() - last_plan_time_).toSec() >
             max_replan_interval_;
}

bool StreamingNavfnPlanner::transformToGlobalFrame(
    const geometry_msgs::PoseStamped& source,
    geometry_msgs::PoseStamped& transformed) const {
  if (source.header.frame_id.empty()) {
    return false;
  }
  if (source.header.frame_id == global_frame_) {
    transformed = source;
    return true;
  }
  try {
    transform_listener_->transformPose(global_frame_, source, transformed);
    return true;
  } catch (const tf::TransformException& error) {
    ROS_WARN_THROTTLE(1.0, "StreamingNavfnPlanner TF transform failed: %s",
                      error.what());
    return false;
  }
}

}  // namespace lste_topo_access

PLUGINLIB_EXPORT_CLASS(lste_topo_access::PersistentTebLocalPlanner,
                       nav_core::BaseLocalPlanner)
PLUGINLIB_EXPORT_CLASS(lste_topo_access::StreamingNavfnPlanner,
                       nav_core::BaseGlobalPlanner)
