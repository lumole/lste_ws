#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Goal arbitration diagnostics, route validation, and target recovery helpers."""

import copy
import json
import math
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from tf.transformations import quaternion_from_euler, quaternion_matrix

from goal_manager_route_validation import validate_target_route as validate_target_route_request
from goal_manager_modes import STATE_LOCKED
from goal_manager_target_utils import wrap_angle

class GoalManagerGoalArbitrationMixin:
    def publish_goal_arbitration(self, event: str, **fields):
        """Publish mission/execution arbitration decisions for run logs.

        The final-goal topic intentionally carries only executable poses.  A
        separate event stream records why a perception intent was accepted,
        deferred, or rejected, so a stopped vehicle can be diagnosed without
        inferring policy decisions from velocity samples alone.
        """
        payload = {
            "event": str(event),
            "task_id": str(self.current_task_id),
            "mission_id": str(getattr(self, "current_mission_id", "")),
            "task_version": str(getattr(self, "current_task_version", "")),
            "source": str(self.goal_source),
            "mode": str(self.effective_mode),
            "controller_mode": str(self.controller_mode),
            "target_state": str(self.target_execution_state),
            "target_epoch": int(self.target_observation_epoch),
            "target_blocked": bool(self.target_blocked),
            "target_failure_count": int(self.target_failure_count),
            "target_observation_gate": (
                None
                if getattr(self, "target_observation_gate", None) is None
                else self.target_observation_gate.snapshot().__dict__
            ),
            "target_viewpoint_ledger": (
                None
                if getattr(self, "target_approach_transaction", None) is None
                else self.target_approach_transaction.viewpoint_ledger.snapshot().to_dict()
            ),
            "target_reinspection_pending": bool(
                getattr(self, "target_reinspection_pending", False)
            ),
        }
        payload.update(fields)
        try:
            self.pub_goal_arbitration.publish(
                String(data=json.dumps(payload, sort_keys=True))
            )
        except (TypeError, ValueError):
            rospy.logwarn_throttle(5.0, "GoalManager: arbitration event serialization failed")

    def _pose_in_frame(self, pose: PoseStamped, target_frame: str) -> Optional[PoseStamped]:
        """Transform a planar pose without introducing a tf message helper."""
        if pose is None:
            return None
        source_frame = (pose.header.frame_id or "odom").strip().lstrip("/") or "odom"
        target_frame = (target_frame or "map").strip().lstrip("/") or "map"
        if source_frame == target_frame:
            return pose
        transform = self.lookup_transform(target_frame, source_frame, rospy.Time(0))
        if transform is None:
            return None
        rotation = transform.transform.rotation
        matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        point = matrix[:3, :3].dot(
            np.array([
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            ], dtype=float)
        )
        translation = transform.transform.translation
        out = PoseStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = target_frame
        out.pose.position.x = float(point[0] + translation.x)
        out.pose.position.y = float(point[1] + translation.y)
        out.pose.position.z = float(point[2] + translation.z)
        source_yaw = self.yaw_from_pose(pose) or 0.0
        transform_yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        qx, qy, qz, qw = quaternion_from_euler(
            0.0, 0.0, wrap_angle(source_yaw + transform_yaw)
        )
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        return out

    def pose_distance(self, first: PoseStamped, second: PoseStamped) -> Optional[float]:
        """Measure two mission poses in one frame for lifecycle matching."""
        if first is None or second is None:
            return None
        first_frame = self._frame_name(first.header.frame_id)
        second_frame = self._frame_name(second.header.frame_id)
        comparable = first
        if first_frame != second_frame:
            comparable = self._pose_in_frame(first, second_frame)
            if comparable is None:
                return None
        return math.hypot(
            float(comparable.pose.position.x) - float(second.pose.position.x),
            float(comparable.pose.position.y) - float(second.pose.position.y),
        )

    def validate_target_route(
        self,
        goal: PoseStamped,
        now: float,
        force: bool = False,
    ) -> Optional[bool]:
        """Delegate Navfn validation to the focused route-validation module."""
        return validate_target_route_request(self, goal, now, force=force)
    def target_reacquisition_goal(self, now: float) -> Optional[PoseStamped]:
        """Make one bounded forward continuation after a target is lost.

        The camera saw the target along ``target_last_heading``.  Continuing
        along that bearing is the only evidence-backed way to bring a small,
        distant object back into view.  The previous reverse-heading sweep
        pointed the vehicle away from the detected object and made the normal
        pipeline abandon valid sightings near desks.
        """
        if (
            self.target_blocked
            # An active target segment already owns the same visual evidence.
            # A loss-driven sweep must never preempt it merely because one
            # detector interval exceeded the freshness threshold.
            or self.target_last_goal is not None
            or self.target_reacquire_duration <= 0.0
            or self.target_reacquire_attempts >= self.target_reacquire_max_attempts
            or self.target_last_seen is None
            or self.target_last_heading is None
            or self.latest_pose is None
        ):
            return None
        evidence_timeout = self.target_evidence_timeout()
        if now - self.target_last_seen < evidence_timeout:
            return None
        if self.target_reacquire_started is None:
            heading = self.target_last_heading
            x = self.latest_pose.x + self.target_reacquire_distance * math.cos(heading)
            y = self.latest_pose.y + self.target_reacquire_distance * math.sin(heading)
            requested_goal_odom = self.make_goal_pose((x, y, 0.0), heading)
            requested_goal = self._pose_in_frame(requested_goal_odom, "map")
            if requested_goal is None:
                self.goal_source = "target_reacquisition_waiting_route"
                self.publish_goal_arbitration(
                    "target_reacquisition_deferred",
                    reason="tf_unavailable",
                )
                return None
            # Reacquisition is still a target mission, so it must meet the
            # same Navfn endpoint contract as a normal visual segment. This
            # also makes a dynamic map change observable before it is sent to
            # the persistent planner rather than as a later route failure.
            route_status = self.validate_target_route(
                requested_goal, now, force=True
            )
            if (
                route_status is not True
                or self.target_route_validation_last_endpoint is None
            ):
                self.goal_source = "target_reacquisition_waiting_route"
                self.publish_goal_arbitration(
                    "target_reacquisition_deferred",
                    reason=(
                        "navfn_empty_plan"
                        if route_status is False
                        else "route_validation_unavailable"
                    ),
                    requested_goal=[
                        round(float(requested_goal.pose.position.x), 3),
                        round(float(requested_goal.pose.position.y), 3),
                    ],
                    goal_frame=(requested_goal.header.frame_id or "map"),
                )
                return None
            self.target_reacquire_goal = copy.deepcopy(
                self.target_route_validation_last_endpoint
            )
            self.target_reacquire_started = now
            self.target_reacquire_attempts += 1
            endpoint_adjustment = self.pose_distance(
                requested_goal, self.target_reacquire_goal
            )
            self.publish_goal_arbitration(
                "target_reacquisition_route_validated",
                evidence_level=(
                    "confirmed_track"
                    if self.target_follow_confirmed else "direct_candidate"
                ),
                requested_goal=[
                    round(float(requested_goal.pose.position.x), 3),
                    round(float(requested_goal.pose.position.y), 3),
                ],
                navfn_returned_goal=[
                    round(float(self.target_reacquire_goal.pose.position.x), 3),
                    round(float(self.target_reacquire_goal.pose.position.y), 3),
                ],
                navfn_endpoint_adjustment=round(
                    float(endpoint_adjustment or 0.0), 4
                ),
                navfn_endpoint_contract=True,
            )
            rospy.loginfo(
                "GoalManager: target %s; Navfn-validated continuation %d/%d",
                "lost" if self.target_follow_confirmed else "candidate needs reinspection",
                self.target_reacquire_attempts,
                self.target_reacquire_max_attempts,
            )
        if now - self.target_reacquire_started < self.target_reacquire_duration:
            self.goal_source = "target_reacquisition_sweep"
            return self.target_reacquire_goal
        rospy.loginfo("GoalManager: target reacquisition sweep expired; resume global exploration")
        self.target_reacquire_goal = None
        self.target_reacquire_started = None
        return None

    def fresh_global_frontier_goal(self, now: float) -> Optional[PoseStamped]:
        if (
            self.frontier_replan_pending_id > 0
            and self.frontier_replan_ready_id != self.frontier_replan_pending_id
        ):
            return None
        if not self.global_frontier_enabled or self.latest_global_frontier_goal is None:
            return None
        stamp = self.latest_global_frontier_goal.header.stamp
        if self.global_frontier_max_age > 0.0 and stamp:
            age = now - stamp.to_sec()
            if age > self.global_frontier_max_age:
                return None
        return self.latest_global_frontier_goal

    def _complete_target_task(
        self, now: float, det, score_threshold: float, completion_evidence: str
    ):
        """Publish one identity-bearing completion transaction."""
        approach_transaction = getattr(
            self, "target_approach_transaction", None
        )
        if approach_transaction is not None:
            approach_transaction.mark_completed(now, completion_evidence)
        event = (
            "target_close_confirmed"
            if completion_evidence == "close_box"
            else "target_terminal_observation_confirmed"
        )
        fields = {
            "target_track_id": self.target_track_id,
            "approach_track_id": self.target_approach_track_id,
            "score": round(float(det.score), 4),
            "score_threshold": round(float(score_threshold), 4),
            "box": [round(float(det.w), 4), round(float(det.h), 4)],
            "hits": int(self.target_close_hits),
            "hold_seconds": round(
                max(0.0, float(now - self.target_close_since))
                if self.target_close_since is not None
                else 0.0,
                3,
            ),
        }
        self.publish_goal_arbitration(event, **fields)
        self.publish_goal_arbitration(
            "target_task_completed",
            completion_evidence=completion_evidence,
            **fields,
        )
        rospy.loginfo(
            "GoalManager: target completion evidence=%s hits=%d hold=%.1fs, "
            "publish task_done",
            completion_evidence,
            self.target_close_hits,
            max(0.0, now - self.target_close_since)
            if self.target_close_since is not None
            else 0.0,
        )
        self.pub_task_done.publish(Bool(data=True))
        self.task_done_published = True

    def maybe_publish_task_done(self, now: float):
        """Close a task from independent fresh visual observations.

        Completion must come from a close target in several fresh detector
        messages, not merely from elapsed state time. LOCKED-only completion
        remains available as an explicit strict mode. Counting distinct
        detector messages prevents the 5 Hz Goal Manager timer from treating
        one stale box as several observations. Once confirmed the velocity
        mux stops all commands atomically, preserving the view instead of
        turning away from a target that has already been reached.
        """
        if self.task_done_published:
            return
        if self.target_done_require_locked and self.current_state != STATE_LOCKED:
            self.reset_target_close_confirmation()
            return
        # Exhausting the current viewpoint ledger is a revalidation boundary,
        # not completion evidence.  A false or distant track must receive a
        # new local observation transaction; it must never publish task_done
        # merely because its bounded route portfolio ran out.
        portfolio_complete = getattr(
            self, "target_viewpoint_portfolio_complete", None
        )
        if callable(portfolio_complete) and portfolio_complete(now):
            det = self.target_detection_for_track(self.latest_dets)
            if det is None or not self.target_detection_is_close(det):
                revalidate = getattr(
                    self, "revalidate_target_viewpoint_portfolio", None
                )
                if callable(revalidate):
                    revalidate(now)
                return
        if not self.target_close_completion_eligible():
            # A lone or unconfirmed image must not complete the task.  A
            # confirmed direct close observation is an equally valid terminal
            # viewpoint when the configured policy permits it: requiring a
            # further blind approach can place the goal inside furniture and
            # makes the robot lose an already-visible target.
            self.reset_target_close_confirmation()
            return
        det = self.target_detection_for_track(self.latest_dets)
        if self.latest_dets is None:
            self.reset_target_close_confirmation()
            return
        stamp = self.latest_dets.header.stamp
        stamp_sec = stamp.to_sec()
        stamp_key = (
            stamp.secs,
            stamp.nsecs,
            int(getattr(self.latest_dets.header, "seq", 0) or 0),
        )
        fresh_detector_message = self.target_close_last_stamp != stamp_key
        if (
            self.target_done_max_detection_age > 0.0
            and now - stamp_sec > self.target_done_max_detection_age
        ):
            # A slow detector can legitimately leave its last message stale
            # before the next independent frame arrives. It must not create a
            # new close vote, but an already started terminal observation may
            # retain the stationary viewpoint for its bounded grace period.
            if (
                self.target_close_last_seen is None
                or now - self.target_close_last_seen
                > self.target_close_confirmation_grace()
            ):
                self.reset_target_close_confirmation()
            return
        close_score_threshold = self.target_close_score_threshold()
        close_enough = self.target_detection_is_close(det)
        if close_enough and self.target_completed_segments >= 1:
            # This is a post-terminal fact, not a score latch. It allows the
            # same track to bridge a detector-size dip without treating the
            # next frame as proof that the target disappeared.
            self.target_terminal_close_candidate_seen = True
        if not close_enough:
            terminal_observation_eligible = getattr(
                self, "target_terminal_observation_eligible", None
            )
            if (
                fresh_detector_message
                and callable(terminal_observation_eligible)
                and terminal_observation_eligible(det)
            ):
                if self.target_close_last_stamp != stamp_key:
                    self.target_close_last_stamp = stamp_key
                    self.target_close_hits += 1
                self.target_close_last_seen = now
                if self.target_close_since is None:
                    self.target_close_since = now
                    self.publish_goal_arbitration(
                        "target_terminal_observation_started",
                        target_track_id=self.target_track_id,
                        approach_track_id=self.target_approach_track_id,
                        score=round(float(det.score), 4),
                        score_threshold=round(float(close_score_threshold), 4),
                        box=[round(float(det.w), 4), round(float(det.h), 4)],
                        hits=int(self.target_close_hits),
                        required_hits=int(self.target_done_min_fresh_hits),
                        required_hold_seconds=round(
                            float(self.target_done_min_hold_time), 3
                        ),
                    )
                    rospy.loginfo(
                        "GoalManager: terminal target observation started "
                        "score=%.3f box=(%.3f,%.3f) hits=%d/%d",
                        float(det.score), float(det.w), float(det.h),
                        self.target_close_hits, self.target_done_min_fresh_hits,
                    )
                    return
                if (
                    self.target_close_hits >= self.target_done_min_fresh_hits
                    and (now - self.target_close_since)
                    >= self.target_done_min_hold_time
                ):
                    self._complete_target_task(
                        now,
                        det,
                        close_score_threshold,
                        "terminal_track_continuity",
                    )
                return
            # A fresh negative frame is an explicit observation transaction
            # boundary.  Keeping the old close vote through every later
            # negative frame turned a detector dip into a 20 s navigation hold
            # and produced the long zero-velocity stalls seen in the office
            # run.  Stale messages remain harmless; the next fresh frame may
            # start a new close transaction.
            if fresh_detector_message:
                self.reset_target_close_confirmation()
                if self.navigation_hold_active:
                    self.set_navigation_hold(False, "fresh_close_evidence_rejected")
            return
        if self.target_close_last_stamp != stamp_key:
            self.target_close_last_stamp = stamp_key
            self.target_close_hits += 1
        self.target_close_last_seen = now
        if self.target_close_since is None:
            self.target_close_since = now
            self.publish_goal_arbitration(
                "target_close_confirmation_started",
                target_track_id=self.target_track_id,
                approach_track_id=self.target_approach_track_id,
                score=round(float(det.score), 4),
                score_threshold=round(float(close_score_threshold), 4),
                box=[round(float(det.w), 4), round(float(det.h), 4)],
                hits=int(self.target_close_hits),
                required_hits=int(self.target_done_min_fresh_hits),
                required_hold_seconds=round(float(self.target_done_min_hold_time), 3),
            )
            rospy.loginfo(
                "GoalManager: close target confirmation started score=%.3f threshold=%.3f "
                "box=(%.3f,%.3f) hits=%d/%d",
                float(det.score), float(close_score_threshold),
                float(det.w), float(det.h),
                self.target_close_hits, self.target_done_min_fresh_hits,
            )
            return
        if (
            self.target_close_hits >= self.target_done_min_fresh_hits
            and (now - self.target_close_since) >= self.target_done_min_hold_time
        ):
            self._complete_target_task(
                now, det, close_score_threshold, "close_box"
            )
