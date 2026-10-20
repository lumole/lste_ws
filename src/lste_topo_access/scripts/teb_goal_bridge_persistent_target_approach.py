"""Commit verified target-approach terminals for persistent TEB execution."""

import rospy


class TebGoalBridgePersistentTargetApproachMixin:
    def _target_terminal_contract_locked(self):
        """Build a semantic target contract for a non-frontier terminal.

        Persistent target segments reuse the native MoveBase action that may
        have been created for a frontier route.  Its immutable transport
        contract must not be reused for the semantic target observation
        boundary: doing so publishes the target terminal as a frontier ACK and
        leaves Global Frontier waiting for a release that this path does not
        own.
        """
        base = getattr(self, "active_action_contract", None)
        contract = dict(base) if hasattr(base, "items") else {}
        latest_source = str(
            getattr(self, "latest_intent_source", "") or "target_follow"
        )
        if not latest_source.startswith("target_"):
            latest_source = "target_follow"
        contract.update(
            {
                "route_id": 0,
                "transaction_id": int(
                    getattr(self, "latest_goal_transaction_id", 0) or 0
                ),
                "graph_transaction_id": int(
                    getattr(self, "latest_graph_transaction_id", 0) or 0
                ),
                "route_kind": "direct_goal",
                "mission_route_kind": "direct_goal",
                "graph_action": "reinspect_target",
                "graph_obligation_kind": "target_observation",
                "source": latest_source,
                "priority": max(
                    2, int(getattr(self, "latest_intent_priority", 0) or 0)
                ),
                "target_epoch": int(getattr(self, "latest_target_epoch", 0) or 0),
                "target_track_id": str(
                    getattr(self, "latest_target_track_id", "") or ""
                ),
                "target_viewpoint_candidate_id": str(
                    getattr(self, "latest_target_viewpoint_candidate_id", "")
                    or ""
                ),
                "target_viewpoint_attempt_id": str(
                    getattr(self, "latest_target_viewpoint_attempt_id", "")
                    or ""
                ),
            }
        )
        return contract

    def _handle_persistent_target_approach_locked(self, transaction_id, message):
        """Forward an identity-bound persistent target arrival to GoalManager."""
        if not self.persistent_execution or self.task_done:
            return
        if (
            self.latest_goal is None
            or self.latest_intent_priority < 2
            or self.persistent_target_approach_reported_transaction
            == int(self.latest_goal_transaction_id)
        ):
            self.publish_bridge_status(
                "persistent_target_approach_ignored",
                received_transaction_id=transaction_id,
                latest_transaction_id=int(self.latest_goal_transaction_id),
            )
            return
        latest_global = self._goal_in_global_frame(self.latest_goal)
        if latest_global is None or not self._same_goal(latest_global, message):
            self.publish_bridge_status(
                "persistent_target_approach_ignored",
                reason="endpoint_mismatch",
                received_transaction_id=transaction_id,
                latest_transaction_id=int(self.latest_goal_transaction_id),
                received_goal=[
                    round(message.pose.position.x, 3),
                    round(message.pose.position.y, 3),
                ],
                latest_goal=(
                    None
                    if latest_global is None
                    else [
                        round(latest_global.pose.position.x, 3),
                        round(latest_global.pose.position.y, 3),
                    ]
                ),
            )
            self._republish_persistent_target_request_locked(
                "stale_target_approach_endpoint"
            )
            return
        strict_transaction_match = (
            transaction_id > 0
            and transaction_id == int(self.latest_goal_transaction_id)
            and self.persistent_installed_target_goal is not None
            and transaction_id == int(self.persistent_installed_target_transaction)
            and self._same_goal(self.persistent_installed_target_goal, message)
        )
        if not strict_transaction_match:
            self.publish_bridge_status(
                "persistent_target_approach_ignored",
                reason="missing_matching_installation",
                received_transaction_id=transaction_id,
                latest_transaction_id=int(self.latest_goal_transaction_id),
            )
            return
        # This is a semantic target-segment boundary, not the end of the
        # persistent MoveBase lease.  The next viewpoint must be able to enter
        # the same Navfn/TEB stream immediately; using the generic termination
        # transaction here would revoke the action generation and leave the
        # successor blocked behind a controller-release ACK that target routes
        # do not produce.
        terminal_contract = self._target_terminal_contract_locked()
        terminal_goal = self._publish_execution_terminal_locked(
            self.latest_goal,
            action_contract=terminal_contract,
        )
        if terminal_goal is None:
            return
        self.persistent_target_approach_reported_transaction = int(
            self.latest_goal_transaction_id
        )
        # This is a logical terminal of the target route. Persistent Navfn/TEB
        # keeps the native MoveBase action alive so the next mission command
        # can stream in without a cancel/restart gap.
        self.persistent_target_terminal_boundary_transaction = int(
            self.latest_goal_transaction_id
        )
        self.terminal_count += 1
        self.publish_bridge_status(
            "persistent_target_approach_terminal",
            transaction_id=int(self.latest_goal_transaction_id),
            received_transaction_id=transaction_id,
            identity_mode="transaction",
            target_epoch=int(self.latest_target_epoch),
            target_track_id=self.latest_target_track_id,
            terminal_goal=[
                round(terminal_goal.pose.position.x, 3),
                round(terminal_goal.pose.position.y, 3),
            ],
            lifecycle="single_action_target_approach",
        )
        # The persistent target reuses the MoveBase action that was originally
        # opened for a frontier route.  Its semantic terminal is non-frontier,
        # but Global Frontier still needs an idempotent release ACK for that
        # predecessor route before it can plan the next room/work item.
        predecessor = getattr(self, "active_action_contract", None)
        if hasattr(predecessor, "get"):
            try:
                predecessor_route_id = int(predecessor.get("route_id", 0) or 0)
            except (TypeError, ValueError):
                predecessor_route_id = 0
            predecessor_source = str(
                predecessor.get("source", "") or ""
            ).strip().lower()
            release_ack = getattr(
                self, "_publish_frontier_release_ack_locked", None
            )
            identity_builder = getattr(
                self, "_route_lease_watchdog_identity_locked", None
            )
            if (
                predecessor_route_id > 0
                and predecessor_source == "global_slam_frontier"
                and callable(release_ack)
                and callable(identity_builder)
            ):
                release_ack(
                    predecessor_route_id,
                    identity_builder(predecessor),
                    str(predecessor.get("route_kind", "") or ""),
                    release_phase="target_terminal_semantic_boundary",
                )
                self.publish_bridge_status(
                    "target_terminal_frontier_predecessor_released",
                    released_route_id=predecessor_route_id,
                    release_phase="target_terminal_semantic_boundary",
                )
        rospy.loginfo(
            "TEB goal bridge forwarded persistent target approach: "
            "transaction=%d received=%d track=%s",
            self.latest_goal_transaction_id,
            transaction_id,
            self.latest_target_track_id,
        )
