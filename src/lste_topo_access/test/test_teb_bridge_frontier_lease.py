#!/usr/bin/env python3
"""Regression tests for graph-to-controller frontier lease handoff."""

import ast
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
STATUS_SOURCE = SCRIPTS / "teb_goal_bridge_frontier_status.py"
HEALTH_SOURCE = SCRIPTS / "teb_goal_bridge_action_health.py"
DISPATCH_SOURCE = SCRIPTS / "teb_goal_bridge_action_dispatch.py"


class GoalStatusStub:
    PREEMPTED = 2


def load_class(source, class_name, namespace):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    owner = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    exec(
        compile(ast.Module(body=[owner], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[class_name]


STATUS_MIXIN = load_class(
    STATUS_SOURCE,
    "TebGoalBridgeFrontierStatusMixin",
    {"json": json, "math": __import__("math"),
     "rospy": SimpleNamespace(logwarn=lambda *_args, **_kwargs: None)},
)
HEALTH_MIXIN = load_class(
    HEALTH_SOURCE,
    "TebGoalBridgeActionHealthMixin",
    {
        "time": SimpleNamespace(monotonic=lambda: 100.0),
        "now_for": lambda _owner: 100.0,
        "GoalStatus": GoalStatusStub,
        "rospy": SimpleNamespace(logwarn=lambda *_args, **_kwargs: None),
    },
)


class ActiveDispatchStub:
    pass


class RetryPolicyStub:
    pass


class ActionClientStub:
    pass


DISPATCH_MIXIN = load_class(
    DISPATCH_SOURCE,
    "TebGoalBridgeActionDispatchMixin",
    {
        "rospy": SimpleNamespace(
            loginfo_throttle=lambda *_args, **_kwargs: None,
        ),
        "TebGoalBridgeActionActiveDispatchMixin": ActiveDispatchStub,
        "TebGoalBridgeActionRetryPolicyMixin": RetryPolicyStub,
        "TebGoalBridgeActionClientMixin": ActionClientStub,
    },
)


class FakeActionClient:
    def __init__(self):
        self.cancel_count = 0

    def cancel_goal(self):
        self.cancel_count += 1


class FakeBridge(STATUS_MIXIN, HEALTH_MIXIN, DISPATCH_MIXIN):
    def __init__(self):
        self.lock = threading.RLock()
        self.persistent_execution = False
        self.task_done = False
        self.action_active = True
        self.active_intent_source = "global_slam_frontier"
        self.active_intent_priority = 0
        self.active_route_kind = "portal_transition"
        self.active_route_id = 17
        self.latest_intent_source = "global_slam_frontier"
        self.latest_intent_priority = 0
        self.latest_route_id = 17
        self.latest_goal = object()
        self.last_dispatched_goal = SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=1.0, y=2.0),
            )
        )
        self.frontier_lease_released_route_id = 0
        self.frontier_lease_released_reason = ""
        self.frontier_observation_completion_pending = None
        self.frontier_continuous_prefetch_handoff_pending = None
        self.frontier_controller_lease_releases = {}
        self.handoff_requested = False
        self.action_generation = 4
        self.last_result_status = None
        self.last_result_monotonic = 0.0
        self.persistent_target_pending_transaction = 0
        self.action_client = FakeActionClient()
        self.statuses = []
        self.health_cleared = False
        self.pending_terminal_prepare = None
        self.pending_lease_release_contract = None
        self.frontier_portal_wait = False
        self.frontier_portal_wait_route_id = 0
        self.terminal_dispatch_timer = None
        self.dispatches = []

    def _is_active_mode(self):
        return True

    def _clear_target_failure_locked(self, _reason):
        pass

    def _clear_failed_route_lease_locked(self):
        pass

    def _clear_action_health_locked(self):
        self.health_cleared = True
        self.active_route_id = 0
        self.active_intent_source = "unknown"
        self.active_intent_priority = 0
        self.active_route_kind = ""

    def publish_bridge_status(self, event, **fields):
        self.statuses.append((event, fields))

    def events(self):
        return [event for event, _fields in self.statuses]

    def _transport_handoff_blocks_dispatch_locked(self):
        return False

    def _action_server_ready(self):
        return True

    def _target_failure_blocks_latest_locked(self):
        return False

    def _dispatch_active_action_locked(self, reason):
        self.dispatches.append(("active", reason))

    def _dispatch_inactive_action_locked(self, force, reason):
        self.dispatches.append(("inactive", reason, force))


# Verbatim shape of the release the bridge could not consume in run
# runtime/manual_teleop/20261015_174533 (sim 35.77): GlobalFrontier released
# the probe route's controller lease without a transport terminal because a
# successor was already queued, so it marked the release ``semantic_only``.
SEMANTIC_RELEASE_ROUTE_2 = {
    "event": "controller_lease_released",
    "route_id": 2,
    "released_route_id": 2,
    "released_route_kind": "frontier_endpoint",
    "transaction_id": 0,
    "terminal_received": False,
    "controller_pending": True,
    "release_phase": "semantic_only",
    "release_reason": "route_release",
    "action_generation": 0,
    "graph_transaction_id": 6,
    "lifecycle_transaction_id": 2584637941852747396477038143873465788,
}

# The predecessor contract the bridge recorded for that same route when it
# revoked the action generation for the successor.
PREDECESSOR_ROUTE_2 = {
    "source": "global_slam_frontier",
    "route_id": 2,
    "generation": 3,
    "transaction_id": 6,
    "graph_transaction_id": 6,
}


class TebBridgeFrontierLeaseTest(unittest.TestCase):
    def test_unavailable_active_route_keeps_controller_until_terminal(self):
        bridge = FakeBridge()

        bridge.on_frontier_status(SimpleNamespace(data=json.dumps({
            "event": "frontier_route_unavailable",
            "route_id": 17,
            "reason": "durable_identity_not_in_current_frontier_snapshot",
        })))

        self.assertEqual(bridge.action_client.cancel_count, 0)
        self.assertFalse(bridge.health_cleared)
        self.assertEqual(bridge.frontier_lease_released_route_id, 0)
        self.assertEqual(bridge.latest_route_id, 17)
        self.assertIsNotNone(bridge.latest_goal)
        self.assertEqual(
            [event for event, _fields in bridge.statuses],
            ["frontier_route_unavailable_deferred"],
        )
        self.assertEqual(
            bridge.statuses[-1][1]["controller_lease"],
            "held_until_terminal",
        )

    def test_unavailable_idle_route_releases_controller_lease(self):
        bridge = FakeBridge()
        bridge.action_active = False

        bridge.on_frontier_status(SimpleNamespace(data=json.dumps({
            "event": "frontier_route_unavailable",
            "route_id": 17,
            "reason": "durable_identity_not_in_current_frontier_snapshot",
        })))

        self.assertEqual(bridge.action_client.cancel_count, 0)
        self.assertTrue(bridge.health_cleared)
        self.assertEqual(bridge.frontier_lease_released_route_id, 17)
        self.assertEqual(bridge.latest_route_id, 0)
        self.assertIsNone(bridge.latest_goal)
        self.assertEqual(
            [event for event, _fields in bridge.statuses],
            ["controller_lease_released", "frontier_route_unavailable"],
        )
        self.assertEqual(
            bridge.statuses[-1][1]["released_route_id"],
            17,
        )

    def test_superseded_release_is_reported_instead_of_silently_dropped(self):
        """A release the bridge ignores must say so on its status topic.

        A withheld successor dispatch and a stalled robot are the same picture
        on the wire -- the vehicle does not move -- so the branch that discards
        the release is the only place that can attribute the hold.  Without
        this event a held crossing can only be diagnosed by re-reading source.
        """
        bridge = FakeBridge()
        bridge.active_route_id = 9
        bridge.latest_route_id = 9

        bridge.on_frontier_status(SimpleNamespace(data=json.dumps({
            "event": "controller_lease_released",
            "route_id": 7,
            "released_route_id": 7,
            "released_route_kind": "portal_transition",
            "transaction_id": 1,
            "terminal_received": True,
            "controller_pending": True,
            "release_phase": "awaiting_transport_ack",
            "lifecycle_transaction_id": 11,
            "action_generation": 2,
            "graph_transaction_id": 3,
        })))

        reported = [
            fields for event, fields in bridge.statuses
            if event == "controller_lease_release_ignored"
        ]
        self.assertEqual(len(reported), 1)
        self.assertEqual(
            reported[0]["reason"], "release_superseded_by_newer_route",
        )
        self.assertEqual(reported[0]["route_id"], 7)
        self.assertEqual(reported[0]["active_route_id"], 9)
        # The newer route must still not be cancelled by the stale release.
        self.assertEqual(bridge.frontier_lease_released_route_id, 0)

    def _feed(self, bridge, payload):
        bridge.on_frontier_status(SimpleNamespace(data=json.dumps(payload)))

    def _arm_predecessor_hold(self, bridge, contract=None):
        bridge.pending_lease_release_contract = {
            "old_contract": dict(contract or PREDECESSOR_ROUTE_2),
        }

    def test_semantic_only_predecessor_release_answers_a_later_hold(self):
        """A release stated without a transport terminal still answers a hold.

        Recorded in runtime/manual_teleop/20261015_174533: GlobalFrontier
        invalidated the probe route at the standoff (sim 35.77) and released
        its controller lease as ``semantic_only`` -- a successor was already
        queued, so it expected no acknowledgement.  The bridge's own hold for
        that predecessor was armed 1.3 s later (sim 36.06) when the crossing
        route's command arrived, and the release it was waiting for had
        already been published and dropped.  Dispatch was withheld for the
        next 120 s -- GlobalFrontier's DISPATCHED lifecycle timeout upstream,
        not a bridge window -- which then aborted the portal transaction
        (`portal_route_released_before_crossing`) for a crossing that was
        never allowed to start.
        """
        bridge = FakeBridge()
        self._feed(bridge, SEMANTIC_RELEASE_ROUTE_2)

        # The release is not a transport boundary, and applying it as one
        # would clear state for an action move_base may still own.  It is
        # retained as the statement it is, and reported as ignored.
        self.assertFalse(bridge.health_cleared)
        self.assertEqual(bridge.action_client.cancel_count, 0)
        self.assertEqual(bridge.events(), ["controller_lease_release_ignored"])
        self.assertEqual(
            bridge.statuses[-1][1]["reason"],
            "release_without_transport_terminal",
        )

        self._arm_predecessor_hold(bridge)
        bridge.latest_route_id = 4

        bridge.dispatch_locked(False, "coalesced_global_goal")

        # The predecessor contract stays for a later authoritative release,
        # but it no longer withholds the successor.
        self.assertTrue(
            bridge.pending_lease_release_contract["hold_answered"]
        )
        self.assertNotIn(
            "dispatch_deferred_controller_lease_release", bridge.events()
        )
        cleared = [
            fields for event, fields in bridge.statuses
            if event == "controller_lease_release_hold_cleared"
        ]
        self.assertEqual(len(cleared), 1)
        self.assertEqual(cleared[0]["route_id"], 2)
        self.assertEqual(cleared[0]["latest_route_id"], 4)
        self.assertEqual(
            cleared[0]["reason"], "predecessor_release_already_stated",
        )
        self.assertEqual(cleared[0]["release_phase"], "semantic_only")
        self.assertEqual(
            bridge.dispatches, [("active", "coalesced_global_goal")]
        )

        # The answer is reported once, not on every later dispatch tick.
        bridge.dispatch_locked(False, "coalesced_global_goal")
        self.assertEqual(
            [
                event for event, _fields in bridge.statuses
                if event == "controller_lease_release_hold_cleared"
            ],
            ["controller_lease_release_hold_cleared"],
        )

    def test_release_arriving_after_the_hold_was_armed_also_clears_it(self):
        """The other ordering: the hold first, GlobalFrontier's release next."""
        bridge = FakeBridge()
        self._arm_predecessor_hold(bridge)
        bridge.latest_route_id = 4

        bridge.dispatch_locked(False, "coalesced_global_goal")
        self.assertIsNotNone(bridge.pending_lease_release_contract)
        self.assertEqual(bridge.dispatches, [])
        self.assertEqual(
            bridge.events(), ["dispatch_deferred_controller_lease_release"]
        )

        self._feed(bridge, SEMANTIC_RELEASE_ROUTE_2)
        bridge.statuses.clear()
        bridge.dispatch_locked(False, "coalesced_global_goal")

        self.assertIn("controller_lease_release_hold_cleared", bridge.events())
        self.assertEqual(
            bridge.dispatches, [("active", "coalesced_global_goal")]
        )

    def test_hold_survives_while_the_predecessor_release_is_missing(self):
        """Silence about the predecessor is still not a release."""
        bridge = FakeBridge()
        self._arm_predecessor_hold(
            bridge,
            dict(PREDECESSOR_ROUTE_2, route_id=9, graph_transaction_id=9),
        )
        bridge.latest_route_id = 12

        bridge.dispatch_locked(False, "coalesced_global_goal")

        self.assertIsNotNone(bridge.pending_lease_release_contract)
        self.assertEqual(bridge.dispatches, [])
        self.assertEqual(
            bridge.events(), ["dispatch_deferred_controller_lease_release"]
        )

    def test_a_replayed_route_is_not_dispatched_by_its_own_release(self):
        """The hold still refuses to resurrect the route it was armed for."""
        bridge = FakeBridge()
        self._feed(bridge, SEMANTIC_RELEASE_ROUTE_2)
        self._arm_predecessor_hold(bridge)
        bridge.latest_route_id = 2

        bridge.dispatch_locked(False, "coalesced_global_goal")

        self.assertIsNotNone(bridge.pending_lease_release_contract)
        self.assertEqual(bridge.dispatches, [])

    def test_a_differing_generation_is_not_the_released_predecessor(self):
        """Identity is the route, but a stated generation must still agree."""
        bridge = FakeBridge()
        self._feed(bridge, dict(SEMANTIC_RELEASE_ROUTE_2, action_generation=7))
        self._arm_predecessor_hold(bridge)
        bridge.latest_route_id = 4

        bridge.dispatch_locked(False, "coalesced_global_goal")

        self.assertIsNotNone(bridge.pending_lease_release_contract)
        self.assertEqual(bridge.dispatches, [])

    def test_endpoint_terminal_allows_release_while_action_transport_is_active(self):
        """A persistent endpoint terminal is an explicit lease boundary."""
        bridge = FakeBridge()

        bridge.on_frontier_status(SimpleNamespace(data=json.dumps({
            "event": "frontier_route_unavailable",
            "route_id": 17,
            "reason": "selected_graph_obligation_not_executable_in_snapshot",
            "released_controller_route": {
                "route_id": 17,
                "controller_pending": True,
                "terminal_received": True,
            },
        })))

        self.assertEqual(bridge.action_client.cancel_count, 1)
        self.assertTrue(bridge.health_cleared)
        self.assertEqual(bridge.frontier_lease_released_route_id, 17)
        self.assertEqual(
            [event for event, _fields in bridge.statuses],
            ["controller_lease_released", "frontier_route_unavailable"],
        )
        self.assertIn(
            "endpoint_terminal_",
            bridge.statuses[-1][1]["reason"],
        )


if __name__ == "__main__":
    unittest.main()
