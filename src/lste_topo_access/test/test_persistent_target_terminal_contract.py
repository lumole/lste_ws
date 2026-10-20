"""Keep semantic target terminals out of the frontier lease protocol."""

import ast
from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "teb_goal_bridge_persistent_target_approach.py"
)


def load_contract_method():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "TebGoalBridgePersistentTargetApproachMixin"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_target_terminal_contract_locked"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[method.name]


BUILD = load_contract_method()


class TargetTerminalContractFixture:
    _target_terminal_contract_locked = BUILD

    active_action_contract = {
        "route_id": 21,
        "generation": 27,
        "source": "global_slam_frontier",
        "priority": 0,
        "route_kind": "frontier_endpoint",
    }
    latest_goal_transaction_id = 23
    latest_graph_transaction_id = 96
    latest_intent_source = "target_reacquisition_sweep"
    latest_intent_priority = 2
    latest_target_epoch = 4
    latest_target_track_id = "yellow_cup:yellow cup:1"
    latest_target_viewpoint_candidate_id = "viewpoint:2"
    latest_target_viewpoint_attempt_id = "attempt:3"


class TargetTerminalContractTest(unittest.TestCase):
    def test_target_terminal_does_not_reuse_frontier_route_identity(self):
        contract = TargetTerminalContractFixture()._target_terminal_contract_locked()

        self.assertEqual(contract["route_id"], 0)
        self.assertEqual(contract["source"], "target_reacquisition_sweep")
        self.assertEqual(contract["priority"], 2)
        self.assertEqual(contract["route_kind"], "direct_goal")
        self.assertEqual(contract["graph_action"], "reinspect_target")
        self.assertEqual(contract["graph_obligation_kind"], "target_observation")
        self.assertEqual(contract["transaction_id"], 23)


if __name__ == "__main__":
    unittest.main()
