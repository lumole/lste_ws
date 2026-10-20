#!/usr/bin/env python3

"""Interpret the actuator boundary's status contract for route accounting.

This node plans and validates routes but never publishes ``/cmd_vel``.  A route
reaches the robot only after the goal manager, the TEB bridge and the command
mux have each accepted it, so the frontier's own clock cannot tell whether a
route failed or whether its commands were never forwarded at all.  The mux
already publishes an observation contract at that boundary
(``/lste/cmd_vel_mux/status`` with ``selected_mode``, ``block_reason`` and the
per-source ``active`` flag), which is exactly the missing fact.

The distinction matters for the failure authority split in
``global_frontier_route_lease``: elapsed time inside this node is evidence
about *its* route only while the boundary is actually carrying that route.  A
human on teleop, a mission-level navigation hold, a completed task or a slice
reset all leave the vehicle moving for reasons this node does not own, and a
progress watchdog that convicts a route then is reporting on someone else's
driving.
"""

import json
import threading
import time


# Every steady state in which the boundary withholds the selected
# controller's commands without that controller having failed.  These are
# ownership facts, not planning facts.
#
# Two absences are deliberate.  A rejected command contract is not one of
# them: there the frontier's own command *was* produced and refused, which is
# real evidence about its route and keeps the historical watchdog authority.
# A mode-switch announcement is not one either: that record already carries
# the newly selected mode, which is the ownership fact itself.
OWNERSHIP_BLOCK_REASONS = (
    "source_not_selected",
    "navigation_hold",
    "task_complete",
    "hard_reset",
    "hard_reset_hold",
)

# The TEB goal bridge is the last hop before ``/move_base``.  It withholds its
# own dispatch while it waits for this node to release the predecessor's
# controller lease, and it says so explicitly on its status topic
# (``dispatch_deferred_controller_lease_release``).  A route held there has not
# been handed to the controller at all, so elapsed time inside this node is not
# evidence about it -- the live two-room run convicted a crossing route as
# stalled 12.2 s in while the bridge had never dispatched the goal, which then
# aborted the portal transaction for want of an egress anchor.
#
# The freeze is bounded by the bridge's own escape: its route-lease watchdog
# releases the hold after ``_ROUTE_LEASE_WATCHDOG_SECONDS``.  Past that window
# the silence is evidence again, so the historical authority returns instead of
# the hold suppressing every stall forever.
DISPATCH_DEFERRAL_EVENT = "dispatch_deferred_controller_lease_release"
BRIDGE_DISPATCH_DEFERRAL_WINDOW_SECONDS = 10.0


class RouteActuatorStatus:
    """Track whether the actuator boundary is forwarding this node's routes."""

    def __init__(self, source="teb"):
        self.source = str(source or "teb").strip().lower() or "teb"
        self._lock = threading.Lock()
        # ``received`` distinguishes "the boundary says it is carrying another
        # owner" from "no boundary status was ever seen".  A composition
        # without the mux keeps the historical watchdog authority rather than
        # suppressing every stall for want of a contract nobody publishes.
        self._received = False
        self._selected_mode = ""
        self._block_reason = "none"
        self._last_source = ""
        self._source_active = False
        self._sequence = 0
        self._dispatch_deferred = None

    def observe_dispatch(self, payload):
        """Record one TEB goal bridge status record; True when it changes a fact.

        Only the bridge's own dispatch bookkeeping is read here.  A deferral is
        stored with the wall clock that observed it, and an actual dispatch
        clears it; every other bridge event is somebody else's protocol.
        """
        if not isinstance(payload, dict):
            return False
        event = str(payload.get("event", "") or "").strip()
        if event == "dispatch":
            with self._lock:
                if self._dispatch_deferred is None:
                    return False
                self._dispatch_deferred = None
            return True
        if event != DISPATCH_DEFERRAL_EVENT:
            return False
        with self._lock:
            self._dispatch_deferred = {
                "route_id": payload.get("route_id"),
                "reason": payload.get("reason"),
                "deferred_reason": payload.get("deferred_reason"),
                "observed_at": time.monotonic(),
            }
        return True

    def dispatch_is_deferred(self):
        """Return whether the bridge is withholding dispatch right now."""
        with self._lock:
            return self._dispatch_deferred_locked()[0]

    def _dispatch_deferred_locked(self):
        """Decide a dispatch hold while the caller already holds the lock.

        Returns ``(held, record)``.  The hold expires after the bridge's own
        route-lease watchdog window, because past that point the bridge is no
        longer withholding anything this node can wait for.
        """
        record = self._dispatch_deferred
        if not isinstance(record, dict):
            return False, None
        observed_at = record.get("observed_at")
        try:
            age = time.monotonic() - float(observed_at)
        except (TypeError, ValueError):
            return False, record
        return age <= BRIDGE_DISPATCH_DEFERRAL_WINDOW_SECONDS, record

    def apply(self, message_data):
        """Record one mux status record; return True when it decides a change."""
        try:
            payload = json.loads(message_data)
        except (TypeError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        mode = str(payload.get("selected_mode", "") or "").strip().lower()
        if not mode:
            # Any other JSON on this topic is not an actuator decision.
            return False
        source = str(payload.get("source", "") or "").strip().lower()
        block_reason = str(
            payload.get("block_reason", "none") or "none"
        ).strip().lower()
        with self._lock:
            changed = (
                not self._received
                or mode != self._selected_mode
                or block_reason != self._block_reason
                or source != self._last_source
            )
            self._received = True
            self._selected_mode = mode
            self._block_reason = block_reason
            self._last_source = source
            if source == self.source:
                self._source_active = bool(payload.get("active", False))
            try:
                self._sequence = int(payload.get("sequence", 0) or 0)
            except (TypeError, ValueError):
                pass
        return changed

    def _forwarding_locked(self):
        """Decide forwarding while the caller already holds the lock."""
        if not self._received:
            # Fail open: without an actuator contract there is nothing to
            # attribute, and the historical authority must not change.  A
            # bridge dispatch hold is an explicit local fact and is honoured
            # even then: it says the goal never left this stack.
            return not self._dispatch_deferred_locked()[0]
        if self._selected_mode != self.source:
            return False
        if self._dispatch_deferred_locked()[0]:
            return False
        return self._block_reason not in OWNERSHIP_BLOCK_REASONS

    def forwarding(self):
        """Return True when the boundary carries this node's route commands."""
        with self._lock:
            return self._forwarding_locked()

    def diagnostic(self):
        """Return the fields a status payload needs to explain a suppression."""
        with self._lock:
            held, record = self._dispatch_deferred_locked()
            return {
                "route_actuator_source": self.source,
                "route_actuator_received": bool(self._received),
                "route_actuator_selected_mode": str(self._selected_mode),
                "route_actuator_block_reason": str(self._block_reason),
                "route_actuator_source_active": bool(self._source_active),
                "route_actuator_forwarding": self._forwarding_locked(),
                "route_actuator_sequence": int(self._sequence),
                "route_actuator_dispatch_deferred": bool(held),
                "route_actuator_dispatch_deferred_route_id": (
                    None if not isinstance(record, dict) else record.get("route_id")
                ),
                "route_actuator_dispatch_deferred_reason": (
                    None if not isinstance(record, dict) else record.get("reason")
                ),
                "route_actuator_dispatch_deferral_window_seconds": (
                    BRIDGE_DISPATCH_DEFERRAL_WINDOW_SECONDS
                ),
            }


__all__ = [
    "BRIDGE_DISPATCH_DEFERRAL_WINDOW_SECONDS",
    "DISPATCH_DEFERRAL_EVENT",
    "OWNERSHIP_BLOCK_REASONS",
    "RouteActuatorStatus",
]
