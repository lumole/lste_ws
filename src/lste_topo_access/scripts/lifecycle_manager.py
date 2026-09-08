#!/usr/bin/env python3
"""Single-threaded event-queue lifecycle coordination.

The ROS transport layer may call ``enqueue`` from any thread.  The owner of
the lifecycle calls ``tick`` from one fixed-rate timer; only that call may
consume events, change state, advance a transaction, or invoke business
handlers.
"""

from dataclasses import dataclass
from enum import Enum
import queue
import threading
import uuid
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from clock_provider import ClockProvider


class State(Enum):
    """Common lifecycle states shared by task and route coordinators."""

    IDLE = "IDLE"
    DISPATCHED = "DISPATCHED"
    EXECUTION_DONE = "EXECUTION_DONE"
    SYNCING = "SYNCING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class EventType(Enum):
    """Ingress facts accepted by the lifecycle queue."""

    TASK_STARTED = "TASK_STARTED"
    TASK_UPDATED = "TASK_UPDATED"
    TASK_COMPLETED = "TASK_COMPLETED"
    SCORES_UPDATED = "SCORES_UPDATED"
    CAMERA_INFO_UPDATED = "CAMERA_INFO_UPDATED"
    DEPTH_UPDATED = "DEPTH_UPDATED"
    FRONTIER_UPDATED = "FRONTIER_UPDATED"
    FRONTIERS_UPDATED = "FRONTIERS_UPDATED"
    CONTROLLER_MODE_UPDATED = "CONTROLLER_MODE_UPDATED"
    CMD_VEL_UPDATED = "CMD_VEL_UPDATED"
    ACCESS_MODE_UPDATED = "ACCESS_MODE_UPDATED"
    ACCESS_GOAL_UPDATED = "ACCESS_GOAL_UPDATED"
    FIXED_GOAL_UPDATED = "FIXED_GOAL_UPDATED"
    GLOBAL_FRONTIER_UPDATED = "GLOBAL_FRONTIER_UPDATED"
    GLOBAL_FRONTIER_COMMAND = "GLOBAL_FRONTIER_COMMAND"
    GLOBAL_FRONTIER_STATUS = "GLOBAL_FRONTIER_STATUS"
    TEB_GOAL_TERMINAL = "TEB_GOAL_TERMINAL"
    TEB_GOAL_FAILURE = "TEB_GOAL_FAILURE"
    NAV_DISPATCHED = "NAV_DISPATCHED"
    NAV_REACHED = "NAV_REACHED"
    NAV_FAILED = "NAV_FAILED"
    EXECUTION_TERMINAL = "EXECUTION_TERMINAL"
    SYNC_REQUESTED = "SYNC_REQUESTED"
    SYNC_DONE = "SYNC_DONE"
    MAP_UPDATED = "MAP_UPDATED"
    COSTMAP_UPDATED = "COSTMAP_UPDATED"
    POSE_UPDATED = "POSE_UPDATED"
    SCAN_UPDATED = "SCAN_UPDATED"
    DETECTIONS_UPDATED = "DETECTIONS_UPDATED"
    STATE_UPDATED = "STATE_UPDATED"
    GOAL_UPDATED = "GOAL_UPDATED"
    REPLAN_REQUESTED = "REPLAN_REQUESTED"
    RECOVERY_OBSERVED = "RECOVERY_OBSERVED"
    NAVFN_RESULT = "NAVFN_RESULT"
    TURN_STATUS = "TURN_STATUS"
    BRIDGE_GOAL = "BRIDGE_GOAL"
    BRIDGE_GOAL_COMMAND = "BRIDGE_GOAL_COMMAND"
    BRIDGE_INTENT = "BRIDGE_INTENT"
    BRIDGE_FRONTIER_STATUS = "BRIDGE_FRONTIER_STATUS"
    BRIDGE_TURN_STATUS = "BRIDGE_TURN_STATUS"
    BRIDGE_TEB_FEEDBACK = "BRIDGE_TEB_FEEDBACK"
    BRIDGE_PLANNER_COMMAND = "BRIDGE_PLANNER_COMMAND"
    BRIDGE_NAVFN_PLAN = "BRIDGE_NAVFN_PLAN"
    BRIDGE_COSTMAP = "BRIDGE_COSTMAP"
    BRIDGE_POSE = "BRIDGE_POSE"
    BRIDGE_MODE = "BRIDGE_MODE"
    BRIDGE_TASK_DONE = "BRIDGE_TASK_DONE"
    BRIDGE_TARGET_RESULT = "BRIDGE_TARGET_RESULT"
    BRIDGE_FRONTIER_ENDPOINT = "BRIDGE_FRONTIER_ENDPOINT"
    ACTION_FEEDBACK = "ACTION_FEEDBACK"
    ACTION_DONE = "ACTION_DONE"
    BRIDGE_WAKE = "BRIDGE_WAKE"
    RESET = "RESET"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class Event:
    """Immutable fact gathered by an external callback."""

    type: EventType
    transaction_id: int
    payload: Any = None
    created_at: float = 0.0


@dataclass(frozen=True)
class TickResult:
    """Bounded observability for one lifecycle tick."""

    processed: int
    discarded_stale: int
    discarded_future: int
    timed_out: bool
    state: State
    transaction_id: int
    adopted_future: int = 0


class LifecycleManager:
    """Own one lifecycle and serialize all of its state transitions.

    ``event_handler`` runs only from ``tick`` and may return a ``State`` to
    request a transition.  A caller can also use ``transition_to`` from code
    that is already executing inside ``tick``.  External callbacks should use
    only ``make_event`` and ``enqueue``.
    """

    _DEFAULT_TRANSITIONS = {
        (State.IDLE, EventType.TASK_STARTED): State.DISPATCHED,
        (State.IDLE, EventType.TASK_COMPLETED): State.COMPLETED,
        (State.IDLE, EventType.NAV_DISPATCHED): State.DISPATCHED,
        (State.DISPATCHED, EventType.NAV_REACHED): State.EXECUTION_DONE,
        (State.DISPATCHED, EventType.EXECUTION_TERMINAL): State.EXECUTION_DONE,
        (State.DISPATCHED, EventType.NAV_FAILED): State.FAILED,
        (State.EXECUTION_DONE, EventType.SYNC_REQUESTED): State.SYNCING,
        (State.SYNCING, EventType.SYNC_DONE): State.IDLE,
        (State.EXECUTION_DONE, EventType.TASK_COMPLETED): State.COMPLETED,
        (State.SYNCING, EventType.TASK_COMPLETED): State.COMPLETED,
        (State.DISPATCHED, EventType.TASK_COMPLETED): State.COMPLETED,
        (State.FAILED, EventType.RESET): State.IDLE,
        (State.COMPLETED, EventType.RESET): State.IDLE,
    }

    # These inputs are sampled state, not causal control events. Keep one
    # pending sample per transaction/stream so a slow compute phase cannot
    # retain an unbounded history of sensor messages. A costmap delta stream is
    # coalesced separately; losing an intermediate patch becomes a resync
    # marker instead of an invalid partially reconstructed grid.
    _COALESCED_EVENT_TYPES = frozenset({
        EventType.MAP_UPDATED,
        EventType.COSTMAP_UPDATED,
        EventType.POSE_UPDATED,
        EventType.SCAN_UPDATED,
        EventType.STATE_UPDATED,
        EventType.DETECTIONS_UPDATED,
        EventType.SCORES_UPDATED,
        EventType.CAMERA_INFO_UPDATED,
        EventType.DEPTH_UPDATED,
        EventType.FRONTIER_UPDATED,
        EventType.FRONTIERS_UPDATED,
        EventType.CONTROLLER_MODE_UPDATED,
        EventType.CMD_VEL_UPDATED,
        EventType.GLOBAL_FRONTIER_UPDATED,
        EventType.BRIDGE_TEB_FEEDBACK,
        EventType.BRIDGE_PLANNER_COMMAND,
        EventType.BRIDGE_NAVFN_PLAN,
        EventType.BRIDGE_COSTMAP,
        EventType.BRIDGE_POSE,
        EventType.TURN_STATUS,
    })
    _TRANSACTION_UNIQUE_BITS = 64
    _TRANSACTION_ID_LOCK = threading.Lock()
    _LAST_TRANSACTION_ID = 0

    def __init__(
        self,
        event_handler: Optional[Callable[[Event], Optional[State]]] = None,
        transition_handler: Optional[Callable[[State, State, Optional[Event]], None]] = None,
        timeout_handler: Optional[Callable[[Event], None]] = None,
        time_fn: Optional[Callable[[], float]] = None,
        time_provider: Optional[Any] = None,
        timeouts: Optional[Mapping[State, float]] = None,
        transitions: Optional[Mapping[Tuple[State, EventType], State]] = None,
        max_inbox_size: int = 1024,
        max_events_per_tick: int = 256,
    ):
        # A zero/unset capacity used to mean an unbounded queue. Keep a hard
        # lower bound so callback bursts cannot grow process memory forever.
        self._max_inbox_size = max(1, int(max_inbox_size))
        self._causal_reserve = max(
            1, min(32, self._max_inbox_size // 4 or 1)
        )
        self._sample_capacity = max(
            0, self._max_inbox_size - self._causal_reserve
        )
        self.inbox = queue.Queue(maxsize=self._max_inbox_size)
        self._state_lock = threading.RLock()
        self._inbox_state_lock = threading.RLock()
        self._tick_lock = threading.Lock()
        if time_fn is not None and time_provider is not None:
            raise ValueError("provide time_fn or time_provider, not both")
        selected_time = time_fn or time_provider or ClockProvider()
        self._time_fn = (
            selected_time
            if callable(selected_time)
            else selected_time.now
        )
        self.time_provider = selected_time
        self._event_handler = event_handler
        self._transition_handler = transition_handler
        self._timeout_handler = timeout_handler
        self._timeouts = {
            state: float(timeout)
            for state, timeout in (timeouts or {}).items()
            if timeout is not None and float(timeout) >= 0.0
        }
        self._transitions = dict(self._DEFAULT_TRANSITIONS)
        if transitions:
            self._transitions.update(transitions)
        self.current_state = State.IDLE
        self.current_transaction_id = 0
        self.state_entry_time = self._time_fn()
        self._next_transaction_id = 0
        self._last_tick_time = self.state_entry_time
        self._tick_active = False
        self._has_ticked = False
        self._reset_failed_on_next_tick = False
        self._max_events_per_tick = max(1, int(max_events_per_tick))
        self._coalesced_events = {}
        self._coalesced_markers = set()
        self._highest_accepted_transaction_id = 0
        self.coalesced_events = 0
        self.processed_events = 0
        self.discarded_stale_events = 0
        self.discarded_future_events = 0
        self.dropped_events = 0
        self.dropped_sample_events = 0
        self.dropped_causal_events = 0

    def now(self) -> float:
        """Return the clock used by lifecycle state and event timestamps."""
        return float(self._time_fn())

    def begin_transaction(
        self,
        initial_state: State = State.IDLE,
        now: Optional[float] = None,
    ) -> int:
        """Start a newer lifecycle transaction from the FSM owner.

        Construction-time initialization is the one exception to the tick
        ownership rule.  Once the manager has processed its first tick, a
        transaction boundary is a state transition and must be requested from
        the timer-owned compute phase.
        """
        if self._has_ticked and not self._tick_active:
            raise RuntimeError("transactions are owned by tick()")
        with self._state_lock:
            candidate = self._new_transaction_id()
            if candidate <= self.current_transaction_id:
                candidate = self.current_transaction_id + 1
            previous = self.current_state
            self._next_transaction_id = candidate
            self.current_transaction_id = candidate
            self.current_state = initial_state
            self.state_entry_time = self._time_fn() if now is None else float(now)
            self._observe_transaction_id(candidate)
        if (
            self._tick_active
            and self._transition_handler is not None
            and previous != initial_state
        ):
            self._transition_handler(previous, initial_state, None)
        return candidate

    @classmethod
    def _new_transaction_id(cls) -> int:
        """Return a globally unique ID whose numeric order follows UUID time.

        ``UUID.int`` is lexicographically laid out with ``time_low`` first,
        so it wraps in numeric order every 2**32 UUID ticks even though the
        UUID timestamp continues forward.  Put the canonical 60-bit UUID1
        timestamp first and retain 64 bits of UUID identity as a tie-breaker.
        """
        identifier = uuid.uuid1()
        candidate = (int(identifier.time) << cls._TRANSACTION_UNIQUE_BITS) | (
            int(identifier.int) & ((1 << cls._TRANSACTION_UNIQUE_BITS) - 1)
        )
        with cls._TRANSACTION_ID_LOCK:
            if candidate <= cls._LAST_TRANSACTION_ID:
                candidate = cls._LAST_TRANSACTION_ID + 1
            cls._LAST_TRANSACTION_ID = candidate
        return candidate

    @classmethod
    def _observe_transaction_id(cls, transaction_id: int):
        """Advance the local HLC high-water mark after remote adoption."""
        with cls._TRANSACTION_ID_LOCK:
            if int(transaction_id) > cls._LAST_TRANSACTION_ID:
                cls._LAST_TRANSACTION_ID = int(transaction_id)

    def make_event(
        self,
        event_type: EventType,
        payload: Any = None,
        transaction_id: Optional[int] = None,
        created_at: Optional[float] = None,
    ) -> Event:
        """Construct an event without changing lifecycle state."""
        with self._state_lock:
            tx = (
                self.current_transaction_id
                if transaction_id is None
                else int(transaction_id)
            )
        return Event(
            type=event_type,
            transaction_id=tx,
            payload=payload,
            created_at=self._time_fn() if created_at is None else float(created_at),
        )

    def adopt_transaction(
        self,
        transaction_id: int,
        initial_state: State = State.IDLE,
        now: Optional[float] = None,
    ) -> bool:
        """Adopt a newer transaction received from another lifecycle owner.

        Adoption is intended for the compute side of a tick. It preserves one
        transaction identity across ROS nodes without allowing an older
        latched message to move the lifecycle backwards.
        """
        candidate = int(transaction_id)
        if candidate <= 0:
            return False
        if self._has_ticked and not self._tick_active:
            raise RuntimeError("transaction adoption is owned by tick()")
        with self._state_lock:
            if candidate < self.current_transaction_id:
                return False
            if candidate == self.current_transaction_id:
                return True
            previous = self.current_state
            self.current_transaction_id = candidate
            self._next_transaction_id = max(self._next_transaction_id, candidate)
            self.current_state = initial_state
            self.state_entry_time = self._time_fn() if now is None else float(now)
            self._observe_transaction_id(candidate)
        if (
            self._tick_active
            and self._transition_handler is not None
            and previous != initial_state
        ):
            self._transition_handler(previous, initial_state, None)
        return True

    def enqueue(self, event: Event) -> bool:
        """Push one immutable fact; this is the only callback-safe method."""
        if not isinstance(event, Event):
            raise TypeError("event must be an Event")
        key = self._coalescing_key(event)
        if key is not None:
            with self._inbox_state_lock:
                previous = self._coalesced_events.get(key)
                if (
                    key not in self._coalesced_markers
                    and self.inbox.qsize() >= self._sample_capacity
                ):
                    self.dropped_events += 1
                    self.dropped_sample_events += 1
                    return False
                self._coalesced_events[key] = self._coalesced_replacement(
                    previous, event, key
                )
                if key in self._coalesced_markers:
                    self._highest_accepted_transaction_id = max(
                        self._highest_accepted_transaction_id,
                        int(event.transaction_id),
                    )
                    self.coalesced_events += 1
                    return True
                marker = Event(
                    event.type,
                    event.transaction_id,
                    ("__coalesced__", key[2]),
                    event.created_at,
                )
                try:
                    self.inbox.put_nowait(marker)
                except queue.Full:
                    self._coalesced_events.pop(key, None)
                    self.dropped_events += 1
                    self.dropped_sample_events += 1
                    return False
                self._coalesced_markers.add(key)
                self._highest_accepted_transaction_id = max(
                    self._highest_accepted_transaction_id,
                    int(event.transaction_id),
                )
                return True
        with self._inbox_state_lock:
            try:
                self.inbox.put_nowait(event)
            except queue.Full:
                self.dropped_events += 1
                self.dropped_causal_events += 1
                return False
            self._highest_accepted_transaction_id = max(
                self._highest_accepted_transaction_id,
                int(event.transaction_id),
            )
        return True

    @classmethod
    def _coalescing_key(cls, event: Event):
        if (
            isinstance(event.payload, (tuple, list))
            and event.payload
            and event.payload[0] == "__coalesced__"
        ):
            stream = event.payload[1] if len(event.payload) > 1 else None
            return event.type, int(event.transaction_id), stream
        if event.type not in cls._COALESCED_EVENT_TYPES:
            if event.type != EventType.ACTION_FEEDBACK:
                return None
            payload = event.payload
            # Keep the action's active callback as a causal event. Feedback
            # samples for one generation are state and can be latest-sampled.
            if (
                not isinstance(payload, (tuple, list))
                or len(payload) < 2
                or payload[0] != "feedback"
            ):
                return None
            stream = ("feedback", payload[1])
        elif isinstance(event.payload, (tuple, list)) and event.payload:
            if event.type == EventType.COSTMAP_UPDATED:
                kind = event.payload[0]
                if kind not in ("full", "delta", "resync"):
                    return None
                stream = "full" if kind == "full" else "delta"
            else:
                stream = None
        elif event.type == EventType.COSTMAP_UPDATED:
            # A bare event is treated as a full-grid sample for defensive
            # recovery. Normal markers carry their stream explicitly.
            stream = "full"
        else:
            stream = None
        return event.type, int(event.transaction_id), stream

    @staticmethod
    def _coalesced_replacement(previous: Optional[Event], event: Event, key):
        """Return the bounded replacement for one latest-sample stream."""
        if (
            previous is not None
            and key[0] == EventType.COSTMAP_UPDATED
            and key[2] == "delta"
        ):
            # Applying only the newest OccupancyGridUpdate would skip cells
            # changed by the discarded patches. Invalidate the reconstruction
            # and wait for the next authoritative full grid instead.
            return Event(
                event.type,
                event.transaction_id,
                ("resync", None),
                max(previous.created_at, event.created_at),
            )
        return event

    def _materialize_event(self, marker: Event) -> Event:
        key = self._coalescing_key(marker)
        if key is None:
            return marker
        with self._inbox_state_lock:
            event = self._coalesced_events.pop(key, None)
            self._coalesced_markers.discard(key)
        if event is None:
            return Event(marker.type, marker.transaction_id, None, marker.created_at)
        return event

    def enqueue_type(
        self,
        event_type: EventType,
        payload: Any = None,
        transaction_id: Optional[int] = None,
    ) -> bool:
        """Construct and enqueue one fact for terse ROS ingress callbacks."""
        return self.enqueue(self.make_event(event_type, payload, transaction_id))

    def transition_to(
        self,
        new_state: State,
        event: Optional[Event] = None,
        now: Optional[float] = None,
    ) -> bool:
        """Change state; callers must be inside the single tick owner."""
        if not isinstance(new_state, State):
            raise TypeError("new_state must be a State")
        if not self._tick_active:
            raise RuntimeError("state transitions are owned by tick()")
        with self._state_lock:
            if new_state == self.current_state:
                return False
            previous = self.current_state
            self.current_state = new_state
            self.state_entry_time = self._time_fn() if now is None else float(now)
            self._reset_failed_on_next_tick = new_state == State.FAILED
        if self._transition_handler is not None:
            self._transition_handler(previous, new_state, event)
        return True

    def is_in(self, *states: State) -> bool:
        """Read-only state query for code that must avoid lifecycle booleans."""
        return self.current_state in states

    def check_timeouts(self, now: float) -> bool:
        """Force a timed-out active state to ``FAILED``."""
        timeout = self._timeouts.get(self.current_state)
        if timeout is None or now - self.state_entry_time < timeout:
            return False
        timeout_event = Event(
            EventType.TIMEOUT,
            self.current_transaction_id,
            {"state": self.current_state.value, "timeout": timeout},
            now,
        )
        if self._timeout_handler is not None:
            self._timeout_handler(timeout_event)
        self.transition_to(State.FAILED, timeout_event, now=now)
        return True

    def tick(
        self,
        now: Optional[float] = None,
        max_events: Optional[int] = None,
        compute_handler: Optional[Callable[[float], None]] = None,
    ) -> TickResult:
        """Run one complete Gather-Compute-Scatter cycle.

        ``compute_handler`` is invoked before ``tick`` releases its ownership
        flag.  This keeps timer-driven planning, action dispatch, and output
        publication in the same single-threaded ownership window as event
        application and state transitions.
        """
        if not self._tick_lock.acquire(False):
            return TickResult(
                0,
                0,
                0,
                False,
                self.current_state,
                self.current_transaction_id,
                0,
            )
        try:
            self._tick_active = True
            self._has_ticked = True
            tick_time = self._time_fn() if now is None else float(now)
            if self.current_state == State.FAILED and self._reset_failed_on_next_tick:
                self._reset_failed_on_next_tick = False
                self.begin_transaction(State.IDLE, now=tick_time)
            processed = stale = future = adopted_future = 0
            event_limit = (
                self._max_events_per_tick
                if max_events is None
                else max(0, int(max_events))
            )
            while processed + stale + future < event_limit:
                with self._inbox_state_lock:
                    try:
                        marker = self.inbox.get_nowait()
                    except queue.Empty:
                        break
                    event = self._materialize_event(marker)
                with self._inbox_state_lock:
                    highest_queued_transaction = (
                        self._highest_accepted_transaction_id
                    )
                if (
                    event.transaction_id < highest_queued_transaction
                    and highest_queued_transaction > self.current_transaction_id
                ):
                    self.discarded_stale_events += 1
                    stale += 1
                    continue
                if event.transaction_id < self.current_transaction_id:
                    self.discarded_stale_events += 1
                    stale += 1
                    continue
                if event.transaction_id != self.current_transaction_id:
                    # A message carrying a newer lifecycle identity is the
                    # first observation of a new distributed transaction.
                    # Adopt it before dispatching the fact; dropping it here
                    # would lose the very event that tells this node to move
                    # to the newer transaction.  Only older events are stale.
                    if event.transaction_id > self.current_transaction_id:
                        self.adopt_transaction(
                            event.transaction_id,
                            State.IDLE,
                            now=tick_time,
                        )
                        adopted_future += 1
                    else:
                        self.discarded_future_events += 1
                        future += 1
                        continue
                requested_state = None
                if self._event_handler is not None:
                    requested_state = self._event_handler(event)
                if requested_state is None:
                    requested_state = self._transitions.get(
                        (self.current_state, event.type)
                    )
                if requested_state is not None:
                    self.transition_to(requested_state, event, now=tick_time)
                self.processed_events += 1
                processed += 1
            if compute_handler is not None and self.current_state != State.FAILED:
                compute_handler(tick_time)
            timed_out = self.check_timeouts(tick_time)
            self._last_tick_time = tick_time
            return TickResult(
                processed,
                stale,
                future,
                timed_out,
                self.current_state,
                self.current_transaction_id,
                adopted_future,
            )
        finally:
            self._tick_active = False
            self._tick_lock.release()


__all__ = ["Event", "EventType", "LifecycleManager", "State", "TickResult"]
