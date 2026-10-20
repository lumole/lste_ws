"""Physical-progress observation for an active frontier route."""

import math
from types import SimpleNamespace

import numpy as np

import rospy

from global_frontier_models import ActiveRouteObservation
from global_frontier_physical_progress import physical_progress_decision
from global_frontier_portal_certification import portal_certification_window
from global_frontier_portal_probes import PortalObservationProbe
from global_frontier_structural_boundaries import structural_boundary_candidates
from global_frontier_grid import frontier_mask
from global_frontier_topology import copy_component_evidence


class GlobalFrontierExecutionObservationMixin:
    """Collect only map, odometry, and local-information progress evidence."""

    def observe_live_portal_openings_from_map(self, message, robot_map, now):
        """Run the live doorway pass when full planning is intentionally deferred."""
        build_context = getattr(self, "_build_frontier_map_context", None)
        if not callable(build_context) or message is None or robot_map is None:
            return 0
        map_context = build_context(message)
        if map_context is None:
            return 0
        unknown = map_context.unknown
        known_free = map_context.known_free
        if (
            not isinstance(unknown, np.ndarray)
            or not isinstance(known_free, np.ndarray)
            or unknown.shape != known_free.shape
        ):
            return 0
        resolution = max(1e-6, float(message.info.resolution))
        rows, cols = np.indices(unknown.shape)
        world_x = (
            float(message.info.origin.position.x) + (cols + 0.5) * resolution
        )
        world_y = (
            float(message.info.origin.position.y) + (rows + 0.5) * resolution
        )
        distance_steps = np.hypot(
            world_x - float(robot_map[0]),
            world_y - float(robot_map[1]),
        ) / resolution
        route_steps = np.where(known_free, distance_steps, -1.0)
        snapshot = SimpleNamespace(
            message=message,
            map_context=map_context,
            route_graph=SimpleNamespace(
                route_steps=route_steps,
                frontier=frontier_mask(known_free, unknown),
            ),
            robot_map=robot_map,
            now=now,
        )
        return self.observe_live_portal_openings(snapshot)

    def observe_live_portal_openings(self, snapshot):
        """Register doorway evidence while any route is being executed.

        This pass is deliberately observation-only.  It consumes the same
        frontier/structural helpers as normal selection, limits evidence to
        the current sensor horizon, and writes durable PortalProbe identities;
        it never changes ``active_frontier`` or publishes a replacement goal.
        """
        if snapshot is None:
            return 0
        components = getattr(snapshot.map_context, "components", None)
        structural_occupied = getattr(components, "structural_occupied", None)
        labels = getattr(components, "labels", None)
        ledger = getattr(self, "portal_probe_ledger", None)
        register = getattr(self, "register_portal_probe", None)
        place_id = getattr(self, "current_physical_place_id", None)
        if (
            components is None
            or structural_occupied is None
            or ledger is None
            or not callable(register)
            or place_id is None
        ):
            return 0

        message = snapshot.message
        map_context = snapshot.map_context
        route_graph = snapshot.route_graph
        unknown = map_context.unknown
        occupied = map_context.occupied
        known_free = ~(unknown | occupied)
        steps = route_graph.route_steps
        if (
            not isinstance(unknown, np.ndarray)
            or not isinstance(occupied, np.ndarray)
            or not isinstance(steps, np.ndarray)
            or unknown.shape != occupied.shape
            or unknown.shape != steps.shape
        ):
            return 0

        projector_factory = getattr(self, "planar_xy_projector", None)
        map_frame = getattr(message.header, "frame_id", "map") or "map"
        map_to_physical = (
            None
            if projector_factory is None
            else projector_factory("odom", map_frame)
        )
        request = SimpleNamespace(
            message=message,
            components=components,
            unknown=unknown,
            occupied=occupied,
            now=snapshot.now,
            map_to_physical_xy=map_to_physical,
        )

        resolution = max(1e-6, float(message.info.resolution))
        horizon = getattr(self, "scan_observation_horizon", None)
        try:
            horizon = float(horizon)
        except (TypeError, ValueError):
            horizon = 0.0
        horizon_cells = (
            None if horizon <= 0.0 else horizon / resolution + 1.0
        )

        def in_sensor_horizon(cell):
            try:
                row, col = int(cell[0]), int(cell[1])
                route_steps = float(steps[row, col])
            except (IndexError, TypeError, ValueError):
                return False
            if horizon_cells is None:
                return route_steps >= 0.0
            cell_x = (
                float(message.info.origin.position.x)
                + (col + 0.5) * resolution
            )
            cell_y = (
                float(message.info.origin.position.y)
                + (row + 0.5) * resolution
            )
            return math.hypot(
                cell_x - float(snapshot.robot_map[0]),
                cell_y - float(snapshot.robot_map[1]),
            ) <= float(horizon) + resolution

        before = {
            int(record.get("id"))
            for record in ledger.snapshot()
            if record.get("id") is not None
        }
        # A doorway may already be known free on both sides by the time the
        # robot passes it.  Structural evidence preserves that opening even
        # though no frontier cell remains to trigger the normal detector.
        source_label = None
        component_at_pose = getattr(self, "component_at_map_position", None)
        if callable(component_at_pose):
            try:
                component = component_at_pose(
                    message,
                    components,
                    ~(unknown | occupied),
                    snapshot.robot_map[0],
                    snapshot.robot_map[1],
                )
                source_label = None if component is None else int(component["label"])
            except (AttributeError, IndexError, TypeError, ValueError):
                source_label = None
        if (
            isinstance(labels, np.ndarray)
            and source_label is not None
            and source_label > 0
        ):
            throat_radius, window_margin = portal_certification_window(
                resolution,
                getattr(self, "region_topology_clearance", 0.50),
                getattr(self, "place_furniture_max_span_m", 2.5),
            )
            route_radius = max(
                1,
                int(math.ceil(
                    float(getattr(self, "clearance", 0.35)) / resolution
                )),
            )
            portal_max_opening_m = float(getattr(self, "portal_max_opening_m", 1.8))
            portal_max_opening_cells = int(math.ceil(portal_max_opening_m / resolution))
            boundaries = structural_boundary_candidates(
                ~(unknown | occupied),
                structural_occupied,
                steps,
                unknown=unknown,
                labels=labels,
                source_label=source_label,
                support_radius=throat_radius,
                minimum_wall_span_cells=throat_radius,
                wall_search_radius=window_margin,
                minimum_opening_cells=2 * route_radius + 1,
                maximum_opening_cells=portal_max_opening_cells,
                resolution=resolution,
                max_opening_m=portal_max_opening_m,
            )
            for boundary in boundaries:
                if not in_sensor_horizon(boundary.source_cell):
                    continue
                # Live background recognition must not promote a wall crack or
                # an unknown room boundary into a doorway.  Require a known
                # free far-side cell and a distinct structural core beyond the
                # throat.  Unknown-side openings remain owned by the normal
                # frontier/WorkItem selector, where a source-side probe can be
                # deliberately routed and certified.
                destination = boundary.destination_cell
                if not (
                    0 <= int(destination[0]) < known_free.shape[0]
                    and 0 <= int(destination[1]) < known_free.shape[1]
                    and bool(known_free[destination])
                ):
                    continue
                destination_label = None
                for distance in range(1, int(window_margin) + 1):
                    row = int(destination[0]) + distance * int(boundary.normal[0])
                    col = int(destination[1]) + distance * int(boundary.normal[1])
                    if not (
                        0 <= row < labels.shape[0]
                        and 0 <= col < labels.shape[1]
                    ):
                        break
                    label = int(labels[row, col])
                    if label > 0:
                        destination_label = label
                        break
                if destination_label is None or destination_label == source_label:
                    continue
                self._register_live_structural_probe(request, boundary)

        after = {
            int(record.get("id")): record
            for record in ledger.snapshot()
            if record.get("id") is not None
        }
        new_ids = sorted(set(after) - before)
        for probe_id in new_ids:
            record = after[probe_id]
            publish = getattr(self, "publish_status", None)
            if callable(publish):
                publish(
                    "portal_opening_observed",
                    probe_id=int(probe_id),
                    source=str(record.get("observation_source", "live_scan")),
                    source_place_id=int(record.get("source_place_id", place_id)),
                    gate=[
                        round(float(value), 3)
                        for value in record.get("physical_gate_xy", ())
                    ],
                    map_epoch=record.get("last_map_epoch"),
                )
        return len(new_ids)

    def _register_live_structural_probe(self, request, boundary):
        """Register one known-free wall opening without scheduling motion."""
        probe = PortalObservationProbe(
            tuple(boundary.opening_cell),
            tuple(boundary.normal),
            None,
            "source",
            "live_structural_scan",
        )
        register = getattr(self, "register_portal_probe", None)
        if callable(register):
            register(request, probe)

    def observe_active_route(
        self, message, steps, validation, unknown, components, row, col, x, y,
        robot_map, now, portal_transition,
    ):
        """Update route-progress evidence and return the current map facts."""
        if hasattr(self, "observe_passive_portal_plane_crossing"):
            self.observe_passive_portal_plane_crossing(now)
        if portal_transition:
            self.observe_portal_gate_approach(now)
            # Portal topology is advanced by continuous physical evidence, not
            # by an arbitrary controller terminal that may arrive late or not
            # at all after the base has already crossed the doorway.
            portal_destination = self.observe_active_portal_crossing(now)
            if portal_destination is not None and self.active_frontier is None:
                # Persistent portal crossing may release the old semantic
                # route in the same serialized tick. Do not continue below
                # with reset route fields or let the old watchdog process a
                # route that has already been committed and released.
                return None
        distance = math.hypot(x - robot_map[0], y - robot_map[1])
        if (
            portal_transition
            and distance <= self.endpoint_terminal_wait_radius
        ):
            transaction = getattr(self, "portal_transaction", None)
            if transaction is not None:
                transaction.destination_standoff(self.active_route_id)
        information = self.frontier_information(unknown, row, col)
        component = None
        if not portal_transition:
            component = self.topology_component_evidence(
                message, components, row, col, steps=steps,
            )
            if component is not None:
                self.active_frontier_component = copy_component_evidence(component)
            observation_ready = self.active_frontier_observation_ready(distance)
            observation_session_started = (
                observation_ready
                and self.active_observation_session_started_at is None
            )
            observed_region = self.region_memory.observe(
                x,
                y,
                information,
                robot_map[0],
                robot_map[1],
                now,
                component=component,
                region_id=self.active_frontier_region_id,
                observation_ready=observation_ready,
                observation_session_started=observation_session_started,
                physical_robot_xy=(
                    None
                    if getattr(self, "pose_odom", None) is None
                    else (float(self.pose_odom.x), float(self.pose_odom.y))
                ),
            )
            if observation_session_started and observed_region is not None:
                self.active_observation_session_started_at = now
                self.publish_status(
                    "frontier_observation_session_started",
                    route_id=int(self.active_route_id),
                    region_id=int(observed_region["id"]),
                    goal=[round(float(x), 3), round(float(y), 3)],
                    observer=[
                        round(float(robot_map[0]), 3),
                        round(float(robot_map[1]), 3),
                    ],
                    observation_distance=round(float(distance), 3),
                )
                rospy.loginfo(
                    "Global frontier began observation session route_id=%d "
                    "region_id=%d distance=%.2fm",
                    self.active_route_id,
                    observed_region["id"],
                    distance,
                )

        route_distance = float(steps[row, col]) * float(message.info.resolution)
        if validation is not None:
            costmap_distance = self.candidate_costmap_distance(validation, x, y)
            if costmap_distance is not None:
                route_distance = costmap_distance

        route_progress = (
            self.active_best_path_distance is None
            or route_distance < self.active_best_path_distance - self.progress_epsilon
        )
        goal_progress = (
            self.active_best_goal_distance is None
            or distance < self.active_best_goal_distance - self.progress_epsilon
        )
        if route_progress:
            self.active_best_path_distance = route_distance
        if goal_progress:
            # Kept for old status readers; this is explicitly the physical
            # distance to the current fixed endpoint.
            self.active_best_goal_distance = distance
            self.active_best_distance = distance

        odom_detour_progress = self._active_route_odom_detour_progress()
        odom_coverage_progress = self._entered_novel_active_odom_cell()
        self._record_active_route_progress(
            now,
            route_progress,
            goal_progress,
            odom_detour_progress,
            odom_coverage_progress,
        )
        self._update_active_route_position(robot_map)
        return ActiveRouteObservation(
            distance=distance,
            information=information,
            component=component,
            route_distance=route_distance,
        )

    def observe_portal_gate_approach(self, now):
        """Start the bounded edge action after physically reaching its gate.

        The map frame moves during online SLAM. The gate was projected to
        ``odom`` when this route became active, so the check remains attached
        to one physical doorway. The existing endpoint envelope is reused as
        the gate tolerance; this is a lifecycle boundary, not a new tuning
        parameter.
        """
        if (
            self.active_route_kind != "portal_transition"
            or self.active_portal_gate_approached_at is not None
            or self.active_portal_gate_odom_xy is None
            or self.pose_odom is None
        ):
            return False
        gate_x, gate_y = self.active_portal_gate_odom_xy
        distance = math.hypot(
            float(self.pose_odom.x) - float(gate_x),
            float(self.pose_odom.y) - float(gate_y),
        )
        # ``completed_radius`` already defines the physical footprint around
        # a reached viewpoint. It tolerates the small map/odom residual left
        # by an online SLAM update while remaining smaller than a room-scale
        # route. Do not introduce a second doorway-distance parameter.
        gate_tolerance = max(
            float(self.endpoint_terminal_wait_radius),
            float(self.completed_radius),
        )
        if distance > gate_tolerance:
            return False
        self.active_portal_gate_approached_at = float(now)
        transaction = getattr(self, "portal_transaction", None)
        if transaction is not None:
            transition = transaction.gate_reached(self.active_route_id)
            if transition is not None:
                self.publish_status(
                    "portal_transaction_phase",
                    transaction_id=int(transition.transaction_id),
                    route_id=int(transition.route_id),
                    state=transition.state,
                    transition=transition.last_transition,
                    reason=transition.last_reason,
                )
        self.publish_status(
            "portal_edge_started_at_gate",
            route_id=int(self.active_route_id),
            portal_gate=[round(float(gate_x), 3), round(float(gate_y), 3)],
            gate_distance=round(float(distance), 3),
            gate_tolerance=round(float(gate_tolerance), 3),
        )
        return True

    def _active_route_odom_detour_progress(self):
        """Accept only a new maximum odom excursion during an active route."""
        if self.active_start_odom_xy is None or self.pose_odom is None:
            return False
        distance = math.hypot(
            float(self.pose_odom.x) - self.active_start_odom_xy[0],
            float(self.pose_odom.y) - self.active_start_odom_xy[1],
        )
        if distance <= self.active_best_detour_odom_distance + self.progress_epsilon:
            return False
        self.active_best_detour_odom_distance = distance
        return True

    def _record_active_route_progress(
        self, now, route_progress, goal_progress, odom_detour_progress,
        odom_coverage_progress,
    ):
        """Renew the watchdog only from the physical-progress contract.

        ``route_progress`` and ``goal_progress`` are map-derived diagnostics.
        They may improve solely because SLAM moved the coordinate frame or
        relabelled free space, which was the cause of the long Level 4 stall.
        Keep recording that fact, but never let it hide a stationary base from
        the route watchdog.
        """
        decision = physical_progress_decision(
            route_progress=route_progress,
            goal_progress=goal_progress,
            odom_detour_progress=odom_detour_progress,
            odom_novel_coverage=odom_coverage_progress,
        )
        if not decision.renew_watchdog:
            if decision.signal != "none":
                self.active_last_progress_signal = decision.signal
            return
        self.active_progress_time = now
        self.active_last_progress_signal = decision.signal

    def _update_active_route_position(self, robot_map):
        """Keep the last map position only after a meaningful displacement."""
        self.remember_active_route_position(robot_map)
        if self.active_last_robot_xy is None:
            self.active_last_robot_xy = (robot_map[0], robot_map[1])
        elif math.hypot(
            robot_map[0] - self.active_last_robot_xy[0],
            robot_map[1] - self.active_last_robot_xy[1],
        ) >= self.progress_epsilon:
            self.active_last_robot_xy = (robot_map[0], robot_map[1])
