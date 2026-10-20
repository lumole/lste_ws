"""Place-graph admission policy for one safe frontier approach.

This mixin deliberately owns only the discrete decision about whether an
approach is local, a certified portal action, or a source-side probe.  Route
geometry and candidate scoring remain in their respective stages.
"""

from global_frontier_portal_certification import (
    certified_route_place_transitions,
    portal_certification_window,
    route_place_transition_candidates,
)
from global_frontier_portal_certification_wall import structural_wall_opening_cell
from global_frontier_portal_probes import PortalObservationProbe
from global_frontier_topology import place_action_tier, route_place_hops


class GlobalFrontierCandidatePlacePolicyMixin:
    """Keep place-graph admission separate from endpoint construction."""

    def _classify_candidate_approach(
        self, request, target_row, target_col, frontier_row, frontier_col,
        context, portal_sources,
    ):
        """Return the permitted place action for one safe approach."""
        components = request.components
        if (
            context.covered_place_reachable is not None
            and not context.covered_place_reachable[target_row, target_col]
        ):
            target_label = int(components.labels[target_row, target_col])
            counter = (
                "last_observed_place_reentry_skips"
                if target_label in context.observed_reentry_labels
                else "last_covered_place_transit_skips"
            )
            setattr(self, counter, getattr(self, counter) + 1)
            return None

        place_hops = None
        portal_gate_xy = None
        structural_probe = None
        if context.place_graph_ready and getattr(
            self, "certified_portals_enabled", True,
        ):
            known_free = ~(request.unknown | request.occupied)
            throat_radius_cells, window_margin_cells = (
                portal_certification_window(
                    request.message.info.resolution,
                    getattr(self, "region_topology_clearance", 0.50),
                    getattr(self, "place_furniture_max_span_m", 2.5),
                )
            )
            raw_place_hops = route_place_hops(
                components.labels,
                request.steps,
                target_row,
                target_col,
                source_label=context.source_label,
            )
            if raw_place_hops is None:
                self.last_place_graph_unresolved_candidates += 1
                structural_probe = self.unresolved_portal_probe(
                    request, frontier_row, frontier_col,
                )
                return (
                    None if structural_probe is None
                    else ("probe", 0, None, structural_probe)
                )
            transitions = certified_route_place_transitions(
                components.labels,
                request.steps,
                known_free,
                target_row,
                target_col,
                context.source_label,
                throat_radius_cells,
                window_margin_cells=window_margin_cells,
                cache=self._portal_transition_certification_cache,
                **self._portal_wall_support_kwargs(request.message, components)
            )
            place_hops = len(transitions)
            for transition in transitions:
                gate_xy = self.cell_xy(
                    request.message, *transition.portal_cell,
                )
                destination_xy = self.cell_xy(
                    request.message, *transition.destination_cell,
                )
                if self.sealed_portal_reentry(
                    gate_xy,
                    destination_xy,
                    map_to_physical_xy=request.map_to_physical_xy,
                ) is not None:
                    return None
            if transitions:
                portal_gate_xy = self.cell_xy(
                    request.message, *transitions[0].portal_cell,
                )
            # A labelled transition may be absent or uncertified while a
            # wall-bounded unknown opening is already observable. Keep it as a
            # source-side probe instead of discarding the only route to fresh
            # doorway evidence.
            if context.source_place_observed and place_hops in (None, 0):
                if getattr(self, "enable_dedicated_portal_probe_transaction", True):
                    structural_probe = self.unresolved_portal_probe(
                        request, frontier_row, frontier_col,
                    )
                    if structural_probe is not None:
                        return "probe", 0, portal_gate_xy, structural_probe
            if raw_place_hops > place_hops:
                self.last_uncertified_place_transition_skips += (
                    raw_place_hops - place_hops
                )
                if (
                    getattr(components, "structural_occupied", None) is not None
                    and place_hops == 0
                ):
                    raw_transitions = route_place_transition_candidates(
                        components.labels,
                        request.steps,
                        target_row,
                        target_col,
                        source_label=context.source_label,
                    )
                    if raw_transitions:
                        raw_transition = raw_transitions[0]
                        opening_cell = structural_wall_opening_cell(
                            components.structural_occupied,
                            raw_transition,
                            throat_radius_cells,
                            minimum_wall_span_cells=throat_radius_cells,
                            known_free=known_free,
                            minimum_opening_cells=2,
                            wall_search_radius=window_margin_cells,
                        )
                        if opening_cell is None:
                            opening_cell = raw_transition.portal_cell

                        d_row = (raw_transition.destination_cell[0] - raw_transition.source_cell[0]) if (raw_transition.source_cell and raw_transition.destination_cell) else 0
                        d_col = (raw_transition.destination_cell[1] - raw_transition.source_cell[1]) if (raw_transition.source_cell and raw_transition.destination_cell) else 0
                        if abs(d_row) >= abs(d_col):
                            normal = (1 if d_row >= 0 else -1, 0)
                        else:
                            normal = (0, 1 if d_col >= 0 else -1)

                        probe_candidate = PortalObservationProbe(
                            tuple(opening_cell),
                            tuple(normal),
                            None,
                            "source",
                            "structural_boundary",
                        )
                        register = getattr(self, "register_portal_probe", None)
                        if callable(register):
                            structural_probe = register(request, probe_candidate)
                        else:
                            structural_probe = probe_candidate

                        if structural_probe is not None:
                            gate_xy = self.cell_xy(
                                request.message, opening_cell[0], opening_cell[1],
                            )
                            if getattr(self, "enable_dedicated_portal_probe_transaction", True):
                                return "probe", 0, gate_xy, structural_probe
                    else:
                        if getattr(self, "enable_dedicated_portal_probe_transaction", True):
                            structural_probe = self.unresolved_portal_probe(
                                request, frontier_row, frontier_col,
                            )
                            if structural_probe is not None:
                                return "probe", 0, None, structural_probe
                    if getattr(self, "enable_dedicated_portal_probe_transaction", True):
                        return None
            if (
                (context.source_place_id is not None or context.source_place_observed)
                and place_hops >= 1
            ):
                self.last_cross_place_endpoint_deferrals += 1
                self._queue_first_portal_action(
                    request, transitions, frontier_row, frontier_col, portal_sources,
                )
                return None
            if place_hops > self.place_graph_hop_limit:
                self.last_place_graph_hop_skips += 1
                if request.allow_portal_transitions and not transitions:
                    self.last_portal_missing_transition_goals += 1
                self._queue_first_portal_action(
                    request, transitions, frontier_row, frontier_col, portal_sources,
                )
                return None

        action_tier = place_action_tier(place_hops, context.place_graph_ready)
        return (
            None if action_tier is None
            else (action_tier, place_hops, portal_gate_xy, structural_probe)
        )
