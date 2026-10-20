"""Continuous physical evidence for a directed portal transaction."""

import math

from global_frontier_portal_crossing import (
    check_portal_plane_crossing,
    portal_crossing_depth,
)


class GlobalFrontierPortalLiveCrossingMixin:
    """Commit a crossed graph edge before the local action terminal arrives."""

    def observe_active_portal_crossing(self, now=None):
        """Commit a physical portal crossing without waiting for TEB success.

        A portal is a graph-edge transaction, while a TEB terminal only says
        that its local action has stopped. Once odometry proves a crossing,
        retaining the old source place until a final few centimetres of TEB
        motion creates the repeated-room failure this graph prevents.
        """
        if (
            self.active_route_kind != "portal_transition"
            or self.active_frontier is None
            or getattr(self, "active_portal_crossing_observed", False)
            or getattr(self, "active_portal_crossing_rejected", False)
        ):
            return None
        crossing = self.active_portal_crossing_evidence()
        if not crossing.verified:
            return None
        destination = self.record_portal_place_arrival(
            self.active_frontier,
            now=now,
            evidence_source="continuous_odom",
            retry_on_unconfirmed=False,
        )
        if (
            destination is not None
            and bool(getattr(self, "persistent_execution", False))
            and getattr(self, "active_frontier", None) is not None
        ):
            released_route_id = int(getattr(self, "active_route_id", 0) or 0)
            release = getattr(self, "release_active_frontier", None)
            if callable(release):
                release(
                    discard_prefetch=True,
                    preserve_place_departure=False,
                    queue_successor=True,
                )
                publish = getattr(self, "publish_status", None)
                if callable(publish):
                    publish(
                        "portal_crossing_route_released",
                        route_id=released_route_id,
                        destination_place_id=int(destination.get("id", 0) or 0),
                        reason="physical_portal_crossing_committed",
                        controller_lease="persistent_stream",
                    )
            return destination
        return destination

    def observe_passive_portal_plane_crossing(self, now=None):
        """Commit certified portal and Place transition when base crosses doorway plane."""
        if not getattr(self, "passive_plane_crossing_certification", True):
            return None
        pose = getattr(self, "pose_odom", None)
        if pose is None:
            return None
        try:
            robot_xy = (float(pose.x), float(pose.y))
        except (AttributeError, TypeError, ValueError):
            return None

        if now is None:
            now = (
                getattr(self, "now_for", lambda _: 0.0)(self)
                if hasattr(self, "now_for")
                else 0.0
            )

        probe_ledger = getattr(self, "portal_probe_ledger", None)
        hypothesis_ledger = getattr(self, "portal_hypothesis_ledger", None)
        if probe_ledger is None and hypothesis_ledger is None:
            return None

        clearance = float(getattr(self, "clearance", 0.52))
        lateral_tol = max(1.2, clearance * 2.0)
        depth = portal_crossing_depth(clearance)

        if not hasattr(self, "_portal_plane_prev_signed"):
            self._portal_plane_prev_signed = {}

        # 1. Collect candidate doorways to check
        candidates = []
        if probe_ledger is not None:
            records = getattr(probe_ledger, "_records", {})
            for probe in list(records.values()):
                state = str(probe.get("state", "")).strip().lower()
                if state in ("crossed", "failed"):
                    continue
                portal_id = probe.get("portal_id")
                if portal_id is not None and hypothesis_ledger is not None:
                    hypo = hypothesis_ledger.get(portal_id)
                    if hypo and str(hypo.get("state", "")).strip().lower() == "crossed":
                        continue
                gate_xy = probe.get("physical_gate_xy") or probe.get("physical_gate")
                normal_xy = probe.get("normal_xy")
                if gate_xy is None or normal_xy is None:
                    continue
                dest_xy = (
                    float(gate_xy[0]) + float(normal_xy[0]) * depth,
                    float(gate_xy[1]) + float(normal_xy[1]) * depth,
                )
                candidates.append({
                    "kind": "probe",
                    "id": int(probe["id"]),
                    "gate_xy": (float(gate_xy[0]), float(gate_xy[1])),
                    "dest_xy": dest_xy,
                    "source_place_id": probe.get("source_place_id"),
                    "portal_id": portal_id,
                    "map_gate_xy": probe.get("map_gate_xy"),
                    "raw": probe,
                })

        if hypothesis_ledger is not None:
            records = getattr(hypothesis_ledger, "_records", {})
            for hypo in list(records.values()):
                if str(hypo.get("state", "")).strip().lower() in ("crossed", "failed"):
                    continue
                gate_xy = hypo.get("physical_gate")
                dest_xy = hypo.get("physical_destination")
                if gate_xy is None or dest_xy is None:
                    continue
                hypo_id = int(hypo["id"])
                if any(c.get("portal_id") == hypo_id for c in candidates):
                    continue
                candidates.append({
                    "kind": "hypo",
                    "id": hypo_id,
                    "gate_xy": (float(gate_xy[0]), float(gate_xy[1])),
                    "dest_xy": (float(dest_xy[0]), float(dest_xy[1])),
                    "source_place_id": hypo.get("source_place_id"),
                    "portal_id": hypo_id,
                    "map_gate_xy": hypo.get("map_gate"),
                    "raw": hypo,
                })

        committed = None
        for cand in candidates:
            key = (cand["kind"], cand["id"])
            prev_signed = self._portal_plane_prev_signed.get(key)
            crossed, current_signed, _lat = check_portal_plane_crossing(
                robot_xy,
                cand["gate_xy"],
                cand["dest_xy"],
                previous_signed_distance=prev_signed,
                lateral_tolerance=lateral_tol,
                max_approach_distance=3.0,
                allow_source_depth=depth,
            )
            if current_signed is not None:
                self._portal_plane_prev_signed[key] = current_signed

            if crossed:
                committed = self._commit_passive_portal_plane_crossing(
                    cand, robot_xy, now=now,
                )

        return committed

    def _commit_passive_portal_plane_crossing(self, cand, robot_xy, now):
        """Perform atomic certification and Place transition for a crossed portal plane."""
        hypothesis_ledger = getattr(self, "portal_hypothesis_ledger", None)
        probe_ledger = getattr(self, "portal_probe_ledger", None)
        gate_xy = cand["gate_xy"]
        dest_xy = cand["dest_xy"]
        source_place_id = cand["source_place_id"]
        if source_place_id is None:
            source_place_id = getattr(self, "current_physical_place_id", None) or 1
        try:
            source_place_id = int(source_place_id)
        except (TypeError, ValueError):
            source_place_id = 1

        portal_id = cand.get("portal_id")
        created = False
        hypo_record = None

        if hypothesis_ledger is not None:
            if portal_id is not None:
                hypo_record = hypothesis_ledger.get(portal_id)
            if hypo_record is None:
                certify_fn = getattr(hypothesis_ledger, "certify", None)
                if callable(certify_fn):
                    hypo_record, created = certify_fn(
                        source_place_id=source_place_id,
                        gate_xy=gate_xy,
                        destination_xy=dest_xy,
                        now=now,
                        task_version=str(getattr(self, "current_task_version", "") or ""),
                        map_gate_xy=cand.get("map_gate_xy"),
                    )
                    if hypo_record is not None:
                        portal_id = int(hypo_record["id"])
            if portal_id is not None:
                crossed_fn = getattr(hypothesis_ledger, "crossed", None)
                if callable(crossed_fn):
                    crossed_fn(portal_id, now=now)

        probe_id = cand["id"] if cand["kind"] == "probe" else None
        if probe_ledger is not None and probe_id is not None:
            if portal_id is not None:
                bind_fn = getattr(probe_ledger, "bind_portal", None)
                if callable(bind_fn):
                    bind_fn(probe_id, portal_id)
            certify_probe = getattr(probe_ledger, "certify_probe", None)
            if callable(certify_probe):
                certify_probe(
                    probe_id,
                    portal_id=portal_id,
                    now=now,
                    reason="passive_plane_crossing_certified",
                )
            else:
                rec = probe_ledger.get(probe_id)
                if rec is not None:
                    rec["state"] = "certified"
                    if portal_id is not None:
                        rec["portal_id"] = portal_id
                    rec["last_finished_at"] = float(now)
                    rec["last_reason"] = "passive_plane_crossing_certified"
            settle_work = getattr(self, "_settle_portal_probe_work_item", None)
            if callable(settle_work):
                probe_rec = probe_ledger.get(probe_id)
                if probe_rec is not None:
                    settle_work(probe_rec, now, "passive_plane_crossing_certified")

        publish = getattr(self, "publish_status", None)
        if callable(publish):
            if created or portal_id is not None:
                publish(
                    "portal_hypothesis_certified",
                    portal_id=portal_id,
                    portal_probe_id=probe_id,
                    source_place_id=source_place_id,
                    gate=[round(gate_xy[0], 3), round(gate_xy[1], 3)],
                    destination=[round(dest_xy[0], 3), round(dest_xy[1], 3)],
                    evidence_source="passive_plane_crossing",
                )
            if probe_id is not None:
                publish(
                    "portal_probe_promoted",
                    portal_probe_id=int(probe_id),
                    portal_id=portal_id,
                    source_place_id=source_place_id,
                    evidence="passive_plane_crossing",
                    crossing_required=False,
                )

        return self._commit_passive_place_transition(
            portal_id=portal_id,
            source_place_id=source_place_id,
            gate_xy=gate_xy,
            dest_xy=dest_xy,
            robot_xy=robot_xy,
            now=now,
        )

    def _commit_passive_place_transition(
        self, portal_id, source_place_id, gate_xy, dest_xy, robot_xy, now,
    ):
        memory = getattr(self, "region_memory", None)
        if memory is None:
            return None

        hypothesis_ledger = getattr(self, "portal_hypothesis_ledger", None)
        hypo = hypothesis_ledger.get(portal_id) if (hypothesis_ledger and portal_id) else None
        destination_place_id = hypo.get("destination_place_id") if hypo else None

        new_region = None
        tier = "new"

        if destination_place_id is not None:
            enter_bound = getattr(memory, "enter_bound_portal_destination", None)
            if callable(enter_bound):
                new_region, tier = enter_bound(
                    destination_place_id,
                    dest_xy[0],
                    dest_xy[1],
                    now,
                    entry_portal=gate_xy,
                    physical_xy=robot_xy,
                    physical_entry_portal=gate_xy,
                    source_place_id=source_place_id,
                )
            if new_region is None:
                new_region = memory.by_id(destination_place_id)
        else:
            cand_fn = getattr(memory, "candidate_tier", None)
            target_place_id = None
            if callable(cand_fn):
                t, cand_region = cand_fn(dest_xy[0], dest_xy[1], 0.0, physical_xy=robot_xy)
                if cand_region is not None and int(cand_region["id"]) != int(source_place_id):
                    target_place_id = int(cand_region["id"])
                    tier = t

            if target_place_id is not None:
                enter_bound = getattr(memory, "enter_bound_portal_destination", None)
                if callable(enter_bound):
                    new_region, tier = enter_bound(
                        target_place_id,
                        dest_xy[0],
                        dest_xy[1],
                        now,
                        entry_portal=gate_xy,
                        physical_xy=robot_xy,
                        physical_entry_portal=gate_xy,
                        source_place_id=source_place_id,
                    )
                if new_region is None:
                    new_region = memory.by_id(target_place_id)
            else:
                enter_dest = getattr(memory, "enter_new_portal_destination", None)
                if callable(enter_dest):
                    new_region, tier = enter_dest(
                        dest_xy[0],
                        dest_xy[1],
                        now,
                        entry_portal=gate_xy,
                        physical_xy=robot_xy,
                        physical_entry_portal=gate_xy,
                        source_place_id=source_place_id,
                    )
                else:
                    enter_fn = getattr(memory, "enter", None)
                    if callable(enter_fn):
                        new_region, tier = enter_fn(
                            dest_xy[0],
                            dest_xy[1],
                            now,
                            entry_portal=gate_xy,
                            physical_xy=robot_xy,
                            physical_entry_portal=gate_xy,
                            source_place_id=source_place_id,
                            force_new=True,
                        )

        if new_region is None:
            return None

        new_place_id = int(new_region["id"])

        if hypothesis_ledger is not None and portal_id is not None:
            bind_fn = getattr(hypothesis_ledger, "bind_destination", None)
            if callable(bind_fn):
                bind_fn(portal_id, new_place_id, now=now)

        self.current_physical_place_id = new_place_id
        self.active_frontier_region_id = new_place_id
        if hasattr(self, "active_work_item_place_id"):
            self.active_work_item_place_id = new_place_id

        work_items = getattr(self, "place_work_items", None)
        active_item_id = getattr(self, "active_work_item_id", None)
        if work_items is not None:
            if active_item_id is not None:
                reattribute_fn = getattr(work_items, "reattribute_work_item", None)
                if callable(reattribute_fn):
                    reattribute_fn(active_item_id, new_place_id, now=now)
            reattribute_near = getattr(work_items, "reattribute_items_near", None)
            if callable(reattribute_near):
                reattribute_near(source_place_id, new_place_id, robot_xy, now=now)

        publish = getattr(self, "publish_status", None)
        if callable(publish):
            publish(
                "portal_place_entered",
                portal_id=portal_id,
                region_id=new_place_id,
                region_tier=str(tier),
                reason="passive_plane_crossing_committed",
                evidence_source="passive_plane_crossing",
            )
        return new_region

