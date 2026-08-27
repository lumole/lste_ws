#!/usr/bin/env python3
"""Validate the office-building benchmark contract before launching it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on ROS host setup
    raise SystemExit("PyYAML is required: %s" % exc)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "worlds/benchmark/office_building_v1_manifest.yaml"


class Validation:
    def __init__(self) -> None:
        self.errors = []
        self.warnings = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def resolve_workspace_path(value: str, root: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_model_dirs() -> list[Path]:
    paths = []
    for value in os.environ.get("GAZEBO_MODEL_PATH", "").split(":"):
        if value:
            paths.append(Path(value))
    paths.extend(
        [
            Path.home() / ".gazebo/models",
            Path("/usr/share/gazebo-11/models"),
        ]
    )
    return paths


def model_exists(model_name: str) -> bool:
    return any((base / model_name / "model.sdf").is_file() for base in available_model_dirs())


def parse_required_level(manifest: dict, level: str, result: Validation) -> None:
    levels = manifest.get("levels", {}) or {}
    if level not in levels:
        result.error("unknown level %r; expected one of %s" % (level, ", ".join(sorted(levels))))


def world_for_level(manifest: dict, level: str, result: Validation) -> str:
    """Resolve the committed world artifact for one deterministic level."""
    levels = manifest.get("levels", {}) or {}
    profile = levels.get(level, {}) or {}
    world_value = str(profile.get("world") or manifest.get("world") or "").strip()
    if not world_value:
        result.error("manifest has no world path for level %r" % level)
    return world_value


def validate(manifest_path: Path, level: str, runtime: bool) -> tuple[Validation, dict]:
    result = Validation()
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        result.error("cannot read manifest %s: %s" % (manifest_path, exc))
        return result, {}

    parse_required_level(manifest, level, result)
    world_value = world_for_level(manifest, level, result)
    if not world_value:
        return result, manifest
    world_path = resolve_workspace_path(world_value, ROOT)
    if not world_path.is_file():
        result.error("world file does not exist: %s" % world_path)
        return result, manifest

    try:
        root = ET.parse(world_path).getroot()
    except (OSError, ET.ParseError) as exc:
        result.error("cannot parse world %s: %s" % (world_path, exc))
        return result, manifest
    world = root.find("world")
    if world is None:
        result.error("world XML has no <world> element")
        return result, manifest
    expected_world_name = str(manifest.get("world_name", "")).strip()
    if expected_world_name and world.get("name") != expected_world_name:
        result.error(
            "world name mismatch: manifest=%r XML=%r"
            % (expected_world_name, world.get("name"))
        )

    model_nodes = list(world.findall("model"))
    include_nodes = list(world.findall("include"))
    names = [node.get("name") for node in model_nodes]
    names.extend(node.findtext("name") for node in include_nodes)
    duplicate_names = sorted({name for name in names if name and names.count(name) > 1})
    if duplicate_names:
        result.error("duplicate world entity names: %s" % ", ".join(duplicate_names))
    if "pro3" in names:
        result.error("benchmark world must not contain Pro3; the launcher spawns it")
    if "ground_plane" not in names:
        result.error("world has no ground_plane model")
    if "office_building_structure" not in names:
        result.error("world has no office_building_structure model")

    include_uris = []
    for include in include_nodes:
        uri = (include.findtext("uri") or "").strip()
        if not uri.startswith("model://"):
            result.error("include %r does not use model:// URI" % (include.findtext("name")))
            continue
        include_uris.append(uri[len("model://") :])
    for model_name in sorted(set(include_uris)):
        if not model_exists(model_name):
            result.error(
                "required Gazebo model is not installed locally: %s "
                "(searched GAZEBO_MODEL_PATH and ~/.gazebo/models)" % model_name
            )

    required_models = manifest.get("required_models", []) or []
    for model_name in required_models:
        if not model_exists(str(model_name)):
            result.error("manifest required model is unavailable: %s" % model_name)

    building = manifest.get("building", {}) or {}
    bounds = building.get("bounds_m", [])
    if len(bounds) != 4:
        result.error("building.bounds_m must be [min_x, min_y, max_x, max_y]")
    else:
        min_x, min_y, max_x, max_y = [float(value) for value in bounds]
        target = (manifest.get("targets", {}) or {}).get("primary", {}) or {}
        target_xy = target.get("pose_xy", [])
        if len(target_xy) != 2:
            result.error("targets.primary.pose_xy must contain x and y")
        elif not (min_x <= float(target_xy[0]) <= max_x and min_y <= float(target_xy[1]) <= max_y):
            result.error("primary target lies outside building bounds: %s" % target_xy)

    target = (manifest.get("targets", {}) or {}).get("primary", {}) or {}
    target_model = str(target.get("model", "")).strip()
    target_name = str(target.get("pose_xy", "")).strip()
    if target_model != "cup_yellow":
        result.warning("primary target model is %r, not the default cup_yellow" % target_model)
    if target_model and target_model not in names:
        result.error("primary target model %r is not an entity in the world" % target_model)
    target_room = str(target.get("room", "")).strip()
    room_ids = {str(room.get("id", "")).strip() for room in manifest.get("rooms", []) or []}
    if target_room not in room_ids:
        result.error("primary target room %r is not declared in rooms" % target_room)

    for profile_name, profile in (manifest.get("robot_profiles", {}) or {}).items():
        pose = profile.get("pose", [])
        if len(pose) != 3:
            result.error("robot profile %s pose must be [x, y, yaw]" % profile_name)
    door_ids = {str(door.get("id", "")).strip() for door in manifest.get("doors", []) or []}
    for room in manifest.get("rooms", []) or []:
        for door_id in room.get("entry_doors", []) or []:
            if door_id not in door_ids:
                result.error("room %s references unknown door %s" % (room.get("id"), door_id))

    actual_hash = sha256(world_path)
    declared_hash = str(manifest.get("world_sha256", "")).strip()
    if declared_hash and declared_hash != "generated_at_build_time" and declared_hash != actual_hash:
        result.error("world_sha256 mismatch: manifest=%s actual=%s" % (declared_hash, actual_hash))

    if runtime:
        required_topics = ((manifest.get("acceptance", {}) or {}).get("require_topics", []) or [])
        try:
            output = subprocess.check_output(["rostopic", "list"], text=True, stderr=subprocess.STDOUT, timeout=5)
            topics = set(output.splitlines())
            for topic in required_topics:
                if str(topic) not in topics:
                    result.error("required runtime topic is not currently advertised: %s" % topic)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            result.error("runtime topic check failed: %s" % exc)

    details = {
        "manifest": str(manifest_path),
        "world": str(world_path),
        "world_sha256": actual_hash,
        "world_name": world.get("name"),
        "level": level,
        "world_entities": len(names),
        "model_includes": len(include_nodes),
        "errors": result.errors,
        "warnings": result.warnings,
    }
    return result, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--level", default="level_2")
    parser.add_argument("--runtime", action="store_true", help="also require ROS topics to be advertised")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    result, details = validate(manifest_path, args.level, args.runtime)
    if args.as_json:
        print(json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("office_building validation: %s" % ("PASS" if not result.errors else "FAIL"))
        print("  world: %s" % details.get("world", "unknown"))
        print("  sha256: %s" % details.get("world_sha256", "unknown"))
        print("  level: %s" % args.level)
        for warning in result.warnings:
            print("  WARN: %s" % warning)
        for error in result.errors:
            print("  ERROR: %s" % error)
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
