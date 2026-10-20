# Workspace Rules

## Code Organization

- Keep ROS entrypoints thin: initialize the node, compose focused modules, and
  start the runtime. Do not accumulate callbacks, parameter parsing, mutable
  state setup, geometry, and control policy in one script.
- When a module has multiple independent responsibilities or is becoming hard
  to navigate, split it by responsibility with explicit names (for example
  `*_parameters.py`, `*_state.py`, `*_ros.py`, `*_routes.py`, or `*_control.py`).
  Preserve the existing public ROS interfaces unless the task explicitly
  changes them.
- Do not split merely to reduce line count. Keep closely related logic together
  and avoid very small wrapper files or deep mixin chains that force readers to
  jump across many modules.
- Build commands from named arrays or helper functions instead of maintaining
  long, shell-escaped command strings, so each argument remains easy to edit
  and review.
- Add a focused structural or behavioral regression test whenever a running
  node is split, and install every sibling Python module needed by an
  install-space or devel-space launch.

## Documentation Naming

- Except for the fixed index `docs/introduction.md`, files under `docs/` use
  the `YYYYMMDD_topic.extension` format; Markdown documents use
  `YYYYMMDD_topic.md`.
- The date prefix is written as year, month, and day in that order. Keep the
  fixed index path unchanged when updating its links.

## Fixed-Goal Test Logs

- Every `rltest` invocation must create one timestamped parent directory under
  `runtime/rl_fixed_goal_test/logs/` using the format `YYYYMMDD_HHMMSS`.
- All program logs for that invocation must be `.log` files inside that one
  timestamped directory. Do not write the formal experiment logs directly to
  `~/.ros/log` or mix files from multiple runs in one directory.
- The timestamped directory must contain one separate log for every launched
  process. Every log filename must repeat the same timestamp as its parent,
  using `YYYYMMDD_HHMMSS_<process>.log` (for example
  `20260723_191530_lifecycle.log`, `20260723_191530_navigation.log`, or
  `20260723_191530_frontier_manager.log`).
- The lifecycle file is named `YYYYMMDD_HHMMSS_lifecycle.log` and must record
  the resolved startup configuration, controller, speed, initial pose, target,
  world, Git revision, and start/stop events.
- Each process writes only its own log file. Log lines must include a timestamp,
  log level, process name, event, and relevant parameters.
- Retain test log directories for 15 days unless a test-specific configuration
  explicitly sets a different retention period.

## Failure Evidence

- `lste_navigation_metrics` must keep a bounded in-memory evidence ring and
  write failure episodes to the current run directory, never to a shared
  global log.
- Every episode uses one `<run_timestamp>-F####` failure ID in
  `failure_started`, related events, and `failure_snapshot_ready` records.
- A failure snapshot must retain the pre/post context needed to distinguish
  planner no-path, local obstacle, controller stall, TF/transform failure,
  ownership race, and exhausted recovery. It must contain summarized pose,
  goal/route transaction, command/mux, scan, Navfn/TEB, costmap, feedback,
  and recovery state; raw images and full costmap arrays are not required.
- Each closed episode must also be written as one atomic
  `<run_timestamp>_failure_<sequence>.json` artifact in the same run directory;
  the path must be included in `failure_snapshot_ready` and the experiment
  record so an investigator can open one failure without parsing a large log.
- Transient candidate/planning waits are context events, not failures. Only a
  terminal failure, explicit route failure, unexpected preemption, or a
  sustained no-progress/lease wait may start a failure episode.

<!-- FAST_INDEX_START -->
## Fast Navigation & Asset Directory
- **Canonical Worlds & Generators**:
  - T-Junction Office (Primary Benchmark): Source `scripts/tests/generate_t_junction_world.py` -> outputs `worlds/t_junction_office_no_pro3.world`, manifest `worlds/t_junction_office_manifest.yaml`.
  - Complex Office (Large Benchmark): Source `scripts/tests/office_building/generate_world.py` -> `worlds/benchmark/office_building_v1_level_*.world`.
  - Always modify the generator script first, regenerate, and run `gz sdf -k <world_file>` to validate.
- **Gazebo Model Placement Rules**:
  - `cup_yellow`: Target yellow cup, desk $Z=0.75$, scale=(1.5, 1.5, 1.5).
  - `desk_brown`: Office desk, top surface $Z=0.72$, length 1.6m, width 0.8m.
  - `monitor_1`, `monitor_2`: Computer monitors, base $Z=0.72$, screen flanking offset $\pm 0.5\text{m}$.
  - `table_conference_2`: Large conference table, top surface $Z=0.75$.
  - `fire_hydrant`: Ground safety item $Z=0.0$, must tuck into wall corners (墙脚).
  - `target_room_occluder`: Screen divider box (1.6 x 0.25 x 1.6m), $Z=0.8$.
- **Core Runtime Commands**:
  - Stack restart: `./run_demo_teb.sh restart` (autonomous TEB) or `CONTROLLER_MODE=teleop ./run_demo_teb.sh restart`.
  - Switch controller: `rostopic pub -1 /lste/controller_mode std_msgs/String "data: 'teb'"` (or `'teleop'`).
  - Diagnostics: `python3 .agents/skills/lste-status-debug/scripts/inspect_demo_status.py`.
- **Detailed Full Index**: Refer to `docs/20261019_project_fast_index.md`.
<!-- FAST_INDEX_END -->
