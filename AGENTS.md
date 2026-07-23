# Workspace Rules

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
