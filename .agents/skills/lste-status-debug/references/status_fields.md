# LSTE Telemetry Fields & Schema Reference

Detailed field descriptions for `status.jsonl` and `bridge_status.jsonl`.

## 1. `status.jsonl` Payload Schema

Each record contains `wall`, `sim`, `event`, and `payload`:

| Path | Type | Description |
|---|---|---|
| `payload.work_item_id` | `int \| null` | Currently active WorkItem ID |
| `payload.work_item_state` | `string` | `"unresolved"` \| `"resolved"` |
| `payload.goal` | `[float, float] \| null` | Active navigation goal in `map` frame `[x, y]` |
| `payload.route_id` | `int` | Sequential ID of dispatched route |
| `payload.route_kind` | `string` | `"frontier_endpoint"` \| `"portal_probe"` \| `"portal_crossing"` |
| `payload.place_id` | `int` | ID of current topological Place |
| `payload.place_progress.phase` | `string` | `"observe"` (collecting evidence) or `"cross"` (crossing portal) |
| `payload.place_progress.unresolved_work_items` | `int` | Remaining items in current place |
| `payload.work_items` | `list[dict]` | Full list of WorkItems in memory |
| `payload.work_items[i].id` | `int` | WorkItem ID (corresponds to RViz `W{id}`) |
| `payload.work_items[i].state` | `string` | `"unresolved"` or `"resolved"` |
| `payload.work_items[i].anchor_xy` | `[float, float]` | 2D anchor location in odom/map coordinates |
| `payload.work_items[i].normal_xy` | `[float, float] \| null` | 2D viewing normal |
| `payload.work_items[i].attempts` | `list[dict]` | Attempts made to observe this WorkItem |
| `payload.work_items[i].attempts[j].state` | `string` | `"active"` \| `"succeeded"` \| `"failed"` |
| `payload.portal_transaction` | `dict` | State of portal crossing (`state`, `destination`, `gate`) |

## 2. `bridge_status.jsonl` Payload Schema

| Path | Type | Description |
|---|---|---|
| `payload.mode` | `string` | `"teb"` (autonomous navigation) or `"teleop"` (manual keyboard) |
| `payload.active_goal` | `[float, float, float] \| null` | `[x, y, yaw]` passed to move_base |
| `payload.reason` | `string` | Handoff / completion reason (e.g. `move_base_succeeded_terminal`) |
| `payload.turn_supervisor_state` | `string` | `"PASS_THROUGH"` or `"TURNING"` |
| `payload.navigation_hold` | `bool` | Whether motion commands are inhibited |
| `payload.stalled` | `bool` | Whether controller has detected zero progress |
