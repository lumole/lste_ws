"""Controller and task lifecycle callbacks for the TEB bridge."""


class TebGoalBridgeMissionControlMixin:
    def on_mode(self, message):
        mode = message.data.strip().lower()
        if not mode:
            return
        with self.lock:
            previous = self.mode
            self.mode = mode
            if previous == self.active_mode and mode != self.active_mode:
                self.cancel_locked("controller_switched_to_%s" % mode)
            elif mode == self.active_mode and previous != self.active_mode:
                self.dispatch_locked(force=True, reason="controller_switched_to_teb")

    def on_task_done(self, message):
        with self.lock:
            previous = self.task_done
            self.task_done = bool(message.data)
            if self.task_done:
                if self.persistent_execution:
                    # PersistentTebLocalPlanner observes this latch and returns
                    # the one action lease naturally, without a stop/preempt gap.
                    self.publish_bridge_status(
                        "persistent_task_done_waiting_for_action_terminal"
                    )
                else:
                    self.cancel_locked("task_done")
            elif previous and self._is_active_mode():
                self.persistent_installed_target_goal = None
                self.persistent_installed_target_transaction = 0
                self.persistent_target_approach_reported_transaction = 0
                self.dispatch_locked(force=True, reason="new_task")
