"""Progress-only hook; does not change prompts, actions, budgets or model responses."""
import time


class ProgressMixin:
    def query(self):
        started = time.monotonic()
        print(f'SWE DECISION {self.n_calls + 1}: requesting model', flush=True)
        reply = super().query()
        print(f'SWE DECISION {self.n_calls}: response received in {time.monotonic()-started:.1f}s; '
              f'actions={len(reply.get("extra", {}).get("actions", []))}', flush=True)
        return reply

    def execute_actions(self, message):
        print('SWE TOOLS: executing selected action(s)', flush=True)
        replies = super().execute_actions(message)
        print(f'SWE TOOLS: {len(replies)} observation(s) saved', flush=True)
        return replies
