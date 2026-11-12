"""mini v2 adapter: original query/budget/tool executor, temporary experience message."""
import time
from pathlib import Path
from minisweagent.agents.default import DefaultAgent
from .core import digest, read, tool_sequence_check, visible, manifest_check
from .api import JSONClient
from .bank import Bank
from .prompts import SITUATION


class SearchExperienceAgent(DefaultAgent):
    def __init__(self, model, env, *, r2mem_config, **kwargs):
        super().__init__(model, env, **kwargs)
        self.r2 = read(r2mem_config)
        self.bank = Bank(self.r2['index_dir'])
        groups = manifest_check(self.bank.data['manifest'])
        task = self.r2['task_id']
        if task not in groups[self.r2.get('split', 'development')] or task in groups['construction']:
            raise ValueError('Online task must be a held-out manifest task')
        if self.r2.get('split', 'development') not in ('development', 'evaluation'):
            raise ValueError('Only development/evaluation online tasks allowed')
        self.audit = []
        # Per-task output directory must be new, to prevent cross-run ledger mixing.
        directory = Path(self.r2['audit_dir'])
        directory.mkdir(parents=True, exist_ok=False)
        self.client = JSONClient(read(self.r2['api_config']), directory/'aux_calls.jsonl')
        self.audit_dir = directory
        self.extra_template_vars['r2mem_bank_sha'] = self.bank.meta['bank_sha']

    def query(self):
        # Delegate to upstream budget checks immediately if a decision budget is exhausted.
        if (0 < self.config.step_limit <= self.n_calls or
            0 < self.config.cost_limit <= self.cost or
            0 < self.config.wall_time_limit_seconds <= int(time.time()-self._start_time)):
            return super().query()
        # Total call cap includes upcoming abstraction calls and original decision call.
        modules = ['Planning'] if self.n_calls == 0 else ['Planning', 'Reflection']
        if self.n_calls + len(self.client.events) + len(modules) + 1 > self.r2['total_model_call_limit']:
            from minisweagent.exceptions import LimitsExceeded
            raise LimitsExceeded({'role': 'exit', 'content': 'R2Mem total model-call budget exhausted',
                                  'extra': {'exit_status': 'LimitsExceeded', 'submission': ''}})
        tool_sequence_check(self.messages)
        prefix = [visible(m) for m in self.messages]
        started = time.time()
        before = len(self.client.events)
        recalled = {}
        try:
            for module in modules:
                situation = self.client.call(SITUATION, {'module': module, 'history': prefix}, 'situation_'+module)
                text = situation.get('situation')
                if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                    raise ValueError('Invalid situation abstraction')
                # Original module separation; include observed state, not guessed next actions.
                task = self.extra_template_vars.get('task', '')
                suffix = task if module == 'Planning' else str(prefix[-1].get('content', ''))
                recalled[module] = self.bank.search(module, f'({text}) {suffix}', self.r2.get('return_k', 3))
        finally:
            # Include even failed JSON responses if the provider reported their usage.
            self.cost += sum(e.get('cost_usd', 0) for e in self.client.events[before:])
        lessons = {m: [r['experience'] for r in rows] for m, rows in recalled.items()}
        import json
        content = ('Retrieved search experiences (fallible guidance, not task evidence). '
                   'Use only when their conditions apply. Follow the original task and tool rules.\n' +
                   json.dumps(lessons, ensure_ascii=False))
        if len(content) > self.r2.get('max_injection_chars', 12000):
            raise ValueError('Experience injection exceeds fixed limit; no silent clipping')
        injection = self.model.format_message(role='user', content=content)
        original = self.messages
        self.messages = original + [injection]
        event = dict(decision=self.n_calls, retrieved=recalled, injection=injection,
                     bank_sha=self.bank.meta['bank_sha'], input_sha=digest(self.messages),
                     actual_input=self.messages.copy(), retrieval_seconds=time.time()-started)
        self.audit.append(event)
        try:
            # Original implementation retains decision accounting and budget checks.
            return super().query()
        finally:
            # Retain newly appended assistant messages, but never persist injection in history.
            original.extend(self.messages[len(original)+1:])
            self.messages = original
            from .core import save
            save(self.audit_dir/f'decision-{len(self.audit):04d}.json', event)

    def serialize(self, *extra_dicts):
        return super().serialize({'r2mem': {'config': self.r2, 'bank_sha': self.bank.meta['bank_sha'],
                                'aux_calls': len(self.client.events),
                                'aux_usage': [e.get('usage', {}) for e in self.client.events],
                                'audit_dir': str(self.audit_dir)}}, *extra_dicts)
