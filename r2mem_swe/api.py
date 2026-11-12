"""Explicit OpenAI-compatible endpoint; bounded requests, usage ledger, no implicit retries."""
import json
import os
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse
from .core import digest


class JSONClient:
    def __init__(self, config, ledger):
        self.config = config
        self.ledger = Path(ledger)
        self.events = []
        self.prior_calls = sum(1 for _ in self.ledger.open()) if self.ledger.exists() else 0
        url = config['base_url'].rstrip('/')
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https') or parsed.username or parsed.password or parsed.query:
            raise ValueError('Endpoint must be a plain http(s) base URL without credentials/query')
        self.url = url + '/chat/completions'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for name in ('input_usd_per_million', 'output_usd_per_million'):
            if config.get(name, -1) < 0:
                raise ValueError('Set explicit prices (0 for local, GPU time accounted separately)')

    def call(self, system, payload, purpose):
        if self.prior_calls + len(self.events) >= self.config.get('max_calls', 10000):
            raise RuntimeError('Auxiliary API call limit reached')
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
        request = dict(model=self.config['model'], messages=messages, temperature=0,
                       max_tokens=self.config.get('max_tokens', 4096))
        options = self.config.get('request_options', {})
        if not isinstance(options, dict) or set(options) - {'thinking', 'response_format'}:
            raise ValueError('Only thinking and response_format request options are allowed')
        request.update(options)
        if len(json.dumps(request)) > self.config.get('max_request_chars', 200000):
            raise ValueError('Context exceeds configured character guard; no silent truncation')
        key = os.getenv(self.config.get('api_key_env', 'R2MEM_API_KEY'), '')
        if not key and not self.config.get('allow_no_key', False):
            raise ValueError('API key environment variable is not configured')
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        event = dict(purpose=purpose, request_sha=digest(request), model=self.config['model'],
                     base_url=self.config['base_url'], started=time.time(), messages=messages)
        self.events.append(event)
        try:
            req = urllib.request.Request(self.url, data=json.dumps(request).encode(), headers=headers)
            with self.opener.open(req, timeout=self.config.get('timeout_seconds', 180)) as response:
                result = json.load(response)
            event['cost_note'] = self.config.get('cost_note', 'configured token prices')
            event['usage'] = result.get('usage', {})
            event['returned_model'] = result.get('model')
            usage = event['usage']
            if not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
                raise ValueError('Missing token usage; cannot silently undercount cost')
            event['cost_usd'] = (usage['prompt_tokens'] * self.config['input_usd_per_million'] +
                                 usage['completion_tokens'] * self.config['output_usd_per_million']) / 1e6
            choice = result['choices'][0]
            event['finish_reason'] = choice.get('finish_reason')
            event['response'] = choice['message'].get('content')
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Incomplete JSON generation')
            value = json.loads(event['response'])
            if not isinstance(value, dict):
                raise ValueError('JSON response must be an object')
            return value
        except Exception as exc:
            # Do not persist HTTP error bodies that could echo headers or secrets.
            event['error_type'] = type(exc).__name__
            raise
        finally:
            event['seconds'] = time.time() - event['started']
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger.open('a') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
