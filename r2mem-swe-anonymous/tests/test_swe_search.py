"""Offline unit tests; no models, containers, GPUs or remote APIs."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from r2mem_swe.core import (DIMENSIONS, extract, evaluation_check, manifest_check,
                            tool_sequence_check, digest, save)
from r2mem_swe.pipeline import pin_directory

MANIFEST = {'construction': ['toy__repo-1'], 'development': ['toy__repo-2'], 'evaluation': ['toy__repo-3']}
TRACE = {'info': {'resolved': True, 'gold_patch': 'SECRET'}, 'messages': [
    {'role': 'system', 'content': 'Use shell'}, {'role': 'user', 'content': 'Locate the bug'},
    {'role': 'assistant', 'content': 'Find the entry point.\n```bash\ngrep entry src.py\n```'},
    {'role': 'user', 'content': 'No match'},
    {'role': 'assistant', 'content': 'That guessed symbol is absent; inspect the known entry file.\n```bash\ncat main.py\n```'},
    {'role': 'user', 'content': 'FUTURE OUTPUT'}, {'role': 'exit', 'content': 'Budget'}]}


def evaluation(record):
    i = record['target_message']
    text = record['context'][i]['content']
    citation = dict(message=i, start=0, end=4, quote=text[:4])
    keys = ('current_query', 'info_needs', 'tool_strategy') if record['module'] == 'Planning' else (
        'interpretation', 'enough', 'enough_scope', 'new_request', 'next_phase')
    extraction = {k: dict(value='example', status='explicit', evidence=[citation]) for k in keys}
    if record['module'] == 'Reflection':
        extraction['enough']['value'] = False
        extraction['enough_scope']['value'] = 'local_information_need'
        extraction['next_phase']['value'] = 'continue_search'
    return dict(module=record['module'], scope='repository_search', extraction=extraction,
                rubrics={d: dict(score=3, reason='grounded', advice='specific', evidence=[citation])
                         for d in DIMENSIONS[record['module']]})


class CoreTests(unittest.TestCase):
    def test_prefix_boundary(self):
        records = extract('toy__repo-1', TRACE, MANIFEST)
        self.assertEqual([r['module'] for r in records], ['Planning', 'Reflection', 'Planning'])
        self.assertNotIn('No match', json.dumps(records[0]))
        self.assertNotIn('FUTURE OUTPUT', json.dumps(records[1]))
        self.assertNotIn('SECRET', json.dumps(records))
        self.assertEqual(records[1]['observation_messages'], [3])

    def test_native_multi_action_is_one_decision(self):
        trace = copy.deepcopy(TRACE)
        trace['messages'][2] = dict(role='assistant', content=None, extra={'actions': [
            {'command': 'cat a'}, {'command': 'cat b'}]})
        records = extract('toy__repo-1', trace, MANIFEST)
        self.assertEqual(len(records), 3)
        self.assertIn('cat b', records[0]['context'][2]['actions_text'])

    def test_split_leakage(self):
        bad = copy.deepcopy(MANIFEST)
        bad['evaluation'] += bad['construction']
        with self.assertRaises(ValueError):
            manifest_check(bad)
        bad['evaluation'] = ['psf__requests-1921']
        with self.assertRaises(ValueError):
            manifest_check(bad)
        with self.assertRaises(ValueError):
            extract('toy__repo-3', TRACE, MANIFEST)

    def test_thresholds_and_null(self):
        record = extract('toy__repo-1', TRACE, MANIFEST)[0]
        for scores, expected in [([3,3,2,2], 'high'), ([1,1,1,2], 'low'), ([2,2,2,3], 'middle'), ([3,3,None,3], 'excluded')]:
            value = evaluation(record)
            for dimension, score in zip(DIMENSIONS['Planning'], scores):
                value['rubrics'][dimension]['score'] = score
            self.assertEqual(evaluation_check(value, record)['quality'], expected)

    def test_invalid_citation_and_score(self):
        record = extract('toy__repo-1', TRACE, MANIFEST)[0]
        value = evaluation(record)
        value['extraction']['current_query']['evidence'][0]['quote'] = 'wrong'
        with self.assertRaises(ValueError):
            evaluation_check(value, record)
        value = evaluation(record)
        value['rubrics'][DIMENSIONS['Planning'][0]]['score'] = True
        with self.assertRaises(ValueError):
            evaluation_check(value, record)

    def test_missing_reflection_is_not_failure(self):
        record = extract('toy__repo-1', TRACE, MANIFEST)[1]
        value = evaluation(record)
        for field in value['extraction'].values():
            field.update(status='absent', value=None, evidence=[])
        self.assertEqual(evaluation_check(value, record)['quality'], 'excluded')

    def test_unique_quote_resolves_character_span(self):
        from r2mem_swe.core import citation_check
        citation = {'message':0, 'quote':'unique phrase'}
        citation_check(citation, [{'content':'prefix unique phrase suffix'}])
        self.assertEqual(citation['start'], 7)
        self.assertEqual(citation['end'], 20)
        with self.assertRaises(ValueError):
            citation_check({'message':0,'quote':'repeat'}, [{'content':'repeat repeat'}])

    def test_tool_pairing(self):
        messages = [dict(role='assistant', tool_calls=[{'id':'a'}, {'id':'b'}]),
                    dict(role='tool', tool_call_id='a'), dict(role='tool', tool_call_id='b')]
        tool_sequence_check(messages)
        with self.assertRaises(ValueError):
            tool_sequence_check(messages[:-1])
        with self.assertRaises(ValueError):
            tool_sequence_check([dict(role='tool', tool_call_id='unknown')])

    def test_resume_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            pin_directory(directory, {'version':1})
            pin_directory(directory, {'version':1})
            with self.assertRaises(ValueError):
                pin_directory(directory, {'version':2})


# Tests below run against the real pinned DefaultAgent when its light dependencies exist.
try:
    from r2mem_swe.agent import SearchExperienceAgent
    from minisweagent.agents.default import DefaultAgent
    HAS_MINI = True
except ImportError:
    HAS_MINI = False


class ReplayModel:
    def __init__(self, failure=False):
        self.inputs = []
        self.failure = failure
    def query(self, messages):
        self.inputs.append(copy.deepcopy(messages))
        if self.failure:
            raise ValueError('deliberate provider failure')
        return dict(role='assistant', content='read next file', extra={'cost':0.4, 'actions':[]})
    def format_message(self, **kwargs):
        return kwargs


class FixedAuxiliary:
    def __init__(self):
        self.events = []
    def call(self, *args):
        self.events.append({'cost_usd': 0.1})
        return {'situation':'unknown call path'}


class FixedBank:
    meta = {'bank_sha':'fixed-test-bank'}
    def search(self, module, query, k):
        return [{'experience':'IF path unknown THEN inspect callers', 'id': module}]


@unittest.skipUnless(HAS_MINI, 'Set PYTHONPATH to pinned mini checkout and install light dependencies')
class AdapterTests(unittest.TestCase):
    def create(self, directory, failure=False):
        agent = SearchExperienceAgent.__new__(SearchExperienceAgent)
        DefaultAgent.__init__(agent, ReplayModel(failure), object(), system_template='system',
                              instance_template='{{task}}', step_limit=3, cost_limit=3)
        agent.messages = copy.deepcopy(TRACE['messages'][:2])
        agent.r2 = {'total_model_call_limit':20}
        agent.client, agent.bank, agent.audit = FixedAuxiliary(), FixedBank(), []
        agent.audit_dir = Path(directory)
        return agent

    def test_original_query_accounting_and_no_history_pollution(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.create(directory)
            original = agent.messages
            agent.query()
            self.assertIs(agent.messages, original)
            self.assertEqual(len(agent.messages), 3)
            self.assertEqual(agent.n_calls, 1)
            self.assertAlmostEqual(agent.cost, .5)
            self.assertIn('Retrieved search experiences', agent.model.inputs[0][-1]['content'])
            self.assertNotIn('Retrieved search experiences', json.dumps(agent.messages))
            self.assertTrue((Path(directory)/'decision-0001.json').exists())

    def test_exception_restores_history(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.create(directory, failure=True)
            original = copy.deepcopy(agent.messages)
            with self.assertRaises(ValueError):
                agent.query()
            self.assertEqual(agent.messages, original)
            self.assertAlmostEqual(agent.cost, .1)

    def test_budget_before_auxiliary_call(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.create(directory)
            agent.n_calls = 3
            with self.assertRaises(Exception):
                agent.query()
            self.assertEqual(len(agent.client.events), 0)
            agent.n_calls = 0
            agent.r2['total_model_call_limit'] = 1
            with self.assertRaises(Exception):
                agent.query()
            self.assertEqual(len(agent.client.events), 0)

class PipelineTests(unittest.TestCase):
    def test_resume_and_bank_provenance(self):
        from r2mem_swe.pipeline import evaluate, learn
        config = {'base_url':'http://127.0.0.1:1/v1', 'model':'test-only',
                  'input_usd_per_million':0, 'output_usd_per_million':0, 'allow_no_key':True}
        record = extract('toy__repo-1', TRACE, MANIFEST)[0]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = {'manifest':MANIFEST, 'records':[record]}
            save(root/'records.json', data)
            save(root/'api.json', config)
            with patch('r2mem_swe.pipeline.JSONClient.call', return_value=evaluation(record)) as call:
                evaluate(root/'records.json', root/'api.json', root/'eval')
                evaluate(root/'records.json', root/'api.json', root/'eval')
                self.assertEqual(call.call_count, 1)
            lesson = {'skip':False, 'summary':'specific search diagnosis', 'situation':'missing call path',
                      'experience':'IF call path unknown THEN inspect the callers'}
            with patch('r2mem_swe.pipeline.JSONClient.call', return_value=lesson) as call:
                learn(root/'records.json', root/'eval', root/'api.json', root/'learn')
                learn(root/'records.json', root/'eval', root/'api.json', root/'learn')
                self.assertEqual(call.call_count, 1)
            bank = json.loads((root/'learn/bank.json').read_text())
            self.assertEqual(bank['manifest'], MANIFEST)
            self.assertEqual(bank['experiences'][0]['source_task'], 'toy__repo-1')
            self.assertEqual(bank['experiences'][0]['quality'], 'high')

    def test_invalid_evaluator_does_not_complete_or_score_zero(self):
        from r2mem_swe.pipeline import evaluate
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            save(root/'records.json', {'manifest': MANIFEST, 'records': extract('toy__repo-1', TRACE, MANIFEST)[:1]})
            save(root/'api.json', {'base_url':'http://127.0.0.1:1/v1', 'model':'test-only',
                                  'input_usd_per_million':0, 'output_usd_per_million':0})
            with patch('r2mem_swe.pipeline.JSONClient.call', return_value={'bad':'format'}):
                with self.assertRaises(ValueError):
                    evaluate(root/'records.json', root/'api.json', root/'eval')
            self.assertFalse((root/'eval/complete.json').exists())
            self.assertEqual(len(list((root/'eval').glob('*.raw.json'))), 1)


if __name__ == '__main__':
    unittest.main()
