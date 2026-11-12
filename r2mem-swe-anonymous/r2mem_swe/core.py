"""Pure data operations. No trajectory command is ever executed."""
import hashlib
import json
from pathlib import Path

DIMENSIONS = {
    'Planning': ['Info Needs Coverage', 'Info Needs Non-Redundancy',
                 'Tool–Info Alignment', 'Planning Efficiency'],
    'Reflection': ['Sufficiency Judgment Accuracy', 'Minimal Sufficiency Recognition',
                   'Follow-up Query Quality', 'Answer Completeness Awareness'],
}
SCOPES = {'repository_search', 'diagnostic_evidence', 'mixed', 'out_of_scope'}
INSPECTED = {'django__django-16569', 'psf__requests-1921', 'psf__requests-2317'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def manifest_check(manifest):
    groups = {}
    for name in ('construction', 'development', 'evaluation'):
        values = manifest[name]
        if not isinstance(values, list) or not all(isinstance(x, str) and x for x in values):
            raise ValueError(f'{name}: expected nonempty string IDs')
        if len(values) != len(set(values)):
            raise ValueError(f'{name}: duplicate task IDs')
        groups[name] = set(values)
    if not groups['construction']:
        raise ValueError('Construction task IDs must be explicitly selected')
    for a, b in [('construction', 'development'), ('construction', 'evaluation'), ('development', 'evaluation')]:
        if groups[a] & groups[b]:
            raise ValueError(f'Split overlap: {a}/{b}')
    if groups['evaluation'] & INSPECTED:
        raise ValueError('Previously inspected examples cannot enter held-out evaluation')
    return groups


def visible(message):
    """Keep exact visible content and actions, exclude outcome/usage/provider internals."""
    result = {k: message[k] for k in ('role', 'content', 'tool_calls', 'tool_call_id') if k in message}
    actions = message.get('extra', {}).get('actions')
    if actions is not None:
        result['actions'] = actions
    if result.get('content') is None:
        result['content'] = ''
    if actions is not None:
        result['actions_text'] = json.dumps(actions, ensure_ascii=False, sort_keys=True)
    if not isinstance(result.get('content', ''), str):
        raise ValueError('Only text trajectories supported; multimodal content needs an explicit adapter')
    return result


def extract(task_id, trajectory, manifest):
    groups = manifest_check(manifest)
    if task_id not in groups['construction']:
        raise ValueError('Only construction tasks may build experiences')
    messages = trajectory['messages']
    if len(messages) < 2 or messages[0]['role'] != 'system':
        raise ValueError('Expected mini trajectory with initial system/task messages')
    if messages[1]['role'] != 'user':
        raise ValueError('Missing initial task message')
    # No trajectory info, final score, submission, or gold patch enters records.
    public = [visible(m) for m in messages]
    turns = [i for i, m in enumerate(public) if m['role'] == 'assistant']
    records = []
    for ordinal, i in enumerate(turns):
        records.append(dict(task_id=task_id, decision_id=ordinal, module='Planning',
                            target_message=i, context=public[:i+1],
                            trajectory_sha=digest(trajectory), manifest_sha=digest(manifest)))
        end = turns[ordinal+1] if ordinal+1 < len(turns) else None
        observations = list(range(i+1, end if end is not None else len(public)))
        observations = [j for j in observations if public[j]['role'] in ('tool', 'user')]
        if end is not None and observations:
            records.append(dict(task_id=task_id, decision_id=ordinal, module='Reflection',
                                target_message=end, observation_messages=observations,
                                context=public[:end+1], trajectory_sha=digest(trajectory),
                                manifest_sha=digest(manifest)))
    for row in records:
        row['record_id'] = digest(row)
    return records


def citation_check(citation, context):
    if not isinstance(citation, dict):
        raise ValueError('Citation must be an object')
    i = citation.get('message')
    if type(i) is not int or not 0 <= i < len(context):
        raise ValueError('Invalid evidence message index')
    field = citation.get('field', 'content')
    if field not in ('content', 'actions_text'):
        raise ValueError('Unknown evidence field')
    text = context[i].get(field, '')
    quote = citation.get('quote')
    if not isinstance(quote, str) or not quote:
        raise ValueError('Evidence quote must be nonempty')
    if 'start' not in citation and 'end' not in citation:
        start = text.find(quote)
        if start < 0 or text.find(quote, start+1) != -1:
            raise ValueError('Quote absent or ambiguous; give a unique longer quote or explicit span')
        citation.update(start=start, end=start+len(quote))
    start, end = citation.get('start'), citation.get('end')
    if any(type(x) is not int for x in (start, end)) or not 0 <= start < end <= len(text):
        raise ValueError('Invalid character span')
    if text[start:end] != quote:
        raise ValueError('Evidence quote does not match the exact character span')


def evaluation_check(result, record):
    """Fail closed on model formatting/scoring failures. Never turn failures into 0."""
    if result.get('module') != record['module'] or result.get('scope') not in SCOPES:
        raise ValueError('Wrong module/scope')
    fields = result.get('extraction')
    if not isinstance(fields, dict):
        raise ValueError('Missing extraction')
    required = ('current_query', 'info_needs', 'tool_strategy') if record['module'] == 'Planning' else (
        'interpretation', 'enough', 'enough_scope', 'new_request', 'next_phase')
    for key in required:
        field = fields.get(key, {})
        if field.get('status') not in ('explicit', 'action_only', 'absent', 'ambiguous'):
            raise ValueError(f'Missing extraction status: {key}')
        cites = field.get('evidence', [])
        for cite in cites:
            citation_check(cite, record['context'])
        if field['status'] in ('explicit', 'action_only') and not any(c['message'] == record['target_message'] for c in cites):
            raise ValueError(f'Grounded extraction must cite target assistant reply: {key}')
        if record['module'] == 'Reflection' and field['status'] == 'explicit' and not any(
                c['message'] == record['target_message'] and c.get('field', 'content') == 'content' for c in cites):
            raise ValueError('Explicit reflection must cite assistant text')
        if field['status'] in ('absent', 'ambiguous') and field.get('value') is not None:
            raise ValueError('Uncertain fields must be null')
    if record['module'] == 'Reflection':
        enough = fields['enough'].get('value')
        if enough is not None and type(enough) is not bool:
            raise ValueError('enough must be boolean/null')
        if fields['enough_scope'].get('value') not in (None, 'local_information_need', 'task_completion'):
            raise ValueError('Invalid enough_scope')
        if enough is not None and fields['enough_scope'].get('value') is None:
            raise ValueError('Sufficiency judgment needs scope')
        if fields['next_phase'].get('value') not in (None, 'continue_search', 'edit', 'validate', 'submit', 'unknown'):
            raise ValueError('Invalid next_phase')
    rubrics = result.get('rubrics', {})
    if set(rubrics) != set(DIMENSIONS[record['module']]):
        raise ValueError('Expected exactly the original four dimensions')
    scores = []
    for dimension in DIMENSIONS[record['module']]:
        item = rubrics[dimension]
        score = item.get('score')
        if score is not None and (type(score) is not int or score not in range(4)):
            raise ValueError('Score must be integer 0–3 or null')
        if not item.get('reason') or not item.get('advice'):
            raise ValueError('Every dimension needs reason and advice')
        evidence = item.get('evidence', [])
        if score is not None and not evidence:
            raise ValueError('Scored dimensions need evidence')
        for cite in evidence:
            citation_check(cite, record['context'])
        scores.append(score)
    # Reflection cannot be inferred solely from an action or tool output.
    no_reflection = record['module'] == 'Reflection' and not any(
        fields[k]['status'] == 'explicit' for k in ('interpretation', 'enough', 'new_request'))
    total = None if no_reflection or None in scores or result['scope'] == 'out_of_scope' else sum(scores)
    result['total_score'] = total
    result['quality'] = ('excluded' if total is None else 'high' if total >= 10 else 'low' if total <= 5 else 'middle')
    return result


def tool_sequence_check(messages):
    """Never inject between a native tool call and its result."""
    pending = set()
    for message in messages:
        if message['role'] == 'assistant':
            if pending:
                raise ValueError('Unresolved previous tool calls')
            pending = {c['id'] for c in (message.get('tool_calls') or [])}
        elif message['role'] == 'tool':
            identifier = message.get('tool_call_id')
            if identifier not in pending:
                raise ValueError('Unpaired tool result')
            pending.remove(identifier)
    if pending:
        raise ValueError('Cannot inject before all tool results are available')
