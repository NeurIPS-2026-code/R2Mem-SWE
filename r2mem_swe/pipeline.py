"""Resumable offline stages; each validated record is saved before proceeding."""
from pathlib import Path
from .core import digest, read, save, extract, evaluation_check, manifest_check
from .prompts import EVALUATOR, LEARNER
from .api import JSONClient


def prepare(manifest_path, trajectories, output):
    manifest = read(manifest_path)
    manifest_check(manifest)
    if Path(output).exists():
        raise FileExistsError(output)
    records = []
    seen = set()
    for path in trajectories:
        task = Path(path).name.removesuffix('.traj.json')
        if task in seen:
            raise ValueError('Duplicate trajectory task')
        seen.add(task)
        records += extract(task, read(path), manifest)
    if not records:
        raise ValueError('No decision records found')
    save(output, {'schema_version': 1, 'manifest': manifest, 'records': records})


def pin_directory(directory, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'provenance.json'
    if path.exists() and read(path) != provenance:
        raise ValueError('Resume inputs/config/prompt changed; use a new output directory')
    if not path.exists():
        save(path, provenance)
    return directory


def evaluate(records_path, config_path, output):
    data, cfg = read(records_path), read(config_path)
    manifest_check(data['manifest'])
    directory = pin_directory(output, {'records_sha': digest(data), 'config_sha': digest(cfg),
                                       'prompt_sha': digest(EVALUATOR)})
    client = JSONClient(cfg, directory/'calls.jsonl')
    for index, record in enumerate(data['records'], 1):
        path = directory/(record['record_id']+'.json')
        if path.exists():
            cached = read(path)
            if cached['record'] != record:
                raise ValueError('Cached record mismatch')
            evaluation_check(cached['evaluation'], record)
            continue
        print(f'EVALUATE {index}/{len(data["records"])} {record["task_id"]} {record["module"]}', flush=True)
        value = client.call(EVALUATOR, record, 'evaluate')
        save(directory/(record['record_id']+'.raw.json'), value)
        value = evaluation_check(value, record)
        save(path, {'record': record, 'evaluation': value})
    save(directory/'complete.json', {'records_sha': digest(data), 'count': len(data['records'])})


def learn(records_path, evaluations, config_path, output):
    data, cfg = read(records_path), read(config_path)
    manifest_check(data['manifest'])
    evaluations = Path(evaluations)
    complete = read(evaluations/'complete.json')
    if complete['records_sha'] != digest(data):
        raise ValueError('Evaluations do not match records')
    entries = [read(evaluations/(r['record_id']+'.json')) for r in data['records']]
    provenance = {'records_sha': digest(data), 'evaluations_sha': digest(entries),
                  'config_sha': digest(cfg), 'prompt_sha': digest(LEARNER)}
    directory = pin_directory(output, provenance)
    client = JSONClient(cfg, directory/'calls.jsonl')
    experiences = []
    for entry in entries:
        record, value = entry['record'], entry['evaluation']
        evaluation_check(value, record)
        if value['quality'] not in ('high', 'low'):
            continue
        path = directory/(record['record_id']+'.json')
        if path.exists():
            lesson = read(path)
        else:
            print('LEARN', record['record_id'][:12], value['quality'], flush=True)
            lesson = client.call(LEARNER, entry, 'learn')
        if type(lesson.get('skip')) is not bool:
            raise ValueError('Learner must explicitly choose skip')
        if lesson['skip']:
            if not lesson.get('reason'):
                raise ValueError('Skipped lesson needs reason')
        else:
            for key in ('summary', 'situation', 'experience'):
                if not isinstance(lesson.get(key), str) or not 0 < len(lesson[key]) <= 1600:
                    raise ValueError('Invalid/oversized lesson field: ' + key)
            if not lesson['experience'].startswith('IF ') or ' THEN ' not in lesson['experience']:
                raise ValueError('Experience must be IF ... THEN ...')
            if record['task_id'] in str(lesson) or '```' in str(lesson):
                raise ValueError('Task-specific or code-block content in abstract lesson')
            experiences.append(dict(id=record['record_id'], module=record['module'],
                                    quality=value['quality'], source_task=record['task_id'],
                                    **{k: lesson[k] for k in ('summary', 'situation', 'experience')}))
        save(path, lesson)
    # Abstract text still requires a leakage/quality audit before formal evaluation.
    save(directory/'bank.json', {'schema_version': 1, 'provenance': provenance,
                               'manifest': data['manifest'], 'experiences': experiences})
