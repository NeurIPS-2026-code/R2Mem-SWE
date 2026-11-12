"""Sequential Verified baseline + official scoring. Resume never resamples a saved prediction."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2mem_swe.core import read, save, digest, INSPECTED
from r2mem_swe import MINI_COMMIT
MINI_CHECKOUT = Path(os.environ.get('MINI_CHECKOUT', 'external/mini-swe-agent')).resolve()
HARNESS_CHECKOUT = Path(os.environ.get('HARNESS_CHECKOUT', 'external/SWE-bench')).resolve()
HARNESS_PYTHON = Path(os.environ.get('HARNESS_PYTHON', '.venv-harness/bin/python')).resolve()
DOCKER_STORAGE_PATH = Path(os.environ.get('DOCKER_STORAGE_PATH', '/var/lib/docker'))
DATASET = 'princeton-nlp/SWE-bench_Verified'
DATASET_REVISION = 'c104f840cc67f8b6eec6f759ebc8b2693d585d4a'
HARNESS = '53448dd9e6c8868e6349b06dca29da80ae6b7187'
DOCKER = [os.environ.get('DOCKER_EXECUTABLE', 'docker')]
CONFIGS = ['configs/swe_qwen25.example.yaml', 'configs/swe_verified_baseline.yaml']


def execute(cmd, log, timeout, cwd=ROOT):
    with Path(log).open('a') as stream:
        process = subprocess.Popen(cmd, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f'Command exit {code}; inspect {log}')


def protocol_hash():
    paths = [*CONFIGS, 'r2mem_swe/run_one.py', 'r2mem_swe/progress.py',
             'scripts/swe_verified_batch.py', 'scripts/swe_official_grade.py']
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    return digest({'mini': MINI_COMMIT, 'harness': HARNESS, 'files': hashes})


def init_data(out):
    meta = out/'inventory.json'
    if meta.exists():
        inventory = read(meta)
        if inventory['protocol_hash'] != protocol_hash():
            raise RuntimeError('Protocol/code changed; use a new output directory')
        rows = read(out/'official-data.json')
        if digest(rows) != inventory['data_hash']:
            raise RuntimeError('Dataset snapshot changed')
        return rows, inventory
    from datasets import load_dataset
    revision = DATASET_REVISION
    print('FETCH VERIFIED:', revision, flush=True)
    rows = list(load_dataset(DATASET, split='test', revision=revision))
    rows.sort(key=lambda x: x['instance_id'])
    if len(rows) != 500 or len({x['instance_id'] for x in rows}) != 500:
        raise RuntimeError('Expected exactly 500 unique Verified test instances')
    # First task per repository, lexicographic ordering. Fixed before any result.
    pilot = []
    seen = set()
    for row in rows:
        if row['repo'] not in seen and row['instance_id'] not in INSPECTED:
            seen.add(row['repo'])
            pilot.append(row['instance_id'])
    pilot = pilot[:3]
    inventory = dict(dataset=DATASET, revision=revision, data_hash=digest(rows),
                     protocol_hash=protocol_hash(), ids=[r['instance_id'] for r in rows],
                     pilot=pilot, previously_inspected=sorted(INSPECTED),
                     release='anonymous-adaptation-v1')
    save(out/'official-data.json', rows)  # Never mounted or passed to model.
    save(out/'manifest.json', {'benchmark': inventory['ids']})
    save(meta, inventory)
    return rows, inventory


def summarize(out, inventory, write=True):
    results = []
    errors = []
    for iid in inventory['ids']:
        directory = out/'tasks'/iid
        if (directory/'done.json').exists():
            results.append(read(directory/'done.json'))
        elif (directory/'error.json').exists():
            errors.append(iid)
    heldout = [r for r in results if r['instance_id'] not in INSPECTED and r['instance_id'] not in inventory['pilot']]
    summary = dict(total=500, completed=len(results), resolved=sum(r['resolved'] for r in results),
                   empty_patches=sum(r['empty_patch'] for r in results),
                   context_budget_exhausted=sum(r.get('exit_status') == 'ContextWindowExceededError' for r in results),
                   infrastructure_or_unclassified_errors=errors,
                   pending=500-len(results), final_resolved_rate=(sum(r['resolved'] for r in results)/500 if len(results)==500 else None),
                   development_ids=sorted(set(inventory['pilot']) | INSPECTED),
                   heldout_completed=len(heldout), heldout_resolved=sum(r['resolved'] for r in heldout),
                   note='Incomplete runs have no final score. Pilot/inspected cases excluded from future held-out comparison.')
    if not write:
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    save(out/'summary.json', summary)
    with (out/'predictions.jsonl').open('w') as f:
        for iid in inventory['ids']:
            p = out/'tasks'/iid/'agent/prediction.json'
            if p.exists():
                f.write(json.dumps(read(p))+'\n')
    print('PROGRESS:', json.dumps(summary), flush=True)
    return summary


def image_for(iid):
    return f"swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_').lower()}:latest"


def image_info(tag):
    p = subprocess.run(DOCKER+['image', 'inspect', tag], text=True, capture_output=True)
    return json.loads(p.stdout)[0] if p.returncode == 0 else None


def recover_context_exhaustion(slot, iid):
    """Preserve original mini terminal event; never infer success or export an unsent patch."""
    trajpath = slot/f'agent/{iid}.traj.json'
    if not trajpath.exists():
        return False
    trajectory = read(trajpath)
    info = trajectory.get('info', {})
    messages = trajectory.get('messages', [])
    if not messages or info.get('exit_status') != 'ContextWindowExceededError':
        return False
    last = messages[-1]
    extra = last.get('extra', {})
    if last.get('role') != 'exit' or extra.get('exit_status') != 'ContextWindowExceededError':
        return False
    if info.get('submission') or extra.get('submission'):
        raise RuntimeError('Unexpected nonempty context-limit submission; inspect manually')
    run = read(slot/'agent/run.json')
    if run['task_id'] != iid or run['mode'] != 'baseline':
        raise RuntimeError('Context recovery requires matching baseline task')
    save(slot/'agent/prediction.json', {'instance_id': iid,
         'model_name_or_path': run['config']['model']['model_name'], 'model_patch': ''})
    save(slot/'context-budget.json', {'exit_status': 'ContextWindowExceededError',
         'trajectory_sha': digest(trajectory), 'time': time.time(),
         'reason': 'Context budget exhausted without submission; original trajectory retained; no resampling'})
    print('CONTEXT BUDGET EXHAUSTED:', iid, '(empty submission; continue scoring pipeline)', flush=True)
    return True


def process(row, out, gold_check=False):
    iid = row['instance_id']
    slot = out/'tasks'/iid
    slot.mkdir(parents=True, exist_ok=True)
    tag = image_for(iid)
    old = image_info(tag)
    if old is None:
        if shutil.disk_usage(DOCKER_STORAGE_PATH).free < 150*1024**3:
            raise RuntimeError('Less than 150 GiB free on configured Docker storage filesystem; stopped before pulling')
        print('PULL:', iid, flush=True)
        execute(DOCKER+['pull', tag], slot/'pull.log', 3600)
    info = image_info(tag)
    if info is None:
        raise RuntimeError('Image unavailable after pull')
    pin = slot/'image.json'
    if pin.exists() and read(pin)['id'] != info['Id']:
        raise RuntimeError('Image changed since previous attempt; refusing mixed environment')
    if not pin.exists():
        save(pin, {'id': info['Id'], 'repo_digests': info.get('RepoDigests'), 'tag': tag,
                   'owned': old is None})
    save(slot/'instance.json', {'instance_id': iid, 'problem_statement': row['problem_statement'],
                               'image_name': info['Id']})
    save(slot/'official-instance.json', row)
    predpath = slot/'agent/prediction.json'
    start = time.time()
    if not predpath.exists() and not recover_context_exhaustion(slot, iid):
        if (slot/'agent').exists():
            (slot/'agent').rename(slot/f'agent-interrupted-{time.time_ns()}')
        try:
            execute([sys.executable, '-u', '-m', 'r2mem_swe', 'run',
                     '--mini-checkout', str(MINI_CHECKOUT), '--manifest', str(out/'manifest.json'),
                     '--instance', str(slot/'instance.json'), '--config', 'swebench_backticks.yaml', *CONFIGS,
                     '--mode', 'baseline', '--split', 'benchmark', '--total-model-calls', '60',
                     '--output', str(slot/'agent')], slot/'agent.log', 3900)
        except RuntimeError:
            if not recover_context_exhaustion(slot, iid):
                raise
        save(slot/'generation-time.json', {'seconds': time.time()-start})
    pred = read(predpath)
    if pred['instance_id'] != iid:
        raise RuntimeError('Prediction ID mismatch')
    prediction_sha = digest(pred)
    if (slot/'prediction-hash.json').exists() and read(slot/'prediction-hash.json')['sha'] != prediction_sha:
        raise RuntimeError('Saved prediction changed; refusing to reuse old grading')
    save(slot/'prediction-hash.json', {'sha': prediction_sha})
    grade = [str(HARNESS_PYTHON), '-u', str(ROOT/'scripts/swe_official_grade.py'),
             '--checkout', str(HARNESS_CHECKOUT), '--instance', str(slot/'official-instance.json'),
             '--prediction', str(predpath)]
    if gold_check and not (out/'gold-control-passed.json').exists():
        execute(grade+['--output', str(slot/'gold-control'), '--gold-check'], slot/'gold-control.log', 2100)
        save(out/'gold-control-passed.json', {'instance_id': iid, 'report': read(slot/'gold-control/result.json')})
    if not pred['model_patch'].strip():
        report = {'resolved': False, 'reason': 'empty_patch; official harness excludes empty predictions'}
    else:
        execute(grade+['--output', str(slot/'grading')], slot/'grading.log', 2100)
        report = read(slot/'grading/result.json')
    trajectory = read(slot/f'agent/{iid}.traj.json')
    usages = []
    for message in trajectory.get('messages', []):
        response = message.get('extra', {}).get('response', {})
        if isinstance(response, dict) and isinstance(response.get('usage'), dict):
            usages.append(response['usage'])
    token_usage = {key: sum(u[key] for u in usages) if usages and all(isinstance(u.get(key), int) for u in usages) else None
                   for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    save(slot/'done.json', {'instance_id': iid, 'resolved': report['resolved'],
                          'empty_patch': not pred['model_patch'].strip(), 'report': report, 'prediction_sha': prediction_sha,
                          'exit_status': trajectory['info'].get('exit_status'),
                          'model_stats': trajectory['info'].get('model_stats'),
                          'token_usage': token_usage, 'calls_with_usage': len(usages),
                          'generation_seconds': read(slot/'generation-time.json')['seconds'] if (slot/'generation-time.json').exists() else None})
    if (slot/'error.json').exists():
        (slot/'error.json').unlink()
    cleanup_image(slot)


def cleanup_image(slot):
    # Only the exact reference pulled by this run; no prune/force or old images.
    pin = read(slot/'image.json')
    info = image_info(pin['tag'])
    if pin['owned'] and info and info['Id'] == pin['id']:
        p = subprocess.run(DOCKER+['image', 'rm', pin['tag']], capture_output=True, text=True)
        (slot/'image-cleanup.log').write_text(p.stdout+p.stderr)
        if p.returncode:
            print('Image retained (in use or shared):', pin['tag'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['pilot', 'full', 'status'])
    p.add_argument('--output', default='outputs/swe-r2mem/verified-qwen25-7b-v1')
    p.add_argument('--retry-errors', action='store_true', help='Retry only infrastructure/incomplete tasks, never completed model failures')
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == 'status':
        summarize(out, read(out/'inventory.json'), write=False)
        return
    with (out/'batch.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another runner is active for this output directory')
        rows, inventory = init_data(out)
        if a.stage == 'full' and not (out/'pilot-passed.json').exists():
            raise RuntimeError('Run pilot successfully before full')
        selected = set(inventory['pilot'] if a.stage == 'pilot' else inventory['ids'])
        failures = []
        for row in rows:
            iid = row['instance_id']
            if iid not in selected:
                continue
            slot = out/'tasks'/iid
            if (slot/'done.json').exists():
                cleanup_image(slot)
                continue
            if (slot/'error.json').exists() and not a.retry_errors:
                failures.append(iid)
                continue
            print('START:', iid, time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
            try:
                process(row, out, gold_check=(a.stage == 'pilot' and iid == inventory['pilot'][0]))
            except Exception as e:
                save(slot/'error.json', {'error': repr(e), 'time': time.time()})
                print('ERROR:', iid, repr(e), flush=True)
                failures.append(iid)
                # Pilot fail-fast; full stops after three infrastructure errors in this invocation.
                if a.stage == 'pilot' or len(failures) >= 3:
                    break
            summarize(out, inventory)
        summary = summarize(out, inventory)
        if a.stage == 'pilot' and not failures and (out/'gold-control-passed.json').exists():
            save(out/'pilot-passed.json', {'ids': inventory['pilot'], 'protocol_hash': inventory['protocol_hash']})
            print('SWE PILOT PIPELINE PASSED (not a model-success claim)', flush=True)
        elif a.stage == 'full' and summary['completed'] == 500:
            print('SWE VERIFIED 500 COMPLETE', flush=True)
        else:
            raise SystemExit('Run incomplete; inspect errors and resume with --retry-errors after fixing infrastructure')

if __name__ == '__main__':
    main()
