"""Isolated adapter to pinned official grading; never imported by the agent."""
import argparse
import hashlib
import signal
import json
import os
from pathlib import Path
import subprocess

COMMIT = "53448dd9e6c8868e6349b06dca29da80ae6b7187"
p = argparse.ArgumentParser()
p.add_argument('--checkout', required=True)
p.add_argument('--instance', required=True)
p.add_argument('--prediction', required=True)
p.add_argument('--output', required=True)
p.add_argument('--timeout', type=int, default=1800)
p.add_argument('--gold-check', action='store_true')
a = p.parse_args()
assert subprocess.check_output(['git', '-C', a.checkout, 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
assert not subprocess.check_output(['git', '-C', a.checkout, 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
import swebench
assert Path(swebench.__file__).resolve().is_relative_to(Path(a.checkout).resolve())
import docker
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec.test_spec import make_test_spec
row = json.loads(Path(a.instance).read_text())
pred = json.loads(Path(a.prediction).read_text())
assert pred['instance_id'] == row['instance_id']
if a.gold_check:
    pred = dict(pred, model_patch=row['patch'], model_name_or_path='gold-control')
else:
    pred = dict(pred, model_name_or_path=pred.get('model_name_or_path', 'agent'))
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
os.chdir(out)
run_id = 'r2mem-' + hashlib.sha256(str(out).encode()).hexdigest()[:12]
def interrupted(signum, frame):
    raise KeyboardInterrupt('Grading interrupted; official finally cleans up its container')
signal.signal(signal.SIGTERM, interrupted)
client = docker.from_env(timeout=180)
# Infrastructure resource limits only. Official test script, patch application,
# parsers and FAIL_TO_PASS/PASS_TO_PASS grading remain untouched.
class LimitedContainers:
    def __init__(self, collection):
        self.collection = collection

    def __getattr__(self, name):
        return getattr(self.collection, name)

    def create(self, *args, **kwargs):
        kwargs.update(network_mode='none', nano_cpus=2_000_000_000,
                      mem_limit='8g', memswap_limit='8g', pids_limit=512)
        return self.collection.create(*args, **kwargs)

class LimitedClient:
    def __init__(self, real):
        self.real = real
        self.containers = LimitedContainers(real.containers)

    def __getattr__(self, name):
        return getattr(self.real, name)

client = LimitedClient(client)
spec = make_test_spec(row, namespace='swebench')
result = run_instance(spec, pred, False, False, client, run_id, timeout=a.timeout)
if result is None:
    from swebench.harness.constants import APPLY_PATCH_FAIL
    logs = list(out.glob(f'logs/run_evaluation/{run_id}/*/*/run_instance.log'))
    # Rejected model patch is a model failure; other missing reports remain errors.
    if len(logs) == 1 and APPLY_PATCH_FAIL in logs[0].read_text():
        report = {'resolved': False, 'reason': 'official_patch_application_failed'}
    else:
        raise RuntimeError('Official harness produced no report; inspect logs, do not score as resolved/unresolved')
else:
    report = result[1][row['instance_id']]
(out/'result.json').write_text(json.dumps(report, indent=2))
print('OFFICIAL REPORT:', row['instance_id'], report, flush=True)
if a.gold_check and not report['resolved']:
    raise RuntimeError('Gold control failed: evaluation environment is not validated')
