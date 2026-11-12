"""Explicit single-instance runner; only this module starts a task environment."""
import importlib
import subprocess
import sys
from pathlib import Path
from . import MINI_COMMIT
from .core import read, save, manifest_check, digest


def verify_checkout(path):
    path = Path(path).resolve()
    revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != MINI_COMMIT or dirty:
        raise ValueError('mini checkout must match the pinned commit with clean tracked files')
    sys.path.insert(0, str(path/'src'))
    return path


def run(args):
    checkout = verify_checkout(args.mini_checkout)
    import yaml
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import get_config_from_spec
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment
    module = importlib.import_module('minisweagent.agents.default')
    if not Path(module.__file__).resolve().is_relative_to(checkout):
        raise ValueError('Imported mini does not match pinned checkout')
    manifest = read(args.manifest)
    if args.split == 'benchmark':
        if args.mode != 'baseline' or set(manifest) != {'benchmark'}:
            raise ValueError('Benchmark inventory is baseline-only, never an experience manifest')
        ids = manifest['benchmark']
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Invalid benchmark inventory')
        groups = {'benchmark': set(ids)}
    else:
        groups = manifest_check(manifest)
    instance = read(args.instance)
    task_id = instance['instance_id']
    if task_id not in groups[args.split]:
        raise ValueError('Instance ID not in requested manifest split')
    if args.mode == 'experience' and args.split == 'construction':
        raise ValueError('Experience runs must not use construction tasks')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new task output directory')
    from minisweagent.utils.serialize import recursive_merge
    config = recursive_merge(*[get_config_from_spec(spec) for spec in args.config])
    if config.get('model', {}).get('model_kwargs', {}).get('api_key') not in (None, '', 'local'):
        raise ValueError('Store real API keys in environment variables, never config files')
    agent_cfg = dict(config['agent'])
    agent_cfg.pop('agent_class', None)
    agent_cfg['output_path'] = output/(task_id+'.traj.json')
    if args.total_model_calls < 1 or not agent_cfg.get('step_limit', 0):
        raise ValueError('Specify a positive total call cap and agent.step_limit')
    if args.mode == 'baseline':
        agent_cfg['step_limit'] = min(agent_cfg['step_limit'], args.total_model_calls)
    else:
        if not args.r2mem_config:
            raise ValueError('--r2mem-config is required')
        r2 = read(args.r2mem_config)
        if r2['task_id'] != task_id or r2.get('split', 'development') != args.split:
            raise ValueError('R2Mem task/split mismatch')
        if r2['total_model_call_limit'] != args.total_model_calls:
            raise ValueError('R2Mem total call budget mismatch')
        bank = read(Path(r2['index_dir'])/'bank.json')
        if bank['manifest'] != manifest:
            raise ValueError('Bank and runner split manifests differ')
    output.mkdir(parents=True)
    # Deliberately exclude gold patches/tests from what is passed to the model.
    save(output/'run.json', {'mini_commit': MINI_COMMIT, 'mode': args.mode,
                            'manifest_sha': digest(manifest), 'task_id': task_id,
                            'total_model_call_limit': args.total_model_calls,
                            'config': config})
    from .progress import ProgressMixin
    class LoggedBaseline(ProgressMixin, DefaultAgent):
        pass
    model = get_model(config=config['model'])
    env = get_sb_environment(config, instance)
    if args.mode == 'experience':
        from .agent import SearchExperienceAgent
        class LoggedExperience(ProgressMixin, SearchExperienceAgent):
            pass
        agent = LoggedExperience(model, env, r2mem_config=args.r2mem_config, **agent_cfg)
    else:
        agent = LoggedBaseline(model, env, **agent_cfg)
    result = agent.run(instance['problem_statement'])
    save(output/'prediction.json', {'instance_id': task_id, 'model_name_or_path': config['model']['model_name'],
                                   'model_patch': result.get('submission', '')})
    print('Agent finished. Patch saved; SWE-bench test-harness evaluation has NOT been run.')
