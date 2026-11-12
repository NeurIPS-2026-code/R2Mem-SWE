import argparse
from pathlib import Path
from .core import read, manifest_check


def main():
    parser = argparse.ArgumentParser(description='R2Mem search adaptation: no implicit API calls or task execution')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Offline trajectory prefix extraction, stdlib only')
    p.add_argument('--manifest', required=True)
    p.add_argument('--trajectories', nargs='+', required=True)
    p.add_argument('--output', required=True)
    for name in ('evaluate', 'learn'):
        p = commands.add_parser(name, help='Calls the explicitly configured evaluator/learner endpoint')
        p.add_argument('--records', required=True)
        p.add_argument('--api-config', required=True)
        p.add_argument('--output', required=True)
        if name == 'learn':
            p.add_argument('--evaluations', required=True)
    p = commands.add_parser('index', help='Local embedding only; no network download')
    p.add_argument('--bank', required=True)
    p.add_argument('--embedding-model', required=True)
    p.add_argument('--output', required=True)
    p = commands.add_parser('preflight', help='Read-only runtime check; no model/API/container starts')
    p.add_argument('--mini-checkout', required=True)
    p = commands.add_parser('run', help='EXPLICIT execution: starts one SWE-bench environment and model calls')
    p.add_argument('--mini-checkout', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--instance', required=True, help='Local JSON for exactly one SWE-bench instance')
    p.add_argument('--config', nargs='+', required=True, help='mini YAML base + overrides, same for both arms')
    p.add_argument('--mode', choices=['baseline', 'experience'], required=True)
    p.add_argument('--split', choices=['construction', 'development', 'evaluation', 'benchmark'], required=True)
    p.add_argument('--total-model-calls', type=int, required=True)
    p.add_argument('--r2mem-config')
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        from .pipeline import prepare
        prepare(args.manifest, args.trajectories, args.output)
    elif args.command == 'evaluate':
        from .pipeline import evaluate
        evaluate(args.records, args.api_config, args.output)
    elif args.command == 'learn':
        from .pipeline import learn
        learn(args.records, args.evaluations, args.api_config, args.output)
    elif args.command == 'index':
        from .bank import create_index
        create_index(args.bank, args.embedding_model, args.output)
    elif args.command == 'run':
        from .run_one import run
        run(args)
    else:
        import importlib.util
        import shutil
        import subprocess
        from . import MINI_COMMIT
        revision = subprocess.check_output(['git', '-C', args.mini_checkout, 'rev-parse', 'HEAD'], text=True).strip()
        if revision != MINI_COMMIT:
            raise ValueError(f'Unvalidated mini version: {revision}, expected {MINI_COMMIT}')
        for module in ('minisweagent', 'sentence_transformers', 'numpy'):
            print(module, 'available' if importlib.util.find_spec(module) else 'MISSING')
        print('docker:', shutil.which('docker') or 'MISSING; benchmark execution not ready')
        print('Pinned mini checkout verified. No models/APIs/containers started.')


if __name__ == '__main__':
    main()
