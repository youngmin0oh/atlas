#!/usr/bin/env python3
"""Benchmark several methods on one circuit over repeated runs.

Each run stops as soon as a design satisfies every target specification, so the
two reported quantities are the success rate and the number of SPICE
evaluations to the first success.

    python run_benchmark.py --circuit cascode_miller_opamp_gf180 \
        --methods ATLAS WEIBO SCBO --budget 500 --n-runs 10
"""

import argparse
import json
import os
import time

import numpy as np
import yaml

from topbo.circuit_eval import CircuitEvaluator
from topbo.optimizers import get_optimizer
from run_experiment import CIRCUITS, METHODS, n_init_for


def sample_spec_targets(yaml_path, n_runs, seed=42):
    """One target-spec dict per run.

    Each entry of target_specs is a [low, high] range. The shipped circuit
    files pin low == high, so every run gets the same targets and the runs
    differ only by their optimizer seed; a widened range instead draws a fresh
    target per run.
    """
    with open(yaml_path) as f:
        target_specs = yaml.full_load(f)['target_specs']

    rng = np.random.RandomState(seed)
    targets = []
    for _ in range(n_runs):
        targets.append({
            name: lo if abs(hi - lo) < 1e-12 else rng.uniform(lo, hi)
            for name, (lo, hi) in sorted(target_specs.items())
        })
    return targets


def run_one(yaml_path, spec_target, method, budget, seed, log_root, surrogate,
            opt_kwargs):
    evaluator = CircuitEvaluator(yaml_path)
    for name, value in spec_target.items():
        if name in evaluator.constraints:
            evaluator.constraints[name] = value

    optimizer = get_optimizer(method, **(opt_kwargs or {}))
    _, results = optimizer.optimize(
        evaluator, budget, n_init=n_init_for(evaluator, budget), seed=seed,
        log_root=log_root, surrogate=surrogate, early_stop=True,
    )
    return results


def run_benchmark(circuit, n_runs, budget, methods, seed=42, surrogate='auto',
                  log_root='logs/benchmark', opt_kwargs=None):
    yaml_path = CIRCUITS[circuit]
    targets = sample_spec_targets(yaml_path, n_runs, seed=seed)

    print(f'{circuit}: {n_runs} runs x {len(methods)} methods, '
          f'budget {budget}, base seed {seed}')

    rows = []
    for run_idx, spec_target in enumerate(targets):
        print(f'\nrun {run_idx + 1}/{n_runs}')
        for method in methods:
            run_seed = seed + run_idx
            t0 = time.time()
            try:
                results = run_one(
                    yaml_path, spec_target, method, budget, run_seed,
                    os.path.join(log_root, f'run{run_idx:02d}'), surrogate,
                    opt_kwargs)
                first = results.get('first_feasible_iter')
                success = results.get('success', False)
            except Exception as exc:
                print(f'  {method}: failed ({exc})')
                first, success = None, False
            elapsed = time.time() - t0

            rows.append({
                'run': run_idx, 'method': method, 'budget': budget,
                'seed': run_seed, 'first_feasible': first, 'success': success,
                'spec_target': {k: float(v) for k, v in spec_target.items()},
                'wall_time_s': round(elapsed, 1),
            })
            status = f'feasible@{first}' if first is not None else 'not found'
            print(f'  {method:<8} {status:<16} {elapsed:6.0f}s')

    print(f'\n{"method":<10}{"SR":<10}{"mean evals to first success":<30}')
    for method in methods:
        got = [r for r in rows if r['method'] == method]
        hits = [r['first_feasible'] for r in got if r['first_feasible'] is not None]
        mean = f'{np.mean(hits):.0f} +/- {np.std(hits):.0f}' if hits else '-'
        print(f'{method:<10}{len(hits)}/{len(got):<8}{mean:<30}')

    os.makedirs(log_root, exist_ok=True)
    out = os.path.join(log_root, 'benchmark_results.json')
    with open(out, 'w') as f:
        json.dump({'circuit': circuit, 'budget': budget, 'seed': seed,
                   'methods': methods, 'results': rows}, f, indent=2)
    print(f'\nwrote {out}')
    return rows


def main():
    parser = argparse.ArgumentParser(description='ATLAS benchmark')
    parser.add_argument('--circuit', required=True, choices=list(CIRCUITS))
    parser.add_argument('--methods', nargs='+', default=['ATLAS', 'WEIBO'],
                        choices=METHODS)
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--surrogate', default='auto', choices=['gp', 'nn', 'auto'])
    parser.add_argument('--log-root', default='logs/benchmark')

    group = parser.add_argument_group('ATLAS settings')
    group.add_argument('--mapper-n-cubes', type=int, default=10)
    group.add_argument('--mapper-overlap', type=float, default=0.3)
    group.add_argument('--mapper-min-cluster', type=int, default=3)
    group.add_argument('--mapper-dbscan-eps', type=float, default=0.3)
    group.add_argument('--top-k-frac', type=float, default=None)
    group.add_argument('--no-region-discount', action='store_true')
    group.add_argument('--no-centroid-guidance', action='store_true')
    group.add_argument('--no-lambda-decay', action='store_true')
    args = parser.parse_args()

    opt_kwargs = {
        'mapper_n_cubes': args.mapper_n_cubes,
        'mapper_overlap': args.mapper_overlap,
        'mapper_min_cluster': args.mapper_min_cluster,
        'mapper_dbscan_eps': args.mapper_dbscan_eps,
        'use_component_discount': not args.no_region_discount,
        'use_centroid_guidance': not args.no_centroid_guidance,
        'use_lambda_decay': not args.no_lambda_decay,
    }
    if args.top_k_frac is not None:
        opt_kwargs['top_k_frac'] = args.top_k_frac

    run_benchmark(args.circuit, args.n_runs, args.budget, args.methods,
                  seed=args.seed, surrogate=args.surrogate,
                  log_root=args.log_root, opt_kwargs=opt_kwargs)


if __name__ == '__main__':
    main()
