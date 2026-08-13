#!/usr/bin/env python3
"""Run one sizing experiment.

    python run_experiment.py --circuit two_stage_opamp_gf180 --method ATLAS \
        --budget 500 --seed 42
"""

import argparse
import os

from topbo.circuit_eval import CircuitEvaluator
from topbo.optimizers import get_optimizer


YAML_DIR = 'eval_engines/ngspice/ngspice_inputs/yaml_files'

CIRCUITS = {
    'two_stage_opamp_gf180': os.path.join(YAML_DIR, 'two_stage_opamp_gf180.yaml'),
    'cascode_miller_opamp_gf180': os.path.join(YAML_DIR, 'cascode_miller_opamp_gf180.yaml'),
    'comparator_gf180': os.path.join(YAML_DIR, 'comparator_gf180.yaml'),
    'ldo_sky130': os.path.join(YAML_DIR, 'ldo_sky130.yaml'),
}

METHODS = ['ATLAS', 'WEIBO', 'TuRBO', 'CMAES', 'Random', 'MACE', 'SCBO']


def n_init_for(evaluator, budget):
    """Initial design size: 3 per design parameter, capped at 50."""
    return max(min(3 * evaluator.n_params, budget // 3, 50), 10)


def run_single(circuit_name, method_name, budget, seed, log_root='logs',
               surrogate='auto', gp_time_limit=10.0, early_stop=True, **kwargs):
    evaluator = CircuitEvaluator(
        CIRCUITS[circuit_name],
        root_dir=os.path.join('tmp', f'{circuit_name}_{method_name}_{seed}'),
    )
    optimizer = get_optimizer(method_name, **kwargs)
    _, results = optimizer.optimize(
        evaluator, budget, n_init=n_init_for(evaluator, budget), seed=seed,
        log_root=log_root, surrogate=surrogate, early_stop=early_stop,
        gp_time_limit=gp_time_limit,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description='ATLAS experiment runner')
    parser.add_argument('--circuit', required=True, choices=list(CIRCUITS))
    parser.add_argument('--method', default='ATLAS', choices=METHODS)
    parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-root', default='logs')
    parser.add_argument('--surrogate', default='auto', choices=['gp', 'nn', 'auto'],
                        help='auto starts on the GP and switches to the NN '
                             'ensemble only once GP fitting gets slow')
    parser.add_argument('--gp-time-limit', type=float, default=10.0,
                        help='seconds of GP fitting that trigger the switch in auto mode')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='keep sampling after the first feasible design')
    args = parser.parse_args()

    results = run_single(
        args.circuit, args.method, args.budget, args.seed,
        log_root=args.log_root, surrogate=args.surrogate,
        gp_time_limit=args.gp_time_limit, early_stop=not args.no_early_stop,
    )
    ff = results.get('first_feasible_iter')
    print(f'{args.method} on {args.circuit}: '
          + (f'feasible at evaluation {ff}' if ff is not None
             else f'no feasible design within {args.budget} evaluations'))


if __name__ == '__main__':
    main()
