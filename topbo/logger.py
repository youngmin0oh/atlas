"""Structured experiment logging system.

Each run creates a directory:
    logs/{circuit}_{method}_B{budget}_S{seed}_{timestamp}/
        config.json          # Experiment configuration
        progress.log         # Human-readable iteration log
        iterations.jsonl     # Machine-readable per-iteration data
        tda_snapshots/       # TDA visualization PNGs (if enabled)
        results.json         # Final summary
"""

import json
import logging
import os
import time
from datetime import datetime

import numpy as np


class RunLogger:
    """Manages all logging for a single optimization run."""

    def __init__(self, circuit_name, method_name, budget, seed, n_init,
                 spec_names, constraints, directions, param_names,
                 log_root='logs'):
        self.circuit_name = circuit_name
        self.method_name = method_name
        self.budget = budget
        self.seed = seed
        self.n_init = n_init
        self.spec_names = spec_names
        self.constraints = dict(constraints)  # snapshot: immune to curriculum changes
        self.directions = directions
        self.param_names = param_names

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_id = f'{circuit_name}_{method_name}_B{budget}_S{seed}_{ts}'
        self.run_dir = os.path.join(log_root, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # File handles
        self.jsonl_path = os.path.join(self.run_dir, 'iterations.jsonl')
        self.jsonl_file = open(self.jsonl_path, 'w')

        # Python logger for progress.log + console
        self.log = logging.getLogger(self.run_id)
        self.log.setLevel(logging.DEBUG)
        self.log.handlers.clear()

        fh = logging.FileHandler(os.path.join(self.run_dir, 'progress.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))

        import sys
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        ch.flush = sys.stdout.flush  # force immediate flush

        self.log.addHandler(fh)
        self.log.addHandler(ch)

        # Tracking state
        self.start_time = time.time()
        self.iter_times = []
        self.n_feasible = 0
        self.first_feasible_iter = None
        self.best_fom = -np.inf
        self.best_fom_iter = None

        # Save config
        self._save_config()

    def _save_config(self):
        config = {
            'circuit': self.circuit_name,
            'method': self.method_name,
            'budget': self.budget,
            'seed': self.seed,
            'n_init': self.n_init,
            'param_names': self.param_names,
            'spec_names': self.spec_names,
            'constraints': {k: float(v) for k, v in self.constraints.items()},
            'directions': self.directions,
            'start_time': datetime.now().isoformat(),
        }
        with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def log_start(self):
        self.log.info(f'{"="*60}')
        self.log.info(f'RUN: {self.run_id}')
        self.log.info(f'  Circuit: {self.circuit_name} | Method: {self.method_name}')
        self.log.info(f'  Budget: {self.budget} | n_init: {self.n_init} | Seed: {self.seed}')
        self.log.info(f'  Params ({len(self.param_names)}): {self.param_names}')
        self.log.info(f'  Specs ({len(self.spec_names)}): {self.spec_names}')
        self.log.info(f'  Log dir: {self.run_dir}')
        self.log.info(f'{"="*60}')

    def log_init_sample(self, i, specs, feasible, sim_time):
        """Log one initial LHS sample."""
        phase = 'INIT'
        status = 'FEAS' if feasible else 'INFEAS'
        if feasible:
            self.n_feasible += 1
            if self.first_feasible_iter is None:
                self.first_feasible_iter = i
        spec_str = '  '.join(f'{k}={v:.4g}' for k, v in specs.items()
                             if not k.startswith('_'))
        pvt_str = ''
        if '_pvt_corners_total' in specs:
            pvt_str = f' | pvt={specs["_pvt_corners_pass"]}/{specs["_pvt_corners_total"]}'
        self.log.info(f'[{phase}] {i+1:>3d}/{self.n_init} {status} | {spec_str} | sim={sim_time:.2f}s{pvt_str}')

        # JSONL
        entry = {
            'iter': i, 'phase': 'init', 'feasible': feasible,
            'specs': {k: _safe_float(v) for k, v in specs.items()
                      if not k.startswith('_')},
            'sim_time': round(sim_time, 3),
        }
        if '_pvt_corners_total' in specs:
            entry['pvt_corners_pass'] = specs['_pvt_corners_pass']
            entry['pvt_corners_total'] = specs['_pvt_corners_total']
        self.jsonl_file.write(json.dumps(entry) + '\n')

    def log_init_summary(self, total_time):
        self.log.info(f'[INIT] Done: {self.n_init} samples in {total_time:.1f}s, '
                      f'{self.n_feasible} feasible')

    def log_iteration(self, t, specs, feasible, fom,
                      sim_time, gp_time, tda_time, acq_time,
                      beta0=0, n_F_hat=0, lambda_t=0.0):
        """Log one BO iteration."""
        iter_total = sim_time + gp_time + tda_time + acq_time
        self.iter_times.append(iter_total)

        if feasible:
            self.n_feasible += 1
            if self.first_feasible_iter is None:
                self.first_feasible_iter = t
                self.log.info(f'[ITER {t:>3d}] *** FIRST FEASIBLE FOUND ***')
            if fom > self.best_fom:
                self.best_fom = fom
                self.best_fom_iter = t

        status = 'FEAS' if feasible else 'INFEAS'

        # Compact spec summary: only show violated specs for infeasible
        if feasible:
            spec_str = f'FoM={fom:.4g}'
        else:
            violated = []
            for name in self.spec_names:
                val = specs.get(name, np.nan)
                c = self.constraints[name]
                d = self.directions[name]
                if d == 'min' and val < c:
                    violated.append(f'{name}={val:.3g}<{c:.3g}')
                elif d == 'max' and val > c:
                    violated.append(f'{name}={val:.3g}>{c:.3g}')
            spec_str = ', '.join(violated[:3])
            if len(violated) > 3:
                spec_str += f' (+{len(violated)-3})'

        pvt_str = ''
        if '_pvt_corners_total' in specs:
            pvt_str = f' pvt={specs["_pvt_corners_pass"]}/{specs["_pvt_corners_total"]}'

        timing_str = f'sim={sim_time:.2f}s fit={gp_time:.2f}s'
        if tda_time > 0:
            timing_str += f' tda={tda_time:.3f}s'

        msg = (f'[ITER {t:>3d}/{self.budget}] {status} | {spec_str} | '
               f'{timing_str} | feas={self.n_feasible} β₀={beta0} |F̂|={n_F_hat}{pvt_str}')
        self.log.info(msg)  # every iteration to console

        # JSONL
        entry = {
            'iter': t, 'phase': 'bo', 'feasible': feasible,
            'fom': _safe_float(fom),
            'specs': {k: _safe_float(v) for k, v in specs.items()
                      if not k.startswith('_')},
            'sim_time': round(sim_time, 3),
            'fit_time': round(gp_time, 3),
            'tda_time': round(tda_time, 4),
            'acq_time': round(acq_time, 3),
            'beta0': beta0,
            'n_F_hat': n_F_hat,
            'lambda_t': round(lambda_t, 6),
            'n_feasible_so_far': self.n_feasible,
            'best_fom_so_far': _safe_float(self.best_fom),
        }
        if '_pvt_corners_total' in specs:
            entry['pvt_corners_pass'] = specs['_pvt_corners_pass']
            entry['pvt_corners_total'] = specs['_pvt_corners_total']
        self.jsonl_file.write(json.dumps(entry) + '\n')
        self.jsonl_file.flush()

    def log_finish(self):
        """Log final summary and save results.json."""
        total_time = time.time() - self.start_time
        self.log.info(f'{"="*60}')
        self.log.info(f'FINISHED: {self.run_id}')
        self.log.info(f'  Total time: {total_time:.1f}s ({total_time/60:.1f}min)')
        self.log.info(f'  Feasible found: {self.n_feasible}/{self.budget}')
        self.log.info(f'  First feasible: iter {self.first_feasible_iter}')
        self.log.info(f'  Best FoM: {self.best_fom:.4g} (iter {self.best_fom_iter})')
        if self.iter_times:
            self.log.info(f'  Avg iter time: {np.mean(self.iter_times):.2f}s')
        self.log.info(f'  Results: {self.run_dir}/results.json')
        self.log.info(f'{"="*60}')

        results = {
            'run_id': self.run_id,
            'circuit': self.circuit_name,
            'method': self.method_name,
            'budget': self.budget,
            'seed': self.seed,
            'total_time_sec': round(total_time, 1),
            'n_feasible': self.n_feasible,
            'first_feasible_iter': self.first_feasible_iter,
            'best_fom': _safe_float(self.best_fom),
            'best_fom_iter': self.best_fom_iter,
            'success': self.n_feasible > 0,
            'avg_iter_time': round(np.mean(self.iter_times), 3) if self.iter_times else None,
        }
        with open(os.path.join(self.run_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        self.jsonl_file.close()
        return results

    def close(self):
        """Cleanup."""
        if not self.jsonl_file.closed:
            self.jsonl_file.close()
        for h in self.log.handlers[:]:
            h.close()
            self.log.removeHandler(h)


def _safe_float(v):
    """Convert to JSON-safe float."""
    if v is None:
        return None
    v = float(v)
    if np.isnan(v) or np.isinf(v):
        return None
    return round(v, 6)
