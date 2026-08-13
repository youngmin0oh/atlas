"""Circuit evaluation wrapper for SPICE simulation (TT-only or full PVT)."""

import os
import numpy as np
import yaml
from collections import OrderedDict

from eval_engines.ngspice.CircuitClass import CircuitClass
from topbo.utils import (
    parse_spec_constraints, detect_circuit_type, check_feasibility,
)


class CircuitEvaluator:
    """Wraps CircuitClass for single-point evaluation.

    Supports TT-only (default) or full PVT multi-corner evaluation.
    In PVT mode: TT first, then all corners if TT passes.
    Returns worst-case specs when PVT is run, TT specs otherwise.
    """

    def __init__(self, yaml_path, root_dir=None, pvt=False):
        self.yaml_path = yaml_path
        self.base_path = os.getcwd()
        self.pvt = pvt

        with open(yaml_path, 'r') as f:
            self.config = yaml.full_load(f)

        all_netlists = self.config['dsn_netlist']

        # Always create TT-only simulator
        self.sim_env = CircuitClass(
            yaml_path=yaml_path,
            path=self.base_path,
            num_process=1,
            root_dir=root_dir,
            design_netlists=[all_netlists[0]],
        )

        # PVT mode: also create multi-corner simulator
        if pvt and len(all_netlists) > 1:
            pvt_netlists = all_netlists[1:]
            max_parallel = min(len(pvt_netlists), 8)
            self.sim_env_pvt = CircuitClass(
                yaml_path=yaml_path,
                path=self.base_path,
                num_process=max_parallel,
                root_dir=root_dir,
                design_netlists=pvt_netlists,
            )
            # Per-corner simulators for selective evaluation
            self.corner_sims = []
            self.corner_names = []
            for netlist in pvt_netlists:
                sim = CircuitClass(
                    yaml_path=yaml_path,
                    path=self.base_path,
                    num_process=1,
                    root_dir=root_dir,
                    design_netlists=[netlist],
                )
                self.corner_sims.append(sim)
                # Extract corner name from netlist path
                name = os.path.splitext(os.path.basename(netlist))[0]
                self.corner_names.append(name)
            self.n_pvt_corners = len(pvt_netlists)
        else:
            self.sim_env_pvt = None
            self.corner_sims = []
            self.corner_names = []
            self.n_pvt_corners = 0

        # Parameter info
        params = self.config['params']
        self.param_names = list(params.keys())
        self.param_bounds = np.array([[v[0], v[1]] for v in params.values()])
        self.n_params = len(self.param_names)

        # Spec info
        self.target_specs = self.config['target_specs']
        self.spec_names, self.constraints, self.directions = parse_spec_constraints(
            self.target_specs
        )
        self.normalize_vals = np.array(self.config.get('normalize', [1.0] * len(self.spec_names)))

        # Circuit type
        self.circuit_type = detect_circuit_type(yaml_path)

        # Precision for parameter rounding
        self.prec_params = self.config.get('prec_params', [9] * self.n_params)

        # Auto-detect log-scale params
        self.log_scale = np.array([
            (self.param_bounds[i, 1] / (self.param_bounds[i, 0] + 1e-30) > 10)
            and self.param_bounds[i, 0] > 0
            for i in range(self.n_params)
        ])

        # Known-good initial parameters from YAML
        raw_init = self.config.get('init_params')
        if raw_init is not None:
            self.init_params = np.array(raw_init, dtype=float)
        else:
            self.init_params = None

    def _round_params(self, x):
        rounded = np.zeros_like(x)
        for i, (val, prec) in enumerate(zip(x, self.prec_params)):
            rounded[i] = round(float(val), prec)
        return rounded

    def _tt_spec_satisfaction(self, tt_result):
        """Count how many specs TT passes and compute total violation."""
        n_pass = 0
        total_viol = 0.0
        for name in self.spec_names:
            val = tt_result.get(name, np.nan)
            c = self.constraints[name]
            if np.isnan(val):
                total_viol += 1.0
                continue
            if self.directions[name] == 'min':
                margin = (val - c) / (abs(c) + 1e-10)
            else:
                margin = (c - val) / (abs(c) + 1e-10)
            if margin >= 0:
                n_pass += 1
            else:
                total_viol += abs(margin)
        return n_pass, total_viol

    def evaluate_tt(self, x):
        """Evaluate TT corner only, regardless of PVT mode.

        Always returns TT-only specs (no worst-case aggregation).
        Use this for GP surrogate training in PVT mode so the model
        learns a clean, consistent landscape.
        """
        x = np.clip(x, self.param_bounds[:, 0], self.param_bounds[:, 1])
        x = self._round_params(x)
        state = OrderedDict(zip(self.param_names, x.tolist()))
        try:
            _, specs_list, _ = self.sim_env.run(state)
            tt_specs = specs_list[0]
            if tt_specs is None:
                return self._penalty_specs()
            return self._remap_specs(tt_specs)
        except Exception as e:
            print(f"[CircuitEvaluator] TT eval failed: {e}")
            return self._penalty_specs()

    def evaluate(self, x):
        """Evaluate a single design point.

        TT-only mode: returns TT specs.
        PVT mode: Soft TT gate — runs corners if TT passes ≥ half specs
                  or total violation is small. Returns worst-case specs.

        Metadata keys (not stored in surrogate):
            _pvt_pass: bool, all PVT corners pass
            _pvt_corners_pass / _pvt_corners_total: corner counts
        """
        x = np.clip(x, self.param_bounds[:, 0], self.param_bounds[:, 1])
        x = self._round_params(x)
        state = OrderedDict(zip(self.param_names, x.tolist()))

        try:
            # Always evaluate TT corner first
            _, specs_list, _ = self.sim_env.run(state)
            tt_specs = specs_list[0]
            if tt_specs is None:
                return self._penalty_specs()
            tt_result = self._remap_specs(tt_specs)

            if not self.pvt or self.sim_env_pvt is None:
                return tt_result

            # PVT mode: soft TT gate — run corners if TT is close to feasible
            n_total = len(self.sim_env_pvt.base_design_names) + 1
            tt_feasible = check_feasibility(tt_result, self.constraints, self.directions)
            n_specs_pass, total_viol = self._tt_spec_satisfaction(tt_result)
            n_specs = len(self.spec_names)

            # Soft gate: skip corners only if TT is very far from feasible
            # (passes fewer than half specs OR total violation > 2.0)
            if not tt_feasible and (n_specs_pass < n_specs // 2 or total_viol > 2.0):
                tt_result['_pvt_pass'] = False
                tt_result['_pvt_corners_pass'] = 0
                tt_result['_pvt_corners_total'] = n_total
                tt_result['_tt_specs_pass'] = n_specs_pass
                return tt_result

            # TT passes soft gate to run all PVT corners
            _, pvt_specs_list, _ = self.sim_env_pvt.run(state)
            all_specs = [tt_specs] + list(pvt_specs_list)

            # Count per-corner pass/fail
            n_pass = 0
            for corner_specs in all_specs:
                if corner_specs is not None:
                    remapped = self._remap_specs(corner_specs)
                    if check_feasibility(remapped, self.constraints, self.directions):
                        n_pass += 1

            # Return worst-case specs for surrogate learning
            wc = self._worst_case_specs(all_specs)
            wc['_pvt_pass'] = (n_pass == n_total)
            wc['_pvt_corners_pass'] = n_pass
            wc['_pvt_corners_total'] = n_total
            wc['_tt_specs_pass'] = n_specs_pass
            return wc

        except Exception as e:
            print(f"[CircuitEvaluator] Simulation failed: {e}")
            return self._penalty_specs()

    def evaluate_batch(self, X):
        return [self.evaluate(x) for x in X]

    def _remap_specs(self, raw_specs):
        """Remap raw spec dict to match target_specs keys."""
        result = {}
        raw_keys = {k.lower(): k for k in raw_specs}
        for target_key in self.spec_names:
            base_key = target_key
            for suffix in ['_min', '_max']:
                if base_key.endswith(suffix):
                    base_key = base_key[:-len(suffix)]
                    break
            if target_key in raw_specs:
                result[target_key] = float(raw_specs[target_key])
            elif base_key in raw_specs:
                result[target_key] = float(raw_specs[base_key])
            elif base_key.lower() in raw_keys:
                result[target_key] = float(raw_specs[raw_keys[base_key.lower()]])
            else:
                found = False
                for suffix in ['_min', '_max']:
                    if target_key.endswith(suffix):
                        raw_key_candidate = target_key[:-len(suffix)]
                        if raw_key_candidate in raw_specs:
                            result[target_key] = float(raw_specs[raw_key_candidate])
                            found = True
                            break
                if not found:
                    result[target_key] = np.nan
        return result

    def _worst_case_specs(self, specs_list):
        """Return worst-case specs across all PVT corners."""
        remapped = []
        for specs in specs_list:
            if specs is None:
                return self._penalty_specs()
            remapped.append(self._remap_specs(specs))

        result = {}
        for name in self.spec_names:
            vals = [r[name] for r in remapped if not np.isnan(r.get(name, np.nan))]
            if not vals:
                result[name] = np.nan
            elif self.directions[name] == 'min':
                result[name] = min(vals)
            else:
                result[name] = max(vals)
        return result

    def _penalty_specs(self):
        result = {}
        for name in self.spec_names:
            if self.directions[name] == 'min':
                result[name] = -1e10
            else:
                result[name] = 1e10
        return result

    def get_spec_array(self, specs_dict):
        return np.array([specs_dict.get(name, np.nan) for name in self.spec_names])

    def get_constraint_array(self):
        return np.array([self.constraints[name] for name in self.spec_names])

    def get_direction_signs(self):
        return np.array([1.0 if self.directions[n] == 'min' else -1.0
                         for n in self.spec_names])
