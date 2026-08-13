"""Utility functions for TopoBO."""

import numpy as np
from scipy.stats import qmc


def latin_hypercube_sampling(bounds, n_samples, seed=None, log_scale=None):
    """Generate LHS samples within given bounds.

    Args:
        bounds: (d, 2) array of [lower, upper] bounds per dimension.
        n_samples: Number of samples to generate.
        seed: Random seed.
        log_scale: (d,) bool array. True = sample in log scale for that dim.

    Returns:
        (n_samples, d) array of samples.
    """
    d = bounds.shape[0]
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    if log_scale is not None:
        samples = np.zeros((n_samples, d))
        for i in range(d):
            if log_scale[i] and lower[i] > 0:
                log_lo, log_hi = np.log10(lower[i]), np.log10(upper[i])
                samples[:, i] = 10 ** (unit_samples[:, i] * (log_hi - log_lo) + log_lo)
            else:
                samples[:, i] = unit_samples[:, i] * (upper[i] - lower[i]) + lower[i]
        return samples
    return qmc.scale(unit_samples, lower, upper)


def normalize_to_unit(X, bounds):
    """Normalize X from [lower, upper] to [0, 1]."""
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    rng = upper - lower
    rng = np.where(rng == 0, 1.0, rng)
    return (X - lower) / rng


def unnormalize_from_unit(X_unit, bounds):
    """Unnormalize X from [0, 1] to [lower, upper]."""
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return X_unit * (upper - lower) + lower


def parse_spec_constraints(target_specs):
    """Parse YAML target_specs into constraint info.

    For _min specs: constraint is value >= first element (lower bound).
    For _max specs: constraint is value <= second element (upper bound).

    Returns:
        spec_names: List of spec names (sorted).
        constraints: Dict of spec_name -> threshold value.
        directions: Dict of spec_name -> 'min' or 'max'.
    """
    spec_names = sorted(target_specs.keys())
    constraints = {}
    directions = {}
    for name in spec_names:
        vals = target_specs[name]
        if name.endswith('_min'):
            constraints[name] = vals[0]  # value >= this
            directions[name] = 'min'
        elif name.endswith('_max'):
            constraints[name] = vals[1]  # value <= this
            directions[name] = 'max'
        else:
            constraints[name] = vals[0]
            directions[name] = 'min'
    return spec_names, constraints, directions


def check_feasibility(specs, constraints, directions):
    """Check if a spec dict satisfies all constraints.

    Args:
        specs: Dict of spec_name -> value.
        constraints: Dict of spec_name -> threshold.
        directions: Dict of spec_name -> 'min' or 'max'.

    Returns:
        True if all constraints are satisfied.
    """
    for name in constraints:
        val = specs.get(name, None)
        if val is None or np.isnan(val):
            return False
        if directions[name] == 'min' and val < constraints[name]:
            return False
        if directions[name] == 'max' and val > constraints[name]:
            return False
    return True


def compute_constraint_violation(specs, constraints, directions):
    """Compute normalized constraint violation for each spec.

    Returns a dict of spec_name -> violation (negative = violated, positive = satisfied).
    """
    violations = {}
    for name in constraints:
        val = specs.get(name, None)
        c = constraints[name]
        if val is None or np.isnan(val):
            violations[name] = -1.0
            continue
        if directions[name] == 'min':
            violations[name] = (val - c) / (abs(c) + 1e-10)
        else:
            violations[name] = (c - val) / (abs(c) + 1e-10)
    return violations


def compute_fom(specs, circuit_type, constraints=None, directions=None):
    """Compute Figure of Merit.

    - All specs satisfied to FoM = 10
    - Otherwise to FoM = sum of violated margins (negative).
      Satisfied specs contribute 0 (not positive), violated specs contribute negative.

    Args:
        specs: Dict of spec_name -> value.
        circuit_type: One of 'opamp', 'comparator', 'ldo'.
        constraints: Dict of spec_name -> threshold.
        directions: Dict of spec_name -> 'min' or 'max'.

    Returns:
        FoM value. 10 when feasible, negative otherwise.
    """
    if constraints is None or directions is None:
        return 0.0

    all_satisfied = True
    violation_sum = 0.0
    for name in constraints:
        val = specs.get(name, np.nan)
        c = constraints[name]
        if np.isnan(val):
            all_satisfied = False
            violation_sum += -1.0
            continue
        if directions[name] == 'min':
            margin = (val - c) / (abs(c) + 1e-6)
        else:
            margin = (c - val) / (abs(c) + 1e-6)

        if margin < 0:
            all_satisfied = False
            violation_sum += margin  # negative contribution
        # satisfied specs: contribute 0

    return 10.0 if all_satisfied else violation_sum


def detect_circuit_type(yaml_path):
    """Detect circuit type from YAML filename."""
    name = yaml_path.lower()
    if 'comparator' in name:
        return 'comparator'
    elif 'ldo' in name:
        return 'ldo'
    else:
        return 'opamp'


class ExperimentLogger:
    """Logs optimization progress per iteration."""

    def __init__(self, spec_names, constraints, directions, circuit_type):
        self.spec_names = spec_names
        self.constraints = constraints
        self.directions = directions
        self.circuit_type = circuit_type
        self.history_X = []
        self.history_specs = []
        self.history_feasible = []
        self.history_fom = []

    def log(self, x, specs):
        self.history_X.append(x.copy())
        self.history_specs.append(dict(specs))
        feas = check_feasibility(specs, self.constraints, self.directions)
        self.history_feasible.append(feas)
        fom = compute_fom(specs, self.circuit_type, self.constraints, self.directions) if feas else -np.inf
        self.history_fom.append(fom)

    def first_feasible_iter(self):
        """Return iteration index of first feasible point, or None."""
        for i, f in enumerate(self.history_feasible):
            if f:
                return i
        return None

    def best_feasible_fom(self):
        """Return best FoM among feasible designs."""
        feasible_foms = [f for f, feas in zip(self.history_fom, self.history_feasible) if feas]
        return max(feasible_foms) if feasible_foms else -np.inf

    def success(self):
        return any(self.history_feasible)

    def n_feasible(self):
        return sum(self.history_feasible)

    def to_dict(self):
        return {
            'X': np.array(self.history_X),
            'specs': self.history_specs,
            'feasible': self.history_feasible,
            'fom': self.history_fom,
            'first_feasible': self.first_feasible_iter(),
            'best_fom': self.best_feasible_fom(),
            'success': self.success(),
            'n_feasible': self.n_feasible(),
        }
