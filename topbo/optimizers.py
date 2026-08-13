"""Optimizers: ATLAS (TopoBO) and the baselines used in the paper."""

import sys
import time
import numpy as np

from topbo.utils import (
    latin_hypercube_sampling, normalize_to_unit, check_feasibility,
    compute_fom, ExperimentLogger,
)
from topbo.gp_model import GPSurrogate
from topbo.nn_model import NNSurrogate
from topbo.mapper import MapperAnalyzer
from topbo.acquisition import compute_wei, compute_lambda_t, compute_mace
from topbo.logger import RunLogger


N_CANDIDATES = 50000


def has_feasible_so_far(fom_np):
    """True once any evaluated design has satisfied every constraint.

    compute_fom returns 10 for a feasible design and a negative violation sum
    otherwise, so a non-negative entry marks the first success.
    """
    finite = fom_np[np.isfinite(fom_np)]
    return len(finite) > 0 and np.any(finite >= 0.0)


def _best_violation_score(Y_np, spec_names, constraints, directions):
    """Best min-over-specs log margin across all observations.

    Drives stagnation detection: no improvement here means the search is stuck.
    """
    n = len(Y_np[spec_names[0]])
    h_all = np.full(n, np.inf)
    for name in spec_names:
        vals = Y_np[name]
        c = constraints[name]
        eps = abs(c) * 1e-10 + 1e-30
        if directions[name] == 'min':
            h_spec = np.log(np.maximum(vals, eps) / (abs(c) + 1e-30))
        else:
            h_spec = np.log((abs(c) + 1e-30) / np.maximum(vals, eps))
        h_all = np.minimum(h_all, h_spec)
    return np.nanmax(h_all)


class BaseOptimizer:

    def __init__(self, name):
        self.name = name

    def optimize(self, evaluator, budget, n_init=20, seed=42,
                 log_root='logs', surrogate='auto', early_stop=False,
                 gp_time_limit=10.0):
        raise NotImplementedError


class _GPBasedOptimizer(BaseOptimizer):
    """Shared BO driver: initial design, surrogate fit, candidates, evaluation.

    Subclasses supply _select_next, which scores a candidate pool and returns
    the design to simulate.
    """

    # Full surrogate refit every REFIT_INTERVAL steps; fine-tune in between.
    REFIT_INTERVAL = 5
    # Iterations without progress before falling back to a heuristic restart.
    RESTART_PATIENCE = 50

    def optimize(self, evaluator, budget, n_init=20, seed=42,
                 log_root='logs', surrogate='auto', early_stop=False,
                 gp_time_limit=10.0):
        np.random.seed(seed)

        # Target specifications. Feasibility, FoM and early stopping are always
        # judged against these; the curriculum below relaxes a separate working
        # copy that only ever reaches the surrogate.
        target_constraints = dict(evaluator.constraints)

        rl = RunLogger(
            circuit_name=evaluator.yaml_path.split('/')[-1].replace('.yaml', ''),
            method_name=self.name,
            budget=budget, seed=seed, n_init=n_init,
            spec_names=evaluator.spec_names,
            constraints=target_constraints,
            directions=evaluator.directions,
            param_names=evaluator.param_names,
            log_root=log_root,
        )
        rl.log_start()

        mem_logger = ExperimentLogger(
            evaluator.spec_names, target_constraints,
            evaluator.directions, evaluator.circuit_type,
        )
        beta0_history = []
        bounds = evaluator.param_bounds
        log_scale = getattr(evaluator, 'log_scale', None)

        X_init = self._initial_design(evaluator, bounds, log_scale, n_init, seed)

        X_data = []
        Y_data = {name: [] for name in evaluator.spec_names}
        fom_data = []

        t_init_start = time.time()
        early_found = False
        for i in range(n_init):
            t0 = time.time()
            specs = evaluator.evaluate(X_init[i])
            sim_time = time.time() - t0
            mem_logger.log(X_init[i], specs)

            feas = check_feasibility(specs, target_constraints, evaluator.directions)
            rl.log_init_sample(i, specs, feas, sim_time)

            X_data.append(X_init[i])
            for name in evaluator.spec_names:
                Y_data[name].append(specs.get(name, np.nan))
            fom = compute_fom(specs, evaluator.circuit_type,
                              target_constraints, evaluator.directions) if feas else np.nan
            fom_data.append(fom)

            if early_stop and feas:
                rl.log.info(f'[EARLY STOP] Feasible found during init at sample {i}')
                early_found = True
                break

        X_data = np.array(X_data)
        Y_np = {name: np.array(vals) for name, vals in Y_data.items()}
        fom_np = np.array(fom_data)
        rl.log_init_summary(time.time() - t_init_start)

        if early_found:
            results = rl.log_finish()
            rl.close()
            return mem_logger, results

        model, nn_shadow, auto_mode = self._build_surrogate(
            surrogate, evaluator, bounds, rl, gp_time_limit)
        switched_to_nn = False

        curriculum_active = True
        best_h_so_far = -np.inf
        stagnation_counter = 0

        for t in range(n_init, budget):
            sys.stdout.flush()

            if curriculum_active:
                if has_feasible_so_far(fom_np):
                    working = dict(target_constraints)
                    curriculum_active = False
                else:
                    working = self._relax_constraints(
                        evaluator, target_constraints, Y_np, t, n_init, budget)
                for m in (model, nn_shadow):
                    if m is not None:
                        m.constraints = dict(working)

            t0 = time.time()
            has_feasible = has_feasible_so_far(fom_np)
            incremental = (t > n_init) and ((t - n_init) % self.REFIT_INTERVAL != 0)
            model.fit(X_data, Y_np,
                      fom_values=fom_np if has_feasible else None,
                      incremental=incremental)
            gp_time = time.time() - t0

            if auto_mode and not switched_to_nn:
                nn_shadow.constraints = dict(model.constraints)
                nn_shadow.fit(X_data, Y_np,
                              fom_values=fom_np if has_feasible else None,
                              incremental=incremental)
                if gp_time > gp_time_limit:
                    rl.log.info(f'[AUTO] GP fit took {gp_time:.1f}s at iter {t} '
                                f'(n={X_data.shape[0]}); switching to NN ensemble.')
                    model, nn_shadow, switched_to_nn = nn_shadow, None, True

            if stagnation_counter >= self.RESTART_PATIENCE and not has_feasible:
                x_next = self._restart_candidate(
                    evaluator, X_data, Y_np, target_constraints, bounds,
                    log_scale, seed, t, stagnation_counter, rl)
                stagnation_counter = 0
                tda_time = acq_time = 0.0
                beta0, n_F_hat, lambda_t = 0, 0, 0.0
                beta0_history.append(beta0)
            else:
                X_cand = self._make_candidates(
                    evaluator, X_data, Y_np, target_constraints, bounds,
                    log_scale, seed, t, n_init, budget, has_feasible)

                t0 = time.time()
                feas_foms = fom_np[np.isfinite(fom_np) & (fom_np >= 0.0)]
                best_f = float(np.max(feas_foms)) if len(feas_foms) > 0 else -np.inf
                sel = self._select_next(model, X_cand, bounds, t, budget, best_f,
                                        observed_Y=Y_np)
                x_next = sel['x_next']
                tda_time = sel.get('tda_time', 0.0)
                acq_time = time.time() - t0 - tda_time

                beta0 = sel.get('beta0', 0)
                n_F_hat = len(sel.get('F_hat', []))
                lambda_t = sel.get('lambda_t', 0.0)
                beta0_history.append(beta0)

            t0 = time.time()
            specs = evaluator.evaluate(x_next)
            sim_time = time.time() - t0
            mem_logger.log(x_next, specs)

            feas = check_feasibility(specs, target_constraints, evaluator.directions)
            # Keep the violation-based FoM for infeasible designs too, so the
            # surrogate has a continuous ranking signal instead of NaN.
            fom = compute_fom(specs, evaluator.circuit_type,
                              target_constraints, evaluator.directions)

            rl.log_iteration(t, specs, feas, fom,
                             sim_time=sim_time, gp_time=gp_time,
                             tda_time=tda_time, acq_time=acq_time,
                             beta0=beta0, n_F_hat=n_F_hat, lambda_t=lambda_t)

            if early_stop and feas:
                rl.log.info(f'[EARLY STOP] Feasible found at iter {t}.')
                break

            X_data = np.vstack([X_data, x_next.reshape(1, -1)])
            for name in evaluator.spec_names:
                Y_np[name] = np.append(Y_np[name], specs.get(name, np.nan))
            fom_np = np.append(fom_np, fom)

            current_h = _best_violation_score(
                Y_np, evaluator.spec_names, target_constraints,
                evaluator.directions)
            if current_h > best_h_so_far + 0.01:
                best_h_so_far = current_h
                stagnation_counter = 0
            else:
                stagnation_counter += 1

        results = rl.log_finish()
        rl.close()
        return mem_logger, results

    @staticmethod
    def _initial_design(evaluator, bounds, log_scale, n_init, seed):
        """Initial samples: LHS, seeded with the YAML reference design if given.

        When the circuit YAML ships an init_params vector, 60% of the budget is
        spent on perturbations around it and the rest on global LHS.
        """
        init_params = getattr(evaluator, 'init_params', None)
        if init_params is None:
            return latin_hypercube_sampling(bounds, n_init, seed=seed,
                                            log_scale=log_scale)

        n_local = max(n_init * 3 // 5, 1)
        n_lhs = max(n_init - n_local - 1, 0)
        rng = np.random.RandomState(seed + 12345)
        param_range = bounds[:, 1] - bounds[:, 0]

        X_local = []
        for _ in range(n_local):
            radius = rng.uniform(0.03, 0.15)
            X_local.append(_perturb(init_params, bounds, param_range, log_scale,
                                    radius, rng))
        X_local = np.array(X_local) if X_local else np.empty((0, bounds.shape[0]))

        blocks = [init_params.reshape(1, -1), X_local]
        if n_lhs > 0:
            blocks.append(latin_hypercube_sampling(bounds, n_lhs, seed=seed,
                                                   log_scale=log_scale))
        return np.vstack(blocks)

    @staticmethod
    def _build_surrogate(surrogate, evaluator, bounds, rl, gp_time_limit):
        """Instantiate the surrogate, plus a shadow NN when in auto mode."""
        args = (evaluator.spec_names, evaluator.constraints,
                evaluator.directions, bounds)
        if surrogate == 'nn':
            rl.log.info('[MODEL] Using NN ensemble surrogate')
            return NNSurrogate(*args), None, False

        auto_mode = (surrogate == 'auto')
        if auto_mode:
            rl.log.info('[MODEL] Using GP surrogate [auto: switches to NN if '
                        'GP fitting exceeds %.0fs]', gp_time_limit)
        else:
            rl.log.info('[MODEL] Using GP surrogate (BoTorch)')
        shadow = NNSurrogate(*args) if auto_mode else None
        return GPSurrogate(*args), shadow, auto_mode

    @staticmethod
    def _relax_constraints(evaluator, target_constraints, Y_np, t, n_init, budget):
        """Constraint set handed to the surrogate while no design is feasible.

        Specs whose best observation is still far from target are relaxed more,
        so the search does not farm the easy specs and ignore the hard ones.
        The relaxation decays to zero over the budget. Returns a new dict; the
        targets used to judge feasibility are never touched.
        """
        progress = (t - n_init) / max(budget - n_init, 1)
        base_relax = max(0.0, 1.0 - progress)
        relaxed = {}

        for name in evaluator.spec_names:
            orig = target_constraints[name]
            valid = Y_np[name][~np.isnan(Y_np[name])]
            if len(valid) == 0:
                spec_relax = 0.5 * base_relax
            else:
                if evaluator.directions[name] == 'min':
                    ratio = np.max(valid) / (abs(orig) + 1e-30)
                else:
                    best = np.min(valid[valid > 0]) if np.any(valid > 0) else valid.min()
                    ratio = (abs(orig) + 1e-30) / (abs(best) + 1e-30)
                if ratio >= 1.0:
                    spec_relax = 0.1 * base_relax
                else:
                    spec_relax = min(0.7, (1.0 - ratio) * 0.8) * base_relax

            if evaluator.directions[name] == 'min':
                relaxed[name] = orig * (1.0 - spec_relax)
            else:
                relaxed[name] = orig * (1.0 + spec_relax)

        return relaxed

    def _make_candidates(self, evaluator, X_data, Y_np, target_constraints,
                         bounds, log_scale, seed, t, n_init, budget, has_feasible):
        """Candidate pool: global LHS plus local perturbations around anchors.

        With a multi-region Mapper graph the anchors are the region centroids
        (Section 3.5); the per-anchor share follows 1/sqrt(1 + v_k) so
        under-visited regions get more of the local budget. The perturbation
        radius decays linearly over the budget, Eq. (10).
        """
        global_frac = 0.6 if has_feasible else 0.8
        n_global = int(N_CANDIDATES * global_frac)
        n_local = N_CANDIDATES - n_global
        X_global = latin_hypercube_sampling(bounds, n_global, seed=seed + t,
                                            log_scale=log_scale)
        if len(X_data) == 0:
            return np.vstack([X_global,
                              latin_hypercube_sampling(bounds, n_local,
                                                       seed=seed + t + 1,
                                                       log_scale=log_scale)])

        progress = (t - n_init) / max(budget - n_init, 1)
        radius = 0.2 * (1.0 - 0.7 * progress)
        param_range = bounds[:, 1] - bounds[:, 0]
        rng = np.random.RandomState(seed + t + 99999)

        multi_score = _multi_spec_score(evaluator, Y_np, target_constraints)
        tda_res = getattr(self, '_cached_tda_result', None)
        use_centroids = (getattr(self, 'use_centroid_guidance', False)
                         and tda_res is not None
                         and len(tda_res.components) > 1
                         and len(getattr(tda_res, 'centroids', [])) > 0)

        if use_centroids:
            anchors = [c for c in tda_res.centroids]
            anchors.append(X_data[np.argmax(multi_score)])
            weights = np.ones(len(anchors))
            for k in range(len(tda_res.components)):
                weights[k] = 1.0 / (1.0 + self._comp_visit_counts.get(k, 0)) ** 0.5
            weights /= weights.sum()
            n_per_anchor = np.maximum((weights * n_local).astype(int), 1)
            n_per_anchor[-1] = n_local - n_per_anchor[:-1].sum()
        else:
            top_idx = np.argsort(multi_score)[-min(5, X_data.shape[0]):]
            anchors = [X_data[i] for i in top_idx]
            n_per_anchor = np.full(len(anchors), max(1, n_local // len(anchors)))

        X_locals = []
        for i, anchor in enumerate(anchors):
            n_pts = int(n_per_anchor[i]) if i < len(n_per_anchor) else 1
            if n_pts <= 0:
                continue
            X_loc = anchor + rng.randn(n_pts, bounds.shape[0]) * param_range * radius
            X_locals.append(np.clip(X_loc, bounds[:, 0], bounds[:, 1]))

        X_local = np.vstack(X_locals)[:n_local] if X_locals else \
            latin_hypercube_sampling(bounds, n_local, seed=seed + t + 1,
                                     log_scale=log_scale)
        return np.vstack([X_global, X_local])

    @staticmethod
    def _restart_candidate(evaluator, X_data, Y_np, target_constraints, bounds,
                           log_scale, seed, t, stagnation_counter, rl):
        """Heuristic query used when the acquisition has stalled.

        Cycles through local refinement, crossover, a bottleneck-targeted move,
        and a return to the reference design.
        """
        rng = np.random.RandomState(seed + t + 77777)
        param_range = bounds[:, 1] - bounds[:, 0]
        score = _multi_spec_score(evaluator, Y_np, target_constraints, cap=1.5)
        mode = (stagnation_counter // _GPBasedOptimizer.RESTART_PATIENCE) % 4

        if mode == 0:
            best_idx = int(np.argmax(score))
            child = _perturb(X_data[best_idx], bounds, param_range, log_scale,
                             rng.uniform(0.03, 0.10), rng)
            rl.log.info(f'[ITER {t:>3d}] RESTART - local refinement around best')
        elif mode == 1:
            top = np.argsort(score)[-min(10, X_data.shape[0]):]
            a, b = rng.choice(top, size=2, replace=False)
            w = rng.uniform(0.2, 0.8)
            child = w * X_data[a] + (1 - w) * X_data[b]
            child += rng.randn(bounds.shape[0]) * param_range * 0.05
            rl.log.info(f'[ITER {t:>3d}] RESTART - crossover of top designs')
        elif mode == 2:
            bottleneck = _bottleneck_spec(evaluator, Y_np, target_constraints)
            vals = Y_np[bottleneck]
            if evaluator.directions[bottleneck] == 'min':
                idx = int(np.nanargmax(vals))
            else:
                filled = np.where(np.isnan(vals), np.inf, vals)
                idx = int(np.argmin(filled))
            child = X_data[idx] + rng.randn(bounds.shape[0]) * param_range * \
                rng.uniform(0.05, 0.15)
            rl.log.info(f'[ITER {t:>3d}] RESTART - targeting {bottleneck}')
        else:
            init_params = getattr(evaluator, 'init_params', None)
            if init_params is not None:
                child = init_params + rng.randn(bounds.shape[0]) * param_range * \
                    rng.uniform(0.05, 0.15)
                rl.log.info(f'[ITER {t:>3d}] RESTART - around reference design')
            else:
                child = latin_hypercube_sampling(bounds, 1, seed=seed + t + 88888,
                                                 log_scale=log_scale)[0]
                rl.log.info(f'[ITER {t:>3d}] RESTART - global LHS draw')

        return np.clip(child, bounds[:, 0], bounds[:, 1])

    def _select_next(self, model, X_cand, bounds, t, T, best_f, observed_Y=None):
        raise NotImplementedError


def _perturb(center, bounds, param_range, log_scale, radius, rng):
    """Gaussian step around center, taken in log space for wide-range params."""
    if log_scale is None:
        return np.clip(center + rng.randn(bounds.shape[0]) * param_range * radius,
                       bounds[:, 0], bounds[:, 1])
    out = np.zeros(bounds.shape[0])
    for i in range(bounds.shape[0]):
        if log_scale[i] and bounds[i, 0] > 0:
            log_range = np.log10(bounds[i, 1]) - np.log10(bounds[i, 0])
            out[i] = 10 ** (np.log10(center[i]) + rng.randn() * log_range * radius)
        else:
            out[i] = center[i] + rng.randn() * param_range[i] * radius
    return np.clip(out, bounds[:, 0], bounds[:, 1])


def _multi_spec_score(evaluator, Y_np, constraints, cap=1.0):
    """Per-observation count of how close each spec is to its target."""
    n = len(Y_np[evaluator.spec_names[0]])
    score = np.zeros(n)
    for name in evaluator.spec_names:
        vals = Y_np[name]
        c = constraints[name]
        if evaluator.directions[name] == 'min':
            score += np.clip(vals / (abs(c) + 1e-30), 0, cap)
        else:
            score += np.clip((abs(c) + 1e-30) / np.maximum(vals, 1e-30), 0, cap)
    return score


def _bottleneck_spec(evaluator, Y_np, constraints):
    """Spec whose best observation sits furthest from its target."""
    ratios = {}
    for name in evaluator.spec_names:
        valid = Y_np[name][~np.isnan(Y_np[name])]
        c = constraints[name]
        if len(valid) == 0:
            ratios[name] = 0.0
        elif evaluator.directions[name] == 'min':
            ratios[name] = np.max(valid) / (abs(c) + 1e-30)
        else:
            best = np.min(valid[valid > 0]) if np.any(valid > 0) else abs(c) * 10
            ratios[name] = (abs(c) + 1e-30) / (best + 1e-30)
    return min(ratios, key=ratios.get)


def _select_near_feasible(h, X_cand, min_frac=0.10, max_frac=0.20):
    """Near-feasible set F_hat: the top-k candidates by h, Eq. (5).

    k tracks the number of predicted-feasible candidates but is clamped to
    [min_frac, max_frac] of the pool, so F_hat stays dense enough for Mapper
    even before any candidate reaches h > 0.
    """
    n = len(X_cand)
    min_k = max(20, int(min_frac * n))
    max_k = max(min_k, int(max_frac * n))
    k = np.clip((h > 0).sum(), min_k, max_k)

    mask = np.zeros(n, dtype=bool)
    mask[np.argpartition(h, -k)[-k:]] = True
    return X_cand[mask], h[mask], mask


class TopoBO(_GPBasedOptimizer):
    """ATLAS: topology-aware BO.

    Each refresh builds a Mapper graph over the surrogate-predicted feasible
    set, classifies candidates as bridge / frontier / interior, and scores them
    with the TAL acquisition of Eq. (9),

        alpha_TAL(x) = d_k(x) * alpha_wEI(x) + lambda_t * delta_beta0(x) * p(x).
    """

    TDA_REFIT_INTERVAL = 3
    STABILITY_WINDOW = 10

    def __init__(self, lambda_0=None, gamma=1.0,
                 mapper_n_cubes=10, mapper_overlap=0.3, mapper_min_cluster=3,
                 mapper_dbscan_eps=0.3, mapper_eps_multiplier=1.0,
                 top_k_frac=None, use_component_discount=True,
                 use_centroid_guidance=True, use_lambda_decay=True):
        super().__init__('ATLAS')
        self.lambda_0 = lambda_0
        self.gamma = gamma
        self.top_k_frac = top_k_frac
        self.use_component_discount = use_component_discount
        self.use_centroid_guidance = use_centroid_guidance
        self.use_lambda_decay = use_lambda_decay

        self.topo_analyzer = MapperAnalyzer(
            n_cubes=mapper_n_cubes,
            overlap=mapper_overlap,
            min_cluster_size=mapper_min_cluster,
            dbscan_eps=mapper_dbscan_eps,
            eps_multiplier=mapper_eps_multiplier,
        )

        self._cached_tda_result = None
        self._cached_beta0 = 0
        self._comp_visit_counts = {}
        self._beta0_history = []

    def _select_next(self, model, X_cand, bounds, t, T, best_f, observed_Y=None):
        h = model.compute_h(X_cand)

        if self.top_k_frac is None:
            F_hat, h_F_hat, _ = _select_near_feasible(h, X_cand)
        else:
            half = self.top_k_frac / 2
            F_hat, h_F_hat, _ = _select_near_feasible(
                h, X_cand, min_frac=self.top_k_frac - half,
                max_frac=self.top_k_frac + half)

        delta_beta0 = np.zeros(len(X_cand))
        tda_result = None
        beta0 = 0
        tda_time = 0.0

        if len(F_hat) >= 6:
            F_hat_norm = normalize_to_unit(F_hat, bounds)
            X_cand_norm = normalize_to_unit(X_cand, bounds)

            if t % self.TDA_REFIT_INTERVAL == 0 or self._cached_tda_result is None:
                t0 = time.time()
                tda_result = self.topo_analyzer.analyze(
                    F_hat_norm, h_F_hat, X_original=F_hat)
                if tda_result is not None:
                    if tda_result.beta0 != self._cached_beta0:
                        self._comp_visit_counts.clear()
                    self._cached_tda_result = tda_result
                    self._cached_beta0 = tda_result.beta0
                tda_time = time.time() - t0

            tda_result = self._cached_tda_result
            if tda_result is not None:
                delta_beta0 = self.topo_analyzer.classify_candidates(
                    X_cand_norm, tda_result, F_hat_norm)
                beta0 = tda_result.beta0
        else:
            delta_beta0 = np.ones(len(X_cand))

        wei = compute_wei(model, X_cand, best_f, observed_Y=observed_Y)

        if self.lambda_0 is None:
            positive = wei[wei > 0]
            self.lambda_0 = max(np.median(positive) if len(positive) else 1.0, 1e-6)

        comp_assignment = self._assign_regions(X_cand, F_hat, bounds, tda_result)
        if comp_assignment is not None and self.use_component_discount:
            visits = np.array([self._comp_visit_counts.get(int(c), 0)
                               for c in comp_assignment])
            wei = wei / np.sqrt(1.0 + visits)

        lambda_t = self._schedule_lambda(t, T, beta0)

        tal = wei + lambda_t * delta_beta0 * model.compute_p(X_cand)
        best_idx = int(np.argmax(tal))

        if comp_assignment is not None:
            k = int(comp_assignment[best_idx])
            self._comp_visit_counts[k] = self._comp_visit_counts.get(k, 0) + 1

        return {
            'x_next': X_cand[best_idx],
            'tda_time': tda_time,
            'beta0': beta0,
            'lambda_t': lambda_t,
            'F_hat': F_hat,
        }

    @staticmethod
    def _assign_regions(X_cand, F_hat, bounds, tda_result):
        """Nearest Mapper region per candidate, or None when K <= 1, Eq. (7)."""
        if tda_result is None or len(tda_result.components) <= 1 or len(F_hat) == 0:
            return None
        from scipy.spatial.distance import cdist

        X_norm = normalize_to_unit(X_cand, bounds)
        F_norm = normalize_to_unit(F_hat, bounds)
        K = len(tda_result.components)
        dists = np.full((len(X_cand), K), np.inf)
        for k, idx in enumerate(tda_result.components):
            if len(idx) == 0:
                continue
            dists[:, k] = cdist(X_norm, F_norm[idx]).min(axis=1)
        return np.argmin(dists, axis=1)

    def _schedule_lambda(self, t, T, beta0):
        """Hold-then-decay lambda, Eq. (8), modulated by beta0 stability."""
        self._beta0_history.append(beta0)
        lambda_t = compute_lambda_t(t, T, self.lambda_0, self.gamma) \
            if self.use_lambda_decay else self.lambda_0

        if len(self._beta0_history) >= self.STABILITY_WINDOW:
            recent = self._beta0_history[-self.STABILITY_WINDOW:]
            if len(set(recent)) == 1:
                lambda_t *= 0.5
            elif max(recent) - min(recent) >= 2:
                lambda_t = min(lambda_t * 1.5, self.lambda_0)
        return lambda_t


class WEIBO(BaseOptimizer):
    """WEIBO: Vanilla Weighted Expected Improvement BO baseline.

    Faithful to the original WEI formulation: acquisition = EI(x) * PoF(x).
    No enhancements: plain LHS candidates, fixed GP surrogate, no curriculum,
    no stagnation restart, no per-spec anchors.
    """

    def __init__(self):
        super().__init__('WEIBO')

    def optimize(self, evaluator, budget, n_init=20, seed=42,
                 log_root='logs', surrogate='gp',
                 early_stop=False, **kwargs):
        # Vanilla WEIBO always uses GP regardless of caller's surrogate arg
        surrogate = 'gp'
        np.random.seed(seed)

        rl = RunLogger(
            circuit_name=evaluator.yaml_path.split('/')[-1].replace('.yaml', ''),
            method_name=self.name,
            budget=budget, seed=seed, n_init=n_init,
            spec_names=evaluator.spec_names,
            constraints=evaluator.constraints,
            directions=evaluator.directions,
            param_names=evaluator.param_names,
            log_root=log_root,
        )
        rl.log_start()

        mem_logger = ExperimentLogger(
            evaluator.spec_names, evaluator.constraints,
            evaluator.directions, evaluator.circuit_type,
        )
        bounds = evaluator.param_bounds
        log_scale = getattr(evaluator, 'log_scale', None)

        # Initial design: plain LHS.
        X_init = latin_hypercube_sampling(bounds, n_init, seed=seed, log_scale=log_scale)
        X_data = []
        Y_data = {name: [] for name in evaluator.spec_names}
        fom_data = []

        t_init_start = time.time()
        for i in range(n_init):
            t0 = time.time()
            specs = evaluator.evaluate(X_init[i])
            sim_time = time.time() - t0
            mem_logger.log(X_init[i], specs)
            feas = check_feasibility(specs, evaluator.constraints, evaluator.directions)
            rl.log_init_sample(i, specs, feas, sim_time)
            X_data.append(X_init[i])
            for name in evaluator.spec_names:
                Y_data[name].append(specs.get(name, np.nan))
            fom = compute_fom(specs, evaluator.circuit_type,
                              evaluator.constraints, evaluator.directions) if feas else np.nan
            fom_data.append(fom)
            if early_stop and feas:
                rl.log.info(f'[EARLY STOP] Feasible found during init at sample {i}')
                results = rl.log_finish()
                rl.close()
                return mem_logger, results

        X_data = np.array(X_data)
        Y_np = {name: np.array(vals) for name, vals in Y_data.items()}
        fom_np = np.array(fom_data)
        rl.log_init_summary(time.time() - t_init_start)

        # BO loop: EI weighted by probability of feasibility.
        if surrogate == 'nn':
            model = NNSurrogate(evaluator.spec_names, evaluator.constraints,
                                evaluator.directions, bounds)
        else:
            model = GPSurrogate(evaluator.spec_names, evaluator.constraints,
                                evaluator.directions, bounds)

        for t in range(n_init, budget):
            sys.stdout.flush()
            t0 = time.time()
            fom_valid = fom_np.copy()
            fom_valid[np.isnan(fom_valid)] = -np.inf
            has_feasible = np.any(np.isfinite(fom_valid) & (fom_valid > -np.inf))
            model.fit(X_data, Y_np, fom_values=fom_np if has_feasible else None)
            gp_time = time.time() - t0

            # Pure LHS candidates (no local perturbation anchors)
            X_cand = latin_hypercube_sampling(bounds, N_CANDIDATES, seed=seed + t,
                                              log_scale=log_scale)

            t0 = time.time()
            best_f = np.nanmax(fom_np) if has_feasible else -np.inf
            # Vanilla WEI: EI(x) * PoF(x)
            # When no feasible point exists, fom_gp is None so EI=0.
            # Fall back to pure PoF for feasibility search (Lyu et al. 2018).
            p = model.compute_p(X_cand)
            if has_feasible:
                ei = model.compute_ei(X_cand, best_f)
                wei = ei * p
            else:
                wei = p
            best_idx = np.argmax(wei)
            x_next = X_cand[best_idx]
            acq_time = time.time() - t0

            t0 = time.time()
            specs = evaluator.evaluate(x_next)
            sim_time = time.time() - t0
            mem_logger.log(x_next, specs)
            feas = check_feasibility(specs, evaluator.constraints, evaluator.directions)
            fom = compute_fom(specs, evaluator.circuit_type,
                              evaluator.constraints, evaluator.directions) if feas else np.nan
            rl.log_iteration(t, specs, feas, fom,
                             sim_time=sim_time, gp_time=gp_time,
                             tda_time=0, acq_time=acq_time)
            if early_stop and feas:
                rl.log.info(f'[EARLY STOP] Feasible found at iter {t}.')
                break
            X_data = np.vstack([X_data, x_next.reshape(1, -1)])
            for name in evaluator.spec_names:
                Y_np[name] = np.append(Y_np[name], specs.get(name, np.nan))
            fom_np = np.append(fom_np, fom)

        results = rl.log_finish()
        rl.close()
        return mem_logger, results

class TuRBOptimizer(BaseOptimizer):
    """TuRBO-1: Trust Region Bayesian Optimization baseline.

    Follows the official BoTorch TuRBO tutorial implementation:
    - GP trained on all data within the current TR epoch
    - TR center at global best of current epoch
    - Lengthscale-weighted TR bounds (normalized by mean, then geometric mean)
    - Thompson sampling with perturbation mask
    - On TR collapse, restart with new LHS points (old data discarded)
    """

    def __init__(self, n_trust_regions=1):
        super().__init__('TuRBO')
        self.n_trust_regions = n_trust_regions

    def optimize(self, evaluator, budget, n_init=20, seed=42,
                 log_root='logs', surrogate='nn',
                 early_stop=False, gp_time_limit=10.0):
        import math
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.generation import MaxPosteriorSampling
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.constraints import Interval

        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float64

        rl = RunLogger(
            circuit_name=evaluator.yaml_path.split('/')[-1].replace('.yaml', ''),
            method_name=self.name,
            budget=budget, seed=seed, n_init=n_init,
            spec_names=evaluator.spec_names,
            constraints=evaluator.constraints,
            directions=evaluator.directions,
            param_names=evaluator.param_names,
            log_root=log_root,
        )
        rl.log_start()

        mem_logger = ExperimentLogger(
            evaluator.spec_names, evaluator.constraints,
            evaluator.directions, evaluator.circuit_type,
        )
        bounds = evaluator.param_bounds
        log_scale = getattr(evaluator, 'log_scale', None)
        n_dims = bounds.shape[0]

        # TuRBO-1 settings, batch size 1.
        batch_size = 1
        length_init = 0.8
        length_min = 0.5 ** 7
        length_max = 1.6
        success_tol = 10  # per tutorial
        failure_tol = math.ceil(max(4.0 / batch_size, float(n_dims) / batch_size))

        lb = bounds[:, 0]
        ub = bounds[:, 1]

        # Normalization helpers, log-aware for wide-range parameters.
        def _to_unit(X_raw):
            """Normalize raw params to [0,1]^d, using log for log-scale dims."""
            X_out = np.empty_like(X_raw)
            for d_i in range(n_dims):
                if log_scale is not None and log_scale[d_i] and lb[d_i] > 0:
                    log_lo, log_hi = np.log10(lb[d_i]), np.log10(ub[d_i])
                    X_out[..., d_i] = (np.log10(X_raw[..., d_i]) - log_lo) / (log_hi - log_lo)
                else:
                    X_out[..., d_i] = (X_raw[..., d_i] - lb[d_i]) / (ub[d_i] - lb[d_i])
            return X_out

        def _from_unit(X_unit):
            """Unnormalize [0,1]^d back to raw params."""
            X_out = np.empty_like(X_unit)
            for d_i in range(n_dims):
                if log_scale is not None and log_scale[d_i] and lb[d_i] > 0:
                    log_lo, log_hi = np.log10(lb[d_i]), np.log10(ub[d_i])
                    X_out[..., d_i] = 10 ** (X_unit[..., d_i] * (log_hi - log_lo) + log_lo)
                else:
                    X_out[..., d_i] = X_unit[..., d_i] * (ub[d_i] - lb[d_i]) + lb[d_i]
            return X_out

        # Objective: clamped sum of feasibility margins, higher is better.
        def _compute_obj(specs_dict):
            total = 0.0
            for name in evaluator.spec_names:
                val = specs_dict.get(name, np.nan)
                c = evaluator.constraints[name]
                if np.isnan(val):
                    total -= 1.0
                    continue
                if evaluator.directions[name] == 'min':
                    margin = (val - c) / (abs(c) + 1e-6)
                else:
                    margin = (c - val) / (abs(c) + 1e-6)
                total += max(-1.0, min(1.0, margin))
            return total

        # Evaluate one design and record it.
        def _eval_and_log(x_raw, iter_idx):
            """Evaluate x_raw, log results, return (specs, feas, fom, obj_val, sim_time)."""
            t0 = time.time()
            specs = evaluator.evaluate(x_raw)
            sim_time = time.time() - t0
            mem_logger.log(x_raw, specs)
            feas = check_feasibility(specs, evaluator.constraints, evaluator.directions)
            fom = compute_fom(specs, evaluator.circuit_type,
                              evaluator.constraints, evaluator.directions) if feas else np.nan
            obj_val = _compute_obj(specs)
            return specs, feas, fom, obj_val, sim_time

        # Early-stop check.
        def _check_early_stop(x_raw, feas, iter_idx):
            """Return True if early stop triggered."""
            if not (early_stop and feas):
                return False
            rl.log.info(f'[EARLY STOP] Feasible found at iter {iter_idx}')
            return True

        # Multi-restart trust-region loop.
        # All data across all restarts (for logging / mem_logger)
        X_all = np.empty((0, n_dims))
        fom_all = np.array([])
        Y_all = {name: np.array([]) for name in evaluator.spec_names}
        obj_all = np.array([])

        t = 0  # global evaluation counter

        while t < budget:
            # Start a trust-region epoch.
            early_found = False
            length = length_init
            n_success = 0
            n_failure = 0

            n_pts = min(n_init, budget - t)
            if n_pts <= 0:
                break
            X_epoch_raw = latin_hypercube_sampling(
                bounds, n_pts, seed=seed + t, log_scale=log_scale)

            # Collect epoch init data
            X_epoch = []
            obj_epoch = []
            is_init_phase = (t == 0)

            t_init_start = time.time()
            early_found = False
            for i in range(n_pts):
                if t >= budget:
                    break
                t0 = time.time()
                specs, feas, fom, obj_val, sim_time = _eval_and_log(X_epoch_raw[i], t)

                if is_init_phase:
                    rl.log_init_sample(i, specs, feas, sim_time)
                else:
                    rl.log_iteration(t, specs, feas, fom,
                                     sim_time=sim_time, gp_time=0, tda_time=0, acq_time=0)

                X_epoch.append(X_epoch_raw[i])
                obj_epoch.append(obj_val)
                X_all = np.vstack([X_all, X_epoch_raw[i].reshape(1, -1)])
                for name in evaluator.spec_names:
                    Y_all[name] = np.append(Y_all[name], specs.get(name, np.nan))
                fom_all = np.append(fom_all, fom)
                obj_all = np.append(obj_all, obj_val)
                t += 1

                if _check_early_stop(X_epoch_raw[i], feas, t - 1):
                    early_found = True
                    break

            if is_init_phase:
                rl.log_init_summary(time.time() - t_init_start)

            if early_found:
                break

            X_epoch = np.array(X_epoch)
            obj_epoch = np.array(obj_epoch)
            best_obj = float(np.max(obj_epoch))

            # Inner loop for this trust region.
            restart_triggered = False
            while t < budget and not restart_triggered:
                sys.stdout.flush()

                # Normalize epoch data to [0,1]^d
                X_epoch_norm = _to_unit(X_epoch)
                obj_epoch_t = torch.tensor(obj_epoch, dtype=dtype, device=device).unsqueeze(-1)

                # TR center = best point in this epoch
                x_center = torch.tensor(
                    X_epoch_norm[np.argmax(obj_epoch)], dtype=dtype, device=device)

                # Fit the GP with manual standardization and a noise floor.
                t0 = time.time()
                try:
                    train_X = torch.tensor(X_epoch_norm, dtype=dtype, device=device)
                    train_Y = (obj_epoch_t - obj_epoch_t.mean()) / (obj_epoch_t.std() + 1e-8)

                    likelihood = GaussianLikelihood(
                        noise_constraint=Interval(1e-8, 1e-3)).to(device)
                    covar_module = ScaleKernel(
                        MaternKernel(nu=2.5, ard_num_dims=n_dims,
                                     lengthscale_constraint=Interval(0.005, 4.0)),
                    ).to(device)
                    gp = SingleTaskGP(
                        train_X, train_Y,
                        covar_module=covar_module,
                        likelihood=likelihood,
                    ).to(device)
                    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                    fit_gpytorch_mll(mll)
                except Exception as e:
                    rl.log.info(f'[TuRBO] GP fit failed: {e}, using random sample')
                    # Fallback: uniform random within TR (no lengthscale scaling)
                    tr_lb = np.clip(x_center.cpu().numpy() - length / 2.0, 0.0, 1.0)
                    tr_ub = np.clip(x_center.cpu().numpy() + length / 2.0, 0.0, 1.0)
                    rng = np.random.RandomState(seed + t)
                    x_next_norm = rng.uniform(tr_lb, tr_ub)
                    x_next = _from_unit(x_next_norm)
                    x_next = np.clip(x_next, lb, ub)

                    specs, feas, fom, obj_val, sim_time = _eval_and_log(x_next, t)
                    rl.log_iteration(t, specs, feas, fom,
                                     sim_time=sim_time, gp_time=time.time() - t0,
                                     tda_time=0, acq_time=0)

                    X_epoch = np.vstack([X_epoch, x_next.reshape(1, -1)])
                    obj_epoch = np.append(obj_epoch, obj_val)
                    X_all = np.vstack([X_all, x_next.reshape(1, -1)])
                    for name in evaluator.spec_names:
                        Y_all[name] = np.append(Y_all[name], specs.get(name, np.nan))
                    fom_all = np.append(fom_all, fom)
                    obj_all = np.append(obj_all, obj_val)
                    t += 1

                    # Update TR state (per tutorial)
                    if obj_val > best_obj + 1e-3 * math.fabs(best_obj):
                        n_success += 1
                        n_failure = 0
                    else:
                        n_success = 0
                        n_failure += 1
                    if n_success == success_tol:
                        length = min(2.0 * length, length_max)
                        n_success = 0
                    elif n_failure == failure_tol:
                        length /= 2.0
                        n_failure = 0
                    best_obj = max(best_obj, obj_val)
                    if length < length_min:
                        restart_triggered = True
                        rl.log.info(f'[TuRBO] TR restart (collapsed below {length_min:.4f})')
                    if _check_early_stop(x_next, feas, t - 1):
                        early_found = True
                        break
                    continue

                gp_time = time.time() - t0

                # Trust-region bounds weighted by the fitted lengthscales.
                weights = gp.covar_module.base_kernel.lengthscale.squeeze().detach()
                weights = weights / weights.mean()
                weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
                tr_lb = torch.clamp(x_center - weights * length / 2.0, 0.0, 1.0)
                tr_ub = torch.clamp(x_center + weights * length / 2.0, 0.0, 1.0)

                # Thompson sampling over a perturbation mask.
                t0 = time.time()
                n_cands = min(5000, max(2000, 200 * n_dims))
                sobol = torch.quasirandom.SobolEngine(n_dims, scramble=True, seed=seed + t)
                pert = sobol.draw(n_cands).to(dtype=dtype, device=device)
                pert = tr_lb + (tr_ub - tr_lb) * pert

                # Perturbation mask: each dim perturbed with prob min(20/d, 1)
                prob_perturb = min(20.0 / n_dims, 1.0)
                mask = torch.rand(n_cands, n_dims, dtype=dtype, device=device) <= prob_perturb
                # Ensure at least one dim is perturbed per candidate
                ind = torch.where(mask.sum(dim=1) == 0)[0]
                if len(ind) > 0:
                    mask[ind, torch.randint(0, n_dims, size=(len(ind),), device=device)] = 1

                X_cand = x_center.expand(n_cands, n_dims).clone()
                X_cand[mask] = pert[mask]

                thompson = MaxPosteriorSampling(model=gp, replacement=False)
                with torch.no_grad():
                    x_next_torch = thompson(X_cand, num_samples=1)
                x_next_norm = x_next_torch.squeeze(0).cpu().numpy()
                x_next = _from_unit(x_next_norm)
                x_next = np.clip(x_next, lb, ub)
                acq_time = time.time() - t0

                # Evaluate.
                specs, feas, fom, obj_val, sim_time = _eval_and_log(x_next, t)
                rl.log_iteration(t, specs, feas, fom,
                                 sim_time=sim_time, gp_time=gp_time,
                                 tda_time=0, acq_time=acq_time)

                X_epoch = np.vstack([X_epoch, x_next.reshape(1, -1)])
                obj_epoch = np.append(obj_epoch, obj_val)
                X_all = np.vstack([X_all, x_next.reshape(1, -1)])
                for name in evaluator.spec_names:
                    Y_all[name] = np.append(Y_all[name], specs.get(name, np.nan))
                fom_all = np.append(fom_all, fom)
                obj_all = np.append(obj_all, obj_val)
                t += 1

                # Update the trust-region state.
                if obj_val > best_obj + 1e-3 * math.fabs(best_obj):
                    n_success += 1
                    n_failure = 0
                else:
                    n_success = 0
                    n_failure += 1

                if n_success == success_tol:
                    length = min(2.0 * length, length_max)
                    n_success = 0
                    rl.log.info(f'[TuRBO] TR expand to length={length:.4f}')
                elif n_failure == failure_tol:
                    length /= 2.0
                    n_failure = 0
                    rl.log.info(f'[TuRBO] TR shrink to length={length:.4f}')

                best_obj = max(best_obj, obj_val)

                if length < length_min:
                    restart_triggered = True
                    rl.log.info(f'[TuRBO] TR restart (collapsed below {length_min:.4f})')

                if _check_early_stop(x_next, feas, t - 1):
                    early_found = True
                    break

            if early_found:
                break

        results = rl.log_finish()
        rl.close()
        return mem_logger, results


class _DEStopEarly(Exception):
    """Raised to force scipy DE / CMA-ES to stop when budget exhausted or feasible found."""
    pass

class RandomSearchOpt(BaseOptimizer):
    """Pure random search baseline (no surrogate)."""

    def __init__(self):
        super().__init__('Random')

    def optimize(self, evaluator, budget, n_init=20, seed=42,
                 log_root='logs', surrogate='nn',
                 early_stop=False, **kwargs):
        np.random.seed(seed)

        rl = RunLogger(
            circuit_name=evaluator.yaml_path.split('/')[-1].replace('.yaml', ''),
            method_name=self.name,
            budget=budget, seed=seed, n_init=0,
            spec_names=evaluator.spec_names,
            constraints=evaluator.constraints,
            directions=evaluator.directions,
            param_names=evaluator.param_names,
            log_root=log_root,
        )
        rl.log_start()

        mem_logger = ExperimentLogger(
            evaluator.spec_names, evaluator.constraints,
            evaluator.directions, evaluator.circuit_type,
        )

        bounds_lo = evaluator.param_bounds[:, 0]
        bounds_hi = evaluator.param_bounds[:, 1]

        for i in range(budget):
            x = np.random.uniform(bounds_lo, bounds_hi)

            t0 = time.time()
            specs = evaluator.evaluate(x)
            sim_time = time.time() - t0
            mem_logger.log(x, specs)

            feas = check_feasibility(specs, evaluator.constraints, evaluator.directions)
            fom = compute_fom(specs, evaluator.circuit_type,
                              evaluator.constraints, evaluator.directions) if feas else np.nan

            rl.log_iteration(i, specs, feas, fom,
                             sim_time=sim_time, gp_time=0, tda_time=0, acq_time=0)

            if early_stop and feas:
                break

        results = rl.log_finish()
        rl.close()
        return mem_logger, results

class CMAESOpt(BaseOptimizer):
    """CMA-ES baseline using pycma."""

    def __init__(self, popsize=None):
        super().__init__('CMAES')
        self._popsize = popsize  # None = let cma choose default

    def optimize(self, evaluator, budget, n_init=20, seed=42,
                 log_root='logs', surrogate='nn',
                 early_stop=False, **kwargs):
        import cma

        np.random.seed(seed)

        rl = RunLogger(
            circuit_name=evaluator.yaml_path.split('/')[-1].replace('.yaml', ''),
            method_name=self.name,
            budget=budget, seed=seed, n_init=0,
            spec_names=evaluator.spec_names,
            constraints=evaluator.constraints,
            directions=evaluator.directions,
            param_names=evaluator.param_names,
            log_root=log_root,
        )
        rl.log_start()

        mem_logger = ExperimentLogger(
            evaluator.spec_names, evaluator.constraints,
            evaluator.directions, evaluator.circuit_type,
        )

        bounds_lo = evaluator.param_bounds[:, 0]
        bounds_hi = evaluator.param_bounds[:, 1]

        # Initial point: center of bounds
        x0 = 0.5 * (bounds_lo + bounds_hi)
        sigma0 = 0.3 * np.max(bounds_hi - bounds_lo)

        eval_count = [0]
        found_feasible = [False]

        def objective(x):
            if eval_count[0] >= budget:
                raise _DEStopEarly()

            x_clipped = np.clip(x, bounds_lo, bounds_hi)

            t0 = time.time()
            specs = evaluator.evaluate(x_clipped)
            sim_time = time.time() - t0
            mem_logger.log(x_clipped, specs)

            feas = check_feasibility(specs, evaluator.constraints, evaluator.directions)
            fom = compute_fom(specs, evaluator.circuit_type,
                              evaluator.constraints, evaluator.directions) if feas else np.nan

            rl.log_iteration(eval_count[0], specs, feas, fom,
                             sim_time=sim_time, gp_time=0, tda_time=0, acq_time=0)
            eval_count[0] += 1

            if early_stop and feas:
                found_feasible[0] = True
                raise _DEStopEarly()

            # Normalized constraint violation as cost
            cost = 0.0
            for name in evaluator.spec_names:
                val = specs.get(name, np.nan)
                c = evaluator.constraints[name]
                if np.isnan(val):
                    cost += 10.0
                    continue
                if evaluator.directions[name] == 'min':
                    cost += max(0, c - val) / (abs(c) + 1e-6)
                else:
                    cost += max(0, val - c) / (abs(c) + 1e-6)

            if cost < 1e-6:
                fom_val = compute_fom(specs, evaluator.circuit_type,
                                      evaluator.constraints, evaluator.directions)
                cost = -fom_val * 1e-6
            return cost

        opts = {
            'seed': seed,
            'maxfevals': budget,
            'bounds': [bounds_lo.tolist(), bounds_hi.tolist()],
            'verbose': -9,  # silent
            'tolfun': 0,
            'tolx': 0,
        }
        if self._popsize is not None:
            opts['popsize'] = self._popsize

        try:
            cma.fmin2(objective, x0, sigma0, options=opts)
        except _DEStopEarly:
            pass

        results = rl.log_finish()
        rl.close()
        return mem_logger, results

class MACEOpt(BaseOptimizer):
    """MACE: Multi-Acquisition Constrained Ensemble baseline.

    Combines EI, PI, and UCB acquisition functions in a weighted ensemble,
    multiplied by feasibility probability. Based on the MACE framework
    (Zhang et al., TCAS-I 2022) adapted to sequential (batch-size=1) setting.
    """

    def __init__(self, alpha=None, batch_size=10):
        super().__init__('MACE')
        self.alpha = alpha or [0.4, 0.3, 0.3]
        self.batch_size = batch_size

    def optimize(self, evaluator, budget, n_init=20, seed=42,
                 log_root='logs', surrogate='gp',
                 early_stop=False, **kwargs):
        np.random.seed(seed)

        rl = RunLogger(
            circuit_name=evaluator.yaml_path.split('/')[-1].replace('.yaml', ''),
            method_name=self.name,
            budget=budget, seed=seed, n_init=n_init,
            spec_names=evaluator.spec_names,
            constraints=evaluator.constraints,
            directions=evaluator.directions,
            param_names=evaluator.param_names,
            log_root=log_root,
        )
        rl.log_start()

        mem_logger = ExperimentLogger(
            evaluator.spec_names, evaluator.constraints,
            evaluator.directions, evaluator.circuit_type,
        )
        bounds = evaluator.param_bounds
        log_scale = getattr(evaluator, 'log_scale', None)

        # Initial design: LHS.
        X_init = latin_hypercube_sampling(bounds, n_init, seed=seed, log_scale=log_scale)
        X_data = []
        Y_data = {name: [] for name in evaluator.spec_names}
        fom_data = []

        t_init_start = time.time()
        for i in range(n_init):
            t0 = time.time()
            specs = evaluator.evaluate(X_init[i])
            sim_time = time.time() - t0
            mem_logger.log(X_init[i], specs)
            feas = check_feasibility(specs, evaluator.constraints, evaluator.directions)
            rl.log_init_sample(i, specs, feas, sim_time)
            X_data.append(X_init[i])
            for name in evaluator.spec_names:
                Y_data[name].append(specs.get(name, np.nan))
            fom = compute_fom(specs, evaluator.circuit_type,
                              evaluator.constraints, evaluator.directions) if feas else np.nan
            fom_data.append(fom)
            if early_stop and feas:
                rl.log.info(f'[EARLY STOP] Feasible found during init at sample {i}')
                results = rl.log_finish()
                rl.close()
                return mem_logger, results

        X_data = np.array(X_data)
        Y_np = {name: np.array(vals) for name, vals in Y_data.items()}
        fom_np = np.array(fom_data)
        rl.log_init_summary(time.time() - t_init_start)

        # BO loop over the MACE acquisition ensemble.
        model = GPSurrogate(evaluator.spec_names, evaluator.constraints,
                            evaluator.directions, bounds)
        q = self.batch_size

        t = n_init
        while t < budget:
            sys.stdout.flush()
            batch_q = min(q, budget - t)

            # Fit surrogate once per batch
            t0 = time.time()
            fom_valid = fom_np.copy()
            fom_valid[np.isnan(fom_valid)] = -np.inf
            has_feasible = np.any(np.isfinite(fom_valid) & (fom_valid > -np.inf))
            model.fit(X_data, Y_np, fom_values=fom_np if has_feasible else None)
            gp_time = time.time() - t0

            X_cand = latin_hypercube_sampling(bounds, N_CANDIDATES, seed=seed + t,
                                              log_scale=log_scale)

            t0 = time.time()
            best_f = np.nanmax(fom_np) if has_feasible else -np.inf
            p = model.compute_p(X_cand)

            if has_feasible:
                beta_t = 2.0 + np.sqrt(np.log(2.0 * (t + 1)))
                acq = compute_mace(model, X_cand, best_f,
                                   alpha=self.alpha, beta_t=beta_t) * p
            else:
                acq = p

            # Select top-q candidates
            top_indices = np.argsort(acq)[-batch_q:][::-1]
            acq_time = time.time() - t0

            # Evaluate batch
            stopped = False
            for i, idx in enumerate(top_indices):
                x_next = X_cand[idx]
                t0 = time.time()
                specs = evaluator.evaluate(x_next)
                sim_time = time.time() - t0
                mem_logger.log(x_next, specs)
                feas = check_feasibility(specs, evaluator.constraints, evaluator.directions)
                fom = compute_fom(specs, evaluator.circuit_type,
                                  evaluator.constraints, evaluator.directions) if feas else np.nan

                iter_gp = gp_time if i == 0 else 0
                iter_acq = acq_time if i == 0 else 0
                rl.log_iteration(t, specs, feas, fom,
                                 sim_time=sim_time, gp_time=iter_gp,
                                 tda_time=0, acq_time=iter_acq)

                X_data = np.vstack([X_data, x_next.reshape(1, -1)])
                for name in evaluator.spec_names:
                    Y_np[name] = np.append(Y_np[name], specs.get(name, np.nan))
                fom_np = np.append(fom_np, fom)
                t += 1

                if early_stop and feas:
                    rl.log.info(f'[EARLY STOP] Feasible found at iter {t - 1}.')
                    stopped = True
                    break
            if stopped:
                break

        results = rl.log_finish()
        rl.close()
        return mem_logger, results


class SCBOOpt(BaseOptimizer):
    """SCBO: Scalable Constrained Bayesian Optimization baseline.

    Faithful implementation of Eriksson & Poloczek (NeurIPS 2021).
    Key differences from TuRBO:
    - Independent GP per constraint + objective GP (not a single merged objective)
    - Thompson sampling weighted by feasibility probability from constraint GPs
    - TR center = best feasible point (not best objective)
    - Lengthscale-weighted TR bounds (same as TuRBO)
    - Multi-restart epochs on TR collapse
    """

    def __init__(self):
        super().__init__('SCBO')

    def optimize(self, evaluator, budget, n_init=20, seed=42,
                 log_root='logs', surrogate='gp',
                 early_stop=False, **kwargs):
        import math
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.constraints import Interval

        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float64

        rl = RunLogger(
            circuit_name=evaluator.yaml_path.split('/')[-1].replace('.yaml', ''),
            method_name=self.name,
            budget=budget, seed=seed, n_init=n_init,
            spec_names=evaluator.spec_names,
            constraints=evaluator.constraints,
            directions=evaluator.directions,
            param_names=evaluator.param_names,
            log_root=log_root,
        )
        rl.log_start()

        mem_logger = ExperimentLogger(
            evaluator.spec_names, evaluator.constraints,
            evaluator.directions, evaluator.circuit_type,
        )
        bounds = evaluator.param_bounds
        log_scale = getattr(evaluator, 'log_scale', None)
        n_dims = bounds.shape[0]

        # Trust-region settings, following TuRBO.
        batch_size = 1
        length_init = 0.8
        length_min = 0.5 ** 7
        length_max = 1.6
        success_tol = 3  # SCBO uses 3 (stricter than TuRBO's 10)
        failure_tol = math.ceil(max(4.0 / batch_size, float(n_dims) / batch_size))

        lb = bounds[:, 0]
        ub = bounds[:, 1]

        # Normalization helpers, as in TuRBO.
        def _to_unit(X_raw):
            X_out = np.empty_like(X_raw)
            for d_i in range(n_dims):
                if log_scale is not None and log_scale[d_i] and lb[d_i] > 0:
                    log_lo, log_hi = np.log10(lb[d_i]), np.log10(ub[d_i])
                    X_out[..., d_i] = (np.log10(X_raw[..., d_i]) - log_lo) / (log_hi - log_lo)
                else:
                    X_out[..., d_i] = (X_raw[..., d_i] - lb[d_i]) / (ub[d_i] - lb[d_i])
            return X_out

        def _from_unit(X_unit):
            X_out = np.empty_like(X_unit)
            for d_i in range(n_dims):
                if log_scale is not None and log_scale[d_i] and lb[d_i] > 0:
                    log_lo, log_hi = np.log10(lb[d_i]), np.log10(ub[d_i])
                    X_out[..., d_i] = 10 ** (X_unit[..., d_i] * (log_hi - log_lo) + log_lo)
                else:
                    X_out[..., d_i] = X_unit[..., d_i] * (ub[d_i] - lb[d_i]) + lb[d_i]
            return X_out

        # Objective: clamped sum of feasibility margins, as in TuRBO.
        def _compute_obj(specs_dict):
            total = 0.0
            for name in evaluator.spec_names:
                val = specs_dict.get(name, np.nan)
                c = evaluator.constraints[name]
                if np.isnan(val):
                    total -= 1.0
                    continue
                if evaluator.directions[name] == 'min':
                    margin = (val - c) / (abs(c) + 1e-6)
                else:
                    margin = (c - val) / (abs(c) + 1e-6)
                total += max(-1.0, min(1.0, margin))
            return total

        # Per-spec constraint value; positive means satisfied.
        def _compute_constraint_values(specs_dict):
            """Return dict of spec_name -> constraint margin (positive = feasible)."""
            vals = {}
            for name in evaluator.spec_names:
                v = specs_dict.get(name, np.nan)
                c = evaluator.constraints[name]
                if np.isnan(v):
                    vals[name] = -1.0
                elif evaluator.directions[name] == 'min':
                    vals[name] = (v - c) / (abs(c) + 1e-6)
                else:
                    vals[name] = (c - v) / (abs(c) + 1e-6)
            return vals

        # Evaluate one design and record it.
        def _eval_and_log(x_raw, iter_idx):
            t0 = time.time()
            specs = evaluator.evaluate(x_raw)
            sim_time = time.time() - t0
            mem_logger.log(x_raw, specs)
            feas = check_feasibility(specs, evaluator.constraints, evaluator.directions)
            fom = compute_fom(specs, evaluator.circuit_type,
                              evaluator.constraints, evaluator.directions) if feas else np.nan
            obj_val = _compute_obj(specs)
            c_vals = _compute_constraint_values(specs)
            return specs, feas, fom, obj_val, c_vals, sim_time

        # Early-stop check.
        def _check_early_stop(x_raw, feas, iter_idx):
            if not (early_stop and feas):
                return False
            rl.log.info(f'[EARLY STOP] Feasible found at iter {iter_idx}')
            return True

        # Fit a single GP.
        def _fit_one_gp(train_X_t, train_Y_t):
            """Fit a SingleTaskGP with Matern 5/2 ARD kernel. Returns GP or None."""
            try:
                train_Y_std = (train_Y_t - train_Y_t.mean()) / (train_Y_t.std() + 1e-8)
                likelihood = GaussianLikelihood(
                    noise_constraint=Interval(1e-8, 1e-3)).to(device)
                covar_module = ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=n_dims,
                                 lengthscale_constraint=Interval(0.005, 4.0)),
                ).to(device)
                gp = SingleTaskGP(
                    train_X_t, train_Y_std,
                    covar_module=covar_module,
                    likelihood=likelihood,
                ).to(device)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                return gp
            except Exception:
                return None

        # Probability of feasibility from the constraint GPs.
        def _compute_pof(constraint_gps, X_cand_t, c_means_epoch):
            """Probability of feasibility from independent constraint GPs.

            For each constraint GP, PoF_s = Phi(mu_s / sigma_s) where positive = feasible.
            Overall PoF = prod_s PoF_s.
            """
            n = X_cand_t.shape[0]
            log_pof = torch.zeros(n, dtype=dtype, device=device)
            for s_name, c_gp in constraint_gps.items():
                if c_gp is None:
                    continue
                with torch.no_grad():
                    posterior = c_gp.posterior(X_cand_t)
                    mu = posterior.mean.squeeze(-1)
                    sigma = posterior.variance.squeeze(-1).sqrt().clamp(min=1e-8)
                    # mu is standardized: un-standardize threshold
                    # Constraint satisfied when raw value > 0
                    # Standardized threshold: (0 - mean) / std of training data
                    c_train = c_means_epoch[s_name]
                    threshold = (0.0 - c_train['mean']) / (c_train['std'] + 1e-8)
                    z = (mu - threshold) / sigma
                    # Phi(z) via sigmoid approximation for GPU efficiency
                    pof_s = torch.sigmoid(1.7 * z)  # close to Phi(z)
                    log_pof += torch.log(pof_s + 1e-30)
            return torch.exp(log_pof)

        # Multi-restart trust-region loop.
        t = 0
        early_found = False

        while t < budget:
            # Start a trust-region epoch.
            length = length_init
            n_success = 0
            n_failure = 0

            n_pts = min(n_init, budget - t)
            if n_pts <= 0:
                break
            X_epoch_raw = latin_hypercube_sampling(
                bounds, n_pts, seed=seed + t, log_scale=log_scale)

            X_epoch = []
            obj_epoch = []
            c_epoch = {name: [] for name in evaluator.spec_names}
            feas_epoch = []
            fom_epoch = []
            is_init_phase = (t == 0)

            t_init_start = time.time()
            for i in range(n_pts):
                if t >= budget:
                    break
                specs, feas, fom, obj_val, c_vals, sim_time = _eval_and_log(X_epoch_raw[i], t)

                if is_init_phase:
                    rl.log_init_sample(i, specs, feas, sim_time)
                else:
                    rl.log_iteration(t, specs, feas, fom,
                                     sim_time=sim_time, gp_time=0, tda_time=0, acq_time=0)

                X_epoch.append(X_epoch_raw[i])
                obj_epoch.append(obj_val)
                feas_epoch.append(feas)
                fom_epoch.append(fom)
                for name in evaluator.spec_names:
                    c_epoch[name].append(c_vals[name])
                t += 1

                if _check_early_stop(X_epoch_raw[i], feas, t - 1):
                    early_found = True
                    break

            if is_init_phase:
                rl.log_init_summary(time.time() - t_init_start)

            if early_found:
                break

            X_epoch = np.array(X_epoch)
            obj_epoch = np.array(obj_epoch)
            feas_epoch = np.array(feas_epoch)
            fom_epoch = np.array(fom_epoch)
            c_epoch_np = {name: np.array(vals) for name, vals in c_epoch.items()}

            # Best objective: prefer feasible, then best obj among feasible
            if np.any(feas_epoch):
                fom_tmp = np.where(feas_epoch, obj_epoch, -np.inf)
                best_idx_epoch = np.argmax(fom_tmp)
            else:
                best_idx_epoch = np.argmax(obj_epoch)
            best_obj = obj_epoch[best_idx_epoch]

            # Inner loop for this trust region.
            restart_triggered = False
            while t < budget and not restart_triggered:
                sys.stdout.flush()

                # Normalize epoch data to [0,1]^d
                X_epoch_norm = _to_unit(X_epoch)
                train_X_t = torch.tensor(X_epoch_norm, dtype=dtype, device=device)

                # TR center = best feasible point, or least-violating
                if np.any(feas_epoch):
                    fom_tmp = np.where(feas_epoch, obj_epoch, -np.inf)
                    center_idx = np.argmax(fom_tmp)
                else:
                    center_idx = np.argmax(obj_epoch)
                x_center = torch.tensor(
                    X_epoch_norm[center_idx], dtype=dtype, device=device)

                # Fit the objective GP and one GP per constraint.
                t0 = time.time()
                obj_epoch_t = torch.tensor(
                    obj_epoch, dtype=dtype, device=device).unsqueeze(-1)
                obj_gp = _fit_one_gp(train_X_t, obj_epoch_t)

                constraint_gps = {}
                c_means_epoch = {}
                for name in evaluator.spec_names:
                    c_vals_t = torch.tensor(
                        c_epoch_np[name], dtype=dtype, device=device).unsqueeze(-1)
                    c_means_epoch[name] = {
                        'mean': float(c_vals_t.mean()),
                        'std': float(c_vals_t.std() + 1e-8),
                    }
                    constraint_gps[name] = _fit_one_gp(train_X_t, c_vals_t)

                if obj_gp is None:
                    # GP fit failed — fallback to random within TR
                    rl.log.info('[SCBO] GP fit failed, sampling at random')
                    tr_lb_np = np.clip(x_center.cpu().numpy() - length / 2.0, 0.0, 1.0)
                    tr_ub_np = np.clip(x_center.cpu().numpy() + length / 2.0, 0.0, 1.0)
                    rng = np.random.RandomState(seed + t)
                    x_next_norm = rng.uniform(tr_lb_np, tr_ub_np)
                    x_next = _from_unit(x_next_norm)
                    x_next = np.clip(x_next, lb, ub)
                    gp_time = time.time() - t0
                    acq_time = 0

                    specs, feas, fom, obj_val, c_vals, sim_time = _eval_and_log(x_next, t)
                    rl.log_iteration(t, specs, feas, fom,
                                     sim_time=sim_time, gp_time=gp_time,
                                     tda_time=0, acq_time=acq_time)

                    X_epoch = np.vstack([X_epoch, x_next.reshape(1, -1)])
                    obj_epoch = np.append(obj_epoch, obj_val)
                    feas_epoch = np.append(feas_epoch, feas)
                    fom_epoch = np.append(fom_epoch, fom)
                    for name in evaluator.spec_names:
                        c_epoch_np[name] = np.append(c_epoch_np[name], c_vals[name])
                    t += 1

                    if obj_val > best_obj + 1e-3 * math.fabs(best_obj):
                        n_success += 1
                        n_failure = 0
                    else:
                        n_success = 0
                        n_failure += 1
                    if n_success == success_tol:
                        length = min(2.0 * length, length_max)
                        n_success = 0
                    elif n_failure == failure_tol:
                        length /= 2.0
                        n_failure = 0
                    best_obj = max(best_obj, obj_val)
                    if length < length_min:
                        restart_triggered = True
                        rl.log.info(f'[SCBO] TR restart (collapsed below {length_min:.4f})')
                    if _check_early_stop(x_next, feas, t - 1):
                        early_found = True
                        break
                    continue

                gp_time = time.time() - t0

                # Trust-region bounds from the objective GP lengthscales.
                weights = obj_gp.covar_module.base_kernel.lengthscale.squeeze().detach()
                weights = weights / weights.mean()
                weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
                tr_lb = torch.clamp(x_center - weights * length / 2.0, 0.0, 1.0)
                tr_ub = torch.clamp(x_center + weights * length / 2.0, 0.0, 1.0)

                # Sobol candidates with a perturbation mask, as in TuRBO.
                t0 = time.time()
                n_cands = min(5000, max(2000, 200 * n_dims))
                sobol = torch.quasirandom.SobolEngine(n_dims, scramble=True, seed=seed + t)
                pert = sobol.draw(n_cands).to(dtype=dtype, device=device)
                pert = tr_lb + (tr_ub - tr_lb) * pert

                prob_perturb = min(20.0 / n_dims, 1.0)
                mask = torch.rand(n_cands, n_dims, dtype=dtype, device=device) <= prob_perturb
                ind = torch.where(mask.sum(dim=1) == 0)[0]
                if len(ind) > 0:
                    mask[ind, torch.randint(0, n_dims, size=(len(ind),), device=device)] = 1

                X_cand = x_center.expand(n_cands, n_dims).clone()
                X_cand[mask] = pert[mask]

                # Thompson sample the objective GP, weighted by the
                # probability of feasibility from the constraint GPs.
                pof = _compute_pof(constraint_gps, X_cand, c_means_epoch)
                with torch.no_grad():
                    posterior = obj_gp.posterior(X_cand)
                    ts_sample = posterior.rsample(torch.Size([1])).squeeze(0).squeeze(-1)
                    # Constrained TS: objective sample * PoF
                    acq_values = ts_sample * pof

                best_cand_idx = torch.argmax(acq_values)
                x_next_norm = X_cand[best_cand_idx].cpu().numpy()
                x_next = _from_unit(x_next_norm)
                x_next = np.clip(x_next, lb, ub)
                acq_time = time.time() - t0

                # Evaluate.
                specs, feas, fom, obj_val, c_vals, sim_time = _eval_and_log(x_next, t)
                rl.log_iteration(t, specs, feas, fom,
                                 sim_time=sim_time, gp_time=gp_time,
                                 tda_time=0, acq_time=acq_time)

                X_epoch = np.vstack([X_epoch, x_next.reshape(1, -1)])
                obj_epoch = np.append(obj_epoch, obj_val)
                feas_epoch = np.append(feas_epoch, feas)
                fom_epoch = np.append(fom_epoch, fom)
                for name in evaluator.spec_names:
                    c_epoch_np[name] = np.append(c_epoch_np[name], c_vals[name])
                t += 1

                # Feasibility-aware trust-region adaptation.
                # Success = found new feasible OR improved objective among feasible
                improved = False
                if feas and np.any(feas_epoch[:-1]):
                    old_best_feas_obj = np.max(obj_epoch[:-1][feas_epoch[:-1]])
                    improved = obj_val > old_best_feas_obj + 1e-3 * math.fabs(old_best_feas_obj)
                elif feas and not np.any(feas_epoch[:-1]):
                    improved = True  # first feasible point
                elif not feas and not np.any(feas_epoch):
                    improved = obj_val > best_obj + 1e-3 * math.fabs(best_obj)

                if improved:
                    n_success += 1
                    n_failure = 0
                else:
                    n_success = 0
                    n_failure += 1

                if n_success == success_tol:
                    length = min(2.0 * length, length_max)
                    n_success = 0
                    rl.log.info(f'[SCBO] TR expand -> length={length:.4f}')
                elif n_failure == failure_tol:
                    length /= 2.0
                    n_failure = 0
                    rl.log.info(f'[SCBO] TR shrink -> length={length:.4f}')

                best_obj = max(best_obj, obj_val)

                if length < length_min:
                    restart_triggered = True
                    rl.log.info(f'[SCBO] TR restart (collapsed below {length_min:.4f})')

                if _check_early_stop(x_next, feas, t - 1):
                    early_found = True
                    break

            if early_found:
                break

        results = rl.log_finish()
        rl.close()
        return mem_logger, results


def get_optimizer(method_name, **kwargs):
    """Build an optimizer by the name used on the command line."""
    optimizers = {
        'ATLAS': lambda: TopoBO(
            lambda_0=kwargs.get('lambda_0'),
            gamma=kwargs.get('gamma', 1.0),
            mapper_n_cubes=kwargs.get('mapper_n_cubes', 10),
            mapper_overlap=kwargs.get('mapper_overlap', 0.3),
            mapper_min_cluster=kwargs.get('mapper_min_cluster', 3),
            mapper_dbscan_eps=kwargs.get('mapper_dbscan_eps', 0.3),
            mapper_eps_multiplier=kwargs.get('mapper_eps_multiplier', 1.0),
            top_k_frac=kwargs.get('top_k_frac'),
            use_component_discount=kwargs.get('use_component_discount', True),
            use_centroid_guidance=kwargs.get('use_centroid_guidance', True),
            use_lambda_decay=kwargs.get('use_lambda_decay', True),
        ),
        'WEIBO': WEIBO,
        'TuRBO': TuRBOptimizer,
        'CMAES': CMAESOpt,
        'Random': RandomSearchOpt,
        'MACE': lambda: MACEOpt(alpha=kwargs.get('alpha'),
                                batch_size=kwargs.get('batch_size', 10)),
        'SCBO': SCBOOpt,
    }
    if method_name not in optimizers:
        raise ValueError(f'Unknown method {method_name}. '
                         f'Choose from {list(optimizers)}')
    return optimizers[method_name]()
