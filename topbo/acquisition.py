"""Acquisition functions."""

import numpy as np


def compute_wei(gp, X_cand, best_f, observed_Y=None):
    """Weighted Expected Improvement: wEI(x) = EI(x) * p(x).

    When no feasible point exists (best_f = -inf):
      - Uses bottleneck-aware acquisition that strongly weights the hardest
        spec relative to the best observed value, preventing the optimizer
        from getting trapped in basins where easy specs are satisfied but
        the bottleneck spec is far from its target.

    Args:
        gp: GPSurrogate instance.
        X_cand: (n, d) array of candidate points.
        best_f: Best feasible FoM so far (-inf if no feasible).
        observed_Y: Optional dict of spec_name -> array of observed values.
                    Used to compute bottleneck weights from historical data.

    Returns:
        wei: (n,) array of wEI values.
    """
    p = gp.compute_p(X_cand)
    ei = gp.compute_ei(X_cand, best_f)

    if np.isinf(best_f) and best_f < 0:
        means, stds = gp.predict(X_cand)
        n = X_cand.shape[0]
        n_specs = len(gp.spec_names)
        margins = np.zeros((n, n_specs))
        for i, name in enumerate(gp.spec_names):
            c = gp.constraints[name]
            eps = abs(c) * 1e-10 + 1e-30
            if gp.directions[name] == 'min':
                margins[:, i] = np.log(np.maximum(means[name], eps) / (abs(c) + 1e-30))
            else:
                margins[:, i] = np.log((abs(c) + 1e-30) / np.maximum(means[name], eps))

        # Bottleneck-aware weighting using OBSERVED data (not candidate stats).
        # Compute how close the best observed value is to each constraint.
        # Specs where the best observation is still far from the constraint
        # get much higher weight — these are the true bottlenecks.
        spec_weights = np.ones(n_specs)
        if observed_Y is not None:
            obs_best_ratios = np.zeros(n_specs)
            for i, name in enumerate(gp.spec_names):
                c = gp.constraints[name]
                vals = observed_Y[name]
                valid = vals[~np.isnan(vals)]
                if len(valid) == 0:
                    obs_best_ratios[i] = 0.0
                    continue
                if gp.directions[name] == 'min':
                    best_val = np.max(valid)
                    obs_best_ratios[i] = best_val / (abs(c) + 1e-30)
                else:
                    best_val = np.min(valid)
                    obs_best_ratios[i] = (abs(c) + 1e-30) / (best_val + 1e-30)

            # Bottleneck weight: specs far from constraint get exponentially
            # higher weight. ratio < 1 means unsatisfied.
            # weight = exp(-2 * ratio) to ratio=0.1 gets weight 4.5x of ratio=1.0
            clamped = np.clip(obs_best_ratios, 0.01, 5.0)
            spec_weights = np.exp(-2.0 * clamped)
            # Ensure minimum weight so no spec is completely ignored
            spec_weights = np.maximum(spec_weights, 0.01)
        else:
            # Fallback: use candidate satisfaction rates
            for i in range(n_specs):
                sat_rate = np.mean(margins[:, i] > 0)
                spec_weights[i] = 1.0 / (sat_rate + 0.05)

        spec_weights /= spec_weights.sum()

        sigmoid_sum = np.zeros(n)
        for i in range(n_specs):
            col = margins[:, i]
            spread = np.std(col) + 1e-10
            z = np.clip(3.0 * col / spread, -30, 30)
            sigmoid_sum += spec_weights[i] * (1.0 / (1.0 + np.exp(-z)))

        # Worst-margin bonus: reward improving the bottleneck spec
        worst_margin = np.min(margins, axis=1)
        wm_norm = (worst_margin - worst_margin.min()) / (worst_margin.max() - worst_margin.min() + 1e-10)

        h_combined = 0.6 * sigmoid_sum / (sigmoid_sum.max() + 1e-10) + 0.4 * wm_norm
        h_norm = (h_combined - h_combined.min()) / (h_combined.max() - h_combined.min() + 1e-10)
        return 0.5 * h_norm + 0.5 * p
    return ei * p

def compute_lambda_t(t, T, lambda_0, gamma=1.0):
    """Compute step-decaying topology weight.

    Maintains full lambda_0 for the first 60% of budget, then decays
    quadratically to 0. This keeps topology exploration active longer
    than linear decay, which was turning off too early.

    Args:
        t: Current iteration.
        T: Total budget.
        lambda_0: Initial lambda (scale-matched to median wEI).
        gamma: Decay exponent (default 1.0).

    Returns:
        lambda_t: Current topology weight.
    """
    ratio = min(t / T, 1.0)
    hold_fraction = 0.6  # keep full lambda for first 60%
    if ratio <= hold_fraction:
        return lambda_0
    # Quadratic decay over remaining 40%
    decay_progress = (ratio - hold_fraction) / (1.0 - hold_fraction)
    return lambda_0 * (1.0 - decay_progress) ** 2

def compute_mace(gp, X_cand, best_f, alpha=None, beta_t=2.0):
    """MACE ensemble acquisition: weighted combination of EI, PI, UCB.

    Single posterior call — EI, PI, UCB all derived from the same (mu, sigma).
    All three are normalized to [0, 1] before combining so the weights
    are scale-invariant.

    Args:
        gp: GPSurrogate instance.
        X_cand: (n, d) array of candidate points.
        best_f: Current best feasible FoM (-inf if none).
        alpha: (3,) weights for [EI, PI, UCB]. Default [0.4, 0.3, 0.3].
        beta_t: UCB exploration parameter.

    Returns:
        mace: (n,) array of MACE acquisition values.
    """
    from scipy.stats import norm
    import torch

    if alpha is None:
        alpha = [0.4, 0.3, 0.3]

    if gp.fom_gp is None:
        return np.zeros(X_cand.shape[0])

    # Single posterior pass for mu, sigma
    n = X_cand.shape[0]
    mu = np.empty(n)
    sigma = np.empty(n)
    batch_size = getattr(gp, 'PREDICT_BATCH_SIZE', 2048)
    device = getattr(gp, 'device', 'cpu')
    dtype = getattr(gp, 'dtype', torch.float64)

    gp.fom_gp.eval()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.tensor(X_cand[start:end], dtype=dtype, device=device)
        with torch.no_grad():
            posterior = gp.fom_gp.posterior(batch)
            mu[start:end] = posterior.mean.squeeze(-1).cpu().numpy()
            sigma[start:end] = posterior.variance.squeeze(-1).sqrt().cpu().numpy()

    sigma = np.maximum(sigma, 1e-10)
    z = (mu - best_f) / sigma

    # EI, PI, UCB from same mu/sigma — no extra GP calls
    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
    pi = norm.cdf(z)
    ucb = mu + np.sqrt(beta_t) * sigma

    def _normalize(a):
        lo, hi = a.min(), a.max()
        if hi - lo < 1e-30:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    return alpha[0] * _normalize(ei) + alpha[1] * _normalize(pi) + alpha[2] * _normalize(ucb)
