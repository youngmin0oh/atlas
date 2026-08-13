"""GP surrogate model using BoTorch (independent GP per spec + FoM)."""

import numpy as np
import torch
from scipy.stats import norm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize


def _matern_ard(d):
    """Matern 5/2 kernel with one lengthscale per design parameter."""
    return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))


class GPSurrogate:
    """Independent GP surrogate for each spec function.

    Uses Matern 5/2 kernel with ARD lengthscales, fitted via MLE.
    """

    def __init__(self, spec_names, constraints, directions, param_bounds):
        """
        Args:
            spec_names: List of spec names (sorted).
            constraints: Dict of spec_name -> threshold.
            directions: Dict of spec_name -> 'min' or 'max'.
            param_bounds: (d, 2) array of parameter bounds.
        """
        self.spec_names = spec_names
        self.constraints = constraints
        self.directions = directions
        self.param_bounds = param_bounds
        self.n_specs = len(spec_names)
        self.d = param_bounds.shape[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64

        # BoTorch bounds as tensor
        self.bounds_tensor = torch.tensor(
            param_bounds.T, dtype=self.dtype, device=self.device)

        self.gps = {}       # spec_name -> SingleTaskGP
        self.fom_gp = None  # GP for FoM (optional)
        self._fitted = False

    def fit(self, X, Y_dict, fom_values=None, incremental=False):
        """Fit independent GPs on observed data.

        Args:
            X: (n, d) numpy array of parameter vectors.
            Y_dict: Dict of spec_name -> (n,) numpy array of observed values.
            fom_values: Optional (n,) array of FoM values for feasible points.
        """
        tkw = dict(dtype=self.dtype, device=self.device)
        train_X = torch.tensor(X, **tkw)
        self.gps = {}

        for name in self.spec_names:
            y = Y_dict[name]
            # Replace NaN with worst-case values
            y_clean = np.nan_to_num(y, nan=0.0)
            train_Y = torch.tensor(y_clean, **tkw).unsqueeze(-1)

            try:
                gp = self._make_gp(train_X, train_Y)
                fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
                self.gps[name] = gp
            except Exception as e:
                # MLE can fail on degenerate data; fall back to prior hyperparameters
                print(f'[GPSurrogate] MLL fit failed for {name}: {e}')
                self.gps[name] = self._make_gp(train_X, train_Y)

        # Fit FoM GP if we have feasible observations
        if fom_values is not None and len(fom_values) > 0:
            valid = ~np.isnan(fom_values) & ~np.isinf(fom_values)
            if valid.sum() >= 2:
                fom_X = torch.tensor(X[valid], **tkw)
                fom_Y = torch.tensor(fom_values[valid], **tkw).unsqueeze(-1)
                try:
                    self.fom_gp = self._make_gp(fom_X, fom_Y)
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(
                        self.fom_gp.likelihood, self.fom_gp))
                except Exception:
                    self.fom_gp = None

        self._fitted = True

    def _make_gp(self, train_X, train_Y):
        return SingleTaskGP(
            train_X, train_Y,
            covar_module=_matern_ard(self.d),
            input_transform=Normalize(d=self.d, bounds=self.bounds_tensor),
            outcome_transform=Standardize(m=1),
        ).to(self.device)

    # Max test points per batch to avoid GPU OOM on large candidate sets
    PREDICT_BATCH_SIZE = 4096

    def predict(self, X):
        """Predict mean and std for each spec.

        Args:
            X: (n, d) numpy array.

        Returns:
            means: Dict of spec_name -> (n,) array of posterior means.
            stds: Dict of spec_name -> (n,) array of posterior stds.
        """
        n = X.shape[0]
        means = {name: np.empty(n) for name in self.spec_names}
        stds = {name: np.empty(n) for name in self.spec_names}

        for name in self.spec_names:
            gp = self.gps[name]
            gp.eval()
            for start in range(0, n, self.PREDICT_BATCH_SIZE):
                end = min(start + self.PREDICT_BATCH_SIZE, n)
                batch = torch.tensor(
                    X[start:end], dtype=self.dtype, device=self.device)
                with torch.no_grad():
                    posterior = gp.posterior(batch)
                    means[name][start:end] = \
                        posterior.mean.squeeze(-1).cpu().numpy()
                    stds[name][start:end] = np.maximum(
                        posterior.variance.squeeze(-1).sqrt().cpu().numpy(),
                        1e-10)
        return means, stds

    def compute_h(self, X):
        """Surrogate feasibility score: h(x) = min_s (mu_s - c_s) / sigma_s.

        For _max specs, this becomes (c_s - mu_s) / sigma_s.

        Args:
            X: (n, d) numpy array.

        Returns:
            h: (n,) array. h > 0 means predicted feasible.
        """
        means, stds = self.predict(X)
        n = X.shape[0]
        h_per_spec = np.zeros((n, self.n_specs))

        for i, name in enumerate(self.spec_names):
            c = self.constraints[name]
            if self.directions[name] == 'min':
                h_per_spec[:, i] = (means[name] - c) / stds[name]
            else:
                h_per_spec[:, i] = (c - means[name]) / stds[name]

        return np.min(h_per_spec, axis=1)

    def compute_p(self, X):
        """Feasibility probability: p(x) = prod_s Phi(h_s(x)).

        Args:
            X: (n, d) numpy array.

        Returns:
            p: (n,) array of feasibility probabilities in [0, 1].
        """
        means, stds = self.predict(X)
        n = X.shape[0]
        log_p = np.zeros(n)

        for name in self.spec_names:
            c = self.constraints[name]
            if self.directions[name] == 'min':
                z = (means[name] - c) / stds[name]
            else:
                z = (c - means[name]) / stds[name]
            log_p += norm.logcdf(z)

        return np.exp(log_p)

    def compute_ei(self, X, best_f):
        """Expected Improvement for FoM.

        Args:
            X: (n, d) numpy array.
            best_f: Current best feasible FoM.

        Returns:
            ei: (n,) array of EI values.
        """
        if self.fom_gp is None:
            return np.zeros(X.shape[0])

        n = X.shape[0]
        mu = np.empty(n)
        sigma = np.empty(n)
        self.fom_gp.eval()
        for start in range(0, n, self.PREDICT_BATCH_SIZE):
            end = min(start + self.PREDICT_BATCH_SIZE, n)
            batch = torch.tensor(
                X[start:end], dtype=self.dtype, device=self.device)
            with torch.no_grad():
                posterior = self.fom_gp.posterior(batch)
                mu[start:end] = posterior.mean.squeeze(-1).cpu().numpy()
                sigma[start:end] = posterior.variance.squeeze(-1).sqrt().cpu().numpy()

        sigma = np.maximum(sigma, 1e-10)
        z = (mu - best_f) / sigma
        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        return ei

    def predict_uncertainty(self, X):
        """Total predictive uncertainty across all specs.

        Args:
            X: (n, d) numpy array.

        Returns:
            uncertainty: (n,) array of summed variances.
        """
        means, stds = self.predict(X)
        n = X.shape[0]
        total_var = np.zeros(n)
        for name in self.spec_names:
            total_var += stds[name] ** 2
        return total_var
