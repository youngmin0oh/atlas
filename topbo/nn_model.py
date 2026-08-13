"""Probabilistic NN ensemble surrogate.

A drop-in alternative to GPSurrogate for the regime where exact GP inference
becomes the bottleneck: fitting is O(n) in the dataset size and prediction cost
is constant. Uncertainty combines ensemble disagreement (epistemic) with the
per-model predicted variance (aleatoric).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm


class ProbabilisticMLP(nn.Module):
    """MLP mapping design parameters to a per-spec mean and log-variance."""

    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, output_dim)
        self.logvar_head = nn.Linear(prev, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=1.0 / (2.0 * m.in_features ** 0.5))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.trunk(x)
        return self.mean_head(h), torch.clamp(self.logvar_head(h), -10.0, 0.5)


class NNSurrogate:
    """Ensemble of ProbabilisticMLPs; same interface as GPSurrogate."""

    def __init__(self, spec_names, constraints, directions, param_bounds,
                 n_ensemble=10, hidden_dims=(256, 256), lr=1e-3,
                 train_epochs=200, incremental_epochs=30):
        self.spec_names = spec_names
        self.constraints = constraints
        self.directions = directions
        self.param_bounds = param_bounds
        self.n_specs = len(spec_names)
        self.d = param_bounds.shape[0]
        self.train_epochs = train_epochs
        self.incremental_epochs = incremental_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.models = [
            ProbabilisticMLP(self.d, self.n_specs, hidden_dims).to(self.device)
            for _ in range(n_ensemble)
        ]
        self.optimizers = [
            torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
            for m in self.models
        ]

        zeros_in = torch.zeros(self.d, device=self.device)
        zeros_out = torch.zeros(self.n_specs, device=self.device)
        self.input_mean, self.input_std = zeros_in, torch.ones_like(zeros_in)
        self.output_mean, self.output_std = zeros_out, torch.ones_like(zeros_out)

        self._fitted = False

    def fit(self, X, Y_dict, fom_values=None, incremental=False):
        """Train every ensemble member; `incremental` runs a short fine-tune."""
        n = X.shape[0]
        Y = np.column_stack([np.nan_to_num(Y_dict[name], nan=0.0)
                             for name in self.spec_names])
        params = torch.tensor(X, dtype=torch.float32, device=self.device)
        specs = torch.tensor(Y, dtype=torch.float32, device=self.device)

        self.input_mean, self.input_std = params.mean(dim=0), params.std(dim=0)
        self.output_mean, self.output_std = specs.mean(dim=0), specs.std(dim=0)

        x_all = self._normalize_input(params)
        y_all = (specs - self.output_mean) / (self.output_std + 1e-8)

        epochs = self.incremental_epochs if (incremental and self._fitted) \
            else self.train_epochs
        batch_size = min(n, 64)
        for _ in range(epochs):
            order = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                self._train_step(x_all[idx], y_all[idx])

        self._fitted = True

    def _train_step(self, x, y):
        """One bootstrap-resampled Gaussian NLL step per ensemble member."""
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            pick = torch.randint(0, len(x), (len(x),), device=self.device)
            mean, logvar = model(x[pick])
            loss = 0.5 * (logvar + (y[pick] - mean) ** 2 / torch.exp(logvar)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _normalize_input(self, x):
        return (x - self.input_mean) / (self.input_std + 1e-8)

    def predict(self, X):
        """Posterior mean and standard deviation per spec, in original units."""
        params = torch.as_tensor(X, device=self.device, dtype=torch.float32)
        if params.dim() == 1:
            params = params.unsqueeze(0)
        x = self._normalize_input(params)

        means, logvars = [], []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                m, lv = model(x)
            means.append(m)
            logvars.append(lv)
        means = torch.stack(means)
        logvars = torch.stack(logvars)

        scale = self.output_std + 1e-8
        mu = means.mean(dim=0) * scale + self.output_mean
        epistemic = means.std(dim=0) * scale
        aleatoric = torch.exp(logvars).mean(dim=0).sqrt() * scale
        sigma = torch.sqrt(epistemic ** 2 + aleatoric ** 2)

        mu_np, sigma_np = mu.cpu().numpy(), sigma.cpu().numpy()
        return ({name: mu_np[:, i] for i, name in enumerate(self.spec_names)},
                {name: np.maximum(sigma_np[:, i], 1e-10)
                 for i, name in enumerate(self.spec_names)})

    def compute_h(self, X):
        """Feasibility score as a min-over-specs log margin.

        The ensemble's predictive scale varies far more across specs than a
        GP's does, so the margins are made comparable by taking the log ratio
        to the constraint rather than dividing by sigma.
        """
        means, _ = self.predict(X)
        h = np.zeros((X.shape[0], self.n_specs))
        for i, name in enumerate(self.spec_names):
            c = self.constraints[name]
            floor = abs(c) * 1e-10 + 1e-30
            if self.directions[name] == 'min':
                h[:, i] = np.log(np.maximum(means[name], floor) / (abs(c) + 1e-30))
            else:
                h[:, i] = np.log((abs(c) + 1e-30) / np.maximum(means[name], floor))
        return np.min(h, axis=1)

    def compute_p(self, X):
        """Joint feasibility probability, product of per-spec normal CDFs."""
        means, stds = self.predict(X)
        log_p = np.zeros(X.shape[0])
        for name in self.spec_names:
            c = self.constraints[name]
            if self.directions[name] == 'min':
                z = (means[name] - c) / stds[name]
            else:
                z = (c - means[name]) / stds[name]
            log_p += norm.logcdf(z)

        # When every candidate is deeply infeasible the probabilities underflow
        # to zero and stop ranking anything; shift them back into range.
        if log_p.max() < -20:
            log_p = log_p - log_p.max()
        return np.exp(log_p)

    def compute_ei(self, X, best_f):
        """No separate FoM model is fitted, so wEI reduces to its p(x) factor."""
        return np.zeros(X.shape[0])

    def predict_uncertainty(self, X):
        _, stds = self.predict(X)
        return sum(stds[name] ** 2 for name in self.spec_names)
