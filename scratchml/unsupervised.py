"""Unsupervised learning algorithms."""
import numpy as np


class KMeans:
    """Lloyd's algorithm with k-means++ seeding and several restarts."""

    def __init__(self, k, n_init=10, max_iter=300, seed=0):
        self.k, self.n_init, self.max_iter, self.seed = k, n_init, max_iter, seed

    def _seed_centers(self, X, rng):
        centers = [X[rng.integers(len(X))]]
        for _ in range(self.k - 1):
            d = ((X[:, None] - np.array(centers)[None]) ** 2).sum(axis=2).min(axis=1)
            centers.append(X[rng.choice(len(X), p=d / d.sum())])
        return np.array(centers)

    def _run(self, X, rng):
        centers = self._seed_centers(X, rng)
        for _ in range(self.max_iter):
            labels = ((X[:, None] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
            new = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
                            for j in range(self.k)])
            done = np.allclose(new, centers)
            centers = new
            if done:
                break
        d = ((X[:, None] - centers[None]) ** 2).sum(axis=2)
        return centers, d.argmin(axis=1), d.min(axis=1).sum()

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        runs = [self._run(X, rng) for _ in range(self.n_init)]
        self.centers_, self.labels_, self.inertia_ = min(runs, key=lambda r: r[2])
        return self


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = vt[: self.n_components]
        variance = s**2 / (len(X) - 1)
        self.explained_variance_ratio_ = (variance / variance.sum())[: self.n_components]
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


class GaussianMixture:
    """Full covariance Gaussian mixture fitted with expectation maximization."""

    def __init__(self, k, max_iter=200, tol=1e-4, reg=1e-6, seed=0):
        self.k, self.max_iter, self.tol, self.reg, self.seed = k, max_iter, tol, reg, seed

    def _log_prob(self, X):
        d = X.shape[1]
        out = np.empty((len(X), self.k))
        for j in range(self.k):
            L = np.linalg.cholesky(self.cov[j])
            z = np.linalg.solve(L, (X - self.means[j]).T)
            out[:, j] = (np.log(self.weights[j]) - np.log(np.diag(L)).sum()
                         - 0.5 * (d * np.log(2 * np.pi) + (z**2).sum(axis=0)))
        return out

    def fit(self, X):
        labels = KMeans(self.k, seed=self.seed).fit(X).labels_
        resp = np.eye(self.k)[labels]
        previous = -np.inf
        for _ in range(self.max_iter):
            self._m_step(X, resp)
            log_p = self._log_prob(X)
            top = log_p.max(axis=1, keepdims=True)
            log_norm = top + np.log(np.exp(log_p - top).sum(axis=1, keepdims=True))
            resp = np.exp(log_p - log_norm)
            self.log_likelihood_ = log_norm.mean()
            if abs(self.log_likelihood_ - previous) < self.tol:
                break
            previous = self.log_likelihood_
        self.labels_ = resp.argmax(axis=1)
        return self

    def _m_step(self, X, resp):
        n_k = resp.sum(axis=0) + 1e-12
        self.weights = n_k / len(X)
        self.means = resp.T @ X / n_k[:, None]
        self.cov = np.empty((self.k, X.shape[1], X.shape[1]))
        for j in range(self.k):
            c = X - self.means[j]
            self.cov[j] = (resp[:, j, None] * c).T @ c / n_k[j] + self.reg * np.eye(X.shape[1])
