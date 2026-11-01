"""Supervised learning algorithms. All follow fit(X, y) and predict(X)."""
import numpy as np


def _one_hot(y, k):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1
    return out


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


class RidgeRegression:
    """Linear regression with an L2 penalty, solved in closed form. alpha=0 is OLS."""

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.lstsq(A, Xc.T @ yc, rcond=None)[0]
        self.intercept_ = y_mean - x_mean @ self.coef_
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class LogisticRegression:
    """Multiclass logistic regression trained with full batch gradient descent."""

    def __init__(self, lr=0.5, epochs=2000, l2=1e-4):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        self.w, self.b = np.zeros((X.shape[1], k)), np.zeros(k)
        Y = _one_hot(y, k)
        for _ in range(self.epochs):
            error = _softmax(X @ self.w + self.b) - Y
            self.w -= self.lr * (X.T @ error / len(X) + 2 * self.l2 * self.w)
            self.b -= self.lr * error.mean(axis=0)
        return self

    def predict(self, X):
        return self.classes_[(X @ self.w + self.b).argmax(axis=1)]


class LinearSVM:
    """One vs rest linear SVM trained with averaged Pegasos subgradient descent."""

    def __init__(self, lam=1e-4, epochs=50, seed=0):
        self.lam, self.epochs, self.seed = lam, epochs, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        n, d = X.shape
        self.w, self.b = np.zeros((len(self.classes_), d)), np.zeros(len(self.classes_))
        for c, cls in enumerate(self.classes_):
            target = np.where(y == cls, 1.0, -1.0)
            w, b, t = np.zeros(d), 0.0, 0
            w_sum, b_sum, kept = np.zeros(d), 0.0, 0
            # Starting t past zero avoids the huge first steps of the raw 1 / (lam * t) schedule.
            offset = int(1 / self.lam) // 10
            for epoch in range(self.epochs):
                for i in rng.permutation(n):
                    t += 1
                    lr = 1 / (self.lam * (t + offset))
                    margin = target[i] * (X[i] @ w + b)
                    w *= 1 - lr * self.lam
                    if margin < 1:
                        w += lr * target[i] * X[i]
                        b += lr * target[i]
                    if epoch >= self.epochs // 2:
                        w_sum += w
                        b_sum += b
                        kept += 1
            self.w[c], self.b[c] = w_sum / kept, b_sum / kept
        return self

    def predict(self, X):
        return self.classes_[(X @ self.w.T + self.b).argmax(axis=1)]


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.classes_, self.y = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        d = ((X[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2)
        nearest = self.y[np.argpartition(d, self.k - 1, axis=1)[:, : self.k]]
        votes = np.apply_along_axis(np.bincount, 1, nearest, minlength=len(self.classes_))
        return self.classes_[votes.argmax(axis=1)]


class GaussianNB:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Variance smoothing matches the usual fix for constant features.
        smooth = 1e-9 * X.var(axis=0).max()
        self.mean = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.var = np.array([X[y == c].var(axis=0) + smooth for c in self.classes_])
        self.log_prior = np.log([np.mean(y == c) for c in self.classes_])
        return self

    def predict(self, X):
        ll = -0.5 * (np.log(2 * np.pi * self.var)[None] + (X[:, None] - self.mean) ** 2 / self.var[None])
        return self.classes_[(ll.sum(axis=2) + self.log_prior).argmax(axis=1)]


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
        self.value, self.feature, self.threshold = value, feature, threshold
        self.left, self.right = left, right


class _CARTTree:
    """CART tree. target is class probabilities (gini) or real values (mse)."""

    def __init__(self, criterion, max_depth, min_samples_split, max_features, rng):
        self.criterion, self.max_depth = criterion, max_depth
        self.min_samples_split, self.max_features, self.rng = min_samples_split, max_features, rng

    def fit(self, X, Y):
        self.root = self._grow(X, Y, 0)
        return self

    def _impurity_totals(self, sums, counts, sq_sums=None):
        """Weighted impurity contribution of one side, summed over all candidates."""
        if self.criterion == "gini":
            return counts - (sums**2).sum(axis=1) / counts
        return sq_sums - (sums**2).sum(axis=1) / counts

    def _best_split(self, X, Y):
        n, d = X.shape
        features = np.arange(d)
        if self.max_features and self.max_features < d:
            features = self.rng.choice(d, self.max_features, replace=False)
        best = (None, None, self._impurity_totals(Y.sum(0, keepdims=True), np.array([n]), np.array([(Y**2).sum()]))[0] - 1e-12)
        for f in features:
            order = np.argsort(X[:, f], kind="stable")
            x_sorted, Y_sorted = X[order, f], Y[order]
            valid = np.nonzero(x_sorted[1:] > x_sorted[:-1])[0]
            if not len(valid):
                continue
            left_sum = np.cumsum(Y_sorted, axis=0)[valid]
            left_n = valid + 1.0
            right_sum = Y_sorted.sum(axis=0) - left_sum
            right_n = n - left_n
            sq = np.cumsum((Y_sorted**2).sum(axis=1))
            left = self._impurity_totals(left_sum, left_n, sq[valid])
            right = self._impurity_totals(right_sum, right_n, sq[-1] - sq[valid])
            total = left + right
            i = total.argmin()
            if total[i] < best[2]:
                cut = (x_sorted[valid[i]] + x_sorted[valid[i] + 1]) / 2
                best = (f, cut, total[i])
        return best[0], best[1]

    def _grow(self, X, Y, depth):
        value = Y.mean(axis=0)
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            return _Node(value=value)
        feature, threshold = self._best_split(X, Y)
        if feature is None:
            return _Node(value=value)
        mask = X[:, feature] <= threshold
        if mask.all() or not mask.any():
            return _Node(value=value)
        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._grow(X[mask], Y[mask], depth + 1),
            right=self._grow(X[~mask], Y[~mask], depth + 1),
        )

    def predict_values(self, X):
        out = np.empty((len(X), self._width()))
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if not len(idx):
                continue
            if node.feature is None:
                out[idx] = node.value
                continue
            mask = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[mask]))
            stack.append((node.right, idx[~mask]))
        return out

    def _width(self):
        node = self.root
        while node.feature is not None:
            node = node.left
        return len(node.value)


class DecisionTree:
    def __init__(self, max_depth=20, min_samples_split=2, max_features=None, seed=0):
        self.max_depth, self.min_samples_split = max_depth, min_samples_split
        self.max_features, self.seed = max_features, seed

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        self.tree = _CARTTree("gini", self.max_depth, self.min_samples_split, self.max_features,
                              np.random.default_rng(self.seed))
        self.tree.fit(X, _one_hot(y, len(self.classes_)))
        return self

    def predict_proba(self, X):
        return self.tree.predict_values(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class RandomForest:
    def __init__(self, n_trees=100, max_depth=20, seed=0):
        self.n_trees, self.max_depth, self.seed = n_trees, max_depth, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.classes_, y = np.unique(y, return_inverse=True)
        Y = _one_hot(y, len(self.classes_))
        features = max(1, int(np.sqrt(X.shape[1])))
        self.trees = []
        for _ in range(self.n_trees):
            sample = rng.integers(0, len(X), len(X))
            tree = _CARTTree("gini", self.max_depth, 2, features, rng)
            self.trees.append(tree.fit(X[sample], Y[sample]))
        return self

    def predict(self, X):
        proba = sum(t.predict_values(X) for t in self.trees)
        return self.classes_[proba.argmax(axis=1)]


class GradientBoosting:
    """Softmax gradient boosting with one regression tree per class per round.

    Leaf values use the Newton step from Friedman's multiclass algorithm, which
    converges much faster than plain gradient averages.
    """

    def __init__(self, n_rounds=100, lr=0.1, max_depth=3, seed=0):
        self.n_rounds, self.lr, self.max_depth, self.seed = n_rounds, lr, max_depth, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.classes_, y = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        Y = _one_hot(y, k)
        self.base = np.log(Y.mean(axis=0) + 1e-12)
        scores = np.tile(self.base, (len(X), 1))
        self.rounds = []
        for _ in range(self.n_rounds):
            p = _softmax(scores)
            trees = []
            for c in range(k):
                residual = (Y[:, c] - p[:, c])[:, None]
                tree = _CARTTree("mse", self.max_depth, 2, None, rng).fit(X, residual)
                self._newton_leaves(tree.root, X, residual[:, 0], p[:, c], k)
                scores[:, c] += self.lr * tree.predict_values(X)[:, 0]
                trees.append(tree)
            self.rounds.append(trees)
        return self

    def _newton_leaves(self, node, X, residual, p, k):
        if node.feature is None:
            denom = np.sum(p * (1 - p)) + 1e-12
            node.value = np.array([(k - 1) / k * residual.sum() / denom])
            return
        mask = X[:, node.feature] <= node.threshold
        self._newton_leaves(node.left, X[mask], residual[mask], p[mask], k)
        self._newton_leaves(node.right, X[~mask], residual[~mask], p[~mask], k)

    def predict(self, X):
        scores = np.tile(self.base, (len(X), 1))
        for trees in self.rounds:
            for c, tree in enumerate(trees):
                scores[:, c] += self.lr * tree.predict_values(X)[:, 0]
        return self.classes_[scores.argmax(axis=1)]
