"""Compares every algorithm with its scikit-learn counterpart and writes results/.

Run from the repo root: python -m benchmarks.run_benchmarks
Classifiers use stratified 5 fold cross validation on datasets bundled with scikit-learn.
"""
import json
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, datasets, ensemble, linear_model, mixture, naive_bayes, neighbors, neural_network, svm, tree
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from original.SyntheticData import createSyntheticData
from scratchml.ai import CliffWalking, greedy_path_length, q_learning
from scratchml.nn import Adam, Conv2D, Dense, Dropout, Flatten, MaxPool2D, ReLU, Sequential
from scratchml.supervised import (DecisionTree, GaussianNB, GradientBoosting, KNN, LinearSVM,
                                  LogisticRegression, RandomForest, RidgeRegression)
from scratchml.unsupervised import GaussianMixture, KMeans

OUT = pathlib.Path("results")
OUT.mkdir(exist_ok=True)


class MLP:
    def __init__(self, hidden=128, epochs=60, seed=0):
        self.hidden, self.epochs, self.seed = hidden, epochs, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.model = Sequential([Dense(X.shape[1], self.hidden, l2=1e-4, rng=rng), ReLU(), Dropout(0.1, rng=rng),
                                 Dense(self.hidden, len(self.classes_), rng=rng)])
        self.model.fit(X, y, Adam(0.002), epochs=self.epochs, batch_size=32, seed=self.seed)
        return self

    def predict(self, X):
        return self.classes_[self.model.predict(X)]


PAIRS = {
    "Logistic regression": (lambda: LogisticRegression(), lambda: linear_model.LogisticRegression(max_iter=2000)),
    "Linear SVM": (lambda: LinearSVM(), lambda: svm.LinearSVC()),
    "k nearest neighbors": (lambda: KNN(), lambda: neighbors.KNeighborsClassifier()),
    "Gaussian naive Bayes": (lambda: GaussianNB(), lambda: naive_bayes.GaussianNB()),
    "Decision tree": (lambda: DecisionTree(), lambda: tree.DecisionTreeClassifier(random_state=0)),
    "Random forest": (lambda: RandomForest(n_trees=100), lambda: ensemble.RandomForestClassifier(random_state=0)),
    "Gradient boosting": (lambda: GradientBoosting(n_rounds=100),
                          lambda: ensemble.GradientBoostingClassifier(random_state=0)),
    "Neural network (MLP)": (lambda: MLP(), lambda: neural_network.MLPClassifier((128,), max_iter=300, random_state=0)),
}
DATA = {"Digits": datasets.load_digits, "Breast cancer": datasets.load_breast_cancer, "Wine": datasets.load_wine}


def cross_val(make, X, y):
    scores, start = [], time.time()
    for train, test in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        scaler = StandardScaler().fit(X[train])
        scores.append(np.mean(make().fit(scaler.transform(X[train]), y[train]).predict(scaler.transform(X[test])) == y[test]))
    return float(np.mean(scores)), float(np.std(scores)), time.time() - start


def classification():
    rows = []
    for algo, (ours, theirs) in PAIRS.items():
        for name, load in DATA.items():
            X, y = load(return_X_y=True)
            a, a_sd, a_t = cross_val(ours, X, y)
            b, b_sd, b_t = cross_val(theirs, X, y)
            rows.append(dict(algorithm=algo, dataset=name, scratch=a, scratch_sd=a_sd, sklearn=b, sklearn_sd=b_sd,
                             scratch_seconds=a_t, sklearn_seconds=b_t))
            print(f"{algo:24s}{name:15s}{a:.4f}  sklearn {b:.4f}  ({a_t:.1f}s vs {b_t:.1f}s)", flush=True)
    return rows


def cnn_digits():
    """Trains the from scratch CNN on the 8x8 digits images."""
    scores = []
    X, y = datasets.load_digits(return_X_y=True)
    X = X.reshape(-1, 1, 8, 8) / 16.0
    for seed, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)):
        rng = np.random.default_rng(seed)
        model = Sequential([Conv2D(1, 16, rng=rng), ReLU(), MaxPool2D(), Conv2D(16, 32, rng=rng), ReLU(), MaxPool2D(),
                            Flatten(), Dropout(0.25, rng=rng), Dense(32 * 2 * 2, 10, rng=rng)])
        model.fit(X[train], y[train], Adam(0.002), epochs=25, batch_size=32, seed=seed)
        scores.append(np.mean(model.predict(X[test]) == y[test]))
        print("cnn fold", seed, round(scores[-1], 4), flush=True)
    return float(np.mean(scores)), float(np.std(scores))


def regression_and_clustering():
    X, y = datasets.load_diabetes(return_X_y=True)
    r2_ours, r2_theirs = [], []
    for train, test in KFold(5, shuffle=True, random_state=0).split(X):
        r2_ours.append(r2_score(y[test], RidgeRegression(1.0).fit(X[train], y[train]).predict(X[test])))
        r2_theirs.append(r2_score(y[test], linear_model.Ridge(1.0).fit(X[train], y[train]).predict(X[test])))
    X, y = datasets.load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return dict(
        ridge_r2=[float(np.mean(r2_ours)), float(np.mean(r2_theirs))],
        kmeans_ari=[adjusted_rand_score(y, KMeans(3).fit(X).labels_),
                    adjusted_rand_score(y, cluster.KMeans(3, n_init=10, random_state=0).fit(X).labels_)],
        gmm_ari=[adjusted_rand_score(y, GaussianMixture(3).fit(X).labels_),
                 adjusted_rand_score(y, mixture.GaussianMixture(3, random_state=0).fit_predict(X))],
    )


def random_search(X, y, seed=0, iterations=10000):
    """The original training loop from original/NN.py, reproduced so both can be plotted."""
    rng = np.random.default_rng(seed)
    w1, b1 = 0.1 * rng.normal(size=(2, 3)), np.zeros(3)
    w2, b2 = 0.1 * rng.normal(size=(3, 3)), np.zeros(3)

    def logits(points, a, b, c, d):
        return np.maximum(0, points @ a + b) @ c + d

    def loss(a, b, c, d):
        z = logits(X, a, b, c, d)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1)))

    best = loss(w1, b1, w2, b2)
    for _ in range(iterations):
        c1, c2 = w1 + 0.05 * rng.normal(size=w1.shape), b1 + 0.05 * rng.normal(size=b1.shape)
        c3, c4 = w2 + 0.05 * rng.normal(size=w2.shape), b2 + 0.05 * rng.normal(size=b2.shape)
        value = loss(c1, c2, c3, c4)
        if value < best:
            best, w1, b1, w2, b2 = value, c1, c2, c3, c4
    return lambda points: logits(points, w1, b1, w2, b2).argmax(axis=1)


def spiral_figures():
    np.random.seed(0)
    X, y = createSyntheticData(100, 3)
    before = random_search(X, y)
    rng = np.random.default_rng(0)
    net = Sequential([Dense(2, 64, rng=rng), ReLU(), Dense(64, 64, rng=rng), ReLU(), Dense(64, 3, rng=rng)])
    curve = net.fit(X, y, Adam(0.01), epochs=800, batch_size=32)
    accs = dict(random_search=float(np.mean(before(X) == y)), backprop=float(np.mean(net.predict(X) == y)))

    colors = ["#c2410c", "#0f766e", "#1d4ed8"]
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 400), np.linspace(-1.2, 1.2, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]

    def panel(ax, predict, title):
        ax.contourf(xx, yy, predict(grid).reshape(xx.shape), levels=[-.5, .5, 1.5, 2.5], colors=colors, alpha=.18)
        for k, c in enumerate(colors):
            ax.scatter(*X[y == k].T, s=9, color=c)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), dpi=160)
    panel(axes[0], before, f"Original: random search, {accs['random_search']:.1%} train accuracy")
    panel(axes[1], net.predict, f"Now: backpropagation with Adam, {accs['backprop']:.1%} train accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "spiral_before_after.png")
    plt.close(fig)

    models = [("Logistic regression", LogisticRegression()), ("Linear SVM", LinearSVM()), ("k nearest neighbors", KNN()),
              ("Gaussian naive Bayes", GaussianNB()), ("Decision tree", DecisionTree()),
              ("Random forest", RandomForest(n_trees=50)), ("Gradient boosting", GradientBoosting(n_rounds=60)),
              ("Neural network", None)]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8), dpi=150)
    for ax, (name, model) in zip(axes.ravel(), models):
        predict = net.predict if model is None else model.fit(X, y).predict
        panel(ax, predict, f"{name}: {np.mean(predict(X) == y):.0%}")
    fig.suptitle("Train accuracy on 3 class spiral data. Straight line models cannot separate it.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "classifier_gallery.png")
    plt.close(fig)
    return accs, len(curve)


def rl():
    env = CliffWalking()
    runs = [q_learning(env, episodes=500, seed=s) for s in range(5)]
    return dict(greedy_steps=[greedy_path_length(env, Q) for Q, _ in runs],
                last_50_mean_return=float(np.mean([np.mean(r[-50:]) for _, r in runs])))


if __name__ == "__main__":
    results = dict(spiral=spiral_figures()[0])
    print(results, flush=True)
    results["rl"] = rl()
    results["other"] = regression_and_clustering()
    print(results, flush=True)
    results["cnn_digits"] = cnn_digits()
    results["classification"] = classification()
    (OUT / "benchmarks.json").write_text(json.dumps(results, indent=2))
    print("done")
