"""Each algorithm is checked against scikit-learn or a known answer."""
import numpy as np
import pytest
from sklearn import datasets, decomposition, linear_model
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scratchml.ai import CliffWalking, alphabeta, astar, greedy_path_length, q_learning, winner
from scratchml.supervised import (DecisionTree, GaussianNB, GradientBoosting, KNN, LinearSVM,
                                  LogisticRegression, RandomForest, RidgeRegression)
from scratchml.unsupervised import GaussianMixture, KMeans, PCA


@pytest.fixture(scope="module")
def wine():
    X, y = datasets.load_wine(return_X_y=True)
    a, b, c, d = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    scaler = StandardScaler().fit(a)
    return scaler.transform(a), scaler.transform(b), c, d


@pytest.mark.parametrize("model", [LogisticRegression(), LinearSVM(), KNN(), GaussianNB(),
                                   DecisionTree(), RandomForest(n_trees=30),
                                   GradientBoosting(n_rounds=30)], ids=lambda m: type(m).__name__)
def test_classifiers_on_wine(model, wine):
    a, b, c, d = wine
    assert np.mean(model.fit(a, c).predict(b) == d) > 0.88


@pytest.mark.parametrize("alpha", [0.0, 5.0])
def test_ridge_matches_sklearn(alpha):
    X, y = datasets.load_diabetes(return_X_y=True)
    ours = RidgeRegression(alpha).fit(X, y)
    theirs = linear_model.Ridge(alpha=alpha).fit(X, y) if alpha else linear_model.LinearRegression().fit(X, y)
    assert ours.coef_ == pytest.approx(theirs.coef_, rel=1e-5, abs=1e-5)
    assert ours.intercept_ == pytest.approx(theirs.intercept_, rel=1e-5)


def test_decision_tree_memorizes_training_data():
    X, y = datasets.load_iris(return_X_y=True)
    assert np.mean(DecisionTree().fit(X, y).predict(X) == y) > 0.99


def test_pca_matches_sklearn():
    X = datasets.load_digits().data
    ours = PCA(5).fit(X)
    theirs = decomposition.PCA(5).fit(X)
    assert ours.explained_variance_ratio_ == pytest.approx(theirs.explained_variance_ratio_)
    assert np.abs(ours.transform(X)) == pytest.approx(np.abs(theirs.transform(X)), abs=1e-6)


def test_clustering_recovers_blobs():
    X, y = datasets.make_blobs(600, centers=4, cluster_std=0.6, random_state=0)
    assert adjusted_rand_score(y, KMeans(4).fit(X).labels_) > 0.95
    assert adjusted_rand_score(y, GaussianMixture(4).fit(X).labels_) > 0.95


def test_astar_finds_shortest_path_around_a_wall():
    grid = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
    path = astar(grid, (0, 0), (2, 0))
    assert len(path) - 1 == 6
    assert astar([[0, 1, 0]], (0, 0), (0, 2)) is None


def test_alphabeta_perfect_play():
    assert alphabeta([0] * 9, 1)[0] == 0  # tic tac toe is a draw with perfect play
    assert alphabeta([1, 1, 0, -1, -1, 0, 0, 0, 0], 1) == (1, 2)  # takes the win
    # Perfect play never loses to a random opponent.
    rng = np.random.default_rng(0)
    for game in range(50):
        board, player = [0] * 9, 1 if game % 2 else -1
        while not winner(board) and 0 in board:
            if player == 1:
                move = alphabeta(board, 1)[1]
            else:
                move = int(rng.choice([i for i, v in enumerate(board) if v == 0]))
            board[move] = player
            player = -player
        assert winner(board) != -1


def test_q_learning_finds_optimal_cliff_path():
    env = CliffWalking()
    Q, _ = q_learning(env, episodes=500)
    assert greedy_path_length(env, Q) == 13
