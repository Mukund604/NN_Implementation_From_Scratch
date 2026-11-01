import numpy as np
import pytest

from scratchml.nn import (Adam, BatchNorm, Conv2D, Dense, Dropout, Flatten, MaxPool2D,
                          ReLU, RMSProp, SGD, Sequential, Tanh)


def numeric_check(model, x, y, samples=40, eps=1e-6, seed=0):
    """Compare hand written gradients with central finite differences."""
    rng = np.random.default_rng(seed)
    model.loss_and_gradients(x, y, training=False)
    for p, g in model.params():
        analytic = g.copy()
        for _ in range(samples):
            i = tuple(rng.integers(0, s) for s in p.shape)
            original = p[i]
            p[i] = original + eps
            up = model.loss_and_gradients(x, y, training=False)
            p[i] = original - eps
            down = model.loss_and_gradients(x, y, training=False)
            p[i] = original
            numeric = (up - down) / (2 * eps)
            assert analytic[i] == pytest.approx(numeric, abs=1e-5, rel=1e-3)


def test_dense_gradients():
    rng = np.random.default_rng(0)
    model = Sequential([Dense(4, 8, l2=1e-3, rng=rng), Tanh(), Dense(8, 3, rng=rng)])
    numeric_check(model, rng.normal(size=(16, 4)), rng.integers(0, 3, 16))


def test_cnn_gradients():
    rng = np.random.default_rng(1)
    model = Sequential([Conv2D(1, 3, rng=rng), Tanh(), MaxPool2D(), Conv2D(3, 2, l2=1e-3, rng=rng),
                        Tanh(), Flatten(), Dense(2 * 3 * 3, 3, rng=rng)])
    numeric_check(model, rng.normal(size=(6, 1, 6, 6)), rng.integers(0, 3, 6))


def test_input_gradient_through_conv_and_pool():
    """backward() must also return the correct gradient with respect to the input."""
    rng = np.random.default_rng(2)
    model = Sequential([Conv2D(1, 2, rng=rng), Tanh(), MaxPool2D(), Flatten(), Dense(2 * 2 * 2, 2, rng=rng)])
    x, y = rng.normal(size=(3, 1, 4, 4)), rng.integers(0, 2, 3)
    logits = model.forward(x)
    model.criterion.forward(logits, y)
    grad = model.criterion.backward()
    for layer in reversed(model.layers):
        grad = layer.backward(grad)
    i = (1, 0, 2, 3)
    eps = 1e-6
    x_up, x_down = x.copy(), x.copy()
    x_up[i] += eps
    x_down[i] -= eps
    up = model.criterion.forward(model.forward(x_up), y)
    down = model.criterion.forward(model.forward(x_down), y)
    assert grad[i] == pytest.approx((up - down) / (2 * eps), abs=1e-6)


def test_batchnorm_training_input_gradient():
    rng = np.random.default_rng(3)
    model = Sequential([BatchNorm(3), Dense(3, 2, rng=rng)])
    x, y = rng.normal(size=(8, 3)), rng.integers(0, 2, 8)
    model.criterion.forward(model.forward(x, training=True), y)
    grad = model.criterion.backward()
    for layer in reversed(model.layers):
        grad = layer.backward(grad)
    eps = 1e-6
    x_up, x_down = x.copy(), x.copy()
    x_up[2, 1] += eps
    x_down[2, 1] -= eps
    up = model.criterion.forward(model.forward(x_up, training=True), y)
    down = model.criterion.forward(model.forward(x_down, training=True), y)
    assert grad[2, 1] == pytest.approx((up - down) / (2 * eps), abs=1e-5)


def test_dropout_keeps_expected_scale_and_is_off_at_inference():
    layer = Dropout(0.5, rng=np.random.default_rng(0))
    x = np.ones((20000, 1))
    assert layer.forward(x, training=True).mean() == pytest.approx(1.0, abs=0.05)
    assert np.array_equal(layer.forward(x, training=False), x)


@pytest.mark.parametrize("optimizer", [SGD(0.5, momentum=0.9), RMSProp(0.01), Adam(0.01)])
def test_optimizers_learn_xor(optimizer):
    rng = np.random.default_rng(0)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = Sequential([Dense(2, 16, rng=rng), ReLU(), Dense(16, 2, rng=rng)])
    model.fit(x, y, optimizer, epochs=1000, batch_size=4)
    assert (model.predict(x) == y).all()
