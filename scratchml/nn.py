"""Neural networks with hand written backpropagation.

Every layer implements forward() and backward(). backward() takes the gradient
of the loss with respect to the layer's output and returns the gradient with
respect to its input, which is the chain rule applied one layer at a time.
Images use the (batch, channels, height, width) layout.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Layer:
    def params(self):
        """Return (parameter, gradient) pairs. Arrays are updated in place."""
        return []

    def regularization_loss(self):
        return 0.0


class Dense(Layer):
    def __init__(self, n_in, n_out, l2=0.0, rng=None):
        rng = rng or np.random.default_rng()
        # He initialization keeps activation variance stable through ReLU layers.
        self.w = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)
        self.l2 = l2
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=False):
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad):
        self.dw[:] = self.x.T @ grad + 2 * self.l2 * self.w
        self.db[:] = grad.sum(axis=0)
        return grad @ self.w.T

    def params(self):
        return [(self.w, self.dw), (self.b, self.db)]

    def regularization_loss(self):
        return self.l2 * np.sum(self.w**2)


class ReLU(Layer):
    def forward(self, x, training=False):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Tanh(Layer):
    def forward(self, x, training=False):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out**2)


class Dropout(Layer):
    def __init__(self, rate, rng=None):
        self.rate = rate
        self.rng = rng or np.random.default_rng()

    def forward(self, x, training=False):
        if not training or self.rate == 0:
            self.mask = np.ones_like(x)
            return x
        # Rescale the survivors so the expected activation stays the same.
        self.mask = (self.rng.random(x.shape) >= self.rate) / (1 - self.rate)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class BatchNorm(Layer):
    """Batch normalization for (batch, features) inputs."""

    def __init__(self, n, momentum=0.9, eps=1e-5):
        self.gamma, self.beta = np.ones(n), np.zeros(n)
        self.dgamma, self.dbeta = np.zeros(n), np.zeros(n)
        self.running_mean, self.running_var = np.zeros(n), np.ones(n)
        self.momentum, self.eps = momentum, eps

    def forward(self, x, training=False):
        if training:
            mean, var = x.mean(axis=0), x.var(axis=0)
            m = self.momentum
            self.running_mean = m * self.running_mean + (1 - m) * mean
            self.running_var = m * self.running_var + (1 - m) * var
        else:
            mean, var = self.running_mean, self.running_var
        self.training = training
        self.mean, self.var = mean, var
        self.std = np.sqrt(var + self.eps)
        self.x = x
        self.z = (x - mean) / self.std
        return self.gamma * self.z + self.beta

    def backward(self, grad):
        self.dgamma[:] = (grad * self.z).sum(axis=0)
        self.dbeta[:] = grad.sum(axis=0)
        g = grad * self.gamma
        if not self.training:
            return g / self.std
        # The batch mean and variance also depend on x, so they join the chain rule.
        n = len(self.x)
        centered = self.x - self.mean
        g_mean = g.mean(axis=0)
        dz_dvar = -0.5 * self.z / (self.var + self.eps)
        return (g - g_mean) / self.std + 2 * centered * (g * dz_dvar).sum(axis=0) / n


class Conv2D(Layer):
    """2D convolution with same padding. kernel must be odd."""

    def __init__(self, in_channels, out_channels, kernel=3, l2=0.0, rng=None):
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        rng = rng or np.random.default_rng()
        scale = np.sqrt(2 / (in_channels * kernel**2))
        self.w = rng.normal(0, scale, (out_channels, in_channels, kernel, kernel))
        self.b = np.zeros(out_channels)
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.k, self.pad, self.l2 = kernel, kernel // 2, l2

    def forward(self, x, training=False):
        p = self.pad
        self.padded_shape = (x.shape[0], x.shape[1], x.shape[2] + 2 * p, x.shape[3] + 2 * p)
        padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        self.windows = sliding_window_view(padded, (self.k, self.k), axis=(2, 3))
        out = np.einsum("ncijkl,ockl->noij", self.windows, self.w, optimize=True)
        return out + self.b[None, :, None, None]

    def backward(self, grad):
        self.dw[:] = np.einsum("ncijkl,noij->ockl", self.windows, grad, optimize=True)
        self.dw += 2 * self.l2 * self.w
        self.db[:] = grad.sum(axis=(0, 2, 3))
        per_window = np.einsum("noij,ockl->ncijkl", grad, self.w, optimize=True)
        padded = np.zeros(self.padded_shape)
        h, w = grad.shape[2:]
        for k in range(self.k):
            for l in range(self.k):
                padded[:, :, k:k + h, l:l + w] += per_window[:, :, :, :, k, l]
        p = self.pad
        return padded[:, :, p:-p, p:-p] if p else padded

    def params(self):
        return [(self.w, self.dw), (self.b, self.db)]

    def regularization_loss(self):
        return self.l2 * np.sum(self.w**2)


class MaxPool2D(Layer):
    """2x2 max pooling. Height and width must be even."""

    def forward(self, x, training=False):
        n, c, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError("MaxPool2D needs even height and width")
        self.shape = x.shape
        blocks = x.reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 2, 4, 3, 5)
        self.block_shape = blocks.shape
        blocks = blocks.reshape(n, c, h // 2, w // 2, 4)
        self.idx = blocks.argmax(axis=-1, keepdims=True)
        return np.take_along_axis(blocks, self.idx, axis=-1).squeeze(-1)

    def backward(self, grad):
        n, c, h, w = self.shape
        blocks = np.zeros((n, c, h // 2, w // 2, 4))
        np.put_along_axis(blocks, self.idx, grad[..., None], axis=-1)
        blocks = blocks.reshape(self.block_shape).transpose(0, 1, 2, 4, 3, 5)
        return blocks.reshape(self.shape)


class Flatten(Layer):
    def forward(self, x, training=False):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, grad):
        return grad.reshape(self.shape)


class SoftmaxCrossEntropy:
    """Softmax plus cross entropy, computed from logits for numerical stability."""

    def forward(self, logits, y):
        shifted = logits - logits.max(axis=1, keepdims=True)
        self.probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        self.y = y
        return -np.mean(np.log(self.probs[np.arange(len(y)), y] + 1e-12))

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(len(self.y)), self.y] -= 1
        return grad / len(self.y)


class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr, self.momentum, self.velocity = lr, momentum, {}

    def step(self, params):
        for p, g in params:
            v = self.velocity.setdefault(id(p), np.zeros_like(p))
            v[:] = self.momentum * v - self.lr * g
            p += v


class RMSProp:
    def __init__(self, lr=0.001, decay=0.9, eps=1e-8):
        self.lr, self.decay, self.eps, self.cache = lr, decay, eps, {}

    def step(self, params):
        for p, g in params:
            c = self.cache.setdefault(id(p), np.zeros_like(p))
            c[:] = self.decay * c + (1 - self.decay) * g**2
            p -= self.lr * g / (np.sqrt(c) + self.eps)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m, self.s, self.t = {}, {}, 0

    def step(self, params):
        self.t += 1
        for p, g in params:
            m = self.m.setdefault(id(p), np.zeros_like(p))
            s = self.s.setdefault(id(p), np.zeros_like(p))
            m[:] = self.b1 * m + (1 - self.b1) * g
            s[:] = self.b2 * s + (1 - self.b2) * g**2
            m_hat = m / (1 - self.b1**self.t)
            s_hat = s / (1 - self.b2**self.t)
            p -= self.lr * m_hat / (np.sqrt(s_hat) + self.eps)


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.criterion = SoftmaxCrossEntropy()

    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def loss_and_gradients(self, x, y, training=True):
        loss = self.criterion.forward(self.forward(x, training=training), y)
        loss += sum(layer.regularization_loss() for layer in self.layers)
        grad = self.criterion.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return loss

    def params(self):
        return [pair for layer in self.layers for pair in layer.params()]

    def fit(self, x, y, optimizer, epochs=100, batch_size=32, seed=0):
        rng = np.random.default_rng(seed)
        history = []
        for _ in range(epochs):
            order = rng.permutation(len(x))
            total = 0.0
            for start in range(0, len(x), batch_size):
                batch = order[start:start + batch_size]
                total += self.loss_and_gradients(x[batch], y[batch]) * len(batch)
                optimizer.step(self.params())
            history.append(total / len(x))
        return history

    def predict(self, x, batch_size=256):
        out = [self.forward(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]
        return np.concatenate(out).argmax(axis=1)
