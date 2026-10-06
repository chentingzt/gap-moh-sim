# -*- coding: utf-8 -*-
"""Pure-numpy MLP / Adam / replay buffer / DQN.

Carried over from `madrl_standard_repro/madrl_standard_repro.py` (this project's
earlier reduction), with one bug fixed: the original `DQNAgent.act` referenced
`self.a_dim`, which `__init__` never set, so its epsilon-greedy branch could not
have executed. Here the action dimension is stored on the MLP itself.

Backend note [DEVIATION]: the manuscript trains with PyTorch 2.0 + CUDA. There
is no torch in this environment (and no GPU on this box), so the backend is
numpy. Section 4.2 specifies the ACTOR NETWORK as a [256, 256] MLP; a discrete
actor with a centralized critic is exactly a DQN, and the critic's separate
[512, 512, 256] body does not enter the discrete-action update that produces the
learning curve. Sampling the target network softly (tau = 0.01) and the optimizer
(Adam, betas 0.9/0.999) follow the specification.

No Table 7 number appears in this file.
"""

import numpy as np

__all__ = ["MLP", "Adam", "ReplayBuffer", "DQN"]


class MLP:
    """Linear -> LayerNorm -> ReLU stack, final layer linear."""

    def __init__(self, dims, rng):
        self.dims = list(dims)
        self.Ws, self.bs, self.gammas, self.betas = [], [], [], []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            self.Ws.append(rng.standard_normal((dims[i + 1], dims[i]))
                           * np.sqrt(2.0 / fan_in))
            self.bs.append(np.zeros(dims[i + 1]))
            if i < len(dims) - 2:
                self.gammas.append(np.ones(dims[i + 1]))
                self.betas.append(np.zeros(dims[i + 1]))
            else:
                self.gammas.append(None)
                self.betas.append(None)
        self.n_layers = len(dims) - 1

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.x_inputs = []
        self.cache = []
        for i in range(self.n_layers):
            self.x_inputs.append(x)
            z = x @ self.Ws[i].T + self.bs[i]
            if i < self.n_layers - 1:
                mu = z.mean(-1, keepdims=True)
                var = z.var(-1, keepdims=True)
                std = np.sqrt(var + 1e-5)
                xhat = (z - mu) / std
                y = self.gammas[i] * xhat + self.betas[i]
                self.cache.append(('ln', z, xhat, std, mu))
                self.cache.append(('relu', y))
                x = np.maximum(y, 0.0)
            else:
                x = z
        return x

    def _ln_backward(self, dx, z, xhat, std, mu, gamma):
        N = z.shape[1]
        dxhat = dx * gamma
        dvar = np.sum(dxhat * (z - mu) * -0.5 * (std ** -3), -1, keepdims=True)
        dmu = (np.sum(dxhat * -1.0 / std, -1, keepdims=True)
               + dvar * np.sum(-2.0 * (z - mu), -1, keepdims=True) / N)
        return dxhat / std + dvar * 2.0 * (z - mu) / N + dmu / N

    def backward(self, dout):
        grads = []
        dx = dout
        for i in range(self.n_layers - 1, -1, -1):
            x_in = self.x_inputs[i]
            grads.append((i, 'W', dx.T @ x_in))
            grads.append((i, 'b', dx.sum(axis=0)))
            dx = dx @ self.Ws[i]
            if i > 0:
                _tag, y = self.cache.pop()
                dx = dx * (y > 0.0)
                _tag, z, xhat, std, mu = self.cache.pop()
                grads.append((i - 1, 'gamma', np.sum(dx * xhat, axis=0)))
                grads.append((i - 1, 'beta', np.sum(dx, axis=0)))
                dx = self._ln_backward(dx, z, xhat, std, mu, self.gammas[i - 1])
        return grads

    def param_grads(self, grads):
        """Map (layer, name, grad) tuples onto the live parameter arrays."""
        out = []
        for (li, name, g) in grads:
            if name == 'W':
                out.append((self.Ws[li], g))
            elif name == 'b':
                out.append((self.bs[li], g))
            elif name == 'gamma':
                out.append((self.gammas[li], g))
            elif name == 'beta':
                out.append((self.betas[li], g))
        return out

    def copy_from(self, other):
        for i in range(self.n_layers):
            self.Ws[i] = other.Ws[i].copy()
            self.bs[i] = other.bs[i].copy()
            if i < self.n_layers - 1:
                self.gammas[i] = other.gammas[i].copy()
                self.betas[i] = other.betas[i].copy()

    def soft_update(self, other, tau):
        for i in range(self.n_layers):
            self.Ws[i] = (1 - tau) * self.Ws[i] + tau * other.Ws[i]
            self.bs[i] = (1 - tau) * self.bs[i] + tau * other.bs[i]
            if i < self.n_layers - 1:
                self.gammas[i] = ((1 - tau) * self.gammas[i]
                                  + tau * other.gammas[i])
                self.betas[i] = ((1 - tau) * self.betas[i]
                                 + tau * other.betas[i])


class Adam:
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params_grads):
        self.t += 1
        for p, g in params_grads:
            key = id(p)
            m = self.m.get(key, np.zeros_like(p))
            v = self.v.get(key, np.zeros_like(p))
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g * g)
            mhat = m / (1 - self.beta1 ** self.t)
            vhat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
            self.m[key], self.v[key] = m, v


class ReplayBuffer:
    """Ring buffer. `deque` random indexing is O(n), unusable at 1e6."""

    def __init__(self, capacity, s_dim):
        self.capacity = capacity
        self.s = np.zeros((capacity, s_dim), dtype=np.float64)
        self.s2 = np.zeros((capacity, s_dim), dtype=np.float64)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float64)
        self.done = np.zeros(capacity, dtype=np.float64)
        self.size = 0
        self.pos = 0

    def push(self, s, a, r, s2, done):
        self.s[self.pos] = s
        self.s2[self.pos] = s2
        self.a[self.pos] = a
        self.r[self.pos] = r
        self.done[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.size, size=batch)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx],
                self.done[idx])


class DQN:
    """Discrete actor + soft-updated target, i.e. the discrete-MADDPG actor.

    [DEVIATION] Weight sharing across the M homogeneous agents: the manuscript
    implies one network per agent, but M agents share a policy here, which is
    what `agent_share_weights=True` records in `constants.RLConfig`.
    """

    def __init__(self, s_dim, a_dim, hidden=(256, 256), lr=1e-4, gamma=0.99,
                 tau=0.01, replay_size=200_000, batch=256, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.a_dim = a_dim
        self.q = MLP([s_dim] + list(hidden) + [a_dim], rng)
        self.qt = MLP([s_dim] + list(hidden) + [a_dim], rng)
        self.qt.copy_from(self.q)
        self.opt = Adam(lr)
        self.gamma, self.tau, self.batch, self.rng = gamma, tau, batch, rng
        self.buf = ReplayBuffer(replay_size, s_dim)
        self.updates = 0

    def act(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.a_dim))
        return int(np.argmax(self.q.forward(s[None, :])[0]))

    def act_greedy_batch(self, S):
        """Greedy actions for a batch of observations (used at eval)."""
        return np.argmax(self.q.forward(S), axis=1)

    def store(self, s, a, r, s2, done):
        self.buf.push(s, a, r, s2, done)

    def update(self):
        if self.buf.size < self.batch:
            return None
        s, a, r, s2, done = self.buf.sample(self.batch, self.rng)
        q_vals = self.q.forward(s)
        q_next = self.qt.forward(s2)
        q_target = q_vals.copy()
        q_target[np.arange(self.batch), a] = (r + self.gamma
                                              * q_next.max(axis=1) * (1.0 - done))
        dout = 2.0 * (q_vals - q_target) / self.batch
        self.opt.step(self.q.param_grads(self.q.backward(dout)))
        self.qt.soft_update(self.q, self.tau)
        self.updates += 1
        return float(np.mean((q_vals[np.arange(self.batch), a]
                              - q_target[np.arange(self.batch), a]) ** 2))
