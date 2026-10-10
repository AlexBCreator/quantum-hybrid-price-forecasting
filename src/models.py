"""Hybrid quantum-classical model and its classical twin, written in NumPy.

Architecture mirrors QuantumTransformerModel from
notebooks/02_regression_hybrid_vs_classic.ipynb:

    x (window * features) -> Linear(D, 4) -> clamp[-1, 1]
        hybrid:  * pi -> 4-qubit circuit (AngleEmbedding + 2 BasicEntanglerLayers) -> <Z>
        classic: Linear(4, 4)
    -> Linear(4, 1)  (zero-initialised so both models start from the same prediction)

Training: MSE loss, Adam, gradient-norm clipping 1.0 — same as the notebook.
Quantum-layer gradients: adjoint differentiation (equal to parameter-shift, see tests).
Initialisation follows PyTorch / PennyLane defaults.
"""
import numpy as np

from qsim import circuit, circuit_forward_backward


class Adam:
    def __init__(self, params, lr=0.002, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads, clip=1.0):
        norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-6)) if clip else 1.0
        self.t += 1
        for k, g in grads.items():
            g = g * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class HybridOrClassic:
    def __init__(self, input_dim, use_quantum=True, n_qubits=4, n_layers=2, seed=0, squash="clamp"):
        rng = np.random.default_rng(seed)
        self.q = use_quantum
        self.squash = squash          # "clamp" as in the notebook, or "tanh" (no dead gradients)
        self.rng = rng
        k = 1 / np.sqrt(input_dim)
        self.p = {
            "W1": rng.uniform(-k, k, (n_qubits, input_dim)),
            "b1": rng.uniform(-k, k, n_qubits),
            "Wo": np.zeros((1, n_qubits)),
            "bo": np.zeros(1),
        }
        if use_quantum:
            self.p["Wq"] = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits))
        else:
            k2 = 1 / np.sqrt(n_qubits)
            self.p["W2"] = rng.uniform(-k2, k2, (n_qubits, n_qubits))
            self.p["b2"] = rng.uniform(-k2, k2, n_qubits)

    def _squash(self, pre):
        if self.squash == "tanh":
            h = np.tanh(pre)
            return h, 1 - h ** 2
        return np.clip(pre, -1, 1), ((pre >= -1) & (pre <= 1)).astype(float)

    def predict(self, X):
        p = self.p
        h, _ = self._squash(X @ p["W1"].T + p["b1"])
        z = circuit(h * np.pi, p["Wq"]) if self.q else h @ p["W2"].T + p["b2"]
        return (z @ p["Wo"].T + p["bo"]).ravel()

    def loss_and_grads(self, X, y):
        p = self.p
        pre = X @ p["W1"].T + p["b1"]
        h, mask = self._squash(pre)
        if self.q:
            z, q_backward = circuit_forward_backward(h * np.pi, p["Wq"])
        else:
            z = h @ p["W2"].T + p["b2"]
        out = (z @ p["Wo"].T + p["bo"]).ravel()
        B = len(y)
        dy = (2.0 / B) * (out - y)[:, None]           # (B, 1)
        g = {"Wo": dy.T @ z, "bo": dy.sum(0)}
        dz = dy @ p["Wo"]                             # (B, 4)
        if self.q:
            d_in, g["Wq"] = q_backward(dz)
            dh = d_in * np.pi
        else:
            g["W2"] = dz.T @ h
            g["b2"] = dz.sum(0)
            dh = dz @ p["W2"]
        dh = dh * mask
        g["W1"] = dh.T @ X
        g["b1"] = dh.sum(0)
        return float(np.mean((out - y) ** 2)), g

    def fit(self, X, y, epochs=35, batch_size=16, lr=0.002, shuffle=False):
        opt = Adam(self.p, lr=lr)
        n_batches = len(X) // batch_size              # drop_last=True
        for _ in range(epochs):
            order = self.rng.permutation(len(X)) if shuffle else np.arange(len(X))
            for b in range(n_batches):
                sl = order[b * batch_size:(b + 1) * batch_size]
                _, g = self.loss_and_grads(X[sl], y[sl])
                opt.step(g, clip=1.0)
        return self
