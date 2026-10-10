"""Checks the NumPy simulator and model gradients. Run: python -m pytest tests"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qsim import circuit, circuit_forward_backward, circuit_with_grads  # noqa: E402
from models import HybridOrClassic  # noqa: E402

N = 4
_I, _X = np.eye(2), np.array([[0, 1], [1, 0]])
_Z, _P0, _P1 = np.diag([1, -1]), np.diag([1, 0]), np.diag([0, 1])


def _op(m, w):
    out = np.array([[1]])
    for q in range(N):
        out = np.kron(out, m if q == w else _I)
    return out


def _cnot(c, t):
    a, b = np.array([[1]]), np.array([[1]])
    for q in range(N):
        a = np.kron(a, _P0 if q == c else _I)
        b = np.kron(b, _P1 if q == c else (_X if q == t else _I))
    return a + b


def _rx(t):
    return np.cos(t / 2) * _I - 1j * np.sin(t / 2) * _X


def _reference(x, W):
    """Gate-by-gate dense simulation, written independently of qsim."""
    s = np.zeros(2 ** N, complex)
    s[0] = 1
    for q in range(N):
        s = _op(_rx(x[q]), q) @ s
    for layer in W:
        for q in range(N):
            s = _op(_rx(layer[q]), q) @ s
        for q in range(N):
            s = _cnot(q, (q + 1) % N) @ s
    return np.array([np.real(s.conj() @ _op(_Z, q) @ s) for q in range(N)])


def test_matches_dense_reference():
    rng = np.random.default_rng(0)
    x, W = rng.uniform(-3, 3, (5, N)), rng.uniform(0, 2 * np.pi, (2, N))
    expected = np.array([_reference(xi, W) for xi in x])
    assert np.allclose(circuit(x, W), expected, atol=1e-12)


def test_no_layers_gives_cos():
    x = np.random.default_rng(1).uniform(-3, 3, (5, N))
    assert np.allclose(circuit(x, np.zeros((0, N))), np.cos(x), atol=1e-12)


def test_parameter_shift_matches_finite_differences():
    rng = np.random.default_rng(2)
    x, W = rng.uniform(-3, 3, (5, N)), rng.uniform(0, 2 * np.pi, (2, N))
    _, d_in, d_w = circuit_with_grads(x, W)
    eps = 1e-6
    for q in range(N):
        e = np.zeros(N)
        e[q] = eps
        fd = (circuit(x + e, W) - circuit(x - e, W)) / (2 * eps)
        assert np.allclose(fd, d_in[:, :, q], atol=1e-7)
    for l in range(2):
        for q in range(N):
            e = np.zeros_like(W)
            e[l, q] = eps
            fd = (circuit(x, W + e) - circuit(x, W - e)) / (2 * eps)
            assert np.allclose(fd, d_w[:, :, l, q], atol=1e-7)


def test_adjoint_matches_parameter_shift():
    rng = np.random.default_rng(5)
    x, W, dz = rng.uniform(-3, 3, (16, N)), rng.uniform(0, 2 * np.pi, (2, N)), rng.normal(size=(16, N))
    out, d_in, d_w = circuit_with_grads(x, W)
    out2, backward = circuit_forward_backward(x, W)
    a, b = backward(dz)
    assert np.allclose(out, out2, atol=1e-12)
    assert np.allclose(np.einsum("bo,boi->bi", dz, d_in), a, atol=1e-10)
    assert np.allclose(np.einsum("bo,bolq->lq", dz, d_w), b, atol=1e-10)


def test_model_backprop():
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(8, 6)), rng.normal(size=8)
    for use_quantum, squash in [(True, "clamp"), (False, "clamp"), (True, "tanh"), (False, "tanh")]:
        m = HybridOrClassic(6, use_quantum=use_quantum, seed=3, squash=squash)
        m.p["Wo"] = rng.normal(size=(1, 4))
        m.p["bo"] = rng.normal(size=1)
        m.p["W1"] *= 0.5
        _, g = m.loss_and_grads(X, y)
        eps = 1e-6
        for k, v in m.p.items():
            it = np.nditer(v, flags=["multi_index"])
            for _ in it:
                i = it.multi_index
                old = v[i]
                v[i] = old + eps
                a, _ = m.loss_and_grads(X, y)
                v[i] = old - eps
                b, _ = m.loss_and_grads(X, y)
                v[i] = old
                assert abs((a - b) / (2 * eps) - g[k][i]) < 1e-6
