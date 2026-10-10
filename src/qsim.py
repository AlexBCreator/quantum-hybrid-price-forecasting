"""Minimal NumPy statevector simulator for the 4-qubit circuit used in the project.

Circuit (identical to the PennyLane QNode in notebooks/02_regression_hybrid_vs_classic.ipynb):

    qml.AngleEmbedding(inputs, wires=range(n), rotation="X")
    qml.BasicEntanglerLayers(weights, wires=range(n))      # RX on each wire + CNOT ring
    return [qml.expval(qml.PauliZ(i)) for i in range(n)]

How it is computed:
  * AngleEmbedding of |0...0> gives a product state, built with Kronecker products.
  * BasicEntanglerLayers do not depend on the data, so they are one 2^n x 2^n unitary
    shared by the whole batch.
  * Gradients are exact, via the parameter-shift rule (the method used on real
    quantum hardware): df/dθ = [f(θ + π/2) - f(θ - π/2)] / 2.

tests/test_qsim.py checks the simulator against a dense reference and finite
differences; tests/test_qsim_vs_pennylane.py checks it against PennyLane itself.
"""
import numpy as np

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_P0 = np.diag([1, 0]).astype(complex)
_P1 = np.diag([0, 1]).astype(complex)


def _kron_all(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def _rx(theta):
    return np.cos(theta / 2) * _I - 1j * np.sin(theta / 2) * _X


def _cnot(control, target, n):
    a = _kron_all([_P0 if q == control else _I for q in range(n)])
    b = _kron_all([_P1 if q == control else (_X if q == target else _I) for q in range(n)])
    return a + b


_CNOT_RING = {}


def _cnot_ring(n):
    if n not in _CNOT_RING:
        U = np.eye(2 ** n, dtype=complex)
        pairs = [(0, 1)] if n == 2 else ([(q, (q + 1) % n) for q in range(n)] if n > 2 else [])
        for c, t in pairs:
            U = _cnot(c, t, n) @ U
        _CNOT_RING[n] = U
    return _CNOT_RING[n]


def entangler_unitary(weights):
    """Unitary of BasicEntanglerLayers(weights). weights: (L, n)."""
    L, n = weights.shape
    U = np.eye(2 ** n, dtype=complex)
    ring = _cnot_ring(n)
    for layer in weights:
        U = ring @ _kron_all([_rx(w) for w in layer]) @ U
    return U


def embed(inputs):
    """AngleEmbedding(rotation='X') of |0..0>. inputs: (B, n) -> states (B, 2**n)."""
    B, n = inputs.shape
    c = np.cos(inputs / 2)
    s = -1j * np.sin(inputs / 2)
    state = np.ones((B, 1), dtype=complex)
    for q in range(n):  # wire 0 = most significant bit (PennyLane convention)
        qubit = np.stack([c[:, q], s[:, q]], axis=1)
        state = (state[:, :, None] * qubit[:, None, :]).reshape(B, -1)
    return state


def _z_signs(n):
    bits = (np.arange(2 ** n)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    return 1.0 - 2.0 * bits  # (2**n, n)


def _expval(states, U):
    psi = states @ U.T
    return (np.abs(psi) ** 2) @ _z_signs(U.shape[0].bit_length() - 1)


def circuit(inputs, weights):
    """inputs: (B, n) angles, weights: (L, n). Returns <Z_i>: (B, n)."""
    return _expval(embed(inputs), entangler_unitary(weights))


def circuit_with_grads(inputs, weights):
    """Returns out (B, n), d_out/d_inputs (B, n_out, n_in), d_out/d_weights (B, n_out, L, n)."""
    B, n = inputs.shape
    U = entangler_unitary(weights)
    out = _expval(embed(inputs), U)
    shift = np.pi / 2
    d_in = np.zeros((B, n, n))
    for q in range(n):
        e = np.zeros(n)
        e[q] = shift
        d_in[:, :, q] = (_expval(embed(inputs + e), U) - _expval(embed(inputs - e), U)) / 2
    L = weights.shape[0]
    d_w = np.zeros((B, n, L, n))
    states = embed(inputs)
    for l in range(L):
        for q in range(n):
            e = np.zeros_like(weights)
            e[l, q] = shift
            d_w[:, :, l, q] = (_expval(states, entangler_unitary(weights + e))
                               - _expval(states, entangler_unitary(weights - e))) / 2
    return out, d_in, d_w


def _full(gate, wire, n):
    return _kron_all([gate if q == wire else _I for q in range(n)])


def circuit_forward_backward(inputs, weights):
    """Fast exact gradients (adjoint differentiation) for training.

    Returns out (B, n) and a function backward(dz) -> (d_inputs (B, n), d_weights (L, n)),
    where dz = dLoss/dout. Gives the same numbers as the parameter-shift rule in
    circuit_with_grads (checked in tests), but needs one backward pass instead of
    2 x (number of parameters) extra circuit runs.
    """
    B, n = inputs.shape
    zs = _z_signs(n)
    psi0 = embed(inputs)
    gates, gens, states = [], [], []          # gate matrices, generators (-i X/2), states after
    s = psi0
    ring = _cnot_ring(n)
    for layer in weights:
        for q in range(n):
            G = _full(_rx(layer[q]), q, n)
            s = s @ G.T
            gates.append(G)
            gens.append(_full(-0.5j * _X, q, n))
            states.append(s)
        s = s @ ring.T
        gates.append(ring)
        gens.append(None)
        states.append(s)
    out = (np.abs(s) ** 2) @ zs

    def backward(dz):
        lam = s * (dz @ zs.T)                 # O|phi> with O = sum_i dz_i Z_i (diagonal)
        d_w = np.zeros(weights.shape)
        idx = len(gates) - 1
        for l in reversed(range(weights.shape[0])):
            # ring (no parameter)
            lam = lam @ np.conj(gates[idx])
            idx -= 1
            for q in reversed(range(n)):
                d_w[l, q] = 2 * np.real(np.sum(np.conj(lam) * (states[idx] @ gens[idx].T)))
                lam = lam @ np.conj(gates[idx])
                idx -= 1
        # lam is now U^dagger O U psi0 ; derivative of the product state w.r.t. each input
        c, sn = np.cos(inputs / 2), np.sin(inputs / 2)
        d_in = np.zeros((B, n))
        for q in range(n):
            st = np.ones((B, 1), dtype=complex)
            for w in range(n):
                if w == q:
                    qubit = np.stack([-0.5 * sn[:, w], -0.5j * c[:, w]], axis=1)
                else:
                    qubit = np.stack([c[:, w], -1j * sn[:, w]], axis=1)
                st = (st[:, :, None] * qubit[:, None, :]).reshape(B, -1)
            d_in[:, q] = 2 * np.real(np.sum(np.conj(lam) * st, axis=1))
        return d_in, d_w

    return out, backward
