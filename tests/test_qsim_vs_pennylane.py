"""Compares the NumPy simulator with PennyLane. Skipped if PennyLane is not installed.

Run locally:  pip install pennylane pytest && python -m pytest tests
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qsim import circuit  # noqa: E402

qml = pytest.importorskip("pennylane")


def test_matches_pennylane():
    n, L = 4, 2
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n), rotation="X")
        qml.BasicEntanglerLayers(weights, wires=range(n))
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    rng = np.random.default_rng(0)
    W = rng.uniform(0, 2 * np.pi, (L, n))
    for x in rng.uniform(-np.pi, np.pi, (10, n)):
        expected = np.array(qnode(x, W), dtype=float)
        assert np.allclose(circuit(x[None, :], W)[0], expected, atol=1e-8)
