from __future__ import annotations

import numpy as np

from .problem import OneHotProblem

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_all(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def local_operator(single_qubit_op: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Return an n-qubit operator.

    Educational convention: qubit 0 is the left-most bit in |x0 x1 ...>.
    """
    if not 0 <= qubit < n:
        raise ValueError("Invalid qubit index")
    return _kron_all(
        [single_qubit_op if i == qubit else I2 for i in range(n)]
    )


def paulis(n: int):
    xs = [local_operator(X, i, n) for i in range(n)]
    ys = [local_operator(Y, i, n) for i in range(n)]
    zs = [local_operator(Z, i, n) for i in range(n)]
    return xs, ys, zs


def hamming_weight_operator(n: int) -> np.ndarray:
    """N = sum_i (I - Z_i)/2."""
    _, _, zs = paulis(n)
    ident = np.eye(2**n, dtype=complex)
    return sum((ident - zi) / 2 for zi in zs)


def objective_hamiltonian(problem: OneHotProblem) -> np.ndarray:
    """Hamiltonian for C(x)=sum_i c_i x_i under x_i=(I-Z_i)/2."""
    _, _, zs = paulis(problem.n)
    ident = np.eye(2**problem.n, dtype=complex)
    H = np.zeros_like(ident)
    for c, zi in zip(problem.costs, zs):
        H += c * (ident - zi) / 2
    return H


def penalty_hamiltonian(problem: OneHotProblem, lam: float) -> np.ndarray:
    """Diagonal Hamiltonian for C(x)+lam(sum_i x_i - 1)^2."""
    if lam < 0:
        raise ValueError("Penalty strength must be non-negative")
    diagonal = [
        problem.penalized_objective(x, lam)
        for x in problem.all_bitstrings()
    ]
    return np.diag(np.asarray(diagonal, dtype=float)).astype(complex)


def x_mixer(n: int) -> np.ndarray:
    xs, _, _ = paulis(n)
    return sum(xs)


def xy_mixer(n: int, edges: tuple[tuple[int, int], ...] | None = None) -> np.ndarray:
    """XY mixer on the supplied mixer graph.

    By default uses a nearest-neighbour line graph:
        0 -- 1 -- ... -- n-1
    """
    if edges is None:
        edges = tuple((i, i + 1) for i in range(n - 1))

    xs, ys, _ = paulis(n)
    H = np.zeros((2**n, 2**n), dtype=complex)

    for i, j in edges:
        if i == j or not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"Invalid mixer edge {(i, j)}")
        H += xs[i] @ xs[j] + ys[i] @ ys[j]

    return H
