"""Cross-checks between the NumPy operators and the Qiskit operators.

The NumPy code uses the educational convention (qubit 0 = left-most bit,
most significant); Qiskit is little-endian (qubit 0 = least significant).
The two matrix representations therefore differ by a qubit-reversal
permutation, which these tests apply explicitly.
"""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from constrained_qaoa import (
    OneHotProblem,
    objective_hamiltonian,
    penalty_hamiltonian,
    xy_mixer,
)
from constrained_qaoa.qiskit_circuits import (
    build_penalty_qaoa,
    build_xy_qaoa,
    objective_cost_operator,
    penalty_cost_operator,
    xy_mixer_operator,
)


def bit_reversal_permutation(n: int) -> np.ndarray:
    """Permutation matrix mapping big-endian to little-endian basis order."""
    dim = 2**n
    P = np.zeros((dim, dim))
    for index in range(dim):
        reversed_index = int(format(index, f"0{n}b")[::-1], 2)
        P[reversed_index, index] = 1.0
    return P


@pytest.mark.parametrize(
    "numpy_op, qiskit_op",
    [
        (
            penalty_hamiltonian(OneHotProblem((1.0, 2.0, 4.0)), lam=5.0),
            penalty_cost_operator(lam=5.0),
        ),
        (
            objective_hamiltonian(OneHotProblem((1.0, 2.0, 4.0))),
            objective_cost_operator(),
        ),
        (xy_mixer(3), xy_mixer_operator()),
    ],
    ids=["penalty_hamiltonian", "objective_hamiltonian", "xy_mixer"],
)
def test_qiskit_operators_match_numpy_up_to_endianness(numpy_op, qiskit_op):
    P = bit_reversal_permutation(3)
    np.testing.assert_allclose(
        P @ numpy_op @ P.T,
        qiskit_op.to_matrix(),
        atol=1e-12,
    )


def test_penalty_operator_general_lambda_matches_hardcoded():
    default = penalty_cost_operator(lam=5.0)
    rebuilt = penalty_cost_operator(lam=5.0 + 1e-12)  # forces the general branch
    np.testing.assert_allclose(
        default.to_matrix(), rebuilt.to_matrix(), atol=1e-9
    )


def test_circuit_builders_return_expected_parameter_counts():
    for build, extra in ((build_penalty_qaoa, {"lam": 5.0}), (build_xy_qaoa, {})):
        qc, gamma, beta = build(p=2, **extra)
        assert len(gamma) == 2
        assert len(beta) == 2
        assert qc.num_qubits == 3
