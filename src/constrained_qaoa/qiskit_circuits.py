from __future__ import annotations

"""Optional Qiskit circuit builders.

The exact simulator in this repository does not depend on Qiskit.
These helpers require the optional `qiskit` dependency.
"""

from math import sqrt

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Qiskit is optional. Install with: pip install -e '.[qiskit]'"
    ) from exc


def penalty_cost_operator(lam: float = 5.0) -> SparsePauliOp:
    """3-qubit penalty Hamiltonian.

    H = 8.5 I - 3 Z0 - 3.5 Z1 - 4.5 Z2
        + 2.5 Z0Z1 + 2.5 Z0Z2 + 2.5 Z1Z2

    `from_sparse_list` avoids ambiguity in Pauli-string endianness.
    """
    terms = [
        ("", [], 8.5),
        ("Z", [0], -3.0),
        ("Z", [1], -3.5),
        ("Z", [2], -4.5),
        ("ZZ", [0, 1], 2.5),
        ("ZZ", [0, 2], 2.5),
        ("ZZ", [1, 2], 2.5),
    ]

    if lam != 5.0:
        # Rebuild coefficients analytically for arbitrary lambda:
        # objective costs = (1,2,4), one-hot penalty.
        const = lam
        linear = [1 - lam, 2 - lam, 4 - lam]
        pair = 2 * lam

        # Map x_i=(I-Z_i)/2 to Pauli coefficients.
        const_pauli = const + sum(c / 2 for c in linear) + 3 * pair / 4
        z = [-(linear[i] / 2) - pair / 2 for i in range(3)]
        zz = pair / 4
        terms = [
            ("", [], const_pauli),
            ("Z", [0], z[0]),
            ("Z", [1], z[1]),
            ("Z", [2], z[2]),
            ("ZZ", [0, 1], zz),
            ("ZZ", [0, 2], zz),
            ("ZZ", [1, 2], zz),
        ]

    return SparsePauliOp.from_sparse_list(terms, num_qubits=3)


def objective_cost_operator() -> SparsePauliOp:
    return SparsePauliOp.from_sparse_list(
        [
            ("", [], 3.5),
            ("Z", [0], -0.5),
            ("Z", [1], -1.0),
            ("Z", [2], -2.0),
        ],
        num_qubits=3,
    )


def xy_mixer_operator() -> SparsePauliOp:
    return SparsePauliOp.from_sparse_list(
        [
            ("XX", [0, 1], 1.0),
            ("YY", [0, 1], 1.0),
            ("XX", [1, 2], 1.0),
            ("YY", [1, 2], 1.0),
        ],
        num_qubits=3,
    )


def x_mixer_operator() -> SparsePauliOp:
    return SparsePauliOp.from_sparse_list(
        [
            ("X", [0], 1.0),
            ("X", [1], 1.0),
            ("X", [2], 1.0),
        ],
        num_qubits=3,
    )


def _prepare_w_state(qc: QuantumCircuit) -> None:
    """Prepare (|100>+|010>+|001>)/sqrt(3) in x0,x1,x2 notation.

    Qiskit's amplitude ordering is little-endian, so classical x0 x1 x2
    corresponds to q0,q1,q2 but basis labels appear reversed when printed.
    """
    amplitudes = [0j] * 8
    # Qiskit basis index = q2 q1 q0.  One-hot on q0,q1,q2 -> indices 1,2,4.
    for index in (1, 2, 4):
        amplitudes[index] = 1 / sqrt(3)
    qc.initialize(amplitudes, [0, 1, 2])


def build_penalty_qaoa(p: int = 1, lam: float = 5.0):
    if p <= 0:
        raise ValueError("p must be positive")

    gamma = ParameterVector("gamma", p)
    beta = ParameterVector("beta", p)
    qc = QuantumCircuit(3)
    qc.h(range(3))

    hc = penalty_cost_operator(lam)
    hm = x_mixer_operator()

    for layer in range(p):
        qc.append(PauliEvolutionGate(hc, time=gamma[layer]), range(3))
        qc.append(PauliEvolutionGate(hm, time=beta[layer]), range(3))

    return qc, gamma, beta


def build_xy_qaoa(p: int = 1):
    if p <= 0:
        raise ValueError("p must be positive")

    gamma = ParameterVector("gamma", p)
    beta = ParameterVector("beta", p)
    qc = QuantumCircuit(3)
    _prepare_w_state(qc)

    hc = objective_cost_operator()
    hm = xy_mixer_operator()

    for layer in range(p):
        qc.append(PauliEvolutionGate(hc, time=gamma[layer]), range(3))
        qc.append(PauliEvolutionGate(hm, time=beta[layer]), range(3))

    return qc, gamma, beta
