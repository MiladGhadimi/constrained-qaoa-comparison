import numpy as np

from constrained_qaoa import (
    OneHotProblem,
    hamming_weight_operator,
    objective_hamiltonian,
    penalty_hamiltonian,
    xy_mixer,
)


def basis_index(bits):
    return int("".join(map(str, bits)), 2)


def test_objective_hamiltonian_matches_classical_costs():
    problem = OneHotProblem((1.0, 2.0, 4.0))
    H = objective_hamiltonian(problem)

    diagonal = np.real(np.diag(H))

    for bits in problem.all_bitstrings():
        np.testing.assert_allclose(
            diagonal[basis_index(bits)],
            problem.objective(bits),
        )


def test_penalty_hamiltonian_matches_penalized_objective():
    problem = OneHotProblem((1.0, 2.0, 4.0))
    H = penalty_hamiltonian(problem, lam=5.0)
    diagonal = np.real(np.diag(H))

    for bits in problem.all_bitstrings():
        np.testing.assert_allclose(
            diagonal[basis_index(bits)],
            problem.penalized_objective(bits, lam=5.0),
        )


def test_xy_mixer_preserves_hamming_weight():
    n = 3
    Hm = xy_mixer(n)
    N = hamming_weight_operator(n)

    commutator = Hm @ N - N @ Hm
    np.testing.assert_allclose(commutator, np.zeros_like(commutator), atol=1e-12)
