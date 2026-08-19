import numpy as np
import pytest

from constrained_qaoa import (
    OneHotProblem,
    objective_hamiltonian,
    penalty_hamiltonian,
    x_mixer,
    xy_mixer,
)
from constrained_qaoa.simulation import (
    expectation_value,
    feasible_probability,
    one_hot_w_state,
    plus_state,
    qaoa_state,
    state_probabilities,
)


def test_w_state_is_normalized_and_feasible():
    problem = OneHotProblem((1.0, 2.0, 4.0))
    psi = one_hot_w_state(problem.n)

    np.testing.assert_allclose(np.vdot(psi, psi), 1.0)
    probs = state_probabilities(psi, problem.n)
    assert feasible_probability(probs, problem) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_xy_evolution_preserves_feasible_subspace(seed):
    """For any angles, ideal XY-QAOA never leaves the one-hot subspace."""
    problem = OneHotProblem((1.0, 2.0, 4.0))
    rng = np.random.default_rng(seed)
    params = rng.uniform(0.0, 2 * np.pi, size=4)  # p=2

    psi = qaoa_state(
        params,
        cost_hamiltonian=objective_hamiltonian(problem),
        mixer_hamiltonian=xy_mixer(problem.n),
        initial_state=one_hot_w_state(problem.n),
        p=2,
    )
    probs = state_probabilities(psi, problem.n)
    assert feasible_probability(probs, problem) == pytest.approx(1.0, abs=1e-10)


def test_x_mixer_evolution_leaks_out_of_feasible_subspace():
    """Contrast case: the X mixer does not preserve Hamming weight."""
    problem = OneHotProblem((1.0, 2.0, 4.0))
    psi = qaoa_state(
        np.array([0.7, 0.9]),
        cost_hamiltonian=penalty_hamiltonian(problem, lam=5.0),
        mixer_hamiltonian=x_mixer(problem.n),
        initial_state=one_hot_w_state(problem.n),
        p=1,
    )
    probs = state_probabilities(psi, problem.n)
    assert feasible_probability(probs, problem) < 1.0


def test_zero_angles_return_initial_state_statistics():
    problem = OneHotProblem((1.0, 2.0, 4.0))
    psi = qaoa_state(
        np.zeros(2),
        cost_hamiltonian=objective_hamiltonian(problem),
        mixer_hamiltonian=xy_mixer(problem.n),
        initial_state=one_hot_w_state(problem.n),
        p=1,
    )
    # <H> of the W state = mean of feasible costs = (1+2+4)/3.
    expectation = expectation_value(psi, objective_hamiltonian(problem))
    assert expectation == pytest.approx((1.0 + 2.0 + 4.0) / 3)


def test_qaoa_state_validates_parameters():
    problem = OneHotProblem((1.0, 2.0, 4.0))
    with pytest.raises(ValueError):
        qaoa_state(
            np.zeros(3),  # wrong length for p=1
            objective_hamiltonian(problem),
            x_mixer(problem.n),
            plus_state(problem.n),
            p=1,
        )
