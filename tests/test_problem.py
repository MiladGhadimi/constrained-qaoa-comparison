import pytest

from constrained_qaoa import OneHotProblem


def test_feasible_space_and_optimum():
    problem = OneHotProblem((1.0, 2.0, 4.0))

    assert set(problem.feasible_bitstrings()) == {
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    }

    optimum, cost = problem.classical_optimum()
    assert optimum == (1, 0, 0)
    assert cost == pytest.approx(1.0)


def test_penalty_makes_zero_state_expensive():
    problem = OneHotProblem((1.0, 2.0, 4.0))

    assert problem.objective((0, 0, 0)) == 0.0
    assert problem.penalized_objective((0, 0, 0), lam=5.0) == 5.0
    assert problem.penalized_objective((1, 0, 0), lam=5.0) == 1.0
