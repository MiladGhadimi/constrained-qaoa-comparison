from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.optimize import differential_evolution

from .problem import OneHotProblem


@dataclass(frozen=True)
class OptimizationResult:
    gammas: np.ndarray
    betas: np.ndarray
    expectation: float
    probabilities: dict[str, float]
    feasible_probability: float


def plus_state(n: int) -> np.ndarray:
    return np.ones(2**n, dtype=complex) / np.sqrt(2**n)


def one_hot_w_state(n: int) -> np.ndarray:
    """Uniform superposition of all Hamming-weight-one computational states."""
    psi = np.zeros(2**n, dtype=complex)
    for i in range(n):
        bits = [0] * n
        bits[i] = 1
        index = int("".join(map(str, bits)), 2)
        psi[index] = 1 / np.sqrt(n)
    return psi


def qaoa_state(
    parameters: np.ndarray,
    cost_hamiltonian: np.ndarray,
    mixer_hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    p: int,
) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=float)
    if p <= 0:
        raise ValueError("p must be positive")
    if len(parameters) != 2 * p:
        raise ValueError(f"Expected {2*p} parameters, got {len(parameters)}")

    gammas = parameters[:p]
    betas = parameters[p:]
    psi = np.asarray(initial_state, dtype=complex)

    for gamma, beta in zip(gammas, betas):
        psi = expm(-1j * gamma * cost_hamiltonian) @ psi
        psi = expm(-1j * beta * mixer_hamiltonian) @ psi

    return psi


def expectation_value(state: np.ndarray, operator: np.ndarray) -> float:
    return float(np.real(np.vdot(state, operator @ state)))


def state_probabilities(state: np.ndarray, n: int) -> dict[str, float]:
    probs = np.abs(state) ** 2
    return {
        format(i, f"0{n}b"): float(prob)
        for i, prob in enumerate(probs)
    }


def feasible_probability(
    probabilities: dict[str, float],
    problem: OneHotProblem,
) -> float:
    return float(
        sum(
            prob
            for bitstring, prob in probabilities.items()
            if problem.is_feasible(tuple(int(b) for b in bitstring))
        )
    )


def optimize_qaoa(
    problem: OneHotProblem,
    cost_hamiltonian: np.ndarray,
    mixer_hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    p: int = 1,
    seed: int = 7,
) -> OptimizationResult:
    """Globally optimize a tiny QAOA instance with differential evolution.

    This is deliberately an educational exact-state routine, not a scalable
    production optimizer.
    """

    def objective(theta: np.ndarray) -> float:
        psi = qaoa_state(
            theta,
            cost_hamiltonian,
            mixer_hamiltonian,
            initial_state,
            p,
        )
        return expectation_value(psi, cost_hamiltonian)

    # Conservative generic angle box for this educational example.
    bounds = [(0.0, 2 * np.pi)] * p + [(0.0, np.pi)] * p

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        polish=True,
        tol=1e-10,
    )

    state = qaoa_state(
        result.x,
        cost_hamiltonian,
        mixer_hamiltonian,
        initial_state,
        p,
    )
    probabilities = state_probabilities(state, problem.n)

    return OptimizationResult(
        gammas=np.asarray(result.x[:p]),
        betas=np.asarray(result.x[p:]),
        expectation=float(result.fun),
        probabilities=probabilities,
        feasible_probability=feasible_probability(probabilities, problem),
    )
