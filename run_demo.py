from constrained_qaoa import (
    OneHotProblem,
    objective_hamiltonian,
    penalty_hamiltonian,
    x_mixer,
    xy_mixer,
    optimize_qaoa,
)
from constrained_qaoa.simulation import plus_state, one_hot_w_state


def show_distribution(title, probabilities, threshold=1e-4):
    print(f"\n{title}")
    print("-" * len(title))
    for bitstring, probability in sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        if probability >= threshold:
            print(f"{bitstring}: {probability:.6f}")


def main():
    problem = OneHotProblem((1.0, 2.0, 4.0))
    optimum, optimum_cost = problem.classical_optimum()

    print("Problem")
    print("=======")
    print("min C(x) = x0 + 2 x1 + 4 x2")
    print("subject to x0 + x1 + x2 = 1")
    print(f"Classical optimum: {''.join(map(str, optimum))}, cost={optimum_cost}")

    lam = 5.0

    penalty_result = optimize_qaoa(
        problem=problem,
        cost_hamiltonian=penalty_hamiltonian(problem, lam),
        mixer_hamiltonian=x_mixer(problem.n),
        initial_state=plus_state(problem.n),
        p=1,
        seed=7,
    )

    xy_result = optimize_qaoa(
        problem=problem,
        cost_hamiltonian=objective_hamiltonian(problem),
        mixer_hamiltonian=xy_mixer(problem.n),
        initial_state=one_hot_w_state(problem.n),
        p=1,
        seed=7,
    )

    print("\nPenalty-based QAOA")
    print("==================")
    print(f"gamma* = {penalty_result.gammas}")
    print(f"beta*  = {penalty_result.betas}")
    print(f"<H_cost> = {penalty_result.expectation:.6f}")
    print(f"feasible probability = {penalty_result.feasible_probability:.6f}")
    show_distribution("Output distribution", penalty_result.probabilities)

    print("\nConstraint-preserving XY-QAOA")
    print("==============================")
    print(f"gamma* = {xy_result.gammas}")
    print(f"beta*  = {xy_result.betas}")
    print(f"<H_cost> = {xy_result.expectation:.6f}")
    print(f"feasible probability = {xy_result.feasible_probability:.6f}")
    show_distribution("Output distribution", xy_result.probabilities)

    print("\nTakeaway")
    print("========")
    print(
        "Penalty-QAOA explores the full Hilbert space and penalizes violations; "
        "XY-QAOA preserves one-hot feasibility ideally, but uses a more complex mixer."
    )


if __name__ == "__main__":
    main()
