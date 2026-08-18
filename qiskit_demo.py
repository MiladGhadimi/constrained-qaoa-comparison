"""Build the two parameterized logical QAOA circuits with Qiskit."""

from constrained_qaoa.qiskit_circuits import (
    build_penalty_qaoa,
    build_xy_qaoa,
)


def main():
    penalty_qc, penalty_gamma, penalty_beta = build_penalty_qaoa(p=1, lam=5.0)
    xy_qc, xy_gamma, xy_beta = build_xy_qaoa(p=1)

    print("Penalty-based QAOA circuit")
    print("==========================")
    print(penalty_qc.draw("text"))
    print("Parameters:", list(penalty_gamma), list(penalty_beta))

    print("\nConstraint-preserving XY-QAOA circuit")
    print("======================================")
    print(xy_qc.draw("text"))
    print("Parameters:", list(xy_gamma), list(xy_beta))

    print(
        "\nNext step on hardware: transpile each circuit for the same backend and "
        "compare depth, two-qubit gate count, routing overhead, and execution quality."
    )


if __name__ == "__main__":
    main()
