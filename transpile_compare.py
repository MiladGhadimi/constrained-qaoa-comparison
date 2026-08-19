"""Transpile both QAOA formulations for the same backend and compare cost.

The README argues that the relevant comparison between penalty-based and
constraint-preserving QAOA is *after* compilation, because XY mixers can
require more expensive two-qubit interactions. This script makes that
comparison concrete on a generic 3-qubit line-topology backend.
"""

from __future__ import annotations

import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import generate_preset_pass_manager

from constrained_qaoa.qiskit_circuits import build_penalty_qaoa, build_xy_qaoa

TWO_QUBIT_GATES = {"cx", "cz", "ecr", "swap"}


def transpiled_stats(qc, backend, optimization_level: int):
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level, backend=backend
    )
    isa = pm.run(qc)
    ops = isa.count_ops()
    two_qubit = sum(count for name, count in ops.items() if name in TWO_QUBIT_GATES)
    return {"depth": isa.depth(), "two_qubit_gates": two_qubit, "ops": dict(ops)}


def main() -> None:
    backend = GenericBackendV2(num_qubits=3, coupling_map=[[0, 1], [1, 2]], seed=7)

    penalty_qc, penalty_gamma, penalty_beta = build_penalty_qaoa(p=1, lam=5.0)
    xy_qc, xy_gamma, xy_beta = build_xy_qaoa(p=1)

    # Bind representative angles so state preparation fully decomposes.
    penalty_bound = penalty_qc.assign_parameters(
        {penalty_gamma[0]: 0.4, penalty_beta[0]: 0.8}
    )
    xy_bound = xy_qc.assign_parameters({xy_gamma[0]: 0.4, xy_beta[0]: 0.8})

    print(f"Backend: {backend.name} (line coupling 0-1-2)")
    print(f"Basis gates: {sorted(backend.operation_names)}")

    for level in (1, 3):
        print(f"\nOptimization level {level}")
        print("=" * 22)
        for label, qc in (
            ("Penalty + X mixer  ", penalty_bound),
            ("Objective + XY mixer", xy_bound),
        ):
            stats = transpiled_stats(qc, backend, level)
            print(
                f"{label}: depth={stats['depth']:3d}  "
                f"2q-gates={stats['two_qubit_gates']:3d}  ops={stats['ops']}"
            )

    print(
        "\nNote: the XY circuit's cost includes W-state preparation, which is "
        "part of the honest price of the constraint-preserving formulation."
    )


if __name__ == "__main__":
    main()
