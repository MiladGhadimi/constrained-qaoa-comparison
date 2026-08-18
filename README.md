# Constraint-Aware QAOA: Penalties vs Constraint-Preserving Mixers

A small, interview-friendly project that solves the **same constrained binary optimization problem** in two different ways:

1. **Penalty-based QAOA** with a standard X mixer.
2. **Constraint-preserving QAOA** with an XY mixer.

The project deliberately keeps the optimization problem tiny so every step can be checked analytically, simulated exactly, and explained on a whiteboard.

## Problem

Choose exactly one of three binary options:

\[
x_0,x_1,x_2\in\{0,1\},
\qquad
x_0+x_1+x_2=1,
\]

while minimizing

\[
C(x)=x_0+2x_1+4x_2.
\]

The feasible solutions are

\[
100,\quad010,\quad001,
\]

with costs \(1,2,4\), so the classical optimum is

\[
x^\star=100,\qquad C(x^\star)=1.
\]

## Approach A — Penalty-based QAOA

Move the constraint into the objective:

\[
C_\lambda(x)
=
C(x)+
\lambda(x_0+x_1+x_2-1)^2.
\]

For \(\lambda=5\),

\[
H_C^{\rm pen}
=
8.5I
-3Z_0
-3.5Z_1
-4.5Z_2
+2.5Z_0Z_1
+2.5Z_0Z_2
+2.5Z_1Z_2.
\]

Use the standard mixer

\[
H_M^X=X_0+X_1+X_2
\]

and the usual \(|+\rangle^{\otimes 3}\) initial state.

**Interpretation:** explore the full \(2^3=8\) state space, but make infeasible states expensive.

## Approach B — Constraint-preserving XY-QAOA

Keep only the original objective in the cost Hamiltonian:

\[
H_C^{\rm obj}
=
3.5I-\frac12Z_0-Z_1-2Z_2.
\]

The feasible subspace is

\[
\mathcal H_F
=
\mathrm{span}\{|100\rangle,|010\rangle,|001\rangle\}.
\]

Start from the feasible W state

\[
|W\rangle
=
\frac{|100\rangle+|010\rangle+|001\rangle}{\sqrt3},
\]

and use an XY mixer

\[
H_M^{XY}
=
(X_0X_1+Y_0Y_1)
+
(X_1X_2+Y_1Y_2).
\]

This mixer commutes with the Hamming-weight operator, so ideal evolution stays inside the one-hot feasible subspace.

**Interpretation:** do not punish infeasible states; avoid exploring them.

## Exact p=1 comparison

The repository includes a NumPy/SciPy exact-state simulator so the core results do not depend on Qiskit.

A representative global optimization of the two \(p=1\) landscapes gives approximately:

| Method | Optimized expected cost | Feasible probability | Most likely feasible state |
|---|---:|---:|---|
| Penalty + X mixer | 2.743 | 0.816 | `010` |
| Objective + XY mixer | 1.238 | 1.000 | `100` |

The exact optimizer output can vary slightly with numerical settings, but the structural difference does not:

- penalty-QAOA can populate infeasible states;
- ideal XY-QAOA preserves the one-hot feasible subspace.

This **does not imply that XY mixers are always superior**. On hardware, XY interactions can create more expensive two-qubit circuits and routing overhead. The relevant comparison is therefore algorithmic quality **after compilation/transpilation**, not only at the logical level.

## Repository structure

```text
constrained-qaoa-comparison/
├── README.md
├── pyproject.toml
├── run_demo.py
├── qiskit_demo.py
├── src/
│   └── constrained_qaoa/
│       ├── __init__.py
│       ├── problem.py
│       ├── operators.py
│       ├── simulation.py
│       └── qiskit_circuits.py
└── tests/
    ├── test_problem.py
    └── test_operators.py
```

## Run the exact simulator

```bash
python -m pip install -e .
python run_demo.py
```

## Run the tests

```bash
pytest
```

## Optional Qiskit circuit demo

Install the optional Qiskit dependency:

```bash
python -m pip install -e ".[qiskit]"
python qiskit_demo.py
```

The Qiskit code builds parameterized logical circuits for both formulations. It uses `SparsePauliOp` and `PauliEvolutionGate`, which are useful because the Hamiltonians are expressed naturally as sums of Pauli operators.

### Qiskit bit ordering

This project defines classical strings as `x0 x1 x2`, so `100` means \(x_0=1\). Qiskit displays measured bitstrings in its own conventional ordering, where qubit 0 is typically the right-most displayed bit. Be explicit about this when decoding measured results.

## Interview takeaway

A concise summary:

> Penalty-based QAOA keeps a hardware-friendly X mixer but searches the full Hilbert space and introduces a penalty-strength hyperparameter. Constraint-preserving QAOA can restrict evolution to feasible states and eliminate penalty tuning, but its mixer becomes problem-dependent and can require more costly two-qubit interactions. I would compare both using solution quality, feasible-sampling probability, transpiled two-qubit gate count, depth, and hardware performance.

## Notes

The exact-state simulator uses dense matrices and is intentionally limited to small educational examples. It is not intended as a scalable QAOA implementation.
