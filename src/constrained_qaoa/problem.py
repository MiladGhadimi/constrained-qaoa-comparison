from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

BitString = tuple[int, ...]


@dataclass(frozen=True)
class OneHotProblem:
    """Choose exactly one option while minimizing a linear cost."""

    costs: tuple[float, ...] = (1.0, 2.0, 4.0)

    @property
    def n(self) -> int:
        return len(self.costs)

    def validate(self, x: Iterable[int]) -> BitString:
        bits = tuple(int(v) for v in x)
        if len(bits) != self.n:
            raise ValueError(f"Expected {self.n} bits, got {len(bits)}")
        if any(v not in (0, 1) for v in bits):
            raise ValueError("All decision variables must be binary")
        return bits

    def objective(self, x: Iterable[int]) -> float:
        bits = self.validate(x)
        return float(sum(c * b for c, b in zip(self.costs, bits)))

    def is_feasible(self, x: Iterable[int]) -> bool:
        bits = self.validate(x)
        return sum(bits) == 1

    def penalty(self, x: Iterable[int], lam: float) -> float:
        if lam < 0:
            raise ValueError("Penalty strength must be non-negative")
        bits = self.validate(x)
        return float(lam * (sum(bits) - 1) ** 2)

    def penalized_objective(self, x: Iterable[int], lam: float) -> float:
        return self.objective(x) + self.penalty(x, lam)

    def all_bitstrings(self) -> list[BitString]:
        return list(product((0, 1), repeat=self.n))

    def feasible_bitstrings(self) -> list[BitString]:
        return [x for x in self.all_bitstrings() if self.is_feasible(x)]

    def classical_optimum(self) -> tuple[BitString, float]:
        feasible = self.feasible_bitstrings()
        best = min(feasible, key=self.objective)
        return best, self.objective(best)
