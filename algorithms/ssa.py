"""Vanilla Sparrow Search Algorithm for bounded, continuous minimization.

Equations (3)--(5) and Algorithm 1 of Xue & Shen (2020), DOI
10.1080/21642583.2019.1708830. See REFERENCE_AUDIT.md for timing choices.
"""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class SSAResult:
    """Best evaluated position (D,), scalar fitness, and history (T,)."""

    best_position: Vector
    best_fitness: float
    convergence_curve: Vector


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _fraction(name: str, value: float, low: float, high: float,
              *, open_ends: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    valid = low < value < high if open_ends else low <= value <= high
    if not np.isfinite(value) or not valid:
        interval = f"({low}, {high})" if open_ends else f"[{low}, {high}]"
        raise ValueError(f"{name} must be in {interval}")
    return float(value)


def _bound(name: str, value: ArrayLike, dimensions: int) -> Vector:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(dimensions, float(array))
    if array.shape != (dimensions,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and scalar or shape ({dimensions},)")
    return array.copy()


def _producer_step(position, rank, iterations, warning, threshold, rng):
    # Equation (3): i is one-based fitness rank; Q is a scalar N(0,1).
    if warning < threshold:
        alpha = 1.0 - rng.random()  # (0, 1], excluding division by zero
        return position * np.exp(-rank / (alpha * iterations))
    return position + rng.normal()


def _scrounger_step(position, rank, population_size, worst, producer, rng):
    # Equation (4): |x-Xp| (1,D) @ A+ (D,1), then broadcast along L.
    if rank > population_size / 2:
        return rng.normal() * np.exp((worst - position) / rank**2)
    signs = rng.choice([-1.0, 1.0], size=position.size)
    pseudoinverse = signs / np.dot(signs, signs)
    return producer + np.dot(np.abs(position - producer), pseudoinverse)


def _scout_step(position, fitness, best, best_fitness, worst, worst_fitness, rng):
    # Equation (5): all fitness/position references belong to the same t snapshot.
    if fitness > best_fitness:
        return best + rng.normal() * np.abs(position - best)
    epsilon = np.finfo(float).eps
    denominator = fitness - worst_fitness + epsilon
    # A negative gap of exactly epsilon can cancel the added epsilon in float64.
    if denominator == 0:
        denominator = -epsilon if fitness < worst_fitness else epsilon
    return position + rng.uniform(-1.0, 1.0) * (
        np.abs(position - worst) / denominator
    )


def ssa(
    population_size: int,
    max_iterations: int,
    lower_bound: ArrayLike,
    upper_bound: ArrayLike,
    dimensions: int,
    objective_function: Callable[[Vector], float],
    discoverer_fraction: float = 0.2,
    scout_fraction: float = 0.1,
    safety_threshold: float = 0.8,
    seed: int | None = None,
) -> SSAResult:
    """Minimize a deterministic scalar objective with vanilla SSA.

    Population shape is (N,D). Bounds are finite scalars or arrays of shape
    (D,), with lower < upper coordinatewise and a finite span. N >= 2, D >= 1,
    T >= 0. Discoverer fraction is in (0,1), scout fraction in [0,1], and safety
    threshold in [0.5,1]. Counts use floor(N*fraction), with at least one
    discoverer, at most N-1 discoverers, and at least one scout if fraction > 0.

    Each objective call receives a separate, bounded (D,) float array and must
    return a finite real scalar. Invalid objective values raise ValueError.
    History has length T, excluding initialization. T=0 returns the best initial
    candidate and empty history. Randomness uses a local numpy Generator.

    Each iteration ranks retained candidates, updates producers, chooses the
    best updated producer, updates scroungers, then overwrites selected proposals
    with scout moves from the frozen t snapshot. Better final proposals replace
    old positions. Best-so-far also includes evaluated intermediate producers.
    Bounds are clipped before every evaluation. No plotting or file I/O occurs.
    """
    population_size = _integer("population_size", population_size, 2)
    max_iterations = _integer("max_iterations", max_iterations, 0)
    dimensions = _integer("dimensions", dimensions, 1)
    discoverer_fraction = _fraction("discoverer_fraction", discoverer_fraction,
                                    0, 1, open_ends=True)
    scout_fraction = _fraction("scout_fraction", scout_fraction, 0, 1)
    safety_threshold = _fraction("safety_threshold", safety_threshold, 0.5, 1)
    if not callable(objective_function):
        raise ValueError("objective_function must be callable")
    if seed is not None:
        seed = _integer("seed", seed, 0)
    lower = _bound("lower_bound", lower_bound, dimensions)
    upper = _bound("upper_bound", upper_bound, dimensions)
    with np.errstate(over="ignore"):
        span = upper - lower
    if np.any(lower >= upper) or not np.all(np.isfinite(span)):
        raise ValueError("bounds must satisfy lower_bound < upper_bound with finite spans")

    rng = np.random.default_rng(seed)
    producer_count = min(population_size - 1,
                         max(1, int(population_size * discoverer_fraction)))
    scout_count = max(1, int(population_size * scout_fraction)) if scout_fraction else 0
    best_fitness = float("inf")
    best_position = np.empty(dimensions)

    def evaluate(position: Vector) -> float:
        nonlocal best_fitness, best_position
        value = np.asarray(objective_function(position.copy()))
        if value.ndim != 0 or value.dtype.kind not in "fiu":
            raise ValueError("objective_function must return a finite real scalar")
        score = float(value)
        if not np.isfinite(score):
            raise ValueError("objective_function must return a finite real scalar")
        if score < best_fitness:
            best_fitness, best_position = score, position.copy()
        return score

    def bounded(position) -> Vector:
        result = np.clip(position, lower, upper)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("SSA movement produced NaN coordinates")
        return result

    positions = lower + rng.random((population_size, dimensions)) * span
    fitness = np.array([evaluate(x) for x in positions])
    curve = np.empty(max_iterations)

    for iteration in range(max_iterations):
        order = np.argsort(fitness, kind="stable")
        # Sorting both arrays keeps one-based rank aligned with position/fitness.
        positions, fitness = positions[order].copy(), fitness[order].copy()
        old_best, old_worst = positions[0].copy(), positions[-1].copy()
        proposals = positions.copy()
        warning = rng.random()  # Algorithm 1 draws R2 once per iteration.

        # Evaluate producers to identify Xp^(t+1), not a previous non-producer.
        producer_fitness = np.empty(producer_count)
        for index in range(producer_count):
            proposals[index] = bounded(_producer_step(
                positions[index], index + 1, max_iterations, warning,
                safety_threshold, rng,
            ))
            producer_fitness[index] = evaluate(proposals[index])
        producer = proposals[int(np.argmin(producer_fitness))].copy()

        # Overflowing moves become signed infinities and are clipped to bounds.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for index in range(producer_count, population_size):
                proposals[index] = bounded(_scrounger_step(
                    positions[index], index + 1, population_size, old_worst,
                    producer, rng,
                ))
            scouts = rng.choice(population_size, size=scout_count, replace=False)
            for index in scouts:
                proposals[index] = bounded(_scout_step(
                    positions[index], fitness[index], old_best, fitness[0],
                    old_worst, fitness[-1], rng,
                ))

        new_fitness = np.array([evaluate(x) for x in proposals])
        improved = new_fitness < fitness
        positions[improved] = proposals[improved]
        fitness[improved] = new_fitness[improved]
        curve[iteration] = best_fitness

    return SSAResult(best_position.copy(), best_fitness, curve)
