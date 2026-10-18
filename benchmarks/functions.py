"""Standard benchmarks; formulas checked against the SFU test-function library.

https://www.sfu.ca/~ssurjano/optimization.html
Bounds below follow Xue & Shen (2020), Tables 1--2 (Ackley uses [-32,32]).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def _vector(x: ArrayLike) -> np.ndarray:
    vector = np.asarray(x, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("benchmark input must be a nonempty finite vector of shape (D,)")
    return vector


def sphere(x: ArrayLike) -> float:
    """sum(x_i^2); minimum 0 at zeros."""
    x = _vector(x)
    return float(np.sum(x**2))


def rastrigin(x: ArrayLike) -> float:
    """sum(x_i^2 - 10*cos(2*pi*x_i) + 10); minimum 0 at zeros."""
    x = _vector(x)
    return float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10))


def rosenbrock(x: ArrayLike) -> float:
    """sum(100*(x[i+1]-x[i]^2)^2+(x[i]-1)^2); D>=2, minimum at ones."""
    x = _vector(x)
    if x.size < 2:
        raise ValueError("Rosenbrock requires at least two dimensions")
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2))


def ackley(x: ArrayLike) -> float:
    """Ackley with a=20, b=0.2, c=2*pi; minimum 0 at zeros."""
    x = _vector(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
                 - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e)


def griewank(x: ArrayLike) -> float:
    """sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i))) + 1; i starts at 1."""
    x = _vector(x)
    return float(np.sum(x**2) / 4000
                 - np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1)))) + 1)


@dataclass(frozen=True)
class Benchmark:
    function: Callable
    lower_bound: float
    upper_bound: float
    minimum_dimensions: int = 1


BENCHMARKS = {
    "sphere": Benchmark(sphere, -100, 100),
    "rastrigin": Benchmark(rastrigin, -5.12, 5.12),
    "rosenbrock": Benchmark(rosenbrock, -30, 30, 2),
    "ackley": Benchmark(ackley, -32, 32),
    "griewank": Benchmark(griewank, -600, 600),
}
