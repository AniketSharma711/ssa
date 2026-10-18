# Sparrow Search Algorithm

A reusable NumPy implementation of vanilla SSA for bounded continuous
minimization, following Xue and Shen (2020), equations (3)--(5) and Algorithm 1.
Source checks, reference-code defects, and choices where the paper leaves details
open are recorded in [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md). This is an explicit
interpretation of the published method, not a claim of identical MATLAB results.

## Setup and quick start

Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
python -m experiments.run_ssa
```

Run commands from the project root. The default experiment uses 30 sparrows,
10 dimensions, 100 iterations, and seed 42 on Sphere. These are development
settings. To save convergence CSV, best-position CSV, and a PNG without opening
a graphical window:

```powershell
python -m experiments.run_ssa --benchmark rastrigin --runs 5 --save-csv --save-plot --no-show
```

Outputs default to `results/`; `--output-dir` overrides it. Files with the same
benchmark and seed are overwritten. Use `--help` for all options, including bounds,
population, iterations, fractions, and safety threshold. Independent runs use
consecutive seeds. The displayed standard deviation is the population standard
deviation (`ddof=0`). Plot axes use linear fitness so zero/negative values are valid.

## Python API

```python
import numpy as np
from algorithms import ssa
from benchmarks.functions import sphere

result = ssa(
    population_size=30,
    max_iterations=100,
    lower_bound=np.full(10, -100.0),
    upper_bound=100.0,
    dimensions=10,
    objective_function=sphere,
    discoverer_fraction=0.2,
    scout_fraction=0.1,
    safety_threshold=0.8,
    seed=42,
)
print(result.best_fitness, result.best_position)
```

The objective accepts a vector of shape `(D,)` and returns a finite real scalar.
It should be deterministic. It receives a copy, so accidental mutation does not
change optimizer state. Exceptions from the objective propagate. Nonfinite,
complex, boolean, and vector return values raise `ValueError`.

`SSAResult` contains `best_position` (a separate `(D,)` NumPy array), `best_fitness`
(Python float), and `convergence_curve` (shape `(max_iterations,)`). History records
best-so-far after each complete iteration, excluding initialization; it is
non-increasing. Zero iterations returns the best initialized candidate and empty
history. Every evaluated producer counts toward best-so-far, including one later
overwritten by a scout. Thus the returned best need not be a retained member of
the final population. It always corresponds to an evaluated vector.

Population size must be an integer at least 2; dimension at least 1; iterations
an integer at least 0. Bounds are finite scalars or exact `(D,)` arrays, with
strictly positive, finite coordinate spans. Fixed coordinates are not supported.
Discoverer fraction lies in `(0,1)`; its count is `floor(N*fraction)` clamped to
`[1,N-1]`. Scout fraction lies in `[0,1]`; zero disables scouts, otherwise the count
is `max(1,floor(N*fraction))`. Scouts are sampled uniformly without replacement
each iteration. Fractions outside the paper's typical range are allowed for
experiments. Safety threshold lies in `[0.5,1]`; defaults are configurable choices.
The optional seed is a nonnegative integer. Runs with the same settings and seed
repeat within the same Python/NumPy environment; cross-version or MATLAB random
sequence equivalence is not promised. Global NumPy random state is untouched.

## Movement and iteration semantics

Initialization is coordinatewise uniform within bounds. Each iteration stably
sorts current candidates by ascending fitness and uses one-based rank `i`.
Producers use multiplicative decay when the shared uniform alarm `R2 < ST`;
otherwise they add one scalar standard-normal draw across coordinates. Alpha is
independently uniform in `(0,1]` for each safe producer.

Scroungers above rank `N/2` use a normal draw times
`exp((worst-position)/i**2)`. Other scroungers follow the best **updated producer**,
using the scalar product of their absolute distance with `A+ = A.T/(A@A.T)`,
broadcast across the ones vector. Each sign of `A` is independently ±1.

Scouts use equation (5) with a frozen start-of-iteration population, fitness,
best, and worst. A worse-than-current-best scout uses scalar standard-normal
beta. A tied-best scout uses uniform `K` in `[-1,1]` and the epsilon-protected
fitness difference. Scout proposals overwrite that individual's earlier movement.
This literal reading of the superscript `t` is a documented timing choice.

All proposals are clipped coordinatewise before evaluation, including producers
evaluated to identify the updated producer leader. Final proposals replace old
candidates only if strictly better, interpreting Algorithm 1's acceptance step
per individual. This is ordinary greedy SSA retention, without added personal
best trajectories, MPA memory, or hybrid search operators. Global best-so-far is
tracked independently. See the audit for limitations of these interpretations.

## Benchmarks and verification

Independent benchmark definitions include Sphere (`[-100,100]`), Rastrigin
(`[-5.12,5.12]`), Rosenbrock (`[-30,30]`), Ackley (`[-32,32]`), and Griewank
(`[-600,600]`). All minima are zero: Rosenbrock at ones, others at zeros.
Rosenbrock requires `D >= 2`; other functions support `D >= 1`.

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover exact equation examples, benchmark values, fitness/position consistency,
repeatability, bounds for every objective call, small populations, varied scout
counts, one dimension, equal fitness, epsilon cancellation, input validation,
multiple Sphere seeds, all other benchmarks, and headless CSV/PNG export.
Sphere convergence is a smoke check, not proof of general optimization performance.
Use independent runs to assess stochastic performance; the code makes no promise
that every run reaches a global optimum.

## References

- Xue, J. and Shen, B. (2020), *A novel swarm intelligence optimization approach:
  sparrow search algorithm*, Systems Science & Control Engineering 8(1), 22--34.
  [DOI](https://doi.org/10.1080/21642583.2019.1708830).
- [Publisher-provided paper text on ResearchGate](https://www.researchgate.net/publication/338386846_A_novel_swarm_intelligence_optimization_approach_sparrow_search_algorithm).
- [Lancephil Python reference](https://github.com/Lancephil/Sparrow-Search-Algorithm-Python),
  inspected at commit `0d7c8c20a13712ed2db94eb5a9512bd037353a34`.
- Surjanovic and Bingham, [SFU optimization test-function library](https://www.sfu.ca/~ssurjano/optimization.html).
