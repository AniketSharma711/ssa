# Executed validation

Date: 2026-10-15. Environment: Python 3.11, NumPy 2.3.5, Matplotlib 3.10.8,
pytest 9.1.1, Windows. All commands ran from the project root.

`python -m pytest -q`: **60 passed in 7.68 seconds**. Pytest treats warnings
as errors. `python -m compileall -q algorithms benchmarks experiments tests`
also completed successfully.

The following full experiment commands completed successfully and exported CSV
histories, best positions, and convergence PNGs under `results/`:

```powershell
python -m experiments.run_ssa --benchmark sphere --runs 5 --no-show --save-csv --save-plot
python -m experiments.run_ssa --benchmark rosenbrock --runs 3 --no-show --save-csv --save-plot
python -m experiments.run_ssa --benchmark rastrigin --runs 3 --no-show --save-csv --save-plot
python -m experiments.run_ssa --benchmark ackley --runs 3 --no-show --save-csv --save-plot
python -m experiments.run_ssa --benchmark griewank --runs 3 --no-show --save-csv --save-plot
```

Each used N=30, D=10, T=100, discoverer fraction 0.2, scout fraction 0.1,
and safety threshold 0.8. Seeds started at 42 and increased by one per run.
These are development smoke checks, not replication of the paper's experiments.

- Sphere, seeds 42--46: final fitness values approximately `1.58790e-132`,
  `2.92321e-300`, `1.91977e-55`, `2.73554e-39`, and `1.86945e-101`.
- Rosenbrock, seeds 42--44: `4.33098e-08`, `2.67796e-05`, `1.04213e-05`.
- Rastrigin, seeds 42--44: floating-point objective `0` in each run.
- Ackley, seeds 42--44: approximately `4.44089e-16` in each run.
- Griewank, seeds 42--44: floating-point objective `0` in each run.

Exact floating-point zeros do not certify exact mathematical optima: Rastrigin
and Griewank can round to zero at small nonzero vectors. Sphere's small values
also reflect its favorable origin-centered structure. No performance conclusion
about arbitrary objectives or superiority over other algorithms follows.

The Rosenbrock convergence PNG was visually inspected: title, axes, three seed
curves, and legend rendered correctly. The runner test separately checks exported
CSV row counts, headers, best-position files, and a valid PNG signature.
