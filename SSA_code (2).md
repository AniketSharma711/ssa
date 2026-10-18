# Implementation Brief: Sparrow Search Algorithm (SSA) in Python

## 1. Objective

Implement a faithful, reusable Python implementation of the original **Sparrow Search Algorithm (SSA)** proposed by Jiankai Xue and Bo Shen.

**Primary reference paper**

Xue, J. and Shen, B. (2020), “A Novel Swarm Intelligence Optimization Approach: Sparrow Search Algorithm,” *Systems Science & Control Engineering*, 8(1), 22–34.  
DOI: https://doi.org/10.1080/21642583.2019.1708830

**User-supplied Python reference repository**

https://github.com/Lancephil/Sparrow-Search-Algorithm-Python

Inspect `function.py`, `function_Ver2.py`, the benchmark definitions, and any test scripts. Treat this repository as a secondary reference, not as proof that its equations are correct.

Prioritize fidelity to the original paper over copying a third-party implementation or adding improvements.

## 2. Mandatory source inspection and verification

Before writing the algorithm:

1. Read the original SSA paper, especially the discoverer/producer, joiner/scrounger, and scout/danger-aware equations and the complete algorithm flow.
2. Inspect both `function.py` and `function_Ver2.py` in the supplied repository. Compare equations, indexing, ranking, fitness evaluation order, and boundary handling.
3. Verify each equation against the original paper. If the paper is ambiguous or the repository differs from it, document the discrepancy and explain the choice made.
4. Do not silently reproduce suspicious code or silently change equations to improve benchmark results.
5. If a source file cannot be accessed, report that limitation instead of inventing its behaviour.

### Preliminary issues to audit in the supplied repository

These are review leads. Verify each against the actual source and paper before classifying it as a confirmed error.

- The lower-ranked joiner expression resembles `exp(worse - position / i**2)`. Python evaluates division before subtraction. Check whether the intended expression is `exp((worse - position) / i**2)`.
- The better-ranked joiner update contains an expression involving `1 / (A.T * (A @ A.T))`. Check this against the paper’s pseudoinverse definition. For a nonzero row vector `A`, the conventional pseudoinverse is `A.T / (A @ A.T)`, not the elementwise reciprocal of `A.T * (A @ A.T)`.
- Verify the distribution of `Q`. In the canonical equations it is normally distributed; do not substitute another distribution without justification.
- Check whether the discoverer warning value is sampled once per iteration or separately for each individual.
- Verify scout fitness comparisons and the use of `bestX`, `fMin`, `worse`, and previous/current fitness. Ensure the definitions are consistent with minimization.
- Check NumPy shapes and indexing rather than translating MATLAB-like array expressions mechanically.
- Check array aliasing. In NumPy, `pX = X` does not create an independent copy.
- Check scout selection. A fixed slice of 20 individuals may fail for other population sizes and may not implement the intended configurable proportion.
- The repository’s limited documentation is not evidence that the implementation has been validated.

Record findings in `REFERENCE_AUDIT.md`. Distinguish confirmed errors from uncertain interpretations.

## 3. Recommended project structure

```text
project/
├── algorithms/
│   └── ssa.py
├── benchmarks/
│   └── functions.py
├── experiments/
│   └── run_ssa.py
├── tests/
│   └── test_ssa.py
├── results/
│   └── .gitkeep
├── requirements.txt
├── README.md
└── REFERENCE_AUDIT.md
```

Keep the core algorithm separate from benchmark definitions, experiment configuration, plotting, and result export.

## 4. SSA function interface

Create a reusable function with an interface equivalent to:

```python
ssa(
    population_size,
    max_iterations,
    lower_bound,
    upper_bound,
    dimensions,
    objective_function,
    discoverer_fraction=0.2,
    scout_fraction=0.1,
    safety_threshold=0.8,
    seed=None,
)
```

Use idiomatic Python naming as appropriate. Defaults must be configurable and documented; do not claim these are the only settings used in the original paper.

Return at least:

- `best_position`: one-dimensional NumPy array containing the best solution found.
- `best_fitness`: objective value of that solution.
- `convergence_curve`: best-so-far fitness recorded across iterations.

Document input and output shapes. Support scalar bounds and, where practical, per-dimension bound arrays. Validate incompatible dimensions, invalid bounds, insufficient population sizes, invalid fractions, and invalid parameter values with useful errors.

Use NumPy and `numpy.random.Generator` for controlled randomness. A fixed seed should make runs repeatable within this Python implementation. NumPy and MATLAB need not generate identical random sequences from the same numeric seed.

Assume minimization unless there is a clear reason to support maximization. If maximization is supported, document it and keep ranking and scout conditions consistent.

## 5. Mathematical components

Use the original paper as the authority for exact notation, indexing, and update order. The equations below are included as a guide to verify, not as a reason to ignore discrepancies found in the original source.

### 5.1 Population representation

Each sparrow is a candidate vector:

$$
\mathbf{X}_i = [x_{i,1}, x_{i,2}, \ldots, x_{i,D}]
$$

The population matrix has shape `(N, D)`, where `N` is the number of sparrows and `D` is the number of decision variables. Fitness is a vector of length `N`.

For minimization, smaller objective values are better. At each iteration, identify the current best and worst positions and maintain the global best-so-far separately.

### 5.2 Initialization

Initialize each coordinate uniformly within its bounds:

$$
X_{i,j}^{0} = L_j + r_{i,j}(U_j-L_j),
\qquad r_{i,j}\sim U(0,1)
$$

Evaluate every initial candidate. Initialize current fitness, best-so-far position and fitness, and convergence history.

### 5.3 Discoverer (producer) movement

Verify the discoverer equation against the paper:

$$
X_{i,j}^{t+1} =
\begin{cases}
X_{i,j}^{t}\exp\left(-\dfrac{i}{\alpha T}\right),
& R_2 < ST,\\[6pt]
X_{i,j}^{t}+QL,
& R_2 \ge ST.
\end{cases}
$$

Confirm the precise meaning of index `i` in the exponent; do not confuse the sparrow rank/index with iteration `t`. Confirm the distributions and dimensions of `alpha`, `R2`, `Q`, and `L`. In the canonical presentation, `Q` is normally distributed and `L` is a vector of ones. Check whether the warning value is sampled once per iteration or per individual in the reference formulation.

The safety threshold `ST` typically lies between 0.5 and 1.0. Preserve strict versus inclusive comparisons from the chosen authoritative source.

### 5.4 Joiner (scrounger/follower) movement

Verify the equation and rank convention against the original paper:

$$
X_{i,j}^{t+1} =
\begin{cases}
Q\exp\left(
\dfrac{X_{\mathrm{worst},j}^{t}-X_{i,j}^{t}}{i^2}
\right), & i > N/2,\\[8pt]
X_{P,j}^{t+1}+
\left|X_{i,j}^{t}-X_{P,j}^{t+1}\right| A^+L,
& i\le N/2.
\end{cases}
$$

The notation and update timing may differ across secondary implementations. Confirm what `X_P` represents in the original paper and whether the equation refers to a current or already-updated discoverer position. Do not use an expression whose operator precedence changes the intended numerator.

The paper defines a random sign vector `A` and its pseudoinverse `A+`. For a nonzero row vector, the conventional pseudoinverse is:

$$
A^+ = \frac{A^T}{AA^T}
$$

Verify the vector/matrix shapes and use an algebraically correct implementation. Do not replace the operation with an unrelated random perturbation without documenting the deviation. Confirm the distribution of `Q` from the paper.

### 5.5 Scout (danger-aware) movement

Verify the scout equation directly against the original paper. A commonly reproduced form is:

$$
X_{i,j}^{t+1} =
\begin{cases}
X_{\mathrm{best},j}^{t}
+\beta\left|X_{i,j}^{t}-X_{\mathrm{best},j}^{t}\right|,
& f_i > f_g,\\[6pt]
X_{i,j}^{t}
+K\left(
\dfrac{\left|X_{i,j}^{t}-X_{\mathrm{worst},j}^{t}\right|}
{f_i-f_w+\varepsilon}
\right),
& f_i=f_g.
\end{cases}
$$

Here, `beta` is typically normally distributed, `K` is a random factor in `[-1, 1]`, `f_g` and `f_w` denote best and worst fitness under the paper’s notation, and `epsilon` prevents division by zero.

Verify the exact fitness definitions and comparisons. For minimization, the global best has the smallest objective value and the global worst the largest. Ensure the inequalities are consistent with this convention. Check whether the comparisons use current-best or global-best fitness. Do not silently reverse or reinterpret conditions.

Make the scout proportion configurable rather than hard-coding a fixed number of individuals. Verify the intended selection process in the paper. Determine whether scout updates operate on original positions or positions already changed by discoverer/joiner updates.

### 5.6 Ranking, update order, and fitness memory

Update order affects results. Implement and document the sequence specified by the paper. A likely outline to verify is:

1. Rank candidates and identify current best/worst.
2. Update discoverers.
3. Update joiners using the positions required by the paper.
4. Select and update scouts.
5. Enforce bounds at the specified point.
6. Evaluate the relevant new candidates.
7. Update fitness memory and global best-so-far.
8. Record convergence.

Adjust this sequence if the paper specifies otherwise, and document the reason. Avoid mixing a sorted index with an original population index. If using sorted indices, preserve the mapping between rank and stored candidate.

Do not introduce MPA’s elite matrix, Brownian motion, Lévy flights, memory mechanism, or FADs effect. These are MPA concepts, not SSA components.

## 6. Boundary handling

Candidates must remain within bounds. Verify the intended boundary handling in the paper or chosen authoritative reference. If no method is specified, use a documented policy such as clipping:

$$
X_{i,j}\leftarrow \min\left(U_j,\max(L_j,X_{i,j})\right)
$$

Do not claim clipping is mandated by the paper if it is a chosen implementation policy. Apply boundary handling consistently and test it. Support per-dimension bounds if the interface accepts them.

## 7. Benchmark functions

Implement benchmarks independently in `benchmarks/functions.py`. Each function accepts a one-dimensional NumPy vector and returns a scalar:

- **Sphere:** `f(x) = sum(x_i**2)`; global minimum `0` at the zero vector.
- **Rastrigin:** multimodal; global minimum `0` at the zero vector.
- **Rosenbrock:** curved valley; global minimum `0` at the all-ones vector.
- **Ackley:** multimodal; global minimum `0` at the zero vector.
- **Griewank:** multimodal; global minimum `0` at the zero vector.

Use standard, documented formulas and appropriate bounds. Start validation with Sphere. Do not hard-code a benchmark inside SSA; pass the objective function as an argument. Independently verify formulas and bounds before reusing definitions from the supplied repository.

## 8. Testing and validation

Add automated tests or a validation script checking:

- The returned best position has shape `(dimensions,)`.
- Best fitness is a finite scalar for valid objective functions.
- The convergence curve has documented length and contains finite values.
- For minimization, the best-so-far curve is non-increasing within floating-point tolerance.
- All evaluated candidate positions remain within bounds.
- A fixed seed produces repeatable results within Python.
- Sphere optimization approaches its known optimum under reasonable settings.
- At least one additional benchmark, such as Rastrigin or Rosenbrock, runs successfully.
- Small populations and non-default dimensions/bounds do not cause indexing errors.
- Scout selection works for population sizes other than 100 and scout fractions other than 0.1.
- Equations work in one-dimensional and multi-dimensional search spaces.
- Equal fitness values and the scout denominator are handled safely.

SSA is stochastic; do not expect every run to reach the exact global optimum. Use multiple independent runs for performance evaluation. Do not write tests that assume one exact final fitness for every seed.

Use modest development settings, such as population size 30, 10 dimensions, and 100 iterations, for quick checks. These are development settings, not a claim about the original paper’s experimental configuration.

## 9. Experiment runner

Create `experiments/run_ssa.py` that:

1. Selects a benchmark function.
2. Defines dimension, bounds, population size, iteration count, parameters, and seed.
3. Runs SSA.
4. Prints best fitness and best position.
5. Plots the convergence curve with Matplotlib.
6. Optionally saves results to CSV under `results/`.

Keep plotting and file output outside the core algorithm.

## 10. Code quality and documentation

- Use clear function names, type hints where useful, and concise docstrings.
- Comment major algorithm stages and explain the equations without commenting every line.
- Explain discoverer movement, joiner movement, scout movement, safety threshold, random variables, rank ordering, and global-best tracking.
- Include installation and run instructions in `README.md`.
- Keep dependencies minimal: NumPy and Matplotlib for the initial implementation; pytest only if tests are added.
- Do not add unrelated packages, GPU support, or premature optimization.
- Do not implement MPA or Quantum-Swarm in this task.

## 11. Scope boundaries

Implement **vanilla SSA only**.

Do not add opposition-based learning, adaptive/nonlinear parameter schedules, mutation operators, local-search enhancements, hybridization, quantum-inspired candidate generation, diversity monitoring, or stagnation detection. Such additions would create a modified SSA variant and are outside this task.

Do not claim the implementation is identical to the original paper until equations, update order, and parameter definitions have been checked. If the paper is ambiguous, document the ambiguity and selected interpretation. If the third-party repository disagrees with the paper, prioritize the paper and explain the difference.

## 12. Expected deliverables

Provide:

1. `algorithms/ssa.py` — core SSA implementation.
2. `benchmarks/functions.py` — benchmark functions.
3. `experiments/run_ssa.py` — example experiment and convergence plot.
4. `tests/test_ssa.py` — correctness and reproducibility tests.
5. `requirements.txt` — minimal dependencies.
6. `README.md` — setup, usage, parameters, return values, and references.
7. `REFERENCE_AUDIT.md` — comparison of the paper and supplied Python repository, with confirmed issues, ambiguities, and implementation decisions.

At completion, summarize:

- Which reference sources and files were inspected.
- How each SSA component maps to the original equations.
- Which tests and benchmarks were actually run.
- Discrepancies with the third-party repository and how they were handled.
- Any ambiguities in the paper and implementation choices made.

Do not report tests as passing unless they were actually executed.
