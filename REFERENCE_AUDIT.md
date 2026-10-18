# SSA reference audit

Inspected on 2026-10-15, before implementing the algorithm.

## Sources and access

The original article is Xue and Shen (2020), *A novel swarm intelligence
optimization approach: sparrow search algorithm*, DOI
[10.1080/21642583.2019.1708830](https://doi.org/10.1080/21642583.2019.1708830).
The DOI and direct publisher full-text endpoints failed in the browser tool.
The publisher-provided full text was accessible on
[ResearchGate](https://www.researchgate.net/publication/338386846_A_novel_swarm_intelligence_optimization_approach_sparrow_search_algorithm).
Read section 2.2, equations (1)--(5), Algorithm 1, benchmark Tables 1--2,
and the experimental parameter paragraph. This verifies the published equations
and algorithm text, not the authors' original MATLAB source, which was not supplied
or inspected. The accessible article identifies itself as CC BY 4.0.

Cloned the supplied [Lancephil repository](https://github.com/Lancephil/Sparrow-Search-Algorithm-Python)
at commit `0d7c8c20a13712ed2db94eb5a9512bd037353a34`. Read all five tracked files:
`function.py`, `function_Ver2.py`, `test.py`, `test1.py`, and `README.md`.
The `Parameters` and `fun` functions contain its benchmarks; no separate benchmark
module exists. Neither test script is an automated assertion suite, and both
import `function`, so neither validates `function_Ver2`. The brief's named source
files were all accessible. A local ignored clone is in `.reference/ssa-python`.
No third-party algorithm code was copied into the implementation.

Independently checked benchmark formulas against the original paper and the SFU
pages and linked MATLAB formula listings for
[Sphere](https://www.sfu.ca/~ssurjano/spheref.html),
[Rastrigin](https://www.sfu.ca/~ssurjano/rastr.html),
[Rosenbrock](https://www.sfu.ca/~ssurjano/rosen.html),
[Ackley](https://www.sfu.ca/~ssurjano/ackley.html), and
[Griewank](https://www.sfu.ca/~ssurjano/griewank.html).
The independently written NumPy formulas use paper Tables 1--2 bounds; Ackley's
paper bound is 32 rather than the SFU page's common 32.768. Neither is universal.

## Confirmed reference-code findings

These findings apply to both algorithm files except where noted.

1. **Joiner exponent:** `exp(worse-pX[...] / square(i))` divides only the old
   position. Equation (4) divides the whole worst-minus-position difference.
   This implementation uses parentheses around that difference.
2. **Q and beta distributions:** producer danger moves and starving joiner moves
   use `rand(1)` (uniform [0,1)), but equations (3)--(4) specify normally
   distributed Q. Scouts use `rand(1,dim)` for beta, whereas equation (5) specifies
   standard normal beta. This implementation uses scalar standard normals.
   The paper says normal Q without specifying its mean/variance; N(0,1) is an
   explicit conventional choice. Beta's mean 0 and variance 1 are specified.
3. **Rank origin:** the safe producer loop starts at Python index zero, making
   the first producer's decay factor exactly 1. Algorithm 1 starts `i=1`.
   Joiner threshold and denominator likewise use zero-based indices. We use
   one-based sorted fitness rank throughout, including the exact `i > N/2` test.
4. **Aliasing:** `pFit=fit` and `pX=X` share storage, so movement changes the
   purported historical values. Greedy comparisons of `fit` and `pFit` therefore
   compare the same array. `worse` and `bestXX` are views and may change while
   they are still being used as references. `bestX` is also a view in `function.py`,
   allowing its coordinates to cease matching `fMin`. Ver2 deep-copies later
   improvements to bestX, but leaves initialization and other aliases intact.
   Here all iteration references and the best-so-far solution are independent
   copies; fitness always belongs to the accompanying position.
5. **Producer leader:** `bestXX` is chosen from all fitness entries after updating
   only producers. It can therefore be an old scrounger. The paper identifies
   `Xp^(t+1)` as the optimal producer position. We evaluate updated producers and
   choose the best among those producers only.
6. **Scout fraction:** `c[0:20]` chooses a fixed 20 candidates, or all candidates
   when N<20 (the slice itself does not raise an index error). It does not enforce
   a configurable proportion. The paper describes 10--20%; its reported benchmark
   configuration uses 10%. We sample a configurable number without replacement.
7. **Scout indexing:** `b=sortIndex[0,c[0:20]]` already contains population
   indices, yet scout movement indexes `sortIndex[0,b[j]]` again. A permutation
   composed with itself is still a permutation, so this alone does **not** break
   uniform subset selection or cause an out-of-range index. It is redundant and
   confuses rank/identity tracing. We select sorted row indices exactly once.
8. **Scout state:** the intended pFit/pX history is aliased, while fMin is a
   historical scalar and worse/fmax come from the start of the iteration. These
   references can describe different states. The paper defines fi, fg, fw as
   present/current fitness. We compare all three on one frozen t snapshot,
   independent of best-so-far; the minimization inequalities are not reversed.

## Suspicious expression that is not a confirmed error

The repository uses `1 / (A.T * (A @ A.T))`. For a general nonzero real row A,
this is not the pseudoinverse. **However, its A consists only of ±1**, so each
element satisfies `1/A_j = A_j`; `A @ A.T = D`. The expression therefore equals
`A.T / D` for every A that this repository actually generates. It is algebraically
equivalent in this restricted setting, not a confirmed numerical defect.
The implementation uses `A / dot(A,A)` as a one-dimensional representation of
the column pseudoinverse to state the mathematics directly. Tests verify the
projected scalar and broadcasting, including D=1 in full optimizer runs.

The shared alarm draw `r2` is also **correct**: Algorithm 1 draws it once before
the producer loop. The reference's boundary routine clips, a reasonable policy,
but that does not prove clipping is mandated by the paper.

The reference's Sphere and Rosenbrock formulas match the standard mathematical
definitions. Ver2 adds the two-dimensional six-hump camel function and changes
Sphere's default range/dimension. Those are configuration changes, not proof of
equation correctness. Its F6 lacks the apparent rounding/floor notation of the
paper's step benchmark; because the printed notation is ambiguous and F6 is not
required here, this is left as an unresolved source-definition difference.

## Equation mapping and explicit implementation decisions

- **Initialization / equations (1)--(2):** `(N,D)` positions and `(N,)` fitness;
  independent coordinatewise uniform initialization. We minimize scalar fitness.
- **Equation (3):** ranks 1 through PD, common uniform R2 each iteration,
  alpha uniform `(0,1]` per safe producer, scalar normal Q per alarmed producer,
  and broadcasting along the `(D,)` ones vector. Strict `< ST` and inclusive
  `>= ST` are preserved. Alpha's distribution and Q's standardization are choices
  consistent with the stated ranges; the article does not specify all draws.
- **Equation (4):** worst position is frozen at t; Xp is the best producer after
  its update. The starving branch has normal scalar Q and the full numerator
  divided by the squared one-based rank. The other branch treats the absolute
  distance as a row, multiplies by A+ to obtain a scalar, then multiplies by L.
  The paper's coordinate subscripts mixed with matrix notation are dimensionally
  terse; this row-vector reading preserves its pseudoinverse and ones matrix.
- **Equation (5):** scalar N(0,1) beta and scalar uniform K in `[-1,1]` per scout;
  branches are `fi > fg` and `fi == fg` using current population minimum/maximum.
  Sorted fi cannot be smaller than fg. Equality is exact, as printed. Epsilon is
  float64 machine epsilon; if addition cancels an exactly negative-epsilon gap,
  use negative epsilon to avoid division by zero. This is a numerical guard,
  not an adaptive search rule. All-equal objectives exercise the other zero gap.
- **Algorithm 1:** stably rank, snapshot current references, draw warning, update
  producers, update scroungers, randomly select/update scouts, evaluate final
  proposals, accept strict improvements per individual, record best-so-far.
  Evaluate producers separately to determine Xp and include those evaluated
  positions in the global archive. No personal-best PSO trajectory is introduced.

The paper leaves some procedural details unresolved. The implementation makes
these choices explicit rather than claiming unique reproduction:

1. The acceptance instruction in Algorithm 1 does not clearly say whether to
   retain each individual or only update the global record. We use per-individual
   greedy acceptance, consistent with an ordinary reading of that step and the
   reference repository's apparent intent. The aliasing in that repository is
   not evidence for replacing acceptance with unconditional movement.
2. Scout equations refer to t positions and fitness despite appearing after the
   other movement loops. We read the superscripts literally: scouts overwrite
   their proposals using frozen t positions/fitness, not partially updated
   producer/joiner arrays. Intermediate joiner proposals overwritten by scouts
   are not evaluated. Selecting new uniform scout identities every iteration
   is a documented interpretation; the paper gives a count and random selection
   language without a complete resampling schedule.
3. The article does not prescribe a detailed boundary repair. We clip before
   every objective evaluation. An exponential overflow can yield signed infinity,
   which clipping maps to the corresponding bound; NaN moves raise an error.
   The finite-bounds/span and finite-objective contracts reject unusable inputs.
4. Fractions become integer counts by floor with the clamps documented in the
   API. The paper does not specify rounding for arbitrary small populations.
   Zero scouts is explicitly allowed; at least one producer and scrounger remain.
5. Every objective receives a copy and must return a finite real scalar. T=0
   means initialization only. Convergence excludes initialization and has T
   entries. These are API choices, not claims about MATLAB output shapes.

These choices prioritize the published equations. No opposition learning,
mutation, local search, quantum operators, MPA mechanisms, diversity schedules,
or benchmark-dependent modifications are included.

## Validation procedure

Run `python -m pytest -q` and the headless runner examples in README.md. Tests
use hand-calculated equation cases, bounds-recording objectives, nonzero benchmark
values, fixed-seed reproducibility, and multiple Sphere seeds. They also cover
all five benchmarks and CSV/PNG export. Actual execution results are recorded in
`VALIDATION.md`; smoke-test success is not a replication of the paper's MATLAB
experiments or evidence of superiority over other algorithms.
