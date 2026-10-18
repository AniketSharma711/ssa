"""Run a benchmark, plot convergence, and optionally export CSV/PNG results."""

import argparse
import csv
from pathlib import Path

import numpy as np

from algorithms import ssa
from benchmarks.functions import BENCHMARKS


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="sphere")
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--discoverer-fraction", type=float, default=0.2)
    parser.add_argument("--scout-fraction", type=float, default=0.1)
    parser.add_argument("--safety-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1,
                        help="Independent runs using seed, seed+1, ...")
    parser.add_argument("--lower-bound", type=float)
    parser.add_argument("--upper-bound", type=float)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-plot", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="Use a headless plot backend")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args(argv)
    benchmark = BENCHMARKS[args.benchmark]
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.dimensions < benchmark.minimum_dimensions:
        parser.error(f"{args.benchmark} needs at least {benchmark.minimum_dimensions} dimensions")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.save_csv or args.save_plot:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    scores = []
    try:
        for run in range(args.runs):
            seed = args.seed + run
            try:
                result = ssa(
                    args.population_size, args.iterations,
                    benchmark.lower_bound if args.lower_bound is None else args.lower_bound,
                    benchmark.upper_bound if args.upper_bound is None else args.upper_bound,
                    args.dimensions, benchmark.function, args.discoverer_fraction,
                    args.scout_fraction, args.safety_threshold, seed,
                )
            except ValueError as error:
                parser.error(str(error))
            scores.append(result.best_fitness)
            print(f"Run {run + 1}, seed {seed}: best fitness = {result.best_fitness:.12g}")
            print(f"Best position: {np.array2string(result.best_position, precision=8)}")
            axis.plot(np.arange(1, args.iterations + 1), result.convergence_curve,
                      label=f"seed {seed}")
            if args.save_csv:
                stem = args.output_dir / f"{args.benchmark}_seed{seed}"
                with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(["iteration", "best_fitness"])
                    writer.writerows(enumerate(result.convergence_curve, start=1))
                with Path(f"{stem}_best.csv").open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(["best_fitness", *[f"x{i+1}" for i in range(args.dimensions)]])
                    writer.writerow([result.best_fitness, *result.best_position])
        if args.runs > 1:
            print(f"Fitness across {args.runs} runs: min={min(scores):.12g}, "
                  f"mean={np.mean(scores):.12g}, std={np.std(scores):.12g}")
        axis.set(xlabel="Iteration", ylabel="Best-so-far fitness",
                 title=f"SSA: {args.benchmark} ({args.dimensions} dimensions)")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        if args.save_plot:
            figure.savefig(args.output_dir / f"{args.benchmark}_convergence.png", dpi=160)
        if not args.no_show:
            plt.show()
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
