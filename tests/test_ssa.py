"""Equation regressions and public contract checks; no exact stochastic targets."""

import csv

import numpy as np
import pytest

from algorithms import ssa
from algorithms.ssa import _producer_step, _scout_step, _scrounger_step
from benchmarks.functions import BENCHMARKS, ackley, griewank, rastrigin, rosenbrock, sphere


def run(**overrides):
    arguments = dict(population_size=30, max_iterations=100, lower_bound=-100,
                     upper_bound=100, dimensions=10, objective_function=sphere, seed=42)
    arguments.update(overrides)
    return ssa(**arguments)


@pytest.mark.parametrize("name", BENCHMARKS)
def test_benchmark_optima(name):
    benchmark = BENCHMARKS[name]
    optimum = np.ones(5) if name == "rosenbrock" else np.zeros(5)
    assert benchmark.function(optimum) == pytest.approx(0, abs=1e-12)


def test_benchmark_nonoptimum_values():
    assert sphere([1, 2]) == 5
    assert rastrigin([1, 2]) == pytest.approx(5)
    assert rosenbrock([0, 0]) == 1
    assert ackley([1, 1]) == pytest.approx(20 * (1 - np.exp(-0.2)))
    assert griewank([1, 2]) == pytest.approx(5 / 4000 - np.cos(1) * np.cos(np.sqrt(2)) + 1)


@pytest.mark.parametrize("x", [[], [[1, 2]], [np.nan], [np.inf]])
def test_invalid_benchmark_vectors(x):
    with pytest.raises(ValueError):
        sphere(x)


def test_rosenbrock_rejects_degenerate_one_dimension():
    with pytest.raises(ValueError, match="two dimensions"):
        rosenbrock([1])


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_sphere_convergence_and_result_contract(seed):
    result = run(seed=seed)
    assert result.best_position.shape == (10,)
    assert isinstance(result.best_fitness, float) and np.isfinite(result.best_fitness)
    assert result.convergence_curve.shape == (100,)
    assert np.all(np.isfinite(result.convergence_curve))
    assert np.all(np.diff(result.convergence_curve) <= 1e-12)
    assert result.convergence_curve[-1] == result.best_fitness
    assert sphere(result.best_position) == result.best_fitness
    assert result.best_fitness < 1e-6


def test_reproducibility_and_no_global_rng_mutation():
    np.random.seed(123)
    expected = np.random.random(4)
    np.random.seed(123)
    first, second = run(), run()
    np.testing.assert_array_equal(first.best_position, second.best_position)
    np.testing.assert_array_equal(first.convergence_curve, second.convergence_curve)
    assert first.best_fitness == second.best_fitness
    np.testing.assert_array_equal(np.random.random(4), expected)


@pytest.mark.parametrize("population,dimensions,scouts", [(2, 1, 0), (3, 1, 1),
                                                          (7, 4, 0.37), (31, 3, 0.2)])
def test_all_evaluations_bounded_and_no_objective_aliasing(population, dimensions, scouts):
    lower = np.arange(dimensions) + 2.0
    upper = lower + 0.5
    evaluated = []

    def objective(x):
        assert x.shape == (dimensions,)
        assert np.all(np.isfinite(x)) and np.all(x >= lower) and np.all(x <= upper)
        evaluated.append(x.copy())
        value = sphere(x - lower)
        x[:] = np.nan  # An objective cannot corrupt the algorithm's arrays.
        return value

    result = run(population_size=population, dimensions=dimensions,
                 lower_bound=lower, upper_bound=upper, scout_fraction=scouts,
                 objective_function=objective, max_iterations=12)
    assert result.best_fitness == sphere(result.best_position - lower)
    assert any(np.array_equal(result.best_position, x) for x in evaluated)
    np.testing.assert_array_equal(lower, np.arange(dimensions) + 2.0)


@pytest.mark.parametrize("name", ["rastrigin", "rosenbrock", "ackley", "griewank"])
def test_other_benchmarks_run(name):
    benchmark = BENCHMARKS[name]
    result = run(objective_function=benchmark.function, lower_bound=benchmark.lower_bound,
                 upper_bound=benchmark.upper_bound, max_iterations=20)
    assert np.isfinite(result.best_fitness)
    assert benchmark.function(result.best_position) == result.best_fitness
    assert np.all(np.diff(result.convergence_curve) <= 1e-12)


def test_zero_iterations_returns_best_initial_candidate():
    seen = []

    def objective(x):
        seen.append(x)
        return sphere(x)

    result = run(max_iterations=0, objective_function=objective)
    assert len(seen) == 30 and result.convergence_curve.shape == (0,)
    assert result.best_fitness == min(map(sphere, seen))


@pytest.mark.parametrize("value", [0.0, 1.0, -3.0])
def test_equal_fitness_and_all_scouts(value):
    result = run(objective_function=lambda x: value, population_size=7,
                 scout_fraction=1.0, max_iterations=20)
    assert result.best_fitness == value
    np.testing.assert_array_equal(result.convergence_curve, np.full(20, value))


@pytest.mark.parametrize("overrides,match", [
    ({"population_size": 1}, "population_size"),
    ({"population_size": 3.5}, "population_size"),
    ({"dimensions": 0}, "dimensions"),
    ({"dimensions": True}, "dimensions"),
    ({"max_iterations": -1}, "max_iterations"),
    ({"max_iterations": 1.5}, "max_iterations"),
    ({"discoverer_fraction": 0}, "discoverer_fraction"),
    ({"discoverer_fraction": 1}, "discoverer_fraction"),
    ({"scout_fraction": -0.1}, "scout_fraction"),
    ({"scout_fraction": 1.1}, "scout_fraction"),
    ({"safety_threshold": 0.4}, "safety_threshold"),
    ({"safety_threshold": np.nan}, "safety_threshold"),
    ({"lower_bound": [0, 0]}, "lower_bound"),
    ({"upper_bound": np.inf}, "upper_bound"),
    ({"lower_bound": 100}, "bounds"),
    ({"lower_bound": 101}, "bounds"),
    ({"lower_bound": -1e308, "upper_bound": 1e308}, "finite spans"),
    ({"objective_function": None}, "callable"),
    ({"seed": -1}, "seed"),
])
def test_invalid_parameters(overrides, match):
    with pytest.raises(ValueError, match=match):
        run(**overrides)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [1.0], 1j, "1", True])
def test_invalid_objective_returns(value):
    with pytest.raises(ValueError, match="finite real scalar"):
        run(objective_function=lambda x: value)


class FixedRandom:
    """Known draws make equation tests independent of random-generator order."""

    def random(self):
        return 0.5

    def normal(self):
        return -0.5

    def uniform(self, low, high):
        assert (low, high) == (-1.0, 1.0)
        return 0.25

    def choice(self, values, size):
        assert values == [-1.0, 1.0]
        return np.array([1.0, -1.0])[:size]


def test_producer_equation_rank_and_alarm_equality():
    x = np.array([2.0, -4.0])
    np.testing.assert_allclose(_producer_step(x, 1, 10, 0.3, 0.8, FixedRandom()),
                               x * np.exp(-1 / 5))
    np.testing.assert_array_equal(_producer_step(x, 1, 10, 0.8, 0.8, FixedRandom()),
                                  x - 0.5)


def test_scrounger_exponent_parentheses_and_negative_normal_draw():
    x, worst = np.array([2.0, 4.0]), np.array([6.0, 12.0])
    actual = _scrounger_step(x, 4, 6, worst, np.zeros(2), FixedRandom())
    np.testing.assert_allclose(actual, -0.5 * np.exp(np.array([4.0, 8.0]) / 16))


def test_scrounger_pseudoinverse_shapes_and_half_population_equality():
    x, producer = np.array([5.0, 4.0]), np.array([1.0, 2.0])
    # |x-Xp| @ ([1,-1]^T / 2) = (4-2)/2 = 1, broadcast to both axes.
    actual = _scrounger_step(x, 3, 6, np.zeros(2), producer, FixedRandom())
    np.testing.assert_allclose(actual, [2, 3])


def test_scout_equations_and_epsilon_cancellation():
    rng = FixedRandom()
    x, best, worst = np.array([3.0, 5.0]), np.array([1.0, 2.0]), np.array([6.0, 8.0])
    np.testing.assert_allclose(_scout_step(x, 4, best, 1, worst, 10, rng),
                               best - 0.5 * np.abs(x - best))
    np.testing.assert_allclose(_scout_step(best, 1, best, 1, worst, 10, rng),
                               best + 0.25 * np.abs(best - worst) / (-9 + np.finfo(float).eps))
    assert np.all(np.isfinite(_scout_step(best, 0, best, 0, worst, np.finfo(float).eps, rng)))


def test_scout_count_selection_and_warning_draw(monkeypatch):
    import importlib
    module = importlib.import_module("algorithms.ssa")
    original_scout, original_producer = module._scout_step, module._producer_step
    scouts, warnings = [], []

    def scout(*args):
        scouts.append(args[0].copy())
        return original_scout(*args)

    def producer(*args):
        warnings.append(args[3])
        return original_producer(*args)

    monkeypatch.setattr(module, "_scout_step", scout)
    monkeypatch.setattr(module, "_producer_step", producer)
    run(population_size=11, max_iterations=1, scout_fraction=0.37,
        discoverer_fraction=0.3)
    assert len(scouts) == 4 and len(np.unique(scouts, axis=0)) == 4
    assert len(warnings) == 3 and len(set(warnings)) == 1


def test_extreme_exponent_moves_are_clipped_before_evaluation():
    result = run(population_size=4, dimensions=1, lower_bound=-1e6,
                 upper_bound=1e6, scout_fraction=1, max_iterations=4)
    assert np.isfinite(result.best_fitness)


def test_runner_csv_and_plot(tmp_path, capsys):
    from experiments.run_ssa import main
    main(["--benchmark", "rastrigin", "--dimensions", "2", "--population-size", "7",
          "--iterations", "3", "--runs", "2", "--no-show", "--save-csv",
          "--save-plot", "--output-dir", str(tmp_path)])
    with (tmp_path / "rastrigin_seed42.csv").open() as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["iteration", "best_fitness"] and len(rows) == 4
    assert (tmp_path / "rastrigin_seed43_best.csv").exists()
    assert (tmp_path / "rastrigin_convergence.png").read_bytes().startswith(b"\x89PNG")
    assert "Fitness across 2 runs" in capsys.readouterr().out
