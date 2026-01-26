//! Basic solver tests.

use solverang::{Problem, Solver, SolverConfig};

#[test]
fn test_solver_quadratic() {
    // Solve x^2 - 2 = 0 to get sqrt(2)
    struct SqrtTwo;

    impl Problem for SqrtTwo {
        fn name(&self) -> &str {
            "sqrt(2)"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            1
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] - 2.0]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 2.0 * x[0])]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![factor]
        }
    }

    let problem = SqrtTwo;
    let solver = Solver::new(SolverConfig::default());
    let result = solver.solve(&problem, &[1.5]);

    assert!(result.is_converged());
    let solution = result.solution().expect("should have solution");
    assert!(
        (solution[0] - std::f64::consts::SQRT_2).abs() < 1e-6,
        "Expected sqrt(2), got {}",
        solution[0]
    );
}

#[test]
fn test_solver_2d_system() {
    // Solve x^2 + y^2 = 1, x = y
    // Solution: x = y = 1/sqrt(2)
    struct CircleLine;

    impl Problem for CircleLine {
        fn name(&self) -> &str {
            "circle-line"
        }
        fn residual_count(&self) -> usize {
            2
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![
                (0, 0, 2.0 * x[0]),
                (0, 1, 2.0 * x[1]),
                (1, 0, 1.0),
                (1, 1, -1.0),
            ]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![0.5 * factor, 0.5 * factor]
        }
    }

    let problem = CircleLine;
    let solver = Solver::new(SolverConfig::default());
    let result = solver.solve(&problem, &[0.5, 0.5]);

    assert!(result.is_converged());
    let solution = result.solution().expect("should have solution");
    let expected = 1.0 / std::f64::consts::SQRT_2;
    assert!(
        (solution[0] - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        solution[0]
    );
    assert!(
        (solution[1] - expected).abs() < 1e-6,
        "Expected {}, got {}",
        expected,
        solution[1]
    );
}

#[test]
fn test_solver_with_known_solution() {
    use solverang::test_problems::Rosenbrock;

    let problem = Rosenbrock;
    let solver = Solver::new(SolverConfig::default());
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    assert!(result.is_converged());

    let solution = result.solution().expect("should have solution");
    let known = problem
        .known_solution()
        .expect("should have known solution");

    for (actual, expected) in solution.iter().zip(known.iter()) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            actual
        );
    }
}

#[test]
fn test_solver_configs() {
    use solverang::test_problems::Rosenbrock;

    let problem = Rosenbrock;
    let x0 = problem.initial_point(1.0);

    // Test fast config
    let solver_fast = Solver::new(SolverConfig::fast());
    let result_fast = solver_fast.solve(&problem, &x0);
    assert!(result_fast.is_converged());

    // Test robust config
    let solver_robust = Solver::new(SolverConfig::robust());
    let result_robust = solver_robust.solve(&problem, &x0);
    assert!(result_robust.is_converged());

    // Test loose config
    let solver_loose = Solver::new(SolverConfig::loose());
    let result_loose = solver_loose.solve(&problem, &x0);
    assert!(result_loose.is_converged());
}

#[test]
fn test_solve_result_methods() {
    use solverang::test_problems::Rosenbrock;

    let problem = Rosenbrock;
    let solver = Solver::new(SolverConfig::default());
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    assert!(result.is_converged());
    assert!(result.is_completed());
    assert!(result.solution().is_some());
    assert!(result.residual_norm().is_some());
    assert!(result.iterations().is_some());
    assert!(result.error().is_none());

    // Residual norm should be very small for converged problems
    assert!(result.residual_norm().expect("should have norm") < 1e-6);
}
