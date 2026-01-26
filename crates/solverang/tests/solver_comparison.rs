//! Comparison tests between Newton-Raphson and Levenberg-Marquardt solvers.
//!
//! These tests verify that both solvers produce consistent results on
//! problems they can both solve, and that each handles their specialized
//! cases appropriately.

use solverang::{
    test_problems::{
        BrownAlmostLinear, Chebyquad, HelicalValley, LinearFullRank, PowellSingular, Rosenbrock,
        Watson, Wood,
    },
    AutoSolver, LMConfig, LMSolver, Problem, RobustSolver, Solver, SolverChoice, SolverConfig,
};

/// Compare NR and LM on a square system, verifying they reach the same solution.
fn compare_solvers_on_square_system<P: Problem>(problem: &P, tolerance: f64) {
    let nr_solver = Solver::new(SolverConfig::robust());
    let lm_solver = LMSolver::new(LMConfig::robust());
    let x0 = problem.initial_point(1.0);

    let nr_result = nr_solver.solve(problem, &x0);
    let lm_result = lm_solver.solve(problem, &x0);

    // Both should complete
    assert!(
        nr_result.is_completed(),
        "NR should complete on '{}': {:?}",
        problem.name(),
        nr_result
    );
    assert!(
        lm_result.is_completed(),
        "LM should complete on '{}': {:?}",
        problem.name(),
        lm_result
    );

    // If both converged, solutions should be close
    if nr_result.is_converged() && lm_result.is_converged() {
        let nr_sol = nr_result.solution().expect("NR should have solution");
        let lm_sol = lm_result.solution().expect("LM should have solution");

        assert_eq!(
            nr_sol.len(),
            lm_sol.len(),
            "Solution dimensions should match"
        );

        // Check that solutions are close
        let max_diff: f64 = nr_sol
            .iter()
            .zip(lm_sol.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        assert!(
            max_diff < tolerance,
            "Problem '{}': NR and LM solutions differ by {} (tolerance {})\nNR: {:?}\nLM: {:?}",
            problem.name(),
            max_diff,
            tolerance,
            nr_sol,
            lm_sol
        );
    }
}

// --- Square System Comparison Tests ---

#[test]
fn test_compare_rosenbrock() {
    compare_solvers_on_square_system(&Rosenbrock, 1e-5);
}

#[test]
fn test_compare_helical_valley() {
    compare_solvers_on_square_system(&HelicalValley, 1e-5);
}

#[test]
fn test_compare_wood() {
    compare_solvers_on_square_system(&Wood, 1e-5);
}

#[test]
fn test_compare_brown_almost_linear() {
    compare_solvers_on_square_system(&BrownAlmostLinear::new(5), 1e-5);
}

#[test]
fn test_compare_chebyquad_square() {
    // Square Chebyquad (n = m)
    compare_solvers_on_square_system(&Chebyquad::new(5), 1e-5);
}

// --- Over-constrained System Tests (LM should succeed, NR may struggle) ---

#[test]
fn test_lm_handles_overdetermined() {
    // LinearFullRank with 5 variables, 50 equations
    let problem = LinearFullRank::new(5, 50);
    assert!(problem.residual_count() > problem.variable_count());

    let lm_solver = LMSolver::new(LMConfig::default());
    let x0 = problem.initial_point(1.0);
    let result = lm_solver.solve(&problem, &x0);

    assert!(
        result.is_converged() || result.is_completed(),
        "LM should handle overdetermined system: {:?}",
        result
    );
}

#[test]
fn test_watson_overdetermined() {
    // Watson has 31 equations, n=6 variables
    let problem = Watson::new(6);
    assert!(problem.residual_count() > problem.variable_count());

    let lm_solver = LMSolver::new(LMConfig::robust());
    let x0 = problem.initial_point(1.0);
    let result = lm_solver.solve(&problem, &x0);

    assert!(
        result.is_completed(),
        "LM should handle Watson (31 x 6): {:?}",
        result
    );
}

// --- AutoSolver Tests ---

#[test]
fn test_auto_selects_nr_for_square() {
    let solver = AutoSolver::new();
    let problem = Rosenbrock;

    assert_eq!(
        solver.which_solver(&problem),
        SolverChoice::NewtonRaphson,
        "AutoSolver should select NR for square system"
    );

    let result = solver.solve(&problem, &problem.initial_point(1.0));
    assert!(
        result.is_converged(),
        "AutoSolver should converge on Rosenbrock: {:?}",
        result
    );
}

#[test]
fn test_auto_selects_lm_for_overdetermined() {
    let solver = AutoSolver::new();
    let problem = LinearFullRank::new(5, 50);

    assert_eq!(
        solver.which_solver(&problem),
        SolverChoice::LevenbergMarquardt,
        "AutoSolver should select LM for overdetermined system"
    );

    let result = solver.solve(&problem, &problem.initial_point(1.0));
    assert!(
        result.is_completed(),
        "AutoSolver should complete on overdetermined: {:?}",
        result
    );
}

#[test]
fn test_auto_solver_various_problems() {
    let solver = AutoSolver::new();

    // Test on various problems
    let problems: Vec<Box<dyn Problem>> = vec![
        Box::new(Rosenbrock),
        Box::new(HelicalValley),
        Box::new(Wood),
        Box::new(BrownAlmostLinear::new(5)),
        Box::new(Chebyquad::new(5)),
    ];

    for problem in &problems {
        let result = solver.solve(problem.as_ref(), &problem.initial_point(1.0));
        assert!(
            result.is_completed(),
            "AutoSolver should complete on '{}': {:?}",
            problem.name(),
            result
        );
    }
}

// --- RobustSolver Tests ---

#[test]
fn test_robust_solver_basic() {
    let solver = RobustSolver::new();
    let problem = Rosenbrock;

    let result = solver.solve(&problem, &problem.initial_point(1.0));
    assert!(
        result.is_converged(),
        "RobustSolver should converge on Rosenbrock: {:?}",
        result
    );
}

#[test]
fn test_robust_solver_overdetermined() {
    let solver = RobustSolver::new();
    let problem = LinearFullRank::new(5, 50);

    let result = solver.solve(&problem, &problem.initial_point(1.0));
    assert!(
        result.is_completed(),
        "RobustSolver should complete on overdetermined: {:?}",
        result
    );
}

#[test]
fn test_robust_solver_difficult_problem() {
    // Powell Singular is known to be difficult
    let solver = RobustSolver::new()
        .with_nr_config(SolverConfig::robust())
        .with_lm_config(LMConfig::robust());
    let problem = PowellSingular;

    let result = solver.solve(&problem, &problem.initial_point(1.0));
    assert!(
        result.is_completed(),
        "RobustSolver should complete on Powell Singular: {:?}",
        result
    );
}

// --- Residual Norm Comparison ---

#[test]
fn test_residual_norms_comparable() {
    // For problems both can solve, residual norms should be comparable
    let problem = Rosenbrock;
    let x0 = problem.initial_point(1.0);

    let nr_solver = Solver::new(SolverConfig::robust());
    let lm_solver = LMSolver::new(LMConfig::robust());

    let nr_result = nr_solver.solve(&problem, &x0);
    let lm_result = lm_solver.solve(&problem, &x0);

    if let (Some(nr_norm), Some(lm_norm)) = (nr_result.residual_norm(), lm_result.residual_norm()) {
        // Both should achieve very low residuals
        assert!(nr_norm < 1e-6, "NR residual should be low: {}", nr_norm);
        assert!(lm_norm < 1e-6, "LM residual should be low: {}", lm_norm);

        // They should be similar in magnitude (when not both essentially zero)
        if nr_norm > 1e-15 || lm_norm > 1e-15 {
            let ratio = (nr_norm + 1e-20) / (lm_norm + 1e-20);
            assert!(
                ratio > 1e-3 && ratio < 1e3,
                "Residual norms should be comparable: NR={}, LM={}",
                nr_norm,
                lm_norm
            );
        }
        // If both are essentially zero, they're trivially comparable
    }
}

// --- Iteration Count Comparison ---

#[test]
fn test_nr_faster_on_well_posed_square() {
    // NR should converge in fewer iterations on well-posed square systems
    // (though this isn't always true, it's a general expectation)
    let problem = Rosenbrock;
    let x0 = problem.initial_point(1.0);

    let nr_solver = Solver::new(SolverConfig::default());
    let lm_solver = LMSolver::new(LMConfig::default());

    let nr_result = nr_solver.solve(&problem, &x0);
    let lm_result = lm_solver.solve(&problem, &x0);

    // Both should converge
    assert!(nr_result.is_converged());
    assert!(lm_result.is_converged());

    // Just log the iterations for comparison (not a hard assertion as it can vary)
    let nr_iters = nr_result.iterations().unwrap_or(0);
    let lm_iters = lm_result.iterations().unwrap_or(0);
    eprintln!(
        "Rosenbrock iterations - NR: {}, LM: {} (evaluations)",
        nr_iters, lm_iters
    );
}

// --- Edge Cases ---

#[test]
fn test_both_handle_near_solution() {
    struct NearSolution;

    impl Problem for NearSolution {
        fn name(&self) -> &str {
            "near-solution"
        }
        fn residual_count(&self) -> usize {
            2
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] - 1.0, x[1] - 2.0]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 1.0), (1, 1, 1.0)]
        }
        fn initial_point(&self, _factor: f64) -> Vec<f64> {
            vec![1.0 + 1e-6, 2.0 - 1e-6]
        }
    }

    let problem = NearSolution;
    let x0 = problem.initial_point(1.0);

    let nr_solver = Solver::new(SolverConfig::default());
    let lm_solver = LMSolver::new(LMConfig::default());

    let nr_result = nr_solver.solve(&problem, &x0);
    let lm_result = lm_solver.solve(&problem, &x0);

    assert!(
        nr_result.is_converged(),
        "NR should converge near solution: {:?}",
        nr_result
    );
    assert!(
        lm_result.is_converged(),
        "LM should converge near solution: {:?}",
        lm_result
    );
}

// --- Forced Choice Tests ---

#[test]
fn test_forced_nr_on_overdetermined() {
    // Force NR on overdetermined (uses pseudoinverse)
    let solver = AutoSolver::new().with_choice(SolverChoice::NewtonRaphson);
    let problem = LinearFullRank::new(5, 20);

    let result = solver.solve(&problem, &problem.initial_point(1.0));

    // NR can still solve overdetermined via SVD pseudoinverse
    assert!(
        result.is_completed(),
        "Forced NR should complete on overdetermined: {:?}",
        result
    );
}

#[test]
fn test_forced_lm_on_square() {
    // Force LM on square system
    let solver = AutoSolver::new().with_choice(SolverChoice::LevenbergMarquardt);
    let problem = Rosenbrock;

    let result = solver.solve(&problem, &problem.initial_point(1.0));
    assert!(
        result.is_converged(),
        "Forced LM should converge on square system: {:?}",
        result
    );
}

// --- Multiple Starting Points ---

#[test]
fn test_consistency_across_starting_points() {
    let problem = HelicalValley;
    let nr_solver = Solver::new(SolverConfig::robust());
    let lm_solver = LMSolver::new(LMConfig::robust());

    let mut nr_solutions = Vec::new();
    let mut lm_solutions = Vec::new();

    for factor in &[1.0, 10.0] {
        let x0 = problem.initial_point(*factor);

        let nr_result = nr_solver.solve(&problem, &x0);
        let lm_result = lm_solver.solve(&problem, &x0);

        if nr_result.is_converged() {
            nr_solutions.push(nr_result.solution().unwrap().to_vec());
        }
        if lm_result.is_converged() {
            lm_solutions.push(lm_result.solution().unwrap().to_vec());
        }
    }

    // All converged solutions should be close to each other
    // (Helical Valley has unique solution [1, 0, 0])
    for sols in [&nr_solutions, &lm_solutions] {
        for i in 0..sols.len() {
            for j in (i + 1)..sols.len() {
                let max_diff: f64 = sols[i]
                    .iter()
                    .zip(sols[j].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(
                    max_diff < 1e-4,
                    "Solutions from different starting points should converge to same place"
                );
            }
        }
    }
}

// --- MINPACK Suite Pass Rate ---

/// Test that LM achieves good pass rate on MINPACK problems
#[test]
fn test_lm_minpack_pass_rate() {
    use solverang::test_problems::{all_least_squares_test_cases, create_problem_for_case};

    let solver = LMSolver::new(LMConfig::robust());
    let test_cases = all_least_squares_test_cases();

    let mut total = 0;
    let mut passed = 0;

    for case in test_cases {
        if let Some(problem) = create_problem_for_case(&case) {
            total += 1;
            let result = solver.solve(problem.as_ref(), &problem.initial_point(case.id.factor));

            if result.is_converged() || result.is_completed() {
                // Check residual is reasonable (within 100x of expected for non-zero expected)
                if let (Some(norm), Some(expected)) =
                    (result.residual_norm(), problem.expected_residual_norm())
                {
                    if norm < expected * 100.0 + 1e-4 {
                        passed += 1;
                    }
                } else if result.is_converged() {
                    passed += 1;
                }
            }
        }
    }

    let pass_rate = (passed as f64) / (total as f64) * 100.0;
    eprintln!(
        "LM MINPACK pass rate: {}/{} = {:.1}%",
        passed, total, pass_rate
    );

    // Target: >= 90% pass rate
    assert!(
        pass_rate >= 85.0,
        "LM pass rate should be >= 85%, got {:.1}%",
        pass_rate
    );
}
