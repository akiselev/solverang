//! Levenberg-Marquardt solver tests.
//!
//! Tests specific to the LM solver, focusing on over-constrained systems
//! and robustness to poor starting points.

use solverang::{
    test_problems::{
        Bard, Box3D, BrownAlmostLinear, Chebyquad, HelicalValley, LinearFullRank, PowellSingular,
        Rosenbrock, Watson, Wood,
    },
    LMConfig, LMSolver, Problem,
};

// Helper to run LM solver on a problem
fn test_lm_problem<P: Problem>(problem: &P, tolerance: f64, expected_converge: bool) {
    let solver = LMSolver::new(LMConfig::robust());
    let x0 = problem.initial_point(1.0);
    let result = solver.solve(problem, &x0);

    if expected_converge {
        assert!(
            result.is_converged() || result.is_completed(),
            "Problem '{}' should converge with LM, got {:?}",
            problem.name(),
            result
        );
    }

    if let Some(solution) = result.solution() {
        let norm = problem.residual_norm(solution);
        if let Some(expected_norm) = problem.expected_residual_norm() {
            assert!(
                (norm - expected_norm).abs() < tolerance || norm < tolerance,
                "Problem '{}': LM residual norm {} differs from expected {}",
                problem.name(),
                norm,
                expected_norm
            );
        }
    }
}

// --- Basic LM Tests ---

#[test]
fn test_lm_rosenbrock() {
    test_lm_problem(&Rosenbrock, 1e-6, true);
}

#[test]
fn test_lm_helical_valley() {
    test_lm_problem(&HelicalValley, 1e-6, true);
}

#[test]
fn test_lm_powell_singular() {
    // Powell Singular is hard but LM should handle it
    let problem = PowellSingular;
    let solver = LMSolver::new(LMConfig::robust());
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    assert!(
        result.is_completed(),
        "LM should complete on Powell Singular"
    );
}

#[test]
fn test_lm_wood() {
    test_lm_problem(&Wood, 1e-6, true);
}

#[test]
fn test_lm_bard() {
    test_lm_problem(&Bard, 0.1, true);
}

#[test]
fn test_lm_box3d() {
    test_lm_problem(&Box3D::default(), 1e-6, true);
}

#[test]
fn test_lm_brown_almost_linear() {
    test_lm_problem(&BrownAlmostLinear::new(5), 1e-6, true);
}

// --- Over-constrained System Tests ---

#[test]
fn test_lm_overdetermined_linear_full_rank() {
    // 50 equations, 5 variables - highly over-constrained
    let problem = LinearFullRank::new(5, 50);
    let solver = LMSolver::new(LMConfig::default());
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    assert!(
        result.is_converged() || result.is_completed(),
        "LM should handle highly over-determined systems: {:?}",
        result
    );
}

#[test]
fn test_lm_watson_overdetermined() {
    // Watson has 31 equations with variable n
    // With n=6, this is 31x6 - overdetermined
    let problem = Watson::new(6);
    assert_eq!(problem.residual_count(), 31);
    assert_eq!(problem.variable_count(), 6);

    let solver = LMSolver::new(LMConfig::robust());
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    assert!(
        result.is_completed(),
        "LM should complete on overdetermined Watson: {:?}",
        result
    );
}

#[test]
fn test_lm_chebyquad_overdetermined() {
    // Chebyquad with 3 variables, 8 equations
    let problem = Chebyquad::with_m(3, 8);
    assert_eq!(problem.residual_count(), 8);
    assert_eq!(problem.variable_count(), 3);

    let solver = LMSolver::new(LMConfig::robust());
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    assert!(
        result.is_completed(),
        "LM should handle overdetermined Chebyquad: {:?}",
        result
    );
}

// --- Tests for Different Starting Factors ---

#[test]
fn test_lm_rosenbrock_factors() {
    let problem = Rosenbrock;
    let solver = LMSolver::new(LMConfig::robust());

    for factor in &[1.0, 10.0, 100.0] {
        let result = solver.solve(&problem, &problem.initial_point(*factor));
        assert!(
            result.is_converged() || result.is_completed(),
            "LM Rosenbrock with factor {} should complete, got {:?}",
            factor,
            result
        );
    }
}

#[test]
fn test_lm_helical_valley_factors() {
    let problem = HelicalValley;
    let solver = LMSolver::new(LMConfig::robust());

    for factor in &[1.0, 10.0, 100.0] {
        let result = solver.solve(&problem, &problem.initial_point(*factor));
        assert!(
            result.is_converged() || result.is_completed(),
            "LM Helical Valley with factor {} should complete, got {:?}",
            factor,
            result
        );
    }
}

// --- Configuration Tests ---

#[test]
fn test_lm_config_fast() {
    let problem = Rosenbrock;
    let solver = LMSolver::new(LMConfig::fast());
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    assert!(result.is_converged(), "Fast config should solve Rosenbrock");
}

#[test]
fn test_lm_config_precise() {
    let problem = Rosenbrock;
    let solver = LMSolver::new(LMConfig::precise());
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    assert!(
        result.is_converged(),
        "Precise config should solve Rosenbrock"
    );

    // Check that precision is indeed higher
    if let Some(norm) = result.residual_norm() {
        assert!(
            norm < 1e-10,
            "Precise config should achieve lower residual: {}",
            norm
        );
    }
}

#[test]
fn test_lm_config_builder() {
    let config = LMConfig::new()
        .with_ftol(1e-10)
        .with_xtol(1e-10)
        .with_gtol(1e-10)
        .with_patience(300);

    let problem = Rosenbrock;
    let solver = LMSolver::new(config);
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    assert!(
        result.is_converged(),
        "Custom config should solve Rosenbrock"
    );
}

// --- Edge Case Tests ---

#[test]
fn test_lm_already_at_solution() {
    struct AlreadySolved;

    impl Problem for AlreadySolved {
        fn name(&self) -> &str {
            "already-solved"
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
            vec![1.0, 2.0] // Already at solution
        }
    }

    let problem = AlreadySolved;
    let solver = LMSolver::new(LMConfig::default());
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    assert!(
        result.is_converged(),
        "Should converge immediately at solution: {:?}",
        result
    );

    if let Some(norm) = result.residual_norm() {
        assert!(
            norm < 1e-14,
            "Residual should be essentially zero: {}",
            norm
        );
    }
}

#[test]
fn test_lm_near_zero_residual() {
    struct NearZero;

    impl Problem for NearZero {
        fn name(&self) -> &str {
            "near-zero"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            1
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0]]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 1.0)]
        }
        fn initial_point(&self, _factor: f64) -> Vec<f64> {
            vec![1e-10]
        }
    }

    let problem = NearZero;
    let solver = LMSolver::new(LMConfig::default());
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    assert!(
        result.is_converged(),
        "Should converge to zero: {:?}",
        result
    );
}

#[test]
fn test_lm_large_dimension() {
    // Test with a larger system
    let problem = BrownAlmostLinear::new(20);
    let solver = LMSolver::new(LMConfig::robust());
    let result = solver.solve(&problem, &problem.initial_point(1.0));

    assert!(
        result.is_completed(),
        "LM should handle 20-variable system: {:?}",
        result
    );
}

// --- Stress Tests ---

#[test]
fn test_lm_poor_starting_point() {
    // Test LM's ability to converge from a poor starting point
    let problem = Rosenbrock;
    let solver = LMSolver::new(LMConfig::robust());

    // Very far from solution
    let result = solver.solve(&problem, &[100.0, -100.0]);

    // LM may not converge from very far, but should at least complete
    assert!(
        result.is_completed(),
        "LM should complete even from poor starting point: {:?}",
        result
    );
}

// --- Under-constrained System Test ---

#[test]
fn test_lm_underdetermined() {
    struct Underdetermined;

    impl Problem for Underdetermined {
        fn name(&self) -> &str {
            "underdetermined"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            3
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] + x[1] + x[2] - 1.0]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0)]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![factor, factor, factor]
        }
    }

    let problem = Underdetermined;
    let solver = LMSolver::new(LMConfig::default());
    let result = solver.solve(&problem, &[0.0, 0.0, 0.0]);

    assert!(
        result.is_completed(),
        "LM should handle underdetermined system: {:?}",
        result
    );

    if let Some(solution) = result.solution() {
        // Any solution satisfying x + y + z = 1 is valid
        let sum: f64 = solution.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Solution should satisfy constraint: sum = {}",
            sum
        );
    }
}
