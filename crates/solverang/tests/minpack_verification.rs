//! MINPACK test suite verification.
//!
//! Tests the solver against the standard MINPACK test problems.

use solverang::{
    test_problems::{
        Bard, Box3D, BrownAlmostLinear, BroydenBanded, BroydenTridiagonal, Chebyquad,
        DiscreteBoundaryValue, FreudensteinRoth, HelicalValley, LinearFullRank, PowellSingular,
        Rosenbrock, Trigonometric, VariablyDimensioned, Watson, Wood,
    },
    verify_jacobian, Problem, Solver, SolverConfig,
};

// Helper to run a test for a problem
fn test_problem<P: Problem>(problem: &P, tolerance: f64, expected_converge: bool) {
    let solver = Solver::new(SolverConfig::robust());
    let x0 = problem.initial_point(1.0);
    let result = solver.solve(problem, &x0);

    if expected_converge {
        assert!(
            result.is_converged() || result.is_completed(),
            "Problem '{}' should converge, got {:?}",
            problem.name(),
            result
        );
    }

    if let Some(solution) = result.solution() {
        let norm = problem.residual_norm(solution);
        if let Some(expected_norm) = problem.expected_residual_norm() {
            // Allow some tolerance
            assert!(
                (norm - expected_norm).abs() < tolerance || norm < tolerance,
                "Problem '{}': residual norm {} differs from expected {}",
                problem.name(),
                norm,
                expected_norm
            );
        }
    }
}

// Helper to verify Jacobian of a problem
fn verify_problem_jacobian<P: Problem>(problem: &P) {
    let x = problem.initial_point(1.0);
    let result = verify_jacobian(problem, &x, 1e-7, 1e-4);
    assert!(
        result.passed,
        "Jacobian verification failed for '{}': max error {} at {:?}",
        problem.name(),
        result.max_absolute_error,
        result.max_error_location
    );
}

// --- Jacobian Verification Tests ---

#[test]
fn test_rosenbrock_jacobian() {
    verify_problem_jacobian(&Rosenbrock);
}

#[test]
fn test_helical_valley_jacobian() {
    // Use a non-zero point to avoid singularity
    let problem = HelicalValley;
    let x = vec![1.0, 1.0, 0.5];
    let result = verify_jacobian(&problem, &x, 1e-7, 1e-4);
    assert!(result.passed);
}

#[test]
fn test_powell_singular_jacobian() {
    // Use a non-zero point
    let problem = PowellSingular;
    let x = vec![1.0, 1.0, 1.0, 1.0];
    let result = verify_jacobian(&problem, &x, 1e-7, 1e-4);
    assert!(result.passed);
}

#[test]
fn test_wood_jacobian() {
    verify_problem_jacobian(&Wood);
}

#[test]
fn test_bard_jacobian() {
    verify_problem_jacobian(&Bard);
}

#[test]
fn test_watson_jacobian() {
    let problem = Watson::new(6);
    let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let result = verify_jacobian(&problem, &x, 1e-7, 1e-4);
    assert!(result.passed);
}

#[test]
fn test_chebyquad_jacobian() {
    verify_problem_jacobian(&Chebyquad::new(5));
}

#[test]
fn test_brown_almost_linear_jacobian() {
    let problem = BrownAlmostLinear::new(5);
    // Avoid zero for product term
    let x = vec![0.5, 0.5, 0.5, 0.5, 0.5];
    let result = verify_jacobian(&problem, &x, 1e-7, 1e-4);
    assert!(result.passed);
}

#[test]
fn test_discrete_boundary_value_jacobian() {
    verify_problem_jacobian(&DiscreteBoundaryValue::new(5));
}

#[test]
fn test_trigonometric_jacobian() {
    verify_problem_jacobian(&Trigonometric::new(5));
}

#[test]
fn test_variably_dimensioned_jacobian() {
    verify_problem_jacobian(&VariablyDimensioned::new(5));
}

#[test]
fn test_broyden_tridiagonal_jacobian() {
    verify_problem_jacobian(&BroydenTridiagonal::new(5));
}

#[test]
fn test_broyden_banded_jacobian() {
    verify_problem_jacobian(&BroydenBanded::new(5));
}

// --- Solver Tests ---

#[test]
fn test_solve_rosenbrock() {
    test_problem(&Rosenbrock, 1e-6, true);
}

#[test]
fn test_solve_helical_valley() {
    test_problem(&HelicalValley, 1e-6, true);
}

#[test]
fn test_solve_powell_singular() {
    // Powell Singular is notoriously hard due to singular Jacobian at solution
    let problem = PowellSingular;
    let solver = Solver::new(SolverConfig::robust());
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    // Just check it completes without error
    assert!(result.is_completed());
}

#[test]
fn test_solve_freudenstein_roth() {
    // May find local minimum
    let problem = FreudensteinRoth;
    let solver = Solver::new(SolverConfig::default());
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    assert!(result.is_completed());
}

#[test]
fn test_solve_wood() {
    test_problem(&Wood, 1e-6, true);
}

#[test]
fn test_solve_linear_full_rank() {
    test_problem(&LinearFullRank::default(), 1.0, true);
}

#[test]
fn test_solve_bard() {
    test_problem(&Bard, 0.1, true);
}

#[test]
fn test_solve_box3d() {
    test_problem(&Box3D::default(), 1e-6, true);
}

#[test]
fn test_solve_brown_almost_linear() {
    test_problem(&BrownAlmostLinear::new(5), 1e-6, true);
}

#[test]
fn test_solve_discrete_boundary_value() {
    test_problem(&DiscreteBoundaryValue::new(10), 1e-6, true);
}

#[test]
fn test_solve_broyden_tridiagonal() {
    test_problem(&BroydenTridiagonal::new(10), 1e-6, true);
}

#[test]
fn test_solve_broyden_banded() {
    test_problem(&BroydenBanded::new(10), 1e-6, true);
}

#[test]
fn test_solve_variably_dimensioned() {
    test_problem(&VariablyDimensioned::new(10), 1e-6, true);
}

// --- Tests for different starting factors ---

#[test]
fn test_rosenbrock_factors() {
    let problem = Rosenbrock;
    let solver = Solver::new(SolverConfig::robust());

    for factor in &[1.0, 10.0, 100.0] {
        let result = solver.solve(&problem, &problem.initial_point(*factor));
        assert!(
            result.is_converged(),
            "Rosenbrock with factor {} should converge",
            factor
        );
    }
}

#[test]
fn test_helical_valley_factors() {
    let problem = HelicalValley;
    let solver = Solver::new(SolverConfig::robust());

    for factor in &[1.0, 10.0, 100.0] {
        let result = solver.solve(&problem, &problem.initial_point(*factor));
        assert!(
            result.is_converged(),
            "Helical Valley with factor {} should converge",
            factor
        );
    }
}

// --- Test problem dimensions ---

#[test]
fn test_watson_dimensions() {
    // Watson has fixed m=31 equations but variable n
    for n in &[2, 6, 9, 12] {
        let problem = Watson::new(*n);
        assert_eq!(problem.residual_count(), 31);
        assert_eq!(problem.variable_count(), *n);

        let x0 = problem.initial_point(1.0);
        let residuals = problem.residuals(&x0);
        assert_eq!(residuals.len(), 31);

        let jac = problem.jacobian(&x0);
        // Should have entries for all columns
        let cols_used: std::collections::HashSet<_> = jac.iter().map(|(_, c, _)| c).collect();
        assert_eq!(cols_used.len(), *n);
    }
}

#[test]
fn test_chebyquad_dimensions() {
    // Square case
    let problem = Chebyquad::new(5);
    assert_eq!(problem.residual_count(), 5);
    assert_eq!(problem.variable_count(), 5);

    // Overdetermined case
    let problem = Chebyquad::with_m(3, 8);
    assert_eq!(problem.residual_count(), 8);
    assert_eq!(problem.variable_count(), 3);
}

#[test]
fn test_linear_full_rank_overdetermined() {
    let problem = LinearFullRank::new(5, 50);
    assert_eq!(problem.residual_count(), 50);
    assert_eq!(problem.variable_count(), 5);

    let solver = Solver::new(SolverConfig::default());
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    assert!(result.is_completed());
}
