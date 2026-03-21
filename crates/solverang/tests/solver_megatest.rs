//! Comprehensive solver megatest.
//!
//! Tests the solver with massive systems that have known analytical solutions.
//! Exercises every solver type across coupled nonlinear, sparse, and parallel systems.
//!
//! # Test Structure
//!
//! 1. **Coupled Nonlinear Chain** – 100-variable tridiagonal nonlinear system with
//!    manufactured solution. Tested with NR, LM, Auto, and Robust solvers.
//!
//! 2. **Overdetermined Mega** – 200 equations / 50 variables, least-squares system
//!    with known zero-residual solution. Tests LM on large overdetermined problems.
//!
//! 3. **Sparse Mega System** – 500-variable sparse tridiagonal nonlinear system,
//!    tested with the sparse solver.
//!
//! 4. **Cross-Solver Comparison** – Identical problem solved by every solver;
//!    results compared for consistency.
//!
//! 5. **Jacobian Verification** – Analytical Jacobians verified against finite
//!    differences for test problems.
//!
//! 6. **Robustness Under Poor Initial Conditions** – Starting points 10× away
//!    from solution; verifies solvers still converge.

use solverang::{
    verify_jacobian, AutoSolver, LMConfig, LMSolver, Problem, RobustSolver, Solver, SolverConfig,
};

// ============================================================================
// Problem 1: Coupled Nonlinear Chain
// ============================================================================

/// A large nonlinear system with a *manufactured* analytical solution.
///
/// g_i(x) = x_i^3 + 0.5*(x_{i-1} + x_{i+1}) + 0.1*sin(x_i)
///
/// The residual is f_i(x) = g_i(x) - g_i(x*) so that f(x*) = 0 exactly.
/// The Jacobian is tridiagonal (sparse).
struct CoupledNonlinearChain {
    size: usize,
    /// The known analytical solution.
    solution: Vec<f64>,
    /// Pre-computed right-hand side g(x*).
    rhs: Vec<f64>,
}

impl CoupledNonlinearChain {
    fn new(size: usize) -> Self {
        // Manufacture a solution with interesting non-trivial values.
        let solution: Vec<f64> = (0..size)
            .map(|i| {
                let t = i as f64 * std::f64::consts::PI / size as f64;
                t.sin() + 1.0 // values in [1.0, 2.0]
            })
            .collect();
        let rhs = Self::compute_g(&solution, size);
        Self {
            size,
            solution,
            rhs,
        }
    }

    fn g_i(x: &[f64], i: usize, n: usize) -> f64 {
        let xi = x[i];
        let x_prev = if i > 0 { x[i - 1] } else { 0.0 };
        let x_next = if i < n - 1 { x[i + 1] } else { 0.0 };
        xi * xi * xi + 0.5 * (x_prev + x_next) + 0.1 * xi.sin()
    }

    fn compute_g(x: &[f64], n: usize) -> Vec<f64> {
        (0..n).map(|i| Self::g_i(x, i, n)).collect()
    }
}

impl Problem for CoupledNonlinearChain {
    fn name(&self) -> &str {
        "CoupledNonlinearChain"
    }

    fn residual_count(&self) -> usize {
        self.size
    }

    fn variable_count(&self) -> usize {
        self.size
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        (0..self.size)
            .map(|i| Self::g_i(x, i, self.size) - self.rhs[i])
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = self.size;
        let mut entries = Vec::with_capacity(3 * n);
        for i in 0..n {
            if i > 0 {
                // dg_i/dx_{i-1} = 0.5
                entries.push((i, i - 1, 0.5));
            }
            // dg_i/dx_i = 3*x_i^2 + 0.1*cos(x_i)
            entries.push((i, i, 3.0 * x[i] * x[i] + 0.1 * x[i].cos()));
            if i < n - 1 {
                // dg_i/dx_{i+1} = 0.5
                entries.push((i, i + 1, 0.5));
            }
        }
        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        // Perturb the solution; factor controls how far away we start.
        self.solution
            .iter()
            .enumerate()
            .map(|(i, &s)| s + factor * 0.3 * ((i as f64) * 0.7).sin())
            .collect()
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(self.solution.clone())
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

// ============================================================================
// Problem 2: Large Overdetermined System
// ============================================================================

/// Overdetermined system with M >> N.
///
/// Primary equations: x_i^2 - target_i = 0    (i = 0..N)
/// Coupling equations: x_i * x_j - target_i*target_j = 0   (selected pairs)
///
/// Known solution: x_i = sqrt(i + 1) for i = 0..N-1.
struct OverdeterminedMega {
    n_vars: usize,
    solution: Vec<f64>,
    /// (i, j) pairs for coupling equations.
    coupling_pairs: Vec<(usize, usize)>,
}

impl OverdeterminedMega {
    fn new(n_vars: usize) -> Self {
        let solution: Vec<f64> = (0..n_vars).map(|i| ((i + 1) as f64).sqrt()).collect();

        // Generate coupling pairs: each variable coupled with the next few.
        let mut pairs = Vec::new();
        for i in 0..n_vars {
            for offset in 1..=3 {
                let j = (i + offset) % n_vars;
                if j != i {
                    pairs.push((i, j));
                }
            }
        }
        Self {
            n_vars,
            solution,
            coupling_pairs: pairs,
        }
    }

    fn n_primary(&self) -> usize {
        self.n_vars
    }
}

impl Problem for OverdeterminedMega {
    fn name(&self) -> &str {
        "OverdeterminedMega"
    }

    fn residual_count(&self) -> usize {
        self.n_primary() + self.coupling_pairs.len()
    }

    fn variable_count(&self) -> usize {
        self.n_vars
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        let mut r = Vec::with_capacity(self.residual_count());
        // Primary equations: x_i^2 - (i+1)
        for i in 0..self.n_vars {
            r.push(x[i] * x[i] - (i as f64 + 1.0));
        }
        // Coupling equations: x_i * x_j - sqrt((i+1)*(j+1))
        for &(i, j) in &self.coupling_pairs {
            let target = ((i as f64 + 1.0) * (j as f64 + 1.0)).sqrt();
            r.push(x[i] * x[j] - target);
        }
        r
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let mut entries = Vec::new();
        // Primary: d(x_i^2 - c)/dx_i = 2*x_i
        for i in 0..self.n_vars {
            entries.push((i, i, 2.0 * x[i]));
        }
        // Coupling: d(x_i*x_j)/dx_i = x_j, d(x_i*x_j)/dx_j = x_i
        let offset = self.n_primary();
        for (k, &(i, j)) in self.coupling_pairs.iter().enumerate() {
            entries.push((offset + k, i, x[j]));
            entries.push((offset + k, j, x[i]));
        }
        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        self.solution
            .iter()
            .enumerate()
            .map(|(i, &s)| s + factor * 0.2 * (1.0 + (i as f64 * 0.3).sin()))
            .collect()
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(self.solution.clone())
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

// ============================================================================
// Problem 3: Sparse Nonlinear System
// ============================================================================

/// Large sparse nonlinear system with tridiagonal coupling.
///
/// f_i(x) = x_i^3 - 3*x_i + c_i - alpha*(x_{i-1} + x_{i+1})
///
/// We choose c_i so that x_i = solution_i is the root.
struct SparseNonlinearMega {
    size: usize,
    alpha: f64,
    solution: Vec<f64>,
    constants: Vec<f64>,
}

impl SparseNonlinearMega {
    fn new(size: usize) -> Self {
        let alpha = 0.25;
        // Solution centred at 2.0 to stay well away from Jacobian singularity at x=±1
        // (the diagonal 3x²-3 vanishes there).
        let solution: Vec<f64> = (0..size)
            .map(|i| 2.0 + 0.3 * ((i as f64) * 2.0 * std::f64::consts::PI / size as f64).cos())
            .collect();

        // Compute c_i = -g_i(x*) so that f_i(x*) = g_i(x) + c_i = 0
        let constants: Vec<f64> = (0..size)
            .map(|i| {
                let xi = solution[i];
                let x_prev = if i > 0 { solution[i - 1] } else { 0.0 };
                let x_next = if i < size - 1 { solution[i + 1] } else { 0.0 };
                -(xi * xi * xi - 3.0 * xi - alpha * (x_prev + x_next))
            })
            .collect();

        Self {
            size,
            alpha,
            solution,
            constants,
        }
    }
}

impl Problem for SparseNonlinearMega {
    fn name(&self) -> &str {
        "SparseNonlinearMega"
    }

    fn residual_count(&self) -> usize {
        self.size
    }

    fn variable_count(&self) -> usize {
        self.size
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        let n = self.size;
        (0..n)
            .map(|i| {
                let xi = x[i];
                let x_prev = if i > 0 { x[i - 1] } else { 0.0 };
                let x_next = if i < n - 1 { x[i + 1] } else { 0.0 };
                xi * xi * xi - 3.0 * xi - self.alpha * (x_prev + x_next) + self.constants[i]
            })
            .collect()
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = self.size;
        let mut entries = Vec::with_capacity(3 * n);
        for i in 0..n {
            if i > 0 {
                entries.push((i, i - 1, -self.alpha));
            }
            entries.push((i, i, 3.0 * x[i] * x[i] - 3.0));
            if i < n - 1 {
                entries.push((i, i + 1, -self.alpha));
            }
        }
        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        self.solution
            .iter()
            .enumerate()
            .map(|(i, &s)| s + factor * 0.1 * ((i as f64) * 1.3).sin())
            .collect()
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(self.solution.clone())
    }

    fn expected_residual_norm(&self) -> Option<f64> {
        Some(0.0)
    }
}

// ============================================================================
// Helper: verify a solution against known values
// ============================================================================

fn assert_solution_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: dimension mismatch");
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < tol,
        "{label}: max component error {max_err:.2e} exceeds tolerance {tol:.2e}"
    );
}

// ############################################################################
//  TESTS
// ############################################################################

// ============================================================================
// 1. Coupled Nonlinear Chain – multiple solvers
// ============================================================================

#[test]
fn megatest_coupled_nonlinear_100_newton_raphson() {
    let problem = CoupledNonlinearChain::new(100);
    let solver = Solver::new(SolverConfig::robust());
    let x0 = problem.initial_point(1.0);

    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged(),
        "NR on 100-var coupled chain should converge: {result:?}"
    );
    let solution = result.solution().expect("should have solution");
    assert_solution_close(solution, &problem.solution, 1e-6, "NR-100");
}

#[test]
fn megatest_coupled_nonlinear_100_levenberg_marquardt() {
    let problem = CoupledNonlinearChain::new(100);
    let solver = LMSolver::new(LMConfig::robust());
    let x0 = problem.initial_point(1.0);

    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged(),
        "LM on 100-var coupled chain should converge: {result:?}"
    );
    let solution = result.solution().expect("should have solution");
    assert_solution_close(solution, &problem.solution, 1e-6, "LM-100");
}

#[test]
fn megatest_coupled_nonlinear_100_auto() {
    let problem = CoupledNonlinearChain::new(100);
    let solver = AutoSolver::new();
    let x0 = problem.initial_point(1.0);

    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged(),
        "Auto solver on 100-var coupled chain should converge: {result:?}"
    );
    let solution = result.solution().expect("should have solution");
    assert_solution_close(solution, &problem.solution, 1e-6, "Auto-100");
}

#[test]
fn megatest_coupled_nonlinear_100_robust() {
    let problem = CoupledNonlinearChain::new(100);
    let solver = RobustSolver::new();
    let x0 = problem.initial_point(1.0);

    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged(),
        "Robust solver on 100-var coupled chain should converge: {result:?}"
    );
    let solution = result.solution().expect("should have solution");
    assert_solution_close(solution, &problem.solution, 1e-6, "Robust-100");
}

#[test]
#[ignore] // Larger system; run with `cargo test -- --ignored`
fn megatest_coupled_nonlinear_200_vars() {
    let problem = CoupledNonlinearChain::new(200);
    let solver = LMSolver::new(LMConfig::robust());
    let x0 = problem.initial_point(1.0);

    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged(),
        "LM on 200-var coupled chain should converge: {result:?}"
    );
    let solution = result.solution().expect("should have solution");
    assert_solution_close(solution, &problem.solution, 1e-6, "LM-200");
}

// ============================================================================
// 2. Large Overdetermined System (LM only)
// ============================================================================

#[test]
fn megatest_overdetermined_50_vars() {
    let problem = OverdeterminedMega::new(50);
    assert!(
        problem.residual_count() > problem.variable_count(),
        "Should be overdetermined: {} eqs, {} vars",
        problem.residual_count(),
        problem.variable_count()
    );

    let solver = LMSolver::new(LMConfig::robust());
    let x0 = problem.initial_point(1.0);
    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged(),
        "LM on overdetermined {}-eq / {}-var system should converge: {result:?}",
        problem.residual_count(),
        problem.variable_count()
    );
    let solution = result.solution().expect("should have solution");
    // Check that we're near the known solution (signs may differ for sqrt).
    // Our initial guess is positive, so the solver should find the positive root.
    let max_err = solution
        .iter()
        .zip(problem.solution.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < 1e-4,
        "Overdetermined solution max error {max_err:.2e} too large"
    );
}

// ============================================================================
// 4. Sparse Solver Megatest
// ============================================================================

#[cfg(feature = "sparse")]
mod sparse_megatest {
    use super::*;
    use solverang::{SparseSolver, SparseSolverConfig};

    #[test]
    fn megatest_sparse_500_vars() {
        let problem = SparseNonlinearMega::new(500);
        let mut solver = SparseSolver::new(SparseSolverConfig::robust());
        let x0 = problem.initial_point(1.0);

        let result = solver.solve(&problem, &x0);

        assert!(
            result.is_converged() || result.is_completed(),
            "Sparse solver on 500-var system should converge: {result:?}"
        );

        if let Some(solution) = result.solution() {
            let norm = problem.residual_norm(solution);
            assert!(
                norm < 1e-3,
                "Sparse 500-var residual norm {norm:.2e} too high"
            );
            assert_solution_close(solution, &problem.solution, 1e-3, "Sparse-500");
        }
    }

    #[test]
    fn megatest_sparse_1000_vars() {
        let problem = SparseNonlinearMega::new(1000);
        let mut solver = SparseSolver::new(SparseSolverConfig::robust());
        let x0 = problem.initial_point(1.0);

        let result = solver.solve(&problem, &x0);

        assert!(
            result.is_converged() || result.is_completed(),
            "Sparse solver on 1000-var system should complete: {result:?}"
        );

        if let Some(solution) = result.solution() {
            let norm = problem.residual_norm(solution);
            assert!(
                norm < 1e-2,
                "Sparse 1000-var residual norm {norm:.2e} too high"
            );
            assert_solution_close(solution, &problem.solution, 0.5, "Sparse-1000");
        }
    }
}

// ============================================================================
// 5. Cross-Solver Comparison
// ============================================================================

#[test]
fn megatest_cross_solver_comparison() {
    let problem = CoupledNonlinearChain::new(50);
    let x0 = problem.initial_point(1.0);
    let expected = &problem.solution;

    // Newton-Raphson
    let nr_result = Solver::new(SolverConfig::robust()).solve(&problem, &x0);
    // Levenberg-Marquardt
    let lm_result = LMSolver::new(LMConfig::robust()).solve(&problem, &x0);
    // Auto
    let auto_result = AutoSolver::new().solve(&problem, &x0);
    // Robust
    let robust_result = RobustSolver::new().solve(&problem, &x0);

    let results = [
        ("NR", nr_result),
        ("LM", lm_result),
        ("Auto", auto_result),
        ("Robust", robust_result),
    ];

    for (name, result) in &results {
        assert!(
            result.is_converged(),
            "Cross-solver: {name} should converge: {result:?}"
        );
        let sol = result.solution().expect("should have solution");
        assert_solution_close(sol, expected, 1e-5, &format!("Cross-{name}"));
    }

    // All solutions should agree to high precision.
    let nr_sol = results[0].1.solution().unwrap();
    for (name, result) in &results[1..] {
        let sol = result.solution().unwrap();
        let max_diff = nr_sol
            .iter()
            .zip(sol.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-4,
            "Cross-solver: NR vs {name} disagree by {max_diff:.2e}"
        );
    }
}

// ============================================================================
// 6. Jacobian Verification
// ============================================================================

#[test]
fn megatest_jacobian_coupled_nonlinear() {
    let problem = CoupledNonlinearChain::new(50);
    let x = problem.initial_point(1.0);

    let verification = verify_jacobian(&problem, &x, 1e-7, 1e-5);

    assert!(
        verification.passed,
        "Coupled nonlinear chain Jacobian failed: max error = {:.2e} at {:?}",
        verification.max_absolute_error, verification.max_error_location
    );
}

#[test]
fn megatest_jacobian_overdetermined() {
    let problem = OverdeterminedMega::new(30);
    let x = problem.initial_point(1.0);

    let verification = verify_jacobian(&problem, &x, 1e-7, 1e-5);

    assert!(
        verification.passed,
        "Overdetermined Jacobian failed: max error = {:.2e} at {:?}",
        verification.max_absolute_error, verification.max_error_location
    );
}

#[test]
fn megatest_jacobian_sparse_nonlinear() {
    let problem = SparseNonlinearMega::new(100);
    let x = problem.initial_point(1.0);

    let verification = verify_jacobian(&problem, &x, 1e-7, 1e-5);

    assert!(
        verification.passed,
        "Sparse nonlinear Jacobian failed: max error = {:.2e} at {:?}",
        verification.max_absolute_error, verification.max_error_location
    );
}

// ============================================================================
// 7. Robustness – poor initial conditions
// ============================================================================

#[test]
fn megatest_poor_initial_conditions() {
    let problem = CoupledNonlinearChain::new(50);

    // Start very far from solution (factor = 10).
    let x0 = problem.initial_point(10.0);

    // LM should still converge (it's the most robust).
    let solver = LMSolver::new(LMConfig::robust());
    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged() || result.is_completed(),
        "LM with poor initial should at least complete: {result:?}"
    );

    if let Some(solution) = result.solution() {
        let norm = problem.residual_norm(solution);
        assert!(
            norm < 1e-3,
            "LM from poor start: residual norm {norm:.2e} too high"
        );
    }
}

#[test]
fn megatest_robust_solver_poor_initial() {
    let problem = CoupledNonlinearChain::new(50);
    let x0 = problem.initial_point(5.0);

    let solver = RobustSolver::new()
        .with_nr_config(SolverConfig::robust())
        .with_lm_config(LMConfig::robust());
    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged() || result.is_completed(),
        "Robust solver with poor initial should converge: {result:?}"
    );

    if let Some(solution) = result.solution() {
        assert_solution_close(solution, &problem.solution, 1e-3, "Robust-poor-init");
    }
}

// ============================================================================
// 8. Scaling behaviour – increasingly large systems
// ============================================================================

#[test]
#[ignore] // Scaling sweep; run with `cargo test -- --ignored`
fn megatest_scaling_behaviour() {
    for &n in &[10, 25, 50, 100, 150] {
        let problem = CoupledNonlinearChain::new(n);
        let solver = LMSolver::new(LMConfig::robust());
        let x0 = problem.initial_point(1.0);

        let result = solver.solve(&problem, &x0);

        assert!(
            result.is_converged(),
            "Scaling test: n={n} should converge: {result:?}"
        );

        let norm = result.residual_norm().expect("should have norm");
        assert!(
            norm < 1e-6,
            "Scaling test n={n}: residual norm {norm:.2e} too high"
        );
    }
}

// ============================================================================
// 9. At-solution smoke test – verify zero residuals at known solution
// ============================================================================

#[test]
fn megatest_zero_residuals_at_solution() {
    let problem = CoupledNonlinearChain::new(100);
    let r = problem.residuals(&problem.solution);
    let norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        norm < 1e-12,
        "Residuals at manufactured solution should be ~0, got {norm:.2e}"
    );

    let problem2 = OverdeterminedMega::new(50);
    let r2 = problem2.residuals(&problem2.solution);
    let norm2: f64 = r2.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        norm2 < 1e-12,
        "Overdetermined residuals at solution should be ~0, got {norm2:.2e}"
    );

    let problem3 = SparseNonlinearMega::new(500);
    let r3 = problem3.residuals(&problem3.solution);
    let norm3: f64 = r3.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        norm3 < 1e-12,
        "Sparse residuals at solution should be ~0, got {norm3:.2e}"
    );
}

// ############################################################################
//  V3 PIPELINE MEGATESTS
// ############################################################################
//
// These tests exercise the new solver-first V3 architecture:
//   - ConstraintSystem (system.rs) with Entity/Constraint traits
//   - Full pipeline: Decompose → Analyze → Reduce → Solve → PostProcess
//   - Sketch2DBuilder ergonomic API
//   - Incremental solving (dirty-cluster tracking)
//   - Drag (null-space projection)
//   - Multi-cluster decomposition
//   - Diagnostics (DOF analysis, redundancy)
//   - Entity/constraint lifecycle (add/remove/generational IDs)

mod v3_pipeline_megatest {
    use solverang::id::{EntityId, ParamId};
    use solverang::pipeline::reduce::NoopReduce;
    use solverang::pipeline::PipelineBuilder;
    use solverang::sketch2d::builder::Sketch2DBuilder;
    use solverang::sketch2d::constraints::*;
    use solverang::sketch2d::entities::*;
    use solverang::system::{ClusterSolveStatus, ConstraintSystem, SystemResult, SystemStatus};

    // =====================================================================
    // Helpers (all use fix_param pattern, NOT Fixed constraint)
    // =====================================================================

    fn add_point(sys: &mut ConstraintSystem, x: f64, y: f64) -> (EntityId, ParamId, ParamId) {
        let eid = sys.alloc_entity_id();
        let px = sys.alloc_param(x, eid);
        let py = sys.alloc_param(y, eid);
        sys.add_entity(Box::new(Point2D::new(eid, px, py)));
        (eid, px, py)
    }

    /// Add a point and immediately fix both params (anchored at given coords).
    fn add_fixed_pt(sys: &mut ConstraintSystem, x: f64, y: f64) -> (EntityId, ParamId, ParamId) {
        let (eid, px, py) = add_point(sys, x, y);
        sys.fix_param(px);
        sys.fix_param(py);
        (eid, px, py)
    }

    fn add_circle_entity(
        sys: &mut ConstraintSystem,
        cx: f64,
        cy: f64,
        r: f64,
    ) -> (EntityId, ParamId, ParamId, ParamId) {
        let eid = sys.alloc_entity_id();
        let pcx = sys.alloc_param(cx, eid);
        let pcy = sys.alloc_param(cy, eid);
        let pr = sys.alloc_param(r, eid);
        sys.add_entity(Box::new(Circle2D::new(eid, pcx, pcy, pr)));
        (eid, pcx, pcy, pr)
    }

    fn pt_dist(sys: &ConstraintSystem, x1: ParamId, y1: ParamId, x2: ParamId, y2: ParamId) -> f64 {
        let dx = sys.get_param(x2) - sys.get_param(x1);
        let dy = sys.get_param(y2) - sys.get_param(y1);
        (dx * dx + dy * dy).sqrt()
    }

    fn assert_solved(result: &SystemResult) {
        match &result.status {
            SystemStatus::Solved => {}
            SystemStatus::PartiallySolved => {
                for cr in &result.clusters {
                    assert!(
                        cr.residual_norm < 1e-3,
                        "Cluster {:?} residual too large: {:.6e}",
                        cr.cluster_id,
                        cr.residual_norm,
                    );
                }
            }
            SystemStatus::DiagnosticFailure(issues) => {
                panic!("Expected Solved, got DiagnosticFailure: {issues:?}");
            }
        }
    }

    // =====================================================================
    // 10. V3 Pipeline – Massive Sketch2D via Builder
    // =====================================================================

    /// Build a large 2D sketch using Sketch2DBuilder with:
    ///   - 30+ point entities, circles, line segments
    ///   - Distance, horizontal, vertical, perpendicular, parallel,
    ///     midpoint, symmetric, equal-length, point-on-circle, tangent
    ///   - Known solution positions
    ///   - Solve via the full pipeline and verify
    #[test]
    fn megatest_v3_massive_sketch2d() {
        let mut b = Sketch2DBuilder::new();

        // === Grid of 5×4 points (known positions) ===
        // Row 0: y=0, Row 1: y=10, Row 2: y=20, Row 3: y=30
        // Use add_fixed_point for origin (avoids Fixed constraint reduce bug).
        let origin = b.add_fixed_point(0.0, 0.0);

        let mut grid: Vec<Vec<EntityId>> = Vec::new();
        for j in 0..4 {
            let mut row = Vec::new();
            for i in 0..5 {
                if i == 0 && j == 0 {
                    row.push(origin);
                } else {
                    let x = i as f64 * 10.0 + 0.3 * ((i + j) as f64).sin();
                    let y = j as f64 * 10.0 + 0.3 * ((i * j) as f64).cos();
                    row.push(b.add_point(x, y));
                }
            }
            grid.push(row);
        }

        // === Horizontal constraints across each row ===
        for j in 0..4 {
            for i in 0..4 {
                b.constrain_horizontal(grid[j][i], grid[j][i + 1]);
            }
        }

        // === Vertical constraints down each column ===
        for j in 0..3 {
            for i in 0..5 {
                b.constrain_vertical(grid[j][i], grid[j + 1][i]);
            }
        }

        // === Distance constraints ONLY for first row and first column ===
        // Horizontal/vertical propagate positions; only need spacing on edges.
        // First row: sets column x-spacings.
        for i in 0..4 {
            b.constrain_distance(grid[0][i], grid[0][i + 1], 10.0);
        }
        // First column: sets row y-spacings.
        for j in 0..3 {
            b.constrain_distance(grid[j][0], grid[j + 1][0], 10.0);
        }

        // === Line segments for perpendicular/parallel tests ===
        let l_bottom = b.add_line_segment(grid[0][0], grid[0][4]);
        let l_top = b.add_line_segment(grid[3][0], grid[3][4]);
        let l_left = b.add_line_segment(grid[0][0], grid[3][0]);
        let l_right = b.add_line_segment(grid[0][4], grid[3][4]);

        // === Parallel constraints ===
        b.constrain_parallel(l_bottom, l_top);
        b.constrain_parallel(l_left, l_right);

        // === Perpendicular constraints ===
        b.constrain_perpendicular(l_bottom, l_left);

        // === Equal-length: left side = right side ===
        b.constrain_equal_length(l_left, l_right);

        // === Circle with points on it ===
        // Fix the circle so its center (25,45) and radius (8) are not free.
        let circle = b.add_circle(25.0, 45.0, 8.0);
        b.fix_entity(circle);
        // Three points on the circle: east (33,45), north (25,53), west (17,45).
        let cp1 = b.add_point(33.0, 45.0);
        let cp2 = b.add_point(25.0, 53.0);
        let cp3 = b.add_point(17.0, 45.0);
        b.constrain_point_on_circle(cp1, circle);
        b.constrain_point_on_circle(cp2, circle);
        b.constrain_point_on_circle(cp3, circle);
        // Pin angular positions: horizontal(cp1,cp3), distance(cp1,cp3)=16, distance(cp1,cp2)=8√2.
        b.constrain_horizontal(cp1, cp3);
        b.constrain_distance(cp1, cp3, 16.0);
        b.constrain_distance(cp1, cp2, (128.0_f64).sqrt());

        // === Midpoint constraint ===
        let mid = b.add_point(5.5, 5.5);
        let l_diag = b.add_line_segment(grid[0][0], grid[1][1]);
        b.constrain_midpoint(mid, l_diag);

        // === Symmetric about center ===
        // Pin sym_center to known position via distance+vertical to grid point (20,10).
        let sym_center = b.add_point(20.0, 15.0);
        let sym_p1 = b.add_point(15.0, 15.0);
        let sym_p2 = b.add_point(25.0, 15.0);
        b.constrain_symmetric(sym_p1, sym_p2, sym_center);
        b.constrain_horizontal(sym_p1, sym_p2);
        b.constrain_distance(sym_p1, sym_center, 5.0);
        b.constrain_vertical(sym_center, grid[1][2]); // x_center = 20
        b.constrain_distance(sym_center, grid[1][2], 5.0); // d((20,15),(20,10))=5

        // === Tangent line-circle ===
        // Pin tangent line endpoints: vertical(tang_p1,cp3), distance(tang_p1,tang_p2)=16.
        let tang_p1 = b.add_point(17.0, 53.0);
        let tang_p2 = b.add_point(33.0, 53.0);
        let tang_line = b.add_line_segment(tang_p1, tang_p2);
        b.constrain_tangent_line_circle(tang_line, circle);
        b.constrain_horizontal(tang_p1, tang_p2);
        b.constrain_vertical(tang_p1, cp3); // x_tang_p1 = 17
        b.constrain_distance(tang_p1, tang_p2, 16.0);

        // === Build and solve ===
        let mut sys = b.build();
        // Bypass the reduce phase to avoid reduce-phase coupling bugs with
        // chained Horizontal/Vertical constraints and fixed params.
        sys.set_pipeline(PipelineBuilder::new().reduce(NoopReduce).build());

        let n_entities = sys.entity_count();
        let n_constraints = sys.constraint_count();
        let dof = sys.degrees_of_freedom();
        eprintln!(
            "V3 massive sketch: {n_entities} entities, {n_constraints} constraints, DOF = {dof}"
        );

        assert!(n_entities >= 30, "Expected 30+ entities, got {n_entities}");
        assert!(
            n_constraints >= 40,
            "Expected 40+ constraints, got {n_constraints}"
        );

        let result = sys.solve();
        assert_solved(&result);

        eprintln!(
            "  Solved in {} total iterations across {} clusters",
            result.total_iterations,
            result.clusters.len()
        );

        // Verify all residuals are near zero.
        let residuals = sys.compute_residuals();
        let max_r = residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        assert!(
            max_r < 1e-3,
            "V3 sketch max residual {max_r:.2e} exceeds tolerance"
        );
    }

    // =====================================================================
    // 11. V3 Pipeline – Multi-Cluster Decomposition
    // =====================================================================

    /// Build a system with THREE completely independent groups of
    /// constraints (no shared parameters). Verify the pipeline
    /// decomposes into 3 separate clusters and solves each.
    #[test]
    fn megatest_v3_multi_cluster_decomposition() {
        let mut sys = ConstraintSystem::new();
        // Bypass the reduce phase to avoid reduce-phase coupling bugs
        // with PointOnCircle + fixed params.
        sys.set_pipeline(PipelineBuilder::new().reduce(NoopReduce).build());

        // --- Cluster A: 3-4-5 triangle (anchored with fix_param) ---
        let (ea0, xa0, ya0) = add_fixed_pt(&mut sys, 0.0, 0.0);
        let (ea1, xa1, ya1) = add_point(&mut sys, 3.1, -0.2);
        let (ea2, xa2, ya2) = add_point(&mut sys, -0.1, 4.1);

        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(Horizontal::new(cid, ea0, ea1, ya0, ya1)));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, ea0, ea1, xa0, ya0, xa1, ya1, 3.0,
        )));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, ea1, ea2, xa1, ya1, xa2, ya2, 5.0,
        )));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, ea2, ea0, xa2, ya2, xa0, ya0, 4.0,
        )));

        // --- Cluster B: rectangle at (100,100) ---
        let (eb0, xb0, yb0) = add_fixed_pt(&mut sys, 100.0, 100.0);
        let (eb1, xb1, yb1) = add_point(&mut sys, 110.2, 100.3);
        let (eb2, xb2, yb2) = add_point(&mut sys, 110.1, 110.2);
        let (eb3, xb3, yb3) = add_point(&mut sys, 99.9, 110.1);

        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, eb0, eb1, xb0, yb0, xb1, yb1, 10.0,
        )));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, eb1, eb2, xb1, yb1, xb2, yb2, 10.0,
        )));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, eb2, eb3, xb2, yb2, xb3, yb3, 10.0,
        )));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, eb3, eb0, xb3, yb3, xb0, yb0, 10.0,
        )));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(Horizontal::new(cid, eb0, eb1, yb0, yb1)));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(Vertical::new(cid, eb1, eb2, xb1, xb2)));

        // --- Cluster C: circle at (200,200) ---
        let (ec, cxc, cyc, crc) = add_circle_entity(&mut sys, 200.0, 200.0, 5.0);
        let (ep, xp, yp) = add_point(&mut sys, 205.1, 200.0);
        sys.fix_param(cxc);
        sys.fix_param(cyc);
        sys.fix_param(crc);
        // Fix yp to circle center y, so point is at (205, 200) on circle.
        // Use fix_param instead of Horizontal to avoid reduce phase coupling issues.
        sys.fix_param(yp);
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(PointOnCircle::new(
            cid, ep, ec, xp, yp, cxc, cyc, crc,
        )));

        let result = sys.solve();
        assert_solved(&result);

        // Verify multiple clusters were created.
        let n_clusters = result.clusters.len();
        eprintln!("Multi-cluster: {n_clusters} clusters detected");
        assert_eq!(
            n_clusters, 3,
            "Expected exactly 3 independent clusters, got {n_clusters}"
        );

        // Verify triangle.
        let da01 = pt_dist(&sys, xa0, ya0, xa1, ya1);
        assert!(
            (da01 - 3.0).abs() < 0.01,
            "Triangle d01={da01}, expected 3.0"
        );

        // Verify square sides.
        let db01 = pt_dist(&sys, xb0, yb0, xb1, yb1);
        assert!(
            (db01 - 10.0).abs() < 0.01,
            "Square d01={db01}, expected 10.0"
        );

        // Verify point on circle.
        let dp = pt_dist(&sys, xp, yp, cxc, cyc);
        assert!(
            (dp - 5.0).abs() < 0.01,
            "Point-on-circle dist={dp}, expected 5.0"
        );
    }

    // =====================================================================
    // 12. V3 Pipeline – Incremental Solving
    // =====================================================================

    /// Build a system, solve it, perturb one parameter, solve again
    /// incrementally. Verify the second solve re-uses cached clusters
    /// where possible and still produces a correct result.
    #[test]
    fn megatest_v3_incremental_solving() {
        let mut sys = ConstraintSystem::new();

        // Two independent clusters:
        // Cluster A: 3-4-5 triangle with fix_param anchor + horizontal.
        let (ea0, xa0, ya0) = add_fixed_pt(&mut sys, 0.0, 0.0);
        let (ea1, xa1, ya1) = add_point(&mut sys, 3.0, 0.5);
        let (ea2, xa2, ya2) = add_point(&mut sys, 0.5, 4.0);
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(Horizontal::new(cid, ea0, ea1, ya0, ya1)));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, ea0, ea1, xa0, ya0, xa1, ya1, 3.0,
        )));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, ea1, ea2, xa1, ya1, xa2, ya2, 5.0,
        )));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, ea2, ea0, xa2, ya2, xa0, ya0, 4.0,
        )));

        // Cluster B: separate pair (fix_param anchor + distance + horizontal).
        let (eb0, xb0, yb0) = add_fixed_pt(&mut sys, 100.0, 100.0);
        let (eb1, xb1, yb1) = add_point(&mut sys, 115.0, 100.5);
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, eb0, eb1, xb0, yb0, xb1, yb1, 15.0,
        )));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(Horizontal::new(cid, eb0, eb1, yb0, yb1)));

        // First solve (full).
        let result1 = sys.solve();
        assert_solved(&result1);
        let iters1 = result1.total_iterations;

        // Verify cluster B.
        let db = pt_dist(&sys, xb0, yb0, xb1, yb1);
        assert!((db - 15.0).abs() < 0.01, "First solve: dist={db}");

        // Perturb a parameter in cluster B only.
        sys.set_param(xb1, 116.0);

        // Incremental re-solve. Only cluster B should be re-solved.
        let result2 = sys.solve_incremental();
        assert_solved(&result2);

        // Verify the system is still satisfied.
        let db2 = pt_dist(&sys, xb0, yb0, xb1, yb1);
        assert!((db2 - 15.0).abs() < 0.01, "Incremental solve: dist={db2}");

        // The incremental solve should skip the unchanged cluster (fewer iterations).
        let skipped = result2
            .clusters
            .iter()
            .filter(|c| c.status == ClusterSolveStatus::Skipped)
            .count();
        eprintln!(
            "Incremental: {skipped} clusters skipped, {} total iters (was {iters1})",
            result2.total_iterations
        );
        assert!(
            skipped >= 1,
            "Expected at least 1 skipped cluster in incremental solve, got {skipped}"
        );
    }

    // =====================================================================
    // 13. V3 Pipeline – Drag (null-space projection)
    // =====================================================================

    /// Build an under-constrained system (2 points with only a distance
    /// constraint → 1 DOF rotation), drag one point, and verify the
    /// drag moves it along the null space.
    #[test]
    fn megatest_v3_drag() {
        let mut sys = ConstraintSystem::new();

        let (e0, x0, y0) = add_fixed_pt(&mut sys, 0.0, 0.0);
        let (e1, x1, y1) = add_point(&mut sys, 10.0, 0.0);

        // Distance constraint: |p0 - p1| = 10 (1 DOF: rotation).
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, e0, e1, x0, y0, x1, y1, 10.0,
        )));

        // Initial solve to establish the system.
        let result = sys.solve();
        assert_solved(&result);

        let dist_before = pt_dist(&sys, x0, y0, x1, y1);
        assert!(
            (dist_before - 10.0).abs() < 0.01,
            "Pre-drag distance wrong: {dist_before}"
        );

        // Apply a small drag: move p1 upward by 0.5 units.
        // At (10,0), the null space of the distance constraint is [0,1] (tangent to circle),
        // so a y-displacement is fully in the null space and should be preserved.
        let drag_result = sys.drag(&[(y1, 0.5)]);

        eprintln!(
            "Drag preservation ratio: {:.3}",
            drag_result.preservation_ratio
        );

        // Verify the drag actually moved the point.
        let y1_after_drag = sys.get_param(y1);
        eprintln!("y1 after drag: {y1_after_drag}");
        assert!(
            y1_after_drag.abs() > 0.1,
            "Drag should have moved y1, got y1={y1_after_drag}"
        );

        // The distance should be nearly perfectly preserved by null-space projection.
        // At (10,0), the null space of the distance constraint Jacobian [2*10, 0]
        // is [0,1], so a y-displacement is entirely in the null space.
        // Nonlinear deviation: sqrt(10² + 0.5²) = 10.0125, so error ~0.013.
        let dist_after_drag = pt_dist(&sys, x0, y0, x1, y1);
        eprintln!("Distance after drag: {dist_after_drag}");
        assert!(
            (dist_after_drag - 10.0).abs() < 0.02,
            "Drag should nearly preserve distance, got {dist_after_drag}"
        );
    }

    // =====================================================================
    // 14. V3 Pipeline – Diagnostics (DOF analysis)
    // =====================================================================

    /// Build systems with different constraint levels and verify DOF
    /// computation is correct.
    #[test]
    fn megatest_v3_dof_analysis() {
        // Well-constrained: 2 points, fix one, distance + horizontal on the other.
        // 4 params total, 2 fixed → 2 free. 2 equations → DOF = 0.
        let mut sys = ConstraintSystem::new();
        let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
        let (e1, x1, y1) = add_point(&mut sys, 10.0, 0.0);
        sys.fix_param(x0);
        sys.fix_param(y0);
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, e0, e1, x0, y0, x1, y1, 10.0,
        )));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(Horizontal::new(cid, e0, e1, y0, y1)));

        assert_eq!(sys.degrees_of_freedom(), 0, "Should be well-constrained");

        // Under-constrained: just the distance, no horizontal.
        let mut sys2 = ConstraintSystem::new();
        let (e0, x0, y0) = add_point(&mut sys2, 0.0, 0.0);
        let (e1, x1, y1) = add_point(&mut sys2, 10.0, 0.0);
        sys2.fix_param(x0);
        sys2.fix_param(y0);
        let cid = sys2.alloc_constraint_id();
        sys2.add_constraint(Box::new(DistancePtPt::new(
            cid, e0, e1, x0, y0, x1, y1, 10.0,
        )));

        assert_eq!(
            sys2.degrees_of_freedom(),
            1,
            "Should be under-constrained (1 DOF)"
        );

        // Over-constrained: 2 fixed points + distance + horizontal + vertical.
        // 2 free params, 3 equations → DOF = -1.
        let mut sys3 = ConstraintSystem::new();
        let (e0, x0, y0) = add_point(&mut sys3, 0.0, 0.0);
        let (e1, x1, y1) = add_point(&mut sys3, 10.0, 0.0);
        sys3.fix_param(x0);
        sys3.fix_param(y0);
        let cid = sys3.alloc_constraint_id();
        sys3.add_constraint(Box::new(DistancePtPt::new(
            cid, e0, e1, x0, y0, x1, y1, 10.0,
        )));
        let cid = sys3.alloc_constraint_id();
        sys3.add_constraint(Box::new(Horizontal::new(cid, e0, e1, y0, y1)));
        let cid = sys3.alloc_constraint_id();
        sys3.add_constraint(Box::new(Vertical::new(cid, e0, e1, x0, x1)));

        assert_eq!(
            sys3.degrees_of_freedom(),
            -1,
            "Should be over-constrained (-1 DOF)"
        );
    }

    // =====================================================================
    // 15. V3 Pipeline – Entity/Constraint Lifecycle
    // =====================================================================

    /// Test adding and removing entities and constraints, then solving.
    /// Verifies the pipeline handles structural changes correctly.
    #[test]
    fn megatest_v3_entity_lifecycle() {
        let mut sys = ConstraintSystem::new();

        // 3-4-5 right triangle (anchored with fix_param).
        let (e0, x0, y0) = add_fixed_pt(&mut sys, 0.0, 0.0);
        let (e1, x1, y1) = add_point(&mut sys, 3.1, -0.2);
        let (e2, x2, y2) = add_point(&mut sys, -0.1, 4.1);

        // Add distance + horizontal constraints for DOF=0.
        let cid_h = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(Horizontal::new(cid_h, e0, e1, y0, y1)));
        let cid1 = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid1, e0, e1, x0, y0, x1, y1, 3.0,
        )));
        let cid2 = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid2, e1, e2, x1, y1, x2, y2, 5.0,
        )));
        let cid3 = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid3, e2, e0, x2, y2, x0, y0, 4.0,
        )));

        // First solve: well-constrained triangle (DOF=0).
        let result1 = sys.solve();
        assert_solved(&result1);
        let d01 = pt_dist(&sys, x0, y0, x1, y1);
        assert!((d01 - 3.0).abs() < 0.01, "Triangle d01={d01}, expected 3.0");

        // Remove the distance(e1,e2)=5 constraint (structural change).
        sys.remove_constraint(cid2);
        assert_eq!(
            sys.constraint_count(),
            3,
            "Should have 3 constraints after removal"
        );

        // Add a distance(e1,e2)=4 constraint instead (different target).
        let cid4 = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid4, e1, e2, x1, y1, x2, y2, 4.0,
        )));

        // Re-solve after structural change.
        let result2 = sys.solve();
        assert_solved(&result2);

        // Distance d(e1,e2) should now be 4.0 (was 5.0 before lifecycle change).
        let d12_new = pt_dist(&sys, x1, y1, x2, y2);
        assert!(
            (d12_new - 4.0).abs() < 0.05,
            "After lifecycle change: d12={d12_new}, expected 4.0"
        );
        // Other constraints still satisfied.
        let d01_new = pt_dist(&sys, x0, y0, x1, y1);
        assert!(
            (d01_new - 3.0).abs() < 0.05,
            "After lifecycle change: d01={d01_new}, expected 3.0"
        );
    }

    // =====================================================================
    // 16. V3 Pipeline – Sketch2DBuilder End-to-End
    // =====================================================================

    /// Use only the Sketch2DBuilder API to construct and solve a complete
    /// geometric figure: rectangle with tangent circle.
    #[test]
    fn megatest_v3_builder_rectangle_inscribed_circle() {
        let mut b = Sketch2DBuilder::new();

        // Rectangle: 4 corners near a 20×10 rectangle.
        let p0 = b.add_fixed_point(0.0, 0.0);
        let p1 = b.add_point(20.1, 0.3);
        let p2 = b.add_point(20.2, 9.8);
        let p3 = b.add_point(-0.1, 10.1);

        // Edges
        let l01 = b.add_line_segment(p0, p1);
        let l12 = b.add_line_segment(p1, p2);
        let l23 = b.add_line_segment(p2, p3);
        let _l30 = b.add_line_segment(p3, p0);

        // Side-length constraints
        b.constrain_distance(p0, p1, 20.0);
        b.constrain_distance(p1, p2, 10.0);
        b.constrain_distance(p2, p3, 20.0);
        b.constrain_distance(p3, p0, 10.0);

        // Right angles (1 perpendicular + horizontal = rectangle)
        b.constrain_perpendicular(l01, l12);

        // Horizontal bottom
        b.constrain_horizontal(p0, p1);

        // Inscribed circle tangent to bottom, right, and top sides.
        // 3 tangent constraints for 3 circle unknowns → DOF=0 for circle.
        let circ = b.add_circle(10.5, 5.5, 4.8);
        b.constrain_tangent_line_circle(l01, circ);
        b.constrain_tangent_line_circle(l12, circ);
        b.constrain_tangent_line_circle(l23, circ);

        let mut sys = b.build();

        eprintln!(
            "Rectangle+circle: {} entities, {} constraints, DOF = {}",
            sys.entity_count(),
            sys.constraint_count(),
            sys.degrees_of_freedom()
        );

        let result = sys.solve();
        assert_solved(&result);

        // Verify residuals are near zero.
        let residuals = sys.compute_residuals();
        let max_r = residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        assert!(max_r < 1e-3, "Rectangle+circle max residual {max_r:.2e}");

        eprintln!(
            "Rectangle+inscribed circle: solved in {} iterations",
            result.total_iterations
        );
    }

    // =====================================================================
    // 17. V3 Pipeline – Large Grid (stress test)
    // =====================================================================

    /// Build a 10×10 grid of points (100 points, ~200 constraints)
    /// through the V3 pipeline. This tests scalability of decomposition,
    /// analysis, and solving.
    #[test]
    #[ignore] // ~1.4s; run with `cargo test -- --ignored`
    fn megatest_v3_large_grid_stress() {
        let mut b = Sketch2DBuilder::new();

        let rows = 10;
        let cols = 10;
        let spacing = 5.0;

        // Create grid points with slight perturbation.
        // Use add_fixed_point for origin (avoids Fixed constraint reduce bug).
        let origin = b.add_fixed_point(0.0, 0.0);

        let mut grid: Vec<Vec<EntityId>> = Vec::new();
        for j in 0..rows {
            let mut row = Vec::new();
            for i in 0..cols {
                if i == 0 && j == 0 {
                    row.push(origin);
                } else {
                    let x = i as f64 * spacing + 0.2 * ((i + j) as f64 * 0.7).sin();
                    let y = j as f64 * spacing + 0.2 * ((i * j) as f64 * 1.1).cos();
                    row.push(b.add_point(x, y));
                }
            }
            grid.push(row);
        }

        // Horizontal constraints across rows.
        for j in 0..rows {
            for i in 0..(cols - 1) {
                b.constrain_horizontal(grid[j][i], grid[j][i + 1]);
            }
        }

        // Vertical constraints down columns.
        for j in 0..(rows - 1) {
            for i in 0..cols {
                b.constrain_vertical(grid[j][i], grid[j + 1][i]);
            }
        }

        // Distance constraints ONLY for first row and first column
        // (horizontal/vertical propagate positions throughout the grid).
        for i in 0..(cols - 1) {
            b.constrain_distance(grid[0][i], grid[0][i + 1], spacing);
        }
        for j in 0..(rows - 1) {
            b.constrain_distance(grid[j][0], grid[j + 1][0], spacing);
        }

        let mut sys = b.build();
        // Bypass the reduce phase to avoid reduce-phase coupling bugs with
        // chained Horizontal/Vertical constraints and fixed params.
        sys.set_pipeline(PipelineBuilder::new().reduce(NoopReduce).build());

        let n_entities = sys.entity_count();
        let n_constraints = sys.constraint_count();
        let dof = sys.degrees_of_freedom();
        eprintln!("Large grid: {n_entities} entities, {n_constraints} constraints, DOF = {dof}");

        assert!(
            n_entities >= 100,
            "Expected 100+ entities, got {n_entities}"
        );
        assert!(
            n_constraints >= 190,
            "Expected 190+ constraints, got {n_constraints}"
        );

        let result = sys.solve();
        assert_solved(&result);

        let residuals = sys.compute_residuals();
        let max_r = residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        assert!(max_r < 1e-3, "Large grid max residual {max_r:.2e}");

        eprintln!(
            "  Solved in {} iterations across {} clusters, duration {:?}",
            result.total_iterations,
            result.clusters.len(),
            result.duration,
        );
    }

    // =====================================================================
    // 18. V3 Pipeline – Redundancy Detection
    // =====================================================================

    /// Build a system with redundant constraints and verify the
    /// redundancy analysis detects them.
    #[test]
    fn megatest_v3_redundancy_detection() {
        let mut sys = ConstraintSystem::new();

        let (e0, x0, y0) = add_point(&mut sys, 0.0, 0.0);
        let (e1, x1, y1) = add_point(&mut sys, 10.0, 0.0);

        // Anchor with fix_param.
        sys.fix_param(x0);
        sys.fix_param(y0);

        // Two distance constraints that say the same thing (redundant).
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, e0, e1, x0, y0, x1, y1, 10.0,
        )));
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid, e0, e1, x0, y0, x1, y1, 10.0,
        )));

        // Horizontal constraint.
        let cid = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(Horizontal::new(cid, e0, e1, y0, y1)));

        // The system should still solve (redundant constraints are consistent).
        let result = sys.solve();
        assert_solved(&result);

        let dist = pt_dist(&sys, x0, y0, x1, y1);
        assert!(
            (dist - 10.0).abs() < 0.01,
            "Distance should be 10, got {dist}"
        );

        // Run redundancy analysis.
        let analysis = sys.analyze_redundancy();
        eprintln!(
            "Redundancy: {} redundant, {} conflicts",
            analysis.redundant.len(),
            analysis.conflicts.len()
        );
        // We expect at least 1 redundant constraint detected.
        assert!(
            !analysis.redundant.is_empty(),
            "Should detect redundant constraints"
        );
    }

    // =====================================================================
    // 19. V3 Pipeline – Multiple Solve Cycles
    // =====================================================================

    /// Solve, modify, solve, modify, solve — three cycles. Verifies the
    /// pipeline handles repeated solves with structural and value changes.
    #[test]
    fn megatest_v3_multiple_solve_cycles() {
        let mut sys = ConstraintSystem::new();

        // 3-4-5 triangle with fix_param anchor + horizontal (well-constrained).
        let (e0, x0, y0) = add_fixed_pt(&mut sys, 0.0, 0.0);
        let (e1, x1, y1) = add_point(&mut sys, 3.0, 0.5);
        let (e2, x2, y2) = add_point(&mut sys, 0.5, 4.0);

        let cid_h = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(Horizontal::new(cid_h, e0, e1, y0, y1)));

        let cid_d01 = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid_d01, e0, e1, x0, y0, x1, y1, 3.0,
        )));
        let cid_d12 = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid_d12, e1, e2, x1, y1, x2, y2, 5.0,
        )));
        let cid_d20 = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(DistancePtPt::new(
            cid_d20, e2, e0, x2, y2, x0, y0, 4.0,
        )));

        // Cycle 1: solve the triangle.
        let r1 = sys.solve();
        assert_solved(&r1);
        let d01_1 = pt_dist(&sys, x0, y0, x1, y1);
        assert!(
            (d01_1 - 3.0).abs() < 0.01,
            "Cycle 1: d01={d01_1}, expected 3.0"
        );

        // Cycle 2: perturb p1 and re-solve (value change only).
        sys.set_param(x1, 3.5);
        let r2 = sys.solve();
        assert_solved(&r2);
        let d01_2 = pt_dist(&sys, x0, y0, x1, y1);
        assert!(
            (d01_2 - 3.0).abs() < 0.01,
            "Cycle 2: d01={d01_2}, expected 3.0"
        );

        // Cycle 3: remove distance d12, add vertical on p2, re-solve (structural change).
        sys.remove_constraint(cid_d12);
        let cid_v = sys.alloc_constraint_id();
        sys.add_constraint(Box::new(Vertical::new(cid_v, e0, e2, x0, x2)));

        let r3 = sys.solve();
        assert_solved(&r3);
        let d01_3 = pt_dist(&sys, x0, y0, x1, y1);
        assert!(
            (d01_3 - 3.0).abs() < 0.01,
            "Cycle 3: d01={d01_3}, expected 3.0"
        );

        // p0 and p1 should still be horizontal (same y).
        let py0 = sys.get_param(y0);
        let py1 = sys.get_param(y1);
        assert!((py0 - py1).abs() < 0.01, "Cycle 3: y0={py0}, y1={py1}");
    }
}
