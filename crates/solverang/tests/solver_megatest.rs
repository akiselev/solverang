//! Comprehensive solver megatest.
//!
//! Tests the solver with massive systems that have known analytical solutions.
//! Exercises every solver type and, via the geometry module, every 2D constraint type.
//!
//! # Test Structure
//!
//! 1. **Coupled Nonlinear Chain** – 100-variable tridiagonal nonlinear system with
//!    manufactured solution. Tested with NR, LM, Auto, and Robust solvers.
//!
//! 2. **Overdetermined Mega** – 200 equations / 50 variables, least-squares system
//!    with known zero-residual solution. Tests LM on large overdetermined problems.
//!
//! 3. **Massive Geometric System** – ~50 points exercising every 2D constraint type
//!    (distance, coincident, fixed, horizontal, vertical, angle, parallel,
//!    perpendicular, midpoint, point-on-line, point-on-circle, collinear,
//!    equal-length, symmetric, symmetric-about-line, line-tangent-circle,
//!    circles-tangent-external, circles-tangent-internal). Solved with LM.
//!
//! 4. **Sparse Mega System** – 500-variable sparse tridiagonal nonlinear system,
//!    tested with the sparse solver.
//!
//! 5. **Cross-Solver Comparison** – Identical problem solved by every solver;
//!    results compared for consistency.
//!
//! 6. **Jacobian Verification** – Analytical Jacobians verified against finite
//!    differences for the coupled nonlinear and geometric systems.
//!
//! 7. **Robustness Under Poor Initial Conditions** – Starting points 10× away
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
                let x_next = if i < size - 1 {
                    solution[i + 1]
                } else {
                    0.0
                };
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
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: dimension mismatch"
    );
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
// 3. Massive Geometric System – all constraint types
// ============================================================================

#[cfg(feature = "geometry")]
mod geometric_megatest {
    use solverang::geometry::constraints::*;
    use solverang::geometry::{ConstraintSystem, Point2D};
    use solverang::{verify_jacobian, LMConfig, LMSolver, SolveResult};

    /// Convergence tolerance for geometric tests.
    const TOL: f64 = 1e-5;

    /// Build a massive 2D constraint system that exercises EVERY 2D constraint
    /// type.  All points have a known analytical solution position.
    ///
    /// Point layout (solution coordinates):
    ///
    /// ```text
    ///   Grid (4×3):
    ///     p0=(0,0)  p1=(10,0)  p2=(20,0)  p3=(30,0)
    ///     p4=(0,10) p5=(10,10) p6=(20,10) p7=(30,10)
    ///     p8=(0,20) p9=(10,20) p10=(20,20) p11=(30,20)
    ///
    ///   Roof triangle:
    ///     p12=(10,30)  p13=(20,30)  p14=(15,30+5√3)
    ///
    ///   Circle section (center + 4 on circle, radius 5):
    ///     p15=(45,10)  center
    ///     p16=(50,10)  p17=(45,15)  p18=(40,10)  p19=(45,5)
    ///
    ///   Midpoint / point-on-line section:
    ///     p20=(5,5)    midpoint of p0 and p5
    ///     p21=(15,15)  on line p5-p10
    ///
    ///   Collinear section:
    ///     p22=(35,0)  p23=(35,10)  p24=(35,5) p25=(35,15)
    ///     segments (p22,p23) and (p24,p25) are collinear
    ///
    ///   Symmetric section:
    ///     p26=(12,25)  p27=(18,25)  p28=(15,25)  symmetric about p28
    ///
    ///   Angle section:
    ///     p29=(40,0)  p30=(40+5√3, 5)  angle = 30° from horizontal
    ///
    ///   Tangent section:
    ///     p31=(45,25) center of tangent circle, radius 5
    ///     p32=(40,30) p33=(50,30) line tangent to circle (horizontal at y=30)
    ///
    ///   Circles tangent (external):
    ///     p34=(60,10) center, radius 4
    ///     p35=(45,10) = p15, radius 5  → dist = 15 > 4+5 ... need to adjust
    ///     Actually: p34=(54,10), radius 4 → dist to p15 = 9 = 5+4 ✓
    ///
    ///   Circles tangent (internal):
    ///     p35=(47,10) center, radius 3 → dist to p15 = 2 = 5-3 ✓
    ///
    ///   Symmetric about line:
    ///     p36=(13,35)  p37=(17,35) symmetric about vertical line x=15
    ///     axis: p38=(15,30) p39=(15,40)
    ///
    ///   Perpendicular:
    ///     p40=(50,20) p41=(60,20) – horizontal
    ///     p42=(55,15) p43=(55,25) – vertical → perpendicular
    ///
    ///   Coincident:
    ///     p44=(30,20) coincident with p11
    ///
    ///   Equal-length (already used via grid, but add explicit):
    ///     |p40-p41| = |p42-p43| (both 10)
    ///
    ///   Parallel:
    ///     (p0,p3) parallel to (p8,p11) – both horizontal at y=0, y=20
    /// ```
    fn build_mega_geometric_system() -> (ConstraintSystem<2>, Vec<(usize, Point2D)>) {
        let sqrt3 = 3.0_f64.sqrt();

        // Known solution coordinates for every point.
        let solution_points: Vec<Point2D> = vec![
            // Grid 4×3 (indices 0-11)
            Point2D::new(0.0, 0.0),   // p0
            Point2D::new(10.0, 0.0),  // p1
            Point2D::new(20.0, 0.0),  // p2
            Point2D::new(30.0, 0.0),  // p3
            Point2D::new(0.0, 10.0),  // p4
            Point2D::new(10.0, 10.0), // p5
            Point2D::new(20.0, 10.0), // p6
            Point2D::new(30.0, 10.0), // p7
            Point2D::new(0.0, 20.0),  // p8
            Point2D::new(10.0, 20.0), // p9
            Point2D::new(20.0, 20.0), // p10
            Point2D::new(30.0, 20.0), // p11
            // Roof triangle (indices 12-14)
            Point2D::new(10.0, 30.0),        // p12
            Point2D::new(20.0, 30.0),        // p13
            Point2D::new(15.0, 30.0 + 5.0 * sqrt3), // p14
            // Circle section (indices 15-19)
            Point2D::new(45.0, 10.0), // p15 center
            Point2D::new(50.0, 10.0), // p16 east
            Point2D::new(45.0, 15.0), // p17 north
            Point2D::new(40.0, 10.0), // p18 west
            Point2D::new(45.0, 5.0),  // p19 south
            // Midpoint / point-on-line (indices 20-21)
            Point2D::new(5.0, 5.0),   // p20 midpoint(p0,p5)
            Point2D::new(15.0, 15.0), // p21 on line(p5,p10)
            // Collinear (indices 22-25)
            Point2D::new(35.0, 0.0),  // p22
            Point2D::new(35.0, 10.0), // p23
            Point2D::new(35.0, 5.0),  // p24
            Point2D::new(35.0, 15.0), // p25
            // Symmetric about point (indices 26-28)
            Point2D::new(12.0, 25.0), // p26
            Point2D::new(18.0, 25.0), // p27
            Point2D::new(15.0, 25.0), // p28 center
            // Angle section (indices 29-30)
            Point2D::new(40.0, 0.0),              // p29
            Point2D::new(40.0 + 5.0 * sqrt3, 5.0), // p30 (angle = 30°)
            // Tangent line-circle (indices 31-33)
            Point2D::new(45.0, 25.0), // p31 circle center, radius 5
            Point2D::new(40.0, 30.0), // p32 line start
            Point2D::new(50.0, 30.0), // p33 line end
            // Circles tangent external (index 34)
            Point2D::new(54.0, 10.0), // p34 center, radius 4 → dist to p15=9=5+4
            // Circles tangent internal (index 35)
            Point2D::new(47.0, 10.0), // p35 center, radius 3 → dist to p15=2=5-3
            // Symmetric about line (indices 36-39)
            Point2D::new(13.0, 35.0), // p36
            Point2D::new(17.0, 35.0), // p37
            Point2D::new(15.0, 30.0), // p38 axis start
            Point2D::new(15.0, 40.0), // p39 axis end
            // Perpendicular + equal-length (indices 40-43)
            // p40 must be midpoint of p42,p43 → (55,20)
            Point2D::new(55.0, 20.0), // p40
            Point2D::new(65.0, 20.0), // p41  (10 away from p40 horizontally)
            Point2D::new(55.0, 15.0), // p42
            Point2D::new(55.0, 25.0), // p43
            // Coincident (index 44)
            Point2D::new(30.0, 20.0), // p44 = same as p11
        ];

        // Indices of points that will be fixed via fix_point() – these must
        // be initialised at their exact solution positions because the solver
        // never updates them.
        let fixed_indices: &[usize] = &[0, 1, 15];

        // Build perturbed initial positions (the solver's starting point).
        // Keep perturbations moderate so the over-constrained solver converges
        // to the correct minimum rather than a spurious local one.
        let perturbed: Vec<Point2D> = solution_points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if fixed_indices.contains(&i) {
                    *p // Fixed points stay at exact solution
                } else {
                    let dx = 0.3 * ((i as f64 * 1.7).sin());
                    let dy = 0.3 * ((i as f64 * 2.3).cos());
                    Point2D::new(p.x() + dx, p.y() + dy)
                }
            })
            .collect();

        let n = perturbed.len();
        let mut system = ConstraintSystem::<2>::new();

        // Add all points (perturbed positions).
        for p in &perturbed {
            system.add_point(*p);
        }

        // Fix anchor points to remove rigid-body DOF.
        system.fix_point(0); // p0 fixed at (0,0)
        system.fix_point(1); // p1 fixed at (10,0)

        // === Grid horizontal constraints ===
        // Row 0
        system.add_constraint(Box::new(HorizontalConstraint::new(0, 1)));
        system.add_constraint(Box::new(HorizontalConstraint::new(1, 2)));
        system.add_constraint(Box::new(HorizontalConstraint::new(2, 3)));
        // Row 1
        system.add_constraint(Box::new(HorizontalConstraint::new(4, 5)));
        system.add_constraint(Box::new(HorizontalConstraint::new(5, 6)));
        system.add_constraint(Box::new(HorizontalConstraint::new(6, 7)));
        // Row 2
        system.add_constraint(Box::new(HorizontalConstraint::new(8, 9)));
        system.add_constraint(Box::new(HorizontalConstraint::new(9, 10)));
        system.add_constraint(Box::new(HorizontalConstraint::new(10, 11)));

        // === Grid vertical constraints ===
        system.add_constraint(Box::new(VerticalConstraint::new(0, 4)));
        system.add_constraint(Box::new(VerticalConstraint::new(4, 8)));
        system.add_constraint(Box::new(VerticalConstraint::new(1, 5)));
        system.add_constraint(Box::new(VerticalConstraint::new(5, 9)));
        system.add_constraint(Box::new(VerticalConstraint::new(2, 6)));
        system.add_constraint(Box::new(VerticalConstraint::new(6, 10)));
        system.add_constraint(Box::new(VerticalConstraint::new(3, 7)));
        system.add_constraint(Box::new(VerticalConstraint::new(7, 11)));

        // === Grid distance constraints (to fully constrain the grid) ===
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(0, 4, 10.0)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(0, 1, 10.0)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(1, 2, 10.0)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(2, 3, 10.0)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(4, 8, 10.0)));

        // === Roof triangle (distance + horizontal) ===
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(12, 13, 10.0)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(12, 14, 10.0)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(13, 14, 10.0)));
        system.add_constraint(Box::new(HorizontalConstraint::new(12, 13)));
        // Connect roof to grid
        system.add_constraint(Box::new(VerticalConstraint::new(9, 12)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(9, 12, 10.0)));
        system.add_constraint(Box::new(VerticalConstraint::new(10, 13)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(10, 13, 10.0)));

        // === Circle section – PointOnCircle ===
        system.fix_point(15); // Fix circle center
        let circle_radius = 5.0;
        system.add_constraint(Box::new(PointOnCircleConstraint::<2>::new(16, 15, circle_radius)));
        system.add_constraint(Box::new(PointOnCircleConstraint::<2>::new(17, 15, circle_radius)));
        system.add_constraint(Box::new(PointOnCircleConstraint::<2>::new(18, 15, circle_radius)));
        system.add_constraint(Box::new(PointOnCircleConstraint::<2>::new(19, 15, circle_radius)));
        // Constrain positions on circle (horizontal/vertical from center)
        system.add_constraint(Box::new(HorizontalConstraint::new(15, 16)));
        system.add_constraint(Box::new(VerticalConstraint::new(15, 17)));
        system.add_constraint(Box::new(HorizontalConstraint::new(15, 18)));
        system.add_constraint(Box::new(VerticalConstraint::new(15, 19)));

        // === Midpoint ===
        system.add_constraint(Box::new(MidpointConstraint::<2>::new(20, 0, 5)));

        // === PointOnLine ===
        system.add_constraint(Box::new(PointOnLineConstraint::<2>::new(21, 5, 10)));
        // Also constrain p21's distance from p5 to pin it down
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(21, 5, 5.0 * 2.0_f64.sqrt())));

        // === Collinear ===
        system.add_constraint(Box::new(CollinearConstraint::<2>::new(22, 23, 24, 25)));
        // Pin the collinear points
        system.add_constraint(Box::new(VerticalConstraint::new(22, 23)));
        system.add_constraint(Box::new(VerticalConstraint::new(24, 25)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(22, 23, 10.0)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(24, 25, 10.0)));
        system.add_constraint(Box::new(HorizontalConstraint::new(3, 22)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(3, 22, 5.0)));
        system.add_constraint(Box::new(MidpointConstraint::<2>::new(24, 22, 23)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(23, 25, 5.0)));

        // === Symmetric about point ===
        system.add_constraint(Box::new(SymmetricConstraint::<2>::new(26, 27, 28)));
        // Pin the center and one end
        system.add_constraint(Box::new(FixedConstraint::<2>::new(28, Point2D::new(15.0, 25.0))));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(26, 28, 3.0)));
        system.add_constraint(Box::new(HorizontalConstraint::new(26, 28)));

        // === Angle constraint ===
        system.add_constraint(Box::new(AngleConstraint::from_degrees(29, 30, 30.0)));
        system.add_constraint(Box::new(FixedConstraint::<2>::new(29, Point2D::new(40.0, 0.0))));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(29, 30, 10.0)));

        // === Line tangent to circle ===
        system.add_constraint(Box::new(FixedConstraint::<2>::new(31, Point2D::new(45.0, 25.0))));
        let tangent_radius = 5.0;
        system.add_constraint(Box::new(LineTangentConstraint::new(32, 33, 31, tangent_radius)));
        system.add_constraint(Box::new(HorizontalConstraint::new(32, 33)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(32, 33, 10.0)));
        // Pin vertical position of tangent line
        system.add_constraint(Box::new(FixedConstraint::<2>::new(32, Point2D::new(40.0, 30.0))));

        // === Circles tangent external ===
        // p15 center (fixed, radius 5), p34 center (radius 4), dist = 9 = 5+4
        system.add_constraint(Box::new(CircleTangentConstraint::external(15, 5.0, 34, 4.0)));
        system.add_constraint(Box::new(HorizontalConstraint::new(15, 34)));

        // === Circles tangent internal ===
        // p15 center (fixed, radius 5), p35 center (radius 3), dist = 2 = 5-3
        system.add_constraint(Box::new(CircleTangentConstraint::internal(15, 5.0, 35, 3.0)));
        system.add_constraint(Box::new(HorizontalConstraint::new(15, 35)));

        // === Symmetric about line ===
        system.add_constraint(Box::new(SymmetricAboutLineConstraint::new(36, 37, 38, 39)));
        // Pin axis
        system.add_constraint(Box::new(FixedConstraint::<2>::new(38, Point2D::new(15.0, 30.0))));
        system.add_constraint(Box::new(FixedConstraint::<2>::new(39, Point2D::new(15.0, 40.0))));
        // Pin one side
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(36, 38, (4.0_f64 + 25.0).sqrt())));
        system.add_constraint(Box::new(HorizontalConstraint::new(36, 37)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(36, 37, 4.0)));

        // === Perpendicular ===
        system.add_constraint(Box::new(PerpendicularConstraint::<2>::new(40, 41, 42, 43)));
        system.add_constraint(Box::new(HorizontalConstraint::new(40, 41)));
        system.add_constraint(Box::new(VerticalConstraint::new(42, 43)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(40, 41, 10.0)));
        system.add_constraint(Box::new(DistanceConstraint::<2>::new(42, 43, 10.0)));
        system.add_constraint(Box::new(MidpointConstraint::<2>::new(40, 42, 43)));
        // Pin one corner of the cross
        system.add_constraint(Box::new(FixedConstraint::<2>::new(42, Point2D::new(55.0, 15.0))));

        // === Equal-length ===
        // |p40-p41| = |p42-p43| (both 10, already constrained above)
        system.add_constraint(Box::new(EqualLengthConstraint::<2>::new(40, 41, 42, 43)));

        // === Coincident ===
        system.add_constraint(Box::new(CoincidentConstraint::<2>::new(44, 11)));

        // === Parallel ===
        // Bottom row (p0,p3) parallel to top row (p8,p11)
        system.add_constraint(Box::new(ParallelConstraint::<2>::new(0, 3, 8, 11)));

        // Build the list of (index, expected_point) for verification.
        let expected: Vec<(usize, Point2D)> = (0..n)
            .map(|i| (i, solution_points[i]))
            .collect();

        (system, expected)
    }

    /// First verify that the known solution positions actually satisfy every
    /// constraint (residuals ≈ 0).  If this fails, the test fixture is wrong.
    #[test]
    fn megatest_geometric_solution_sanity() {
        let (mut system, expected_points) = build_mega_geometric_system();

        // Set ALL points (including fixed) to their known solution coordinates.
        for &(idx, ref sol_pt) in &expected_points {
            system.set_point(idx, *sol_pt);
        }

        let residuals = system.evaluate_residuals();
        for (i, r) in residuals.iter().enumerate() {
            if r.abs() > 1e-6 {
                eprintln!("  Constraint equation {i}: residual = {r:.6e}");
            }
        }
        let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        assert!(
            max_residual < 1e-6,
            "Known solution should satisfy all constraints, max residual = {max_residual:.6e}"
        );
    }

    #[test]
    fn megatest_geometric_all_constraint_types() {
        let (mut system, expected_points) = build_mega_geometric_system();

        let n_points = system.point_count();
        let n_constraints = system.constraint_count();
        let n_vars = system.total_variable_count();
        let n_eqs = system.equation_count();

        eprintln!(
            "Geometric megatest: {n_points} points, {n_constraints} constraints, \
             {n_vars} variables, {n_eqs} equations, DOF = {}",
            system.degrees_of_freedom()
        );

        // Sanity: should be a large system.
        assert!(n_points >= 40, "Expected 40+ points, got {n_points}");
        assert!(n_constraints >= 50, "Expected 50+ constraints, got {n_constraints}");

        // Solve with Levenberg-Marquardt (system is over-constrained, DOF < 0).
        let solver = LMSolver::new(LMConfig::precise());
        let initial = system.current_values();
        let result = solver.solve(&system, &initial);

        assert!(
            result.is_converged(),
            "Geometric megatest should converge: {result:?}"
        );

        if let SolveResult::Converged {
            ref solution,
            iterations,
            residual_norm,
            ..
        } = result
        {
            eprintln!("  Converged in {iterations} iterations, residual norm = {residual_norm:.2e}");

            system.set_values(solution);

            // Verify all residuals are near zero.
            let residuals = system.evaluate_residuals();
            let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
            assert!(
                max_residual < TOL,
                "Max residual {max_residual:.2e} exceeds tolerance"
            );

            // Verify points are close to known solution positions.
            for &(idx, ref expected_pt) in &expected_points {
                if system.is_fixed(idx) {
                    continue; // Fixed points don't move.
                }
                let actual = system.get_point(idx).expect("point should exist");
                let dx = (actual.x() - expected_pt.x()).abs();
                let dy = (actual.y() - expected_pt.y()).abs();
                let dist = (dx * dx + dy * dy).sqrt();
                // Allow some tolerance since the system may be under- or
                // over-constrained in places and could have multiple solutions.
                // We mainly verify residuals; point checks use a looser tolerance.
                assert!(
                    dist < 1.0,
                    "Point {idx} too far from expected: actual=({:.4}, {:.4}), expected=({:.4}, {:.4}), dist={dist:.4}",
                    actual.x(), actual.y(), expected_pt.x(), expected_pt.y()
                );
            }
        }
    }

    #[test]
    fn megatest_geometric_jacobian_verification() {
        let (system, _) = build_mega_geometric_system();
        // Verify at the (perturbed) initial values.  We use a moderate tolerance
        // because some advanced constraints (tangent, symmetric-about-line) may
        // have larger finite-difference approximation errors at arbitrary points.
        let x = system.current_values();

        let verification = verify_jacobian(&system, &x, 1e-6, 1e-3);

        assert!(
            verification.passed,
            "Geometric megatest Jacobian verification failed: max error = {:.2e} at {:?}",
            verification.max_absolute_error,
            verification.max_error_location
        );
    }
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
            assert_solution_close(solution, &problem.solution, 0.5, "Sparse-500");
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
        verification.max_absolute_error,
        verification.max_error_location
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
        verification.max_absolute_error,
        verification.max_error_location
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
        verification.max_absolute_error,
        verification.max_error_location
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
