//! MINPACK test problems solved through the optimization path (M7).
//!
//! Uses LeastSquaresObjective to wrap existing Problem implementations
//! as Objective instances, then solves with BFGS.

use solverang::id::EntityId;
use solverang::optimization::adapters::LeastSquaresObjective;
use solverang::optimization::{ObjectiveId, OptimizationConfig, OptimizationStatus};
use solverang::param::ParamStore;
use solverang::solver::BfgsSolver;
use solverang::test_problems::*;
use solverang::{Objective, Problem};

/// Helper: create a ParamStore with n free params initialized from the problem's initial point.
fn setup_store(problem: &dyn Problem, factor: f64) -> (ParamStore, Vec<solverang::ParamId>) {
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let x0 = problem.initial_point(factor);
    let param_ids: Vec<_> = x0.iter().map(|&v| store.alloc(v, owner)).collect();
    (store, param_ids)
}

/// Helper: solve a problem via BFGS through the LeastSquaresObjective adapter.
fn solve_via_bfgs(
    problem: &dyn Problem,
    factor: f64,
    max_iter: usize,
    tolerance: f64,
) -> (OptimizationStatus, f64, Vec<f64>) {
    let (mut store, param_ids) = setup_store(problem, factor);
    let obj = LeastSquaresObjective::new(
        ClosureProblemRef { problem },
        param_ids.clone(),
        ObjectiveId::new(0, 0),
    );
    let mut config = OptimizationConfig::default();
    config.max_outer_iterations = max_iter;
    config.dual_tolerance = tolerance;

    let result = BfgsSolver::solve(&obj, &mut store, &config);
    let solution: Vec<f64> = param_ids.iter().map(|&pid| store.get(pid)).collect();
    (result.status, result.objective_value, solution)
}

/// Wrapper to make &dyn Problem implement Problem (for LeastSquaresObjective).
struct ClosureProblemRef<'a> {
    problem: &'a dyn Problem,
}

impl Problem for ClosureProblemRef<'_> {
    fn name(&self) -> &str {
        self.problem.name()
    }
    fn residual_count(&self) -> usize {
        self.problem.residual_count()
    }
    fn variable_count(&self) -> usize {
        self.problem.variable_count()
    }
    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        self.problem.residuals(x)
    }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        self.problem.jacobian(x)
    }
    fn initial_point(&self, factor: f64) -> Vec<f64> {
        self.problem.initial_point(factor)
    }
}

// Use relaxed criteria: BFGS on least-squares is less efficient than LM,
// so we check convergence to a reasonable residual rather than exact solution.
const MAX_ITER: usize = 5000;
const TOLERANCE: f64 = 1e-8;
const RESIDUAL_TOL: f64 = 1e-3; // 0.5*||F(x)||^2 < this

macro_rules! minpack_bfgs_test {
    ($name:ident, $problem:expr) => {
        #[test]
        fn $name() {
            let problem = $problem;
            let (status, obj_val, _solution) = solve_via_bfgs(&problem, 1.0, MAX_ITER, TOLERANCE);
            assert!(
                status == OptimizationStatus::Converged || obj_val < RESIDUAL_TOL,
                "{}: status={:?}, f(x)={:.6e} (expected < {:.1e})",
                problem.name(),
                status,
                obj_val,
                RESIDUAL_TOL,
            );
        }
    };
}

minpack_bfgs_test!(minpack_rosenbrock, Rosenbrock);
// Freudenstein-Roth has a well-known local minimum at f≈24.49.
// BFGS converges there from the standard starting point — expected behavior.
#[test]
fn minpack_freudenstein_roth() {
    let problem = FreudensteinRoth;
    let (_status, obj_val, _) = solve_via_bfgs(&problem, 1.0, MAX_ITER, TOLERANCE);
    // Accept either global min (f≈0) or the known local min (f≈24.49)
    assert!(
        obj_val < 25.0,
        "FreudensteinRoth: f(x)={:.6e} (expected < 25.0, known local min ≈ 24.49)",
        obj_val,
    );
}
minpack_bfgs_test!(minpack_powell_singular, PowellSingular);
minpack_bfgs_test!(minpack_helical_valley, HelicalValley);
minpack_bfgs_test!(minpack_bard, Bard);
minpack_bfgs_test!(minpack_kowalik_osborne, KowalikOsborne);
minpack_bfgs_test!(minpack_box3d, Box3D::default());
minpack_bfgs_test!(minpack_jennrich_sampson, JennrichSampson::new(10));
minpack_bfgs_test!(minpack_brown_dennis, BrownDennis::default());
minpack_bfgs_test!(minpack_chebyquad, Chebyquad::new(6));
minpack_bfgs_test!(minpack_linear_full_rank, LinearFullRank::new(5, 10));
minpack_bfgs_test!(minpack_linear_rank1, LinearRank1::new(5, 10));
minpack_bfgs_test!(minpack_osborne1, Osborne1);

// Gradient FD verification for the LeastSquaresObjective adapter
#[test]
fn gradient_fd_verification() {
    let problem = Rosenbrock;
    let (mut store, param_ids) = setup_store(&problem, 1.0);
    let obj = LeastSquaresObjective::new(
        ClosureProblemRef { problem: &problem },
        param_ids.clone(),
        ObjectiveId::new(0, 0),
    );

    let eps = 1e-7;
    let tol = 1e-4;
    let grad = obj.gradient(&store);

    for (pid, analytical) in &grad {
        let orig = store.get(*pid);

        store.set(*pid, orig + eps);
        let f_plus = obj.value(&store);
        store.set(*pid, orig - eps);
        let f_minus = obj.value(&store);
        store.set(*pid, orig);

        let fd = (f_plus - f_minus) / (2.0 * eps);
        assert!(
            (analytical - fd).abs() < tol,
            "Gradient FD mismatch: analytical={}, fd={} (param {:?})",
            analytical,
            fd,
            pid
        );
    }
}
