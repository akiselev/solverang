//! Tests for optimization solvers (M3: BFGS, M4: ALM).

use solverang::optimization::{Objective, ObjectiveId, OptimizationConfig, OptimizationStatus};
use solverang::param::ParamStore;
use solverang::solver::BfgsSolver;
use solverang::id::{EntityId, ParamId};

// =========================================================================
// Test objectives
// =========================================================================

/// f(x) = (x - 3)^2, minimum at x = 3
struct Quadratic1D {
    param: ParamId,
}

impl Objective for Quadratic1D {
    fn id(&self) -> ObjectiveId { ObjectiveId::new(0, 0) }
    fn name(&self) -> &str { "quadratic_1d" }
    fn param_ids(&self) -> &[ParamId] { std::slice::from_ref(&self.param) }
    fn value(&self, store: &ParamStore) -> f64 {
        let x = store.get(self.param);
        (x - 3.0).powi(2)
    }
    fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
        let x = store.get(self.param);
        vec![(self.param, 2.0 * (x - 3.0))]
    }
}

/// Rosenbrock: f(x,y) = 100(y - x^2)^2 + (1 - x)^2, minimum at (1, 1)
struct Rosenbrock {
    params: [ParamId; 2], // [px, py]
}

impl Rosenbrock {
    fn new(px: ParamId, py: ParamId) -> Self {
        Self { params: [px, py] }
    }
    fn px(&self) -> ParamId { self.params[0] }
    fn py(&self) -> ParamId { self.params[1] }
}

impl Objective for Rosenbrock {
    fn id(&self) -> ObjectiveId { ObjectiveId::new(0, 0) }
    fn name(&self) -> &str { "rosenbrock" }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn value(&self, store: &ParamStore) -> f64 {
        let x = store.get(self.px());
        let y = store.get(self.py());
        100.0 * (y - x * x).powi(2) + (1.0 - x).powi(2)
    }
    fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
        let x = store.get(self.px());
        let y = store.get(self.py());
        vec![
            (self.px(), -400.0 * x * (y - x * x) - 2.0 * (1.0 - x)),
            (self.py(), 200.0 * (y - x * x)),
        ]
    }
}

/// N-dimensional quadratic: f(x) = sum_i (x_i - (i+1))^2
struct QuadraticND {
    params: Vec<ParamId>,
}

impl Objective for QuadraticND {
    fn id(&self) -> ObjectiveId { ObjectiveId::new(0, 0) }
    fn name(&self) -> &str { "quadratic_nd" }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn value(&self, store: &ParamStore) -> f64 {
        self.params
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let x = store.get(p);
                let target = (i + 1) as f64;
                (x - target).powi(2)
            })
            .sum()
    }
    fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
        self.params
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let x = store.get(p);
                let target = (i + 1) as f64;
                (p, 2.0 * (x - target))
            })
            .collect()
    }
}

// =========================================================================
// BFGS tests (M3)
// =========================================================================

#[test]
fn bfgs_1d_quadratic_converges() {
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(0.0, owner); // start at 0

    let obj = Quadratic1D { param: px };
    let config = OptimizationConfig::default();

    let result = BfgsSolver::solve(&obj, &mut store, &config);

    assert_eq!(result.status, OptimizationStatus::Converged);
    assert!(result.outer_iterations < 10, "took {} iterations", result.outer_iterations);
    let x = store.get(px);
    assert!((x - 3.0).abs() < 1e-6, "x = {}, expected 3.0", x);
    assert!(result.objective_value < 1e-12);
}

#[test]
fn bfgs_rosenbrock_converges() {
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(-1.2, owner);
    let py = store.alloc(1.0, owner);

    let obj = Rosenbrock::new(px, py);
    let mut config = OptimizationConfig::default();
    config.max_outer_iterations = 2000; // Rosenbrock needs more iterations
    config.dual_tolerance = 1e-6;

    let result = BfgsSolver::solve(&obj, &mut store, &config);

    assert_eq!(result.status, OptimizationStatus::Converged,
        "BFGS did not converge on Rosenbrock after {} iterations (grad_norm = {})",
        result.outer_iterations, result.kkt_residual.dual);
    let x = store.get(px);
    let y = store.get(py);
    assert!((x - 1.0).abs() < 1e-4, "x = {}, expected 1.0", x);
    assert!((y - 1.0).abs() < 1e-4, "y = {}, expected 1.0", y);
}

#[test]
fn bfgs_nd_quadratic_converges() {
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let n = 10;
    let params: Vec<ParamId> = (0..n).map(|_| store.alloc(0.0, owner)).collect();

    let obj = QuadraticND { params: params.clone() };
    let config = OptimizationConfig::default();

    let result = BfgsSolver::solve(&obj, &mut store, &config);

    assert_eq!(result.status, OptimizationStatus::Converged);
    for (i, &p) in params.iter().enumerate() {
        let x = store.get(p);
        let target = (i + 1) as f64;
        assert!((x - target).abs() < 1e-5, "x[{}] = {}, expected {}", i, x, target);
    }
}

#[test]
fn bfgs_already_at_minimum() {
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(3.0, owner); // already at minimum

    let obj = Quadratic1D { param: px };
    let config = OptimizationConfig::default();

    let result = BfgsSolver::solve(&obj, &mut store, &config);

    assert_eq!(result.status, OptimizationStatus::Converged);
    assert_eq!(result.outer_iterations, 0); // should detect immediately
}

#[test]
fn bfgs_zero_variables() {
    // All params fixed → 0 free variables
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(5.0, owner);
    store.fix(px);

    let obj = Quadratic1D { param: px };
    let config = OptimizationConfig::default();

    let result = BfgsSolver::solve(&obj, &mut store, &config);

    assert_eq!(result.status, OptimizationStatus::Converged);
    assert_eq!(result.outer_iterations, 0);
    // Value should be (5 - 3)^2 = 4.0 since param is fixed
    assert!((result.objective_value - 4.0).abs() < 1e-12);
}
