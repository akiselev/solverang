//! Tests for optimization solvers (M3: BFGS, M4: ALM).

use solverang::id::{ConstraintId, EntityId, ParamId};
use solverang::optimization::{Objective, ObjectiveId, OptimizationConfig, OptimizationStatus};
use solverang::param::ParamStore;
use solverang::solver::{AlmSolver, BfgsSolver};

// Re-import Constraint trait for ALM tests
use solverang::constraint::Constraint;

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

// =========================================================================
// ALM test constraints
// =========================================================================

/// Linear equality constraint: x + y = target
struct LinearEqualityConstraint {
    id: ConstraintId,
    params: [ParamId; 2], // [px, py]
    target: f64,
}

impl LinearEqualityConstraint {
    fn new(id: ConstraintId, px: ParamId, py: ParamId, target: f64) -> Self {
        Self { id, params: [px, py], target }
    }
    fn px(&self) -> ParamId { self.params[0] }
    fn py(&self) -> ParamId { self.params[1] }
}

impl Constraint for LinearEqualityConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }
    fn name(&self) -> &str {
        "linear_equality"
    }
    fn entity_ids(&self) -> &[EntityId] {
        &[]
    }
    fn param_ids(&self) -> &[ParamId] {
        &self.params
    }
    fn equation_count(&self) -> usize {
        1
    }
    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        let x = store.get(self.px());
        let y = store.get(self.py());
        vec![x + y - self.target]
    }
    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![(0, self.px(), 1.0), (0, self.py(), 1.0)]
    }
}

// =========================================================================
// ALM tests (M4)
// =========================================================================

#[test]
fn alm_rosenbrock_with_linear_constraint() {
    // min f(x,y) = 100(y-x^2)^2 + (1-x)^2
    // s.t. x + y = 1
    // The constrained minimum is NOT at (1,1) — the constraint changes it.
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(0.0, owner);
    let py = store.alloc(0.5, owner);

    let obj = Rosenbrock::new(px, py);

    let cid = ConstraintId::new(0, 0);
    let constraint = LinearEqualityConstraint::new(cid, px, py, 1.0);
    let constraints: Vec<&dyn Constraint> = vec![&constraint];

    let mut config = OptimizationConfig::default();
    config.max_outer_iterations = 50;
    config.max_inner_iterations = 500;
    config.outer_tolerance = 1e-5;
    config.dual_tolerance = 1e-4;

    let result = AlmSolver::solve(&obj, &constraints, &mut store, &config);

    assert!(
        result.status.is_converged(),
        "ALM did not converge: status={:?}, outer_iters={}, kkt={:?}",
        result.status,
        result.outer_iterations,
        result.kkt_residual
    );

    // Check constraint satisfaction: x + y ≈ 1
    let x = store.get(px);
    let y = store.get(py);
    assert!(
        (x + y - 1.0).abs() < 1e-4,
        "Constraint violated: x+y = {} (expected 1.0)",
        x + y
    );

    // Check multiplier is present
    let multipliers = result.multipliers.lambda_for_constraint(cid);
    assert!(multipliers.is_some(), "No multiplier returned for constraint");
    let lambda = multipliers.unwrap();
    assert_eq!(lambda.len(), 1);
    // Multiplier should be non-zero (constraint is active)
    assert!(
        lambda[0].abs() > 1e-6,
        "Multiplier too small: {} (expected non-zero for active constraint)",
        lambda[0]
    );
}

#[test]
fn alm_quadratic_with_constraint() {
    // min f(x,y) = (x-3)^2 + (y-4)^2
    // s.t. x + y = 5
    // Optimal: x=2.5, y=2.5 (project (3,4) onto x+y=5 line)
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(0.0, owner);
    let py = store.alloc(0.0, owner);

    // 2D quadratic objective
    struct Quad2D {
        params: [ParamId; 2],
        targets: [f64; 2],
    }
    impl Objective for Quad2D {
        fn id(&self) -> ObjectiveId { ObjectiveId::new(0, 0) }
        fn name(&self) -> &str { "quad2d" }
        fn param_ids(&self) -> &[ParamId] { &self.params }
        fn value(&self, store: &ParamStore) -> f64 {
            let x = store.get(self.params[0]);
            let y = store.get(self.params[1]);
            (x - self.targets[0]).powi(2) + (y - self.targets[1]).powi(2)
        }
        fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
            let x = store.get(self.params[0]);
            let y = store.get(self.params[1]);
            vec![
                (self.params[0], 2.0 * (x - self.targets[0])),
                (self.params[1], 2.0 * (y - self.targets[1])),
            ]
        }
    }

    let obj = Quad2D {
        params: [px, py],
        targets: [3.0, 4.0],
    };

    let cid = ConstraintId::new(0, 0);
    let constraint = LinearEqualityConstraint::new(cid, px, py, 5.0);
    let constraints: Vec<&dyn Constraint> = vec![&constraint];

    let config = OptimizationConfig::default();
    let result = AlmSolver::solve(&obj, &constraints, &mut store, &config);

    assert!(
        result.status.is_converged(),
        "ALM did not converge: {:?}",
        result.status
    );

    let x = store.get(px);
    let y = store.get(py);

    // Analytical solution: project (3,4) onto x+y=5
    // x* = 3 - λ/2, y* = 4 - λ/2, with x+y=5: 7 - λ = 5, so λ = 2
    // x* = 2, y* = 3
    assert!(
        (x - 2.0).abs() < 1e-3,
        "x = {} (expected 2.0)",
        x
    );
    assert!(
        (y - 3.0).abs() < 1e-3,
        "y = {} (expected 3.0)",
        y
    );
    assert!(
        (x + y - 5.0).abs() < 1e-4,
        "Constraint: x+y = {} (expected 5.0)",
        x + y
    );

    // Multiplier should be ≈ 2.0 (from analytical solution)
    let lambda = result
        .multipliers
        .lambda_for_constraint(cid)
        .expect("multiplier present");
    assert!(
        (lambda[0] - 2.0).abs() < 0.5,
        "λ = {} (expected ≈ 2.0)",
        lambda[0]
    );
}

#[test]
fn alm_already_feasible() {
    // Start at a point that already satisfies the constraint
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(2.0, owner);
    let py = store.alloc(3.0, owner); // x + y = 5 already

    struct SimpleQuad { params: [ParamId; 2] }
    impl Objective for SimpleQuad {
        fn id(&self) -> ObjectiveId { ObjectiveId::new(0, 0) }
        fn name(&self) -> &str { "simple" }
        fn param_ids(&self) -> &[ParamId] { &self.params }
        fn value(&self, store: &ParamStore) -> f64 {
            let x = store.get(self.params[0]);
            let y = store.get(self.params[1]);
            (x - 2.0).powi(2) + (y - 3.0).powi(2)
        }
        fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
            let x = store.get(self.params[0]);
            let y = store.get(self.params[1]);
            vec![
                (self.params[0], 2.0 * (x - 2.0)),
                (self.params[1], 2.0 * (y - 3.0)),
            ]
        }
    }

    let obj = SimpleQuad { params: [px, py] };
    let cid = ConstraintId::new(0, 0);
    let constraint = LinearEqualityConstraint::new(cid, px, py, 5.0);
    let constraints: Vec<&dyn Constraint> = vec![&constraint];

    let config = OptimizationConfig::default();
    let result = AlmSolver::solve(&obj, &constraints, &mut store, &config);

    // Should converge quickly — already at the optimum
    assert!(result.status.is_converged());
    assert!(
        result.kkt_residual.primal < 1e-5,
        "primal violation = {}",
        result.kkt_residual.primal
    );
}
