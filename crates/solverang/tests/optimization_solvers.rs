//! Tests for optimization solvers (M3: BFGS, M4: ALM).

use solverang::id::{ConstraintId, EntityId, ParamId};
use solverang::optimization::{
    InequalityFn, Objective, ObjectiveId, OptimizationAlgorithm, OptimizationConfig,
    OptimizationStatus,
};
use solverang::param::ParamStore;
use solverang::solver::{AlmSolver, BfgsBSolver, BfgsSolver, TrustRegionSolver};

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

// =========================================================================
// Wolfe line search tests (M2)
// =========================================================================

#[test]
fn wolfe_rosenbrock_converges() {
    // Rosenbrock from the standard starting point with default (Wolfe) line search.
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(-1.2, owner);
    let py = store.alloc(1.0, owner);

    let obj = Rosenbrock::new(px, py);
    let mut config = OptimizationConfig::default();
    config.max_outer_iterations = 2000;
    config.dual_tolerance = 1e-6;

    let result = BfgsSolver::solve(&obj, &mut store, &config);

    assert_eq!(
        result.status,
        OptimizationStatus::Converged,
        "BFGS+Wolfe did not converge on Rosenbrock after {} iterations (grad_norm = {})",
        result.outer_iterations,
        result.kkt_residual.dual
    );
    let x = store.get(px);
    let y = store.get(py);
    assert!((x - 1.0).abs() < 1e-4, "x = {}, expected 1.0", x);
    assert!((y - 1.0).abs() < 1e-4, "y = {}, expected 1.0", y);
}

#[test]
fn wolfe_10d_quadratic_iteration_count() {
    // 10D separable quadratic. Verify convergence and that iteration count is
    // reasonable (Wolfe provides better curvature pairs than pure Armijo).
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let n = 10;
    let params: Vec<ParamId> = (0..n).map(|_| store.alloc(0.0, owner)).collect();

    let obj = QuadraticND { params: params.clone() };
    let config = OptimizationConfig::default();

    let result = BfgsSolver::solve(&obj, &mut store, &config);

    assert_eq!(result.status, OptimizationStatus::Converged);
    assert!(
        result.outer_iterations <= 50,
        "10D quadratic took {} iterations with Wolfe search",
        result.outer_iterations
    );
    for (i, &p) in params.iter().enumerate() {
        let x = store.get(p);
        let target = (i + 1) as f64;
        assert!((x - target).abs() < 1e-5, "x[{}] = {}, expected {}", i, x, target);
    }
}

// =========================================================================
// Dimension-independent convergence scaling tests (M3)
// =========================================================================

#[test]
fn bfgs_2d_quadratic_relative_tolerance_converges() {
    // 2D quadratic with default config (relative_tolerance = true)
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let params: Vec<ParamId> = (0..2).map(|_| store.alloc(0.0, owner)).collect();

    let obj = QuadraticND { params: params.clone() };
    let config = OptimizationConfig::default();
    assert!(config.relative_tolerance, "default should be relative");

    let result = BfgsSolver::solve(&obj, &mut store, &config);

    assert_eq!(result.status, OptimizationStatus::Converged);
    for (i, &p) in params.iter().enumerate() {
        let x = store.get(p);
        let target = (i + 1) as f64;
        assert!((x - target).abs() < 1e-5, "x[{}] = {}, expected {}", i, x, target);
    }
}

#[test]
fn bfgs_absolute_tolerance_backward_compat() {
    // Setting relative_tolerance = false reproduces exact previous behavior
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(0.0, owner);

    let obj = Quadratic1D { param: px };
    let mut config = OptimizationConfig::default();
    config.relative_tolerance = false;

    let result = BfgsSolver::solve(&obj, &mut store, &config);

    assert_eq!(result.status, OptimizationStatus::Converged);
    let x = store.get(px);
    assert!((x - 3.0).abs() < 1e-6, "x = {}, expected 3.0", x);
    assert!(result.objective_value < 1e-12);
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

    let result = AlmSolver::solve(&obj, &constraints, &[], &mut store, &config);

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
    let result = AlmSolver::solve(&obj, &constraints, &[], &mut store, &config);

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
    let result = AlmSolver::solve(&obj, &constraints, &[], &mut store, &config);

    // Should converge quickly — already at the optimum
    assert!(result.status.is_converged());
    assert!(
        result.kkt_residual.primal < 1e-5,
        "primal violation = {}",
        result.kkt_residual.primal
    );
}

// =========================================================================
// L-BFGS-B tests (M4)
// =========================================================================

#[test]
fn bfgs_b_bounded_quadratic() {
    // min (x-3)^2 + (y-1)^2, x in [0, 2], y in (-inf, +inf)
    // Unconstrained minimum (3, 1); x hits upper bound.
    // Constrained solution: (2, 1).
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(0.0, owner);
    let py = store.alloc(0.0, owner);

    // Set bound on x only.
    store.set_bounds(px, 0.0, 2.0);

    struct BoundedQuad {
        params: [ParamId; 2],
    }
    impl Objective for BoundedQuad {
        fn id(&self) -> ObjectiveId { ObjectiveId::new(0, 0) }
        fn name(&self) -> &str { "bounded_quad" }
        fn param_ids(&self) -> &[ParamId] { &self.params }
        fn value(&self, store: &ParamStore) -> f64 {
            let x = store.get(self.params[0]);
            let y = store.get(self.params[1]);
            (x - 3.0).powi(2) + (y - 1.0).powi(2)
        }
        fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
            let x = store.get(self.params[0]);
            let y = store.get(self.params[1]);
            vec![
                (self.params[0], 2.0 * (x - 3.0)),
                (self.params[1], 2.0 * (y - 1.0)),
            ]
        }
    }

    let obj = BoundedQuad { params: [px, py] };
    let config = OptimizationConfig::default();

    let result = BfgsBSolver::solve(&obj, &mut store, &config);

    assert_eq!(
        result.status,
        OptimizationStatus::Converged,
        "BfgsB did not converge: {:?}",
        result.status
    );
    let x = store.get(px);
    let y = store.get(py);
    // x is clamped at upper bound 2.0
    assert!(
        (x - 2.0).abs() < 1e-5,
        "x = {} (expected 2.0, upper bound active)",
        x
    );
    // y is unconstrained, should reach 1.0
    assert!((y - 1.0).abs() < 1e-5, "y = {} (expected 1.0)", y);
    // Bounds must be respected
    assert!(x <= 2.0 + 1e-10, "x = {} violates upper bound", x);
    assert!(x >= 0.0 - 1e-10, "x = {} violates lower bound", x);
}

#[test]
fn bfgs_b_unconstrained_matches_bfgs() {
    // Same Rosenbrock problem with infinite bounds — should converge to (1, 1).
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(-1.2, owner);
    let py = store.alloc(1.0, owner);

    // Bounds are implicitly infinite (no set_bounds call); this tests that
    // BfgsB with no active bounds converges identically to unconstrained BFGS.

    let obj = Rosenbrock::new(px, py);
    let mut config = OptimizationConfig::default();
    config.max_outer_iterations = 2000;
    config.dual_tolerance = 1e-6;

    let result = BfgsBSolver::solve(&obj, &mut store, &config);

    assert_eq!(
        result.status,
        OptimizationStatus::Converged,
        "BfgsB+infinite bounds did not converge on Rosenbrock after {} iterations (pg_norm = {})",
        result.outer_iterations,
        result.kkt_residual.dual
    );
    let x = store.get(px);
    let y = store.get(py);
    assert!((x - 1.0).abs() < 1e-4, "x = {} (expected 1.0)", x);
    assert!((y - 1.0).abs() < 1e-4, "y = {} (expected 1.0)", y);
}

#[test]
fn bfgs_b_all_bounds_active() {
    // min (x - 5)^2 with x in [0, 2].
    // Unconstrained minimum at x = 5; feasible solution: x = 2 (upper bound).
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(0.5, owner);
    store.set_bounds(px, 0.0, 2.0);

    let obj = Quadratic1D { param: px };

    // Redefine objective targeting 5.0 instead of 3.0.
    struct QuadTarget5 { param: ParamId }
    impl Objective for QuadTarget5 {
        fn id(&self) -> ObjectiveId { ObjectiveId::new(0, 0) }
        fn name(&self) -> &str { "quad_target5" }
        fn param_ids(&self) -> &[ParamId] { std::slice::from_ref(&self.param) }
        fn value(&self, store: &ParamStore) -> f64 {
            let x = store.get(self.param);
            (x - 5.0).powi(2)
        }
        fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
            let x = store.get(self.param);
            vec![(self.param, 2.0 * (x - 5.0))]
        }
    }
    let _ = obj; // suppress unused warning

    let obj5 = QuadTarget5 { param: px };
    let config = OptimizationConfig::default();

    let result = BfgsBSolver::solve(&obj5, &mut store, &config);

    assert_eq!(
        result.status,
        OptimizationStatus::Converged,
        "BfgsB did not converge: {:?}",
        result.status
    );
    let x = store.get(px);
    assert!(
        (x - 2.0).abs() < 1e-5,
        "x = {} (expected 2.0, upper bound active)",
        x
    );
    assert!(x <= 2.0 + 1e-10, "x violates upper bound");
}

// =========================================================================
// ALM inequality constraint tests (M5)
// =========================================================================

/// Linear inequality constraint: a*x + b*y <= rhs, i.e. h(x,y) = a*x + b*y - rhs <= 0
struct LinearInequalityConstraint {
    id: ConstraintId,
    params: [ParamId; 2],
    a: f64,
    b: f64,
    rhs: f64,
}

impl LinearInequalityConstraint {
    fn new(id: ConstraintId, px: ParamId, py: ParamId, a: f64, b: f64, rhs: f64) -> Self {
        Self { id, params: [px, py], a, b, rhs }
    }
    fn px(&self) -> ParamId { self.params[0] }
    fn py(&self) -> ParamId { self.params[1] }
}

impl InequalityFn for LinearInequalityConstraint {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "linear_inequality" }
    fn entity_ids(&self) -> &[EntityId] { &[] }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn inequality_count(&self) -> usize { 1 }
    fn values(&self, store: &ParamStore) -> Vec<f64> {
        let x = store.get(self.px());
        let y = store.get(self.py());
        vec![self.a * x + self.b * y - self.rhs]
    }
    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![(0, self.px(), self.a), (0, self.py(), self.b)]
    }
}

/// 2D quadratic objective: (x - tx)^2 + (y - ty)^2
struct Quad2DTarget {
    params: [ParamId; 2],
    target: [f64; 2],
}

impl Objective for Quad2DTarget {
    fn id(&self) -> ObjectiveId { ObjectiveId::new(0, 0) }
    fn name(&self) -> &str { "quad2d_target" }
    fn param_ids(&self) -> &[ParamId] { &self.params }
    fn value(&self, store: &ParamStore) -> f64 {
        let x = store.get(self.params[0]);
        let y = store.get(self.params[1]);
        (x - self.target[0]).powi(2) + (y - self.target[1]).powi(2)
    }
    fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
        let x = store.get(self.params[0]);
        let y = store.get(self.params[1]);
        vec![
            (self.params[0], 2.0 * (x - self.target[0])),
            (self.params[1], 2.0 * (y - self.target[1])),
        ]
    }
}

#[test]
fn alm_inequality_inactive() {
    // min (x-2)^2 + (y-1)^2 s.t. x + y <= 5
    // Unconstrained minimum (2, 1) satisfies x+y=3 <= 5, so constraint is inactive.
    // Solution: (2, 1).
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(0.0, owner);
    let py = store.alloc(0.0, owner);

    let obj = Quad2DTarget { params: [px, py], target: [2.0, 1.0] };

    let hid = ConstraintId::new(1, 0);
    let ineq = LinearInequalityConstraint::new(hid, px, py, 1.0, 1.0, 5.0);
    let inequalities: Vec<&dyn InequalityFn> = vec![&ineq];

    let mut config = OptimizationConfig::default();
    config.max_outer_iterations = 50;
    config.max_inner_iterations = 500;

    let result = AlmSolver::solve(&obj, &[], &inequalities, &mut store, &config);

    assert!(
        result.status.is_converged(),
        "ALM (inactive ineq) did not converge: {:?}, kkt={:?}",
        result.status,
        result.kkt_residual
    );

    let x = store.get(px);
    let y = store.get(py);
    assert!((x - 2.0).abs() < 1e-3, "x = {} (expected 2.0)", x);
    assert!((y - 1.0).abs() < 1e-3, "y = {} (expected 1.0)", y);

    // Constraint satisfied
    assert!(x + y <= 5.0 + 1e-4, "Constraint violated: x+y = {}", x + y);

    // Multiplier should be ~0 for inactive constraint
    let mu = result.multipliers.lambda_for_constraint(hid);
    if let Some(mu_vals) = mu {
        assert!(
            mu_vals[0] >= -1e-6,
            "Inequality multiplier must be >= 0, got {}",
            mu_vals[0]
        );
    }
}

#[test]
fn alm_inequality_active() {
    // min (x-5)^2 + (y-5)^2 s.t. x + y <= 3
    // Unconstrained minimum (5, 5) violates x+y<=3.
    // Constrained solution: (1.5, 1.5) — constraint active.
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(0.0, owner);
    let py = store.alloc(0.0, owner);

    let obj = Quad2DTarget { params: [px, py], target: [5.0, 5.0] };

    let hid = ConstraintId::new(1, 0);
    let ineq = LinearInequalityConstraint::new(hid, px, py, 1.0, 1.0, 3.0);
    let inequalities: Vec<&dyn InequalityFn> = vec![&ineq];

    let mut config = OptimizationConfig::default();
    config.max_outer_iterations = 100;
    config.max_inner_iterations = 500;
    config.outer_tolerance = 1e-4;
    config.dual_tolerance = 1e-3;

    let result = AlmSolver::solve(&obj, &[], &inequalities, &mut store, &config);

    assert!(
        result.status.is_converged(),
        "ALM (active ineq) did not converge: {:?}, kkt={:?}",
        result.status,
        result.kkt_residual
    );

    let x = store.get(px);
    let y = store.get(py);
    assert!((x - 1.5).abs() < 1e-2, "x = {} (expected 1.5)", x);
    assert!((y - 1.5).abs() < 1e-2, "y = {} (expected 1.5)", y);

    // Constraint satisfied (active at boundary)
    assert!(x + y <= 3.0 + 1e-3, "Constraint violated: x+y = {}", x + y);

    // Multiplier should be positive (active constraint)
    let mu = result.multipliers.lambda_for_constraint(hid);
    assert!(mu.is_some(), "No multiplier returned for active inequality");
    let mu_vals = mu.unwrap();
    assert!(
        mu_vals[0] >= -1e-6,
        "Inequality multiplier must be >= 0, got {}",
        mu_vals[0]
    );
    assert!(
        mu_vals[0] > 1e-4,
        "Active constraint multiplier should be > 0, got {}",
        mu_vals[0]
    );
}

#[test]
fn alm_mixed_equality_inequality() {
    // min x^2 + y^2 s.t. x + y = 1, -x <= 0 (i.e. x >= 0)
    // Equality: x + y = 1. Inequality: h(x) = -x <= 0.
    // Solution: (0.5, 0.5) — equality active, inequality inactive (x=0.5 > 0).
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(0.5, owner);
    let py = store.alloc(0.5, owner);

    struct NormSquared { params: [ParamId; 2] }
    impl Objective for NormSquared {
        fn id(&self) -> ObjectiveId { ObjectiveId::new(0, 0) }
        fn name(&self) -> &str { "norm_squared" }
        fn param_ids(&self) -> &[ParamId] { &self.params }
        fn value(&self, store: &ParamStore) -> f64 {
            let x = store.get(self.params[0]);
            let y = store.get(self.params[1]);
            x * x + y * y
        }
        fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
            let x = store.get(self.params[0]);
            let y = store.get(self.params[1]);
            vec![(self.params[0], 2.0 * x), (self.params[1], 2.0 * y)]
        }
    }

    let obj = NormSquared { params: [px, py] };

    let cid = ConstraintId::new(0, 0);
    let eq_constraint = LinearEqualityConstraint::new(cid, px, py, 1.0);
    let constraints: Vec<&dyn Constraint> = vec![&eq_constraint];

    // h(x, y) = -x <= 0  (enforces x >= 0)
    struct NonNegativityConstraint { id: ConstraintId, param: ParamId }
    impl InequalityFn for NonNegativityConstraint {
        fn id(&self) -> ConstraintId { self.id }
        fn name(&self) -> &str { "non_negativity" }
        fn entity_ids(&self) -> &[EntityId] { &[] }
        fn param_ids(&self) -> &[ParamId] { std::slice::from_ref(&self.param) }
        fn inequality_count(&self) -> usize { 1 }
        fn values(&self, store: &ParamStore) -> Vec<f64> {
            vec![-store.get(self.param)]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.param, -1.0)]
        }
    }

    let hid = ConstraintId::new(2, 0);
    let ineq = NonNegativityConstraint { id: hid, param: px };
    let inequalities: Vec<&dyn InequalityFn> = vec![&ineq];

    let mut config = OptimizationConfig::default();
    config.max_outer_iterations = 50;
    config.max_inner_iterations = 500;
    config.outer_tolerance = 1e-4;
    config.dual_tolerance = 1e-4;

    let result = AlmSolver::solve(&obj, &constraints, &inequalities, &mut store, &config);

    assert!(
        result.status.is_converged(),
        "ALM (mixed eq+ineq) did not converge: {:?}, kkt={:?}",
        result.status,
        result.kkt_residual
    );

    let x = store.get(px);
    let y = store.get(py);

    // Solution: (0.5, 0.5)
    assert!((x - 0.5).abs() < 1e-3, "x = {} (expected 0.5)", x);
    assert!((y - 0.5).abs() < 1e-3, "y = {} (expected 0.5)", y);

    // Equality constraint satisfied
    assert!((x + y - 1.0).abs() < 1e-4, "Equality violated: x+y = {}", x + y);

    // Inequality satisfied: x >= 0
    assert!(x >= -1e-5, "Inequality violated: x = {} (expected >= 0)", x);

    // Inequality multiplier must be >= 0
    let mu = result.multipliers.lambda_for_constraint(hid);
    if let Some(mu_vals) = mu {
        assert!(
            mu_vals[0] >= -1e-6,
            "Inequality multiplier must be >= 0, got {}",
            mu_vals[0]
        );
    }
}

// =========================================================================
// Trust-region tests (M7)
// =========================================================================

#[test]
fn trust_region_rosenbrock_converges() {
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(-1.2, owner);
    let py = store.alloc(1.0, owner);

    let obj = Rosenbrock::new(px, py);
    let mut config = OptimizationConfig::default();
    config.algorithm = OptimizationAlgorithm::TrustRegion;
    config.max_outer_iterations = 2000;
    config.dual_tolerance = 1e-6;

    let result = TrustRegionSolver::solve(&obj, &mut store, &config);

    assert_eq!(
        result.status,
        OptimizationStatus::Converged,
        "TrustRegion did not converge on Rosenbrock after {} iterations (grad_norm = {})",
        result.outer_iterations,
        result.kkt_residual.dual
    );
    let x = store.get(px);
    let y = store.get(py);
    assert!((x - 1.0).abs() < 1e-4, "x = {}, expected 1.0", x);
    assert!((y - 1.0).abs() < 1e-4, "y = {}, expected 1.0", y);
}

#[test]
fn trust_region_10d_quadratic() {
    // 10D separable quadratic — uses dogleg (n=10 < threshold=100).
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let n = 10;
    let params: Vec<ParamId> = (0..n).map(|_| store.alloc(0.0, owner)).collect();

    let obj = QuadraticND { params: params.clone() };
    let mut config = OptimizationConfig::default();
    config.algorithm = OptimizationAlgorithm::TrustRegion;

    let result = TrustRegionSolver::solve(&obj, &mut store, &config);

    assert_eq!(
        result.status,
        OptimizationStatus::Converged,
        "TrustRegion 10D quadratic did not converge after {} iterations",
        result.outer_iterations
    );
    for (i, &p) in params.iter().enumerate() {
        let x = store.get(p);
        let target = (i + 1) as f64;
        assert!((x - target).abs() < 1e-5, "x[{}] = {}, expected {}", i, x, target);
    }
}

#[test]
fn bfgs_50d_quadratic_relative_tolerance_converges() {
    // 50D separable quadratic with default config (relative_tolerance = true)
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let n = 50;
    let params: Vec<ParamId> = (0..n).map(|_| store.alloc(0.0, owner)).collect();

    let obj = QuadraticND { params: params.clone() };
    let config = OptimizationConfig::default();
    assert!(config.relative_tolerance, "default should be relative");

    let result = BfgsSolver::solve(&obj, &mut store, &config);

    assert_eq!(
        result.status,
        OptimizationStatus::Converged,
        "BFGS 50D quadratic did not converge after {} iterations",
        result.outer_iterations
    );
    for (i, &p) in params.iter().enumerate() {
        let x = store.get(p);
        let target = (i + 1) as f64;
        assert!((x - target).abs() < 1e-4, "x[{}] = {}, expected {}", i, x, target);
    }
}

#[test]
fn trust_region_200d_steihaug_cg_path() {
    // 200D separable quadratic — n=200 exceeds tr_subproblem_threshold=100,
    // forcing the Steihaug-CG subproblem solver instead of dogleg.
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let n = 200;
    let params: Vec<ParamId> = (0..n).map(|_| store.alloc(0.0, owner)).collect();

    let obj = QuadraticND { params: params.clone() };
    let mut config = OptimizationConfig::default();
    config.algorithm = OptimizationAlgorithm::TrustRegion;

    let result = TrustRegionSolver::solve(&obj, &mut store, &config);

    assert_eq!(
        result.status,
        OptimizationStatus::Converged,
        "TrustRegion 200D (Steihaug-CG) did not converge after {} iterations",
        result.outer_iterations
    );
    for (i, &p) in params.iter().enumerate() {
        let x = store.get(p);
        let target = (i + 1) as f64;
        assert!((x - target).abs() < 1e-4, "x[{}] = {}, expected {}", i, x, target);
    }
}

#[test]
fn trust_region_beale_converges() {
    // Beale's function: f(x,y) = (1.5 - x + x*y)^2 + (2.25 - x + x*y^2)^2 + (2.625 - x + x*y^3)^2
    // Global minimum at (3.0, 0.5) with f = 0.
    struct Beale {
        params: [ParamId; 2],
    }
    impl Objective for Beale {
        fn id(&self) -> ObjectiveId { ObjectiveId::new(0, 0) }
        fn name(&self) -> &str { "beale" }
        fn param_ids(&self) -> &[ParamId] { &self.params }
        fn value(&self, store: &ParamStore) -> f64 {
            let x = store.get(self.params[0]);
            let y = store.get(self.params[1]);
            let a = 1.5 - x + x * y;
            let b = 2.25 - x + x * y * y;
            let c = 2.625 - x + x * y * y * y;
            a * a + b * b + c * c
        }
        fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)> {
            let x = store.get(self.params[0]);
            let y = store.get(self.params[1]);
            let a = 1.5 - x + x * y;
            let b = 2.25 - x + x * y * y;
            let c = 2.625 - x + x * y * y * y;
            let da_dx = -1.0 + y;
            let db_dx = -1.0 + y * y;
            let dc_dx = -1.0 + y * y * y;
            let da_dy = x;
            let db_dy = 2.0 * x * y;
            let dc_dy = 3.0 * x * y * y;
            vec![
                (self.params[0], 2.0 * (a * da_dx + b * db_dx + c * dc_dx)),
                (self.params[1], 2.0 * (a * da_dy + b * db_dy + c * dc_dy)),
            ]
        }
    }

    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(-1.0, owner);
    let py = store.alloc(0.0, owner);

    let obj = Beale { params: [px, py] };
    let mut config = OptimizationConfig::default();
    config.algorithm = OptimizationAlgorithm::TrustRegion;
    config.max_outer_iterations = 2000;
    config.dual_tolerance = 1e-6;

    let result = TrustRegionSolver::solve(&obj, &mut store, &config);

    assert_eq!(
        result.status,
        OptimizationStatus::Converged,
        "TrustRegion did not converge on Beale's function after {} iterations (grad_norm = {})",
        result.outer_iterations,
        result.kkt_residual.dual
    );
    let x = store.get(px);
    let y = store.get(py);
    assert!((x - 3.0).abs() < 1e-3, "x = {} (expected 3.0)", x);
    assert!((y - 0.5).abs() < 1e-3, "y = {} (expected 0.5)", y);
}

#[test]
fn trust_region_already_at_minimum() {
    // Start at the minimum — should converge at iteration 0.
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let px = store.alloc(3.0, owner); // minimum of (x - 3)^2

    let obj = Quadratic1D { param: px };
    let mut config = OptimizationConfig::default();
    config.algorithm = OptimizationAlgorithm::TrustRegion;

    let result = TrustRegionSolver::solve(&obj, &mut store, &config);

    assert_eq!(result.status, OptimizationStatus::Converged);
    assert_eq!(result.outer_iterations, 0, "should detect convergence immediately");
    let x = store.get(px);
    assert!((x - 3.0).abs() < 1e-10, "x = {}, expected 3.0", x);
}
