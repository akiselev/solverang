# 04 — User-Facing Optimization API and TDD Roadmap

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Core Traits](#2-core-traits)
3. [OptimizationBuilder API](#3-optimizationbuilder-api)
4. [Code Examples](#4-code-examples)
5. [Error Handling Design](#5-error-handling-design)
6. [TDD Roadmap](#6-tdd-roadmap)
7. [Property Tests](#7-property-tests)
8. [Integration Test Strategy](#8-integration-test-strategy)
9. [Extending the MINPACK Suite](#9-extending-the-minpack-suite)

---

## 1. Design Philosophy

The optimization API follows four principles derived from the existing Solverang
patterns:

1. **Trait-first**: Just as `Problem` defines residuals/Jacobian and `Constraint`
   defines residuals/Jacobian-over-`ParamStore`, the new `Objective` trait defines
   a scalar objective, its gradient, and (optionally) its Hessian.

2. **Builder ergonomics**: `OptimizationBuilder` mirrors `Sketch2DBuilder` — it
   allocates parameters, wires entities, and produces a solvable
   `OptimizationProblem` via `.build()`.

3. **Composition over conversion**: Existing `Constraint`s become equality
   constraints directly; no adapter layer is needed for the common case.

4. **Separate problem definition from solver selection**: The
   `OptimizationProblem` struct is solver-agnostic. It can be handed to an
   interior-point solver, an SQP solver, or a penalty-method wrapper around the
   existing LM solver.

---

## 2. Core Traits

### 2.1 The `Objective` Trait

<!-- Decision: Two-level design. The Objective trait defined below is the problem-level (array-based, ObjectiveFunction) interface for standalone use in OptimizationBuilder. The system-level Objective trait (in 01_mathematical_architecture.md) uses ParamStore-based interface with id() and sparse gradient. Adapters convert between levels. Both levels exist and are intentionally distinct. -->

```rust
// crates/solverang/src/optimization/objective.rs

/// A scalar objective function f(x) to be minimized.
///
/// This is the problem-level (builder-facing) interface, analogous to the
/// `Problem` trait for root-finding. The system-level ParamStore-based
/// interface (used by `LagrangianAssembler`) is defined in
/// `01_mathematical_architecture.md` and is a separate trait named `Objective`.
/// This problem-level trait should be named `ObjectiveFunction` to distinguish
/// it from the system-level `Objective`.
///
/// Implementors provide the function value and its gradient.
/// The Hessian is optional — solvers fall back to BFGS approximation if
/// `hessian()` returns `None`. BFGS is the default solver.
pub trait Objective: Send + Sync {
    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;

    /// Number of variables this objective depends on.
    fn variable_count(&self) -> usize;

    /// Evaluate f(x).
    fn value(&self, x: &[f64]) -> f64;

    /// Gradient ∇f(x) as a dense vector of length `variable_count()`.
    fn gradient(&self, x: &[f64]) -> Vec<f64>;

    /// Hessian ∇²f(x) as sparse lower-triangular triplets (row, col, value)
    /// where `row >= col`.
    ///
    /// Returns `None` if the Hessian is not analytically available.
    /// Solvers will use BFGS or finite-difference approximation instead.
    fn hessian(&self, x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
        let _ = x;
        None
    }
}
```

### 2.2 The `InequalityConstraint` Trait

Equality constraints are reused directly from the existing `Constraint` trait
(residual = 0). Inequality constraints need a new trait:

<!-- Decision: The system-level inequality trait is named InequalityFn (not InequalityConstraint or SystemInequalityConstraint). The trait below is the problem-level (array-based) inequality interface for OptimizationBuilder. It keeps the name InequalityConstraint at the problem level to match the existing array-based InequalityConstraint in constraints/inequality.rs. Method names: inequality_count() and values() (matching doc 01). -->

```rust
// crates/solverang/src/optimization/inequality.rs

/// An inequality constraint g(x) <= 0.
///
/// Note: this trait is named `InequalityFn` in `01_mathematical_architecture.md`
/// to avoid collision with the existing `InequalityConstraint` in
/// `constraints/inequality.rs`. Use `InequalityFn` as the canonical name.
///
/// The solver treats this as satisfied when every component of
/// `values(x)` is non-positive.
pub trait InequalityConstraint: Send + Sync {
    /// Human-readable name.
    fn name(&self) -> &str;

    /// Which parameters this constraint depends on.
    fn param_ids(&self) -> &[ParamId];

    /// Number of scalar inequality equations.
    fn inequality_count(&self) -> usize;

    /// Evaluate g(x). Each element should be <= 0 when satisfied.
    fn values(&self, store: &ParamStore) -> Vec<f64>;

    /// Sparse Jacobian: (equation_row, param_id, partial_derivative).
    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)>;
}
```

### 2.3 `OptimizationProblem` — The Assembled Problem

```rust
// crates/solverang/src/optimization/problem.rs

/// A fully assembled optimization problem ready for solving.
///
/// Produced by `OptimizationBuilder::build()`.
pub struct OptimizationProblem {
    /// The scalar objective function.
    pub objective: Box<dyn Objective>,
    /// Equality constraints h(x) = 0 (reusing existing `Constraint` trait).
    pub equality_constraints: Vec<Box<dyn Constraint>>,
    /// Inequality constraints g(x) <= 0.
    pub inequality_constraints: Vec<Box<dyn InequalityConstraint>>,
    /// Parameter store with initial values and fixed/free status.
    pub params: ParamStore,
    /// Variable bounds (lower, upper) indexed by ParamId.
    pub bounds: Vec<(ParamId, f64, f64)>,
}
```

### 2.4 `OptimizationResult`

```rust
// crates/solverang/src/optimization/result.rs

use thiserror::Error;

/// Result of solving an optimization problem.
#[derive(Clone, Debug)]
pub enum OptimizationResult {
    /// Solver converged to a local minimum.
    Converged {
        /// Optimal variable values.
        solution: Vec<f64>,
        /// Optimal objective value f(x*).
        objective_value: f64,
        /// Number of iterations.
        iterations: usize,
        /// Final gradient norm ||∇f(x*)||.
        gradient_norm: f64,
        /// Maximum constraint violation max(0, g_i(x*)) or |h_i(x*)|.
        constraint_violation: f64,
    },

    /// Solver did not converge.
    NotConverged {
        solution: Vec<f64>,
        objective_value: f64,
        iterations: usize,
        gradient_norm: f64,
        constraint_violation: f64,
    },

    /// Solver detected an infeasible problem.
    Infeasible {
        /// Best point found.
        solution: Vec<f64>,
        /// Minimum constraint violation achieved.
        constraint_violation: f64,
    },

    /// Solver failed due to a fatal error.
    Failed {
        error: OptimizationError,
    },
}

impl OptimizationResult {
    pub fn is_converged(&self) -> bool {
        matches!(self, OptimizationResult::Converged { .. })
    }

    pub fn solution(&self) -> Option<&[f64]> {
        match self {
            OptimizationResult::Converged { solution, .. }
            | OptimizationResult::NotConverged { solution, .. }
            | OptimizationResult::Infeasible { solution, .. } => Some(solution),
            OptimizationResult::Failed { .. } => None,
        }
    }

    pub fn objective_value(&self) -> Option<f64> {
        match self {
            OptimizationResult::Converged { objective_value, .. }
            | OptimizationResult::NotConverged { objective_value, .. } => {
                Some(*objective_value)
            }
            _ => None,
        }
    }

    /// Maximum constraint violation at the returned solution point.
    ///
    /// Returns `Some` for `Converged`, `NotConverged`, and `Infeasible`.
    /// Returns `None` for `Failed`.
    pub fn constraint_violation(&self) -> Option<f64> {
        match self {
            OptimizationResult::Converged { constraint_violation, .. }
            | OptimizationResult::NotConverged { constraint_violation, .. }
            | OptimizationResult::Infeasible { constraint_violation, .. } => {
                Some(*constraint_violation)
            }
            OptimizationResult::Failed { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum OptimizationError {
    #[error("no objective function defined")]
    NoObjective,

    #[error("no variables to optimize (all fixed)")]
    NoVariables,

    #[error("gradient contains non-finite values")]
    NonFiniteGradient,

    #[error("Hessian contains non-finite values")]
    NonFiniteHessian,

    #[error("line search failed to find acceptable step")]
    LineSearchFailed,

    #[error("maximum iterations ({0}) exceeded")]
    MaxIterationsExceeded(usize),

    #[error("objective function returned NaN or infinity")]
    NonFiniteObjective,

    #[error(
        "dimension mismatch: gradient has {got} elements, expected {expected}"
    )]
    GradientDimensionMismatch { expected: usize, got: usize },

    #[error("bound violation: lower bound {lower} > upper bound {upper} for parameter {param_index}")]
    InvalidBounds {
        param_index: usize,
        lower: f64,
        upper: f64,
    },
}
```

---

## 3. OptimizationBuilder API

```rust
// crates/solverang/src/optimization/builder.rs

use std::collections::HashMap;
use crate::constraint::Constraint;
use crate::id::{EntityId, ParamId};
use crate::optimization::{
    InequalityConstraint, Objective, OptimizationProblem, OptimizationResult,
};
use crate::param::ParamStore;
use crate::sketch2d::builder::Sketch2DBuilder;
use crate::system::ConstraintSystem;

/// Configuration for the optimization solver.
#[derive(Clone, Debug)]
pub struct OptimizationConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance for gradient norm.
    pub gradient_tolerance: f64,
    /// Convergence tolerance for constraint violation.
    pub constraint_tolerance: f64,
    /// Convergence tolerance for objective function change.
    pub objective_tolerance: f64,
    /// Penalty parameter for constraint violations (penalty method).
    pub initial_penalty: f64,
    /// Penalty growth factor per outer iteration.
    pub penalty_growth: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            gradient_tolerance: 1e-8,
            constraint_tolerance: 1e-8,
            objective_tolerance: 1e-12,
            initial_penalty: 1.0,
            penalty_growth: 10.0,
        }
    }
}

/// Builder for optimization problems.
///
/// Follows the same ergonomic pattern as `Sketch2DBuilder`: allocate
/// parameters, define entities, set up the objective and constraints,
/// then call `.build()` to get an `OptimizationProblem`.
///
/// # Example
///
/// ```rust,ignore
/// use solverang::optimization::{OptimizationBuilder, OptimizationConfig};
///
/// let mut b = OptimizationBuilder::new();
/// let x = b.add_variable(1.0);
/// let y = b.add_variable(1.0);
///
/// // Minimize (x-3)^2 + (y-4)^2
/// b.set_objective(QuadraticObjective { target_x: 3.0, target_y: 4.0, px: x, py: y });
///
/// // Subject to: x + y = 5  (equality)
/// b.add_equality_constraint(SumConstraint { px: x, py: y, target: 5.0 });
///
/// let problem = b.build().unwrap();
/// let result = problem.solve(OptimizationConfig::default());
/// assert!(result.is_converged());
/// ```
pub struct OptimizationBuilder {
    params: ParamStore,
    /// Dummy entity for standalone optimization variables.
    opt_entity: EntityId,
    /// Maps variable index -> ParamId.
    variables: Vec<ParamId>,
    /// The objective function (set via set_objective).
    objective: Option<Box<dyn Objective>>,
    /// Equality constraints (h(x) = 0).
    equality_constraints: Vec<Box<dyn Constraint>>,
    /// Inequality constraints (g(x) <= 0).
    inequality_constraints: Vec<Box<dyn InequalityConstraint>>,
    /// Variable bounds.
    bounds: Vec<(ParamId, f64, f64)>,
}

impl OptimizationBuilder {
    /// Create a new, empty optimization builder.
    pub fn new() -> Self {
        let mut params = ParamStore::new();
        // Use a well-known sentinel entity ID for standalone optimization variables.
        // This must not collide with any EntityId allocated by an imported
        // ConstraintSystem. If `import_constraint_system` is called, the system's
        // entity IDs begin at 0 and increment, so this sentinel must be distinct.
        // TODO: allocate via a shared counter once ConstraintSystem exposes
        // `alloc_entity_id` independently of entity ownership.
        let opt_entity = EntityId::new(u32::MAX, 0);
        Self {
            params,
            opt_entity,
            variables: Vec::new(),
            objective: None,
            equality_constraints: Vec::new(),
            inequality_constraints: Vec::new(),
            bounds: Vec::new(),
        }
    }

    // =================================================================
    // Variable creation
    // =================================================================

    /// Add a free optimization variable with the given initial value.
    /// Returns the `ParamId` that can be used in objective/constraint closures.
    pub fn add_variable(&mut self, initial_value: f64) -> ParamId {
        let pid = self.params.alloc(initial_value, self.opt_entity);
        self.variables.push(pid);
        pid
    }

    /// Add a variable with lower and upper bounds.
    pub fn add_bounded_variable(
        &mut self,
        initial_value: f64,
        lower: f64,
        upper: f64,
    ) -> ParamId {
        let pid = self.add_variable(initial_value);
        self.bounds.push((pid, lower, upper));
        pid
    }

    /// Fix a variable (exclude from optimization).
    pub fn fix_variable(&mut self, param: ParamId) {
        self.params.fix(param);
    }

    // =================================================================
    // Objective
    // =================================================================

    /// Set the objective function to minimize.
    ///
    /// Only one objective can be set. Calling this again replaces the
    /// previous objective.
    pub fn set_objective(&mut self, objective: impl Objective + 'static) {
        self.objective = Some(Box::new(objective));
    }

    /// Set a weighted multi-objective: minimize sum(w_i * f_i(x)).
    ///
    /// This is sugar for a single `WeightedObjective` that sums
    /// individual objectives with the given weights.
    pub fn set_weighted_objective(
        &mut self,
        objectives: Vec<(f64, Box<dyn Objective>)>,
    ) {
        self.objective = Some(Box::new(WeightedObjective { objectives }));
    }

    // =================================================================
    // Constraints
    // =================================================================

    /// Add an equality constraint h(x) = 0.
    ///
    /// This accepts any existing `Constraint` from the sketch2d module
    /// or a custom implementation.
    pub fn add_equality_constraint(
        &mut self,
        constraint: impl Constraint + 'static,
    ) {
        self.equality_constraints.push(Box::new(constraint));
    }

    /// Add an inequality constraint g(x) <= 0.
    pub fn add_inequality_constraint(
        &mut self,
        constraint: impl InequalityConstraint + 'static,
    ) {
        self.inequality_constraints.push(Box::new(constraint));
    }

    // =================================================================
    // Build
    // =================================================================

    /// Validate and produce the optimization problem.
    ///
    /// Returns `Err` with a descriptive `OptimizationError` if the
    /// problem is malformed (no objective, inconsistent bounds, etc.).
    pub fn build(self) -> Result<OptimizationProblem, OptimizationError> {
        let objective = self
            .objective
            .ok_or(OptimizationError::NoObjective)?;

        if self.params.free_param_count() == 0 {
            return Err(OptimizationError::NoVariables);
        }

        // Validate bounds
        for &(pid, lo, hi) in &self.bounds {
            if lo > hi {
                return Err(OptimizationError::InvalidBounds {
                    param_index: pid.raw_index() as usize,
                    lower: lo,
                    upper: hi,
                });
            }
        }

        Ok(OptimizationProblem {
            objective,
            equality_constraints: self.equality_constraints,
            inequality_constraints: self.inequality_constraints,
            params: self.params,
            bounds: self.bounds,
        })
    }

    // =================================================================
    // Interop with ConstraintSystem
    // =================================================================

    /// Import variables and constraints from an existing `ConstraintSystem`.
    ///
    /// This lets you define a Sketch2D geometry, then optimize over it.
    /// All free parameters of the system become optimization variables;
    /// all constraints become equality constraints.
    ///
    /// # Prerequisites
    ///
    /// This method requires `ConstraintSystem` to expose a public iterator over
    /// its alive constraints (e.g., `fn constraints(&self) -> impl Iterator<Item = &dyn Constraint>`).
    /// That method does not yet exist and must be added before this can be
    /// implemented.
    ///
    /// # ParamId Namespace Warning
    ///
    /// After calling this method, do NOT call `add_variable()` on the same
    /// builder. The imported system's `ParamStore` uses `ParamId` indices
    /// starting from 0. Any new variables allocated via `add_variable()` would
    /// produce IDs that collide with the imported IDs. If additional variables
    /// are needed alongside an imported system, a separate builder should be
    /// used and the results combined.
    pub fn import_constraint_system(&mut self, system: &ConstraintSystem) {
        // Implementation delegates to system's param store and constraint list.
        // Omitted for brevity — the key idea is that existing Sketch2D
        // constraints become equality constraints in the optimization.
        todo!()
    }
}

/// A weighted sum of multiple objectives.
///
/// Must be `pub` so that test code and the property tests in Section 7
/// can construct and use it directly.
pub struct WeightedObjective {
    pub objectives: Vec<(f64, Box<dyn Objective>)>,
}

impl Objective for WeightedObjective {
    fn name(&self) -> &str {
        "WeightedObjective"
    }

    fn variable_count(&self) -> usize {
        // All sub-objectives share the same variable space.
        self.objectives
            .first()
            .map(|(_, o)| o.variable_count())
            .unwrap_or(0)
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.objectives
            .iter()
            .map(|(w, obj)| w * obj.value(x))
            .sum()
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let n = self.variable_count();
        let mut grad = vec![0.0; n];
        for (w, obj) in &self.objectives {
            let g = obj.gradient(x);
            for i in 0..n {
                grad[i] += w * g[i];
            }
        }
        grad
    }

    fn hessian(&self, x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
        let mut triplets = Vec::new();
        for (w, obj) in &self.objectives {
            if let Some(h) = obj.hessian(x) {
                for (r, c, v) in h {
                    triplets.push((r, c, w * v));
                }
            } else {
                return None; // If any sub-objective lacks a Hessian, bail out.
            }
        }
        Some(triplets)
    }
}
```

---

## 4. Code Examples

### 4.1 Unconstrained Optimization — Rosenbrock

```rust
use solverang::optimization::{
    Objective, OptimizationBuilder, OptimizationConfig,
};

/// Rosenbrock objective: f(x,y) = (1-x)^2 + 100(y-x^2)^2
///
/// Note: with the array-indexed Objective interface, the struct needs no
/// stored ParamId fields. Variables are accessed by position in x[].
/// The insertion order from add_variable() calls determines x[0], x[1], etc.
struct RosenbrockObjective;

impl Objective for RosenbrockObjective {
    fn name(&self) -> &str { "Rosenbrock" }
    fn variable_count(&self) -> usize { 2 }

    fn value(&self, x: &[f64]) -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        let dy = 200.0 * (x[1] - x[0] * x[0]);
        vec![dx, dy]
    }

    fn hessian(&self, x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
        // Lower-triangular entries only (row >= col).
        // h00 = d^2f/dx^2 = 2 - 400*(y - 3*x^2)
        let h00 = 2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0]);
        // h10 = d^2f/(dy dx) = -400*x  (lower triangle: row=1, col=0)
        let h10 = -400.0 * x[0];
        // h11 = d^2f/dy^2 = 200
        let h11 = 200.0;
        Some(vec![(0, 0, h00), (1, 0, h10), (1, 1, h11)])
    }
}

#[test]
fn test_unconstrained_rosenbrock() {
    let mut b = OptimizationBuilder::new();
    let _px = b.add_variable(-1.2); // x[0]
    let _py = b.add_variable(1.0);  // x[1]
    b.set_objective(RosenbrockObjective);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    assert!((sol[0] - 1.0).abs() < 1e-6);
    assert!((sol[1] - 1.0).abs() < 1e-6);
    assert!(result.objective_value().unwrap() < 1e-12);
}
```

### 4.2 Equality-Constrained Optimization

```rust
use solverang::optimization::{
    Objective, OptimizationBuilder, OptimizationConfig,
};
use solverang::constraint::Constraint;
use solverang::id::{ConstraintId, EntityId, ParamId};
use solverang::param::ParamStore;

/// Minimize f(x,y) = x^2 + y^2 subject to x + y = 1.
struct DistanceToOrigin;

impl Objective for DistanceToOrigin {
    fn name(&self) -> &str { "DistanceToOrigin" }
    fn variable_count(&self) -> usize { 2 }

    fn value(&self, x: &[f64]) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![2.0 * x[0], 2.0 * x[1]]
    }

    fn hessian(&self, _x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
        Some(vec![(0, 0, 2.0), (1, 1, 2.0)])
    }
}

/// Constraint: x + y - 1 = 0.
struct SumEquals {
    id: ConstraintId,
    entity_ids: Vec<EntityId>,
    px: ParamId,
    py: ParamId,
    target: f64,
}

impl Constraint for SumEquals {
    fn id(&self) -> ConstraintId { self.id }
    fn name(&self) -> &str { "SumEquals" }
    fn entity_ids(&self) -> &[EntityId] { &self.entity_ids }
    fn param_ids(&self) -> &[ParamId] { &[self.px, self.py] }
    fn equation_count(&self) -> usize { 1 }

    fn residuals(&self, store: &ParamStore) -> Vec<f64> {
        vec![store.get(self.px) + store.get(self.py) - self.target]
    }

    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![(0, self.px, 1.0), (0, self.py, 1.0)]
    }
}

#[test]
fn test_equality_constrained() {
    let mut b = OptimizationBuilder::new();
    let px = b.add_variable(0.8);
    let py = b.add_variable(0.2);

    b.set_objective(DistanceToOrigin);

    // x + y = 1.
    // ConstraintId and EntityId must be allocated consistently.
    // In production code these come from a ConstraintSystem or a shared counter;
    // here we use well-known sentinel values since no other IDs are in scope.
    let cid = ConstraintId::new(0, 0);
    let eid = EntityId::new(0, 0);
    b.add_equality_constraint(SumEquals {
        id: cid, entity_ids: vec![eid], px, py, target: 1.0,
    });

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    // Analytical solution: x = y = 0.5
    assert!((sol[0] - 0.5).abs() < 1e-6);
    assert!((sol[1] - 0.5).abs() < 1e-6);
}
```

### 4.3 Mixed Equality and Inequality Constraints

```rust
use solverang::optimization::{
    InequalityConstraint, Objective, OptimizationBuilder, OptimizationConfig,
};
use solverang::id::ParamId;
use solverang::param::ParamStore;

/// Minimize f(x,y) = -x - y
/// subject to: x + y <= 1  (g1)
///             x >= 0      (g2: -x <= 0)
///             y >= 0      (g3: -y <= 0)
///
/// Solution: x = 0.5, y = 0.5 (or any point on the x+y=1 edge).
/// Actually: solution is at a vertex of the feasible region on x+y=1.
/// With the linear objective -x-y, any point on x+y=1 with x,y >= 0
/// gives the same optimal value f* = -1.

struct NegSum;

impl Objective for NegSum {
    fn name(&self) -> &str { "NegSum" }
    fn variable_count(&self) -> usize { 2 }

    fn value(&self, x: &[f64]) -> f64 {
        -x[0] - x[1]
    }

    fn gradient(&self, _x: &[f64]) -> Vec<f64> {
        vec![-1.0, -1.0]
    }

    fn hessian(&self, _x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
        Some(vec![]) // Zero Hessian (linear objective).
    }
}

/// g(x) = x + y - 1 <= 0.
struct SumLeq {
    px: ParamId,
    py: ParamId,
}

impl InequalityConstraint for SumLeq {
    fn name(&self) -> &str { "SumLeq1" }
    fn param_ids(&self) -> &[ParamId] { &[self.px, self.py] }
    fn inequality_count(&self) -> usize { 1 }

    fn values(&self, store: &ParamStore) -> Vec<f64> {
        vec![store.get(self.px) + store.get(self.py) - 1.0]
    }

    fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        vec![(0, self.px, 1.0), (0, self.py, 1.0)]
    }
}

#[test]
fn test_mixed_constraints() {
    let mut b = OptimizationBuilder::new();
    let px = b.add_bounded_variable(0.3, 0.0, f64::INFINITY);
    let py = b.add_bounded_variable(0.3, 0.0, f64::INFINITY);

    b.set_objective(NegSum);
    b.add_inequality_constraint(SumLeq { px, py });

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    // x + y should equal 1 at optimum, both non-negative
    assert!((sol[0] + sol[1] - 1.0).abs() < 1e-6);
    assert!(sol[0] >= -1e-10);
    assert!(sol[1] >= -1e-10);
    assert!((result.objective_value().unwrap() - (-1.0)).abs() < 1e-6);
}
```

### 4.4 Reusing Sketch2D Constraints

```rust
use solverang::optimization::{Objective, OptimizationBuilder, OptimizationConfig};
use solverang::sketch2d::builder::Sketch2DBuilder;

/// Find the position of point P that minimizes distance to a target
/// while satisfying a distance constraint to a fixed origin.
///
/// Minimize |P - target|^2  subject to  |P - origin| = R

struct DistToTarget {
    target: [f64; 2],
}

impl Objective for DistToTarget {
    fn name(&self) -> &str { "DistToTarget" }
    fn variable_count(&self) -> usize { 2 }

    fn value(&self, x: &[f64]) -> f64 {
        let dx = x[0] - self.target[0];
        let dy = x[1] - self.target[1];
        dx * dx + dy * dy
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![
            2.0 * (x[0] - self.target[0]),
            2.0 * (x[1] - self.target[1]),
        ]
    }

    fn hessian(&self, _x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
        Some(vec![(0, 0, 2.0), (1, 1, 2.0)])
    }
}

#[test]
fn test_reuse_sketch2d_constraint() {
    // Build a Sketch2D geometry with origin and moving point.
    let mut sketch = Sketch2DBuilder::new();
    let origin = sketch.add_fixed_point(0.0, 0.0);
    let p = sketch.add_point(3.0, 0.0);
    let _dist_cid = sketch.constrain_distance(origin, p, 5.0);

    // Now set up optimization: minimize distance from P to target (4, 3)
    // subject to the Sketch2D distance constraint |P - origin| = 5.
    let mut opt = OptimizationBuilder::new();
    opt.import_constraint_system(sketch.system());
    opt.set_objective(DistToTarget { target: [4.0, 3.0] });

    let problem = opt.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    // The closest point on the circle of radius 5 to (4,3) is
    // (4,3) * 5 / |(4,3)| = (4,3) * 5/5 = (4,3).
    assert!((sol[0] - 4.0).abs() < 1e-6);
    assert!((sol[1] - 3.0).abs() < 1e-6);
}
```

### 4.5 Multi-Objective / Weighted Optimization

```rust
use solverang::optimization::{Objective, OptimizationBuilder, OptimizationConfig};

/// Minimize distance to point A.
struct DistToA {
    ax: f64,
    ay: f64,
}

impl Objective for DistToA {
    fn name(&self) -> &str { "DistToA" }
    fn variable_count(&self) -> usize { 2 }

    fn value(&self, x: &[f64]) -> f64 {
        (x[0] - self.ax).powi(2) + (x[1] - self.ay).powi(2)
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![2.0 * (x[0] - self.ax), 2.0 * (x[1] - self.ay)]
    }

    fn hessian(&self, _x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
        Some(vec![(0, 0, 2.0), (1, 1, 2.0)])
    }
}

/// Minimize distance to point B.
struct DistToB {
    bx: f64,
    by: f64,
}

impl Objective for DistToB {
    fn name(&self) -> &str { "DistToB" }
    fn variable_count(&self) -> usize { 2 }

    fn value(&self, x: &[f64]) -> f64 {
        (x[0] - self.bx).powi(2) + (x[1] - self.by).powi(2)
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![2.0 * (x[0] - self.bx), 2.0 * (x[1] - self.by)]
    }

    fn hessian(&self, _x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
        Some(vec![(0, 0, 2.0), (1, 1, 2.0)])
    }
}

#[test]
fn test_weighted_multi_objective() {
    let mut b = OptimizationBuilder::new();
    let _px = b.add_variable(0.0);
    let _py = b.add_variable(0.0);

    // Minimize 0.3 * |P - A|^2 + 0.7 * |P - B|^2
    // where A = (0, 0) and B = (10, 0).
    // Analytical solution: x = 0.3*0 + 0.7*10 = 7.0, y = 0.
    b.set_weighted_objective(vec![
        (0.3, Box::new(DistToA { ax: 0.0, ay: 0.0 })),
        (0.7, Box::new(DistToB { bx: 10.0, by: 0.0 })),
    ]);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    assert!((sol[0] - 7.0).abs() < 1e-6);
    assert!((sol[1] - 0.0).abs() < 1e-6);
}
```

### 4.6 Bounded Variable Optimization

```rust
use solverang::optimization::{Objective, OptimizationBuilder, OptimizationConfig};

/// Minimize f(x) = (x - 5)^2 subject to 0 <= x <= 3.
/// Solution: x* = 3 (bound is active).
struct QuadFrom5;

impl Objective for QuadFrom5 {
    fn name(&self) -> &str { "QuadFrom5" }
    fn variable_count(&self) -> usize { 1 }

    fn value(&self, x: &[f64]) -> f64 {
        (x[0] - 5.0).powi(2)
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![2.0 * (x[0] - 5.0)]
    }

    fn hessian(&self, _x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
        Some(vec![(0, 0, 2.0)])
    }
}

#[test]
fn test_bounded_optimization() {
    let mut b = OptimizationBuilder::new();
    let _x = b.add_bounded_variable(1.0, 0.0, 3.0);
    b.set_objective(QuadFrom5);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    assert!((sol[0] - 3.0).abs() < 1e-6);
}
```

---

## 5. Error Handling Design

### 5.1 Error Message Philosophy

Errors are categorized into three tiers, following the existing `SolveError` and
`DiagnosticIssue` patterns:

| Tier | When | Example |
|------|------|---------|
| **Build-time** | `build()` returns `Err` | Missing objective, invalid bounds |
| **Solve-time** | `solve()` returns `Failed` variant | NaN in gradient, singular Hessian |
| **Diagnostic** | `diagnose()` returns warnings | Redundant constraints, under-constrained |

### 5.2 Build-Time Errors

```rust
// These are returned from OptimizationBuilder::build()

let mut b = OptimizationBuilder::new();
let x = b.add_variable(1.0);
// Forgot to set objective!
let err = b.build().unwrap_err();
assert_eq!(err.to_string(), "no objective function defined");

// Invalid bounds
let mut b = OptimizationBuilder::new();
let x = b.add_bounded_variable(1.0, 5.0, 3.0); // lower > upper
let err = b.build();
// Error: "bound violation: lower bound 5 > upper bound 3 for parameter 0"
```

### 5.3 Solve-Time Errors

```rust
// Returned as OptimizationResult::Failed

// NaN objective
let result = problem.solve(config);
if let OptimizationResult::Failed { error } = result {
    match error {
        OptimizationError::NonFiniteObjective =>
            println!("Objective returned NaN — check your formula"),
        OptimizationError::NonFiniteGradient =>
            println!("Gradient has NaN — verify analytial derivatives"),
        OptimizationError::LineSearchFailed =>
            println!("Line search failed — try a different starting point"),
        OptimizationError::MaxIterationsExceeded(n) =>
            println!("Did not converge in {n} iterations — increase max_iterations or loosen tolerance"),
        _ => {}
    }
}
```

### 5.4 Diagnostic Warnings

> **REVIEW NOTE:** The `OptDiagnostic` enum and `OptimizationProblem::diagnose()`
> method are referenced here but defined nowhere in this document or the other
> docs in this series. They must be defined before this section can be
> implemented. Suggested definition:
>
> ```rust
> // crates/solverang/src/optimization/diagnostic.rs
>
> #[derive(Debug, Clone)]
> pub enum OptDiagnostic {
>     ActiveBound { param: ParamId, bound_type: BoundType },
>     NearSingularHessian { condition_number: f64 },
>     RedundantEqualityConstraint { index: usize },
>     InactiveInequality { index: usize },
> }
>
> #[derive(Debug, Clone, Copy)]
> pub enum BoundType { Lower, Upper }
> ```
>
> And `OptimizationProblem` needs `pub fn diagnose(&self) -> Vec<OptDiagnostic>`.

```rust
// Non-fatal diagnostic output
let diag = problem.diagnose();
for issue in &diag {
    match issue {
        OptDiagnostic::ActiveBound { param, bound_type } =>
            println!("Parameter {param:?} is at its {bound_type} bound — solution may be on boundary"),
        OptDiagnostic::NearSingularHessian { condition_number } =>
            println!("Hessian condition number {condition_number:.2e} — objective may be nearly flat"),
        OptDiagnostic::RedundantEqualityConstraint { index } =>
            println!("Equality constraint {index} is redundant (linearly dependent)"),
        OptDiagnostic::InactiveInequality { index } =>
            println!("Inequality constraint {index} is inactive (slack > 0)"),
    }
}
```

---

## 6. TDD Roadmap

Each stage follows the Red-Green-Refactor cycle. The stages build incrementally
from foundational types through to full integration.

---

### Stage 1: `Objective` Trait and `OptimizationResult`

**Red — write the test first:**

```rust
// tests/optimization_tests.rs

#[test]
fn test_objective_trait_basic() {
    // A simple quadratic: f(x) = x^2, grad = 2x.
    struct Quadratic;
    impl Objective for Quadratic {
        fn name(&self) -> &str { "x^2" }
        fn variable_count(&self) -> usize { 1 }
        fn value(&self, x: &[f64]) -> f64 { x[0] * x[0] }
        fn gradient(&self, x: &[f64]) -> Vec<f64> { vec![2.0 * x[0]] }
    }

    let obj = Quadratic;
    assert_eq!(obj.value(&[3.0]), 9.0);
    assert_eq!(obj.gradient(&[3.0]), vec![6.0]);
    assert!(obj.hessian(&[3.0]).is_none()); // default returns None
}
```

**Expected failure:** `Objective` trait does not exist; compilation error.

**Green:** Create `crates/solverang/src/optimization/mod.rs` and
`objective.rs` with the trait definition. Create `result.rs` with
`OptimizationResult` and `OptimizationError`.

**Refactor:** Extract common result accessor methods (`.is_converged()`,
`.solution()`, `.objective_value()`) following the same pattern as `SolveResult`.

---

### Stage 2: `OptimizationBuilder` — Variable Allocation

**Red:**

```rust
#[test]
fn test_builder_add_variable() {
    let mut b = OptimizationBuilder::new();
    let x = b.add_variable(3.0);
    let y = b.add_variable(4.0);

    // Two free variables allocated
    assert_eq!(b.variable_count(), 2);
    assert_eq!(b.get_value(x), 3.0);
    assert_eq!(b.get_value(y), 4.0);
}

#[test]
fn test_builder_bounded_variable() {
    let mut b = OptimizationBuilder::new();
    let x = b.add_bounded_variable(1.0, 0.0, 10.0);
    assert_eq!(b.bounds_for(x), Some((0.0, 10.0)));
}

#[test]
fn test_builder_fix_variable() {
    let mut b = OptimizationBuilder::new();
    let x = b.add_variable(5.0);
    b.fix_variable(x);
    assert_eq!(b.free_variable_count(), 0);
}
```

**Expected failure:** `OptimizationBuilder` struct does not exist, and
`variable_count()`, `get_value()`, `bounds_for()`, and `free_variable_count()`
are also undefined. The tests fail to compile for both reasons simultaneously.

**Green:** Implement `OptimizationBuilder` with `ParamStore` delegation AND
add `variable_count()`, `free_variable_count()`, `get_value()`, and
`bounds_for()` in the same step. The builder wraps `ParamStore` for allocation
and tracks variable metadata.

**Refactor:** Confirm that the added accessor methods match the patterns in
`Sketch2DBuilder` for consistency.

---

### Stage 3: `OptimizationBuilder::build()` — Validation

**Red:**

```rust
#[test]
fn test_build_fails_without_objective() {
    let mut b = OptimizationBuilder::new();
    let _ = b.add_variable(1.0);
    let err = b.build().unwrap_err();
    assert_eq!(err, OptimizationError::NoObjective);
}

#[test]
fn test_build_fails_no_variables() {
    let mut b = OptimizationBuilder::new();
    b.set_objective(ConstantObjective); // f(x) = 42, but no variables
    let err = b.build().unwrap_err();
    assert_eq!(err, OptimizationError::NoVariables);
}

#[test]
fn test_build_fails_invalid_bounds() {
    let mut b = OptimizationBuilder::new();
    let _ = b.add_bounded_variable(1.0, 5.0, 3.0); // lo > hi
    b.set_objective(ConstantObjective);
    let err = b.build().unwrap_err();
    assert!(matches!(err, OptimizationError::InvalidBounds { .. }));
}

#[test]
fn test_build_succeeds_simple() {
    let mut b = OptimizationBuilder::new();
    let _ = b.add_variable(1.0);
    b.set_objective(ConstantObjective);
    let problem = b.build();
    assert!(problem.is_ok());
}
```

**Expected failure:** `build()` not implemented or returns wrong errors.

**Green:** Implement the three validation checks in `build()`.

**Refactor:** Consolidate error creation; ensure error messages match the
`Display` impls in `OptimizationError`.

---

### Stage 4: Unconstrained Gradient Descent (Simplest Solver)

**Red:**

```rust
#[test]
fn test_solve_unconstrained_quadratic() {
    // Minimize f(x) = (x-3)^2. Solution: x = 3.
    struct ShiftedQuadratic;
    impl Objective for ShiftedQuadratic {
        fn name(&self) -> &str { "shifted-quad" }
        fn variable_count(&self) -> usize { 1 }
        fn value(&self, x: &[f64]) -> f64 { (x[0] - 3.0).powi(2) }
        fn gradient(&self, x: &[f64]) -> Vec<f64> { vec![2.0 * (x[0] - 3.0)] }
    }

    let mut b = OptimizationBuilder::new();
    let _ = b.add_variable(0.0);
    b.set_objective(ShiftedQuadratic);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    assert!((sol[0] - 3.0).abs() < 1e-6);
}
```

**Expected failure:** `OptimizationProblem::solve()` does not exist.

**Green:** Implement the simplest possible solver — steepest descent with
Armijo backtracking line search. This is intentionally naive; it will be
replaced later, but it establishes the solve interface.

```rust
impl OptimizationProblem {
    pub fn solve(&self, config: OptimizationConfig) -> OptimizationResult {
        // Steepest descent with backtracking line search.
        //
        // Note: `build_solver_mapping()` must be made public on `ParamStore`
        // or the solve logic must go through the pipeline infrastructure.
        // In the current codebase, solver mapping construction happens inside
        // `SolvePipeline`, not directly on `ParamStore`. This needs resolution
        // before implementation; one option is to add a
        // `ParamStore::free_param_values() -> Vec<f64>` method as an alternative.
        let mapping = self.params.build_solver_mapping();
        let n = mapping.len();
        let mut x: Vec<f64> = mapping.col_to_param.iter()
            .map(|&pid| self.params.get(pid))
            .collect();

        for iter in 0..config.max_iterations {
            let val = self.objective.value(&x);
            let grad = self.objective.gradient(&x);
            let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

            if grad_norm < config.gradient_tolerance {
                return OptimizationResult::Converged {
                    solution: x,
                    objective_value: val,
                    iterations: iter,
                    gradient_norm: grad_norm,
                    constraint_violation: 0.0,
                };
            }

            // Backtracking line search
            let mut alpha = 1.0;
            let c = 1e-4;
            let dir_deriv: f64 = grad.iter().map(|g| -g * g).sum();
            loop {
                let x_new: Vec<f64> = x.iter().zip(&grad)
                    .map(|(xi, gi)| xi - alpha * gi)
                    .collect();
                if self.objective.value(&x_new) <= val + c * alpha * dir_deriv {
                    x = x_new;
                    break;
                }
                alpha *= 0.5;
                if alpha < 1e-16 {
                    return OptimizationResult::Failed {
                        error: OptimizationError::LineSearchFailed,
                    };
                }
            }
        }

        OptimizationResult::NotConverged { /* ... */ }
    }
}
```

**Refactor:** Extract line search into a separate module. Add convergence
status logging behind a feature flag (similar to `bool_debug!`).

---

### Stage 5: Unconstrained L-BFGS

<!-- Decision: Stage 5 implements L-BFGS (not Newton) for Rosenbrock. BFGS/L-BFGS is the default optimization method and needs only first derivatives. Newton's method (requiring Hessian) is an optional upgrade, not the default path. -->

**Red:**

```rust
#[test]
fn test_solve_unconstrained_rosenbrock() {
    let mut b = OptimizationBuilder::new();
    let _ = b.add_variable(-1.2);
    let _ = b.add_variable(1.0);
    b.set_objective(RosenbrockObjective);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    assert!((sol[0] - 1.0).abs() < 1e-4);
    assert!((sol[1] - 1.0).abs() < 1e-4);
}
```

**Expected failure:** Gradient descent from Stage 4 is too slow for Rosenbrock.
The test will hit max iterations without meeting the tolerance.

**Green:** Implement L-BFGS. Maintain a history of the last m gradient
differences and parameter steps (m = 20 by default). Use the two-loop
recursion to compute the L-BFGS direction. No Hessian evaluation needed —
this is a first-derivative-only method.

**Refactor:** Introduce `SolverStrategy` enum (`GradientDescent`, `LBFGS`,
`Newton`) that the solver selects based on problem capabilities. L-BFGS is
the default for unconstrained problems.

---

### Stage 6: Augmented Lagrangian for Equality Constraints

<!-- Decision: Stage 6 goes directly to ALM (not penalty method first). ALM reuses the existing NR/LM inner solver, requires no Hessian, and converges with bounded penalty parameter. The detour through a pure penalty method (and Stage 11 upgrade) is skipped. -->

**Red:**

```rust
#[test]
fn test_equality_constrained_min_on_line() {
    // Minimize x^2 + y^2 subject to x + y = 1.
    // Solution: x = y = 0.5.
    let mut b = OptimizationBuilder::new();
    let px = b.add_variable(0.8);
    let py = b.add_variable(0.2);
    b.set_objective(DistanceToOrigin);
    b.add_equality_constraint(/* x + y - 1 = 0 */);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    assert!((sol[0] - 0.5).abs() < 1e-5);
    assert!((sol[1] - 0.5).abs() < 1e-5);
    assert!(result.constraint_violation().unwrap_or(f64::MAX) < 1e-8);
}
```

**Expected failure:** The unconstrained solver ignores constraints entirely.

**Green:** Implement augmented Lagrangian method. The augmented Lagrangian is:

    L_A(x; lambda, rho) = f(x) + lambda^T h(x) + (rho/2) * sum(h_i(x)^2)

Outer loop updates multipliers: `lambda += rho * h(x)`, and increases `rho`
when constraint violation does not decrease sufficiently. Inner loop minimizes
the augmented objective using L-BFGS from Stage 5 (or the existing LM solver).
Converge when `max|h_i(x)| < constraint_tolerance`.

**Refactor:** Extract `AugLagObjective` as a reusable adapter that wraps
the original objective and equality constraints.

---

### Stage 7: Inequality Constraints via Log-Barrier / Slack Variables

**Red:**

```rust
#[test]
fn test_inequality_constrained_bounded_linear() {
    // Minimize -x subject to x <= 3 and x >= 0.
    // Solution: x = 3.
    let mut b = OptimizationBuilder::new();
    let x = b.add_bounded_variable(1.0, 0.0, f64::INFINITY);
    b.set_objective(NegX);
    b.add_inequality_constraint(/* x - 3 <= 0 */);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    assert!((sol[0] - 3.0).abs() < 1e-5);
}
```

**Expected failure:** No inequality handling exists.

**Green:** Implement log-barrier (interior point) method. Convert
`g(x) <= 0` to barrier term `-mu * sum(ln(-g_i(x)))`. Drive `mu -> 0` over
outer iterations. Alternatively, convert to slack variables: `g(x) + s = 0,
s >= 0` and use the penalty method from Stage 6 plus bound enforcement.

**Refactor:** Unify the penalty and barrier methods under a common
`AugmentedLagrangian` framework. Both are instances of the same pattern:
perturbed KKT conditions.

---

### Stage 8: Variable Bounds

**Red:**

```rust
#[test]
fn test_bounds_active_at_solution() {
    // Minimize (x - 5)^2 subject to 0 <= x <= 3.
    // Solution: x = 3.
    let mut b = OptimizationBuilder::new();
    let _ = b.add_bounded_variable(1.0, 0.0, 3.0);
    b.set_objective(QuadFrom5);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    assert!((result.solution().unwrap()[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_bounds_inactive() {
    // Minimize (x - 1)^2 subject to 0 <= x <= 3.
    // Solution: x = 1 (bound inactive).
    let mut b = OptimizationBuilder::new();
    let _ = b.add_bounded_variable(2.0, 0.0, 3.0);
    b.set_objective(QuadFrom1);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    assert!((result.solution().unwrap()[0] - 1.0).abs() < 1e-6);
}
```

**Expected failure:** Solver does not enforce bounds.

**Green:** Implement projected gradient for bounds. After each step,
clamp variables to their bounds. For Newton's method, use a projected
Newton approach or convert bounds to inequality constraints handled by
the barrier method from Stage 7.

**Refactor:** Consolidate bound handling with inequality handling. Bounds
are syntactic sugar for `x - upper <= 0` and `lower - x <= 0`.

---

### Stage 9: Weighted Multi-Objective

> **REVIEW NOTE:** `WeightedObjective` is a pure data structure — it requires
> no new solver capability. The existing gradient descent / Newton solver from
> Stages 4-5 already handles it. This stage tests `set_weighted_objective()`
> as a builder API convenience rather than a solver advancement. It could
> reasonably be folded into Stage 2 (builder ergonomics). It is kept as a
> separate stage only to confirm the end-to-end path works with a
> multi-component objective.

**Red:**

```rust
#[test]
fn test_weighted_objective() {
    let mut b = OptimizationBuilder::new();
    let _ = b.add_variable(0.0);
    let _ = b.add_variable(0.0);

    b.set_weighted_objective(vec![
        (0.3, Box::new(DistToA { ax: 0.0, ay: 0.0 })),
        (0.7, Box::new(DistToB { bx: 10.0, by: 0.0 })),
    ]);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    // Weighted centroid: 0.3*(0,0) + 0.7*(10,0) = (7, 0)
    assert!((sol[0] - 7.0).abs() < 1e-6);
    assert!((sol[1] - 0.0).abs() < 1e-6);
}
```

**Expected failure:** `set_weighted_objective` not implemented.

**Green:** Implement `WeightedObjective` struct that stores
`Vec<(f64, Box<dyn Objective>)>` and sums weighted values/gradients/Hessians.
This is already shown in Section 3 as part of `OptimizationBuilder`.

**Refactor:** Verify that `WeightedObjective` composes Hessians correctly
(only returns `Some` if all sub-objectives provide Hessians).

---

### Stage 10: ConstraintSystem Interop

**Red:**

```rust
#[test]
fn test_import_constraint_system() {
    let mut sketch = Sketch2DBuilder::new();
    let origin = sketch.add_fixed_point(0.0, 0.0);
    let p = sketch.add_point(3.0, 0.0);
    sketch.constrain_distance(origin, p, 5.0);

    let mut opt = OptimizationBuilder::new();
    opt.import_constraint_system(sketch.system());
    opt.set_objective(DistToTarget { target: [4.0, 3.0] });

    let problem = opt.build().unwrap();
    assert!(problem.equality_constraints.len() >= 1);
    assert!(problem.params.free_param_count() >= 2);
}
```

**Expected failure:** `import_constraint_system` is a `todo!()`.

**Green:** Implement `import_constraint_system` by:
1. Cloning the `ParamStore` from the `ConstraintSystem`.
2. Iterating over alive constraints and adding them as equality constraints.
3. Preserving fixed/free status of parameters.

**Refactor:** Handle entity ownership correctly. Ensure that `ParamId`s
remain valid across the boundary. Consider a trait-based abstraction if
the clone approach is too heavy.

---

### Stage 11: ALM Convergence Verification and Tuning

**Red:**

```rust
#[test]
fn test_augmented_lagrangian_convergence_rate() {
    // The penalty method needs mu -> infinity for exact satisfaction.
    // The augmented Lagrangian should converge with bounded mu.
    let mut b = OptimizationBuilder::new();
    let px = b.add_variable(0.8);
    let py = b.add_variable(0.2);
    b.set_objective(DistanceToOrigin);
    b.add_equality_constraint(/* x + y - 1 = 0 */);

    let config = OptimizationConfig {
        max_iterations: 50, // Tighter iteration budget
        ..Default::default()
    };
    let problem = b.build().unwrap();
    let result = problem.solve(config);

    assert!(result.is_converged());
    // Constraint should be satisfied to high precision
    let cv = result.constraint_violation().unwrap_or(f64::MAX);
    assert!(cv < 1e-10, "Constraint violation {cv} too high");
}
```

**Expected failure:** ALM may not converge in 50 iterations with constraint
violation < 1e-10 if multiplier initialization or penalty schedule is wrong.

**Green:** Verify that dual variable updates `lambda_i += rho * h_i(x)` and
the penalty schedule produce tight constraint satisfaction within a bounded
number of outer iterations. Tune `initial_penalty`, `penalty_increase`, and
`max_outer_iterations` in `AugLagConfig`.

**Refactor:** Parameterize the solve method to accept a `SolveStrategy`
enum allowing the caller to choose between augmented Lagrangian and (future)
SQP.

---

### Stage 12: OptimizationResult Diagnostics

**Red:**

```rust
#[test]
fn test_result_reports_gradient_norm() {
    let mut b = OptimizationBuilder::new();
    let _ = b.add_variable(3.0); // Already at the minimum of (x-3)^2
    b.set_objective(ShiftedQuadratic);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    // Gradient should be near zero at the solution
    match &result {
        OptimizationResult::Converged { gradient_norm, .. } => {
            assert!(*gradient_norm < 1e-8);
        }
        _ => panic!("Expected Converged"),
    }
}

#[test]
fn test_infeasible_detection() {
    // Minimize x^2 subject to x = 3 AND x = 5 (contradictory).
    let mut b = OptimizationBuilder::new();
    let px = b.add_variable(4.0);
    b.set_objective(QuadraticObj);
    b.add_equality_constraint(/* x = 3 */);
    b.add_equality_constraint(/* x = 5 */);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    // Accept either Infeasible or NotConverged: the penalty method cannot
    // distinguish true infeasibility from slow convergence — both exhaust
    // iterations with high constraint violation. A tighter test would run
    // a feasibility check (minimize ||h(x)||^2 separately) and confirm
    // the violation does not decrease, but that requires additional solver
    // infrastructure beyond the penalty method.
    assert!(matches!(
        result,
        OptimizationResult::Infeasible { .. }
            | OptimizationResult::NotConverged { .. }
    ));
    // Additionally verify that constraint violation is non-trivially large.
    let cv = result.constraint_violation().unwrap_or(0.0);
    assert!(cv > 1e-4, "Expected large constraint violation for infeasible system, got {cv}");
}
```

**Expected failure:** Diagnostics fields not populated.

**Green:** Populate all `OptimizationResult` fields during solve. Add
post-solve feasibility check: if `constraint_violation > constraint_tolerance`
after max iterations, return `Infeasible` (acknowledged limitation: this
heuristic does not guarantee true infeasibility detection; it only signals
that the solver gave up trying to satisfy constraints).

**Refactor:** Ensure `Display` impl gives a useful summary. The
`constraint_violation()` convenience method was added to `OptimizationResult`
in Section 2.4.

---

## 7. Property Tests

### 7.1 Gradient Correctness via Finite Differences

This is the most critical property test. It mirrors the existing
`check_jacobian_fd` pattern from `tests/common/mod.rs`.

```rust
// tests/optimization_property_tests.rs

use proptest::prelude::*;
use solverang::optimization::Objective;

/// Central finite difference verification for an objective's gradient.
///
/// For each variable j, checks:
///   |∂f/∂x_j (analytical) - (f(x+h*e_j) - f(x-h*e_j)) / (2h)| < tol * (1 + max(|fd|, |ana|))
fn check_gradient_fd(
    obj: &dyn Objective,
    x: &[f64],
    eps: f64,
    tol: f64,
) -> bool {
    let n = obj.variable_count();
    let analytical = obj.gradient(x);
    assert_eq!(analytical.len(), n);

    for j in 0..n {
        let h = eps * (1.0 + x[j].abs());

        let mut x_plus = x.to_vec();
        x_plus[j] += h;
        let f_plus = obj.value(&x_plus);

        let mut x_minus = x.to_vec();
        x_minus[j] -= h;
        let f_minus = obj.value(&x_minus);

        let fd = (f_plus - f_minus) / (2.0 * h);
        let ana = analytical[j];

        let error = (fd - ana).abs();
        let scale = 1.0 + fd.abs().max(ana.abs());
        if error >= tol * scale {
            return false;
        }
    }
    true
}

/// Central finite difference verification for an objective's Hessian.
///
/// For each pair (i,j), checks the Hessian against finite differences
/// of the gradient: H[i,j] ≈ (∇f(x+h*e_j)[i] - ∇f(x-h*e_j)[i]) / (2h)
fn check_hessian_fd(
    obj: &dyn Objective,
    x: &[f64],
    eps: f64,
    tol: f64,
) -> bool {
    let n = obj.variable_count();
    let hessian_triplets = match obj.hessian(x) {
        Some(h) => h,
        None => return true, // No analytical Hessian to check.
    };

    // Build dense Hessian from triplets.
    let mut h_analytical = vec![vec![0.0; n]; n];
    for (i, j, val) in &hessian_triplets {
        h_analytical[*i][*j] += val;
    }

    // Finite difference Hessian from gradient.
    for j in 0..n {
        let h = eps * (1.0 + x[j].abs());

        let mut x_plus = x.to_vec();
        x_plus[j] += h;
        let g_plus = obj.gradient(&x_plus);

        let mut x_minus = x.to_vec();
        x_minus[j] -= h;
        let g_minus = obj.gradient(&x_minus);

        for i in 0..n {
            let fd = (g_plus[i] - g_minus[i]) / (2.0 * h);
            let ana = h_analytical[i][j];

            let error = (fd - ana).abs();
            let scale = 1.0 + fd.abs().max(ana.abs());
            if error >= tol * scale {
                return false;
            }
        }
    }
    true
}
```

### 7.2 Property: Gradient Matches Finite Differences at Random Points

```rust
/// Strategy: generate random points in [-10, 10]^n.
fn arb_point(n: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-10.0_f64..10.0_f64, n)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn prop_rosenbrock_gradient_correct(x in arb_point(2)) {
        let obj = RosenbrockObjective; // unit struct — no fields needed
        prop_assert!(
            check_gradient_fd(&obj, &x, 1e-7, 1e-4),
            "Gradient mismatch at x = {:?}", x
        );
    }

    #[test]
    fn prop_rosenbrock_hessian_correct(x in arb_point(2)) {
        let obj = RosenbrockObjective; // unit struct — no fields needed
        prop_assert!(
            check_hessian_fd(&obj, &x, 1e-5, 1e-3),
            "Hessian mismatch at x = {:?}", x
        );
    }

    #[test]
    fn prop_weighted_objective_gradient_correct(
        x in arb_point(2),
        w1 in 0.01_f64..10.0,
        w2 in 0.01_f64..10.0,
    ) {
        let obj = WeightedObjective {
            objectives: vec![
                (w1, Box::new(DistToA { ax: 1.0, ay: 2.0 })),
                (w2, Box::new(DistToB { bx: 5.0, by: 3.0 })),
            ],
        };
        prop_assert!(
            check_gradient_fd(&obj, &x, 1e-7, 1e-4),
            "Weighted gradient mismatch at x = {:?}", x
        );
    }
}
```

### 7.3 Property: Penalty Objective Gradient Correct

```rust
proptest! {
    #[test]
    fn prop_penalty_objective_gradient(
        x in arb_point(2),
        mu in 0.1_f64..1000.0,
    ) {
        // PenaltyObjective wraps the original objective + mu/2 * sum(h_i^2).
        // Its gradient must match FD.
        let base = DistanceToOrigin;
        let constraint = /* x + y - 1 = 0 */;
        let penalty = PenaltyObjective::new(base, vec![constraint], mu);

        prop_assert!(
            check_gradient_fd(&penalty, &x, 1e-7, 1e-4),
            "Penalty gradient mismatch at mu={mu}, x={x:?}"
        );
    }
}
```

### 7.4 Property: InequalityConstraint Jacobian Correct

```rust
/// Reuses the existing check_jacobian_fd pattern from tests/common/mod.rs,
/// adapted for InequalityConstraint.
fn check_inequality_jacobian_fd(
    constraint: &dyn InequalityConstraint,
    store: &ParamStore,
    eps: f64,
    tol: f64,
) -> bool {
    // Same central FD logic as Constraint Jacobian verification.
    // ...
}

proptest! {
    #[test]
    fn prop_sum_leq_jacobian(
        px_val in -10.0_f64..10.0,
        py_val in -10.0_f64..10.0,
    ) {
        // Allocate params in a real ParamStore using the sentinel opt entity.
        let opt_entity = EntityId::new(u32::MAX, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(px_val, opt_entity);
        let py = store.alloc(py_val, opt_entity);
        let constraint = SumLeq { px, py };
        prop_assert!(check_inequality_jacobian_fd(&constraint, &store, 1e-7, 1e-4));
    }
}
```

### 7.5 Exhaustive Property Tests (Ignored, for Overnight Runs)

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored
    fn prop_rosenbrock_gradient_exhaustive(x in arb_point(2)) {
        let obj = RosenbrockObjective; // unit struct — no fields needed
        prop_assert!(check_gradient_fd(&obj, &x, 1e-7, 1e-5));
    }

    #[test]
    #[ignore]
    fn prop_rosenbrock_hessian_exhaustive(x in arb_point(2)) {
        let obj = RosenbrockObjective; // unit struct — no fields needed
        prop_assert!(check_hessian_fd(&obj, &x, 1e-5, 1e-4));
    }
}
```

---

## 8. Integration Test Strategy

### 8.1 Round-Trip: Build, Solve, Verify

Each integration test follows the pattern:

1. Construct problem via `OptimizationBuilder`
2. Solve
3. Verify solution against known analytical answer
4. Verify KKT conditions (gradient + constraint Jacobian^T * lambda = 0)
5. Verify constraint satisfaction

```rust
// tests/optimization_integration.rs

/// Helper to verify first-order KKT conditions at a solution.
fn verify_kkt(
    problem: &OptimizationProblem,
    result: &OptimizationResult,
    tol: f64,
) {
    let sol = result.solution().unwrap();

    // 1. Gradient of Lagrangian should be near zero
    //    (for unconstrained: just gradient of objective)
    let grad = problem.objective.gradient(sol);
    let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

    if problem.equality_constraints.is_empty()
        && problem.inequality_constraints.is_empty()
    {
        assert!(
            grad_norm < tol,
            "KKT violation: ||∇f|| = {grad_norm} > {tol}"
        );
    }

    // 2. Equality constraints satisfied
    // (More detailed KKT checking with multipliers omitted for brevity)
}
```

### 8.2 Integration with Existing Test Problems

The existing MINPACK test problems express `F(x) = 0` as a least-squares
problem `min ||F(x)||^2`. We can convert any `Problem` into an
`Objective` for the optimization framework:

```rust
/// Adapter: convert a `Problem` (residual-based) into an `Objective`
/// that minimizes ||F(x)||^2.
struct LeastSquaresObjective<P: Problem> {
    problem: P,
}

impl<P: Problem> Objective for LeastSquaresObjective<P> {
    fn name(&self) -> &str {
        self.problem.name()
    }

    fn variable_count(&self) -> usize {
        self.problem.variable_count()
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.problem.sum_of_squares(x)
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        // ∇(||F||^2) = 2 * J^T * F
        let r = self.problem.residuals(x);
        let jac = self.problem.jacobian_dense(x);
        let n = self.problem.variable_count();
        let mut grad = vec![0.0; n];
        for i in 0..r.len() {
            for j in 0..n {
                grad[j] += 2.0 * jac[i][j] * r[i];
            }
        }
        grad
    }

    fn hessian(&self, x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
        // Gauss-Newton approximation: H ≈ 2 * J^T * J.
        // Only lower-triangular entries (i >= j) are stored.
        let jac = self.problem.jacobian_dense(x);
        let n = self.problem.variable_count();
        let m = self.problem.residual_count();
        let mut triplets = Vec::new();
        for i in 0..n {
            for j in 0..=i { // lower triangle only
                let mut val = 0.0;
                for k in 0..m {
                    val += 2.0 * jac[k][i] * jac[k][j];
                }
                if val.abs() > 1e-15 {
                    triplets.push((i, j, val));
                }
            }
        }
        Some(triplets)
    }
}
```

### 8.3 MINPACK Problems as Optimization Tests

```rust
#[test]
fn test_rosenbrock_as_optimization() {
    use solverang::test_problems::Rosenbrock;

    let problem = Rosenbrock;
    let obj = LeastSquaresObjective { problem };

    let mut b = OptimizationBuilder::new();
    for &val in &Rosenbrock.initial_point(1.0) {
        b.add_variable(val);
    }
    b.set_objective(obj);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    assert!((sol[0] - 1.0).abs() < 1e-4);
    assert!((sol[1] - 1.0).abs() < 1e-4);
    assert!(result.objective_value().unwrap() < 1e-8);
}

#[test]
fn test_powell_singular_as_optimization() {
    use solverang::test_problems::PowellSingular;

    let problem = PowellSingular;
    let obj = LeastSquaresObjective { problem };

    let mut b = OptimizationBuilder::new();
    for &val in &PowellSingular.initial_point(1.0) {
        b.add_variable(val);
    }
    b.set_objective(obj);

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    for &v in sol {
        assert!(v.abs() < 1e-3);
    }
}

/// Run all MINPACK least-squares problems through the optimization solver.
///
/// Note: `LeastSquaresObjective<P: Problem>` is generic over a concrete type.
/// `least_squares_problem()` returns `Box<dyn Problem>`, which is a trait object
/// and not directly usable as `P`. Two approaches are possible:
///
/// Option A: make `LeastSquaresObjective` hold `Box<dyn Problem>`:
/// ```rust
/// struct LeastSquaresObjective { problem: Box<dyn Problem> }
/// ```
///
/// Option B: write out each concrete problem type explicitly (Rosenbrock, etc.)
/// in individual test functions.
///
/// Option A is preferred for a sweep test. The implementation below uses it.
#[test]
fn test_all_minpack_as_optimization() {
    use solverang::test_problems::least_squares_problem;

    /// Boxed variant for use with trait objects.
    struct LeastSquaresObjectiveBoxed { problem: Box<dyn Problem> }

    impl Objective for LeastSquaresObjectiveBoxed {
        fn name(&self) -> &str { self.problem.name() }
        fn variable_count(&self) -> usize { self.problem.variable_count() }
        fn value(&self, x: &[f64]) -> f64 { self.problem.sum_of_squares(x) }
        fn gradient(&self, x: &[f64]) -> Vec<f64> {
            let r = self.problem.residuals(x);
            let jac = self.problem.jacobian_dense(x);
            let n = self.problem.variable_count();
            let mut grad = vec![0.0; n];
            for i in 0..r.len() {
                for j in 0..n {
                    grad[j] += 2.0 * jac[i][j] * r[i];
                }
            }
            grad
        }
    }

    for i in 1..=18 {
        let problem = least_squares_problem(i).unwrap();
        let x0 = problem.initial_point(1.0);
        let obj = LeastSquaresObjectiveBoxed { problem };

        let mut b = OptimizationBuilder::new();
        for &val in &x0 {
            b.add_variable(val);
        }
        b.set_objective(obj);
        // Solve and check convergence — specific tolerance depends on problem.
        // Some MINPACK problems have non-zero optimal residuals (data fitting).
        let problem_ref = least_squares_problem(i).unwrap();
        let expected_norm = problem_ref.expected_residual_norm();
        let result = b.build().unwrap().solve(OptimizationConfig::default());
        assert!(
            result.is_converged(),
            "MINPACK problem {} did not converge", i
        );
        if let Some(expected) = expected_norm {
            let actual = result.objective_value().unwrap().sqrt(); // sqrt(||F||^2)
            assert!(
                (actual - expected).abs() < 1e-4,
                "MINPACK problem {} objective mismatch: got {}, expected {}",
                i, actual, expected
            );
        }
    }
}
```

### 8.4 Sketch2D Optimization Integration

```rust
#[test]
fn test_sketch2d_optimization_closest_point_on_circle() {
    // Given a circle centered at origin with radius 5,
    // find the point on the circle closest to (10, 0).
    // Answer: (5, 0).
    let mut sketch = Sketch2DBuilder::new();
    let origin = sketch.add_fixed_point(0.0, 0.0);
    let p = sketch.add_point(3.0, 1.0); // Initial guess
    sketch.constrain_distance(origin, p, 5.0);

    let mut opt = OptimizationBuilder::new();
    opt.import_constraint_system(sketch.system());
    opt.set_objective(DistToTarget { target: [10.0, 0.0] });

    let problem = opt.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    assert!((sol[0] - 5.0).abs() < 1e-5);
    assert!(sol[1].abs() < 1e-5);
}

#[test]
fn test_sketch2d_optimization_max_area_rectangle() {
    // Maximize area of a rectangle inscribed in a circle of radius R.
    // Equivalently, minimize -w*h subject to w^2 + h^2 = (2R)^2.
    // Solution: w = h = R*sqrt(2).
    let r = 5.0;
    let mut b = OptimizationBuilder::new();
    let w = b.add_bounded_variable(3.0, 0.0, 2.0 * r);
    let h = b.add_bounded_variable(4.0, 0.0, 2.0 * r);

    // With the array-indexed Objective interface, no ParamId fields are needed.
    // x[0] = w, x[1] = h by insertion order.
    struct NegArea;
    impl Objective for NegArea {
        fn name(&self) -> &str { "NegArea" }
        fn variable_count(&self) -> usize { 2 }
        fn value(&self, x: &[f64]) -> f64 { -x[0] * x[1] }
        fn gradient(&self, x: &[f64]) -> Vec<f64> { vec![-x[1], -x[0]] }
        fn hessian(&self, _x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
            // f(w,h) = -w*h. Lower triangle only (row >= col).
            // d^2f/dw^2 = 0, d^2f/(dh dw) = -1, d^2f/dh^2 = 0.
            // Only the off-diagonal lower-triangle entry is non-zero.
            Some(vec![(1, 0, -1.0)])
        }
    }

    b.set_objective(NegArea);
    // Diagonal^2 = w^2 + h^2 = (2R)^2 — but using Sketch2D distance constraint
    // between origin and the corner point (w/2, h/2), distance = R.
    // Simplified: add custom equality constraint w^2 + h^2 = 4*R^2.
    // NOTE: this equality constraint is NOT added below, making the example
    // incomplete. Without it, the optimizer will find w=h=10 (the bound
    // corner), not w=h=R*sqrt(2). The TODO is to add the constraint.
    // TODO: implement DiagonalConstraint { pw: w, ph: h, target_sq: 4.0*r*r }

    // TODO: add DiagonalConstraint here before calling build()

    let problem = b.build().unwrap();
    let result = problem.solve(OptimizationConfig::default());

    // NOTE: without the diagonal equality constraint the test below will fail.
    // It is left as a placeholder showing the intended assertion once the
    // constraint is added.
    assert!(result.is_converged());
    let sol = result.solution().unwrap();
    let expected = r * std::f64::consts::SQRT_2;
    assert!((sol[0] - expected).abs() < 1e-3);
    assert!((sol[1] - expected).abs() < 1e-3);
}
```

---

## 9. Extending the MINPACK Suite

### 9.1 Adding Optimization Test Problems

The existing test problem suite under `src/test_problems/` uses the `Problem`
trait (residual-based). For optimization, we add a parallel module:

```
src/
  optimization/
    test_problems/
      mod.rs
      rosenbrock.rs          # f(x,y) = (1-x)^2 + 100(y-x^2)^2
      beale.rs               # f(x,y) = sum((1.5 - x + xy^i)^2)
      booth.rs               # f(x,y) = (x+2y-7)^2 + (2x+y-5)^2
      himmelblau.rs          # f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2
      rastrigin.rs           # f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
      sphere.rs              # f(x) = sum(x_i^2)
      styblinski_tang.rs     # f(x) = sum(x_i^4 - 16x_i^2 + 5x_i) / 2
      constrained_rosen.rs   # Rosenbrock + disk constraint x^2+y^2 <= 2
      hs_problems.rs         # Hock-Schittkowski test suite (constrained)
```

### 9.2 The `OptimizationTestProblem` Trait

```rust
// src/optimization/test_problems/mod.rs

/// A standard optimization test problem with known solution.
pub trait OptimizationTestProblem: Objective {
    /// Standard starting point, optionally scaled.
    fn initial_point(&self, factor: f64) -> Vec<f64>;

    /// Known optimal solution (if available).
    fn known_solution(&self) -> Option<Vec<f64>>;

    /// Known optimal objective value.
    fn known_optimal_value(&self) -> Option<f64>;

    /// Equality constraints for this problem (empty for unconstrained).
    fn equality_constraints(&self) -> Vec<Box<dyn Constraint>> {
        vec![]
    }

    /// Inequality constraints for this problem (empty for unconstrained).
    fn inequality_constraints(&self) -> Vec<Box<dyn InequalityConstraint>> {
        vec![]
    }

    /// Variable bounds (empty for unbounded).
    fn bounds(&self) -> Vec<(usize, f64, f64)> {
        vec![]
    }
}
```

### 9.3 Example: Rosenbrock as `OptimizationTestProblem`

```rust
// src/optimization/test_problems/rosenbrock.rs

pub struct RosenbrockOpt;

impl Objective for RosenbrockOpt {
    fn name(&self) -> &str { "Rosenbrock (optimization)" }
    fn variable_count(&self) -> usize { 2 }

    fn value(&self, x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
            200.0 * (x[1] - x[0].powi(2)),
        ]
    }

    fn hessian(&self, x: &[f64]) -> Option<Vec<(usize, usize, f64)>> {
        // Lower-triangular entries only (row >= col).
        let h00 = 2.0 - 400.0 * x[1] + 1200.0 * x[0].powi(2);
        let h10 = -400.0 * x[0]; // row=1, col=0 (lower triangle)
        let h11 = 200.0;
        Some(vec![(0, 0, h00), (1, 0, h10), (1, 1, h11)])
    }
}

impl OptimizationTestProblem for RosenbrockOpt {
    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![-1.2 * factor, factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(vec![1.0, 1.0])
    }

    fn known_optimal_value(&self) -> Option<f64> {
        Some(0.0)
    }
}
```

### 9.4 Constrained Rosenbrock (disk-constrained)

> **REVIEW NOTE:** This problem was previously labelled "Hock-Schittkowski #1
> style" but that is inaccurate. Standard HS #1 has x^2 + y^2 <= 1.5, which
> puts (1,1) outside the feasible region (1+1=2 > 1.5), so the HS #1 solution
> is NOT (1,1). This problem uses radius_sq = 2.0, which places (1,1) exactly
> on the boundary. It is a valid test problem but is NOT HS #1. The HS label
> has been removed to avoid confusion.

```rust
// src/optimization/test_problems/constrained_rosen.rs

/// Minimize Rosenbrock subject to x^2 + y^2 <= 2.
///
/// The global Rosenbrock minimum x* = (1, 1) has x*^2 + y*^2 = 2, which
/// lies exactly on the constraint boundary. The solution is therefore
/// x* = (1, 1) with f* = 0, and the inequality is active at the solution.
pub struct ConstrainedRosenbrock;

impl OptimizationTestProblem for ConstrainedRosenbrock {
    fn initial_point(&self, factor: f64) -> Vec<f64> {
        vec![-1.2 * factor, factor]
    }

    fn known_solution(&self) -> Option<Vec<f64>> {
        Some(vec![1.0, 1.0])
    }

    fn known_optimal_value(&self) -> Option<f64> {
        Some(0.0)
    }

    fn inequality_constraints(&self) -> Vec<Box<dyn InequalityConstraint>> {
        vec![Box::new(DiskConstraint { px: /* allocated by builder */, py: /* allocated by builder */, radius_sq: 2.0 })]
    }
}

/// g(x) = x^2 + y^2 - R^2 <= 0.
///
/// This constraint requires ParamIds for the x and y variables. In standalone
/// optimization these are obtained from `OptimizationBuilder::add_variable()`
/// and stored in the problem. The `OptimizationTestProblem::inequality_constraints()`
/// method needs a way to receive these IDs; one approach is to pass them at
/// construction time via `ConstrainedRosenbrock { px, py }`.
struct DiskConstraint {
    px: ParamId,
    py: ParamId,
    radius_sq: f64,
}

impl InequalityConstraint for DiskConstraint {
    fn name(&self) -> &str { "DiskConstraint" }

    fn param_ids(&self) -> &[ParamId] {
        &[self.px, self.py]
    }

    fn inequality_count(&self) -> usize { 1 }

    fn values(&self, store: &ParamStore) -> Vec<f64> {
        let x = store.get(self.px);
        let y = store.get(self.py);
        vec![x * x + y * y - self.radius_sq]
    }

    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
        let x = store.get(self.px);
        let y = store.get(self.py);
        vec![(0, self.px, 2.0 * x), (0, self.py, 2.0 * y)]
    }
}
```

### 9.5 Megatest for All Optimization Problems

```rust
// tests/optimization_megatest.rs

use solverang::optimization::test_problems::*;

fn test_optimization_problem<P: OptimizationTestProblem + 'static>(
    problem: P,
    tol: f64,
) {
    let x0 = problem.initial_point(1.0);

    // 1. Verify gradient via FD at initial point
    assert!(
        check_gradient_fd(&problem, &x0, 1e-7, 1e-4),
        "Gradient FD check failed for {}",
        problem.name()
    );

    // 2. Verify Hessian via FD if available
    if problem.hessian(&x0).is_some() {
        assert!(
            check_hessian_fd(&problem, &x0, 1e-5, 1e-3),
            "Hessian FD check failed for {}",
            problem.name()
        );
    }

    // 3. Solve
    let mut b = OptimizationBuilder::new();
    for &val in &x0 {
        b.add_variable(val);
    }
    b.set_objective(problem);
    let opt = b.build().unwrap();
    let result = opt.solve(OptimizationConfig::default());

    assert!(
        result.is_converged(),
        "Problem '{}' did not converge: {:?}",
        opt.objective.name(),
        result,
    );

    // 4. Check against known solution
    if let Some(known) = problem.known_solution() {
        let sol = result.solution().unwrap();
        for (i, (&got, &expected)) in sol.iter().zip(known.iter()).enumerate() {
            assert!(
                (got - expected).abs() < tol,
                "Problem '{}': x[{i}] = {got}, expected {expected}",
                problem.name(),
            );
        }
    }

    // 5. Check objective value
    if let Some(expected_val) = problem.known_optimal_value() {
        let val = result.objective_value().unwrap();
        assert!(
            (val - expected_val).abs() < tol,
            "Problem '{}': f(x*) = {val}, expected {expected_val}",
            problem.name(),
        );
    }
}

#[test]
fn megatest_rosenbrock() {
    test_optimization_problem(RosenbrockOpt, 1e-4);
}

#[test]
fn megatest_beale() {
    test_optimization_problem(BealeOpt, 1e-4);
}

// ... etc for all test problems
```

---

## Summary

The TDD roadmap progresses through 12 stages:

| Stage | Capability Added | Key Test |
|-------|-----------------|----------|
| 1 | `Objective` trait, `OptimizationResult` | Trait impl compiles and evaluates |
| 2 | `OptimizationBuilder` variable allocation | Variables tracked correctly |
| 3 | `build()` validation | Missing objective / invalid bounds rejected |
| 4 | Unconstrained gradient descent | Solves shifted quadratic |
| 5 | Newton's method with Hessian | Solves Rosenbrock |
| 6 | Equality constraints (penalty) | min x^2+y^2 s.t. x+y=1 |
| 7 | Inequality constraints (barrier) | Bounded linear program |
| 8 | Variable bounds | Active bound detection |
| 9 | Weighted multi-objective | Weighted centroid problem |
| 10 | ConstraintSystem interop | Import Sketch2D constraints |
| 11 | Augmented Lagrangian upgrade | Tighter convergence in fewer iters |
| 12 | Diagnostics and infeasibility | Gradient norm, infeasible detection |

Each stage starts with a failing test, implements the minimal code to pass it,
then refactors. Property tests run alongside, verifying gradient and Hessian
correctness at every stage via finite differences.
