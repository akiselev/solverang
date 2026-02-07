# Plan 1: Explicit Optimization Objectives

## Status: PROPOSAL
## Priority: High (most value with least architectural disruption)
## Depends on: Plan A (ProblemBase trait)
## Feature flag: `optimization`

---

## 1. Motivation

Solverang currently solves **feasibility** problems: find x such that F(x) = 0. But many
real-world problems are **optimization** problems: minimize f(x) subject to constraints.
The existing least-squares machinery (min ||F(x)||^2) is a special case, but users cannot
express:

- **Minimize cost**: min f(x) s.t. g(x) = 0, h(x) >= 0
- **Maximize throughput**: max f(x) s.t. constraints
- **Multi-objective**: min [f1(x), f2(x)] with Pareto trade-offs
- **Weighted soft constraints**: minimize violation of soft constraints while satisfying
  hard constraints

This plan adds first-class optimization objectives that compose with the existing
constraint infrastructure.

## 2. Design

### 2.1 The `OptimizationProblem` Trait

```rust
/// A problem with an explicit optimization objective.
///
/// Extends ContinuousProblem (which provides equality constraints via residuals)
/// with an objective function to minimize and optional inequality constraints.
pub trait OptimizationProblem: ProblemBase {
    /// Evaluate the objective function f(x) to minimize.
    ///
    /// The solver will seek to minimize this value subject to constraints.
    /// For maximization, negate the objective.
    fn objective(&self, x: &[f64]) -> f64;

    /// Gradient of the objective function: df/dx.
    ///
    /// Returns (variable_index, partial_derivative) pairs.
    /// Only non-zero entries needed.
    fn objective_gradient(&self, x: &[f64]) -> Vec<(usize, f64)>;

    /// Number of equality constraints: g(x) = 0.
    fn equality_count(&self) -> usize;

    /// Evaluate equality constraints. Returns g(x) where g(x) = 0 is required.
    fn equalities(&self, x: &[f64]) -> Vec<f64>;

    /// Jacobian of equality constraints (sparse triplets).
    fn equality_jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)>;

    /// Number of inequality constraints: h(x) >= 0.
    fn inequality_count(&self) -> usize { 0 }

    /// Evaluate inequality constraints. Returns h(x) where h(x) >= 0 is required.
    fn inequalities(&self, x: &[f64]) -> Vec<f64> { vec![] }

    /// Jacobian of inequality constraints (sparse triplets).
    fn inequality_jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> { vec![] }

    /// Initial point.
    fn initial_point(&self, factor: f64) -> Vec<f64>;

    /// Sense of optimization.
    fn sense(&self) -> OptimizationSense { OptimizationSense::Minimize }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationSense {
    Minimize,
    Maximize,
}
```

### 2.2 Solver Approaches

Three solving strategies, ordered by implementation complexity:

#### Strategy A: Penalty Method (simplest, build first)

Convert the optimization problem into an unconstrained one that existing solvers can handle:

```
minimize f(x) + (mu/2) * ||g(x)||^2 + (mu/2) * ||max(0, -h(x))||^2
```

This reuses the existing `Problem` trait by constructing a synthetic residual system:

```rust
/// Converts an OptimizationProblem into a standard Problem via penalty method.
pub struct PenaltyTransform<P: OptimizationProblem> {
    inner: P,
    penalty_weight: f64,  // mu, increased iteratively
}

impl<P: OptimizationProblem> Problem for PenaltyTransform<P> {
    // residuals = [sqrt(f(x)), sqrt(mu)*g_1(x), ..., sqrt(mu)*h_1(x), ...]
    // Solving min ||residuals||^2 = min f(x) + mu*||g||^2 + mu*penalty(h)
}
```

This allows immediate use with LMSolver (which minimizes ||F(x)||^2).

#### Strategy B: Augmented Lagrangian (medium complexity)

More sophisticated than penalty: maintains Lagrange multiplier estimates.

```rust
pub struct AugmentedLagrangianSolver {
    inner_solver: AutoSolver,
    max_outer_iterations: usize,
    penalty_growth: f64,   // typically 10.0
    initial_penalty: f64,  // mu_0
    tolerance: f64,
}

impl AugmentedLagrangianSolver {
    pub fn solve<P: OptimizationProblem>(&self, problem: &P, x0: &[f64]) -> OptimizationResult {
        let mut x = x0.to_vec();
        let mut lambda = vec![0.0; problem.equality_count()];    // equality multipliers
        let mut mu_ineq = vec![0.0; problem.inequality_count()]; // inequality multipliers
        let mut penalty = self.initial_penalty;

        for outer_iter in 0..self.max_outer_iterations {
            // Inner solve: minimize augmented Lagrangian as a Problem
            let al_problem = AugmentedLagrangianProblem {
                inner: problem,
                lambda: &lambda,
                mu_ineq: &mu_ineq,
                penalty,
            };
            let result = self.inner_solver.solve(&al_problem, &x);

            // Update multipliers
            // Update penalty
            // Check convergence of constraints
        }
    }
}
```

#### Strategy C: Interior Point / Barrier Method (advanced)

For problems with many inequality constraints (LP-like structure). Solves a sequence
of barrier problems:

```
minimize f(x) - tau * sum(log(h_i(x)))  s.t. g(x) = 0
```

As tau -> 0, the solution approaches the constrained optimum. This requires solving
a KKT system at each step, which is a Newton-like step — close to existing NR machinery.

### 2.3 Result Type

```rust
/// Result of an optimization solve.
#[derive(Clone, Debug)]
pub enum OptimizationResult {
    /// Found an optimal (or locally optimal) solution.
    Optimal {
        solution: Vec<f64>,
        objective_value: f64,
        iterations: usize,
        constraint_violation: f64,
    },
    /// Solution satisfies constraints but optimality not guaranteed.
    Feasible {
        solution: Vec<f64>,
        objective_value: f64,
        iterations: usize,
        constraint_violation: f64,
    },
    /// Could not find a feasible solution.
    Infeasible {
        best_solution: Vec<f64>,
        constraint_violation: f64,
        iterations: usize,
    },
    /// Solver error.
    Failed { error: SolveError },
}
```

## 3. Integration with Existing Code

### 3.1 Bridge: `Problem` -> `OptimizationProblem`

```rust
/// Wraps a standard feasibility Problem as an optimization problem
/// with a zero objective (pure constraint satisfaction).
impl<T: Problem> OptimizationProblem for FeasibilityWrapper<T> {
    fn objective(&self, _x: &[f64]) -> f64 { 0.0 }
    fn objective_gradient(&self, _x: &[f64]) -> Vec<(usize, f64)> { vec![] }
    fn equality_count(&self) -> usize { self.inner.residual_count() }
    fn equalities(&self, x: &[f64]) -> Vec<f64> { self.inner.residuals(x) }
    // ...
}
```

### 3.2 Bridge: `OptimizationProblem` -> `Problem`

The penalty/AL transforms produce a standard `Problem`, so all existing solvers
(NR, LM, Sparse, Parallel, JIT) can solve the inner subproblems.

### 3.3 Geometric Constraint Integration

The geometry module's `ConstraintSystem` currently implements `Problem` for feasibility.
Adding optimization enables:

```rust
// Future: find the geometry that satisfies all constraints AND minimizes total perimeter
let system = ConstraintSystemBuilder::<2>::new()
    .point(Point2D::new(0.0, 0.0))
    .point(Point2D::new(10.0, 0.0))
    .point(Point2D::new(5.0, 5.0))
    .distance(0, 1, 10.0)
    .distance(1, 2, 8.0)
    .minimize(|points| {  // objective
        perimeter(points[0], points[1], points[2])
    })
    .build_optimization();
```

## 4. File Layout

```
crates/solverang/src/
├── optimization/                  # NEW (feature: "optimization")
│   ├── mod.rs
│   ├── problem.rs                 # OptimizationProblem trait, OptimizationSense
│   ├── result.rs                  # OptimizationResult
│   ├── penalty.rs                 # PenaltyTransform (Strategy A)
│   ├── augmented_lagrangian.rs    # AugmentedLagrangianSolver (Strategy B)
│   └── barrier.rs                 # BarrierSolver (Strategy C, later)
```

## 5. Implementation Phases

### Phase 1: Core trait + Penalty method
- Define `OptimizationProblem` trait
- Implement `PenaltyTransform` that wraps into `Problem`
- Users can immediately use LMSolver on penalty-transformed problems
- Add basic tests with known optima (Rosenbrock with constraints, etc.)

### Phase 2: Augmented Lagrangian solver
- Implement outer-inner loop
- Multiplier updates (equality and inequality)
- Convergence diagnostics
- Test on CUTE/CUTEst-style problems

### Phase 3: Interior Point (optional, future)
- Barrier function
- KKT system formulation
- Mehrotra predictor-corrector for LP/QP special cases

### Phase 4: Integration with geometry and modeling layers
- `ConstraintSystemBuilder::minimize()` / `maximize()`
- Expose in modeling DSL (Plan 5)

## 6. Interaction with Concurrent Work

| Active Work Area      | Impact                                                    |
|-----------------------|-----------------------------------------------------------|
| Existing solvers      | None. Penalty transform produces a standard `Problem`.    |
| Geometry module       | Future extension only (Phase 4). No immediate changes.    |
| Inequality transform  | Complementary. `SlackVariableTransform` handles one form  |
|                       | of inequality; optimization needs a different approach    |
|                       | (penalty/barrier). Both can coexist.                      |
| JIT solver            | Opportunity: JIT-compile objective + gradient evaluation. |
| Auto-Jacobian macros  | Opportunity: `#[auto_gradient]` for objective functions.  |

## 7. Test Strategy

- **Known-optimum problems**: Rosenbrock constrained, simple QP, circle packing
- **Regression**: All existing `Problem`-based tests remain unchanged
- **Bridge tests**: Wrap existing `Problem` as `OptimizationProblem`, verify same results
- **Convergence benchmarks**: Compare penalty vs AL on a standard problem set

## 8. Open Questions

1. **Hessian information**: Should `OptimizationProblem` provide second-order information
   (Hessian of Lagrangian)? Useful for interior point but adds API complexity. Could be
   a separate `SecondOrderOptimizationProblem` extension trait.

2. **Constraint qualification**: Should we detect/warn about violated constraint
   qualifications (LICQ, MFCQ) that can cause solver issues?

3. **Multi-objective**: Should `objective()` return `Vec<f64>` for multi-objective?
   Or is that a separate `MultiObjectiveProblem` trait?

## 9. Acceptance Criteria

- [ ] `OptimizationProblem` trait compiles with clear documentation
- [ ] `PenaltyTransform` wraps into `Problem` and solves with LMSolver
- [ ] At least 3 constrained optimization test problems pass
- [ ] `FeasibilityWrapper` bridge preserves existing solver behavior
- [ ] `OptimizationResult` type with conversion to `SolveResult`
