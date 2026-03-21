# Extending Solverang: Unified Nonlinear Solving + Optimization

## Current State

Solverang handles two problem classes, unified under the `Problem` trait:

| Class | Formulation | Solver |
|-------|-------------|--------|
| Root-finding | F(x) = 0 | Newton-Raphson |
| Nonlinear least-squares | min ‖F(x)‖² | Levenberg-Marquardt |

Both express problems as **residual vectors** F(x) with **Jacobians** J(x). This is clean but
insufficient for general optimization.

### What's Missing

General optimization is: **min f(x) subject to constraints**

| Concept | Current Support | Needed |
|---------|----------------|--------|
| Scalar objective f(x) | No (only residual vectors) | Yes |
| Gradient ∇f(x) | No (only Jacobian of residuals) | Yes |
| Hessian ∇²f(x) | No | Optional (quasi-Newton approximation) |
| Equality constraints c(x) = 0 | Yes (via `Constraint` trait) | Already there |
| Inequality constraints g(x) ≤ 0 | No | Yes |
| Bound constraints l ≤ x ≤ u | Partial (`ParamStore::fix`) | Full bounds |

### Existing Architecture (V1 + V3)

**V1 — Problem-trait solvers:**
- `Problem` trait: `residuals()`, `jacobian()`, `initial_point()`
- Solvers: Newton-Raphson, Levenberg-Marquardt, Auto, Robust, Parallel, Sparse
- Result: `SolveResult { Converged | NotConverged | Failed }`

**V3 — Constraint system (CAD-focused):**
- `Entity` / `Constraint` / `ParamStore` with generational IDs
- 5-phase pipeline: Decompose → Analyze → Reduce → Solve → PostProcess
- Domain builders: Sketch2D, Sketch3D, Assembly
- `Constraint::is_soft()` / `weight()` — ad-hoc soft constraint support

V3 constructs a `Problem` adapter and delegates to V1 solvers.


## Core Mathematical Insight

Root-finding, least-squares, and optimization are all instances of **one** structure:

```
minimize    f(x)
subject to  c(x) = 0,  g(x) ≤ 0,  l ≤ x ≤ u
```

- **Root-finding** F(x) = 0: f(x) = ½‖F(x)‖², no constraints
- **Least-squares**: same, but F has exploitable residual structure
  (J → Jᵀr for gradient, JᵀJ ≈ Hessian via Gauss-Newton)
- **Unconstrained optimization**: direct f(x), no constraints
- **Constrained optimization**: f(x) + constraints

The key difference isn't the problem type — it's **what derivative structure is available**:
- Residual structure (F + J) lets Gauss-Newton/LM exploit the sum-of-squares form
- Scalar structure (f + ∇f) feeds BFGS/L-BFGS
- Both are valid views of the same underlying problem

These are also bidirectionally connected:
1. **Least-squares IS optimization**: f(x) = ½‖F(x)‖², ∇f = Jᵀr, ∇²f ≈ JᵀJ
2. **Unconstrained optimization IS root-finding**: necessary condition ∇f(x) = 0
3. **Constrained optimization IS root-finding** via KKT: ∇f + λᵀ∇c = 0, c(x) = 0


## Design: Unified Problem Trait

**No downstream users. No backward compatibility concerns.**

One trait. Two optional derivative paths. Everything else composes.

```rust
pub trait Problem: Send + Sync {
    fn name(&self) -> &str;
    fn variable_count(&self) -> usize;
    fn initial_point(&self) -> Vec<f64>;

    // ═══ Objective: implement at least one path ═══

    // ── Residual path (root-finding, least-squares) ──
    // Implies f(x) = ½‖F(x)‖². Enables Gauss-Newton/LM.

    fn residual_count(&self) -> usize { 0 }
    fn residuals(&self, _x: &[f64]) -> Vec<f64> { vec![] }
    fn residual_jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> { vec![] }

    // ── Scalar path (general optimization) ──
    // Direct f(x) + ∇f. Enables BFGS/L-BFGS/trust-region.
    // Defaults derived from residuals when present.

    fn objective(&self, x: &[f64]) -> f64 {
        // Default: ½‖F(x)‖²
        let r = self.residuals(x);
        0.5 * r.iter().map(|v| v * v).sum::<f64>()
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        // Default: Jᵀr (gradient of ½‖F‖² from residual structure)
        let r = self.residuals(x);
        let jac = self.residual_jacobian(x);
        let n = self.variable_count();
        let mut grad = vec![0.0; n];
        for (row, col, val) in jac {
            if col < n && row < r.len() {
                grad[col] += val * r[row];
            }
        }
        grad
    }

    fn hessian(&self, _x: &[f64]) -> Option<Vec<(usize, usize, f64)>> { None }

    // ═══ Constraints ═══

    fn bounds(&self) -> Option<Vec<(f64, f64)>> { None }

    fn equality_count(&self) -> usize { 0 }
    fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> { vec![] }
    fn equality_jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> { vec![] }

    fn inequality_count(&self) -> usize { 0 }
    fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> { vec![] }
    fn inequality_jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> { vec![] }

    // ═══ Classification (auto-derived) ═══

    fn has_residual_structure(&self) -> bool { self.residual_count() > 0 }

    fn is_constrained(&self) -> bool {
        self.equality_count() > 0
            || self.inequality_count() > 0
            || self.bounds().is_some()
    }
}
```

### How it works

- A **root-finding problem** implements `residuals()` + `residual_jacobian()` and gets
  `objective()` / `gradient()` for free via defaults.
- An **optimization problem** implements `objective()` + `gradient()` and leaves residual
  methods at their defaults (empty).
- Both can add constraints and bounds.
- Solvers check `has_residual_structure()` to pick algorithms.


## Unified Result Type

```rust
pub enum SolveResult {
    Converged {
        solution: Vec<f64>,
        objective_value: f64,
        gradient_norm: f64,
        residual_norm: Option<f64>,  // only for residual problems
        iterations: usize,
    },
    NotConverged {
        solution: Vec<f64>,
        objective_value: f64,
        gradient_norm: f64,
        residual_norm: Option<f64>,
        iterations: usize,
    },
    Failed {
        error: SolveError,
    },
}
```

`SolveError` gains new variants for optimization-specific failures (e.g., unbounded objective,
constraint violation, etc.).


## V3 Constraint System Changes

### Replace ad-hoc soft constraints with proper classification

Current `is_soft()` / `weight()` become:

```rust
pub enum ConstraintKind {
    Equality,                    // c(x) = 0
    Inequality,                  // c(x) ≤ 0
    Penalty { weight: f64 },     // add weight * c(x)² to objective
}

pub trait Constraint: Send + Sync {
    fn kind(&self) -> ConstraintKind { ConstraintKind::Equality }
    fn id(&self) -> ConstraintId;
    fn name(&self) -> &str;
    fn entity_ids(&self) -> &[EntityId];
    fn param_ids(&self) -> &[ParamId];
    fn equation_count(&self) -> usize;
    fn residuals(&self, store: &ParamStore) -> Vec<f64>;
    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)>;
    // is_soft() and weight() removed
}
```

### Add objective support to V3

```rust
pub trait ObjectiveFunction: Send + Sync {
    fn param_ids(&self) -> &[ParamId];
    fn value(&self, store: &ParamStore) -> f64;
    fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)>;
}
```

`ConstraintSystem` gains:
- `add_objective(obj: Box<dyn ObjectiveFunction>)`
- `add_inequality_constraint(c: Box<dyn Constraint>)`

### ParamStore gets bounds

```rust
impl ParamStore {
    fn set_bounds(&mut self, id: ParamId, lower: f64, upper: f64);
    fn bounds(&self, id: ParamId) -> (f64, f64);  // default: (-∞, +∞)
}
```


## Solver Selection (AutoSolver)

```
has_residuals && no constraints     → NR (square) or LM (overdetermined)
has_residuals && bounds only        → Bounded LM
no residuals && no constraints      → BFGS / L-BFGS
no residuals && bounds only         → L-BFGS-B
equality or inequality constraints  → SQP
```

### New solver implementations needed

| Solver | Algorithm | Best For |
|--------|-----------|----------|
| **BFGS** | Quasi-Newton + Wolfe line search | Small-medium unconstrained |
| **L-BFGS** | Limited-memory BFGS | Large unconstrained (1000+ vars) |
| **L-BFGS-B** | L-BFGS with box constraints | Bound-constrained |
| **Trust-Region** | Trust-region Newton/quasi-Newton | Robust unconstrained |
| **SQP** | Sequential Quadratic Programming | Constrained optimization |

Existing solvers (NR, LM, Parallel, Sparse) adapt to the new Problem trait — they just
use the residual path methods.


## Full Architecture

```
┌──────────────────────────────────────────┐
│     V3: ConstraintSystem                 │
│  Entity + Constraint + ObjectiveFunction │
│  ParamStore (with bounds)                │
│  ConstraintKind: Eq / Ineq / Penalty     │
│  Pipeline: Decompose→Analyze→Reduce→Solve│
└────────────────┬─────────────────────────┘
                 │ constructs
                 ▼
┌──────────────────────────────────────────┐
│     Unified Problem trait                │
│  residual path  ←→  scalar path          │
│  bounds + equality + inequality          │
└────────────────┬─────────────────────────┘
                 │ dispatches to
                 ▼
┌──────────────────────────────────────────┐
│     Solvers                              │
│  Residual:     Newton-Raphson, LM        │
│  Gradient:     BFGS, L-BFGS, TrustRegion │
│  Constrained:  SQP, Aug. Lagrangian      │
│  Meta:         Auto, Parallel, Sparse    │
└──────────────────────────────────────────┘
```


## CAD Kernel Use Cases This Enables

- **Closest point on surface**: min ‖P - S(u,v)‖² — unconstrained BFGS in UV space
- **Surface fitting/approximation**: min Σ‖S(uᵢ,vᵢ) - Pᵢ‖² — LM with bound constraints on UV
- **Mesh quality optimization**: min quality_metric s.t. vertices on surface — constrained SQP
- **Fillet/chamfer sizing**: min deviation s.t. tangency constraints — constrained
- **Tolerance-based operations**: min error s.t. tolerance bounds — L-BFGS-B


## Implementation Phases

### Phase 1 — Core unification
- Redesign `Problem` trait (unified, two derivative paths)
- Redesign `SolveResult` (objective_value, gradient_norm)
- Migrate Newton-Raphson and LM to new trait
- Migrate all test problems (MINPACK, NIST)
- BFGS solver (most generally useful new algorithm)
- Standard optimization test problems (Rosenbrock, Beale, Booth, etc.)

### Phase 2 — Scale + bounds
- L-BFGS for large problems
- L-BFGS-B for bound-constrained
- Trust-region method
- ParamStore bounds support

### Phase 3 — Constraints + V3 integration
- SQP solver
- Augmented Lagrangian
- `ConstraintKind` enum (replace is_soft/weight)
- `ObjectiveFunction` trait for V3
- V3 pipeline integration

### Phase 4 — Validation
- CUTEst-style standard test suite
- Geometric kernel benchmarks (closest point, surface fitting)
- Jacobian/gradient verification utilities


## Design Decision: Single Trait vs. Capability Traits

**Chosen: single unified trait.**

Arguments for:
- One trait to learn and implement
- Solver APIs take `&dyn Problem` uniformly
- Default implementations express the math directly (Jᵀr, ½‖F‖²)
- No diamond inheritance or dynamic dispatch complexity
- Auto-classification via `has_residual_structure()` / `is_constrained()`

Arguments against (split `Objective` / `Residuals` / `Constrained` traits):
- Each solver gets exactly what it needs (no unused methods)
- Type system enforces contracts at compile time
- Cleaner individual traits

The single-trait approach wins because Rust's trait object system makes multi-trait composition
with dynamic dispatch painful. Runtime classification via `has_residual_structure()` is simple
and the mathematical defaults connect the two paths naturally.
