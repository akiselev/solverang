# Solverang Optimization Extension: Architectural Synthesis

## Executive Summary

Solverang's architecture extends naturally to optimization. The key insight: **optimization's inner loop IS root-finding** — the existing NR/LM solvers become the workhorse inside optimization algorithms (ALM, SQP, IPM). This means the extension is additive, not transformative.

The design adds three concepts — objectives, inequality constraints, and dual variables (Lagrange multipliers) — without modifying any existing trait. The macro system gains `#[objective]` and `#[inequality]` attributes alongside `#[residual]`, generating gradients and optionally Hessians via the same `Expr::differentiate()` that already exists.

Implementation order (this document): Phase 1 includes both unconstrained (BFGS/L-BFGS) and constrained (ALM with NR/LM inner loop), then L-BFGS-B with bounds, then SQP, then IPM.

<!-- Decision: Phase 1 includes BOTH BFGS (unconstrained) AND ALM (constrained). Doc 07 Phase 1 and this document are now aligned. -->

---

## 1. Unified Abstraction Model

### The Mathematical Bridge

Every root-finding problem `F(x) = 0` is equivalent to `min ||F(x)||²`. Conversely, every constrained optimization problem's KKT conditions form a (structured) root-finding system. This bidirectional relationship is the architectural foundation:

```
Root-Finding:  F(x) = 0              → Newton-Raphson, Levenberg-Marquardt
Optimization:  min f(x) s.t. g=0, h≤0  → KKT system → root-finding inner loop
```

The existing solver infrastructure handles the inner loop. A new outer layer manages objectives, multipliers, and algorithmic strategy.

### New Traits (Additive, Non-Breaking)

<!-- Decision: Two-level design resolved. System-level uses ParamStore-based interface (InequalityFn, Objective with id()/param_ids()/sparse gradient). Problem-level uses array-based interface (ObjectiveFunction, InequalityConstraint in OptimizationBuilder). Adapters convert between levels. ConstraintHessian method is residual_hessian(equation_row: usize, store: &ParamStore) — index first, store second. -->

```rust
/// Scalar objective function to minimize.
///
/// Uses `ParamStore` for value/gradient (consistent with `Constraint` trait).
/// Hessian is separated into `ObjectiveHessian` to make BFGS fallback the
/// default rather than a conscious opt-out.
pub trait Objective: Send + Sync {
    fn id(&self) -> ObjectiveId;
    fn name(&self) -> &str;
    fn param_ids(&self) -> &[ParamId];
    fn value(&self, store: &ParamStore) -> f64;
    /// Sparse gradient: (param_id, df/dp) pairs. Only non-zero entries needed.
    fn gradient(&self, store: &ParamStore) -> Vec<(ParamId, f64)>;
}

/// Optional Hessian extension (opt-in for second-order methods).
/// When absent, solvers use BFGS/L-BFGS quasi-Newton approximation.
pub trait ObjectiveHessian: Objective {
    fn hessian_entries(&self, store: &ParamStore) -> Vec<(ParamId, ParamId, f64)>;
}

/// Optional Hessian extension for existing Constraint trait.
pub trait ConstraintHessian: Constraint {
    fn residual_hessian(
        &self,
        equation_row: usize,
        store: &ParamStore,
    ) -> Option<Vec<(ParamId, ParamId, f64)>>;
}

/// System-level inequality constraint h(x) ≤ 0.
/// Parallels the existing `Constraint` trait (ParamId-based, ParamStore-reading).
pub trait InequalityFn: Send + Sync {
    fn id(&self) -> ConstraintId;
    fn name(&self) -> &str;
    fn entity_ids(&self) -> &[EntityId];
    fn param_ids(&self) -> &[ParamId];
    fn inequality_count(&self) -> usize;
    /// Evaluate h(x). Each element should be ≤ 0 when the constraint is satisfied.
    fn values(&self, store: &ParamStore) -> Vec<f64>;
    fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)>;
}
```

**Why these choices:**
- `Objective` returns `(ParamId, f64)` gradient entries — consistent with `Constraint::jacobian` returning `(usize, ParamId, f64)`
- `ObjectiveHessian` is a separate trait (not a method on `Objective`) so that BFGS fallback is the default, not a conscious opt-out
- `InequalityFn` uses `values` (should be ≤ 0) mirroring `Constraint::residuals` (should be = 0); the name `InequalityFn` avoids collision with the existing array-based `InequalityConstraint`
- The existing `Constraint` trait maps to equality constraints with zero modifications

### Two-Level Design (Preserving Existing Infrastructure)

The codebase already has a Problem-level `InequalityConstraint` trait (in `constraints/inequality.rs`) with array-indexed interface. This is preserved:

| Level | Equality | Inequality | Objective |
|-------|----------|------------|-----------|
| **Problem-level** (standalone, `&[f64]`) | `Problem` trait | `InequalityConstraint` + `InequalityProblem` (existing) | NEW: `ObjectiveFunction` trait |
| **System-level** (ParamId, `ParamStore`) | `Constraint` trait | NEW: `InequalityFn` | NEW: `Objective` trait |

Adapters convert between levels when needed.

### New ID Types

```rust
/// Generational index for an objective function.
/// Uses the project-wide `Generation = u32` type alias (see id.rs).
pub struct ObjectiveId { pub(crate) index: u32, pub(crate) generation: Generation }

/// Semantic address for a Lagrange multiplier (dual variable).
/// No generational bookkeeping — multipliers are recomputed each solve.
pub struct MultiplierId { pub constraint_id: ConstraintId, pub equation_row: usize }
```

Multipliers are stored in a separate `MultiplierStore` (NOT in `ParamStore`) because they lack entity ownership, fixed/free semantics, and change-tracking that parameters have.

---

## 2. Macro Extension

### Core Insight: Zero Changes to `Expr::differentiate()`

The existing `differentiate(&self, var_id: usize) -> Expr` returns the same type it operates on. This means:

```rust
// Second derivative = just call differentiate twice with intermediate simplification
let gradient_i = expr.differentiate(var_i).simplify();
let hessian_ij = gradient_i.differentiate(var_j).simplify();
```

No new AST nodes, no new differentiation rules. The challenge is purely expression-size management.

### New Attributes

```rust
// Existing (unchanged):
#[auto_jacobian(array_param = "x")]
impl MyConstraint {
    #[residual]  // generates jacobian_entries()
    fn residual(&self, x: &[f64]) -> f64 { ... }
}

// New general-purpose attribute:
#[auto_diff(array_param = "x")]
impl MyObjective {
    #[objective]  // generates gradient() only (BFGS default)
    fn value(&self, x: &[f64]) -> f64 { ... }
}

#[auto_diff(array_param = "x")]
impl MyObjectiveExact {
    #[objective(hessian = "exact")]  // generates gradient() + hessian_entries() (opt-in, N≤30)
    fn value(&self, x: &[f64]) -> f64 { ... }
}

#[auto_diff(array_param = "x")]
impl MyInequality {
    #[inequality]  // generates values() + jacobian()
    fn value(&self, x: &[f64]) -> f64 { ... }
}
```

`#[auto_jacobian]` stays as a backward-compatible alias for `#[auto_diff]`.

### Hessian Strategy

<!-- Decision: #[objective] generates gradient only by default. #[objective(hessian = "exact")] opts in to macro-generated exact Hessian for N≤30. BFGS is the default optimization method. Doc 05's concern about AST explosion is addressed by making exact Hessians opt-in rather than default. -->

`#[objective]` generates a gradient method only. BFGS is the default solver and needs only first derivatives. To opt in to macro-generated exact Hessians for small problems:

| Problem Size | Default Hessian Strategy | Override |
|---|---|---|
| N ≤ 30 | BFGS approximation (default) | `#[objective(hessian = "exact")]` to generate exact symbolic Hessian |
| N > 30 | BFGS approximation | `#[objective(hessian = "exact")]` possible but not recommended |
| N > 50 | L-BFGS approximation | `#[objective(hessian = "hvp")]` for Hessian-vector products |

The N=30 threshold for `hessian = "exact"` is an **unvalidated estimate** requiring empirical measurement of compile times before production use.

### Required Simplification Enhancements

The existing `simplify()` handles `0+x`, `0*x`, `1*x`, constant folding, etc. Second derivatives need:

1. **`Sub(x, x) → 0`** when structurally equal (product rule produces these)
2. **`Div(x, x) → 1`** when structurally equal (quotient rule produces these)
3. **Multi-pass simplification** to fixed point (one pass may reveal new opportunities)

This requires implementing structural equality comparison on `Expr`.

### Compile-Time Warning for Non-Smooth Expressions

The macro should detect `Abs` in `#[objective]` expressions and emit:
```
warning: `abs()` in objective creates non-differentiable point at zero.
         Consider using a smooth approximation (Huber loss, softabs).
```

### JIT Extensions

New opcodes:
- `StoreGradient(var_idx, register)` — write gradient entry
- `StoreHessian(row_idx, col_idx, register)` — write Hessian entry (upper triangle)
- `StoreObjective(register)` — write scalar objective value
- `StoreInequalityValue(ineq_idx, register)` — write inequality value

New compiled type: `CompiledOptimization` parallel to existing `CompiledConstraints`.

---

## 3. Pipeline & System Design

### Unified ConstraintSystem (Extended, Not Replaced)

```rust
impl ConstraintSystem {
    // Existing (unchanged):
    pub fn solve(&mut self) -> SystemResult { ... }

    // New: optimization entry points
    pub fn set_objective(&mut self, objective: Box<dyn Objective>) { ... }
    pub fn clear_objective(&mut self) { ... }
    pub fn add_inequality(&mut self, ineq: Box<dyn InequalityFn>) { ... }
    pub fn set_opt_config(&mut self, config: OptimizationConfig) { ... }
    pub fn optimize(&mut self) -> OptimizationResult { ... }

    // New: post-solve multiplier queries
    pub fn multiplier(&self, constraint: ConstraintId) -> Option<&[f64]> { ... }
    pub fn multipliers(&self) -> &MultiplierStore { ... }
}
```

When no objective is set, `optimize()` returns `OptimizationError::NoObjective`. When no inequalities exist, it is equality-constrained optimization. The `solve()` method is completely unchanged — calling `solve()` on a system with an objective set ignores the objective and solves constraints only. `solve()` and `optimize()` are distinct entry points with distinct semantics.

### Extended Pipeline

<!-- Decision: Pipeline order is Classify → Decompose → Analyze → MultiplierInit → Solve → PostProcess. Classify runs first so decomposition is objective-aware. -->

```
EXISTING:    Decompose --> Analyze --> Reduce --> Solve --> PostProcess

OPTIMIZATION:
  Classify --> Decompose* --> Analyze* --> Reduce --> MultiplierInit --> Solve* --> PostProcess*
      |              |             |                       |               |              |
      v              v             v                       v               v              v
 ProblemClass   OptDecomp    OptAnalysis            MultiplierState   OptSolution    OptResult
```

Phases marked with `*` have modified behavior in optimization mode. `Classify` and `MultiplierInit` are new and only activate when an objective is present.

- **Classify** (new, Phase 0): Determines problem type (pure constraints, unconstrained opt, equality-constrained opt, mixed-constrained opt). Zero-cost when no objective present. Must run before Decompose so that objective-coupled variables are correctly grouped.
- **MultiplierInit** (new): Initializes Lagrange multipliers. Strategies: zero, least-squares estimate, warm-start from previous solve.
- **Solve** (extended): Dispatches to NR/LM for constraints-only, or ALM/BFGS/SQP/IPM for optimization.

### Decomposition with Objectives

Objectives typically couple all variables, defeating decomposition. Strategy:
1. **Default: monolithic** — all objective-touching params in one cluster
2. **When detectable: partial separability** — if `f(x) = f₁(x₁) + f₂(x₂)`, decompose into independent sub-problems
3. **Constraint-only clusters**: constraints that don't interact with the objective can still be decomposed and solved independently first

### Lagrangian Assembly at Runtime

The Lagrangian `L(x,λ,μ) = f(x) + Σ λᵢgᵢ(x) + Σ μⱼhⱼ(x)` is never materialized as a single expression. Instead, a `LagrangianAssembler` iterates over registered components:

```rust
struct LagrangianAssembler<'a> {
    objective: &'a dyn Objective,
    equalities: &'a [Box<dyn Constraint>],
    inequalities: &'a [Box<dyn InequalityFn>],
    lambda: &'a [f64],  // equality multipliers
    mu: &'a [f64],      // inequality multipliers
}

impl LagrangianAssembler {
    fn hessian_of_lagrangian(&self, store: &ParamStore) -> SparseMatrix {
        // H_L = H_f + Σ λᵢ H_{gᵢ} + Σ μⱼ H_{hⱼ}
        // Each term comes from the corresponding trait's hessian_entries()
        // Falls back to BFGS approximation when exact Hessians unavailable
    }
}
```

This mirrors how the current system assembles the global Jacobian from per-constraint entries.

### Algorithm Selection

```
No objective          → NR/LM (existing path)
Unconstrained         → L-BFGS (gradient-only) or Newton (with Hessian)
Equality-constrained  → ALM with NR/LM inner loop
Inequality-constrained → IPM (log-barrier) or SQP (with QP subproblem)
Mixed                 → SQP or IPM based on problem structure
```

### Inequality Handling: Log-Barrier over Slack Variables

The existing `SlackVariableTransform` converts `g(x) ≥ 0` to `g(x) - s² = 0`. This creates rank-deficient Jacobians when constraints become active (`s → 0` means the Jacobian column `-2s → 0`).

For optimization, use log-barrier (interior point) instead:
```
min f(x) - μ Σ ln(-hⱼ(x))
```
The barrier term is smooth, prevents the iterate from violating inequalities, and μ → 0 recovers the original problem. The existing `SlackVariableTransform` is kept for backward compatibility but not used in the optimization pipeline.

---

## 4. Implementation Order

### Phase 1: Foundation + BFGS + ALM (~1500 LOC)

Phase 1 includes **both** unconstrained optimization (BFGS/L-BFGS) and constrained optimization (ALM with NR/LM inner loop). BFGS is the default solver for unconstrained problems; ALM reuses the existing NR/LM solver for constrained problems. Neither requires Hessian infrastructure.

1. `Objective` trait + `ObjectiveId`
2. `InequalityFn` trait (system-level, `ParamStore`-based)
3. `MultiplierStore` + `MultiplierId`
4. `#[objective]` attribute in macro (gradient generation only; BFGS default)
5. `ConstraintSystem.set_objective()` / `.add_inequality()` / `.optimize()`
6. BFGS solver (unconstrained, gradient-only)
7. ALM solver (outer loop managing multipliers, inner loop = existing LM)
8. `OptimizationResult` type

**Test problems**: Rosenbrock (unconstrained via BFGS), equality-constrained Rosenbrock (ALM), Hock-Schittkowski #1

### Phase 2: L-BFGS-B + Bounds (~1500 LOC)

1. Variable bounds in `ParamStore` (lower/upper per param)
2. L-BFGS-B solver (standalone, no subproblem solver needed)
3. Bounded variable handling in pipeline
4. `#[inequality]` attribute in macro

**Test problems**: Box-constrained optimization, bounded MINPACK problems

### Phase 3: SQP + Exact Hessians (~2500 LOC)

<!-- Decision: BFGS is the default Hessian strategy; exact Hessians are opt-in via #[objective(hessian = "exact")] for N≤30. SQP in Phase 3 uses BFGS Hessian approximation by default. ObjectiveHessian + ConstraintHessian traits are optional extensions. If exact Hessians are used, Expr::differentiate2() must include CSE (see doc 05). -->

1. `ObjectiveHessian` + `ConstraintHessian` traits (unconditional)
2. `Expr::differentiate2()` + simplification enhancements (conditional on empirical Hessian feasibility — see Hessian Strategy note in Section 2)
3. `#[objective]` Hessian generation in macro (conditional — requires CSE pass to be viable)
4. Hessian JIT opcodes (conditional on step 3)
5. SQP solver with Rust QP subproblem solver (Clarabel.rs or OSQP — both surveyed in `06_state_of_the_art.md`)
6. `LagrangianAssembler` for Hessian of Lagrangian

**Test problems**: Full Hock-Schittkowski suite, CUTEst subset

### Phase 4: IPM + Polish (~2000 LOC)

1. Interior point method (log-barrier)
2. Algorithm auto-selection in pipeline
3. Sensitivity analysis (multiplier exposure)
4. Implicit differentiation for parametric optimization
5. Comprehensive benchmarking

**Test problems**: Large-scale constrained problems, COPS benchmark set

---

## 5. TDD Roadmap (12 Stages)

Each stage follows red → green → refactor. The detailed test code, including exact assertions and solve patterns, is in `04_api_and_tdd.md`.

<!-- Decision: Stage 5 = L-BFGS for unconstrained Rosenbrock; Stage 6 = ALM for equality-constrained problems (reusing NR/LM inner loop). No separate penalty method stage. -->

| Stage | Red (Write Test) | Green (Minimal Implementation) |
|-------|-------------------|-------------------------------|
| 1 | `Objective` trait compiles with `ObjectiveId` | Trait + ID type definitions |
| 2 | `OptimizationBuilder` allocates variables | Builder with `ParamStore` delegation |
| 3 | `build()` validation rejects missing objective / bad bounds | Error types + validation checks |
| 4 | Simple quadratic `(x-3)^2` converges (gradient descent) | Steepest descent + Armijo backtracking |
| 5 | Rosenbrock converges (L-BFGS) | L-BFGS solver |
| 6 | Equality-constrained problem converges (ALM) | ALM outer loop + existing LM inner loop |
| 7 | Inequality constraints handled (log-barrier) | Interior point / barrier method |
| 8 | Variable bounds enforced at solution | Projected gradient or bound-as-inequality |
| 9 | Weighted multi-objective converges | `WeightedObjective` scalarization adapter |
| 10 | Sketch2D constraints imported into optimization | `import_constraint_system()` implementation |
| 11 | ALM convergence with bounded `mu` verified | Verify multiplier update correctness |
| 12 | `OptimizationResult` diagnostics populated; infeasibility reported | Diagnostic enrichment |

### Property Tests

Two flavors are needed, matching the two API levels:

- **System-level** (`ParamStore`-based `Objective`): perturb the `ParamStore`, re-evaluate.
- **Problem-level** (array-based `Objective` from `OptimizationBuilder`): perturb the `x: &[f64]` vector.

```rust
// Problem-level gradient verification (matches 04_api_and_tdd.md patterns)
fn check_gradient_fd(obj: &dyn Objective, x: &[f64], eps: f64, tol: f64) -> bool {
    let n = obj.variable_count();
    let analytical = obj.gradient(x);
    assert_eq!(analytical.len(), n);
    for j in 0..n {
        let h = eps * (1.0 + x[j].abs());
        let mut x_plus = x.to_vec(); x_plus[j] += h;
        let mut x_minus = x.to_vec(); x_minus[j] -= h;
        let fd = (obj.value(&x_plus) - obj.value(&x_minus)) / (2.0 * h);
        let scale = 1.0 + fd.abs().max(analytical[j].abs());
        if (fd - analytical[j]).abs() > tol * scale { return false; }
    }
    true
}

// Hessian verification: finite-difference the gradient to get the Hessian
fn check_hessian_fd(obj: &dyn Objective, x: &[f64], eps: f64, tol: f64) -> bool {
    // Only called when obj.hessian(x) returns Some(...)
    // ...
    true
}
```

Convention: 256 cases default, 100k `#[ignore]` exhaustive (matching existing proptests).

### MINPACK Reuse via Adapter

The adapter wraps a `Problem` as a Problem-level `Objective` (array-based). It cannot use the System-level `Objective` because `Problem` has no `ParamStore` or `ParamId` concept.

```rust
/// Converts any existing Problem into an Objective (minimizing 0.5 * ||F(x)||²).
/// Uses the Problem-level (array-based) Objective API from OptimizationBuilder.
struct LeastSquaresObjective<P: Problem> { inner: P }

impl<P: Problem> Objective for LeastSquaresObjective<P> {
    fn name(&self) -> &str { self.inner.name() }
    fn variable_count(&self) -> usize { self.inner.variable_count() }

    fn value(&self, x: &[f64]) -> f64 {
        let r = self.inner.residuals(x);
        r.iter().map(|ri| ri * ri).sum::<f64>() * 0.5
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        // J^T * r  (Jacobian transpose times residual vector)
        let r = self.inner.residuals(x);
        let n = self.inner.variable_count();
        let mut grad = vec![0.0; n];
        for (row, col, jac_val) in self.inner.jacobian(x) {
            grad[col] += jac_val * r[row];
        }
        grad
    }
}
```

This immediately gives 18+ test problems for the optimization path via the existing MINPACK suite.

---

## 6. Per-Algorithm Derivative Requirements

| Algorithm | Objective | Gradient | Hessian | Eq. Constraint | Eq. Jacobian | Ineq. Constraint | Ineq. Jacobian |
|-----------|-----------|----------|---------|----------------|--------------|-------------------|----------------|
| **ALM** | ✓ | ✓ | — | ✓ (existing) | ✓ (existing) | ✓ | ✓ |
| **L-BFGS-B** | ✓ | ✓ | — | — | — | bounds only | — |
| **SQP** | ✓ | ✓ | ✓ (or BFGS) | ✓ | ✓ | ✓ | ✓ |
| **IPM** | ✓ | ✓ | ✓ (or BFGS) | ✓ | ✓ | ✓ | ✓ |
| **Penalty** | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |

"—" means not needed. "✓ (existing)" means the existing `Constraint` trait already provides this.

---

## 7. Key Risks and Mitigations

The following table is derived from `05_risks_and_alternatives.md`, which also provides the two most critical recommendations: (1) do not generate Hessians via macro, and (2) replace slack variables with log-barrier for optimization. Both are reflected in the mitigations below.

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hessian expression blowup (N>30) | Compile-time explosion | BFGS default, exact Hessian opt-in only after empirical validation; HVP via forward-mode AD as alternative |
| `Abs` derivative NaN at zero | Silent solver failure | Compile-time warning in macro; `smooth_abs(x, eps)` utility; improve error messages to distinguish non-smooth from numerical issues |
| Slack variable rank deficiency | Solver divergence at active constraints | Log-barrier (IPM) for optimization path; `SlackVariableTransform` kept for constraint-satisfaction-only use |
| Objective defeats decomposition | Performance regression on large problems | Monolithic fallback + partial separability detection |
| Macro can't handle control flow | Objectives with branches, loops, or dynamic structure fail at compile time | `ManualObjective` escape hatch; longer term: forward-mode AD (dual numbers) as runtime fallback — not optional for non-trivial objectives (see doc 05) |
| Non-smooth objectives (L1, minimax) | Incorrect or undefined derivatives | Out of scope for v1; smooth approximations documented (`log_sum_exp`, `softabs`); improved error messages that name the non-differentiable point |
| Saddle point convergence | Newton's method converges to saddle, not minimum | Hessian positive-definiteness check for any Newton-based optimization step; BFGS maintains positive-definiteness by construction (another argument for BFGS as default) |
| Unbounded problems | Solver iterates to infinity | Gradient norm and step size monitoring; explicit `UnboundedDescent` error after K consecutive steps with growing step size |
| Degenerate constraint qualification (LICQ violated) | Lagrange multipliers not unique; solver may diverge | Post-solve SVD check on constraint Jacobian; warn user when multipliers are likely non-unique |

---

## 8. What Solverang Can Uniquely Offer

Based on the state-of-the-art survey in `06_state_of_the_art.md`, Solverang's combination of:

1. **Compile-time symbolic differentiation** — no mainstream optimization framework generates derivatives at compile time; CasADi and JAX do so at runtime; Enzyme (nightly Rust) is the closest future analog
2. **JIT compilation of derivative code** — CasADi generates C code (offline), Solverang generates Cranelift IR (online); Optim.jl (Julia) achieves comparable JIT-compiled derivatives via the Julia runtime
3. **Graph-aware decomposition coupled to optimization** — solvers such as IPOPT and SNOPT operate on the full problem without structural decomposition; Solverang's cluster-based decomposition is unique among integrated CAD+optimization systems (though multi-block methods like ALADIN address this in distributed settings)
4. **Tight coupling between constraint solver and optimizer** — surveyed CAD tools (SolidWorks, CATIA, FreeCAD) all use black-box optimization over the constraint solver; Spatial CDS's simulation mode is the closest commercial analog
5. **Implicit differentiation available from the existing Jacobian factorization** — the factorized `J` from constraint solving is already available; one back-substitution per design parameter gives `dx/dp`

...positions Solverang uniquely for **parametric design optimization** — identified in `07_integration_patterns.md` as the highest-relevance CAD use case ("HIGHEST" tier).

### Rust Ecosystem Notes (from `06_state_of_the_art.md`)

For Phase 3 QP subproblem solver, two Rust-native candidates exist:
- **Clarabel.rs** (Apache-2.0): interior-point for conic programs; handles QP, SOCP, SDP; LDL factorization on sparse KKT system
- **OSQP** (Apache-2.0): ADMM-based QP with warm-starting; very fast for parametric sequences of related QPs (the SQP use case)

For framework reference: **argmin** (pure Rust) provides well-tested line search and trust-region implementations that could serve as reference implementations or be integrated directly.

When `#[autodiff]` stabilizes in the Rust compiler (Enzyme-based, active as of 2024), Solverang could offer a dual AD pathway: compile-time symbolic AD for constraint systems with known structure, LLVM-level AD for user-defined objectives with arbitrary control flow.

---

## Document Index

| File | Contents |
|------|----------|
| `01_mathematical_architecture.md` | Trait hierarchy, Lagrangian assembly, derivative requirements |
| `02_macro_extensions.md` | `#[objective]`/`#[inequality]`, second-order differentiation, JIT opcodes |
| `03_pipeline_design.md` | Pipeline phases, decomposition, multiplier flow, algorithm selection |
| `04_api_and_tdd.md` | API examples, builder pattern, 12-stage TDD roadmap, property tests |
| `05_risks_and_alternatives.md` | Failure modes, runtime AD comparison, scaling limits |
| `06_state_of_the_art.md` | Solver survey (IPOPT, SNOPT, Knitro, etc.), 2022-2025 papers |
| `07_integration_patterns.md` | CAD use cases, solver interplay, phased rollout |
