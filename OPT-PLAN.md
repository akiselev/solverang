# Optimizer Enhancement Plan

## Overview

Enhance solverang's optimizer from its current state (unconstrained L-BFGS +
equality-constrained ALM) to a production-grade optimization toolkit supporting
bound constraints, inequality constraints, trust-region methods, and
macro-generated Hessians.

The plan adds six capabilities in dependency order: (1) expanded macro functions,
(2) strong Wolfe line search with Armijo fallback, (3) dimension-independent
convergence scaling, (4) L-BFGS-B with bounds in ParamStore, (5) inequality
constraints in ALM via direct penalty, (6) opt-in Hessian generation via
`#[hessian]`, and (7) trust-region solver with auto-selected dogleg/Steihaug-CG.

## Planning Context

### Decision Log

| Decision | Reasoning Chain |
|---|---|
| Bounds in ParamStore | Bounds are intrinsic to parameters (radius > 0, angle ∈ [0,2π]) -> storing in OptimizationConfig would decouple bounds from their parameters -> ParamStore already tracks fixed/free status, so bounds extend the same concept -> reusable across all solvers without extra plumbing |
| Direct penalty for ALM inequalities | Slack variables double the variable count for each inequality -> log-barrier is a different algorithm class (interior point) -> direct penalty (Birgin & Martínez) parallels existing equality penalty code, shares the same ρ schedule, and adds zero extra variables -> simplest extension with proven convergence |
| Wolfe default with Armijo fallback | Pure Wolfe replacement loses the simplicity of Armijo for trivial problems -> some problems have expensive gradient evaluations where the curvature check wastes a function call -> Wolfe-then-Armijo-fallback provides best quality BFGS pairs when possible, degrades gracefully when Wolfe zoom fails |
| Opt-in Hessian via `#[hessian]` | Always-on Hessian doubles symbolic differentiation at compile time for every objective -> most objectives only need gradients (BFGS/L-BFGS) -> opt-in `#[hessian]` avoids unused codegen while making Hessians available for trust-region/Newton when wanted |
| Both dogleg and Steihaug-CG | Dogleg is fast but forms full Hessian (O(n²) memory) -> Steihaug-CG works matrix-free but has higher per-iteration cost for small problems -> auto-select by dimension (configurable threshold, default 100) covers both regimes |
| Configurable TR subproblem threshold | Problem characteristics vary -> a geometric kernel may have 10 variables (dogleg optimal) or 1000 (Steihaug required) -> hardcoded cutoff forces suboptimal choice -> config field costs one line and lets users tune |
| Integration tests only for solvers | Existing pattern in `tests/optimization_solvers.rs` uses known problems (Rosenbrock, quadratics) with convergence assertions -> property-based tests for optimizers require generating convex objectives which is fragile -> follow established pattern |
| trybuild + FD for macro tests | trybuild catches parsing/codegen regressions at compile time -> FD verification catches numerical errors in derivative rules -> neither alone is sufficient since a macro can compile but produce wrong derivatives |

### Rejected Alternatives

| Alternative | Why Rejected |
|---|---|
| Bounds in OptimizationConfig | Decouples bound information from the parameter it describes; requires passing bounds map separately to every solver; doesn't survive parameter ID remapping |
| Bounds as a separate BoundsMap struct | Maximum flexibility but adds plumbing: every solver signature gains a parameter, ConstraintSystem needs to own it, and the user must register bounds in two places |
| Slack variables for inequalities | Doubles variable count per inequality constraint; introduces coupling between slack and original variables that complicates the BFGS inner solve; more code for equivalent convergence |
| Log-barrier interior point for inequalities | Different algorithm class (interior point vs. ALM); requires strictly feasible starting point which CAD problems often lack; significantly more complex implementation with separate step computation |
| Replace Armijo entirely with Wolfe | Loses fast-path for simple problems where the extra curvature check is wasted; Wolfe zoom can fail on flat objectives where Armijo backtracking succeeds trivially |
| Always-on Hessian generation | Doubles compile time for every `#[objective]` even when only gradient is needed; BFGS quasi-Newton is the default solver and ignores exact Hessians |
| Feature-gated Hessian | Cargo features are workspace-wide; can't opt in per-struct; `#[hessian]` attribute gives per-method control |
| Dogleg only for trust region | Fails on large-dimension problems (O(n²) Hessian storage); CAD systems can have high-dimensional parameter spaces from complex assemblies |
| Steihaug-CG only for trust region | Excessive overhead for small problems (2-10 variables) where dogleg solves in one step; most individual CAD constraints are small |

### Constraints & Assumptions

- Rust edition 2021, no nightly-only features
- All new code must be `Send + Sync` (existing trait bounds on `Objective`, `InequalityFn`)
- Sparse gradient/Jacobian representation (`Vec<(ParamId, f64)>`) preserved throughout
- `ParamEntry` is `pub(crate)` — bounds fields don't affect public API surface
- Macro crate uses `proc-macro2`, `quote`, `syn` with features: full, extra-traits, parsing, visit
- No new external dependencies — all algorithms implemented from scratch
- Testing convention: integration tests in `crates/solverang/tests/` following existing patterns

### Known Risks

| Risk | Mitigation | Anchor |
|---|---|---|
| Hessian codegen expression explosion | Apply existing `simplify()` to second derivatives; same algebraic identities (0*x=0, 1*x=x) prevent blowup | `crates/macros/src/expr.rs:204-296` — `simplify()` handles Add/Mul/Div/Pow zero/one cases |
| L-BFGS-B projection breaks curvature pairs | Project gradient difference `y = P(g_new) - P(g_old)` not raw difference; skip update when projected curvature `s^T y` too small | `crates/solverang/src/solver/bfgs.rs:159` — existing `s^T y > 1e-10` guard |
| ALM penalty causes inner BFGS ill-conditioning as ρ→10⁶ | Existing `rho_max` cap at 1e6 limits conditioning; inner tolerance loosens proportionally | `crates/solverang/src/solver/alm.rs:181` — `rho = (rho * growth).min(config.rho_max)` |
| Wolfe zoom may not terminate on non-smooth objectives | Fallback to Armijo after zoom failure (decision: Wolfe-then-Armijo) ensures progress | N/A — new code, fallback by design |
| Trust-region Hessian approximation with L-BFGS | When exact Hessian unavailable, use L-BFGS compact representation `B = γI + V M V^T` for dogleg Newton point | N/A — new code |

## Invisible Knowledge

### Architecture

```
                    ConstraintSystem::optimize()
                              |
                    ┌─────────┼──────────┐
                    │         │          │
               [no constr] [eq+ineq]  [auto]
                    │         │          │
                    v         v          v
              ┌──────────┐  ┌──────┐  ┌──────────────┐
              │BfgsSolver│  │ ALM  │  │TrustRegionSolver│
              │(or BfgsB)│  │      │  │(dogleg/CG)      │
              └──────────┘  └──┬───┘  └────────────────┘
                    │          │              │
                    │    ┌─────┼─────┐        │
                    │    │BFGS inner │        │
                    │    │  loop     │        │
                    │    └───────────┘        │
                    │                         │
              ┌─────┴─────────────────────────┘
              │
         line_search.rs
         (Wolfe → Armijo fallback)
```

### Data Flow

```
User sets Objective + Constraints + Inequalities + Bounds on ParamStore
                          │
    optimize() reads ParamStore → builds solver mapping → dispatches to solver
                          │
    Solver reads params via ParamStore::get(), writes via ParamStore::set()
                          │
    Bounds enforced by L-BFGS-B projection: clamp(x, lower, upper) after each step
                          │
    ALM outer loop: penalty ρ grows, multipliers λ/μ update, inner BFGS converges
                          │
    Result: OptimizationResult with KKT residuals, multipliers, constraint violations
```

### Why This Structure

- `line_search.rs` is a standalone module because both BFGS and trust-region may
  use it (trust-region can fall back to line search when radius is large)
- `bfgs_b.rs` is separate from `bfgs.rs` rather than adding bounds logic to
  the existing solver, because the projection logic (gradient projection,
  Cauchy point on bounds, active set) fundamentally changes the iteration
- Trust-region is a parallel solver to BFGS, not a wrapper, because the step
  computation (subproblem solve) replaces the line search entirely

### Invariants

- `ParamStore` bounds `lower <= value <= upper` must hold after every
  `BfgsBSolver` iteration; violation indicates a projection bug
- ALM inequality multipliers `μ` must remain non-negative (`max(0, ...)` update);
  negative μ would encourage constraint violation
- Wolfe curvature condition `|∇f(x+αd)·d| ≤ c₂|∇f(x)·d|` must hold when
  Wolfe succeeds; violation produces poor L-BFGS curvature pairs
- Hessian entries from macro must be lower-triangle only (i ≥ j); duplicate
  entries produce incorrect Newton steps

### Tradeoffs

- **Bounds in ParamStore vs. separate storage**: Coupling bounds to parameters
  means every `ParamEntry` grows by 16 bytes (two f64). Acceptable since entry
  count is bounded by entity count (typically < 10K in CAD).
- **Direct penalty vs. slack**: Direct penalty avoids extra variables but the
  `max(0, h + μ/ρ)` function is non-smooth at the switching point. Smoothness
  improves as ρ → ∞ which the ALM loop provides.
- **Wolfe + Armijo fallback**: Two code paths increase maintenance. The Wolfe
  path covers 95%+ of iterations; the Armijo path exists only for degenerate
  cases where zoom cannot bracket.

## Milestones

### Milestone 1: Expand `#[auto_diff]` Macro Functions

**Files**:
- `crates/macros/src/expr.rs`
- `crates/macros/src/parse.rs`
- `crates/macros/tests/trybuild/` (new test fixtures)

**Requirements**:

- Add `Asin`, `Acos`, `Sinh`, `Cosh`, `Tanh` variants to `Expr` enum
- Implement `differentiate()` for each:
  - `d(asin(e)) = de / sqrt(1 - e²)`
  - `d(acos(e)) = -de / sqrt(1 - e²)`
  - `d(sinh(e)) = cosh(e) * de`
  - `d(cosh(e)) = sinh(e) * de`
  - `d(tanh(e)) = de / cosh²(e)`
- Add `to_tokens()` branches: `.asin()`, `.acos()`, `.sinh()`, `.cosh()`, `.tanh()`
- Add `simplify()` branches (constant folding)
- Add `collect_variables_into()` match arms
- Parse `.asin()`, `.acos()`, `.sinh()`, `.cosh()`, `.tanh()` method calls in `parse.rs`
- Parse `f64::asin()`, `f64::acos()`, `f64::sinh()`, `f64::cosh()`, `f64::tanh()` function calls

**Acceptance Criteria**:

- `#[auto_diff]` with `x[0].sinh()` compiles and generates correct gradient
- All five new functions have correct derivatives verified by finite difference
- trybuild tests confirm valid inputs compile and invalid inputs produce errors

**Tests**:

- **Test files**: `crates/macros/src/expr.rs` (unit tests), `crates/solverang/tests/macro_functions.rs` (new)
- **Test type**: unit + integration (FD verification + trybuild)
- **Backing**: user-specified
- **Scenarios**:
  - Normal: each function in isolation (e.g., `x[0].sinh()` gradient matches FD)
  - Composed: `x[0].sinh().acos()` chain rule produces correct gradient
  - Edge: `Expr::Const(0.0).asin()` simplifies to constant
  - Error (trybuild): unsupported method `.foobar()` produces compile error

**Code Intent**:

- Add 5 enum variants to `Expr` in `expr.rs` after existing `Exp` variant
- Add `differentiate()` match arms following existing `Sin`/`Cos`/`Tan` pattern
- Add `to_tokens()` match arms following existing pattern (`.asin()`, etc.)
- Add `simplify()` match arms with constant folding (e.g., `Asin(Const(v)) => Const(v.asin())`)
- Add `collect_variables_into()` match arms in the existing unary arm group
- In `parse.rs` `parse_method_call()`: add `"asin" | "acos" | "sinh" | "cosh" | "tanh"` arms
- In `parse.rs` `parse_function_call()`: add `Some("asin") | Some("acos") | ...` arms
- New integration test file with FD verification for each function
- trybuild pass/fail test fixtures

**Code Changes**: *(to be filled by Developer)*

---

### Milestone 2: Strong Wolfe Line Search with Armijo Fallback

**Files**:
- `crates/solverang/src/solver/line_search.rs` (new)
- `crates/solverang/src/solver/bfgs.rs`
- `crates/solverang/src/solver/mod.rs`
- `crates/solverang/src/optimization/config.rs`
- `crates/solverang/tests/optimization_solvers.rs`

**Flags**: `complex-algorithm`, `needs-rationale`

**Requirements**:

- Implement strong Wolfe line search (Nocedal & Wright Algorithm 3.5/3.6):
  - Bracketing phase: find interval [α_lo, α_hi] containing step satisfying Wolfe
  - Zoom phase: bisect interval until strong Wolfe conditions met
  - Strong Wolfe conditions: sufficient decrease (Armijo) AND curvature `|∇f(x+αd)·d| ≤ c₂|∇f(x)·d|`
- Add `wolfe_c2: f64` config field (default: 0.9 for quasi-Newton methods)
- Fallback: if zoom fails to find Wolfe step after max bisections, return best Armijo step found during bracketing
- Replace `armijo_line_search` calls in `bfgs.rs` with `line_search` that tries Wolfe first
- Keep the `armijo_line_search` function available (used by fallback path)

**Acceptance Criteria**:

- Rosenbrock converges in fewer iterations than pure Armijo (measure before/after)
- Strong Wolfe conditions verified at each accepted step via debug assertion
- Fallback to Armijo produces valid descent step when Wolfe zoom fails
- All existing BFGS tests continue to pass

**Tests**:

- **Test files**: `crates/solverang/tests/optimization_solvers.rs` (extend existing)
- **Test type**: integration
- **Backing**: user-specified
- **Scenarios**:
  - Normal: Rosenbrock converges, Wolfe conditions hold
  - Normal: 10D quadratic, verify iteration count ≤ Armijo iteration count
  - Edge: flat objective (gradient ≈ 0 everywhere), line search returns minimal step without hanging
  - Edge: very steep objective, step size α << 1 still satisfies Wolfe

**Code Intent**:

- New file `line_search.rs`:
  - `pub fn strong_wolfe_search(...)` — bracketing + zoom, returns `(alpha, f_alpha, grad_alpha)`
  - `fn zoom(...)` — bisection refinement within bracket
  - `pub fn armijo_search(...)` — extracted from existing `bfgs.rs` function
  - `pub fn line_search(...)` — tries Wolfe, falls back to Armijo on failure
- Modify `config.rs`: add `wolfe_c2: f64` field (default 0.9)
- Modify `bfgs.rs`: replace inline Armijo with call to `line_search::line_search()`
- Modify `mod.rs`: add `pub mod line_search;`
- Existing `armijo_line_search` function in `bfgs.rs` removed (moved to `line_search.rs`)

**Code Changes**: *(to be filled by Developer)*

---

### Milestone 3: Dimension-Independent Convergence Scaling

**Files**:
- `crates/solverang/src/optimization/config.rs`
- `crates/solverang/src/solver/bfgs.rs`
- `crates/solverang/src/solver/alm.rs`
- `crates/solverang/tests/optimization_solvers.rs`

**Requirements**:

- Add `relative_tolerance: bool` config field (default: `true`)
- When enabled, convergence checks use relative norms:
  - BFGS dual: `grad_norm / max(1.0, f.abs()) < dual_tolerance`
  - ALM primal: `violation_norm / max(1.0, n_eq as f64).sqrt() < outer_tolerance`
  - ALM dual: `dual_norm / max(1.0, n as f64).sqrt() < dual_tolerance`
- When disabled, behavior is unchanged (absolute tolerance, backward compatible)
- Initial gradient norm captured at iteration 0 for relative scaling

**Acceptance Criteria**:

- 2D and 50D quadratics converge with same config when `relative_tolerance: true`
- Setting `relative_tolerance: false` reproduces exact previous behavior
- No existing tests break

**Tests**:

- **Test files**: `crates/solverang/tests/optimization_solvers.rs` (extend)
- **Test type**: integration
- **Backing**: user-specified
- **Scenarios**:
  - Normal: 2D quadratic converges with default (relative) tolerances
  - Normal: 50D quadratic converges with same config, similar iteration count
  - Edge: objective value near zero (`f ≈ 0`), `max(1.0, f.abs())` prevents division issues

**Code Intent**:

- `config.rs`: add `pub relative_tolerance: bool` to `OptimizationConfig`, default `true`
- `bfgs.rs` line 73: replace `grad_norm < config.dual_tolerance` with scaled check
- `alm.rs` line 134: scale `violation_norm` and `dual_norm` by dimension
- Both files: capture reference values at iteration 0 for relative comparison

**Code Changes**: *(to be filled by Developer)*

---

### Milestone 4: ParamStore Bounds and L-BFGS-B Solver

**Files**:
- `crates/solverang/src/param/store.rs`
- `crates/solverang/src/solver/bfgs_b.rs` (new)
- `crates/solverang/src/solver/mod.rs`
- `crates/solverang/src/optimization/config.rs`
- `crates/solverang/src/system.rs`
- `crates/solverang/tests/optimization_solvers.rs`

**Flags**: `complex-algorithm`, `needs-rationale`

**Requirements**:

- Add `lower: f64` and `upper: f64` fields to `ParamEntry` (default: `-INFINITY`, `+INFINITY`)
- Add `ParamStore` methods: `set_bounds(id, lower, upper)`, `bounds(id) -> (f64, f64)`, `has_finite_bounds(id) -> bool`
- Implement L-BFGS-B algorithm (projected gradient with L-BFGS direction):
  - Project iterate onto feasible box: `x_i = clamp(x_i, l_i, u_i)` after each step
  - Compute generalized Cauchy point (GCP): identify active bounds along steepest descent path
  - Projected gradient for convergence check: `||P(x - g) - x||` where P is box projection
  - L-BFGS direction computed in reduced space (free variables only)
  - Line search (Wolfe from M2) operates on projected step
- Add `OptimizationAlgorithm::BfgsB` variant
- Update `Auto` dispatch: if any free parameter has finite bounds → BfgsB, else → Bfgs
- Re-export `BfgsBSolver` from `solver/mod.rs`

**Acceptance Criteria**:

- `min (x-3)² + (y-1)²` with `x ∈ [0, 2]` converges to `(2, 1)`, not `(3, 1)`
- Unconstrained problem (infinite bounds) converges identically to BFGS
- All bounds satisfied at every iteration (verified by debug assertion)
- Active-set identification correct: free variables at solution match expected set

**Tests**:

- **Test files**: `crates/solverang/tests/optimization_solvers.rs` (extend)
- **Test type**: integration
- **Backing**: user-specified
- **Scenarios**:
  - Normal: bounded Rosenbrock with `x ∈ [-5, 0.5]` — solution at bound
  - Normal: 10D quadratic with box bounds, all bounds inactive — matches BFGS
  - Edge: all parameters at bounds (zero free dimensions) — immediate convergence
  - Edge: single bound active — solution on boundary face
  - Error: `lower > upper` panics or returns error

**Code Intent**:

- `store.rs`: add `lower: f64, upper: f64` to `ParamEntry` (default ±INFINITY); add `set_bounds()`, `bounds()`, `has_finite_bounds()` pub methods
- New `bfgs_b.rs`: `BfgsBSolver` struct with `solve()` method
  - Reuse `lbfgs_direction()` from `bfgs.rs` (make it `pub(crate)`)
  - `fn project(x: &mut [f64], lower: &[f64], upper: &[f64])` — box projection
  - `fn projected_gradient_norm(x, grad, lower, upper)` — convergence metric
  - `fn generalized_cauchy_point(x, grad, lower, upper, ...)` — GCP computation
  - Line search via `line_search::line_search()` from M2
- `config.rs`: add `BfgsB` to `OptimizationAlgorithm`
- `system.rs`: update `optimize()` Auto dispatch to check bounds
- `mod.rs`: add `pub mod bfgs_b;` and re-export

**Code Changes**: *(to be filled by Developer)*

---

### Milestone 5: Inequality Constraints in ALM

**Files**:
- `crates/solverang/src/solver/alm.rs`
- `crates/solverang/src/system.rs`
- `crates/solverang/src/optimization/result.rs`
- `crates/solverang/tests/optimization_solvers.rs`

**Flags**: `complex-algorithm`

**Requirements**:

- Extend `AlmSolver::solve()` to accept `&[&dyn InequalityFn]`
- Augmented Lagrangian with inequalities (Birgin & Martínez formulation):
  - `L_A = f(x) + Σ_eq [λ_i g_i + ρ/2 g_i²] + Σ_ineq (ρ/2) [max(0, h_j + μ_j/ρ)² - (μ_j/ρ)²]`
- Multiplier update: `μ_{j,k+1} = max(0, μ_{j,k} + ρ · h_j(x_k))`
  - Non-negativity enforced (inequality multipliers must be ≥ 0)
- Primal feasibility: `max(||g(x)||, max_j max(0, h_j(x)))`
- Complementarity residual: `max_j |μ_j · h_j(x)|`
- Update `KktResidual.complementarity` (currently hardcoded to 0.0)
- Extend `AugmentedLagrangianObjective` value() and gradient() to include inequality terms
- Wire `system.rs` `optimize()` to pass inequalities to ALM
- Auto dispatch: if has inequalities → ALM (regardless of equalities)

**Acceptance Criteria**:

- `min (x-2)² + (y-1)²` s.t. `x + y ≤ 3` converges to `(2, 1)` (inactive constraint)
- `min (x-5)² + (y-5)²` s.t. `x + y ≤ 3` converges to `(1.5, 1.5)` (active constraint)
- Mixed equality + inequality problem converges
- Inequality multipliers μ ≥ 0 at solution
- Complementarity `|μ · h| < tolerance` at convergence

**Tests**:

- **Test files**: `crates/solverang/tests/optimization_solvers.rs` (extend)
- **Test type**: integration
- **Backing**: user-specified
- **Scenarios**:
  - Normal: single inequality, inactive at solution
  - Normal: single inequality, active at solution
  - Normal: multiple inequalities, subset active
  - Normal: mixed equality + inequality
  - Edge: infeasible (equality and inequality contradict)

**Code Intent**:

- `alm.rs`: extend `AlmSolver::solve()` signature to `(objective, eq_constraints, ineq_constraints, store, config)`
  - Add `mu` vector for inequality multipliers (init 0)
  - Extend `AugmentedLagrangianObjective` to hold `&[&dyn InequalityFn]` and `mu`/`rho`
  - `value()`: add `(ρ/2) * max(0, h_j + μ_j/ρ)² - (μ_j/ρ)²` terms
  - `gradient()`: add `J_h^T * max(0, μ + ρ·h)` contribution
  - Multiplier update: `μ_j = max(0, μ_j + ρ * h_j)`
  - KKT: compute complementarity as `max |μ_j * h_j|`
  - Primal: `max(eq_violation, max_ineq_violation)`
- `system.rs`: collect `&dyn InequalityFn` from `self.inequalities`, pass to ALM
  - Update Auto dispatch to include inequalities
- `result.rs`: no structural changes (KktResidual already has complementarity field)

**Code Changes**: *(to be filled by Developer)*

---

### Milestone 6: Opt-in Hessian Generation via `#[hessian]`

**Files**:
- `crates/macros/src/expr.rs`
- `crates/macros/src/codegen.rs`
- `crates/macros/src/lib.rs`
- `crates/solverang/tests/macro_hessian.rs` (new)

**Flags**: `complex-algorithm`

**Requirements**:

- Add `#[hessian]` marker attribute recognized by `#[auto_diff]`
- When `#[hessian]` is present alongside `#[objective]`, generate `hessian_entries()`:
  - `fn hessian_entries(&self, x: &[f64]) -> Vec<(usize, usize, f64)>` (lower-triangle)
  - Symbolic second derivatives: differentiate each first derivative expression wrt all variables
  - Apply `simplify()` to second derivatives to control expression size
  - Only emit entries where i ≥ j (lower triangle)
  - Filter zero entries (threshold 1e-30)
- Without `#[hessian]`, behavior unchanged (gradient only)

**Acceptance Criteria**:

- Rosenbrock Hessian matches finite-difference Hessian to 1e-6 relative error
- Quadratic `x² + 2xy + 3y²` produces exact Hessian `[[2, 2], [2, 6]]`
- `#[objective]` without `#[hessian]` still generates only `gradient_entries()`
- trybuild: `#[hessian]` on `#[residual]` method produces compile error

**Tests**:

- **Test files**: `crates/solverang/tests/macro_hessian.rs` (new), `crates/macros/src/expr.rs` (unit)
- **Test type**: integration (FD) + unit + trybuild
- **Backing**: user-specified
- **Scenarios**:
  - Normal: quadratic with known Hessian — exact match
  - Normal: Rosenbrock — FD verification
  - Normal: trigonometric objective — Hessian of `sin(x)*cos(y)` matches FD
  - Edge: linear objective `3*x[0] + 2*x[1]` — Hessian is all zeros
  - Edge: single-variable `x[0]^3` — Hessian is `6*x[0]`
  - Error (trybuild): `#[hessian]` without `#[objective]` fails

**Code Intent**:

- `expr.rs`: no new code needed — second derivative uses existing `differentiate()` twice, then `simplify()`
- `codegen.rs`: new `pub fn generate_hessian_entries(expr, variables) -> Vec<(String, String, TokenStream)>` — for each variable pair (i, j) where i ≥ j, differentiate gradient[j] wrt variable i
  - New `pub fn generate_hessian_method(entries) -> TokenStream` — builds the method body
- `lib.rs`:
  - Add `#[proc_macro_attribute] pub fn hessian(...)` — marker attribute
  - Add `find_hessian_methods()` — checks for `#[hessian]` attribute
  - In `generate_auto_diff_impl()`: if `#[hessian]` present AND `#[objective]` present, generate both `gradient_entries()` and `hessian_entries()`; if `#[hessian]` without `#[objective]`, error
  - Hessian codegen: for each variable j, take the gradient expression for j, differentiate wrt each variable i ≥ j, simplify, generate tokens

**Code Changes**: *(to be filled by Developer)*

---

### Milestone 7: Trust-Region Solver (Dogleg + Steihaug-CG)

**Files**:
- `crates/solverang/src/solver/trust_region.rs` (new)
- `crates/solverang/src/solver/mod.rs`
- `crates/solverang/src/optimization/config.rs`
- `crates/solverang/src/system.rs`
- `crates/solverang/tests/optimization_solvers.rs`

**Flags**: `complex-algorithm`, `needs-rationale`

**Requirements**:

- Implement trust-region method with two subproblem solvers:
  - **Dogleg** (for n < threshold): combines Cauchy point and Newton point
    - Cauchy point: `p_c = -(g^T g / g^T B g) * g`
    - Newton point: `p_n = -B^{-1} g` (using exact Hessian if `ObjectiveHessian`, else L-BFGS compact form)
    - Dogleg path: interpolate between origin → Cauchy → Newton, clipped to trust radius
  - **Steihaug-CG** (for n ≥ threshold): truncated conjugate gradient
    - CG on `B p = -g`, stop at trust-region boundary or negative curvature
    - Matrix-free: only needs Hessian-vector products (from Hessian entries or L-BFGS)
- Trust-region radius update (standard):
  - `ρ_k = (f(x_k) - f(x_k + p_k)) / (m(0) - m(p_k))` (actual vs. predicted reduction)
  - `ρ_k < 0.25`: shrink Δ by 0.25
  - `ρ_k > 0.75` and step at boundary: expand Δ by 2.0
  - Accept step if `ρ_k > η` (η = 0.1)
- Config additions:
  - `OptimizationAlgorithm::TrustRegion` variant
  - `trust_region_init: f64` (default: 1.0) — initial radius Δ₀
  - `trust_region_max: f64` (default: 100.0) — maximum radius
  - `tr_subproblem_threshold: usize` (default: 100) — dogleg vs. Steihaug-CG cutoff
- Auto dispatch: `TrustRegion` must be explicitly selected (not part of Auto)
- If `ObjectiveHessian` is implemented by the objective, use exact Hessian; else use L-BFGS approximation

**Acceptance Criteria**:

- Rosenbrock converges from standard start `(-1.2, 1.0)`
- Trust radius contracts on rejected steps, expands on good steps (verified by log)
- Dogleg used for n < threshold, Steihaug-CG for n ≥ threshold
- Results match BFGS on well-conditioned quadratics
- Ill-conditioned problems (Beale's function) converge where BFGS may struggle

**Tests**:

- **Test files**: `crates/solverang/tests/optimization_solvers.rs` (extend)
- **Test type**: integration
- **Backing**: user-specified
- **Scenarios**:
  - Normal: Rosenbrock with trust region (exact Hessian via manual impl)
  - Normal: 10D quadratic — dogleg path, exact convergence
  - Normal: 200D quadratic — Steihaug-CG path (above threshold)
  - Edge: already at minimum — immediate convergence
  - Edge: Cauchy point outside trust region — step truncated to boundary

**Code Intent**:

- New `trust_region.rs`:
  - `pub struct TrustRegionSolver;`
  - `pub fn solve(objective, store, config) -> OptimizationResult`
  - `fn dogleg_step(grad, hessian, delta) -> Vec<f64>` — dogleg subproblem
  - `fn steihaug_cg(grad, hess_vec_product, delta, max_cg_iters) -> Vec<f64>` — Steihaug-CG
  - `fn actual_reduction(f_old, f_new) -> f64`
  - `fn predicted_reduction(grad, hessian, step) -> f64`
  - `fn update_radius(rho, delta, step_norm, delta_max) -> f64`
  - Hessian source: check if objective implements `ObjectiveHessian` via trait object downcast; if not, build L-BFGS compact representation from gradient history
- `config.rs`: add `TrustRegion` variant, `trust_region_init`, `trust_region_max`, `tr_subproblem_threshold` fields
- `system.rs`: add `TrustRegion` dispatch arm in `optimize()`
- `mod.rs`: add `pub mod trust_region;`, re-export `TrustRegionSolver`

**Code Changes**: *(to be filled by Developer)*

---

### Milestone 8: Documentation

**Delegated to**: @agent-technical-writer (mode: post-implementation)

**Source**: `## Invisible Knowledge` section of this plan

**Files**:

- `crates/solverang/src/solver/CLAUDE.md` (update index)
- `crates/solverang/src/solver/README.md` (architecture, data flow, invariants)
- `crates/solverang/src/optimization/CLAUDE.md` (update index with new types)
- `crates/macros/src/CLAUDE.md` (update with Hessian codegen)

**Requirements**:

Delegate to Technical Writer. Key deliverables:
- CLAUDE.md: Pure navigation index (tabular format)
- README.md: Architecture diagrams, data flow, invariants from Invisible Knowledge

**Acceptance Criteria**:

- CLAUDE.md is tabular index only (no prose sections)
- README.md exists in solver directory with architecture diagram
- README.md is self-contained (no external references)
- New files (line_search.rs, bfgs_b.rs, trust_region.rs) appear in CLAUDE.md index

## Milestone Dependencies

```
M1 (macro functions)
  \
   --> M6 (Hessian generation) --> M7 (trust region, uses Hessian)
  /                                  ^
M2 (Wolfe line search) ─────────────┘
  |                      \
  v                       v
M3 (scaling) ---------> M4 (L-BFGS-B, uses Wolfe + bounds)
  |
  v
M5 (ALM inequalities, uses Wolfe via inner BFGS)

M8 (documentation) -- after all others
```

**Parallelizable waves**:
- Wave 1: M1, M2, M3 (independent)
- Wave 2: M4, M5 (depend on M2+M3; M4 also needs M1 indirectly for test coverage)
- Wave 3: M6 (depends on M1)
- Wave 4: M7 (depends on M2, M6)
- Wave 5: M8 (after all)
