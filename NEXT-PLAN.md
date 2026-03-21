# Phase 1: Solverang Optimization Extension (BFGS + ALM)

## Overview

Add constrained optimization (`min f(x) s.t. g(x)=0, h(x)≤0`) to the Solverang
constraint solver. Phase 1 delivers: `Objective` + `InequalityFn` traits,
`#[objective]` macro attribute (gradient-only, BFGS default), BFGS solver
(unconstrained), ALM solver (constrained, reusing existing NR/LM as inner loop),
multiplier extraction for sensitivity analysis, DRC repair and rubber-banding in
autopcb-router, `minimize`/`subject_to` in the spec language, and a Channel MCP
server for agent feedback loops.

Approach B selected: new `src/optimization/` module within the existing solverang
crate for traits/config/result, solvers in `solver/`, pipeline extensions in
`pipeline/`. Zero changes to existing `solve()` code path.

## Planning Context

### Decision Log

| Decision | Reasoning Chain |
|----------|----------------|
| Separate `optimization/` module (Approach B) | Optimization concepts (objectives, multipliers) are orthogonal to constraint satisfaction -> mixing them into `constraint/` increases cognitive load -> separate module gives clean feature-gating without crate split overhead |
| `InequalityFn` trait name | Existing `InequalityConstraint` in `constraints/inequality.rs` uses array `&[f64]` API -> new system-level trait needs different name to avoid collision -> `InequalityFn` is short and distinct -> user confirmed |
| BFGS default, no macro Hessians in Phase 1 | Symbolic Hessian AST grows O(N²) -> untested compile-time impact for N>20 -> BFGS needs only gradients (already proven via `differentiate()`) -> defer Hessian generation to Phase 3 after empirical validation -> user confirmed opt-in at N≤30 later |
| ALM + BFGS both in Phase 1 | ALM reuses existing NR/LM as inner loop (zero new solver code for inner loop) -> validates full trait→system→pipeline path -> BFGS covers unconstrained case independently -> both needed for DRC repair (constrained) and placement (unconstrained) -> user chose both |
| `residual_hessian(equation_row, store)` | Index-first matches `jacobian()` convention `(row, ParamId, f64)` -> name `residual_hessian` clarifies per-residual scope vs full Lagrangian Hessian -> used by `LagrangianAssembler` code -> user confirmed |
| MultiplierId as semantic `{constraint_id, equation_row}` | No allocator/free-list needed -> natural mapping to constraint-equation pairs -> multipliers are ephemeral (recomputed each solve) so generational lifecycle is wasted complexity -> user confirmed |
| Classify phase before Decompose | Problem type (has objective? inequalities?) is known from registered components -> informs objective-aware variable clustering in Decompose -> doesn't need DOF info (computed later in Analyze) -> user confirmed |
| `opt_config: OptimizationConfig` (not Option) | `Option + unwrap_or_default()` silently absorbs future default changes -> direct field matches existing `SystemConfig` pattern -> user confirmed |
| Bare `#[inequality]` defaults to `upper = 0.0` (h(x) ≤ 0) | Standard mathematical convention -> most constraints are h(x) ≤ 0 -> less surprising for optimization users -> user confirmed |
| Two-level Objective with adapters | System-level `Objective` (ParamStore, sparse) for ConstraintSystem integration -> Problem-level `ObjectiveFunction` (&[f64], dense) for standalone use -> matches existing `Constraint` vs `Problem` pattern -> user confirmed |
| Log-barrier for inequalities (not slack) | `SlackVariableTransform` creates rank-deficient Jacobian when `s→0` (`-2s→0` column) -> NR falls back to SVD pseudoinverse (silent wrong answer, not error) -> log-barrier is smooth and prevents constraint violation -> ALM Phase 1 uses barrier for inequality handling |
| TDD straight to ALM (skip penalty) | Penalty method is a special case of ALM (fixed multipliers) -> going directly to ALM tests the full infrastructure -> user confirmed |
| DRC repair: endpoints only | Moving trace endpoints is simpler and less invasive -> vertex insertion/deletion requires topology changes -> endpoint-only covers majority of clearance violations -> vertex ops deferred to Phase 2+ |
| Rubber-band: silent geometric fallback | Solverang divergence on complex obstacles should not crash the router -> geometric rubber-banding is proven fallback -> log warning for diagnosis -> strict mode deferred to Phase 2+ |
| Spec language in NEXT-PLAN | User wants full Phase 1 vision in one plan -> spec language `minimize`/`subject_to` and Channel MCP server included as later milestones -> depends on core solver being stable (M1-M9) |

### Rejected Alternatives

| Alternative | Why Rejected |
|-------------|-------------|
| Approach A (monolithic) | Pollutes `constraint/mod.rs` with optimization concepts -> `ParamStore` grows responsibilities -> harder to feature-gate optimization code |
| Approach C (new crate) | Over-engineering for Phase 1 -> complicates workspace -> harder to access private internals of solverang (e.g., `ParamStore` fields) |
| Exact Hessians in Phase 1 | Untested compile-time impact -> AST blowup for N>30 -> BFGS sufficient for PCB-scale problems (4-40 variables) -> revisit Phase 3 with empirical data |
| Slack variables for optimization inequalities | Rank-deficient Jacobian when active (`-2s→0`) -> NR pseudoinverse gives silent wrong answers -> log-barrier is smooth and better-conditioned |
| Penalty method before ALM | Penalty is a degenerate ALM (fixed multipliers) -> extra TDD stage with no lasting infrastructure -> straight to ALM is more efficient |
| Nesterov's accelerated gradient in Phase 1 | Requires momentum state management and restart logic -> BFGS with Armijo line search is simpler and proven -> Nesterov deferred to Phase 4 |

### Constraints & Assumptions

- Solverang workspace: `crates/solverang/` (solver) + `crates/macros/` (proc macros)
- Autopcb workspace: `crates/autopcb-router/` (router) + `crates/autopcb-placement/` (placer)
- Existing test pattern: proptest with 256 default / 100k `#[ignore]` exhaustive
- Existing Jacobian verification: `verify_jacobian()` with finite differences
- Feature flag: `solverang` in autopcb-router Cargo.toml (already exists, stubs present)
- No breaking changes concern (no downstream users outside org)
- Rust edition 2021, stable toolchain

### Known Risks

| Risk | Mitigation | Anchor |
|------|-----------|--------|
| ALM outer loop divergence (λ→∞) | Bound multipliers `\|λ\| < max_multiplier`, detect divergence via `\|λ_{k+1}\| > 10 * \|λ_k\|`, fallback to penalty method | New code in `solver/alm.rs` |
| Non-smooth objective (abs()) produces NaN | Compile-time detection of `Abs` in `#[objective]` expressions, emit warning | `crates/macros/src/expr.rs:182-190` — Abs differentiation produces `e/\|e\|` which is 0/0 at e=0 |
| DRC repair creates out-of-bounds endpoints | Add board-containment constraints automatically alongside clearance constraints | `autopcb-router/src/drc/repair.rs` — stub already has `PcbIr` access for board bounds |
| Rubber-band divergence on complex obstacles | Silent fallback to geometric rubber-banding, log warning | `autopcb-router/src/optimize/rubber_band.rs` — existing geometric path proven |
| `#[objective]` macro rejects control flow | Expected limitation — `parse.rs` rejects `if/else` at parse time -> provide `ManualObjective` escape hatch for complex objectives | `crates/macros/src/parse.rs:152` — catch-all rejects unsupported expression types |
| Objective defeats decomposition (single cluster) | Accept monolithic default for Phase 1 -> partial separability detection deferred to Phase 4 | Acceptable because PCB-scale problems are small (4-40 variables) |
| LM `convert_termination` string matching fragility | Inner loop tolerance checked separately via residual norm, not LM's termination message | `crates/solverang/src/solver/levenberg_marquardt.rs` — `msg.contains("residual")` pattern |

## Invisible Knowledge

### Architecture

```
Existing path (unchanged):
  ConstraintSystem::solve() -> Pipeline -> NR/LM -> SolveResult

New optimization path:
  ConstraintSystem::optimize()
    -> Classify (problem type: unconstrained/equality/inequality/mixed)
    -> Decompose (monolithic if objective present)
    -> Analyze (DOF + algorithm selection: BFGS or ALM)
    -> MultiplierInit (zero or warm-start)
    -> Solve:
        BFGS (unconstrained) | ALM outer loop:
          [update λ, μ] -> NR/LM inner solve on augmented Lagrangian -> [check KKT]
    -> PostProcess (extract multipliers, sensitivity)
    -> OptimizationResult { x, f(x), multipliers, sensitivity, status }
```

### Data Flow

```
User defines:
  Objective (value + gradient)  ─┐
  Constraints (existing)        ─┤─> ConstraintSystem
  InequalityFn (values + jac)   ─┘         │
                                     optimize()
                                            │
                                    ┌───────▼──────┐
                                    │  Classify     │ (has objective? inequalities?)
                                    │  Decompose    │ (monolithic if objective couples all)
                                    │  Analyze      │ (select BFGS or ALM)
                                    │  MultiplierInit│ (zero-init for λ, μ)
                                    │  Solve        │ (BFGS or ALM outer + NR/LM inner)
                                    │  PostProcess  │ (extract multipliers)
                                    └───────┬──────┘
                                            │
                                    OptimizationResult
                                      ├── objective_value: f64
                                      ├── param_values: Vec<f64>
                                      ├── multipliers: MultiplierStore
                                      │     └── λ_i = ∂f*/∂b_i (sensitivity)
                                      ├── constraint_violations: Vec<f64>
                                      └── status: Converged | MaxIter | Infeasible
```

### Why This Structure

The `optimization/` module is separate from `constraint/` because:
- Objectives are scalar-valued (gradient is a vector), constraints are vector-valued (Jacobian is a matrix) — different derivative shapes
- Multipliers are ephemeral (recomputed each solve), parameters are persistent (stored between solves) — different lifecycle
- Optimization adds an outer loop (ALM) around existing inner solvers — architectural layering, not modification

If reorganized to merge optimization into constraint, the `Constraint` trait would need optional `is_objective()` methods, the `ParamStore` would gain multiplier storage it doesn't own, and the pipeline would have conditional paths everywhere instead of clean phase separation.

### Invariants

- `MultiplierStore` is cleared before each `optimize()` call — stale multipliers from previous solves must never leak
- `Objective::gradient()` returns sparse entries `(ParamId, f64)` — zero entries must NOT be included (affects sparsity structure)
- `InequalityFn::values()` returns values that should be ≤ 0 when satisfied — positive values indicate violation
- ALM inner solver tolerance must be tighter than outer tolerance — otherwise multiplier updates are based on inaccurate solutions
- Multiplier ordering in `MultiplierStore` matches constraint registration order — `LagrangianAssembler` indexes by offset

### Tradeoffs

- **BFGS over exact Hessians**: Loses quadratic convergence near the solution, but avoids compile-time explosion and works for all problem sizes. For PCB-scale (4-40 vars), BFGS converges in ~50 iterations vs ~10 for Newton — acceptable.
- **Monolithic decomposition**: All variables in one cluster when objective present. Loses parallel sub-problem solving. But PCB optimization problems are small enough that single-cluster solve is fast.
- **Log-barrier over slack variables**: Prevents constraint boundary approach (iterate stays strictly feasible). This means the solution is always slightly inside the feasible region, not exactly on the boundary. For PCB clearance (0.2mm tolerance), this is fine.

## Milestones

### Milestone 1: Foundation Types

**Files**:
- `crates/solverang/src/id.rs`
- `crates/solverang/src/optimization/mod.rs` (NEW)
- `crates/solverang/src/optimization/objective.rs` (NEW)
- `crates/solverang/src/optimization/inequality.rs` (NEW)
- `crates/solverang/src/optimization/multiplier_store.rs` (NEW)
- `crates/solverang/src/optimization/config.rs` (NEW)
- `crates/solverang/src/optimization/result.rs` (NEW)
- `crates/solverang/src/lib.rs`

**Flags**: `conformance`, `needs-rationale`

**Requirements**:
- Add `ObjectiveId` generational index to `id.rs` (same pattern as `EntityId`)
- Add `MultiplierId` struct with `constraint_id: ConstraintId` + `equation_row: usize`
- Define `Objective` trait: `id()`, `name()`, `param_ids()`, `value(&ParamStore) -> f64`, `gradient(&ParamStore) -> Vec<(ParamId, f64)>`
- Define `ObjectiveHessian: Objective` trait: `hessian_entries(&ParamStore) -> Vec<(ParamId, ParamId, f64)>`
- Define `InequalityFn` trait: `id()`, `name()`, `entity_ids()`, `param_ids()`, `inequality_count()`, `values(&ParamStore) -> Vec<f64>`, `jacobian(&ParamStore) -> Vec<(usize, ParamId, f64)>`
- Define `MultiplierStore`: indexed by `MultiplierId`, methods `set()`, `get()`, `clear()`, `lambda_for_constraint()`, `mu_for_inequality()`
- Define `OptimizationConfig`: algorithm enum (BFGS, ALM), outer/inner tolerances, max iterations, multiplier init strategy
- Define `OptimizationResult`: objective_value, status enum (Converged, MaxIterationsReached, Infeasible, Diverged), multiplier_store, constraint_violations, kkt_residual
- Export all new types from `lib.rs`

**Acceptance Criteria**:
- All new types compile
- `ObjectiveId` follows generational index pattern (index + generation)
- `MultiplierId` is `{constraint_id, equation_row}`
- `Objective` and `InequalityFn` are object-safe (`dyn Objective` works)
- `OptimizationConfig::default()` produces sensible values

**Tests**:
- **Test files**: `crates/solverang/tests/optimization_types.rs` (NEW)
- **Test type**: unit
- **Backing**: TDD Stage 1 (doc 04)
- **Scenarios**:
  - Normal: ObjectiveId creation and comparison
  - Normal: MultiplierStore set/get round-trip
  - Normal: OptimizationConfig default values
  - Edge: MultiplierStore get for non-existent ID returns None

**Code Intent**:
- New file `optimization/mod.rs`: re-export sub-modules (objective, inequality, multiplier_store, config, result)
- New file `optimization/objective.rs`: `Objective` trait + `ObjectiveHessian` trait
- New file `optimization/inequality.rs`: `InequalityFn` trait
- New file `optimization/multiplier_store.rs`: `MultiplierId` struct + `MultiplierStore` struct with HashMap<MultiplierId, f64>
- New file `optimization/config.rs`: `OptimizationConfig` struct + `OptimizationAlgorithm` enum + builder methods
- New file `optimization/result.rs`: `OptimizationResult` struct + `OptimizationStatus` enum + `KktResidual` struct
- Modify `id.rs`: add `ObjectiveId` following `EntityId` pattern
- Modify `lib.rs`: add `pub mod optimization;` and re-exports

---

### Milestone 2: ConstraintSystem Extension

**Files**:
- `crates/solverang/src/system.rs`

**Flags**: `conformance`

**Requirements**:
- Add `objective: Option<Box<dyn Objective>>` field to `ConstraintSystem`
- Add `inequalities: Vec<Option<Box<dyn InequalityFn>>>` field
- Add `opt_config: OptimizationConfig` field (direct, not Option)
- Add `set_objective(obj: Box<dyn Objective>)` method
- Add `clear_objective()` method
- Add `add_inequality(ineq: Box<dyn InequalityFn>) -> ConstraintId` method
- Add `set_opt_config(config: OptimizationConfig)` method
- Add `optimize() -> OptimizationResult` method (stub: returns NotImplemented status)
- Add `multiplier(cid: ConstraintId) -> Option<&[f64]>` accessor
- Existing `solve()` must be completely unchanged

**Acceptance Criteria**:
- `set_objective()` stores objective
- `add_inequality()` stores inequality and returns a ConstraintId
- `optimize()` returns a result (stub for now)
- `solve()` behavior is identical (regression test existing tests)
- All existing tests pass unchanged

**Tests**:
- **Test files**: `crates/solverang/tests/optimization_types.rs` (extend)
- **Test type**: integration
- **Backing**: TDD Stage 2-3 (doc 04)
- **Scenarios**:
  - Normal: set_objective + optimize returns result
  - Normal: add_inequality stores and returns ID
  - Error: optimize without objective returns error
  - Regression: existing solve() tests still pass

**Code Intent**:
- Modify `system.rs` `ConstraintSystem` struct: add 4 new fields
- Add 6 new methods on `impl ConstraintSystem`
- `optimize()` initially returns `OptimizationResult { status: NotImplemented, .. }`
- Ensure `Default` for `OptimizationConfig` is used for initial field value

---

### Milestone 3: BFGS Solver

**Files**:
- `crates/solverang/src/solver/bfgs.rs` (NEW)
- `crates/solverang/src/solver/mod.rs`

**Flags**: `complex-algorithm`, `needs-rationale`, `performance`

**Requirements**:
- Implement L-BFGS solver for unconstrained optimization
- Input: `Objective` trait (value + gradient), initial point, config
- Algorithm: L-BFGS two-loop recursion with Armijo backtracking line search
- Memory: store last `m` pairs (s_k, y_k), default m=10
- Convergence: `||gradient|| < tolerance` or max iterations
- Return `OptimizationResult` with objective value, solution, iterations, status

**Acceptance Criteria**:
- Quadratic `(x-3)^2` converges in <10 iterations from x=0
- 2D Rosenbrock converges to (1,1) within tolerance 1e-6
- Gradient is only called (never Hessian)
- Line search satisfies Armijo sufficient decrease condition

**Tests**:
- **Test files**: `crates/solverang/tests/optimization_solvers.rs` (NEW)
- **Test type**: integration + property-based
- **Backing**: TDD Stage 4-5 (doc 04)
- **Scenarios**:
  - Normal: 1D quadratic converges in <10 iterations
  - Normal: 2D Rosenbrock converges to (1, 1) within 1e-6
  - Normal: N-dimensional quadratic (N=10) converges
  - Edge: Already at minimum (gradient ≈ 0) returns immediately
  - Edge: Flat objective (gradient = 0 everywhere) reports MaxIterations
  - Property: gradient FD verification for test objectives (proptest, 256 cases)

**Code Intent**:
- New file `solver/bfgs.rs`: `BfgsConfig` struct (memory size, tolerance, max_iter, line search params) + `BfgsSolver` struct + `solve()` method
- L-BFGS two-loop recursion: maintain deque of (s_k, y_k) pairs, compute H_k * g_k via two-loop algorithm
- Armijo line search: backtrack from step=1.0 with factor 0.5 until `f(x + α*d) ≤ f(x) + c₁*α*g^T*d` where c₁=1e-4
- Modify `solver/mod.rs`: add `pub mod bfgs;` and re-export `BfgsSolver`, `BfgsConfig`

---

### Milestone 4: ALM Solver

**Files**:
- `crates/solverang/src/solver/alm.rs` (NEW)
- `crates/solverang/src/solver/mod.rs`

**Flags**: `complex-algorithm`, `needs-rationale`, `error-handling`

**Requirements**:
- Implement Augmented Lagrangian Method for equality-constrained optimization
- Outer loop: update multipliers λ and penalty ρ, solve inner subproblem
- Inner loop: reuse existing LMSolver to minimize `L_A(x, λ, ρ) = f(x) + λ^T g(x) + (ρ/2) ||g(x)||^2`
- Inner problem is formulated as: find x that minimizes the augmented Lagrangian (least-squares on augmented residuals)
- Multiplier update: `λ_{k+1} = λ_k + ρ * g(x_k)`
- Penalty update: `ρ_{k+1} = min(ρ_k * penalty_growth, ρ_max)` when constraint violation doesn't decrease sufficiently
- Convergence: primal feasibility `||g(x)|| < tol_primal` AND dual feasibility `||∇_x L|| < tol_dual`
- Divergence detection: `||λ_{k+1}|| > 10 * ||λ_k||` triggers fallback to pure penalty (freeze λ)
- Return multipliers in `OptimizationResult`

**Acceptance Criteria**:
- Rosenbrock + equality constraint `x + y = 1` converges to constrained minimum
- Multiplier for the equality constraint is recoverable and has correct sign
- Inner solver (LM) is called, not re-implemented
- Divergence detected and handled (penalty fallback) on ill-conditioned problem
- KKT residual reported in result

**Tests**:
- **Test files**: `crates/solverang/tests/optimization_solvers.rs` (extend)
- **Test type**: integration + property-based
- **Backing**: TDD Stage 6 (doc 04)
- **Scenarios**:
  - Normal: Rosenbrock + `x + y = 1` converges, multiplier ≈ known value
  - Normal: Quadratic + 2 equality constraints converges
  - Edge: Already feasible (constraints satisfied at start)
  - Edge: Infeasible (conflicting constraints) detected and reported
  - Error: Divergence detected, fallback to penalty mode
  - Property: gradient FD verification of augmented Lagrangian (proptest, 256 cases)

**Code Intent**:
- New file `solver/alm.rs`: `AlmConfig` struct (outer_tol, inner_tol, max_outer_iter, rho_init, rho_growth, rho_max, max_multiplier) + `AlmSolver` struct + `solve()` method
- `AugmentedLagrangianProblem` adapter: wraps `Objective` + equality constraints into a `Problem` for LMSolver — residuals are `[√ρ * g_i(x) + λ_i/√ρ]` and objective gradient terms
- Outer loop: solve inner -> check KKT -> update λ -> optionally increase ρ -> repeat
- Multiplier extraction: populate `MultiplierStore` with final λ values
- Modify `solver/mod.rs`: add `pub mod alm;` and re-export

---

### Milestone 5: Pipeline Integration

**Files**:
- `crates/solverang/src/pipeline/mod.rs`
- `crates/solverang/src/pipeline/classify.rs` (NEW)
- `crates/solverang/src/pipeline/multiplier_init.rs` (NEW)
- `crates/solverang/src/system.rs`

**Flags**: `conformance`

**Requirements**:
- Add `Classify` pipeline phase: determines problem type from registered components
- Add `MultiplierInit` pipeline phase: initializes multipliers (zero-init for Phase 1)
- Pipeline order: Classify → Decompose → Analyze → MultiplierInit → Solve → PostProcess
- `ConstraintSystem::optimize()` wired to use the extended pipeline
- Algorithm selection: no objective → existing `solve()` path; unconstrained → BFGS; equality-constrained → ALM
- Zero overhead when no objective present (existing `solve()` path unchanged)

**Acceptance Criteria**:
- Unconstrained problem dispatches to BFGS
- Equality-constrained problem dispatches to ALM
- No-objective case uses existing NR/LM path (no regression)
- MultiplierInit produces zero-initialized multipliers
- End-to-end: `set_objective()` + `optimize()` produces correct result

**Tests**:
- **Test files**: `crates/solverang/tests/optimization_solvers.rs` (extend)
- **Test type**: integration
- **Backing**: TDD Stage 6 (doc 04)
- **Scenarios**:
  - Normal: Unconstrained → BFGS dispatched
  - Normal: Equality-constrained → ALM dispatched
  - Normal: No objective → existing solve path (regression)
  - Normal: End-to-end Rosenbrock optimization through pipeline

**Code Intent**:
- New file `pipeline/classify.rs`: `ProblemType` enum (Unconstrained, EqualityConstrained, InequalityConstrained, Mixed) + `classify()` function
- New file `pipeline/multiplier_init.rs`: `MultiplierInitStrategy` enum (Zero, WarmStart) + `init_multipliers()` function
- Modify `pipeline/mod.rs`: register new phases
- Modify `system.rs`: wire `optimize()` to pipeline → classify → dispatch to BFGS or ALM

---

### Milestone 6: Macro Extension (#[objective])

**Files**:
- `crates/macros/src/lib.rs`
- `crates/macros/src/codegen.rs`
- `crates/macros/src/expr.rs`

**Flags**: `complex-algorithm`, `needs-rationale`

**Requirements**:
- Add `#[auto_diff]` attribute as superset of `#[auto_jacobian]` (backward-compatible alias)
- Add `#[objective]` inner attribute: generates `gradient_entries(&self, x: &[f64]) -> Vec<(usize, f64)>` method
- Add `#[inequality]` inner attribute: generates `constraint_values()` + `constraint_jacobian()` methods
- Bare `#[inequality]` defaults to `upper = 0.0` (h(x) ≤ 0)
- `#[objective(hessian = "exact")]` emits compile error in Phase 1: "Exact Hessian generation deferred to Phase 3"
- Detect `Abs` in `#[objective]` expressions and emit compile-time warning
- Simplification enhancements: `Sub(x, x) → 0`, `Div(x, x) → 1` for structurally equal expressions

**Acceptance Criteria**:
- `#[objective]` on `f(x) = (x[0]-1)^2 + (x[1]-2)^2` generates correct gradient
- `#[inequality(upper = 0.0)]` on `g(x) = x[0] + x[1] - 10` generates correct values + Jacobian
- Bare `#[inequality]` defaults to upper=0.0
- `#[objective(hessian = "exact")]` produces helpful compile error
- `abs()` in `#[objective]` produces warning
- Gradient verified via finite differences on Rosenbrock
- `#[auto_jacobian]` still works (backward compatible)

**Tests**:
- **Test files**: `crates/solverang/tests/macro_tests.rs` (extend)
- **Test type**: integration + property-based
- **Backing**: doc 02
- **Scenarios**:
  - Normal: Simple quadratic gradient correct
  - Normal: Rosenbrock gradient correct (FD verification)
  - Normal: Inequality values + Jacobian correct
  - Edge: Bare `#[inequality]` defaults to upper=0.0
  - Error: `#[objective(hessian="exact")]` compile error
  - Warning: `abs()` in objective triggers warning
  - Regression: existing `#[auto_jacobian]` + `#[residual]` tests pass

**Code Intent**:
- Modify `lib.rs`: add `#[proc_macro_attribute] pub fn auto_diff(...)` entry point that delegates to same logic as `auto_jacobian` but also recognizes `#[objective]` and `#[inequality]`
- New function `generate_gradient_entries()` in `codegen.rs`: for each variable, compute `expr.differentiate(var_id).simplify()`, collect non-zero entries
- New function `generate_gradient_method()` in `codegen.rs`: emit `fn gradient_entries(&self, x: &[f64]) -> Vec<(usize, f64)>` method body
- New function `generate_inequality_methods()` in `codegen.rs`: emit `constraint_values()` + `constraint_jacobian()` + `constraint_bounds()`
- Modify `expr.rs` `simplify()`: add `Sub(a, b) if structurally_equal(a, b) => Const(0.0)` and `Div(a, b) if structurally_equal(a, b) => Const(1.0)`. Add `Expr::structurally_equal(&self, other: &Expr) -> bool` method.
- Add `Abs` detection in objective context: walk expression tree, if `Abs` variant found, emit `proc_macro::Diagnostic` warning

---

### Milestone 7: MINPACK Adapter + Property Tests

**Files**:
- `crates/solverang/tests/optimization_property_tests.rs` (NEW)
- `crates/solverang/tests/minpack_optimization.rs` (NEW)
- `crates/solverang/src/optimization/adapters.rs` (NEW)

**Flags**: `conformance`

**Requirements**:
- `LeastSquaresObjective` adapter: wraps any `Problem` as an `Objective` (minimizing `0.5 * ||F(x)||^2`)
- Gradient: `J^T * r` (Jacobian transpose times residual)
- All 18 MINPACK test problems converge via BFGS through the adapter
- Property tests: gradient FD verification for `Objective` implementations (256 cases)
- Property tests: inequality Jacobian FD verification for `InequalityFn` implementations (256 cases)

**Acceptance Criteria**:
- 18/18 MINPACK problems converge to known solutions within 1e-4
- Gradient FD error < 1e-5 for all test objectives
- Jacobian FD error < 1e-5 for all test inequalities
- Exhaustive variants (100k cases) pass under `#[ignore]`

**Tests**:
- **Test files**: `crates/solverang/tests/minpack_optimization.rs`, `crates/solverang/tests/optimization_property_tests.rs`
- **Test type**: property-based + integration
- **Backing**: existing MINPACK convention + doc 04
- **Scenarios**:
  - Normal: Each MINPACK problem converges via BFGS
  - Property: Random starting points → gradient FD match (256 cases)
  - Property: Random starting points → inequality Jacobian FD match (256 cases)
  - Exhaustive: Same with 100k cases (`#[ignore]`)

**Code Intent**:
- New file `optimization/adapters.rs`: `LeastSquaresObjective<P: Problem>` struct + `Objective` impl (value = `0.5 * sum(r_i^2)`, gradient = `J^T * r`)
- New test file `minpack_optimization.rs`: iterate over existing test problem constructors, wrap in `LeastSquaresObjective`, run BFGS, check convergence
- New test file `optimization_property_tests.rs`: `check_gradient_fd(obj, store, eps, tol)` helper + proptest strategies

---

### Milestone 8: DRC Repair (autopcb-router)

**Files**:
- `/home/kiselev/git/altium-cli-simplified/crates/autopcb-router/src/drc/repair.rs`

**Flags**: `error-handling`, `needs-rationale`

**Requirements**:
- Implement `repair_with_solverang()` function (currently a stub)
- For each clearance violation: extract adjacent geometry, set up ConstraintSystem with:
  - Objective: minimize sum of squared endpoint displacements
  - Constraints: clearance >= required distance for each obstacle pair
  - Board containment: endpoints stay within board bounds
- Solve with ALM (equality constraints from clearance)
- Apply adjusted endpoint positions to route solution
- If solver diverges or result violates board bounds: leave violation unrepaired, log warning
- Feature-gated: `#[cfg(feature = "solverang")]`

**Acceptance Criteria**:
- Simple violation (trace too close to pad, 2 objects) → endpoint moves, violation cleared
- Board containment respected (endpoint doesn't go off-board)
- Solver divergence → graceful fallback (no crash, warning logged)
- Feature flag: without `solverang` feature, function returns unrepaired (existing behavior)

**Tests**:
- **Test files**: `/home/kiselev/git/altium-cli-simplified/crates/autopcb-router/tests/drc_repair.rs` (NEW)
- **Test type**: integration
- **Backing**: doc 09
- **Scenarios**:
  - Normal: Single clearance violation fixed by endpoint displacement
  - Normal: Multiple violations, all repaired
  - Edge: Already-clear trace → no displacement
  - Error: Impossible constraint (trace between two close pads) → graceful failure
  - Feature: Without solverang feature → no-op fallback

**Code Intent**:
- Modify `repair.rs`: replace stub with real implementation
- Create `RepairConstraintSystem`: helper that builds a `ConstraintSystem` from `DrcViolation` + surrounding geometry
- Objective: `DisplacementObjective` — sum of `(x_i - x_i_original)^2 + (y_i - y_i_original)^2`
- Constraints: `ClearanceConstraint` — `distance(endpoint, obstacle) >= required_clearance`
- Board bounds: `ContainmentConstraint` — `x >= x_min, x <= x_max, y >= y_min, y <= y_max`
- Solve with `system.optimize()`, extract new endpoint positions, update `RouteSolution`

---

### Milestone 9: Rubber-Band with Clearance (autopcb-router)

**Files**:
- `/home/kiselev/git/altium-cli-simplified/crates/autopcb-router/src/optimize/rubber_band.rs`

**Flags**: `error-handling`

**Requirements**:
- Implement `rubber_band_solverang()` function (currently falls back to geometric)
- Objective: minimize total trace segment length (sum of Euclidean distances between consecutive vertices)
- Constraints: clearance to all nearby obstacles >= DRC minimum
- Fixed: pad endpoints pinned (not movable)
- Solve with ALM
- If solver diverges: fall back to geometric rubber-banding, log warning
- Feature-gated: `#[cfg(feature = "solverang")]`

**Acceptance Criteria**:
- Slack trace (extra bends) → vertices pulled toward straight line while maintaining clearance
- Clearance-hugging: trace settles at minimum clearance distance from obstacle
- Pad endpoints don't move
- Divergence → geometric fallback (no crash)

**Tests**:
- **Test files**: `/home/kiselev/git/altium-cli-simplified/crates/autopcb-router/tests/rubber_band.rs` (NEW)
- **Test type**: integration
- **Backing**: doc 09
- **Scenarios**:
  - Normal: Slack trace shortened while maintaining clearance
  - Normal: Clearance-hugging near obstacle
  - Edge: Already-optimal trace → no movement
  - Error: Complex obstacle layout → divergence → geometric fallback

**Code Intent**:
- Modify `rubber_band.rs`: implement `rubber_band_solverang()` body
- Create `TraceLengthObjective`: sum of `sqrt((x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2)` for consecutive vertex pairs
- Constraints: `ObstacleClearanceConstraint` for each (vertex, nearby_obstacle) pair
- Fixed endpoints: use `fix_param()` on pad vertex parameters
- Solve with `system.optimize()`, extract new vertex positions
- Fallback: catch divergence/error, call existing `rubber_band_geometric()` instead

---

### Milestone 10: Spec Language `minimize`/`subject_to`

**Files**:
- `/home/kiselev/git/altium-cli-simplified/crates/altium-format-spec/src/parser.rs`
- `/home/kiselev/git/altium-cli-simplified/crates/altium-format-spec/src/compiler.rs`
- `/home/kiselev/git/altium-cli-simplified/crates/altium-format-spec/src/ops.rs`
- `/home/kiselev/git/altium-cli-simplified/docs/ops-lang-spec.md`

**Flags**: `conformance`, `needs-rationale`

**Requirements**:
- New `minimize { expr }` block in placement section of spec language
- New `subject_to { constraint_list }` block following minimize
- Constraint list supports: `clearance { ... }`, `containment { ... }`, existing constraint types
- Constraints can be marked `relaxable: true` with optional `priority: low|medium|high` and `min: value`
- Compiler lowers `minimize` block to `set_objective()` + `add_inequality()` calls on ConstraintSystem
- Solver output includes sensitivity report (JSON) with multiplier values per constraint

**Acceptance Criteria**:
- `minimize { wirelength }` compiles and sets HPWL objective
- `subject_to { clearance { all: 0.5mm } }` compiles and adds clearance constraints
- `relaxable: true` annotation preserved in output for agent consumption
- Sensitivity report JSON includes constraint name, multiplier value, and residual

**Tests**:
- **Test files**: `/home/kiselev/git/altium-cli-simplified/crates/altium-format-spec/tests/optimization_spec.rs` (NEW)
- **Test type**: integration
- **Backing**: doc 10
- **Scenarios**:
  - Normal: Parse `minimize { wirelength }` block
  - Normal: Parse `subject_to { clearance { all: 0.5mm } }`
  - Normal: Compile and execute optimization spec
  - Edge: `minimize` without `subject_to` (unconstrained)
  - Error: Invalid expression in minimize block

**Code Intent**:
- Modify `parser.rs`: add `MinimizeBlock` and `SubjectToBlock` AST nodes + parsing rules
- Modify `ops.rs`: add `Op::Minimize { objective_expr, constraints }` variant
- Modify `compiler.rs`: lower `Op::Minimize` to `system.set_objective(...)` + `system.add_inequality(...)` + `system.optimize()` calls
- Add `SensitivityReport` struct: JSON-serializable, contains per-constraint `{name, multiplier, residual, relaxable, priority}`
- Update `docs/ops-lang-spec.md`: document new syntax

---

### Milestone 11: Channel MCP Server

**Files**:
- `/home/kiselev/git/altium-cli-simplified/crates/autopcb-channel/` (NEW crate)
- `/home/kiselev/git/altium-cli-simplified/Cargo.toml` (workspace member)

**Flags**: `needs-rationale`

**Requirements**:
- New crate `autopcb-channel`: MCP server implementing `claude/channel` capability
- Listens on local HTTP port for solver results
- When solver completes: pushes `OptimizationResult` + `SensitivityReport` as channel event
- Channel event includes: objective value, constraint violations, multiplier values, suggested relaxations
- Two-way: Claude can call `send_feedback` tool to submit modified constraints back
- Integration: autopcb CLI can launch the channel server alongside solver

**Acceptance Criteria**:
- Channel server starts and registers with Claude Code
- Solver result posted to local HTTP endpoint → channel event delivered
- Event contains parseable JSON with multiplier values
- `send_feedback` tool callable from Claude Code

**Tests**:
- **Test files**: `/home/kiselev/git/altium-cli-simplified/crates/autopcb-channel/tests/channel_test.rs` (NEW)
- **Test type**: integration
- **Backing**: doc 10
- **Scenarios**:
  - Normal: POST solver result → event emitted
  - Normal: send_feedback tool returns modified constraints
  - Edge: Solver timeout → partial result event

**Code Intent**:
- New crate `autopcb-channel` with MCP server (TypeScript or Rust — decide based on MCP SDK maturity)
- HTTP listener on configurable port (default 8788)
- Event format: `<channel source="autopcb" job_id="..." status="completed">{ JSON result }</channel>`
- `send_feedback` tool: accepts `{job_id, modified_constraints}`, writes to shared file or sends to solver process
- CLI integration: `--with-channel` flag on `altium-cli` commands

---

### Milestone 12: Documentation

**Delegated to**: @agent-technical-writer (mode: post-implementation)

**Source**: `## Invisible Knowledge` section of this plan

**Files**:
- `crates/solverang/src/optimization/README.md` (NEW)
- `crates/solverang/src/optimization/CLAUDE.md` (NEW)
- `crates/solverang/src/solver/README.md` (UPDATE)

**Requirements**:
- CLAUDE.md: tabular index of optimization module files
- README.md: architecture diagram, data flow, invariants, tradeoffs from Invisible Knowledge
- Solver README update: document BFGS and ALM alongside existing NR/LM

**Acceptance Criteria**:
- CLAUDE.md is tabular index only (no prose)
- README.md contains architecture diagram from Invisible Knowledge
- All invariants from Invisible Knowledge section documented

## Milestone Dependencies

```
M1 (types) ──┬──> M2 (system) ──> M5 (pipeline) ──> M7 (tests+adapter)
              │                                           │
              ├──> M3 (BFGS) ──────────────────> M5       ├──> M8 (DRC repair)
              │                                           │
              ├──> M4 (ALM) ───────────────────> M5       ├──> M9 (rubber-band)
              │                                           │
              └──> M6 (macro) ─────────────────> M7       ├──> M10 (spec lang)
                                                          │
                                                          └──> M11 (channel)
                                                                │
                                                          M12 (docs) <── all
```

**Parallel waves**:
- Wave 1: M1 (foundation types)
- Wave 2: M2, M3, M4, M6 (all depend only on M1, can run in parallel)
- Wave 3: M5, M7 (depend on M2+M3+M4 and M6 respectively)
- Wave 4: M8, M9, M10, M11 (all depend on M5+M7, can run in parallel)
- Wave 5: M12 (documentation, after all implementation)
