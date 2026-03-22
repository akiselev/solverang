# Optimization Module

Constrained optimization (`min f(x) s.t. g(x)=0, h(x)≤0`) built on top of
the existing constraint-satisfaction infrastructure.

## Architecture

```
Existing path (unchanged):
  ConstraintSystem::solve() → Pipeline → NR/LM → SolveResult

Optimization path:
  ConstraintSystem::optimize()
    → Classify (unconstrained? equality-constrained?)
    → Dispatch:
        BFGS (unconstrained) | ALM outer loop:
          [update λ, ρ] → BFGS inner solve on augmented Lagrangian → [check KKT]
    → OptimizationResult { f(x*), multipliers, kkt_residual, status }
```

## Data Flow

```
User defines:
  Objective (value + gradient)  ─┐
  Constraints (existing)        ─┤─→ ConstraintSystem
  InequalityFn (values + jac)   ─┘         │
                                      optimize()
                                            │
                                  OptimizationResult
                                    ├── objective_value: f64
                                    ├── multipliers: MultiplierStore
                                    │     └── λ_i = ∂f*/∂b_i (sensitivity)
                                    ├── kkt_residual: {primal, dual, complementarity}
                                    └── status: Converged | MaxIter | Infeasible
```

## Module Structure

| File | Purpose |
|------|---------|
| `objective.rs` | `Objective` trait (value + gradient) + `ObjectiveHessian` (opt-in) |
| `inequality.rs` | `InequalityFn` trait (h(x) ≤ 0, parallels `Constraint`) |
| `multiplier_store.rs` | `MultiplierId` (semantic: constraint+row) + `MultiplierStore` |
| `config.rs` | `OptimizationConfig` (algorithm, tolerances, ALM params) |
| `result.rs` | `OptimizationResult` + `OptimizationStatus` + `KktResidual` |
| `adapters.rs` | `LeastSquaresObjective` (wraps `Problem` as `Objective`) |

## Key Design Decisions

**Separate from constraint module.** Objectives are scalar (gradient = vector),
constraints are vector (Jacobian = matrix) — different derivative shapes.
Multipliers are ephemeral (recomputed each solve), parameters are persistent.

**BFGS default, no Hessians in Phase 1.** L-BFGS needs only gradients. Exact
Hessians deferred to Phase 3 after compile-time impact is measured empirically.

**MultiplierStore uses semantic addressing** `{constraint_id, equation_row}`,
not generational indices. Multipliers are ephemeral — no allocation lifecycle.

**Log-barrier for inequalities** (not slack variables). Slack variables create
rank-deficient Jacobians when constraints become active.

## Invariants

- `MultiplierStore` cleared before each `optimize()` — stale multipliers never leak
- `Objective::gradient()` returns sparse `(ParamId, f64)` — zero entries must NOT be included
- `InequalityFn::values()` returns values ≤ 0 when satisfied — positive = violation
- ALM inner tolerance must be tighter than outer tolerance
- Multiplier ordering matches constraint registration order

## Solvers

| Algorithm | File | Use Case |
|-----------|------|----------|
| L-BFGS | `solver/bfgs.rs` | Unconstrained (gradient-only, Armijo line search) |
| ALM | `solver/alm.rs` | Equality-constrained (BFGS inner loop, multiplier updates) |

## Macro Support

`#[auto_diff]` with `#[objective]` generates `gradient_entries()` via symbolic
differentiation — same `Expr::differentiate()` as `#[residual]` Jacobians.
