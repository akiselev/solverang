# Optimization Module

Constrained optimization (`min f(x) s.t. g(x)=0, h(x)≤0`) built on top of
the existing constraint-satisfaction infrastructure.

## Architecture

```
Existing path (unchanged):
  ConstraintSystem::solve() → Pipeline → NR/LM → SolveResult

Optimization path:
  ConstraintSystem::optimize()
    → Classify (unconstrained? bounded? constrained?)
    → Dispatch:
        BFGS (unconstrained, no bounds)
        L-BFGS-B (unconstrained with bounds)
        ALM (equality/inequality constrained):
          [update λ/μ, ρ] → BFGS inner solve on augmented Lagrangian → [check KKT]
        TrustRegion (explicit selection, dogleg/Steihaug-CG)
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

**BFGS default, opt-in Hessians via `#[hessian]`.** L-BFGS needs only gradients
and is the default for unconstrained problems. `#[hessian]` on an `#[objective]`
method generates `hessian_entries()` — opt-in avoids double compile-time cost
for objectives that don't need second-order information.

**MultiplierStore uses semantic addressing** `{constraint_id, equation_row}`,
not generational indices. Multipliers are ephemeral — no allocation lifecycle.

**Direct penalty for ALM inequalities** (Birgin & Martínez formulation) —
`(ρ/2)[max(0, h + μ/ρ)² − (μ/ρ)²]` adds zero extra variables. Log-barrier
requires a strictly feasible starting point (CAD problems often lack one).
Slack variables double the variable count per inequality.

## Invariants

- `MultiplierStore` cleared before each `optimize()` — stale multipliers never leak
- `Objective::gradient()` returns sparse `(ParamId, f64)` — zero entries must NOT be included
- `InequalityFn::values()` returns values ≤ 0 when satisfied — positive = violation
- ALM inner tolerance must be tighter than outer tolerance
- Multiplier ordering matches constraint registration order

## Solvers

| Algorithm | File | Use Case |
|-----------|------|----------|
| L-BFGS | `solver/bfgs.rs` | Unconstrained, no bounds (gradient-only, Wolfe line search) |
| L-BFGS-B | `solver/bfgs_b.rs` | Box-constrained (projected gradient, bounds from ParamStore) |
| ALM | `solver/alm.rs` | Equality + inequality constrained (BFGS inner loop, multiplier updates) |
| Trust-Region | `solver/trust_region.rs` | Unconstrained with optional exact Hessian (dogleg / Steihaug-CG) |
| Line Search | `solver/line_search.rs` | Shared Wolfe + Armijo fallback (used by BFGS, L-BFGS-B) |

## Macro Support

`#[auto_diff]` with `#[objective]` generates `gradient_entries()` via symbolic
differentiation — same `Expr::differentiate()` as `#[residual]` Jacobians.
Adding `#[hessian]` alongside `#[objective]` generates `hessian_entries()`
(lower-triangle sparse second derivatives) for trust-region / Newton methods.
