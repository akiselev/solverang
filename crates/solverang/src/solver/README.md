# Solvers

## Overview

Two distinct solver families live here: root-finding solvers (Newton-Raphson, Levenberg-Marquardt, and variants) that solve F(x)=0, and optimization solvers (BFGS, L-BFGS-B, ALM, trust-region) that minimize f(x) subject to constraints. They share the `ParamStore` abstraction but have separate result types and convergence criteria.

## Architecture

```
ConstraintSystem::optimize()
          |
          +-- [no constraints, no bounds] --> BfgsSolver
          |
          +-- [has bounds, no constraints] --> BfgsBSolver
          |
          +-- [eq or ineq constraints]    --> AlmSolver
          |                                       |
          |                                [BFGS inner loop]
          |
          +-- [explicit TrustRegion]      --> TrustRegionSolver
                                               |
                                    +----------+----------+
                                    |                     |
                               [n < threshold]       [n >= threshold]
                               dogleg subproblem     Steihaug-CG
```

All gradient-based solvers share `line_search.rs`. ALM's inner loop calls `BfgsSolver` (not `BfgsBSolver`) — inner iterates are unconstrained; bounds apply only at the outer level.

## Data Flow

```
User: ParamStore with values + optional bounds + optional constraints/inequalities
  |
optimize() builds solver mapping (free params only → dense index)
  |
Solver reads params via ParamStore::get()
  |
Gradient evaluated as sparse Vec<(ParamId, f64)>, densified for solver
  |
Line search or trust-region step computed in dense space
  |
Updated dense values written back via ParamStore::set()
  |
BfgsBSolver: clamp(x_i, lower_i, upper_i) after every write (projection)
  |
ALM outer loop: penalty ρ grows, multipliers λ/μ updated
  |
Result: OptimizationResult with KktResidual (primal, dual, complementarity)
```

## Design Decisions

**`line_search.rs` is a standalone module** — both `BfgsSolver` and `BfgsBSolver` use it, and trust-region could fall back to it when the radius is large. Keeping it separate avoids circular dependencies between the solver files.

**`bfgs_b.rs` is separate from `bfgs.rs`** — the projection logic (gradient projection, Cauchy point on bounds, active-set tracking) fundamentally changes the iteration structure. Merging them would require branching inside every inner loop step, reducing readability without reducing code.

**`trust_region.rs` is a parallel solver, not a wrapper around BFGS** — the step computation (dogleg or Steihaug-CG subproblem solve) replaces the line search entirely. There is no descent direction to search along; the step and its length are determined together by the subproblem.

**Auto dispatch checks bounds before constraints** — if free parameters have finite bounds but no constraints, `BfgsBSolver` is chosen. If any constraints or inequalities are present, `AlmSolver` is chosen regardless of bounds. `TrustRegion` must be selected explicitly; it is not part of Auto.

**Direct penalty for ALM inequalities** — the Birgin & Martínez formulation `(ρ/2)[max(0, h + μ/ρ)² − (μ/ρ)²]` adds zero extra variables. Slack variables would double the variable count per inequality and couple slacks to the original parameters. Log-barrier is a different algorithm class requiring strictly feasible starting points, which CAD problems often lack.

**Wolfe default with Armijo fallback** — pure Armijo backtracking is fast for trivial problems but produces poor L-BFGS curvature pairs. Pure Wolfe can fail on flat objectives where the zoom cannot bracket. The combined strategy tries strong Wolfe (N&W Algorithm 3.5/3.6), falls back to the best Armijo step found during bracketing, and always makes progress.

**L-BFGS compact approximation for trust-region** — when no exact `ObjectiveHessian` is available, `TrustRegionSolver` uses the scaled identity `B ≈ γI` where `γ = yᵀy / sᵀy` from the most recent L-BFGS pair. The Newton point in dogleg is obtained via the full two-loop recursion from `bfgs::lbfgs_direction`.

**Dogleg vs. Steihaug-CG split at n = 100 (configurable)** — dogleg forms a full dense Hessian approximation O(n²). Steihaug-CG is matrix-free but has higher per-iteration cost for small n. The `tr_subproblem_threshold` config field sets the crossover; 100 covers most individual CAD constraint sub-problems (< 100 vars) while routing large assembly problems through CG.

## Invariants

- `BfgsBSolver`: after every parameter write, `lower_i <= x_i <= upper_i` must hold. A projection bug produces values outside the box.
- ALM inequality multipliers `μ_j` must remain non-negative (`max(0, ...)` update). Negative `μ` would penalize constraint satisfaction rather than violation.
- Wolfe curvature condition `|∇f(x+αd)·d| ≤ c₂|∇f(x)·d|` must hold when `strong_wolfe_search` returns success. Violation produces bad L-BFGS curvature pairs and degrades quasi-Newton convergence.
- L-BFGS curvature pairs are skipped when `sᵀy < 1e-10` (existing guard in `bfgs.rs`). For `BfgsBSolver`, `y` uses the projected gradient difference, not the raw difference; skipping on raw `sᵀy` would retain bad pairs.
- Hessian entries from `#[hessian]` macro are lower-triangle only (i >= j). Duplicate entries from symmetric positions produce incorrect Newton steps in `TrustRegionSolver`.

## Solver Quick Reference

| Solver | Algorithm | Problem Type | Derivatives |
|--------|-----------|-------------|-------------|
| `Solver` (NR) | Newton-Raphson | Square F(x)=0 | Jacobian |
| `LMSolver` | Levenberg-Marquardt | Least-squares | Jacobian |
| `AutoSolver` | NR or LM (auto) | Any F(x)=0 | Jacobian |
| `RobustSolver` | NR → LM fallback | Any F(x)=0 | Jacobian |
| `ParallelSolver` | Decompose + parallel | Independent sub-problems | Jacobian |
| `SparseSolver` | Sparse factorization | Large sparse F(x)=0 | Jacobian |
| `BfgsSolver` | L-BFGS | Unconstrained optimization | Gradient |
| `BfgsBSolver` | L-BFGS-B | Box-constrained optimization | Gradient |
| `AlmSolver` | Augmented Lagrangian | Equality + inequality constrained | Gradient |
| `TrustRegionSolver` | Dogleg / Steihaug-CG | Unconstrained (exact Hessian optional) | Gradient + optional Hessian |
