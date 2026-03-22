# Solvers

| Solver | Algorithm | Problem Type | Derivatives Needed |
|--------|-----------|-------------|-------------------|
| `Solver` (NR) | Newton-Raphson | Square F(x)=0 | Jacobian |
| `LMSolver` | Levenberg-Marquardt | Least-squares | Jacobian |
| `AutoSolver` | NR or LM (auto) | Any F(x)=0 | Jacobian |
| `RobustSolver` | NR→LM fallback | Any F(x)=0 | Jacobian |
| `ParallelSolver` | Decompose+parallel | Independent sub-problems | Jacobian |
| `SparseSolver` | Sparse factorization | Large sparse F(x)=0 | Jacobian |
| **`BfgsSolver`** | L-BFGS | **Unconstrained optimization** | **Gradient** |
| **`AlmSolver`** | Augmented Lagrangian | **Equality-constrained optimization** | **Gradient** |

## Optimization Solvers (Phase 1)

**`BfgsSolver`** (`bfgs.rs`): L-BFGS two-loop recursion with Armijo backtracking
line search. Gradient-only (no Hessian). Memory=10 past gradient pairs.
Convergence when `||∇f|| < tolerance`.

**`AlmSolver`** (`alm.rs`): Augmented Lagrangian Method. Outer loop updates
multipliers (λ += ρ*g) and penalty (ρ *= growth). Inner loop uses `BfgsSolver`
on the augmented Lagrangian `L_A = f + λᵀg + (ρ/2)||g||²`. Convergence when
primal feasibility `||g|| < tol` AND dual feasibility `||∇L|| < tol`.

ALM reuses existing constraint infrastructure — `Constraint` trait objects serve
as equality constraints directly.
