# CLAUDE.md

## Overview

Solver implementations for nonlinear systems (Newton-Raphson, LM, BFGS variants, trust-region) and augmented Lagrangian constrained optimization.

## Index

| File | Contents (WHAT) | Read When (WHEN) |
| --- | --- | --- |
| `mod.rs` | Module root, re-exports all public solver types | Adding a new solver or tracing an import |
| `newton_raphson.rs` | `Solver` — Newton-Raphson for square F(x)=0 | Debugging NR convergence, changing damping |
| `levenberg_marquardt.rs` | `LMSolver` — Levenberg-Marquardt for over/under-determined systems | Debugging LM convergence |
| `auto.rs` | `AutoSolver`, `RobustSolver` — automatic solver selection | Changing dispatch heuristics |
| `parallel.rs` | `ParallelSolver` — decomposes into independent sub-problems | Working with parallel solve |
| `sparse_solver.rs` | `SparseSolver` — sparse factorization for large systems | Tuning large sparse solves |
| `bfgs.rs` | `BfgsSolver` — L-BFGS unconstrained optimization, shared helpers (`lbfgs_direction`, `update_lbfgs_history`, `dense_gradient`) | Debugging gradient-based convergence, modifying line search integration |
| `bfgs_b.rs` | `BfgsBSolver` — L-BFGS-B with box constraints; projected gradient, GCP, active-set | Adding bound-constrained problems, debugging projection bugs |
| `line_search.rs` | `strong_wolfe_search`, `armijo_search`, `line_search` — Wolfe with Armijo fallback | Changing step-size strategy, debugging step acceptance |
| `trust_region.rs` | `TrustRegionSolver` — dogleg (n < threshold) or Steihaug-CG (n >= threshold) subproblems | Using exact Hessians, tuning trust-region radius |
| `alm.rs` | `AlmSolver` — Augmented Lagrangian for equality + inequality constraints | Debugging constrained optimization, multiplier updates |
| `config.rs` | `SolverConfig` — Newton-Raphson / LM configuration | Tuning NR/LM tolerances |
| `lm_config.rs` | `LMConfig` — Levenberg-Marquardt damping parameters | Tuning LM damping |
| `lm_adapter.rs` | Adapter bridging LM to `Problem` trait | Implementing new LM-compatible problems |
| `result.rs` | `SolveResult`, `SolveError` — NR/LM result types | Processing root-finding output |
| `jit_solver.rs` | `JITSolver` (feature = "jit") — JIT-compiled constraint evaluation | Working with JIT compilation |
| `README.md` | Architecture diagram, data flow, invariants, design decisions | Understanding solver dispatch, debugging convergence |
