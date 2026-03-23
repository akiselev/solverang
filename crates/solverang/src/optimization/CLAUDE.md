# CLAUDE.md

## Overview

Types for defining optimization problems: objectives, constraints, configuration, and results.

## Index

| File | Contents (WHAT) | Read When (WHEN) |
| --- | --- | --- |
| `mod.rs` | Module root, `ObjectiveId` type, re-exports | Adding new optimization types |
| `objective.rs` | `Objective` + `ObjectiveHessian` traits | Implementing custom objectives; `ObjectiveHessian` enables exact Hessian for trust-region |
| `inequality.rs` | `InequalityFn` trait (h(x) ≤ 0) | Adding inequality constraints |
| `multiplier_store.rs` | `MultiplierStore`, `MultiplierId` — sensitivity data from ALM | Accessing dual variables / sensitivity after solve |
| `config.rs` | `OptimizationConfig`, `OptimizationAlgorithm` (`Auto`, `Bfgs`, `BfgsB`, `Alm`, `TrustRegion`), `MultiplierInitStrategy`; fields include `wolfe_c2`, `relative_tolerance`, `trust_region_init`, `trust_region_max`, `tr_subproblem_threshold` | Changing algorithm selection, tuning tolerances, configuring trust-region or line search |
| `result.rs` | `OptimizationResult`, `KktResidual` (primal, dual, complementarity), `OptimizationStatus` | Processing solver output, checking convergence |
| `adapters.rs` | `LeastSquaresObjective` — adapts `Problem` trait to `Objective` | Wrapping existing least-squares problems for optimization solvers |
| `README.md` | Architecture decisions for the optimization subsystem | Understanding solver dispatch and ALM structure |
| `../solver/bfgs.rs` | L-BFGS unconstrained solver | Debugging gradient convergence |
| `../solver/bfgs_b.rs` | L-BFGS-B box-constrained solver | Debugging bound enforcement |
| `../solver/alm.rs` | ALM equality + inequality constrained solver | Debugging constrained convergence, multiplier updates |
| `../solver/trust_region.rs` | Trust-region solver (dogleg / Steihaug-CG) | Using exact Hessians, tuning radius |
| `../solver/line_search.rs` | Strong Wolfe + Armijo fallback line search | Debugging step acceptance |
