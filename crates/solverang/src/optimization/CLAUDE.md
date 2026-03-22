| File | What | When to read |
|------|------|-------------|
| `mod.rs` | Module root, `ObjectiveId` type, re-exports | Adding new optimization types |
| `objective.rs` | `Objective` + `ObjectiveHessian` traits | Implementing custom objectives |
| `inequality.rs` | `InequalityFn` trait (h(x) ≤ 0) | Adding inequality constraints |
| `multiplier_store.rs` | `MultiplierId` + `MultiplierStore` | Accessing sensitivity data |
| `config.rs` | `OptimizationConfig`, algorithm selection | Tuning solver parameters |
| `result.rs` | `OptimizationResult`, `KktResidual`, status enum | Processing solver output |
| `adapters.rs` | `LeastSquaresObjective` (Problem → Objective) | Using existing test problems |
| `../solver/bfgs.rs` | L-BFGS solver (unconstrained) | Debugging convergence |
| `../solver/alm.rs` | ALM solver (equality-constrained) | Debugging constrained optimization |
