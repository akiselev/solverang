# Solverang TODO

## JIT

### Multi-residual `#[auto_jacobian]` support
The macro currently handles a single `#[residual]` method returning `f64`. Real problems have multiple residuals (Rosenbrock has 2, BroydenTridiagonal has N). Need to support either multiple `#[residual]` methods or a method returning `Vec<f64>` so the JIT path works for non-trivial problems without per-residual struct wrapping.

### JITSolver auto-detection
`JITSolver::solve()` should detect that a Problem has `lower_to_compiled_constraints` and use it automatically. Currently the user must manually call `lower_to_compiled_constraints()` → `JITCompiler::compile()` → `jit_fn.evaluate_residuals()`. Make JIT transparent: `solver.solve(&problem, &x0)` compiles if beneficial, falls back otherwise.

### Fused residual + Jacobian evaluation
Compile a single `evaluate_both(vars, residuals, jacobian)` function that loads each variable once and computes both outputs in one pass. Currently residuals and Jacobian are compiled and called separately, redundantly loading the same variables. Expected ~1.5-2x improvement on top of current JIT speedup.

### Direct dense Jacobian assembly
JIT currently writes Jacobian in COO format, then the solver copies into a dense nalgebra `DMatrix`. Instead, compute dense matrix offsets at JIT compile time (`col * m + row`, column-major) and have the JIT write directly into the matrix storage. Eliminates per-iteration allocation and copy.

### JIT threshold calibration
The hardcoded `jit_threshold: 1000` is arbitrary with no empirical backing. Use benchmark data to find break-even points. Consider a two-tier threshold: higher for one-shot solves, lower for interactive/re-solve scenarios where compilation amortizes over many calls.

### Compiled Newton steps (N < 30)
For small square systems, compile the entire Newton iteration (residual → Jacobian → LU decompose → back-substitute → update) into a single native function. Symbolic LU at compile time determines non-zero structure; emit fully unrolled factorization. Eliminates per-iteration solver overhead. Useful for hot-loop cases like the PCB autorouter.

### Remove dead `StoreJacobian` opcode
`ConstraintOp::StoreJacobian { row, col, src }` reads the source register but discards the value in cranelift codegen. All actual Jacobian storage goes through `StoreJacobianIndexed`. Remove the dead variant.

## Ergonomics

### Better solve failure diagnostics
When solve fails, report which constraints are unsatisfied and by how much. Currently returns `NotConverged` with just the residual norm — no breakdown of which equations are the problem.

### Builder API for multi-residual Problems
Ergonomic way to compose a Problem from multiple residual functions without manually implementing the trait. Something like:
```rust
let problem = ProblemBuilder::new(2) // 2 variables
    .residual(|x| 10.0 * (x[1] - x[0].powi(2)))
    .residual(|x| 1.0 - x[0])
    .build();
```

## Documentation

### End-to-end tutorial
Show the PCB toolkit use case: define a physics equation, annotate with `#[auto_jacobian]`, JIT-compile, call from an autorouter loop. This is the headline demo.

### Crate-level docs
Document the two paths: `Problem` trait (low-level, manual residuals/Jacobian) vs `ConstraintSystem` (high-level, entities + constraints + auto-decomposition).

## Publish

### Clean public API surface
Audit what's `pub` vs `pub(crate)`. The `__jit_reexports` module is already `#[doc(hidden)]` but there may be other internal types leaking.

### Changelog
Write a changelog covering the current feature set before publishing to crates.io.
