# Changelog

## [Unreleased]

## [0.1.0] - 2026-03-21

### Added
- Multiple solver algorithms: Newton-Raphson, Levenberg-Marquardt, AutoSolver, RobustSolver, ParallelSolver, SparseSolver
- JIT compilation via Cranelift for constraint evaluation (`jit` feature)
  - Fused residual + Jacobian evaluation in single native function
  - Direct dense Jacobian assembly (column-major, no COO copy)
  - Compiled Newton steps for small systems (N < 30)
  - Automatic JIT detection in JITSolver::solve()
- `#[auto_jacobian]` proc macro for automatic symbolic differentiation (`macros` feature)
  - Multi-residual support (multiple `#[residual]` methods)
  - JIT opcode lowering generation
- ProblemBuilder API for closure-based problem construction
- Solve failure diagnostics with per-equation residual breakdown
- Problem decomposition into independent sub-problems
- Jacobian verification via finite differences
- V3 ConstraintSystem with Sketch2D/3D builders
- 40+ MINPACK test problems
