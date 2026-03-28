# Solverang -- Repository Status

*Last updated: 2026-03-26*

## Overview

Solverang is a domain-agnostic numerical solver for nonlinear systems and least-squares problems. It has two personalities: a low-level `Problem` trait for raw equation systems, and a high-level V3 constraint system (`ConstraintSystem`, `Sketch2DBuilder`) where you describe entities and constraints and the solver figures out the rest.

**Architecture**: V3 "solver-first" -- the solver core never imports a geometry type. Domain-specific modules (sketch2d, sketch3d, assembly) implement the `Entity` and `Constraint` extension traits.

## Codebase

| Component | Files | Lines |
|-----------|-------|-------|
| solverang library (`crates/solverang/src/`) | ~130 | ~49,400 |
| macros crate (`crates/macros/src/`) | 5 | ~1,400 |
| Integration tests (`crates/solverang/tests/`) | 17 | ~12,600 |
| Benchmarks (`crates/solverang/benches/`) | 3 | ~1,300 |

**All tests passing, zero compiler warnings.**

## Module Map

### Core (Tier 1)

| Module | What it does |
|--------|-------------|
| `id` | Generational index types: ParamId, EntityId, ConstraintId, ClusterId |
| `param` | ParamStore -- parameter allocation, fixing, get/set; SolverMapping |
| `entity` | Entity trait -- named parameter groups |
| `constraint` | Constraint trait -- residuals + Jacobian entries keyed by ParamId |
| `graph` | Bipartite entity-constraint graph, clustering, decomposition, DOF analysis, redundancy detection |
| `decomposition` | Union-find component extraction for the low-level Problem API |
| `system` | ConstraintSystem -- the top-level orchestrator for V3 |
| `pipeline` | Pluggable solve pipeline: Decompose, Analyze, Reduce, Solve, PostProcess |
| `problem` | Low-level Problem trait (residuals, Jacobian, dimensions) |

### Solving Infrastructure (Tier 2)

| Module | What it does |
|--------|-------------|
| `solve` | ReducedSubProblem adapter, null-space drag, branch selection, closed-form solvers |
| `reduce` | Symbolic reduction: substitute fixed params, merge coincident, eliminate trivials |
| `dataflow` | ChangeTracker + SolutionCache for incremental re-solving with warm starts |
| `solver` | Newton-Raphson, Levenberg-Marquardt, AutoSolver, RobustSolver, ParallelSolver, SparseSolver, JITSolver |
| `jacobian` | Finite-difference Jacobian, verification, sparse patterns, CSR matrices |
| `constraints` | Inequality constraints, slack variable transforms, bounds |

### Domain Plugins (Tier 3)

| Module | What it does |
|--------|-------------|
| `sketch2d` | 2D sketch: Point2D, LineSegment2D, Circle2D, Arc2D + 16 constraint types (incl. SymmetricAboutLine) + Sketch2DBuilder (concentric, tangent_circle_circle, collinear, equal_radius, symmetric_about_line builder methods added) |
| `sketch3d` | 3D sketch: Point3D, LineSegment3D, Plane, Axis3D + 8 constraint types |
| `assembly` | Rigid-body assembly: RigidBody (quaternion orientation) + Mate, Coaxial, Insert, Gear |

### Other

| Module | What it does |
|--------|-------------|
| `test_problems` | 18 MGH least-squares + 14 nonlinear equation problems + NIST StRD (feature-gated) |
| `jit` | Cranelift JIT compilation for constraint evaluation (feature-gated) |

### Macros Crate

`solverang_macros` -- `#[auto_jacobian]` procedural macro for automatic Jacobian generation via symbolic differentiation. Supports arithmetic, trig, sqrt, pow, atan2, chain rule. No control flow.

## Feature Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `std` | yes | Standard library support |
| `macros` | yes | `#[auto_jacobian]` procedural macro |
| `sparse` | yes | Sparse matrix support via faer |
| `parallel` | yes | Parallel solving via rayon |
| `jit` | yes | Cranelift JIT compilation |
| `nist` | yes | NIST StRD regression test problems |

## Test Suite

| Category | Files | What they cover |
|----------|-------|-----------------|
| Unit tests (embedded) | — | All modules |
| Solver megatest | 1 | 100-var chains, overdetermined, sparse mega, cross-solver, robustness |
| Property tests (sketch2d) | 1 | Proptest: satisfaction, Jacobian, DOF, decomposition, coordinate invariance |
| Property tests (sketch3d) | 1 | Proptest: 3D constraint properties |
| Property tests (assembly) | 1 | Proptest: rigid body, quaternion, assembly constraints |
| Property tests (general) | 1 | Proptest: solver properties |
| Contract tests | 1 | Design-by-contract: trait compliance for all constraint/entity types |
| Solver comparison | 1 | NR vs LM vs AutoSolver consistency |
| Sparse tests | 1 | Sparse Jacobian, CSR, faer integration |
| LM tests | 1 | Levenberg-Marquardt edge cases |
| Parallel tests | 1 | Decomposition, component independence |
| Macro tests | 1 | `#[auto_jacobian]` symbolic differentiation |
| MINPACK verification | 1 | Reference validation against MINPACK |
| Solver basic tests | 1 | Basic solver functionality |
| Doc-tests | — | Inline examples in lib.rs |

## Benchmarks

| Suite | What it measures |
|-------|-----------------|
| `comprehensive.rs` | NR vs LM vs AutoSolver, sparse vs dense crossover, parallel speedup |
| `scaling.rs` | Solver scaling across problem sizes, decomposition vs monolithic |
| `nist_benchmarks.rs` | MGH problem suite performance (requires `--features nist`) |

## Known Issues

1. **Decomposition cascading** -- when the solver decomposes into sub-clusters, solutions from earlier clusters don't always propagate to dependent clusters. Can cause `SystemStatus::Solved` to be returned despite non-zero residuals on cross-cluster constraints. Workaround: ensure free entities connect directly to fixed entities.

## Documentation

- `docs/plans/solver-first-v3.md` -- V3 architecture blueprint
- `docs/plans/testing/` -- 11 testing strategy documents
- `docs/plans/jit/` -- Three-level JIT implementation plan
- `docs/notes/` -- Research notes on solvers, JIT, differential dataflow, SOTA survey
- `TESTING_STRATEGY.md` -- Comprehensive testing strategy overview
- `lib.rs` -- Crate-level docs with runnable examples

## CI/CD

GitHub Actions workflows:
- `ci.yml` -- build, test, clippy, doc-tests on push/PR
- `release.yml` -- full test suite + crates.io publish on version tags
