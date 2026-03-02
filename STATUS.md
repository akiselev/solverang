# Solverang -- Repository Status

*Last updated: 2026-03-02*

## Overview

Solverang is a domain-agnostic numerical solver for nonlinear systems and least-squares problems. It has two personalities: a low-level `Problem` trait for raw equation systems, and a high-level V3 constraint system (`ConstraintSystem`, `Sketch2DBuilder`) where you describe entities and constraints and the solver figures out the rest.

**Architecture**: V3 "solver-first" -- the solver core never imports a geometry type. Domain-specific modules (sketch2d, sketch3d, assembly, geometry) implement the `Entity` and `Constraint` extension traits.

## Codebase

| Component | Files | Lines |
|-----------|-------|-------|
| solverang library (`crates/solverang/src/`) | ~130 | ~49,400 |
| macros crate (`crates/macros/src/`) | 5 | ~1,400 |
| Integration tests (`crates/solverang/tests/`) | 17 | ~12,600 |
| Benchmarks (`crates/solverang/benches/`) | 3 | ~1,300 |

**1,242 tests, all passing.** 14 compiler warnings (dead_code on entity ID struct fields, one unused import, one useless comparison).

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
| `geometry` | Legacy 2D/3D constraint library using `GeometricConstraint<D>` trait with `Point<D>`, `Line<D>`, `Circle`, builder API. 16 constraint types. Feature-gated. |
| `sketch2d` | V3 2D sketch: Point2D, LineSegment2D, Circle2D, Arc2D + 15 constraint types + Sketch2DBuilder |
| `sketch3d` | V3 3D sketch: Point3D, LineSegment3D, Plane, Axis3D + 8 constraint types |
| `assembly` | V3 rigid-body assembly: RigidBody (quaternion orientation) + Mate, Coaxial, Insert, Gear |

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
| `geometry` | yes | Legacy geometric constraint library |
| `sparse` | yes | Sparse matrix support via faer |
| `parallel` | yes | Parallel solving via rayon |
| `jit` | yes | Cranelift JIT compilation |
| `nist` | yes | NIST StRD regression test problems |

## Test Suite

17 test files, 1,242 tests total.

| Category | Files | Tests | What they cover |
|----------|-------|-------|-----------------|
| Unit tests (embedded) | — | 784 | All modules |
| Solver megatest | 1 | 14 | 100-var chains, overdetermined, massive geometric, sparse mega, cross-solver, robustness |
| Property tests (sketch2d) | 1 | 92 | Proptest: satisfaction, Jacobian, DOF, decomposition, coordinate invariance |
| Property tests (sketch3d) | 1 | 16 | Proptest: 3D constraint properties |
| Property tests (assembly) | 1 | 38 | Proptest: rigid body, quaternion, assembly constraints |
| Property tests (geometry) | 1 | 20 | Proptest: legacy geometry constraints |
| Property tests (general) | 1 | 31 | Proptest: solver properties |
| Contract tests | 1 | 25 | Design-by-contract: trait compliance for all constraint/entity types |
| Solver comparison | 1 | 10 | NR vs LM vs AutoSolver consistency |
| Sparse tests | 1 | 31 | Sparse Jacobian, CSR, faer integration |
| LM tests | 1 | 20 | Levenberg-Marquardt edge cases |
| Parallel tests | 1 | 12 | Decomposition, component independence |
| Macro tests | 1 | 10 | `#[auto_jacobian]` symbolic differentiation |
| MINPACK verification | 1 | 25 | Reference validation against MINPACK |
| Legacy geometry tests | 2 | — | Old geometry module integration tests |
| Solver basic tests | 1 | 5 | Basic solver functionality |
| Doc-tests | — | 18 | Inline examples in lib.rs |

## Benchmarks

| Suite | What it measures |
|-------|-----------------|
| `comprehensive.rs` | NR vs LM vs AutoSolver, sparse vs dense crossover, parallel speedup, geometric systems |
| `scaling.rs` | Solver scaling across problem sizes, decomposition vs monolithic |
| `nist_benchmarks.rs` | MGH problem suite performance (requires `--features nist`) |

## Known Issues

1. **14 compiler warnings** -- mostly dead_code on struct fields that store EntityId for documentation/debugging but aren't read in hot paths. One unused import (`ConstraintId` in pipeline/types.rs), one useless comparison (`duration.as_nanos() >= 0` in system.rs).

2. **Decomposition cascading** -- when the solver decomposes into sub-clusters, solutions from earlier clusters don't always propagate to dependent clusters. Can cause `SystemStatus::Solved` to be returned despite non-zero residuals on cross-cluster constraints. Workaround: ensure free entities connect directly to fixed entities.

3. **Dual geometry systems** -- the legacy `geometry` module and the V3 `sketch2d`/`sketch3d`/`assembly` modules coexist. The legacy module uses a separate `GeometricConstraint<D>` trait that operates on `&[Point<D>]`; the V3 modules use the core `Constraint` trait with `ParamId`. Both work, but having two parallel systems is confusing and should be consolidated.

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
