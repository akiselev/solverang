# Solverang Generic Constraint Solver — Evolution Plans

This directory contains architectural plans for evolving solverang from a continuous
nonlinear solver into a generic constraint solver supporting multiple problem paradigms.

## Plan Overview

| Plan | Title | Feature Flag | Depends On | Status |
|------|-------|-------------|------------|--------|
| [A](00-problem-trait-expansion.md) | Problem Trait Expansion | core | — | Proposal |
| [1](01-optimization-objectives.md) | Optimization Objectives | `optimization` | A | Proposal |
| [2](02-integer-variables-branch-and-bound.md) | Integer Variables (B&B) | `mixed-integer` | A, 1 | Proposal |
| [3](03-finite-domain-csp.md) | Finite-Domain CSP | `csp` | A | Proposal |
| [4](04-problem-classifier-solver-dispatch.md) | Problem Classifier & Dispatch | `dispatch` | A | Proposal |
| [5](05-modeling-dsl.md) | Modeling DSL | `model` | A, 4 | Proposal |

## Dependency Graph

```
                    Plan A: ProblemBase
                   (foundation for all)
                    /     |     \     \
                   /      |      \     \
            Plan 1     Plan 3   Plan 4  \
         Optimization   CSP    Dispatch  \
              |                  |        \
            Plan 2            Plan 5       \
            MIP/B&B          Model DSL     Plan 5
         (uses Plan 1       (uses Plan 4
          for relaxations)   for dispatch)
```

## Implementation Order

**Recommended sequence** (minimizes blocking, maximizes early value):

1. **Plan A** — Must come first; defines `ProblemBase`, `VariableDomain`
2. **Plan 1** — High value, moderate effort, builds on existing NR/LM
3. **Plan 4** (Phase 1 only) — Classifier + registry with existing solvers
4. **Plan 3** (Phase 1-2) — CSP is independent of continuous, can parallelize
5. **Plan 2** (Phase 1) — Basic B&B using Plan 1's optimization infrastructure
6. **Plan 5** (Phase 1) — Expression tree + differentiation for continuous DSL
7. Remaining phases of Plans 2-5 in parallel

Plans 1 and 3 can be developed **in parallel** after Plan A lands, since they are
independent (continuous optimization vs discrete CSP).

## Concurrency with Other Agents

These plans are designed to be safe for concurrent development:

- **No existing files are modified** until Phase 3 of Plan A (optional deprecation)
- Each plan introduces **new modules behind feature flags**
- The existing `Problem` trait, all solvers, geometry module, JIT, and macros
  continue to work unchanged
- New code **layers on top** via blanket implementations and adapter patterns
- The `SolverRegistry` (Plan 4) uses registration, not modification of existing solvers

## Key Design Decisions

1. **Additive architecture**: New traits and modules are added alongside existing ones.
   The `Problem` trait is never removed or modified.

2. **Feature flags for isolation**: Each plan's code lives behind a feature flag,
   ensuring the core crate stays lean and compilation is fast for users who only
   need continuous solving.

3. **Blanket implementations**: `impl ProblemBase for T where T: Problem` bridges
   old and new. Existing code gets new capabilities for free.

4. **Adapter pattern**: Existing solvers get `AnySolver` adapters for the dispatch
   system. The solvers themselves are never changed.

5. **Expression trees**: The modeling DSL uses runtime expression trees for flexibility
   (runtime problem definition, JIT compilation, serialization) while the existing
   `#[auto_jacobian]` macro continues to provide zero-cost compile-time differentiation.
