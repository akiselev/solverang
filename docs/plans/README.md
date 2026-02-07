# Solverang Generic Constraint Solver — Evolution Plans (v2)

This directory contains architectural plans for evolving solverang from a continuous
nonlinear solver into a generic constraint solver supporting multiple problem paradigms.

**v2 revision**: Removed the unified `AnySolver` / `SolverRegistry` / `SolverOutput`
dispatch layer (v1 Plan 4). Solvers compose through typed protocols, not type-erased
dispatch. The DSL (Plan 5) is the user-facing integration point. Plan 6 (Hybrid Solver
Composition) is the solver-facing integration point for problems spanning paradigms.

## Plan Overview

| Plan | Title | Feature Flag | Depends On | Status |
|------|-------|-------------|------------|--------|
| [A](00-problem-trait-expansion.md) | Problem Metadata Layer | core | — | Proposal v2 |
| [1](01-optimization-objectives.md) | Optimization Objectives | `optimization` | A | Proposal v2 |
| [2](02-integer-variables-branch-and-bound.md) | Integer Variables (B&B) | `mixed-integer` | A, 1 | Proposal v2 |
| [3](03-finite-domain-csp.md) | Finite-Domain CSP | `csp` | A | Proposal v2 |
| [4](04-problem-classifier-solver-dispatch.md) | Problem Classifier | — | A | Proposal v2 |
| [5](05-modeling-dsl.md) | Modeling DSL | `model` | A, 4 | Proposal v2 |
| [6](06-hybrid-solver-composition.md) | Hybrid Solver Composition | `hybrid` | A, 4, 1, 3 | Proposal |

## Architecture: How the Plans Integrate

The plans do NOT integrate through a single `solve()` function or generic solver trait.
That approach requires type erasure and produces an API worse than typed solvers.

Instead, integration happens at two levels:

### User-facing integration: the DSL (Plan 5)

```
Model::new()
    .var("x")                          // continuous
    .integer("slot", 0, 10)            // discrete
    .minimize(cost_expr)               // objective
    .subject_to("clearance", expr)     // constraint
    .build()                           // ← COMPILE-TIME DISPATCH
         │
         │  build() inspects variable types and constraints,
         │  calls ProblemClassifier (Plan 4), emits the right typed object:
         │
         ├── pure continuous     → impl Problem        → AutoSolver
         ├── continuous + obj    → impl OptimProblem    → AugLagSolver (Plan 1)
         ├── pure discrete       → impl DiscreteProblem → CSPSolver (Plan 3)
         └── mixed               → impl HybridProblem   → HybridSolver (Plan 6)
```

### Solver-facing integration: composition protocols (Plan 6)

When a problem spans discrete + continuous, solvers communicate through protocols:

```
HybridSolver (Plan 6)
    │
    ├── Benders: discrete master ↔ continuous sub-problem
    │   Communication: BendersCut (conflict explanation)
    │
    └── DPLL(T): CSP search + TheorySolver callbacks
        Communication: Conflict (nogood for CSP learning)

    Each sub-solver keeps its typed API:
    ├── CSPSolver.solve(&dyn DiscreteProblem) → CSPResult
    ├── AutoSolver.solve(&dyn Problem) → SolveResult
    └── AugLagSolver.solve(&dyn OptimProblem) → OptimResult
```

## Dependency Graph

```
                    Plan A: ProblemBase
                   (metadata for all)
                   /    |    \      \
                  /     |     \      \
           Plan 1    Plan 3   Plan 4  \
        Optimization  CSP   Classifier \
             |         |        |       \
           Plan 2      |     Plan 5      \
           MIP/B&B     |    Model DSL     \
                       |                   \
                       └────── Plan 6 ──────┘
                        Hybrid Composition
                      (Benders / DPLL(T))
                     uses Plan 1 + Plan 3
```

## Implementation Order

1. **Plan A** — Small. ProblemBase metadata trait + VariableDomain + blanket impl
2. **Plan 1** — High value. OptimizationProblem + PenaltyTransform + AugLag solver
3. **Plan 4** — Small. Classifier + VariablePartition (needed by Plans 5 and 6)
4. **Plan 3** — Independent of Plan 1. CSP solver with TheorySolver hook
5. **Plan 2** — Uses Plan 1 relaxations. B&B for MINLP
6. **Plan 6** — Requires Plans 1 + 3. Benders decomposition + DPLL(T)
7. **Plan 5** — The capstone. DSL that emits the right typed object

Plans 1 and 3 can be developed **in parallel** after Plan A lands.

## Key Design Decisions (v2)

1. **No unified `solve()` function**. Each solver has its own typed API. Users who know
   their problem type call the right solver directly. Users who want automation use the
   DSL (Plan 5).

2. **Composition through protocols, not inheritance**. The `HybridProblem` trait (Plan 6)
   defines how to decompose a mixed problem into typed sub-problems. `TheorySolver` and
   `BendersCut` are the communication formats between solvers.

3. **ProblemBase is metadata, not a solving interface**. It enables classification (Plan 4)
   and structural analysis (VariablePartition) but is never passed to `solve()`.

4. **The DSL is the integration point for users**. `Model::build()` determines the
   problem type at build time and returns a typed object — no runtime dispatch needed.

5. **Additive architecture**. The existing `Problem` trait, all current solvers, geometry
   module, JIT, and macros continue to work unchanged. Everything new is layered on top.

## Concurrency with Other Agents

- **No existing files are modified** — all plans add new modules
- Each plan uses **feature flags** for isolation
- Blanket impls bridge old traits to new metadata (`Problem` → `ProblemBase`)
- Existing solvers are used as-is by the composition framework (Plan 6)
