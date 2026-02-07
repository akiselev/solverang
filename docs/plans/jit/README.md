# JIT Implementation Plan

Three-level plan for taking the Cranelift JIT from proof-of-concept to
production feature. Each level builds on the previous.

See `docs/notes/jit-integration-analysis.md` for the analysis that motivates
this plan.

## Levels

### [Level 1: Make It Work](level-1-make-it-work.md)

Fix correctness bugs, complete constraint coverage, connect JIT to the public API.

- Fix trig functions (replace Taylor approximations with libm calls)
- Remove dead `StoreJacobian` opcode
- Implement `Lowerable` for all 18 constraint types
- Bridge the type-erasure gap via `try_lower_*` on `GeometricConstraint<D>`
- Add `ConstraintSystem::try_compile()` and `JITSolver::solve_constraint_system()`
- Equivalence tests for every constraint type

**~1,200 LOC** | Steps 1-5 | No dependencies outside this repo

### [Level 2: Make It Fast](level-2-make-it-fast.md)

Measurable, benchmarked speedup on realistic workloads.

- JIT benchmark suite (grid systems at 25 to 2500 variables)
- Fused residual + Jacobian evaluation (single pass, shared variable loads)
- Direct dense Jacobian assembly (skip COO intermediate)
- Per-cluster incremental compilation (recompile only dirty clusters)
- Empirically calibrated JIT threshold

**~1,500 LOC** | Steps 1-5 | Depends on Level 1

### [Level 3: Make It Transformative](level-3-make-it-transformative.md)

Capabilities that interpreted code cannot match.

- Automatic Jacobian derivation via forward-mode AD on the opcode IR
- Compiled Newton steps for small clusters (< 30 vars)
- Symbolic sparsity analysis (static structure, ordering, coloring)
- SIMD vectorization of same-type constraint groups
- First-class integration with V3 architecture (`Constraint` trait, `ParamStore`,
  `ChangeTracker`)

**~3,000 LOC** | Steps 1-5 | Depends on Level 2 + V3 development

## Decision Points

| After... | Decide... |
|----------|-----------|
| Level 1 complete | Are benchmarks worth building? (If the project pivots to V3 immediately, skip Level 2 on the old architecture and build JIT into V3 from the start.) |
| Level 2 benchmarks | Is evaluation time a significant fraction of total solve time? (If not, SIMD and fused eval have low ROI — focus on autodiff and V3 integration.) |
| Level 3 Step 1 (autodiff) | Can we remove all hand-written Jacobian lowerings? (If yes, this halves the per-constraint JIT code.) |
| Level 3 Step 2 (compiled steps) | Do small clusters benefit measurably? (If not, the complexity isn't justified.) |

## Grand Total

~5,700 new LOC across all three levels, touching ~20 files.
