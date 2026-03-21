# Solverang Optimization: Phases 2-4

Deferred work beyond Phase 1 (BFGS + ALM). Implement only when Phase 1 is
validated and concrete use cases demand it.

---

## Phase 2: L-BFGS-B + Variable Bounds (~1500 LOC)

**Prerequisite**: Phase 1 complete and passing all tests.

**Scope**:
1. Variable bounds in ParamStore (lower/upper per param)
2. L-BFGS-B solver (projected gradient with limited-memory BFGS)
3. Bounded variable handling in pipeline
4. `#[inequality]` attribute fully tested with log-barrier
5. Inequality constraint TDD stages 7-8 from doc 04

**Test problems**: Box-constrained Rosenbrock, bounded MINPACK problems.

**Why deferred**: Phase 1's ALM handles equalities. Bounds need projected
gradient infrastructure that isn't needed for DRC repair or rubber-banding.

---

## Phase 3: SQP + Exact Hessians (~2500 LOC)

**Prerequisite**: Phase 2 complete. Empirical validation of macro Hessian
compile times at N=10, N=20, N=30.

**Scope**:
1. `ObjectiveHessian` + `ConstraintHessian` traits activated
2. `Expr::differentiate2()` + enhanced simplification (Sub(x,x)→0, Div(x,x)→1)
3. `#[objective(hessian = "exact")]` enabled in macro (N≤30 only)
4. Hessian JIT opcodes (StoreHessian, StoreGradient)
5. SQP solver with Rust QP subproblem solver (Clarabel.rs or OSQP)
6. `LagrangianAssembler` for Hessian of Lagrangian
7. Full Hock-Schittkowski test suite (HS #1-#116)

**Decision gate**: If macro-generated Hessians exceed 10s compile time at N=20,
abandon exact Hessians and use BFGS-only for all problem sizes. Redesign Phase 3
around forward-mode AD for Hessian-vector products instead.

**Why deferred**: BFGS is sufficient for PCB-scale problems (4-40 variables).
SQP's quadratic convergence only matters for larger or more constrained problems.

---

## Phase 4: IPM + Polish (~2000 LOC)

**Prerequisite**: Phase 3 complete.

**Scope**:
1. Interior point method (log-barrier with KKT system)
2. Algorithm auto-selection in pipeline (BFGS/ALM/SQP/IPM based on structure)
3. Sensitivity analysis exposure (multiplier → agent feedback)
4. Implicit differentiation for parametric optimization
5. Comprehensive benchmarking vs IPOPT on CUTEst/COPS
6. Nesterov's accelerated gradient for placement (replaces LM on unconstrained)
7. Weighted-Average wirelength model (replaces LSE in autopcb placer)

**Why deferred**: IPM is the most complex solver. Algorithm auto-selection
requires all solvers to exist. Sensitivity analysis needs mature multiplier
infrastructure. WA wirelength is an improvement but not blocking.

---

## Phase 1b: Spec Language + Channel (after Phase 1 core)

**Prerequisite**: Phase 1 milestones 1-12 (core solver + autopcb integration).

**Scope**:
1. `minimize { expr } subject_to { constraints }` syntax in spec language
2. Spec compiler lowers optimization blocks to Solverang `optimize()` calls
3. Sensitivity report JSON output from solver
4. Channel MCP server (`autopcb-channel`) for agent feedback loop
5. Agent decision algorithm documentation

**Why separated**: Cross-repo work (altium-format-spec, new MCP server crate)
that depends on the core solver being stable. Can be developed in parallel once
Phase 1 milestones 1-9 are done.

See docs/optimization/10_agent_feedback_loop.md for full channel architecture.
