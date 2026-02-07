# JIT Integration Analysis

## Executive Summary

The JIT module is an **architecturally disconnected proof-of-concept**. It compiles,
all 23 unit tests pass, and the Cranelift codegen works for simple cases — but it is
unreachable from the actual public API, has correctness bugs in trigonometric functions,
and covers only 6 of 18 geometric constraint types. The `JITSolver.solve()` method
literally hard-codes a fallback to interpreted evaluation on every call.

The deeper question — *is there even a point to JIT-compiling the solver?* — has a
nuanced answer. The current approach (JIT-compiling residual/Jacobian evaluation) targets
the wrong bottleneck for most problem sizes, but becomes genuinely valuable for large
systems. The real payoff requires rethinking what gets compiled.

---

## 1. Current State: What Exists

### Architecture (4-phase pipeline)

```
Geometric Constraints     ConstraintOp IR      Cranelift IR       Native x86_64/aarch64
(Lowerable trait)    →    (register-based   →  (SSA form)     →   machine code
                           opcodes)
```

### Files

| File | Purpose | LOC |
|------|---------|-----|
| `jit/mod.rs` | Config, `jit_available()`, re-exports | 150 |
| `jit/opcodes.rs` | 17 opcode types, `CompiledConstraints`, validation | 488 |
| `jit/lower.rs` | `Lowerable` trait, `OpcodeEmitter`, `lower_constraints()` | 459 |
| `jit/cranelift.rs` | Cranelift backend, `JITCompiler`, `JITFunction` | 708 |
| `jit/geometry_lowering.rs` | `Lowerable` impls for 6 constraint types | 517 |
| `solver/jit_solver.rs` | `JITSolver`, Newton's method with JIT eval | 422 |
| **Total** | | **~2,744** |

### What Works

- **Compilation**: `cargo check --features jit,geometry` succeeds.
- **Unit tests**: All 23 JIT tests pass (cranelift creation, opcode emission,
  constraint lowering, distance residual correctness, solver fallback).
- **Residual evaluation**: JIT-compiled residuals produce correct results for
  distance constraints (verified: `(0,0)→(3,4)` at target 5 gives residual 0).
- **Jacobian lowering**: Sparsity patterns and `StoreJacobianIndexed` opcodes
  are emitted correctly for the 6 implemented constraint types.
- **Platform detection**: `jit_available()` correctly gates on x86_64/aarch64.

### What's Broken

1. **Architectural disconnect** (`jit_solver.rs:94-97`):
   ```rust
   // For now, we always use interpreted since we can't automatically lower
   // arbitrary Problem implementations.
   self.solve_interpreted(problem, x0)
   ```
   `JITSolver.solve()` **never** uses JIT. It always falls back to interpreted.
   The only way to use JIT is to manually call `solve_with_jit()` with a
   pre-compiled `JITFunction` — which requires the caller to manually lower
   constraints, something no public API facilitates.

2. **Type erasure wall**: `ConstraintSystem<D>` stores constraints as
   `Vec<Box<dyn GeometricConstraint<D>>>`. The `Lowerable` trait is implemented
   on concrete types (`DistanceConstraint<2>`, etc.) but there's no way to
   downcast from `dyn GeometricConstraint<D>` to the concrete type for lowering.

3. **Trig approximations are wrong** (`cranelift.rs:349-408`):
   - `approximate_sin/cos`: Taylor series around 0 with no range reduction.
     Diverges for |x| > ~3. Constraint angles routinely span 0–2π.
   - `approximate_atan2`: Just computes `atan(y/x)` — wrong for 3 of 4 quadrants.
   - `approximate_atan`: Rational approx `x/(1+0.28x²)` — only valid for |x| < 1.

4. **`StoreJacobian` is a no-op** (`cranelift.rs:324-329`): The COO-format
   Jacobian store opcode reads the source register but discards it. Only
   `StoreJacobianIndexed` actually writes to memory.

5. **Missing constraint coverage**: Only 6 of 18 geometric constraints have
   `Lowerable` implementations:
   - **Have**: Distance, Coincident, Fixed, Horizontal, Vertical, Angle
   - **Missing**: Parallel, Perpendicular, Midpoint, PointOnLine, PointOnCircle,
     PointOnCircleVariableRadius, CircleTangent, LineTangent, Symmetric,
     SymmetricAboutLine, Collinear, EqualLength

6. **No performance validation**: Zero benchmarks exercise the JIT path. The
   claimed "2-5x speedup on 1000+ variables" is unsubstantiated.

---

## 2. Is There a Point to JIT Compilation?

### Where the Time Goes in a Solver

For a constraint system with N variables, M residuals, and I iterations:

| Phase | Cost | Dominates When |
|-------|------|----------------|
| Residual evaluation | O(M) per iteration | Never (linear, fast) |
| Jacobian evaluation | O(M × nnz_per_row) per iter | Rarely |
| Linear solve (J·δ = -r) | O(N³) square dense, O(MN²) QR, O(nnz·N) sparse | Almost always |
| Decomposition/graph | O(N + M) one-time | Never |

**The current JIT targets residual and Jacobian evaluation** — which is almost never
the bottleneck. For a 100-variable problem with 100 constraints and 50 iterations,
residual evaluation does ~5,000 arithmetic ops but the LU factorization (O(N³) ≈
1,000,000) dominates. JIT-compiling the residuals saves microseconds in a
millisecond solve.

### When JIT Actually Helps

JIT becomes valuable when:

1. **Virtual dispatch overhead dominates** — systems with thousands of tiny
   constraints where the per-constraint vtable call + branch misprediction cost
   exceeds the arithmetic cost. This is real in CAD: a 500-point sketch with 1500
   constraints means 1500 virtual calls per iteration × 50 iterations = 75,000
   indirect calls. JIT compiles them into one straight-line function.

2. **Jacobian sparsity is exploited** — the JIT knows the sparsity pattern at
   compile time and emits only the stores for non-zero entries. The interpreted path
   builds a `Vec<(usize, usize, f64)>` with allocation overhead every iteration.

3. **The system is re-solved many times** — drag operations in CAD, where the same
   constraint topology is solved at 60 Hz. JIT compilation amortizes across
   thousands of solves, and each solve avoids dispatch overhead.

4. **SIMD/vectorization opportunities** — Cranelift can auto-vectorize the fused
   residual computation in ways that per-constraint Rust code cannot (the constraint
   boundary prevents LLVM from seeing across vtable calls).

### Verdict: Conditional Yes

JIT is **not** useful for one-shot solves of moderate systems (< 200 variables).
It **is** valuable for:
- Interactive CAD constraint solving (same topology, many re-solves)
- Large systems (1000+ constraints) where dispatch overhead is measurable
- Real-time applications requiring predictable, low-latency evaluation

The current implementation targets the right operations (residual + Jacobian) but the
architectural disconnect means it can never actually be used.

---

## 3. How to Take It to the Next Level

### Level 1: Make It Work (Fix bugs, connect to API)

**Goal**: JIT is usable from the public API and produces correct results.

**A. Fix trig functions** — Replace Taylor approximations with libm calls via
Cranelift's `call_indirect` or implement proper range reduction (Cody-Waite
reduction to [-π/4, π/4] then minimax polynomial). Alternatively, since Cranelift
0.116 supports `libcall` for math functions, use those directly.

**B. Fix StoreJacobian no-op** — Either implement the COO storage path or remove
the opcode entirely (the indexed variant works).

**C. Bridge the type-erasure gap** — Two options:
  - **Option 1: Add `Lowerable` to `GeometricConstraint`** — Add a
    `fn try_lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext) -> bool`
    method with a default impl returning `false`. Override in concrete types.
    Then `ConstraintSystem::compile()` iterates constraints and lowers those that
    support it, falling back to interpreted for the rest.
  - **Option 2: Enum dispatch** — Replace `Box<dyn GeometricConstraint<D>>` with
    a constraint enum. Each variant can be pattern-matched for lowering. This is
    faster and JIT-friendly but less extensible.

**D. Complete constraint coverage** — Implement `Lowerable` for the remaining 12
constraint types. Most are simple (Parallel, Perpendicular, Midpoint are just
linear combinations).

**E. Add equivalence tests** — For every constraint type, verify that JIT evaluation
matches interpreted evaluation to within 1e-12 across a range of inputs.

### Level 2: Make It Fast (Performance engineering)

**Goal**: Measurable speedup on realistic workloads.

**A. Fused evaluation** — Currently, residuals and Jacobian are compiled as separate
functions. For Newton's method, we always need both. Compile a single
`evaluate_both(vars, residuals, jacobian)` function that loads each variable once
and computes both outputs in a single pass. This halves memory traffic.

**B. Sparse Jacobian direct assembly** — Instead of returning COO triplets that the
solver converts to a dense matrix, have the JIT function write directly into the
solver's matrix storage (CSR or dense column-major). Eliminate the intermediate
allocation entirely.

**C. Incremental recompilation** — When a constraint is added/removed, only recompile
the affected cluster (not the whole system). The decomposition module already
identifies independent sub-problems — leverage this for per-cluster JIT functions.

**D. Benchmarks** — Create benchmark problems at 100, 500, 1000, 5000 constraints.
Measure:
  - JIT compilation time vs problem size
  - Per-iteration evaluation time (JIT vs interpreted)
  - End-to-end solve time including compilation overhead
  - Break-even point (how many iterations before JIT wins)

### Level 3: Make It Transformative (Beyond evaluation speedup)

**Goal**: JIT enables capabilities that interpreted code cannot match.

**A. Automatic differentiation via JIT** — The opcode IR is a perfect input for
automatic differentiation. Instead of hand-writing Jacobian lowering for each
constraint, transform the residual opcodes into Jacobian opcodes automatically
using forward-mode AD on the IR. This eliminates the most tedious and error-prone
part of adding new constraint types: you write `lower_residual()` once and get
the Jacobian for free.

**B. Compile the full Newton step** — Go beyond evaluating residuals/Jacobian.
For small-to-medium clusters (< 50 vars), compile the entire Newton step:
residual → Jacobian → LU decomposition → back-substitution → solution update.
The LU decomposition has known sparsity structure, so the JIT can eliminate all
branching and indirection, producing a single straight-line function for one
full Newton iteration.

**C. Symbolic sparsity analysis at compile time** — The opcode IR reveals exactly
which variables each residual depends on. Use this for:
  - Static Jacobian structure: allocate once, never check bounds
  - Optimal variable ordering for sparse factorization
  - Graph coloring for compressed Jacobian evaluation

**D. SIMD vectorization** — Group constraints by type and emit SIMD instructions
(AVX2/NEON) to evaluate 4 constraints simultaneously. Distance constraints are
all structurally identical — evaluate 4 in parallel using 256-bit vector ops.

**E. Integration with V3 architecture** — The V3 plan (solver-first-v3.md) introduces
`ParamStore`, `Entity`, and `Constraint` traits. The JIT system should target these
new traits, not the old `GeometricConstraint<D>`:
  - `Constraint::residuals(&self, store: &ParamStore)` → lowerable via the new
    `Constraint` trait having an optional `lower()` method
  - `ReducedSubProblem` (the bridge between new and old APIs) could carry a
    pre-compiled `JITFunction` for the cluster it represents
  - `ChangeTracker` knows which clusters are dirty → only recompile dirty clusters

---

## 4. Priority Recommendation

Given the V3 rearchitecture is the next major milestone, the pragmatic path is:

1. **Don't fix the current JIT-to-ConstraintSystem bridge** — `ConstraintSystem<D>`
   is being replaced. Connecting JIT to it is wasted effort.

2. **Do fix the trig bugs** — These are correctness issues that affect any future
   use of the Cranelift backend.

3. **Do build JIT into V3 from the start** — The new `Constraint` trait should have
   an optional `fn lower(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext)`
   method with a default no-op impl. The new `ConstraintSystem` should check at
   solve time whether all constraints in a cluster support lowering, and if so,
   compile a JIT function for that cluster.

4. **Do implement automatic Jacobian derivation** — This is the killer feature.
   It makes adding new constraint types trivial and eliminates an entire class of
   bugs (incorrect hand-derived Jacobians).

5. **Do benchmark before optimizing** — Get real numbers on where time is spent
   in 1000+ constraint systems before investing in SIMD or fused evaluation.
