# Level 3: Make It Transformative

**Goal**: JIT enables capabilities that interpreted code fundamentally cannot match.

**Prerequisite**: Level 2 complete (JIT is measurably faster with benchmarks).

---

## Overview

Levels 1 and 2 make JIT a faster version of the interpreted path. Level 3 makes
it do things the interpreted path *cannot do at all*:

- **Automatic Jacobian derivation** — write `lower_residual()` once, get the
  Jacobian for free via forward-mode AD on the opcode IR. Eliminates the most
  error-prone part of adding new constraint types.
- **Compiled Newton steps** — for small clusters, compile the entire iteration
  (residual → Jacobian → linear solve → update) into a single native function.
- **Symbolic sparsity analysis** — extract structural information from the IR to
  optimize sparse matrix operations.
- **SIMD vectorization** — group structurally identical constraints and evaluate
  them in parallel using vector instructions.
- **V3 architecture integration** — make JIT a first-class citizen of the new
  `Constraint`/`Entity`/`ParamStore` architecture.

**Estimated effort**: ~3,000 new LOC across ~12 files.

---

## Step 1: Automatic Jacobian Derivation from Opcode IR

**Files**: new `jit/autodiff.rs`, modifications to `jit/lower.rs`

This is the single highest-leverage feature in the entire JIT roadmap. Today,
every constraint type requires two hand-written lowering methods: `lower_residual()`
and `lower_jacobian()`. The Jacobian implementations are tedious, error-prone,
and must be kept in sync with the residual code. Automatic differentiation
eliminates this entirely.

### Why the opcode IR is perfect for AD

The `ConstraintOp` IR is:
- A straight-line sequence of register-to-register operations (no control flow)
- Every operation has a known derivative rule
- The register model makes the chain rule trivial to apply

This is the *textbook* setting for forward-mode automatic differentiation.

### 1a. Forward-mode AD on opcode streams

Forward-mode AD computes `∂(output)/∂(input_i)` by propagating derivative
"tangent" values alongside the primal computation. For each primal register `r`,
we create a tangent register `dr` that holds `∂r/∂x_i`.

**Derivative rules for each opcode**:

| Opcode | Primal | Tangent (∂/∂x_i) |
|--------|--------|-------------------|
| `LoadVar { dst, var_idx }` | `r = vars[j]` | `dr = (j == i) ? 1.0 : 0.0` |
| `LoadConst { dst, value }` | `r = c` | `dr = 0.0` |
| `Add { dst, a, b }` | `r = a + b` | `dr = da + db` |
| `Sub { dst, a, b }` | `r = a - b` | `dr = da - db` |
| `Mul { dst, a, b }` | `r = a * b` | `dr = da*b + a*db` |
| `Div { dst, a, b }` | `r = a / b` | `dr = (da*b - a*db) / b²` |
| `Neg { dst, src }` | `r = -s` | `dr = -ds` |
| `Sqrt { dst, src }` | `r = √s` | `dr = ds / (2√s)` |
| `Sin { dst, src }` | `r = sin(s)` | `dr = cos(s) * ds` |
| `Cos { dst, src }` | `r = cos(s)` | `dr = -sin(s) * ds` |
| `Atan2 { dst, y, x }` | `r = atan2(y,x)` | `dr = (x*dy - y*dx) / (x²+y²)` |
| `Abs { dst, src }` | `r = |s|` | `dr = sign(s) * ds` |
| `Max { dst, a, b }` | `r = max(a,b)` | `dr = (a>=b) ? da : db` |
| `Min { dst, a, b }` | `r = min(a,b)` | `dr = (a<=b) ? da : db` |
| `StoreResidual { idx, src }` | output[idx] = r | `J[idx][i] = dr` |

### 1b. The `differentiate` function

```rust
/// Derive Jacobian opcodes from residual opcodes via forward-mode AD.
///
/// For each variable index `var_idx` in `0..n_vars`, this creates a
/// derivative trace that computes ∂(residuals)/∂(var[var_idx]).
///
/// The output is a stream of opcodes that:
/// 1. Computes all primal values (same as the residual stream)
/// 2. For each variable, propagates tangent values through the computation
/// 3. Stores the resulting Jacobian entries via StoreJacobianIndexed
pub fn differentiate(
    residual_ops: &[ConstraintOp],
    n_vars: usize,
    n_residuals: usize,
    active_vars: &[u32],  // Which variable indices have non-zero derivatives
) -> (Vec<ConstraintOp>, Vec<JacobianEntry>) {
    // ...
}
```

**Key optimization**: Not every residual depends on every variable. The
`active_vars` parameter (populated from `Lowerable::variable_indices()`) avoids
generating zero-derivative traces. For a distance constraint on 2 points out of
1000, we only differentiate w.r.t. 4 variables, not 2000.

### 1c. Sparse tangent propagation

Naive forward-mode AD with N variables requires N passes through the computation.
For sparse Jacobians, we can do much better:

```rust
fn differentiate_sparse(
    residual_ops: &[ConstraintOp],
    dependency_map: &HashMap<u32, Vec<u32>>,  // residual_idx → active var indices
) -> (Vec<ConstraintOp>, Vec<JacobianEntry>) {
    // For each residual, determine which variables it depends on
    // (by analyzing LoadVar ops in the residual's computation).
    // Only propagate tangents for those variables.
    //
    // For a system with M residuals each depending on K variables:
    //   Naive: O(M * N) tangent ops
    //   Sparse: O(M * K) tangent ops
    //
    // K is typically 2-6 for geometric constraints, so this is a massive win.
}
```

### 1d. Dead tangent elimination

After generating the tangent trace, run a simple dead-code elimination pass:
any tangent register that is never read (because it's a constant zero that got
optimized away) can be removed along with the ops that compute it.

```rust
fn eliminate_dead_tangents(ops: &mut Vec<ConstraintOp>) {
    // 1. Find all registers that are read (appear as src/a/b in any op)
    // 2. Remove ops whose dst register is never read
    // 3. Repeat until fixpoint (removing one op may make others dead)
}
```

### 1e. Integration: auto-derive Jacobians during lowering

Modify `lower_constraints()` in `jit/lower.rs`:

```rust
pub fn lower_constraints<L: Lowerable>(
    constraints: &[L],
    n_vars: usize,
    dimension: usize,
) -> CompiledConstraints {
    let n_residuals: usize = constraints.iter().map(|c| c.residual_count()).sum();
    let mut cc = CompiledConstraints::new(n_vars, n_residuals);

    // Lower residuals (same as before)
    let mut residual_emitter = OpcodeEmitter::new();
    let mut ctx = LoweringContext::new(dimension, n_vars);
    for constraint in constraints {
        constraint.lower_residual(&mut residual_emitter, &ctx);
        for _ in 0..constraint.residual_count() { ctx.next_residual(); }
    }
    cc.residual_ops = residual_emitter.into_ops();

    // NEW: Auto-derive Jacobian from residual ops
    let active_vars = collect_active_vars(&cc.residual_ops);
    let (jacobian_ops, pattern) = differentiate(
        &cc.residual_ops, n_vars, n_residuals, &active_vars,
    );
    cc.jacobian_ops = jacobian_ops;
    cc.jacobian_pattern = pattern;
    cc.jacobian_nnz = cc.jacobian_pattern.len();

    cc
}
```

### 1f. Make `lower_jacobian()` optional

With auto-derivation, the `Lowerable` trait can make `lower_jacobian()` optional:

```rust
pub trait Lowerable {
    fn lower_residual(&self, emitter: &mut OpcodeEmitter, ctx: &LoweringContext);
    fn residual_count(&self) -> usize;
    fn variable_indices(&self) -> Vec<usize>;

    /// Optional hand-written Jacobian lowering.
    ///
    /// If not provided (default), the Jacobian is automatically derived
    /// from lower_residual() via forward-mode AD.
    fn lower_jacobian(&self, _emitter: &mut OpcodeEmitter, _ctx: &LoweringContext) {
        // Default: no-op. Auto-derivation will be used instead.
    }

    /// Whether this constraint provides a hand-written Jacobian.
    fn has_custom_jacobian(&self) -> bool { false }
}
```

### 1g. Validation: AD vs hand-written

For every constraint that has both a hand-written `lower_jacobian()` and the
auto-derived version, add a cross-validation test:

```rust
#[test]
fn test_autodiff_matches_handwritten_distance() {
    let constraints = vec![DistanceConstraint::<2>::new(0, 1, 5.0)];

    // Lower with hand-written Jacobian
    let cc_hand = lower_constraints_with_hand_jacobian(&constraints, 4, 2);

    // Lower with auto-derived Jacobian
    let cc_auto = lower_constraints_with_auto_jacobian(&constraints, 4, 2);

    // Compile both, evaluate at multiple test points
    // Assert Jacobian values match to 1e-12
}
```

Run this for **all** constraint types to validate the AD implementation against
known-correct hand-written Jacobians.

### 1h. Remove hand-written Jacobian lowerings (optional, after validation)

Once auto-derivation is validated, the hand-written `lower_jacobian()` impls in
`geometry_lowering.rs` can be removed. Keep them as tests (auto-derive, compare
to hand-written), but stop using them in production.

This cuts `geometry_lowering.rs` roughly in half and means new constraint types
only need `lower_residual()`.

**Acceptance criteria**: Auto-derived Jacobians match hand-written Jacobians to
< 1e-12 for all constraint types. New constraints only require `lower_residual()`.
Full test suite passes.

---

## Step 2: Compiled Newton Steps

**Files**: new `jit/compiled_step.rs`, modifications to `solver/jit_solver.rs`

**Problem**: The JIT compiles residual and Jacobian evaluation but the linear
solve (`J·δ = -r`) is still general-purpose. For small clusters (N < ~30 vars),
the overhead of forming the matrix, calling LU decomposition, and doing
back-substitution is significant relative to the evaluation cost.

**Approach**: For small clusters where the Jacobian sparsity structure is known
at compile time, emit a fully unrolled LU decomposition and back-substitution
as native code.

### 2a. When this is appropriate

This optimization is only viable for small, dense clusters:
- N ≤ 30 variables (LU has N³/3 operations, so 30³/3 = 9000 ops — acceptable
  for JIT)
- Square or nearly-square systems (N ≈ M)
- Known sparsity structure (many zeros in J can be skipped)

For larger clusters, the general-purpose linear solver is more appropriate.

### 2b. Symbolic LU decomposition

Given the Jacobian sparsity pattern (known at compile time), perform symbolic
LU factorization to determine:
1. Which entries in L and U are structurally non-zero
2. The fill-in pattern (entries that start zero but become non-zero during LU)
3. The elimination order

```rust
struct SymbolicLU {
    /// Non-zero positions in L (below diagonal)
    l_pattern: Vec<(usize, usize)>,
    /// Non-zero positions in U (above diagonal, including diagonal)
    u_pattern: Vec<(usize, usize)>,
    /// Permutation for partial pivoting (optional)
    perm: Vec<usize>,
}

fn symbolic_lu(pattern: &[(usize, usize)], n: usize) -> SymbolicLU {
    // Analyze sparsity to determine non-zero structure of L and U.
    // No numerical computation — this is done at compile time.
}
```

### 2c. Emit unrolled LU + solve

```rust
fn emit_lu_solve(
    emitter: &mut OpcodeEmitter,
    symbolic: &SymbolicLU,
    j_regs: &HashMap<(usize, usize), Reg>,  // Jacobian entry registers
    r_regs: &[Reg],                          // Residual registers
) -> Vec<Reg> {
    // Emit opcodes for:
    // 1. LU decomposition (only non-zero entries, fully unrolled)
    //    - For each column k:
    //      - For each row i > k with L[i][k] non-zero:
    //        - Compute multiplier: L[i][k] = A[i][k] / A[k][k]
    //        - Update row i: A[i][j] -= L[i][k] * A[k][j] for non-zero j
    //
    // 2. Forward substitution: L·y = -r
    //    - Only operate on non-zero entries of L
    //
    // 3. Back substitution: U·δ = y
    //    - Only operate on non-zero entries of U
    //
    // 4. Return δ registers (the Newton step)
}
```

### 2d. Compile a full Newton iteration

```rust
fn compile_newton_step(
    compiler: &mut JITCompiler,
    compiled: &CompiledConstraints,
) -> Result<FuncId, JITError> {
    // Function signature:
    //   fn(vars: *const f64, delta: *mut f64) -> f64
    //   Returns residual norm (for convergence check).
    //
    // Emitted code:
    // 1. Load all variables
    // 2. Compute residuals (from residual_ops)
    // 3. Compute Jacobian entries (from jacobian_ops or auto-derived)
    // 4. Compute residual norm = sqrt(sum(r_i²))
    // 5. LU decompose Jacobian (symbolic structure, numerical values)
    // 6. Solve for delta
    // 7. Store delta to output buffer
    // 8. Return residual norm
}
```

### 2e. Use compiled steps in the solver

```rust
impl JITSolver {
    fn solve_compiled_step(
        &self,
        step_fn: &CompiledNewtonStep,
        x0: &[f64],
    ) -> SolveResult {
        let n = step_fn.variable_count();
        let mut x = x0.to_vec();
        let mut delta = vec![0.0; n];

        for iteration in 0..self.config.max_iterations {
            // One call does: evaluate residuals + Jacobian + LU solve
            let norm = step_fn.execute(&x, &mut delta);

            if norm < self.config.tolerance {
                return SolveResult::Converged { ... };
            }

            // Update: x += delta
            for i in 0..n {
                x[i] += delta[i];
            }
        }

        SolveResult::NotConverged { ... }
    }
}
```

### 2f. Size gate

Add logic to `JITSolver` to select the compiled-step path only for small clusters:

```rust
const MAX_COMPILED_STEP_VARS: usize = 30;

fn should_compile_full_step(n_vars: usize, n_residuals: usize) -> bool {
    n_vars <= MAX_COMPILED_STEP_VARS
        && n_residuals <= MAX_COMPILED_STEP_VARS * 2
        && n_vars == n_residuals  // Square systems only
}
```

**Acceptance criteria**: Compiled Newton steps produce the same solutions as the
standard solver to within 1e-10. Benchmarks show speedup on small clusters (< 30
vars). Falls back to standard path for larger clusters.

---

## Step 3: Symbolic Sparsity Analysis

**Files**: new `jit/sparsity.rs`

**Problem**: The current solver discovers the Jacobian sparsity pattern at runtime
by evaluating the Jacobian and inspecting which entries are non-zero. The JIT
system knows the pattern at compile time but doesn't expose it for broader use.

**Approach**: Extract structural information from the opcode IR and make it
available to the solver infrastructure for:
1. Static Jacobian allocation (allocate once, reuse every iteration)
2. Optimal variable ordering for sparse factorization
3. Graph coloring for compressed Jacobian evaluation

### 3a. Static dependency analysis

```rust
/// Analyze an opcode stream to determine which residuals depend on which variables.
///
/// Returns a map: residual_index → set of variable indices that affect it.
pub fn analyze_dependencies(
    ops: &[ConstraintOp],
) -> HashMap<u32, HashSet<u32>> {
    // Walk the opcode stream.
    // Track which variables each register transitively depends on.
    // When a StoreResidual is encountered, record the variable set.
    //
    // This is a forward dataflow analysis:
    //   deps[LoadVar { var_idx }] = { var_idx }
    //   deps[LoadConst] = {}
    //   deps[Add { a, b }] = deps[a] ∪ deps[b]
    //   deps[Mul { a, b }] = deps[a] ∪ deps[b]
    //   ... etc for all ops ...
    //   deps[StoreResidual { idx, src }] → record deps[src] for residual idx
}
```

### 3b. Structural Jacobian matrix

```rust
/// Structural sparsity pattern of the Jacobian matrix.
///
/// Unlike the numerical Jacobian (which has concrete f64 values), this
/// represents which entries are structurally non-zero — i.e., which entries
/// *could* be non-zero for some variable values.
pub struct JacobianStructure {
    /// Number of rows (residuals).
    pub m: usize,
    /// Number of columns (variables).
    pub n: usize,
    /// Structurally non-zero entries: (row, col).
    pub nonzeros: Vec<(usize, usize)>,
    /// CSR row pointers (for efficient row access).
    pub row_ptr: Vec<usize>,
    /// CSR column indices.
    pub col_ind: Vec<usize>,
}

impl JacobianStructure {
    /// Build from dependency analysis.
    pub fn from_dependencies(deps: &HashMap<u32, HashSet<u32>>, m: usize, n: usize) -> Self {
        // ...
    }

    /// Compute fill-reducing ordering (e.g., AMD, COLAMD).
    pub fn compute_ordering(&self) -> Vec<usize> {
        // Approximate Minimum Degree ordering.
        // This reorders variables to minimize fill-in during LU/Cholesky.
        // Well-known algorithm — use or port a simple implementation.
    }

    /// Compute graph coloring for compressed Jacobian evaluation.
    pub fn compute_coloring(&self) -> Vec<usize> {
        // Greedy graph coloring on the column intersection graph.
        // Two columns with the same color have no shared rows.
        // This enables evaluating multiple Jacobian columns simultaneously.
    }
}
```

### 3c. Pre-allocated Jacobian workspace

```rust
/// Pre-allocated workspace for Jacobian evaluation.
///
/// Created once from the JacobianStructure, reused every iteration.
/// Eliminates per-iteration allocation of Vec<(usize, usize, f64)>.
pub struct JacobianWorkspace {
    /// CSR values array — filled by JIT, read by linear solver.
    pub values: Vec<f64>,
    /// CSR row pointers (constant after creation).
    pub row_ptr: Vec<usize>,
    /// CSR column indices (constant after creation).
    pub col_ind: Vec<usize>,
}
```

### 3d. Integration with compiled Jacobian

Have the JIT's `write_jacobian_dense` (from Level 2) write into the
`JacobianWorkspace` CSR arrays instead:

```rust
impl JITFunction {
    pub fn write_jacobian_csr(&self, vars: &[f64], workspace: &mut JacobianWorkspace) {
        // JIT writes directly into workspace.values.
        // The structure (row_ptr, col_ind) is fixed and matches the JIT's output order.
        unsafe {
            (self.jacobian_csr_fn)(vars.as_ptr(), workspace.values.as_mut_ptr());
        }
    }
}
```

**Acceptance criteria**: Sparsity analysis correctly identifies the Jacobian
structure. Pre-allocated workspace eliminates per-iteration allocation. Variable
ordering produces measurably less fill-in than natural ordering on 50x50+ systems.

---

## Step 4: SIMD Vectorization

**Files**: new `jit/simd.rs`, modifications to `jit/cranelift.rs`

**Problem**: Constraints of the same type (e.g., 100 distance constraints) perform
structurally identical computations on different data. The current JIT evaluates
them sequentially, one at a time.

**Approach**: Group same-type constraints and emit SIMD instructions to evaluate
4 (AVX2) or 2 (NEON) constraints simultaneously.

### 4a. Constraint grouping

```rust
/// Group constraints by their "structural signature" (same opcode sequence).
fn group_by_structure(constraints: &[impl Lowerable]) -> Vec<Vec<usize>> {
    // Two constraints have the same structure if their residual lowering
    // produces the same sequence of opcodes (ignoring register numbers
    // and constant values, but matching opcode types and variable patterns).
    //
    // For example: all DistanceConstraint<2> instances produce:
    //   LoadVar, LoadVar, LoadVar, LoadVar, Sub, Sub, Mul, Mul, Add, Sqrt, LoadConst, Sub, StoreResidual
    // regardless of which points they reference.
}
```

### 4b. SIMD opcode emission

For a group of 4 structurally identical constraints, emit vector instructions:

```rust
fn emit_simd_group(
    builder: &mut FunctionBuilder,
    group: &[&CompiledSingleConstraint],
) {
    // Instead of:
    //   r0 = load vars[p1.x]  (constraint 0)
    //   r1 = load vars[p1.x]  (constraint 1)
    //   r2 = load vars[p1.x]  (constraint 2)
    //   r3 = load vars[p1.x]  (constraint 3)
    //   ... compute each independently ...
    //
    // Emit:
    //   v0 = gather [vars[c0.p1.x], vars[c1.p1.x], vars[c2.p1.x], vars[c3.p1.x]]
    //   v1 = gather [vars[c0.p1.y], vars[c1.p1.y], vars[c2.p1.y], vars[c3.p1.y]]
    //   ... vector arithmetic ...
    //   scatter v_result → [residuals[c0.idx], residuals[c1.idx], ...]
}
```

### 4c. Cranelift SIMD support

Cranelift supports SIMD types (`I8X16`, `F32X4`, `F64X2`). For AVX2 on x86_64,
we can use `F64X4` (256-bit). For NEON on aarch64, `F64X2` (128-bit).

```rust
fn vector_width() -> usize {
    if cfg!(target_arch = "x86_64") { 4 }  // AVX2: 256-bit / 64-bit = 4
    else if cfg!(target_arch = "aarch64") { 2 }  // NEON: 128-bit / 64-bit = 2
    else { 1 }  // No vectorization
}
```

### 4d. Gather/scatter for non-contiguous data

The variable arrays are not laid out for SIMD access — constraint 0's variables
are at different offsets than constraint 1's. We need gather loads:

```rust
// Gather 4 f64 values from non-contiguous memory locations
fn emit_gather_f64x4(
    builder: &mut FunctionBuilder,
    base: Value,
    offsets: [i32; 4],
) -> Value {
    // Load each value individually, then pack into a vector.
    // On x86_64 with AVX2, this becomes VGATHERDPD.
    // On platforms without gather support, it's 4 scalar loads + insert.
    let v0 = builder.ins().load(F64, MemFlags::trusted(), base, offsets[0]);
    let v1 = builder.ins().load(F64, MemFlags::trusted(), base, offsets[1]);
    let v2 = builder.ins().load(F64, MemFlags::trusted(), base, offsets[2]);
    let v3 = builder.ins().load(F64, MemFlags::trusted(), base, offsets[3]);

    // Build vector (depends on Cranelift's vector construction API)
    // This may require a helper sequence.
}
```

### 4e. Handling remainder groups

If there are 17 distance constraints, process 4 groups of 4 = 16 with SIMD,
then 1 remainder constraint with scalar code.

### 4f. Expected speedup

SIMD vectorization gives a theoretical 2-4x speedup on the evaluation phase.
However:
- Gather/scatter overhead may reduce this to 1.5-2.5x on non-contiguous data
- The evaluation phase is already not the bottleneck for most problem sizes
- The real win is combining SIMD with fused evaluation (Level 2) so that the
  entire residual + Jacobian computation is vectorized

This step should only be pursued after Level 2 benchmarks confirm that evaluation
time is a significant fraction of total solve time for the target workloads.

**Acceptance criteria**: SIMD evaluation produces identical results to scalar
evaluation. Benchmarks show measurable improvement on systems with 100+ same-type
constraints. Falls back to scalar on unsupported platforms.

---

## Step 5: V3 Architecture Integration

**Files**: modifications span the V3 codebase (which doesn't exist yet — this
step is designed to be built alongside V3)

**Problem**: The current JIT targets the old `GeometricConstraint<D>` trait and
`ConstraintSystem<D>`, both of which are being replaced by V3's `Constraint`
trait and new `ConstraintSystem`. The JIT must be designed into V3 from the start,
not bolted on afterward.

### 5a. Add optional lowering to the V3 `Constraint` trait

In the new `constraint/mod.rs`:

```rust
pub trait Constraint: Send + Sync {
    // ... existing V3 methods (id, name, entity_ids, param_ids, etc.) ...

    /// Lower this constraint's residual to JIT opcodes.
    ///
    /// If not implemented (default), the constraint cannot be JIT-compiled
    /// and the system falls back to interpreted evaluation for the cluster
    /// containing this constraint.
    #[cfg(feature = "jit")]
    fn lower_residual(
        &self,
        _emitter: &mut crate::jit::OpcodeEmitter,
        _param_mapping: &crate::jit::ParamMapping,
    ) -> bool {
        false
    }

    // Note: lower_jacobian() is NOT in the trait. Jacobians are auto-derived
    // from lower_residual() via the AD system (Step 1).
}
```

**Key difference from the old API**: The `LoweringContext` is replaced by a
`ParamMapping` that maps `ParamId`s to variable indices in the JIT function's
input array. This aligns with V3's `ParamStore`/`SolverMapping` architecture.

```rust
/// Maps ParamIds to variable indices for JIT compilation.
pub struct ParamMapping {
    map: HashMap<ParamId, u32>,
}

impl ParamMapping {
    pub fn var_index(&self, param: ParamId) -> Option<u32> {
        self.map.get(&param).copied()
    }
}
```

### 5b. `ReducedSubProblem` carries JIT functions

In V3, `ReducedSubProblem` is the bridge between the new `Constraint` API and
the old `Problem` trait. It's the natural place to cache JIT functions:

```rust
pub(crate) struct ReducedSubProblem<'a> {
    store: &'a ParamStore,
    mapping: SolverMapping,
    constraints: Vec<&'a dyn Constraint>,
    initial_values: Vec<f64>,

    /// Cached JIT function for this cluster (if all constraints support lowering).
    #[cfg(feature = "jit")]
    jit_fn: Option<JITFunction>,
}

impl Problem for ReducedSubProblem<'_> {
    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        #[cfg(feature = "jit")]
        if let Some(ref jit) = self.jit_fn {
            let mut residuals = vec![0.0; self.residual_count()];
            jit.evaluate_residuals(x, &mut residuals);
            return residuals;
        }

        // Interpreted fallback
        // ...
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        #[cfg(feature = "jit")]
        if let Some(ref jit) = self.jit_fn {
            // Use direct dense or CSR assembly
        }

        // Interpreted fallback
        // ...
    }
}
```

### 5c. `ChangeTracker` triggers recompilation

V3's `ChangeTracker` knows which clusters are dirty (constraints added/removed,
topology changed). Connect this to the `ClusterJIT` cache from Level 2:

```rust
impl ConstraintSystem {
    pub fn solve(&mut self) -> SystemResult {
        let dirty_clusters = self.tracker.dirty_clusters();

        for cluster_id in dirty_clusters {
            // Recompile JIT for this cluster
            self.jit_cache.invalidate(cluster_id);
        }

        // Solve each cluster (using JIT if available)
        for cluster in &self.clusters {
            let jit_fn = self.jit_cache.get_or_compile(cluster, &self.constraints);
            let sub = self.build_sub_problem(cluster, jit_fn);
            // ...
        }
    }
}
```

### 5d. Sketch2D constraints implement `lower_residual`

The V3 geometry plugin (`sketch2d`) implements `lower_residual()` for all its
constraint types. With auto-derivation (Step 1), this is the only JIT method
each constraint needs:

```rust
// In sketch2d/constraints/distance.rs
impl Constraint for DistanceConstraint {
    // ... standard V3 methods ...

    #[cfg(feature = "jit")]
    fn lower_residual(
        &self,
        emitter: &mut OpcodeEmitter,
        mapping: &ParamMapping,
    ) -> bool {
        let p1_x = emitter.load_var(mapping.var_index(self.p1_x)?);
        let p1_y = emitter.load_var(mapping.var_index(self.p1_y)?);
        let p2_x = emitter.load_var(mapping.var_index(self.p2_x)?);
        let p2_y = emitter.load_var(mapping.var_index(self.p2_y)?);

        let dx = emitter.sub(p2_x, p1_x);
        let dy = emitter.sub(p2_y, p1_y);
        let dist_sq = emitter.add(emitter.square(dx), emitter.square(dy));
        let dist = emitter.safe_distance(dist_sq, 1e-15);
        let target = emitter.const_f64(self.target);
        let residual = emitter.sub(dist, target);
        emitter.store_residual(/* current index */, residual);
        true
    }
}
```

**Acceptance criteria**: JIT is a first-class feature of V3. Adding a new
constraint type requires implementing only `lower_residual()` (no Jacobian code).
The `ConstraintSystem` automatically compiles clusters when all constraints
support lowering, recompiles dirty clusters incrementally, and falls back
gracefully when any constraint doesn't support JIT.

---

## Dependency Graph

```
Step 1 (autodiff) ──────────────────────┐
    │                                    │
    └── Step 2 (compiled Newton step)    │
                                         │
Step 3 (sparsity) ──────────────────────┤
                                         │
Step 4 (SIMD) ─ (optional, data-driven) │
                                         │
Step 5 (V3 integration) ────────────────┘
        ↑
        depends on V3 being built
```

Step 1 (autodiff) is the highest priority and highest leverage. It should be
done first. Step 5 is done alongside V3 development, not after.

Steps 2, 3, and 4 are independent of each other but all depend on having solid
benchmarks from Level 2 to justify the investment.

---

## Files Changed Summary

| File | Change |
|------|--------|
| `jit/autodiff.rs` | New: forward-mode AD on opcode streams (~500 LOC) |
| `jit/compiled_step.rs` | New: symbolic LU, compiled Newton iteration (~400 LOC) |
| `jit/sparsity.rs` | New: dependency analysis, JacobianStructure (~300 LOC) |
| `jit/simd.rs` | New: SIMD grouping and vectorized emission (~400 LOC) |
| `jit/lower.rs` | Auto-derive Jacobians, make lower_jacobian optional (~100 LOC) |
| `jit/cranelift.rs` | SIMD instruction emission helpers (~200 LOC) |
| `jit/mod.rs` | Re-exports, ParamMapping (~50 LOC) |
| `solver/jit_solver.rs` | Compiled step integration (~100 LOC) |
| V3 `constraint/mod.rs` | Add lower_residual() to Constraint trait (~50 LOC) |
| V3 `solve/sub_problem.rs` | JIT caching in ReducedSubProblem (~100 LOC) |
| V3 `system.rs` | ChangeTracker ↔ ClusterJIT integration (~100 LOC) |
| V3 `sketch2d/constraints/*.rs` | lower_residual() for each constraint (~700 LOC) |

**Estimated total**: ~3,000 new LOC.

---

## Priority Order

1. **Step 1 (Autodiff)** — Do first. Highest leverage: eliminates half of the
   per-constraint JIT code and removes an entire class of bugs. Required for
   Step 5 (V3 integration assumes auto-derivation).

2. **Step 5 (V3 integration)** — Do alongside V3 development. Not a separate
   phase but a design constraint woven into V3 from the start.

3. **Step 3 (Sparsity)** — Do when tackling large systems. The analysis is useful
   even without JIT (feeds into sparse solver ordering).

4. **Step 2 (Compiled steps)** — Do when benchmarks show small-cluster overhead
   matters. High implementation complexity for narrow applicability.

5. **Step 4 (SIMD)** — Do last, only if benchmarks show evaluation is still a
   bottleneck after all other optimizations. Most speculative ROI.
