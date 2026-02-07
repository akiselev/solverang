# Level 2: Make It Fast

**Goal**: Measurable, benchmarked speedup on realistic workloads.

**Prerequisite**: Level 1 complete (JIT is correct and reachable from the API).

---

## Overview

Level 1 makes JIT work. Level 2 makes it worth using. The focus is on reducing
per-iteration overhead through three techniques: fused evaluation (compute
residuals and Jacobian in one pass), direct sparse matrix assembly (skip the COO
intermediate), and per-cluster compilation (compile only what changed). All claims
are validated by a new JIT benchmark suite.

**Estimated effort**: ~1,500 new/modified LOC across 6 files.

---

## Step 1: Establish Baselines with JIT Benchmarks

**Files**: new `benches/jit_benchmarks.rs`

Before optimizing anything, we need numbers. The benchmark suite measures the
quantities that matter for deciding when JIT is worthwhile.

### 1a. Benchmark structure

Use Criterion.rs (already a dev dependency). Create synthetic constraint systems
at controlled sizes:

```rust
fn make_grid_system(n: usize) -> (ConstraintSystem<2>, Vec<f64>) {
    // Create an n×n grid of points with:
    // - Horizontal distance constraints between adjacent columns
    // - Vertical distance constraints between adjacent rows
    // - One fixed point at (0,0) to anchor the system
    //
    // Total: n² points → 2n² variables
    //        ~2n(n-1) distance constraints + 1 fixed constraint
    //
    // This is a realistic CAD-like workload.
}
```

### 1b. What to measure

**Compilation overhead**:
```rust
fn bench_jit_compile_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compile");
    for n in [5, 10, 20, 50] {
        let (system, _x0) = make_grid_system(n);
        group.bench_function(&format!("{}x{}_grid", n, n), |b| {
            b.iter(|| system.try_compile().unwrap())
        });
    }
}
```

**Per-iteration evaluation (residuals only)**:
```rust
fn bench_residual_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("residual_eval");
    for n in [5, 10, 20, 50] {
        let (system, x0) = make_grid_system(n);
        let jit_fn = system.try_compile().unwrap();

        group.bench_function(&format!("interpreted_{}x{}", n, n), |b| {
            b.iter(|| system.residuals(&x0))
        });
        group.bench_function(&format!("jit_{}x{}", n, n), |b| {
            let mut residuals = vec![0.0; jit_fn.residual_count()];
            b.iter(|| jit_fn.evaluate_residuals(&x0, &mut residuals))
        });
    }
}
```

**Per-iteration evaluation (Jacobian only)**:
```rust
fn bench_jacobian_evaluation(c: &mut Criterion) {
    // Same pattern: interpreted vs JIT, across grid sizes.
}
```

**End-to-end solve time (including compilation)**:
```rust
fn bench_end_to_end_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_e2e");
    for n in [5, 10, 20, 50] {
        let (system, x0) = make_grid_system(n);

        group.bench_function(&format!("interpreted_{}x{}", n, n), |b| {
            b.iter(|| {
                let mut solver = JITSolver::new(JITConfig::always_interpreted());
                solver.solve_constraint_system(&system, &x0)
            })
        });
        group.bench_function(&format!("jit_{}x{}", n, n), |b| {
            b.iter(|| {
                let mut solver = JITSolver::new(JITConfig::always_jit());
                solver.solve_constraint_system(&system, &x0)
            })
        });
    }
}
```

**Break-even analysis**:
```rust
fn bench_breakeven(c: &mut Criterion) {
    // For a 20x20 grid:
    // Measure: compile_time + N * jit_iteration_time
    // vs:      N * interpreted_iteration_time
    // Find N where JIT wins.
    // Output this as a custom metric.
}
```

**Re-solve amortization** (simulates interactive drag):
```rust
fn bench_drag_simulation(c: &mut Criterion) {
    // Compile once, then re-solve 100 times with slightly perturbed x0.
    // This is the interactive CAD use case.
    let (system, x0) = make_grid_system(20);
    let jit_fn = system.try_compile().unwrap();

    group.bench_function("drag_100_resolves_jit", |b| {
        b.iter(|| {
            let mut solver = JITSolver::new(JITConfig::always_jit());
            for i in 0..100 {
                let mut x = x0.clone();
                x[0] += 0.01 * i as f64; // Simulate dragging point 0
                solver.solve_with_jit(&jit_fn, &x);
            }
        })
    });
}
```

### 1c. Expected results and decision criteria

| Grid | Variables | Constraints | Expected: JIT wins? |
|------|-----------|-------------|---------------------|
| 5x5 | 50 | ~41 | No (compilation overhead > savings) |
| 10x10 | 200 | ~181 | Maybe (borderline) |
| 20x20 | 800 | ~761 | Likely (dispatch overhead matters) |
| 50x50 | 5000 | ~4901 | Yes (significant dispatch + allocation) |

If benchmarks show JIT is slower even at 50x50 for single solves, the value
proposition narrows to re-solve amortization only. That's fine — it just changes
what we optimize next.

**Acceptance criteria**: Benchmark suite runs via `cargo bench --features jit,geometry`.
Results are committed as a baseline comparison in `docs/notes/jit-benchmarks.md`.

---

## Step 2: Fused Residual + Jacobian Evaluation

**Files**: `jit/cranelift.rs`, `jit/opcodes.rs`, `solver/jit_solver.rs`

**Problem**: Currently, residuals and Jacobian are compiled as separate functions.
During Newton iteration, both are evaluated every iteration. Each function loads
the same variables from memory independently.

**Approach**: Compile a third function `evaluate_both(vars, residuals, jacobian)`
that performs a single pass, loading each variable once and computing both outputs.

### 2a. Add `compile_fused` method to `JITCompiler`

```rust
fn compile_fused(
    &mut self,
    compiled: &CompiledConstraints,
) -> Result<FuncId, JITError> {
    // Function signature: fn(vars: *const f64, residuals: *mut f64, jacobian: *mut f64)
    // Three pointer arguments.

    // Strategy:
    // 1. Emit residual opcodes first (they compute intermediate values)
    // 2. Emit Jacobian opcodes second (they can reuse loaded variables)
    // 3. The key optimization: share LoadVar registers between the two
    //    opcode streams by renumbering Jacobian registers to avoid conflicts
    //    with residual registers.
}
```

### 2b. Register renumbering for fused compilation

The residual and Jacobian opcode streams use independent register namespaces.
For fused compilation, we need to merge them:

```rust
fn renumber_ops(ops: &[ConstraintOp], offset: u16) -> Vec<ConstraintOp> {
    // Add `offset` to every register reference in the opcode stream.
    // This shifts the Jacobian registers to avoid collisions with residual registers.
}
```

Then the fused function is:

```
[residual ops with registers 0..R]
[jacobian ops with registers R+1..R+J, sharing LoadVar values where possible]
```

### 2c. Share variable loads

An optimization pass before Cranelift compilation:

```rust
fn share_loads(
    residual_ops: &[ConstraintOp],
    jacobian_ops: &mut Vec<ConstraintOp>,
    register_offset: u16,
) {
    // Build a map: var_idx → register in residual ops
    // For each LoadVar in jacobian ops, if the same var_idx was loaded
    // in residual ops, replace the LoadVar with a reference to the
    // residual register (no offset needed since it's in the shared space).
    // Remove the redundant LoadVar.
}
```

This eliminates redundant memory loads. For a distance constraint with 4 variables,
this saves 4 loads per constraint per iteration.

### 2d. Add `evaluate_both` to `JITFunction`

```rust
impl JITFunction {
    pub fn evaluate_both(&self, vars: &[f64], residuals: &mut [f64], jacobian: &mut [f64]) {
        unsafe {
            (self.fused_fn)(vars.as_ptr(), residuals.as_mut_ptr(), jacobian.as_mut_ptr());
        }
    }
}
```

### 2e. Use fused evaluation in `solve_with_jit`

In `jit_solver.rs`, replace the separate calls:

```rust
// Before:
jit_fn.evaluate_residuals(x.as_slice(), &mut residuals);
jit_fn.evaluate_jacobian(x.as_slice(), &mut jac_values);

// After:
jit_fn.evaluate_both(x.as_slice(), &mut residuals, &mut jac_values);
```

### 2f. Benchmarks

Add to `jit_benchmarks.rs`:

```rust
fn bench_fused_vs_separate(c: &mut Criterion) {
    // Compare: evaluate_residuals + evaluate_jacobian
    //     vs:  evaluate_both
    // Across grid sizes.
}
```

**Acceptance criteria**: Fused evaluation is measurably faster than separate calls
on 20x20+ grids. All existing correctness tests still pass.

---

## Step 3: Direct Sparse Jacobian Assembly

**Files**: `jit/cranelift.rs`, `solver/jit_solver.rs`

**Problem**: The JIT Jacobian function writes values into a flat `f64` array in
COO order. The solver then iterates this array and copies each entry into a
dense `DMatrix<f64>`. This intermediate step allocates and involves bounds checks.

**Approach**: For problems where the solver uses a dense matrix (the common case
in `jit_solver.rs`), have the JIT function write directly into the dense matrix's
column-major storage.

### 3a. Direct-to-dense Jacobian compilation

When the Jacobian sparsity pattern is known at compile time (it always is in our
system), we can compute the offset into a column-major dense matrix at JIT compile
time:

```rust
fn compile_jacobian_direct_dense(
    &mut self,
    compiled: &CompiledConstraints,
    m: usize, // rows (residuals)
    n: usize, // cols (variables)
) -> Result<FuncId, JITError> {
    // For each StoreJacobianIndexed { output_idx, src }:
    // Look up the (row, col) from jacobian_pattern[output_idx].
    // Compute dense offset = col * m + row (column-major for nalgebra).
    // Emit a store to jacobian_ptr + offset * 8.
    //
    // This replaces the flat COO array with direct writes to the matrix.
}
```

### 3b. Add a `write_jacobian_dense` method to `JITFunction`

```rust
impl JITFunction {
    /// Write Jacobian directly into a nalgebra DMatrix's storage.
    ///
    /// The matrix must be m×n (residuals × variables) and stored column-major.
    pub fn write_jacobian_dense(&self, vars: &[f64], matrix: &mut DMatrix<f64>) {
        debug_assert_eq!(matrix.nrows(), self.n_residuals);
        debug_assert_eq!(matrix.ncols(), self.n_vars);
        // Zero the matrix first (only non-zero entries are written)
        matrix.fill(0.0);
        unsafe {
            (self.jacobian_dense_fn)(vars.as_ptr(), matrix.as_mut_ptr());
        }
    }
}
```

### 3c. Update `solve_with_jit` to use direct assembly

```rust
// Before:
let mut jac_values = vec![0.0; jit_fn.jacobian_nnz()];
jit_fn.evaluate_jacobian(x.as_slice(), &mut jac_values);
let mut j = DMatrix::zeros(m, n);
for (entry, &val) in jit_fn.jacobian_pattern().iter().zip(jac_values.iter()) {
    j[(entry.row as usize, entry.col as usize)] = val;
}

// After:
let mut j = DMatrix::zeros(m, n);
jit_fn.write_jacobian_dense(x.as_slice(), &mut j);
```

This eliminates:
- The `jac_values` allocation
- The `zip` + bounds-checked indexing loop
- One pass over the data

### 3d. Optional: CSR assembly for sparse solver integration

If the `SparseSolver` is used, provide a CSR-direct variant. This is lower
priority since `JITSolver` currently uses dense matrices, but document the
approach:

```rust
// Future: JIT writes directly into CSR values array.
// The row_ptr and col_ind arrays are known at compile time
// (they come from the sparsity pattern).
// Only the values array is written at runtime.
```

**Acceptance criteria**: Direct dense assembly is faster than COO-then-copy on
20x20+ grids. Benchmarks show the improvement. Correctness tests pass.

---

## Step 4: Per-Cluster Incremental Compilation

**Files**: `solver/jit_solver.rs`, `jit/mod.rs`

**Problem**: When a single constraint changes in a large system, the entire
constraint system must be recompiled. For interactive CAD (add a constraint,
drag a point), this means unnecessary recompilation of unchanged clusters.

**Approach**: Leverage the existing decomposition module to compile each
independent cluster separately. Cache compiled functions per cluster. Recompile
only dirty clusters.

### 4a. `ClusterJIT` cache

```rust
/// Cached JIT compilation for a decomposed constraint system.
pub struct ClusterJIT {
    /// Per-cluster compiled functions, indexed by cluster ID.
    /// None = not yet compiled or cluster changed.
    clusters: Vec<Option<JITFunction>>,

    /// Hash of each cluster's constraint composition.
    /// Used to detect when recompilation is needed.
    fingerprints: Vec<u64>,
}

impl ClusterJIT {
    pub fn new() -> Self { ... }

    /// Compile or reuse JIT functions for each cluster.
    pub fn compile_clusters(
        &mut self,
        components: &[Component],
        system: &ConstraintSystem<D>,
    ) -> Vec<Option<&JITFunction>> {
        for (i, component) in components.iter().enumerate() {
            let fingerprint = hash_cluster(component, system);
            if self.fingerprints.get(i) != Some(&fingerprint) {
                // Cluster changed — recompile
                let jit_fn = compile_cluster(component, system);
                self.clusters[i] = jit_fn;
                self.fingerprints[i] = fingerprint;
            }
        }
        // Return references to compiled functions
    }
}
```

### 4b. Cluster fingerprinting

A cluster's fingerprint is a hash of:
- The set of constraint types and their parameters (target distances, angles, etc.)
- The variable-to-constraint connectivity (which variables each constraint touches)

This does **not** include variable values (which change every iteration) — only
the structural topology that determines the compiled code.

```rust
fn hash_cluster(component: &Component, system: &ConstraintSystem<D>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    // Hash constraint types and parameters
    for &ci in &component.constraint_indices {
        system.constraints[ci].type_id().hash(&mut hasher);
        // Hash parameters that affect compiled code (e.g., target distance)
    }

    // Hash connectivity
    for &ci in &component.constraint_indices {
        for &vi in &component.variable_indices {
            // Hash the (constraint, variable) edge
        }
    }

    hasher.finish()
}
```

### 4c. Integration with parallel solver

The parallel solver already decomposes and solves clusters independently. Add
JIT as an option for per-cluster solving:

```rust
// In parallel solver's per-component solve:
fn solve_component(
    component: &Component,
    jit_fn: Option<&JITFunction>,
    config: &SolverConfig,
) -> SolveResult {
    if let Some(jit) = jit_fn {
        // Use JIT-compiled evaluation
        jit_solver.solve_with_jit(jit, &x0)
    } else {
        // Fall back to interpreted
        solver.solve(&component_problem, &x0)
    }
}
```

### 4d. Benchmarks

```rust
fn bench_incremental_recompile(c: &mut Criterion) {
    // 1. Create 20x20 grid → ~4 independent clusters
    // 2. Compile all clusters
    // 3. Modify one constraint in one cluster
    // 4. Measure: recompile_all vs recompile_dirty_only
}
```

**Acceptance criteria**: Modifying one constraint in one cluster only recompiles
that cluster. The `ClusterJIT` cache correctly identifies unchanged clusters.
Benchmarks show incremental recompilation is faster than full recompilation on
multi-cluster systems.

---

## Step 5: Calibrate JIT Threshold

**Files**: `jit/mod.rs`

**Problem**: The current `jit_threshold` of 1000 (residuals × estimated iterations)
is an arbitrary guess with no empirical backing.

**Approach**: Use benchmark data from Steps 1-4 to determine the actual
break-even point.

### 5a. Analyze benchmark results

From the benchmark data, extract:
- `t_compile(N)`: JIT compilation time as a function of constraint count N
- `t_jit_iter(N)`: Per-iteration time with JIT
- `t_interp_iter(N)`: Per-iteration time with interpreted evaluation
- Typical iteration count `I` for convergence

The break-even is where:
```
t_compile(N) + I * t_jit_iter(N) < I * t_interp_iter(N)
```

Solving for N:
```
N_breakeven where t_compile(N) < I * (t_interp_iter(N) - t_jit_iter(N))
```

### 5b. Update threshold

Replace the hardcoded threshold with an empirically-derived value. Consider
making it a function of constraint count (not residuals × iterations), since
compilation cost is proportional to constraint count:

```rust
impl JITConfig {
    pub fn default() -> Self {
        Self {
            // Empirically determined: JIT wins for single-solve when
            // constraint_count > N (from benchmarks).
            jit_threshold: EMPIRICAL_VALUE,

            // For re-solve scenarios (drag), JIT wins much earlier:
            jit_threshold_resolves: EMPIRICAL_VALUE_2,
            ..
        }
    }
}
```

### 5c. Two-tier threshold

Add a `resolve_count_hint` parameter that lets callers indicate how many times
the system will be re-solved with the same topology:

```rust
pub fn should_jit(&self, n_constraints: usize, resolve_count_hint: usize) -> bool {
    if self.force_jit { return true; }
    if self.force_interpreted { return false; }

    if resolve_count_hint > 10 {
        // Interactive mode: lower threshold since compilation amortizes
        n_constraints > self.jit_threshold_interactive
    } else {
        // One-shot mode: higher threshold
        n_constraints > self.jit_threshold_oneshot
    }
}
```

**Acceptance criteria**: Threshold values are backed by benchmark data. The
default config produces optimal or near-optimal decisions for the benchmark
suite workloads.

---

## Dependency Graph

```
Step 1 (benchmarks) ─────────────────────────────────────────┐
    │                                                         │
    ├── Step 2 (fused eval)                                   │
    │       │                                                 │
    │       └── re-run benchmarks ────────────────────────────┤
    │                                                         │
    ├── Step 3 (direct assembly)                              │
    │       │                                                 │
    │       └── re-run benchmarks ────────────────────────────┤
    │                                                         │
    ├── Step 4 (per-cluster)                                  │
    │       │                                                 │
    │       └── re-run benchmarks ────────────────────────────┤
    │                                                         │
    └─────────────────────────────────────────────────────────┘
                                                              │
                                                    Step 5 (calibrate threshold)
```

Step 1 must come first. Steps 2, 3, 4 are independent of each other but each
requires re-running benchmarks to measure impact. Step 5 comes last, using
all accumulated data.

---

## Files Changed Summary

| File | Change |
|------|--------|
| `benches/jit_benchmarks.rs` | New benchmark file (~400 LOC) |
| `jit/cranelift.rs` | Fused compilation, direct dense assembly (~300 LOC) |
| `jit/opcodes.rs` | Register renumbering utility (~50 LOC) |
| `solver/jit_solver.rs` | Use fused eval, direct assembly, cluster JIT (~200 LOC) |
| `jit/mod.rs` | `ClusterJIT` cache, threshold calibration (~250 LOC) |
| `docs/notes/jit-benchmarks.md` | Benchmark results and analysis |

**Estimated total**: ~1,500 new/modified LOC.
