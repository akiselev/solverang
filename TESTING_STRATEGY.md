# Testing Strategy Analysis for Solverang

## Current State

Solverang already has a solid testing foundation:

| Category | What Exists |
|----------|-------------|
| **Unit tests** | ~412 `#[test]` functions embedded in 96 source files |
| **Integration tests** | 11 test files (~4,500 lines) in `crates/solverang/tests/` |
| **Property-based tests** | `proptest` v1.4 with custom strategies, 100–500 cases per property |
| **Reference tests** | MINPACK verification suite + 30+ NIST StRD problems |
| **Benchmarks** | 3 Criterion suites (scaling, comprehensive, NIST) |
| **Jacobian verification** | Analytical vs. finite-difference checks throughout |

The `.gitignore` already lists `mutants.out*/`, indicating `cargo-mutants` has been
used before but is not integrated into CI.

---

## Testing Strategies to Add

### 1. Mutation Testing (`cargo-mutants`)

**What it does:** Systematically modifies (mutates) source code — flipping `+` to `-`,
removing conditionals, replacing constants — and verifies that at least one test
fails for each mutation. Surviving mutants indicate weak or missing test assertions.

**Why it matters for solverang:** Numerical code often has subtle bugs where a sign
flip or off-by-one in an index still produces "close enough" results. Mutation testing
catches tests that assert too loosely.

**How to set it up:**

```bash
cargo install cargo-mutants
cargo mutants -p solverang --features geometry,parallel,sparse
```

**Priority targets for mutation testing:**
- `src/solver/newton_raphson.rs` — line search logic, convergence checks
- `src/solver/lm_adapter.rs` — parameter mapping to/from `levenberg-marquardt` crate
- `src/jacobian/numeric.rs` — finite-difference step sizes and formulas
- `src/decomposition.rs` — union-find algorithm, graph construction
- `src/geometry/constraints/` — each constraint's residual and Jacobian formulas

**Specific areas likely to have surviving mutants:**
- Tolerance comparisons (`< 1e-10` vs `<= 1e-10`)
- Sign of Jacobian entries (especially cross-terms in coupled constraints)
- Off-by-one in loop bounds for banded/sparse patterns
- Backtrack factor and Armijo condition in Newton-Raphson line search

---

### 2. Fuzzing (`cargo-fuzz` / `libFuzzer`)

**What it does:** Generates random byte streams, interprets them as structured inputs,
and runs code paths looking for panics, hangs, and memory safety violations. Unlike
property testing, fuzzing is open-ended and runs for minutes to hours.

**Why it matters for solverang:** The JIT compiler (`src/jit/`) contains `unsafe` code
with `std::mem::transmute` on function pointers. The solver handles arbitrary
floating-point inputs including NaN, infinity, and subnormals. Fuzzing is the best
tool for finding crashes in these areas.

**Setup:**

```bash
cargo install cargo-fuzz
cargo fuzz init  # creates fuzz/ directory
```

**Recommended fuzz targets:**

```
fuzz/fuzz_targets/
├── fuzz_solver_inputs.rs      # Random residual/Jacobian pairs → solver
├── fuzz_jit_compiler.rs       # Random constraint descriptions → JIT compile + execute
├── fuzz_constraint_system.rs  # Random point/constraint combos → geometry solver
└── fuzz_macro_input.rs        # Random expression trees → macro expansion
```

**Example fuzz target for solver inputs:**

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use solverang::{LMSolver, LMConfig, Problem};

struct FuzzProblem { data: Vec<f64>, n: usize }

impl Problem for FuzzProblem {
    fn name(&self) -> &str { "fuzz" }
    fn residual_count(&self) -> usize { self.n }
    fn variable_count(&self) -> usize { self.n }
    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        x.iter().zip(self.data.iter()).map(|(a, b)| a - b).collect()
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        (0..self.n).map(|i| (i, i, 1.0)).collect()
    }
    fn initial_point(&self, _: f64) -> Vec<f64> { vec![0.0; self.n] }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 || data.len() % 8 != 0 { return; }
    let floats: Vec<f64> = data.chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let n = floats.len().min(20);
    let problem = FuzzProblem { data: floats[..n].to_vec(), n };
    let solver = LMSolver::new(LMConfig::default());
    let _ = solver.solve(&problem, &vec![0.0; n]);
});
```

**Critical fuzz target — JIT unsafe code:**

The `cranelift.rs` module does:
```rust
residual_fn: unsafe { std::mem::transmute::<*const u8, JITFnPtr>(residual_ptr) },
```
A fuzz target should feed arbitrary constraint descriptions into the JIT compiler
and verify it either compiles successfully or returns an error — never segfaults.

---

### 3. Snapshot Testing (`insta`)

**What it does:** Captures the output of a computation and saves it to a file. Future
test runs compare against the saved snapshot. Changes require explicit `cargo insta
review` approval.

**Why it matters for solverang:** Solver output (iteration count, final residual,
solution vector) should be deterministic for a given problem and initial point.
Snapshot tests catch unintended behavioral regressions that tolerance-based assertions
miss — for example, a refactor that changes convergence from 5 iterations to 50 while
still meeting the tolerance.

**Setup:**

```toml
# Cargo.toml [dev-dependencies]
insta = { version = "1.39", features = ["yaml"] }
```

**What to snapshot:**

| Target | What to Capture |
|--------|----------------|
| NIST StRD problems | `{ solution, residual_norm, iterations }` for all 30+ problems |
| MINPACK problems | `{ solution, residual_norm, iterations }` for each starting point |
| Solver comparison | Full `SolveResult` for each solver on each test problem |
| JIT vs interpreted | Residual and Jacobian vectors for same inputs |
| Decomposition | Component assignment for known constraint graphs |
| Macro expansion | Generated Jacobian code (proc-macro output) |

**Example:**

```rust
use insta::assert_yaml_snapshot;
use solverang::test_problems::Rosenbrock;

#[test]
fn snapshot_rosenbrock_newton() {
    let problem = Rosenbrock::new();
    let solver = Solver::new(SolverConfig::default());
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    // Captures solution, residual_norm, iterations to snapshots/snapshot_rosenbrock_newton.snap
    assert_yaml_snapshot!(result);
}
```

**Particularly valuable for:**
- Detecting when a dependency update (e.g., `nalgebra`, `levenberg-marquardt`) silently
  changes numerical behavior
- Catching JIT codegen changes that produce different floating-point results
- Documenting expected performance characteristics per problem

---

### 4. Property-Based Testing Expansion

The existing property tests in `property_tests.rs` are good but focused on basic
invariants. Here are specific property expansions:

**A. Solver Idempotency**
```rust
// Solving from a converged solution should return the same solution
proptest! {
    fn prop_solver_idempotent(target in small_vec_strategy(1, 5)) {
        let problem = LinearProblem { target };
        let result1 = solver.solve(&problem, &x0);
        if let Converged { solution, .. } = result1 {
            let result2 = solver.solve(&problem, &solution);
            // result2 should converge in 0-1 iterations with same solution
        }
    }
}
```

**B. Solver Determinism**
```rust
// Same inputs should always produce same outputs
fn prop_solver_deterministic(target, x0) {
    let result1 = solver.solve(&problem, &x0);
    let result2 = solver.solve(&problem, &x0);
    assert_eq!(result1, result2); // Bitwise equal, not tolerance
}
```

**C. Jacobian Transpose Symmetry for Least-Squares**
```rust
// For least-squares: J^T * J should be symmetric positive semi-definite
fn prop_jtj_symmetric(problem, x) {
    let j = problem.jacobian_dense(x);
    let jtj = j.transpose() * j;
    assert!(is_symmetric(&jtj));
    assert!(eigenvalues_nonnegative(&jtj));
}
```

**D. Constraint Composition Properties**
```rust
// Adding a redundant constraint should not change the solution
// Removing an already-satisfied constraint should not change the solution
// Constraint residuals should scale linearly with deviation from solution
```

**E. Numerical Gradient Check with Higher-Order Differences**
The current verification uses first-order finite differences. Add central differences
and Richardson extrapolation for tighter Jacobian validation:
```rust
fn verify_jacobian_central(problem, x, epsilon) -> JacobianVerification {
    // f'(x) ≈ (f(x+h) - f(x-h)) / (2h)  — second-order accurate
}
```

---

### 5. Error Path / Negative Testing

The codebase defines 8 error variants in `SolveError` but several are never triggered
in tests:

**Untested error paths:**

| Error | Status | Suggested Test |
|-------|--------|----------------|
| `MaxIterationsExceeded` | Defined but never returned | Create problem with very slow convergence, set `max_iterations = 1` |
| `LineSearchFailed` | Returned in NR but no test triggers it | Create problem where all step sizes increase the residual |
| `SingularJacobian` | Tested in NR inline tests | Add property test with randomly generated singular Jacobians |
| `NonFiniteResiduals` | Tested | Expand with proptest strategies that inject NaN at random positions |
| `NonFiniteJacobian` | Tested | Same expansion as above |
| `DimensionMismatch` | Tested | Add tests for dynamic dimension changes mid-solve |

**Example — force line search failure:**
```rust
/// Problem where the residual increases for any step along Newton direction
struct LineSearchTrap;
impl Problem for LineSearchTrap {
    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        // Residual that has a saddle point at x=0 where Newton direction
        // points toward increasing residual
        vec![x[0].powi(3)]
    }
    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, 0, 3.0 * x[0].powi(2))]
    }
    fn initial_point(&self, _: f64) -> Vec<f64> { vec![0.0] }  // Jacobian is singular here
}
```

**Silent failure modes to test:**
- Jacobian entries with `row >= residual_count` or `col >= variable_count` are silently
  dropped in `jacobian_dense()`. Add a test that verifies this behavior (or changes it
  to return an error).
- Duplicate `(row, col)` entries in sparse Jacobian — last value wins silently.
- `residual_count()` returning different values across calls.

---

### 6. Configuration Boundary Testing

Solver configurations have numeric parameters with implicit valid ranges. None of
these boundaries are tested:

```rust
// SolverConfig parameters to boundary-test:
max_iterations: 0, 1, usize::MAX
tolerance: 0.0, f64::MIN_POSITIVE, f64::EPSILON, f64::NAN, f64::INFINITY
max_line_search_iterations: 0
min_step_size: 0.0, 1.0, f64::INFINITY
armijo_c: 0.0, 1.0, -1.0
backtrack_factor: 0.0, 0.5, 1.0, 2.0
svd_tolerance: 0.0, f64::NAN

// LMConfig parameters:
patience: 0
ftol/xtol/gtol: 0.0, f64::NAN, f64::INFINITY, negative values
```

These should be tested with proptest strategies that generate boundary values:
```rust
fn config_strategy() -> impl Strategy<Value = SolverConfig> {
    (0usize..1000, prop::num::f64::POSITIVE, 0usize..100, ...)
        .prop_map(|(max_iter, tol, ...)| SolverConfig { ... })
}
```

---

### 7. Contract / Design-by-Contract Testing

Verify invariants at module boundaries using debug assertions that become tests:

**Problem trait contract:**
```rust
fn validate_problem<P: Problem>(p: &P) -> Result<(), String> {
    let m = p.residual_count();
    let n = p.variable_count();
    assert!(m > 0, "residual_count must be positive");
    assert!(n > 0, "variable_count must be positive");
    let x0 = p.initial_point(1.0);
    assert_eq!(x0.len(), n, "initial_point length must equal variable_count");
    let r = p.residuals(&x0);
    assert_eq!(r.len(), m, "residuals length must equal residual_count");
    let j = p.jacobian(&x0);
    for (row, col, val) in &j {
        assert!(*row < m, "Jacobian row out of bounds");
        assert!(*col < n, "Jacobian col out of bounds");
        assert!(val.is_finite(), "Jacobian entry must be finite");
    }
    Ok(())
}
```

This can be run as a property test over all registered test problems:
```rust
#[test]
fn all_test_problems_satisfy_contract() {
    for problem in all_test_problems() {
        validate_problem(&problem).unwrap();
    }
}
```

---

### 8. Cross-Validation / Oracle Testing

Compare solverang against external solvers to verify correctness:

**A. JIT vs. Interpreted Oracle**
```rust
#[test]
fn jit_matches_interpreted() {
    for problem in geometric_test_problems() {
        let x = problem.initial_point(1.0);
        let interpreted_r = problem.residuals(&x);
        let interpreted_j = problem.jacobian(&x);
        let jit = JITCompiler::compile(&problem);
        let jit_r = jit.residuals(&x);
        let jit_j = jit.jacobian(&x);
        assert_eq_float_vec(&interpreted_r, &jit_r, 1e-12);
        assert_eq_float_vec(&interpreted_j, &jit_j, 1e-12);
    }
}
```

**B. Sparse vs. Dense Oracle**
```rust
#[test]
fn sparse_solver_matches_dense() {
    // Same problem solved by SparseSolver and Solver should produce same solution
}
```

**C. Parallel vs. Sequential Oracle**
```rust
#[test]
fn parallel_matches_sequential() {
    // ParallelSolver and sequential solver should produce identical results
}
```

---

### 9. Concurrency / Thread Safety Testing

The `parallel` feature uses Rayon for parallel decomposition solving. Additional
concurrency tests:

**A. Data Race Detection (run under Miri or ThreadSanitizer):**
```bash
RUSTFLAGS="-Z sanitizer=thread" cargo test -p solverang --features parallel
```

**B. Stress Tests:**
```rust
#[test]
fn parallel_solver_under_contention() {
    // Run many parallel solves concurrently to detect race conditions
    let handles: Vec<_> = (0..100).map(|_| {
        std::thread::spawn(|| {
            let problem = Rosenbrock::new();
            let solver = ParallelSolver::new();
            solver.solve(&problem, &problem.initial_point(1.0))
        })
    }).collect();
    for h in handles {
        assert!(h.join().unwrap().is_converged());
    }
}
```

**C. Determinism Under Parallelism:**
Floating-point addition is not associative. Verify parallel results are bitwise
identical across runs (or document acceptable deviation).

---

### 10. Performance Regression Testing

Beyond Criterion benchmarks, add automated performance gates:

**A. Use `criterion`'s `--save-baseline` for CI:**
```bash
cargo bench --features parallel,sparse -- --save-baseline main
# After changes:
cargo bench --features parallel,sparse -- --baseline main
```

**B. Test algorithmic complexity:**
```rust
#[test]
fn solver_scales_linearly_with_sparse_size() {
    let times: Vec<_> = [100, 200, 400, 800].iter().map(|&n| {
        let problem = generate_sparse_problem(n);
        let start = Instant::now();
        solver.solve(&problem, &x0);
        start.elapsed()
    }).collect();
    // Verify roughly O(n) scaling, not O(n²) or worse
    let ratio = times[3].as_secs_f64() / times[0].as_secs_f64();
    assert!(ratio < 20.0, "Scaling worse than expected: {ratio}x for 8x size");
}
```

---

## Prioritized Recommendations

| Priority | Strategy | Effort | Impact | Rationale |
|----------|----------|--------|--------|-----------|
| **1** | **Fuzzing the JIT compiler** | Medium | Critical | `unsafe` transmute + extern fn calls = potential memory safety bugs |
| **2** | **Mutation testing** | Low | High | Just run `cargo mutants`; immediately reveals weak assertions |
| **3** | **Error path testing** | Low | High | 2 error variants are dead code; others have gaps |
| **4** | **Snapshot testing (NIST/MINPACK)** | Low | Medium | Catch regressions from dependency updates |
| **5** | **Configuration boundary tests** | Low | Medium | Invalid configs could cause infinite loops or panics |
| **6** | **Cross-validation (JIT vs interpreted)** | Medium | High | Only way to verify JIT correctness systematically |
| **7** | **Property test expansion** | Medium | Medium | Idempotency, determinism, and composition properties |
| **8** | **Contract testing** | Low | Medium | One-time setup validates all future test problems |
| **9** | **Concurrency stress tests** | Medium | Medium | Rayon use needs thread-safety validation |
| **10** | **Performance regression tests** | Medium | Low | Prevent silent algorithmic regressions |

---

## Tool Summary

| Tool | Crate / Command | Purpose |
|------|----------------|---------|
| `cargo-mutants` | `cargo install cargo-mutants` | Mutation testing |
| `cargo-fuzz` | `cargo install cargo-fuzz` | libFuzzer-based fuzzing |
| `insta` | `insta = "1.39"` in dev-deps | Snapshot testing |
| `proptest` | Already in use | Property-based testing (expand) |
| `criterion` | Already in use | Performance benchmarks (add baselines) |
| Miri | `cargo +nightly miri test` | Undefined behavior detection |
| ThreadSanitizer | `RUSTFLAGS="-Z sanitizer=thread"` | Data race detection |
| `cargo-tarpaulin` / `llvm-cov` | `cargo install cargo-tarpaulin` | Code coverage metrics |
