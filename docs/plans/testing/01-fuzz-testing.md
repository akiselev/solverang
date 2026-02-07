# Plan 01: Fuzz Testing

## Goal

Fuzz testing is the highest-priority addition because solverang contains **`unsafe` code**
in the JIT module (`src/jit/cranelift.rs`) that uses `std::mem::transmute` to convert
raw pointers into function pointers. A malformed constraint description that produces
bad machine code could cause a segfault or memory corruption. Fuzzing is the only
practical way to explore this enormous input space.

Beyond the JIT, every solver must handle arbitrary `f64` inputs — NaN, infinity,
subnormals, and extreme magnitudes — without panicking. Property tests cover structured
random inputs; fuzzing covers the truly adversarial edge.

## Tool Selection

**Recommendation: `cargo-fuzz`** (wraps libFuzzer)

| Criterion | `cargo-fuzz` | `afl.rs` |
|-----------|-------------|----------|
| Setup complexity | Low (one command) | Medium (requires AFL++ install) |
| Structured fuzzing | `arbitrary` crate integration | Requires manual |
| CI integration | Simple (`cargo fuzz run -- -max_total_time=N`) | More complex |
| Coverage feedback | Built-in (SanitizerCoverage) | Built-in (AFL instrumentation) |
| Rust ecosystem support | First-class | Good but less common |

`cargo-fuzz` is the standard in the Rust ecosystem. If deeper exploration is needed
later, `afl.rs` can be added as a complement — they find different bugs due to
different mutation strategies.

## Setup Steps

### 1. Install cargo-fuzz

```bash
cargo install cargo-fuzz
```

### 2. Initialize fuzz directory

```bash
cd crates/solverang
cargo fuzz init
```

This creates:

```
crates/solverang/fuzz/
├── Cargo.toml
├── fuzz_targets/
│   └── (targets go here)
└── corpus/
    └── (seed inputs go here)
```

### 3. Configure fuzz/Cargo.toml

```toml
[package]
name = "solverang-fuzz"
version = "0.0.0"
publish = false
edition = "2021"

[package.metadata]
cargo-fuzz = true

[dependencies]
libfuzzer-sys = "0.4"
arbitrary = { version = "1", features = ["derive"] }
solverang = { path = "..", features = ["geometry", "jit", "sparse", "parallel", "macros"] }

[[bin]]
name = "fuzz_solver_inputs"
path = "fuzz_targets/fuzz_solver_inputs.rs"
test = false
doc = false

[[bin]]
name = "fuzz_jit_compiler"
path = "fuzz_targets/fuzz_jit_compiler.rs"
test = false
doc = false

[[bin]]
name = "fuzz_constraint_system"
path = "fuzz_targets/fuzz_constraint_system.rs"
test = false
doc = false

[[bin]]
name = "fuzz_sparse_ops"
path = "fuzz_targets/fuzz_sparse_ops.rs"
test = false
doc = false

[[bin]]
name = "fuzz_jacobian"
path = "fuzz_targets/fuzz_jacobian.rs"
test = false
doc = false
```

### 4. Update .gitignore

Add to the workspace `.gitignore`:

```
# Fuzz corpus and artifacts
**/fuzz/corpus/
**/fuzz/artifacts/
```

## Fuzz Targets

### Target 1: Solver with Arbitrary Inputs

**File:** `fuzz/fuzz_targets/fuzz_solver_inputs.rs`

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use solverang::{LMSolver, LMConfig, Problem, SolveResult};

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Target values for the linear problem (1-20 dimensions)
    targets: Vec<f64>,
    /// Initial point
    x0: Vec<f64>,
    /// Solver config overrides
    max_iterations: u16,
    patience: u8,
}

struct FuzzLinearProblem {
    target: Vec<f64>,
}

impl Problem for FuzzLinearProblem {
    fn name(&self) -> &str { "fuzz-linear" }
    fn residual_count(&self) -> usize { self.target.len() }
    fn variable_count(&self) -> usize { self.target.len() }
    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        x.iter().zip(&self.target).map(|(a, b)| a - b).collect()
    }
    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        (0..self.target.len()).map(|i| (i, i, 1.0)).collect()
    }
    fn initial_point(&self, f: f64) -> Vec<f64> { vec![f; self.target.len()] }
}

fuzz_target!(|input: FuzzInput| {
    // Constrain dimensions to avoid OOM
    if input.targets.is_empty() || input.targets.len() > 20 { return; }
    if input.x0.len() != input.targets.len() { return; }

    let problem = FuzzLinearProblem { target: input.targets };
    let config = LMConfig {
        max_iterations: input.max_iterations as usize,
        patience: input.patience as usize,
        ..LMConfig::default()
    };
    let solver = LMSolver::new(config);

    // Must not panic — any result variant is acceptable
    let _result = solver.solve(&problem, &input.x0);
});
```

### Target 2: JIT Compiler (Critical — Unsafe Code)

**File:** `fuzz/fuzz_targets/fuzz_jit_compiler.rs`

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use solverang::geometry::{ConstraintSystem, ConstraintSystemBuilder, Point2D};
use solverang::geometry::constraints::*;

#[derive(Debug, Arbitrary)]
struct FuzzConstraints {
    points: Vec<(f64, f64)>,
    constraint_ops: Vec<ConstraintOp>,
}

#[derive(Debug, Arbitrary)]
enum ConstraintOp {
    Distance { p1: u8, p2: u8, dist: f64 },
    Coincident { p1: u8, p2: u8 },
    Horizontal { p1: u8, p2: u8 },
    Vertical { p1: u8, p2: u8 },
    Fixed { p: u8 },
}

fuzz_target!(|input: FuzzConstraints| {
    // Bound dimensions
    if input.points.is_empty() || input.points.len() > 20 { return; }
    if input.constraint_ops.is_empty() || input.constraint_ops.len() > 30 { return; }

    let num_points = input.points.len();
    let mut system = ConstraintSystem::<2>::new();

    for &(x, y) in &input.points {
        // Clamp to avoid extreme values
        let x = x.clamp(-1e6, 1e6);
        let y = y.clamp(-1e6, 1e6);
        if !x.is_finite() || !y.is_finite() { return; }
        system.add_point(Point2D::new(x, y));
    }

    for op in &input.constraint_ops {
        match op {
            ConstraintOp::Distance { p1, p2, dist } => {
                let i = (*p1 as usize) % num_points;
                let j = (*p2 as usize) % num_points;
                if i != j && dist.is_finite() && *dist > 0.0 {
                    system.add_constraint(Box::new(DistanceConstraint::<2>::new(i, j, *dist)));
                }
            }
            ConstraintOp::Coincident { p1, p2 } => {
                let i = (*p1 as usize) % num_points;
                let j = (*p2 as usize) % num_points;
                if i != j {
                    system.add_constraint(Box::new(CoincidentConstraint::<2>::new(i, j)));
                }
            }
            ConstraintOp::Horizontal { p1, p2 } => {
                let i = (*p1 as usize) % num_points;
                let j = (*p2 as usize) % num_points;
                if i != j {
                    system.add_constraint(Box::new(HorizontalConstraint::new(i, j)));
                }
            }
            ConstraintOp::Vertical { p1, p2 } => {
                let i = (*p1 as usize) % num_points;
                let j = (*p2 as usize) % num_points;
                if i != j {
                    system.add_constraint(Box::new(VerticalConstraint::new(i, j)));
                }
            }
            ConstraintOp::Fixed { p } => {
                let i = (*p as usize) % num_points;
                let pt = Point2D::new(
                    input.points[i].0.clamp(-1e6, 1e6),
                    input.points[i].1.clamp(-1e6, 1e6),
                );
                system.add_constraint(Box::new(FixedConstraint::<2>::new(i, pt)));
            }
        }
    }

    // Evaluate residuals and Jacobian — these go through JIT if enabled
    let x = system.current_values();
    let _residuals = system.residuals(&x);
    let _jacobian = system.jacobian(&x);

    // Also try solving — must not segfault
    use solverang::{LMSolver, LMConfig};
    let solver = LMSolver::new(LMConfig { max_iterations: 20, ..LMConfig::default() });
    let _result = solver.solve(&system, &x);
});
```

### Target 3: Constraint System Builder

**File:** `fuzz/fuzz_targets/fuzz_constraint_system.rs`

Exercises the builder API with arbitrary combinations of points, constraints, and
fixed/free designations. Tests that the system never panics regardless of constraint
configuration (over-constrained, under-constrained, contradictory).

### Target 4: Sparse Matrix Operations

**File:** `fuzz/fuzz_targets/fuzz_sparse_ops.rs`

Feeds arbitrary sparse triplet `(row, col, value)` entries into the sparse Jacobian
path. Tests out-of-bounds indices, duplicate entries, empty matrices, and extreme values.

### Target 5: Jacobian Computation

**File:** `fuzz/fuzz_targets/fuzz_jacobian.rs`

Tests `jacobian_dense()`, `verify_jacobian()`, and finite-difference computation with
arbitrary problem dimensions and evaluation points. Verifies no panics from dimension
mismatches or non-finite values.

## Structured Fuzzing with `arbitrary`

Instead of interpreting raw bytes, derive `Arbitrary` for structured inputs:

```rust
#[derive(Debug, Arbitrary)]
struct SolverFuzzInput {
    problem_type: ProblemType,
    dimensions: u8,        // 1..20
    values: Vec<f64>,
    config: FuzzConfig,
}

#[derive(Debug, Arbitrary)]
enum ProblemType {
    Linear,
    Quadratic,
    Coupled,
    Rosenbrock,
}

#[derive(Debug, Arbitrary)]
struct FuzzConfig {
    max_iterations: u16,
    tolerance: f64,
}
```

This dramatically improves fuzzing efficiency — the fuzzer explores meaningful
input variations instead of wasting time on structurally invalid byte sequences.

## Corpus Management

### Seed from existing tests

```bash
# Create seed corpus from existing test inputs
mkdir -p fuzz/corpus/fuzz_solver_inputs
mkdir -p fuzz/corpus/fuzz_jit_compiler

# Generate seeds programmatically from test problems
cargo test --features nist -- --nocapture generate_fuzz_seeds 2>/dev/null
```

Write a helper test that serializes existing test problem inputs into the corpus
directory. This gives the fuzzer a head start with known-good inputs to mutate from.

### Corpus hygiene

```bash
# Minimize corpus after a fuzzing session
cargo fuzz cmin fuzz_solver_inputs
cargo fuzz cmin fuzz_jit_compiler
```

Commit the minimized corpus to the repository so CI runs start from a good baseline.

## CI Integration

### Time-boxed fuzz runs in nightly CI

```yaml
# In .github/workflows/nightly.yml
fuzz:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      target: [fuzz_solver_inputs, fuzz_jit_compiler, fuzz_constraint_system]
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@nightly
    - run: cargo install cargo-fuzz
    - run: |
        cd crates/solverang
        cargo fuzz run ${{ matrix.target }} -- \
          -max_total_time=900 \
          -jobs=4 \
          -workers=4
    - uses: actions/upload-artifact@v4
      if: failure()
      with:
        name: fuzz-artifacts-${{ matrix.target }}
        path: crates/solverang/fuzz/artifacts/
```

Each target gets 15 minutes of fuzzing with 4 parallel workers. Total: ~45 minutes
of CI time per nightly run.

## Triage Process

1. **Crash found**: `cargo fuzz` saves crashing input to `fuzz/artifacts/<target>/`
2. **Reproduce**: `cargo fuzz run <target> fuzz/artifacts/<target>/<crash-file>`
3. **Minimize**: `cargo fuzz tmin <target> fuzz/artifacts/<target>/<crash-file>`
4. **Diagnose**: Run under ASAN/MSAN for memory safety bugs, or debug for logic bugs
5. **Fix**: Patch the code
6. **Regression test**: Add minimized input as a regular `#[test]`:
   ```rust
   #[test]
   fn regression_crash_xyz() {
       let input = include_bytes!("../fuzz/regressions/crash_xyz");
       // Parse and run the input, assert no panic
   }
   ```
7. **Add to corpus**: Keep the minimized crash input in `fuzz/corpus/` so the fuzzer
   starts from this known-interesting input.

## Success Metrics

| Metric | Target |
|--------|--------|
| Fuzz targets implemented | 5 |
| Hours of fuzzing without new crashes | 100+ |
| JIT-specific coverage | All `unsafe` blocks reached |
| Corpus size per target | 500+ minimized inputs |
| CI fuzz time per nightly | 45 minutes |

## Estimated Effort

| Task | Time |
|------|------|
| Install and initialize cargo-fuzz | 30 min |
| Write 5 fuzz targets | 4-6 hours |
| Create seed corpus from existing tests | 2 hours |
| CI integration | 1 hour |
| Initial fuzzing campaign (local) | 8+ hours of wall time |
| Document triage process | 1 hour |
| **Total active effort** | **~10 hours** |
