# Plan 04: Snapshot Testing

## Goal

Snapshot testing captures a computation's output and saves it to a file. Future test
runs compare against the saved snapshot. Changes require explicit review and approval.

This catches **behavioral regressions** that tolerance-based assertions miss:
- A refactor that changes convergence from 5 iterations to 50 (but still meets tolerance)
- A dependency update (`nalgebra`, `levenberg-marquardt`) that silently changes numerical behavior
- JIT codegen changes that produce subtly different floating-point results
- Decomposition algorithm changes that assign constraints to different sub-problems

## Tool Selection

**`insta` v1.39** — the standard Rust snapshot testing crate.

Features to enable:
- `yaml` — human-readable YAML snapshots for structured data
- `redactions` — mask non-deterministic or platform-dependent fields

## Setup Steps

### 1. Add to Cargo.toml

```toml
[dev-dependencies]
insta = { version = "1.39", features = ["yaml", "redactions"] }
serde = { version = "1", features = ["derive"] }  # If not already present
```

### 2. Install CLI tool

```bash
cargo install cargo-insta
```

### 3. Snapshot directory structure

`insta` automatically creates snapshot files next to the test file:

```
crates/solverang/tests/
├── snapshot_tests.rs
└── snapshots/
    ├── snapshot_tests__nist_rosenbrock.snap
    ├── snapshot_tests__nist_powell.snap
    ├── snapshot_tests__solver_comparison_rosenbrock.snap
    └── ...
```

### 4. .gitignore update

```
# Pending snapshots (not yet reviewed)
*.snap.new
```

Committed `.snap` files are version-controlled — they ARE the expected output.

## Snapshot Targets

### 1. NIST StRD Problems (30+ problems)

```rust
use insta::assert_yaml_snapshot;
use serde::Serialize;
use solverang::test_problems::nist::*;
use solverang::{LMSolver, LMConfig, SolveResult};

#[derive(Serialize)]
struct SolverSnapshot {
    problem: String,
    converged: bool,
    residual_norm: f64,
    iterations: usize,
    solution: Vec<f64>,
}

impl SolverSnapshot {
    fn from_result(name: &str, result: &SolveResult) -> Self {
        match result {
            SolveResult::Converged { solution, residual_norm, iterations } => Self {
                problem: name.to_string(),
                converged: true,
                residual_norm: round(*residual_norm, 10),
                iterations: *iterations,
                solution: solution.iter().map(|v| round(*v, 10)).collect(),
            },
            SolveResult::NotConverged { solution, residual_norm, iterations } => Self {
                problem: name.to_string(),
                converged: false,
                residual_norm: round(*residual_norm, 10),
                iterations: *iterations,
                solution: solution.iter().map(|v| round(*v, 10)).collect(),
            },
            SolveResult::Failed { .. } => Self {
                problem: name.to_string(),
                converged: false,
                residual_norm: f64::NAN,
                iterations: 0,
                solution: vec![],
            },
        }
    }
}

/// Round to N significant digits for cross-platform reproducibility.
fn round(v: f64, digits: u32) -> f64 {
    if !v.is_finite() { return v; }
    let factor = 10f64.powi(digits as i32);
    (v * factor).round() / factor
}

macro_rules! nist_snapshot_test {
    ($name:ident, $problem:expr) => {
        #[test]
        fn $name() {
            let problem = $problem;
            let solver = LMSolver::new(LMConfig::default());
            let result = solver.solve(&problem, &problem.initial_point(1.0));
            let snapshot = SolverSnapshot::from_result(stringify!($name), &result);
            assert_yaml_snapshot!(snapshot);
        }
    };
}

nist_snapshot_test!(nist_misra1a, Misra1a::new());
nist_snapshot_test!(nist_chwirut2, Chwirut2::new());
nist_snapshot_test!(nist_eckerle4, Eckerle4::new());
// ... one for each of the 30+ NIST problems
```

### 2. MINPACK Problems

```rust
#[test]
fn snapshot_minpack_rosenbrock_start1() {
    let problem = Rosenbrock::new();
    let result = solver.solve(&problem, &[-1.2, 1.0]);
    let snapshot = SolverSnapshot::from_result("rosenbrock_start1", &result);
    assert_yaml_snapshot!(snapshot);
}

#[test]
fn snapshot_minpack_rosenbrock_start2() {
    let problem = Rosenbrock::new();
    let result = solver.solve(&problem, &[10.0, 10.0]);
    let snapshot = SolverSnapshot::from_result("rosenbrock_start2", &result);
    assert_yaml_snapshot!(snapshot);
}
```

### 3. Solver Comparison

```rust
#[derive(Serialize)]
struct SolverComparisonSnapshot {
    problem: String,
    results: Vec<(String, SolverSnapshot)>,
}

#[test]
fn snapshot_solver_comparison_rosenbrock() {
    let problem = Rosenbrock::new();
    let x0 = problem.initial_point(1.0);

    let results = vec![
        ("newton_raphson", Solver::new(Default::default()).solve(&problem, &x0)),
        ("levenberg_marquardt", LMSolver::new(Default::default()).solve(&problem, &x0)),
        ("auto", AutoSolver::new().solve(&problem, &x0)),
        ("robust", RobustSolver::new().solve(&problem, &x0)),
    ];

    let snapshots: Vec<_> = results.iter()
        .map(|(name, r)| (name.to_string(), SolverSnapshot::from_result(name, r)))
        .collect();

    assert_yaml_snapshot!(SolverComparisonSnapshot {
        problem: "rosenbrock".into(),
        results: snapshots,
    });
}
```

### 4. JIT vs Interpreted

```rust
#[derive(Serialize)]
struct EvalSnapshot {
    residuals: Vec<f64>,
    jacobian_entries: Vec<(usize, usize, f64)>,
}

#[test]
fn snapshot_jit_vs_interpreted_triangle() {
    let system = build_triangle_system();
    let x = system.current_values();

    let interpreted = EvalSnapshot {
        residuals: system.residuals(&x),
        jacobian_entries: system.jacobian(&x),
    };
    assert_yaml_snapshot!("triangle_interpreted", interpreted);

    // JIT version should produce identical snapshot
    let jit_system = JITCompiler::compile(&system).unwrap();
    let jit_eval = EvalSnapshot {
        residuals: jit_system.residuals(&x),
        jacobian_entries: jit_system.jacobian(&x),
    };
    assert_yaml_snapshot!("triangle_jit", jit_eval);
}
```

### 5. Problem Decomposition

```rust
#[derive(Serialize)]
struct DecompositionSnapshot {
    num_components: usize,
    component_sizes: Vec<usize>,
    variable_assignments: Vec<usize>,  // Which component each variable belongs to
}

#[test]
fn snapshot_decomposition_independent_triangles() {
    let system = build_two_independent_triangles();
    let decomp = decompose(&system);
    let snapshot = DecompositionSnapshot {
        num_components: decomp.components().len(),
        component_sizes: decomp.components().iter().map(|c| c.variable_count()).collect(),
        variable_assignments: decomp.variable_assignments().to_vec(),
    };
    assert_yaml_snapshot!(snapshot);
}
```

### 6. Macro Expansion

```rust
// In the macros crate tests
#[test]
fn snapshot_auto_jacobian_expansion() {
    // Capture the generated code from #[auto_jacobian]
    // This requires a compile-time test or using trybuild
    // Alternative: snapshot the Jacobian VALUES produced by macro-generated code
    let problem = MacroGeneratedProblem::new();
    let jac = problem.jacobian(&[1.0, 2.0, 3.0]);
    assert_yaml_snapshot!(jac);
}
```

## Snapshot Serialization: Handling Floats

### Problem: Cross-platform float reproducibility

IEEE 754 operations can produce slightly different results across platforms due to
different FMA availability, compiler optimizations, and math library implementations.

### Solution: Round to significant digits

```rust
fn round_for_snapshot(v: f64, significant_digits: u32) -> f64 {
    if !v.is_finite() || v == 0.0 { return v; }
    let magnitude = v.abs().log10().floor() as i32;
    let factor = 10f64.powi(significant_digits as i32 - 1 - magnitude);
    (v * factor).round() / factor
}
```

Use 10 significant digits — enough to detect real changes but tolerant of platform
differences in the least significant bits.

### Alternative: Use `insta` redactions

```rust
assert_yaml_snapshot!(result, {
    ".residual_norm" => insta::rounded_redaction(6),
    ".solution[]" => insta::rounded_redaction(8),
});
```

## Workflow

### Development cycle

```bash
# Run tests — new snapshots create .snap.new files
cargo test

# Review pending snapshots interactively
cargo insta review

# Accept all pending snapshots
cargo insta accept

# Reject all pending snapshots
cargo insta reject
```

### When a snapshot changes

1. **Intentional change** (algorithm improvement, dependency update):
   - Review the diff in `cargo insta review`
   - Verify the new behavior is correct
   - Accept: `cargo insta accept`
   - Commit the updated `.snap` files

2. **Unintentional change** (regression):
   - Investigate why the output changed
   - Fix the bug
   - Reject: `cargo insta reject`

## CI Integration

```yaml
# In ci.yml
- name: Run tests (snapshot mode)
  run: cargo test
  env:
    INSTA_UPDATE: no  # Fail on snapshot mismatches instead of creating .snap.new
```

In CI, `INSTA_UPDATE=no` makes snapshot mismatches a hard failure. Developers must
run `cargo insta review` locally and commit updated snapshots.

## Snapshot Review Process

### When to accept changes

- Algorithm intentionally changed (documented in PR description)
- Dependency updated (check changelogs for relevant changes)
- New feature added (new snapshots, no existing ones changed)
- Tolerance improvement (solution closer to known answer)

### When to reject changes

- Iteration count increased significantly (performance regression)
- Solution accuracy decreased
- Previously converging problem now fails
- No corresponding code change explains the snapshot change

## Estimated Effort

| Task | Time |
|------|------|
| Add `insta` + `serde` to dev-dependencies | 15 min |
| Write snapshot infrastructure (helpers, macros) | 2 hours |
| NIST problem snapshots (30+ tests) | 2-3 hours |
| MINPACK problem snapshots | 1 hour |
| Solver comparison snapshots | 1-2 hours |
| JIT vs interpreted snapshots | 1 hour |
| Decomposition snapshots | 1 hour |
| Review and accept initial snapshots | 1 hour |
| CI integration | 30 min |
| **Total** | **~10-12 hours** |
