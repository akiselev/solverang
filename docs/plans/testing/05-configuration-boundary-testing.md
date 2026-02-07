# Plan 05: Configuration Boundary Testing

## Goal

Solver configurations (`SolverConfig`, `LMConfig`, `SparseSolverConfig`,
`ParallelSolverConfig`) have numeric parameters with implicit valid ranges. None of
these boundaries are currently tested. Invalid configurations can cause:

- **Infinite loops** (`max_iterations = 0` with a loop that checks `iter < max`)
- **NaN propagation** (`tolerance = NaN` makes all comparisons false)
- **Division by zero** (`backtrack_factor = 0.0`)
- **Nonsensical behavior** (`armijo_c = -1.0` accepts steps that increase the residual)

## Audit of Configuration Parameters

### `SolverConfig` (Newton-Raphson)

| Parameter | Type | Default | Valid Range | What Happens at Boundary |
|-----------|------|---------|-------------|-------------------------|
| `max_iterations` | `usize` | 100 | 0..=usize::MAX | 0: never iterates (immediate NotConverged?) |
| `tolerance` | `f64` | 1e-10 | >0 | 0.0: never converges; NaN: never converges; Inf: always converges |
| `max_line_search_iterations` | `usize` | 20 | 0..=usize::MAX | 0: line search never runs → full Newton step always |
| `min_step_size` | `f64` | 1e-14 | ≥0 | 0.0: line search never stops; Inf: first backtrack stops |
| `armijo_c` | `f64` | 1e-4 | (0, 1) | 0.0: any step accepted; 1.0: requires full decrease; >1: impossible |
| `backtrack_factor` | `f64` | 0.5 | (0, 1) | 0.0: step becomes 0; 1.0: step never shrinks (infinite loop); >1: step grows |
| `svd_tolerance` | `f64` | 1e-10 | ≥0 | 0.0: no singular values rejected; NaN: all rejected |

### `LMConfig` (Levenberg-Marquardt)

| Parameter | Type | Default | Valid Range | What Happens at Boundary |
|-----------|------|---------|-------------|-------------------------|
| `max_iterations` | `usize` | 200 | >0 | 0: no iterations |
| `patience` | `usize` | 10 | >0 | 0: immediate patience exhaustion |
| `ftol` | `f64` | 1e-10 | ≥0 | 0.0: never satisfied; NaN: comparison always false |
| `xtol` | `f64` | 1e-10 | ≥0 | Same as ftol |
| `gtol` | `f64` | 1e-10 | ≥0 | Same as ftol |
| `scale_diag` | `bool` | true | N/A | N/A |

### `SparseSolverConfig`

| Parameter | Type | Default | Valid Range | What Happens at Boundary |
|-----------|------|---------|-------------|-------------------------|
| `max_iterations` | `usize` | 200 | >0 | 0: no iterations |
| `tolerance` | `f64` | 1e-10 | >0 | Same issues as above |

### `ParallelSolverConfig`

| Parameter | Type | Default | Valid Range | What Happens at Boundary |
|-----------|------|---------|-------------|-------------------------|
| `num_threads` | `Option<usize>` | None | >0 or None | Some(0): Rayon may panic or use 1 thread |

## Boundary Value Categories

### Category 1: Zero values

```rust
#[test]
fn test_solver_max_iterations_zero() {
    let config = SolverConfig { max_iterations: 0, ..Default::default() };
    let solver = Solver::new(config);
    let problem = Rosenbrock::new();
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    // Expected: NotConverged with 0 iterations, or immediate return
    assert!(!result.is_converged());
}

#[test]
fn test_solver_tolerance_zero() {
    let config = SolverConfig { tolerance: 0.0, ..Default::default() };
    let solver = Solver::new(config);
    let problem = LinearProblem { target: vec![1.0] };
    let result = solver.solve(&problem, &[0.0]);
    // Expected: runs max_iterations then returns NotConverged
    // (exact 0.0 residual is nearly impossible with floats)
}

#[test]
fn test_lm_patience_zero() {
    let config = LMConfig { patience: 0, ..Default::default() };
    let solver = LMSolver::new(config);
    let problem = Rosenbrock::new();
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    // Expected: immediate stop after first iteration without improvement
}
```

### Category 2: Extreme values

```rust
#[test]
fn test_solver_max_iterations_max() {
    let config = SolverConfig { max_iterations: usize::MAX, ..Default::default() };
    let solver = Solver::new(config);
    let problem = LinearProblem { target: vec![1.0] };
    // Should converge quickly regardless of huge max_iterations
    let result = solver.solve(&problem, &[0.0]);
    assert!(result.is_converged());
}

#[test]
fn test_solver_max_iterations_one() {
    let config = SolverConfig { max_iterations: 1, ..Default::default() };
    let solver = Solver::new(config);
    let problem = Rosenbrock::new();
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    // Expected: either converged (if 1 step suffices) or NotConverged
    // Should definitely not panic or loop
}
```

### Category 3: Special float values

```rust
#[test]
fn test_solver_tolerance_nan() {
    let config = SolverConfig { tolerance: f64::NAN, ..Default::default() };
    let solver = Solver::new(config);
    let problem = LinearProblem { target: vec![1.0] };
    let result = solver.solve(&problem, &[0.0]);
    // NaN comparisons always false → should hit max_iterations
    // Must NOT infinite loop
    assert!(!result.is_converged());
}

#[test]
fn test_solver_tolerance_infinity() {
    let config = SolverConfig { tolerance: f64::INFINITY, ..Default::default() };
    let solver = Solver::new(config);
    let problem = Rosenbrock::new();
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    // Every residual < Infinity → should converge on first iteration
    assert!(result.is_converged());
}

#[test]
fn test_solver_tolerance_negative() {
    let config = SolverConfig { tolerance: -1.0, ..Default::default() };
    let solver = Solver::new(config);
    let problem = LinearProblem { target: vec![1.0] };
    let result = solver.solve(&problem, &[0.0]);
    // Negative tolerance: no residual is ever < -1.0 → max_iterations
    assert!(!result.is_converged());
}
```

### Category 4: Domain boundary values

```rust
#[test]
fn test_armijo_c_zero() {
    // armijo_c = 0: any step is accepted (no decrease required)
    let config = SolverConfig { armijo_c: 0.0, ..Default::default() };
    let solver = Solver::new(config);
    let problem = Rosenbrock::new();
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    // Should still work, just with weaker line search
}

#[test]
fn test_armijo_c_one() {
    // armijo_c = 1.0: requires full linear decrease — rarely satisfied
    let config = SolverConfig { armijo_c: 1.0, ..Default::default() };
    let solver = Solver::new(config);
    let problem = Rosenbrock::new();
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    // Likely fails to satisfy line search → line search failure
}

#[test]
fn test_backtrack_factor_one() {
    // backtrack_factor = 1.0: step size never shrinks → infinite line search
    let config = SolverConfig {
        backtrack_factor: 1.0,
        max_line_search_iterations: 10,  // Prevent infinite loop
        ..Default::default()
    };
    let solver = Solver::new(config);
    let problem = Rosenbrock::new();
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    // Should exhaust line search iterations
}

#[test]
fn test_backtrack_factor_greater_than_one() {
    // backtrack_factor > 1: step GROWS during line search — dangerous
    let config = SolverConfig {
        backtrack_factor: 2.0,
        max_line_search_iterations: 5,
        ..Default::default()
    };
    let solver = Solver::new(config);
    let problem = Rosenbrock::new();
    let result = solver.solve(&problem, &problem.initial_point(1.0));
    // Should fail gracefully
}
```

## Proptest Configuration Strategy

```rust
use proptest::prelude::*;

fn solver_config_strategy() -> impl Strategy<Value = SolverConfig> {
    (
        0usize..1000,                          // max_iterations
        prop_oneof![               // tolerance
            Just(0.0),
            Just(f64::NAN),
            Just(f64::INFINITY),
            Just(f64::NEG_INFINITY),
            Just(f64::MIN_POSITIVE),
            Just(f64::EPSILON),
            Just(-1.0),
            1e-15f64..1e-1,
        ],
        0usize..100,                           // max_line_search_iterations
        prop_oneof![               // min_step_size
            Just(0.0),
            Just(f64::INFINITY),
            1e-20f64..1.0,
        ],
        prop_oneof![               // armijo_c
            Just(0.0),
            Just(1.0),
            Just(-1.0),
            Just(2.0),
            0.0001f64..0.5,
        ],
        prop_oneof![               // backtrack_factor
            Just(0.0),
            Just(1.0),
            Just(2.0),
            0.01f64..0.99,
        ],
    ).prop_map(|(max_iter, tol, max_ls, min_step, armijo, backtrack)| {
        SolverConfig {
            max_iterations: max_iter,
            tolerance: tol,
            max_line_search_iterations: max_ls,
            min_step_size: min_step,
            armijo_c: armijo,
            backtrack_factor: backtrack,
            ..Default::default()
        }
    })
}

fn lm_config_strategy() -> impl Strategy<Value = LMConfig> {
    (
        0usize..500,                           // max_iterations
        0usize..20,                            // patience
        prop_oneof![Just(0.0), Just(f64::NAN), 1e-15f64..1e-1], // ftol
        prop_oneof![Just(0.0), Just(f64::NAN), 1e-15f64..1e-1], // xtol
        prop_oneof![Just(0.0), Just(f64::NAN), 1e-15f64..1e-1], // gtol
    ).prop_map(|(max_iter, patience, ftol, xtol, gtol)| {
        LMConfig {
            max_iterations: max_iter,
            patience,
            ftol,
            xtol,
            gtol,
            ..Default::default()
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_nr_solver_no_panic_with_any_config(
        config in solver_config_strategy(),
    ) {
        let problem = LinearProblem { target: vec![1.0, 2.0] };
        let solver = Solver::new(config);
        // Must not panic or infinite-loop (proptest has a timeout)
        let _ = solver.solve(&problem, &[0.0, 0.0]);
    }

    #[test]
    fn prop_lm_solver_no_panic_with_any_config(
        config in lm_config_strategy(),
    ) {
        let problem = LinearProblem { target: vec![1.0, 2.0] };
        let solver = LMSolver::new(config);
        let _ = solver.solve(&problem, &[0.0, 0.0]);
    }
}
```

## Suggested Validation: `Config::validate()`

Consider adding validation methods:

```rust
impl SolverConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.tolerance.is_nan() {
            return Err(ConfigError::InvalidTolerance("NaN".into()));
        }
        if self.tolerance < 0.0 {
            return Err(ConfigError::InvalidTolerance("negative".into()));
        }
        if self.backtrack_factor <= 0.0 || self.backtrack_factor >= 1.0 {
            return Err(ConfigError::InvalidBacktrackFactor(self.backtrack_factor));
        }
        if self.armijo_c <= 0.0 || self.armijo_c >= 1.0 {
            return Err(ConfigError::InvalidArmijoC(self.armijo_c));
        }
        Ok(())
    }
}
```

Call `validate()` at solver construction time (or at `solve()` entry) and return
an error instead of exhibiting undefined behavior.

## File Organization

```
crates/solverang/tests/
└── config_boundary_tests.rs    # All configuration boundary tests
```

## Estimated Effort

| Task | Time |
|------|------|
| Audit all config parameters | 1-2 hours |
| Write explicit boundary tests | 3-4 hours |
| Write proptest config strategies | 2 hours |
| Implement Config::validate() (if desired) | 2-3 hours |
| **Total** | **~8-11 hours** |
