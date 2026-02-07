# Plan 02: Mutation Testing

## Goal

Mutation testing verifies that the test suite actually detects real bugs — not just
that it runs without failing. `cargo-mutants` systematically modifies source code
(flipping `+` to `-`, removing conditionals, replacing constants with defaults) and
checks that at least one test fails for each mutation. **Surviving mutants** reveal
tests that are too weak or missing entirely.

This is especially important for numerical code where:
- A sign flip in a Jacobian entry might still produce "close enough" results
- An off-by-one in a loop bound might only affect rare edge cases
- A tolerance comparison (`<` vs `<=`) might never matter in happy-path tests

The `.gitignore` already lists `mutants.out*/`, indicating `cargo-mutants` has been
used before but isn't integrated systematically.

## Tool

**`cargo-mutants`** — the standard Rust mutation testing tool.

```bash
cargo install cargo-mutants
```

Version: 24.x+ (supports workspace filtering, timeouts, and `mutants.toml` config)

## Baseline Run

### Full crate scan

```bash
cd /home/user/solverang
cargo mutants -p solverang --features geometry,parallel,sparse,jit,macros,nist
```

This will take a long time on the full crate (potentially hours). For a quick baseline:

```bash
# Test only a specific module
cargo mutants -p solverang -f src/solver/newton_raphson.rs --timeout 120
```

### Interpreting results

```
$ cargo mutants -p solverang -f src/solver/newton_raphson.rs
...
412 mutants tested: 380 caught, 20 survived, 12 timed out
Mutation score: 92.2% (380/412)
```

- **Caught**: A test failed — good, the test suite detects this bug class
- **Survived**: No test failed — **gap in test coverage**
- **Timed out**: Mutant caused infinite loop — usually counts as "caught"
- **Unviable**: Mutant doesn't compile — not counted

## Priority Modules to Target

### Tier 1: Core solver logic (highest impact)

| File | Why | Likely Surviving Mutants |
|------|-----|------------------------|
| `src/solver/newton_raphson.rs` | Line search, Armijo condition, convergence | `armijo_c` sign, `backtrack_factor` direction, convergence threshold |
| `src/solver/lm_adapter.rs` | Parameter mapping to `levenberg-marquardt` crate | Scale factors, parameter ordering |
| `src/jacobian/numeric.rs` | Finite-difference step size, central vs forward | Step size epsilon, division direction |

### Tier 2: Algorithm correctness

| File | Why | Likely Surviving Mutants |
|------|-----|------------------------|
| `src/decomposition.rs` | Union-find, dependency graph | Union vs find, rank comparison |
| `src/solver/auto_solver.rs` | Algorithm selection heuristics | Threshold comparisons |
| `src/solver/robust_solver.rs` | Fallback logic | Fallback condition |

### Tier 3: Constraint formulas

| File | Why | Likely Surviving Mutants |
|------|-----|------------------------|
| `src/geometry/constraints/distance.rs` | Distance formula, Jacobian | Squared vs unsquared, sign of partials |
| `src/geometry/constraints/angle.rs` | Angle computation | atan2 argument order, normalization |
| `src/geometry/constraints/parallel.rs` | Cross product formula | Component ordering |
| `src/geometry/constraints/perpendicular.rs` | Dot product formula | Sign |

## Configuration: `mutants.toml`

Create `/home/user/solverang/crates/solverang/mutants.toml`:

```toml
# Timeout per mutant (seconds) — numerical code can be slow
timeout_multiplier = 3.0
minimum_test_timeout = 60

# Skip modules that are too slow or not meaningful to mutate
exclude_globs = [
    "src/test_problems/**",    # Test infrastructure, not production code
    "benches/**",              # Benchmarks
    "tests/**",                # Test code itself
]

# Only run relevant tests for each mutant (faster)
# cargo-mutants does this automatically via `--cargo-arg`

# Cap concurrent jobs to avoid OOM with large test suite
jobs = 4
```

## Analyzing Results

### Surviving mutant categories

**1. True gaps — need new tests:**
```
SURVIVED: src/solver/newton_raphson.rs:142: replace `<` with `<=` in `residual < tolerance`
```
This means no test distinguishes between `<` and `<=` for convergence. Write a test
where the residual is exactly equal to the tolerance.

**2. Equivalent mutants — false positives:**
```
SURVIVED: src/solver/newton_raphson.rs:55: replace `x + 0.0` with `x + 1.0`
```
Sometimes a mutation is functionally equivalent (dead code, unreachable branch).
Mark these with `#[mutants::skip]` after confirming.

**3. Performance-only mutants:**
```
SURVIVED: src/solver/newton_raphson.rs:87: replace `/ 2.0` with `/ 1.0`
```
If this only affects convergence speed (not correctness), it may survive because
tests only check final results. Consider adding iteration-count assertions.

### Triage workflow

1. Run `cargo mutants` on a module
2. Review surviving mutants in `mutants.out/caught.txt` and `mutants.out/survived.txt`
3. For each survivor:
   - Is it a true gap? → Write a targeted test
   - Is it equivalent? → Add `#[mutants::skip]` with comment
   - Is it performance-only? → Add an iteration-count or convergence-rate assertion
4. Re-run to confirm the survivor is now caught

## Fixing Gaps: Writing Targeted Tests

### Example: Killing a tolerance boundary mutant

```rust
// Mutant: replace `residual_norm < tolerance` with `residual_norm <= tolerance`
#[test]
fn test_convergence_at_exact_tolerance() {
    // Create a problem that converges to exactly the tolerance
    let config = SolverConfig { tolerance: 1e-6, ..Default::default() };
    let problem = ProblemWithKnownResidual { final_residual: 1e-6 };
    let result = Solver::new(config).solve(&problem, &x0);
    // With `<`: NotConverged (residual == tolerance, not less than)
    // With `<=`: Converged
    // This test pins the behavior to one or the other
    assert!(!result.is_converged()); // or assert!(result.is_converged())
}
```

### Example: Killing a Jacobian sign mutant

```rust
// Mutant: replace `2.0 * x[i]` with `-2.0 * x[i]` in Jacobian
#[test]
fn test_quadratic_jacobian_sign() {
    let problem = QuadraticProblem { target: vec![4.0] };
    let jac = problem.jacobian(&[2.0]);
    assert_eq!(jac, vec![(0, 0, 4.0)]); // 2 * 2.0 = 4.0, not -4.0
}
```

## CI Integration

### Nightly mutation testing (not on every PR — too slow)

```yaml
# .github/workflows/nightly.yml
mutation-testing:
  runs-on: ubuntu-latest
  timeout-minutes: 60
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo install cargo-mutants
    - name: Run mutation testing on core modules
      run: |
        cargo mutants -p solverang \
          --features geometry,parallel,sparse \
          -f src/solver/newton_raphson.rs \
          -f src/solver/lm_adapter.rs \
          -f src/jacobian/numeric.rs \
          -f src/decomposition.rs \
          --timeout 120 \
          --output mutants-report
    - uses: actions/upload-artifact@v4
      with:
        name: mutation-report
        path: mutants-report/
    - name: Check mutation score
      run: |
        # Parse mutation score and fail if below threshold
        SCORE=$(grep -oP 'mutation score: \K[\d.]+' mutants-report/outcomes.json || echo "0")
        if (( $(echo "$SCORE < 85.0" | bc -l) )); then
          echo "Mutation score $SCORE% is below 85% threshold"
          exit 1
        fi
```

### PR-triggered incremental mutation testing

```yaml
# Only mutate files changed in the PR
mutation-incremental:
  runs-on: ubuntu-latest
  if: github.event_name == 'pull_request'
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    - run: |
        CHANGED=$(git diff --name-only origin/main...HEAD -- 'crates/solverang/src/**/*.rs')
        if [ -n "$CHANGED" ]; then
          cargo mutants -p solverang $(echo "$CHANGED" | sed 's/^/-f /')
        fi
```

## Metrics

| Metric | Current (est.) | Target |
|--------|---------------|--------|
| Overall mutation score | Unknown | ≥85% |
| Core solver modules | Unknown | ≥90% |
| Constraint formulas | Unknown | ≥95% |
| Equivalent mutants documented | 0 | All identified |

Track mutation score over time by storing `mutants.out/outcomes.json` as a CI artifact
and graphing the trend.

## Estimated Effort

| Task | Time |
|------|------|
| Install and run baseline | 1 hour |
| Analyze Tier 1 surviving mutants | 3-4 hours |
| Write targeted tests to kill survivors | 4-6 hours |
| Analyze Tier 2-3 modules | 4-6 hours |
| Write additional targeted tests | 4-6 hours |
| CI integration | 2 hours |
| Document `#[mutants::skip]` decisions | 1 hour |
| **Total** | **~20 hours** (spread over multiple sessions) |
