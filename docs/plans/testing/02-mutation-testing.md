# Plan 02: Mutation Testing

## Goal

Mutation testing verifies that the test suite actually detects real bugs -- not just
that it runs without failing. `cargo-mutants` systematically modifies source code
(flipping `+` to `-`, removing conditionals, replacing constants with defaults) and
checks that at least one test fails for each mutation. **Surviving mutants** reveal
tests that are too weak or missing entirely.

This is especially important for solverang because:

- **Numerical code is sign-sensitive.** A sign flip in a Jacobian partial derivative
  might still produce "close enough" results in tolerant tests. Squared constraint
  formulations (`dx^2+dy^2 - d^2`) in the V3 `sketch2d` module are particularly
  vulnerable -- a sign error in the derivative may only matter for certain geometric
  configurations.

- **Closed-form analytical solvers** (`solve/closed_form.rs`) contain math expressions
  where a single sign or operator mutation changes the mathematical meaning (e.g.,
  `+` to `-` in a circle-circle discriminant changes which intersection branch is
  returned).

- **Graph algorithms** contain rank comparisons, index arithmetic, and threshold
  checks where off-by-one mutations may only affect rare topologies.

- **Reduce passes** perform symbolic transformations where an incorrect merge or
  elimination may produce a system that still converges to an answer -- just the
  wrong answer.

The `.gitignore` already lists `mutants.out*/`, indicating `cargo-mutants` has been
used before but is not integrated systematically.

## Tool

**`cargo-mutants`** -- the standard Rust mutation testing tool.

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
cargo mutants -p solverang -f src/solve/closed_form.rs --timeout 120
```

### Interpreting results

```
$ cargo mutants -p solverang -f src/solve/closed_form.rs
...
412 mutants tested: 380 caught, 20 survived, 12 timed out
Mutation score: 92.2% (380/412)
```

- **Caught**: A test failed -- good, the test suite detects this bug class
- **Survived**: No test failed -- **gap in test coverage**
- **Timed out**: Mutant caused infinite loop -- usually counts as "caught"
- **Unviable**: Mutant doesn't compile -- not counted

## Priority Modules to Target

### Tier 1: Highest impact -- math-intensive code with sign-sensitive formulas

These modules contain the most critical numerical logic where a single operator
mutation represents a real bug class. Tier 1 modules should reach 95%+ mutation score.

| File | Lines | Why | Likely Surviving Mutants |
|------|-------|-----|------------------------|
| `src/solve/closed_form.rs` | ~750 | Analytical solvers with sign-sensitive math: circle-circle discriminant, Newton step direction, polar-to-cartesian signs | `+` vs `-` in discriminant, `>` vs `>=` in zero-Jacobian guard, sign of Newton update |
| `src/sketch2d/constraints.rs` | ~2103 | 16 constraint types with squared formulations and hand-coded Jacobians; sign errors in partials may not be caught by tolerant integration tests | Jacobian sign: `2*dx` vs `-2*dx`, squared terms: `dx*dx + dy*dy - d*d` operator mutations, `atan2(dy, dx)` argument order |
| `src/solver/newton_raphson.rs` | ~300 | Line search, Armijo condition, convergence check | `armijo_c` sign, `backtrack_factor` direction, convergence threshold `<` vs `<=` |
| `src/solver/lm_adapter.rs` | ~200 | Parameter mapping to `levenberg-marquardt` crate | Scale factors, parameter ordering |
| `src/jacobian/numeric.rs` | ~200 | Finite-difference step size, central vs forward | Step size epsilon, division direction |

**Specific likely surviving mutants in Tier 1:**

For `solve/closed_form.rs`:
- `d*d - r1*r1 - r2*r2` -- mutating any operator changes the circle-circle intersection
  geometry. Tests must verify specific intersection coordinates, not just "solved vs not".
- `x - f/df` (Newton step) -- mutating `x - f/df` to `x + f/df` reverses the step
  direction. Tests must check the output value, not just convergence status.
- Zero-Jacobian guard: `if df.abs() < eps` -- mutating `<` to `>` skips the guard
  entirely. Need a test with exactly-zero Jacobian.

For `sketch2d/constraints.rs`:
- `DistancePtPt`: `dx*dx + dy*dy - d*d` -- mutating `+` to `-` between `dx*dx` and
  `dy*dy` makes the constraint compute `|dx|^2 - |dy|^2 - d^2` instead of distance.
  Only caught if a test has `dx != 0` AND `dy != 0` AND checks the final position.
- `Angle`: `atan2(dy, dx)` -- swapping arguments computes the complementary angle.
  Need a test with a non-axis-aligned angle.
- Each of the 16 Jacobian implementations: sign flip in any partial derivative.
  Squared formulations double the number of terms (chain rule), increasing mutation
  surface.

### Tier 2: Algorithm correctness -- logic-heavy code

These modules contain complex algorithmic logic where mutations affect correctness
in subtle ways. Target: 90%+ mutation score.

| File | Lines | Why | Likely Surviving Mutants |
|------|-------|-----|------------------------|
| `src/reduce/substitute.rs` | ~350 | Fixed-param elimination: wrong fixed set = wrong elimination | `is_fixed()` check negation, parameter substitution direction |
| `src/reduce/merge.rs` | ~350 | Coincident merging: union-find logic, representative selection | Union direction (a->b vs b->a), path compression |
| `src/reduce/eliminate.rs` | ~350 | Trivial elimination: single-equation detection threshold | `== 1` vs `<= 1` for free param count, elimination value sign |
| `src/graph/dof.rs` | ~400 | SVD-based rank computation, per-entity DOF | Rank threshold comparison, column selection |
| `src/graph/redundancy.rs` | ~400 | Incremental rank test, conflict detection via null-space projection | Tolerance comparison, residual projection sign |
| `src/graph/pattern.rs` | ~400 | Pattern matching catalogue: classification of constraint neighborhoods | Pattern category check, constraint name matching |
| `src/pipeline/solve_phase.rs` | ~500 | Solver dispatch, warm start, convergence check | Warm start usage, convergence threshold |
| `src/dataflow/tracker.rs` | ~350 | Structural vs value change classification | `structural_change` flag logic, dirty set computation |

**Specific likely surviving mutants in Tier 2:**

For `reduce/merge.rs`:
- Union direction: whether `a` maps to `b` or `b` maps to `a` in the substitution
  map. If tests only check that the two params end up equal, either direction passes.
  Need a test that checks which specific `ParamId` survives.

For `graph/dof.rs`:
- SVD rank tolerance: `singular_value > tolerance` vs `singular_value >= tolerance`.
  Need a test with a singular value exactly at the tolerance boundary.
- Per-entity DOF: `free_params - local_rank` -- mutating `-` to `+` produces absurd
  DOF values. Only caught if tests assert specific DOF numbers.

For `dataflow/tracker.rs`:
- `structural_change` flag: `true` vs `false` when adding/removing entities. If tests
  always call full `solve()` (which handles both structural and value changes), the
  distinction is never tested.

### Tier 3: Extended coverage -- domain-specific constraints and algorithms

These modules are important but lower priority because they have more straightforward
test patterns. Target: 85%+ mutation score.

| File | Lines | Why | Likely Surviving Mutants |
|------|-------|-----|------------------------|
| `src/sketch3d/constraints.rs` | ~1200 | 8 constraint types: 3D geometry formulas | Cross product component ordering, sign of normals |
| `src/assembly/constraints.rs` | ~1000 | Quaternion math in Mate, CoaxialAssembly, Insert, Gear | Quaternion component signs, rotation direction |
| `src/solve/branch.rs` | ~200 | Branch selection: L2 distance, residual comparison | Distance metric, comparison direction |
| `src/solve/drag.rs` | ~300 | Null-space projection via SVD | Projection matrix construction, singular value threshold |
| `src/solve/sub_problem.rs` | ~400 | `ReducedSubProblem` adapter: mapping between V3 and legacy solver | Column/row mapping direction, value write-back |
| `src/param/store.rs` | ~380 | Free-list, generation checks, solver mapping | Generation increment, alive flag |
| `src/decomposition.rs` | ~300 | Legacy union-find decomposition | Union vs find, rank comparison |
| `src/solver/auto.rs` | ~200 | Solver selection heuristics | Threshold comparisons |

**Specific likely surviving mutants in Tier 3:**

For `assembly/constraints.rs`:
- Quaternion multiplication order: `q1 * q2` vs `q2 * q1` (quaternion multiplication
  is non-commutative). Tests must use non-trivial rotations to catch this.
- `Gear` constraint ratio: mutating the gear ratio application direction. Need a test
  with a non-unity gear ratio.

For `solve/branch.rs`:
- `ClosestToPrevious` vs `SmallestResidual`: if tests only have single-solution
  problems, branch selection is never exercised. Need multi-solution test cases
  (e.g., circle-circle intersection with two valid branches).

For `param/store.rs`:
- Generation increment: `generation + 1` to `generation + 0` or `generation - 1`.
  This would break stale-ID detection. Need a test that specifically checks generation
  mismatch behavior.

## Configuration: `mutants.toml`

Create `/home/user/solverang/crates/solverang/mutants.toml`:

```toml
# Timeout per mutant (seconds) -- numerical code can be slow
timeout_multiplier = 3.0
minimum_test_timeout = 60

# Skip modules that are too slow or not meaningful to mutate
exclude_globs = [
    "src/test_problems/**",    # Test infrastructure, not production code
    "benches/**",              # Benchmarks
    "tests/**",                # Test code itself
    "src/pipeline/integration_tests.rs",
    "src/pipeline/incremental_tests.rs",
    "src/pipeline/minpack_bridge_tests.rs",
]

# Cap concurrent jobs to avoid OOM with large test suite
jobs = 4
```

## Analyzing Results

### Surviving mutant categories

**1. True gaps -- need new tests:**
```
SURVIVED: src/solve/closed_form.rs:87: replace `x - f_val / df` with `x + f_val / df`
```
This means no test checks whether the Newton step goes in the correct direction.
Write a test with a known scalar equation (e.g., `x^2 - 4 = 0` starting at `x=3`)
and assert the result is `x=2.0`, not `x=4.0`.

**2. Equivalent mutants -- false positives:**
```
SURVIVED: src/solve/closed_form.rs:55: replace `x + 0.0` with `x + 1.0`
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
   - Is it a true gap? Write a targeted test
   - Is it equivalent? Add `#[mutants::skip]` with comment
   - Is it performance-only? Add an iteration-count or convergence-rate assertion
4. Re-run to confirm the survivor is now caught

## Fixing Gaps: Writing Targeted Tests

### Example: Killing a closed-form solver sign mutant

```rust
// Mutant: replace `x - f_val / df` with `x + f_val / df` in solve_scalar()
#[test]
fn test_scalar_newton_step_direction() {
    // x^2 - 4 = 0, starting at x = 3.0
    // f(3) = 5, f'(3) = 6
    // Correct step: x_new = 3 - 5/6 = 2.1667
    // Wrong step:   x_new = 3 + 5/6 = 3.8333
    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let pid = store.alloc(3.0, owner);

    // Build a "x^2 - 4" constraint
    // ... (construct test constraint)

    let pattern = MatchedPattern {
        kind: PatternKind::ScalarSolve,
        entity_ids: vec![owner],
        constraint_indices: vec![0],
        param_ids: vec![pid],
    };

    let result = solve_pattern(&pattern, &[&constraint], &store).unwrap();
    // The new value should be closer to 2.0, not farther from it
    assert!(result.solved);
    let new_val = result.values[0].1;
    assert!((new_val - 2.0).abs() < (3.0 - 2.0).abs(),
        "Newton step should move toward root, got {}", new_val);
}
```

### Example: Killing a squared-formulation Jacobian sign mutant

```rust
// Mutant: replace `2.0 * dx` with `-2.0 * dx` in DistancePtPt Jacobian
#[test]
fn test_distance_pt_pt_jacobian_sign() {
    let mut builder = Sketch2DBuilder::new();
    let p0 = builder.add_point(3.0, 4.0);
    let p1 = builder.add_point(6.0, 8.0);
    // DistancePtPt uses squared formulation: dx^2 + dy^2 - d^2
    // Jacobian w.r.t. p0.x should be -2*dx = -2*(6-3) = -6
    // Jacobian w.r.t. p1.x should be  2*dx =  2*(6-3) =  6
    builder.constrain_distance(p0, p1, 5.0);
    let system = builder.build();

    // Evaluate Jacobian and verify specific entries
    let residuals = system.compute_residuals();
    // With dx=3, dy=4: residual = 9+16-25 = 0 (already satisfied)
    assert!((residuals[0]).abs() < 1e-10);

    // The Jacobian sign matters for convergence direction.
    // A solver starting from a perturbed state should converge correctly.
    // Perturb p0.x and verify solver converges.
}
```

### Example: Killing a tolerance boundary mutant

```rust
// Mutant: replace `residual_norm < tolerance` with `residual_norm <= tolerance`
#[test]
fn test_convergence_at_exact_tolerance() {
    let config = SolverConfig { tolerance: 1e-6, ..Default::default() };
    let problem = ProblemWithKnownResidual { final_residual: 1e-6 };
    let result = Solver::new(config).solve(&problem, &x0);
    // Pins the behavior to one specific interpretation
    assert!(!result.is_converged()); // or assert!(result.is_converged())
}
```

### Example: Killing a DOF analysis rank mutant

```rust
// Mutant: replace `free_params - local_rank` with `free_params + local_rank` in dof.rs
#[test]
fn test_dof_well_constrained_point() {
    let mut system = ConstraintSystem::new();
    // Create a point (2 DOF) with 2 constraints (fixes both DOFs)
    let (eid, px, py) = add_test_point(&mut system, 1.0, 2.0);
    // Fix both: DOF should be 0
    let cid1 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid1, entity_ids: vec![eid], param: px, target: 5.0,
    }));
    let cid2 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid2, entity_ids: vec![eid], param: py, target: 3.0,
    }));
    let _ = system.solve();
    let dof = system.analyze_dof();
    assert_eq!(dof.total_dof, 0, "Well-constrained point should have 0 DOF");
    // With the mutant (free_params + local_rank), DOF would be 4 instead of 0
}
```

### Example: Killing a change tracker mutant

```rust
// Mutant: negate `structural_change` flag in ChangeTracker
#[test]
fn test_add_entity_is_structural_change() {
    let mut system = ConstraintSystem::new();
    let _ = system.solve(); // Initial solve, clears tracker

    let (eid, _px, _py) = add_test_point(&mut system, 1.0, 2.0);
    assert!(system.change_tracker().has_structural_changes(),
        "Adding an entity must be a structural change");
}

#[test]
fn test_set_param_is_not_structural_change() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);
    let _ = system.solve(); // Clears tracker

    system.set_param(px, 5.0);
    assert!(!system.change_tracker().has_structural_changes(),
        "Setting a param value must NOT be a structural change");
    assert!(system.change_tracker().has_any_changes(),
        "Setting a param value must be tracked as a value change");
}
```

## CI Integration

### Nightly mutation testing (not on every PR -- too slow)

```yaml
# .github/workflows/nightly.yml
mutation-testing:
  runs-on: ubuntu-latest
  timeout-minutes: 120
  strategy:
    matrix:
      tier:
        - name: tier1-closed-form
          files: src/solve/closed_form.rs
          threshold: 95
        - name: tier1-sketch2d-constraints
          files: src/sketch2d/constraints.rs
          threshold: 95
        - name: tier1-newton-raphson
          files: src/solver/newton_raphson.rs
          threshold: 95
        - name: tier2-reduce
          files: src/reduce/substitute.rs src/reduce/merge.rs src/reduce/eliminate.rs
          threshold: 90
        - name: tier2-graph
          files: src/graph/dof.rs src/graph/redundancy.rs src/graph/pattern.rs
          threshold: 90
        - name: tier2-pipeline
          files: src/pipeline/solve_phase.rs src/dataflow/tracker.rs
          threshold: 90
        - name: tier3-sketch3d-assembly
          files: src/sketch3d/constraints.rs src/assembly/constraints.rs
          threshold: 85
        - name: tier3-solve
          files: src/solve/branch.rs src/solve/drag.rs src/solve/sub_problem.rs
          threshold: 85
        - name: tier3-legacy
          files: src/solver/lm_adapter.rs src/jacobian/numeric.rs src/decomposition.rs
          threshold: 85
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo install cargo-mutants
    - name: Run mutation testing
      run: |
        FILES=$(echo "${{ matrix.tier.files }}" | sed 's/ / -f /g')
        cargo mutants -p solverang \
          --features geometry,parallel,sparse \
          -f $FILES \
          --timeout 120 \
          --output mutants-report-${{ matrix.tier.name }}
    - uses: actions/upload-artifact@v4
      with:
        name: mutation-report-${{ matrix.tier.name }}
        path: mutants-report-${{ matrix.tier.name }}/
    - name: Check mutation score
      run: |
        SCORE=$(grep -oP 'mutation score: \K[\d.]+' \
          mutants-report-${{ matrix.tier.name }}/outcomes.json || echo "0")
        THRESHOLD=${{ matrix.tier.threshold }}
        if (( $(echo "$SCORE < $THRESHOLD" | bc -l) )); then
          echo "Mutation score $SCORE% is below $THRESHOLD% threshold"
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
| **Tier 1 mutation score** | Unknown | 95%+ |
| Closed-form solvers | Unknown | 95%+ |
| Sketch2D constraint Jacobians | Unknown | 95%+ |
| Newton-Raphson / LM adapter | Unknown | 95%+ |
| **Tier 2 mutation score** | Unknown | 90%+ |
| Reduce passes | Unknown | 90%+ |
| Graph analysis (DOF, redundancy, patterns) | Unknown | 90%+ |
| Pipeline solve phase, dataflow tracker | Unknown | 90%+ |
| **Tier 3 mutation score** | Unknown | 85%+ |
| Sketch3D / Assembly constraints | Unknown | 85%+ |
| Branch selection / drag / sub_problem | Unknown | 85%+ |
| Legacy solver modules | Unknown | 85%+ |
| Equivalent mutants documented | 0 | All identified |

Track mutation score over time by storing `mutants.out/outcomes.json` as a CI artifact
and graphing the trend.

## Estimated Effort

| Task | Time |
|------|------|
| Install and run baseline across all tiers | 2 hours |
| Analyze Tier 1 surviving mutants (closed-form, sketch2d, NR, LM) | 4-6 hours |
| Write targeted tests to kill Tier 1 survivors | 6-8 hours |
| Analyze Tier 2 surviving mutants (reduce, graph, pipeline, dataflow) | 4-6 hours |
| Write targeted tests to kill Tier 2 survivors | 4-6 hours |
| Analyze Tier 3 surviving mutants (sketch3d, assembly, branch, drag, legacy) | 3-4 hours |
| Write targeted tests to kill Tier 3 survivors | 3-4 hours |
| CI integration (nightly + PR-incremental) | 2 hours |
| Document `#[mutants::skip]` decisions | 1 hour |
| **Total** | **~30-38 hours** (spread over multiple sessions) |
