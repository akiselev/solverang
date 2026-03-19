# Solverang V3 Comprehensive Testing Plan

**Status**: Active
**Scope**: All V3 architecture modules (~20,284 LOC production, ~6,125 LOC existing tests)
**Target coverage**: Bring test LOC to approximately 2:1 ratio with new V3 code

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [Test Architecture](#3-test-architecture)
4. [Unit Tests](#4-unit-tests)
5. [Integration Tests](#5-integration-tests)
6. [Property-Based Tests](#6-property-based-tests)
7. [Performance Tests](#7-performance-tests)
8. [Regression Tests](#8-regression-tests)
9. [JIT Testing (Cross-Reference)](#9-jit-testing-cross-reference)
10. [Prioritized Roadmap](#10-prioritized-roadmap)
11. [LOC Estimates](#11-loc-estimates)

---

## 1. Executive Summary

The V3 architecture introduces a trait-based constraint system with ~20k LOC across
13 new modules: `id`, `param`, `entity`, `constraint`, `graph`, `solve`, `reduce`,
`dataflow`, `system`, `sketch2d`, `sketch3d`, `assembly`, and `pipeline`. Existing
tests cover the pipeline integration layer (~1,605 LOC in `incremental_tests.rs` and
`minpack_bridge_tests.rs`) and the `system.rs` unit tests (~30 tests), but leave
critical gaps in geometric constraint correctness, assembly constraints, cross-module
integration, performance benchmarking, and fuzz testing of the new constraint types.

This plan defines a phased approach to close these gaps, organized from
highest-risk/highest-value tests (constraint correctness, Jacobian verification)
through integration and property-based testing, to performance benchmarking and
regression snapshot infrastructure.

**Total estimated new test LOC: ~8,500-10,500** across 4 phases.

---

## 2. Current State Assessment

### 2.1 What Exists

| Location | LOC | Tests | Coverage Area |
|----------|-----|-------|---------------|
| `src/pipeline/incremental_tests.rs` | 927 | 13 | Incremental solving, warm-start, reduction, diagnostics |
| `src/pipeline/minpack_bridge_tests.rs` | 678 | 9 | Legacy `Problem` trait bridging via pipeline |
| `src/system.rs` (inline `#[cfg(test)]`) | ~250 | 30+ | Entity/constraint lifecycle, solve basics, change tracking |
| `tests/geometric_tests.rs` | 464 | ~8 | Legacy `geometry` feature constraint solving |
| `tests/test_3d_constraints.rs` | 389 | ~6 | Legacy 3D constraint solving |
| `tests/property_tests.rs` | 896 | ~15 | Legacy `Problem` trait property tests |
| `tests/lm_tests.rs` | 372 | ~8 | LM solver correctness |
| `tests/solver_tests.rs` | 162 | ~4 | NR solver basics |
| `tests/solver_comparison.rs` | 470 | ~6 | NR vs LM vs Auto comparison |
| `tests/parallel_tests.rs` | 423 | ~6 | Parallel solver |
| `tests/sparse_tests.rs` | 638 | ~8 | Sparse solver |
| `tests/macro_tests.rs` | 413 | ~10 | `auto_jacobian` / `residual` macros |
| `tests/minpack_verification.rs` | 293 | ~5 | NIST/MINPACK verification |
| `benches/comprehensive.rs` | ~300 | - | Legacy benchmarks (NR vs LM vs Sparse) |
| `benches/nist_benchmarks.rs` | ~200 | - | NIST problem benchmarks |
| `benches/scaling.rs` | ~200 | - | Scaling benchmarks |

### 2.2 Critical Gaps

| Gap | Risk | Modules Affected |
|-----|------|------------------|
| No Sketch2D constraint correctness tests | **Critical** | `sketch2d/constraints.rs` (2,103 LOC, 15 constraint types) |
| No Sketch2D Jacobian verification | **Critical** | All 15 squared-formulation Jacobians are unverified against finite differences |
| No Sketch3D constraint correctness tests | **High** | `sketch3d/constraints.rs` (1,175 LOC, 8 constraint types) |
| No assembly constraint tests | **High** | `assembly/constraints.rs` (885 LOC, 4 constraint types + quaternion math) |
| No cross-module integration tests | **High** | Sketch2D -> Pipeline -> Solve -> verify geometry |
| No drag solving tests with real geometry | **Medium** | `solve/drag.rs` (367 LOC) |
| No branch management tests | **Medium** | `solve/branch.rs` (306 LOC) |
| No closed-form solver tests | **Medium** | `solve/closed_form.rs` (898 LOC, 4 pattern solvers) |
| No pattern detection tests | **Medium** | `graph/pattern.rs` with real constraints |
| No redundancy analysis tests | **Medium** | `graph/redundancy.rs` (653 LOC) |
| No DOF analysis tests with real geometry | **Medium** | `graph/dof.rs` |
| No performance benchmarks for V3 pipeline | **Medium** | Pipeline, decompose, reduce, solve |
| No property/fuzz tests for new constraints | **Medium** | All 27 new constraint types |
| No Sketch2DBuilder end-to-end tests | **Low** | `sketch2d/builder.rs` (679 LOC) |
| No reduce module unit tests | **Low** | `reduce/` (1,058 LOC) |
| JIT equivalence tests not written | **Low** (deferred) | `jit/` (covered by separate JIT plan) |

---

## 3. Test Architecture

### 3.1 Directory Layout

```
crates/solverang/
  src/
    sketch2d/
      constraint_tests.rs     # Inline unit tests for each constraint type
    sketch3d/
      constraint_tests.rs     # Inline unit tests for each 3D constraint type
    assembly/
      constraint_tests.rs     # Inline unit tests for assembly constraints
    graph/
      pattern_tests.rs        # Pattern detection unit tests
      redundancy_tests.rs     # Redundancy analysis tests
      dof_tests.rs            # DOF analysis tests
    solve/
      closed_form_tests.rs    # Closed-form solver tests
      branch_tests.rs         # Branch selection tests
      drag_tests.rs           # Drag solving tests
    reduce/
      reduce_tests.rs         # Reduce pass unit tests
    pipeline/
      incremental_tests.rs    # (existing - extend)
      minpack_bridge_tests.rs # (existing - keep)
    system.rs                 # (existing inline tests - extend)
  tests/
    v3_sketch2d_integration.rs   # Sketch2D end-to-end through pipeline
    v3_sketch3d_integration.rs   # Sketch3D end-to-end through pipeline
    v3_assembly_integration.rs   # Assembly end-to-end through pipeline
    v3_pipeline_integration.rs   # Multi-domain pipeline scenarios
    v3_drag_integration.rs       # Drag solving with real geometry
    v3_property_tests.rs         # Property-based tests for V3 constraints
  benches/
    v3_pipeline.rs               # V3 pipeline benchmarks
    v3_sketch2d.rs               # Sketch2D solve benchmarks
    v3_scaling.rs                # V3 scaling benchmarks
```

### 3.2 Test Helpers Module

Create a shared test utilities module to eliminate duplication across test files.
This should live at `src/test_helpers.rs` (gated behind `#[cfg(test)]`) or as a
`dev-dependency` helper crate.

```rust
// Shared helpers needed across multiple test files:

/// Build a ConstraintSystem with N 2D test points at given positions.
fn system_with_points(positions: &[(f64, f64)]) -> (ConstraintSystem, Vec<EntityId>, Vec<(ParamId, ParamId)>)

/// Assert that all residuals of a system are below tolerance after solving.
fn assert_solved(system: &mut ConstraintSystem, tol: f64)

/// Assert parameter value within tolerance.
fn assert_param_near(system: &ConstraintSystem, param: ParamId, expected: f64, tol: f64)

/// Build a Sketch2D system using Sketch2DBuilder and return the ConstraintSystem.
fn sketch2d_triangle(side_a: f64, side_b: f64, side_c: f64) -> ConstraintSystem

/// Verify a V3 Constraint's Jacobian against finite differences.
fn verify_v3_jacobian(constraint: &dyn Constraint, store: &ParamStore, tol: f64) -> bool
```

### 3.3 Test Naming Conventions

All V3 test functions should follow these patterns:

- Unit tests: `test_{module}_{constraint_or_function}_{scenario}`
  - Example: `test_sketch2d_distance_pt_pt_basic_solve`
  - Example: `test_sketch2d_distance_pt_pt_jacobian_vs_finite_diff`
- Integration tests: `test_v3_{domain}_{scenario}`
  - Example: `test_v3_sketch2d_triangle_distance_constrained`
- Property tests: `prop_{module}_{invariant}`
  - Example: `prop_sketch2d_distance_residual_zero_at_target`
- Benchmarks: `bench_v3_{module}_{scenario}`
  - Example: `bench_v3_pipeline_100_point_chain`

### 3.4 Tolerances

Standardize numerical tolerances across all V3 tests:

| Context | Symbol | Value | Rationale |
|---------|--------|-------|-----------|
| Solver convergence | `SOLVE_TOL` | `1e-8` | Matches default LM tolerance |
| Residual check | `RESIDUAL_TOL` | `1e-6` | Post-solve residual verification |
| Jacobian verification | `JACOBIAN_TOL` | `1e-5` | Finite-difference vs analytical |
| Finite-difference step | `FD_STEP` | `1e-7` | Central differences step size |
| Geometric position | `POSITION_TOL` | `1e-6` | Point coordinate comparison |
| Angle tolerance | `ANGLE_TOL` | `1e-6` | Radian-valued comparisons |
| Quaternion norm | `QUAT_TOL` | `1e-8` | Unit quaternion normalization |

---

## 4. Unit Tests

### 4.1 Sketch2D Constraints (`sketch2d/constraint_tests.rs`)

Each of the 15 Sketch2D constraint types requires three categories of unit test:

#### Category A: Residual correctness

Verify that `residuals()` returns zero (within tolerance) when the constraint is
satisfied, and returns a non-zero value when violated.

| Constraint | Test Cases |
|------------|------------|
| `DistancePtPt` | (a) Two points at exact distance -> residual ~0. (b) Points at wrong distance -> residual proportional to (d_actual^2 - d_target^2). (c) Coincident points with target=0 -> residual ~0. (d) Large distances (1e6 scale). |
| `Coincident` | (a) Same position -> residual ~0. (b) Different position -> residual = [dx, dy]. |
| `Fixed` | (a) Point at target -> residual ~0. (b) Point displaced -> residual = [dx, dy]. |
| `Horizontal` | (a) Same y-coordinate -> residual ~0. (b) Different y -> residual = dy. |
| `Vertical` | (a) Same x-coordinate -> residual ~0. (b) Different x -> residual = dx. |
| `Parallel` | (a) Parallel line segments -> residual ~0. (b) Perpendicular segments -> residual != 0. (c) Degenerate (zero-length segment). |
| `Perpendicular` | (a) Perpendicular segments -> residual ~0. (b) Parallel segments -> residual != 0. |
| `Angle` | (a) Segment at target angle -> residual ~0. (b) Various quadrants. (c) 0, pi/2, pi, 3pi/2 boundaries. |
| `Midpoint` | (a) Point at midpoint -> residual ~0. (b) Point displaced from midpoint. |
| `Symmetric` | (a) Symmetric points about axis -> residual ~0. (b) Asymmetric case. |
| `EqualLength` | (a) Equal-length segments -> residual ~0. (b) Unequal lengths -> nonzero. |
| `PointOnCircle` | (a) Point on circle -> residual ~0. (b) Point inside circle -> negative. (c) Point outside -> positive. |
| `TangentLineCircle` | (a) Tangent configuration -> residual ~0. (b) Secant -> nonzero. (c) Non-intersecting -> nonzero. |
| `TangentCircleCircle` | (a) Externally tangent -> residual ~0. (b) Internally tangent -> residual ~0. (c) Overlapping -> nonzero. |
| `DistancePtLine` | (a) Point at target distance from line -> residual ~0. (b) Point on line with target=0. |

#### Category B: Jacobian verification

For each constraint, verify the analytical Jacobian against central finite
differences at 3-5 random configurations:

```rust
#[test]
fn test_distance_pt_pt_jacobian_vs_finite_diff() {
    let store = /* build param store with random values */;
    let constraint = DistancePtPt::new(/* ... */);
    let analytical = constraint.jacobian(&store);
    let numerical = finite_difference_jacobian_v3(&constraint, &store, FD_STEP);
    assert_jacobians_match(analytical, numerical, JACOBIAN_TOL);
}
```

This must be done for all 15 constraints. The squared formulations (DistancePtPt,
EqualLength, PointOnCircle, TangentLineCircle, TangentCircleCircle) are especially
important because the Jacobian of `f^2` differs from the Jacobian of `f`.

#### Category C: Solver convergence

For each constraint type, set up a minimal system that exercises only that constraint
(plus enough fixed constraints to remove remaining DOF), perturb the initial values,
and verify the solver converges to the correct geometry.

**Estimated tests: ~90 (15 constraints x ~6 tests each)**

### 4.2 Sketch3D Constraints (`sketch3d/constraint_tests.rs`)

Same three categories (residual, Jacobian, convergence) for all 8 Sketch3D constraints:

| Constraint | Specific Test Focus |
|------------|-------------------|
| `Distance3D` | 3D distance between points |
| `Coincident3D` | 3-component residual |
| `Fixed3D` | 3-component target |
| `PointOnPlane` | Plane normal dot product formulation |
| `Coplanar` | Multiple points on same plane |
| `Parallel3D` | 3D cross-product based residual |
| `Perpendicular3D` | 3D dot-product based residual |
| `Coaxial` | Axis alignment in 3D |

Special attention: `Parallel3D` and `Perpendicular3D` operate on direction vectors
derived from line segment endpoints. Verify they handle degenerate (zero-length)
segments gracefully.

**Estimated tests: ~48 (8 constraints x ~6 tests each)**

### 4.3 Assembly Constraints (`assembly/constraint_tests.rs`)

Assembly constraints involve quaternion math, which is error-prone. Each constraint
needs Jacobian verification with special attention to quaternion derivatives.

| Constraint | Test Focus |
|------------|-----------|
| `UnitQuaternion` | (a) Normalized quaternion -> residual ~0. (b) Unnormalized -> residual = norm^2 - 1. (c) Jacobian at various orientations. |
| `Mate` | (a) Two bodies with coincident local points -> residual ~0. (b) Bodies displaced -> 3-component residual. (c) Jacobian vs finite differences at 5 random orientations. (d) Verify rotation matrix derivatives (`quat_rotate_derivatives`). |
| `CoaxialAssembly` | (a) Aligned axes -> residual ~0. (b) Misaligned axes. (c) Jacobian at various orientations. |
| `Insert` | (a) Coaxial + axial mate satisfied -> residual ~0. (b) Partial satisfaction. (c) Jacobian (composite of Mate + Coaxial). |
| `Gear` | (a) Correct rotation ratio -> residual ~0. (b) Incorrect ratio. (c) Jacobian (may use finite differences internally -- verify against double finite differences). |

Additional quaternion math tests:
- `quat_to_rotation_matrix`: Verify against known rotation matrices for axis-angle inputs.
- `quat_rotate_derivatives`: Verify all 4 quaternion partial derivatives against finite differences of the rotation.
- Edge cases: Identity quaternion, 180-degree rotations, near-gimbal-lock orientations.

**Estimated tests: ~40**

### 4.4 Graph Module

#### 4.4.1 Pattern Detection (`graph/pattern_tests.rs`)

| Test | Description |
|------|-------------|
| `test_pattern_scalar_single_eq` | 1 constraint + 1 free param -> ScalarSolve |
| `test_pattern_two_distances` | 2 DistancePtPt on same Point2D -> TwoDistances |
| `test_pattern_hv` | Horizontal + Vertical on same point -> HorizontalVertical |
| `test_pattern_distance_angle` | DistancePtPt + Angle on same point -> DistanceAngle |
| `test_pattern_no_match` | 3 constraints on 1 point -> no pattern |
| `test_pattern_mixed_entities` | Patterns across multiple entity types |
| `test_pattern_with_fixed_params` | Fixed params reduce free param count; verify pattern still matches |
| `test_pattern_constraint_name_matching` | Verify `classify_constraint` correctly categorizes all 15 Sketch2D names |

**Estimated tests: ~15**

#### 4.4.2 Redundancy Analysis (`graph/redundancy_tests.rs`)

| Test | Description |
|------|-------------|
| `test_redundancy_none` | Well-constrained system -> no redundant constraints |
| `test_redundancy_duplicate` | Same constraint added twice -> detected |
| `test_redundancy_implied` | Horizontal(A,B) + Horizontal(B,C) + Horizontal(A,C) -> 1 redundant |
| `test_redundancy_conflicting` | Fixed(A, (0,0)) + Fixed(A, (1,1)) -> conflict detected |
| `test_redundancy_over_constrained_triangle` | 3 distances + horizontal + vertical + fix -> identify surplus |
| `test_redundancy_svd_rank` | Verify SVD rank computation matches expected rank |

**Estimated tests: ~10**

#### 4.4.3 DOF Analysis (`graph/dof_tests.rs`)

| Test | Description |
|------|-------------|
| `test_dof_unconstrained_point` | 1 free Point2D -> DOF = 2 |
| `test_dof_fixed_point` | Fixed Point2D -> DOF = 0 |
| `test_dof_two_points_distance` | 2 points + distance -> DOF = 3 (2+2-1) |
| `test_dof_triangle_fully_constrained` | 3 pts + 3 distances + fix + horiz -> DOF = 0 |
| `test_dof_per_entity` | Verify `EntityDof` reports per-entity breakdown |
| `test_dof_quick_vs_full` | `quick_dof` matches `analyze_dof` for several configs |
| `test_dof_3d_tetrahedron` | 4 Point3D + 6 distances + fix -> DOF = 3 (rotation) |
| `test_dof_rigid_body` | RigidBody (7 params) + UnitQuaternion (1 eq) -> DOF = 6 |

**Estimated tests: ~12**

### 4.5 Solve Module

#### 4.5.1 Closed-Form Solvers (`solve/closed_form_tests.rs`)

| Test | Description |
|------|-------------|
| `test_scalar_solve_linear` | Single linear equation -> exact solution |
| `test_scalar_solve_quadratic` | x^2 - 4 = 0 -> x = +/- 2 (branch selection) |
| `test_two_distances_intersecting` | Two circles that intersect -> 2 solutions |
| `test_two_distances_tangent` | Two circles tangent -> 1 solution |
| `test_two_distances_non_intersecting` | Two circles too far apart -> no solution |
| `test_two_distances_concentric` | Same center, different radii -> no solution |
| `test_hv_direct` | Horizontal + Vertical -> direct assignment of x, y |
| `test_distance_angle_quadrant_1` | Distance + angle in Q1 -> polar conversion |
| `test_distance_angle_quadrant_2` | Distance + angle in Q2 |
| `test_distance_angle_quadrant_3` | Distance + angle in Q3 |
| `test_distance_angle_quadrant_4` | Distance + angle in Q4 |
| `test_distance_angle_zero_angle` | Angle = 0 -> point on positive x-axis |

**Estimated tests: ~15**

#### 4.5.2 Branch Selection (`solve/branch_tests.rs`)

| Test | Description |
|------|-------------|
| `test_branch_closest_single_converged` | 1 converged result -> selected |
| `test_branch_closest_two_converged` | 2 converged, pick closest to previous |
| `test_branch_closest_none_converged` | All diverged -> None |
| `test_branch_smallest_residual` | 3 converged, pick smallest residual norm |
| `test_branch_mixed_converged_diverged` | Some converged, some not |
| `test_branch_identical_distances` | Tie-breaking behavior |

**Estimated tests: ~8**

#### 4.5.3 Drag Solving (`solve/drag_tests.rs`)

| Test | Description |
|------|-------------|
| `test_drag_unconstrained` | No constraints -> displacement preserved fully |
| `test_drag_fully_constrained` | DOF = 0 -> displacement projected to zero |
| `test_drag_1dof_horizontal` | Point constrained vertically, drag horizontally -> preserved |
| `test_drag_1dof_vertical` | Point constrained horizontally, drag vertically -> preserved |
| `test_drag_preservation_ratio` | Verify `preservation_ratio` is in [0, 1] |
| `test_drag_null_space_orthogonal` | Projected displacement is orthogonal to constraint tangent space |
| `test_drag_svd_tolerance` | Very small singular values treated as zero |

**Estimated tests: ~10**

#### 4.5.4 Sub-Problem Adapter (`solve/sub_problem_tests.rs`)

| Test | Description |
|------|-------------|
| `test_reduced_sub_problem_basic` | ReducedSubProblem implements Problem correctly |
| `test_reduced_sub_problem_residuals` | Residuals match constraint residuals |
| `test_reduced_sub_problem_jacobian` | Jacobian columns mapped correctly via SolverMapping |
| `test_reduced_sub_problem_fixed_params_excluded` | Fixed params not in variable set |

**Estimated tests: ~6**

### 4.6 Reduce Module (`reduce/reduce_tests.rs`)

| Test | Description |
|------|-------------|
| `test_substitute_fixed_single` | 1 fixed param -> substituted out of constraint |
| `test_substitute_fixed_all_params` | All constraint params fixed -> constraint removed |
| `test_eliminate_trivial_satisfied` | Constraint residual = 0 at current values -> eliminated |
| `test_eliminate_trivial_unsatisfied` | Non-zero residual -> kept |
| `test_merge_equality_simple` | param_a = param_b equality constraint -> merged |
| `test_merge_equality_chain` | a = b, b = c -> all three merged |
| `test_merge_equality_cycle` | a = b, b = a -> handled without infinite loop |
| `test_reduce_combined` | All three passes in sequence on a mixed system |

**Estimated tests: ~12**

### 4.7 DataFlow Module

#### ChangeTracker (extend existing tests in `system.rs`)

| Test | Description |
|------|-------------|
| `test_tracker_param_dirty` | `mark_param_dirty` -> `dirty_params` contains it |
| `test_tracker_structural_add_entity` | Adding entity marks structural change |
| `test_tracker_structural_add_constraint` | Adding constraint marks structural change |
| `test_tracker_structural_remove` | Removing entity/constraint marks structural |
| `test_tracker_compute_dirty_clusters` | Dirty param -> correct cluster marked dirty |
| `test_tracker_clear` | After `clear()`, no changes reported |

#### SolutionCache

| Test | Description |
|------|-------------|
| `test_cache_store_retrieve` | Store and get back identical values |
| `test_cache_invalidate` | Invalidated cluster not returned |
| `test_cache_clear_all` | `clear()` empties cache |
| `test_cache_overwrite` | Second store for same cluster replaces first |

**Estimated tests: ~12**

### 4.8 Param Module

| Test | Description |
|------|-------------|
| `test_param_store_alloc_get_set` | Basic CRUD |
| `test_param_store_generational_id` | Old ID not valid after removal |
| `test_param_store_fixed` | Fixed param reported correctly |
| `test_param_store_solver_mapping` | SolverMapping excludes fixed params |
| `test_param_store_bulk_operations` | Bulk get/set for solver integration |

**Estimated tests: ~8**

---

## 5. Integration Tests

Integration tests live in `crates/solverang/tests/` and exercise the full pipeline
from entity creation through solving to geometric verification.

### 5.1 Sketch2D Integration (`tests/v3_sketch2d_integration.rs`)

These tests use the `Sketch2DBuilder` or direct `ConstraintSystem` API to build
real geometric problems with real Sketch2D entities and constraints, solve them
through the full V3 pipeline, and verify the resulting geometry.

| Test | Entities | Constraints | Verification |
|------|----------|-------------|-------------|
| `test_v3_triangle_3_distances` | 3 Point2D | Fix(p0) + Horizontal(p0,p1) + Distance x3 | Side lengths match targets |
| `test_v3_square_4_points` | 4 Point2D | Fix(p0) + Horizontal x2 + Vertical x2 + Distance x4 | All sides equal, right angles |
| `test_v3_circle_with_points_on_it` | 1 Circle2D + 3 Point2D | PointOnCircle x3 + Fix(circle center) | All points at radius distance |
| `test_v3_tangent_line_circle` | 1 Circle2D + 1 LineSegment2D | TangentLineCircle + Fix(circle) + Fix(line start) | Tangency verified geometrically |
| `test_v3_parallel_lines` | 2 LineSegment2D | Parallel + Fix(line1) + Distance(line2 start, origin) | Slopes match |
| `test_v3_perpendicular_lines` | 2 LineSegment2D | Perpendicular + Fix(line1) + Fix(line2 start) | Dot product of directions = 0 |
| `test_v3_midpoint_constraint` | 3 Point2D | Midpoint(p2, p0, p1) + Fix(p0) + Fix(p1) | p2 = (p0+p1)/2 |
| `test_v3_symmetric_about_vertical` | 3 Point2D | Symmetric + Fix(axis) | Mirror positions verified |
| `test_v3_equal_length_segments` | 2 LineSegment2D | EqualLength + Fix(seg1) + Fix(seg2 start) | Lengths match |
| `test_v3_fully_constrained_mechanism` | 5 Point2D + 2 LineSegment2D | Mix of distance, parallel, perpendicular, fix | DOF = 0, all constraints satisfied |
| `test_v3_under_constrained_solve` | 2 Point2D | Distance only (no fix) | Solver converges, distance satisfied, DOF > 0 |
| `test_v3_over_constrained_detected` | 2 Point2D | Fix(both) + Distance (conflicting) | Diagnostics report over-constraint |
| `test_v3_builder_api_triangle` | 3 Point2D | Via Sketch2DBuilder | Same as manual construction |
| `test_v3_arc_point_on_arc` | 1 Arc2D + 1 Point2D | PointOnCircle (arc as circle) | Point on arc boundary |
| `test_v3_infinite_line_point_on_line` | 1 InfiniteLine2D + 1 Point2D | Collinear | Point on the infinite line |

**Estimated tests: ~15, ~600 LOC**

### 5.2 Sketch3D Integration (`tests/v3_sketch3d_integration.rs`)

| Test | Entities | Constraints | Verification |
|------|----------|-------------|-------------|
| `test_v3_3d_tetrahedron` | 4 Point3D | Fix(p0) + Distance3D x6 | All edge lengths correct |
| `test_v3_3d_point_on_plane` | 1 Plane + 3 Point3D | PointOnPlane x3 + Fix(plane) | All points satisfy plane eq |
| `test_v3_3d_parallel_segments` | 2 LineSegment3D | Parallel3D + Fix(seg1) + Fix(seg2 start) | Cross product = 0 |
| `test_v3_3d_perpendicular_segments` | 2 LineSegment3D | Perpendicular3D + Fix(seg1) + Fix(seg2 start) | Dot product = 0 |
| `test_v3_3d_coplanar_points` | 4 Point3D | Coplanar + Fix(3 points) | 4th point on plane of first 3 |
| `test_v3_3d_mixed_constraints` | 4 Point3D + 1 Plane | Distance3D + PointOnPlane + Fix | Multi-constraint 3D solve |

**Estimated tests: ~8, ~400 LOC**

### 5.3 Assembly Integration (`tests/v3_assembly_integration.rs`)

| Test | Description |
|------|-------------|
| `test_v3_assembly_two_body_mate` | Two RigidBodies with Mate constraint -> contact point coincident |
| `test_v3_assembly_coaxial` | Two bodies with CoaxialAssembly -> axes aligned |
| `test_v3_assembly_insert` | Pin-in-hole (Coaxial + Mate) -> verify both constraints |
| `test_v3_assembly_gear_ratio` | Two bodies with Gear -> rotation ratio maintained |
| `test_v3_assembly_quaternion_normalization` | UnitQuaternion constraint keeps ||q|| = 1 throughout solve |
| `test_v3_assembly_chain` | 3+ bodies chained by mates -> all joints solved |

**Estimated tests: ~8, ~500 LOC**

### 5.4 Pipeline Integration (`tests/v3_pipeline_integration.rs`)

These tests verify the pipeline stages work together correctly for non-trivial
multi-cluster systems.

| Test | Description |
|------|-------------|
| `test_v3_pipeline_two_independent_sketches` | Two separate Sketch2D triangles -> 2 clusters, both solved |
| `test_v3_pipeline_incremental_param_change` | Solve, change 1 param, re-solve -> only dirty cluster re-solved |
| `test_v3_pipeline_incremental_add_constraint` | Solve, add constraint, re-solve -> re-decomposed |
| `test_v3_pipeline_incremental_remove_constraint` | Solve, remove constraint, re-solve -> re-decomposed |
| `test_v3_pipeline_warm_start_fewer_iterations` | Second solve uses fewer iterations than first |
| `test_v3_pipeline_reduce_eliminates_fixed` | Fixed params correctly substituted in reduce phase |
| `test_v3_pipeline_pattern_detected_and_used` | Pattern solver invoked for HV pattern -> exact solution |
| `test_v3_pipeline_redundancy_detected` | Redundant constraint flagged in diagnostics |
| `test_v3_pipeline_dof_reported` | DOF analysis matches expected value |
| `test_v3_pipeline_large_system_30_points` | 30-point chain -> decomposes and solves correctly |

**Estimated tests: ~12, ~600 LOC**

### 5.5 Drag Integration (`tests/v3_drag_integration.rs`)

| Test | Description |
|------|-------------|
| `test_v3_drag_point_on_line` | Point constrained to line, drag perpendicular -> stays on line |
| `test_v3_drag_point_on_circle` | Point on circle, drag outward -> stays on circle |
| `test_v3_drag_triangle_vertex` | Drag one vertex of constrained triangle -> distances preserved |
| `test_v3_drag_fully_constrained` | Drag point in DOF=0 system -> no movement |
| `test_v3_drag_then_solve` | Drag displacement applied, then full re-solve -> consistent |

**Estimated tests: ~6, ~300 LOC**

---

## 6. Property-Based Tests

Property-based tests use `proptest` to verify invariants over randomly generated
inputs. These catch edge cases that hand-written tests miss.

File: `tests/v3_property_tests.rs`

### 6.1 Constraint Invariants

For every constraint type, verify these universal properties:

```rust
// Property 1: Residual is zero when constraint is satisfied
proptest! {
    #[test]
    fn prop_distance_residual_zero_at_target(
        x1 in -1000.0..1000.0f64,
        y1 in -1000.0..1000.0f64,
        angle in 0.0..std::f64::consts::TAU,
        dist in 0.01..1000.0f64,
    ) {
        let x2 = x1 + dist * angle.cos();
        let y2 = y1 + dist * angle.sin();
        // Build constraint with target = dist
        // Assert residual < RESIDUAL_TOL
    }
}

// Property 2: Jacobian is consistent with finite differences
proptest! {
    #[test]
    fn prop_distance_jacobian_matches_fd(
        x1 in -100.0..100.0f64,
        y1 in -100.0..100.0f64,
        x2 in -100.0..100.0f64,
        y2 in -100.0..100.0f64,
        dist in 0.01..100.0f64,
    ) {
        // Build constraint, compute analytical and FD Jacobians
        // Assert max difference < JACOBIAN_TOL
    }
}

// Property 3: Residual changes sign/magnitude appropriately
// Property 4: Jacobian entry count matches expected sparsity
```

### 6.2 System-Level Invariants

| Property | Description |
|----------|-------------|
| `prop_solve_reduces_residual` | For any solvable system, `solve()` reduces total residual norm |
| `prop_dof_non_negative` | DOF is always >= 0 for any valid system |
| `prop_dof_decreases_with_constraints` | Adding a non-redundant constraint decreases DOF by its equation count |
| `prop_cluster_count_leq_constraint_count` | Number of clusters never exceeds number of constraints |
| `prop_incremental_equals_full` | Incremental solve produces same result as fresh solve |
| `prop_fixed_param_unchanged` | Fixed params are never modified by solve |
| `prop_cache_does_not_change_result` | Warm-started solve converges to same solution as cold start |

### 6.3 Parameterized Geometry Generation

Generate random but valid geometric configurations:

```rust
/// Generate a random N-point chain with distance constraints.
fn arb_point_chain(n: usize) -> impl Strategy<Value = ConstraintSystem> { ... }

/// Generate a random polygon with side length constraints.
fn arb_polygon(sides: usize) -> impl Strategy<Value = ConstraintSystem> { ... }

/// Generate a random rigid body with random orientation.
fn arb_rigid_body() -> impl Strategy<Value = (ConstraintSystem, EntityId)> { ... }
```

### 6.4 Fuzz-Like Stress Tests

| Test | Description |
|------|-------------|
| `prop_random_constraint_removal_no_panic` | Build system, remove random constraints, solve -> no panic |
| `prop_random_param_perturbation_converges` | Perturb solved system by small amount, re-solve -> converges |
| `prop_large_random_system_no_panic` | 50+ entities, random constraints -> solve completes without panic |
| `prop_quaternion_normalization_survives` | Random quaternion values -> UnitQuaternion constraint converges |

**Estimated tests: ~35, ~1,200 LOC**

---

## 7. Performance Tests

### 7.1 V3 Pipeline Benchmarks (`benches/v3_pipeline.rs`)

| Benchmark | Description | Sizes |
|-----------|-------------|-------|
| `bench_pipeline_point_chain` | N-point chain with distance constraints | 10, 50, 100, 500 |
| `bench_pipeline_grid` | NxN grid with horizontal + vertical + distance | 5x5, 10x10, 20x20 |
| `bench_pipeline_star` | Central point + N radial distances | 10, 50, 100 |
| `bench_pipeline_incremental_vs_full` | Full solve vs incremental (1 param changed) | 50, 100, 500 |
| `bench_pipeline_decompose` | Decomposition time only (no solve) | 100, 500, 1000 entities |
| `bench_pipeline_reduce` | Reduce phase time only | 50, 100, 500 constraints |
| `bench_pipeline_warm_start_speedup` | Ratio of warm-start to cold-start iterations | 50, 100 |

### 7.2 Sketch2D Benchmarks (`benches/v3_sketch2d.rs`)

| Benchmark | Description | Sizes |
|-----------|-------------|-------|
| `bench_sketch2d_triangle` | Single triangle solve (baseline) | 1 |
| `bench_sketch2d_100_triangles` | 100 independent triangles | 100 |
| `bench_sketch2d_mechanism` | 4-bar linkage mechanism | 1 |
| `bench_sketch2d_residual_eval` | Residual evaluation only (no solve) for N constraints | 100, 500, 1000 |
| `bench_sketch2d_jacobian_eval` | Jacobian evaluation only for N constraints | 100, 500, 1000 |
| `bench_sketch2d_builder_construction` | Sketch2DBuilder construction time | 100, 500 entities |

### 7.3 Scaling Benchmarks (`benches/v3_scaling.rs`)

| Benchmark | Description | Sizes |
|-----------|-------------|-------|
| `bench_scaling_entities` | Solve time vs entity count | 10, 50, 100, 500, 1000 |
| `bench_scaling_constraints_per_entity` | Solve time vs constraint density | 1, 2, 3, 5, 10 per entity |
| `bench_scaling_clusters` | Solve time vs number of independent clusters | 1, 10, 50, 100 |
| `bench_scaling_pattern_vs_iterative` | Pattern solver vs LM for pattern-matchable problems | 10, 50, 100 |
| `bench_scaling_drag` | Drag solve time vs system size | 10, 50, 100, 500 |

### 7.4 Comparison Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `bench_v3_vs_legacy_triangle` | V3 ConstraintSystem vs legacy geometry::ConstraintSystem for same problem |
| `bench_v3_vs_legacy_100_points` | V3 vs legacy for 100-point chain |
| `bench_assembly_vs_sketch` | Assembly Mate constraint vs Sketch3D Coincident3D (same geometric problem) |

**Estimated benchmark LOC: ~800**

---

## 8. Regression Tests

### 8.1 Golden File / Snapshot Testing

For deterministic problems, capture the exact solution as a golden file and
assert future runs match. This catches unintentional changes to solver behavior.

**Strategy**: Use `insta` crate (or similar) for snapshot testing.

| Snapshot | Contents |
|----------|----------|
| `triangle_3_6_8.snap` | Solved coordinates for 3-6-8 triangle |
| `square_10.snap` | Solved coordinates for 10x10 square |
| `tetrahedron_10.snap` | Solved 3D coordinates for regular tetrahedron |
| `assembly_mate.snap` | Solved rigid body positions for basic mate |
| `pipeline_diagnostics.snap` | Diagnostic output for over-constrained system |

### 8.2 Regression Test Cases

These encode previously-encountered bugs or tricky edge cases. Each test should
include a comment explaining what regression it guards against.

| Test | Regression |
|------|-----------|
| `test_regression_squared_distance_zero` | DistancePtPt with target=0 and coincident initial points (Jacobian is zero -- solver must handle) |
| `test_regression_parallel_zero_length` | Parallel constraint on zero-length segment (degenerate direction vector) |
| `test_regression_angle_wrap_around` | Angle constraint near 2*pi boundary (wrap-around discontinuity) |
| `test_regression_quaternion_flip` | Quaternion sign flip during solve (q and -q represent same rotation) |
| `test_regression_incremental_stale_cache` | Stale cache entry after structural change causes wrong result |
| `test_regression_reduce_merge_cycle` | Equality merge with a = b = a cycle |
| `test_regression_pattern_false_positive` | Pattern detector matches a sub-graph that isn't actually solvable in closed form |

**Estimated tests: ~10, ~300 LOC**

### 8.3 NIST/MINPACK Regression via V3

Bridge the existing NIST test problems through the V3 pipeline to verify the
`minpack_bridge_tests.rs` approach works for all 18 MINPACK problems, not just
the 9 currently tested.

**Estimated tests: ~9 additional, ~400 LOC**

---

## 9. JIT Testing (Cross-Reference)

JIT testing is covered by the separate JIT testing plan at
`docs/plans/jit/level-1-make-it-work.md` through `level-3-make-it-transformative.md`.

This V3 testing plan creates the foundation that JIT tests build on. Specifically:

| JIT Test Need | V3 Foundation |
|---------------|---------------|
| JIT equivalence tests (JIT output matches interpreted) | V3 constraint correctness tests provide the reference values |
| JIT fused evaluation benchmarks | V3 pipeline benchmarks provide the baseline |
| JIT compiled Newton step validation | V3 solver convergence tests provide expected solutions |

**Integration point**: When JIT Level 1 tests are implemented, they should import
the V3 test helper functions and reuse the same geometric configurations.

---

## 10. Prioritized Roadmap

### Phase 1: Constraint Correctness (Highest Priority)

**Rationale**: The 27 new constraint types (15 Sketch2D + 8 Sketch3D + 4 Assembly)
are the foundation of the entire system. Incorrect residuals or Jacobians will
cause every downstream component to produce wrong results. This is the single
highest-risk gap.

| Task | Est. LOC | Files |
|------|----------|-------|
| Sketch2D Jacobian verification (all 15) | 600 | `sketch2d/constraint_tests.rs` |
| Sketch2D residual correctness (all 15) | 400 | `sketch2d/constraint_tests.rs` |
| Sketch2D solver convergence (all 15) | 300 | `sketch2d/constraint_tests.rs` |
| Sketch3D Jacobian + residual + convergence (all 8) | 500 | `sketch3d/constraint_tests.rs` |
| Assembly Jacobian + residual + convergence (all 5) | 500 | `assembly/constraint_tests.rs` |
| Quaternion math unit tests | 200 | `assembly/constraint_tests.rs` |
| Test helpers module | 200 | `src/test_helpers.rs` |
| **Phase 1 Total** | **~2,700** | |

**Exit criteria**: Every constraint type has (a) residual = 0 at satisfied config,
(b) Jacobian matches finite differences at 3+ random configs, (c) minimal system
converges under LM solver.

### Phase 2: Integration Tests (High Priority)

**Rationale**: Unit tests verify individual constraints but not their composition.
Real CAD usage involves 5-50 constraints interacting through the pipeline. This
phase catches issues in decomposition, reduction, pattern matching, and the
solve-write-back loop.

| Task | Est. LOC | Files |
|------|----------|-------|
| Sketch2D integration (15 scenarios) | 600 | `tests/v3_sketch2d_integration.rs` |
| Sketch3D integration (8 scenarios) | 400 | `tests/v3_sketch3d_integration.rs` |
| Assembly integration (6 scenarios) | 500 | `tests/v3_assembly_integration.rs` |
| Pipeline integration (12 scenarios) | 600 | `tests/v3_pipeline_integration.rs` |
| Drag integration (6 scenarios) | 300 | `tests/v3_drag_integration.rs` |
| **Phase 2 Total** | **~2,400** | |

**Exit criteria**: All integration tests pass. A triangle, square, and tetrahedron
can be fully constrained and solved through the V3 pipeline with correct
final geometry.

### Phase 3: Solver Internals + Property Tests (Medium Priority)

**Rationale**: Closed-form solvers, branch selection, reduce passes, and graph
analysis are internal details that could harbor subtle bugs. Property-based
tests catch edge cases not covered by example-based tests.

| Task | Est. LOC | Files |
|------|----------|-------|
| Closed-form solver tests (15 tests) | 400 | `solve/closed_form_tests.rs` |
| Branch selection tests (8 tests) | 200 | `solve/branch_tests.rs` |
| Drag unit tests (10 tests) | 250 | `solve/drag_tests.rs` |
| Sub-problem adapter tests (6 tests) | 150 | `solve/sub_problem_tests.rs` |
| Pattern detection tests (15 tests) | 300 | `graph/pattern_tests.rs` |
| Redundancy analysis tests (10 tests) | 250 | `graph/redundancy_tests.rs` |
| DOF analysis tests (12 tests) | 250 | `graph/dof_tests.rs` |
| Reduce module tests (12 tests) | 300 | `reduce/reduce_tests.rs` |
| DataFlow tests (12 tests) | 200 | Extend `system.rs` or `dataflow/` |
| Param store tests (8 tests) | 150 | `param/` |
| Property-based tests (35 tests) | 1,200 | `tests/v3_property_tests.rs` |
| **Phase 3 Total** | **~3,650** | |

**Exit criteria**: All solver internal tests pass. Property tests run with
100+ cases each without failure. Pattern detection correctly identifies all
4 pattern types in isolation and rejects non-matching configurations.

### Phase 4: Performance + Regression (Lower Priority)

**Rationale**: Performance testing establishes baselines and catches regressions.
Snapshot tests lock in known-good behavior. These are less urgent than correctness
but essential before release.

| Task | Est. LOC | Files |
|------|----------|-------|
| V3 pipeline benchmarks | 300 | `benches/v3_pipeline.rs` |
| Sketch2D benchmarks | 200 | `benches/v3_sketch2d.rs` |
| Scaling benchmarks | 200 | `benches/v3_scaling.rs` |
| Snapshot tests (insta) | 200 | Inline or `tests/v3_snapshots.rs` |
| Regression tests | 300 | `tests/v3_regressions.rs` |
| Additional MINPACK bridge tests | 400 | Extend `minpack_bridge_tests.rs` |
| **Phase 4 Total** | **~1,600** | |

**Exit criteria**: Benchmarks run and produce reports. Snapshot tests lock in
deterministic solutions. All regression tests pass.

---

## 11. LOC Estimates

### Summary by Phase

| Phase | Description | Est. LOC | Cumulative |
|-------|-------------|----------|------------|
| Phase 1 | Constraint Correctness | 2,700 | 2,700 |
| Phase 2 | Integration Tests | 2,400 | 5,100 |
| Phase 3 | Internals + Property Tests | 3,650 | 8,750 |
| Phase 4 | Performance + Regression | 1,600 | 10,350 |

### Summary by Test Category

| Category | Est. LOC | Test Count |
|----------|----------|------------|
| Unit tests (constraints) | 2,500 | ~178 |
| Unit tests (solver internals) | 1,500 | ~63 |
| Unit tests (graph/reduce/dataflow) | 1,150 | ~52 |
| Integration tests | 2,400 | ~55 |
| Property-based tests | 1,200 | ~35 |
| Benchmarks | 700 | ~30 benchmark groups |
| Regression / snapshot tests | 700 | ~19 |
| Test helpers | 200 | - |
| **Total** | **~10,350** | **~432** |

### Comparison with Existing Test Code

| Metric | Before | After |
|--------|--------|-------|
| Existing test LOC | 6,125 | 6,125 |
| New V3 test LOC | 0 | ~10,350 |
| Total test LOC | 6,125 | ~16,475 |
| V3 production LOC | 20,284 | 20,284 |
| Test-to-production ratio (V3 only) | 0.08:1 | 0.59:1 |
| Test-to-production ratio (overall) | 0.30:1 | 0.81:1 |

---

## Appendix A: Test Dependencies

```toml
[dev-dependencies]
proptest = "1.4"
insta = "1.34"           # For snapshot testing
approx = "0.5"           # For float comparison macros
rand = "0.8"             # For random test configurations
rand_chacha = "0.3"      # Deterministic RNG for reproducibility
```

## Appendix B: CI Integration

All V3 tests should be added to the CI pipeline:

1. **Unit tests**: `cargo test -p solverang` -- runs on every PR.
2. **Integration tests**: `cargo test -p solverang --test 'v3_*'` -- runs on every PR.
3. **Property tests**: `cargo test -p solverang --test v3_property_tests` -- runs on
   every PR but with reduced case count (`PROPTEST_CASES=50`). Nightly runs use
   `PROPTEST_CASES=1000`.
4. **Benchmarks**: `cargo bench -p solverang --bench 'v3_*'` -- runs nightly, not on
   every PR. Results stored for trend analysis.
5. **Snapshot updates**: `cargo insta review` -- manual step when intentional changes
   are made to solver behavior.

## Appendix C: Test Execution Order

For local development, the recommended execution order is:

```bash
# Fast feedback (< 30s) -- run first
cargo test -p solverang --lib              # Unit tests only

# Medium feedback (< 2min) -- run before pushing
cargo test -p solverang                    # All tests including integration

# Full suite (< 10min) -- run before merging
PROPTEST_CASES=500 cargo test -p solverang
cargo bench -p solverang --bench 'v3_*'    # Optional: check for perf regressions
```

## Appendix D: V3 Module-to-Test Mapping

| Production Module | LOC | Test File(s) | Phase |
|-------------------|-----|--------------|-------|
| `sketch2d/constraints.rs` | 2,103 | `sketch2d/constraint_tests.rs`, `tests/v3_sketch2d_integration.rs` | 1, 2 |
| `sketch2d/entities.rs` | 611 | (covered by constraint tests) | 1 |
| `sketch2d/builder.rs` | 679 | `tests/v3_sketch2d_integration.rs` | 2 |
| `sketch3d/constraints.rs` | 1,175 | `sketch3d/constraint_tests.rs`, `tests/v3_sketch3d_integration.rs` | 1, 2 |
| `sketch3d/entities.rs` | 392 | (covered by constraint tests) | 1 |
| `assembly/constraints.rs` | 885 | `assembly/constraint_tests.rs`, `tests/v3_assembly_integration.rs` | 1, 2 |
| `assembly/entities.rs` | 524 | `assembly/constraint_tests.rs` | 1 |
| `system.rs` | 950 | `system.rs` (inline, existing), `tests/v3_pipeline_integration.rs` | 2 |
| `pipeline/mod.rs` | 818 | `pipeline/incremental_tests.rs` (existing), `tests/v3_pipeline_integration.rs` | 2 |
| `pipeline/analyze.rs` | 434 | `tests/v3_pipeline_integration.rs` | 2 |
| `pipeline/decompose.rs` | 443 | `tests/v3_pipeline_integration.rs` | 2 |
| `pipeline/reduce.rs` | 974 | `reduce/reduce_tests.rs` | 3 |
| `pipeline/solve_phase.rs` | 846 | `tests/v3_pipeline_integration.rs` | 2 |
| `pipeline/post_process.rs` | 191 | `tests/v3_pipeline_integration.rs` | 2 |
| `pipeline/traits.rs` | 107 | (covered by pipeline integration) | 2 |
| `pipeline/types.rs` | 111 | (covered by pipeline integration) | 2 |
| `graph/bipartite.rs` | 416 | `graph/pattern_tests.rs` | 3 |
| `graph/cluster.rs` | 408 | `graph/dof_tests.rs` | 3 |
| `graph/decompose.rs` | 474 | `tests/v3_pipeline_integration.rs` | 2 |
| `graph/dof.rs` | 588 | `graph/dof_tests.rs` | 3 |
| `graph/pattern.rs` | 405 | `graph/pattern_tests.rs` | 3 |
| `graph/redundancy.rs` | 653 | `graph/redundancy_tests.rs` | 3 |
| `solve/closed_form.rs` | 898 | `solve/closed_form_tests.rs` | 3 |
| `solve/branch.rs` | 306 | `solve/branch_tests.rs` | 3 |
| `solve/drag.rs` | 367 | `solve/drag_tests.rs`, `tests/v3_drag_integration.rs` | 2, 3 |
| `solve/sub_problem.rs` | 514 | `solve/sub_problem_tests.rs` | 3 |
| `reduce/substitute.rs` | 293 | `reduce/reduce_tests.rs` | 3 |
| `reduce/eliminate.rs` | 366 | `reduce/reduce_tests.rs` | 3 |
| `reduce/merge.rs` | 374 | `reduce/reduce_tests.rs` | 3 |
| `dataflow/tracker.rs` | 430 | Extend `system.rs` inline tests | 3 |
| `dataflow/cache.rs` | 252 | Extend `system.rs` inline tests | 3 |
| `param/store.rs` | 372 | `param/` inline tests | 3 |
| `id.rs` | 148 | (covered by param + system tests) | 1 |
| `entity/mod.rs` | 53 | (trait, covered by sketch tests) | 1 |
| `constraint/mod.rs` | 68 | (trait, covered by sketch tests) | 1 |
