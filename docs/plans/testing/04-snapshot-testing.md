# Plan 04: Snapshot Testing

## Goal

Snapshot testing captures the exact output of a function and saves it to a file. Future
test runs compare against the saved snapshot. Changes require explicit review and
approval. This catches **behavioral regressions** that tolerance-based assertions miss.

For solverang, snapshot testing serves three purposes:

1. **Numerical regression detection**: A solver change that shifts a converged solution
   by even 1e-15 will cause a snapshot mismatch, prompting review. This is critical
   because the crate has two solver paths (legacy `Problem` trait and V3 pipeline) that
   must produce consistent results.

2. **Diagnostic output stability**: The V3 `SystemResult`, `DofAnalysis`,
   `RedundancyAnalysis`, and `MatchedPattern` outputs are consumed by downstream CAD
   applications. Structural changes to these outputs (field reordering, renamed variants)
   are breaking API changes that snapshot tests will catch.

3. **Cross-architecture consistency**: When both legacy and V3 solvers solve the same
   geometric problem, their results should agree within tolerance. Snapshots lock in
   both results so divergence is immediately visible.

Specific regression classes that snapshots catch:
- A refactor that changes convergence from 5 iterations to 50 (but still meets tolerance)
- A dependency update (`nalgebra`, `levenberg-marquardt`) that silently changes behavior
- JIT codegen changes that produce subtly different floating-point results
- Pipeline phase changes that alter cluster decomposition or solve order
- Reduce pass changes that eliminate different constraint sets
- Pattern matcher changes that match different closed-form templates

## Tool: `insta`

**[insta](https://insta.rs/)** is the standard Rust snapshot testing crate.

```bash
cargo install cargo-insta
```

### Setup

Add to `crates/solverang/Cargo.toml`:

```toml
[dev-dependencies]
insta = { version = "1.38", features = ["yaml", "redactions"] }
serde = { version = "1", features = ["derive"] }  # If not already present
```

### Snapshot directory structure

`insta` automatically creates snapshot files next to the test file:

```
crates/solverang/tests/
├── v3_snapshot_tests.rs
├── closed_form_snapshots.rs
├── graph_snapshots.rs
├── reduce_snapshots.rs
├── sketch2d_snapshots.rs
├── pattern_snapshots.rs
├── nist_snapshots.rs
├── cross_architecture_snapshots.rs
└── snapshots/
    ├── v3_snapshot_tests__triangle_solve.snap
    ├── v3_snapshot_tests__multi_cluster_solve.snap
    ├── closed_form_snapshots__circle_circle_intersection.snap
    ├── graph_snapshots__two_clusters.snap
    ├── graph_snapshots__dof_free_point.snap
    ├── reduce_snapshots__reduce_substitute.snap
    ├── sketch2d_snapshots__sketch2d_rectangle.snap
    ├── pattern_snapshots__pattern_two_distances.snap
    ├── nist_snapshots__rosenbrock_nr.snap
    └── ... (70+ total snapshot files)
```

### `.gitignore` update

```
# Pending snapshots (not yet reviewed)
*.snap.new
```

Committed `.snap` files are version-controlled -- they ARE the expected output.

## Snapshot Categories

### Category 1: V3 Pipeline End-to-End Results

Snapshot the full `SystemResult` from `ConstraintSystem::solve()`, including per-cluster
results and diagnostics. These are the most important snapshots because they exercise the
entire 5-phase pipeline (`Decompose -> Analyze -> Reduce -> Solve -> PostProcess`).

#### 1a. Simple Well-Constrained Sketch

```rust
use insta::{assert_yaml_snapshot, Settings};

/// Snapshot a triangle defined by 3 fixed distances.
#[test]
fn snapshot_triangle_solve() {
    let mut builder = Sketch2DBuilder::new();
    let p0 = builder.add_fixed_point(0.0, 0.0);
    let p1 = builder.add_fixed_point(10.0, 0.0);
    let p2 = builder.add_point(5.0, 1.0); // Initial guess

    builder.constrain_distance(p0, p1, 10.0);
    builder.constrain_distance(p1, p2, 8.0);
    builder.constrain_distance(p2, p0, 6.0);

    let mut system = builder.build();
    let result = system.solve();

    let mut settings = Settings::clone_current();
    settings.add_redaction(".duration", "[DURATION]");
    settings.add_redaction(".clusters[].iterations", "[ITERS]");
    settings.bind(|| {
        assert_yaml_snapshot!("triangle_solve", result);
    });
}
```

Expected snapshot (`snapshots/v3_snapshot_tests__triangle_solve.snap`):

```yaml
---
source: tests/v3_snapshot_tests.rs
expression: result
---
status: Solved
clusters:
  - status: Converged
    iterations: "[ITERS]"
    residual_norm: 0.0
    param_values:
      - id: "ParamId(4, 0)"
        value: 3.6
      - id: "ParamId(5, 0)"
        value: 4.8
total_iterations: "[ITERS]"
duration: "[DURATION]"
```

#### 1b. Over-Constrained System (Redundant Constraints)

```rust
#[test]
fn snapshot_overconstrained_system() {
    let mut builder = Sketch2DBuilder::new();
    let p0 = builder.add_fixed_point(0.0, 0.0);
    let p1 = builder.add_point(5.0, 0.0);

    // Two distance constraints that agree (redundant)
    builder.constrain_distance(p0, p1, 5.0);
    builder.constrain_distance(p0, p1, 5.0);
    builder.constrain_horizontal(p0, p1);

    let mut system = builder.build();
    let result = system.solve();

    let mut settings = Settings::clone_current();
    settings.add_redaction(".duration", "[DURATION]");
    settings.bind(|| {
        assert_yaml_snapshot!("overconstrained_system", result);
    });
}
```

#### 1c. Multi-Cluster System

```rust
#[test]
fn snapshot_multi_cluster_solve() {
    let mut builder = Sketch2DBuilder::new();

    // Cluster 1: two connected points
    let p0 = builder.add_fixed_point(0.0, 0.0);
    let p1 = builder.add_point(1.0, 0.0);
    builder.constrain_distance(p0, p1, 5.0);
    builder.constrain_horizontal(p0, p1);

    // Cluster 2: independent pair
    let p2 = builder.add_fixed_point(100.0, 100.0);
    let p3 = builder.add_point(101.0, 100.0);
    builder.constrain_distance(p2, p3, 3.0);
    builder.constrain_vertical(p2, p3);

    let mut system = builder.build();
    let result = system.solve();

    let mut settings = Settings::clone_current();
    settings.add_redaction(".duration", "[DURATION]");
    settings.add_redaction(".clusters[].iterations", "[ITERS]");
    settings.bind(|| {
        assert_yaml_snapshot!("multi_cluster_solve", result);
    });
}
```

#### 1d. Partially Solved System

```rust
#[test]
fn snapshot_partially_solved() {
    let mut builder = Sketch2DBuilder::new();

    // Cluster 1: solvable
    let p0 = builder.add_fixed_point(0.0, 0.0);
    let p1 = builder.add_point(1.0, 0.0);
    builder.constrain_distance(p0, p1, 5.0);
    builder.constrain_horizontal(p0, p1);

    // Cluster 2: conflicting
    let p2 = builder.add_point(0.0, 0.0);
    builder.constrain_fixed(p2, 5.0, 5.0);
    builder.constrain_fixed(p2, 10.0, 10.0);

    let mut system = builder.build();
    let result = system.solve();

    let mut settings = Settings::clone_current();
    settings.add_redaction(".duration", "[DURATION]");
    settings.add_redaction(".clusters[].iterations", "[ITERS]");
    settings.bind(|| {
        assert_yaml_snapshot!("partially_solved", result);
    });
}
```

#### 1e. Empty System

```rust
#[test]
fn snapshot_empty_system() {
    let system = ConstraintSystem::new();
    let result = system.solve();

    let mut settings = Settings::clone_current();
    settings.add_redaction(".duration", "[DURATION]");
    settings.bind(|| {
        assert_yaml_snapshot!("empty_system", result);
    });
}
```

#### 1f. Incremental Solve (Value Change Only)

```rust
#[test]
fn snapshot_incremental_solve() {
    let mut builder = Sketch2DBuilder::new();
    let p0 = builder.add_fixed_point(0.0, 0.0);
    let p1 = builder.add_point(5.0, 0.0);
    builder.constrain_distance(p0, p1, 5.0);
    builder.constrain_horizontal(p0, p1);

    let mut system = builder.build();
    let _ = system.solve(); // Initial solve

    // Perturb p1.x slightly (value change, not structural)
    system.set_param(/* p1.x ParamId */, 4.5);
    let result = system.solve_incremental();

    let mut settings = Settings::clone_current();
    settings.add_redaction(".duration", "[DURATION]");
    settings.add_redaction(".clusters[].iterations", "[ITERS]");
    settings.bind(|| {
        assert_yaml_snapshot!("incremental_solve", result);
    });
}
```

### Category 2: Closed-Form Solver Results

Snapshot the exact solutions produced by closed-form solvers for known configurations.
These are critical because closed-form solvers bypass the iterative pipeline and must
produce mathematically exact results.

#### 2a. Circle-Circle Intersection

```rust
#[test]
fn snapshot_circle_circle_intersection() {
    // Two circles: center (0,0) r=5, center (6,0) r=5
    // Exact intersections: (3, 4) and (3, -4)
    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let px = store.alloc(1.0, owner);  // Initial guess x
    let py = store.alloc(1.0, owner);  // Initial guess y

    // Build TwoDistances pattern with known intersection
    // (distance to (0,0)=5 and distance to (6,0)=5)

    let pattern = MatchedPattern {
        kind: PatternKind::TwoDistances,
        entity_ids: vec![owner],
        constraint_indices: vec![0, 1],
        param_ids: vec![px, py],
    };

    let result = solve_pattern(&pattern, &constraints, &store);
    assert_yaml_snapshot!("circle_circle_intersection", result);
}
```

#### 2b. Tangent Circles (Single Intersection)

```rust
#[test]
fn snapshot_tangent_circles() {
    // Two circles tangent externally: center (0,0) r=3, center (5,0) r=2
    // Single intersection at (3, 0)
    let result = solve_tangent_circles_case();
    assert_yaml_snapshot!("tangent_circles", result);
}
```

#### 2c. Non-Intersecting Circles (No Solution)

```rust
#[test]
fn snapshot_non_intersecting_circles() {
    // Two circles too far apart: center (0,0) r=1, center (10,0) r=1
    let result = solve_non_intersecting_case();
    assert_yaml_snapshot!("non_intersecting_circles", result);
}
```

#### 2d. Scalar Newton Solve

```rust
#[test]
fn snapshot_scalar_newton() {
    // x^2 - 2 = 0 starting at x = 1.5
    // Should converge to sqrt(2) = 1.41421356...
    let result = solve_scalar_case(1.5, |x| x * x - 2.0, |x| 2.0 * x);
    assert_yaml_snapshot!("scalar_newton_sqrt2", result);
}
```

#### 2e. Polar-to-Cartesian (DistanceAngle)

```rust
#[test]
fn snapshot_distance_angle_solve() {
    // Point at distance 5 from origin, angle pi/4
    // Expected: (5/sqrt(2), 5/sqrt(2)) = (3.5355..., 3.5355...)
    let result = solve_distance_angle_case(5.0, std::f64::consts::FRAC_PI_4);
    assert_yaml_snapshot!("distance_angle_pi4", result);
}
```

#### 2f. Horizontal/Vertical Direct Assignment

```rust
#[test]
fn snapshot_horizontal_vertical_solve() {
    // Point constrained to match another point's x (horizontal) and y (vertical)
    let result = solve_horizontal_vertical_case(3.0, 7.0);
    assert_yaml_snapshot!("horizontal_vertical_solve", result);
}
```

### Category 3: Graph Decomposition Snapshots

Snapshot cluster assignments for known constraint topologies. This catches regressions
in the bipartite graph construction and cluster decomposition algorithms.

#### 3a. Single Connected Component

```rust
#[test]
fn snapshot_single_cluster_graph() {
    // All entities connected in a chain: p0-p1-p2
    let mut system = ConstraintSystem::new();
    let p0 = add_test_point(&mut system, 0.0, 0.0);
    let p1 = add_test_point(&mut system, 1.0, 0.0);
    let p2 = add_test_point(&mut system, 2.0, 0.0);

    add_distance_constraint(&mut system, p0, p1, 1.0);
    add_distance_constraint(&mut system, p1, p2, 1.0);

    let clusters = system.decompose_clusters();
    assert_yaml_snapshot!("single_cluster_chain", clusters);
}
```

#### 3b. Two Disconnected Clusters

```rust
#[test]
fn snapshot_two_clusters() {
    let mut system = ConstraintSystem::new();
    // Cluster A: p0-p1
    let p0 = add_test_point(&mut system, 0.0, 0.0);
    let p1 = add_test_point(&mut system, 1.0, 0.0);
    add_distance_constraint(&mut system, p0, p1, 5.0);

    // Cluster B: p2-p3 (disconnected from A)
    let p2 = add_test_point(&mut system, 10.0, 10.0);
    let p3 = add_test_point(&mut system, 11.0, 10.0);
    add_distance_constraint(&mut system, p2, p3, 3.0);

    let clusters = system.decompose_clusters();
    assert_yaml_snapshot!("two_clusters", clusters);
}
```

#### 3c. Star Topology (Hub Entity)

```rust
#[test]
fn snapshot_star_topology() {
    let mut system = ConstraintSystem::new();
    let hub = add_test_point(&mut system, 0.0, 0.0);
    let spokes: Vec<_> = (0..5).map(|i| {
        add_test_point(&mut system, (i + 1) as f64, 0.0)
    }).collect();

    for &spoke in &spokes {
        add_distance_constraint(&mut system, hub, spoke, 1.0);
    }

    let clusters = system.decompose_clusters();
    assert_yaml_snapshot!("star_topology", clusters);
}
```

#### 3d. Isolated Entities (No Constraints)

```rust
#[test]
fn snapshot_isolated_entities() {
    let mut system = ConstraintSystem::new();
    for i in 0..5 {
        add_test_point(&mut system, i as f64, 0.0);
    }
    // No constraints -- each entity is its own cluster (or no clusters at all)
    let clusters = system.decompose_clusters();
    assert_yaml_snapshot!("isolated_entities", clusters);
}
```

### Category 4: DOF Analysis Snapshots

Snapshot per-entity DOF analysis for standard sketches. These catch regressions in the
SVD-based rank computation (`graph/dof.rs`).

#### 4a. Free Point (2 DOF)

```rust
#[test]
fn snapshot_dof_free_point() {
    let mut system = ConstraintSystem::new();
    let (_eid, _px, _py) = add_test_point(&mut system, 1.0, 2.0);

    let dof = system.analyze_dof();
    assert_yaml_snapshot!("dof_free_point", dof);
}
```

Expected:

```yaml
---
entities:
  - entity_id: "EntityId(0, 0)"
    total_params: 2
    fixed_params: 0
    dof: 2
total_dof: 2
total_free_params: 2
total_equations: 0
```

#### 4b. Well-Constrained Point (0 DOF)

```rust
#[test]
fn snapshot_dof_fixed_point() {
    let mut system = ConstraintSystem::new();
    let (eid, px, py) = add_test_point(&mut system, 1.0, 2.0);

    let cid1 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid1, entity_ids: vec![eid], param: px, target: 5.0,
    }));
    let cid2 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid2, entity_ids: vec![eid], param: py, target: 3.0,
    }));

    let dof = system.analyze_dof();
    assert_yaml_snapshot!("dof_well_constrained_point", dof);
}
```

#### 4c. Over-Constrained System (Negative DOF)

```rust
#[test]
fn snapshot_dof_over_constrained() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

    // 3 constraints on 1 parameter
    for target in [1.0, 2.0, 3.0] {
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid, entity_ids: vec![eid], param: px, target,
        }));
    }

    let dof = system.analyze_dof();
    assert_yaml_snapshot!("dof_over_constrained", dof);
}
```

#### 4d. Full Sketch DOF

```rust
#[test]
fn snapshot_dof_full_sketch() {
    let mut builder = Sketch2DBuilder::new();
    let p0 = builder.add_fixed_point(0.0, 0.0);
    let p1 = builder.add_point(10.0, 0.0);
    let p2 = builder.add_point(5.0, 8.0);

    builder.constrain_distance(p0, p1, 10.0);
    builder.constrain_distance(p1, p2, 8.0);
    builder.constrain_distance(p2, p0, 6.0);
    builder.constrain_horizontal(p0, p1);

    let system = builder.build();
    let dof = system.analyze_dof();
    assert_yaml_snapshot!("dof_full_sketch", dof);
}
```

### Category 5: Redundancy Analysis Snapshots

Snapshot known redundant and conflicting constraint sets to catch regressions in the
incremental rank test and null-space projection (`graph/redundancy.rs`).

#### 5a. Clean System (No Redundancy)

```rust
#[test]
fn snapshot_redundancy_clean() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);
    let cid = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid, entity_ids: vec![eid], param: px, target: 5.0,
    }));

    let result = system.analyze_redundancy();
    assert_yaml_snapshot!("redundancy_clean", result);
}
```

#### 5b. Redundant Constraints

```rust
#[test]
fn snapshot_redundancy_redundant() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);

    // Same constraint twice (redundant, not conflicting)
    for _ in 0..2 {
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid, entity_ids: vec![eid], param: px, target: 5.0,
        }));
    }

    let result = system.analyze_redundancy();
    assert_yaml_snapshot!("redundancy_redundant", result);
}
```

#### 5c. Conflicting Constraints

```rust
#[test]
fn snapshot_redundancy_conflicting() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);

    let cid1 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid1, entity_ids: vec![eid], param: px, target: 5.0,
    }));
    let cid2 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid2, entity_ids: vec![eid], param: px, target: 10.0,
    }));

    let result = system.analyze_redundancy();
    assert_yaml_snapshot!("redundancy_conflicting", result);
}
```

### Category 6: Reduce Pass Snapshots

Snapshot the before/after state of symbolic reduction passes to catch regressions
in the parameter elimination, merging, and constraint simplification logic
(`reduce/substitute.rs`, `reduce/merge.rs`, `reduce/eliminate.rs`).

#### 6a. Fixed-Parameter Substitution

```rust
use serde::Serialize;

#[derive(Serialize)]
struct ReduceSnapshot {
    before_param_count: usize,
    after_param_count: usize,
    before_constraint_count: usize,
    after_constraint_count: usize,
    eliminated_params: Vec<String>,
    eliminated_constraints: Vec<String>,
}

#[test]
fn snapshot_reduce_substitute() {
    let mut system = build_test_system_with_fixed_params();
    let before_params = system.param_count();
    let before_constraints = system.constraint_count();

    let result = system.run_reduce_pass();

    let snapshot = ReduceSnapshot {
        before_param_count: before_params,
        after_param_count: result.reduced_param_count,
        before_constraint_count: before_constraints,
        after_constraint_count: result.remaining_constraints,
        eliminated_params: result.substituted_params.iter().map(|p| format!("{:?}", p)).collect(),
        eliminated_constraints: result.eliminated_constraints.iter().map(|c| format!("{:?}", c)).collect(),
    };
    assert_yaml_snapshot!("reduce_substitute", snapshot);
}
```

#### 6b. Coincident Merge

```rust
#[test]
fn snapshot_reduce_merge() {
    let mut system = build_test_system_with_coincident_points();
    let before_params = system.param_count();

    let result = system.run_reduce_pass();

    let snapshot = ReduceSnapshot {
        before_param_count: before_params,
        after_param_count: result.reduced_param_count,
        // ...
    };
    assert_yaml_snapshot!("reduce_merge", snapshot);
}
```

#### 6c. Trivial Elimination

```rust
#[test]
fn snapshot_reduce_eliminate() {
    let mut system = build_test_system_with_trivial_constraint();
    let before_constraints = system.constraint_count();

    let result = system.run_reduce_pass();

    let snapshot = ReduceSnapshot {
        before_constraint_count: before_constraints,
        after_constraint_count: result.remaining_constraints,
        // ...
    };
    assert_yaml_snapshot!("reduce_eliminate", snapshot);
}
```

### Category 7: Sketch2D End-to-End Snapshots

Build complete sketches via `Sketch2DBuilder`, solve, and snapshot the final geometry.
These are integration-level snapshots that exercise the full V3 stack from high-level
API to final parameter values.

#### 7a. Rectangle

```rust
#[derive(Serialize)]
struct GeometrySnapshot {
    points: Vec<(String, f64, f64)>,
    status: String,
}

#[test]
fn snapshot_sketch2d_rectangle() {
    let mut builder = Sketch2DBuilder::new();
    let p0 = builder.add_fixed_point(0.0, 0.0);
    let p1 = builder.add_point(10.0, 0.0);
    let p2 = builder.add_point(10.0, 5.0);
    let p3 = builder.add_point(0.0, 5.0);

    let l0 = builder.add_line(p0, p1);
    let l1 = builder.add_line(p1, p2);
    let l2 = builder.add_line(p2, p3);
    let l3 = builder.add_line(p3, p0);

    builder.constrain_horizontal(p0, p1);
    builder.constrain_vertical(p1, p2);
    builder.constrain_horizontal(p2, p3);
    builder.constrain_vertical(p3, p0);
    builder.constrain_distance(p0, p1, 10.0);
    builder.constrain_distance(p1, p2, 5.0);

    let mut system = builder.build();
    let result = system.solve();

    let geometry = system.extract_point_coordinates();

    let mut settings = Settings::clone_current();
    settings.add_redaction(".solve_result.duration", "[DURATION]");
    settings.add_redaction(".solve_result.clusters[].iterations", "[ITERS]");
    settings.bind(|| {
        assert_yaml_snapshot!("sketch2d_rectangle", {
            "solve_status" => format!("{:?}", result.status),
            "geometry" => &geometry,
        });
    });
}
```

#### 7b. Circle with Tangent Lines

```rust
#[test]
fn snapshot_sketch2d_circle_tangent() {
    let mut builder = Sketch2DBuilder::new();
    let center = builder.add_point(0.0, 0.0);
    let circle = builder.add_circle(center, 5.0);

    let lp0 = builder.add_point(-10.0, 5.0);
    let lp1 = builder.add_point(10.0, 5.0);
    let line = builder.add_line(lp0, lp1);

    builder.constrain_fixed(center, 0.0, 0.0);
    builder.constrain_tangent_line_circle(line, circle);

    let mut system = builder.build();
    let result = system.solve();

    let mut settings = Settings::clone_current();
    settings.add_redaction(".duration", "[DURATION]");
    settings.bind(|| {
        assert_yaml_snapshot!("sketch2d_circle_tangent", result);
    });
}
```

#### 7c. Symmetric Sketch

```rust
#[test]
fn snapshot_sketch2d_symmetric() {
    let mut builder = Sketch2DBuilder::new();
    let p0 = builder.add_point(3.0, 4.0);
    let p1 = builder.add_point(-2.0, 5.0);  // Initial guess (wrong side)

    let axis_p0 = builder.add_fixed_point(0.0, 0.0);
    let axis_p1 = builder.add_fixed_point(0.0, 10.0);
    let axis = builder.add_line(axis_p0, axis_p1);

    builder.constrain_symmetric(p0, p1, axis);
    builder.constrain_fixed(p0, 3.0, 4.0);

    let mut system = builder.build();
    let result = system.solve();
    // p1 should end up at (-3.0, 4.0)

    let mut settings = Settings::clone_current();
    settings.add_redaction(".duration", "[DURATION]");
    settings.bind(|| {
        assert_yaml_snapshot!("sketch2d_symmetric", result);
    });
}
```

#### 7d. Equal Length Lines

```rust
#[test]
fn snapshot_sketch2d_equal_length() {
    let mut builder = Sketch2DBuilder::new();
    let p0 = builder.add_fixed_point(0.0, 0.0);
    let p1 = builder.add_point(5.0, 0.0);
    let p2 = builder.add_fixed_point(10.0, 0.0);
    let p3 = builder.add_point(13.0, 4.0);

    let l0 = builder.add_line(p0, p1);
    let l1 = builder.add_line(p2, p3);

    builder.constrain_distance(p0, p1, 5.0);
    builder.constrain_equal_length(l0, l1);
    builder.constrain_horizontal(p0, p1);

    let mut system = builder.build();
    let result = system.solve();

    let mut settings = Settings::clone_current();
    settings.add_redaction(".duration", "[DURATION]");
    settings.bind(|| {
        assert_yaml_snapshot!("sketch2d_equal_length", result);
    });
}
```

#### 7e. All 16 Constraint Types

```rust
/// Comprehensive sketch exercising every sketch2d constraint type.
#[test]
fn snapshot_sketch2d_all_constraints() {
    let mut builder = Sketch2DBuilder::new();
    // Build a complex sketch using:
    // DistancePtPt, DistancePtLine, Coincident, Fixed, Horizontal, Vertical,
    // Parallel, Perpendicular, Angle, Midpoint, Symmetric, EqualLength,
    // PointOnCircle, TangentLineCircle, TangentCircleCircle
    // ...

    let mut system = builder.build();
    let result = system.solve();

    let mut settings = Settings::clone_current();
    settings.add_redaction(".duration", "[DURATION]");
    settings.add_redaction(".clusters[].iterations", "[ITERS]");
    settings.bind(|| {
        assert_yaml_snapshot!("sketch2d_all_constraints", result);
    });
}
```

### Category 8: Pattern Matching Snapshots

Snapshot which closed-form patterns are detected for known constraint subgraphs.
This catches regressions in the pattern classification logic (`graph/pattern.rs`).

```rust
#[test]
fn snapshot_pattern_two_distances() {
    let mut system = ConstraintSystem::new();
    let p0 = add_test_point(&mut system, 0.0, 0.0);
    let p1 = add_test_point(&mut system, 5.0, 0.0);
    let p2 = add_test_point(&mut system, 3.0, 1.0); // Free point

    add_distance_constraint(&mut system, p0, p2, 5.0);
    add_distance_constraint(&mut system, p1, p2, 4.0);
    fix_point(&mut system, p0);
    fix_point(&mut system, p1);

    let patterns = system.detect_patterns();
    assert_yaml_snapshot!("pattern_two_distances", patterns);
}

#[test]
fn snapshot_pattern_horizontal_vertical() {
    let mut system = ConstraintSystem::new();
    let p0 = add_test_point(&mut system, 0.0, 0.0);
    let p1 = add_test_point(&mut system, 1.0, 1.0);

    add_horizontal_constraint(&mut system, p0, p1);
    add_vertical_constraint(&mut system, p0, p1);
    fix_point(&mut system, p0);

    let patterns = system.detect_patterns();
    assert_yaml_snapshot!("pattern_horizontal_vertical", patterns);
}

#[test]
fn snapshot_pattern_scalar_solve() {
    let mut system = ConstraintSystem::new();
    let p0 = add_test_point(&mut system, 0.0, 0.0);
    let p1 = add_test_point(&mut system, 1.0, 0.0);

    add_horizontal_constraint(&mut system, p0, p1);
    fix_point(&mut system, p0);

    let patterns = system.detect_patterns();
    assert_yaml_snapshot!("pattern_scalar_solve", patterns);
}

#[test]
fn snapshot_pattern_no_match() {
    // Dense triangle: no simple closed-form pattern
    let mut system = ConstraintSystem::new();
    let p0 = add_test_point(&mut system, 0.0, 0.0);
    let p1 = add_test_point(&mut system, 1.0, 0.0);
    let p2 = add_test_point(&mut system, 0.5, 1.0);

    add_distance_constraint(&mut system, p0, p1, 1.0);
    add_distance_constraint(&mut system, p1, p2, 1.0);
    add_distance_constraint(&mut system, p2, p0, 1.0);

    let patterns = system.detect_patterns();
    assert_yaml_snapshot!("pattern_no_match", patterns);
}
```

### Category 9: Legacy NIST/MINPACK Reference Snapshots

Snapshot the solutions to all 18 MINPACK test problems and 14 nonlinear equation
test problems. These serve as a regression baseline for the legacy solver path.

```rust
use serde::Serialize;
use solverang::test_problems;
use solverang::{LMSolver, LMConfig, Solver, SolverConfig, SolveResult};

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
    if !v.is_finite() || v == 0.0 { return v; }
    let magnitude = v.abs().log10().floor() as i32;
    let factor = 10f64.powi(digits as i32 - 1 - magnitude);
    (v * factor).round() / factor
}

macro_rules! nist_snapshot {
    ($name:ident, $problem:expr, $solver:expr) => {
        #[test]
        fn $name() {
            let problem = $problem;
            let x0 = problem.initial_point(1.0);
            let result = $solver.solve(&problem, &x0);
            let snapshot = SolverSnapshot::from_result(stringify!($name), &result);

            let mut settings = insta::Settings::clone_current();
            settings.set_snapshot_suffix(stringify!($name));
            settings.add_redaction(".iterations", "[ITERS]");
            settings.bind(|| {
                insta::assert_yaml_snapshot!(snapshot);
            });
        }
    };
}

nist_snapshot!(rosenbrock_nr, test_problems::Rosenbrock::new(),
    Solver::new(SolverConfig::default()));
nist_snapshot!(rosenbrock_lm, test_problems::Rosenbrock::new(),
    LMSolver::new(LMConfig::default()));
nist_snapshot!(powell_singular_nr, test_problems::PowellSingular::new(),
    Solver::new(SolverConfig::default()));
nist_snapshot!(powell_singular_lm, test_problems::PowellSingular::new(),
    LMSolver::new(LMConfig::default()));
// ... repeat for all 18 MINPACK + 14 nonlinear test problems
```

### Category 10: Cross-Architecture Comparison

For problems that can be expressed in both legacy and V3 APIs, snapshot both results
and verify they agree. This is the most important category for migration confidence.

```rust
#[test]
fn snapshot_cross_architecture_triangle() {
    // Legacy
    let mut legacy_system = geometry::ConstraintSystemBuilder::<2>::new()
        .name("triangle-legacy")
        .point(Point2D::new(0.0, 0.0))
        .point(Point2D::new(10.0, 0.0))
        .point(Point2D::new(5.0, 1.0))
        .fix(0)
        .fix(1)
        .distance(0, 1, 10.0)
        .distance(1, 2, 8.0)
        .distance(2, 0, 6.0)
        .build();

    let legacy_result = LMSolver::new(LMConfig::default())
        .solve(&legacy_system, &legacy_system.current_values());

    // V3
    let mut builder = Sketch2DBuilder::new();
    let p0 = builder.add_fixed_point(0.0, 0.0);
    let p1 = builder.add_fixed_point(10.0, 0.0);
    let p2 = builder.add_point(5.0, 1.0);
    builder.constrain_distance(p0, p1, 10.0);
    builder.constrain_distance(p1, p2, 8.0);
    builder.constrain_distance(p2, p0, 6.0);
    let mut v3_system = builder.build();
    let v3_result = v3_system.solve();

    // Extract the free-point solution from both
    let legacy_solution = match &legacy_result {
        SolveResult::Converged { solution, .. } => solution.clone(),
        _ => panic!("Legacy solver did not converge"),
    };

    let v3_p2_x = v3_system.get_param(/* p2.x ParamId */);
    let v3_p2_y = v3_system.get_param(/* p2.y ParamId */);

    // Snapshot both for comparison
    assert_yaml_snapshot!("cross_arch_triangle", {
        "legacy_p2" => format!("({:.10}, {:.10})", legacy_solution[4], legacy_solution[5]),
        "v3_p2" => format!("({:.10}, {:.10})", v3_p2_x, v3_p2_y),
    });
}

#[test]
fn snapshot_cross_architecture_rectangle() {
    // Build the same rectangle problem in both APIs and compare
    // ...
    assert_yaml_snapshot!("cross_arch_rectangle", comparison);
}

#[test]
fn snapshot_cross_architecture_circle_tangent() {
    // Build the same circle-tangent problem in both APIs and compare
    // ...
    assert_yaml_snapshot!("cross_arch_circle_tangent", comparison);
}
```

### Category 11: Sketch3D and Assembly Snapshots

Snapshot 3D geometry and assembly results to catch regressions in the 3D constraint
formulations and quaternion-based rigid body math.

```rust
#[test]
fn snapshot_sketch3d_coplanar() {
    // 4 points constrained to a plane: use PointOnPlane constraints
    // Snapshot final 3D coordinates
    let mut system = build_coplanar_test_system();
    let result = system.solve();

    let mut settings = Settings::clone_current();
    settings.add_redaction(".duration", "[DURATION]");
    settings.bind(|| {
        assert_yaml_snapshot!("sketch3d_coplanar", result);
    });
}

#[test]
fn snapshot_sketch3d_perpendicular_lines() {
    // Two line segments constrained to be perpendicular in 3D
    let mut system = build_perpendicular_3d_system();
    let result = system.solve();
    assert_yaml_snapshot!("sketch3d_perpendicular", result);
}

#[test]
fn snapshot_assembly_mate() {
    // Two rigid bodies connected by a Mate constraint
    // Snapshot includes final positions and quaternion orientations
    let mut system = build_mate_assembly();
    let result = system.solve();
    assert_yaml_snapshot!("assembly_mate", result);
}

#[test]
fn snapshot_assembly_coaxial() {
    // Two rigid bodies with a CoaxialAssembly constraint
    let mut system = build_coaxial_assembly();
    let result = system.solve();
    assert_yaml_snapshot!("assembly_coaxial", result);
}

#[test]
fn snapshot_assembly_gear() {
    // Two rigid bodies with a Gear constraint (ratio 2:1)
    let mut system = build_gear_assembly(2.0);
    let result = system.solve();
    assert_yaml_snapshot!("assembly_gear_2to1", result);
}

#[test]
fn snapshot_assembly_insert() {
    // Insert constraint (coaxial + mate combination)
    let mut system = build_insert_assembly();
    let result = system.solve();
    assert_yaml_snapshot!("assembly_insert", result);
}
```

### Category 12: JIT vs Interpreted Comparison (Legacy)

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

### Category 13: Legacy Decomposition Snapshots

```rust
#[derive(Serialize)]
struct DecompositionSnapshot {
    num_components: usize,
    component_sizes: Vec<usize>,
    variable_assignments: Vec<usize>,
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

## Redaction Strategy

Non-deterministic fields must be redacted to prevent flaky snapshots:

| Field | Why Non-Deterministic | Redaction |
|-------|----------------------|-----------|
| `duration` | Wall-clock timing | `"[DURATION]"` |
| `iterations` | May vary by platform/BLAS | `"[ITERS]"` |
| `total_iterations` | Sum of above | `"[ITERS]"` |

Fields that **must NOT** be redacted (changes here indicate real regressions):
- `status` (`Solved`, `PartiallySolved`, `DiagnosticFailure`)
- `residual_norm` (should be deterministic to machine epsilon)
- All parameter values (the actual solution)
- All DOF numbers
- All redundancy analysis results
- All pattern match results
- All cluster assignments

## Snapshot Serialization: Handling Floats

### Cross-platform float reproducibility

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

Use 10 significant digits -- enough to detect real changes but tolerant of platform
differences in the least significant bits.

## Workflow

### Development cycle

```bash
# Run tests -- new snapshots create .snap.new files
cargo test -p solverang

# Review pending snapshots interactively
cargo insta review

# Accept all pending snapshots (use with caution)
cargo insta accept

# Reject all pending snapshots
cargo insta reject
```

### When a snapshot changes

**Intentional change** (algorithm improvement, dependency update):
1. Review the diff in `cargo insta review`
2. Verify the new behavior is correct
3. Accept: `cargo insta accept`
4. Commit the updated `.snap` files

**Unintentional change** (regression):
1. Investigate why the output changed
2. Fix the bug
3. Reject: `cargo insta reject`

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

## CI Integration

```yaml
# In .github/workflows/ci.yml
snapshot-tests:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo install cargo-insta
    - name: Run snapshot tests
      run: cargo insta test -p solverang --features geometry,parallel,sparse
      env:
        INSTA_UPDATE: no  # Fail on mismatches instead of creating .snap.new
    - name: Fail if pending snapshots
      run: |
        if find . -name '*.snap.new' | grep -q .; then
          echo "Pending snapshot changes detected! Run 'cargo insta review' locally."
          find . -name '*.snap.new' -exec echo "  {}" \;
          exit 1
        fi
```

In CI, `INSTA_UPDATE=no` makes snapshot mismatches a hard failure. Developers must
run `cargo insta review` locally and commit updated snapshots.

## File Organization

```
crates/solverang/tests/
├── v3_snapshot_tests.rs             # V3 pipeline end-to-end snapshots (Cat 1)
├── closed_form_snapshots.rs         # Closed-form solver snapshots (Cat 2)
├── graph_snapshots.rs               # Decomposition, DOF, redundancy (Cat 3-5)
├── reduce_snapshots.rs              # Reduce pass before/after (Cat 6)
├── sketch2d_snapshots.rs            # Sketch2D geometry snapshots (Cat 7)
├── pattern_snapshots.rs             # Pattern matching snapshots (Cat 8)
├── nist_snapshots.rs                # NIST/MINPACK reference snapshots (Cat 9)
├── cross_architecture_snapshots.rs  # Legacy vs V3 comparison (Cat 10)
├── sketch3d_assembly_snapshots.rs   # Sketch3D + Assembly snapshots (Cat 11)
├── jit_snapshots.rs                 # JIT vs interpreted comparison (Cat 12)
├── decomposition_snapshots.rs       # Legacy decomposition snapshots (Cat 13)
└── snapshots/
    ├── v3_snapshot_tests__triangle_solve.snap
    ├── v3_snapshot_tests__overconstrained_system.snap
    ├── v3_snapshot_tests__multi_cluster_solve.snap
    ├── v3_snapshot_tests__partially_solved.snap
    ├── v3_snapshot_tests__empty_system.snap
    ├── v3_snapshot_tests__incremental_solve.snap
    ├── closed_form_snapshots__circle_circle_intersection.snap
    ├── closed_form_snapshots__tangent_circles.snap
    ├── closed_form_snapshots__non_intersecting_circles.snap
    ├── closed_form_snapshots__scalar_newton_sqrt2.snap
    ├── closed_form_snapshots__distance_angle_pi4.snap
    ├── closed_form_snapshots__horizontal_vertical_solve.snap
    ├── graph_snapshots__single_cluster_chain.snap
    ├── graph_snapshots__two_clusters.snap
    ├── graph_snapshots__star_topology.snap
    ├── graph_snapshots__isolated_entities.snap
    ├── graph_snapshots__dof_free_point.snap
    ├── graph_snapshots__dof_well_constrained_point.snap
    ├── graph_snapshots__dof_over_constrained.snap
    ├── graph_snapshots__dof_full_sketch.snap
    ├── graph_snapshots__redundancy_clean.snap
    ├── graph_snapshots__redundancy_redundant.snap
    ├── graph_snapshots__redundancy_conflicting.snap
    ├── reduce_snapshots__reduce_substitute.snap
    ├── reduce_snapshots__reduce_merge.snap
    ├── reduce_snapshots__reduce_eliminate.snap
    ├── sketch2d_snapshots__sketch2d_rectangle.snap
    ├── sketch2d_snapshots__sketch2d_circle_tangent.snap
    ├── sketch2d_snapshots__sketch2d_symmetric.snap
    ├── sketch2d_snapshots__sketch2d_equal_length.snap
    ├── sketch2d_snapshots__sketch2d_all_constraints.snap
    ├── pattern_snapshots__pattern_two_distances.snap
    ├── pattern_snapshots__pattern_horizontal_vertical.snap
    ├── pattern_snapshots__pattern_scalar_solve.snap
    ├── pattern_snapshots__pattern_no_match.snap
    ├── nist_snapshots__rosenbrock_nr.snap
    ├── nist_snapshots__rosenbrock_lm.snap
    ├── nist_snapshots__powell_singular_nr.snap
    ├── nist_snapshots__powell_singular_lm.snap
    ├── cross_architecture_snapshots__cross_arch_triangle.snap
    ├── cross_architecture_snapshots__cross_arch_rectangle.snap
    ├── cross_architecture_snapshots__cross_arch_circle_tangent.snap
    ├── sketch3d_assembly_snapshots__sketch3d_coplanar.snap
    ├── sketch3d_assembly_snapshots__sketch3d_perpendicular.snap
    ├── sketch3d_assembly_snapshots__assembly_mate.snap
    ├── sketch3d_assembly_snapshots__assembly_coaxial.snap
    ├── sketch3d_assembly_snapshots__assembly_gear_2to1.snap
    ├── sketch3d_assembly_snapshots__assembly_insert.snap
    ├── jit_snapshots__triangle_interpreted.snap
    ├── jit_snapshots__triangle_jit.snap
    └── ... (70+ total snapshot files)
```

## Metrics

| Metric | Target |
|--------|--------|
| V3 pipeline snapshots | 6+ (solved, overconstrained, multi-cluster, partially-solved, empty, incremental) |
| Closed-form solver snapshots | 6+ (circle-circle, tangent, no-solution, scalar, polar, HV) |
| Graph decomposition snapshots | 4+ (single, multi, star, isolated) |
| DOF analysis snapshots | 4+ (free, constrained, over-constrained, all-fixed) |
| Redundancy analysis snapshots | 3+ (clean, redundant, conflicting) |
| Reduce pass snapshots | 3+ (substitute, merge, eliminate) |
| Sketch2D end-to-end snapshots | 5+ (rectangle, triangle, circle-tangent, symmetric, equal-length) |
| Pattern matching snapshots | 4+ (TwoDistances, HV, ScalarSolve, no-match) |
| NIST/MINPACK snapshots | 32+ (18 least-squares + 14 nonlinear, multiple solvers) |
| Cross-architecture snapshots | 3+ (triangle, rectangle, circle) |
| Sketch3D/Assembly snapshots | 6+ (coplanar, perpendicular, mate, coaxial, gear, insert) |
| JIT vs interpreted snapshots | 2+ |
| Legacy decomposition snapshots | 2+ |
| **Total snapshot files** | **~80+** |

## Estimated Effort

| Task | Time |
|------|------|
| Set up insta dependency and CI | 30 min |
| Write V3 pipeline snapshot tests (Category 1) | 3 hours |
| Write closed-form solver snapshot tests (Category 2) | 2 hours |
| Write graph decomposition snapshot tests (Category 3) | 2 hours |
| Write DOF analysis snapshot tests (Category 4) | 1.5 hours |
| Write redundancy analysis snapshot tests (Category 5) | 1.5 hours |
| Write reduce pass snapshot tests (Category 6) | 2 hours |
| Write Sketch2D end-to-end snapshot tests (Category 7) | 3 hours |
| Write pattern matching snapshot tests (Category 8) | 1.5 hours |
| Write NIST/MINPACK snapshot tests (Category 9) | 2 hours |
| Write cross-architecture comparison tests (Category 10) | 2 hours |
| Write Sketch3D/Assembly snapshot tests (Category 11) | 2 hours |
| Write JIT vs interpreted snapshot tests (Category 12) | 1 hour |
| Write legacy decomposition snapshot tests (Category 13) | 1 hour |
| Initial snapshot generation and review | 2 hours |
| Document redaction strategy | 30 min |
| **Total** | **~26-30 hours** |
