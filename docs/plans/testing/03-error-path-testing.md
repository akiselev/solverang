# Plan 03: Error Path & Negative Testing

## Goal

Error path testing ensures every failure mode in solverang is reachable, tested, and
behaves as documented. The crate now has two distinct error reporting architectures:

1. **Legacy**: `SolveResult` / `SolveError` enum for the `Problem` trait solvers.
2. **V3**: `SystemResult` / `SystemStatus` / `ClusterSolveStatus` / `DiagnosticIssue`
   for the new `ConstraintSystem` pipeline, plus panics in `ParamStore` for invalid IDs.

Both architectures contain error variants that may be dead code, untested, or silently
swallowed. This plan audits every error path in both architectures, identifies gaps,
and provides concrete test strategies for each.

## Audit of Error Paths

### V3 Architecture: `SystemResult` / `SystemStatus`

The V3 `ConstraintSystem::solve()` returns a `SystemResult` (defined in `system.rs`):

```rust
pub struct SystemResult {
    pub status: SystemStatus,
    pub clusters: Vec<ClusterResult>,
    pub total_iterations: usize,
    pub duration: std::time::Duration,
}
```

#### `SystemStatus` Variants

| Variant | Where Created | Test Coverage | Status |
|---------|--------------|---------------|--------|
| `Solved` | `pipeline/mod.rs` -- all clusters converged or skipped | Tested in `system.rs` and `pipeline/mod.rs` tests | OK |
| `PartiallySolved` | `pipeline/mod.rs` -- some converged, some not | Partially tested (accepted alongside `Solved` in assertions) | **Weak** |
| `DiagnosticFailure(Vec<DiagnosticIssue>)` | `pipeline/mod.rs` -- diagnostics + not_converged | **Untested directly** | **Gap** |

#### `ClusterSolveStatus` Variants

| Variant | Where Created | Test Coverage | Status |
|---------|--------------|---------------|--------|
| `Converged` | `pipeline/post_process.rs` | Tested | OK |
| `NotConverged` | `pipeline/post_process.rs` | Partially tested | **Weak** |
| `Skipped` | `pipeline/mod.rs` -- clean cluster or no free vars | Tested in `system.rs` test_solve_with_fixed_param_cluster_skipped | OK |

#### `DiagnosticIssue` Variants

| Variant | Where Created | Test Coverage | Status |
|---------|--------------|---------------|--------|
| `RedundantConstraint { constraint, implied_by }` | `system.rs` diagnose() via `graph/redundancy.rs` | Unknown -- need to verify | **Needs audit** |
| `ConflictingConstraints { constraints }` | `system.rs` diagnose() via `graph/redundancy.rs` | Unknown | **Needs audit** |
| `UnderConstrained { entity, free_directions }` | `system.rs` diagnose() via `graph/dof.rs` | Unknown | **Needs audit** |

### V3 Architecture: `ParamStore` Panics

`ParamStore` (`param/store.rs`) uses `expect()` to panic on invalid `ParamId` access.
These are **intentional panics** (not error returns), consistent with Rust conventions
for index types. Each panic site needs a test confirming the behavior:

| Method | Panic Condition | Test Coverage | Status |
|--------|----------------|---------------|--------|
| `get(id)` | Stale generation or out-of-bounds index | `test_free_and_reuse` tests generation, but no OOB test | **Weak** |
| `set(id, value)` | Stale generation or out-of-bounds index | Not directly tested | **Gap** |
| `free(id)` | Stale generation or out-of-bounds index | Not directly tested | **Gap** |
| `fix(id)` | Stale generation or out-of-bounds index | Not directly tested | **Gap** |
| `unfix(id)` | Stale generation or out-of-bounds index | Not directly tested | **Gap** |
| `is_fixed(id)` | Stale generation or out-of-bounds index | Not directly tested | **Gap** |
| `owner(id)` | Stale generation or out-of-bounds index | Not directly tested | **Gap** |

### V3 Architecture: Silent Failure Modes

| Behavior | Location | Current Handling |
|----------|----------|-----------------|
| `remove_entity` with stale `EntityId` | `system.rs` | Silently does nothing (generation check) |
| `remove_constraint` with stale `ConstraintId` | `system.rs` | Silently does nothing (generation check) |
| Constraint references removed entity's params | `system.rs` | Constraint stays, params freed -- Jacobian may panic |
| Pipeline with empty constraint list | `pipeline/mod.rs` | Returns `Solved` with 0 clusters |
| Reduce pass on already-reduced system | `pipeline/reduce.rs` | Should be idempotent -- needs verification |
| `ChangeTracker.clear()` called twice | `dataflow/tracker.rs` | Second clear is a no-op -- needs verification |
| `SolutionCache.get()` for unknown cluster | `dataflow/cache.rs` | Returns `None` -- needs verification |
| `SolverMapping` with no free params | `param/store.rs` | Empty mapping -- solver should handle gracefully |
| DOF analysis with all fixed params | `graph/dof.rs` | Should return 0 DOF -- needs verification |
| Redundancy analysis with 0 constraints | `graph/redundancy.rs` | Should return clean result -- needs verification |

### Legacy Architecture: `SolveResult` / `SolveError`

| Variant | Where Created | Test Coverage | Status |
|---------|--------------|---------------|--------|
| `Converged { solution, residual_norm, iterations }` | All solvers | Well tested | OK |
| `NotConverged { solution, residual_norm, iterations }` | All solvers | Tested | OK |
| `Failed { error }` | All solvers | Partially tested | **Weak** |

### Legacy Architecture: `SolveError` Variants

| Variant | Where Created | Test Coverage | Status |
|---------|--------------|---------------|--------|
| `MaxIterationsExceeded` | **Nowhere** | None | **DEAD CODE** |
| `LineSearchFailed` | `newton_raphson.rs` line search | None | **Untested** |
| `SingularJacobian` | `newton_raphson.rs` SVD step | Inline unit test only | **Weak** |
| `NonFiniteResiduals` | Solver pre-check | Tested | OK |
| `NonFiniteJacobian` | Solver pre-check | Tested | OK |
| `DimensionMismatch` | Various | Tested | OK |

### Legacy Architecture: Silent Failure Modes

| Behavior | Location | Current Handling |
|----------|----------|-----------------|
| Jacobian entry with `row >= residual_count` | `jacobian_dense()` | Silently dropped |
| Jacobian entry with `col >= variable_count` | `jacobian_dense()` | Silently dropped |
| Duplicate `(row, col)` entries | `jacobian_dense()` | Last value wins |
| `residual_count()` changes between calls | Solver loop | Undefined behavior |
| `residuals()` returns wrong length | Solver loop | Potential index OOB panic |
| Empty Jacobian | Solver | May produce NaN |

## Test Plan: V3 Error Paths

### 1. `SystemStatus::DiagnosticFailure`

Trigger `DiagnosticFailure` by creating a system with redundant constraints that also
fails to converge (both conditions must be true for this status).

```rust
#[test]
fn test_diagnostic_failure_status() {
    let mut system = ConstraintSystem::new();
    let (eid, px, py) = add_test_point(&mut system, 0.0, 0.0);

    // Conflicting constraints: fix px to both 5.0 and 10.0
    let cid1 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid1, entity_ids: vec![eid], param: px, target: 5.0,
    }));
    let cid2 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid2, entity_ids: vec![eid], param: px, target: 10.0,
    }));

    let result = system.solve();
    // Should detect conflicting constraints AND fail to converge
    match result.status {
        SystemStatus::DiagnosticFailure(issues) => {
            assert!(!issues.is_empty(), "Should report diagnostic issues");
        }
        SystemStatus::PartiallySolved => {
            // Also acceptable -- depends on whether diagnostics are collected
        }
        SystemStatus::Solved => {
            panic!("Conflicting constraints should not produce Solved status");
        }
    }
}
```

### 2. `SystemStatus::PartiallySolved`

Create a multi-cluster system where one cluster converges and the other does not.

```rust
#[test]
fn test_partially_solved_status() {
    let mut system = ConstraintSystem::new();

    // Cluster 1: solvable (fix px to 3.0)
    let (eid1, px1, _py1) = add_test_point(&mut system, 0.0, 0.0);
    let cid1 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid1, entity_ids: vec![eid1], param: px1, target: 3.0,
    }));

    // Cluster 2: conflicting (fix px2 to 5.0 AND 10.0)
    let (eid2, px2, _py2) = add_test_point(&mut system, 0.0, 0.0);
    let cid2 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid2, entity_ids: vec![eid2], param: px2, target: 5.0,
    }));
    let cid3 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid3, entity_ids: vec![eid2], param: px2, target: 10.0,
    }));

    let result = system.solve();
    assert_eq!(result.clusters.len(), 2);

    // One cluster should have converged, the other not
    let converged = result.clusters.iter()
        .filter(|c| c.status == ClusterSolveStatus::Converged)
        .count();
    let not_converged = result.clusters.iter()
        .filter(|c| c.status == ClusterSolveStatus::NotConverged)
        .count();
    assert!(converged >= 1, "At least one cluster should converge");
    assert!(not_converged >= 1, "At least one cluster should fail");
}
```

### 3. `DiagnosticIssue::RedundantConstraint`

```rust
#[test]
fn test_redundant_constraint_detection() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);

    // Two identical constraints: both fix px to 5.0
    let cid1 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid1, entity_ids: vec![eid], param: px, target: 5.0,
    }));
    let cid2 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid2, entity_ids: vec![eid], param: px, target: 5.0,
    }));

    let issues = system.diagnose();
    let redundant = issues.iter().filter(|i| matches!(i, DiagnosticIssue::RedundantConstraint { .. })).count();
    assert!(redundant >= 1, "Should detect at least one redundant constraint, got {:?}", issues);
}
```

### 4. `DiagnosticIssue::ConflictingConstraints`

```rust
#[test]
fn test_conflicting_constraint_detection() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);

    // Conflicting: fix px to 5.0 AND 10.0
    let cid1 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid1, entity_ids: vec![eid], param: px, target: 5.0,
    }));
    let cid2 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid2, entity_ids: vec![eid], param: px, target: 10.0,
    }));

    let issues = system.diagnose();
    let conflicts = issues.iter().filter(|i| matches!(i, DiagnosticIssue::ConflictingConstraints { .. })).count();
    assert!(conflicts >= 1, "Should detect conflicting constraints, got {:?}", issues);
}
```

### 5. `DiagnosticIssue::UnderConstrained`

```rust
#[test]
fn test_under_constrained_entity_detection() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);

    // Only constrain px (not py) -- entity has 1 remaining DOF
    let cid = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid, entity_ids: vec![eid], param: px, target: 5.0,
    }));

    let issues = system.diagnose();
    let under = issues.iter()
        .filter(|i| matches!(i, DiagnosticIssue::UnderConstrained { entity, free_directions }
            if *entity == eid && *free_directions > 0))
        .count();
    assert!(under >= 1, "Should detect under-constrained entity, got {:?}", issues);
}
```

### 6. `ParamStore` Stale-ID Panics

Each `ParamStore` method that panics on invalid IDs needs a test confirming the panic:

```rust
#[test]
#[should_panic(expected = "invalid ParamId")]
fn test_param_store_get_stale_id_panics() {
    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let id = store.alloc(1.0, owner);
    store.free(id);
    // Reallocate in the same slot (bumps generation)
    let _new_id = store.alloc(2.0, owner);
    // Now `id` has a stale generation -- get should panic
    store.get(id);
}

#[test]
#[should_panic(expected = "invalid ParamId")]
fn test_param_store_set_stale_id_panics() {
    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let id = store.alloc(1.0, owner);
    store.free(id);
    let _new_id = store.alloc(2.0, owner);
    store.set(id, 99.0);
}

#[test]
#[should_panic(expected = "invalid ParamId")]
fn test_param_store_free_stale_id_panics() {
    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let id = store.alloc(1.0, owner);
    store.free(id);
    let _new_id = store.alloc(2.0, owner);
    store.free(id); // Double free with stale generation
}

#[test]
#[should_panic(expected = "invalid ParamId")]
fn test_param_store_fix_stale_id_panics() {
    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let id = store.alloc(1.0, owner);
    store.free(id);
    let _new_id = store.alloc(2.0, owner);
    store.fix(id);
}

#[test]
#[should_panic(expected = "invalid ParamId")]
fn test_param_store_unfix_stale_id_panics() {
    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let id = store.alloc(1.0, owner);
    store.free(id);
    let _new_id = store.alloc(2.0, owner);
    store.unfix(id);
}

#[test]
#[should_panic(expected = "invalid ParamId")]
fn test_param_store_is_fixed_stale_id_panics() {
    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let id = store.alloc(1.0, owner);
    store.free(id);
    let _new_id = store.alloc(2.0, owner);
    store.is_fixed(id);
}

#[test]
#[should_panic(expected = "invalid ParamId")]
fn test_param_store_owner_stale_id_panics() {
    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let id = store.alloc(1.0, owner);
    store.free(id);
    let _new_id = store.alloc(2.0, owner);
    store.owner(id);
}
```

### 7. `ConstraintSystem` Stale Entity/Constraint ID Handling

The system silently ignores stale IDs in `remove_entity` and `remove_constraint`.
This behavior should be documented and tested:

```rust
#[test]
fn test_remove_entity_stale_id_is_noop() {
    let mut system = ConstraintSystem::new();
    let (eid, _px, _py) = add_test_point(&mut system, 1.0, 2.0);

    system.remove_entity(eid);
    assert_eq!(system.entity_count(), 0);

    // Remove again with the same (now stale) ID -- should be a no-op
    system.remove_entity(eid);
    assert_eq!(system.entity_count(), 0);
}

#[test]
fn test_remove_constraint_stale_id_is_noop() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

    let cid = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid, entity_ids: vec![eid], param: px, target: 5.0,
    }));

    system.remove_constraint(cid);
    assert_eq!(system.constraint_count(), 0);

    // Remove again with the same (now stale) ID
    system.remove_constraint(cid);
    assert_eq!(system.constraint_count(), 0);
}
```

### 8. Constraint Referencing Removed Entity's Params

When an entity is removed, its params are freed, but constraints referencing those
params are not automatically removed. This is a dangerous state that needs testing:

```rust
#[test]
fn test_constraint_survives_entity_removal() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

    let cid = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid, entity_ids: vec![eid], param: px, target: 5.0,
    }));

    // Remove entity but NOT constraint
    system.remove_entity(eid);
    assert_eq!(system.entity_count(), 0);
    assert_eq!(system.constraint_count(), 1); // Constraint still alive

    // Solving should handle this gracefully (not panic)
    let result = std::panic::catch_unwind(
        std::panic::AssertUnwindSafe(|| {
            let mut sys = system; // move
            sys.solve()
        })
    );
    // Document the behavior: either Ok(some_result) or Err(panic)
    // If it panics, we should consider adding validation logic
}
```

### 9. Pipeline Error Propagation (Cluster Failure)

Test what happens when one cluster fails in a multi-cluster system:

```rust
#[test]
fn test_pipeline_one_cluster_fails() {
    let mut system = ConstraintSystem::new();

    // Cluster 1: easy (will converge)
    let (eid1, px1, _) = add_test_point(&mut system, 0.0, 0.0);
    let cid1 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid1, entity_ids: vec![eid1], param: px1, target: 3.0,
    }));

    // Cluster 2: impossible (contradictory)
    let (eid2, px2, _) = add_test_point(&mut system, 0.0, 0.0);
    let cid2 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid2, entity_ids: vec![eid2], param: px2, target: 5.0,
    }));
    let cid3 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid3, entity_ids: vec![eid2], param: px2, target: 10.0,
    }));

    let result = system.solve();

    // Cluster 1 should still have converged
    assert!((system.get_param(px1) - 3.0).abs() < 1e-6,
        "Solvable cluster should still converge even when another cluster fails");

    // Overall status should not be Solved
    assert!(!matches!(result.status, SystemStatus::Solved),
        "System with a failing cluster should not report Solved");
}
```

### 10. `ChangeTracker` Edge Cases

```rust
#[test]
fn test_change_tracker_clear_idempotent() {
    let mut tracker = ChangeTracker::new();
    tracker.mark_param_dirty(ParamId::new(0, 0));
    tracker.mark_entity_added(EntityId::new(0, 0));

    tracker.clear();
    assert!(!tracker.has_any_changes());

    // Second clear should be a no-op, not panic
    tracker.clear();
    assert!(!tracker.has_any_changes());
}

#[test]
fn test_change_tracker_structural_vs_value() {
    let mut tracker = ChangeTracker::new();

    // Value-only change
    tracker.mark_param_dirty(ParamId::new(0, 0));
    assert!(tracker.has_any_changes());
    assert!(!tracker.has_structural_changes());

    // Adding a structural change
    tracker.mark_entity_added(EntityId::new(0, 0));
    assert!(tracker.has_structural_changes());
}

#[test]
fn test_change_tracker_dirty_cluster_computation() {
    use std::collections::HashMap;
    let mut tracker = ChangeTracker::new();
    let pid = ParamId::new(0, 0);
    let cid = ClusterId(0);

    tracker.mark_param_dirty(pid);

    let mut param_to_cluster = HashMap::new();
    param_to_cluster.insert(pid, cid);

    let dirty = tracker.compute_dirty_clusters(&param_to_cluster);
    assert!(dirty.contains(&cid));
}

#[test]
fn test_change_tracker_dirty_param_not_in_any_cluster() {
    use std::collections::HashMap;
    let mut tracker = ChangeTracker::new();
    let pid = ParamId::new(99, 0); // Not in any cluster

    tracker.mark_param_dirty(pid);

    let param_to_cluster: HashMap<ParamId, ClusterId> = HashMap::new();
    let dirty = tracker.compute_dirty_clusters(&param_to_cluster);
    // Should not panic; dirty set may be empty
    assert!(dirty.is_empty());
}
```

### 11. Closed-Form Solver No-Solution Cases

```rust
#[test]
fn test_closed_form_non_intersecting_circles() {
    // Two circles that don't intersect: centers 10 apart, radii 1 each
    // Should return solved=false, not panic
    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let px = store.alloc(0.0, owner);
    let py = store.alloc(0.0, owner);

    // Build TwoDistances pattern with non-intersecting geometry
    // ... (construct appropriate constraints)

    let pattern = MatchedPattern {
        kind: PatternKind::TwoDistances,
        entity_ids: vec![owner],
        constraint_indices: vec![0, 1],
        param_ids: vec![px, py],
    };

    let result = solve_pattern(&pattern, &constraints, &store);
    match result {
        Some(r) => assert!(!r.solved, "Non-intersecting circles should not solve"),
        None => {} // Also acceptable
    }
}

#[test]
fn test_closed_form_zero_jacobian_newton() {
    // Scalar equation where Jacobian is zero at the evaluation point
    // solve_scalar should return solved=false, not divide by zero
    let mut store = ParamStore::new();
    let owner = EntityId::new(0, 0);
    let pid = store.alloc(0.0, owner);

    // Build a constraint where f(0)=1 but f'(0)=0
    // (e.g., residual = x^2 + 1, Jacobian = 2*x, evaluated at x=0)
    // ... (construct constraint)

    let pattern = MatchedPattern {
        kind: PatternKind::ScalarSolve,
        entity_ids: vec![owner],
        constraint_indices: vec![0],
        param_ids: vec![pid],
    };

    let result = solve_pattern(&pattern, &constraints, &store);
    match result {
        Some(r) => assert!(!r.solved, "Zero Jacobian should prevent solving"),
        None => {} // Also acceptable
    }
}
```

### 12. Reduce Pass Error Conditions

```rust
#[test]
fn test_reduce_merge_self_merge() {
    // Coincident constraint where param_a == param_a (self-referencing)
    // Should be detected as trivially satisfied, not create a cycle
    // ... (build constraint system with self-referencing coincident)
}

#[test]
fn test_reduce_merge_chain() {
    // Chain: a==b, b==c => after merge, a, b, c all map to same representative
    // ... (build three params with two coincident constraints)
}

#[test]
fn test_reduce_eliminate_zero_jacobian() {
    // Single-equation constraint where the Jacobian entry for the free param is 0
    // Elimination should skip this constraint (not divide by zero)
    // ... (build constraint with zero partial)
}

#[test]
fn test_reduce_substitute_all_fixed() {
    // All parameters are fixed -- substitution should eliminate everything
    // Result should be an empty reduced system
    // ... (build system, fix all params, run substitute)
}

#[test]
fn test_reduce_idempotent() {
    // Running reduce twice should produce the same result as once
    // ... (build system, reduce, reduce again, compare)
}
```

### 13. DOF Analysis Edge Cases

```rust
#[test]
fn test_dof_analysis_no_constraints() {
    let mut system = ConstraintSystem::new();
    let (_eid, _px, _py) = add_test_point(&mut system, 1.0, 2.0);

    let dof = system.analyze_dof();
    assert_eq!(dof.total_dof, 2, "Unconstrained 2D point should have 2 DOF");
    assert_eq!(dof.total_free_params, 2);
    assert_eq!(dof.total_equations, 0);
}

#[test]
fn test_dof_analysis_over_constrained() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 1.0, 2.0);

    // 3 constraints on 1 parameter = over-constrained
    for target in [1.0, 2.0, 3.0] {
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid, entity_ids: vec![eid], param: px, target,
        }));
    }

    let dof = system.analyze_dof();
    assert!(dof.total_dof < 0, "Over-constrained system should have negative DOF, got {}", dof.total_dof);
}

#[test]
fn test_dof_analysis_all_fixed_params() {
    let mut system = ConstraintSystem::new();
    let (_eid, px, py) = add_test_point(&mut system, 1.0, 2.0);
    system.fix_param(px);
    system.fix_param(py);

    let dof = system.analyze_dof();
    assert_eq!(dof.total_free_params, 0);
    assert_eq!(dof.total_dof, 0);
}
```

### 14. Redundancy Analysis Edge Cases

```rust
#[test]
fn test_redundancy_analysis_no_constraints() {
    let system = ConstraintSystem::new();
    let result = system.analyze_redundancy();
    assert!(result.is_clean());
    assert_eq!(result.equation_count, 0);
}

#[test]
fn test_redundancy_analysis_conflicting_constraints() {
    let mut system = ConstraintSystem::new();
    let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);

    // Fix px to 5 and 10 -- conflicting
    let cid1 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid1, entity_ids: vec![eid], param: px, target: 5.0,
    }));
    let cid2 = system.alloc_constraint_id();
    system.add_constraint(Box::new(FixValueConstraint {
        id: cid2, entity_ids: vec![eid], param: px, target: 10.0,
    }));

    let result = system.analyze_redundancy();
    assert!(!result.is_clean(), "Should detect conflicting constraints");
    assert!(result.rank_deficiency() > 0);
}
```

## Test Plan: Legacy Error Paths

### 15. Dead Code Analysis: `MaxIterationsExceeded`

**Finding:** `MaxIterationsExceeded` is defined in the `SolveError` enum but is never
constructed anywhere in the codebase.

**Recommended action:** Decide whether `NotConverged` vs `Failed(MaxIterationsExceeded)`
is the right semantic. If `NotConverged` always means "hit max iterations", the error
variant is redundant and should be removed. If there is a meaningful difference, wire
it up.

### 16. `LineSearchFailed`

```rust
#[test]
fn test_line_search_failure() {
    // Problem where Newton direction always increases the residual
    struct LineSearchTrap;
    impl Problem for LineSearchTrap {
        fn name(&self) -> &str { "line-search-trap" }
        fn residual_count(&self) -> usize { 1 }
        fn variable_count(&self) -> usize { 1 }
        fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![x[0].powi(3)] }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 3.0 * x[0].powi(2))]
        }
        fn initial_point(&self, _: f64) -> Vec<f64> { vec![1e-10] }
    }

    let solver = Solver::new(SolverConfig {
        max_line_search_iterations: 5,
        ..Default::default()
    });
    let result = solver.solve(&LineSearchTrap, &[1e-10]);
    match result {
        SolveResult::Failed { error } => {
            assert!(matches!(error, SolveError::LineSearchFailed { .. }));
        }
        _ => panic!("Expected line search failure, got {:?}", result),
    }
}
```

### 17. `SingularJacobian`

```rust
#[test]
fn test_singular_jacobian_handling() {
    struct AllZeroJacobian;
    impl Problem for AllZeroJacobian {
        fn name(&self) -> &str { "singular" }
        fn residual_count(&self) -> usize { 2 }
        fn variable_count(&self) -> usize { 2 }
        fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![x[0] * x[0], x[1] * x[1]] }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 0.0), (0, 1, 0.0), (1, 0, 0.0), (1, 1, 0.0)]
        }
        fn initial_point(&self, _: f64) -> Vec<f64> { vec![1.0, 1.0] }
    }

    let solver = Solver::new(SolverConfig::default());
    let result = solver.solve(&AllZeroJacobian, &[1.0, 1.0]);
    assert!(matches!(result, SolveResult::Failed { .. }));
}
```

### 18-20. Legacy Silent Failures

```rust
#[test]
fn test_jacobian_oob_row_dropped() {
    // Document: out-of-bounds entries are silently dropped
    struct OOBJacobian;
    impl Problem for OOBJacobian {
        fn residual_count(&self) -> usize { 2 }
        fn variable_count(&self) -> usize { 2 }
        fn residuals(&self, _: &[f64]) -> Vec<f64> { vec![0.0, 0.0] }
        fn jacobian(&self, _: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 1.0), (1, 1, 1.0), (99, 0, 999.0)]
        }
        fn initial_point(&self, _: f64) -> Vec<f64> { vec![0.0, 0.0] }
    }
    // Should not panic
    let solver = LMSolver::new(LMConfig::default());
    let _ = solver.solve(&OOBJacobian, &[0.0, 0.0]);
}

#[test]
fn test_jacobian_duplicate_entries() {
    // Document: last value wins (overwrite semantics)
    struct DuplicateJacobian;
    impl Problem for DuplicateJacobian {
        fn residual_count(&self) -> usize { 1 }
        fn variable_count(&self) -> usize { 1 }
        fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![x[0] - 1.0] }
        fn jacobian(&self, _: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 1.0), (0, 0, 5.0)] // Duplicate!
        }
        fn initial_point(&self, _: f64) -> Vec<f64> { vec![0.0] }
    }
    // Should not panic
    let solver = LMSolver::new(LMConfig::default());
    let _ = solver.solve(&DuplicateJacobian, &[0.0]);
}
```

## Property-Based Error Testing

```rust
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// V3 ConstraintSystem should never panic on any sequence of valid operations.
    #[test]
    fn prop_v3_system_no_panic(
        num_points in 0usize..10,
        ops in proptest::collection::vec(
            prop_oneof![
                Just(Op::Solve),
                Just(Op::Diagnose),
                (0u8..20).prop_map(|i| Op::RemoveEntity(i)),
                (0u8..20).prop_map(|i| Op::RemoveConstraint(i)),
            ],
            0..20,
        ),
    ) {
        let mut system = ConstraintSystem::new();
        for _ in 0..num_points {
            let eid = system.alloc_entity_id();
            let _px = system.alloc_param(1.0, eid);
        }
        for op in ops {
            let _ = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| {
                    match op {
                        Op::Solve => { system.solve(); }
                        Op::Diagnose => { system.diagnose(); }
                        Op::RemoveEntity(i) => {
                            // This may panic if entity doesn't exist
                        }
                        _ => {}
                    }
                })
            );
        }
    }

    /// Legacy solver should never panic with adversarial Jacobians.
    #[test]
    fn prop_legacy_solver_handles_bad_jacobian(
        row in 0usize..100,
        col in 0usize..100,
        val in prop::num::f64::ANY,
    ) {
        struct BadJacobian { entries: Vec<(usize, usize, f64)> }
        impl Problem for BadJacobian {
            fn name(&self) -> &str { "bad" }
            fn residual_count(&self) -> usize { 2 }
            fn variable_count(&self) -> usize { 2 }
            fn residuals(&self, x: &[f64]) -> Vec<f64> { vec![x[0], x[1]] }
            fn jacobian(&self, _: &[f64]) -> Vec<(usize, usize, f64)> {
                self.entries.clone()
            }
            fn initial_point(&self, _: f64) -> Vec<f64> { vec![1.0, 1.0] }
        }

        let problem = BadJacobian { entries: vec![(row, col, val)] };
        let _ = std::panic::catch_unwind(|| {
            let solver = LMSolver::new(LMConfig::default());
            solver.solve(&problem, &[1.0, 1.0])
        });
    }
}
```

## Suggested Code Changes

| Silent Failure | Recommended Change |
|---------------|-------------------|
| OOB Jacobian entries silently dropped (legacy) | Add `debug_assert!` in `jacobian_dense()`, or return `Result` |
| Duplicate entries overwrite (legacy) | Document as intentional (additive semantics may be better for FEM) |
| Wrong residual length (legacy) | Add dimension check in solver loop, return `DimensionMismatch` |
| `MaxIterationsExceeded` dead code (legacy) | Either wire up or remove the variant |
| `remove_entity` leaves dangling constraint refs (V3) | Consider validating constraint refs on solve, or auto-removing |
| `ParamStore` panics on stale IDs (V3) | Consider returning `Result` instead of panicking |
| `ChangeTracker` wrong structural flag (V3) | Add debug assertions in tracker mutations |

## File Organization

```
crates/solverang/tests/
├── error_path_tests.rs           # Legacy error variant and negative tests
├── v3_error_path_tests.rs        # V3 SystemStatus, DiagnosticIssue tests
├── v3_stale_id_tests.rs          # ParamStore and ConstraintSystem stale-ID tests
├── v3_pipeline_error_tests.rs    # Pipeline error propagation tests
├── silent_failure_tests.rs       # OOB, duplicates, dimension changes (legacy)
└── property_tests.rs             # Add adversarial property tests here
```

## Estimated Effort

| Task | Time |
|------|------|
| Audit all V3 error paths (code reading) | 3 hours |
| Audit all legacy error paths (code reading) | 1 hour |
| Write V3 SystemStatus/DiagnosticIssue tests | 3-4 hours |
| Write V3 ParamStore stale-ID panic tests | 2 hours |
| Write V3 ConstraintSystem stale-ID tests | 2 hours |
| Write V3 pipeline error propagation tests | 2-3 hours |
| Write V3 ChangeTracker/SolutionCache edge case tests | 2 hours |
| Write V3 closed-form no-solution tests | 1-2 hours |
| Write V3 reduce pass error tests | 2 hours |
| Write V3 DOF/redundancy edge case tests | 2 hours |
| Write legacy error variant trigger tests | 2-3 hours |
| Write legacy silent failure tests | 1-2 hours |
| Add property-based error tests | 2 hours |
| Decide on `MaxIterationsExceeded` fate | 30 min discussion |
| Implement code changes (if any) | 2-4 hours |
| **Total** | **~28-34 hours** |
