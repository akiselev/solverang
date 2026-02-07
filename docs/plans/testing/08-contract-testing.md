# Plan 08: Contract / Design-by-Contract Testing

## Goal

Define and enforce interface contracts for every public trait and critical data
structure in solverang, covering both the legacy `Problem` trait and all V3
traits (`Entity`, `Constraint`, `Decompose`, `Analyze`, `Reduce`, `SolveCluster`,
`PostProcess`), the `ParamStore`, the `ConstraintGraph`, the `Sketch2DBuilder`,
and every concrete constraint implementation across sketch2d, sketch3d, and
assembly modules. Contract validators are reusable functions that can be
embedded in `debug_assert!` blocks, invoked from integration tests, and applied
to every concrete type.

---

## Part A: Legacy Contracts

### A.1 `Problem` Trait Contract

Source: `crates/solverang/src/problem.rs`

The `Problem` trait has the following implicit contracts documented in its
docstring but not enforced at runtime.

| Contract | Assertion |
|----------|-----------|
| `residuals()` length | `residuals(x).len() == residual_count()` |
| `jacobian()` row bounds | All `(row, _, _)` in `jacobian(x)` have `row < residual_count()` |
| `jacobian()` col bounds | All `(_, col, _)` in `jacobian(x)` have `col < variable_count()` |
| `jacobian()` finite values | All `(_, _, v)` in `jacobian(x)` have `v.is_finite()` |
| `initial_point()` length | `initial_point().len() == variable_count()` |
| Counts consistency | `residual_count() > 0 && variable_count() > 0` |

```rust
/// Validate all contracts of the `Problem` trait at a given point.
///
/// Returns `Ok(())` if all contracts pass, or `Err(violations)` listing
/// every contract that was violated.
pub fn validate_problem_contract(
    problem: &dyn Problem,
    x: &[f64],
) -> Result<(), Vec<ContractViolation>> {
    let mut violations = Vec::new();

    // Counts
    if problem.residual_count() == 0 {
        violations.push(ContractViolation::new("residual_count() == 0"));
    }
    if problem.variable_count() == 0 {
        violations.push(ContractViolation::new("variable_count() == 0"));
    }

    // residuals() length
    let r = problem.residuals(x);
    if r.len() != problem.residual_count() {
        violations.push(ContractViolation::new(format!(
            "residuals().len() = {} but residual_count() = {}",
            r.len(), problem.residual_count()
        )));
    }

    // residuals() finiteness
    for (i, val) in r.iter().enumerate() {
        if !val.is_finite() {
            violations.push(ContractViolation::new(format!(
                "residuals()[{}] = {} (not finite)", i, val
            )));
        }
    }

    // jacobian() bounds and finiteness
    let jac = problem.jacobian(x);
    for (i, &(row, col, val)) in jac.iter().enumerate() {
        if row >= problem.residual_count() {
            violations.push(ContractViolation::new(format!(
                "jacobian()[{}]: row {} >= residual_count() {}",
                i, row, problem.residual_count()
            )));
        }
        if col >= problem.variable_count() {
            violations.push(ContractViolation::new(format!(
                "jacobian()[{}]: col {} >= variable_count() {}",
                i, col, problem.variable_count()
            )));
        }
        if !val.is_finite() {
            violations.push(ContractViolation::new(format!(
                "jacobian()[{}]: value {} at ({}, {}) is not finite",
                i, val, row, col
            )));
        }
    }

    // initial_point() length
    if let Some(ip) = problem.initial_point_opt() {
        if ip.len() != problem.variable_count() {
            violations.push(ContractViolation::new(format!(
                "initial_point().len() = {} but variable_count() = {}",
                ip.len(), problem.variable_count()
            )));
        }
    }

    if violations.is_empty() { Ok(()) } else { Err(violations) }
}
```

### A.2 `GeometricConstraint` Trait Contract (Legacy)

| Contract | Assertion |
|----------|-----------|
| `dimension()` matches template param | `D == dimension()` |
| `residual()` returns finite values | All residuals are finite |
| `point_count()` consistency | Referenced point indices < point_count |

```rust
pub fn validate_geometric_constraint_contract<const D: usize>(
    constraint: &dyn GeometricConstraint<D>,
    points: &[[f64; D]],
) -> Result<(), Vec<ContractViolation>> {
    let mut violations = Vec::new();

    // Residual finiteness
    let residuals = constraint.residual(points);
    for (i, r) in residuals.iter().enumerate() {
        if !r.is_finite() {
            violations.push(ContractViolation::new(format!(
                "residual()[{}] = {} (not finite)", i, r
            )));
        }
    }

    // Jacobian bounds
    let jac = constraint.jacobian(points);
    let max_row = constraint.equation_count();
    let max_col = points.len() * D;
    for &(row, col, val) in &jac {
        if row >= max_row {
            violations.push(ContractViolation::new(format!(
                "jacobian entry: row {} >= equation_count {}", row, max_row
            )));
        }
        if col >= max_col {
            violations.push(ContractViolation::new(format!(
                "jacobian entry: col {} >= total_params {}", col, max_col
            )));
        }
        if !val.is_finite() {
            violations.push(ContractViolation::new(format!(
                "jacobian entry ({}, {}) = {} (not finite)", row, col, val
            )));
        }
    }

    if violations.is_empty() { Ok(()) } else { Err(violations) }
}
```

### A.3 Apply Legacy Validators to All Concrete Types

```rust
#[test]
fn test_all_legacy_problems_satisfy_contracts() {
    for (name, problem) in all_test_problems() {
        let x = problem.initial_point(1.0);
        let result = validate_problem_contract(&*problem, &x);
        assert!(result.is_ok(), "{}: {}", name,
            result.unwrap_err().iter().map(|v| v.message.as_str()).collect::<Vec<_>>().join("; "));
    }
}

#[test]
fn test_all_geometric_constraints_satisfy_contracts() {
    for (name, constraint, points) in all_geometric_test_constraints() {
        let result = validate_geometric_constraint_contract(&*constraint, &points);
        assert!(result.is_ok(), "{}: {}", name,
            result.unwrap_err().iter().map(|v| v.message.as_str()).collect::<Vec<_>>().join("; "));
    }
}
```

---

## Part B: V3 Constraint Trait Contract

Source: `crates/solverang/src/constraint/mod.rs`

### B.1 `Constraint` Trait Contracts

The V3 `Constraint` trait returns Jacobian entries as `(row, ParamId, value)`
instead of `(row, col, value)`. The `ParamStore` maps `ParamId`s to column
indices at solve time.

| Contract | Assertion |
|----------|-----------|
| Nonzero equation count | `equation_count() > 0` |
| Residuals length | `residuals(store).len() == equation_count()` |
| Residuals finiteness | All values in `residuals(store)` are finite |
| Jacobian row bounds | All `(row, _, _)` have `row < equation_count()` |
| Jacobian ParamId validity | All `(_, pid, _)` have `pid` in `param_ids()` |
| Jacobian value finiteness | All `(_, _, v)` have `v.is_finite()` |
| param_ids nonempty | `!param_ids().is_empty()` |
| entity_ids nonempty | `!entity_ids().is_empty()` |
| No duplicate param_ids | `param_ids()` contains no duplicates |

```rust
/// Validate all contracts of the V3 `Constraint` trait.
pub fn validate_constraint_contract(
    constraint: &dyn Constraint,
    store: &ParamStore,
) -> Result<(), Vec<ContractViolation>> {
    let mut violations = Vec::new();

    // equation_count > 0
    if constraint.equation_count() == 0 {
        violations.push(ContractViolation::new("equation_count() == 0"));
    }

    // residuals length
    let residuals = constraint.residuals(store);
    if residuals.len() != constraint.equation_count() {
        violations.push(ContractViolation::new(format!(
            "residuals().len() = {} but equation_count() = {}",
            residuals.len(), constraint.equation_count()
        )));
    }

    // residuals finiteness
    for (i, r) in residuals.iter().enumerate() {
        if !r.is_finite() {
            violations.push(ContractViolation::new(format!(
                "residuals()[{}] = {} (not finite)", i, r
            )));
        }
    }

    // Jacobian validation
    let jac = constraint.jacobian(store);
    let valid_params: std::collections::HashSet<_> =
        constraint.param_ids().iter().copied().collect();

    for (idx, &(row, pid, val)) in jac.iter().enumerate() {
        // Row bounds
        if row >= constraint.equation_count() {
            violations.push(ContractViolation::new(format!(
                "jacobian[{}]: row {} >= equation_count() {}",
                idx, row, constraint.equation_count()
            )));
        }

        // ParamId in param_ids()
        if !valid_params.contains(&pid) {
            violations.push(ContractViolation::new(format!(
                "jacobian[{}]: ParamId {:?} not in param_ids()",
                idx, pid
            )));
        }

        // Value finiteness
        if !val.is_finite() {
            violations.push(ContractViolation::new(format!(
                "jacobian[{}]: value {} at (row={}, param={:?}) is not finite",
                idx, val, row, pid
            )));
        }
    }

    // param_ids nonempty
    if constraint.param_ids().is_empty() {
        violations.push(ContractViolation::new("param_ids() is empty"));
    }

    // entity_ids nonempty
    if constraint.entity_ids().is_empty() {
        violations.push(ContractViolation::new("entity_ids() is empty"));
    }

    // No duplicate param_ids
    let param_ids = constraint.param_ids();
    let unique: std::collections::HashSet<_> = param_ids.iter().collect();
    if unique.len() != param_ids.len() {
        violations.push(ContractViolation::new(format!(
            "param_ids() contains duplicates: {} unique out of {} total",
            unique.len(), param_ids.len()
        )));
    }

    if violations.is_empty() { Ok(()) } else { Err(violations) }
}
```

---

## Part C: V3 Entity Trait Contract

Source: `crates/solverang/src/entity/mod.rs`

| Contract | Assertion |
|----------|-----------|
| `params()` returns valid ParamIds | All returned ParamIds are valid in the ParamStore |
| `id()` is consistent | `id()` returns the same value on repeated calls |
| `params()` nonempty | `!params().is_empty()` |
| `params()` no duplicates | No duplicate ParamIds |

```rust
/// Validate all contracts of the V3 `Entity` trait.
pub fn validate_entity_contract(
    entity: &dyn Entity,
    store: &ParamStore,
) -> Result<(), Vec<ContractViolation>> {
    let mut violations = Vec::new();

    // id() consistency
    let id1 = entity.id();
    let id2 = entity.id();
    if id1 != id2 {
        violations.push(ContractViolation::new(format!(
            "id() not consistent: {:?} vs {:?}", id1, id2
        )));
    }

    // params() nonempty
    let params = entity.params();
    if params.is_empty() {
        violations.push(ContractViolation::new("params() is empty"));
    }

    // params() no duplicates
    let unique: std::collections::HashSet<_> = params.iter().collect();
    if unique.len() != params.len() {
        violations.push(ContractViolation::new(format!(
            "params() has duplicates: {} unique out of {} total",
            unique.len(), params.len()
        )));
    }

    // All param IDs are valid in the store
    for &pid in params {
        if !store.is_valid(pid) {
            violations.push(ContractViolation::new(format!(
                "params() contains invalid ParamId {:?}", pid
            )));
        } else {
            let val = store.get(pid);
            if !val.is_finite() {
                violations.push(ContractViolation::new(format!(
                    "param {:?} has non-finite value {}", pid, val
                )));
            }
        }
    }

    if violations.is_empty() { Ok(()) } else { Err(violations) }
}
```

---

## Part D: V3 `ParamStore` Contracts

Source: `crates/solverang/src/param/store.rs`

| Contract | Assertion |
|----------|-----------|
| `get()` with valid ID | Returns a finite `f64` value |
| `get()` with stale ID | Panics (debug) or returns error |
| `set()` with valid ID | Subsequent `get()` returns the set value |
| `set()` with stale ID | Panics (debug) or returns error |
| `alloc()` returns unique ID | Two consecutive `alloc()` return different IDs |
| `free()` followed by `alloc()` | Reuses slot, higher generation |
| `len()` consistency | `len()` matches number of allocated (non-freed) slots |

```rust
/// Validate ParamStore contracts after a sequence of operations.
pub fn validate_param_store_contract(
    store: &ParamStore,
    expected_ids: &[ParamId],
) -> Result<(), Vec<ContractViolation>> {
    let mut violations = Vec::new();

    // All expected IDs are valid and return finite values
    for &pid in expected_ids {
        if !store.is_valid(pid) {
            violations.push(ContractViolation::new(format!(
                "Expected valid ParamId {:?} but is_valid returned false", pid
            )));
        } else {
            let val = store.get(pid);
            if !val.is_finite() {
                violations.push(ContractViolation::new(format!(
                    "ParamId {:?} has non-finite value {}", pid, val
                )));
            }
        }
    }

    // Length consistency
    if store.len() != expected_ids.len() {
        violations.push(ContractViolation::new(format!(
            "store.len() = {} but expected {} active params",
            store.len(), expected_ids.len()
        )));
    }

    if violations.is_empty() { Ok(()) } else { Err(violations) }
}

#[test]
fn test_param_store_get_stale_id_panics() {
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let id1 = store.alloc(42.0, owner);
    store.free(id1);
    // Accessing freed ID should panic in debug mode
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        store.get(id1)
    }));
    assert!(result.is_err(), "get() with stale ID should panic");
}

#[test]
fn test_param_store_set_stale_id_panics() {
    let owner = EntityId::new(0, 0);
    let mut store = ParamStore::new();
    let id1 = store.alloc(42.0, owner);
    store.free(id1);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        store.set(id1, 99.0)
    }));
    assert!(result.is_err(), "set() with stale ID should panic");
}
```

---

## Part E: V3 `ConstraintGraph` Contracts

Source: `crates/solverang/src/graph/mod.rs`

| Contract | Assertion |
|----------|-----------|
| All constraint entities registered | For every constraint, all `entity_ids()` are in the graph |
| `param_to_constraints` consistent | If constraint C has param P, then `param_to_constraints[P]` contains C |
| Cluster completeness | Every registered constraint appears in exactly one cluster |
| Cluster separation | No two clusters share a `ParamId` |

```rust
/// Validate ConstraintGraph structural contracts.
pub fn validate_constraint_graph_contract(
    constraints: &[Option<Box<dyn Constraint>>],
    entities: &[Option<Box<dyn Entity>>],
    clusters: &[ClusterData],
    store: &ParamStore,
) -> Result<(), Vec<ContractViolation>> {
    let mut violations = Vec::new();

    // All constraint entities are registered
    let entity_ids_present: std::collections::HashSet<_> = entities.iter()
        .filter_map(|e| e.as_ref().map(|e| e.id()))
        .collect();

    for (idx, c_opt) in constraints.iter().enumerate() {
        if let Some(c) = c_opt {
            for &eid in c.entity_ids() {
                if !entity_ids_present.contains(&eid) {
                    violations.push(ContractViolation::new(format!(
                        "Constraint [{}] references entity {:?} not in the graph",
                        idx, eid
                    )));
                }
            }
        }
    }

    // Cluster completeness: every active constraint in exactly one cluster
    let mut constraint_cluster: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for (ci, cluster) in clusters.iter().enumerate() {
        for &constraint_idx in &cluster.constraint_indices {
            if let Some(prev) = constraint_cluster.insert(constraint_idx, ci) {
                violations.push(ContractViolation::new(format!(
                    "Constraint index {} in clusters {} and {}",
                    constraint_idx, prev, ci
                )));
            }
        }
    }
    for (idx, c_opt) in constraints.iter().enumerate() {
        if c_opt.is_some() && !constraint_cluster.contains_key(&idx) {
            violations.push(ContractViolation::new(format!(
                "Active constraint index {} not in any cluster", idx
            )));
        }
    }

    // Cluster separation: no shared ParamIds between clusters
    let mut param_cluster: std::collections::HashMap<ParamId, usize> =
        std::collections::HashMap::new();
    for (ci, cluster) in clusters.iter().enumerate() {
        for &pid in &cluster.param_ids {
            if let Some(prev) = param_cluster.insert(pid, ci) {
                if prev != ci {
                    violations.push(ContractViolation::new(format!(
                        "ParamId {:?} shared between clusters {} and {}",
                        pid, prev, ci
                    )));
                }
            }
        }
    }

    if violations.is_empty() { Ok(()) } else { Err(violations) }
}
```

---

## Part F: V3 `SolvePipeline` Phase Contracts

Source: `crates/solverang/src/pipeline/traits.rs`,
`crates/solverang/src/pipeline/types.rs`

### F.1 Decompose Phase Post-Conditions

| Contract | Assertion |
|----------|-----------|
| Non-overlapping clusters | No constraint index appears in more than one `ClusterData` |
| Complete coverage | Every active constraint appears in some cluster |
| Param consistency | `ClusterData::param_ids` matches union of constraint param_ids |
| Entity consistency | `ClusterData::entity_ids` matches union of constraint entity_ids |

```rust
pub fn validate_decompose_postconditions(
    clusters: &[ClusterData],
    constraints: &[Option<Box<dyn Constraint>>],
) -> Result<(), Vec<ContractViolation>> {
    let mut violations = Vec::new();

    // Non-overlapping
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for cluster in clusters {
        for &idx in &cluster.constraint_indices {
            if !seen.insert(idx) {
                violations.push(ContractViolation::new(format!(
                    "Constraint index {} in multiple clusters", idx
                )));
            }
        }
    }

    // Complete coverage
    for (idx, c_opt) in constraints.iter().enumerate() {
        if c_opt.is_some() && !seen.contains(&idx) {
            violations.push(ContractViolation::new(format!(
                "Active constraint {} not in any cluster", idx
            )));
        }
    }

    // Param consistency per cluster
    for cluster in clusters {
        let expected_params: std::collections::HashSet<_> = cluster.constraint_indices.iter()
            .filter_map(|&idx| constraints[idx].as_ref())
            .flat_map(|c| c.param_ids().to_vec())
            .collect();
        let actual_params: std::collections::HashSet<_> =
            cluster.param_ids.iter().copied().collect();
        if expected_params != actual_params {
            let missing: Vec<_> = expected_params.difference(&actual_params).collect();
            let extra: Vec<_> = actual_params.difference(&expected_params).collect();
            violations.push(ContractViolation::new(format!(
                "Cluster {:?}: param_ids mismatch. Missing: {:?}, Extra: {:?}",
                cluster.id, missing, extra
            )));
        }
    }

    if violations.is_empty() { Ok(()) } else { Err(violations) }
}
```

### F.2 Reduce Phase Post-Conditions

| Contract | Assertion |
|----------|-----------|
| No variable count increase | `reduced.active_param_ids.len() <= cluster.param_ids.len()` |
| Removed + active = original | `removed_constraints` + `active_constraint_indices` covers all original constraints |
| Eliminated params have values | Every `(ParamId, f64)` in `eliminated_params` has a finite value |
| Merge map targets are active | All values in `merge_map` are in `active_param_ids` or `eliminated_params` |

```rust
pub fn validate_reduce_postconditions(
    reduced: &ReducedCluster,
    original: &ClusterData,
) -> Result<(), Vec<ContractViolation>> {
    let mut violations = Vec::new();

    // No variable count increase
    if reduced.active_param_ids.len() > original.param_ids.len() {
        violations.push(ContractViolation::new(format!(
            "Reduce increased variable count: {} -> {}",
            original.param_ids.len(), reduced.active_param_ids.len()
        )));
    }

    // Eliminated params are finite
    for (pid, val) in &reduced.eliminated_params {
        if !val.is_finite() {
            violations.push(ContractViolation::new(format!(
                "Eliminated param {:?} has non-finite value {}", pid, val
            )));
        }
    }

    // Removed + active covers original
    let mut all_constraints: std::collections::HashSet<usize> =
        reduced.active_constraint_indices.iter().copied().collect();
    all_constraints.extend(reduced.removed_constraints.iter());
    all_constraints.extend(reduced.trivially_violated.iter());
    let original_set: std::collections::HashSet<usize> =
        original.constraint_indices.iter().copied().collect();
    if all_constraints != original_set {
        let missing: Vec<_> = original_set.difference(&all_constraints).collect();
        let extra: Vec<_> = all_constraints.difference(&original_set).collect();
        violations.push(ContractViolation::new(format!(
            "Reduce coverage mismatch. Missing: {:?}, Extra: {:?}", missing, extra
        )));
    }

    if violations.is_empty() { Ok(()) } else { Err(violations) }
}
```

### F.3 Solve Phase Post-Conditions

| Contract | Assertion |
|----------|-----------|
| All free params have values | Every `ParamId` in `reduced.active_param_ids` appears in `solution.param_values` |
| Values are finite | All `(_, value)` in `param_values` are finite |
| Residual norm is finite | `solution.residual_norm.is_finite()` |
| Status matches residual | If `status == Converged`, then `residual_norm < tolerance` |

```rust
pub fn validate_solve_postconditions(
    solution: &ClusterSolution,
    reduced: &ReducedCluster,
    tolerance: f64,
) -> Result<(), Vec<ContractViolation>> {
    let mut violations = Vec::new();

    // All free params have values
    let solved_params: std::collections::HashSet<_> =
        solution.param_values.iter().map(|(pid, _)| pid).collect();
    for &pid in &reduced.active_param_ids {
        if !solved_params.contains(&pid) {
            violations.push(ContractViolation::new(format!(
                "Free param {:?} has no value in solution", pid
            )));
        }
    }

    // Values are finite
    for (pid, val) in &solution.param_values {
        if !val.is_finite() {
            violations.push(ContractViolation::new(format!(
                "Param {:?} solved to non-finite value {}", pid, val
            )));
        }
    }

    // Residual norm is finite
    if !solution.residual_norm.is_finite() {
        violations.push(ContractViolation::new(format!(
            "Residual norm is not finite: {}", solution.residual_norm
        )));
    }

    // Status matches residual
    if solution.status == ClusterSolveStatus::Converged {
        if solution.residual_norm > tolerance {
            violations.push(ContractViolation::new(format!(
                "Status is Converged but residual_norm {} > tolerance {}",
                solution.residual_norm, tolerance
            )));
        }
    }

    if violations.is_empty() { Ok(()) } else { Err(violations) }
}
```

---

## Part G: `Sketch2DBuilder` Contracts

Source: `crates/solverang/src/sketch2d/builder.rs`

| Contract | Assertion |
|----------|-----------|
| Entity count matches | `system.entity_count()` == number of `add_point` + `add_line` + `add_circle` calls |
| Constraint count matches | `system.constraint_count()` == number of `constrain_*` calls |
| No orphan params | Every allocated ParamId belongs to exactly one entity |
| Fixed points have fixed params | Fixed points' params are marked as fixed in the store |

```rust
pub fn validate_sketch2d_builder_contract(
    builder_entity_count: usize,
    builder_constraint_count: usize,
    system: &ConstraintSystem,
) -> Result<(), Vec<ContractViolation>> {
    let mut violations = Vec::new();

    if system.entity_count() != builder_entity_count {
        violations.push(ContractViolation::new(format!(
            "Entity count mismatch: builder added {} but system has {}",
            builder_entity_count, system.entity_count()
        )));
    }

    if system.constraint_count() != builder_constraint_count {
        violations.push(ContractViolation::new(format!(
            "Constraint count mismatch: builder added {} but system has {}",
            builder_constraint_count, system.constraint_count()
        )));
    }

    if violations.is_empty() { Ok(()) } else { Err(violations) }
}
```

---

## Part H: Apply Contract Validators to All Concrete Types

### H.1 Sketch2D -- All 16 Constraint Types

```rust
#[test]
fn test_all_sketch2d_constraints_satisfy_contracts() {
    let test_cases: Vec<(&str, Box<dyn Constraint>, ParamStore)> = vec![
        build_distance_pt_pt_test_case(),
        build_distance_pt_line_test_case(),
        build_coincident_test_case(),
        build_fixed_test_case(),
        build_horizontal_test_case(),
        build_vertical_test_case(),
        build_parallel_test_case(),
        build_perpendicular_test_case(),
        build_angle_test_case(),
        build_midpoint_test_case(),
        build_symmetric_test_case(),
        build_equal_length_test_case(),
        build_point_on_circle_test_case(),
        build_tangent_line_circle_test_case(),
        build_tangent_circle_circle_test_case(),
        build_point_on_line_test_case(),
    ];

    for (name, constraint, store) in &test_cases {
        let result = validate_constraint_contract(constraint.as_ref(), store);
        assert!(result.is_ok(), "sketch2d/{}: {}", name,
            result.unwrap_err().iter()
                .map(|v| v.message.as_str())
                .collect::<Vec<_>>().join("; "));
    }
}
```

### H.2 Sketch3D -- All 8 Constraint Types

```rust
#[test]
fn test_all_sketch3d_constraints_satisfy_contracts() {
    let test_cases: Vec<(&str, Box<dyn Constraint>, ParamStore)> = vec![
        build_distance3d_test_case(),
        build_point_on_plane_test_case(),
        build_coplanar_test_case(),
        build_parallel3d_test_case(),
        build_perpendicular3d_test_case(),
        build_angle3d_test_case(),
        build_coincident3d_test_case(),
        build_coaxial_test_case(),
    ];

    for (name, constraint, store) in &test_cases {
        let result = validate_constraint_contract(constraint.as_ref(), store);
        assert!(result.is_ok(), "sketch3d/{}: {}", name,
            result.unwrap_err().iter()
                .map(|v| v.message.as_str())
                .collect::<Vec<_>>().join("; "));
    }
}
```

### H.3 Assembly -- All 4 Constraint Types + UnitQuaternion

```rust
#[test]
fn test_all_assembly_constraints_satisfy_contracts() {
    let test_cases: Vec<(&str, Box<dyn Constraint>, ParamStore)> = vec![
        build_mate_test_case(),
        build_coaxial_assembly_test_case(),
        build_insert_test_case(),
        build_gear_test_case(),
        build_unit_quaternion_test_case(),
    ];

    for (name, constraint, store) in &test_cases {
        let result = validate_constraint_contract(constraint.as_ref(), store);
        assert!(result.is_ok(), "assembly/{}: {}", name,
            result.unwrap_err().iter()
                .map(|v| v.message.as_str())
                .collect::<Vec<_>>().join("; "));
    }
}
```

### H.4 Entity Contracts for All Geometry Modules

```rust
#[test]
fn test_all_entities_satisfy_contracts() {
    let entity_test_cases = [
        // Sketch2D
        build_point2d_entity(),
        build_line_segment2d_entity(),
        build_circle2d_entity(),
        // Sketch3D
        build_point3d_entity(),
        build_line3d_entity(),
        build_plane_entity(),
        build_circle3d_entity(),
        // Assembly
        build_rigid_body_entity(),
    ];

    for (name, entity, store) in &entity_test_cases {
        let result = validate_entity_contract(entity.as_ref(), store);
        assert!(result.is_ok(), "entity/{}: {}", name,
            result.unwrap_err().iter()
                .map(|v| v.message.as_str())
                .collect::<Vec<_>>().join("; "));
    }
}
```

---

## Part I: Debug Assertions for Runtime Enforcement

Contract validators should also be embedded as `debug_assert!` calls in hot
paths to catch violations during development and testing without impacting
release performance.

### I.1 In `ConstraintSystem::solve()`

```rust
impl ConstraintSystem {
    pub fn solve(&mut self) -> SystemResult {
        // Pre-conditions
        debug_assert!(!self.constraints.iter().all(|c| c.is_none()),
            "solve() called with no constraints");

        let result = self.pipeline.execute(
            &self.constraints,
            &self.entities,
            &self.store,
            &self.config,
        );

        // Post-conditions
        #[cfg(debug_assertions)]
        for cluster in &result.clusters {
            if cluster.status == ClusterSolveStatus::Converged {
                debug_assert!(
                    cluster.residual_norm < self.config.solver_config.tolerance * 10.0,
                    "Converged cluster has residual {} > tolerance {}",
                    cluster.residual_norm, self.config.solver_config.tolerance
                );
            }
        }

        result
    }
}
```

### I.2 In `ParamStore::get()` / `set()`

```rust
impl ParamStore {
    pub fn get(&self, id: ParamId) -> f64 {
        debug_assert!(self.is_valid(id),
            "get() with stale ParamId {:?}", id);
        // ... actual get ...
    }

    pub fn set(&mut self, id: ParamId, value: f64) {
        debug_assert!(self.is_valid(id),
            "set() with stale ParamId {:?}", id);
        debug_assert!(value.is_finite(),
            "set() with non-finite value {} for {:?}", value, id);
        // ... actual set ...
    }
}
```

### I.3 In `Constraint::jacobian()` Default Wrapper

```rust
/// Wrapper that validates constraint contracts at runtime (debug only).
#[cfg(debug_assertions)]
pub fn checked_jacobian(
    constraint: &dyn Constraint,
    store: &ParamStore,
) -> Vec<(usize, ParamId, f64)> {
    let jac = constraint.jacobian(store);
    let valid_params: std::collections::HashSet<_> =
        constraint.param_ids().iter().copied().collect();
    for &(row, pid, val) in &jac {
        debug_assert!(row < constraint.equation_count(),
            "Jacobian row {} >= equation_count() {}", row, constraint.equation_count());
        debug_assert!(valid_params.contains(&pid),
            "Jacobian entry references unknown ParamId {:?}", pid);
        debug_assert!(val.is_finite(),
            "Jacobian entry at ({}, {:?}) = {} (not finite)", row, pid, val);
    }
    jac
}
```

---

## Part J: ContractViolation Type

```rust
/// A single contract violation with a human-readable description.
#[derive(Clone, Debug)]
pub struct ContractViolation {
    pub message: String,
}

impl ContractViolation {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl std::fmt::Display for ContractViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Contract violation: {}", self.message)
    }
}
```

---

## Part K: Comprehensive Contract Test Suite

```rust
/// Run all contract validators against a fully populated ConstraintSystem.
pub fn validate_system_contracts(
    system: &ConstraintSystem,
) -> Result<(), Vec<ContractViolation>> {
    let mut all_violations = Vec::new();

    // Validate all entities
    for entity_opt in system.entities() {
        if let Some(entity) = entity_opt {
            if let Err(violations) = validate_entity_contract(
                entity.as_ref(), system.store()
            ) {
                all_violations.extend(violations);
            }
        }
    }

    // Validate all constraints
    for constraint_opt in system.constraints() {
        if let Some(constraint) = constraint_opt {
            if let Err(violations) = validate_constraint_contract(
                constraint.as_ref(), system.store()
            ) {
                all_violations.extend(violations);
            }
        }
    }

    // Validate param store
    let active_ids: Vec<ParamId> = system.entities().iter()
        .filter_map(|e| e.as_ref())
        .flat_map(|e| e.params().to_vec())
        .collect();
    if let Err(violations) = validate_param_store_contract(
        system.store(), &active_ids
    ) {
        all_violations.extend(violations);
    }

    if all_violations.is_empty() { Ok(()) } else { Err(all_violations) }
}

#[test]
fn test_sketch2d_builder_system_satisfies_all_contracts() {
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(10.0, 0.0);
    let p2 = b.add_point(5.0, 8.0);
    b.constrain_distance(p0, p1, 10.0);
    b.constrain_distance(p1, p2, 8.0);
    b.constrain_distance(p2, p0, 6.0);

    let system = b.build();
    let result = validate_system_contracts(&system);
    assert!(result.is_ok(), "Contract violations: {:?}", result.unwrap_err());
}

#[test]
fn test_assembly_system_satisfies_all_contracts() {
    // Build a simple 2-body assembly system with a Mate constraint
    // ... and validate all contracts
}

#[test]
fn test_contracts_hold_after_solve() {
    // Build, solve, then re-validate -- contracts must still hold
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(10.0, 0.0);
    b.constrain_distance(p0, p1, 5.0);
    let mut system = b.build();

    let pre_result = validate_system_contracts(&system);
    assert!(pre_result.is_ok(), "Pre-solve: {:?}", pre_result.unwrap_err());

    let _ = system.solve();

    let post_result = validate_system_contracts(&system);
    assert!(post_result.is_ok(), "Post-solve: {:?}", post_result.unwrap_err());
}

#[test]
fn test_contracts_hold_after_incremental_edit() {
    // Build, solve, modify, solve again, validate
    let mut b = Sketch2DBuilder::new();
    let p0 = b.add_fixed_point(0.0, 0.0);
    let p1 = b.add_point(5.0, 0.0);
    b.constrain_distance(p0, p1, 10.0);
    let mut system = b.build();
    let _ = system.solve();

    // Modify
    system.set_param(/* p1.x */, 3.0);
    let _ = system.solve();

    let result = validate_system_contracts(&system);
    assert!(result.is_ok(), "Post-incremental: {:?}", result.unwrap_err());
}
```

---

## File Organization

```
crates/solverang/src/
  contracts/
    mod.rs                        -- ContractViolation type, re-exports
    problem_contract.rs           -- Legacy Problem trait validator
    geometric_contract.rs         -- Legacy GeometricConstraint validator
    constraint_contract.rs        -- V3 Constraint trait validator
    entity_contract.rs            -- V3 Entity trait validator
    param_store_contract.rs       -- ParamStore validator
    graph_contract.rs             -- ConstraintGraph validator
    pipeline_contract.rs          -- Decompose/Reduce/Solve phase validators
    builder_contract.rs           -- Sketch2DBuilder validator
    system_contract.rs            -- Full-system validator (orchestrator)

crates/solverang/tests/
  contract_tests/
    mod.rs
    legacy_contracts.rs           -- Problem + GeometricConstraint tests
    sketch2d_contracts.rs         -- All 16 sketch2d constraint types
    sketch3d_contracts.rs         -- All 8 sketch3d constraint types
    assembly_contracts.rs         -- All 5 assembly constraint types
    entity_contracts.rs           -- All entity types across all modules
    param_store_contracts.rs      -- ParamStore edge cases
    pipeline_phase_contracts.rs   -- Decompose, Reduce, Solve phase post-conditions
    system_lifecycle_contracts.rs -- Build, solve, edit, re-solve lifecycle
```

## Estimated Effort

| Task | Time |
|------|------|
| ContractViolation type and utility module | 1 hour |
| Legacy Problem trait validator | 2-3 hours |
| Legacy GeometricConstraint validator | 1-2 hours |
| V3 Constraint trait validator | 2-3 hours |
| V3 Entity trait validator | 1-2 hours |
| ParamStore validator + stale-ID tests | 2-3 hours |
| ConstraintGraph validator | 2-3 hours |
| Pipeline phase validators (decompose, reduce, solve) | 3-4 hours |
| Sketch2DBuilder validator | 1-2 hours |
| Full-system validator (orchestrator) | 2 hours |
| Apply validators to all 16 sketch2d constraints | 3-4 hours |
| Apply validators to all 8 sketch3d constraints | 2 hours |
| Apply validators to all 5 assembly constraints | 1-2 hours |
| Apply validators to all entity types | 1-2 hours |
| Debug assertion integration | 2-3 hours |
| System lifecycle tests (build, solve, edit, re-solve) | 2-3 hours |
| **Total** | **~26-36 hours** |
