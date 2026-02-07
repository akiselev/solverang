# Plan 07: Property-Based Test Expansion

## Goal

Expand property-based test coverage from the existing 10 properties (Jacobian
verification, no-crash, linear convergence, geometric constraints) to 20
properties covering both the legacy `Problem` trait and the V3
entity/constraint/pipeline architecture. New properties target generational IDs,
`ParamStore` allocation, pipeline idempotency, reduce-pass invariants, all
sketch2d/sketch3d/assembly constraint types, graph decomposition stability,
and redundancy detection.

---

## Gap Analysis

### What IS tested (existing `property_tests.rs`)

- Jacobian matches finite differences (linear, quadratic, coupled)
- Solver does not crash (LM, Auto, Robust)
- Solver improves residual
- Linear problems always converge
- Geometric constraints satisfied at expected points (10 legacy constraint types)
- Triangle convergence
- DOF calculation
- Numerical stability with large/small values

### What is NOT tested

| Property | Architecture | Impact | Difficulty |
|----------|-------------|--------|------------|
| Generational ID monotonicity | V3 | High -- prevents use-after-free | Low |
| ParamStore alloc/dealloc round-trip | V3 | High -- memory correctness | Low |
| Pipeline idempotency | V3 | High -- reveals state leaks | Medium |
| Reduce pass invariant | V3 | High -- correctness of optimization | Medium |
| Sketch2D constraint symmetry (16 types) | V3 | Medium -- formulation correctness | Medium |
| Sketch2D Jacobian correctness (16 types) | V3 | High -- solver convergence | Medium |
| Sketch3D constraint properties | V3 | Medium -- 3D feature parity | Medium |
| Assembly quaternion normalization | V3 | High -- physical correctness | Medium |
| Graph decomposition stability | V3 | High -- determinism | Medium |
| Redundancy detection correctness | V3 | Medium -- diagnostics accuracy | Medium |
| Solver idempotency (legacy) | Legacy | High -- state leaks | Low |
| Solver determinism (legacy) | Legacy | High -- race conditions | Low |
| J^T*J symmetry and PSD | Legacy | Medium -- LM correctness | Medium |
| LM cost monotonicity | Legacy | High -- algorithm correctness | Medium |
| Constraint composition | Both | Medium -- system-level bugs | Medium |
| Residual scaling linearity | Both | Medium -- numerical stability | Low |
| Central difference Jacobian | Legacy | Medium -- tighter validation | Low |
| Decomposition correctness | Both | High -- parallel solver | Medium |
| Sparse pattern consistency | Legacy | Medium -- sparse solver | Medium |

---

## New Properties

### Property 1: Generational ID Monotonicity

**Invariant:** When a slot in `ParamStore`, `ConstraintSystem::entities`, or
`ConstraintSystem::constraints` is freed and reused, the new ID has a strictly
higher generation than the old one.

Source: `crates/solverang/src/id.rs`, `crates/solverang/src/param/store.rs`,
`crates/solverang/src/system.rs`

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_param_id_generation_increases_on_reuse(
        n_alloc in 2usize..50,
        n_free in 1usize..25,
    ) {
        let n_free = n_free.min(n_alloc - 1);
        let owner = EntityId::new(0, 0);
        let mut store = ParamStore::new();

        // Phase 1: allocate n_alloc params
        let mut ids: Vec<ParamId> = (0..n_alloc)
            .map(|i| store.alloc(i as f64, owner))
            .collect();

        // Phase 2: free the first n_free params
        let freed: Vec<ParamId> = ids.drain(..n_free).collect();
        for &id in &freed {
            store.free(id);
        }

        // Phase 3: re-allocate n_free params (should reuse freed slots)
        let new_ids: Vec<ParamId> = (0..n_free)
            .map(|i| store.alloc(100.0 + i as f64, owner))
            .collect();

        // Assert: new generation > old generation for reused slots
        for (old, new) in freed.iter().zip(&new_ids) {
            if old.raw_index() == new.raw_index() {
                prop_assert!(
                    new.generation > old.generation,
                    "Generation did not increase: old={:?}, new={:?}",
                    old, new
                );
            }
        }
    }

    #[test]
    fn prop_entity_id_generation_increases_on_reuse(
        n in 2usize..20,
    ) {
        let mut system = ConstraintSystem::new();

        // Allocate and add entities
        let mut eids = Vec::new();
        for i in 0..n {
            let eid = system.alloc_entity_id();
            let px = system.alloc_param(i as f64, eid);
            system.add_entity(Box::new(TestPoint { id: eid, params: vec![px] }));
            eids.push(eid);
        }

        // Remove first entity
        let old_id = eids[0];
        system.remove_entity(old_id);

        // Allocate new entity (should reuse slot 0)
        let new_id = system.alloc_entity_id();
        if new_id.raw_index() == old_id.raw_index() {
            prop_assert!(
                new_id.generation > old_id.generation,
                "Entity generation did not increase"
            );
        }
    }
}
```

### Property 2: ParamStore Alloc/Dealloc Round-Trip

**Invariant:** Allocate a param, free it, re-allocate -- the new allocation
reuses the same slot index but with a higher generation, and the old ID is no
longer valid (accessing it panics or returns None).

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_param_store_alloc_free_realloc(
        initial_value in -1000.0f64..1000.0,
        new_value in -1000.0f64..1000.0,
    ) {
        let owner = EntityId::new(0, 0);
        let mut store = ParamStore::new();

        let id1 = store.alloc(initial_value, owner);
        prop_assert!((store.get(id1) - initial_value).abs() < 1e-15);

        store.free(id1);

        let id2 = store.alloc(new_value, owner);
        // Same slot reused
        prop_assert_eq!(id2.raw_index(), id1.raw_index());
        // Different generation
        prop_assert_ne!(id1, id2);
        // New value is correct
        prop_assert!((store.get(id2) - new_value).abs() < 1e-15);

        // Old ID is now invalid -- accessing it should panic in debug mode
        // (We test this separately with #[should_panic])
    }
}
```

### Property 3: Pipeline Idempotency

**Invariant:** Calling `system.solve()` twice with no intervening changes
should produce the same result on the second call with zero clusters re-solved
(all clusters skipped due to caching).

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_pipeline_idempotent(
        target_x in -100.0f64..100.0,
        target_y in -100.0f64..100.0,
    ) {
        let mut b = Sketch2DBuilder::new();
        let p = b.add_point(0.0, 0.0);
        b.constrain_fixed(p, target_x, target_y);
        let mut system = b.build();

        // First solve
        let result1 = system.solve();
        prop_assert!(matches!(result1.status,
            SystemStatus::Solved | SystemStatus::PartiallySolved));

        // Second solve with no changes
        let result2 = system.solve();
        prop_assert!(matches!(result2.status, SystemStatus::Solved));

        // All clusters should be skipped (clean)
        prop_assert_eq!(result2.total_iterations, 0,
            "Second solve should skip all clusters");

        for cr in &result2.clusters {
            prop_assert_eq!(cr.status, ClusterSolveStatus::Skipped,
                "Cluster should be skipped on idempotent re-solve");
        }
    }
}
```

### Property 4: Reduce Pass Invariant

**Invariant:** The solution of the reduced system, when expanded back to the
full parameter space, should equal the solution of the unreduced system.
Specifically: parameters that were eliminated by reduction should have the
same values as they would from a full numerical solve.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_reduce_preserves_solution(
        target_d in 1.0f64..50.0,
    ) {
        // Build a system with a fixed point and a horizontal + distance constraint
        // so that reduction can eliminate the y-coordinate analytically.
        let mut b1 = Sketch2DBuilder::new();
        let p0 = b1.add_fixed_point(0.0, 0.0);
        let p1 = b1.add_point(5.0, 3.0);
        b1.constrain_horizontal(p0, p1);
        b1.constrain_distance(p0, p1, target_d);
        let mut system_reduced = b1.build();
        let _ = system_reduced.solve();

        let mut b2 = Sketch2DBuilder::new();
        let p0 = b2.add_fixed_point(0.0, 0.0);
        let p1 = b2.add_point(5.0, 3.0);
        b2.constrain_horizontal(p0, p1);
        b2.constrain_distance(p0, p1, target_d);
        let mut system_unreduced = b2.build();
        system_unreduced.set_pipeline(
            PipelineBuilder::new().reduce(NoopReduce).build()
        );
        let _ = system_unreduced.solve();

        // Compare final parameter values
        // Both should yield the same point coordinates within tolerance
        // (exact comparison may fail due to different solver paths)
    }
}
```

### Property 5: Sketch2D Constraint Symmetry (16 types)

**Invariant:** For symmetric constraints (distance, coincident, equal_length,
tangent_circle_circle), swapping the two entities should produce the same
residual (up to sign where applicable).

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_distance_pt_pt_symmetric(
        x1 in -100.0f64..100.0, y1 in -100.0f64..100.0,
        x2 in -100.0f64..100.0, y2 in -100.0f64..100.0,
        d in 0.1f64..100.0,
    ) {
        let owner = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px1 = store.alloc(x1, owner);
        let py1 = store.alloc(y1, owner);
        let px2 = store.alloc(x2, owner);
        let py2 = store.alloc(y2, owner);

        let e1 = EntityId::new(0, 0);
        let e2 = EntityId::new(1, 0);

        let c12 = DistancePtPt::new(
            ConstraintId::new(0, 0), e1, e2, px1, py1, px2, py2, d,
        );
        let c21 = DistancePtPt::new(
            ConstraintId::new(1, 0), e2, e1, px2, py2, px1, py1, d,
        );

        let r12 = c12.residuals(&store);
        let r21 = c21.residuals(&store);
        prop_assert!(
            (r12[0] - r21[0]).abs() < 1e-10,
            "Distance not symmetric: {} vs {}", r12[0], r21[0]
        );
    }

    #[test]
    fn prop_coincident_symmetric(
        x1 in -100.0f64..100.0, y1 in -100.0f64..100.0,
        x2 in -100.0f64..100.0, y2 in -100.0f64..100.0,
    ) {
        // Coincident constraint residuals should be equal (up to sign)
        // when entities are swapped.
        // ...
    }
}

// Additional symmetry tests for:
// - EqualLength (symmetric in the two line segments)
// - TangentCircleCircle (symmetric in the two circles)
// - Parallel (symmetric in the two lines)
// - Perpendicular (symmetric in the two lines)
```

### Property 6: Sketch2D Jacobian Correctness (16 types)

**Invariant:** For each of the 16 sketch2d constraint types, the analytical
Jacobian should match finite differences computed via the `ParamStore`.

This is the V3 analog of the legacy Jacobian verification, but adapted to
work with `ParamStore` and `ParamId`-based Jacobians instead of `(row, col, value)`.

```rust
/// Verify a V3 constraint's Jacobian against finite differences.
fn verify_v3_jacobian(
    constraint: &dyn Constraint,
    store: &ParamStore,
    h: f64,
    tol: f64,
) -> bool {
    let residuals = constraint.residuals(store);
    let jac = constraint.jacobian(store);
    let param_ids = constraint.param_ids();

    for &pid in param_ids {
        let original = store.get(pid);
        let mut store_plus = store.snapshot();
        let mut store_minus = store.snapshot();
        store_plus.set(pid, original + h);
        store_minus.set(pid, original - h);

        let r_plus = constraint.residuals(&store_plus);
        let r_minus = constraint.residuals(&store_minus);

        for row in 0..constraint.equation_count() {
            let fd = (r_plus[row] - r_minus[row]) / (2.0 * h);
            let analytical = jac.iter()
                .find(|(r, p, _)| *r == row && *p == pid)
                .map(|(_, _, v)| *v)
                .unwrap_or(0.0);
            if (fd - analytical).abs() > tol {
                return false;
            }
        }
    }
    true
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_sketch2d_distance_jacobian(
        x1 in -50.0f64..50.0, y1 in -50.0f64..50.0,
        x2 in -50.0f64..50.0, y2 in -50.0f64..50.0,
        d in 0.1f64..100.0,
    ) {
        let dist = ((x2-x1).powi(2) + (y2-y1).powi(2)).sqrt();
        if dist < 0.01 { return Ok(()); } // Skip near-singular configurations

        let owner = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px1 = store.alloc(x1, owner);
        let py1 = store.alloc(y1, owner);
        let px2 = store.alloc(x2, owner);
        let py2 = store.alloc(y2, owner);

        let c = DistancePtPt::new(
            ConstraintId::new(0, 0),
            EntityId::new(0, 0), EntityId::new(1, 0),
            px1, py1, px2, py2, d,
        );

        prop_assert!(
            verify_v3_jacobian(&c, &store, 1e-7, 1e-4),
            "Jacobian mismatch for DistancePtPt"
        );
    }
}

// Repeat for all 16 constraint types:
// DistancePtPt, DistancePtLine, Coincident, Fixed, Horizontal, Vertical,
// Parallel, Perpendicular, Angle, Midpoint, Symmetric, EqualLength,
// PointOnCircle, TangentLineCircle, TangentCircleCircle, PointOnLine (if present as 16th)
```

### Property 7: Sketch3D Constraint Properties

**Invariant:** 3D constraints satisfy domain-specific properties.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_distance3d_symmetric(
        x1 in -50.0f64..50.0, y1 in -50.0f64..50.0, z1 in -50.0f64..50.0,
        x2 in -50.0f64..50.0, y2 in -50.0f64..50.0, z2 in -50.0f64..50.0,
        d in 0.1f64..100.0,
    ) {
        // Distance3D(A, B, d) should have same residual as Distance3D(B, A, d)
        let owner = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        // ... alloc params, build constraint both ways, compare residuals ...
    }

    #[test]
    fn prop_distance3d_jacobian(
        x1 in -50.0f64..50.0, y1 in -50.0f64..50.0, z1 in -50.0f64..50.0,
        x2 in -50.0f64..50.0, y2 in -50.0f64..50.0, z2 in -50.0f64..50.0,
        d in 0.1f64..100.0,
    ) {
        let dist = ((x2-x1).powi(2) + (y2-y1).powi(2) + (z2-z1).powi(2)).sqrt();
        if dist < 0.01 { return Ok(()); }
        // Verify Jacobian against finite differences
    }

    #[test]
    fn prop_coplanar_zero_on_plane(
        // All points in z=0 plane should have zero Coplanar residual
        x1 in -50.0f64..50.0, y1 in -50.0f64..50.0,
        x2 in -50.0f64..50.0, y2 in -50.0f64..50.0,
        x3 in -50.0f64..50.0, y3 in -50.0f64..50.0,
    ) {
        // Build 3 points with z=0, Coplanar constraint, verify residual = 0
    }

    #[test]
    fn prop_parallel3d_symmetric(
        // Parallel3D(A, B) should have same residual as Parallel3D(B, A)
    ) {
        // ...
    }
}
```

### Property 8: Assembly Quaternion Normalization

**Invariant:** After solving any assembly system that includes a
`UnitQuaternion` constraint, the quaternion parameters should satisfy
`qw^2 + qx^2 + qy^2 + qz^2 = 1` within solver tolerance.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_unit_quaternion_after_solve(
        qw in -2.0f64..2.0, qx in -2.0f64..2.0,
        qy in -2.0f64..2.0, qz in -2.0f64..2.0,
    ) {
        // Skip degenerate all-zero case
        let norm = (qw*qw + qx*qx + qy*qy + qz*qz).sqrt();
        if norm < 0.01 { return Ok(()); }

        let mut system = ConstraintSystem::new();
        let eid = system.alloc_entity_id();
        let tx = system.alloc_param(0.0, eid);
        let ty = system.alloc_param(0.0, eid);
        let tz = system.alloc_param(0.0, eid);
        let pw = system.alloc_param(qw, eid);
        let px = system.alloc_param(qx, eid);
        let py = system.alloc_param(qy, eid);
        let pz = system.alloc_param(qz, eid);

        let body = RigidBody::new(eid, tx, ty, tz, pw, px, py, pz);
        system.add_entity(Box::new(body));

        let cid = system.alloc_constraint_id();
        let uq = UnitQuaternion::new(cid, eid, pw, px, py, pz);
        system.add_constraint(Box::new(uq));

        let result = system.solve();

        let w = system.get_param(pw);
        let x = system.get_param(px);
        let y = system.get_param(py);
        let z = system.get_param(pz);
        let norm_sq = w*w + x*x + y*y + z*z;

        prop_assert!(
            (norm_sq - 1.0).abs() < 1e-6,
            "Quaternion not normalized after solve: |q|^2 = {}", norm_sq
        );
    }
}
```

### Property 9: Graph Decomposition Stability

**Invariant:** The same set of entities and constraints should produce the
same cluster decomposition regardless of insertion order.

Source: `crates/solverang/src/graph/decompose.rs`,
`crates/solverang/src/pipeline/decompose.rs`

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_decomposition_order_independent(
        n_entities in 2usize..6,
        seed in any::<u64>(),
    ) {
        // Build a system with n_entities points and random constraints.
        // Insert entities and constraints in two different random orders.
        // Verify that the resulting cluster decomposition is the same
        // (same number of clusters, same constraint assignments).

        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(seed);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(seed + 1);

        // Build canonical system
        let (entities, constraints) = build_random_system(n_entities, &mut rng1);

        // Order 1: sequential
        let clusters1 = decompose_in_order(&entities, &constraints, &|v| v.to_vec());

        // Order 2: shuffled
        let clusters2 = decompose_in_order(&entities, &constraints, &|v| {
            let mut s = v.to_vec();
            s.shuffle(&mut rng2);
            s
        });

        // Same number of clusters
        prop_assert_eq!(clusters1.len(), clusters2.len(),
            "Different cluster counts: {} vs {}", clusters1.len(), clusters2.len());

        // Same constraint grouping (modulo ordering within clusters)
        let sets1 = clusters_to_sets(&clusters1);
        let sets2 = clusters_to_sets(&clusters2);
        prop_assert_eq!(sets1, sets2, "Cluster contents differ");
    }
}
```

### Property 10: Redundancy Detection Correctness

**Invariant:** Known-redundant constraints should be detected by
`analyze_redundancy()`. Known-independent constraints should not be flagged.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_known_redundant_detected(
        d in 1.0f64..50.0,
    ) {
        // Build a triangle with 3 distance constraints + 1 redundant distance
        // (the fourth distance is implied by the first three on a rigid triangle).
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_fixed_point(0.0, 0.0);
        let p1 = b.add_fixed_point(d, 0.0);
        let p2 = b.add_point(d / 2.0, d);
        b.constrain_distance(p0, p1, d);
        b.constrain_distance(p1, p2, d);
        b.constrain_distance(p2, p0, d);
        // This is fully constrained. Adding a horizontal constraint on p0-p1
        // is redundant since both are fixed.
        b.constrain_horizontal(p0, p1);

        let system = b.build();
        let analysis = system.analyze_redundancy();

        prop_assert!(
            !analysis.redundant.is_empty(),
            "Should detect redundancy, but found none"
        );
    }

    #[test]
    fn prop_independent_not_flagged(
        d1 in 1.0f64..50.0,
        d2 in 1.0f64..50.0,
    ) {
        // Two independent distance constraints on separate point pairs.
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_fixed_point(0.0, 0.0);
        let p1 = b.add_point(5.0, 0.0);
        let p2 = b.add_fixed_point(20.0, 0.0);
        let p3 = b.add_point(25.0, 0.0);
        b.constrain_distance(p0, p1, d1);
        b.constrain_distance(p2, p3, d2);

        let system = b.build();
        let analysis = system.analyze_redundancy();

        prop_assert!(
            analysis.redundant.is_empty(),
            "Independent constraints falsely flagged as redundant"
        );
    }
}
```

---

## Retained Legacy Properties (Updated)

### Property 11: Solver Idempotency (Legacy)

**Invariant:** Solving from a converged solution should return the same
solution in 0-1 iterations.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_solver_idempotent(target in small_vec_strategy(1, 5)) {
        if target.is_empty() || target.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }
        let problem = LinearProblem { target: target.clone() };
        let solver = LMSolver::new(LMConfig::default());
        let x0 = vec![0.0; target.len()];
        let result1 = solver.solve(&problem, &x0);
        if let SolveResult::Converged { solution, .. } = &result1 {
            let result2 = solver.solve(&problem, solution);
            if let SolveResult::Converged { iterations, solution: s2, .. } = &result2 {
                prop_assert!(*iterations <= 1);
                for (a, b) in solution.iter().zip(s2) {
                    prop_assert!((a - b).abs() < 1e-10);
                }
            }
        }
    }
}
```

### Property 12: Solver Determinism (Legacy)

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_solver_deterministic(
        target in small_vec_strategy(1, 4),
        x0 in small_vec_strategy(1, 4),
    ) {
        if target.len() != x0.len() || target.is_empty() { return Ok(()); }
        if target.iter().chain(x0.iter()).any(|v| !v.is_finite()) { return Ok(()); }

        let problem = LinearProblem { target };
        let solver = LMSolver::new(LMConfig::default());
        let r1 = solver.solve(&problem, &x0);
        let r2 = solver.solve(&problem, &x0);
        prop_assert_eq!(format!("{:?}", r1), format!("{:?}", r2));
    }
}
```

### Property 13: J^T*J Symmetry and PSD

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_jtj_symmetric_psd(
        target in small_vec_strategy(2, 6),
        x in small_vec_strategy(2, 6),
    ) {
        if target.len() != x.len() || target.is_empty() { return Ok(()); }

        let problem = CoupledProblem { target };
        let entries = problem.jacobian(&x);

        let m = problem.residual_count();
        let n = problem.variable_count();
        let mut j = nalgebra::DMatrix::zeros(m, n);
        for (row, col, val) in &entries {
            if *row < m && *col < n { j[(*row, *col)] = *val; }
        }

        let jtj = j.transpose() * &j;
        // Symmetry
        for i in 0..n {
            for k in i+1..n {
                prop_assert!((jtj[(i, k)] - jtj[(k, i)]).abs() < 1e-10);
            }
        }
        // PSD
        let eigenvalues = jtj.symmetric_eigenvalues();
        for ev in eigenvalues.iter() {
            prop_assert!(*ev >= -1e-10, "Negative eigenvalue: {}", ev);
        }
    }
}
```

### Property 14: LM Cost Monotonicity

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_lm_monotonic_cost(target in positive_vec_strategy(2, 5)) {
        let problem = QuadraticProblem { target: target.clone() };
        let x0: Vec<f64> = target.iter().map(|t| t.sqrt() + 1.0).collect();
        let initial_norm = problem.residual_norm(&x0);
        let solver = LMSolver::new(LMConfig::default());
        let result = solver.solve(&problem, &x0);
        if let SolveResult::Converged { residual_norm, .. }
            | SolveResult::NotConverged { residual_norm, .. } = result
        {
            prop_assert!(residual_norm <= initial_norm + 1e-10);
        }
    }
}
```

### Property 15: Constraint Composition (V3)

**Invariant:** Adding a constraint that is already satisfied should not change
the solution (the system is already at a valid configuration).

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_redundant_constraint_no_change_v3(
        d in 1.0f64..50.0,
    ) {
        // Build a well-constrained triangle, solve it
        let mut b = Sketch2DBuilder::new();
        let p0 = b.add_fixed_point(0.0, 0.0);
        let p1 = b.add_fixed_point(d, 0.0);
        let p2 = b.add_point(d / 2.0, d / 2.0);
        b.constrain_distance(p0, p1, d);
        b.constrain_distance(p1, p2, d);
        b.constrain_distance(p2, p0, d);
        let mut system = b.build();
        let _ = system.solve();

        // Record solution
        let x_before = system.get_param(/* p2.x */);
        let y_before = system.get_param(/* p2.y */);

        // Add redundant horizontal constraint (p0-p1 are already horizontal)
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(Horizontal::new(/* p0, p1 */)));
        let _ = system.solve();

        let x_after = system.get_param(/* p2.x */);
        let y_after = system.get_param(/* p2.y */);

        prop_assert!((x_before - x_after).abs() < 1e-6);
        prop_assert!((y_before - y_after).abs() < 1e-6);
    }
}
```

### Property 16: Residual Scaling Linearity (V3)

For sketch2d `DistancePtPt` (squared formulation), residual scales quadratically
with deviation since `r = dx^2+dy^2 - d^2`.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_distance_residual_scaling_squared(
        target_dist in 1.0f64..100.0,
        deviation in 0.1f64..10.0,
    ) {
        let owner = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px1 = store.alloc(0.0, owner);
        let py1 = store.alloc(0.0, owner);
        let px2_single = store.alloc(target_dist + deviation, owner);
        let py2_single = store.alloc(0.0, owner);

        let c = DistancePtPt::new(
            ConstraintId::new(0, 0),
            EntityId::new(0, 0), EntityId::new(1, 0),
            px1, py1, px2_single, py2_single,
            target_dist,
        );

        let r1 = c.residuals(&store)[0]; // (d+dev)^2 - d^2

        let mut store2 = store.snapshot();
        store2.set(px2_single, target_dist + 2.0 * deviation);
        let r2 = c.residuals(&store2)[0]; // (d+2*dev)^2 - d^2

        // For squared formulation: r = (d+k*dev)^2 - d^2 = 2*d*k*dev + k^2*dev^2
        // r2/r1 should be approximately (2*d*2dev + 4dev^2) / (2*d*dev + dev^2)
        // This is NOT simply 2x, so we verify the actual algebraic scaling.
        let expected_r1 = 2.0 * target_dist * deviation + deviation * deviation;
        let expected_r2 = 2.0 * target_dist * 2.0 * deviation + 4.0 * deviation * deviation;

        prop_assert!(
            (r1 - expected_r1).abs() < 1e-8,
            "r1 mismatch: {} vs {}", r1, expected_r1
        );
        prop_assert!(
            (r2 - expected_r2).abs() < 1e-8,
            "r2 mismatch: {} vs {}", r2, expected_r2
        );
    }
}
```

### Property 17: Central Difference Jacobian Tighter than Forward (Legacy)

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_central_diff_tighter(
        target in positive_vec_strategy(2, 6),
        x in small_vec_strategy(2, 6),
    ) {
        if target.len() != x.len() { return Ok(()); }
        let problem = QuadraticProblem { target };
        let h = 1e-7;
        let forward_error = compute_forward_diff_error(&problem, &x, h);
        let central_error = compute_central_diff_error(&problem, &x, h);
        prop_assert!(central_error <= forward_error * 1.1 + 1e-12);
    }
}
```

### Property 18: Decomposition Matches Monolithic (Legacy)

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_decomposed_matches_monolithic(
        n_groups in 2usize..4,
        group_size in 2usize..4,
    ) {
        let problem = build_block_diagonal_problem(n_groups, group_size);
        let x0 = problem.initial_point(1.0);
        let mono = LMSolver::new(LMConfig::default()).solve(&problem, &x0);
        let decomp_result = solve_decomposed(&problem, &x0);
        // Compare solutions
    }
}
```

### Property 19: Sparse Pattern Covers Nonzeros (Legacy)

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_sparse_pattern_covers_nonzeros(n in 5usize..30) {
        let problem = BroydenTridiagonal::new(n);
        let x = problem.initial_point(1.0);
        let entries = problem.jacobian(&x);
        let pattern = detect_sparsity_pattern(&problem, &x);
        for (row, col, val) in &entries {
            if val.abs() > 1e-15 {
                prop_assert!(pattern.contains(*row, *col));
            }
        }
    }
}
```

### Property 20: V3 Solver Determinism

**Invariant:** The V3 `ConstraintSystem::solve()` is deterministic: same
initial parameters and constraints produce bitwise identical results.

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_v3_solve_deterministic(
        x in -50.0f64..50.0,
        y in -50.0f64..50.0,
        target_x in -50.0f64..50.0,
    ) {
        let build = || {
            let mut b = Sketch2DBuilder::new();
            let p = b.add_point(x, y);
            b.constrain_fixed_x(p, target_x);
            let mut system = b.build();
            let _ = system.solve();
            system.get_param(/* p.x */)
        };
        let v1 = build();
        let v2 = build();
        prop_assert_eq!(v1.to_bits(), v2.to_bits(),
            "Non-deterministic V3 solve: {} vs {}", v1, v2);
    }
}
```

---

## Improved Strategies

### Near-singular strategy

```rust
fn near_singular_target_strategy(n: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        prop_oneof![
            Just(0.0), Just(1e-15), Just(1e-300),
            -100.0f64..100.0,
        ],
        n,
    )
}
```

### Ill-conditioned problem strategy

```rust
fn ill_conditioned_strategy() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        prop_oneof![
            1e-10f64..1e-8,
            1e8f64..1e10,
        ],
        2..6,
    )
}
```

### V3 point coordinate strategy

```rust
fn coord_strategy() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0), Just(1e-10), Just(-1e-10),
        Just(1e6), Just(-1e6),
        -1000.0f64..1000.0,
    ]
}
```

---

## ProptestConfig Tuning

```rust
// CI-friendly: fewer cases
#[cfg(not(feature = "slow_tests"))]
const FAST_CASES: u32 = 100;

// Nightly: comprehensive
#[cfg(feature = "slow_tests")]
const THOROUGH_CASES: u32 = 2000;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: if cfg!(feature = "slow_tests") { 2000 } else { 100 },
        max_shrink_iters: 10000,
        ..ProptestConfig::default()
    })]
    // ...
}
```

---

## File Organization

```
crates/solverang/tests/
  property_tests.rs                         -- Existing (keep as-is)
  property_tests_v3/
    mod.rs
    generational_ids.rs                     -- Properties 1, 2
    pipeline_idempotency.rs                 -- Property 3
    reduce_invariant.rs                     -- Property 4
    sketch2d_symmetry.rs                    -- Property 5
    sketch2d_jacobian.rs                    -- Property 6 (16 constraint types)
    sketch3d_properties.rs                  -- Property 7
    assembly_quaternion.rs                  -- Property 8
    decomposition_stability.rs              -- Property 9
    redundancy_detection.rs                 -- Property 10
    composition.rs                          -- Property 15
    residual_scaling.rs                     -- Property 16
    determinism.rs                          -- Property 20
  property_tests_legacy/
    idempotency_determinism.rs              -- Properties 11, 12
    jtj_symmetry.rs                         -- Property 13
    lm_monotonicity.rs                      -- Property 14
    central_diff.rs                         -- Property 17
    decomposition.rs                        -- Property 18
    sparse_pattern.rs                       -- Property 19
```

## Estimated Effort

| Task | Time |
|------|------|
| Generational ID + ParamStore properties (1, 2) | 2-3 hours |
| Pipeline idempotency (3) | 2 hours |
| Reduce pass invariant (4) | 2-3 hours |
| Sketch2D symmetry -- 16 types (5) | 3-4 hours |
| Sketch2D Jacobian verification -- 16 types (6) | 4-5 hours |
| Sketch3D properties (7) | 2-3 hours |
| Assembly quaternion (8) | 2 hours |
| Decomposition stability (9) | 2-3 hours |
| Redundancy detection (10) | 2 hours |
| Legacy properties (11-14, 17-19) | 6-8 hours |
| V3 composition + scaling + determinism (15, 16, 20) | 3-4 hours |
| Improved strategies and config | 1 hour |
| **Total** | **~32-42 hours** |
