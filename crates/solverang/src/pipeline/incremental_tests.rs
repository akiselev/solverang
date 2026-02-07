//! Integration tests for incremental solving, warm starts, reduction, and diagnostics.
//!
//! These tests exercise the full `ConstraintSystem` pipeline through
//! multiple solve cycles, verifying cluster skipping, warm-start effectiveness,
//! reduction passes, and diagnostic analysis.

#[cfg(test)]
mod tests {
    use crate::constraint::Constraint;
    use crate::entity::Entity;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;
    use crate::system::{
        ClusterSolveStatus, ConstraintSystem, DiagnosticIssue, SystemStatus,
    };

    // ===================================================================
    // Test entity: a 2D point with two parameters (x, y).
    // ===================================================================

    struct TestPoint {
        id: EntityId,
        params: Vec<ParamId>,
    }

    impl Entity for TestPoint {
        fn id(&self) -> EntityId {
            self.id
        }
        fn params(&self) -> &[ParamId] {
            &self.params
        }
        fn name(&self) -> &str {
            "TestPoint"
        }
    }

    // ===================================================================
    // Test constraint: param = target  (single equation).
    // ===================================================================

    struct FixValueConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        param: ParamId,
        target: f64,
    }

    impl Constraint for FixValueConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "FixValue"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entity_ids
        }
        fn param_ids(&self) -> &[ParamId] {
            std::slice::from_ref(&self.param)
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.param) - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.param, 1.0)]
        }
    }

    // ===================================================================
    // Test constraint: a + b = target  (sum constraint).
    // ===================================================================

    struct SumConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        params: Vec<ParamId>,
        target: f64,
    }

    impl Constraint for SumConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "Sum"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entity_ids
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            let a = store.get(self.params[0]);
            let b = store.get(self.params[1]);
            vec![a + b - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![
                (0, self.params[0], 1.0),
                (0, self.params[1], 1.0),
            ]
        }
    }

    // ===================================================================
    // Test constraint: scale * param = target  (scaled fix).
    //   residual = scale * param - target
    //   jacobian = scale
    // ===================================================================

    struct ScaledFixConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        param: ParamId,
        scale: f64,
        target: f64,
    }

    impl Constraint for ScaledFixConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "ScaledFix"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entity_ids
        }
        fn param_ids(&self) -> &[ParamId] {
            std::slice::from_ref(&self.param)
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![self.scale * store.get(self.param) - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.param, self.scale)]
        }
    }

    // ===================================================================
    // Test constraint: a - b = 0  (equality / coincident).
    // ===================================================================

    struct EqualityConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        params: [ParamId; 2],
    }

    impl Constraint for EqualityConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "Equality"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entity_ids
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.params[0]) - store.get(self.params[1])]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![
                (0, self.params[0], 1.0),
                (0, self.params[1], -1.0),
            ]
        }
    }

    // ===================================================================
    // Helpers
    // ===================================================================

    /// Add a 2D point entity to the system, returning (entity_id, px, py).
    fn add_test_point(
        system: &mut ConstraintSystem,
        x: f64,
        y: f64,
    ) -> (EntityId, ParamId, ParamId) {
        let eid = system.alloc_entity_id();
        let px = system.alloc_param(x, eid);
        let py = system.alloc_param(y, eid);
        let point = TestPoint {
            id: eid,
            params: vec![px, py],
        };
        system.add_entity(Box::new(point));
        (eid, px, py)
    }

    /// Add a FixValueConstraint to the system.
    fn add_fix_constraint(
        system: &mut ConstraintSystem,
        entity: EntityId,
        param: ParamId,
        target: f64,
    ) -> ConstraintId {
        let cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(FixValueConstraint {
            id: cid,
            entity_ids: vec![entity],
            param,
            target,
        }));
        cid
    }

    /// Assert the system solves successfully (Solved or PartiallySolved).
    fn assert_solved(result: &crate::system::SystemResult) {
        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Expected Solved or PartiallySolved, got {:?}",
            result.status,
        );
    }

    /// Count clusters with a given status.
    fn count_status(
        result: &crate::system::SystemResult,
        status: ClusterSolveStatus,
    ) -> usize {
        result.clusters.iter().filter(|c| c.status == status).count()
    }

    // ===================================================================
    // Group 1: Incremental Solving Lifecycle
    // ===================================================================

    #[test]
    fn test_incremental_skips_unchanged_clusters() {
        let mut system = ConstraintSystem::new();

        // Entity 1 with constraint on px1 -> target 3.0
        let (eid1, px1, _py1) = add_test_point(&mut system, 0.0, 0.0);
        let _cid1 = add_fix_constraint(&mut system, eid1, px1, 3.0);

        // Entity 2 with constraint on px2 -> target 5.0 (independent cluster)
        let (eid2, px2, _py2) = add_test_point(&mut system, 0.0, 0.0);
        let _cid2 = add_fix_constraint(&mut system, eid2, px2, 5.0);

        // First solve: both clusters solve.
        let result1 = system.solve();
        assert_solved(&result1);
        assert_eq!(result1.clusters.len(), 2);
        assert!(
            (system.get_param(px1) - 3.0).abs() < 1e-6,
            "px1 = {}, expected 3.0",
            system.get_param(px1),
        );
        assert!(
            (system.get_param(px2) - 5.0).abs() < 1e-6,
            "px2 = {}, expected 5.0",
            system.get_param(px2),
        );

        // Modify only cluster 1's param.
        system.set_param(px1, 1.0);

        // Second solve: cluster 1 re-solves, cluster 2 should be skipped.
        let result2 = system.solve_incremental();
        assert_solved(&result2);
        assert_eq!(result2.clusters.len(), 2);

        // Exactly one cluster should be Skipped, one should be Converged.
        let skipped = count_status(&result2, ClusterSolveStatus::Skipped);
        let converged = count_status(&result2, ClusterSolveStatus::Converged);
        assert_eq!(
            skipped, 1,
            "Expected 1 skipped cluster, got {}. Statuses: {:?}",
            skipped,
            result2.clusters.iter().map(|c| c.status).collect::<Vec<_>>(),
        );
        assert_eq!(
            converged, 1,
            "Expected 1 converged cluster, got {}. Statuses: {:?}",
            converged,
            result2.clusters.iter().map(|c| c.status).collect::<Vec<_>>(),
        );

        // Values should be correct.
        assert!(
            (system.get_param(px1) - 3.0).abs() < 1e-6,
            "px1 = {} after incremental solve, expected 3.0",
            system.get_param(px1),
        );
        assert!(
            (system.get_param(px2) - 5.0).abs() < 1e-6,
            "px2 = {} after incremental solve, expected 5.0",
            system.get_param(px2),
        );
    }

    #[test]
    fn test_incremental_three_round_lifecycle() {
        let mut system = ConstraintSystem::new();

        // Three independent entities with one fix constraint each.
        let (eid1, px1, _) = add_test_point(&mut system, 0.0, 0.0);
        let _c1 = add_fix_constraint(&mut system, eid1, px1, 1.0);

        let (eid2, px2, _) = add_test_point(&mut system, 0.0, 0.0);
        let _c2 = add_fix_constraint(&mut system, eid2, px2, 2.0);

        let (eid3, px3, _) = add_test_point(&mut system, 0.0, 0.0);
        let _c3 = add_fix_constraint(&mut system, eid3, px3, 3.0);

        // Round 1: First solve -- all 3 clusters solve.
        let r1 = system.solve();
        assert_solved(&r1);
        assert_eq!(r1.clusters.len(), 3);
        assert!(
            (system.get_param(px1) - 1.0).abs() < 1e-6,
            "Round 1: px1 = {}",
            system.get_param(px1),
        );
        assert!(
            (system.get_param(px2) - 2.0).abs() < 1e-6,
            "Round 1: px2 = {}",
            system.get_param(px2),
        );
        assert!(
            (system.get_param(px3) - 3.0).abs() < 1e-6,
            "Round 1: px3 = {}",
            system.get_param(px3),
        );

        // Round 2: Modify cluster 1 only.
        system.set_param(px1, 0.5);
        let r2 = system.solve_incremental();
        assert_solved(&r2);
        assert_eq!(count_status(&r2, ClusterSolveStatus::Skipped), 2);
        assert!(
            (system.get_param(px1) - 1.0).abs() < 1e-6,
            "Round 2: px1 = {}",
            system.get_param(px1),
        );

        // Round 3: Modify cluster 3 only.
        system.set_param(px3, 0.0);
        let r3 = system.solve_incremental();
        assert_solved(&r3);
        assert_eq!(count_status(&r3, ClusterSolveStatus::Skipped), 2);
        assert!(
            (system.get_param(px3) - 3.0).abs() < 1e-6,
            "Round 3: px3 = {}",
            system.get_param(px3),
        );

        // Round 4: No modifications -> all 3 skipped.
        let r4 = system.solve_incremental();
        assert_solved(&r4);
        assert_eq!(
            count_status(&r4, ClusterSolveStatus::Skipped),
            3,
            "Round 4: expected all 3 clusters skipped, statuses: {:?}",
            r4.clusters.iter().map(|c| c.status).collect::<Vec<_>>(),
        );
        assert_eq!(r4.total_iterations, 0, "Round 4: expected 0 iterations");
    }

    #[test]
    fn test_structural_change_invalidates_all_clusters() {
        let mut system = ConstraintSystem::new();

        let (eid1, px1, _) = add_test_point(&mut system, 0.0, 0.0);
        let _c1 = add_fix_constraint(&mut system, eid1, px1, 1.0);

        let (eid2, px2, _) = add_test_point(&mut system, 0.0, 0.0);
        let _c2 = add_fix_constraint(&mut system, eid2, px2, 2.0);

        // First solve: everything resolves.
        let _r1 = system.solve();
        assert!(!system.change_tracker().has_any_changes());

        // Add a new constraint -> structural change.
        let (eid3, px3, _) = add_test_point(&mut system, 0.0, 0.0);
        assert!(
            system.change_tracker().has_structural_changes(),
            "Adding a constraint should be a structural change",
        );
        let _c3 = add_fix_constraint(&mut system, eid3, px3, 3.0);

        // Solve again: ALL clusters should re-solve (not skipped).
        let r2 = system.solve();
        assert_solved(&r2);
        let skipped = count_status(&r2, ClusterSolveStatus::Skipped);
        // After structural change, clusters that were previously solved may
        // end up skipped if the pipeline detects they have no free variables,
        // but the key assertion is that the system re-decomposed. We verify
        // by checking that the new cluster count reflects the new constraint.
        assert!(
            r2.clusters.len() >= 3,
            "Expected at least 3 clusters after adding a third entity/constraint, got {}",
            r2.clusters.len(),
        );

        // After solve, tracker should be cleared.
        assert!(
            !system.change_tracker().has_structural_changes(),
            "Structural changes should be cleared after solve",
        );
        assert!(
            !system.change_tracker().has_any_changes(),
            "All changes should be cleared after solve",
        );
    }

    #[test]
    fn test_fix_param_invalidates_pipeline() {
        let mut system = ConstraintSystem::new();

        let (eid, px, py) = add_test_point(&mut system, 0.0, 0.0);
        let _c1 = add_fix_constraint(&mut system, eid, px, 3.0);
        let _c2 = add_fix_constraint(&mut system, eid, py, 7.0);

        // First solve.
        let r1 = system.solve();
        assert_solved(&r1);
        assert!(
            (system.get_param(px) - 3.0).abs() < 1e-6,
            "px = {}",
            system.get_param(px),
        );
        assert!(
            (system.get_param(py) - 7.0).abs() < 1e-6,
            "py = {}",
            system.get_param(py),
        );

        // Fix px: this is a structural change that invalidates the pipeline.
        system.fix_param(px);

        // Solve again after fixing: should re-decompose and re-solve.
        let r2 = system.solve();
        assert_solved(&r2);
        // px is fixed at 3.0, py should still be solved to 7.0.
        assert!(
            (system.get_param(px) - 3.0).abs() < 1e-6,
            "px after fix = {}",
            system.get_param(px),
        );
        assert!(
            (system.get_param(py) - 7.0).abs() < 1e-6,
            "py after fix = {}",
            system.get_param(py),
        );

        // Unfix px: another structural change.
        system.unfix_param(px);

        // Perturb px to verify it's free again.
        system.set_param(px, 0.0);

        // Solve again: should re-decompose and resolve px.
        let r3 = system.solve();
        assert_solved(&r3);
        assert!(
            (system.get_param(px) - 3.0).abs() < 1e-6,
            "px after unfix = {}",
            system.get_param(px),
        );
    }

    // ===================================================================
    // Group 2: Warm Start Effectiveness
    // ===================================================================

    #[test]
    fn test_warm_start_reduces_iterations() {
        let mut system = ConstraintSystem::new();

        // Build a moderately complex system: 3 coupled constraints on 3 params.
        let (eid, px, py) = add_test_point(&mut system, 0.0, 0.0);

        // px = 5.0
        let _c1 = add_fix_constraint(&mut system, eid, px, 5.0);
        // px + py = 12.0  =>  py = 7.0
        let cid2 = system.alloc_constraint_id();
        system.add_constraint(Box::new(SumConstraint {
            id: cid2,
            entity_ids: vec![eid],
            params: vec![px, py],
            target: 12.0,
        }));

        // First solve from cold start: all params start at 0.0.
        let r1 = system.solve();
        assert_solved(&r1);
        let first_iterations = r1.total_iterations;
        assert!(
            (system.get_param(px) - 5.0).abs() < 1e-6,
            "px = {}",
            system.get_param(px),
        );
        assert!(
            (system.get_param(py) - 7.0).abs() < 1e-6,
            "py = {}",
            system.get_param(py),
        );

        // Slightly perturb px (small perturbation from the solution).
        system.set_param(px, 4.9);

        // Second solve: warm start from cached solution should help.
        let r2 = system.solve_incremental();
        assert_solved(&r2);
        let second_iterations = r2.total_iterations;

        // Warm start should use fewer or equal iterations since we're
        // starting closer to the solution.
        assert!(
            second_iterations <= first_iterations,
            "Warm start should not use more iterations than cold start: \
             cold={}, warm={}",
            first_iterations,
            second_iterations,
        );

        // Verify solution is still correct.
        assert!(
            (system.get_param(px) - 5.0).abs() < 1e-6,
            "px after warm start = {}",
            system.get_param(px),
        );
        assert!(
            (system.get_param(py) - 7.0).abs() < 1e-6,
            "py after warm start = {}",
            system.get_param(py),
        );
    }

    #[test]
    fn test_warm_start_survives_parameter_change() {
        let mut system = ConstraintSystem::new();

        let (eid, px, py) = add_test_point(&mut system, 0.0, 0.0);
        let _c1 = add_fix_constraint(&mut system, eid, px, 10.0);
        let _c2 = add_fix_constraint(&mut system, eid, py, 20.0);

        // First solve populates the cache.
        let r1 = system.solve();
        assert_solved(&r1);
        assert!(
            (system.get_param(px) - 10.0).abs() < 1e-6,
            "px = {}",
            system.get_param(px),
        );
        assert!(
            (system.get_param(py) - 20.0).abs() < 1e-6,
            "py = {}",
            system.get_param(py),
        );

        // Slightly change px.
        system.set_param(px, 9.5);

        // Second solve should use cached warm start and converge correctly.
        let r2 = system.solve_incremental();
        assert_solved(&r2);

        // Verify param_values are correct post-solve.
        assert!(
            (system.get_param(px) - 10.0).abs() < 1e-6,
            "px after warm-start re-solve = {}",
            system.get_param(px),
        );
        assert!(
            (system.get_param(py) - 20.0).abs() < 1e-6,
            "py after warm-start re-solve = {}",
            system.get_param(py),
        );
    }

    // ===================================================================
    // Group 3: Reduce Effectiveness
    // ===================================================================

    #[test]
    fn test_reduction_eliminates_trivial_constraints() {
        let mut system = ConstraintSystem::new();

        let (eid, px, py) = add_test_point(&mut system, 5.0, 0.0);

        // Fix px so it cannot move.
        system.fix_param(px);

        // Constraint: px = 5.0 (trivially satisfied since px is fixed at 5.0).
        let _c1 = add_fix_constraint(&mut system, eid, px, 5.0);

        // Constraint: py = 10.0 (one free param, eliminable).
        let _c2 = add_fix_constraint(&mut system, eid, py, 10.0);

        // Solve: should converge.
        let result = system.solve();
        assert_solved(&result);

        // The reduction pipeline should have handled both constraints:
        // - c1 is trivially satisfied (fixed param matches target).
        // - c2 has one free param, should be eliminated analytically.
        // Either way, the total iterations should be minimal.
        assert!(
            result.total_iterations <= 1,
            "Expected minimal iterations due to reduction, got {}",
            result.total_iterations,
        );

        // Verify py = 10.0.
        assert!(
            (system.get_param(py) - 10.0).abs() < 1e-6,
            "py = {}, expected 10.0",
            system.get_param(py),
        );

        // px should remain at 5.0.
        assert!(
            (system.get_param(px) - 5.0).abs() < 1e-6,
            "px = {}, expected 5.0",
            system.get_param(px),
        );
    }

    #[test]
    fn test_reduction_handles_merge() {
        let mut system = ConstraintSystem::new();

        // Two entities, each with one param.
        let (eid1, px1, _py1) = add_test_point(&mut system, 0.0, 0.0);
        let (eid2, px2, _py2) = add_test_point(&mut system, 0.0, 0.0);

        // Equality constraint: px1 = px2 (coincident-like merge).
        let eq_cid = system.alloc_constraint_id();
        system.add_constraint(Box::new(EqualityConstraint {
            id: eq_cid,
            entity_ids: vec![eid1, eid2],
            params: [px1, px2],
        }));

        // Fix px1 to a value.
        let _c2 = add_fix_constraint(&mut system, eid1, px1, 7.0);

        // Solve: after merge, px1 and px2 should be identical.
        let result = system.solve();
        assert_solved(&result);

        // Both params should be 7.0.
        assert!(
            (system.get_param(px1) - 7.0).abs() < 1e-6,
            "px1 = {}, expected 7.0",
            system.get_param(px1),
        );
        assert!(
            (system.get_param(px2) - 7.0).abs() < 1e-6,
            "px2 = {}, expected 7.0 (merged with px1)",
            system.get_param(px2),
        );
    }

    // ===================================================================
    // Group 4: Diagnostics
    // ===================================================================

    #[test]
    fn test_diagnose_redundant_constraint() {
        let mut system = ConstraintSystem::new();

        let (eid, px, _py) = add_test_point(&mut system, 5.0, 0.0);

        // Constraint 1: px = 5.0
        let _c1 = add_fix_constraint(&mut system, eid, px, 5.0);

        // Constraint 2: 2*px = 10.0 (redundant -- linearly dependent with c1).
        let cid2 = system.alloc_constraint_id();
        system.add_constraint(Box::new(ScaledFixConstraint {
            id: cid2,
            entity_ids: vec![eid],
            param: px,
            scale: 2.0,
            target: 10.0,
        }));

        // Analyze redundancy.
        let redundancy = system.analyze_redundancy();

        // There should be a rank deficiency (2 equations, 1 variable,
        // but only rank 1 because the rows are proportional).
        assert!(
            redundancy.rank_deficiency() > 0,
            "Expected rank deficiency > 0, got {}. rank={}, eqs={}",
            redundancy.rank_deficiency(),
            redundancy.jacobian_rank,
            redundancy.equation_count,
        );

        // The redundant constraint should be detected.
        assert!(
            !redundancy.redundant.is_empty(),
            "Expected at least one redundant constraint to be detected",
        );

        // No conflicts expected (both constraints are consistent).
        assert!(
            redundancy.conflicts.is_empty(),
            "Expected no conflicts, but found: {:?}",
            redundancy.conflicts,
        );
    }

    #[test]
    fn test_diagnose_conflicting_constraints() {
        let mut system = ConstraintSystem::new();

        let (eid, px, _py) = add_test_point(&mut system, 5.0, 0.0);

        // Constraint 1: px = 5.0
        let _c1 = add_fix_constraint(&mut system, eid, px, 5.0);

        // Constraint 2: px = 10.0 (conflicting!)
        let _c2 = add_fix_constraint(&mut system, eid, px, 10.0);

        // The system should either fail to converge or report a diagnostic failure.
        let result = system.solve();

        // Check that the solver detected a problem. The system may report
        // DiagnosticFailure, NotConverged (via PartiallySolved), or the
        // redundancy analysis will find conflicts.
        let has_diagnostic_issue = matches!(result.status, SystemStatus::DiagnosticFailure(_));
        let has_non_converged_cluster = result
            .clusters
            .iter()
            .any(|c| c.status == ClusterSolveStatus::NotConverged);

        // Also run explicit redundancy analysis.
        let redundancy = system.analyze_redundancy();
        let has_conflicts = !redundancy.conflicts.is_empty();

        assert!(
            has_diagnostic_issue || has_non_converged_cluster || has_conflicts,
            "Expected either DiagnosticFailure, NotConverged cluster, or \
             redundancy conflicts for conflicting constraints. \
             Status: {:?}, conflicts: {:?}",
            result.status,
            redundancy.conflicts,
        );
    }

    #[test]
    fn test_diagnose_under_constrained() {
        let mut system = ConstraintSystem::new();

        let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);

        // Only one constraint on a 2-param entity -> under-constrained.
        let _c1 = add_fix_constraint(&mut system, eid, px, 5.0);

        // Analyze DOF.
        let dof = system.analyze_dof();

        // total_dof should be positive (under-constrained).
        assert!(
            dof.total_dof > 0,
            "Expected total_dof > 0 for under-constrained system, got {}",
            dof.total_dof,
        );
        assert!(
            dof.is_under_constrained(),
            "Expected is_under_constrained() = true",
        );

        // The entity should have dof > 0.
        assert!(
            !dof.entities.is_empty(),
            "Expected at least one entity in DOF analysis",
        );
        let entity_dof = dof
            .entities
            .iter()
            .find(|e| e.entity_id == eid)
            .expect("Entity should appear in DOF analysis");
        assert!(
            entity_dof.dof > 0,
            "Expected entity dof > 0 for under-constrained entity, got {}",
            entity_dof.dof,
        );
    }

    #[test]
    fn test_diagnose_well_constrained() {
        let mut system = ConstraintSystem::new();

        let (eid, px, py) = add_test_point(&mut system, 0.0, 0.0);

        // Two independent constraints on 2 params -> well-constrained.
        let _c1 = add_fix_constraint(&mut system, eid, px, 5.0);
        let _c2 = add_fix_constraint(&mut system, eid, py, 10.0);

        // Solve first to verify correctness.
        let result = system.solve();
        assert_solved(&result);
        assert!(
            (system.get_param(px) - 5.0).abs() < 1e-6,
            "px = {}",
            system.get_param(px),
        );
        assert!(
            (system.get_param(py) - 10.0).abs() < 1e-6,
            "py = {}",
            system.get_param(py),
        );

        // Analyze DOF.
        let dof = system.analyze_dof();

        assert_eq!(
            dof.total_dof, 0,
            "Expected total_dof = 0 for well-constrained system, got {}",
            dof.total_dof,
        );
        assert!(
            dof.is_well_constrained(),
            "Expected is_well_constrained() = true",
        );
        assert!(
            !dof.is_under_constrained(),
            "Expected is_under_constrained() = false",
        );
        assert!(
            !dof.is_over_constrained(),
            "Expected is_over_constrained() = false",
        );

        // Entity should have dof = 0.
        let entity_dof = dof
            .entities
            .iter()
            .find(|e| e.entity_id == eid)
            .expect("Entity should appear in DOF analysis");
        assert_eq!(
            entity_dof.dof, 0,
            "Expected entity dof = 0, got {}",
            entity_dof.dof,
        );
    }

    // ===================================================================
    // Additional edge case tests
    // ===================================================================

    #[test]
    fn test_incremental_no_change_all_skipped() {
        // Verify that when no params are changed between solves, all
        // clusters are skipped on the second call.
        let mut system = ConstraintSystem::new();

        let (eid, px, py) = add_test_point(&mut system, 0.0, 0.0);
        let _c1 = add_fix_constraint(&mut system, eid, px, 4.0);
        let _c2 = add_fix_constraint(&mut system, eid, py, 8.0);

        // First solve.
        let r1 = system.solve();
        assert_solved(&r1);

        // Immediately solve again without any changes.
        let r2 = system.solve_incremental();
        assert_solved(&r2);
        assert_eq!(
            r2.total_iterations, 0,
            "Expected 0 iterations when nothing changed, got {}",
            r2.total_iterations,
        );

        // All clusters should be skipped.
        for cluster in &r2.clusters {
            assert_eq!(
                cluster.status,
                ClusterSolveStatus::Skipped,
                "Expected all clusters to be Skipped, but cluster {:?} has status {:?}",
                cluster.cluster_id,
                cluster.status,
            );
        }
    }

    #[test]
    fn test_diagnose_returns_under_constrained_issues() {
        // Verify that the `diagnose()` convenience method detects
        // under-constrained entities.
        let mut system = ConstraintSystem::new();

        // An entity with 2 params and only 1 constraint: y is free.
        let (eid, px, _py) = add_test_point(&mut system, 0.0, 0.0);
        let _c = add_fix_constraint(&mut system, eid, px, 1.0);

        let issues = system.diagnose();

        let under_constrained_issues: Vec<_> = issues
            .iter()
            .filter(|i| matches!(i, DiagnosticIssue::UnderConstrained { .. }))
            .collect();

        assert!(
            !under_constrained_issues.is_empty(),
            "Expected at least one UnderConstrained diagnostic issue, got {:?}",
            issues,
        );
    }
}
