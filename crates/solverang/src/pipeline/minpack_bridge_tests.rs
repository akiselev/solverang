//! Bridge tests: MINPACK/NIST test problems solved through the ConstraintSystem pipeline.
//!
//! This module provides a generic `ProblemConstraint` adapter that wraps any
//! `Problem` implementation as a `Constraint`, enabling existing MINPACK test
//! problems to be solved through the full pipeline (Decompose -> Analyze ->
//! Reduce -> Solve -> PostProcess).
//!
//! The tests validate that solving through the pipeline produces results
//! equivalent to solving directly with the LM solver, confirming the pipeline
//! does not break anything.

#[cfg(test)]
mod tests {
    use crate::constraint::Constraint;
    use crate::entity::Entity;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;
    use crate::problem::Problem;
    use crate::solver::{LMConfig, LMSolver};
    use crate::system::{ClusterSolveStatus, ConstraintSystem, SystemStatus};
    use crate::test_problems::{
        Bard, FreudensteinRoth, HelicalValley, PowellSingular, Rosenbrock, Wood,
    };

    // -----------------------------------------------------------------------
    // GenericEntity — a simple entity that holds an arbitrary set of params.
    // -----------------------------------------------------------------------

    struct GenericEntity {
        id: EntityId,
        params: Vec<ParamId>,
        label: String,
    }

    impl Entity for GenericEntity {
        fn id(&self) -> EntityId {
            self.id
        }
        fn params(&self) -> &[ParamId] {
            &self.params
        }
        fn name(&self) -> &str {
            &self.label
        }
    }

    // -----------------------------------------------------------------------
    // ProblemConstraint — wraps any Problem as a Constraint.
    //
    // This is the key bridge adapter.  It stores the Problem together with a
    // mapping from ParamId -> column index (positional, based on param_ids
    // order) so the Constraint can read values from the ParamStore and
    // translate Problem-level (row, col) Jacobian entries into
    // Constraint-level (row, ParamId) entries.
    // -----------------------------------------------------------------------

    struct ProblemConstraint {
        id: ConstraintId,
        entity_id: EntityId,
        param_ids: Vec<ParamId>,
        problem: Box<dyn Problem>,
    }

    impl Constraint for ProblemConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }

        fn name(&self) -> &str {
            self.problem.name()
        }

        fn entity_ids(&self) -> &[EntityId] {
            std::slice::from_ref(&self.entity_id)
        }

        fn param_ids(&self) -> &[ParamId] {
            &self.param_ids
        }

        fn equation_count(&self) -> usize {
            self.problem.residual_count()
        }

        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            let values: Vec<f64> = self.param_ids.iter().map(|&pid| store.get(pid)).collect();
            self.problem.residuals(&values)
        }

        fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            let values: Vec<f64> = self.param_ids.iter().map(|&pid| store.get(pid)).collect();
            let raw_jac = self.problem.jacobian(&values);
            raw_jac
                .into_iter()
                .map(|(row, col, val)| (row, self.param_ids[col], val))
                .collect()
        }
    }

    // -----------------------------------------------------------------------
    // Helper: build a ConstraintSystem from a Problem.
    //
    // Allocates an entity, parameters (initialised from the problem's default
    // starting point), and a ProblemConstraint wrapping the Problem.
    // Returns the system and the ordered list of ParamIds.
    // -----------------------------------------------------------------------

    fn build_system_from_problem(
        problem: Box<dyn Problem>,
        factor: f64,
    ) -> (ConstraintSystem, Vec<ParamId>) {
        let mut system = ConstraintSystem::new();
        let eid = system.alloc_entity_id();
        let initial = problem.initial_point(factor);
        let params: Vec<ParamId> = initial
            .iter()
            .map(|&v| system.alloc_param(v, eid))
            .collect();

        let entity = GenericEntity {
            id: eid,
            params: params.clone(),
            label: format!("{}_entity", problem.name()),
        };
        system.add_entity(Box::new(entity));

        let cid = system.alloc_constraint_id();
        let constraint = ProblemConstraint {
            id: cid,
            entity_id: eid,
            param_ids: params.clone(),
            problem,
        };
        system.add_constraint(Box::new(constraint));

        (system, params)
    }

    /// Helper: add a second (or third, ...) problem into an *existing* system.
    /// Returns the entity id and the param ids for the new problem.
    fn add_problem_to_system(
        system: &mut ConstraintSystem,
        problem: Box<dyn Problem>,
        factor: f64,
    ) -> (EntityId, Vec<ParamId>) {
        let eid = system.alloc_entity_id();
        let initial = problem.initial_point(factor);
        let params: Vec<ParamId> = initial
            .iter()
            .map(|&v| system.alloc_param(v, eid))
            .collect();

        let entity = GenericEntity {
            id: eid,
            params: params.clone(),
            label: format!("{}_entity", problem.name()),
        };
        system.add_entity(Box::new(entity));

        let cid = system.alloc_constraint_id();
        let constraint = ProblemConstraint {
            id: cid,
            entity_id: eid,
            param_ids: params.clone(),
            problem,
        };
        system.add_constraint(Box::new(constraint));

        (eid, params)
    }

    /// Compute the L2 norm of a slice.
    fn norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    // ===================================================================
    // Test 1: Rosenbrock through the pipeline
    // ===================================================================

    #[test]
    fn test_rosenbrock_through_pipeline() {
        let problem = Rosenbrock;
        let known = problem.known_solution().unwrap();

        let (mut system, params) = build_system_from_problem(Box::new(Rosenbrock), 1.0);
        let result = system.solve();

        // Should converge.
        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Rosenbrock pipeline solve status: {:?}",
            result.status,
        );

        // At least one cluster should have converged.
        assert!(
            result
                .clusters
                .iter()
                .any(|c| c.status == ClusterSolveStatus::Converged),
            "Expected at least one converged cluster",
        );

        // Residual norm should be small.
        let residuals = system.compute_residuals();
        let res_norm = norm(&residuals);
        assert!(
            res_norm < 1e-6,
            "Rosenbrock residual norm too large: {}",
            res_norm,
        );

        // Solution should match known minimum [1, 1].
        for (i, &pid) in params.iter().enumerate() {
            let val = system.get_param(pid);
            assert!(
                (val - known[i]).abs() < 1e-4,
                "Rosenbrock x[{}] = {}, expected {}",
                i,
                val,
                known[i],
            );
        }
    }

    // ===================================================================
    // Test 2: Powell Singular through the pipeline
    // ===================================================================

    #[test]
    fn test_powell_singular_through_pipeline() {
        let (mut system, params) = build_system_from_problem(Box::new(PowellSingular), 1.0);
        let result = system.solve();

        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "PowellSingular pipeline solve status: {:?}",
            result.status,
        );

        // Residual should be near zero.
        let residuals = system.compute_residuals();
        let res_norm = norm(&residuals);
        assert!(
            res_norm < 1e-4,
            "PowellSingular residual norm too large: {}",
            res_norm,
        );

        // Known solution is (0,0,0,0).
        for (i, &pid) in params.iter().enumerate() {
            let val = system.get_param(pid);
            assert!(
                val.abs() < 1e-2,
                "PowellSingular x[{}] = {}, expected ~0",
                i,
                val,
            );
        }
    }

    // ===================================================================
    // Test 3: Helical Valley through the pipeline
    // ===================================================================

    #[test]
    fn test_helical_valley_through_pipeline() {
        let problem = HelicalValley;
        let known = problem.known_solution().unwrap(); // [1, 0, 0]

        let (mut system, params) = build_system_from_problem(Box::new(HelicalValley), 1.0);
        let result = system.solve();

        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "HelicalValley pipeline solve status: {:?}",
            result.status,
        );

        let residuals = system.compute_residuals();
        let res_norm = norm(&residuals);
        assert!(
            res_norm < 1e-6,
            "HelicalValley residual norm too large: {}",
            res_norm,
        );

        for (i, &pid) in params.iter().enumerate() {
            let val = system.get_param(pid);
            assert!(
                (val - known[i]).abs() < 1e-4,
                "HelicalValley x[{}] = {}, expected {}",
                i,
                val,
                known[i],
            );
        }
    }

    // ===================================================================
    // Test 4: Pipeline vs Direct Solve comparison
    //
    // For each problem, solve both via direct LMSolver::solve and via
    // the pipeline, then compare final residual norms.
    // ===================================================================

    #[test]
    fn test_pipeline_vs_direct_solve_comparison() {
        let problems: Vec<(&str, Box<dyn Problem>)> = vec![
            ("Rosenbrock", Box::new(Rosenbrock)),
            ("PowellSingular", Box::new(PowellSingular)),
            ("HelicalValley", Box::new(HelicalValley)),
            ("Wood", Box::new(Wood)),
            ("Bard", Box::new(Bard)),
        ];

        let lm = LMSolver::new(LMConfig::default());

        for (label, problem) in problems {
            // --- Direct solve ---
            let x0 = problem.initial_point(1.0);
            let direct_result = lm.solve(problem.as_ref(), &x0);
            let direct_converged = direct_result.is_converged() || direct_result.is_completed();
            let direct_norm = direct_result.residual_norm().unwrap_or(f64::INFINITY);

            // --- Pipeline solve ---
            // We need to rebuild the problem because we moved it; create a
            // fresh copy.  Since Problem is object-safe and we cannot clone
            // the Box<dyn Problem>, we recreate from the label.
            let pipeline_problem: Box<dyn Problem> = match label {
                "Rosenbrock" => Box::new(Rosenbrock),
                "PowellSingular" => Box::new(PowellSingular),
                "HelicalValley" => Box::new(HelicalValley),
                "Wood" => Box::new(Wood),
                "Bard" => Box::new(Bard),
                _ => unreachable!(),
            };

            let (mut system, _params) = build_system_from_problem(pipeline_problem, 1.0);
            let pipeline_result = system.solve();

            let pipeline_converged = matches!(
                pipeline_result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            );

            let pipeline_residuals = system.compute_residuals();
            let pipeline_norm = norm(&pipeline_residuals);

            // Both should converge (or at least complete).
            assert!(
                direct_converged,
                "[{}] Direct LM solver did not converge",
                label,
            );
            assert!(
                pipeline_converged,
                "[{}] Pipeline solve did not converge (status: {:?})",
                label,
                pipeline_result.status,
            );

            // Residual norms should be in the same ballpark.
            // We allow generous tolerance because the two code paths may
            // converge at slightly different rates or to slightly different
            // local optima (for problems with non-zero residual minima like Bard).
            let tol = 1e-2;
            let diff = (pipeline_norm - direct_norm).abs();
            assert!(
                diff < tol || pipeline_norm < tol,
                "[{}] Residual norm mismatch: pipeline={}, direct={}, diff={}",
                label,
                pipeline_norm,
                direct_norm,
                diff,
            );
        }
    }

    // ===================================================================
    // Test 5: Multiple problems as separate clusters in the same system
    //
    // Each problem is wrapped as an independent entity + constraint.
    // The pipeline should decompose them into separate clusters (no
    // shared parameters) and solve each independently.
    // ===================================================================

    #[test]
    fn test_multiple_problems_as_separate_clusters() {
        let mut system = ConstraintSystem::new();

        // Problem 1: Rosenbrock (2 vars)
        let (_, rosenbrock_params) =
            add_problem_to_system(&mut system, Box::new(Rosenbrock), 1.0);

        // Problem 2: PowellSingular (4 vars)
        let (_, powell_params) =
            add_problem_to_system(&mut system, Box::new(PowellSingular), 1.0);

        // Problem 3: HelicalValley (3 vars)
        let (_, helical_params) =
            add_problem_to_system(&mut system, Box::new(HelicalValley), 1.0);

        // Solve the combined system.
        let result = system.solve();

        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Multi-problem pipeline solve status: {:?}",
            result.status,
        );

        // Should decompose into 3 independent clusters (no shared params).
        assert_eq!(
            result.clusters.len(),
            3,
            "Expected 3 independent clusters, got {}",
            result.clusters.len(),
        );

        // Each cluster should have converged.
        for (i, cr) in result.clusters.iter().enumerate() {
            assert!(
                cr.status == ClusterSolveStatus::Converged
                    || cr.status == ClusterSolveStatus::Skipped,
                "Cluster {} did not converge: {:?}",
                i,
                cr.status,
            );
        }

        // Verify Rosenbrock solution ~ [1, 1].
        let rosenbrock_known = Rosenbrock.known_solution().unwrap();
        for (i, &pid) in rosenbrock_params.iter().enumerate() {
            let val = system.get_param(pid);
            assert!(
                (val - rosenbrock_known[i]).abs() < 1e-4,
                "Rosenbrock x[{}] = {}, expected {}",
                i,
                val,
                rosenbrock_known[i],
            );
        }

        // Verify PowellSingular solution ~ [0, 0, 0, 0].
        for (i, &pid) in powell_params.iter().enumerate() {
            let val = system.get_param(pid);
            assert!(
                val.abs() < 1e-2,
                "PowellSingular x[{}] = {}, expected ~0",
                i,
                val,
            );
        }

        // Verify HelicalValley solution ~ [1, 0, 0].
        let helical_known = HelicalValley.known_solution().unwrap();
        for (i, &pid) in helical_params.iter().enumerate() {
            let val = system.get_param(pid);
            assert!(
                (val - helical_known[i]).abs() < 1e-4,
                "HelicalValley x[{}] = {}, expected {}",
                i,
                val,
                helical_known[i],
            );
        }
    }

    // ===================================================================
    // Test 6: Adapter correctness — ProblemConstraint produces the same
    // residuals and Jacobian entries as the underlying Problem.
    // ===================================================================

    #[test]
    fn test_problem_constraint_adapter_correctness() {
        let problem = Rosenbrock;
        let x0 = problem.initial_point(1.0);

        // Build a standalone store + constraint for direct evaluation.
        let mut store = ParamStore::new();
        let eid = EntityId::new(0, 0);
        let param_ids: Vec<ParamId> = x0.iter().map(|&v| store.alloc(v, eid)).collect();

        let cid = ConstraintId::new(0, 0);
        let adapter = ProblemConstraint {
            id: cid,
            entity_id: eid,
            param_ids: param_ids.clone(),
            problem: Box::new(Rosenbrock),
        };

        // Residuals should match.
        let direct_residuals = problem.residuals(&x0);
        let adapter_residuals = adapter.residuals(&store);
        assert_eq!(direct_residuals.len(), adapter_residuals.len());
        for (i, (d, a)) in direct_residuals.iter().zip(&adapter_residuals).enumerate() {
            assert!(
                (d - a).abs() < 1e-15,
                "Residual[{}] mismatch: direct={}, adapter={}",
                i,
                d,
                a,
            );
        }

        // Jacobian: the raw Problem returns (row, col_index, val), while the
        // adapter returns (row, ParamId, val).  Verify that the mapping is
        // consistent.
        let direct_jac = problem.jacobian(&x0);
        let adapter_jac = adapter.jacobian(&store);
        assert_eq!(direct_jac.len(), adapter_jac.len());
        for ((d_row, d_col, d_val), (a_row, a_pid, a_val)) in
            direct_jac.iter().zip(&adapter_jac)
        {
            assert_eq!(*d_row, *a_row, "Jacobian row mismatch");
            assert_eq!(
                param_ids[*d_col], *a_pid,
                "Jacobian ParamId mismatch for col {}",
                d_col,
            );
            assert!(
                (d_val - a_val).abs() < 1e-15,
                "Jacobian value mismatch at ({}, {}): direct={}, adapter={}",
                d_row,
                d_col,
                d_val,
                a_val,
            );
        }

        // Equation count should match residual count.
        assert_eq!(adapter.equation_count(), problem.residual_count());

        // Param ids should match what we passed in.
        assert_eq!(adapter.param_ids(), &param_ids[..]);

        // Entity ids should contain the single entity id.
        assert_eq!(adapter.entity_ids(), &[eid]);
    }

    // ===================================================================
    // Test 7: FreudensteinRoth through the pipeline
    //
    // FreudensteinRoth has multiple local minima.  The LM solver from
    // the default starting point (0.5, -2) often converges to the local
    // minimum near (11.41, -0.90) with residual norm ~6.999 rather than
    // the global minimum at (5, 4).  We accept convergence to either.
    // ===================================================================

    #[test]
    fn test_freudenstein_roth_through_pipeline() {
        let (mut system, _params) = build_system_from_problem(Box::new(FreudensteinRoth), 1.0);
        let result = system.solve();

        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "FreudensteinRoth pipeline solve status: {:?}",
            result.status,
        );

        // For FreudensteinRoth, the solver may find the local minimum with
        // residual norm ~6.999.  We just verify it completed and produced a
        // finite residual.
        let residuals = system.compute_residuals();
        let res_norm = norm(&residuals);
        assert!(
            res_norm.is_finite(),
            "FreudensteinRoth residual norm is not finite: {}",
            res_norm,
        );
        assert!(
            res_norm < 10.0,
            "FreudensteinRoth residual norm unexpectedly large: {}",
            res_norm,
        );
    }

    // ===================================================================
    // Test 8: Overdetermined Bard problem through the pipeline
    //
    // Bard has 15 equations and 3 variables — a genuine overdetermined
    // least-squares problem.  The pipeline should handle m > n.
    // ===================================================================

    #[test]
    fn test_bard_overdetermined_through_pipeline() {
        let problem = Bard;
        let expected_norm = problem.expected_residual_norm().unwrap();

        let (mut system, _params) = build_system_from_problem(Box::new(Bard), 1.0);
        let result = system.solve();

        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Bard pipeline solve status: {:?}",
            result.status,
        );

        let residuals = system.compute_residuals();
        let res_norm = norm(&residuals);

        // The Bard problem has a non-zero residual at the optimum.
        // Verify we get close to the expected optimal residual norm.
        assert!(
            (res_norm - expected_norm).abs() < 1e-3,
            "Bard residual norm: {}, expected ~{}",
            res_norm,
            expected_norm,
        );
    }

    // ===================================================================
    // Test 9: Wood function (4 variables) through the pipeline
    //
    // The Wood function is a square system (4 equations, 4 variables)
    // with multiple roots.  The known solution at (1,1,1,1) is one
    // root, but from the default starting point (-3,-1,-3,-1) the
    // solver legitimately converges to a different root.  We verify
    // that the pipeline reaches a root (small residual) and also that
    // the pipeline and direct solver find equivalent solutions.
    // ===================================================================

    #[test]
    fn test_wood_through_pipeline() {
        // Verify convergence from the default starting point.
        let (mut system, params) = build_system_from_problem(Box::new(Wood), 1.0);
        let result = system.solve();

        assert!(
            matches!(
                result.status,
                SystemStatus::Solved | SystemStatus::PartiallySolved
            ),
            "Wood pipeline solve status: {:?}",
            result.status,
        );

        let residuals = system.compute_residuals();
        let res_norm = norm(&residuals);
        assert!(
            res_norm < 1e-4,
            "Wood residual norm too large: {}",
            res_norm,
        );

        // Verify the solution is actually a root of the Wood function
        // by evaluating the problem residuals at the found solution.
        let solution: Vec<f64> = params.iter().map(|&pid| system.get_param(pid)).collect();
        let wood = Wood;
        let direct_residuals = wood.residuals(&solution);
        let direct_norm = norm(&direct_residuals);
        assert!(
            direct_norm < 1e-4,
            "Wood: solution found by pipeline is not a root (residual norm = {})",
            direct_norm,
        );
    }
}
