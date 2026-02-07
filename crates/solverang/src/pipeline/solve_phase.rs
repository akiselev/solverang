//! Default solve-phase implementations for the pipeline.
//!
//! This module provides two implementations of the [`SolveCluster`] trait:
//!
//! - [`DefaultSolve`]: Tries closed-form solvers for matched patterns first,
//!   then falls back to numerical Levenberg-Marquardt for remaining constraints.
//! - [`NumericalOnlySolve`]: Skips closed-form entirely and goes straight to
//!   numerical LM solving. Useful for benchmarking closed-form vs. numerical.

use std::collections::HashSet;

use crate::constraint::Constraint;
use crate::id::ParamId;
use crate::param::{ParamStore, SolverMapping};
use crate::problem::Problem;
use crate::solve::closed_form::solve_pattern;
use crate::solve::ReducedSubProblem;
use crate::solver::{LMSolver, SolveResult};
use crate::system::{ClusterSolveStatus, SystemConfig};

use super::traits::SolveCluster;
use super::types::{ClusterAnalysis, ClusterSolution, ReducedCluster};

// ---------------------------------------------------------------------------
// DefaultSolve
// ---------------------------------------------------------------------------

/// Default solve strategy: closed-form first, then numerical LM fallback.
///
/// For each matched pattern in the cluster analysis, the solver attempts a
/// closed-form solution. Constraints and parameters handled by successful
/// closed-form solves are removed from the remaining set. Any leftover
/// constraints are solved numerically using Levenberg-Marquardt.
pub struct DefaultSolve;

impl SolveCluster for DefaultSolve {
    fn solve_cluster(
        &self,
        reduced: &ReducedCluster,
        analysis: &ClusterAnalysis,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &ParamStore,
        warm_start: Option<&[f64]>,
        config: &SystemConfig,
    ) -> ClusterSolution {
        // 1. Early exit for trivially violated clusters.
        if !reduced.trivially_violated.is_empty() {
            return ClusterSolution {
                cluster_id: reduced.cluster_id,
                status: ClusterSolveStatus::NotConverged,
                param_values: Vec::new(),
                mapping: None,
                numerical_solution: None,
                iterations: 0,
                residual_norm: f64::INFINITY,
            };
        }

        // 2. Collect constraint references from active indices.
        let active_set: HashSet<usize> = reduced.active_constraint_indices.iter().copied().collect();
        let active_refs: Vec<(usize, &dyn Constraint)> = reduced
            .active_constraint_indices
            .iter()
            .filter_map(|&idx| {
                constraints.get(idx).and_then(|opt| {
                    opt.as_ref().map(|c| (idx, c.as_ref()))
                })
            })
            .collect();

        // Build a lookup from constraint index to &dyn Constraint.
        let constraint_by_idx = |idx: usize| -> Option<&dyn Constraint> {
            active_refs
                .iter()
                .find(|(i, _)| *i == idx)
                .map(|(_, c)| *c)
        };

        // Track which constraints and params remain after closed-form.
        let mut remaining_constraint_indices: HashSet<usize> = active_set.clone();
        let mut remaining_param_ids: HashSet<ParamId> =
            reduced.active_param_ids.iter().copied().collect();

        let mut closed_form_values: Vec<(ParamId, f64)> = Vec::new();
        let mut closed_form_store = store.snapshot();

        // 3. Try closed-form patterns.
        for pattern in &analysis.patterns {
            // Check that all of the pattern's constraint indices are still in the active set.
            let all_active = pattern
                .constraint_indices
                .iter()
                .all(|idx| remaining_constraint_indices.contains(idx));
            if !all_active {
                continue;
            }

            // Build the flat constraint refs slice for the pattern.
            let pattern_constraints: Vec<&dyn Constraint> = pattern
                .constraint_indices
                .iter()
                .filter_map(|&idx| constraint_by_idx(idx))
                .collect();

            if pattern_constraints.len() != pattern.constraint_indices.len() {
                // Some constraints couldn't be found; skip this pattern.
                continue;
            }

            if let Some(result) = solve_pattern(pattern, &pattern_constraints, &closed_form_store) {
                if result.solved {
                    // Apply values to our working snapshot so subsequent
                    // patterns see updated values.
                    for &(pid, val) in &result.values {
                        closed_form_store.set(pid, val);
                    }
                    closed_form_values.extend(&result.values);

                    // Remove pattern's constraints and params from remaining sets.
                    for &cidx in &pattern.constraint_indices {
                        remaining_constraint_indices.remove(&cidx);
                    }
                    for &pid in &pattern.param_ids {
                        remaining_param_ids.remove(&pid);
                    }
                }
            }
        }

        // 4. Numerical solve for remaining constraints.
        let remaining_constraint_refs: Vec<&dyn Constraint> = remaining_constraint_indices
            .iter()
            .filter_map(|&idx| constraint_by_idx(idx))
            .collect();
        let remaining_params: Vec<ParamId> = reduced
            .active_param_ids
            .iter()
            .copied()
            .filter(|pid| remaining_param_ids.contains(pid))
            .collect();

        let mut numerical_values: Vec<(ParamId, f64)> = Vec::new();
        let mut numerical_solution: Option<Vec<f64>> = None;
        let mut mapping: Option<SolverMapping> = None;
        let mut iterations: usize = 0;
        let mut residual_norm: f64 = 0.0;
        let mut status = ClusterSolveStatus::Converged;

        if !remaining_constraint_refs.is_empty() && !remaining_params.is_empty() {
            // Build sub-problem using the updated snapshot so closed-form
            // values are visible as current parameter values.
            let sub = ReducedSubProblem::new(
                &closed_form_store,
                remaining_constraint_refs,
                &remaining_params,
            );

            if sub.variable_count() == 0 {
                // No free variables left. Check if there are residual violations.
                if sub.residual_count() > 0 {
                    let x0 = sub.initial_point(1.0);
                    let r = sub.residuals(&x0);
                    residual_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
                }
                status = ClusterSolveStatus::Skipped;
            } else {
                // Determine initial point.
                let x0 = if let Some(ws) = warm_start {
                    if ws.len() == sub.variable_count() {
                        ws.to_vec()
                    } else {
                        sub.initial_point(1.0)
                    }
                } else {
                    sub.initial_point(1.0)
                };

                let solver = LMSolver::new(config.lm_config.clone());
                let result = solver.solve(&sub, &x0);
                let sub_mapping = sub.mapping().clone();

                match result {
                    SolveResult::Converged {
                        solution,
                        iterations: iters,
                        residual_norm: rn,
                    } => {
                        // Map solution back to param values.
                        for (col, &pid) in sub_mapping.col_to_param.iter().enumerate() {
                            numerical_values.push((pid, solution[col]));
                        }
                        numerical_solution = Some(solution);
                        iterations = iters;
                        residual_norm = rn;
                        status = ClusterSolveStatus::Converged;
                    }
                    SolveResult::NotConverged {
                        solution,
                        iterations: iters,
                        residual_norm: rn,
                    } => {
                        for (col, &pid) in sub_mapping.col_to_param.iter().enumerate() {
                            numerical_values.push((pid, solution[col]));
                        }
                        numerical_solution = Some(solution);
                        iterations = iters;
                        residual_norm = rn;
                        status = ClusterSolveStatus::NotConverged;
                    }
                    SolveResult::Failed { .. } => {
                        iterations = 0;
                        residual_norm = f64::INFINITY;
                        status = ClusterSolveStatus::NotConverged;
                    }
                }

                mapping = Some(sub_mapping);
            }
        } else if remaining_constraint_refs.is_empty() && !closed_form_values.is_empty() {
            // 6. Fully closed-form case: all constraints were handled.
            // Compute the final residual norm by evaluating all original constraints.
            let mut total_sq = 0.0;
            for &idx in &reduced.active_constraint_indices {
                if let Some(c) = constraint_by_idx(idx) {
                    let r = c.residuals(&closed_form_store);
                    total_sq += r.iter().map(|v| v * v).sum::<f64>();
                }
            }
            residual_norm = total_sq.sqrt();
            status = ClusterSolveStatus::Converged;
        } else if remaining_constraint_refs.is_empty() && closed_form_values.is_empty() {
            // No active constraints at all after reduction -> Skipped.
            status = ClusterSolveStatus::Skipped;
        }

        // 5. Combine results: closed-form + numerical + eliminated params.
        let mut param_values = Vec::with_capacity(
            closed_form_values.len()
                + numerical_values.len()
                + reduced.eliminated_params.len(),
        );
        param_values.extend(&closed_form_values);
        param_values.extend(&numerical_values);
        param_values.extend(&reduced.eliminated_params);

        ClusterSolution {
            cluster_id: reduced.cluster_id,
            status,
            param_values,
            mapping,
            numerical_solution,
            iterations,
            residual_norm,
        }
    }
}

// ---------------------------------------------------------------------------
// NumericalOnlySolve
// ---------------------------------------------------------------------------

/// Numerical-only solve strategy: skip closed-form, go straight to LM.
///
/// This is useful for benchmarking to compare closed-form vs. purely
/// numerical solving performance and accuracy.
pub struct NumericalOnlySolve;

impl SolveCluster for NumericalOnlySolve {
    fn solve_cluster(
        &self,
        reduced: &ReducedCluster,
        _analysis: &ClusterAnalysis,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &ParamStore,
        warm_start: Option<&[f64]>,
        config: &SystemConfig,
    ) -> ClusterSolution {
        // Early exit for trivially violated clusters.
        if !reduced.trivially_violated.is_empty() {
            return ClusterSolution {
                cluster_id: reduced.cluster_id,
                status: ClusterSolveStatus::NotConverged,
                param_values: Vec::new(),
                mapping: None,
                numerical_solution: None,
                iterations: 0,
                residual_norm: f64::INFINITY,
            };
        }

        // Collect constraint refs.
        let constraint_refs: Vec<&dyn Constraint> = reduced
            .active_constraint_indices
            .iter()
            .filter_map(|&idx| {
                constraints.get(idx).and_then(|opt| {
                    opt.as_ref().map(|c| c.as_ref())
                })
            })
            .collect();

        if constraint_refs.is_empty() || reduced.active_param_ids.is_empty() {
            // Nothing to solve.
            let mut param_values = Vec::new();
            param_values.extend(&reduced.eliminated_params);
            return ClusterSolution {
                cluster_id: reduced.cluster_id,
                status: ClusterSolveStatus::Skipped,
                param_values,
                mapping: None,
                numerical_solution: None,
                iterations: 0,
                residual_norm: 0.0,
            };
        }

        let sub = ReducedSubProblem::new(store, constraint_refs, &reduced.active_param_ids);

        if sub.variable_count() == 0 {
            let mut residual_norm = 0.0;
            if sub.residual_count() > 0 {
                let x0 = sub.initial_point(1.0);
                let r = sub.residuals(&x0);
                residual_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            }
            let mut param_values = Vec::new();
            param_values.extend(&reduced.eliminated_params);
            return ClusterSolution {
                cluster_id: reduced.cluster_id,
                status: ClusterSolveStatus::Skipped,
                param_values,
                mapping: None,
                numerical_solution: None,
                iterations: 0,
                residual_norm,
            };
        }

        let x0 = if let Some(ws) = warm_start {
            if ws.len() == sub.variable_count() {
                ws.to_vec()
            } else {
                sub.initial_point(1.0)
            }
        } else {
            sub.initial_point(1.0)
        };

        let solver = LMSolver::new(config.lm_config.clone());
        let result = solver.solve(&sub, &x0);
        let sub_mapping = sub.mapping().clone();

        let (status, numerical_values, numerical_solution, iterations, residual_norm) = match result
        {
            SolveResult::Converged {
                solution,
                iterations,
                residual_norm,
            } => {
                let vals: Vec<(ParamId, f64)> = sub_mapping
                    .col_to_param
                    .iter()
                    .enumerate()
                    .map(|(col, &pid)| (pid, solution[col]))
                    .collect();
                (
                    ClusterSolveStatus::Converged,
                    vals,
                    Some(solution),
                    iterations,
                    residual_norm,
                )
            }
            SolveResult::NotConverged {
                solution,
                iterations,
                residual_norm,
            } => {
                let vals: Vec<(ParamId, f64)> = sub_mapping
                    .col_to_param
                    .iter()
                    .enumerate()
                    .map(|(col, &pid)| (pid, solution[col]))
                    .collect();
                (
                    ClusterSolveStatus::NotConverged,
                    vals,
                    Some(solution),
                    iterations,
                    residual_norm,
                )
            }
            SolveResult::Failed { .. } => (
                ClusterSolveStatus::NotConverged,
                Vec::new(),
                None,
                0,
                f64::INFINITY,
            ),
        };

        let mut param_values =
            Vec::with_capacity(numerical_values.len() + reduced.eliminated_params.len());
        param_values.extend(&numerical_values);
        param_values.extend(&reduced.eliminated_params);

        ClusterSolution {
            cluster_id: reduced.cluster_id,
            status,
            param_values,
            mapping: Some(sub_mapping),
            numerical_solution,
            iterations,
            residual_norm,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::Constraint;
    use crate::graph::pattern::{MatchedPattern, PatternKind};
    use crate::id::{ClusterId, ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;
    use crate::system::SystemConfig;
    use std::collections::HashMap;

    // -----------------------------------------------------------------------
    // Test constraint: fix a parameter to a target value.
    // Residual: param - target
    // Jacobian: d(residual)/d(param) = 1.0
    // -----------------------------------------------------------------------

    struct FixValueConstraint {
        id: ConstraintId,
        entity: EntityId,
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
            std::slice::from_ref(&self.entity)
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

    fn dummy_entity() -> EntityId {
        EntityId::new(0, 0)
    }

    fn default_config() -> SystemConfig {
        SystemConfig::default()
    }

    /// Helper to build a ReducedCluster with sane defaults.
    fn make_reduced(
        cluster_id: ClusterId,
        active_constraint_indices: Vec<usize>,
        active_param_ids: Vec<ParamId>,
    ) -> ReducedCluster {
        ReducedCluster {
            cluster_id,
            active_constraint_indices,
            active_param_ids,
            eliminated_params: Vec::new(),
            removed_constraints: Vec::new(),
            merge_map: HashMap::new(),
            trivially_violated: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Test: cluster with no active constraints after reduction -> Skipped
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_active_constraints_returns_skipped() {
        let store = ParamStore::new();
        let config = default_config();
        let solver = DefaultSolve;

        let reduced = make_reduced(ClusterId(0), vec![], vec![]);
        let analysis = ClusterAnalysis {
            cluster_id: ClusterId(0),
            ..Default::default()
        };

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![];

        let solution =
            solver.solve_cluster(&reduced, &analysis, &constraints, &store, None, &config);

        assert_eq!(solution.status, ClusterSolveStatus::Skipped);
        assert_eq!(solution.iterations, 0);
    }

    // -----------------------------------------------------------------------
    // Test: trivially violated cluster -> NotConverged
    // -----------------------------------------------------------------------

    #[test]
    fn test_trivially_violated_returns_not_converged() {
        let store = ParamStore::new();
        let config = default_config();
        let solver = DefaultSolve;

        let mut reduced = make_reduced(ClusterId(0), vec![0], vec![]);
        reduced.trivially_violated = vec![0];

        let analysis = ClusterAnalysis {
            cluster_id: ClusterId(0),
            ..Default::default()
        };

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![];

        let solution =
            solver.solve_cluster(&reduced, &analysis, &constraints, &store, None, &config);

        assert_eq!(solution.status, ClusterSolveStatus::NotConverged);
        assert_eq!(solution.iterations, 0);
        assert!(solution.residual_norm.is_infinite());
    }

    // -----------------------------------------------------------------------
    // Test: warm_start is used when provided
    // -----------------------------------------------------------------------

    #[test]
    fn test_warm_start_is_used() {
        let eid = dummy_entity();
        let mut store = ParamStore::new();
        // Start far from the target so that the default initial point differs
        // from the warm start.
        let px = store.alloc(100.0, eid);

        let c = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity: eid,
            param: px,
            target: 5.0,
        };

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(Box::new(c))];

        let reduced = make_reduced(ClusterId(0), vec![0], vec![px]);
        let analysis = ClusterAnalysis {
            cluster_id: ClusterId(0),
            ..Default::default()
        };

        let config = default_config();
        let solver = DefaultSolve;

        // Warm start close to the target: the solver should converge quickly.
        let warm = vec![4.9];
        let solution = solver.solve_cluster(
            &reduced,
            &analysis,
            &constraints,
            &store,
            Some(&warm),
            &config,
        );

        assert_eq!(solution.status, ClusterSolveStatus::Converged);
        // The solution should be close to 5.0.
        let solved_val = solution
            .param_values
            .iter()
            .find(|(pid, _)| *pid == px)
            .map(|(_, v)| *v)
            .unwrap();
        assert!(
            (solved_val - 5.0).abs() < 1e-6,
            "solved value {solved_val} not close to target 5.0"
        );
    }

    // -----------------------------------------------------------------------
    // Test: eliminated_params are included in param_values
    // -----------------------------------------------------------------------

    #[test]
    fn test_eliminated_params_included_in_param_values() {
        let eid = dummy_entity();
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, eid);
        let py = store.alloc(0.0, eid);

        // px is solved numerically (constrained), py is eliminated.
        let c = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity: eid,
            param: px,
            target: 3.0,
        };

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(Box::new(c))];

        let mut reduced = make_reduced(ClusterId(0), vec![0], vec![px]);
        reduced.eliminated_params = vec![(py, 42.0)];

        let analysis = ClusterAnalysis {
            cluster_id: ClusterId(0),
            ..Default::default()
        };

        let config = default_config();
        let solver = DefaultSolve;

        let solution =
            solver.solve_cluster(&reduced, &analysis, &constraints, &store, None, &config);

        // Check that eliminated param py is in the output.
        let py_val = solution
            .param_values
            .iter()
            .find(|(pid, _)| *pid == py)
            .map(|(_, v)| *v);
        assert_eq!(py_val, Some(42.0), "eliminated param py should be 42.0");

        // Check that numerically solved param px is also present.
        let px_val = solution
            .param_values
            .iter()
            .find(|(pid, _)| *pid == px)
            .map(|(_, v)| *v);
        assert!(px_val.is_some(), "solved param px should be present");
    }

    // -----------------------------------------------------------------------
    // Test: NumericalOnlySolve trivially violated
    // -----------------------------------------------------------------------

    #[test]
    fn test_numerical_only_trivially_violated() {
        let store = ParamStore::new();
        let config = default_config();
        let solver = NumericalOnlySolve;

        let mut reduced = make_reduced(ClusterId(0), vec![0], vec![]);
        reduced.trivially_violated = vec![0];

        let analysis = ClusterAnalysis {
            cluster_id: ClusterId(0),
            ..Default::default()
        };

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![];

        let solution =
            solver.solve_cluster(&reduced, &analysis, &constraints, &store, None, &config);

        assert_eq!(solution.status, ClusterSolveStatus::NotConverged);
        assert!(solution.residual_norm.is_infinite());
    }

    // -----------------------------------------------------------------------
    // Test: NumericalOnlySolve no active constraints -> Skipped
    // -----------------------------------------------------------------------

    #[test]
    fn test_numerical_only_no_constraints_skipped() {
        let store = ParamStore::new();
        let config = default_config();
        let solver = NumericalOnlySolve;

        let reduced = make_reduced(ClusterId(0), vec![], vec![]);
        let analysis = ClusterAnalysis {
            cluster_id: ClusterId(0),
            ..Default::default()
        };
        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![];

        let solution =
            solver.solve_cluster(&reduced, &analysis, &constraints, &store, None, &config);

        assert_eq!(solution.status, ClusterSolveStatus::Skipped);
    }

    // -----------------------------------------------------------------------
    // Test: NumericalOnlySolve basic convergence
    // -----------------------------------------------------------------------

    #[test]
    fn test_numerical_only_solve_converges() {
        let eid = dummy_entity();
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);

        let c = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity: eid,
            param: px,
            target: 7.0,
        };

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(Box::new(c))];

        let reduced = make_reduced(ClusterId(0), vec![0], vec![px]);
        let analysis = ClusterAnalysis {
            cluster_id: ClusterId(0),
            ..Default::default()
        };

        let config = default_config();
        let solver = NumericalOnlySolve;

        let solution =
            solver.solve_cluster(&reduced, &analysis, &constraints, &store, None, &config);

        assert_eq!(solution.status, ClusterSolveStatus::Converged);
        let val = solution
            .param_values
            .iter()
            .find(|(pid, _)| *pid == px)
            .map(|(_, v)| *v)
            .unwrap();
        assert!(
            (val - 7.0).abs() < 1e-6,
            "expected value near 7.0, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: DefaultSolve with a closed-form pattern covering all constraints
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_solve_closed_form_only() {
        let eid = dummy_entity();
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);

        let c = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity: eid,
            param: px,
            target: 5.0,
        };

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(Box::new(c))];

        let reduced = make_reduced(ClusterId(0), vec![0], vec![px]);

        // A ScalarSolve pattern covering the single constraint.
        let pattern = MatchedPattern {
            kind: PatternKind::ScalarSolve,
            entity_ids: vec![eid],
            constraint_indices: vec![0],
            param_ids: vec![px],
        };

        let analysis = ClusterAnalysis {
            cluster_id: ClusterId(0),
            patterns: vec![pattern],
            ..Default::default()
        };

        let config = default_config();
        let solver = DefaultSolve;

        let solution =
            solver.solve_cluster(&reduced, &analysis, &constraints, &store, None, &config);

        assert_eq!(solution.status, ClusterSolveStatus::Converged);
        assert_eq!(solution.iterations, 0, "closed-form should need 0 iterations");

        let val = solution
            .param_values
            .iter()
            .find(|(pid, _)| *pid == px)
            .map(|(_, v)| *v)
            .unwrap();
        assert!(
            (val - 5.0).abs() < 1e-10,
            "closed-form should solve to target 5.0, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: NumericalOnlySolve includes eliminated params
    // -----------------------------------------------------------------------

    #[test]
    fn test_numerical_only_includes_eliminated_params() {
        let eid = dummy_entity();
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);
        let py = store.alloc(0.0, eid);

        let c = FixValueConstraint {
            id: ConstraintId::new(0, 0),
            entity: eid,
            param: px,
            target: 3.0,
        };

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(Box::new(c))];

        let mut reduced = make_reduced(ClusterId(0), vec![0], vec![px]);
        reduced.eliminated_params = vec![(py, 99.0)];

        let analysis = ClusterAnalysis {
            cluster_id: ClusterId(0),
            ..Default::default()
        };

        let config = default_config();
        let solver = NumericalOnlySolve;

        let solution =
            solver.solve_cluster(&reduced, &analysis, &constraints, &store, None, &config);

        let py_val = solution
            .param_values
            .iter()
            .find(|(pid, _)| *pid == py)
            .map(|(_, v)| *v);
        assert_eq!(py_val, Some(99.0));
    }
}
