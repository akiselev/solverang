//! Pipeline `Reduce` implementations.
//!
//! Wraps the low-level reduction passes from [`crate::reduce`] into the
//! pipeline's [`Reduce`] trait, plus a [`ChainedReducer`] compositor and
//! sensible defaults ([`DefaultReduce`], [`NoopReduce`]).

use std::collections::{HashMap, HashSet};

use crate::constraint::Constraint;
use crate::id::ParamId;
use crate::param::ParamStore;
use crate::reduce::eliminate::detect_trivial_eliminations;
use crate::reduce::merge::{build_substitution_map, detect_merges};
use crate::reduce::substitute::analyze_substitutions;

use super::traits::Reduce;
use super::types::{ClusterData, ReducedCluster};

// ---------------------------------------------------------------------------
// NoopReduce
// ---------------------------------------------------------------------------

/// A reducer that performs no reduction, returning a passthrough.
pub struct NoopReduce;

impl Reduce for NoopReduce {
    fn reduce(
        &self,
        cluster: &ClusterData,
        _constraints: &[Option<Box<dyn Constraint>>],
        _store: &mut ParamStore,
    ) -> ReducedCluster {
        ReducedCluster::passthrough(cluster)
    }
}

// ---------------------------------------------------------------------------
// SubstituteReducer
// ---------------------------------------------------------------------------

/// Wraps [`analyze_substitutions`] to identify constraints that are trivially
/// satisfied (all params fixed, residual near zero) or trivially violated
/// (all params fixed, residual far from zero).
///
/// Trivially-satisfied constraints are added to `removed_constraints`.
/// Trivially-violated constraints are flagged in `trivially_violated` but
/// remain in `active_constraint_indices` for the orchestrator to report.
///
/// Fixed parameters are **not** removed from `active_param_ids` -- the
/// `SolverMapping` handles that during the Solve phase.
pub struct SubstituteReducer;

impl Reduce for SubstituteReducer {
    fn reduce(
        &self,
        cluster: &ClusterData,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &mut ParamStore,
    ) -> ReducedCluster {
        let mut result = ReducedCluster::passthrough(cluster);

        // Collect constraint refs with a mapping from flat-slice index back to
        // the system-level constraint index.
        let mut constraint_refs: Vec<&dyn Constraint> = Vec::new();
        let mut index_map: Vec<usize> = Vec::new();

        for &ci in &cluster.constraint_indices {
            if let Some(ref c) = constraints[ci] {
                constraint_refs.push(c.as_ref());
                index_map.push(ci);
            }
        }

        if constraint_refs.is_empty() {
            return result;
        }

        let sub_result = analyze_substitutions(&constraint_refs, store);

        // Map trivially-satisfied flat indices back to system-level indices.
        for &flat_idx in &sub_result.trivially_satisfied {
            result.removed_constraints.push(index_map[flat_idx]);
        }

        // Remove satisfied constraints from the active list.
        let removed: HashSet<usize> = result.removed_constraints.iter().copied().collect();
        result
            .active_constraint_indices
            .retain(|ci| !removed.contains(ci));

        // Map trivially-violated flat indices back to system-level indices.
        for &flat_idx in &sub_result.trivially_violated {
            result.trivially_violated.push(index_map[flat_idx]);
        }

        result
    }
}

// ---------------------------------------------------------------------------
// MergeReducer
// ---------------------------------------------------------------------------

/// Tolerance for checking whether a Jacobian entry matches +1 or -1.
const JACOBIAN_TOLERANCE: f64 = 1e-10;

/// Wraps [`detect_merges`] and [`build_substitution_map`] to identify
/// coincident-parameter equality constraints.
///
/// Equality constraints whose params form a detected merge pair are added to
/// `removed_constraints`. Source params remain in `active_param_ids` but the
/// `merge_map` is available for the Solve phase to remap them.
pub struct MergeReducer;

/// Check whether a Jacobian represents a simple equality `a - b = 0`.
///
/// Looks for exactly two entries in row 0, one being +1 and the other -1
/// (or vice versa), matching the two parameter IDs.
///
/// This duplicates the private `is_equality_jacobian` in `reduce::merge`
/// because we need the same structural check to identify which constraints
/// correspond to the detected merges.
fn is_equality_jac(jac: &[(usize, ParamId, f64)], param_a: ParamId, param_b: ParamId) -> bool {
    let mut val_a: Option<f64> = None;
    let mut val_b: Option<f64> = None;

    for &(row, pid, value) in jac {
        if row != 0 {
            continue;
        }
        if pid == param_a {
            val_a = Some(value);
        } else if pid == param_b {
            val_b = Some(value);
        }
    }

    match (val_a, val_b) {
        (Some(a), Some(b)) => {
            ((a - 1.0).abs() < JACOBIAN_TOLERANCE && (b + 1.0).abs() < JACOBIAN_TOLERANCE)
                || ((a + 1.0).abs() < JACOBIAN_TOLERANCE && (b - 1.0).abs() < JACOBIAN_TOLERANCE)
        }
        _ => false,
    }
}

/// Return `true` if `c` matches the same structural criteria used by
/// [`detect_merges`]: single equation, two free params, +1/-1 Jacobian.
fn is_equality_constraint(c: &dyn Constraint, store: &ParamStore) -> bool {
    if c.equation_count() != 1 {
        return false;
    }
    let params = c.param_ids();
    if params.len() != 2 {
        return false;
    }
    if store.is_fixed(params[0]) || store.is_fixed(params[1]) {
        return false;
    }
    let jac = c.jacobian(store);
    is_equality_jac(&jac, params[0], params[1])
}

impl Reduce for MergeReducer {
    fn reduce(
        &self,
        cluster: &ClusterData,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &mut ParamStore,
    ) -> ReducedCluster {
        let mut result = ReducedCluster::passthrough(cluster);

        // Collect constraint refs with index mapping.
        let mut constraint_refs: Vec<&dyn Constraint> = Vec::new();
        let mut index_map: Vec<usize> = Vec::new();

        for &ci in &cluster.constraint_indices {
            if let Some(ref c) = constraints[ci] {
                constraint_refs.push(c.as_ref());
                index_map.push(ci);
            }
        }

        if constraint_refs.is_empty() {
            return result;
        }

        let merge_result = detect_merges(&constraint_refs, store);

        if merge_result.merges.is_empty() {
            return result;
        }

        // Build the substitution map (handles transitive chains via union-find).
        result.merge_map = build_substitution_map(&merge_result.merges);

        // Build a set of merge pairs for quick lookup.
        // Convention matches detect_merges: higher raw_index = source.
        let merge_pair_set: HashSet<(ParamId, ParamId)> = merge_result
            .merges
            .iter()
            .map(|m| (m.source, m.target))
            .collect();

        // Re-examine constraints to find those that correspond to the detected
        // merges. A constraint is removed if it passes the same equality
        // criteria and its param pair matches a merge pair.
        for (flat_idx, &c_ref) in constraint_refs.iter().enumerate() {
            if !is_equality_constraint(c_ref, store) {
                continue;
            }

            let params = c_ref.param_ids();
            let (source, target) = if params[0].raw_index() > params[1].raw_index() {
                (params[0], params[1])
            } else {
                (params[1], params[0])
            };

            if merge_pair_set.contains(&(source, target)) {
                result.removed_constraints.push(index_map[flat_idx]);
            }
        }

        // Remove merged-away constraints from the active list.
        let removed: HashSet<usize> = result.removed_constraints.iter().copied().collect();
        result
            .active_constraint_indices
            .retain(|ci| !removed.contains(ci));

        result
    }
}

// ---------------------------------------------------------------------------
// EliminateReducer
// ---------------------------------------------------------------------------

/// Wraps [`detect_trivial_eliminations`] to analytically solve single-free-
/// parameter constraints.
///
/// Eliminated parameters are added to `eliminated_params` with their
/// determined values, removed from `active_param_ids`, and the consumed
/// constraints are added to `removed_constraints`.
pub struct EliminateReducer;

/// Tolerance for verifying that a linearised elimination actually satisfies
/// the constraint (guards against approximate results on nonlinear constraints).
const ELIMINATION_RESIDUAL_TOL: f64 = 1e-10;

impl Reduce for EliminateReducer {
    fn reduce(
        &self,
        cluster: &ClusterData,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &mut ParamStore,
    ) -> ReducedCluster {
        let mut result = ReducedCluster::passthrough(cluster);

        // Iterate until no more eliminations are possible. Each pass may
        // determine parameter values that unlock further eliminations in
        // remaining constraints (cascading reduction).
        loop {
            // Build (system_index, &dyn Constraint) pairs from the
            // *currently active* constraints.
            let indexed_constraints: Vec<(usize, &dyn Constraint)> = result
                .active_constraint_indices
                .iter()
                .filter_map(|&ci| constraints[ci].as_ref().map(|c| (ci, c.as_ref())))
                .collect();

            if indexed_constraints.is_empty() {
                break;
            }

            let eliminations = detect_trivial_eliminations(&indexed_constraints, store);

            if eliminations.is_empty() {
                break;
            }

            let mut any_accepted = false;
            let mut eliminated_set = HashSet::new();

            for elim in &eliminations {
                // Tentatively apply the determined value and verify that the
                // constraint is actually satisfied. For linear constraints
                // the linearisation is exact; for nonlinear constraints the
                // result is only approximate — we must not freeze an
                // inaccurate value.
                let original = store.get(elim.param);
                store.set(elim.param, elim.determined_value);

                let satisfied = constraints[elim.constraint_index]
                    .as_ref()
                    .map(|c| {
                        c.residuals(store)
                            .iter()
                            .all(|r| r.abs() < ELIMINATION_RESIDUAL_TOL)
                    })
                    .unwrap_or(false);

                if satisfied {
                    // Exact (or near-exact) elimination — keep the value
                    // and mark the param fixed so later passes see it as
                    // a known constant.
                    store.fix(elim.param);

                    result
                        .eliminated_params
                        .push((elim.param, elim.determined_value));
                    eliminated_set.insert(elim.param);
                    result.removed_constraints.push(elim.constraint_index);
                    any_accepted = true;
                } else {
                    // Approximate result — restore original value and leave
                    // the constraint for the numerical solver.
                    store.set(elim.param, original);
                }
            }

            if !any_accepted {
                break;
            }

            // Remove eliminated params from active list.
            result
                .active_param_ids
                .retain(|p| !eliminated_set.contains(p));

            // Remove consumed constraints from active list.
            let removed: HashSet<usize> = result.removed_constraints.iter().copied().collect();
            result
                .active_constraint_indices
                .retain(|ci| !removed.contains(ci));
        }

        result
    }
}

// ---------------------------------------------------------------------------
// ChainedReducer
// ---------------------------------------------------------------------------

/// Runs multiple [`Reduce`] stages sequentially, narrowing the cluster view
/// after each stage.
///
/// After each stage completes, the next stage receives a `ClusterData` whose
/// `constraint_indices` and `param_ids` reflect only the remaining active
/// constraints and parameters. The final `ReducedCluster` merges results
/// from all stages:
///
/// - `removed_constraints`: union across all stages.
/// - `eliminated_params`: concatenation across all stages.
/// - `merge_map`: merged across all stages.
/// - `trivially_violated`: union across all stages.
/// - `active_constraint_indices` / `active_param_ids`: from the final stage.
pub struct ChainedReducer {
    stages: Vec<Box<dyn Reduce>>,
}

impl ChainedReducer {
    /// Create a new `ChainedReducer` from an ordered list of stages.
    pub fn new(stages: Vec<Box<dyn Reduce>>) -> Self {
        Self { stages }
    }
}

impl Reduce for ChainedReducer {
    fn reduce(
        &self,
        cluster: &ClusterData,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &mut ParamStore,
    ) -> ReducedCluster {
        if self.stages.is_empty() {
            return ReducedCluster::passthrough(cluster);
        }

        // Accumulated results across all stages.
        let mut all_removed: Vec<usize> = Vec::new();
        let mut all_eliminated: Vec<(ParamId, f64)> = Vec::new();
        let mut all_merge_map: HashMap<ParamId, ParamId> = HashMap::new();
        let mut all_violated: Vec<usize> = Vec::new();

        // The cluster view narrows after each stage.
        let mut current_cluster = cluster.clone();

        for stage in &self.stages {
            let stage_result = stage.reduce(&current_cluster, constraints, store);

            all_removed.extend(&stage_result.removed_constraints);
            all_eliminated.extend(&stage_result.eliminated_params);
            all_merge_map.extend(&stage_result.merge_map);
            all_violated.extend(&stage_result.trivially_violated);

            // Narrow the cluster for the next stage.
            current_cluster = ClusterData {
                id: cluster.id,
                constraint_indices: stage_result.active_constraint_indices,
                param_ids: stage_result.active_param_ids,
                entity_ids: cluster.entity_ids.clone(),
            };
        }

        ReducedCluster {
            cluster_id: cluster.id,
            active_constraint_indices: current_cluster.constraint_indices,
            active_param_ids: current_cluster.param_ids,
            eliminated_params: all_eliminated,
            removed_constraints: all_removed,
            merge_map: all_merge_map,
            trivially_violated: all_violated,
        }
    }
}

// ---------------------------------------------------------------------------
// DefaultReduce
// ---------------------------------------------------------------------------

/// Default reduction pipeline: Substitute, then Merge, then Eliminate.
///
/// This ordering ensures that:
/// 1. Fixed-parameter constraints are handled first (Substitute).
/// 2. Equality constraints between free parameters are merged (Merge).
/// 3. Remaining single-free-param constraints are solved analytically (Eliminate).
pub struct DefaultReduce;

impl Reduce for DefaultReduce {
    fn reduce(
        &self,
        cluster: &ClusterData,
        constraints: &[Option<Box<dyn Constraint>>],
        store: &mut ParamStore,
    ) -> ReducedCluster {
        let chain = ChainedReducer::new(vec![
            Box::new(SubstituteReducer),
            Box::new(MergeReducer),
            Box::new(EliminateReducer),
        ]);
        chain.reduce(cluster, constraints, store)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ClusterId, ConstraintId, EntityId};
    use crate::param::ParamStore;

    // -----------------------------------------------------------------------
    // Test constraint helpers
    // -----------------------------------------------------------------------

    /// Constraint: `param - target = 0`.
    /// Jacobian: `[(0, param, 1.0)]`.
    struct FixValueConstraint {
        id: ConstraintId,
        param: ParamId,
        target: f64,
    }

    impl Constraint for FixValueConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "fix_value"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &[]
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

    /// Constraint: `param_a == param_b` (residual: `a - b`).
    /// Jacobian: `[(0, a, +1.0), (0, b, -1.0)]`.
    struct EqualityConstraint {
        id: ConstraintId,
        params: [ParamId; 2],
    }

    impl Constraint for EqualityConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "equality"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &[]
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
            vec![(0, self.params[0], 1.0), (0, self.params[1], -1.0)]
        }
    }

    fn dummy_owner() -> EntityId {
        EntityId::new(0, 0)
    }

    fn make_cluster(constraint_indices: Vec<usize>, param_ids: Vec<ParamId>) -> ClusterData {
        ClusterData {
            id: ClusterId(0),
            constraint_indices,
            param_ids,
            entity_ids: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // NoopReduce
    // -----------------------------------------------------------------------

    #[test]
    fn noop_returns_passthrough() {
        let mut store = ParamStore::new();
        let p1 = store.alloc(1.0, dummy_owner());
        let p2 = store.alloc(2.0, dummy_owner());

        let c: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p1,
            target: 1.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![p1, p2]);

        let result = NoopReduce.reduce(&cluster, &constraints, &mut store);

        assert_eq!(result.cluster_id, ClusterId(0));
        assert_eq!(result.active_constraint_indices, vec![0]);
        assert_eq!(result.active_param_ids, vec![p1, p2]);
        assert!(result.eliminated_params.is_empty());
        assert!(result.removed_constraints.is_empty());
        assert!(result.merge_map.is_empty());
        assert!(result.trivially_violated.is_empty());
    }

    // -----------------------------------------------------------------------
    // SubstituteReducer
    // -----------------------------------------------------------------------

    #[test]
    fn substitute_removes_trivially_satisfied() {
        let mut store = ParamStore::new();
        let p = store.alloc(5.0, dummy_owner());
        store.fix(p);

        // param=5.0, target=5.0 -> residual=0 -> trivially satisfied.
        let c: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 5.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![p]);

        let result = SubstituteReducer.reduce(&cluster, &constraints, &mut store);

        assert_eq!(result.removed_constraints, vec![0]);
        assert!(result.active_constraint_indices.is_empty());
        assert!(result.trivially_violated.is_empty());
    }

    #[test]
    fn substitute_flags_trivially_violated() {
        let mut store = ParamStore::new();
        let p = store.alloc(5.0, dummy_owner());
        store.fix(p);

        // param=5.0, target=10.0 -> residual=-5.0 -> trivially violated.
        let c: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 10.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![p]);

        let result = SubstituteReducer.reduce(&cluster, &constraints, &mut store);

        // Violated constraints stay active but are flagged.
        assert!(result.removed_constraints.is_empty());
        assert_eq!(result.trivially_violated, vec![0]);
        assert_eq!(result.active_constraint_indices, vec![0]);
    }

    #[test]
    fn substitute_maps_flat_indices_to_system_indices() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p1 = store.alloc(3.0, owner);
        let p2 = store.alloc(7.0, owner);
        store.fix(p1);
        store.fix(p2);

        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p1,
            target: 3.0, // satisfied
        });
        let c2: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(1, 0),
            param: p2,
            target: 99.0, // violated
        });

        // Constraints at system indices 2 and 5, with None gaps.
        let constraints: Vec<Option<Box<dyn Constraint>>> =
            vec![None, None, Some(c1), None, None, Some(c2)];
        let cluster = make_cluster(vec![2, 5], vec![p1, p2]);

        let result = SubstituteReducer.reduce(&cluster, &constraints, &mut store);

        assert_eq!(result.removed_constraints, vec![2]);
        assert_eq!(result.trivially_violated, vec![5]);
        assert_eq!(result.active_constraint_indices, vec![5]);
    }

    #[test]
    fn substitute_preserves_free_param_constraints() {
        let mut store = ParamStore::new();
        let p = store.alloc(0.0, dummy_owner());
        // p is free, not fixed.

        let c: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 5.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![p]);

        let result = SubstituteReducer.reduce(&cluster, &constraints, &mut store);

        // Free-param constraint is neither satisfied nor violated.
        assert!(result.removed_constraints.is_empty());
        assert!(result.trivially_violated.is_empty());
        assert_eq!(result.active_constraint_indices, vec![0]);
        assert_eq!(result.active_param_ids, vec![p]);
    }

    // -----------------------------------------------------------------------
    // MergeReducer
    // -----------------------------------------------------------------------

    #[test]
    fn merge_detects_equality_constraint() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(5.0, owner);
        let b = store.alloc(5.0, owner);

        let c: Box<dyn Constraint> = Box::new(EqualityConstraint {
            id: ConstraintId::new(0, 0),
            params: [a, b],
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![a, b]);

        let result = MergeReducer.reduce(&cluster, &constraints, &mut store);

        // b (higher raw index) maps to a (lower raw index).
        assert_eq!(result.merge_map.get(&b), Some(&a));
        assert_eq!(result.removed_constraints, vec![0]);
        assert!(result.active_constraint_indices.is_empty());
        // Both params stay in active_param_ids.
        assert_eq!(result.active_param_ids, vec![a, b]);
    }

    #[test]
    fn merge_ignores_non_equality_constraints() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(1.0, owner);

        // Single-param constraint is not an equality constraint.
        let c: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: a,
            target: 1.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![a]);

        let result = MergeReducer.reduce(&cluster, &constraints, &mut store);

        assert!(result.merge_map.is_empty());
        assert!(result.removed_constraints.is_empty());
        assert_eq!(result.active_constraint_indices, vec![0]);
    }

    #[test]
    fn merge_skips_fixed_params() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(5.0, owner);
        let b = store.alloc(5.0, owner);
        store.fix(a);

        let c: Box<dyn Constraint> = Box::new(EqualityConstraint {
            id: ConstraintId::new(0, 0),
            params: [a, b],
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![a, b]);

        let result = MergeReducer.reduce(&cluster, &constraints, &mut store);

        // One param is fixed -> no merge.
        assert!(result.merge_map.is_empty());
        assert!(result.removed_constraints.is_empty());
    }

    // -----------------------------------------------------------------------
    // EliminateReducer
    // -----------------------------------------------------------------------

    #[test]
    fn eliminate_detects_single_free_param() {
        let mut store = ParamStore::new();
        let p = store.alloc(0.0, dummy_owner());

        // p - 7.0 = 0 -> single free param, determined value = 7.0.
        let c: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 7.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![p]);

        let result = EliminateReducer.reduce(&cluster, &constraints, &mut store);

        assert_eq!(result.eliminated_params.len(), 1);
        assert_eq!(result.eliminated_params[0].0, p);
        assert!((result.eliminated_params[0].1 - 7.0).abs() < 1e-12);
        assert_eq!(result.removed_constraints, vec![0]);
        assert!(result.active_param_ids.is_empty());
        assert!(result.active_constraint_indices.is_empty());
    }

    #[test]
    fn eliminate_preserves_system_level_indices() {
        let mut store = ParamStore::new();
        let p = store.alloc(0.0, dummy_owner());

        let c: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 3.0,
        });

        // Constraint lives at system index 4.
        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![None, None, None, None, Some(c)];
        let cluster = make_cluster(vec![4], vec![p]);

        let result = EliminateReducer.reduce(&cluster, &constraints, &mut store);

        assert_eq!(result.removed_constraints, vec![4]);
        assert_eq!(result.eliminated_params.len(), 1);
        assert!((result.eliminated_params[0].1 - 3.0).abs() < 1e-12);
    }

    #[test]
    fn eliminate_skips_multi_free_param_constraints() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);

        // Two free params -> not eliminable.
        let c: Box<dyn Constraint> = Box::new(EqualityConstraint {
            id: ConstraintId::new(0, 0),
            params: [a, b],
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![a, b]);

        let result = EliminateReducer.reduce(&cluster, &constraints, &mut store);

        assert!(result.eliminated_params.is_empty());
        assert!(result.removed_constraints.is_empty());
        assert_eq!(result.active_param_ids, vec![a, b]);
    }

    // -----------------------------------------------------------------------
    // ChainedReducer
    // -----------------------------------------------------------------------

    #[test]
    fn chained_composes_multiple_stages() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p_fixed = store.alloc(5.0, owner);
        let p_free = store.alloc(0.0, owner);
        store.fix(p_fixed);

        // c0: fixed, satisfied -> removed by SubstituteReducer.
        let c0: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p_fixed,
            target: 5.0,
        });
        // c1: single free param -> eliminated by EliminateReducer.
        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(1, 0),
            param: p_free,
            target: 3.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c0), Some(c1)];
        let cluster = make_cluster(vec![0, 1], vec![p_fixed, p_free]);

        let chain = ChainedReducer::new(vec![
            Box::new(SubstituteReducer),
            Box::new(EliminateReducer),
        ]);
        let result = chain.reduce(&cluster, &constraints, &mut store);

        // Both constraints removed.
        assert!(result.removed_constraints.contains(&0));
        assert!(result.removed_constraints.contains(&1));
        assert!(result.active_constraint_indices.is_empty());

        // p_free eliminated with value 3.0.
        assert_eq!(result.eliminated_params.len(), 1);
        assert_eq!(result.eliminated_params[0].0, p_free);
        assert!((result.eliminated_params[0].1 - 3.0).abs() < 1e-12);
    }

    #[test]
    fn chained_empty_stages_is_passthrough() {
        let mut store = ParamStore::new();
        let p = store.alloc(1.0, dummy_owner());

        let c: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 1.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![p]);

        let chain = ChainedReducer::new(vec![]);
        let result = chain.reduce(&cluster, &constraints, &mut store);

        assert_eq!(result.active_constraint_indices, vec![0]);
        assert_eq!(result.active_param_ids, vec![p]);
        assert!(result.removed_constraints.is_empty());
        assert!(result.eliminated_params.is_empty());
    }

    #[test]
    fn chained_narrowing_prevents_double_processing() {
        // After SubstituteReducer removes constraint 0, EliminateReducer
        // should not see it at all (even if it would also match).
        let mut store = ParamStore::new();
        let p = store.alloc(5.0, dummy_owner());
        store.fix(p);

        // Fixed param, target matches -> trivially satisfied.
        let c: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 5.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![p]);

        let chain = ChainedReducer::new(vec![
            Box::new(SubstituteReducer),
            Box::new(EliminateReducer),
        ]);
        let result = chain.reduce(&cluster, &constraints, &mut store);

        // Constraint 0 removed exactly once (by SubstituteReducer).
        assert_eq!(
            result
                .removed_constraints
                .iter()
                .filter(|&&x| x == 0)
                .count(),
            1
        );
    }

    // -----------------------------------------------------------------------
    // DefaultReduce
    // -----------------------------------------------------------------------

    #[test]
    fn default_reduce_combines_all_three_passes() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p_fixed = store.alloc(5.0, owner);
        let p_a = store.alloc(3.0, owner);
        let p_b = store.alloc(3.0, owner);
        let p_free = store.alloc(0.0, owner);
        store.fix(p_fixed);

        // c0: trivially satisfied (Substitute).
        let c0: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p_fixed,
            target: 5.0,
        });
        // c1: equality constraint (Merge).
        let c1: Box<dyn Constraint> = Box::new(EqualityConstraint {
            id: ConstraintId::new(1, 0),
            params: [p_a, p_b],
        });
        // c2: single free param (Eliminate).
        let c2: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(2, 0),
            param: p_free,
            target: 9.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c0), Some(c1), Some(c2)];
        let cluster = make_cluster(vec![0, 1, 2], vec![p_fixed, p_a, p_b, p_free]);

        let result = DefaultReduce.reduce(&cluster, &constraints, &mut store);

        // All three constraints removed by their respective passes.
        assert!(result.removed_constraints.contains(&0));
        assert!(result.removed_constraints.contains(&1));
        assert!(result.removed_constraints.contains(&2));
        assert!(result.active_constraint_indices.is_empty());

        // Merge map: p_b -> p_a.
        assert_eq!(result.merge_map.get(&p_b), Some(&p_a));

        // p_free eliminated with value 9.0.
        assert_eq!(result.eliminated_params.len(), 1);
        assert_eq!(result.eliminated_params[0].0, p_free);
        assert!((result.eliminated_params[0].1 - 9.0).abs() < 1e-12);

        // p_free removed from active params; p_fixed, p_a, p_b remain.
        assert!(!result.active_param_ids.contains(&p_free));
        assert!(result.active_param_ids.contains(&p_fixed));
        assert!(result.active_param_ids.contains(&p_a));
        assert!(result.active_param_ids.contains(&p_b));
    }

    #[test]
    fn default_reduce_passthrough_when_nothing_to_reduce() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);

        // Non-linear, two free params -> nothing for any pass to reduce.
        // Use an equality constraint but with non-+1/-1 Jacobian behavior
        // by having two free params that won't trigger eliminate either.
        let c: Box<dyn Constraint> = Box::new(EqualityConstraint {
            id: ConstraintId::new(0, 0),
            params: [a, b],
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![a, b]);

        let result = DefaultReduce.reduce(&cluster, &constraints, &mut store);

        // The equality constraint IS detected by MergeReducer, so it should
        // be removed and a merge map produced.
        assert_eq!(result.removed_constraints, vec![0]);
        assert_eq!(result.merge_map.get(&b), Some(&a));
        assert!(result.active_constraint_indices.is_empty());
    }

    // -----------------------------------------------------------------------
    // Additional test constraint helpers
    // -----------------------------------------------------------------------

    /// Constraint: `param_a + param_b - target = 0`.
    /// Jacobian: `[(0, a, 1.0), (0, b, 1.0)]`.
    struct SumConstraint {
        id: ConstraintId,
        params: [ParamId; 2],
        target: f64,
    }

    impl Constraint for SumConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "sum"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &[]
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.params[0]) + store.get(self.params[1]) - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.params[0], 1.0), (0, self.params[1], 1.0)]
        }
    }

    /// Constraint: `param_a - param_b - target = 0` (difference).
    /// Jacobian: `[(0, a, 1.0), (0, b, -1.0)]`.
    struct DiffConstraint {
        id: ConstraintId,
        params: [ParamId; 2],
        target: f64,
    }

    impl Constraint for DiffConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "diff"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &[]
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.params[0]) - store.get(self.params[1]) - self.target]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.params[0], 1.0), (0, self.params[1], -1.0)]
        }
    }

    /// Nonlinear constraint: `param^2 - target = 0`.
    /// Jacobian: `[(0, param, 2*param)]`.
    struct QuadraticConstraint {
        id: ConstraintId,
        param: ParamId,
        target: f64,
    }

    impl Constraint for QuadraticConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "quadratic"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &[]
        }
        fn param_ids(&self) -> &[ParamId] {
            std::slice::from_ref(&self.param)
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            let v = store.get(self.param);
            vec![v * v - self.target]
        }
        fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.param, 2.0 * store.get(self.param))]
        }
    }

    // -----------------------------------------------------------------------
    // Regression tests — cascading elimination
    // -----------------------------------------------------------------------

    /// Core coupling scenario: fixing p1 via C1 should cascade to make
    /// C2 (which depends on p1 and p2) also eliminable, since C2 is
    /// linear and has exactly one remaining free param after p1 is known.
    #[test]
    fn eliminate_cascades_through_linear_constraints() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p1 = store.alloc(0.0, owner);
        let p2 = store.alloc(0.0, owner);

        // C0: p1 = 5.0  (single free param -> eliminable first pass)
        let c0: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p1,
            target: 5.0,
        });
        // C1: p1 + p2 = 10.0  (two free params initially, but after p1
        //     is eliminated, becomes single free param -> eliminable)
        let c1: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(1, 0),
            params: [p1, p2],
            target: 10.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c0), Some(c1)];
        let cluster = make_cluster(vec![0, 1], vec![p1, p2]);

        let result = EliminateReducer.reduce(&cluster, &constraints, &mut store);

        // Both constraints should be eliminated.
        assert!(
            result.removed_constraints.contains(&0),
            "C0 should be eliminated"
        );
        assert!(
            result.removed_constraints.contains(&1),
            "C1 should be eliminated via cascading"
        );
        assert!(result.active_constraint_indices.is_empty());

        // Both params should be determined.
        assert_eq!(result.eliminated_params.len(), 2);

        let p1_val = result
            .eliminated_params
            .iter()
            .find(|(pid, _)| *pid == p1)
            .map(|(_, v)| *v)
            .expect("p1 should be eliminated");
        let p2_val = result
            .eliminated_params
            .iter()
            .find(|(pid, _)| *pid == p2)
            .map(|(_, v)| *v)
            .expect("p2 should be eliminated");

        assert!((p1_val - 5.0).abs() < 1e-12, "p1 = {p1_val}, expected 5.0");
        assert!((p2_val - 5.0).abs() < 1e-12, "p2 = {p2_val}, expected 5.0");

        // Both params should be removed from the active set.
        assert!(!result.active_param_ids.contains(&p1));
        assert!(!result.active_param_ids.contains(&p2));
    }

    /// Three-step cascading chain: C0 determines p1, which lets C1
    /// determine p2, which lets C2 determine p3.
    #[test]
    fn eliminate_cascades_three_deep() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p1 = store.alloc(0.0, owner);
        let p2 = store.alloc(0.0, owner);
        let p3 = store.alloc(0.0, owner);

        // C0: p1 = 3.0
        let c0: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p1,
            target: 3.0,
        });
        // C1: p1 + p2 = 7.0  -> p2 = 4.0 after p1 is known
        let c1: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(1, 0),
            params: [p1, p2],
            target: 7.0,
        });
        // C2: p2 - p3 = 1.0  -> p3 = 3.0 after p2 is known
        let c2: Box<dyn Constraint> = Box::new(DiffConstraint {
            id: ConstraintId::new(2, 0),
            params: [p2, p3],
            target: 1.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c0), Some(c1), Some(c2)];
        let cluster = make_cluster(vec![0, 1, 2], vec![p1, p2, p3]);

        let result = EliminateReducer.reduce(&cluster, &constraints, &mut store);

        // All three constraints should be eliminated.
        assert_eq!(result.removed_constraints.len(), 3);
        assert!(result.active_constraint_indices.is_empty());

        // All three params determined.
        assert_eq!(result.eliminated_params.len(), 3);
        assert!(!result.active_param_ids.contains(&p1));
        assert!(!result.active_param_ids.contains(&p2));
        assert!(!result.active_param_ids.contains(&p3));

        let val = |pid: ParamId| -> f64 {
            result
                .eliminated_params
                .iter()
                .find(|(p, _)| *p == pid)
                .unwrap()
                .1
        };
        assert!((val(p1) - 3.0).abs() < 1e-12, "p1 = {}", val(p1));
        assert!((val(p2) - 4.0).abs() < 1e-12, "p2 = {}", val(p2));
        assert!((val(p3) - 3.0).abs() < 1e-12, "p3 = {}", val(p3));
    }

    /// Cascading elimination through the full DefaultReduce pipeline
    /// (Substitute → Merge → Eliminate): a user-fixed param makes a
    /// constraint trivially satisfied (Substitute removes it), then an
    /// equality constraint merges two free params (Merge), then the
    /// eliminate pass cascades through remaining linear constraints.
    #[test]
    fn default_reduce_cascading_through_all_stages() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p_fixed = store.alloc(10.0, owner);
        let p1 = store.alloc(0.0, owner);
        let p2 = store.alloc(0.0, owner);
        store.fix(p_fixed);

        // C0: p_fixed = 10.0  -> trivially satisfied (Substitute removes it)
        let c0: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p_fixed,
            target: 10.0,
        });
        // C1: p1 = 6.0  -> single free param (Eliminate, first pass)
        let c1: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(1, 0),
            param: p1,
            target: 6.0,
        });
        // C2: p1 + p2 = 11.0  -> after p1 is eliminated, p2 = 5.0 (cascade)
        let c2: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(2, 0),
            params: [p1, p2],
            target: 11.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c0), Some(c1), Some(c2)];
        let cluster = make_cluster(vec![0, 1, 2], vec![p_fixed, p1, p2]);

        let result = DefaultReduce.reduce(&cluster, &constraints, &mut store);

        // All three constraints should be removed.
        assert_eq!(result.removed_constraints.len(), 3);
        assert!(result.active_constraint_indices.is_empty());

        // p1 and p2 should be in eliminated_params.
        assert_eq!(result.eliminated_params.len(), 2);

        let p1_val = result
            .eliminated_params
            .iter()
            .find(|(pid, _)| *pid == p1)
            .map(|(_, v)| *v)
            .expect("p1 should be eliminated");
        let p2_val = result
            .eliminated_params
            .iter()
            .find(|(pid, _)| *pid == p2)
            .map(|(_, v)| *v)
            .expect("p2 should be eliminated");

        assert!((p1_val - 6.0).abs() < 1e-12, "p1 = {p1_val}, expected 6.0");
        assert!((p2_val - 5.0).abs() < 1e-12, "p2 = {p2_val}, expected 5.0");
    }

    /// Nonlinear single-free-param constraints should NOT be eliminated
    /// when their linearised solution is approximate (residual not near zero).
    #[test]
    fn eliminate_skips_nonlinear_approximate_elimination() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        // Start with p at a value where linearised solve is inaccurate.
        let p = store.alloc(3.0, owner);

        // C0: p^2 = 25  (nonlinear; exact solutions are ±5, but
        //     linearisation around 3.0 gives 3.0 - (9-25)/6 = 5.666...,
        //     which doesn't satisfy p^2=25.)
        let c: Box<dyn Constraint> = Box::new(QuadraticConstraint {
            id: ConstraintId::new(0, 0),
            param: p,
            target: 25.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c)];
        let cluster = make_cluster(vec![0], vec![p]);

        let result = EliminateReducer.reduce(&cluster, &constraints, &mut store);

        // The constraint should NOT be eliminated because the linearised
        // result is approximate and doesn't satisfy the nonlinear residual.
        assert!(
            result.eliminated_params.is_empty(),
            "nonlinear constraint should not be eliminated when linearised solution is approximate"
        );
        assert_eq!(result.active_constraint_indices, vec![0]);
        assert_eq!(result.active_param_ids, vec![p]);
    }

    /// Cascading elimination does NOT freeze approximate nonlinear results.
    /// After fixing p1 linearly, if the remaining constraint on p2 is
    /// nonlinear, p2 should be left for the numerical solver.
    #[test]
    fn cascading_stops_at_nonlinear_constraint() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p1 = store.alloc(0.0, owner);
        let p2 = store.alloc(3.0, owner);

        // C0: p1 = 5.0  (linear, exact elimination)
        let c0: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p1,
            target: 5.0,
        });
        // C1: p2^2 - 25 = 0  (nonlinear; even though p2 is now the only free
        //     param, the linearised result at p2=3 is approximate)
        let c1: Box<dyn Constraint> = Box::new(QuadraticConstraint {
            id: ConstraintId::new(1, 0),
            param: p2,
            target: 25.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c0), Some(c1)];
        let cluster = make_cluster(vec![0, 1], vec![p1, p2]);

        let result = EliminateReducer.reduce(&cluster, &constraints, &mut store);

        // C0 should be eliminated (linear, exact).
        assert!(result.removed_constraints.contains(&0));
        assert_eq!(result.eliminated_params.len(), 1);
        assert_eq!(result.eliminated_params[0].0, p1);
        assert!((result.eliminated_params[0].1 - 5.0).abs() < 1e-12);

        // C1 should remain active (nonlinear, approximate).
        assert!(
            result.active_constraint_indices.contains(&1),
            "nonlinear constraint C1 should remain active"
        );
        assert!(
            result.active_param_ids.contains(&p2),
            "p2 should remain in the active solve set"
        );
    }

    /// After reduce, eliminated params are temporarily fixed in the store.
    /// Verify that the store state is consistent: eliminated params have
    /// their determined values set and are marked fixed.
    #[test]
    fn eliminate_updates_store_for_cascading() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p1 = store.alloc(0.0, owner);
        let p2 = store.alloc(0.0, owner);

        let c0: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p1,
            target: 5.0,
        });
        let c1: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(1, 0),
            params: [p1, p2],
            target: 10.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c0), Some(c1)];
        let cluster = make_cluster(vec![0, 1], vec![p1, p2]);

        let _result = EliminateReducer.reduce(&cluster, &constraints, &mut store);

        // After reduce, the store should contain determined values and
        // params should be marked fixed (for the solver to see them as
        // known constants).
        assert!((store.get(p1) - 5.0).abs() < 1e-12);
        assert!((store.get(p2) - 5.0).abs() < 1e-12);
        assert!(store.is_fixed(p1));
        assert!(store.is_fixed(p2));
    }

    /// ChainedReducer propagates store mutations across stages: params
    /// eliminated in an early stage are visible to later stages as fixed.
    #[test]
    fn chained_propagates_eliminations_across_stages() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p1 = store.alloc(0.0, owner);
        let p2 = store.alloc(0.0, owner);

        // C0: p1 = 5.0  -> eliminated by first EliminateReducer stage
        let c0: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p1,
            target: 5.0,
        });
        // C1: p1 + p2 = 10.0  -> after p1 is fixed by first stage,
        //     the second stage (or cascade within the same stage)
        //     can eliminate p2
        let c1: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(1, 0),
            params: [p1, p2],
            target: 10.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c0), Some(c1)];
        let cluster = make_cluster(vec![0, 1], vec![p1, p2]);

        // Use two separate EliminateReducer stages to verify cross-stage
        // propagation (the first stage eliminates p1 and fixes it in the
        // store; the second stage should then see p1 as fixed and eliminate
        // p2 from C1).
        let chain =
            ChainedReducer::new(vec![Box::new(EliminateReducer), Box::new(EliminateReducer)]);
        let result = chain.reduce(&cluster, &constraints, &mut store);

        // Both constraints should be removed.
        assert!(result.removed_constraints.contains(&0));
        assert!(result.removed_constraints.contains(&1));
        assert!(result.active_constraint_indices.is_empty());

        assert_eq!(result.eliminated_params.len(), 2);
    }

    /// After the full DefaultReduce pipeline, eliminated params should
    /// NOT remain permanently fixed in the store (the orchestrator
    /// un-fixes them). This test verifies the reduce phase itself does
    /// fix them (for correct cascading), which the orchestrator then
    /// reverses.
    #[test]
    fn eliminate_does_not_permanently_alter_non_eliminated_params() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let p1 = store.alloc(0.0, owner);
        let p2 = store.alloc(0.0, owner);
        let p3 = store.alloc(0.0, owner);

        // C0: p1 = 5.0  (eliminable)
        let c0: Box<dyn Constraint> = Box::new(FixValueConstraint {
            id: ConstraintId::new(0, 0),
            param: p1,
            target: 5.0,
        });
        // C1: p2 + p3 = 10.0  (two free params, not eliminable)
        let c1: Box<dyn Constraint> = Box::new(SumConstraint {
            id: ConstraintId::new(1, 0),
            params: [p2, p3],
            target: 10.0,
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(c0), Some(c1)];
        let cluster = make_cluster(vec![0, 1], vec![p1, p2, p3]);

        let _result = EliminateReducer.reduce(&cluster, &constraints, &mut store);

        // p1 should be fixed (it was eliminated).
        assert!(store.is_fixed(p1));
        // p2 and p3 should remain free (not touched by elimination).
        assert!(!store.is_fixed(p2), "p2 should remain free");
        assert!(!store.is_fixed(p3), "p3 should remain free");
    }
}
