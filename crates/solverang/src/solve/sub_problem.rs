//! [`ReducedSubProblem`] — bridges trait-based constraints to the existing
//! [`Problem`](crate::problem::Problem) trait.
//!
//! This is the crucial adapter that makes all existing solvers work with the
//! new entity/constraint/param system. Each cluster of coupled constraints
//! becomes a `ReducedSubProblem` that implements `Problem`. The solver sees
//! a standard nonlinear system with column-indexed variables and row-indexed
//! residuals — it never touches `ParamId` or `Entity` directly.

use crate::constraint::Constraint;
use crate::id::ParamId;
use crate::param::{ParamStore, SolverMapping};
use crate::problem::Problem;

/// A reduced sub-problem for a single cluster, ready for numerical solving.
///
/// Implements the existing [`Problem`] trait so that all existing solvers
/// (Newton-Raphson, Levenberg-Marquardt, Auto, Robust, Parallel, Sparse)
/// work unchanged.
///
/// # Construction
///
/// A `ReducedSubProblem` is built from:
/// - A reference to the shared [`ParamStore`] (read-only for snapshots)
/// - The set of constraints in this cluster
/// - The set of parameter IDs that are free (solvable) in this cluster
///
/// The constructor builds a [`SolverMapping`] that maps between `ParamId`s
/// and the dense column indices that the solver works with.
///
/// # How it works
///
/// When the solver calls `residuals(x)` or `jacobian(x)`:
/// 1. The solver's `x` vector is written into a snapshot of the `ParamStore`
///    using the `SolverMapping` (column index -> `ParamId`).
/// 2. Each constraint evaluates its residuals/Jacobian by reading from the
///    snapshot via `ParamId`.
/// 3. The Jacobian's `ParamId`-based entries are mapped back to column
///    indices via `SolverMapping`.
///
/// This two-way mapping is the bridge between the solver's dense column
/// world and the constraint system's `ParamId` world.
pub struct ReducedSubProblem<'a> {
    /// Reference to the shared parameter store.
    store: &'a ParamStore,
    /// Mapping between free params in this cluster and solver column indices.
    mapping: SolverMapping,
    /// Constraints that belong to this cluster.
    constraints: Vec<&'a dyn Constraint>,
    /// Initial parameter values (extracted at construction time).
    initial_values: Vec<f64>,
    /// Human-readable name for diagnostics.
    name: String,
}

impl<'a> ReducedSubProblem<'a> {
    /// Create a new sub-problem for the given constraints and parameters.
    ///
    /// `param_ids` should include all parameters touched by the constraints
    /// in this cluster. Fixed parameters are automatically excluded from the
    /// solver mapping (they remain at their current values in the snapshot).
    ///
    /// # Arguments
    ///
    /// * `store` - The shared parameter store (values are snapshotted during solving)
    /// * `constraints` - The constraints in this cluster
    /// * `param_ids` - All parameter IDs relevant to this cluster (free + fixed)
    pub fn new(
        store: &'a ParamStore,
        constraints: Vec<&'a dyn Constraint>,
        param_ids: &[ParamId],
    ) -> Self {
        let mapping = store.build_solver_mapping_for(param_ids);
        let initial_values = store.extract_free_values(&mapping);
        let name = format!("cluster({}c/{}v)", constraints.len(), mapping.len());
        Self {
            store,
            mapping,
            constraints,
            initial_values,
            name,
        }
    }

    /// The solver mapping for this sub-problem.
    ///
    /// After solving, use this mapping to write the solution back into the
    /// `ParamStore` via [`ParamStore::write_free_values`].
    pub fn mapping(&self) -> &SolverMapping {
        &self.mapping
    }

    /// The constraints in this sub-problem.
    pub fn constraints(&self) -> &[&'a dyn Constraint] {
        &self.constraints
    }

    /// Build a snapshot of the param store with solver values `x` applied.
    ///
    /// This is the core mechanism: clone the store, overwrite free params
    /// with the solver's current iterate, and hand it to constraints.
    fn apply_to_snapshot(&self, x: &[f64]) -> ParamStore {
        let mut snapshot = self.store.snapshot();
        for (col, &param_id) in self.mapping.col_to_param.iter().enumerate() {
            snapshot.set(param_id, x[col]);
        }
        snapshot
    }
}

impl Problem for ReducedSubProblem<'_> {
    fn name(&self) -> &str {
        &self.name
    }

    fn variable_count(&self) -> usize {
        self.mapping.col_to_param.len()
    }

    fn residual_count(&self) -> usize {
        self.constraints.iter().map(|c| c.equation_count()).sum()
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        let snapshot = self.apply_to_snapshot(x);

        let mut residuals = Vec::with_capacity(self.residual_count());
        for constraint in &self.constraints {
            let r = constraint.residuals(&snapshot);
            let w = constraint.weight();
            if (w - 1.0).abs() > f64::EPSILON {
                residuals.extend(r.iter().map(|v| v * w));
            } else {
                residuals.extend(r);
            }
        }
        residuals
    }

    fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
        let snapshot = self.apply_to_snapshot(x);

        let mut entries = Vec::new();
        let mut row_offset = 0;

        for constraint in &self.constraints {
            let w = constraint.weight();
            for (local_row, param_id, value) in constraint.jacobian(&snapshot) {
                // Map ParamId -> column index. If the param is fixed or not in
                // this cluster's mapping, skip it (its column doesn't exist in
                // the solver's variable space).
                if let Some(&col) = self.mapping.param_to_col.get(&param_id) {
                    let weighted = if (w - 1.0).abs() > f64::EPSILON {
                        value * w
                    } else {
                        value
                    };
                    entries.push((row_offset + local_row, col, weighted));
                }
            }
            row_offset += constraint.equation_count();
        }
        entries
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        self.initial_values.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ConstraintId, EntityId};

    // -------------------------------------------------------------------
    // Test constraint: f(a, b) = a + b - target  (one equation, two params)
    // -------------------------------------------------------------------
    struct SumConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        params: Vec<ParamId>,
        target: f64,
    }

    impl SumConstraint {
        fn new(id: ConstraintId, entity: EntityId, params: Vec<ParamId>, target: f64) -> Self {
            Self {
                id,
                entity_ids: vec![entity],
                params,
                target,
            }
        }
    }

    impl Constraint for SumConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "SumConstraint"
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
            vec![(0, self.params[0], 1.0), (0, self.params[1], 1.0)]
        }
    }

    // -------------------------------------------------------------------
    // Test constraint: f(a) = a - target  (fix a single param)
    // -------------------------------------------------------------------
    struct FixValueConstraint {
        id: ConstraintId,
        entity_ids: Vec<EntityId>,
        param: ParamId,
        target: f64,
        weight: f64,
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
        fn weight(&self) -> f64 {
            self.weight
        }
    }

    /// Helper: create a dummy EntityId for tests.
    fn dummy_entity() -> EntityId {
        EntityId::new(0, 0)
    }

    #[test]
    fn test_sub_problem_dimensions() {
        let mut store = ParamStore::new();
        let owner = dummy_entity();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);

        let c_id = ConstraintId::new(0, 0);
        let constraint = SumConstraint::new(c_id, owner, vec![a, b], 5.0);
        let constraints: Vec<&dyn Constraint> = vec![&constraint];

        let sub = ReducedSubProblem::new(&store, constraints, &[a, b]);
        assert_eq!(sub.variable_count(), 2);
        assert_eq!(sub.residual_count(), 1);
        assert_eq!(sub.name(), "cluster(1c/2v)");
    }

    #[test]
    fn test_sub_problem_residuals() {
        let mut store = ParamStore::new();
        let owner = dummy_entity();
        let a = store.alloc(3.0, owner);
        let b = store.alloc(4.0, owner);

        let c_id = ConstraintId::new(0, 0);
        let constraint = SumConstraint::new(c_id, owner, vec![a, b], 10.0);
        let constraints: Vec<&dyn Constraint> = vec![&constraint];

        let sub = ReducedSubProblem::new(&store, constraints, &[a, b]);

        // Initial values are [3.0, 4.0]; residual = 3+4-10 = -3
        let x0 = sub.initial_point(1.0);
        assert_eq!(x0.len(), 2);
        let r = sub.residuals(&x0);
        assert_eq!(r.len(), 1);
        assert!((r[0] - (-3.0)).abs() < 1e-12);

        // At x = [5.0, 5.0]; residual = 5+5-10 = 0
        let r = sub.residuals(&[5.0, 5.0]);
        assert!(r[0].abs() < 1e-12);
    }

    #[test]
    fn test_sub_problem_jacobian() {
        let mut store = ParamStore::new();
        let owner = dummy_entity();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);

        let c_id = ConstraintId::new(0, 0);
        let constraint = SumConstraint::new(c_id, owner, vec![a, b], 5.0);
        let constraints: Vec<&dyn Constraint> = vec![&constraint];

        let sub = ReducedSubProblem::new(&store, constraints, &[a, b]);
        let jac = sub.jacobian(&[1.0, 2.0]);

        // Sum constraint: d/da = 1, d/db = 1
        assert_eq!(jac.len(), 2);
        // Both entries are in row 0
        assert!(jac.iter().all(|(row, _, _)| *row == 0));
        // Both derivatives are 1.0
        assert!(jac.iter().all(|(_, _, val)| (*val - 1.0).abs() < 1e-12));
        // Columns 0 and 1
        let cols: Vec<usize> = jac.iter().map(|(_, c, _)| *c).collect();
        assert!(cols.contains(&0));
        assert!(cols.contains(&1));
    }

    #[test]
    fn test_sub_problem_with_fixed_param() {
        let mut store = ParamStore::new();
        let owner = dummy_entity();
        let a = store.alloc(3.0, owner);
        let b = store.alloc(7.0, owner);

        // Fix parameter b
        store.fix(b);

        let c_id = ConstraintId::new(0, 0);
        let constraint = SumConstraint::new(c_id, owner, vec![a, b], 10.0);
        let constraints: Vec<&dyn Constraint> = vec![&constraint];

        let sub = ReducedSubProblem::new(&store, constraints, &[a, b]);

        // Only 'a' is free
        assert_eq!(sub.variable_count(), 1);
        assert_eq!(sub.residual_count(), 1);

        // Initial value of free param 'a' is 3.0
        let x0 = sub.initial_point(1.0);
        assert_eq!(x0.len(), 1);
        assert!((x0[0] - 3.0).abs() < 1e-12);

        // Residual: a + b_fixed(7.0) - 10 = 3 + 7 - 10 = 0
        let r = sub.residuals(&x0);
        assert!(r[0].abs() < 1e-12);

        // Jacobian: only the column for 'a' appears (derivative = 1.0)
        let jac = sub.jacobian(&x0);
        assert_eq!(jac.len(), 1);
        assert_eq!(jac[0].0, 0); // row 0
        assert_eq!(jac[0].1, 0); // col 0 (only free param)
        assert!((jac[0].2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sub_problem_weight() {
        let mut store = ParamStore::new();
        let owner = dummy_entity();
        let a = store.alloc(5.0, owner);

        let c_id = ConstraintId::new(0, 0);
        let constraint = FixValueConstraint {
            id: c_id,
            entity_ids: vec![owner],
            param: a,
            target: 3.0,
            weight: 2.5,
        };
        let constraints: Vec<&dyn Constraint> = vec![&constraint];

        let sub = ReducedSubProblem::new(&store, constraints, &[a]);

        // Residual without weight: 5-3 = 2; with weight 2.5: 5.0
        let r = sub.residuals(&[5.0]);
        assert!((r[0] - 5.0).abs() < 1e-12);

        // Jacobian without weight: 1.0; with weight 2.5: 2.5
        let jac = sub.jacobian(&[5.0]);
        assert_eq!(jac.len(), 1);
        assert!((jac[0].2 - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_sub_problem_multiple_constraints() {
        let mut store = ParamStore::new();
        let owner = dummy_entity();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);

        let c1 = SumConstraint::new(ConstraintId::new(0, 0), owner, vec![a, b], 5.0);
        let c2 = FixValueConstraint {
            id: ConstraintId::new(1, 0),
            entity_ids: vec![owner],
            param: a,
            target: 3.0,
            weight: 1.0,
        };

        let constraints: Vec<&dyn Constraint> = vec![&c1, &c2];
        let sub = ReducedSubProblem::new(&store, constraints, &[a, b]);

        assert_eq!(sub.variable_count(), 2);
        assert_eq!(sub.residual_count(), 2); // 1 from c1 + 1 from c2

        let r = sub.residuals(&[3.0, 2.0]);
        // c1: 3+2-5 = 0
        assert!(r[0].abs() < 1e-12);
        // c2: 3-3 = 0
        assert!(r[1].abs() < 1e-12);
    }

    #[test]
    fn test_sub_problem_mapping_accessor() {
        let mut store = ParamStore::new();
        let owner = dummy_entity();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);

        let c_id = ConstraintId::new(0, 0);
        let constraint = SumConstraint::new(c_id, owner, vec![a, b], 3.0);
        let constraints: Vec<&dyn Constraint> = vec![&constraint];

        let sub = ReducedSubProblem::new(&store, constraints, &[a, b]);
        let mapping = sub.mapping();

        assert_eq!(mapping.len(), 2);
        assert!(!mapping.is_empty());
        assert!(mapping.param_to_col.contains_key(&a));
        assert!(mapping.param_to_col.contains_key(&b));
    }

    #[test]
    fn test_sub_problem_no_free_params() {
        let mut store = ParamStore::new();
        let owner = dummy_entity();
        let a = store.alloc(5.0, owner);
        let b = store.alloc(5.0, owner);
        store.fix(a);
        store.fix(b);

        let c_id = ConstraintId::new(0, 0);
        let constraint = SumConstraint::new(c_id, owner, vec![a, b], 10.0);
        let constraints: Vec<&dyn Constraint> = vec![&constraint];

        let sub = ReducedSubProblem::new(&store, constraints, &[a, b]);

        assert_eq!(sub.variable_count(), 0);
        assert_eq!(sub.residual_count(), 1);
        assert!(sub.mapping().is_empty());
    }

    #[test]
    fn test_sub_problem_snapshot_isolation() {
        // Verify that evaluating residuals does not mutate the original store.
        let mut store = ParamStore::new();
        let owner = dummy_entity();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);

        let c_id = ConstraintId::new(0, 0);
        let constraint = SumConstraint::new(c_id, owner, vec![a, b], 5.0);
        let constraints: Vec<&dyn Constraint> = vec![&constraint];

        let sub = ReducedSubProblem::new(&store, constraints, &[a, b]);

        // Evaluate at a different point
        let _r = sub.residuals(&[99.0, 99.0]);

        // Original store is unchanged
        assert!((store.get(a) - 1.0).abs() < 1e-12);
        assert!((store.get(b) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_sub_problem_row_offsets_in_jacobian() {
        // With two constraints, the second constraint's rows should be offset.
        let mut store = ParamStore::new();
        let owner = dummy_entity();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);

        let c1 = SumConstraint::new(ConstraintId::new(0, 0), owner, vec![a, b], 5.0);
        let c2 = SumConstraint::new(ConstraintId::new(1, 0), owner, vec![a, b], 3.0);

        let constraints: Vec<&dyn Constraint> = vec![&c1, &c2];
        let sub = ReducedSubProblem::new(&store, constraints, &[a, b]);

        let jac = sub.jacobian(&[1.0, 2.0]);

        // c1 produces row 0 entries; c2 produces row 1 entries
        let rows: Vec<usize> = jac.iter().map(|(r, _, _)| *r).collect();
        assert!(rows.contains(&0)); // from c1
        assert!(rows.contains(&1)); // from c2 (offset by c1.equation_count() == 1)
    }
}
