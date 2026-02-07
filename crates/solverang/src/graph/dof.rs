//! Per-entity degrees-of-freedom analysis using null-space projection.
//!
//! This module computes effective degrees of freedom (DOF) for individual
//! entities and for the full system.  The global DOF equals the number of
//! free parameters minus the Jacobian rank.  Per-entity DOF is obtained by
//! restricting the Jacobian to each entity's parameter columns and computing
//! the rank of that sub-matrix.
//!
//! # Algorithm
//!
//! 1. Build the dense Jacobian matrix for the full system.
//! 2. Compute the global rank via SVD.
//! 3. For each entity, select the Jacobian columns corresponding to its free
//!    parameters and compute the local rank.
//! 4. `entity_dof = free_entity_params - local_rank`.

use std::collections::HashMap;

use nalgebra::DMatrix;

use crate::constraint::Constraint;
use crate::entity::Entity;
use crate::id::EntityId;
use crate::param::{ParamStore, SolverMapping};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// DOF analysis result for a single entity.
#[derive(Clone, Debug)]
pub struct EntityDof {
    /// The entity's generational ID.
    pub entity_id: EntityId,
    /// Total parameters the entity owns (including fixed).
    pub total_params: usize,
    /// Number of fixed (immovable) parameters.
    pub fixed_params: usize,
    /// Effective degrees of freedom:
    /// `free_params - rank(Jacobian restricted to entity columns)`.
    pub dof: usize,
}

/// DOF analysis for a cluster or the full system.
#[derive(Clone, Debug)]
pub struct DofAnalysis {
    /// Per-entity DOF breakdown.
    pub entities: Vec<EntityDof>,
    /// Total DOF across the system (can be negative if over-constrained).
    pub total_dof: i32,
    /// Total free (non-fixed) parameters.
    pub total_free_params: usize,
    /// Total number of equations (Jacobian rows).
    pub total_equations: usize,
}

impl DofAnalysis {
    /// True when the system is exactly constrained (zero DOF).
    pub fn is_well_constrained(&self) -> bool {
        self.total_dof == 0
    }

    /// True when there are more equations than the Jacobian can support
    /// (total_dof < 0, indicating redundancy / over-constraint).
    pub fn is_over_constrained(&self) -> bool {
        self.total_dof < 0
    }

    /// True when the system has remaining freedom (total_dof > 0).
    pub fn is_under_constrained(&self) -> bool {
        self.total_dof > 0
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Compute degrees of freedom for entities.
///
/// Global DOF is `free_params - rank(Jacobian)`.
///
/// Per-entity DOF is computed by restricting the Jacobian to the columns
/// belonging to each entity's free parameters, then:
///
/// `entity_dof = entity_free_params - rank(restricted_Jacobian)`
///
/// This counts how many of the entity's free parameters remain unconstrained
/// by the current set of constraints.
///
/// # Note
///
/// Because constraints that span multiple entities contribute rank to each
/// entity independently, the sum of per-entity DOFs may exceed the global
/// DOF.  This is expected and simply reflects shared constraint coupling.
pub fn analyze_dof(
    entities: &[&dyn Entity],
    constraints: &[(usize, &dyn Constraint)],
    store: &ParamStore,
    mapping: &SolverMapping,
) -> DofAnalysis {
    let ncols = mapping.len();
    let tolerance = 1e-10;

    // Count total equations and build the dense Jacobian.
    let mut total_equations = 0usize;
    let mut row_offsets: Vec<usize> = Vec::with_capacity(constraints.len());
    for (_, c) in constraints {
        row_offsets.push(total_equations);
        total_equations += c.equation_count();
    }

    let jac = if total_equations > 0 && ncols > 0 {
        Some(build_dense_jacobian(
            constraints,
            &row_offsets,
            store,
            mapping,
            total_equations,
            ncols,
        ))
    } else {
        None
    };

    // Global rank.
    let global_rank = jac
        .as_ref()
        .map(|j| compute_rank(j, tolerance))
        .unwrap_or(0);
    let total_dof = ncols as i32 - global_rank as i32;

    // --- Per-entity DOF ---
    // Pre-compute which columns belong to each entity for fast lookup.
    let entity_cols = compute_entity_columns(entities, store, mapping);

    let entity_dofs: Vec<EntityDof> = entities
        .iter()
        .map(|entity| {
            let eid = entity.id();
            let all_params = entity.params();
            let total_params = all_params.len();
            let fixed_params = all_params.iter().filter(|&&pid| store.is_fixed(pid)).count();
            let free_cols = entity_cols.get(&eid).cloned().unwrap_or_default();
            let free_count = free_cols.len();

            let dof = if free_count == 0 || total_equations == 0 {
                // No free params or no constraints: DOF = free param count.
                free_count
            } else if let Some(ref full_jac) = jac {
                // Build the sub-Jacobian restricted to this entity's columns.
                let sub_jac = extract_columns(full_jac, &free_cols);
                let local_rank = compute_rank(&sub_jac, tolerance);
                free_count.saturating_sub(local_rank)
            } else {
                free_count
            };

            EntityDof {
                entity_id: eid,
                total_params,
                fixed_params,
                dof,
            }
        })
        .collect();

    DofAnalysis {
        entities: entity_dofs,
        total_dof,
        total_free_params: ncols,
        total_equations,
    }
}

/// Quick DOF estimate without SVD (just count equations vs variables).
///
/// Returns `free_param_count - equation_count`.  This is an upper bound on
/// the true DOF because it assumes all equations are independent.
pub fn quick_dof(free_param_count: usize, equation_count: usize) -> i32 {
    free_param_count as i32 - equation_count as i32
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a dense Jacobian matrix from sparse constraint triplets.
fn build_dense_jacobian(
    constraints: &[(usize, &dyn Constraint)],
    row_offsets: &[usize],
    store: &ParamStore,
    mapping: &SolverMapping,
    nrows: usize,
    ncols: usize,
) -> DMatrix<f64> {
    let mut jac = DMatrix::zeros(nrows, ncols);

    for (ci, (_, constraint)) in constraints.iter().enumerate() {
        let row_start = row_offsets[ci];
        for (local_row, param_id, value) in constraint.jacobian(store) {
            if let Some(&col) = mapping.param_to_col.get(&param_id) {
                let global_row = row_start + local_row;
                if global_row < nrows && col < ncols {
                    jac[(global_row, col)] = value;
                }
            }
        }
    }

    jac
}

/// Compute the numerical rank of a matrix using SVD.
fn compute_rank(matrix: &DMatrix<f64>, tolerance: f64) -> usize {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return 0;
    }
    let svd = matrix.clone().svd(false, false);
    svd.singular_values
        .iter()
        .filter(|&&s| s > tolerance)
        .count()
}

/// Map each entity to its free-parameter column indices in the solver mapping.
fn compute_entity_columns(
    entities: &[&dyn Entity],
    store: &ParamStore,
    mapping: &SolverMapping,
) -> HashMap<EntityId, Vec<usize>> {
    let mut map = HashMap::with_capacity(entities.len());
    for entity in entities {
        let cols: Vec<usize> = entity
            .params()
            .iter()
            .filter(|&&pid| !store.is_fixed(pid))
            .filter_map(|pid| mapping.param_to_col.get(pid).copied())
            .collect();
        map.insert(entity.id(), cols);
    }
    map
}

/// Extract a column subset from a matrix, returning a new dense matrix.
fn extract_columns(matrix: &DMatrix<f64>, cols: &[usize]) -> DMatrix<f64> {
    let nrows = matrix.nrows();
    let ncols = cols.len();
    let mut sub = DMatrix::zeros(nrows, ncols);
    for (j, &col) in cols.iter().enumerate() {
        for i in 0..nrows {
            sub[(i, j)] = matrix[(i, col)];
        }
    }
    sub
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    // -- Stub types ----------------------------------------------------------

    struct StubEntity {
        id: EntityId,
        params: Vec<ParamId>,
    }

    impl Entity for StubEntity {
        fn id(&self) -> EntityId {
            self.id
        }
        fn params(&self) -> &[ParamId] {
            &self.params
        }
        fn name(&self) -> &str {
            "stub"
        }
    }

    struct StubConstraint {
        id: ConstraintId,
        entities: Vec<EntityId>,
        params: Vec<ParamId>,
        neq: usize,
        residual_fn: Box<dyn Fn(&ParamStore) -> Vec<f64> + Send + Sync>,
        jacobian_fn: Box<dyn Fn(&ParamStore) -> Vec<(usize, ParamId, f64)> + Send + Sync>,
    }

    impl Constraint for StubConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "stub"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entities
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            self.neq
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            (self.residual_fn)(store)
        }
        fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            (self.jacobian_fn)(store)
        }
    }

    // -- Tests ---------------------------------------------------------------

    #[test]
    fn test_quick_dof() {
        assert_eq!(quick_dof(4, 2), 2);
        assert_eq!(quick_dof(2, 2), 0);
        assert_eq!(quick_dof(1, 3), -2);
        assert_eq!(quick_dof(0, 0), 0);
    }

    #[test]
    fn test_unconstrained_entity() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);
        let py = store.alloc(2.0, eid);
        let mapping = store.build_solver_mapping();

        let entity = StubEntity {
            id: eid,
            params: vec![px, py],
        };

        let constraints: Vec<(usize, &dyn Constraint)> = vec![];
        let entities: Vec<&dyn Entity> = vec![&entity];
        let result = analyze_dof(&entities, &constraints, &store, &mapping);

        assert_eq!(result.total_dof, 2);
        assert_eq!(result.total_free_params, 2);
        assert_eq!(result.total_equations, 0);
        assert!(result.is_under_constrained());
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].dof, 2);
        assert_eq!(result.entities[0].total_params, 2);
        assert_eq!(result.entities[0].fixed_params, 0);
    }

    #[test]
    fn test_fully_constrained_entity() {
        // Two params, two independent constraints: x=1, y=2.
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);
        let py = store.alloc(2.0, eid);
        let mapping = store.build_solver_mapping();

        let entity = StubEntity {
            id: eid,
            params: vec![px, py],
        };

        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![px],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(px) - 1.0]),
            jacobian_fn: Box::new(move |_| vec![(0, px, 1.0)]),
        };
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![eid],
            params: vec![py],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(py) - 2.0]),
            jacobian_fn: Box::new(move |_| vec![(0, py, 1.0)]),
        };

        let constraints: Vec<(usize, &dyn Constraint)> = vec![(0, &c0), (1, &c1)];
        let entities: Vec<&dyn Entity> = vec![&entity];
        let result = analyze_dof(&entities, &constraints, &store, &mapping);

        assert_eq!(result.total_dof, 0);
        assert!(result.is_well_constrained());
        assert_eq!(result.entities[0].dof, 0);
    }

    #[test]
    fn test_partially_constrained() {
        // Two params, one constraint: x=1.  y is free.
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);
        let py = store.alloc(2.0, eid);
        let mapping = store.build_solver_mapping();

        let entity = StubEntity {
            id: eid,
            params: vec![px, py],
        };

        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![px],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(px) - 1.0]),
            jacobian_fn: Box::new(move |_| vec![(0, px, 1.0)]),
        };

        let constraints: Vec<(usize, &dyn Constraint)> = vec![(0, &c0)];
        let entities: Vec<&dyn Entity> = vec![&entity];
        let result = analyze_dof(&entities, &constraints, &store, &mapping);

        assert_eq!(result.total_dof, 1);
        assert!(result.is_under_constrained());
        assert_eq!(result.entities[0].dof, 1);
    }

    #[test]
    fn test_fixed_params_excluded() {
        // Two params, px is fixed.  One constraint on py.
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);
        let py = store.alloc(2.0, eid);
        store.fix(px);
        let mapping = store.build_solver_mapping();

        let entity = StubEntity {
            id: eid,
            params: vec![px, py],
        };

        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![py],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(py) - 2.0]),
            jacobian_fn: Box::new(move |_| vec![(0, py, 1.0)]),
        };

        let constraints: Vec<(usize, &dyn Constraint)> = vec![(0, &c0)];
        let entities: Vec<&dyn Entity> = vec![&entity];
        let result = analyze_dof(&entities, &constraints, &store, &mapping);

        assert_eq!(result.total_free_params, 1);
        assert_eq!(result.total_dof, 0);
        assert_eq!(result.entities[0].total_params, 2);
        assert_eq!(result.entities[0].fixed_params, 1);
        assert_eq!(result.entities[0].dof, 0);
    }

    #[test]
    fn test_two_entities_shared_constraint() {
        // Entity A (px), Entity B (py), constraint: px + py = 0.
        let eid_a = EntityId::new(0, 0);
        let eid_b = EntityId::new(1, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid_a);
        let py = store.alloc(-1.0, eid_b);
        let mapping = store.build_solver_mapping();

        let entity_a = StubEntity {
            id: eid_a,
            params: vec![px],
        };
        let entity_b = StubEntity {
            id: eid_b,
            params: vec![py],
        };

        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid_a, eid_b],
            params: vec![px, py],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(px) + s.get(py)]),
            jacobian_fn: Box::new(move |_| vec![(0, px, 1.0), (0, py, 1.0)]),
        };

        let constraints: Vec<(usize, &dyn Constraint)> = vec![(0, &c0)];
        let entities: Vec<&dyn Entity> = vec![&entity_a, &entity_b];
        let result = analyze_dof(&entities, &constraints, &store, &mapping);

        // Global: 2 free params, 1 equation, rank 1 -> DOF = 1.
        assert_eq!(result.total_dof, 1);

        // Per-entity with sub-Jacobian approach:
        // Entity A: sub-Jac [1.0], rank 1 -> DOF = 1-1 = 0
        // Entity B: sub-Jac [1.0], rank 1 -> DOF = 1-1 = 0
        // Sum = 0, which is less than global DOF = 1.
        // This reflects that each entity alone is constrained, but they
        // can move together along the null direction.
        assert_eq!(result.entities[0].dof, 0);
        assert_eq!(result.entities[1].dof, 0);
    }

    #[test]
    fn test_dof_analysis_helpers() {
        let analysis = DofAnalysis {
            entities: vec![],
            total_dof: 0,
            total_free_params: 4,
            total_equations: 4,
        };
        assert!(analysis.is_well_constrained());
        assert!(!analysis.is_over_constrained());
        assert!(!analysis.is_under_constrained());
    }

    #[test]
    fn test_over_constrained() {
        let analysis = DofAnalysis {
            entities: vec![],
            total_dof: -2,
            total_free_params: 2,
            total_equations: 4,
        };
        assert!(analysis.is_over_constrained());
    }

    #[test]
    fn test_compute_rank_internal() {
        let m = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(compute_rank(&m, 1e-10), 2);

        let empty = DMatrix::<f64>::zeros(0, 0);
        assert_eq!(compute_rank(&empty, 1e-10), 0);
    }
}
