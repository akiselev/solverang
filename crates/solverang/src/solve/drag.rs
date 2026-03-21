//! Null-space projection for under-constrained drag.
//!
//! When a constraint system is under-constrained (more free parameters than
//! equations), the user can drag entities along the remaining degrees of
//! freedom. This module projects a desired displacement onto the null space
//! of the constraint Jacobian so the displacement preserves all constraints.
//!
//! # Algorithm
//!
//! 1. Build the dense Jacobian **J** at the current configuration.
//! 2. Compute the SVD: **J = U S V^T**.
//! 3. The null space of **J** is spanned by columns of **V** whose
//!    corresponding singular values are below `tolerance`.
//! 4. Project the displacement: **d_proj = N N^T d** where **N** is the
//!    null-space basis.

use nalgebra::DMatrix;

use crate::constraint::Constraint;
use crate::id::ParamId;
use crate::param::{ParamStore, SolverMapping};

/// Result of a drag operation.
#[derive(Clone, Debug)]
pub struct DragResult {
    /// The projected displacement (in solver variable space).
    pub projected_displacement: Vec<f64>,
    /// How much of the original displacement was preserved (0.0 to 1.0).
    pub preservation_ratio: f64,
}

/// Build a dense Jacobian matrix from trait-based constraints.
///
/// Rows correspond to constraint equations (in order), columns to free
/// parameters (as mapped by `mapping`).
fn build_dense_jacobian(
    constraints: &[&dyn Constraint],
    store: &ParamStore,
    mapping: &SolverMapping,
) -> DMatrix<f64> {
    let ncols = mapping.len();

    // Count total equation rows.
    let nrows: usize = constraints.iter().map(|c| c.equation_count()).sum();

    let mut j = DMatrix::zeros(nrows, ncols);

    let mut row_offset = 0;
    for &c in constraints {
        let entries = c.jacobian(store);
        for (local_row, param_id, value) in entries {
            if let Some(&col) = mapping.param_to_col.get(&param_id) {
                let global_row = row_offset + local_row;
                if global_row < nrows && col < ncols {
                    j[(global_row, col)] = value;
                }
            }
        }
        row_offset += c.equation_count();
    }

    j
}

/// Project a desired parameter displacement onto the constraint manifold's
/// null space, so that the displacement satisfies all constraints.
///
/// This is used for interactive dragging: the user wants to move a point,
/// and we project their intent onto the directions allowed by the constraints.
///
/// # Algorithm
///
/// 1. Build Jacobian **J** at the current point.
/// 2. Compute null space **N** of **J** (via SVD: columns of **V**
///    corresponding to singular values below `tolerance`).
/// 3. Project displacement: **d_proj = N * N^T * d**.
///
/// # Arguments
///
/// * `constraints` - The active constraints.
/// * `store` - Current parameter values.
/// * `mapping` - Mapping from `ParamId` to solver column indices.
/// * `desired_displacement` - `(param, delta)` pairs describing the user's
///   intended move.
/// * `tolerance` - Singular values below this threshold are treated as zero
///   when determining the null space.
///
/// # Returns
///
/// A [`DragResult`] containing the projected displacement in solver variable
/// space and the preservation ratio (how much of the original displacement
/// survived projection).
pub fn project_drag(
    constraints: &[&dyn Constraint],
    store: &ParamStore,
    mapping: &SolverMapping,
    desired_displacement: &[(ParamId, f64)],
    tolerance: f64,
) -> DragResult {
    let n = mapping.len();

    // Edge case: no free parameters.
    if n == 0 {
        return DragResult {
            projected_displacement: Vec::new(),
            preservation_ratio: 0.0,
        };
    }

    // Build the displacement vector in solver variable space.
    let mut d = vec![0.0; n];
    for &(pid, delta) in desired_displacement {
        if let Some(&col) = mapping.param_to_col.get(&pid) {
            d[col] = delta;
        }
    }

    let d_norm_sq: f64 = d.iter().map(|x| x * x).sum();

    // Edge case: zero displacement.
    if d_norm_sq < f64::EPSILON * f64::EPSILON {
        return DragResult {
            projected_displacement: vec![0.0; n],
            preservation_ratio: 1.0,
        };
    }

    // Edge case: no constraints -- everything is free.
    if constraints.is_empty() {
        return DragResult {
            projected_displacement: d,
            preservation_ratio: 1.0,
        };
    }

    // Build the dense Jacobian.
    let j = build_dense_jacobian(constraints, store, mapping);

    // SVD of J.
    let svd = j.svd(false, true);

    // Extract V^T; the null space is the rows of V^T corresponding to
    // near-zero singular values.
    let v_t = match svd.v_t {
        Some(ref vt) => vt,
        None => {
            // If SVD did not compute V, return the original displacement.
            return DragResult {
                projected_displacement: d,
                preservation_ratio: 1.0,
            };
        }
    };

    let singular_values = &svd.singular_values;

    // Project onto the null space by *removing* the range-space components.
    // The thin SVD gives V^T with min(m,n) rows — exactly the range-space
    // basis vectors. We compute: d_proj = d - sum_i (v_i . d) * v_i
    // for all i where singular_values[i] > tolerance.
    let mut projected = d.clone();

    for i in 0..singular_values.len().min(v_t.nrows()) {
        if singular_values[i] >= tolerance {
            // Row i of V^T is a range-space basis vector. Remove it.
            let dot: f64 = (0..n).map(|k| v_t[(i, k)] * d[k]).sum();
            for k in 0..n {
                projected[k] -= dot * v_t[(i, k)];
            }
        }
    }

    let proj_norm_sq: f64 = projected.iter().map(|x| x * x).sum();
    let preservation_ratio = if d_norm_sq > f64::EPSILON {
        (proj_norm_sq / d_norm_sq).sqrt().min(1.0)
    } else {
        0.0
    };

    DragResult {
        projected_displacement: projected,
        preservation_ratio,
    }
}

/// Convenience function: apply the projected drag result back to the store.
///
/// After calling [`project_drag`], use this to actually move the parameters.
pub fn apply_drag(store: &mut ParamStore, mapping: &SolverMapping, result: &DragResult) {
    for (col, &pid) in mapping.col_to_param.iter().enumerate() {
        if col < result.projected_displacement.len() {
            let current = store.get(pid);
            store.set(pid, current + result.projected_displacement[col]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ConstraintId, EntityId, ParamId};

    // --- Stub constraint for testing ---

    struct FixYConstraint {
        id: ConstraintId,
        entity: EntityId,
        y_param: ParamId,
        target_y: f64,
    }

    impl Constraint for FixYConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "fix_y"
        }
        fn entity_ids(&self) -> &[EntityId] {
            std::slice::from_ref(&self.entity)
        }
        fn param_ids(&self) -> &[ParamId] {
            std::slice::from_ref(&self.y_param)
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            vec![store.get(self.y_param) - self.target_y]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![(0, self.y_param, 1.0)]
        }
    }

    #[test]
    fn test_no_constraints_full_displacement() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, eid);
        let py = store.alloc(0.0, eid);

        let mapping = store.build_solver_mapping();
        let constraints: Vec<&dyn Constraint> = vec![];

        let result = project_drag(
            &constraints,
            &store,
            &mapping,
            &[(px, 1.0), (py, 2.0)],
            1e-10,
        );

        assert!((result.projected_displacement[0] - 1.0).abs() < 1e-10);
        assert!((result.projected_displacement[1] - 2.0).abs() < 1e-10);
        assert!((result.preservation_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fully_constrained_zero_displacement() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(5.0, eid);

        let mapping = store.build_solver_mapping();

        // Constraint: px = 5.0 (fixes the single parameter)
        let c = FixYConstraint {
            id: ConstraintId::new(0, 0),
            entity: eid,
            y_param: px,
            target_y: 5.0,
        };
        let constraints: Vec<&dyn Constraint> = vec![&c];

        let result = project_drag(&constraints, &store, &mapping, &[(px, 3.0)], 1e-10);

        // The parameter is fully constrained, so displacement should be zero.
        assert!(result.projected_displacement[0].abs() < 1e-10);
        assert!(result.preservation_ratio < 0.01);
    }

    #[test]
    fn test_partially_constrained_projection() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(0.0, eid);
        let py = store.alloc(0.0, eid);

        let mapping = store.build_solver_mapping();

        // Constraint: py = 0 (fixes y, leaves x free)
        let c = FixYConstraint {
            id: ConstraintId::new(0, 0),
            entity: eid,
            y_param: py,
            target_y: 0.0,
        };
        let constraints: Vec<&dyn Constraint> = vec![&c];

        // Desired: move both x and y by 1.0
        let result = project_drag(
            &constraints,
            &store,
            &mapping,
            &[(px, 1.0), (py, 1.0)],
            1e-10,
        );

        // x should be preserved (free direction), y should be zeroed (constrained).
        let col_x = mapping.param_to_col[&px];
        let col_y = mapping.param_to_col[&py];
        assert!(
            (result.projected_displacement[col_x] - 1.0).abs() < 1e-10,
            "x displacement should be preserved"
        );
        assert!(
            result.projected_displacement[col_y].abs() < 1e-10,
            "y displacement should be zeroed"
        );
    }

    #[test]
    fn test_apply_drag() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(3.0, eid);
        let py = store.alloc(4.0, eid);

        let mapping = store.build_solver_mapping();
        let result = DragResult {
            projected_displacement: vec![1.0, -2.0],
            preservation_ratio: 1.0,
        };

        apply_drag(&mut store, &mapping, &result);

        assert!((store.get(px) - 4.0).abs() < 1e-10);
        assert!((store.get(py) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_displacement() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let _px = store.alloc(0.0, eid);

        let mapping = store.build_solver_mapping();
        let constraints: Vec<&dyn Constraint> = vec![];

        let result = project_drag(&constraints, &store, &mapping, &[], 1e-10);

        assert!(result
            .projected_displacement
            .iter()
            .all(|x| x.abs() < 1e-15));
        assert!((result.preservation_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_free_params() {
        let store = ParamStore::new();
        let mapping = store.build_solver_mapping();
        let constraints: Vec<&dyn Constraint> = vec![];

        let result = project_drag(&constraints, &store, &mapping, &[], 1e-10);

        assert!(result.projected_displacement.is_empty());
    }
}
