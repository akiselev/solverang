//! Jacobian rank analysis to detect redundant and conflicting constraints.
//!
//! This module examines the numerical rank of the Jacobian matrix to identify
//! constraints that are redundant (implied by others) or conflicting (inconsistent
//! with others). The analysis uses SVD (Singular Value Decomposition) via
//! [`nalgebra::DMatrix`] to determine the rank and the left null-space residual
//! projection.
//!
//! # Algorithm
//!
//! 1. Build a dense Jacobian matrix from sparse constraint triplets.
//! 2. Compute the thin SVD of the Jacobian.
//! 3. Determine the numerical rank from singular values above a tolerance.
//! 4. If rank < equation_count, identify which constraint blocks are dependent
//!    using an incremental rank test.
//! 5. Compute the null-space residual projection to distinguish redundant
//!    (consistent) from conflicting (inconsistent) constraints.

use nalgebra::DMatrix;

use crate::constraint::Constraint;
use crate::id::ConstraintId;
use crate::param::{ParamStore, SolverMapping};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of redundancy analysis.
#[derive(Clone, Debug)]
pub struct RedundancyAnalysis {
    /// Constraints that are redundant (implied by others).
    pub redundant: Vec<RedundantConstraint>,
    /// Groups of conflicting constraints.
    pub conflicts: Vec<ConflictGroup>,
    /// Numerical rank of the Jacobian.
    pub jacobian_rank: usize,
    /// Total number of equations (rows in the Jacobian).
    pub equation_count: usize,
    /// Number of free variables (columns in the Jacobian).
    pub variable_count: usize,
}

impl RedundancyAnalysis {
    /// True if no redundancies or conflicts were detected.
    pub fn is_clean(&self) -> bool {
        self.redundant.is_empty() && self.conflicts.is_empty()
    }

    /// Number of rank-deficient directions (`equation_count - jacobian_rank`).
    pub fn rank_deficiency(&self) -> usize {
        self.equation_count.saturating_sub(self.jacobian_rank)
    }
}

/// A constraint identified as redundant.
#[derive(Clone, Debug)]
pub struct RedundantConstraint {
    /// The constraint's generational ID.
    pub id: ConstraintId,
    /// The constraint's index in the system's constraint vector.
    pub index: usize,
}

/// A group of constraints that conflict with each other.
#[derive(Clone, Debug)]
pub struct ConflictGroup {
    /// IDs of the conflicting constraints.
    pub constraint_ids: Vec<ConstraintId>,
    /// Indices of the conflicting constraints in the system's constraint vector.
    pub constraint_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Analyze constraints for redundancy and conflicts using Jacobian rank analysis.
///
/// Uses SVD to compute the numerical rank of the Jacobian. If rank < equation_count,
/// some constraints are redundant or conflicting.
///
/// A redundant constraint has its Jacobian row in the span of other rows and
/// its residual is consistent (near zero when others are satisfied).
///
/// A conflicting constraint has its Jacobian row in the span of other rows but
/// its residual is inconsistent (non-zero even when others are satisfied).
///
/// # Arguments
///
/// * `constraints` - Pairs of `(system_index, constraint_ref)`.
/// * `store`       - Parameter store with current values.
/// * `mapping`     - Solver mapping from `ParamId` to column index.
/// * `tolerance`   - Threshold for near-zero singular values and residuals.
pub fn analyze_redundancy(
    constraints: &[(usize, &dyn Constraint)],
    store: &ParamStore,
    mapping: &SolverMapping,
    tolerance: f64,
) -> RedundancyAnalysis {
    let ncols = mapping.len();

    // Build row ranges: (start_row, end_row) for each constraint.
    let mut total_equations = 0usize;
    let mut row_ranges: Vec<(usize, usize)> = Vec::with_capacity(constraints.len());
    for (_, c) in constraints {
        let neq = c.equation_count();
        row_ranges.push((total_equations, total_equations + neq));
        total_equations += neq;
    }

    // Degenerate cases.
    if total_equations == 0 || ncols == 0 || constraints.is_empty() {
        return RedundancyAnalysis {
            redundant: Vec::new(),
            conflicts: Vec::new(),
            jacobian_rank: 0,
            equation_count: total_equations,
            variable_count: ncols,
        };
    }

    // --- 1. Build dense Jacobian ---
    let jac = build_dense_jacobian(
        constraints,
        &row_ranges,
        store,
        mapping,
        total_equations,
        ncols,
    );

    // --- 2-3. Compute rank via SVD ---
    let rank = compute_rank(&jac, tolerance);

    if rank >= total_equations {
        // Full row-rank: no redundancy.
        return RedundancyAnalysis {
            redundant: Vec::new(),
            conflicts: Vec::new(),
            jacobian_rank: rank,
            equation_count: total_equations,
            variable_count: ncols,
        };
    }

    // --- 4. Identify dependent constraint blocks ---
    let dependent_blocks = identify_dependent_blocks(&jac, &row_ranges, tolerance);

    // --- 5. Compute null-space residual projection ---
    // Collect all residuals in global row order.
    let all_residuals = collect_residuals(constraints, store, total_equations);

    // r_null = r - U_r * (U_r^T * r) where U_r is the first `rank` columns of U.
    let r_null = compute_null_residual(&jac, &all_residuals, rank, tolerance);

    // --- 6. Classify each dependent block ---
    let mut redundant = Vec::new();
    let mut conflicting_indices: Vec<usize> = Vec::new();

    for &ci in &dependent_blocks {
        let (row_start, row_end) = row_ranges[ci];

        // Check the null-space residual component for this block's rows.
        let max_null_component = (row_start..row_end)
            .map(|r| r_null[r].abs())
            .fold(0.0_f64, f64::max);

        if max_null_component < tolerance {
            // Consistent: this constraint is redundant.
            let (idx, c) = &constraints[ci];
            redundant.push(RedundantConstraint {
                id: c.id(),
                index: *idx,
            });
        } else {
            // Inconsistent: this constraint participates in a conflict.
            conflicting_indices.push(ci);
        }
    }

    // Build conflict groups from the conflicting constraint indices.
    let conflicts = build_conflict_groups(constraints, &conflicting_indices);

    RedundancyAnalysis {
        redundant,
        conflicts,
        jacobian_rank: rank,
        equation_count: total_equations,
        variable_count: ncols,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a dense Jacobian matrix from sparse constraint triplets.
fn build_dense_jacobian(
    constraints: &[(usize, &dyn Constraint)],
    row_ranges: &[(usize, usize)],
    store: &ParamStore,
    mapping: &SolverMapping,
    nrows: usize,
    ncols: usize,
) -> DMatrix<f64> {
    let mut jac = DMatrix::zeros(nrows, ncols);

    for (ci, (_, constraint)) in constraints.iter().enumerate() {
        let (row_start, _) = row_ranges[ci];
        let triplets = constraint.jacobian(store);
        for (local_row, param_id, value) in triplets {
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
///
/// A singular value is considered non-zero if it exceeds `tolerance`.
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

/// Collect all constraint residuals into a single vector in global row order.
fn collect_residuals(
    constraints: &[(usize, &dyn Constraint)],
    store: &ParamStore,
    total_equations: usize,
) -> Vec<f64> {
    let mut residuals = Vec::with_capacity(total_equations);
    for (_, c) in constraints {
        residuals.extend(c.residuals(store));
    }
    residuals
}

/// Identify dependent constraint blocks via incremental rank testing.
///
/// Processes blocks one at a time, checking whether adding a block increases
/// the rank of the accumulated sub-matrix. Returns the slice indices of
/// blocks that did NOT increase the rank.
fn identify_dependent_blocks(
    jac: &DMatrix<f64>,
    row_ranges: &[(usize, usize)],
    tolerance: f64,
) -> Vec<usize> {
    let ncols = jac.ncols();
    let mut dependent = Vec::new();
    let mut accumulated_rows: Vec<usize> = Vec::new();
    let mut current_rank = 0usize;

    for (ci, &(row_start, row_end)) in row_ranges.iter().enumerate() {
        let block_size = row_end - row_start;
        if block_size == 0 {
            continue;
        }

        // Build accumulated matrix including this block.
        let candidate_rows: Vec<usize> = accumulated_rows
            .iter()
            .copied()
            .chain(row_start..row_end)
            .collect();

        let mut test_matrix = DMatrix::zeros(candidate_rows.len(), ncols);
        for (i, &r) in candidate_rows.iter().enumerate() {
            for c in 0..ncols {
                test_matrix[(i, c)] = jac[(r, c)];
            }
        }

        let new_rank = compute_rank(&test_matrix, tolerance);
        if new_rank > current_rank {
            current_rank = new_rank;
            accumulated_rows.extend(row_start..row_end);
        } else {
            dependent.push(ci);
        }
    }

    dependent
}

/// Compute the null-space component of the residual vector.
///
/// Uses the thin SVD to project away the column-space component:
///
/// `r_null = r - U_r * (U_r^T * r)`
///
/// where `U_r` is the first `rank` columns of `U`.
fn compute_null_residual(
    jac: &DMatrix<f64>,
    residuals: &[f64],
    rank: usize,
    tolerance: f64,
) -> Vec<f64> {
    let m = jac.nrows();
    let mut r_null = residuals.to_vec();

    if rank == 0 || m == 0 {
        return r_null;
    }

    let svd = jac.clone().svd(true, false);
    let u = match svd.u.as_ref() {
        Some(u) => u,
        None => return r_null,
    };

    let usable_cols = rank.min(u.ncols());
    for col_idx in 0..usable_cols {
        if svd.singular_values[col_idx] <= tolerance {
            continue;
        }
        let u_col = u.column(col_idx);
        let coeff: f64 = (0..m).map(|row| u_col[row] * residuals[row]).sum();
        for row in 0..m {
            r_null[row] -= coeff * u_col[row];
        }
    }

    r_null
}

/// Group conflicting constraint indices into [`ConflictGroup`]s by shared
/// parameters. Two conflicting constraints are placed in the same group if
/// they share at least one parameter.
fn build_conflict_groups(
    constraints: &[(usize, &dyn Constraint)],
    conflicting: &[usize],
) -> Vec<ConflictGroup> {
    if conflicting.is_empty() {
        return Vec::new();
    }

    // Union-find over conflicting indices.
    let n = conflicting.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    // Merge indices that share at least one parameter.
    for i in 0..n {
        let params_i = constraints[conflicting[i]].1.param_ids();
        for j in (i + 1)..n {
            let params_j = constraints[conflicting[j]].1.param_ids();
            if params_i.iter().any(|p| params_j.contains(p)) {
                union(&mut parent, i, j);
            }
        }
    }

    // Collect groups.
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(conflicting[i]);
    }

    groups
        .into_values()
        .map(|members| {
            let mut ids = Vec::with_capacity(members.len());
            let mut indices = Vec::with_capacity(members.len());
            for &ci in &members {
                let (idx, c) = &constraints[ci];
                ids.push(c.id());
                indices.push(*idx);
            }
            ConflictGroup {
                constraint_ids: ids,
                constraint_indices: indices,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    // -- Stub constraint for testing -----------------------------------------

    struct TestConstraint {
        id: ConstraintId,
        entities: Vec<EntityId>,
        params: Vec<ParamId>,
        neq: usize,
        residual_fn: Box<dyn Fn(&ParamStore) -> Vec<f64> + Send + Sync>,
        jacobian_fn: Box<dyn Fn(&ParamStore) -> Vec<(usize, ParamId, f64)> + Send + Sync>,
    }

    impl Constraint for TestConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "test"
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

    // -- Helper --------------------------------------------------------------

    fn setup_2param_store() -> (ParamStore, EntityId, ParamId, ParamId) {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let p0 = store.alloc(3.0, eid);
        let p1 = store.alloc(4.0, eid);
        (store, eid, p0, p1)
    }

    // -- Tests ---------------------------------------------------------------

    #[test]
    fn test_no_redundancy() {
        let (store, eid, p0, p1) = setup_2param_store();
        let mapping = store.build_solver_mapping();

        let c0 = TestConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![p0],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(p0) - 3.0]),
            jacobian_fn: Box::new(move |_| vec![(0, p0, 1.0)]),
        };
        let c1 = TestConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![eid],
            params: vec![p1],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(p1) - 4.0]),
            jacobian_fn: Box::new(move |_| vec![(0, p1, 1.0)]),
        };

        let pairs: Vec<(usize, &dyn Constraint)> = vec![(0, &c0), (1, &c1)];
        let result = analyze_redundancy(&pairs, &store, &mapping, 1e-10);

        assert!(result.is_clean());
        assert_eq!(result.jacobian_rank, 2);
        assert_eq!(result.equation_count, 2);
        assert_eq!(result.variable_count, 2);
        assert_eq!(result.rank_deficiency(), 0);
    }

    #[test]
    fn test_redundant_constraint() {
        // c0: x = 3, c1: y = 4, c2: x = 3 (duplicate of c0).
        let (store, eid, p0, p1) = setup_2param_store();
        let mapping = store.build_solver_mapping();

        let c0 = TestConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![p0],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(p0) - 3.0]),
            jacobian_fn: Box::new(move |_| vec![(0, p0, 1.0)]),
        };
        let c1 = TestConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![eid],
            params: vec![p1],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(p1) - 4.0]),
            jacobian_fn: Box::new(move |_| vec![(0, p1, 1.0)]),
        };
        let c2 = TestConstraint {
            id: ConstraintId::new(2, 0),
            entities: vec![eid],
            params: vec![p0],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(p0) - 3.0]),
            jacobian_fn: Box::new(move |_| vec![(0, p0, 1.0)]),
        };

        let pairs: Vec<(usize, &dyn Constraint)> = vec![(0, &c0), (1, &c1), (2, &c2)];
        let result = analyze_redundancy(&pairs, &store, &mapping, 1e-10);

        assert_eq!(result.jacobian_rank, 2);
        assert_eq!(result.equation_count, 3);
        assert_eq!(result.rank_deficiency(), 1);
        assert!(!result.redundant.is_empty(), "should detect redundancy");
        assert!(result.conflicts.is_empty(), "no conflicts expected");
    }

    #[test]
    fn test_conflicting_constraints() {
        // c0: x = 3, c1: x = 5 (same Jacobian row, inconsistent residuals).
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let p0 = store.alloc(3.0, eid);
        let mapping = store.build_solver_mapping();

        let c0 = TestConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![p0],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(p0) - 3.0]),
            jacobian_fn: Box::new(move |_| vec![(0, p0, 1.0)]),
        };
        let c1 = TestConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![eid],
            params: vec![p0],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(p0) - 5.0]),
            jacobian_fn: Box::new(move |_| vec![(0, p0, 1.0)]),
        };

        let pairs: Vec<(usize, &dyn Constraint)> = vec![(0, &c0), (1, &c1)];
        let result = analyze_redundancy(&pairs, &store, &mapping, 1e-10);

        assert_eq!(result.jacobian_rank, 1);
        assert_eq!(result.equation_count, 2);
        assert!(!result.conflicts.is_empty(), "should detect conflict");
    }

    #[test]
    fn test_empty_constraints() {
        let store = ParamStore::new();
        let mapping = store.build_solver_mapping();
        let pairs: Vec<(usize, &dyn Constraint)> = vec![];
        let result = analyze_redundancy(&pairs, &store, &mapping, 1e-10);

        assert!(result.is_clean());
        assert_eq!(result.jacobian_rank, 0);
        assert_eq!(result.equation_count, 0);
        assert_eq!(result.variable_count, 0);
    }

    #[test]
    fn test_compute_rank_identity() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(compute_rank(&m, 1e-10), 2);
    }

    #[test]
    fn test_compute_rank_singular() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert_eq!(compute_rank(&m, 1e-10), 1);
    }

    #[test]
    fn test_compute_rank_zero() {
        let m = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(compute_rank(&m, 1e-10), 0);
    }

    #[test]
    fn test_rank_deficiency_helper() {
        let r = RedundancyAnalysis {
            redundant: vec![],
            conflicts: vec![],
            jacobian_rank: 3,
            equation_count: 5,
            variable_count: 4,
        };
        assert_eq!(r.rank_deficiency(), 2);
    }

    #[test]
    fn test_multi_equation_constraint_redundancy() {
        // c0: 2-equation constraint on (p0, p1): x-1=0, y-2=0
        // c1: 1-equation duplicate on p0: x-1=0
        let (store, eid, p0, p1) = setup_2param_store();
        // Set param values to satisfy c0 exactly.
        let mut store = store;
        store.set(p0, 1.0);
        store.set(p1, 2.0);
        let mapping = store.build_solver_mapping();

        let c0 = TestConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![p0, p1],
            neq: 2,
            residual_fn: Box::new(move |s| vec![s.get(p0) - 1.0, s.get(p1) - 2.0]),
            jacobian_fn: Box::new(move |_| vec![(0, p0, 1.0), (1, p1, 1.0)]),
        };
        let c1 = TestConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![eid],
            params: vec![p0],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(p0) - 1.0]),
            jacobian_fn: Box::new(move |_| vec![(0, p0, 1.0)]),
        };

        let pairs: Vec<(usize, &dyn Constraint)> = vec![(0, &c0), (1, &c1)];
        let result = analyze_redundancy(&pairs, &store, &mapping, 1e-10);

        assert_eq!(result.jacobian_rank, 2);
        assert_eq!(result.equation_count, 3);
        assert!(!result.redundant.is_empty());
        assert!(result.conflicts.is_empty());
    }
}
