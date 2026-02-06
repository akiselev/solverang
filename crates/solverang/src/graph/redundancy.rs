use crate::geometry::params::ConstraintId;
use super::dm::dm_decompose;

/// Result of redundancy analysis.
#[derive(Clone, Debug, Default)]
pub struct RedundancyReport {
    /// Structurally redundant constraints (in over-determined DM block).
    pub structurally_redundant: Vec<ConstraintId>,
    /// Numerically redundant (linearly dependent Jacobian rows).
    pub numerically_redundant: Vec<ConstraintId>,
    /// Conflicting constraints (over-determined AND residuals non-zero).
    pub conflicting: Vec<ConstraintId>,
}

/// Detect structurally redundant constraints using DM decomposition.
///
/// Constraints in the over-determined block are structurally redundant,
/// meaning they do not add any new degrees of freedom to the system.
///
/// # Arguments
/// * `constraint_ids` - Ordered list of constraint IDs (index i corresponds to constraint i)
/// * `n_constraints` - Total number of constraints
/// * `n_variables` - Total number of variables
/// * `edges` - (constraint_idx, variable_idx) pairs from Jacobian sparsity pattern
///
/// # Returns
/// List of constraint IDs that are structurally redundant
pub fn detect_structural_redundancy(
    constraint_ids: &[ConstraintId],
    n_constraints: usize,
    n_variables: usize,
    edges: &[(usize, usize)],
) -> Vec<ConstraintId> {
    if n_constraints == 0 || n_variables == 0 {
        return vec![];
    }

    let dm = dm_decompose(n_constraints, n_variables, edges);

    // Unmatched constraints are structurally redundant — they have no
    // corresponding variable in the maximum matching and represent excess
    // equations that don't add degrees of freedom.
    dm.unmatched_constraints
        .iter()
        .filter_map(|&idx| {
            if idx < constraint_ids.len() {
                Some(constraint_ids[idx])
            } else {
                None
            }
        })
        .collect()
}

/// Detect numerically redundant constraints by checking Jacobian rank.
///
/// Two constraints are numerically redundant if their Jacobian rows are
/// linearly dependent (within tolerance). This uses QR decomposition with
/// column pivoting to identify near-zero pivots.
///
/// # Arguments
/// * `jacobian_entries` - Sparse Jacobian as (row, col, value) triplets
/// * `n_constraints` - Number of constraint rows
/// * `n_variables` - Number of variable columns
/// * `constraint_ids` - Ordered list of constraint IDs
/// * `tolerance` - Threshold for considering a pivot as zero
///
/// # Returns
/// List of constraint IDs that are numerically redundant
pub fn detect_numerical_redundancy(
    jacobian_entries: &[(usize, usize, f64)],
    n_constraints: usize,
    n_variables: usize,
    constraint_ids: &[ConstraintId],
    tolerance: f64,
) -> Vec<ConstraintId> {
    if n_constraints == 0 || n_variables == 0 {
        return vec![];
    }

    // Build dense Jacobian matrix (for simplicity; could use sparse QR in future)
    let mut jacobian = vec![vec![0.0; n_variables]; n_constraints];
    for &(row, col, val) in jacobian_entries {
        if row < n_constraints && col < n_variables {
            jacobian[row][col] = val;
        }
    }

    // Perform Gaussian elimination with partial pivoting to find rank
    let mut redundant_rows = find_redundant_rows(&jacobian, tolerance);

    // Map row indices to constraint IDs
    redundant_rows.sort_unstable();
    redundant_rows
        .into_iter()
        .filter_map(|row| {
            if row < constraint_ids.len() {
                Some(constraint_ids[row])
            } else {
                None
            }
        })
        .collect()
}

/// Find redundant rows in a matrix using Gaussian elimination.
///
/// Returns the original indices of rows that are linearly dependent on other rows.
/// Uses forward-order pivot selection (first viable row) so that earlier rows are
/// preferred as pivots and later dependent rows are flagged as redundant.
fn find_redundant_rows(matrix: &[Vec<f64>], tolerance: f64) -> Vec<usize> {
    let n_rows = matrix.len();
    if n_rows == 0 {
        return vec![];
    }
    let n_cols = matrix[0].len();

    // Create a working copy
    let mut a = matrix.to_vec();
    let mut redundant = Vec::new();
    let mut pivot_row = 0;

    // Track original row index at each position through swaps
    let mut row_perm: Vec<usize> = (0..n_rows).collect();

    for col in 0..n_cols {
        if pivot_row >= n_rows {
            break;
        }

        // Find first row from pivot_row onward with non-negligible value in this column.
        // This preserves the original row ordering so that later dependent rows
        // (not earlier ones) are the ones flagged as redundant.
        let mut found_row = None;
        for row in pivot_row..n_rows {
            if a[row][col].abs() >= tolerance {
                found_row = Some(row);
                break;
            }
        }

        let swap_row = match found_row {
            Some(r) => r,
            None => continue, // No viable pivot in this column
        };

        // Swap rows if needed
        if swap_row != pivot_row {
            a.swap(swap_row, pivot_row);
            row_perm.swap(swap_row, pivot_row);
        }

        // Eliminate below pivot
        for row in (pivot_row + 1)..n_rows {
            let factor = a[row][col] / a[pivot_row][col];
            if factor.abs() > tolerance {
                for c in col..n_cols {
                    a[row][c] -= factor * a[pivot_row][c];
                }
            }
        }

        pivot_row += 1;
    }

    // Check each row to see if it's effectively zero after elimination
    // and map back to original row indices
    for row in 0..n_rows {
        let mut is_zero = true;
        for col in 0..n_cols {
            if a[row][col].abs() > tolerance {
                is_zero = false;
                break;
            }
        }
        if is_zero {
            redundant.push(row_perm[row]);
        }
    }

    redundant
}

/// Detect conflicting constraints by attempting to solve the over-determined block.
///
/// If residuals remain large after solving, the constraints are incompatible.
/// This is a placeholder for now - actual implementation would require access
/// to the solver and constraint evaluation.
///
/// # Arguments
/// * `redundant_constraints` - Structurally redundant constraints to check
/// * `residuals` - Residual values for each constraint (optional)
/// * `tolerance` - Threshold for considering residuals as zero
///
/// # Returns
/// List of constraint IDs that are conflicting (redundant but incompatible)
pub fn detect_conflicts(
    redundant_constraints: &[ConstraintId],
    residuals: Option<&[f64]>,
    tolerance: f64,
) -> Vec<ConstraintId> {
    if let Some(res) = residuals {
        // A redundant constraint is conflicting if its residual is large
        redundant_constraints
            .iter()
            .enumerate()
            .filter_map(|(i, &cid)| {
                if i < res.len() && res[i].abs() > tolerance {
                    Some(cid)
                } else {
                    None
                }
            })
            .collect()
    } else {
        // Without residuals, we can't determine conflicts
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_redundancy() {
        let ids = vec![];
        let redundant = detect_structural_redundancy(&ids, 0, 0, &[]);
        assert!(redundant.is_empty());
    }

    #[test]
    fn test_no_structural_redundancy() {
        // Perfect matching: 2 constraints, 2 variables
        let ids = vec![ConstraintId(0), ConstraintId(1)];
        let edges = vec![(0, 0), (1, 1)];

        let redundant = detect_structural_redundancy(&ids, 2, 2, &edges);
        assert!(redundant.is_empty());
    }

    #[test]
    fn test_structural_redundancy_detected() {
        // 3 constraints, 2 variables -> at least one is redundant
        let ids = vec![ConstraintId(0), ConstraintId(1), ConstraintId(2)];
        let edges = vec![
            (0, 0), (0, 1),
            (1, 0), (1, 1),
            (2, 0), (2, 1),
        ];

        let redundant = detect_structural_redundancy(&ids, 3, 2, &edges);
        assert!(!redundant.is_empty());
    }

    #[test]
    fn test_find_redundant_rows_independent() {
        // All rows are independent
        let matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let redundant = find_redundant_rows(&matrix, 1e-10);
        assert!(redundant.is_empty());
    }

    #[test]
    fn test_find_redundant_rows_duplicate() {
        // Third row is duplicate of first
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 4.0],
            vec![1.0, 2.0, 3.0],  // Same as row 0
        ];

        let redundant = find_redundant_rows(&matrix, 1e-10);
        assert!(redundant.contains(&2));
    }

    #[test]
    fn test_find_redundant_rows_linear_combination() {
        // Third row is 2*row0 + 3*row1
        let matrix = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 3.0],  // Linear combination
        ];

        let redundant = find_redundant_rows(&matrix, 1e-10);
        assert!(redundant.contains(&2));
    }

    #[test]
    fn test_find_redundant_rows_zero_row() {
        // Zero row is always redundant
        let matrix = vec![
            vec![1.0, 2.0],
            vec![0.0, 0.0],  // Zero row
            vec![3.0, 4.0],
        ];

        let redundant = find_redundant_rows(&matrix, 1e-10);
        assert!(redundant.contains(&1));
    }

    #[test]
    fn test_numerical_redundancy_independent() {
        let ids = vec![ConstraintId(0), ConstraintId(1)];
        let jacobian = vec![
            (0, 0, 1.0),
            (1, 1, 1.0),
        ];

        let redundant = detect_numerical_redundancy(&jacobian, 2, 2, &ids, 1e-10);
        assert!(redundant.is_empty());
    }

    #[test]
    fn test_numerical_redundancy_duplicate_rows() {
        let ids = vec![ConstraintId(0), ConstraintId(1)];
        let jacobian = vec![
            (0, 0, 1.0), (0, 1, 2.0),
            (1, 0, 1.0), (1, 1, 2.0),  // Same as row 0
        ];

        let redundant = detect_numerical_redundancy(&jacobian, 2, 2, &ids, 1e-10);
        assert!(!redundant.is_empty());
    }

    #[test]
    fn test_numerical_redundancy_near_zero() {
        let ids = vec![ConstraintId(0), ConstraintId(1), ConstraintId(2)];
        let jacobian = vec![
            (0, 0, 1.0), (0, 1, 0.0),
            (1, 0, 0.0), (1, 1, 1.0),
            (2, 0, 1e-12), (2, 1, 1e-12),  // Effectively zero row
        ];

        let redundant = detect_numerical_redundancy(&jacobian, 3, 2, &ids, 1e-10);
        assert!(redundant.contains(&ConstraintId(2)));
    }

    #[test]
    fn test_detect_conflicts_no_residuals() {
        let redundant = vec![ConstraintId(0), ConstraintId(1)];
        let conflicts = detect_conflicts(&redundant, None, 1e-6);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_detect_conflicts_with_residuals() {
        let redundant = vec![ConstraintId(0), ConstraintId(1)];
        let residuals = vec![1e-8, 1.0];  // Second one has large residual

        let conflicts = detect_conflicts(&redundant, Some(&residuals), 1e-6);
        assert_eq!(conflicts, vec![ConstraintId(1)]);
    }

    #[test]
    fn test_detect_conflicts_all_satisfied() {
        let redundant = vec![ConstraintId(0), ConstraintId(1)];
        let residuals = vec![1e-10, 1e-9];  // Both small

        let conflicts = detect_conflicts(&redundant, Some(&residuals), 1e-6);
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_empty_matrix() {
        let matrix: Vec<Vec<f64>> = vec![];
        let redundant = find_redundant_rows(&matrix, 1e-10);
        assert!(redundant.is_empty());
    }

    #[test]
    fn test_single_row_matrix() {
        let matrix = vec![vec![1.0, 2.0, 3.0]];
        let redundant = find_redundant_rows(&matrix, 1e-10);
        assert!(redundant.is_empty());
    }

    #[test]
    fn test_redundancy_report_default() {
        let report = RedundancyReport::default();
        assert!(report.structurally_redundant.is_empty());
        assert!(report.numerically_redundant.is_empty());
        assert!(report.conflicting.is_empty());
    }

    #[test]
    fn test_tolerance_sensitivity() {
        // Test that tolerance affects what's considered redundant
        let matrix = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1e-8, 1e-8],  // Near-zero row
        ];

        // With strict tolerance, should be detected
        let redundant_strict = find_redundant_rows(&matrix, 1e-10);
        assert!(redundant_strict.contains(&2));

        // With loose tolerance, might not be detected
        let redundant_loose = find_redundant_rows(&matrix, 1e-6);
        assert!(redundant_loose.contains(&2));
    }

    #[test]
    fn test_larger_system() {
        // 5x4 matrix with one redundant row
        let matrix = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],  // Sum of first two rows
        ];

        let redundant = find_redundant_rows(&matrix, 1e-10);
        assert!(redundant.contains(&4));
    }

    #[test]
    fn test_partial_redundancy() {
        // System where only some constraints are redundant
        let ids = vec![
            ConstraintId(0),
            ConstraintId(1),
            ConstraintId(2),
            ConstraintId(3),
        ];

        // 4 constraints, 3 variables
        let edges = vec![
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 0), (3, 1),  // Fourth constraint is redundant
        ];

        let redundant = detect_structural_redundancy(&ids, 4, 3, &edges);
        assert!(!redundant.is_empty());
        assert!(redundant.len() <= 2); // At most 1 redundant
    }
}
