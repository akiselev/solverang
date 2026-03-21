//! Sparse Jacobian representation in COO (Coordinate) format.
//!
//! This module provides both COO format for assembly and CSR format
//! for efficient matrix operations and linear system solving.

use super::pattern::SparsityPattern;

/// Sparse Jacobian matrix in COO (Coordinate) format.
///
/// This format stores only non-zero entries as (row, col, value) triplets.
/// It is efficient for assembly and conversion to other sparse formats.
#[derive(Clone, Debug, Default)]
pub struct SparseJacobian {
    /// Row indices of non-zero entries.
    pub rows: Vec<usize>,
    /// Column indices of non-zero entries.
    pub cols: Vec<usize>,
    /// Values of non-zero entries.
    pub values: Vec<f64>,
    /// Number of rows (m = number of equations).
    pub nrows: usize,
    /// Number of columns (n = number of variables).
    pub ncols: usize,
}

impl SparseJacobian {
    /// Create a new empty sparse Jacobian with the given dimensions.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
            values: Vec::new(),
            nrows,
            ncols,
        }
    }

    /// Create a sparse Jacobian with pre-allocated capacity.
    pub fn with_capacity(nrows: usize, ncols: usize, capacity: usize) -> Self {
        Self {
            rows: Vec::with_capacity(capacity),
            cols: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            nrows,
            ncols,
        }
    }

    /// Create from triplet vectors.
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: Vec<(usize, usize, f64)>) -> Self {
        let len = triplets.len();
        let mut rows = Vec::with_capacity(len);
        let mut cols = Vec::with_capacity(len);
        let mut values = Vec::with_capacity(len);

        for (row, col, val) in triplets {
            rows.push(row);
            cols.push(col);
            values.push(val);
        }

        Self {
            rows,
            cols,
            values,
            nrows,
            ncols,
        }
    }

    /// Add a non-zero entry to the Jacobian.
    ///
    /// Panics if row >= nrows or col >= ncols.
    pub fn add_entry(&mut self, row: usize, col: usize, value: f64) {
        assert!(
            row < self.nrows,
            "row index {} out of bounds (nrows = {})",
            row,
            self.nrows
        );
        assert!(
            col < self.ncols,
            "col index {} out of bounds (ncols = {})",
            col,
            self.ncols
        );

        self.rows.push(row);
        self.cols.push(col);
        self.values.push(value);
    }

    /// Try to add an entry, returning false if out of bounds.
    pub fn try_add_entry(&mut self, row: usize, col: usize, value: f64) -> bool {
        if row >= self.nrows || col >= self.ncols {
            return false;
        }

        self.rows.push(row);
        self.cols.push(col);
        self.values.push(value);
        true
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Check if the Jacobian is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get entry at (row, col), or 0.0 if not present.
    ///
    /// Note: This is O(nnz) - for frequent lookups, convert to a dense format.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        for (i, (&r, &c)) in self.rows.iter().zip(self.cols.iter()).enumerate() {
            if r == row && c == col {
                return self.values[i];
            }
        }
        0.0
    }

    /// Convert to dense matrix (row-major).
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];

        for (i, (&row, &col)) in self.rows.iter().zip(self.cols.iter()).enumerate() {
            if row < self.nrows && col < self.ncols {
                dense[row][col] = self.values[i];
            }
        }

        dense
    }

    /// Convert to triplet vector.
    pub fn to_triplets(&self) -> Vec<(usize, usize, f64)> {
        self.rows
            .iter()
            .zip(self.cols.iter())
            .zip(self.values.iter())
            .map(|((&r, &c), &v)| (r, c, v))
            .collect()
    }

    /// Iterate over entries as (row, col, value) tuples.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.rows
            .iter()
            .zip(self.cols.iter())
            .zip(self.values.iter())
            .map(|((&r, &c), &v)| (r, c, v))
    }

    /// Clear all entries while preserving dimensions and capacity.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.cols.clear();
        self.values.clear();
    }

    /// Remove entries where the absolute value is below the threshold.
    pub fn drop_below(&mut self, threshold: f64) {
        let mut i = 0;
        while i < self.values.len() {
            if self.values[i].abs() < threshold {
                self.rows.swap_remove(i);
                self.cols.swap_remove(i);
                self.values.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Compute the Frobenius norm: sqrt(sum(values^2)).
    pub fn frobenius_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Check if all values are finite (not NaN or infinity).
    pub fn is_finite(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }

    /// Extract the sparsity pattern from this Jacobian.
    pub fn pattern(&self) -> SparsityPattern {
        let coords: Vec<(usize, usize)> = self
            .rows
            .iter()
            .zip(self.cols.iter())
            .map(|(&r, &c)| (r, c))
            .collect();
        SparsityPattern::from_coordinates(self.nrows, self.ncols, &coords)
    }

    /// Convert to CSR format for efficient operations.
    pub fn to_csr(&self) -> CsrMatrix {
        CsrMatrix::from_coo(self)
    }

    /// Create from triplets with a pre-existing pattern.
    ///
    /// If the pattern matches, this enables fast value-only updates.
    /// Returns None if the triplets don't match the pattern structure.
    pub fn from_triplets_with_pattern(
        triplets: &[(usize, usize, f64)],
        pattern: &SparsityPattern,
    ) -> Option<Self> {
        // Verify all triplet positions exist in the pattern
        for &(row, col, _) in triplets {
            if !pattern.has_entry(row, col) {
                return None;
            }
        }

        let nrows = pattern.nrows;
        let ncols = pattern.ncols;
        let len = triplets.len();
        let mut rows = Vec::with_capacity(len);
        let mut cols = Vec::with_capacity(len);
        let mut values = Vec::with_capacity(len);

        for &(row, col, val) in triplets {
            rows.push(row);
            cols.push(col);
            values.push(val);
        }

        Some(Self {
            rows,
            cols,
            values,
            nrows,
            ncols,
        })
    }

    /// Sparsity ratio: nnz / (nrows * ncols).
    pub fn sparsity_ratio(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 0.0;
        }
        self.nnz() as f64 / total as f64
    }

    /// Check if the matrix should be treated as sparse (below density threshold).
    pub fn is_sparse(&self, density_threshold: f64) -> bool {
        self.sparsity_ratio() < density_threshold
    }
}

/// Sparse matrix in CSR (Compressed Sparse Row) format.
///
/// CSR format is efficient for:
/// - Matrix-vector multiplication
/// - Row slicing
/// - Sparse linear system solving
#[derive(Clone, Debug, Default)]
pub struct CsrMatrix {
    /// Values of non-zero entries in row-major order.
    pub values: Vec<f64>,
    /// Column indices for each value.
    pub col_indices: Vec<usize>,
    /// Row pointers: row_ptr\[i\]..row_ptr\[i+1\] gives the range of entries in row i.
    pub row_ptr: Vec<usize>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

impl CsrMatrix {
    /// Create a new empty CSR matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptr: vec![0; nrows + 1],
            nrows,
            ncols,
        }
    }

    /// Create from COO format (SparseJacobian).
    ///
    /// Duplicate entries are summed.
    pub fn from_coo(coo: &SparseJacobian) -> Self {
        if coo.is_empty() {
            return Self::new(coo.nrows, coo.ncols);
        }

        // Group entries by row and sort by column within each row
        let mut row_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); coo.nrows];

        for (i, (&row, &col)) in coo.rows.iter().zip(coo.cols.iter()).enumerate() {
            if row < coo.nrows && col < coo.ncols {
                row_entries[row].push((col, coo.values[i]));
            }
        }

        // Sort each row by column and merge duplicates
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = Vec::with_capacity(coo.nrows + 1);

        row_ptr.push(0);

        for row_cols in &mut row_entries {
            row_cols.sort_by_key(|&(col, _)| col);

            // Merge duplicates by summing values
            let mut merged: Vec<(usize, f64)> = Vec::new();
            for &(col, val) in row_cols.iter() {
                if let Some(last) = merged.last_mut() {
                    if last.0 == col {
                        last.1 += val;
                        continue;
                    }
                }
                merged.push((col, val));
            }

            for (col, val) in merged {
                col_indices.push(col);
                values.push(val);
            }
            row_ptr.push(values.len());
        }

        Self {
            values,
            col_indices,
            row_ptr,
            nrows: coo.nrows,
            ncols: coo.ncols,
        }
    }

    /// Create from triplets directly.
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let coo = SparseJacobian::from_triplets(nrows, ncols, triplets.to_vec());
        Self::from_coo(&coo)
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Check if the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get entries for a specific row.
    pub fn row(&self, row: usize) -> Option<(&[usize], &[f64])> {
        if row >= self.nrows {
            return None;
        }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        Some((&self.col_indices[start..end], &self.values[start..end]))
    }

    /// Get entry at (row, col), or 0.0 if not present.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.nrows || col >= self.ncols {
            return 0.0;
        }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        // Binary search for column in this row
        match self.col_indices[start..end].binary_search(&col) {
            Ok(idx) => self.values[start + idx],
            Err(_) => 0.0,
        }
    }

    /// Matrix-vector multiplication: y = A * x.
    ///
    /// Returns None if x has wrong length.
    pub fn mul_vec(&self, x: &[f64]) -> Option<Vec<f64>> {
        if x.len() != self.ncols {
            return None;
        }

        let mut result = vec![0.0; self.nrows];
        for (row, result_elem) in result.iter_mut().enumerate() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            let mut sum = 0.0;
            for (idx, &col) in self.col_indices[start..end].iter().enumerate() {
                sum += self.values[start + idx] * x[col];
            }
            *result_elem = sum;
        }
        Some(result)
    }

    /// Transposed matrix-vector multiplication: y = A^T * x.
    ///
    /// Returns None if x has wrong length.
    pub fn mul_vec_transpose(&self, x: &[f64]) -> Option<Vec<f64>> {
        if x.len() != self.nrows {
            return None;
        }

        let mut result = vec![0.0; self.ncols];
        for (row, &x_row) in x.iter().enumerate().take(self.nrows) {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for (idx, &col) in self.col_indices[start..end].iter().enumerate() {
                result[col] += self.values[start + idx] * x_row;
            }
        }
        Some(result)
    }

    /// Convert to dense matrix.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        for (row, dense_row) in dense.iter_mut().enumerate() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for (idx, &col) in self.col_indices[start..end].iter().enumerate() {
                dense_row[col] = self.values[start + idx];
            }
        }
        dense
    }

    /// Extract the sparsity pattern.
    pub fn pattern(&self) -> SparsityPattern {
        SparsityPattern {
            row_ptr: self.row_ptr.clone(),
            col_idx: self.col_indices.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// Update values from triplets while keeping the same pattern.
    ///
    /// This is an efficient operation when the sparsity pattern is unchanged
    /// (only values differ). Returns false if the pattern doesn't match.
    pub fn update_values(&mut self, triplets: &[(usize, usize, f64)]) -> bool {
        // First, zero out all values
        for val in &mut self.values {
            *val = 0.0;
        }

        // Then accumulate new values
        for &(row, col, val) in triplets {
            if row >= self.nrows || col >= self.ncols {
                return false;
            }

            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            match self.col_indices[start..end].binary_search(&col) {
                Ok(idx) => {
                    self.values[start + idx] += val;
                }
                Err(_) => {
                    // Entry not in pattern
                    return false;
                }
            }
        }
        true
    }

    /// Check if all values are finite.
    pub fn is_finite(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }

    /// Compute the squared Frobenius norm: sum(values^2).
    pub fn frobenius_norm_squared(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum()
    }

    /// Compute the Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.frobenius_norm_squared().sqrt()
    }

    /// Get the number of entries in each row (useful for detecting empty rows).
    pub fn row_nnz(&self) -> Vec<usize> {
        (0..self.nrows)
            .map(|row| self.row_ptr[row + 1] - self.row_ptr[row])
            .collect()
    }

    /// Check if any row is completely empty.
    pub fn has_empty_row(&self) -> bool {
        self.row_nnz().contains(&0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let jac = SparseJacobian::new(3, 4);
        assert_eq!(jac.nrows, 3);
        assert_eq!(jac.ncols, 4);
        assert_eq!(jac.nnz(), 0);
        assert!(jac.is_empty());
    }

    #[test]
    fn test_add_entry() {
        let mut jac = SparseJacobian::new(3, 3);
        jac.add_entry(0, 0, 1.0);
        jac.add_entry(1, 1, 2.0);
        jac.add_entry(2, 2, 3.0);

        assert_eq!(jac.nnz(), 3);
        assert!((jac.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((jac.get(1, 1) - 2.0).abs() < 1e-10);
        assert!((jac.get(2, 2) - 3.0).abs() < 1e-10);
        assert!((jac.get(0, 1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_triplets() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (0, 1, 3.0)];
        let jac = SparseJacobian::from_triplets(2, 2, triplets);

        assert_eq!(jac.nrows, 2);
        assert_eq!(jac.ncols, 2);
        assert_eq!(jac.nnz(), 3);
    }

    #[test]
    fn test_to_dense() {
        let mut jac = SparseJacobian::new(2, 2);
        jac.add_entry(0, 0, 1.0);
        jac.add_entry(0, 1, 2.0);
        jac.add_entry(1, 0, 3.0);
        jac.add_entry(1, 1, 4.0);

        let dense = jac.to_dense();
        assert_eq!(dense.len(), 2);
        assert_eq!(dense[0], vec![1.0, 2.0]);
        assert_eq!(dense[1], vec![3.0, 4.0]);
    }

    #[test]
    fn test_frobenius_norm() {
        let mut jac = SparseJacobian::new(2, 2);
        jac.add_entry(0, 0, 3.0);
        jac.add_entry(1, 1, 4.0);

        let norm = jac.frobenius_norm();
        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_drop_below() {
        let mut jac = SparseJacobian::new(2, 2);
        jac.add_entry(0, 0, 1.0);
        jac.add_entry(0, 1, 1e-12);
        jac.add_entry(1, 1, 2.0);

        jac.drop_below(1e-10);

        assert_eq!(jac.nnz(), 2);
        assert!((jac.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((jac.get(0, 1) - 0.0).abs() < 1e-10);
        assert!((jac.get(1, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_finite() {
        let mut jac = SparseJacobian::new(2, 2);
        jac.add_entry(0, 0, 1.0);
        jac.add_entry(1, 1, 2.0);
        assert!(jac.is_finite());

        jac.add_entry(0, 1, f64::NAN);
        assert!(!jac.is_finite());
    }

    #[test]
    #[should_panic(expected = "row index")]
    fn test_add_entry_out_of_bounds_row() {
        let mut jac = SparseJacobian::new(2, 2);
        jac.add_entry(5, 0, 1.0);
    }

    #[test]
    fn test_try_add_entry() {
        let mut jac = SparseJacobian::new(2, 2);
        assert!(jac.try_add_entry(0, 0, 1.0));
        assert!(!jac.try_add_entry(5, 0, 1.0));
        assert_eq!(jac.nnz(), 1);
    }

    #[test]
    fn test_iter() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0)];
        let jac = SparseJacobian::from_triplets(2, 2, triplets.clone());

        let collected: Vec<_> = jac.iter().collect();
        assert_eq!(collected, triplets);
    }

    #[test]
    fn test_clear() {
        let mut jac = SparseJacobian::new(2, 2);
        jac.add_entry(0, 0, 1.0);
        jac.add_entry(1, 1, 2.0);
        assert_eq!(jac.nnz(), 2);

        jac.clear();
        assert_eq!(jac.nnz(), 0);
        assert_eq!(jac.nrows, 2);
        assert_eq!(jac.ncols, 2);
    }

    #[test]
    fn test_pattern_extraction() {
        let mut jac = SparseJacobian::new(3, 3);
        jac.add_entry(0, 1, 1.0);
        jac.add_entry(1, 0, 2.0);
        jac.add_entry(2, 2, 3.0);

        let pattern = jac.pattern();
        assert_eq!(pattern.nrows, 3);
        assert_eq!(pattern.ncols, 3);
        assert_eq!(pattern.nnz(), 3);
        assert!(pattern.has_entry(0, 1));
        assert!(pattern.has_entry(1, 0));
        assert!(pattern.has_entry(2, 2));
        assert!(!pattern.has_entry(0, 0));
    }

    #[test]
    fn test_sparsity_ratio() {
        let mut jac = SparseJacobian::new(4, 4);
        jac.add_entry(0, 0, 1.0);
        jac.add_entry(1, 1, 2.0);

        let ratio = jac.sparsity_ratio();
        assert!((ratio - 2.0 / 16.0).abs() < 1e-10);
        assert!(jac.is_sparse(0.5));
    }

    #[test]
    fn test_csr_from_coo() {
        let mut jac = SparseJacobian::new(3, 3);
        jac.add_entry(0, 1, 1.0);
        jac.add_entry(0, 2, 2.0);
        jac.add_entry(1, 0, 3.0);
        jac.add_entry(2, 2, 4.0);

        let csr = jac.to_csr();
        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.ncols, 3);
        assert_eq!(csr.nnz(), 4);

        assert!((csr.get(0, 1) - 1.0).abs() < 1e-10);
        assert!((csr.get(0, 2) - 2.0).abs() < 1e-10);
        assert!((csr.get(1, 0) - 3.0).abs() < 1e-10);
        assert!((csr.get(2, 2) - 4.0).abs() < 1e-10);
        assert!((csr.get(0, 0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_duplicate_merge() {
        let triplets = vec![(0, 0, 1.0), (0, 0, 2.0), (1, 1, 3.0)];
        let csr = CsrMatrix::from_triplets(2, 2, &triplets);

        // Duplicates should be summed
        assert!((csr.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((csr.get(1, 1) - 3.0).abs() < 1e-10);
        assert_eq!(csr.nnz(), 2);
    }

    #[test]
    fn test_csr_mul_vec() {
        // Matrix: [[1, 2], [3, 4]]
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)];
        let csr = CsrMatrix::from_triplets(2, 2, &triplets);

        let x = vec![1.0, 2.0];
        let y = csr.mul_vec(&x).expect("should succeed");

        // y = [[1, 2], [3, 4]] * [1, 2] = [5, 11]
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_mul_vec_transpose() {
        // Matrix: [[1, 2], [3, 4]]
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)];
        let csr = CsrMatrix::from_triplets(2, 2, &triplets);

        let x = vec![1.0, 2.0];
        let y = csr.mul_vec_transpose(&x).expect("should succeed");

        // y = [[1, 3], [2, 4]] * [1, 2] = [7, 10]
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_wrong_dimension() {
        let triplets = vec![(0, 0, 1.0)];
        let csr = CsrMatrix::from_triplets(2, 2, &triplets);

        assert!(csr.mul_vec(&[1.0]).is_none());
        assert!(csr.mul_vec_transpose(&[1.0]).is_none());
    }

    #[test]
    fn test_csr_update_values() {
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)];
        let mut csr = CsrMatrix::from_triplets(2, 2, &triplets);

        let new_triplets = vec![(0, 0, 10.0), (0, 1, 20.0), (1, 0, 30.0)];
        assert!(csr.update_values(&new_triplets));

        assert!((csr.get(0, 0) - 10.0).abs() < 1e-10);
        assert!((csr.get(0, 1) - 20.0).abs() < 1e-10);
        assert!((csr.get(1, 0) - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_update_values_wrong_pattern() {
        let triplets = vec![(0, 0, 1.0)];
        let mut csr = CsrMatrix::from_triplets(2, 2, &triplets);

        // Try to update with entry not in pattern
        let new_triplets = vec![(0, 0, 10.0), (1, 1, 20.0)];
        assert!(!csr.update_values(&new_triplets));
    }

    #[test]
    fn test_csr_row_nnz() {
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (2, 0, 3.0)];
        let csr = CsrMatrix::from_triplets(3, 2, &triplets);

        let nnz = csr.row_nnz();
        assert_eq!(nnz, vec![2, 0, 1]);
        assert!(csr.has_empty_row());
    }

    #[test]
    fn test_csr_to_dense() {
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)];
        let csr = CsrMatrix::from_triplets(2, 2, &triplets);

        let dense = csr.to_dense();
        assert_eq!(dense[0], vec![1.0, 2.0]);
        assert_eq!(dense[1], vec![3.0, 4.0]);
    }

    #[test]
    fn test_csr_empty() {
        let csr = CsrMatrix::new(3, 4);
        assert!(csr.is_empty());
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.ncols, 4);
    }

    #[test]
    fn test_csr_row_access() {
        let triplets = vec![(0, 1, 1.0), (0, 3, 2.0), (1, 0, 3.0)];
        let csr = CsrMatrix::from_triplets(3, 4, &triplets);

        let (cols, vals) = csr.row(0).expect("row 0 should exist");
        assert_eq!(cols, &[1, 3]);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);

        let (cols, vals) = csr.row(1).expect("row 1 should exist");
        assert_eq!(cols, &[0]);
        assert!((vals[0] - 3.0).abs() < 1e-10);

        let (cols, vals) = csr.row(2).expect("row 2 should exist");
        assert!(cols.is_empty());
        assert!(vals.is_empty());

        assert!(csr.row(3).is_none());
    }
}
