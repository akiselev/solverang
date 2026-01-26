//! Sparsity pattern representation for Jacobian matrices.
//!
//! This module provides a `SparsityPattern` type that represents the structural
//! sparsity of a matrix without storing values. This is useful for:
//!
//! - Caching Jacobian structure for incremental solving (pattern rarely changes)
//! - Efficiently updating only the values when the structure is known
//! - Pre-allocating storage for sparse operations

use std::collections::HashMap;

/// Compressed Sparse Row (CSR) format sparsity pattern.
///
/// This structure describes which elements of a sparse matrix are non-zero
/// without storing the actual values. It uses CSR format which is efficient
/// for row-wise access and matrix-vector products.
///
/// # CSR Format
///
/// For a matrix with entries at positions (0,1), (0,3), (1,0), (1,2), (2,3):
/// ```text
/// row_ptr = [0, 2, 4, 5]  // Row 0 has entries at indices 0..2, row 1 at 2..4, etc.
/// col_idx = [1, 3, 0, 2, 3]  // Column indices of each entry
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparsityPattern {
    /// Row pointers: row_ptr\[i\]..row_ptr\[i+1\] gives the range of entries in row i.
    pub row_ptr: Vec<usize>,
    /// Column indices for each non-zero entry.
    pub col_idx: Vec<usize>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

impl SparsityPattern {
    /// Create a new empty sparsity pattern.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            row_ptr: vec![0; nrows + 1],
            col_idx: Vec::new(),
            nrows,
            ncols,
        }
    }

    /// Create a sparsity pattern from COO (coordinate) format triplets.
    ///
    /// The triplets are (row, col) pairs indicating non-zero positions.
    /// Duplicate entries are merged (only one entry is created).
    pub fn from_coordinates(nrows: usize, ncols: usize, coordinates: &[(usize, usize)]) -> Self {
        if coordinates.is_empty() {
            return Self::new(nrows, ncols);
        }

        // Count entries per row and deduplicate
        let mut row_entries: Vec<Vec<usize>> = vec![Vec::new(); nrows];

        for &(row, col) in coordinates {
            if row < nrows && col < ncols {
                row_entries[row].push(col);
            }
        }

        // Sort and deduplicate each row
        for row_cols in &mut row_entries {
            row_cols.sort_unstable();
            row_cols.dedup();
        }

        // Build CSR structure
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::new();

        row_ptr.push(0);
        for row_cols in &row_entries {
            col_idx.extend_from_slice(row_cols);
            row_ptr.push(col_idx.len());
        }

        Self {
            row_ptr,
            col_idx,
            nrows,
            ncols,
        }
    }

    /// Create a sparsity pattern from triplets with values.
    ///
    /// This is a convenience method that ignores the values and only uses positions.
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let coordinates: Vec<(usize, usize)> = triplets.iter().map(|&(r, c, _)| (r, c)).collect();
        Self::from_coordinates(nrows, ncols, &coordinates)
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    /// Sparsity ratio: nnz / (nrows * ncols).
    ///
    /// Returns 0.0 if the matrix has zero dimensions.
    pub fn sparsity_ratio(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 0.0;
        }
        self.nnz() as f64 / total as f64
    }

    /// Check if the matrix is considered sparse (sparsity ratio below threshold).
    pub fn is_sparse(&self, threshold: f64) -> bool {
        self.sparsity_ratio() < threshold
    }

    /// Get the column indices for a specific row.
    ///
    /// Returns an empty slice if the row is out of bounds.
    pub fn row_indices(&self, row: usize) -> &[usize] {
        if row >= self.nrows {
            return &[];
        }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        &self.col_idx[start..end]
    }

    /// Check if a position has a non-zero entry.
    pub fn has_entry(&self, row: usize, col: usize) -> bool {
        if row >= self.nrows || col >= self.ncols {
            return false;
        }
        self.row_indices(row).binary_search(&col).is_ok()
    }

    /// Get the linear index of an entry in the values array.
    ///
    /// Returns None if the entry doesn't exist in the pattern.
    pub fn entry_index(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.nrows {
            return None;
        }
        let start = self.row_ptr[row];
        let row_cols = self.row_indices(row);
        row_cols.binary_search(&col).ok().map(|pos| start + pos)
    }

    /// Number of entries in a specific row.
    pub fn row_nnz(&self, row: usize) -> usize {
        if row >= self.nrows {
            return 0;
        }
        self.row_ptr[row + 1] - self.row_ptr[row]
    }

    /// Iterate over (row, col) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.nrows).flat_map(move |row| {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            self.col_idx[start..end].iter().map(move |&col| (row, col))
        })
    }

    /// Create a mapping from (row, col) to value index.
    ///
    /// This is useful for fast value updates when the pattern is cached.
    pub fn index_map(&self) -> HashMap<(usize, usize), usize> {
        let mut map = HashMap::with_capacity(self.nnz());
        for (idx, (row, col)) in self.iter().enumerate() {
            map.insert((row, col), idx);
        }
        map
    }

    /// Check if two patterns are structurally identical.
    pub fn same_structure(&self, other: &SparsityPattern) -> bool {
        self.nrows == other.nrows
            && self.ncols == other.ncols
            && self.row_ptr == other.row_ptr
            && self.col_idx == other.col_idx
    }

    /// Get rows that have at least one entry.
    pub fn non_empty_rows(&self) -> Vec<usize> {
        (0..self.nrows)
            .filter(|&row| self.row_nnz(row) > 0)
            .collect()
    }

    /// Get columns that have at least one entry.
    pub fn non_empty_cols(&self) -> Vec<usize> {
        let mut cols: Vec<usize> = self.col_idx.to_vec();
        cols.sort_unstable();
        cols.dedup();
        cols
    }

    /// Check if any row has zero entries (potential singularity indicator).
    pub fn has_empty_row(&self) -> bool {
        (0..self.nrows).any(|row| self.row_nnz(row) == 0)
    }

    /// Check if any column has zero entries (potential singularity indicator).
    pub fn has_empty_col(&self) -> bool {
        let non_empty = self.non_empty_cols();
        non_empty.len() < self.ncols
    }

    /// Convert to COO format (coordinate list).
    pub fn to_coordinates(&self) -> Vec<(usize, usize)> {
        self.iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let pattern = SparsityPattern::new(3, 4);
        assert_eq!(pattern.nrows, 3);
        assert_eq!(pattern.ncols, 4);
        assert_eq!(pattern.nnz(), 0);
        assert_eq!(pattern.row_ptr.len(), 4); // nrows + 1
    }

    #[test]
    fn test_from_coordinates() {
        let coords = vec![(0, 1), (0, 3), (1, 0), (1, 2), (2, 3)];
        let pattern = SparsityPattern::from_coordinates(3, 4, &coords);

        assert_eq!(pattern.nrows, 3);
        assert_eq!(pattern.ncols, 4);
        assert_eq!(pattern.nnz(), 5);

        assert!(pattern.has_entry(0, 1));
        assert!(pattern.has_entry(0, 3));
        assert!(pattern.has_entry(1, 0));
        assert!(pattern.has_entry(1, 2));
        assert!(pattern.has_entry(2, 3));
        assert!(!pattern.has_entry(0, 0));
        assert!(!pattern.has_entry(2, 2));
    }

    #[test]
    fn test_from_triplets() {
        let triplets = vec![(0, 1, 1.0), (0, 3, 2.0), (1, 0, 3.0)];
        let pattern = SparsityPattern::from_triplets(2, 4, &triplets);

        assert_eq!(pattern.nnz(), 3);
        assert!(pattern.has_entry(0, 1));
        assert!(pattern.has_entry(0, 3));
        assert!(pattern.has_entry(1, 0));
    }

    #[test]
    fn test_duplicate_entries() {
        let coords = vec![(0, 1), (0, 1), (0, 1), (1, 0)];
        let pattern = SparsityPattern::from_coordinates(2, 2, &coords);

        assert_eq!(pattern.nnz(), 2); // Duplicates merged
    }

    #[test]
    fn test_row_indices() {
        let coords = vec![(0, 1), (0, 3), (1, 0), (1, 2)];
        let pattern = SparsityPattern::from_coordinates(2, 4, &coords);

        assert_eq!(pattern.row_indices(0), &[1, 3]);
        assert_eq!(pattern.row_indices(1), &[0, 2]);
        assert_eq!(pattern.row_indices(5), &[]); // Out of bounds
    }

    #[test]
    fn test_entry_index() {
        let coords = vec![(0, 1), (0, 3), (1, 0), (1, 2)];
        let pattern = SparsityPattern::from_coordinates(2, 4, &coords);

        assert_eq!(pattern.entry_index(0, 1), Some(0));
        assert_eq!(pattern.entry_index(0, 3), Some(1));
        assert_eq!(pattern.entry_index(1, 0), Some(2));
        assert_eq!(pattern.entry_index(1, 2), Some(3));
        assert_eq!(pattern.entry_index(0, 0), None);
    }

    #[test]
    fn test_sparsity_ratio() {
        let coords = vec![(0, 0), (1, 1)]; // 2 entries in 4x4 matrix
        let pattern = SparsityPattern::from_coordinates(4, 4, &coords);

        let ratio = pattern.sparsity_ratio();
        assert!((ratio - 2.0 / 16.0).abs() < 1e-10);
        assert!(pattern.is_sparse(0.5));
        assert!(!pattern.is_sparse(0.1));
    }

    #[test]
    fn test_zero_dimension() {
        let pattern = SparsityPattern::new(0, 5);
        assert_eq!(pattern.sparsity_ratio(), 0.0);
    }

    #[test]
    fn test_has_empty_row_col() {
        let coords = vec![(0, 0), (0, 1)]; // Row 1 is empty
        let pattern = SparsityPattern::from_coordinates(2, 3, &coords);

        assert!(pattern.has_empty_row());
        assert!(pattern.has_empty_col()); // Column 2 is empty
    }

    #[test]
    fn test_same_structure() {
        let coords1 = vec![(0, 1), (1, 0)];
        let coords2 = vec![(0, 1), (1, 0)];
        let coords3 = vec![(0, 1), (1, 1)];

        let p1 = SparsityPattern::from_coordinates(2, 2, &coords1);
        let p2 = SparsityPattern::from_coordinates(2, 2, &coords2);
        let p3 = SparsityPattern::from_coordinates(2, 2, &coords3);

        assert!(p1.same_structure(&p2));
        assert!(!p1.same_structure(&p3));
    }

    #[test]
    fn test_iter() {
        let coords = vec![(1, 0), (0, 1), (1, 2)];
        let pattern = SparsityPattern::from_coordinates(2, 3, &coords);

        let collected: Vec<_> = pattern.iter().collect();
        // Should be in row-major order
        assert_eq!(collected, vec![(0, 1), (1, 0), (1, 2)]);
    }

    #[test]
    fn test_index_map() {
        let coords = vec![(0, 1), (1, 0), (1, 2)];
        let pattern = SparsityPattern::from_coordinates(2, 3, &coords);

        let map = pattern.index_map();
        assert_eq!(map.get(&(0, 1)), Some(&0));
        assert_eq!(map.get(&(1, 0)), Some(&1));
        assert_eq!(map.get(&(1, 2)), Some(&2));
        assert_eq!(map.get(&(0, 0)), None);
    }

    #[test]
    fn test_out_of_bounds_coords() {
        let coords = vec![(0, 0), (10, 0), (0, 10)];
        let pattern = SparsityPattern::from_coordinates(2, 2, &coords);

        // Only (0, 0) should be included
        assert_eq!(pattern.nnz(), 1);
        assert!(pattern.has_entry(0, 0));
    }

    #[test]
    fn test_to_coordinates() {
        let coords = vec![(0, 1), (1, 0), (1, 2)];
        let pattern = SparsityPattern::from_coordinates(2, 3, &coords);

        let result = pattern.to_coordinates();
        assert_eq!(result, vec![(0, 1), (1, 0), (1, 2)]);
    }

    #[test]
    fn test_non_empty_rows_cols() {
        let coords = vec![(0, 1), (2, 3)]; // Row 1 and cols 0, 2 are empty
        let pattern = SparsityPattern::from_coordinates(3, 4, &coords);

        assert_eq!(pattern.non_empty_rows(), vec![0, 2]);
        assert_eq!(pattern.non_empty_cols(), vec![1, 3]);
    }
}
