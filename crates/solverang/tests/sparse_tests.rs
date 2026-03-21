//! Tests for the sparse solver and sparse matrix operations.

use solverang::{
    CsrMatrix, Problem, SolveError, SparseSolver, SparseSolverConfig, SparsityPattern,
};

/// Large diagonal problem to test sparse operations.
struct LargeDiagonalProblem {
    size: usize,
}

impl Problem for LargeDiagonalProblem {
    fn name(&self) -> &str {
        "large-diagonal"
    }

    fn residual_count(&self) -> usize {
        self.size
    }

    fn variable_count(&self) -> usize {
        self.size
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| xi - (i as f64 + 1.0))
            .collect()
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        (0..self.size).map(|i| (i, i, 1.0)).collect()
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        vec![0.0; self.size]
    }
}

/// Tridiagonal problem (sparse but connected).
struct TridiagonalProblem {
    size: usize,
}

impl Problem for TridiagonalProblem {
    fn name(&self) -> &str {
        "tridiagonal"
    }

    fn residual_count(&self) -> usize {
        self.size
    }

    fn variable_count(&self) -> usize {
        self.size
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        let n = self.size;
        let mut r = Vec::with_capacity(n);

        for i in 0..n {
            let xi = x[i];
            let left = if i > 0 { x[i - 1] } else { 0.0 };
            let right = if i < n - 1 { x[i + 1] } else { 0.0 };
            r.push(-left + 2.0 * xi - right - 1.0);
        }

        r
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = self.size;
        let mut entries = Vec::new();

        for i in 0..n {
            if i > 0 {
                entries.push((i, i - 1, -1.0));
            }
            entries.push((i, i, 2.0));
            if i < n - 1 {
                entries.push((i, i + 1, -1.0));
            }
        }

        entries
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        vec![0.0; self.size]
    }
}

/// Banded problem with variable bandwidth.
struct BandedProblem {
    size: usize,
    bandwidth: usize,
}

impl Problem for BandedProblem {
    fn name(&self) -> &str {
        "banded"
    }

    fn residual_count(&self) -> usize {
        self.size
    }

    fn variable_count(&self) -> usize {
        self.size
    }

    fn residuals(&self, x: &[f64]) -> Vec<f64> {
        let n = self.size;
        let bw = self.bandwidth;
        let mut r = Vec::with_capacity(n);

        for i in 0..n {
            let start = i.saturating_sub(bw);
            let end = (i + bw).min(n - 1) + 1;
            let sum: f64 = x[start..end].iter().sum();
            r.push(sum - (i as f64 + 1.0));
        }

        r
    }

    fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
        let n = self.size;
        let bw = self.bandwidth;
        let mut entries = Vec::new();

        for i in 0..n {
            for j in i.saturating_sub(bw)..=(i + bw).min(n - 1) {
                entries.push((i, j, 1.0));
            }
        }

        entries
    }

    fn initial_point(&self, _factor: f64) -> Vec<f64> {
        vec![0.0; self.size]
    }
}

// ============================================================================
// Sparse Solver Tests
// ============================================================================

#[test]
fn test_sparse_solver_diagonal() {
    let problem = LargeDiagonalProblem { size: 100 };
    let mut solver = SparseSolver::default_solver();
    let x0 = problem.initial_point(1.0);

    let result = solver.solve(&problem, &x0);

    assert!(result.is_converged(), "Result: {:?}", result);

    if let Some(solution) = result.solution() {
        for (i, &xi) in solution.iter().enumerate() {
            assert!(
                (xi - (i as f64 + 1.0)).abs() < 1e-6,
                "x[{}] should be {}, got {}",
                i,
                i + 1,
                xi
            );
        }
    }

    // Linear problem should converge in 1-2 iterations
    assert!(result.iterations().unwrap_or(1000) <= 5);
}

#[test]
fn test_sparse_solver_tridiagonal() {
    let problem = TridiagonalProblem { size: 50 };
    let mut solver = SparseSolver::default_solver();
    let x0 = problem.initial_point(1.0);

    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged() || result.is_completed(),
        "Result: {:?}",
        result
    );

    // Verify solution satisfies equations
    if let Some(solution) = result.solution() {
        let residuals = problem.residuals(solution);
        let norm: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();
        assert!(norm < 1e-4, "Residual norm {} too high", norm);
    }
}

#[test]
fn test_sparse_solver_banded() {
    let problem = BandedProblem {
        size: 30,
        bandwidth: 2,
    };
    let mut solver = SparseSolver::new(SparseSolverConfig::robust());
    let x0 = problem.initial_point(1.0);

    let result = solver.solve(&problem, &x0);

    assert!(
        result.is_converged() || result.is_completed(),
        "Result: {:?}",
        result
    );
}

#[test]
fn test_sparse_solver_dimension_mismatch() {
    let problem = LargeDiagonalProblem { size: 10 };
    let mut solver = SparseSolver::default_solver();

    let result = solver.solve(&problem, &[1.0, 2.0]);

    assert!(!result.is_completed());
    assert_eq!(
        result.error(),
        Some(&SolveError::DimensionMismatch {
            expected: 10,
            got: 2
        })
    );
}

#[test]
fn test_sparse_solver_pattern_caching() {
    let problem = LargeDiagonalProblem { size: 20 };
    let mut solver = SparseSolver::new(SparseSolverConfig {
        use_pattern_cache: true,
        ..Default::default()
    });

    assert!(solver.cached_pattern().is_none());

    let x0 = problem.initial_point(1.0);
    let _ = solver.solve(&problem, &x0);

    assert!(solver.cached_pattern().is_some());

    // Pattern should have correct dimensions
    let pattern = solver.cached_pattern().unwrap();
    assert_eq!(pattern.nrows, 20);
    assert_eq!(pattern.ncols, 20);

    solver.clear_pattern_cache();
    assert!(solver.cached_pattern().is_none());
}

#[test]
fn test_sparse_solver_config_presets() {
    let fast = SparseSolverConfig::fast();
    assert!(!fast.line_search);
    assert_eq!(fast.max_iterations, 100);

    let robust = SparseSolverConfig::robust();
    assert!(robust.line_search);
    assert_eq!(robust.max_iterations, 500);
}

// ============================================================================
// Sparsity Pattern Tests
// ============================================================================

#[test]
fn test_sparsity_pattern_creation() {
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
}

#[test]
fn test_sparsity_pattern_from_triplets() {
    let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (0, 1, 3.0)];
    let pattern = SparsityPattern::from_triplets(2, 2, &triplets);

    assert_eq!(pattern.nnz(), 3);
    assert!(pattern.has_entry(0, 0));
    assert!(pattern.has_entry(0, 1));
    assert!(pattern.has_entry(1, 1));
    assert!(!pattern.has_entry(1, 0));
}

#[test]
fn test_sparsity_pattern_duplicate_handling() {
    let coords = vec![(0, 0), (0, 0), (0, 0), (1, 1)];
    let pattern = SparsityPattern::from_coordinates(2, 2, &coords);

    assert_eq!(pattern.nnz(), 2); // Duplicates merged
}

#[test]
fn test_sparsity_pattern_row_indices() {
    let coords = vec![(0, 1), (0, 3), (1, 0), (1, 2)];
    let pattern = SparsityPattern::from_coordinates(2, 4, &coords);

    assert_eq!(pattern.row_indices(0), &[1, 3]);
    assert_eq!(pattern.row_indices(1), &[0, 2]);
    assert_eq!(pattern.row_indices(5), &[]); // Out of bounds
}

#[test]
fn test_sparsity_pattern_entry_index() {
    let coords = vec![(0, 1), (0, 3), (1, 0), (1, 2)];
    let pattern = SparsityPattern::from_coordinates(2, 4, &coords);

    assert_eq!(pattern.entry_index(0, 1), Some(0));
    assert_eq!(pattern.entry_index(0, 3), Some(1));
    assert_eq!(pattern.entry_index(1, 0), Some(2));
    assert_eq!(pattern.entry_index(1, 2), Some(3));
    assert_eq!(pattern.entry_index(0, 0), None);
}

#[test]
fn test_sparsity_pattern_sparsity_ratio() {
    let coords = vec![(0, 0), (1, 1)];
    let pattern = SparsityPattern::from_coordinates(4, 4, &coords);

    let ratio = pattern.sparsity_ratio();
    assert!((ratio - 2.0 / 16.0).abs() < 1e-10);
    assert!(pattern.is_sparse(0.5));
    assert!(!pattern.is_sparse(0.1));
}

#[test]
fn test_sparsity_pattern_empty_rows_cols() {
    let coords = vec![(0, 0), (0, 1)];
    let pattern = SparsityPattern::from_coordinates(2, 3, &coords);

    assert!(pattern.has_empty_row());
    assert!(pattern.has_empty_col());

    let non_empty_rows = pattern.non_empty_rows();
    assert_eq!(non_empty_rows, vec![0]);

    let non_empty_cols = pattern.non_empty_cols();
    assert_eq!(non_empty_cols, vec![0, 1]);
}

#[test]
fn test_sparsity_pattern_same_structure() {
    let coords1 = vec![(0, 1), (1, 0)];
    let coords2 = vec![(0, 1), (1, 0)];
    let coords3 = vec![(0, 1), (1, 1)];

    let p1 = SparsityPattern::from_coordinates(2, 2, &coords1);
    let p2 = SparsityPattern::from_coordinates(2, 2, &coords2);
    let p3 = SparsityPattern::from_coordinates(2, 2, &coords3);

    assert!(p1.same_structure(&p2));
    assert!(!p1.same_structure(&p3));
}

// ============================================================================
// CSR Matrix Tests
// ============================================================================

#[test]
fn test_csr_from_triplets() {
    let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)];
    let csr = CsrMatrix::from_triplets(2, 2, &triplets);

    assert_eq!(csr.nrows, 2);
    assert_eq!(csr.ncols, 2);
    assert_eq!(csr.nnz(), 4);

    assert!((csr.get(0, 0) - 1.0).abs() < 1e-10);
    assert!((csr.get(0, 1) - 2.0).abs() < 1e-10);
    assert!((csr.get(1, 0) - 3.0).abs() < 1e-10);
    assert!((csr.get(1, 1) - 4.0).abs() < 1e-10);
}

#[test]
fn test_csr_duplicate_summing() {
    let triplets = vec![(0, 0, 1.0), (0, 0, 2.0), (0, 0, 3.0)];
    let csr = CsrMatrix::from_triplets(1, 1, &triplets);

    // Duplicates should be summed
    assert!((csr.get(0, 0) - 6.0).abs() < 1e-10);
    assert_eq!(csr.nnz(), 1);
}

#[test]
fn test_csr_matrix_vector_multiply() {
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
fn test_csr_matrix_vector_transpose() {
    // Matrix: [[1, 2], [3, 4]]
    let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)];
    let csr = CsrMatrix::from_triplets(2, 2, &triplets);

    let x = vec![1.0, 2.0];
    let y = csr.mul_vec_transpose(&x).expect("should succeed");

    // y = A^T * x = [[1, 3], [2, 4]] * [1, 2] = [7, 10]
    assert!((y[0] - 7.0).abs() < 1e-10);
    assert!((y[1] - 10.0).abs() < 1e-10);
}

#[test]
fn test_csr_dimension_check() {
    let triplets = vec![(0, 0, 1.0)];
    let csr = CsrMatrix::from_triplets(2, 2, &triplets);

    assert!(csr.mul_vec(&[1.0]).is_none()); // Wrong size
    assert!(csr.mul_vec_transpose(&[1.0]).is_none()); // Wrong size
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
fn test_csr_update_values_pattern_mismatch() {
    let triplets = vec![(0, 0, 1.0)];
    let mut csr = CsrMatrix::from_triplets(2, 2, &triplets);

    // Try to add entry not in pattern
    let new_triplets = vec![(0, 0, 10.0), (1, 1, 20.0)];
    assert!(!csr.update_values(&new_triplets));
}

#[test]
fn test_csr_row_access() {
    let triplets = vec![(0, 1, 1.0), (0, 3, 2.0), (1, 0, 3.0), (2, 2, 4.0)];
    let csr = CsrMatrix::from_triplets(3, 4, &triplets);

    let (cols, vals) = csr.row(0).expect("row should exist");
    assert_eq!(cols, &[1, 3]);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);

    let (cols, vals) = csr.row(1).expect("row should exist");
    assert_eq!(cols, &[0]);
    assert!((vals[0] - 3.0).abs() < 1e-10);

    assert!(csr.row(10).is_none());
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
fn test_csr_frobenius_norm() {
    let triplets = vec![(0, 0, 3.0), (1, 1, 4.0)];
    let csr = CsrMatrix::from_triplets(2, 2, &triplets);

    let norm = csr.frobenius_norm();
    assert!((norm - 5.0).abs() < 1e-10);
}

#[test]
fn test_csr_is_finite() {
    let triplets = vec![(0, 0, 1.0), (1, 1, 2.0)];
    let csr = CsrMatrix::from_triplets(2, 2, &triplets);
    assert!(csr.is_finite());

    let bad_triplets = vec![(0, 0, f64::NAN)];
    let bad_csr = CsrMatrix::from_triplets(1, 1, &bad_triplets);
    assert!(!bad_csr.is_finite());
}

// ============================================================================
// should_use_sparse Tests
// ============================================================================

#[test]
fn test_should_use_sparse() {
    // Diagonal matrix is very sparse
    let sparse_problem = LargeDiagonalProblem { size: 100 };
    assert!(solverang::solver::should_use_sparse(&sparse_problem, 0.1));

    // Small dense problem
    struct DenseProblem;
    impl Problem for DenseProblem {
        fn name(&self) -> &str {
            "dense"
        }
        fn residual_count(&self) -> usize {
            2
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] - 1.0, x[1] - 2.0]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            // Full 2x2 matrix
            vec![(0, 0, 1.0), (0, 1, 0.5), (1, 0, 0.5), (1, 1, 1.0)]
        }
        fn initial_point(&self, _f: f64) -> Vec<f64> {
            vec![0.0, 0.0]
        }
    }

    // 4 entries in 2x2 = 100% dense
    assert!(!solverang::solver::should_use_sparse(&DenseProblem, 0.5));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_problem() {
    struct EmptyProblem;
    impl Problem for EmptyProblem {
        fn name(&self) -> &str {
            "empty"
        }
        fn residual_count(&self) -> usize {
            0
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, _x: &[f64]) -> Vec<f64> {
            vec![]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![]
        }
        fn initial_point(&self, _f: f64) -> Vec<f64> {
            vec![0.0, 0.0]
        }
    }

    let mut solver = SparseSolver::default_solver();
    let result = solver.solve(&EmptyProblem, &[0.0, 0.0]);
    assert!(!result.is_completed());
    assert_eq!(result.error(), Some(&SolveError::NoEquations));
}

#[test]
fn test_no_variables() {
    struct NoVarsProblem;
    impl Problem for NoVarsProblem {
        fn name(&self) -> &str {
            "no-vars"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            0
        }
        fn residuals(&self, _x: &[f64]) -> Vec<f64> {
            vec![1.0]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![]
        }
        fn initial_point(&self, _f: f64) -> Vec<f64> {
            vec![]
        }
    }

    let mut solver = SparseSolver::default_solver();
    let result = solver.solve(&NoVarsProblem, &[]);
    assert!(!result.is_completed());
    assert_eq!(result.error(), Some(&SolveError::NoVariables));
}

#[test]
fn test_non_finite_initial() {
    let problem = LargeDiagonalProblem { size: 3 };
    let mut solver = SparseSolver::default_solver();

    let result = solver.solve(&problem, &[f64::NAN, 0.0, 0.0]);
    assert!(!result.is_completed());
}

#[test]
fn test_zero_dimension_pattern() {
    let pattern = SparsityPattern::new(0, 5);
    assert_eq!(pattern.sparsity_ratio(), 0.0);
    assert_eq!(pattern.nnz(), 0);
}
