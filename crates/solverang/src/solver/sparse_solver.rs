//! Sparse solver optimized for large systems with low density Jacobians.
//!
//! This module provides a Newton-Raphson solver variant that uses sparse
//! matrix operations for efficiency with large, sparse constraint systems.
//!
//! # When to Use
//!
//! The sparse solver is beneficial when:
//! - The Jacobian matrix is large (1000+ variables)
//! - The Jacobian is sparse (less than 30% non-zero entries)
//! - Each constraint involves only a few variables
//!
//! For dense or small problems, the regular Newton-Raphson solver is faster.

use crate::jacobian::{CsrMatrix, SparsityPattern};
use crate::problem::Problem;
use crate::solver::result::{SolveError, SolveResult};

/// Configuration for the sparse solver.
#[derive(Clone, Debug)]
pub struct SparseSolverConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,

    /// Convergence tolerance for residual norm.
    pub tolerance: f64,

    /// Sparsity threshold: switch to sparse solver when nnz/total < this value.
    /// Default: 0.3 (30% non-zero entries).
    pub sparsity_threshold: f64,

    /// Enable pattern caching for incremental solving.
    /// When true, the sparsity pattern is computed once and reused.
    pub use_pattern_cache: bool,

    /// Tolerance for linear solver (relative error).
    pub linear_tolerance: f64,

    /// Enable backtracking line search.
    pub line_search: bool,

    /// Backtracking factor for line search.
    pub backtrack_factor: f64,

    /// Maximum line search iterations.
    pub max_line_search_iterations: usize,

    /// Minimum step size before giving up line search.
    pub min_step_size: f64,
}

impl Default for SparseSolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-8,
            sparsity_threshold: 0.3,
            use_pattern_cache: true,
            linear_tolerance: 1e-10,
            line_search: true,
            backtrack_factor: 0.5,
            max_line_search_iterations: 20,
            min_step_size: 1e-12,
        }
    }
}

impl SparseSolverConfig {
    /// Configuration for fast convergence on well-behaved sparse problems.
    pub fn fast() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            use_pattern_cache: true,
            line_search: false,
            ..Default::default()
        }
    }

    /// Configuration for robust convergence on difficult sparse problems.
    pub fn robust() -> Self {
        Self {
            max_iterations: 500,
            tolerance: 1e-10,
            use_pattern_cache: true,
            line_search: true,
            backtrack_factor: 0.5,
            max_line_search_iterations: 30,
            min_step_size: 1e-14,
            ..Default::default()
        }
    }
}

/// Sparse solver for large, sparse constraint systems.
///
/// This solver uses sparse matrix operations for efficiency with systems
/// where most Jacobian entries are zero.
///
/// # Algorithm
///
/// The solver uses Newton-Raphson iteration with sparse linear algebra:
///
/// 1. Compute residuals F(x)
/// 2. Compute sparse Jacobian J(x)
/// 3. Solve J * dx = -F using sparse QR or LU decomposition
/// 4. Update x = x + alpha * dx (with optional line search)
/// 5. Repeat until ||F(x)|| < tolerance
pub struct SparseSolver {
    config: SparseSolverConfig,
    cached_pattern: Option<SparsityPattern>,
}

impl SparseSolver {
    /// Create a new sparse solver with the given configuration.
    pub fn new(config: SparseSolverConfig) -> Self {
        Self {
            config,
            cached_pattern: None,
        }
    }

    /// Create a sparse solver with default configuration.
    pub fn default_solver() -> Self {
        Self::new(SparseSolverConfig::default())
    }

    /// Get the solver configuration.
    pub fn config(&self) -> &SparseSolverConfig {
        &self.config
    }

    /// Get the cached sparsity pattern, if any.
    pub fn cached_pattern(&self) -> Option<&SparsityPattern> {
        self.cached_pattern.as_ref()
    }

    /// Clear the cached sparsity pattern.
    pub fn clear_pattern_cache(&mut self) {
        self.cached_pattern = None;
    }

    /// Solve a problem starting from the given initial point.
    pub fn solve<P: Problem + ?Sized>(&mut self, problem: &P, x0: &[f64]) -> SolveResult {
        let n = problem.variable_count();
        let m = problem.residual_count();

        // Validate dimensions
        if n == 0 {
            return SolveResult::Failed {
                error: SolveError::NoVariables,
            };
        }

        if m == 0 {
            return SolveResult::Failed {
                error: SolveError::NoEquations,
            };
        }

        if x0.len() != n {
            return SolveResult::Failed {
                error: SolveError::DimensionMismatch {
                    expected: n,
                    got: x0.len(),
                },
            };
        }

        // Check initial point for non-finite values
        if x0.iter().any(|v| !v.is_finite()) {
            return SolveResult::Failed {
                error: SolveError::NonFiniteResiduals,
            };
        }

        let mut x = x0.to_vec();

        for iteration in 0..self.config.max_iterations {
            // Compute residuals
            let residuals = problem.residuals(&x);

            // Check for non-finite residuals
            if residuals.iter().any(|r| !r.is_finite()) {
                return SolveResult::Failed {
                    error: SolveError::NonFiniteResiduals,
                };
            }

            let norm: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

            // Check convergence
            if norm < self.config.tolerance {
                return SolveResult::Converged {
                    solution: x,
                    iterations: iteration,
                    residual_norm: norm,
                };
            }

            // Compute Jacobian
            let jacobian_triplets = problem.jacobian(&x);

            // Check for non-finite Jacobian entries
            if jacobian_triplets.iter().any(|(_, _, v)| !v.is_finite()) {
                return SolveResult::Failed {
                    error: SolveError::NonFiniteJacobian,
                };
            }

            // Build CSR matrix
            let csr = CsrMatrix::from_triplets(m, n, &jacobian_triplets);

            // Check if sparse solving is appropriate
            let sparsity = csr.nnz() as f64 / (m * n) as f64;
            let use_sparse = sparsity < self.config.sparsity_threshold;

            // Update pattern cache if needed
            if self.config.use_pattern_cache && self.cached_pattern.is_none() {
                self.cached_pattern = Some(csr.pattern());
            }

            // Solve linear system J * delta = -residuals
            let neg_residuals: Vec<f64> = residuals.iter().map(|r| -r).collect();
            let delta = if use_sparse {
                self.solve_sparse(&csr, &neg_residuals)
            } else {
                self.solve_dense(&csr, &neg_residuals)
            };

            let delta = match delta {
                Some(d) => d,
                None => {
                    return SolveResult::Failed {
                        error: SolveError::SingularJacobian,
                    };
                }
            };

            // Line search (optional)
            let alpha = if self.config.line_search {
                match self.line_search(problem, &x, &delta, norm) {
                    Some(a) => a,
                    None => {
                        return SolveResult::NotConverged {
                            solution: x,
                            iterations: iteration,
                            residual_norm: norm,
                        };
                    }
                }
            } else {
                1.0
            };

            // Update solution
            for (xi, di) in x.iter_mut().zip(delta.iter()) {
                *xi += alpha * di;
            }
        }

        // Did not converge within max iterations
        let residuals = problem.residuals(&x);
        let norm: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

        SolveResult::NotConverged {
            solution: x,
            iterations: self.config.max_iterations,
            residual_norm: norm,
        }
    }

    /// Solve using the problem's default initial point.
    pub fn solve_from_initial<P: Problem + ?Sized>(
        &mut self,
        problem: &P,
        factor: f64,
    ) -> SolveResult {
        let x0 = problem.initial_point(factor);
        self.solve(problem, &x0)
    }

    /// Solve the linear system using sparse operations.
    #[cfg(feature = "sparse")]
    fn solve_sparse(&self, csr: &CsrMatrix, rhs: &[f64]) -> Option<Vec<f64>> {
        use faer::prelude::*;
        use faer::sparse::{SparseColMat, Triplet};

        let m = csr.nrows;
        let n = csr.ncols;

        if csr.is_empty() {
            return None;
        }

        // Convert CSR to faer's triplet format
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(csr.nnz());
        for row in 0..m {
            let start = csr.row_ptr[row];
            let end = csr.row_ptr[row + 1];
            for (idx, &col) in csr.col_indices[start..end].iter().enumerate() {
                triplets.push(Triplet::new(row, col, csr.values[start + idx]));
            }
        }

        // Build faer sparse matrix (CSC format)
        let sparse_mat = match SparseColMat::<usize, f64>::try_new_from_triplets(m, n, &triplets) {
            Ok(mat) => mat,
            Err(_) => return None,
        };

        // Use sparse QR for rectangular systems, sparse LU for square
        if m == n {
            // Square system - try sparse LU
            match sparse_mat.sp_lu() {
                Ok(lu) => {
                    let mut result = faer::Col::<f64>::zeros(n);
                    for (i, &val) in rhs.iter().enumerate() {
                        if i < n {
                            result[i] = val;
                        }
                    }
                    lu.solve_in_place(result.as_mut());
                    Some(result.iter().copied().collect())
                }
                Err(_) => {
                    // Fall back to QR
                    self.solve_sparse_qr(&sparse_mat, rhs, m, n)
                }
            }
        } else {
            // Rectangular - use QR
            self.solve_sparse_qr(&sparse_mat, rhs, m, n)
        }
    }

    /// Solve using sparse QR decomposition (for overdetermined systems).
    #[cfg(feature = "sparse")]
    fn solve_sparse_qr(
        &self,
        sparse_mat: &faer::sparse::SparseColMat<usize, f64>,
        rhs: &[f64],
        m: usize,
        n: usize,
    ) -> Option<Vec<f64>> {
        use faer::prelude::*;

        if m >= n {
            // Overdetermined or square: use sparse QR
            match sparse_mat.sp_qr() {
                Ok(qr) => {
                    // Create a column vector with the RHS, padded or truncated as needed
                    let mut rhs_col = faer::Col::<f64>::zeros(m);
                    for (i, &val) in rhs.iter().enumerate() {
                        if i < m {
                            rhs_col[i] = val;
                        }
                    }

                    // Solve least squares problem
                    qr.solve_lstsq_in_place(rhs_col.as_mut());

                    // Extract first n components
                    Some(rhs_col.iter().take(n).copied().collect())
                }
                Err(_) => None,
            }
        } else {
            // Underdetermined: use minimum norm solution via normal equations
            // This is less common, fall back to dense
            None
        }
    }

    #[cfg(not(feature = "sparse"))]
    fn solve_sparse(&self, csr: &CsrMatrix, rhs: &[f64]) -> Option<Vec<f64>> {
        // Without sparse feature, fall back to dense
        self.solve_dense(csr, rhs)
    }

    /// Solve the linear system using dense operations.
    fn solve_dense(&self, csr: &CsrMatrix, rhs: &[f64]) -> Option<Vec<f64>> {
        use nalgebra::{DMatrix, DVector};

        let m = csr.nrows;
        let n = csr.ncols;

        // Convert to dense matrix
        let dense = csr.to_dense();
        let mut j = DMatrix::zeros(m, n);
        for (row_idx, row) in dense.iter().enumerate() {
            for (col_idx, &val) in row.iter().enumerate() {
                j[(row_idx, col_idx)] = val;
            }
        }

        let b = DVector::from_column_slice(rhs);

        if m == n {
            // Square system: try LU first
            if let Some(solution) = j.clone().lu().solve(&b) {
                return Some(solution.as_slice().to_vec());
            }
        }

        // Use SVD for rectangular or singular systems
        let svd = j.svd(true, true);
        svd.solve(&b, self.config.linear_tolerance)
            .ok()
            .map(|s| s.as_slice().to_vec())
    }

    /// Backtracking line search with Armijo condition.
    fn line_search<P: Problem + ?Sized>(
        &self,
        problem: &P,
        x: &[f64],
        delta: &[f64],
        f0: f64,
    ) -> Option<f64> {
        let mut alpha = 1.0;
        let rho = self.config.backtrack_factor;
        let armijo_c = 1e-4; // Standard Armijo constant

        for _ in 0..self.config.max_line_search_iterations {
            let x_new: Vec<f64> = x
                .iter()
                .zip(delta.iter())
                .map(|(&xi, &di)| xi + alpha * di)
                .collect();

            let residuals = problem.residuals(&x_new);

            // Check for non-finite values
            if residuals.iter().any(|r| !r.is_finite()) {
                alpha *= rho;
                if alpha < self.config.min_step_size {
                    return None;
                }
                continue;
            }

            let f_new: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

            // Armijo condition
            if f_new <= f0 * (1.0 - armijo_c * alpha) {
                return Some(alpha);
            }

            alpha *= rho;

            if alpha < self.config.min_step_size {
                // Accept the small step anyway
                return Some(alpha);
            }
        }

        Some(alpha)
    }
}

impl Default for SparseSolver {
    fn default() -> Self {
        Self::default_solver()
    }
}

/// Determine if a problem should use sparse solving based on Jacobian density.
pub fn should_use_sparse<P: Problem + ?Sized>(problem: &P, threshold: f64) -> bool {
    let x0 = problem.initial_point(1.0);
    let jacobian = problem.jacobian(&x0);

    let m = problem.residual_count();
    let n = problem.variable_count();
    let total = m * n;

    if total == 0 {
        return false;
    }

    let nnz = jacobian.len();
    let density = nnz as f64 / total as f64;

    density < threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple sparse problem: diagonal system
    struct DiagonalProblem {
        size: usize,
    }

    impl Problem for DiagonalProblem {
        fn name(&self) -> &str {
            "diagonal"
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
            // Identity matrix - very sparse for large n
            (0..self.size).map(|i| (i, i, 1.0)).collect()
        }

        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![0.0 * factor; self.size]
        }
    }

    // Tridiagonal problem
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

                // -x_{i-1} + 2*x_i - x_{i+1} = 1
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

        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![0.0 * factor; self.size]
        }
    }

    #[test]
    fn test_diagonal_problem() {
        let problem = DiagonalProblem { size: 10 };
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
    }

    #[test]
    fn test_tridiagonal_problem() {
        let problem = TridiagonalProblem { size: 10 };
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
    fn test_should_use_sparse() {
        let sparse_problem = DiagonalProblem { size: 100 };
        assert!(should_use_sparse(&sparse_problem, 0.1));

        // A dense 2x2 problem wouldn't be sparse
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
                vec![(0, 0, 1.0), (0, 1, 0.0), (1, 0, 0.0), (1, 1, 1.0)]
            }
            fn initial_point(&self, _f: f64) -> Vec<f64> {
                vec![0.0, 0.0]
            }
        }

        // 4 entries in 2x2 = 100% dense
        assert!(!should_use_sparse(&DenseProblem, 0.5));
    }

    #[test]
    fn test_dimension_mismatch() {
        let problem = DiagonalProblem { size: 5 };
        let mut solver = SparseSolver::default_solver();

        let result = solver.solve(&problem, &[1.0, 2.0]); // Wrong size

        assert!(!result.is_completed());
        assert_eq!(
            result.error(),
            Some(&SolveError::DimensionMismatch {
                expected: 5,
                got: 2
            })
        );
    }

    #[test]
    fn test_pattern_caching() {
        let problem = DiagonalProblem { size: 5 };
        let mut solver = SparseSolver::new(SparseSolverConfig {
            use_pattern_cache: true,
            ..Default::default()
        });

        assert!(solver.cached_pattern().is_none());

        let x0 = problem.initial_point(1.0);
        let _ = solver.solve(&problem, &x0);

        assert!(solver.cached_pattern().is_some());

        solver.clear_pattern_cache();
        assert!(solver.cached_pattern().is_none());
    }

    #[test]
    fn test_config_presets() {
        let fast = SparseSolverConfig::fast();
        assert!(!fast.line_search);
        assert_eq!(fast.max_iterations, 100);

        let robust = SparseSolverConfig::robust();
        assert!(robust.line_search);
        assert_eq!(robust.max_iterations, 500);
    }

    // Empty problem
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

    #[test]
    fn test_empty_problem() {
        let problem = EmptyProblem;
        let mut solver = SparseSolver::default_solver();

        let result = solver.solve(&problem, &[0.0, 0.0]);
        assert!(!result.is_completed());
        assert_eq!(result.error(), Some(&SolveError::NoEquations));
    }

    struct NoVariablesProblem;

    impl Problem for NoVariablesProblem {
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

    #[test]
    fn test_no_variables() {
        let problem = NoVariablesProblem;
        let mut solver = SparseSolver::default_solver();

        let result = solver.solve(&problem, &[]);
        assert!(!result.is_completed());
        assert_eq!(result.error(), Some(&SolveError::NoVariables));
    }

    #[test]
    fn test_large_sparse_problem() {
        // Test with a larger system to verify scaling
        let problem = DiagonalProblem { size: 100 };
        let mut solver = SparseSolver::default_solver();
        let x0 = problem.initial_point(1.0);

        let result = solver.solve(&problem, &x0);
        assert!(result.is_converged(), "Result: {:?}", result);

        // Should converge in very few iterations for this linear problem
        assert!(result.iterations().unwrap_or(1000) < 10);
    }
}
