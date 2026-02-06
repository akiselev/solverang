//! Solver selection heuristics.
//!
//! This module provides logic to automatically select the most appropriate solver
//! for a given sub-problem based on its characteristics (size, nonlinearity, sparsity).

use crate::geometry::constraint::Nonlinearity;

/// Solver selection strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverSelection {
    /// Newton-Raphson solver (fast for well-posed problems)
    NewtonRaphson,
    /// Levenberg-Marquardt solver (robust for difficult problems)
    LevenbergMarquardt,
    /// Robust solver (NR with LM fallback)
    Robust,
    /// Sparse solver (for large sparse systems)
    Sparse,
    /// Auto-selection (lets the solver choose)
    Auto,
}

/// Select the best solver for a sub-problem based on its characteristics.
///
/// # Arguments
///
/// * `n_variables` - Number of variables in the problem
/// * `n_equations` - Number of equations in the problem
/// * `max_nonlinearity` - Highest nonlinearity level among all constraints
/// * `sparsity_ratio` - Fraction of non-zero Jacobian entries (0.0 = dense, 1.0 = very sparse)
///
/// # Returns
///
/// The recommended solver type.
///
/// # Selection Heuristic
///
/// The selection follows these rules in order:
/// 1. Trivial problems (≤2 variables): Newton-Raphson
/// 2. All linear constraints: Newton-Raphson (exact in one step)
/// 3. Large and sparse (>100 vars, <10% sparsity): Sparse solver
/// 4. High nonlinearity: Levenberg-Marquardt (more robust)
/// 5. Default: Robust (NR with LM fallback)
pub fn select_solver(
    n_variables: usize,
    _n_equations: usize,
    max_nonlinearity: Nonlinearity,
    sparsity_ratio: f64,
) -> SolverSelection {
    // Trivial problems: just use NR
    if n_variables <= 2 {
        return SolverSelection::NewtonRaphson;
    }

    // All linear constraints: one NR step suffices
    if max_nonlinearity == Nonlinearity::Linear {
        return SolverSelection::NewtonRaphson;
    }

    // Large and sparse: use sparse solver
    if n_variables > 100 && sparsity_ratio < 0.1 {
        return SolverSelection::Sparse;
    }

    // High nonlinearity: LM is more robust
    if matches!(max_nonlinearity, Nonlinearity::High) {
        return SolverSelection::LevenbergMarquardt;
    }

    // Default: robust (NR with LM fallback)
    SolverSelection::Robust
}

/// Estimate Jacobian sparsity ratio.
///
/// The sparsity ratio is the fraction of non-zero entries in the Jacobian matrix.
/// - 0.0 means fully sparse (no non-zero entries, degenerate)
/// - 1.0 means fully dense (all entries are non-zero)
///
/// # Arguments
///
/// * `n_equations` - Number of rows in the Jacobian
/// * `n_variables` - Number of columns in the Jacobian
/// * `n_nonzero` - Number of non-zero entries
///
/// # Returns
///
/// The sparsity ratio in [0.0, 1.0].
pub fn estimate_sparsity(n_equations: usize, n_variables: usize, n_nonzero: usize) -> f64 {
    let total = n_equations * n_variables;
    if total == 0 {
        return 1.0; // Degenerate case: treat as dense
    }
    (n_nonzero as f64) / (total as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_trivial() {
        // Trivial problem: 1 variable
        let selection = select_solver(1, 1, Nonlinearity::Moderate, 0.5);
        assert_eq!(selection, SolverSelection::NewtonRaphson);

        // Trivial problem: 2 variables
        let selection = select_solver(2, 2, Nonlinearity::High, 0.5);
        assert_eq!(selection, SolverSelection::NewtonRaphson);
    }

    #[test]
    fn test_select_linear() {
        // All linear constraints
        let selection = select_solver(10, 10, Nonlinearity::Linear, 0.5);
        assert_eq!(selection, SolverSelection::NewtonRaphson);

        let selection = select_solver(100, 100, Nonlinearity::Linear, 0.1);
        assert_eq!(selection, SolverSelection::NewtonRaphson);
    }

    #[test]
    fn test_select_sparse() {
        // Large and sparse: >100 vars, <10% sparsity
        let selection = select_solver(200, 200, Nonlinearity::Moderate, 0.05);
        assert_eq!(selection, SolverSelection::Sparse);

        // Large but not sparse enough
        let selection = select_solver(200, 200, Nonlinearity::Moderate, 0.15);
        assert_eq!(selection, SolverSelection::Robust);

        // Sparse but not large enough
        let selection = select_solver(50, 50, Nonlinearity::Moderate, 0.05);
        assert_eq!(selection, SolverSelection::Robust);
    }

    #[test]
    fn test_select_high_nonlinearity() {
        // High nonlinearity: prefer LM
        let selection = select_solver(10, 10, Nonlinearity::High, 0.5);
        assert_eq!(selection, SolverSelection::LevenbergMarquardt);

        // But trivial problems still use NR
        let selection = select_solver(2, 2, Nonlinearity::High, 0.5);
        assert_eq!(selection, SolverSelection::NewtonRaphson);

        // And sparse problems use sparse solver
        let selection = select_solver(200, 200, Nonlinearity::High, 0.05);
        assert_eq!(selection, SolverSelection::Sparse);
    }

    #[test]
    fn test_select_moderate() {
        // Moderate nonlinearity: default to Robust
        let selection = select_solver(10, 10, Nonlinearity::Moderate, 0.5);
        assert_eq!(selection, SolverSelection::Robust);

        let selection = select_solver(50, 50, Nonlinearity::Moderate, 0.3);
        assert_eq!(selection, SolverSelection::Robust);
    }

    #[test]
    fn test_estimate_sparsity() {
        // Fully dense: all entries non-zero
        let sparsity = estimate_sparsity(10, 10, 100);
        assert!((sparsity - 1.0).abs() < 1e-10);

        // Half dense
        let sparsity = estimate_sparsity(10, 10, 50);
        assert!((sparsity - 0.5).abs() < 1e-10);

        // Very sparse: 5% non-zero
        let sparsity = estimate_sparsity(100, 100, 500);
        assert!((sparsity - 0.05).abs() < 1e-10);

        // Empty matrix: degenerate case
        let sparsity = estimate_sparsity(0, 0, 0);
        assert_eq!(sparsity, 1.0);
    }

    #[test]
    fn test_estimate_sparsity_rectangular() {
        // Rectangular matrix: 10x20 = 200 total entries, 50 non-zero
        let sparsity = estimate_sparsity(10, 20, 50);
        assert!((sparsity - 0.25).abs() < 1e-10);

        // Overdetermined: 20x10 = 200 total entries, 100 non-zero
        let sparsity = estimate_sparsity(20, 10, 100);
        assert!((sparsity - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_solver_selection_variants() {
        // Test all enum variants
        let selections = vec![
            SolverSelection::NewtonRaphson,
            SolverSelection::LevenbergMarquardt,
            SolverSelection::Robust,
            SolverSelection::Sparse,
            SolverSelection::Auto,
        ];

        // Test that they're distinct
        for (i, s1) in selections.iter().enumerate() {
            for (j, s2) in selections.iter().enumerate() {
                if i == j {
                    assert_eq!(s1, s2);
                } else {
                    assert_ne!(s1, s2);
                }
            }
        }
    }

    #[test]
    fn test_solver_selection_debug() {
        // Test Debug trait
        let s = format!("{:?}", SolverSelection::NewtonRaphson);
        assert!(s.contains("NewtonRaphson"));

        let s = format!("{:?}", SolverSelection::Robust);
        assert!(s.contains("Robust"));
    }

    #[test]
    fn test_solver_selection_clone_copy() {
        let s1 = SolverSelection::LevenbergMarquardt;
        let s2 = s1; // Copy
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_edge_cases() {
        // Zero variables (degenerate)
        let selection = select_solver(0, 10, Nonlinearity::Moderate, 0.5);
        assert_eq!(selection, SolverSelection::NewtonRaphson); // ≤2 rule

        // Exactly at threshold: 100 variables
        let selection = select_solver(100, 100, Nonlinearity::Moderate, 0.05);
        assert_eq!(selection, SolverSelection::Robust); // Not >100

        // Exactly at threshold: 101 variables
        let selection = select_solver(101, 101, Nonlinearity::Moderate, 0.05);
        assert_eq!(selection, SolverSelection::Sparse); // >100

        // Exactly at sparsity threshold: 0.1
        let selection = select_solver(200, 200, Nonlinearity::Moderate, 0.1);
        assert_eq!(selection, SolverSelection::Robust); // Not <0.1

        // Just below threshold: 0.09
        let selection = select_solver(200, 200, Nonlinearity::Moderate, 0.09);
        assert_eq!(selection, SolverSelection::Sparse); // <0.1
    }
}
