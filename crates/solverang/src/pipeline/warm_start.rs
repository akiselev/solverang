//! Warm starting and regularization for under-constrained systems.
//!
//! This module provides utilities for handling under-constrained systems
//! by adding regularization terms that prefer solutions close to the current state.

/// Default regularization weight for under-constrained systems.
///
/// This weight is added to the Tikhonov regularization term:
///   minimize ||F(x)||² + λ||x - x₀||²
///
/// The value is chosen to be:
/// - Small enough to not interfere with constraint satisfaction
/// - Large enough to prevent wild jumps in unconstrained degrees of freedom
pub const DEFAULT_LAMBDA: f64 = 1e-6;

/// Add regularization terms to residuals for under-constrained systems.
///
/// For under-constrained systems (more variables than equations), we add
/// Tikhonov regularization to prefer the solution closest to the current state:
///   minimize ||F(x)||² + λ||x - x_initial||²
///
/// This is implemented by augmenting the residual vector with additional terms.
///
/// # Arguments
///
/// * `original_residuals` - Residuals from the constraint system
/// * `x` - Current variable values
/// * `x_initial` - Initial variable values (warm start point)
/// * `lambda` - Regularization weight (typically `DEFAULT_LAMBDA`)
///
/// # Returns
///
/// Augmented residual vector with regularization terms appended.
///
/// # Example
///
/// ```
/// use solverang::pipeline::warm_start::{regularize_residuals, DEFAULT_LAMBDA};
///
/// let residuals = vec![1.0, 2.0]; // Original constraints
/// let x = vec![0.5, 1.5, 2.5];     // Current state
/// let x0 = vec![0.0, 1.0, 2.0];    // Initial state
///
/// let augmented = regularize_residuals(&residuals, &x, &x0, DEFAULT_LAMBDA);
///
/// // First 2 are original residuals, next 3 are regularization
/// assert_eq!(augmented.len(), 5);
/// assert_eq!(augmented[0], 1.0);
/// assert_eq!(augmented[1], 2.0);
/// // augmented[2] = lambda * (x[0] - x0[0]) = 1e-6 * 0.5
/// ```
pub fn regularize_residuals(
    original_residuals: &[f64],
    x: &[f64],
    x_initial: &[f64],
    lambda: f64,
) -> Vec<f64> {
    assert_eq!(
        x.len(),
        x_initial.len(),
        "x and x_initial must have the same length"
    );

    let mut residuals = original_residuals.to_vec();

    // Append regularization terms: λ(x - x₀)
    for i in 0..x.len() {
        residuals.push(lambda * (x[i] - x_initial[i]));
    }

    residuals
}

/// Add regularization terms to Jacobian for under-constrained systems.
///
/// Augments the Jacobian matrix with rows corresponding to the regularization
/// terms in the residuals. For the regularization term λ(xᵢ - x₀ᵢ), the Jacobian
/// entry is simply λ at row (n_original_equations + i), column i.
///
/// # Arguments
///
/// * `original_entries` - Sparse Jacobian entries from the constraint system
/// * `n_original_equations` - Number of original equations (before regularization)
/// * `n_variables` - Number of variables
/// * `lambda` - Regularization weight (typically `DEFAULT_LAMBDA`)
///
/// # Returns
///
/// Augmented Jacobian entries with regularization diagonal added.
///
/// # Example
///
/// ```
/// use solverang::pipeline::warm_start::{regularize_jacobian, DEFAULT_LAMBDA};
///
/// let jac = vec![
///     (0, 0, 2.0), (0, 1, 1.0),  // First equation
///     (1, 1, 3.0), (1, 2, -1.0), // Second equation
/// ];
///
/// let augmented = regularize_jacobian(&jac, 2, 3, DEFAULT_LAMBDA);
///
/// // Original 4 entries + 3 diagonal regularization entries
/// assert_eq!(augmented.len(), 7);
/// ```
pub fn regularize_jacobian(
    original_entries: &[(usize, usize, f64)],
    n_original_equations: usize,
    n_variables: usize,
    lambda: f64,
) -> Vec<(usize, usize, f64)> {
    let mut entries = original_entries.to_vec();

    // Add diagonal entries for regularization
    // ∂/∂xᵢ [λ(xᵢ - x₀ᵢ)] = λ
    for i in 0..n_variables {
        entries.push((n_original_equations + i, i, lambda));
    }

    entries
}

/// Check if a problem is under-constrained.
///
/// A problem is under-constrained if it has more variables than equations,
/// meaning there are degrees of freedom (infinite solutions).
///
/// # Arguments
///
/// * `n_equations` - Number of equations
/// * `n_variables` - Number of variables
///
/// # Returns
///
/// `true` if the system is under-constrained (DOF > 0).
pub fn is_underconstrained(n_equations: usize, n_variables: usize) -> bool {
    n_variables > n_equations
}

/// Calculate degrees of freedom for a system.
///
/// # Arguments
///
/// * `n_equations` - Number of equations
/// * `n_variables` - Number of variables
///
/// # Returns
///
/// Number of degrees of freedom. Can be negative (over-constrained).
pub fn degrees_of_freedom(n_equations: usize, n_variables: usize) -> i32 {
    (n_variables as i32) - (n_equations as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_lambda() {
        assert_eq!(DEFAULT_LAMBDA, 1e-6);
        assert!(DEFAULT_LAMBDA > 0.0);
        assert!(DEFAULT_LAMBDA < 1e-5);
    }

    #[test]
    fn test_regularize_residuals() {
        let original = vec![1.0, 2.0];
        let x = vec![0.5, 1.5, 2.5];
        let x0 = vec![0.0, 1.0, 2.0];
        let lambda = 1e-3;

        let augmented = regularize_residuals(&original, &x, &x0, lambda);

        // Should have 2 original + 3 regularization = 5 total
        assert_eq!(augmented.len(), 5);

        // Original residuals unchanged
        assert_eq!(augmented[0], 1.0);
        assert_eq!(augmented[1], 2.0);

        // Regularization terms: λ(x - x0)
        assert!((augmented[2] - 1e-3 * 0.5).abs() < 1e-10);
        assert!((augmented[3] - 1e-3 * 0.5).abs() < 1e-10);
        assert!((augmented[4] - 1e-3 * 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_regularize_residuals_zero_displacement() {
        let original = vec![1.0];
        let x = vec![2.0, 3.0];
        let x0 = vec![2.0, 3.0]; // Same as x
        let lambda = 1e-3;

        let augmented = regularize_residuals(&original, &x, &x0, lambda);

        // Regularization terms should be zero
        assert_eq!(augmented.len(), 3);
        assert_eq!(augmented[1], 0.0);
        assert_eq!(augmented[2], 0.0);
    }

    #[test]
    fn test_regularize_jacobian() {
        let original = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0), (1, 2, -1.0)];
        let lambda = 1e-3;

        let augmented = regularize_jacobian(&original, 2, 3, lambda);

        // Should have 4 original + 3 diagonal = 7 total
        assert_eq!(augmented.len(), 7);

        // Original entries unchanged
        assert_eq!(augmented[0], (0, 0, 2.0));
        assert_eq!(augmented[1], (0, 1, 1.0));
        assert_eq!(augmented[2], (1, 1, 3.0));
        assert_eq!(augmented[3], (1, 2, -1.0));

        // Diagonal regularization entries
        assert_eq!(augmented[4], (2, 0, lambda));
        assert_eq!(augmented[5], (3, 1, lambda));
        assert_eq!(augmented[6], (4, 2, lambda));
    }

    #[test]
    fn test_regularize_jacobian_empty() {
        let original = vec![];
        let lambda = 1e-3;

        let augmented = regularize_jacobian(&original, 0, 2, lambda);

        // Should have 2 diagonal entries
        assert_eq!(augmented.len(), 2);
        assert_eq!(augmented[0], (0, 0, lambda));
        assert_eq!(augmented[1], (1, 1, lambda));
    }

    #[test]
    fn test_is_underconstrained() {
        // More variables than equations
        assert!(is_underconstrained(2, 3));
        assert!(is_underconstrained(5, 10));

        // Equal (well-constrained)
        assert!(!is_underconstrained(5, 5));

        // More equations than variables (over-constrained)
        assert!(!is_underconstrained(10, 5));
    }

    #[test]
    fn test_degrees_of_freedom() {
        // Under-constrained: positive DOF
        assert_eq!(degrees_of_freedom(2, 5), 3);
        assert_eq!(degrees_of_freedom(0, 3), 3);

        // Well-constrained: zero DOF
        assert_eq!(degrees_of_freedom(5, 5), 0);

        // Over-constrained: negative DOF
        assert_eq!(degrees_of_freedom(10, 5), -5);
        assert_eq!(degrees_of_freedom(3, 0), -3);
    }

    #[test]
    #[should_panic(expected = "x and x_initial must have the same length")]
    fn test_regularize_residuals_mismatched_lengths() {
        let original = vec![1.0];
        let x = vec![1.0, 2.0];
        let x0 = vec![1.0]; // Different length
        regularize_residuals(&original, &x, &x0, 1e-3);
    }

    #[test]
    fn test_regularize_with_default_lambda() {
        let original = vec![0.0];
        let x = vec![1.0];
        let x0 = vec![0.0];

        let augmented = regularize_residuals(&original, &x, &x0, DEFAULT_LAMBDA);

        assert_eq!(augmented.len(), 2);
        assert_eq!(augmented[0], 0.0);
        assert!((augmented[1] - DEFAULT_LAMBDA).abs() < 1e-10);
    }

    #[test]
    fn test_regularization_preserves_convergence() {
        // At the solution (residuals = 0), regularization should be small
        let original = vec![0.0, 0.0]; // Constraints satisfied
        let x = vec![1.0, 2.0];
        let x0 = vec![1.1, 2.1]; // Slight displacement

        let augmented = regularize_residuals(&original, &x, &x0, DEFAULT_LAMBDA);

        // Regularization terms should be tiny
        for i in 2..augmented.len() {
            assert!(augmented[i].abs() < 1e-5);
        }
    }
}
