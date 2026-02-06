use super::params::ConstraintId;

/// Minimum epsilon for safe division and numerical stability.
pub const MIN_EPSILON: f64 = 1e-10;

/// Classification of constraint nonlinearity.
/// Helps the solver selection heuristic choose the appropriate algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nonlinearity {
    /// Linear constraint (e.g., horizontal, vertical, coincident).
    /// Can be solved in one Newton-Raphson step.
    Linear,

    /// Quadratic or mildly nonlinear (e.g., distance, point-on-circle).
    /// Usually converges well with Newton-Raphson.
    Moderate,

    /// Highly nonlinear (e.g., tangent, curvature continuity, angle).
    /// May require Levenberg-Marquardt or careful initialization.
    High,
}

/// A geometric constraint operating on the flat parameter vector.
///
/// Constraints receive the full parameter store as `&[f64]` and know which
/// indices they depend on. They produce scalar equations (residuals) and
/// their Jacobian entries in sparse format.
pub trait Constraint: Send + Sync {
    /// Unique identifier for this constraint instance.
    fn id(&self) -> ConstraintId;

    /// Human-readable name for this constraint type.
    fn name(&self) -> &'static str;

    /// Number of scalar equations this constraint produces.
    fn equation_count(&self) -> usize;

    /// Which parameter indices this constraint depends on.
    /// Used for graph construction and decomposition.
    fn dependencies(&self) -> &[usize];

    /// Evaluate residuals given the full parameter vector.
    /// Must return exactly `equation_count()` values.
    ///
    /// # Arguments
    /// * `params` - The full parameter vector from ParameterStore
    ///
    /// # Returns
    /// Vector of residual values. When the constraint is satisfied,
    /// all residuals should be zero (or near-zero).
    fn residuals(&self, params: &[f64]) -> Vec<f64>;

    /// Compute sparse Jacobian entries.
    ///
    /// # Arguments
    /// * `params` - The full parameter vector from ParameterStore
    ///
    /// # Returns
    /// Vector of (row, col, value) tuples where:
    /// - `row` is the local equation index (0..equation_count())
    /// - `col` is the global parameter index
    /// - `value` is the partial derivative ∂residual[row]/∂params[col]
    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)>;

    /// Whether this is a soft/preference constraint.
    /// Soft constraints are minimized rather than zeroed.
    /// Default: false (hard constraint).
    fn is_soft(&self) -> bool {
        false
    }

    /// Priority weight for weighted least-squares.
    /// Higher weight = higher priority.
    /// Default: 1.0.
    fn weight(&self) -> f64 {
        1.0
    }

    /// Hint about the nonlinearity of this constraint.
    /// Helps the solver selection heuristic.
    /// Default: Moderate.
    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Moderate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nonlinearity_variants() {
        let linear = Nonlinearity::Linear;
        let moderate = Nonlinearity::Moderate;
        let high = Nonlinearity::High;

        assert_eq!(linear, Nonlinearity::Linear);
        assert_eq!(moderate, Nonlinearity::Moderate);
        assert_eq!(high, Nonlinearity::High);

        assert_ne!(linear, moderate);
        assert_ne!(moderate, high);
    }

    #[test]
    fn test_min_epsilon_value() {
        assert_eq!(MIN_EPSILON, 1e-10);
        assert!(MIN_EPSILON > 0.0);
        assert!(MIN_EPSILON < 1e-9);
    }

    #[test]
    fn test_min_epsilon_usage() {
        // Test that MIN_EPSILON can be used for safe division
        let denominator: f64 = 1e-12;
        let safe_denom = if denominator.abs() < MIN_EPSILON {
            MIN_EPSILON
        } else {
            denominator
        };

        assert_eq!(safe_denom, MIN_EPSILON);

        let large_denom: f64 = 5.0;
        let safe_large = if large_denom.abs() < MIN_EPSILON {
            MIN_EPSILON
        } else {
            large_denom
        };

        assert_eq!(safe_large, 5.0);
    }

    // Mock constraint for testing trait default methods
    struct MockConstraint {
        id: ConstraintId,
        deps: Vec<usize>,
    }

    impl Constraint for MockConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }

        fn name(&self) -> &'static str {
            "MockConstraint"
        }

        fn equation_count(&self) -> usize {
            1
        }

        fn dependencies(&self) -> &[usize] {
            &self.deps
        }

        fn residuals(&self, _params: &[f64]) -> Vec<f64> {
            vec![0.0]
        }

        fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![]
        }
    }

    #[test]
    fn test_constraint_trait_defaults() {
        let constraint = MockConstraint {
            id: ConstraintId(42),
            deps: vec![0, 1, 2],
        };

        // Test default implementations
        assert!(!constraint.is_soft());
        assert_eq!(constraint.weight(), 1.0);
        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::Moderate);

        // Test required methods
        assert_eq!(constraint.id(), ConstraintId(42));
        assert_eq!(constraint.name(), "MockConstraint");
        assert_eq!(constraint.equation_count(), 1);
        assert_eq!(constraint.dependencies(), &[0, 1, 2]);
    }

    #[test]
    fn test_nonlinearity_clone_copy() {
        let n = Nonlinearity::High;
        let n2 = n; // Copy
        assert_eq!(n, n2);
    }

    #[test]
    fn test_nonlinearity_debug() {
        // Test Debug trait
        let s = format!("{:?}", Nonlinearity::Linear);
        assert!(s.contains("Linear"));
    }
}
