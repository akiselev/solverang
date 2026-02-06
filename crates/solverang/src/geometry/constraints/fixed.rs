//! Fixed constraint: p = target_values

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Constrains a point to be at a specific fixed location.
///
/// Equations: p[i] = target[i] for each coordinate i
///
/// Works for both 2D and 3D points (determined by ParamRange.count).
pub struct FixedConstraint {
    id: ConstraintId,
    p: ParamRange,
    target_values: Vec<f64>,
    deps: Vec<usize>,
}

impl FixedConstraint {
    /// Create a new fixed constraint.
    ///
    /// # Panics
    /// Panics if target_values.len() != p.count.
    pub fn new(id: ConstraintId, p: ParamRange, target_values: Vec<f64>) -> Self {
        assert_eq!(
            target_values.len(),
            p.count,
            "FixedConstraint: target_values length must match point dimensionality"
        );

        let deps: Vec<usize> = p.iter().collect();

        Self {
            id,
            p,
            target_values,
            deps,
        }
    }
}

impl Constraint for FixedConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "fixed"
    }

    fn equation_count(&self) -> usize {
        self.p.count
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let dim = self.p.count;
        let mut residuals = Vec::with_capacity(dim);

        for i in 0..dim {
            residuals.push(params[self.p.start + i] - self.target_values[i]);
        }

        residuals
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.p.count;
        let mut jacobian = Vec::with_capacity(dim);

        // For each equation i: residual[i] = p[i] - target[i]
        // d/dp[i] = 1
        for i in 0..dim {
            jacobian.push((i, self.p.start + i, 1.0));
        }

        jacobian
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Linear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_satisfied_2d() {
        let p = ParamRange { start: 0, count: 2 };
        let constraint = FixedConstraint::new(ConstraintId(0), p, vec![5.0, 3.0]);

        // Point at (5, 3)
        let params = vec![5.0, 3.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!((residuals[0]).abs() < 1e-10);
        assert!((residuals[1]).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_unsatisfied_2d() {
        let p = ParamRange { start: 0, count: 2 };
        let constraint = FixedConstraint::new(ConstraintId(0), p, vec![10.0, 20.0]);

        // Point at (0, 0) but should be at (10, 20)
        let params = vec![0.0, 0.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!((residuals[0] - (-10.0)).abs() < 1e-10);
        assert!((residuals[1] - (-20.0)).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_3d() {
        let p = ParamRange { start: 0, count: 3 };
        let constraint = FixedConstraint::new(ConstraintId(0), p, vec![1.0, 2.0, 3.0]);

        // Point at (4, 5, 6)
        let params = vec![4.0, 5.0, 6.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 3);
        assert!((residuals[0] - 3.0).abs() < 1e-10);
        assert!((residuals[1] - 3.0).abs() < 1e-10);
        assert!((residuals[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_equation_count() {
        let p = ParamRange { start: 0, count: 2 };
        let constraint = FixedConstraint::new(ConstraintId(0), p, vec![0.0, 0.0]);

        assert_eq!(constraint.equation_count(), 2);
    }

    #[test]
    fn test_dependencies() {
        let p = ParamRange { start: 0, count: 2 };
        let constraint = FixedConstraint::new(ConstraintId(0), p, vec![0.0, 0.0]);

        assert_eq!(constraint.dependencies(), &[0, 1]);
    }

    #[test]
    fn test_jacobian_2d() {
        let p = ParamRange { start: 0, count: 2 };
        let constraint = FixedConstraint::new(ConstraintId(0), p, vec![5.0, 3.0]);

        let params = vec![0.0, 0.0];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 2);

        // d(eq0)/d(p.x) = 1, d(eq1)/d(p.y) = 1
        assert!(jac.contains(&(0, 0, 1.0)));
        assert!(jac.contains(&(1, 1, 1.0)));
    }

    #[test]
    fn test_fixed_with_offset_param_range() {
        // Point stored at indices 5-6 in parameter vector
        let p = ParamRange { start: 5, count: 2 };
        let constraint = FixedConstraint::new(ConstraintId(0), p, vec![10.0, 20.0]);

        // Create a params vector with dummy values, the point at indices 5-6
        let params = vec![0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0];

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);

        let jac = constraint.jacobian(&params);
        // Jacobian should reference columns 5 and 6
        assert!(jac.contains(&(0, 5, 1.0)));
        assert!(jac.contains(&(1, 6, 1.0)));
    }

    #[test]
    #[should_panic(expected = "must match point dimensionality")]
    fn test_mismatched_target_length() {
        let p = ParamRange { start: 0, count: 2 };
        FixedConstraint::new(ConstraintId(0), p, vec![1.0, 2.0, 3.0]);
    }
}
