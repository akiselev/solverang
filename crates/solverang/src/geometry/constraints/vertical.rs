//! Vertical constraint: p1.x = p2.x (2D points)

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Constrains two 2D points to have the same x-coordinate.
///
/// Equation: p1.x = p2.x
///
/// This is a 2D-only constraint.
pub struct VerticalConstraint {
    id: ConstraintId,
    p1: ParamRange,
    p2: ParamRange,
    deps: Vec<usize>,
}

impl VerticalConstraint {
    /// Create a new vertical constraint.
    ///
    /// # Panics
    /// Panics if p1 or p2 are not 2D points (count != 2).
    pub fn new(id: ConstraintId, p1: ParamRange, p2: ParamRange) -> Self {
        assert_eq!(p1.count, 2, "VerticalConstraint requires 2D points");
        assert_eq!(p2.count, 2, "VerticalConstraint requires 2D points");

        let deps = vec![p1.start, p2.start]; // Only x-coordinates

        Self { id, p1, p2, deps }
    }
}

impl Constraint for VerticalConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "vertical"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        // Residual: p1.x - p2.x
        vec![params[self.p1.start] - params[self.p2.start]]
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, self.p1.start, 1.0),  // d/dp1_x = 1
            (0, self.p2.start, -1.0), // d/dp2_x = -1
        ]
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Linear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertical_satisfied() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = VerticalConstraint::new(ConstraintId(0), p1, p2);

        // Both points have x = 5
        let params = vec![5.0, 0.0, 5.0, 10.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!((residuals[0]).abs() < 1e-10);
    }

    #[test]
    fn test_vertical_unsatisfied() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = VerticalConstraint::new(ConstraintId(0), p1, p2);

        // p1.x = 0, p2.x = 5
        let params = vec![0.0, 0.0, 5.0, 10.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!((residuals[0] - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_equation_count() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = VerticalConstraint::new(ConstraintId(0), p1, p2);

        assert_eq!(constraint.equation_count(), 1);
    }

    #[test]
    fn test_dependencies() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = VerticalConstraint::new(ConstraintId(0), p1, p2);

        // Only x-coordinates (indices 0 and 2)
        assert_eq!(constraint.dependencies(), &[0, 2]);
    }

    #[test]
    fn test_jacobian() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = VerticalConstraint::new(ConstraintId(0), p1, p2);

        let params = vec![0.0, 0.0, 5.0, 10.0];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 2);

        // d/dp1.x = 1, d/dp2.x = -1
        assert!(jac.contains(&(0, 0, 1.0)));
        assert!(jac.contains(&(0, 2, -1.0)));
    }

    #[test]
    #[should_panic(expected = "requires 2D points")]
    fn test_3d_points_rejected() {
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        VerticalConstraint::new(ConstraintId(0), p1, p2);
    }
}
