//! Horizontal constraint: p1.y = p2.y (2D points)

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Constrains two 2D points to have the same y-coordinate.
///
/// Equation: p1.y = p2.y
///
/// This is a 2D-only constraint.
pub struct HorizontalConstraint {
    id: ConstraintId,
    p1: ParamRange,
    p2: ParamRange,
    deps: Vec<usize>,
}

impl HorizontalConstraint {
    /// Create a new horizontal constraint.
    ///
    /// # Panics
    /// Panics if p1 or p2 are not 2D points (count != 2).
    pub fn new(id: ConstraintId, p1: ParamRange, p2: ParamRange) -> Self {
        assert_eq!(p1.count, 2, "HorizontalConstraint requires 2D points");
        assert_eq!(p2.count, 2, "HorizontalConstraint requires 2D points");

        let deps = vec![p1.start + 1, p2.start + 1]; // Only y-coordinates

        Self { id, p1, p2, deps }
    }
}

impl Constraint for HorizontalConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "horizontal"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        // Residual: p1.y - p2.y
        vec![params[self.p1.start + 1] - params[self.p2.start + 1]]
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, self.p1.start + 1, 1.0),  // d/dp1_y = 1
            (0, self.p2.start + 1, -1.0), // d/dp2_y = -1
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
    fn test_horizontal_satisfied() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = HorizontalConstraint::new(ConstraintId(0), p1, p2);

        // Both points have y = 5
        let params = vec![0.0, 5.0, 10.0, 5.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!((residuals[0]).abs() < 1e-10);
    }

    #[test]
    fn test_horizontal_unsatisfied() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = HorizontalConstraint::new(ConstraintId(0), p1, p2);

        // p1.y = 0, p2.y = 5
        let params = vec![0.0, 0.0, 10.0, 5.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!((residuals[0] - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_equation_count() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = HorizontalConstraint::new(ConstraintId(0), p1, p2);

        assert_eq!(constraint.equation_count(), 1);
    }

    #[test]
    fn test_dependencies() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = HorizontalConstraint::new(ConstraintId(0), p1, p2);

        // Only y-coordinates (indices 1 and 3)
        assert_eq!(constraint.dependencies(), &[1, 3]);
    }

    #[test]
    fn test_jacobian() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = HorizontalConstraint::new(ConstraintId(0), p1, p2);

        let params = vec![0.0, 0.0, 10.0, 5.0];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 2);

        // d/dp1.y = 1, d/dp2.y = -1
        assert!(jac.contains(&(0, 1, 1.0)));
        assert!(jac.contains(&(0, 3, -1.0)));
    }

    #[test]
    #[should_panic(expected = "requires 2D points")]
    fn test_3d_points_rejected() {
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        HorizontalConstraint::new(ConstraintId(0), p1, p2);
    }
}
