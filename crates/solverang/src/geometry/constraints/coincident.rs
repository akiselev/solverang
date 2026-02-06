//! Coincident constraint: p1 = p2

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Constrains two points to be at the same location.
///
/// Equations: p1[i] = p2[i] for each coordinate i
///
/// Works for both 2D and 3D points (determined by ParamRange.count).
pub struct CoincidentConstraint {
    id: ConstraintId,
    p1: ParamRange,
    p2: ParamRange,
    deps: Vec<usize>,
}

impl CoincidentConstraint {
    /// Create a new coincident constraint.
    ///
    /// # Panics
    /// Panics if p1 and p2 have different dimensionality (count).
    pub fn new(id: ConstraintId, p1: ParamRange, p2: ParamRange) -> Self {
        assert_eq!(
            p1.count, p2.count,
            "CoincidentConstraint requires points of same dimensionality"
        );

        let mut deps = Vec::new();
        deps.extend(p1.iter());
        deps.extend(p2.iter());

        Self { id, p1, p2, deps }
    }
}

impl Constraint for CoincidentConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "coincident"
    }

    fn equation_count(&self) -> usize {
        self.p1.count
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let dim = self.p1.count;
        let mut residuals = Vec::with_capacity(dim);

        for i in 0..dim {
            residuals.push(params[self.p1.start + i] - params[self.p2.start + i]);
        }

        residuals
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.p1.count;
        let mut jacobian = Vec::with_capacity(2 * dim);

        // For each equation i: residual[i] = p1[i] - p2[i]
        // d/dp1[i] = 1, d/dp2[i] = -1
        for i in 0..dim {
            jacobian.push((i, self.p1.start + i, 1.0));
            jacobian.push((i, self.p2.start + i, -1.0));
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
    fn test_coincident_satisfied_2d() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = CoincidentConstraint::new(ConstraintId(0), p1, p2);

        // Both points at (1, 2)
        let params = vec![1.0, 2.0, 1.0, 2.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!((residuals[0]).abs() < 1e-10);
        assert!((residuals[1]).abs() < 1e-10);
    }

    #[test]
    fn test_coincident_unsatisfied_2d() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = CoincidentConstraint::new(ConstraintId(0), p1, p2);

        // p1 at (1, 2), p2 at (4, 6)
        let params = vec![1.0, 2.0, 4.0, 6.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!((residuals[0] - (-3.0)).abs() < 1e-10);
        assert!((residuals[1] - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_coincident_3d() {
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let constraint = CoincidentConstraint::new(ConstraintId(0), p1, p2);

        // Both points at (1, 2, 3)
        let params = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 3);
        assert!((residuals[0]).abs() < 1e-10);
        assert!((residuals[1]).abs() < 1e-10);
        assert!((residuals[2]).abs() < 1e-10);
    }

    #[test]
    fn test_equation_count_2d() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = CoincidentConstraint::new(ConstraintId(0), p1, p2);

        assert_eq!(constraint.equation_count(), 2);
    }

    #[test]
    fn test_equation_count_3d() {
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let constraint = CoincidentConstraint::new(ConstraintId(0), p1, p2);

        assert_eq!(constraint.equation_count(), 3);
    }

    #[test]
    fn test_dependencies() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = CoincidentConstraint::new(ConstraintId(0), p1, p2);

        assert_eq!(constraint.dependencies(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_jacobian_2d() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = CoincidentConstraint::new(ConstraintId(0), p1, p2);

        let params = vec![1.0, 2.0, 4.0, 6.0];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 4);

        // Equation 0: p1.x - p2.x
        assert!(jac.contains(&(0, 0, 1.0)));
        assert!(jac.contains(&(0, 2, -1.0)));

        // Equation 1: p1.y - p2.y
        assert!(jac.contains(&(1, 1, 1.0)));
        assert!(jac.contains(&(1, 3, -1.0)));
    }

    #[test]
    fn test_jacobian_3d() {
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let constraint = CoincidentConstraint::new(ConstraintId(0), p1, p2);

        let params = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 6);

        // Verify all entries
        assert!(jac.contains(&(0, 0, 1.0)));
        assert!(jac.contains(&(0, 3, -1.0)));
        assert!(jac.contains(&(1, 1, 1.0)));
        assert!(jac.contains(&(1, 4, -1.0)));
        assert!(jac.contains(&(2, 2, 1.0)));
        assert!(jac.contains(&(2, 5, -1.0)));
    }

    #[test]
    #[should_panic(expected = "same dimensionality")]
    fn test_mismatched_dimensions() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 3 };
        CoincidentConstraint::new(ConstraintId(0), p1, p2);
    }
}
