//! Midpoint constraint: mid = (start + end) / 2

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Constrains a point to be at the midpoint of two other points.
///
/// Equations: mid[i] = (start[i] + end[i]) / 2 for each coordinate i
///
/// Works for both 2D and 3D points (determined by ParamRange.count).
pub struct MidpointConstraint {
    id: ConstraintId,
    mid: ParamRange,
    start: ParamRange,
    end: ParamRange,
    deps: Vec<usize>,
}

impl MidpointConstraint {
    /// Create a new midpoint constraint.
    ///
    /// # Panics
    /// Panics if mid, start, and end don't all have the same dimensionality.
    pub fn new(
        id: ConstraintId,
        mid: ParamRange,
        start: ParamRange,
        end: ParamRange,
    ) -> Self {
        assert_eq!(
            mid.count, start.count,
            "MidpointConstraint requires all points to have same dimensionality"
        );
        assert_eq!(
            mid.count, end.count,
            "MidpointConstraint requires all points to have same dimensionality"
        );

        let mut deps = Vec::new();
        deps.extend(mid.iter());
        deps.extend(start.iter());
        deps.extend(end.iter());

        Self {
            id,
            mid,
            start,
            end,
            deps,
        }
    }
}

impl Constraint for MidpointConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "midpoint"
    }

    fn equation_count(&self) -> usize {
        self.mid.count
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let dim = self.mid.count;
        let mut residuals = Vec::with_capacity(dim);

        for i in 0..dim {
            let mid_val = params[self.mid.start + i];
            let start_val = params[self.start.start + i];
            let end_val = params[self.end.start + i];

            residuals.push(mid_val - (start_val + end_val) / 2.0);
        }

        residuals
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.mid.count;
        let mut jacobian = Vec::with_capacity(3 * dim);

        // For each equation i: residual[i] = mid[i] - (start[i] + end[i]) / 2
        // d/dmid[i] = 1, d/dstart[i] = -0.5, d/dend[i] = -0.5
        for i in 0..dim {
            jacobian.push((i, self.mid.start + i, 1.0));
            jacobian.push((i, self.start.start + i, -0.5));
            jacobian.push((i, self.end.start + i, -0.5));
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
    fn test_midpoint_satisfied_2d() {
        let mid = ParamRange { start: 0, count: 2 };
        let start = ParamRange { start: 2, count: 2 };
        let end = ParamRange { start: 4, count: 2 };
        let constraint = MidpointConstraint::new(ConstraintId(0), mid, start, end);

        // mid = (5, 5), start = (0, 0), end = (10, 10)
        let params = vec![5.0, 5.0, 0.0, 0.0, 10.0, 10.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!((residuals[0]).abs() < 1e-10);
        assert!((residuals[1]).abs() < 1e-10);
    }

    #[test]
    fn test_midpoint_unsatisfied_2d() {
        let mid = ParamRange { start: 0, count: 2 };
        let start = ParamRange { start: 2, count: 2 };
        let end = ParamRange { start: 4, count: 2 };
        let constraint = MidpointConstraint::new(ConstraintId(0), mid, start, end);

        // mid = (0, 0), start = (0, 0), end = (10, 10)
        // Expected midpoint is (5, 5)
        let params = vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!((residuals[0] - (-5.0)).abs() < 1e-10);
        assert!((residuals[1] - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_midpoint_3d() {
        let mid = ParamRange { start: 0, count: 3 };
        let start = ParamRange { start: 3, count: 3 };
        let end = ParamRange { start: 6, count: 3 };
        let constraint = MidpointConstraint::new(ConstraintId(0), mid, start, end);

        // mid = (5, 5, 5), start = (0, 0, 0), end = (10, 10, 10)
        let params = vec![5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 3);
        for r in &residuals {
            assert!(r.abs() < 1e-10);
        }
    }

    #[test]
    fn test_equation_count() {
        let mid = ParamRange { start: 0, count: 2 };
        let start = ParamRange { start: 2, count: 2 };
        let end = ParamRange { start: 4, count: 2 };
        let constraint = MidpointConstraint::new(ConstraintId(0), mid, start, end);

        assert_eq!(constraint.equation_count(), 2);
    }

    #[test]
    fn test_dependencies() {
        let mid = ParamRange { start: 0, count: 2 };
        let start = ParamRange { start: 2, count: 2 };
        let end = ParamRange { start: 4, count: 2 };
        let constraint = MidpointConstraint::new(ConstraintId(0), mid, start, end);

        assert_eq!(constraint.dependencies(), &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_jacobian_2d() {
        let mid = ParamRange { start: 0, count: 2 };
        let start = ParamRange { start: 2, count: 2 };
        let end = ParamRange { start: 4, count: 2 };
        let constraint = MidpointConstraint::new(ConstraintId(0), mid, start, end);

        let params = vec![5.0, 5.0, 0.0, 0.0, 10.0, 10.0];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 6); // 2 equations * 3 terms each

        // Check expected entries
        let expected = vec![
            (0, 0, 1.0),   // d(eq0)/d(mid.x)
            (0, 2, -0.5),  // d(eq0)/d(start.x)
            (0, 4, -0.5),  // d(eq0)/d(end.x)
            (1, 1, 1.0),   // d(eq1)/d(mid.y)
            (1, 3, -0.5),  // d(eq1)/d(start.y)
            (1, 5, -0.5),  // d(eq1)/d(end.y)
        ];

        for exp in &expected {
            assert!(jac.contains(exp), "Missing entry {:?}", exp);
        }
    }

    #[test]
    #[should_panic(expected = "same dimensionality")]
    fn test_mismatched_dimensions() {
        let mid = ParamRange { start: 0, count: 2 };
        let start = ParamRange { start: 2, count: 2 };
        let end = ParamRange { start: 4, count: 3 };
        MidpointConstraint::new(ConstraintId(0), mid, start, end);
    }
}
