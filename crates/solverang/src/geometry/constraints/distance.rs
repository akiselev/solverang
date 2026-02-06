//! Distance constraint: ||p1 - p2|| = target

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity, MIN_EPSILON};

/// Constrains the distance between two points to a target value.
///
/// Equation: sqrt(sum((p1[i] - p2[i])²)) = target
///
/// Works for both 2D and 3D points (determined by ParamRange.count).
pub struct DistanceConstraint {
    id: ConstraintId,
    p1: ParamRange,
    p2: ParamRange,
    target_distance: f64,
    deps: Vec<usize>,
}

impl DistanceConstraint {
    /// Create a new distance constraint.
    ///
    /// # Panics
    /// Panics if p1 and p2 have different dimensionality (count).
    pub fn new(id: ConstraintId, p1: ParamRange, p2: ParamRange, target_distance: f64) -> Self {
        assert_eq!(
            p1.count, p2.count,
            "DistanceConstraint requires points of same dimensionality"
        );

        let mut deps = Vec::new();
        deps.extend(p1.iter());
        deps.extend(p2.iter());

        Self {
            id,
            p1,
            p2,
            target_distance,
            deps,
        }
    }
}

impl Constraint for DistanceConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "distance"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let dim = self.p1.count;

        // Compute squared distance
        let mut dist_sq = 0.0;
        for i in 0..dim {
            let diff = params[self.p1.start + i] - params[self.p2.start + i];
            dist_sq += diff * diff;
        }

        let dist = dist_sq.sqrt();
        vec![dist - self.target_distance]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.p1.count;

        // Compute distance
        let mut dist_sq = 0.0;
        for i in 0..dim {
            let diff = params[self.p1.start + i] - params[self.p2.start + i];
            dist_sq += diff * diff;
        }

        let dist = dist_sq.sqrt().max(MIN_EPSILON);

        let mut jacobian = Vec::with_capacity(2 * dim);

        // Jacobian: d/dp1[i] = (p1[i] - p2[i]) / dist
        //           d/dp2[i] = -(p1[i] - p2[i]) / dist
        for i in 0..dim {
            let diff = params[self.p1.start + i] - params[self.p2.start + i];
            let deriv = diff / dist;

            jacobian.push((0, self.p1.start + i, deriv));
            jacobian.push((0, self.p2.start + i, -deriv));
        }

        jacobian
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Moderate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_satisfied_2d() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, 5.0);

        // Points at distance 5: (0,0) and (3,4)
        let params = vec![0.0, 0.0, 3.0, 4.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!((residuals[0]).abs() < 1e-10, "Expected residual ≈ 0, got {}", residuals[0]);
    }

    #[test]
    fn test_distance_unsatisfied_2d() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, 5.0);

        // Points at distance 10: (0,0) and (10,0)
        let params = vec![0.0, 0.0, 10.0, 0.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!((residuals[0] - 5.0).abs() < 1e-10, "Expected residual = 5.0, got {}", residuals[0]);
    }

    #[test]
    fn test_distance_3d() {
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, 1.0);

        // Points at distance 1: (0,0,0) and (1,0,0)
        let params = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!((residuals[0]).abs() < 1e-10);
    }

    #[test]
    fn test_equation_count() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, 5.0);

        assert_eq!(constraint.equation_count(), 1);
    }

    #[test]
    fn test_dependencies() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, 5.0);

        assert_eq!(constraint.dependencies(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_jacobian_2d() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, 5.0);

        // Points: (0,0) and (3,4) -> distance = 5
        let params = vec![0.0, 0.0, 3.0, 4.0];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 4);

        // d/dx1 = -3/5 = -0.6, d/dy1 = -4/5 = -0.8
        // d/dx2 = 3/5 = 0.6, d/dy2 = 4/5 = 0.8

        // Find entries by column
        let dx1 = jac.iter().find(|&&(_, col, _)| col == 0).unwrap().2;
        let dy1 = jac.iter().find(|&&(_, col, _)| col == 1).unwrap().2;
        let dx2 = jac.iter().find(|&&(_, col, _)| col == 2).unwrap().2;
        let dy2 = jac.iter().find(|&&(_, col, _)| col == 3).unwrap().2;

        assert!((dx1 - (-0.6)).abs() < 1e-10);
        assert!((dy1 - (-0.8)).abs() < 1e-10);
        assert!((dx2 - 0.6).abs() < 1e-10);
        assert!((dy2 - 0.8).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "same dimensionality")]
    fn test_mismatched_dimensions() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 3 };
        DistanceConstraint::new(ConstraintId(0), p1, p2, 5.0);
    }

    #[test]
    fn test_zero_distance_safe() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = DistanceConstraint::new(ConstraintId(0), p1, p2, 0.0);

        // Coincident points - should use MIN_EPSILON for safe division
        let params = vec![1.0, 2.0, 1.0, 2.0];

        let jac = constraint.jacobian(&params);
        // Should not panic and should produce finite values
        for &(_, _, val) in &jac {
            assert!(val.is_finite());
        }
    }
}
