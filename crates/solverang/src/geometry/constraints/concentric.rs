//! Concentric constraint: two circles/arcs share the same center.

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Concentric constraint: center1 = center2.
///
/// Enforces that two circular entities (circles, arcs, ellipses) have the same center point.
/// This is a linear constraint with D equations (where D is the dimension).
///
/// # Entity Layout
/// - `center1`: ParamRange for first center [cx, cy] (2D) or [cx, cy, cz] (3D)
/// - `center2`: ParamRange for second center (same dimension)
///
/// # Equations
///
/// For each coordinate i in 0..D:
/// `params[center1.start + i] - params[center2.start + i] = 0`
///
/// # Jacobian
///
/// For equation i (coordinate i):
/// - `d/d(center1[i]) = 1.0`
/// - `d/d(center2[i]) = -1.0`
#[derive(Clone, Debug)]
pub struct ConcentricConstraint {
    id: ConstraintId,
    /// First center point range.
    center1: ParamRange,
    /// Second center point range.
    center2: ParamRange,
    /// Cached dependency list.
    dependencies: Vec<usize>,
}

impl ConcentricConstraint {
    /// Create a new concentric constraint.
    ///
    /// # Arguments
    /// * `id` - Unique constraint identifier
    /// * `center1` - Parameter range for the first center
    /// * `center2` - Parameter range for the second center
    ///
    /// # Panics
    /// Panics if center1.count != center2.count (dimension mismatch).
    pub fn new(id: ConstraintId, center1: ParamRange, center2: ParamRange) -> Self {
        assert_eq!(
            center1.count, center2.count,
            "Centers must have the same dimension, got {} and {}",
            center1.count, center2.count
        );

        let dim = center1.count;
        assert!(
            dim == 2 || dim == 3,
            "Centers must be 2D or 3D, got {} params",
            dim
        );

        // Build dependency list
        let mut dependencies = Vec::with_capacity(center1.count + center2.count);
        dependencies.extend(center1.iter());
        dependencies.extend(center2.iter());

        Self {
            id,
            center1,
            center2,
            dependencies,
        }
    }
}

impl Constraint for ConcentricConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "Concentric"
    }

    fn equation_count(&self) -> usize {
        self.center1.count
    }

    fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let dim = self.center1.count;
        let mut residuals = Vec::with_capacity(dim);

        for i in 0..dim {
            let c1_i = params[self.center1.start + i];
            let c2_i = params[self.center2.start + i];
            residuals.push(c1_i - c2_i);
        }

        residuals
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.center1.count;
        let mut entries = Vec::with_capacity(dim * 2);

        for i in 0..dim {
            // Equation i: c1[i] - c2[i] = 0
            // d/dc1[i] = 1.0
            entries.push((i, self.center1.start + i, 1.0));
            // d/dc2[i] = -1.0
            entries.push((i, self.center2.start + i, -1.0));
        }

        entries
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Linear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concentric_2d_satisfied() {
        let id = ConstraintId(0);
        let center1 = ParamRange { start: 0, count: 2 };
        let center2 = ParamRange { start: 3, count: 2 };
        let constraint = ConcentricConstraint::new(id, center1, center2);

        // Circle1: [cx=5, cy=10, r=3], Circle2: [cx=5, cy=10, r=7]
        let params = vec![5.0, 10.0, 3.0, 5.0, 10.0, 7.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!(residuals[0].abs() < 1e-10, "x residual = {}", residuals[0]);
        assert!(residuals[1].abs() < 1e-10, "y residual = {}", residuals[1]);
    }

    #[test]
    fn test_concentric_2d_not_satisfied() {
        let id = ConstraintId(0);
        let center1 = ParamRange { start: 0, count: 2 };
        let center2 = ParamRange { start: 3, count: 2 };
        let constraint = ConcentricConstraint::new(id, center1, center2);

        // Circle1: [5, 10, 3], Circle2: [8, 12, 7]
        let params = vec![5.0, 10.0, 3.0, 8.0, 12.0, 7.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        // x: 5 - 8 = -3
        // y: 10 - 12 = -2
        assert!((residuals[0] - (-3.0)).abs() < 1e-10);
        assert!((residuals[1] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_concentric_3d_satisfied() {
        let id = ConstraintId(0);
        let center1 = ParamRange { start: 0, count: 3 };
        let center2 = ParamRange { start: 4, count: 3 };
        let constraint = ConcentricConstraint::new(id, center1, center2);

        // Sphere1: [1, 2, 3, r=5], Sphere2: [1, 2, 3, r=10]
        let params = vec![1.0, 2.0, 3.0, 5.0, 1.0, 2.0, 3.0, 10.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 3);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);
        assert!(residuals[2].abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_2d() {
        let id = ConstraintId(0);
        let center1 = ParamRange { start: 0, count: 2 };
        let center2 = ParamRange { start: 3, count: 2 };
        let constraint = ConcentricConstraint::new(id, center1, center2);

        let params = vec![5.0, 10.0, 3.0, 8.0, 12.0, 7.0];
        let jac = constraint.jacobian(&params);

        assert_eq!(jac.len(), 4); // 2 equations * 2 params each

        // Organize by (row, col)
        let mut found = std::collections::HashMap::new();
        for (row, col, val) in &jac {
            found.insert((*row, *col), *val);
        }

        // Equation 0 (x): c1.x - c2.x = 0
        assert!((found[&(0, 0)] - 1.0).abs() < 1e-10, "d/dc1.x");
        assert!((found[&(0, 3)] - (-1.0)).abs() < 1e-10, "d/dc2.x");

        // Equation 1 (y): c1.y - c2.y = 0
        assert!((found[&(1, 1)] - 1.0).abs() < 1e-10, "d/dc1.y");
        assert!((found[&(1, 4)] - (-1.0)).abs() < 1e-10, "d/dc2.y");
    }

    #[test]
    fn test_jacobian_3d() {
        let id = ConstraintId(0);
        let center1 = ParamRange { start: 0, count: 3 };
        let center2 = ParamRange { start: 4, count: 3 };
        let constraint = ConcentricConstraint::new(id, center1, center2);

        let params = vec![1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 10.0];
        let jac = constraint.jacobian(&params);

        assert_eq!(jac.len(), 6); // 3 equations * 2 params each

        let mut found = std::collections::HashMap::new();
        for (row, col, val) in &jac {
            found.insert((*row, *col), *val);
        }

        // Equation 0 (x)
        assert!((found[&(0, 0)] - 1.0).abs() < 1e-10);
        assert!((found[&(0, 4)] - (-1.0)).abs() < 1e-10);

        // Equation 1 (y)
        assert!((found[&(1, 1)] - 1.0).abs() < 1e-10);
        assert!((found[&(1, 5)] - (-1.0)).abs() < 1e-10);

        // Equation 2 (z)
        assert!((found[&(2, 2)] - 1.0).abs() < 1e-10);
        assert!((found[&(2, 6)] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dependencies() {
        let id = ConstraintId(0);
        let center1 = ParamRange { start: 10, count: 2 };
        let center2 = ParamRange { start: 20, count: 2 };
        let constraint = ConcentricConstraint::new(id, center1, center2);

        let deps = constraint.dependencies();
        assert_eq!(deps, &[10, 11, 20, 21]);
    }

    #[test]
    fn test_equation_count() {
        let id = ConstraintId(0);
        let center1 = ParamRange { start: 0, count: 2 };
        let center2 = ParamRange { start: 2, count: 2 };
        let constraint = ConcentricConstraint::new(id, center1, center2);

        assert_eq!(constraint.equation_count(), 2);
    }

    #[test]
    fn test_name() {
        let id = ConstraintId(0);
        let center1 = ParamRange { start: 0, count: 2 };
        let center2 = ParamRange { start: 2, count: 2 };
        let constraint = ConcentricConstraint::new(id, center1, center2);

        assert_eq!(constraint.name(), "Concentric");
    }

    #[test]
    fn test_nonlinearity() {
        let id = ConstraintId(0);
        let center1 = ParamRange { start: 0, count: 2 };
        let center2 = ParamRange { start: 2, count: 2 };
        let constraint = ConcentricConstraint::new(id, center1, center2);

        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::Linear);
    }

    #[test]
    #[should_panic(expected = "Centers must have the same dimension")]
    fn test_dimension_mismatch() {
        let id = ConstraintId(0);
        let center1 = ParamRange { start: 0, count: 2 };
        let center2 = ParamRange { start: 2, count: 3 };
        ConcentricConstraint::new(id, center1, center2);
    }

    #[test]
    #[should_panic(expected = "Centers must be 2D or 3D")]
    fn test_invalid_dimension() {
        let id = ConstraintId(0);
        let center1 = ParamRange { start: 0, count: 4 };
        let center2 = ParamRange { start: 4, count: 4 };
        ConcentricConstraint::new(id, center1, center2);
    }
}
