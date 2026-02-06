//! Equal radius constraint: two circles/arcs have the same radius.

use crate::geometry::params::ConstraintId;
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Equal radius constraint: r1 = r2.
///
/// This is a simple linear constraint between two radius parameters.
/// Typically used to enforce that two circles or arcs have the same size.
///
/// # Equation
///
/// `params[r1] - params[r2] = 0`
///
/// # Jacobian
///
/// - `d/dr1 = 1.0`
/// - `d/dr2 = -1.0`
#[derive(Clone, Debug)]
pub struct EqualRadiusConstraint {
    id: ConstraintId,
    /// Index of the first radius parameter.
    r1_idx: usize,
    /// Index of the second radius parameter.
    r2_idx: usize,
    /// Cached dependency list.
    dependencies: Vec<usize>,
}

impl EqualRadiusConstraint {
    /// Create a new equal radius constraint.
    ///
    /// # Arguments
    /// * `id` - Unique constraint identifier
    /// * `r1_idx` - Global parameter index of the first radius
    /// * `r2_idx` - Global parameter index of the second radius
    pub fn new(id: ConstraintId, r1_idx: usize, r2_idx: usize) -> Self {
        let dependencies = vec![r1_idx, r2_idx];

        Self {
            id,
            r1_idx,
            r2_idx,
            dependencies,
        }
    }
}

impl Constraint for EqualRadiusConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "EqualRadius"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let r1 = params[self.r1_idx];
        let r2 = params[self.r2_idx];
        vec![r1 - r2]
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, self.r1_idx, 1.0),
            (0, self.r2_idx, -1.0),
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
    fn test_equal_radius_satisfied() {
        let id = ConstraintId(0);
        let constraint = EqualRadiusConstraint::new(id, 2, 5);

        // params: [cx1, cy1, r1, cx2, cy2, r2]
        // r1 = params[2] = 3.0, r2 = params[5] = 3.0
        let params = vec![0.0, 0.0, 3.0, 10.0, 10.0, 3.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_equal_radius_not_satisfied() {
        let id = ConstraintId(0);
        let constraint = EqualRadiusConstraint::new(id, 2, 5);

        // r1 = 3.0, r2 = 5.0
        let params = vec![0.0, 0.0, 3.0, 10.0, 10.0, 5.0];

        let residuals = constraint.residuals(&params);
        // residual = 3.0 - 5.0 = -2.0
        assert!((residuals[0] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_jacobian() {
        let id = ConstraintId(0);
        let constraint = EqualRadiusConstraint::new(id, 2, 5);

        let params = vec![0.0, 0.0, 3.0, 10.0, 10.0, 5.0];
        let jac = constraint.jacobian(&params);

        assert_eq!(jac.len(), 2);

        let mut found = std::collections::HashMap::new();
        for (row, col, val) in &jac {
            assert_eq!(*row, 0);
            found.insert(*col, *val);
        }

        assert_eq!(found.len(), 2);
        assert!((found[&2] - 1.0).abs() < 1e-10, "d/dr1");
        assert!((found[&5] - (-1.0)).abs() < 1e-10, "d/dr2");
    }

    #[test]
    fn test_dependencies() {
        let id = ConstraintId(0);
        let constraint = EqualRadiusConstraint::new(id, 10, 20);

        let deps = constraint.dependencies();
        assert_eq!(deps, &[10, 20]);
    }

    #[test]
    fn test_equation_count() {
        let id = ConstraintId(0);
        let constraint = EqualRadiusConstraint::new(id, 0, 1);

        assert_eq!(constraint.equation_count(), 1);
    }

    #[test]
    fn test_name() {
        let id = ConstraintId(0);
        let constraint = EqualRadiusConstraint::new(id, 0, 1);

        assert_eq!(constraint.name(), "EqualRadius");
    }

    #[test]
    fn test_nonlinearity() {
        let id = ConstraintId(0);
        let constraint = EqualRadiusConstraint::new(id, 0, 1);

        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::Linear);
    }

    #[test]
    fn test_zero_radius() {
        let id = ConstraintId(0);
        let constraint = EqualRadiusConstraint::new(id, 0, 1);

        // Both radii zero
        let params = vec![0.0, 0.0];

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_negative_radius() {
        // The constraint doesn't prevent negative radii - that's the job
        // of a ParamRange constraint or the solver bounds
        let id = ConstraintId(0);
        let constraint = EqualRadiusConstraint::new(id, 0, 1);

        let params = vec![-5.0, -5.0];

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10);
    }
}
