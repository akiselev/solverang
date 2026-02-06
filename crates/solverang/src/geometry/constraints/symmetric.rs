//! Symmetric constraints: point symmetry and line symmetry

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Constrains two points to be symmetric about a center point.
///
/// Equations: p1[i] + p2[i] = 2 * center[i] for each coordinate i
///
/// Works for both 2D and 3D points (determined by ParamRange.count).
pub struct SymmetricConstraint {
    id: ConstraintId,
    p1: ParamRange,
    p2: ParamRange,
    center: ParamRange,
    deps: Vec<usize>,
}

impl SymmetricConstraint {
    /// Create a new symmetric constraint.
    ///
    /// # Panics
    /// Panics if p1, p2, and center don't all have the same dimensionality.
    pub fn new(
        id: ConstraintId,
        p1: ParamRange,
        p2: ParamRange,
        center: ParamRange,
    ) -> Self {
        assert_eq!(
            p1.count, p2.count,
            "SymmetricConstraint requires all points to have same dimensionality"
        );
        assert_eq!(
            p1.count, center.count,
            "SymmetricConstraint requires all points to have same dimensionality"
        );

        let mut deps = Vec::new();
        deps.extend(p1.iter());
        deps.extend(p2.iter());
        deps.extend(center.iter());

        Self {
            id,
            p1,
            p2,
            center,
            deps,
        }
    }
}

impl Constraint for SymmetricConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "symmetric"
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
            let p1_val = params[self.p1.start + i];
            let p2_val = params[self.p2.start + i];
            let center_val = params[self.center.start + i];

            residuals.push(p1_val + p2_val - 2.0 * center_val);
        }

        residuals
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.p1.count;
        let mut jacobian = Vec::with_capacity(3 * dim);

        // For each equation i: residual[i] = p1[i] + p2[i] - 2*center[i]
        // d/dp1[i] = 1, d/dp2[i] = 1, d/dcenter[i] = -2
        for i in 0..dim {
            jacobian.push((i, self.p1.start + i, 1.0));
            jacobian.push((i, self.p2.start + i, 1.0));
            jacobian.push((i, self.center.start + i, -2.0));
        }

        jacobian
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Linear
    }
}

/// Constrains two points to be symmetric about a line axis (2D only).
///
/// Two conditions must be satisfied:
/// 1. Midpoint of P1-P2 lies on the axis
/// 2. Line P1-P2 is perpendicular to axis
pub struct SymmetricAboutLineConstraint {
    id: ConstraintId,
    p1: ParamRange,
    p2: ParamRange,
    axis_start: ParamRange,
    axis_end: ParamRange,
    deps: Vec<usize>,
}

impl SymmetricAboutLineConstraint {
    /// Create a new symmetric-about-line constraint.
    ///
    /// # Panics
    /// Panics if any point is not 2D (count != 2).
    pub fn new(
        id: ConstraintId,
        p1: ParamRange,
        p2: ParamRange,
        axis_start: ParamRange,
        axis_end: ParamRange,
    ) -> Self {
        assert_eq!(p1.count, 2, "SymmetricAboutLineConstraint requires 2D points");
        assert_eq!(p2.count, 2, "SymmetricAboutLineConstraint requires 2D points");
        assert_eq!(axis_start.count, 2, "SymmetricAboutLineConstraint requires 2D points");
        assert_eq!(axis_end.count, 2, "SymmetricAboutLineConstraint requires 2D points");

        let mut deps = Vec::new();
        deps.extend(p1.iter());
        deps.extend(p2.iter());
        deps.extend(axis_start.iter());
        deps.extend(axis_end.iter());

        Self {
            id,
            p1,
            p2,
            axis_start,
            axis_end,
            deps,
        }
    }
}

impl Constraint for SymmetricAboutLineConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "symmetric_about_line"
    }

    fn equation_count(&self) -> usize {
        2
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        // Extract point coordinates
        let p1x = params[self.p1.start];
        let p1y = params[self.p1.start + 1];
        let p2x = params[self.p2.start];
        let p2y = params[self.p2.start + 1];
        let ax = params[self.axis_start.start];
        let ay = params[self.axis_start.start + 1];
        let bx = params[self.axis_end.start];
        let by = params[self.axis_end.start + 1];

        // Midpoint of P1-P2
        let mx = (p1x + p2x) / 2.0;
        let my = (p1y + p2y) / 2.0;

        // Axis direction
        let ax_dx = bx - ax;
        let ax_dy = by - ay;

        // P1-P2 direction
        let p_dx = p2x - p1x;
        let p_dy = p2y - p1y;

        // Equation 1: Midpoint on axis (cross product = 0)
        // (M - A) × (B - A) = 0
        let mid_cross = (mx - ax) * ax_dy - (my - ay) * ax_dx;

        // Equation 2: P1-P2 perpendicular to axis (dot product = 0)
        // (P2 - P1) · (B - A) = 0
        let perp_dot = p_dx * ax_dx + p_dy * ax_dy;

        vec![mid_cross, perp_dot]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        // Extract point coordinates
        let p1x = params[self.p1.start];
        let p1y = params[self.p1.start + 1];
        let p2x = params[self.p2.start];
        let p2y = params[self.p2.start + 1];
        let ax = params[self.axis_start.start];
        let ay = params[self.axis_start.start + 1];
        let bx = params[self.axis_end.start];
        let by = params[self.axis_end.start + 1];

        // Midpoint
        let mx = (p1x + p2x) / 2.0;
        let my = (p1y + p2y) / 2.0;

        // Axis direction
        let ax_dx = bx - ax;
        let ax_dy = by - ay;

        // P1-P2 direction
        let p_dx = p2x - p1x;
        let p_dy = p2y - p1y;

        // Equation 1: mid_cross = (Mx - Ax)(By - Ay) - (My - Ay)(Bx - Ax)
        // where Mx = (P1x + P2x)/2, My = (P1y + P2y)/2
        //
        // d/dP1x = 0.5 * ax_dy, d/dP1y = -0.5 * ax_dx
        // d/dP2x = 0.5 * ax_dy, d/dP2y = -0.5 * ax_dx
        // d/dAx = -ax_dy + (My - Ay) = My - By (simplified)
        // d/dAy = ax_dx - (Mx - Ax) = Bx - Mx
        // d/dBx = -(My - Ay)
        // d/dBy = (Mx - Ax)

        // Equation 2: perp_dot = (P2x - P1x)(Bx - Ax) + (P2y - P1y)(By - Ay)
        // d/dP1x = -ax_dx, d/dP1y = -ax_dy
        // d/dP2x = ax_dx,  d/dP2y = ax_dy
        // d/dAx = -p_dx,   d/dAy = -p_dy
        // d/dBx = p_dx,    d/dBy = p_dy

        vec![
            // Equation 1: midpoint on axis
            (0, self.p1.start, 0.5 * ax_dy),
            (0, self.p1.start + 1, -0.5 * ax_dx),
            (0, self.p2.start, 0.5 * ax_dy),
            (0, self.p2.start + 1, -0.5 * ax_dx),
            (0, self.axis_start.start, my - by),
            (0, self.axis_start.start + 1, bx - mx),
            (0, self.axis_end.start, -(my - ay)),
            (0, self.axis_end.start + 1, mx - ax),
            // Equation 2: P1-P2 perpendicular to axis
            (1, self.p1.start, -ax_dx),
            (1, self.p1.start + 1, -ax_dy),
            (1, self.p2.start, ax_dx),
            (1, self.p2.start + 1, ax_dy),
            (1, self.axis_start.start, -p_dx),
            (1, self.axis_start.start + 1, -p_dy),
            (1, self.axis_end.start, p_dx),
            (1, self.axis_end.start + 1, p_dy),
        ]
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Moderate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_satisfied_2d() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let center = ParamRange { start: 4, count: 2 };
        let constraint = SymmetricConstraint::new(ConstraintId(0), p1, p2, center);

        // p1 = (0, 0), p2 = (10, 10), center = (5, 5)
        let params = vec![0.0, 0.0, 10.0, 10.0, 5.0, 5.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!((residuals[0]).abs() < 1e-10);
        assert!((residuals[1]).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_unsatisfied_2d() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let center = ParamRange { start: 4, count: 2 };
        let constraint = SymmetricConstraint::new(ConstraintId(0), p1, p2, center);

        // p1 = (0, 0), p2 = (10, 10), center = (0, 0) (not at midpoint)
        // Expected: p1 + p2 - 2*center = 10, 10
        let params = vec![0.0, 0.0, 10.0, 10.0, 0.0, 0.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!((residuals[0] - 10.0).abs() < 1e-10);
        assert!((residuals[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_3d() {
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let center = ParamRange { start: 6, count: 3 };
        let constraint = SymmetricConstraint::new(ConstraintId(0), p1, p2, center);

        // p1 = (0, 0, 0), p2 = (10, 10, 10), center = (5, 5, 5)
        let params = vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 3);
        for r in &residuals {
            assert!(r.abs() < 1e-10);
        }
    }

    #[test]
    fn test_symmetric_equation_count() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let center = ParamRange { start: 4, count: 2 };
        let constraint = SymmetricConstraint::new(ConstraintId(0), p1, p2, center);

        assert_eq!(constraint.equation_count(), 2);
    }

    #[test]
    fn test_symmetric_dependencies() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let center = ParamRange { start: 4, count: 2 };
        let constraint = SymmetricConstraint::new(ConstraintId(0), p1, p2, center);

        assert_eq!(constraint.dependencies(), &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_symmetric_jacobian() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let center = ParamRange { start: 4, count: 2 };
        let constraint = SymmetricConstraint::new(ConstraintId(0), p1, p2, center);

        let params = vec![0.0, 0.0, 10.0, 10.0, 5.0, 5.0];

        let jac = constraint.jacobian(&params);
        assert_eq!(jac.len(), 6); // 2 equations * 3 terms each

        let expected = vec![
            (0, 0, 1.0),   // d(eq0)/d(p1.x)
            (0, 2, 1.0),   // d(eq0)/d(p2.x)
            (0, 4, -2.0),  // d(eq0)/d(center.x)
            (1, 1, 1.0),   // d(eq1)/d(p1.y)
            (1, 3, 1.0),   // d(eq1)/d(p2.y)
            (1, 5, -2.0),  // d(eq1)/d(center.y)
        ];

        for exp in &expected {
            assert!(jac.contains(exp), "Missing entry {:?}", exp);
        }
    }

    #[test]
    fn test_symmetric_about_line_satisfied() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let axis_start = ParamRange { start: 4, count: 2 };
        let axis_end = ParamRange { start: 6, count: 2 };
        let constraint = SymmetricAboutLineConstraint::new(
            ConstraintId(0),
            p1,
            p2,
            axis_start,
            axis_end,
        );

        // p1 = (-1, 0), p2 = (1, 0), axis: y-axis from (0, -5) to (0, 5)
        let params = vec![-1.0, 0.0, 1.0, 0.0, 0.0, -5.0, 0.0, 5.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!((residuals[0]).abs() < 1e-10, "midpoint residual: {}", residuals[0]);
        assert!((residuals[1]).abs() < 1e-10, "perpendicular residual: {}", residuals[1]);
    }

    #[test]
    fn test_symmetric_about_line_diagonal() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let axis_start = ParamRange { start: 4, count: 2 };
        let axis_end = ParamRange { start: 6, count: 2 };
        let constraint = SymmetricAboutLineConstraint::new(
            ConstraintId(0),
            p1,
            p2,
            axis_start,
            axis_end,
        );

        // p1 = (0, 1), p2 = (1, 0), axis: y=x from (0, 0) to (1, 1)
        let params = vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!((residuals[0]).abs() < 1e-10, "midpoint residual: {}", residuals[0]);
        assert!((residuals[1]).abs() < 1e-10, "perpendicular residual: {}", residuals[1]);
    }

    #[test]
    fn test_symmetric_about_line_equation_count() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let axis_start = ParamRange { start: 4, count: 2 };
        let axis_end = ParamRange { start: 6, count: 2 };
        let constraint = SymmetricAboutLineConstraint::new(
            ConstraintId(0),
            p1,
            p2,
            axis_start,
            axis_end,
        );

        assert_eq!(constraint.equation_count(), 2);
    }

    #[test]
    fn test_symmetric_about_line_dependencies() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let axis_start = ParamRange { start: 4, count: 2 };
        let axis_end = ParamRange { start: 6, count: 2 };
        let constraint = SymmetricAboutLineConstraint::new(
            ConstraintId(0),
            p1,
            p2,
            axis_start,
            axis_end,
        );

        assert_eq!(constraint.dependencies(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    #[should_panic(expected = "same dimensionality")]
    fn test_symmetric_mismatched_dimensions() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let center = ParamRange { start: 4, count: 3 };
        SymmetricConstraint::new(ConstraintId(0), p1, p2, center);
    }

    #[test]
    #[should_panic(expected = "requires 2D points")]
    fn test_symmetric_about_line_3d_rejected() {
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let axis_start = ParamRange { start: 6, count: 3 };
        let axis_end = ParamRange { start: 9, count: 3 };
        SymmetricAboutLineConstraint::new(ConstraintId(0), p1, p2, axis_start, axis_end);
    }
}
