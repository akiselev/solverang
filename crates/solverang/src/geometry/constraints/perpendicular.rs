//! Perpendicular constraint: two line segments are perpendicular.

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Perpendicular constraint: two line segments are perpendicular.
///
/// Two line segments are perpendicular when their direction vectors have zero dot product.
///
/// Works in any dimension:
/// - d1 = p2 - p1, d2 = p4 - p3
/// - Residual: d1 · d2 = sum(d1[i] * d2[i]) = 0
/// - 1 equation
pub struct PerpendicularConstraint {
    id: ConstraintId,
    line1_start: ParamRange,
    line1_end: ParamRange,
    line2_start: ParamRange,
    line2_end: ParamRange,
    deps: Vec<usize>,
}

impl PerpendicularConstraint {
    /// Create a new perpendicular constraint from two line ParamRanges.
    ///
    /// For a Line2D (count=4): start = [0..2), end = [2..4)
    /// For a Line3D (count=6): start = [0..3), end = [3..6)
    pub fn new(id: ConstraintId, line1: ParamRange, line2: ParamRange) -> Self {
        assert!(
            line1.count == line2.count,
            "Lines must have the same dimension"
        );
        assert!(
            line1.count == 4 || line1.count == 6,
            "Line must be 2D (4 params) or 3D (6 params), got {}",
            line1.count
        );

        let dim = line1.count / 2;
        let line1_start = ParamRange {
            start: line1.start,
            count: dim,
        };
        let line1_end = ParamRange {
            start: line1.start + dim,
            count: dim,
        };
        let line2_start = ParamRange {
            start: line2.start,
            count: dim,
        };
        let line2_end = ParamRange {
            start: line2.start + dim,
            count: dim,
        };

        let mut deps = Vec::new();
        for i in line1_start.iter() {
            deps.push(i);
        }
        for i in line1_end.iter() {
            deps.push(i);
        }
        for i in line2_start.iter() {
            deps.push(i);
        }
        for i in line2_end.iter() {
            deps.push(i);
        }

        Self {
            id,
            line1_start,
            line1_end,
            line2_start,
            line2_end,
            deps,
        }
    }

    /// Create from four point ParamRanges.
    /// Line1 = p1→p2, Line2 = p3→p4
    pub fn from_points(
        id: ConstraintId,
        p1: ParamRange,
        p2: ParamRange,
        p3: ParamRange,
        p4: ParamRange,
    ) -> Self {
        assert_eq!(p1.count, p2.count, "Points must have same dimension");
        assert_eq!(p1.count, p3.count, "Points must have same dimension");
        assert_eq!(p1.count, p4.count, "Points must have same dimension");

        let mut deps = Vec::new();
        for i in p1.iter() {
            deps.push(i);
        }
        for i in p2.iter() {
            deps.push(i);
        }
        for i in p3.iter() {
            deps.push(i);
        }
        for i in p4.iter() {
            deps.push(i);
        }

        Self {
            id,
            line1_start: p1,
            line1_end: p2,
            line2_start: p3,
            line2_end: p4,
            deps,
        }
    }

    fn dimension(&self) -> usize {
        self.line1_start.count
    }
}

impl Constraint for PerpendicularConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "Perpendicular"
    }

    fn equation_count(&self) -> usize {
        1 // One equation for all dimensions
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        let mut dot = 0.0;

        for i in 0..dim {
            let d1_i = params[self.line1_end.start + i] - params[self.line1_start.start + i];
            let d2_i = params[self.line2_end.start + i] - params[self.line2_start.start + i];
            dot += d1_i * d2_i;
        }

        vec![dot]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.dimension();
        let mut jac = Vec::new();

        // f = sum(d1[i] * d2[i])
        // where d1[i] = p2[i] - p1[i], d2[i] = p4[i] - p3[i]
        //
        // df/dp1[i] = -d2[i]
        // df/dp2[i] = d2[i]
        // df/dp3[i] = -d1[i]
        // df/dp4[i] = d1[i]

        for i in 0..dim {
            let d1_i = params[self.line1_end.start + i] - params[self.line1_start.start + i];
            let d2_i = params[self.line2_end.start + i] - params[self.line2_start.start + i];

            jac.push((0, self.line1_start.start + i, -d2_i)); // df/dp1[i]
            jac.push((0, self.line1_end.start + i, d2_i));    // df/dp2[i]
            jac.push((0, self.line2_start.start + i, -d1_i)); // df/dp3[i]
            jac.push((0, self.line2_end.start + i, d1_i));    // df/dp4[i]
        }

        jac
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Moderate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perpendicular_2d_satisfied() {
        // Horizontal and vertical lines
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint = PerpendicularConstraint::new(ConstraintId(0), line1, line2);

        let params = vec![
            0.0, 0.0, // line1 start: (0, 0)
            1.0, 0.0, // line1 end: (1, 0) - horizontal
            5.0, 0.0, // line2 start: (5, 0)
            5.0, 1.0, // line2 end: (5, 1) - vertical
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_perpendicular_2d_unsatisfied() {
        // Two lines with slope 1 - not perpendicular
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint = PerpendicularConstraint::new(ConstraintId(0), line1, line2);

        let params = vec![
            0.0, 0.0, // line1 start: (0, 0)
            1.0, 1.0, // line1 end: (1, 1)
            5.0, 0.0, // line2 start: (5, 0)
            6.0, 1.0, // line2 end: (6, 1)
        ];

        let residuals = constraint.residuals(&params);
        // dot = 1*1 + 1*1 = 2
        assert!((residuals[0] - 2.0).abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_perpendicular_3d_satisfied() {
        // Line along x-axis and line along y-axis
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint = PerpendicularConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);

        let params = vec![
            0.0, 0.0, 0.0, // p1
            1.0, 0.0, 0.0, // p2 - along x
            5.0, 5.0, 5.0, // p3
            5.0, 6.0, 5.0, // p4 - along y
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_perpendicular_3d_unsatisfied() {
        // Two parallel lines along x-axis
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint = PerpendicularConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);

        let params = vec![
            0.0, 0.0, 0.0, // p1
            1.0, 0.0, 0.0, // p2 - along x
            5.0, 5.0, 5.0, // p3
            6.0, 5.0, 5.0, // p4 - also along x
        ];

        let residuals = constraint.residuals(&params);
        // dot = 1*1 + 0*0 + 0*0 = 1
        assert!((residuals[0] - 1.0).abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_equation_count() {
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint_2d = PerpendicularConstraint::new(ConstraintId(0), line1, line2);
        assert_eq!(constraint_2d.equation_count(), 1);

        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint_3d = PerpendicularConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);
        assert_eq!(constraint_3d.equation_count(), 1);
    }

    #[test]
    fn test_dependency_count() {
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint_2d = PerpendicularConstraint::new(ConstraintId(0), line1, line2);
        assert_eq!(constraint_2d.dependencies().len(), 8);

        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint_3d = PerpendicularConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);
        assert_eq!(constraint_3d.dependencies().len(), 12);
    }
}
