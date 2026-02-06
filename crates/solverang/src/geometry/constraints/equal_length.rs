//! Equal length constraint: two line segments have equal length.

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Equal length constraint: ||p2-p1|| = ||p4-p3||
///
/// This constraint enforces that two line segments have the same length.
/// We compare SQUARED lengths to avoid sqrt (cleaner Jacobian).
///
/// Works in any dimension:
/// - Residual: sum((p2[i]-p1[i])²) - sum((p4[i]-p3[i])²) = 0
/// - 1 equation
pub struct EqualLengthConstraint {
    id: ConstraintId,
    line1_start: ParamRange,
    line1_end: ParamRange,
    line2_start: ParamRange,
    line2_end: ParamRange,
    deps: Vec<usize>,
}

impl EqualLengthConstraint {
    /// Create a new equal length constraint from two line ParamRanges.
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

impl Constraint for EqualLengthConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "EqualLength"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let dim = self.dimension();
        let mut len1_sq = 0.0;
        let mut len2_sq = 0.0;

        for i in 0..dim {
            let d1_i = params[self.line1_end.start + i] - params[self.line1_start.start + i];
            let d2_i = params[self.line2_end.start + i] - params[self.line2_start.start + i];
            len1_sq += d1_i * d1_i;
            len2_sq += d2_i * d2_i;
        }

        vec![len1_sq - len2_sq]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.dimension();
        let mut jac = Vec::new();

        // f = sum((p2[i]-p1[i])²) - sum((p4[i]-p3[i])²)
        //
        // df/dp1[i] = -2*(p2[i]-p1[i])
        // df/dp2[i] = 2*(p2[i]-p1[i])
        // df/dp3[i] = 2*(p4[i]-p3[i])
        // df/dp4[i] = -2*(p4[i]-p3[i])

        for i in 0..dim {
            let d1_i = params[self.line1_end.start + i] - params[self.line1_start.start + i];
            let d2_i = params[self.line2_end.start + i] - params[self.line2_start.start + i];

            jac.push((0, self.line1_start.start + i, -2.0 * d1_i)); // df/dp1[i]
            jac.push((0, self.line1_end.start + i, 2.0 * d1_i));    // df/dp2[i]
            jac.push((0, self.line2_start.start + i, 2.0 * d2_i));  // df/dp3[i]
            jac.push((0, self.line2_end.start + i, -2.0 * d2_i));   // df/dp4[i]
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
    fn test_equal_length_2d_satisfied() {
        // Two lines of equal length
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint = EqualLengthConstraint::new(ConstraintId(0), line1, line2);

        let params = vec![
            0.0, 0.0, // line1 start: (0, 0)
            3.0, 4.0, // line1 end: (3, 4) - length = 5
            10.0, 0.0, // line2 start: (10, 0)
            10.0, 5.0, // line2 end: (10, 5) - length = 5
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_equal_length_2d_unsatisfied() {
        // Two lines of different lengths
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint = EqualLengthConstraint::new(ConstraintId(0), line1, line2);

        let params = vec![
            0.0, 0.0, // line1 start: (0, 0)
            1.0, 0.0, // line1 end: (1, 0) - length = 1
            10.0, 0.0, // line2 start: (10, 0)
            13.0, 0.0, // line2 end: (13, 0) - length = 3
        ];

        let residuals = constraint.residuals(&params);
        // len1_sq = 1, len2_sq = 9, residual = 1 - 9 = -8
        assert!((residuals[0] + 8.0).abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_equal_length_3d_satisfied() {
        // Two lines of equal length in 3D
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint = EqualLengthConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);

        let params = vec![
            0.0, 0.0, 0.0, // p1
            1.0, 0.0, 0.0, // p2 - length = 1
            5.0, 5.0, 5.0, // p3
            5.0, 6.0, 5.0, // p4 - length = 1
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_equal_length_3d_unsatisfied() {
        // Two lines of different lengths
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint = EqualLengthConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);

        let params = vec![
            0.0, 0.0, 0.0, // p1
            1.0, 0.0, 0.0, // p2 - length = 1
            5.0, 5.0, 5.0, // p3
            8.0, 5.0, 5.0, // p4 - length = 3
        ];

        let residuals = constraint.residuals(&params);
        // len1_sq = 1, len2_sq = 9, residual = 1 - 9 = -8
        assert!((residuals[0] + 8.0).abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_equation_count() {
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint_2d = EqualLengthConstraint::new(ConstraintId(0), line1, line2);
        assert_eq!(constraint_2d.equation_count(), 1);

        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint_3d = EqualLengthConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);
        assert_eq!(constraint_3d.equation_count(), 1);
    }

    #[test]
    fn test_dependency_count() {
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint_2d = EqualLengthConstraint::new(ConstraintId(0), line1, line2);
        assert_eq!(constraint_2d.dependencies().len(), 8);

        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint_3d = EqualLengthConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);
        assert_eq!(constraint_3d.dependencies().len(), 12);
    }
}
