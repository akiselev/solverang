//! Angle constraint: line makes a specific angle with horizontal (2D only).

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Angle constraint: line from p1 to p2 makes target angle with horizontal.
///
/// This is a 2D-only constraint.
///
/// Uses the formulation: dy*cos(θ) - dx*sin(θ) = 0
/// where dx = p2.x - p1.x, dy = p2.y - p1.y, θ = target angle
///
/// This avoids atan2 discontinuities.
///
/// 1 equation
pub struct AngleConstraint {
    id: ConstraintId,
    line_start: ParamRange,
    line_end: ParamRange,
    target_radians: f64,
    deps: Vec<usize>,
}

impl AngleConstraint {
    /// Create a new angle constraint with target angle in radians.
    pub fn new(id: ConstraintId, p1: ParamRange, p2: ParamRange, target_radians: f64) -> Self {
        assert_eq!(p1.count, 2, "Angle constraint is 2D only");
        assert_eq!(p2.count, 2, "Angle constraint is 2D only");

        let mut deps = Vec::new();
        for i in p1.iter() {
            deps.push(i);
        }
        for i in p2.iter() {
            deps.push(i);
        }

        Self {
            id,
            line_start: p1,
            line_end: p2,
            target_radians,
            deps,
        }
    }

    /// Create a new angle constraint with target angle in degrees.
    pub fn from_degrees(id: ConstraintId, p1: ParamRange, p2: ParamRange, degrees: f64) -> Self {
        Self::new(id, p1, p2, degrees.to_radians())
    }
}

impl Constraint for AngleConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "Angle"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let p1_x = params[self.line_start.start];
        let p1_y = params[self.line_start.start + 1];
        let p2_x = params[self.line_end.start];
        let p2_y = params[self.line_end.start + 1];

        let dx = p2_x - p1_x;
        let dy = p2_y - p1_y;

        let cos_target = self.target_radians.cos();
        let sin_target = self.target_radians.sin();

        // dy*cos(θ) - dx*sin(θ) = 0
        let residual = dy * cos_target - dx * sin_target;

        vec![residual]
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        let cos_target = self.target_radians.cos();
        let sin_target = self.target_radians.sin();

        // f = dy*cos(θ) - dx*sin(θ)
        // where dx = p2_x - p1_x, dy = p2_y - p1_y
        //
        // df/dp1_x = sin(θ)
        // df/dp1_y = -cos(θ)
        // df/dp2_x = -sin(θ)
        // df/dp2_y = cos(θ)

        vec![
            (0, self.line_start.start, sin_target),      // df/dp1_x
            (0, self.line_start.start + 1, -cos_target), // df/dp1_y
            (0, self.line_end.start, -sin_target),       // df/dp2_x
            (0, self.line_end.start + 1, cos_target),    // df/dp2_y
        ]
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::High
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_angle_horizontal() {
        // Line from (0,0) to (1,0) should be at angle 0
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = AngleConstraint::new(ConstraintId(0), p1, p2, 0.0);

        let params = vec![
            0.0, 0.0, // p1: (0, 0)
            1.0, 0.0, // p2: (1, 0)
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_angle_45_degrees() {
        // Line from (0,0) to (1,1) should be at 45 degrees
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = AngleConstraint::from_degrees(ConstraintId(0), p1, p2, 45.0);

        let params = vec![
            0.0, 0.0, // p1: (0, 0)
            1.0, 1.0, // p2: (1, 1)
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_angle_90_degrees() {
        // Line from (0,0) to (0,1) should be at 90 degrees
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = AngleConstraint::new(ConstraintId(0), p1, p2, PI / 2.0);

        let params = vec![
            0.0, 0.0, // p1: (0, 0)
            0.0, 1.0, // p2: (0, 1)
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_angle_unsatisfied() {
        // Line at 0 degrees but constraint expects 90 degrees
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = AngleConstraint::new(ConstraintId(0), p1, p2, PI / 2.0);

        let params = vec![
            0.0, 0.0, // p1: (0, 0)
            1.0, 0.0, // p2: (1, 0) - horizontal
        ];

        let residuals = constraint.residuals(&params);
        // dy=0, dx=1, cos(90)=0, sin(90)=1
        // residual = 0*0 - 1*1 = -1
        assert!((residuals[0] + 1.0).abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_angle_180_degrees() {
        // Line from (0,0) to (-1,0) should be at 180 degrees
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = AngleConstraint::new(ConstraintId(0), p1, p2, PI);

        let params = vec![
            0.0, 0.0,  // p1: (0, 0)
            -1.0, 0.0, // p2: (-1, 0)
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_equation_count() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = AngleConstraint::new(ConstraintId(0), p1, p2, 0.0);
        assert_eq!(constraint.equation_count(), 1);
    }

    #[test]
    fn test_dependency_count() {
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let constraint = AngleConstraint::new(ConstraintId(0), p1, p2, 0.0);
        assert_eq!(constraint.dependencies().len(), 4); // 2 points × 2 coords
    }
}
