//! Point on line constraint: a point must lie on a line.

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Point on line constraint: point p lies on line through (ls, le).
///
/// Uses cross product formulation: (p - ls) × (le - ls) = 0
///
/// # 2D
/// - 1 equation: (p-ls).x * (le-ls).y - (p-ls).y * (le-ls).x = 0
///
/// # 3D
/// - 2 equations: two independent components of the cross product
pub struct PointOnLineConstraint {
    id: ConstraintId,
    point: ParamRange,
    line_start: ParamRange,
    line_end: ParamRange,
    deps: Vec<usize>,
}

impl PointOnLineConstraint {
    /// Create a new point-on-line constraint.
    pub fn new(
        id: ConstraintId,
        point: ParamRange,
        line_start: ParamRange,
        line_end: ParamRange,
    ) -> Self {
        assert_eq!(
            point.count, line_start.count,
            "Point and line must have same dimension"
        );
        assert_eq!(
            point.count, line_end.count,
            "Point and line must have same dimension"
        );

        let mut deps = Vec::new();
        for i in point.iter() {
            deps.push(i);
        }
        for i in line_start.iter() {
            deps.push(i);
        }
        for i in line_end.iter() {
            deps.push(i);
        }

        Self {
            id,
            point,
            line_start,
            line_end,
            deps,
        }
    }

    fn dimension(&self) -> usize {
        self.point.count
    }
}

impl Constraint for PointOnLineConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "PointOnLine"
    }

    fn equation_count(&self) -> usize {
        match self.dimension() {
            2 => 1,
            3 => 2,
            _ => panic!("Unsupported dimension"),
        }
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let dim = self.dimension();

        if dim == 2 {
            // 2D case
            let p_x = params[self.point.start];
            let p_y = params[self.point.start + 1];
            let ls_x = params[self.line_start.start];
            let ls_y = params[self.line_start.start + 1];
            let le_x = params[self.line_end.start];
            let le_y = params[self.line_end.start + 1];

            // Vector from line start to point
            let v_x = p_x - ls_x;
            let v_y = p_y - ls_y;

            // Direction vector of line
            let d_x = le_x - ls_x;
            let d_y = le_y - ls_y;

            // 2D cross product: v × d
            let cross = v_x * d_y - v_y * d_x;

            vec![cross]
        } else {
            // 3D case
            let p_x = params[self.point.start];
            let p_y = params[self.point.start + 1];
            let p_z = params[self.point.start + 2];
            let ls_x = params[self.line_start.start];
            let ls_y = params[self.line_start.start + 1];
            let ls_z = params[self.line_start.start + 2];
            let le_x = params[self.line_end.start];
            let le_y = params[self.line_end.start + 1];
            let le_z = params[self.line_end.start + 2];

            let v_x = p_x - ls_x;
            let v_y = p_y - ls_y;
            let v_z = p_z - ls_z;

            let d_x = le_x - ls_x;
            let d_y = le_y - ls_y;
            let d_z = le_z - ls_z;

            // 3D cross product v × d (use 2 independent components)
            let cross_yz = v_y * d_z - v_z * d_y;
            let cross_zx = v_z * d_x - v_x * d_z;

            vec![cross_yz, cross_zx]
        }
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.dimension();

        if dim == 2 {
            // 2D case
            let p_x = params[self.point.start];
            let p_y = params[self.point.start + 1];
            let ls_x = params[self.line_start.start];
            let ls_y = params[self.line_start.start + 1];
            let le_x = params[self.line_end.start];
            let le_y = params[self.line_end.start + 1];

            let _v_x = p_x - ls_x;
            let _v_y = p_y - ls_y;
            let d_x = le_x - ls_x;
            let d_y = le_y - ls_y;

            // f = v_x * d_y - v_y * d_x
            //   = (p_x - ls_x) * (le_y - ls_y) - (p_y - ls_y) * (le_x - ls_x)
            //
            // df/dp_x = d_y = le_y - ls_y
            // df/dp_y = -d_x = -(le_x - ls_x)
            // df/dls_x = -d_y + v_y = -d_y + (p_y - ls_y) = p_y - le_y
            // df/dls_y = d_x - v_x = (le_x - ls_x) - (p_x - ls_x) = le_x - p_x
            // df/dle_x = -v_y = -(p_y - ls_y) = ls_y - p_y
            // df/dle_y = v_x = p_x - ls_x

            vec![
                (0, self.point.start, d_y),             // df/dp_x
                (0, self.point.start + 1, -d_x),        // df/dp_y
                (0, self.line_start.start, p_y - le_y), // df/dls_x
                (0, self.line_start.start + 1, le_x - p_x), // df/dls_y
                (0, self.line_end.start, ls_y - p_y),   // df/dle_x
                (0, self.line_end.start + 1, p_x - ls_x), // df/dle_y
            ]
        } else {
            // 3D case
            let p_x = params[self.point.start];
            let p_y = params[self.point.start + 1];
            let p_z = params[self.point.start + 2];
            let ls_x = params[self.line_start.start];
            let ls_y = params[self.line_start.start + 1];
            let ls_z = params[self.line_start.start + 2];
            let le_x = params[self.line_end.start];
            let le_y = params[self.line_end.start + 1];
            let le_z = params[self.line_end.start + 2];

            let v_x = p_x - ls_x;
            let v_y = p_y - ls_y;
            let v_z = p_z - ls_z;
            let d_x = le_x - ls_x;
            let d_y = le_y - ls_y;
            let d_z = le_z - ls_z;

            let mut jac = Vec::new();

            // Eq0: cross_yz = v_y * d_z - v_z * d_y
            jac.push((0, self.point.start + 1, d_z));
            jac.push((0, self.point.start + 2, -d_y));
            jac.push((0, self.line_start.start + 1, -d_z + v_z));
            jac.push((0, self.line_start.start + 2, d_y - v_y));
            jac.push((0, self.line_end.start + 1, -v_z));
            jac.push((0, self.line_end.start + 2, v_y));

            // Eq1: cross_zx = v_z * d_x - v_x * d_z
            jac.push((1, self.point.start + 2, d_x));
            jac.push((1, self.point.start, -d_z));
            jac.push((1, self.line_start.start + 2, -d_x + v_x));
            jac.push((1, self.line_start.start, d_z - v_z));
            jac.push((1, self.line_end.start + 2, -v_x));
            jac.push((1, self.line_end.start, v_z));

            jac
        }
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Moderate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_on_line_2d_satisfied() {
        // Point (2,2) on line from (0,0) to (4,4)
        let p = ParamRange { start: 0, count: 2 };
        let ls = ParamRange { start: 2, count: 2 };
        let le = ParamRange { start: 4, count: 2 };
        let constraint = PointOnLineConstraint::new(ConstraintId(0), p, ls, le);

        let params = vec![
            2.0, 2.0, // p: (2, 2)
            0.0, 0.0, // ls: (0, 0)
            4.0, 4.0, // le: (4, 4)
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_point_on_line_2d_unsatisfied() {
        // Point (2,3) NOT on line from (0,0) to (4,0)
        let p = ParamRange { start: 0, count: 2 };
        let ls = ParamRange { start: 2, count: 2 };
        let le = ParamRange { start: 4, count: 2 };
        let constraint = PointOnLineConstraint::new(ConstraintId(0), p, ls, le);

        let params = vec![
            2.0, 3.0, // p: (2, 3) - off the line!
            0.0, 0.0, // ls: (0, 0)
            4.0, 0.0, // le: (4, 0) - horizontal line
        ];

        let residuals = constraint.residuals(&params);
        // v = (2, 3), d = (4, 0), cross = 2*0 - 3*4 = -12
        assert!((residuals[0] + 12.0).abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_point_on_line_3d_satisfied() {
        // Point (1,1,1) on line from (0,0,0) to (2,2,2)
        let p = ParamRange { start: 0, count: 3 };
        let ls = ParamRange { start: 3, count: 3 };
        let le = ParamRange { start: 6, count: 3 };
        let constraint = PointOnLineConstraint::new(ConstraintId(0), p, ls, le);

        let params = vec![
            1.0, 1.0, 1.0, // p: (1, 1, 1)
            0.0, 0.0, 0.0, // ls: (0, 0, 0)
            2.0, 2.0, 2.0, // le: (2, 2, 2)
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!(residuals[0].abs() < 1e-10, "residual[0] = {}", residuals[0]);
        assert!(residuals[1].abs() < 1e-10, "residual[1] = {}", residuals[1]);
    }

    #[test]
    fn test_point_on_line_3d_unsatisfied() {
        // Point (1,2,3) NOT on line from (0,0,0) to (1,0,0)
        let p = ParamRange { start: 0, count: 3 };
        let ls = ParamRange { start: 3, count: 3 };
        let le = ParamRange { start: 6, count: 3 };
        let constraint = PointOnLineConstraint::new(ConstraintId(0), p, ls, le);

        let params = vec![
            1.0, 2.0, 3.0, // p: (1, 2, 3)
            0.0, 0.0, 0.0, // ls: (0, 0, 0)
            1.0, 0.0, 0.0, // le: (1, 0, 0) - along x-axis
        ];

        let residuals = constraint.residuals(&params);
        let total = residuals[0].abs() + residuals[1].abs();
        assert!(total > 1.0, "total residual = {}", total);
    }

    #[test]
    fn test_equation_count() {
        let p = ParamRange { start: 0, count: 2 };
        let ls = ParamRange { start: 2, count: 2 };
        let le = ParamRange { start: 4, count: 2 };
        let constraint_2d = PointOnLineConstraint::new(ConstraintId(0), p, ls, le);
        assert_eq!(constraint_2d.equation_count(), 1);

        let p3 = ParamRange { start: 0, count: 3 };
        let ls3 = ParamRange { start: 3, count: 3 };
        let le3 = ParamRange { start: 6, count: 3 };
        let constraint_3d = PointOnLineConstraint::new(ConstraintId(0), p3, ls3, le3);
        assert_eq!(constraint_3d.equation_count(), 2);
    }

    #[test]
    fn test_dependency_count() {
        let p = ParamRange { start: 0, count: 2 };
        let ls = ParamRange { start: 2, count: 2 };
        let le = ParamRange { start: 4, count: 2 };
        let constraint_2d = PointOnLineConstraint::new(ConstraintId(0), p, ls, le);
        assert_eq!(constraint_2d.dependencies().len(), 6); // 3 points × 2 coords

        let p3 = ParamRange { start: 0, count: 3 };
        let ls3 = ParamRange { start: 3, count: 3 };
        let le3 = ParamRange { start: 6, count: 3 };
        let constraint_3d = PointOnLineConstraint::new(ConstraintId(0), p3, ls3, le3);
        assert_eq!(constraint_3d.dependencies().len(), 9); // 3 points × 3 coords
    }
}
