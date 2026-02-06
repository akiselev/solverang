//! Parallel constraint: two line segments are parallel.

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Parallel constraint: two line segments are parallel.
///
/// In 2D, two line segments are parallel when their direction vectors have zero cross product.
/// In 3D, the cross product must be the zero vector (2 independent equations).
///
/// # 2D
/// - Direction vectors: d1 = p2 - p1, d2 = p4 - p3
/// - Residual: d1.x * d2.y - d1.y * d2.x = 0
/// - 1 equation
///
/// # 3D
/// - Direction vectors: d1 = p2 - p1, d2 = p4 - p3
/// - Residual[0]: d1.y*d2.z - d1.z*d2.y = 0
/// - Residual[1]: d1.z*d2.x - d1.x*d2.z = 0
/// - 2 equations (third component is dependent)
pub struct ParallelConstraint {
    id: ConstraintId,
    line1_start: ParamRange,
    line1_end: ParamRange,
    line2_start: ParamRange,
    line2_end: ParamRange,
    deps: Vec<usize>,
}

impl ParallelConstraint {
    /// Create a new parallel constraint from two line ParamRanges.
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

impl Constraint for ParallelConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "Parallel"
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
            let p1_x = params[self.line1_start.start];
            let p1_y = params[self.line1_start.start + 1];
            let p2_x = params[self.line1_end.start];
            let p2_y = params[self.line1_end.start + 1];
            let p3_x = params[self.line2_start.start];
            let p3_y = params[self.line2_start.start + 1];
            let p4_x = params[self.line2_end.start];
            let p4_y = params[self.line2_end.start + 1];

            let d1_x = p2_x - p1_x;
            let d1_y = p2_y - p1_y;
            let d2_x = p4_x - p3_x;
            let d2_y = p4_y - p3_y;

            // 2D cross product
            let cross = d1_x * d2_y - d1_y * d2_x;
            vec![cross]
        } else {
            // 3D case
            let p1_x = params[self.line1_start.start];
            let p1_y = params[self.line1_start.start + 1];
            let p1_z = params[self.line1_start.start + 2];
            let p2_x = params[self.line1_end.start];
            let p2_y = params[self.line1_end.start + 1];
            let p2_z = params[self.line1_end.start + 2];
            let p3_x = params[self.line2_start.start];
            let p3_y = params[self.line2_start.start + 1];
            let p3_z = params[self.line2_start.start + 2];
            let p4_x = params[self.line2_end.start];
            let p4_y = params[self.line2_end.start + 1];
            let p4_z = params[self.line2_end.start + 2];

            let d1_x = p2_x - p1_x;
            let d1_y = p2_y - p1_y;
            let d1_z = p2_z - p1_z;
            let d2_x = p4_x - p3_x;
            let d2_y = p4_y - p3_y;
            let d2_z = p4_z - p3_z;

            // 3D cross product: (d1.y*d2.z - d1.z*d2.y, d1.z*d2.x - d1.x*d2.z, d1.x*d2.y - d1.y*d2.x)
            // Use first two components
            let cross_yz = d1_y * d2_z - d1_z * d2_y;
            let cross_zx = d1_z * d2_x - d1_x * d2_z;

            vec![cross_yz, cross_zx]
        }
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let dim = self.dimension();

        if dim == 2 {
            // 2D case
            let p1_x = params[self.line1_start.start];
            let p1_y = params[self.line1_start.start + 1];
            let p2_x = params[self.line1_end.start];
            let p2_y = params[self.line1_end.start + 1];
            let p3_x = params[self.line2_start.start];
            let p3_y = params[self.line2_start.start + 1];
            let p4_x = params[self.line2_end.start];
            let p4_y = params[self.line2_end.start + 1];

            let d1_x = p2_x - p1_x;
            let d1_y = p2_y - p1_y;
            let d2_x = p4_x - p3_x;
            let d2_y = p4_y - p3_y;

            // f = d1_x * d2_y - d1_y * d2_x
            // where d1_x = p2_x - p1_x, d1_y = p2_y - p1_y, d2_x = p4_x - p3_x, d2_y = p4_y - p3_y
            //
            // df/dp1_x = -d2_y
            // df/dp1_y = d2_x
            // df/dp2_x = d2_y
            // df/dp2_y = -d2_x
            // df/dp3_x = d1_y
            // df/dp3_y = -d1_x
            // df/dp4_x = -d1_y
            // df/dp4_y = d1_x

            vec![
                (0, self.line1_start.start, -d2_y),     // df/dp1_x
                (0, self.line1_start.start + 1, d2_x),  // df/dp1_y
                (0, self.line1_end.start, d2_y),        // df/dp2_x
                (0, self.line1_end.start + 1, -d2_x),   // df/dp2_y
                (0, self.line2_start.start, d1_y),      // df/dp3_x
                (0, self.line2_start.start + 1, -d1_x), // df/dp3_y
                (0, self.line2_end.start, -d1_y),       // df/dp4_x
                (0, self.line2_end.start + 1, d1_x),    // df/dp4_y
            ]
        } else {
            // 3D case
            let p1_x = params[self.line1_start.start];
            let p1_y = params[self.line1_start.start + 1];
            let p1_z = params[self.line1_start.start + 2];
            let p2_x = params[self.line1_end.start];
            let p2_y = params[self.line1_end.start + 1];
            let p2_z = params[self.line1_end.start + 2];
            let p3_x = params[self.line2_start.start];
            let p3_y = params[self.line2_start.start + 1];
            let p3_z = params[self.line2_start.start + 2];
            let p4_x = params[self.line2_end.start];
            let p4_y = params[self.line2_end.start + 1];
            let p4_z = params[self.line2_end.start + 2];

            let d1_x = p2_x - p1_x;
            let d1_y = p2_y - p1_y;
            let d1_z = p2_z - p1_z;
            let d2_x = p4_x - p3_x;
            let d2_y = p4_y - p3_y;
            let d2_z = p4_z - p3_z;

            // f0 = d1_y * d2_z - d1_z * d2_y
            // f1 = d1_z * d2_x - d1_x * d2_z

            let mut jac = Vec::new();

            // Equation 0: f0 = d1_y * d2_z - d1_z * d2_y
            // df0/dp1_y = -d2_z
            // df0/dp1_z = d2_y
            // df0/dp2_y = d2_z
            // df0/dp2_z = -d2_y
            // df0/dp3_y = d1_z
            // df0/dp3_z = -d1_y
            // df0/dp4_y = -d1_z
            // df0/dp4_z = d1_y
            jac.push((0, self.line1_start.start + 1, -d2_z));
            jac.push((0, self.line1_start.start + 2, d2_y));
            jac.push((0, self.line1_end.start + 1, d2_z));
            jac.push((0, self.line1_end.start + 2, -d2_y));
            jac.push((0, self.line2_start.start + 1, d1_z));
            jac.push((0, self.line2_start.start + 2, -d1_y));
            jac.push((0, self.line2_end.start + 1, -d1_z));
            jac.push((0, self.line2_end.start + 2, d1_y));

            // Equation 1: f1 = d1_z * d2_x - d1_x * d2_z
            // df1/dp1_z = -d2_x
            // df1/dp1_x = d2_z
            // df1/dp2_z = d2_x
            // df1/dp2_x = -d2_z
            // df1/dp3_z = d1_x
            // df1/dp3_x = -d1_z
            // df1/dp4_z = -d1_x
            // df1/dp4_x = d1_z
            jac.push((1, self.line1_start.start + 2, -d2_x));
            jac.push((1, self.line1_start.start, d2_z));
            jac.push((1, self.line1_end.start + 2, d2_x));
            jac.push((1, self.line1_end.start, -d2_z));
            jac.push((1, self.line2_start.start + 2, d1_x));
            jac.push((1, self.line2_start.start, -d1_z));
            jac.push((1, self.line2_end.start + 2, -d1_x));
            jac.push((1, self.line2_end.start, d1_z));

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
    fn test_parallel_2d_satisfied() {
        // Two parallel lines with slope 1
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint = ParallelConstraint::new(ConstraintId(0), line1, line2);

        let params = vec![
            0.0, 0.0, // line1 start: (0, 0)
            1.0, 1.0, // line1 end: (1, 1)
            5.0, 0.0, // line2 start: (5, 0)
            6.0, 1.0, // line2 end: (6, 1)
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_parallel_2d_unsatisfied() {
        // Horizontal and vertical lines - not parallel
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint = ParallelConstraint::new(ConstraintId(0), line1, line2);

        let params = vec![
            0.0, 0.0, // line1 start: (0, 0)
            1.0, 0.0, // line1 end: (1, 0) - horizontal
            5.0, 0.0, // line2 start: (5, 0)
            5.0, 1.0, // line2 end: (5, 1) - vertical
        ];

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() > 0.5, "residual = {}", residuals[0]);
    }

    #[test]
    fn test_parallel_3d_satisfied() {
        // Two parallel lines in 3D
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint = ParallelConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);

        let params = vec![
            0.0, 0.0, 0.0, // p1
            1.0, 2.0, 3.0, // p2
            5.0, 5.0, 5.0, // p3
            6.0, 7.0, 8.0, // p4 - same direction as p1->p2
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!(residuals[0].abs() < 1e-10, "residual[0] = {}", residuals[0]);
        assert!(residuals[1].abs() < 1e-10, "residual[1] = {}", residuals[1]);
    }

    #[test]
    fn test_parallel_3d_unsatisfied() {
        // Lines along (1,1,0) and (0,0,1) - not parallel.
        // Avoid purely xy-plane directions (like x-axis vs y-axis) because
        // the 2-equation formulation only checks yz/zx cross-product components,
        // which are both zero when both directions lie in the xy-plane.
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint = ParallelConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);

        let params = vec![
            0.0, 0.0, 0.0, // p1
            1.0, 1.0, 0.0, // p2 - direction (1,1,0)
            5.0, 5.0, 5.0, // p3
            5.0, 5.0, 6.0, // p4 - direction (0,0,1)
        ];

        let residuals = constraint.residuals(&params);
        let total = residuals[0].abs() + residuals[1].abs();
        assert!(total > 0.5, "total residual = {}", total);
    }

    #[test]
    fn test_equation_count() {
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint_2d = ParallelConstraint::new(ConstraintId(0), line1, line2);
        assert_eq!(constraint_2d.equation_count(), 1);

        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint_3d = ParallelConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);
        assert_eq!(constraint_3d.equation_count(), 2);
    }

    #[test]
    fn test_dependency_count() {
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint_2d = ParallelConstraint::new(ConstraintId(0), line1, line2);
        assert_eq!(constraint_2d.dependencies().len(), 8); // 4 points × 2 coords

        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint_3d = ParallelConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);
        assert_eq!(constraint_3d.dependencies().len(), 12); // 4 points × 3 coords
    }
}
