//! Collinear constraint: two line segments lie on the same line.

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Collinear constraint: line segments (p1,p2) and (p3,p4) lie on the same line.
///
/// Two line segments are collinear when:
/// - The direction p2-p1 is parallel to p3-p1
/// - The direction p2-p1 is parallel to p4-p1
///
/// # 2D
/// - Eq1: (p2-p1) × (p3-p1) = 0
/// - Eq2: (p2-p1) × (p4-p1) = 0
/// - 2 equations
///
/// # 3D
/// - Each cross product has 2 independent components
/// - 4 equations total
pub struct CollinearConstraint {
    id: ConstraintId,
    line1_start: ParamRange,
    line1_end: ParamRange,
    line2_start: ParamRange,
    line2_end: ParamRange,
    deps: Vec<usize>,
}

impl CollinearConstraint {
    /// Create a new collinear constraint from two line ParamRanges.
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

impl Constraint for CollinearConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "Collinear"
    }

    fn equation_count(&self) -> usize {
        match self.dimension() {
            2 => 2,
            3 => 4,
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

            // Direction vector along line1
            let d_x = p2_x - p1_x;
            let d_y = p2_y - p1_y;

            // Vectors from p1 to p3 and p4
            let v3_x = p3_x - p1_x;
            let v3_y = p3_y - p1_y;
            let v4_x = p4_x - p1_x;
            let v4_y = p4_y - p1_y;

            // Cross products (2D)
            let cross1 = d_x * v3_y - d_y * v3_x; // (p2-p1) × (p3-p1)
            let cross2 = d_x * v4_y - d_y * v4_x; // (p2-p1) × (p4-p1)

            vec![cross1, cross2]
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

            let d_x = p2_x - p1_x;
            let d_y = p2_y - p1_y;
            let d_z = p2_z - p1_z;

            let v3_x = p3_x - p1_x;
            let v3_y = p3_y - p1_y;
            let v3_z = p3_z - p1_z;
            let v4_x = p4_x - p1_x;
            let v4_y = p4_y - p1_y;
            let v4_z = p4_z - p1_z;

            // Cross product d × v3 (2 independent components)
            let cross1_yz = d_y * v3_z - d_z * v3_y;
            let cross1_zx = d_z * v3_x - d_x * v3_z;

            // Cross product d × v4 (2 independent components)
            let cross2_yz = d_y * v4_z - d_z * v4_y;
            let cross2_zx = d_z * v4_x - d_x * v4_z;

            vec![cross1_yz, cross1_zx, cross2_yz, cross2_zx]
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

            let d_x = p2_x - p1_x;
            let d_y = p2_y - p1_y;
            let v3_x = p3_x - p1_x;
            let v3_y = p3_y - p1_y;
            let v4_x = p4_x - p1_x;
            let v4_y = p4_y - p1_y;

            let mut jac = Vec::new();

            // Eq0: cross1 = d_x * v3_y - d_y * v3_x
            //     = (p2_x - p1_x) * (p3_y - p1_y) - (p2_y - p1_y) * (p3_x - p1_x)
            jac.push((0, self.line1_start.start, -v3_y + d_y));     // df0/dp1_x
            jac.push((0, self.line1_start.start + 1, v3_x - d_x));  // df0/dp1_y
            jac.push((0, self.line1_end.start, v3_y));              // df0/dp2_x
            jac.push((0, self.line1_end.start + 1, -v3_x));         // df0/dp2_y
            jac.push((0, self.line2_start.start, -d_y));            // df0/dp3_x
            jac.push((0, self.line2_start.start + 1, d_x));         // df0/dp3_y

            // Eq1: cross2 = d_x * v4_y - d_y * v4_x
            jac.push((1, self.line1_start.start, -v4_y + d_y));     // df1/dp1_x
            jac.push((1, self.line1_start.start + 1, v4_x - d_x));  // df1/dp1_y
            jac.push((1, self.line1_end.start, v4_y));              // df1/dp2_x
            jac.push((1, self.line1_end.start + 1, -v4_x));         // df1/dp2_y
            jac.push((1, self.line2_end.start, -d_y));              // df1/dp4_x
            jac.push((1, self.line2_end.start + 1, d_x));           // df1/dp4_y

            jac
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

            let d_x = p2_x - p1_x;
            let d_y = p2_y - p1_y;
            let d_z = p2_z - p1_z;
            let v3_x = p3_x - p1_x;
            let v3_y = p3_y - p1_y;
            let v3_z = p3_z - p1_z;
            let v4_x = p4_x - p1_x;
            let v4_y = p4_y - p1_y;
            let v4_z = p4_z - p1_z;

            let mut jac = Vec::new();

            // Eq0: cross1_yz = d_y * v3_z - d_z * v3_y
            jac.push((0, self.line1_start.start + 1, -v3_z + d_z));
            jac.push((0, self.line1_start.start + 2, v3_y - d_y));
            jac.push((0, self.line1_end.start + 1, v3_z));
            jac.push((0, self.line1_end.start + 2, -v3_y));
            jac.push((0, self.line2_start.start + 1, -d_z));
            jac.push((0, self.line2_start.start + 2, d_y));

            // Eq1: cross1_zx = d_z * v3_x - d_x * v3_z
            jac.push((1, self.line1_start.start + 2, -v3_x + d_x));
            jac.push((1, self.line1_start.start, v3_z - d_z));
            jac.push((1, self.line1_end.start + 2, v3_x));
            jac.push((1, self.line1_end.start, -v3_z));
            jac.push((1, self.line2_start.start + 2, -d_x));
            jac.push((1, self.line2_start.start, d_z));

            // Eq2: cross2_yz = d_y * v4_z - d_z * v4_y
            jac.push((2, self.line1_start.start + 1, -v4_z + d_z));
            jac.push((2, self.line1_start.start + 2, v4_y - d_y));
            jac.push((2, self.line1_end.start + 1, v4_z));
            jac.push((2, self.line1_end.start + 2, -v4_y));
            jac.push((2, self.line2_end.start + 1, -d_z));
            jac.push((2, self.line2_end.start + 2, d_y));

            // Eq3: cross2_zx = d_z * v4_x - d_x * v4_z
            jac.push((3, self.line1_start.start + 2, -v4_x + d_x));
            jac.push((3, self.line1_start.start, v4_z - d_z));
            jac.push((3, self.line1_end.start + 2, v4_x));
            jac.push((3, self.line1_end.start, -v4_z));
            jac.push((3, self.line2_end.start + 2, -d_x));
            jac.push((3, self.line2_end.start, d_z));

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
    fn test_collinear_2d_satisfied() {
        // Four points on the same line y = x
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let p3 = ParamRange { start: 4, count: 2 };
        let p4 = ParamRange { start: 6, count: 2 };
        let constraint = CollinearConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);

        let params = vec![
            0.0, 0.0, // p1: (0, 0)
            1.0, 1.0, // p2: (1, 1)
            2.0, 2.0, // p3: (2, 2)
            3.0, 3.0, // p4: (3, 3)
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);
        assert!(residuals[0].abs() < 1e-10, "residual[0] = {}", residuals[0]);
        assert!(residuals[1].abs() < 1e-10, "residual[1] = {}", residuals[1]);
    }

    #[test]
    fn test_collinear_2d_unsatisfied() {
        // Points not on the same line
        let p1 = ParamRange { start: 0, count: 2 };
        let p2 = ParamRange { start: 2, count: 2 };
        let p3 = ParamRange { start: 4, count: 2 };
        let p4 = ParamRange { start: 6, count: 2 };
        let constraint = CollinearConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);

        let params = vec![
            0.0, 0.0, // p1: (0, 0)
            1.0, 0.0, // p2: (1, 0) - horizontal
            0.0, 1.0, // p3: (0, 1) - not on line!
            1.0, 1.0, // p4: (1, 1)
        ];

        let residuals = constraint.residuals(&params);
        let total = residuals[0].abs() + residuals[1].abs();
        assert!(total > 0.5, "total residual = {}", total);
    }

    #[test]
    fn test_collinear_3d_satisfied() {
        // Four points on the line t*(1,1,1)
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint = CollinearConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);

        let params = vec![
            0.0, 0.0, 0.0, // p1
            1.0, 1.0, 1.0, // p2
            2.0, 2.0, 2.0, // p3
            3.0, 3.0, 3.0, // p4
        ];

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 4);
        for (i, &r) in residuals.iter().enumerate() {
            assert!(r.abs() < 1e-10, "residual[{}] = {}", i, r);
        }
    }

    #[test]
    fn test_collinear_3d_unsatisfied() {
        // Points not collinear.
        // Use direction (1,1,1) so cross-product yz/zx components are non-zero
        // (axis-aligned directions like (1,0,0) only produce a non-zero xy component,
        // which is the dependent component omitted by the 2-equation formulation).
        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint = CollinearConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);

        let params = vec![
            0.0, 0.0, 0.0, // p1
            1.0, 1.0, 1.0, // p2 - direction (1,1,1)
            1.0, 0.0, 0.0, // p3 - not collinear!
            0.0, 1.0, 0.0, // p4 - not collinear!
        ];

        let residuals = constraint.residuals(&params);
        let total: f64 = residuals.iter().map(|r| r.abs()).sum();
        assert!(total > 0.5, "total residual = {}", total);
    }

    #[test]
    fn test_equation_count() {
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint_2d = CollinearConstraint::new(ConstraintId(0), line1, line2);
        assert_eq!(constraint_2d.equation_count(), 2);

        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint_3d = CollinearConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);
        assert_eq!(constraint_3d.equation_count(), 4);
    }

    #[test]
    fn test_dependency_count() {
        let line1 = ParamRange { start: 0, count: 4 };
        let line2 = ParamRange { start: 4, count: 4 };
        let constraint_2d = CollinearConstraint::new(ConstraintId(0), line1, line2);
        assert_eq!(constraint_2d.dependencies().len(), 8);

        let p1 = ParamRange { start: 0, count: 3 };
        let p2 = ParamRange { start: 3, count: 3 };
        let p3 = ParamRange { start: 6, count: 3 };
        let p4 = ParamRange { start: 9, count: 3 };
        let constraint_3d = CollinearConstraint::from_points(ConstraintId(0), p1, p2, p3, p4);
        assert_eq!(constraint_3d.dependencies().len(), 12);
    }
}
