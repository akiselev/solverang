//! Bezier curve continuity constraints (G0, G1, G2).

use crate::geometry::params::{ConstraintId, ParamRange};
use crate::geometry::constraint::{Constraint, Nonlinearity, MIN_EPSILON};

/// G0 (positional) continuity: end of one curve equals start of another.
///
/// # Entity Parameters
/// - curve1_end: Point2D [x, y] — last control point of first curve (2 params)
/// - curve2_start: Point2D [x, y] — first control point of second curve (2 params)
///
/// # Equations (2 for 2D)
/// - curve1_end.x = curve2_start.x
/// - curve1_end.y = curve2_start.y
#[derive(Clone, Debug)]
pub struct G0ContinuityConstraint {
    id: ConstraintId,
    curve1_end: ParamRange,
    curve2_start: ParamRange,
    deps: Vec<usize>,
}

impl G0ContinuityConstraint {
    /// Create a new G0 continuity constraint.
    ///
    /// # Arguments
    /// - `id`: Unique constraint identifier
    /// - `curve1_end`: ParamRange for the last control point of curve1
    /// - `curve2_start`: ParamRange for the first control point of curve2
    pub fn new(id: ConstraintId, curve1_end: ParamRange, curve2_start: ParamRange) -> Self {
        assert_eq!(curve1_end.count, 2, "2D control point must have 2 parameters");
        assert_eq!(curve2_start.count, 2, "2D control point must have 2 parameters");

        let mut deps = Vec::with_capacity(4);
        deps.extend(curve1_end.iter());
        deps.extend(curve2_start.iter());

        Self {
            id,
            curve1_end,
            curve2_start,
            deps,
        }
    }

    /// Convenience constructor from two CubicBezier2D param ranges.
    ///
    /// Automatically extracts P3 of bezier1 and P0 of bezier2.
    /// CubicBezier2D layout: [x0,y0, x1,y1, x2,y2, x3,y3] — 8 params
    pub fn from_beziers(id: ConstraintId, bezier1: ParamRange, bezier2: ParamRange) -> Self {
        assert_eq!(bezier1.count, 8, "CubicBezier2D must have 8 parameters");
        assert_eq!(bezier2.count, 8, "CubicBezier2D must have 8 parameters");

        let curve1_end = ParamRange {
            start: bezier1.start + 6,  // P3: offsets 6,7
            count: 2,
        };
        let curve2_start = ParamRange {
            start: bezier2.start,       // P0: offsets 0,1
            count: 2,
        };

        Self::new(id, curve1_end, curve2_start)
    }
}

impl Constraint for G0ContinuityConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "G0Continuity"
    }

    fn equation_count(&self) -> usize {
        2  // 2D
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let x1 = params[self.curve1_end.start];
        let y1 = params[self.curve1_end.start + 1];
        let x2 = params[self.curve2_start.start];
        let y2 = params[self.curve2_start.start + 1];

        vec![
            x1 - x2,
            y1 - y2,
        ]
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, self.curve1_end.start, 1.0),
            (0, self.curve2_start.start, -1.0),
            (1, self.curve1_end.start + 1, 1.0),
            (1, self.curve2_start.start + 1, -1.0),
        ]
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Linear
    }
}

/// G1 (tangent) continuity: G0 + tangent vectors are collinear.
///
/// For cubic Bezier curves:
/// - Tangent at t=1 of curve A: proportional to (P3_a - P2_a)
/// - Tangent at t=0 of curve B: proportional to (P1_b - P0_b)
///
/// # Entity Parameters
/// - bezier1: CubicBezier2D [x0,y0, x1,y1, x2,y2, x3,y3] — 8 params
/// - bezier2: CubicBezier2D [x0,y0, x1,y1, x2,y2, x3,y3] — 8 params
///
/// # Equations (3 for 2D)
/// - G0 x: P3_a.x = P0_b.x
/// - G0 y: P3_a.y = P0_b.y
/// - Tangent collinearity: (P3_a - P2_a) × (P1_b - P0_b) = 0 (2D cross product)
#[derive(Clone, Debug)]
pub struct G1ContinuityConstraint {
    id: ConstraintId,
    bezier1: ParamRange,
    bezier2: ParamRange,
    deps: Vec<usize>,
}

impl G1ContinuityConstraint {
    /// Create a new G1 continuity constraint between two cubic Bezier curves.
    pub fn new(id: ConstraintId, bezier1: ParamRange, bezier2: ParamRange) -> Self {
        assert_eq!(bezier1.count, 8, "CubicBezier2D must have 8 parameters");
        assert_eq!(bezier2.count, 8, "CubicBezier2D must have 8 parameters");

        // Dependencies: P2, P3 from bezier1 + P0, P1 from bezier2
        let mut deps = Vec::with_capacity(8);
        deps.extend((bezier1.start + 4)..=(bezier1.start + 7));  // P2, P3
        deps.extend((bezier2.start)..=(bezier2.start + 3));      // P0, P1

        Self {
            id,
            bezier1,
            bezier2,
            deps,
        }
    }
}

impl Constraint for G1ContinuityConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "G1Continuity"
    }

    fn equation_count(&self) -> usize {
        3  // 2 for G0 + 1 for tangent collinearity
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        // P2_a, P3_a from bezier1
        let p2a_x = params[self.bezier1.start + 4];
        let p2a_y = params[self.bezier1.start + 5];
        let p3a_x = params[self.bezier1.start + 6];
        let p3a_y = params[self.bezier1.start + 7];

        // P0_b, P1_b from bezier2
        let p0b_x = params[self.bezier2.start];
        let p0b_y = params[self.bezier2.start + 1];
        let p1b_x = params[self.bezier2.start + 2];
        let p1b_y = params[self.bezier2.start + 3];

        // G0: P3_a = P0_b
        let g0_x = p3a_x - p0b_x;
        let g0_y = p3a_y - p0b_y;

        // Tangent vectors
        let ta_x = p3a_x - p2a_x;
        let ta_y = p3a_y - p2a_y;
        let tb_x = p1b_x - p0b_x;
        let tb_y = p1b_y - p0b_y;

        // 2D cross product: ta × tb = ta_x * tb_y - ta_y * tb_x
        let cross = ta_x * tb_y - ta_y * tb_x;

        vec![g0_x, g0_y, cross]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        // P2_a, P3_a from bezier1
        let p2a_x = params[self.bezier1.start + 4];
        let p2a_y = params[self.bezier1.start + 5];
        let p3a_x = params[self.bezier1.start + 6];
        let p3a_y = params[self.bezier1.start + 7];

        // P0_b, P1_b from bezier2
        let p0b_x = params[self.bezier2.start];
        let p0b_y = params[self.bezier2.start + 1];
        let p1b_x = params[self.bezier2.start + 2];
        let p1b_y = params[self.bezier2.start + 3];

        // Tangent vectors
        let ta_x = p3a_x - p2a_x;
        let ta_y = p3a_y - p2a_y;
        let tb_x = p1b_x - p0b_x;
        let tb_y = p1b_y - p0b_y;

        let mut jac = Vec::with_capacity(16);

        // Equation 0: g0_x = p3a_x - p0b_x
        jac.push((0, self.bezier1.start + 6, 1.0));   // d/dp3a_x
        jac.push((0, self.bezier2.start, -1.0));      // d/dp0b_x

        // Equation 1: g0_y = p3a_y - p0b_y
        jac.push((1, self.bezier1.start + 7, 1.0));   // d/dp3a_y
        jac.push((1, self.bezier2.start + 1, -1.0));  // d/dp0b_y

        // Equation 2: cross = ta_x * tb_y - ta_y * tb_x
        // where ta = P3_a - P2_a, tb = P1_b - P0_b

        // d(cross)/dp2a_x = d(cross)/dta_x * (-1) = -tb_y
        jac.push((2, self.bezier1.start + 4, -tb_y));

        // d(cross)/dp2a_y = d(cross)/dta_y * (-1) = tb_x
        jac.push((2, self.bezier1.start + 5, tb_x));

        // d(cross)/dp3a_x = d(cross)/dta_x * 1 = tb_y
        jac.push((2, self.bezier1.start + 6, tb_y));

        // d(cross)/dp3a_y = d(cross)/dta_y * 1 = -tb_x
        jac.push((2, self.bezier1.start + 7, -tb_x));

        // d(cross)/dp0b_x = d(cross)/dtb_x * (-1) = ta_y
        jac.push((2, self.bezier2.start, ta_y));

        // d(cross)/dp0b_y = d(cross)/dtb_y * (-1) = -ta_x
        jac.push((2, self.bezier2.start + 1, -ta_x));

        // d(cross)/dp1b_x = d(cross)/dtb_x * 1 = -ta_y
        jac.push((2, self.bezier2.start + 2, -ta_y));

        // d(cross)/dp1b_y = d(cross)/dtb_y * 1 = ta_x
        jac.push((2, self.bezier2.start + 3, ta_x));

        jac
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::High  // Cross product is nonlinear
    }
}

/// G2 (curvature) continuity: G1 + curvatures match at junction.
///
/// For cubic Bezier, curvature at t=1 of curve A:
/// ```text
/// κ_a = (2/3) * |cross(P3-P2, P3-P1)| / |P3-P2|³
/// ```
/// At t=0 of curve B:
/// ```text
/// κ_b = (2/3) * |cross(P1-P0, P2-P0)| / |P1-P0|³
/// ```
///
/// # Entity Parameters
/// - bezier1: CubicBezier2D — 8 params
/// - bezier2: CubicBezier2D — 8 params
///
/// # Equations (4 for 2D)
/// - G0 (2 equations)
/// - G1 tangent collinearity (1 equation)
/// - Curvature match (1 equation):
///   cross(P3a-P2a, P3a-P1a) * |P1b-P0b|³ - cross(P1b-P0b, P2b-P0b) * |P3a-P2a|³ = 0
#[derive(Clone, Debug)]
pub struct G2ContinuityConstraint {
    id: ConstraintId,
    bezier1: ParamRange,
    bezier2: ParamRange,
    deps: Vec<usize>,
}

impl G2ContinuityConstraint {
    /// Create a new G2 continuity constraint between two cubic Bezier curves.
    pub fn new(id: ConstraintId, bezier1: ParamRange, bezier2: ParamRange) -> Self {
        assert_eq!(bezier1.count, 8, "CubicBezier2D must have 8 parameters");
        assert_eq!(bezier2.count, 8, "CubicBezier2D must have 8 parameters");

        // Dependencies: P1, P2, P3 from bezier1 + P0, P1, P2 from bezier2
        let mut deps = Vec::with_capacity(12);
        deps.extend((bezier1.start + 2)..=(bezier1.start + 7));  // P1, P2, P3
        deps.extend((bezier2.start)..=(bezier2.start + 5));      // P0, P1, P2

        Self {
            id,
            bezier1,
            bezier2,
            deps,
        }
    }

    /// Helper: 2D cross product of two 2D vectors.
    fn cross_2d(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
        ax * by - ay * bx
    }
}

impl Constraint for G2ContinuityConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "G2Continuity"
    }

    fn equation_count(&self) -> usize {
        4  // 2 for G0 + 1 for G1 + 1 for curvature
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        // P1_a, P2_a, P3_a from bezier1
        let p1a_x = params[self.bezier1.start + 2];
        let p1a_y = params[self.bezier1.start + 3];
        let p2a_x = params[self.bezier1.start + 4];
        let p2a_y = params[self.bezier1.start + 5];
        let p3a_x = params[self.bezier1.start + 6];
        let p3a_y = params[self.bezier1.start + 7];

        // P0_b, P1_b, P2_b from bezier2
        let p0b_x = params[self.bezier2.start];
        let p0b_y = params[self.bezier2.start + 1];
        let p1b_x = params[self.bezier2.start + 2];
        let p1b_y = params[self.bezier2.start + 3];
        let p2b_x = params[self.bezier2.start + 4];
        let p2b_y = params[self.bezier2.start + 5];

        // G0: P3_a = P0_b
        let g0_x = p3a_x - p0b_x;
        let g0_y = p3a_y - p0b_y;

        // G1: tangent collinearity
        let ta_x = p3a_x - p2a_x;
        let ta_y = p3a_y - p2a_y;
        let tb_x = p1b_x - p0b_x;
        let tb_y = p1b_y - p0b_y;
        let g1_cross = Self::cross_2d(ta_x, ta_y, tb_x, tb_y);

        // G2: curvature match
        // For curve A at t=1:
        let v1_x = p3a_x - p2a_x;
        let v1_y = p3a_y - p2a_y;
        let v2_x = p3a_x - p1a_x;
        let v2_y = p3a_y - p1a_y;
        let cross_a = Self::cross_2d(v1_x, v1_y, v2_x, v2_y);

        // For curve B at t=0:
        let u1_x = p1b_x - p0b_x;
        let u1_y = p1b_y - p0b_y;
        let u2_x = p2b_x - p0b_x;
        let u2_y = p2b_y - p0b_y;
        let cross_b = Self::cross_2d(u1_x, u1_y, u2_x, u2_y);

        // Magnitude cubed
        let mag_a_sq = v1_x * v1_x + v1_y * v1_y;
        let mag_a_cubed = mag_a_sq.max(MIN_EPSILON) * mag_a_sq.max(MIN_EPSILON).sqrt();

        let mag_b_sq = u1_x * u1_x + u1_y * u1_y;
        let mag_b_cubed = mag_b_sq.max(MIN_EPSILON) * mag_b_sq.max(MIN_EPSILON).sqrt();

        // Curvature equation: cross_a * mag_b³ - cross_b * mag_a³ = 0
        let g2_curv = cross_a * mag_b_cubed - cross_b * mag_a_cubed;

        vec![g0_x, g0_y, g1_cross, g2_curv]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        // P1_a, P2_a, P3_a from bezier1
        let p1a_x = params[self.bezier1.start + 2];
        let p1a_y = params[self.bezier1.start + 3];
        let p2a_x = params[self.bezier1.start + 4];
        let p2a_y = params[self.bezier1.start + 5];
        let p3a_x = params[self.bezier1.start + 6];
        let p3a_y = params[self.bezier1.start + 7];

        // P0_b, P1_b, P2_b from bezier2
        let p0b_x = params[self.bezier2.start];
        let p0b_y = params[self.bezier2.start + 1];
        let p1b_x = params[self.bezier2.start + 2];
        let p1b_y = params[self.bezier2.start + 3];
        let p2b_x = params[self.bezier2.start + 4];
        let p2b_y = params[self.bezier2.start + 5];

        let mut jac = Vec::new();

        // === Equation 0: G0 x ===
        jac.push((0, self.bezier1.start + 6, 1.0));   // d/dp3a_x
        jac.push((0, self.bezier2.start, -1.0));      // d/dp0b_x

        // === Equation 1: G0 y ===
        jac.push((1, self.bezier1.start + 7, 1.0));   // d/dp3a_y
        jac.push((1, self.bezier2.start + 1, -1.0));  // d/dp0b_y

        // === Equation 2: G1 tangent collinearity ===
        let ta_x = p3a_x - p2a_x;
        let ta_y = p3a_y - p2a_y;
        let tb_x = p1b_x - p0b_x;
        let tb_y = p1b_y - p0b_y;

        jac.push((2, self.bezier1.start + 4, -tb_y));  // d/dp2a_x
        jac.push((2, self.bezier1.start + 5, tb_x));   // d/dp2a_y
        jac.push((2, self.bezier1.start + 6, tb_y));   // d/dp3a_x
        jac.push((2, self.bezier1.start + 7, -tb_x));  // d/dp3a_y
        jac.push((2, self.bezier2.start, ta_y));       // d/dp0b_x
        jac.push((2, self.bezier2.start + 1, -ta_x));  // d/dp0b_y
        jac.push((2, self.bezier2.start + 2, -ta_y));  // d/dp1b_x
        jac.push((2, self.bezier2.start + 3, ta_x));   // d/dp1b_y

        // === Equation 3: G2 curvature ===
        // This is complex - compute derivatives step by step

        let v1_x = p3a_x - p2a_x;
        let v1_y = p3a_y - p2a_y;
        let v2_x = p3a_x - p1a_x;
        let v2_y = p3a_y - p1a_y;
        let cross_a = Self::cross_2d(v1_x, v1_y, v2_x, v2_y);

        let u1_x = p1b_x - p0b_x;
        let u1_y = p1b_y - p0b_y;
        let u2_x = p2b_x - p0b_x;
        let u2_y = p2b_y - p0b_y;
        let cross_b = Self::cross_2d(u1_x, u1_y, u2_x, u2_y);

        let mag_a_sq = v1_x * v1_x + v1_y * v1_y;
        let mag_a = mag_a_sq.max(MIN_EPSILON).sqrt();
        let mag_a_cubed = mag_a * mag_a * mag_a;

        let mag_b_sq = u1_x * u1_x + u1_y * u1_y;
        let mag_b = mag_b_sq.max(MIN_EPSILON).sqrt();
        let mag_b_cubed = mag_b * mag_b * mag_b;

        // f = cross_a * mag_b³ - cross_b * mag_a³

        // Derivatives w.r.t. P1_a (affects v2 and cross_a)
        // v2 = P3_a - P1_a, so dv2/dp1a = -1
        // cross_a = v1_x*v2_y - v1_y*v2_x
        // dcross_a/dp1a_x = dcross_a/dv2_x * (-1) = (-v1_y)*(-1) = v1_y
        // dcross_a/dp1a_y = dcross_a/dv2_y * (-1) = v1_x*(-1) = -v1_x
        let dcross_a_dp1a_x = v1_y;
        let dcross_a_dp1a_y = -v1_x;
        jac.push((3, self.bezier1.start + 2, dcross_a_dp1a_x * mag_b_cubed));
        jac.push((3, self.bezier1.start + 3, dcross_a_dp1a_y * mag_b_cubed));

        // Derivatives w.r.t. P2_a (affects v1, cross_a, mag_a)
        // v1 = P3_a - P2_a, so dv1/dp2a = -1
        // dcross_a/dp2a_x = dcross_a/dv1_x * (-1) = v2_y * (-1) = -v2_y
        // dcross_a/dp2a_y = dcross_a/dv1_y * (-1) = (-v2_x) * (-1) = v2_x
        let dcross_a_dp2a_x = -v2_y;
        let dcross_a_dp2a_y = v2_x;
        let dmag_a_cubed_dp2a_x = -3.0 * mag_a * v1_x;
        let dmag_a_cubed_dp2a_y = -3.0 * mag_a * v1_y;
        jac.push((3, self.bezier1.start + 4, dcross_a_dp2a_x * mag_b_cubed - cross_b * dmag_a_cubed_dp2a_x));
        jac.push((3, self.bezier1.start + 5, dcross_a_dp2a_y * mag_b_cubed - cross_b * dmag_a_cubed_dp2a_y));

        // Derivatives w.r.t. P3_a (affects v1, v2, cross_a, mag_a)
        let dcross_a_dp3a_x = v2_y - v1_y;
        let dcross_a_dp3a_y = -v2_x + v1_x;
        let dmag_a_cubed_dp3a_x = 3.0 * mag_a * v1_x;
        let dmag_a_cubed_dp3a_y = 3.0 * mag_a * v1_y;
        jac.push((3, self.bezier1.start + 6, dcross_a_dp3a_x * mag_b_cubed - cross_b * dmag_a_cubed_dp3a_x));
        jac.push((3, self.bezier1.start + 7, dcross_a_dp3a_y * mag_b_cubed - cross_b * dmag_a_cubed_dp3a_y));

        // Derivatives w.r.t. P0_b (affects u1, u2, cross_b, mag_b)
        // u1 = P1_b - P0_b, du1/dp0b = -1; u2 = P2_b - P0_b, du2/dp0b = -1
        // dcross_b/dp0b_x = u2_y*(-1) + (-u1_y)*(-1) = -u2_y + u1_y = u1_y - u2_y
        // dcross_b/dp0b_y = (-u2_x)*(-1) + u1_x*(-1) = u2_x - u1_x
        let dcross_b_dp0b_x = u1_y - u2_y;
        let dcross_b_dp0b_y = u2_x - u1_x;
        let dmag_b_cubed_dp0b_x = -3.0 * mag_b * u1_x;
        let dmag_b_cubed_dp0b_y = -3.0 * mag_b * u1_y;
        jac.push((3, self.bezier2.start, cross_a * dmag_b_cubed_dp0b_x - dcross_b_dp0b_x * mag_a_cubed));
        jac.push((3, self.bezier2.start + 1, cross_a * dmag_b_cubed_dp0b_y - dcross_b_dp0b_y * mag_a_cubed));

        // Derivatives w.r.t. P1_b (affects u1, cross_b, mag_b)
        // u1 = P1_b - P0_b, so du1/dp1b = +1
        // dcross_b/dp1b_x = u2_y * (+1) = u2_y
        // dcross_b/dp1b_y = (-u2_x) * (+1) = -u2_x
        let dcross_b_dp1b_x = u2_y;
        let dcross_b_dp1b_y = -u2_x;
        let dmag_b_cubed_dp1b_x = 3.0 * mag_b * u1_x;
        let dmag_b_cubed_dp1b_y = 3.0 * mag_b * u1_y;
        jac.push((3, self.bezier2.start + 2, cross_a * dmag_b_cubed_dp1b_x - dcross_b_dp1b_x * mag_a_cubed));
        jac.push((3, self.bezier2.start + 3, cross_a * dmag_b_cubed_dp1b_y - dcross_b_dp1b_y * mag_a_cubed));

        // Derivatives w.r.t. P2_b (affects u2 and cross_b)
        // u2 = P2_b - P0_b, so du2/dp2b = +1
        // dcross_b/dp2b_x = (-u1_y) * (+1) = -u1_y
        // dcross_b/dp2b_y = u1_x * (+1) = u1_x
        let dcross_b_dp2b_x = -u1_y;
        let dcross_b_dp2b_y = u1_x;
        jac.push((3, self.bezier2.start + 4, -dcross_b_dp2b_x * mag_a_cubed));
        jac.push((3, self.bezier2.start + 5, -dcross_b_dp2b_y * mag_a_cubed));

        jac
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::High  // Curvature is highly nonlinear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g0_continuity_satisfied() {
        let id = ConstraintId(0);
        let curve1_end = ParamRange { start: 0, count: 2 };
        let curve2_start = ParamRange { start: 2, count: 2 };

        let constraint = G0ContinuityConstraint::new(id, curve1_end, curve2_start);

        let params = vec![5.0, 3.0, 5.0, 3.0];  // Same point
        let residuals = constraint.residuals(&params);

        assert_eq!(residuals.len(), 2);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);
    }

    #[test]
    fn test_g0_continuity_not_satisfied() {
        let id = ConstraintId(0);
        let curve1_end = ParamRange { start: 0, count: 2 };
        let curve2_start = ParamRange { start: 2, count: 2 };

        let constraint = G0ContinuityConstraint::new(id, curve1_end, curve2_start);

        let params = vec![5.0, 3.0, 10.0, 8.0];  // Different points
        let residuals = constraint.residuals(&params);

        assert!((residuals[0] - (-5.0)).abs() < 1e-10);
        assert!((residuals[1] - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_g0_from_beziers() {
        let id = ConstraintId(0);
        let bezier1 = ParamRange { start: 0, count: 8 };
        let bezier2 = ParamRange { start: 8, count: 8 };

        let constraint = G0ContinuityConstraint::from_beziers(id, bezier1, bezier2);

        // Bezier1 P3 at indices 6,7; Bezier2 P0 at indices 8,9
        let mut params = vec![0.0; 16];
        params[6] = 10.0;  // P3 x
        params[7] = 5.0;   // P3 y
        params[8] = 10.0;  // P0 x
        params[9] = 5.0;   // P0 y

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10);
        assert!(residuals[1].abs() < 1e-10);
    }

    #[test]
    fn test_g1_continuity_satisfied() {
        let id = ConstraintId(0);
        let bezier1 = ParamRange { start: 0, count: 8 };
        let bezier2 = ParamRange { start: 8, count: 8 };

        let constraint = G1ContinuityConstraint::new(id, bezier1, bezier2);

        // Bezier1: P2=(0,0), P3=(2,0)  => tangent = (2,0)
        // Bezier2: P0=(2,0), P1=(4,0)  => tangent = (2,0)
        let mut params = vec![0.0; 16];
        params[4] = 0.0; params[5] = 0.0;  // P2_a
        params[6] = 2.0; params[7] = 0.0;  // P3_a
        params[8] = 2.0; params[9] = 0.0;  // P0_b
        params[10] = 4.0; params[11] = 0.0;  // P1_b

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 3);
        assert!(residuals[0].abs() < 1e-10, "G0 x");
        assert!(residuals[1].abs() < 1e-10, "G0 y");
        assert!(residuals[2].abs() < 1e-10, "G1 cross product");
    }

    #[test]
    fn test_g1_continuity_not_collinear() {
        let id = ConstraintId(0);
        let bezier1 = ParamRange { start: 0, count: 8 };
        let bezier2 = ParamRange { start: 8, count: 8 };

        let constraint = G1ContinuityConstraint::new(id, bezier1, bezier2);

        // Bezier1: P2=(0,0), P3=(2,0)  => tangent = (2,0)
        // Bezier2: P0=(2,0), P1=(2,2)  => tangent = (0,2)
        // Cross product = 2*2 - 0*0 = 4 (not zero)
        let mut params = vec![0.0; 16];
        params[4] = 0.0; params[5] = 0.0;  // P2_a
        params[6] = 2.0; params[7] = 0.0;  // P3_a
        params[8] = 2.0; params[9] = 0.0;  // P0_b
        params[10] = 2.0; params[11] = 2.0;  // P1_b

        let residuals = constraint.residuals(&params);
        assert!(residuals[2].abs() > 1.0, "Should have non-zero cross product");
    }

    #[test]
    fn test_g1_jacobian_numerical() {
        let id = ConstraintId(0);
        let bezier1 = ParamRange { start: 0, count: 8 };
        let bezier2 = ParamRange { start: 8, count: 8 };

        let constraint = G1ContinuityConstraint::new(id, bezier1, bezier2);

        let mut params = vec![0.0; 16];
        params[4] = 1.0; params[5] = 0.5;
        params[6] = 3.0; params[7] = 1.0;
        params[8] = 3.0; params[9] = 1.0;
        params[10] = 5.0; params[11] = 2.0;

        let jac = constraint.jacobian(&params);
        let h = 1e-7;

        for &(row, col, analytical) in &jac {
            let mut params_plus = params.clone();
            params_plus[col] += h;
            let res_plus = constraint.residuals(&params_plus);

            let mut params_minus = params.clone();
            params_minus[col] -= h;
            let res_minus = constraint.residuals(&params_minus);

            let numerical = (res_plus[row] - res_minus[row]) / (2.0 * h);
            let error = (analytical - numerical).abs();

            assert!(
                error < 1e-4,
                "Jacobian mismatch at ({},{}): analytical={}, numerical={}, error={}",
                row, col, analytical, numerical, error
            );
        }
    }

    #[test]
    fn test_g2_continuity_satisfied() {
        let id = ConstraintId(0);
        let bezier1 = ParamRange { start: 0, count: 8 };
        let bezier2 = ParamRange { start: 8, count: 8 };

        let constraint = G2ContinuityConstraint::new(id, bezier1, bezier2);

        // Straight line segment with G2 continuity
        // Bezier1: P1=(0,0), P2=(1,0), P3=(2,0)
        // Bezier2: P0=(2,0), P1=(3,0), P2=(4,0)
        let mut params = vec![0.0; 16];
        params[2] = 0.0; params[3] = 0.0;  // P1_a
        params[4] = 1.0; params[5] = 0.0;  // P2_a
        params[6] = 2.0; params[7] = 0.0;  // P3_a
        params[8] = 2.0; params[9] = 0.0;  // P0_b
        params[10] = 3.0; params[11] = 0.0;  // P1_b
        params[12] = 4.0; params[13] = 0.0;  // P2_b

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 4);
        assert!(residuals[0].abs() < 1e-10, "G0 x");
        assert!(residuals[1].abs() < 1e-10, "G0 y");
        assert!(residuals[2].abs() < 1e-10, "G1");
        assert!(residuals[3].abs() < 1e-10, "G2 curvature");
    }

    #[test]
    fn test_g2_jacobian_finite() {
        let id = ConstraintId(0);
        let bezier1 = ParamRange { start: 0, count: 8 };
        let bezier2 = ParamRange { start: 8, count: 8 };

        let constraint = G2ContinuityConstraint::new(id, bezier1, bezier2);

        let mut params = vec![0.0; 16];
        for i in 0..16 {
            params[i] = (i as f64) * 0.5;
        }

        let jac = constraint.jacobian(&params);
        for (row, col, val) in &jac {
            assert!(val.is_finite(), "Non-finite Jacobian at ({},{})", row, col);
        }
    }

    #[test]
    fn test_constraint_metadata() {
        let id = ConstraintId(42);
        let range1 = ParamRange { start: 0, count: 2 };
        let range2 = ParamRange { start: 2, count: 2 };

        let g0 = G0ContinuityConstraint::new(id, range1, range2);
        assert_eq!(g0.id(), ConstraintId(42));
        assert_eq!(g0.name(), "G0Continuity");
        assert_eq!(g0.equation_count(), 2);
        assert_eq!(g0.nonlinearity_hint(), Nonlinearity::Linear);

        let bezier1 = ParamRange { start: 0, count: 8 };
        let bezier2 = ParamRange { start: 8, count: 8 };

        let g1 = G1ContinuityConstraint::new(id, bezier1, bezier2);
        assert_eq!(g1.name(), "G1Continuity");
        assert_eq!(g1.equation_count(), 3);
        assert_eq!(g1.nonlinearity_hint(), Nonlinearity::High);

        let g2 = G2ContinuityConstraint::new(id, bezier1, bezier2);
        assert_eq!(g2.name(), "G2Continuity");
        assert_eq!(g2.equation_count(), 4);
        assert_eq!(g2.nonlinearity_hint(), Nonlinearity::High);
    }
}
