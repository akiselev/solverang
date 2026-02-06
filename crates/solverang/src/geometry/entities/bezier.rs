/// Evaluate a 2D cubic Bezier curve at parameter t ∈ [0, 1].
/// CubicBezier2D params: [x0, y0, x1, y1, x2, y2, x3, y3].
/// B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
pub fn evaluate_cubic_2d(params: &[f64], t: f64) -> Vec<f64> {
    let p0x = params[0];
    let p0y = params[1];
    let p1x = params[2];
    let p1y = params[3];
    let p2x = params[4];
    let p2y = params[5];
    let p3x = params[6];
    let p3y = params[7];

    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;

    let x = mt3 * p0x + 3.0 * mt2 * t * p1x + 3.0 * mt * t2 * p2x + t3 * p3x;
    let y = mt3 * p0y + 3.0 * mt2 * t * p1y + 3.0 * mt * t2 * p2y + t3 * p3y;

    vec![x, y]
}

/// Evaluate tangent vector for a 2D cubic Bezier at parameter t.
/// B'(t) = 3[(1-t)²(P1-P0) + 2(1-t)t(P2-P1) + t²(P3-P2)]
pub fn tangent_cubic_2d(params: &[f64], t: f64) -> Vec<f64> {
    let p0x = params[0];
    let p0y = params[1];
    let p1x = params[2];
    let p1y = params[3];
    let p2x = params[4];
    let p2y = params[5];
    let p3x = params[6];
    let p3y = params[7];

    let t2 = t * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;

    let dx = 3.0 * (mt2 * (p1x - p0x) + 2.0 * mt * t * (p2x - p1x) + t2 * (p3x - p2x));
    let dy = 3.0 * (mt2 * (p1y - p0y) + 2.0 * mt * t * (p2y - p1y) + t2 * (p3y - p2y));

    vec![dx, dy]
}

/// Evaluate curvature at parameter t for 2D cubic Bezier.
/// κ = |B' × B''| / |B'|³
/// For 2D, cross product magnitude is |x'y'' - y'x''|.
pub fn curvature_cubic_2d(params: &[f64], t: f64) -> f64 {
    let p0x = params[0];
    let p0y = params[1];
    let p1x = params[2];
    let p1y = params[3];
    let p2x = params[4];
    let p2y = params[5];
    let p3x = params[6];
    let p3y = params[7];

    // First derivative
    let t2 = t * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;

    let dx = 3.0 * (mt2 * (p1x - p0x) + 2.0 * mt * t * (p2x - p1x) + t2 * (p3x - p2x));
    let dy = 3.0 * (mt2 * (p1y - p0y) + 2.0 * mt * t * (p2y - p1y) + t2 * (p3y - p2y));

    // Second derivative
    // B''(t) = 6[(1-t)(P2-2P1+P0) + t(P3-2P2+P1)]
    let ddx = 6.0 * (mt * (p2x - 2.0 * p1x + p0x) + t * (p3x - 2.0 * p2x + p1x));
    let ddy = 6.0 * (mt * (p2y - 2.0 * p1y + p0y) + t * (p3y - 2.0 * p2y + p1y));

    // Cross product magnitude in 2D
    let cross = dx * ddy - dy * ddx;

    // |B'|³
    let speed = (dx * dx + dy * dy).sqrt();
    let speed3 = speed * speed * speed;

    if speed3 < 1e-10 {
        0.0 // Degenerate case
    } else {
        cross.abs() / speed3
    }
}

/// Evaluate a 2D quadratic Bezier curve at parameter t ∈ [0, 1].
/// QuadBezier2D params: [x0, y0, x1, y1, x2, y2].
/// Q(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
pub fn evaluate_quad_2d(params: &[f64], t: f64) -> Vec<f64> {
    let p0x = params[0];
    let p0y = params[1];
    let p1x = params[2];
    let p1y = params[3];
    let p2x = params[4];
    let p2y = params[5];

    let t2 = t * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;

    let x = mt2 * p0x + 2.0 * mt * t * p1x + t2 * p2x;
    let y = mt2 * p0y + 2.0 * mt * t * p1y + t2 * p2y;

    vec![x, y]
}

/// Evaluate tangent vector for a 2D quadratic Bezier at parameter t.
/// Q'(t) = 2[(1-t)(P1-P0) + t(P2-P1)]
pub fn tangent_quad_2d(params: &[f64], t: f64) -> Vec<f64> {
    let p0x = params[0];
    let p0y = params[1];
    let p1x = params[2];
    let p1y = params[3];
    let p2x = params[4];
    let p2y = params[5];

    let mt = 1.0 - t;

    let dx = 2.0 * (mt * (p1x - p0x) + t * (p2x - p1x));
    let dy = 2.0 * (mt * (p1y - p0y) + t * (p2y - p1y));

    vec![dx, dy]
}

/// Evaluate a 3D cubic Bezier curve at parameter t ∈ [0, 1].
/// CubicBezier3D params: [x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3].
pub fn evaluate_cubic_3d(params: &[f64], t: f64) -> Vec<f64> {
    let p0x = params[0];
    let p0y = params[1];
    let p0z = params[2];
    let p1x = params[3];
    let p1y = params[4];
    let p1z = params[5];
    let p2x = params[6];
    let p2y = params[7];
    let p2z = params[8];
    let p3x = params[9];
    let p3y = params[10];
    let p3z = params[11];

    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;

    let x = mt3 * p0x + 3.0 * mt2 * t * p1x + 3.0 * mt * t2 * p2x + t3 * p3x;
    let y = mt3 * p0y + 3.0 * mt2 * t * p1y + 3.0 * mt * t2 * p2y + t3 * p3y;
    let z = mt3 * p0z + 3.0 * mt2 * t * p1z + 3.0 * mt * t2 * p2z + t3 * p3z;

    vec![x, y, z]
}

/// Evaluate tangent vector for a 3D cubic Bezier at parameter t.
pub fn tangent_cubic_3d(params: &[f64], t: f64) -> Vec<f64> {
    let p0x = params[0];
    let p0y = params[1];
    let p0z = params[2];
    let p1x = params[3];
    let p1y = params[4];
    let p1z = params[5];
    let p2x = params[6];
    let p2y = params[7];
    let p2z = params[8];
    let p3x = params[9];
    let p3y = params[10];
    let p3z = params[11];

    let t2 = t * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;

    let dx = 3.0 * (mt2 * (p1x - p0x) + 2.0 * mt * t * (p2x - p1x) + t2 * (p3x - p2x));
    let dy = 3.0 * (mt2 * (p1y - p0y) + 2.0 * mt * t * (p2y - p1y) + t2 * (p3y - p2y));
    let dz = 3.0 * (mt2 * (p1z - p0z) + 2.0 * mt * t * (p2z - p1z) + t2 * (p3z - p2z));

    vec![dx, dy, dz]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_cubic_2d_endpoints() {
        // Cubic Bezier from (0,0) to (3,3) with control points at (1,0) and (2,3)
        let params = [0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 3.0, 3.0];

        // At t=0, should be at P0
        let p0 = evaluate_cubic_2d(&params, 0.0);
        assert!((p0[0] - 0.0).abs() < 1e-10);
        assert!((p0[1] - 0.0).abs() < 1e-10);

        // At t=1, should be at P3
        let p1 = evaluate_cubic_2d(&params, 1.0);
        assert!((p1[0] - 3.0).abs() < 1e-10);
        assert!((p1[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_cubic_2d_midpoint() {
        // Straight line as a cubic Bezier (all control points collinear)
        let params = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];

        // At t=0.5, should be at (1.5, 1.5)
        let p = evaluate_cubic_2d(&params, 0.5);
        assert!((p[0] - 1.5).abs() < 1e-10);
        assert!((p[1] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_tangent_cubic_2d() {
        // Horizontal line from (0,0) to (3,0)
        let params = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];

        // Tangent at t=0 should point in +x direction
        let tan0 = tangent_cubic_2d(&params, 0.0);
        assert!(tan0[0] > 0.0);
        assert!(tan0[1].abs() < 1e-10);

        // Tangent at t=1 should also point in +x direction (straight line)
        let tan1 = tangent_cubic_2d(&params, 1.0);
        assert!(tan1[0] > 0.0);
        assert!(tan1[1].abs() < 1e-10);
    }

    #[test]
    fn test_curvature_cubic_2d_line() {
        // Straight line has zero curvature
        let params = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];

        let curv = curvature_cubic_2d(&params, 0.5);
        assert!(curv.abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_quad_2d_endpoints() {
        // Quadratic Bezier from (0,0) to (2,2) with control point at (1,0)
        let params = [0.0, 0.0, 1.0, 0.0, 2.0, 2.0];

        // At t=0, should be at P0
        let p0 = evaluate_quad_2d(&params, 0.0);
        assert!((p0[0] - 0.0).abs() < 1e-10);
        assert!((p0[1] - 0.0).abs() < 1e-10);

        // At t=1, should be at P2
        let p1 = evaluate_quad_2d(&params, 1.0);
        assert!((p1[0] - 2.0).abs() < 1e-10);
        assert!((p1[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_quad_2d_midpoint() {
        // Straight line as a quadratic Bezier
        let params = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        // At t=0.5, should be at (1, 1)
        let p = evaluate_quad_2d(&params, 0.5);
        assert!((p[0] - 1.0).abs() < 1e-10);
        assert!((p[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tangent_quad_2d() {
        // Horizontal line from (0,0) to (2,0)
        let params = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0];

        // Tangent should point in +x direction
        let tan = tangent_quad_2d(&params, 0.5);
        assert!(tan[0] > 0.0);
        assert!(tan[1].abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_cubic_3d_endpoints() {
        // 3D cubic Bezier from (0,0,0) to (3,3,3)
        let params = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 3.0, 3.0];

        // At t=0, should be at P0
        let p0 = evaluate_cubic_3d(&params, 0.0);
        assert!((p0[0] - 0.0).abs() < 1e-10);
        assert!((p0[1] - 0.0).abs() < 1e-10);
        assert!((p0[2] - 0.0).abs() < 1e-10);

        // At t=1, should be at P3
        let p1 = evaluate_cubic_3d(&params, 1.0);
        assert!((p1[0] - 3.0).abs() < 1e-10);
        assert!((p1[1] - 3.0).abs() < 1e-10);
        assert!((p1[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_tangent_cubic_3d() {
        // Straight line in 3D
        let params = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];

        // Tangent should point in (1,1,1) direction
        let tan = tangent_cubic_3d(&params, 0.5);
        assert!(tan[0] > 0.0);
        assert!(tan[1] > 0.0);
        assert!(tan[2] > 0.0);

        // For a straight line, tangent components should be equal
        assert!((tan[0] - tan[1]).abs() < 1e-10);
        assert!((tan[1] - tan[2]).abs() < 1e-10);
    }
}
