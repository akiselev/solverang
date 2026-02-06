use std::f64::consts::PI;

/// Evaluate position at parameter t ∈ [0, 1] on a 2D ellipse.
/// Ellipse2D params: [cx, cy, rx, ry, rotation].
/// angle = 2πt
/// local: (rx*cos(angle), ry*sin(angle))
/// Rotate by rotation and translate by (cx, cy).
pub fn evaluate_2d(params: &[f64], t: f64) -> Vec<f64> {
    let cx = params[0];
    let cy = params[1];
    let rx = params[2];
    let ry = params[3];
    let rotation = params[4];

    let angle = 2.0 * PI * t;
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();

    // Point in local (unrotated) frame
    let local_x = rx * cos_angle;
    let local_y = ry * sin_angle;

    // Rotate by rotation angle
    let cos_rot = rotation.cos();
    let sin_rot = rotation.sin();

    let rotated_x = local_x * cos_rot - local_y * sin_rot;
    let rotated_y = local_x * sin_rot + local_y * cos_rot;

    // Translate to center
    vec![cx + rotated_x, cy + rotated_y]
}

/// Evaluate tangent vector at parameter t on a 2D ellipse.
/// Derivative of the parameterization with respect to t.
pub fn tangent_2d(params: &[f64], t: f64) -> Vec<f64> {
    let rx = params[2];
    let ry = params[3];
    let rotation = params[4];

    let angle = 2.0 * PI * t;
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();

    let two_pi = 2.0 * PI;

    // Derivative in local frame: d/dt [rx*cos(2πt), ry*sin(2πt)]
    let local_dx = -rx * sin_angle * two_pi;
    let local_dy = ry * cos_angle * two_pi;

    // Rotate by rotation angle
    let cos_rot = rotation.cos();
    let sin_rot = rotation.sin();

    let rotated_dx = local_dx * cos_rot - local_dy * sin_rot;
    let rotated_dy = local_dx * sin_rot + local_dy * cos_rot;

    vec![rotated_dx, rotated_dy]
}

/// Calculate the eccentricity of an ellipse.
/// e = sqrt(1 - (b²/a²)) where a is the semi-major axis and b is the semi-minor axis.
pub fn eccentricity(params: &[f64]) -> f64 {
    let rx = params[2];
    let ry = params[3];

    let a = rx.max(ry);
    let b = rx.min(ry);

    if a == 0.0 {
        return 0.0;
    }

    let ratio = b / a;
    (1.0 - ratio * ratio).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_evaluate_2d_circle() {
        // When rx == ry, should behave like a circle
        let params = [0.0, 0.0, 1.0, 1.0, 0.0];

        // t=0 should be at (1, 0)
        let p0 = evaluate_2d(&params, 0.0);
        assert!((p0[0] - 1.0).abs() < 1e-10);
        assert!(p0[1].abs() < 1e-10);

        // t=0.25 should be at (0, 1)
        let p1 = evaluate_2d(&params, 0.25);
        assert!(p1[0].abs() < 1e-10);
        assert!((p1[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_2d_axis_aligned() {
        // Ellipse with rx=2, ry=1, no rotation
        let params = [0.0, 0.0, 2.0, 1.0, 0.0];

        // t=0 should be at (2, 0)
        let p0 = evaluate_2d(&params, 0.0);
        assert!((p0[0] - 2.0).abs() < 1e-10);
        assert!(p0[1].abs() < 1e-10);

        // t=0.25 should be at (0, 1)
        let p1 = evaluate_2d(&params, 0.25);
        assert!(p1[0].abs() < 1e-10);
        assert!((p1[1] - 1.0).abs() < 1e-10);

        // t=0.5 should be at (-2, 0)
        let p2 = evaluate_2d(&params, 0.5);
        assert!((p2[0] + 2.0).abs() < 1e-10);
        assert!(p2[1].abs() < 1e-10);

        // t=0.75 should be at (0, -1)
        let p3 = evaluate_2d(&params, 0.75);
        assert!(p3[0].abs() < 1e-10);
        assert!((p3[1] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_2d_rotated() {
        // Ellipse with rx=2, ry=1, rotated by π/2
        let params = [0.0, 0.0, 2.0, 1.0, PI / 2.0];

        // After 90° rotation, major axis is along Y, minor along X
        // t=0 should be at (0, 2) (the point that was at (2,0))
        let p0 = evaluate_2d(&params, 0.0);
        assert!(p0[0].abs() < 1e-10);
        assert!((p0[1] - 2.0).abs() < 1e-10);

        // t=0.25 should be at (-1, 0) (the point that was at (0,1))
        let p1 = evaluate_2d(&params, 0.25);
        assert!((p1[0] + 1.0).abs() < 1e-10);
        assert!(p1[1].abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_2d_with_offset() {
        // Ellipse centered at (5, 3) with rx=2, ry=1
        let params = [5.0, 3.0, 2.0, 1.0, 0.0];

        let p0 = evaluate_2d(&params, 0.0);
        assert!((p0[0] - 7.0).abs() < 1e-10);
        assert!((p0[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_tangent_2d() {
        // Circle (rx=ry=1)
        let params = [0.0, 0.0, 1.0, 1.0, 0.0];

        // At t=0 (point (1,0)), tangent should point in +y direction
        let tan0 = tangent_2d(&params, 0.0);
        assert!(tan0[0].abs() < 1e-10);
        assert!(tan0[1] > 0.0);

        // At t=0.25 (point (0,1)), tangent should point in -x direction
        let tan1 = tangent_2d(&params, 0.25);
        assert!(tan1[0] < 0.0);
        assert!(tan1[1].abs() < 1e-10);
    }

    #[test]
    fn test_eccentricity() {
        // Circle should have eccentricity 0
        let params_circle = [0.0, 0.0, 1.0, 1.0, 0.0];
        let e_circle = eccentricity(&params_circle);
        assert!(e_circle.abs() < 1e-10);

        // Ellipse with rx=2, ry=1
        // e = sqrt(1 - 1/4) = sqrt(3/4) = sqrt(3)/2
        let params_ellipse = [0.0, 0.0, 2.0, 1.0, 0.0];
        let e_ellipse = eccentricity(&params_ellipse);
        let expected = (3.0_f64 / 4.0).sqrt();
        assert!((e_ellipse - expected).abs() < 1e-10);
    }
}
