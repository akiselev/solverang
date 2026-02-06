/// Evaluate position at parameter t ∈ [0, 1] on a 2D arc.
/// Arc2D params: [cx, cy, r, start_angle, end_angle].
/// angle = start_angle + t * (end_angle - start_angle)
/// Returns [cx + r*cos(angle), cy + r*sin(angle)].
pub fn evaluate_2d(params: &[f64], t: f64) -> Vec<f64> {
    let cx = params[0];
    let cy = params[1];
    let r = params[2];
    let start_angle = params[3];
    let end_angle = params[4];

    let angle = start_angle + t * (end_angle - start_angle);
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    vec![cx + r * cos_a, cy + r * sin_a]
}

/// Evaluate tangent vector at parameter t on a 2D arc.
/// d/dt of evaluate: sweep * [-r*sin(angle), r*cos(angle)]
/// where sweep = end_angle - start_angle.
pub fn tangent_2d(params: &[f64], t: f64) -> Vec<f64> {
    let r = params[2];
    let start_angle = params[3];
    let end_angle = params[4];

    let sweep = end_angle - start_angle;
    let angle = start_angle + t * sweep;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    vec![-r * sin_a * sweep, r * cos_a * sweep]
}

/// Return the angular sweep of the arc.
pub fn sweep_angle(params: &[f64]) -> f64 {
    let start_angle = params[3];
    let end_angle = params[4];
    end_angle - start_angle
}

/// Return arc length.
pub fn arc_length(params: &[f64]) -> f64 {
    let r = params[2];
    let sweep = sweep_angle(params);
    r * sweep.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_evaluate_2d_quarter_circle() {
        // Quarter circle from 0 to π/2
        let params = [0.0, 0.0, 1.0, 0.0, PI / 2.0];

        // t=0 should be at angle 0 -> (1, 0)
        let p0 = evaluate_2d(&params, 0.0);
        assert!((p0[0] - 1.0).abs() < 1e-10);
        assert!(p0[1].abs() < 1e-10);

        // t=1 should be at angle π/2 -> (0, 1)
        let p1 = evaluate_2d(&params, 1.0);
        assert!(p1[0].abs() < 1e-10);
        assert!((p1[1] - 1.0).abs() < 1e-10);

        // t=0.5 should be at angle π/4 -> (√2/2, √2/2)
        let p_mid = evaluate_2d(&params, 0.5);
        let sqrt2_2 = 2.0_f64.sqrt() / 2.0;
        assert!((p_mid[0] - sqrt2_2).abs() < 1e-10);
        assert!((p_mid[1] - sqrt2_2).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_2d_with_offset() {
        // Arc from π to 2π centered at (5, 3) with radius 2
        let params = [5.0, 3.0, 2.0, PI, 2.0 * PI];

        // t=0 at angle π -> (-1, 0) relative, or (3, 3) absolute
        let p0 = evaluate_2d(&params, 0.0);
        assert!((p0[0] - 3.0).abs() < 1e-10);
        assert!((p0[1] - 3.0).abs() < 1e-10);

        // t=1 at angle 2π -> (1, 0) relative, or (7, 3) absolute
        let p1 = evaluate_2d(&params, 1.0);
        assert!((p1[0] - 7.0).abs() < 1e-10);
        assert!((p1[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_tangent_2d() {
        // Quarter circle from 0 to π/2
        let params = [0.0, 0.0, 1.0, 0.0, PI / 2.0];

        // At t=0 (angle 0, point (1,0)), tangent should point in +y direction
        let tan0 = tangent_2d(&params, 0.0);
        assert!(tan0[0].abs() < 1e-10); // x component ~0
        assert!(tan0[1] > 0.0); // y component positive

        // At t=1 (angle π/2, point (0,1)), tangent should point in -x direction
        let tan1 = tangent_2d(&params, 1.0);
        assert!(tan1[0] < 0.0); // x component negative
        assert!(tan1[1].abs() < 1e-10); // y component ~0
    }

    #[test]
    fn test_sweep_angle() {
        let params = [0.0, 0.0, 1.0, 0.0, PI / 2.0];
        let sweep = sweep_angle(&params);
        assert!((sweep - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_arc_length() {
        // Quarter circle with radius 1
        let params = [0.0, 0.0, 1.0, 0.0, PI / 2.0];
        let len = arc_length(&params);
        assert!((len - PI / 2.0).abs() < 1e-10);

        // Semicircle with radius 2
        let params2 = [0.0, 0.0, 2.0, 0.0, PI];
        let len2 = arc_length(&params2);
        assert!((len2 - 2.0 * PI).abs() < 1e-10);
    }
}
