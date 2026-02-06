use std::f64::consts::PI;

/// Evaluate position at parameter t ∈ [0, 1] on a 2D circle.
/// Circle2D params: [cx, cy, r].
/// Full circle parameterized by angle θ = 2πt.
/// Returns [cx + r*cos(2πt), cy + r*sin(2πt)].
pub fn evaluate_2d(params: &[f64], t: f64) -> Vec<f64> {
    let cx = params[0];
    let cy = params[1];
    let r = params[2];

    let theta = 2.0 * PI * t;
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();

    vec![cx + r * cos_theta, cy + r * sin_theta]
}

/// Evaluate tangent vector at parameter t on a 2D circle.
/// Returns [-r*sin(2πt)*2π, r*cos(2πt)*2π].
pub fn tangent_2d(params: &[f64], t: f64) -> Vec<f64> {
    let r = params[2];

    let theta = 2.0 * PI * t;
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();

    let two_pi = 2.0 * PI;
    vec![-r * sin_theta * two_pi, r * cos_theta * two_pi]
}

/// Evaluate position at parameter t ∈ [0, 1] on a 3D circle.
/// Circle3D params: [cx, cy, cz, nx, ny, nz, r].
/// Circle in 3D with center, normal, radius.
/// Builds orthonormal frame from normal, then parameterizes.
pub fn evaluate_3d(params: &[f64], t: f64) -> Vec<f64> {
    let cx = params[0];
    let cy = params[1];
    let cz = params[2];
    let nx = params[3];
    let ny = params[4];
    let nz = params[5];
    let r = params[6];

    // Build orthonormal frame: given normal n, find two perpendicular unit vectors u, v
    let (u, v) = build_orthonormal_frame([nx, ny, nz]);

    let theta = 2.0 * PI * t;
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();

    // Point = center + r * (cos(θ) * u + sin(θ) * v)
    vec![
        cx + r * (cos_theta * u[0] + sin_theta * v[0]),
        cy + r * (cos_theta * u[1] + sin_theta * v[1]),
        cz + r * (cos_theta * u[2] + sin_theta * v[2]),
    ]
}

/// Build an orthonormal frame from a normal vector.
/// Returns two unit vectors (u, v) that are perpendicular to the normal and to each other.
fn build_orthonormal_frame(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let [nx, ny, nz] = normal;

    // Normalize the normal
    let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
    let n = [nx / n_len, ny / n_len, nz / n_len];

    // Find a vector not parallel to n
    let candidate = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    // u = normalize(candidate × n)
    let cross1 = cross(candidate, n);
    let u = normalize(cross1);

    // v = normalize(n × u)
    let cross2 = cross(n, u);
    let v = normalize(cross2);

    (u, v)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_2d() {
        let params = [0.0, 0.0, 1.0]; // unit circle at origin

        // t=0 should be at (1, 0)
        let p0 = evaluate_2d(&params, 0.0);
        assert!((p0[0] - 1.0).abs() < 1e-10);
        assert!(p0[1].abs() < 1e-10);

        // t=0.25 should be at (0, 1)
        let p1 = evaluate_2d(&params, 0.25);
        assert!(p1[0].abs() < 1e-10);
        assert!((p1[1] - 1.0).abs() < 1e-10);

        // t=0.5 should be at (-1, 0)
        let p2 = evaluate_2d(&params, 0.5);
        assert!((p2[0] + 1.0).abs() < 1e-10);
        assert!(p2[1].abs() < 1e-10);

        // t=0.75 should be at (0, -1)
        let p3 = evaluate_2d(&params, 0.75);
        assert!(p3[0].abs() < 1e-10);
        assert!((p3[1] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_2d_offset() {
        let params = [5.0, 3.0, 2.0]; // circle at (5, 3) with radius 2

        let p0 = evaluate_2d(&params, 0.0);
        assert!((p0[0] - 7.0).abs() < 1e-10);
        assert!((p0[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_tangent_2d() {
        let params = [0.0, 0.0, 1.0]; // unit circle

        // At t=0 (point (1,0)), tangent should point in +y direction
        let tan0 = tangent_2d(&params, 0.0);
        assert!(tan0[0].abs() < 1e-10); // x component ~0
        assert!(tan0[1] > 0.0); // y component positive

        // At t=0.25 (point (0,1)), tangent should point in -x direction
        let tan1 = tangent_2d(&params, 0.25);
        assert!(tan1[0] < 0.0); // x component negative
        assert!(tan1[1].abs() < 1e-10); // y component ~0
    }

    #[test]
    fn test_evaluate_3d_xy_plane() {
        // Circle in XY plane (normal = +Z)
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        let p0 = evaluate_3d(&params, 0.0);
        // Should be at radius 1 from center in XY plane
        let dist_xy = (p0[0] * p0[0] + p0[1] * p0[1]).sqrt();
        assert!((dist_xy - 1.0).abs() < 1e-10);
        assert!(p0[2].abs() < 1e-10); // z should be 0

        let p1 = evaluate_3d(&params, 0.25);
        let dist_xy1 = (p1[0] * p1[0] + p1[1] * p1[1]).sqrt();
        assert!((dist_xy1 - 1.0).abs() < 1e-10);
        assert!(p1[2].abs() < 1e-10);
    }

    #[test]
    fn test_build_orthonormal_frame() {
        // Test with Z-axis normal
        let (u, v) = build_orthonormal_frame([0.0, 0.0, 1.0]);

        // u and v should be unit vectors
        let u_len = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let v_len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((u_len - 1.0).abs() < 1e-10);
        assert!((v_len - 1.0).abs() < 1e-10);

        // u and v should be perpendicular
        let dot_uv = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        assert!(dot_uv.abs() < 1e-10);

        // Both should be perpendicular to normal
        let dot_un = u[2]; // dot with [0,0,1]
        let dot_vn = v[2];
        assert!(dot_un.abs() < 1e-10);
        assert!(dot_vn.abs() < 1e-10);
    }
}
