
/// Cylinder params: [px, py, pz, dx, dy, dz, r]
/// where (px, py, pz) is a point on the axis,
/// (dx, dy, dz) is the axis direction (not necessarily unit length),
/// and r is the radius.

/// Evaluate position on cylinder surface at parameters (u, v).
/// - u ∈ [0, 2π] is the angular parameter around the axis
/// - v ∈ ℝ is the parameter along the axis
///
/// Point = axis_point + v * axis_direction + r * (cos(u) * e1 + sin(u) * e2)
/// where e1, e2 are perpendicular to the axis.
pub fn evaluate(params: &[f64], u: f64, v: f64) -> [f64; 3] {
    let px = params[0];
    let py = params[1];
    let pz = params[2];
    let dx = params[3];
    let dy = params[4];
    let dz = params[5];
    let r = params[6];

    // Build orthonormal frame
    let (e1, e2) = build_orthonormal_frame([dx, dy, dz]);

    let cos_u = u.cos();
    let sin_u = u.sin();

    [
        px + v * dx + r * (cos_u * e1[0] + sin_u * e2[0]),
        py + v * dy + r * (cos_u * e1[1] + sin_u * e2[1]),
        pz + v * dz + r * (cos_u * e1[2] + sin_u * e2[2]),
    ]
}

/// Evaluate outward normal at parameters (u, v).
/// The normal is perpendicular to the axis, pointing radially outward.
pub fn normal_at(params: &[f64], u: f64, _v: f64) -> [f64; 3] {
    let dx = params[3];
    let dy = params[4];
    let dz = params[5];

    // Build orthonormal frame
    let (e1, e2) = build_orthonormal_frame([dx, dy, dz]);

    let cos_u = u.cos();
    let sin_u = u.sin();

    // Normal is just the radial component (already unit length)
    [
        cos_u * e1[0] + sin_u * e2[0],
        cos_u * e1[1] + sin_u * e2[1],
        cos_u * e1[2] + sin_u * e2[2],
    ]
}

/// Calculate signed distance from a point to the cylinder surface.
/// Returns the radial distance minus the radius.
pub fn signed_distance_to_surface(params: &[f64], point: &[f64; 3]) -> f64 {
    let px = params[0];
    let py = params[1];
    let pz = params[2];
    let dx = params[3];
    let dy = params[4];
    let dz = params[5];
    let r = params[6];

    // Normalize axis direction
    let axis_len = (dx * dx + dy * dy + dz * dz).sqrt();
    let ax = dx / axis_len;
    let ay = dy / axis_len;
    let az = dz / axis_len;

    // Vector from axis point to query point
    let qx = point[0] - px;
    let qy = point[1] - py;
    let qz = point[2] - pz;

    // Project onto axis
    let proj_len = qx * ax + qy * ay + qz * az;

    // Closest point on axis
    let closest_x = px + proj_len * ax;
    let closest_y = py + proj_len * ay;
    let closest_z = pz + proj_len * az;

    // Distance from point to axis
    let rad_x = point[0] - closest_x;
    let rad_y = point[1] - closest_y;
    let rad_z = point[2] - closest_z;
    let radial_dist = (rad_x * rad_x + rad_y * rad_y + rad_z * rad_z).sqrt();

    radial_dist - r
}

/// Build an orthonormal frame from an axis direction vector.
/// Returns two unit vectors (e1, e2) that are perpendicular to the axis and to each other.
fn build_orthonormal_frame(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let [dx, dy, dz] = axis;

    // Normalize the axis
    let axis_len = (dx * dx + dy * dy + dz * dz).sqrt();
    let ax = dx / axis_len;
    let ay = dy / axis_len;
    let az = dz / axis_len;

    // Find a vector not parallel to axis
    let candidate = if ax.abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    // e1 = normalize(candidate × axis)
    let cross1 = cross(candidate, [ax, ay, az]);
    let e1 = normalize(cross1);

    // e2 = normalize(axis × e1)
    let cross2 = cross([ax, ay, az], e1);
    let e2 = normalize(cross2);

    (e1, e2)
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
    use std::f64::consts::PI;

    #[test]
    fn test_evaluate_z_axis_cylinder() {
        // Cylinder along Z axis, radius 1
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        // At u=0, v=0, should be at radius 1 in the X direction
        let p1 = evaluate(&params, 0.0, 0.0);
        let dist_xy = (p1[0] * p1[0] + p1[1] * p1[1]).sqrt();
        assert!((dist_xy - 1.0).abs() < 1e-10);
        assert!(p1[2].abs() < 1e-10);

        // At u=π/2, v=0, should be at radius 1 in the Y direction
        let p2 = evaluate(&params, PI / 2.0, 0.0);
        let dist_xy2 = (p2[0] * p2[0] + p2[1] * p2[1]).sqrt();
        assert!((dist_xy2 - 1.0).abs() < 1e-10);
        assert!(p2[2].abs() < 1e-10);

        // At u=0, v=5, should be at radius 1 in X direction, z=5
        let p3 = evaluate(&params, 0.0, 5.0);
        let dist_xy3 = (p3[0] * p3[0] + p3[1] * p3[1]).sqrt();
        assert!((dist_xy3 - 1.0).abs() < 1e-10);
        assert!((p3[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_x_axis_cylinder() {
        // Cylinder along X axis, radius 2
        let params = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0];

        // At u=0, v=0
        let p1 = evaluate(&params, 0.0, 0.0);
        let dist_yz = (p1[1] * p1[1] + p1[2] * p1[2]).sqrt();
        assert!((dist_yz - 2.0).abs() < 1e-10);
        assert!(p1[0].abs() < 1e-10);

        // At u=0, v=3, should be at x=3
        let p2 = evaluate(&params, 0.0, 3.0);
        let dist_yz2 = (p2[1] * p2[1] + p2[2] * p2[2]).sqrt();
        assert!((dist_yz2 - 2.0).abs() < 1e-10);
        assert!((p2[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_at() {
        // Cylinder along Z axis
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        // Normal at u=0 should point in +X direction
        let n1 = normal_at(&params, 0.0, 0.0);
        let n_len = (n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]).sqrt();
        assert!((n_len - 1.0).abs() < 1e-10); // unit length
        assert!(n1[2].abs() < 1e-10); // perpendicular to Z axis

        // Normal at u=π/2 should point in +Y direction
        let n2 = normal_at(&params, PI / 2.0, 0.0);
        let n2_len = (n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]).sqrt();
        assert!((n2_len - 1.0).abs() < 1e-10);
        assert!(n2[2].abs() < 1e-10);
    }

    #[test]
    fn test_signed_distance_to_surface() {
        // Cylinder along Z axis, radius 1
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        // Point outside
        let p1 = [2.0, 0.0, 5.0];
        let dist1 = signed_distance_to_surface(&params, &p1);
        assert!((dist1 - 1.0).abs() < 1e-10);

        // Point on surface
        let p2 = [1.0, 0.0, 3.0];
        let dist2 = signed_distance_to_surface(&params, &p2);
        assert!(dist2.abs() < 1e-10);

        // Point inside
        let p3 = [0.5, 0.0, 2.0];
        let dist3 = signed_distance_to_surface(&params, &p3);
        assert!((dist3 + 0.5).abs() < 1e-10);

        // Point on axis
        let p4 = [0.0, 0.0, 10.0];
        let dist4 = signed_distance_to_surface(&params, &p4);
        assert!((dist4 + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_orthonormal_frame() {
        // Test with Z axis
        let (e1, e2) = build_orthonormal_frame([0.0, 0.0, 1.0]);

        // e1 and e2 should be unit vectors
        let e1_len = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
        let e2_len = (e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]).sqrt();
        assert!((e1_len - 1.0).abs() < 1e-10);
        assert!((e2_len - 1.0).abs() < 1e-10);

        // e1 and e2 should be perpendicular
        let dot_e1e2 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
        assert!(dot_e1e2.abs() < 1e-10);

        // Both should be perpendicular to Z axis
        let dot_e1z = e1[2];
        let dot_e2z = e2[2];
        assert!(dot_e1z.abs() < 1e-10);
        assert!(dot_e2z.abs() < 1e-10);
    }
}
