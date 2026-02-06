
/// Cone params: [px, py, pz, dx, dy, dz, half_angle]
/// where (px, py, pz) is the apex,
/// (dx, dy, dz) is the axis direction,
/// and half_angle is the half-angle of the cone (angle from axis to surface).

/// Evaluate position on cone surface at parameters (u, v).
/// - u ∈ [0, 2π] is the angular parameter around the axis
/// - v ∈ ℝ is the parameter along the axis (distance from apex)
///
/// At distance v from apex, the radius is r(v) = v * tan(half_angle).
/// Point = apex + v * axis_direction + r(v) * (cos(u) * e1 + sin(u) * e2)
/// where e1, e2 are perpendicular to the axis.
pub fn evaluate(params: &[f64], u: f64, v: f64) -> [f64; 3] {
    let px = params[0];
    let py = params[1];
    let pz = params[2];
    let dx = params[3];
    let dy = params[4];
    let dz = params[5];
    let half_angle = params[6];

    // Normalize axis direction
    let axis_len = (dx * dx + dy * dy + dz * dz).sqrt();
    let ax = dx / axis_len;
    let ay = dy / axis_len;
    let az = dz / axis_len;

    // Build orthonormal frame
    let (e1, e2) = build_orthonormal_frame([ax, ay, az]);

    // Radius at parameter v
    let r = v * half_angle.tan();

    let cos_u = u.cos();
    let sin_u = u.sin();

    [
        px + v * ax + r * (cos_u * e1[0] + sin_u * e2[0]),
        py + v * ay + r * (cos_u * e1[1] + sin_u * e2[1]),
        pz + v * az + r * (cos_u * e1[2] + sin_u * e2[2]),
    ]
}

/// Evaluate outward normal at parameters (u, v).
/// The normal is perpendicular to both the tangent along v and the tangent around u.
pub fn normal_at(params: &[f64], u: f64, _v: f64) -> [f64; 3] {
    let dx = params[3];
    let dy = params[4];
    let dz = params[5];
    let half_angle = params[6];

    // Normalize axis direction
    let axis_len = (dx * dx + dy * dy + dz * dz).sqrt();
    let ax = dx / axis_len;
    let ay = dy / axis_len;
    let az = dz / axis_len;

    // Build orthonormal frame
    let (e1, e2) = build_orthonormal_frame([ax, ay, az]);

    let cos_u = u.cos();
    let sin_u = u.sin();

    // Radial direction at this angle
    let rad_x = cos_u * e1[0] + sin_u * e2[0];
    let rad_y = cos_u * e1[1] + sin_u * e2[1];
    let rad_z = cos_u * e1[2] + sin_u * e2[2];

    // For a cone, the normal makes an angle with the radial direction
    // Normal = cos(half_angle) * radial + sin(half_angle) * (-axis)
    let cos_ha = half_angle.cos();
    let sin_ha = half_angle.sin();

    let nx = cos_ha * rad_x - sin_ha * ax;
    let ny = cos_ha * rad_y - sin_ha * ay;
    let nz = cos_ha * rad_z - sin_ha * az;

    // Normalize
    let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
    [nx / n_len, ny / n_len, nz / n_len]
}

/// Calculate signed distance from a point to the cone surface.
/// This is an approximation that works well for points not too far from the surface.
pub fn signed_distance_to_surface(params: &[f64], point: &[f64; 3]) -> f64 {
    let px = params[0];
    let py = params[1];
    let pz = params[2];
    let dx = params[3];
    let dy = params[4];
    let dz = params[5];
    let half_angle = params[6];

    // Normalize axis direction
    let axis_len = (dx * dx + dy * dy + dz * dz).sqrt();
    let ax = dx / axis_len;
    let ay = dy / axis_len;
    let az = dz / axis_len;

    // Vector from apex to query point
    let qx = point[0] - px;
    let qy = point[1] - py;
    let qz = point[2] - pz;

    // Project onto axis
    let proj_len = qx * ax + qy * ay + qz * az;

    // If behind apex (negative v), special handling
    if proj_len < 0.0 {
        // Distance to apex
        return (qx * qx + qy * qy + qz * qz).sqrt();
    }

    // Point on axis at this distance
    let axis_x = px + proj_len * ax;
    let axis_y = py + proj_len * ay;
    let axis_z = pz + proj_len * az;

    // Distance from point to axis
    let rad_x = point[0] - axis_x;
    let rad_y = point[1] - axis_y;
    let rad_z = point[2] - axis_z;
    let radial_dist = (rad_x * rad_x + rad_y * rad_y + rad_z * rad_z).sqrt();

    // Expected radius at this axial position
    let expected_radius = proj_len * half_angle.tan();

    radial_dist - expected_radius
}

/// Build an orthonormal frame from an axis direction vector.
fn build_orthonormal_frame(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let [ax, ay, az] = axis;

    // Find a vector not parallel to axis
    let candidate = if ax.abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    // e1 = normalize(candidate - (candidate·axis)*axis)  (Gram-Schmidt)
    let dot = candidate[0] * ax + candidate[1] * ay + candidate[2] * az;
    let e1 = normalize([
        candidate[0] - dot * ax,
        candidate[1] - dot * ay,
        candidate[2] - dot * az,
    ]);

    // e2 = normalize(axis × e1)
    let e2 = normalize(cross([ax, ay, az], e1));

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
    fn test_evaluate_z_axis_cone() {
        // Cone along Z axis with half-angle 45° (π/4)
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, PI / 4.0];

        // At v=0, should be at apex
        let p0 = evaluate(&params, 0.0, 0.0);
        assert!(p0[0].abs() < 1e-10);
        assert!(p0[1].abs() < 1e-10);
        assert!(p0[2].abs() < 1e-10);

        // At v=1, radius should be tan(45°) = 1
        let p1 = evaluate(&params, 0.0, 1.0);
        let dist_xy = (p1[0] * p1[0] + p1[1] * p1[1]).sqrt();
        assert!((dist_xy - 1.0).abs() < 1e-10);
        assert!((p1[2] - 1.0).abs() < 1e-10);

        // At v=2, radius should be 2
        let p2 = evaluate(&params, 0.0, 2.0);
        let dist_xy2 = (p2[0] * p2[0] + p2[1] * p2[1]).sqrt();
        assert!((dist_xy2 - 2.0).abs() < 1e-10);
        assert!((p2[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_cone_angles() {
        // Cone along Z axis with half-angle 45°
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, PI / 4.0];

        // At v=1, u=0, should be in +X direction
        let p1 = evaluate(&params, 0.0, 1.0);
        assert!(p1[0] > 0.0);
        assert!(p1[1].abs() < 1e-10);

        // At v=1, u=π/2, should be in +Y direction
        let p2 = evaluate(&params, PI / 2.0, 1.0);
        assert!(p2[0].abs() < 1e-10);
        assert!(p2[1] > 0.0);

        // At v=1, u=π, should be in -X direction
        let p3 = evaluate(&params, PI, 1.0);
        assert!(p3[0] < 0.0);
        assert!(p3[1].abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_narrow_cone() {
        // Cone with small half-angle (30°)
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, PI / 6.0];

        // At v=1, radius should be tan(30°) ≈ 0.577
        let p1 = evaluate(&params, 0.0, 1.0);
        let dist_xy = (p1[0] * p1[0] + p1[1] * p1[1]).sqrt();
        let expected_radius = (PI / 6.0).tan();
        assert!((dist_xy - expected_radius).abs() < 1e-10);
    }

    #[test]
    fn test_normal_at() {
        // Cone along Z axis with half-angle 45°
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, PI / 4.0];

        // Normal at v=1, u=0
        let n1 = normal_at(&params, 0.0, 1.0);
        let n_len = (n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]).sqrt();
        assert!((n_len - 1.0).abs() < 1e-10); // unit length

        // For a 45° cone, the normal should make a 45° angle with the radial direction
        // This means the normal has equal X and Z components (for u=0)
        let sqrt2 = 2.0_f64.sqrt();
        assert!((n1[0] - sqrt2 / 2.0).abs() < 1e-9);
        assert!(n1[1].abs() < 1e-10);
        assert!((n1[2] + sqrt2 / 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_signed_distance_to_surface() {
        // Cone along Z axis with half-angle 45°
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, PI / 4.0];

        // Point on surface at v=1, u=0 (should be at (1, 0, 1))
        let p1 = [1.0, 0.0, 1.0];
        let dist1 = signed_distance_to_surface(&params, &p1);
        assert!(dist1.abs() < 1e-10);

        // Point outside (larger radius)
        let p2 = [2.0, 0.0, 1.0];
        let dist2 = signed_distance_to_surface(&params, &p2);
        assert!(dist2 > 0.0);
        assert!((dist2 - 1.0).abs() < 1e-10);

        // Point inside (smaller radius)
        let p3 = [0.5, 0.0, 1.0];
        let dist3 = signed_distance_to_surface(&params, &p3);
        assert!(dist3 < 0.0);
    }

    #[test]
    fn test_signed_distance_apex() {
        // Cone along Z axis
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, PI / 4.0];

        // Point at apex
        let p = [0.0, 0.0, 0.0];
        let dist = signed_distance_to_surface(&params, &p);
        assert!(dist.abs() < 1e-10);
    }
}
