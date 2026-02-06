
/// Torus params: [cx, cy, cz, nx, ny, nz, R, r]
/// where (cx, cy, cz) is the center,
/// (nx, ny, nz) is the axis normal (perpendicular to the plane of the major circle),
/// R is the major radius (distance from center to tube center),
/// r is the minor radius (tube radius).

/// Evaluate position on torus surface at parameters (u, v).
/// - u ∈ [0, 2π] is the angle around the major circle
/// - v ∈ [0, 2π] is the angle around the minor circle (tube)
///
/// Standard torus parameterization:
/// Point = center + (R + r*cos(v)) * (cos(u)*e1 + sin(u)*e2) + r*sin(v)*normal
/// where e1, e2 are in the plane perpendicular to the normal.
pub fn evaluate(params: &[f64], u: f64, v: f64) -> [f64; 3] {
    let cx = params[0];
    let cy = params[1];
    let cz = params[2];
    let nx = params[3];
    let ny = params[4];
    let nz = params[5];
    let big_r = params[6];
    let small_r = params[7];

    // Normalize the normal
    let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
    let n = [nx / n_len, ny / n_len, nz / n_len];

    // Build orthonormal frame in the plane
    let (e1, e2) = build_orthonormal_frame(n);

    let cos_u = u.cos();
    let sin_u = u.sin();
    let cos_v = v.cos();
    let sin_v = v.sin();

    // Point on major circle at angle u with radius R + r*cos(v)
    let radius = big_r + small_r * cos_v;

    [
        cx + radius * (cos_u * e1[0] + sin_u * e2[0]) + small_r * sin_v * n[0],
        cy + radius * (cos_u * e1[1] + sin_u * e2[1]) + small_r * sin_v * n[1],
        cz + radius * (cos_u * e1[2] + sin_u * e2[2]) + small_r * sin_v * n[2],
    ]
}

/// Evaluate outward normal at parameters (u, v).
/// The normal points radially outward from the tube center.
pub fn normal_at(params: &[f64], u: f64, v: f64) -> [f64; 3] {
    let nx = params[3];
    let ny = params[4];
    let nz = params[5];

    // Normalize the normal
    let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
    let n = [nx / n_len, ny / n_len, nz / n_len];

    // Build orthonormal frame
    let (e1, e2) = build_orthonormal_frame(n);

    let cos_u = u.cos();
    let sin_u = u.sin();
    let cos_v = v.cos();
    let sin_v = v.sin();

    // Direction from tube center to surface point
    // The tube center is at angle u on the major circle
    let radial_in_plane_x = cos_u * e1[0] + sin_u * e2[0];
    let radial_in_plane_y = cos_u * e1[1] + sin_u * e2[1];
    let radial_in_plane_z = cos_u * e1[2] + sin_u * e2[2];

    // Normal = cos(v) * radial_direction + sin(v) * axis_normal
    let norm_x = cos_v * radial_in_plane_x + sin_v * n[0];
    let norm_y = cos_v * radial_in_plane_y + sin_v * n[1];
    let norm_z = cos_v * radial_in_plane_z + sin_v * n[2];

    // Already unit length by construction
    [norm_x, norm_y, norm_z]
}

/// Calculate signed distance from a point to the torus surface (approximate).
/// This uses the implicit equation: (sqrt(x² + y²) - R)² + z² - r² = 0
/// (for a torus in the XY plane with axis along Z).
pub fn signed_distance_to_surface(params: &[f64], point: &[f64; 3]) -> f64 {
    let cx = params[0];
    let cy = params[1];
    let cz = params[2];
    let nx = params[3];
    let ny = params[4];
    let nz = params[5];
    let big_r = params[6];
    let small_r = params[7];

    // Normalize the normal
    let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
    let ax = nx / n_len;
    let ay = ny / n_len;
    let az = nz / n_len;

    // Transform point to torus-local coordinates
    // Vector from center to point
    let qx = point[0] - cx;
    let qy = point[1] - cy;
    let qz = point[2] - cz;

    // Project onto axis to get axial distance
    let axial_dist = qx * ax + qy * ay + qz * az;

    // Radial distance in the plane (distance from axis)
    let closest_on_axis_x = qx - axial_dist * ax;
    let closest_on_axis_y = qy - axial_dist * ay;
    let closest_on_axis_z = qz - axial_dist * az;
    let radial_plane_dist =
        (closest_on_axis_x * closest_on_axis_x + closest_on_axis_y * closest_on_axis_y + closest_on_axis_z * closest_on_axis_z).sqrt();

    // Distance from point to major circle
    let dist_to_major_circle = (radial_plane_dist - big_r).abs();

    // Distance to torus surface
    let dist_to_surface = (dist_to_major_circle * dist_to_major_circle + axial_dist * axial_dist).sqrt();

    // Sign: positive if outside, negative if inside
    let sign = if dist_to_surface > small_r {
        1.0
    } else {
        -1.0
    };

    sign * (dist_to_surface - small_r).abs()
}

/// Build an orthonormal frame from a normal vector.
/// Returns two unit vectors (e1, e2) that are perpendicular to the normal and to each other.
fn build_orthonormal_frame(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let [nx, _ny, _nz] = normal;

    // Find a vector not parallel to normal
    let candidate = if nx.abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    // e1 = normalize(candidate - (candidate·normal)*normal)  (Gram-Schmidt)
    let dot = candidate[0] * normal[0] + candidate[1] * normal[1] + candidate[2] * normal[2];
    let e1 = normalize([
        candidate[0] - dot * normal[0],
        candidate[1] - dot * normal[1],
        candidate[2] - dot * normal[2],
    ]);

    // e2 = normalize(normal × e1)
    let e2 = normalize(cross(normal, e1));

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
    fn test_evaluate_z_axis_torus() {
        // Torus in XY plane with axis along Z, R=2, r=1
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        // At u=0, v=0: outermost point in +X direction
        // Should be at (R+r, 0, 0) = (3, 0, 0)
        let p1 = evaluate(&params, 0.0, 0.0);
        assert!((p1[0] - 3.0).abs() < 1e-10);
        assert!(p1[1].abs() < 1e-10);
        assert!(p1[2].abs() < 1e-10);

        // At u=0, v=π: innermost point in +X direction
        // Should be at (R-r, 0, 0) = (1, 0, 0)
        let p2 = evaluate(&params, 0.0, PI);
        assert!((p2[0] - 1.0).abs() < 1e-10);
        assert!(p2[1].abs() < 1e-10);
        assert!(p2[2].abs() < 1e-10);

        // At u=0, v=π/2: top point in +X direction
        // Should be at (R, 0, r) = (2, 0, 1)
        let p3 = evaluate(&params, 0.0, PI / 2.0);
        assert!((p3[0] - 2.0).abs() < 1e-10);
        assert!(p3[1].abs() < 1e-10);
        assert!((p3[2] - 1.0).abs() < 1e-10);

        // At u=0, v=-π/2: bottom point in +X direction
        // Should be at (R, 0, -r) = (2, 0, -1)
        let p4 = evaluate(&params, 0.0, -PI / 2.0);
        assert!((p4[0] - 2.0).abs() < 1e-10);
        assert!(p4[1].abs() < 1e-10);
        assert!((p4[2] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_rotated_torus() {
        // Torus with various u angles
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        // At u=π/2, v=0: outermost point in +Y direction
        // Should be at (0, R+r, 0) = (0, 3, 0)
        let p1 = evaluate(&params, PI / 2.0, 0.0);
        assert!(p1[0].abs() < 1e-10);
        assert!((p1[1] - 3.0).abs() < 1e-10);
        assert!(p1[2].abs() < 1e-10);

        // At u=π, v=0: outermost point in -X direction
        // Should be at (-R-r, 0, 0) = (-3, 0, 0)
        let p2 = evaluate(&params, PI, 0.0);
        assert!((p2[0] + 3.0).abs() < 1e-10);
        assert!(p2[1].abs() < 1e-10);
        assert!(p2[2].abs() < 1e-10);
    }

    #[test]
    fn test_normal_at() {
        // Torus in XY plane
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        // At u=0, v=0: normal should point in +X direction
        let n1 = normal_at(&params, 0.0, 0.0);
        let n_len = (n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]).sqrt();
        assert!((n_len - 1.0).abs() < 1e-10); // unit length
        assert!((n1[0] - 1.0).abs() < 1e-10);
        assert!(n1[1].abs() < 1e-10);
        assert!(n1[2].abs() < 1e-10);

        // At u=0, v=π: normal should point in -X direction
        let n2 = normal_at(&params, 0.0, PI);
        assert!((n2[0] + 1.0).abs() < 1e-10);
        assert!(n2[1].abs() < 1e-10);
        assert!(n2[2].abs() < 1e-10);

        // At u=0, v=π/2: normal should point in +Z direction
        let n3 = normal_at(&params, 0.0, PI / 2.0);
        assert!(n3[0].abs() < 1e-10);
        assert!(n3[1].abs() < 1e-10);
        assert!((n3[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_signed_distance_to_surface() {
        // Torus in XY plane with R=2, r=1
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        // Point on surface (outermost)
        let p1 = [3.0, 0.0, 0.0];
        let dist1 = signed_distance_to_surface(&params, &p1);
        assert!(dist1.abs() < 1e-10);

        // Point outside
        let p2 = [4.0, 0.0, 0.0];
        let dist2 = signed_distance_to_surface(&params, &p2);
        assert!(dist2 > 0.0);

        // Point on surface (innermost)
        let p3 = [1.0, 0.0, 0.0];
        let dist3 = signed_distance_to_surface(&params, &p3);
        assert!(dist3.abs() < 1e-10);

        // Point on surface (top)
        let p4 = [2.0, 0.0, 1.0];
        let dist4 = signed_distance_to_surface(&params, &p4);
        assert!(dist4.abs() < 1e-10);
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
