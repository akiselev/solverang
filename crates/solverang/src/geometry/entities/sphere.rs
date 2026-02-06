
/// Sphere params: [cx, cy, cz, r]

/// Evaluate position on sphere surface at parameters (u, v).
/// Uses spherical coordinates:
/// - u ∈ [0, 2π] is the longitude (azimuthal angle)
/// - v ∈ [-π/2, π/2] is the latitude (polar angle from equator)
///
/// x = cx + r * cos(v) * cos(u)
/// y = cy + r * cos(v) * sin(u)
/// z = cz + r * sin(v)
pub fn evaluate(params: &[f64], u: f64, v: f64) -> [f64; 3] {
    let cx = params[0];
    let cy = params[1];
    let cz = params[2];
    let r = params[3];

    let cos_v = v.cos();
    let sin_v = v.sin();
    let cos_u = u.cos();
    let sin_u = u.sin();

    [
        cx + r * cos_v * cos_u,
        cy + r * cos_v * sin_u,
        cz + r * sin_v,
    ]
}

/// Evaluate outward normal at parameters (u, v).
/// For a sphere, the normal is just (point - center) / r.
pub fn normal_at(params: &[f64], u: f64, v: f64) -> [f64; 3] {
    let cx = params[0];
    let cy = params[1];
    let cz = params[2];
    let r = params[3];

    let point = evaluate(params, u, v);

    // Normal is outward from center
    let nx = (point[0] - cx) / r;
    let ny = (point[1] - cy) / r;
    let nz = (point[2] - cz) / r;

    [nx, ny, nz]
}

/// Calculate signed distance from a point to the sphere surface.
/// Positive means outside, negative means inside.
pub fn signed_distance_to_surface(params: &[f64], point: &[f64; 3]) -> f64 {
    let cx = params[0];
    let cy = params[1];
    let cz = params[2];
    let r = params[3];

    let dx = point[0] - cx;
    let dy = point[1] - cy;
    let dz = point[2] - cz;

    let dist_from_center = (dx * dx + dy * dy + dz * dz).sqrt();
    dist_from_center - r
}

/// Project a point onto the sphere surface.
pub fn project_point(params: &[f64], point: &[f64; 3]) -> [f64; 3] {
    let cx = params[0];
    let cy = params[1];
    let cz = params[2];
    let r = params[3];

    let dx = point[0] - cx;
    let dy = point[1] - cy;
    let dz = point[2] - cz;

    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    if dist < 1e-10 {
        // Point at center, return arbitrary point on surface
        return [cx + r, cy, cz];
    }

    // Normalize direction and scale by radius
    [
        cx + r * dx / dist,
        cy + r * dy / dist,
        cz + r * dz / dist,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_evaluate_cardinal_points() {
        // Unit sphere at origin
        let params = [0.0, 0.0, 0.0, 1.0];

        // Top of sphere (north pole): u=0, v=π/2
        let top = evaluate(&params, 0.0, PI / 2.0);
        assert!(top[0].abs() < 1e-10);
        assert!(top[1].abs() < 1e-10);
        assert!((top[2] - 1.0).abs() < 1e-10);

        // Bottom of sphere (south pole): u=0, v=-π/2
        let bottom = evaluate(&params, 0.0, -PI / 2.0);
        assert!(bottom[0].abs() < 1e-10);
        assert!(bottom[1].abs() < 1e-10);
        assert!((bottom[2] + 1.0).abs() < 1e-10);

        // Point on equator: u=0, v=0 -> (1, 0, 0)
        let eq1 = evaluate(&params, 0.0, 0.0);
        assert!((eq1[0] - 1.0).abs() < 1e-10);
        assert!(eq1[1].abs() < 1e-10);
        assert!(eq1[2].abs() < 1e-10);

        // Point on equator: u=π/2, v=0 -> (0, 1, 0)
        let eq2 = evaluate(&params, PI / 2.0, 0.0);
        assert!(eq2[0].abs() < 1e-10);
        assert!((eq2[1] - 1.0).abs() < 1e-10);
        assert!(eq2[2].abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_offset_sphere() {
        // Sphere at (5, 3, 2) with radius 2
        let params = [5.0, 3.0, 2.0, 2.0];

        // Top of sphere
        let top = evaluate(&params, 0.0, PI / 2.0);
        assert!((top[0] - 5.0).abs() < 1e-10);
        assert!((top[1] - 3.0).abs() < 1e-10);
        assert!((top[2] - 4.0).abs() < 1e-10); // 2 + 2
    }

    #[test]
    fn test_normal_at() {
        // Unit sphere at origin
        let params = [0.0, 0.0, 0.0, 1.0];

        // Normal at top should point up
        let n_top = normal_at(&params, 0.0, PI / 2.0);
        assert!(n_top[0].abs() < 1e-10);
        assert!(n_top[1].abs() < 1e-10);
        assert!((n_top[2] - 1.0).abs() < 1e-10);

        // Normal at (1,0,0) should point in +x direction
        let n_x = normal_at(&params, 0.0, 0.0);
        assert!((n_x[0] - 1.0).abs() < 1e-10);
        assert!(n_x[1].abs() < 1e-10);
        assert!(n_x[2].abs() < 1e-10);
    }

    #[test]
    fn test_signed_distance_to_surface() {
        // Unit sphere at origin
        let params = [0.0, 0.0, 0.0, 1.0];

        // Point outside
        let p1 = [2.0, 0.0, 0.0];
        let dist1 = signed_distance_to_surface(&params, &p1);
        assert!((dist1 - 1.0).abs() < 1e-10);

        // Point on surface
        let p2 = [1.0, 0.0, 0.0];
        let dist2 = signed_distance_to_surface(&params, &p2);
        assert!(dist2.abs() < 1e-10);

        // Point inside
        let p3 = [0.5, 0.0, 0.0];
        let dist3 = signed_distance_to_surface(&params, &p3);
        assert!((dist3 + 0.5).abs() < 1e-10);

        // Point at center
        let p4 = [0.0, 0.0, 0.0];
        let dist4 = signed_distance_to_surface(&params, &p4);
        assert!((dist4 + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_project_point() {
        // Unit sphere at origin
        let params = [0.0, 0.0, 0.0, 1.0];

        // Project a point outside
        let p1 = [2.0, 0.0, 0.0];
        let proj1 = project_point(&params, &p1);
        assert!((proj1[0] - 1.0).abs() < 1e-10);
        assert!(proj1[1].abs() < 1e-10);
        assert!(proj1[2].abs() < 1e-10);

        // Project a point inside
        let p2 = [0.5, 0.0, 0.0];
        let proj2 = project_point(&params, &p2);
        assert!((proj2[0] - 1.0).abs() < 1e-10);
        assert!(proj2[1].abs() < 1e-10);
        assert!(proj2[2].abs() < 1e-10);

        // Project a diagonal point
        let sqrt3 = 3.0_f64.sqrt();
        let p3 = [2.0, 2.0, 2.0];
        let proj3 = project_point(&params, &p3);
        // Should be at (1/√3, 1/√3, 1/√3)
        let expected = 1.0 / sqrt3;
        assert!((proj3[0] - expected).abs() < 1e-10);
        assert!((proj3[1] - expected).abs() < 1e-10);
        assert!((proj3[2] - expected).abs() < 1e-10);
    }
}
