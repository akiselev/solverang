/// Plane params: [px, py, pz, nx, ny, nz]
/// where (px, py, pz) is a point on the plane and (nx, ny, nz) is the normal vector.

/// Return the normal vector of the plane (not normalized).
pub fn normal(params: &[f64]) -> [f64; 3] {
    [params[3], params[4], params[5]]
}

/// Return a point on the plane.
pub fn point_on_plane(params: &[f64]) -> [f64; 3] {
    [params[0], params[1], params[2]]
}

/// Calculate signed distance from a point to the plane.
/// Returns n · (point - p) / |n|, where n is the normal and p is a point on the plane.
/// Positive distance means the point is on the side of the normal.
pub fn signed_distance(params: &[f64], point: &[f64; 3]) -> f64 {
    let px = params[0];
    let py = params[1];
    let pz = params[2];
    let nx = params[3];
    let ny = params[4];
    let nz = params[5];

    // Vector from plane point to query point
    let dx = point[0] - px;
    let dy = point[1] - py;
    let dz = point[2] - pz;

    // Dot product with normal
    let dot = nx * dx + ny * dy + nz * dz;

    // Normalize by length of normal
    let n_len = (nx * nx + ny * ny + nz * nz).sqrt();

    if n_len < 1e-10 {
        0.0 // Degenerate case
    } else {
        dot / n_len
    }
}

/// Calculate unsigned distance from a point to the plane.
pub fn distance(params: &[f64], point: &[f64; 3]) -> f64 {
    signed_distance(params, point).abs()
}

/// Project a point onto the plane.
/// Returns the closest point on the plane to the given point.
pub fn project_point(params: &[f64], point: &[f64; 3]) -> [f64; 3] {
    let dist = signed_distance(params, point);

    let nx = params[3];
    let ny = params[4];
    let nz = params[5];

    // Normalize the normal
    let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
    let nx_norm = nx / n_len;
    let ny_norm = ny / n_len;
    let nz_norm = nz / n_len;

    // Project by moving along the normal by -dist
    [
        point[0] - dist * nx_norm,
        point[1] - dist * ny_norm,
        point[2] - dist * nz_norm,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal() {
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let n = normal(&params);
        assert_eq!(n, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_point_on_plane() {
        let params = [1.0, 2.0, 3.0, 0.0, 0.0, 1.0];
        let p = point_on_plane(&params);
        assert_eq!(p, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_signed_distance_xy_plane() {
        // XY plane at z=0 with normal pointing up
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        // Point above the plane
        let p1 = [1.0, 2.0, 5.0];
        let dist1 = signed_distance(&params, &p1);
        assert!((dist1 - 5.0).abs() < 1e-10);

        // Point below the plane
        let p2 = [1.0, 2.0, -3.0];
        let dist2 = signed_distance(&params, &p2);
        assert!((dist2 + 3.0).abs() < 1e-10);

        // Point on the plane
        let p3 = [1.0, 2.0, 0.0];
        let dist3 = signed_distance(&params, &p3);
        assert!(dist3.abs() < 1e-10);
    }

    #[test]
    fn test_signed_distance_offset_plane() {
        // Plane at z=10 with normal pointing up
        let params = [0.0, 0.0, 10.0, 0.0, 0.0, 1.0];

        // Point above the plane
        let p1 = [0.0, 0.0, 15.0];
        let dist1 = signed_distance(&params, &p1);
        assert!((dist1 - 5.0).abs() < 1e-10);

        // Point on the plane
        let p2 = [5.0, 3.0, 10.0];
        let dist2 = signed_distance(&params, &p2);
        assert!(dist2.abs() < 1e-10);
    }

    #[test]
    fn test_distance() {
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        let p1 = [0.0, 0.0, 5.0];
        let dist1 = distance(&params, &p1);
        assert!((dist1 - 5.0).abs() < 1e-10);

        let p2 = [0.0, 0.0, -3.0];
        let dist2 = distance(&params, &p2);
        assert!((dist2 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_project_point() {
        // XY plane at z=0
        let params = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        // Project a point above the plane
        let p1 = [3.0, 4.0, 5.0];
        let proj1 = project_point(&params, &p1);
        assert!((proj1[0] - 3.0).abs() < 1e-10);
        assert!((proj1[1] - 4.0).abs() < 1e-10);
        assert!(proj1[2].abs() < 1e-10);

        // Project a point below the plane
        let p2 = [1.0, 2.0, -7.0];
        let proj2 = project_point(&params, &p2);
        assert!((proj2[0] - 1.0).abs() < 1e-10);
        assert!((proj2[1] - 2.0).abs() < 1e-10);
        assert!(proj2[2].abs() < 1e-10);
    }

    #[test]
    fn test_project_point_tilted_plane() {
        // Plane through origin with normal (1, 0, 0) - YZ plane
        let params = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let p = [5.0, 3.0, 4.0];
        let proj = project_point(&params, &p);

        // Should project to (0, 3, 4)
        assert!(proj[0].abs() < 1e-10);
        assert!((proj[1] - 3.0).abs() < 1e-10);
        assert!((proj[2] - 4.0).abs() < 1e-10);
    }
}
