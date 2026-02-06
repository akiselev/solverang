/// Return position of a 2D point.
pub fn position_2d(params: &[f64]) -> [f64; 2] {
    [params[0], params[1]]
}

/// Return position of a 3D point.
pub fn position_3d(params: &[f64]) -> [f64; 3] {
    [params[0], params[1], params[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_2d() {
        let params = [1.5, -2.3];
        let pos = position_2d(&params);
        assert_eq!(pos, [1.5, -2.3]);
    }

    #[test]
    fn test_position_3d() {
        let params = [1.5, -2.3, 4.7];
        let pos = position_3d(&params);
        assert_eq!(pos, [1.5, -2.3, 4.7]);
    }
}
