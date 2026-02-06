/// Evaluate position at parameter t ∈ [0, 1] on a line.
/// Line params: [x1, y1, ..., x2, y2, ...] — works for 2D (4 params) or 3D (6 params).
/// Returns lerp between start and end: start + t * (end - start).
pub fn evaluate(params: &[f64], t: f64) -> Vec<f64> {
    let dim = params.len() / 2;
    let mut result = Vec::with_capacity(dim);

    for i in 0..dim {
        let start = params[i];
        let end = params[dim + i];
        result.push(start + t * (end - start));
    }

    result
}

/// Return tangent vector (constant for lines).
/// Tangent is (end - start), not normalized.
pub fn tangent(params: &[f64]) -> Vec<f64> {
    let dim = params.len() / 2;
    let mut result = Vec::with_capacity(dim);

    for i in 0..dim {
        let start = params[i];
        let end = params[dim + i];
        result.push(end - start);
    }

    result
}

/// Return length of the line segment.
pub fn length(params: &[f64]) -> f64 {
    let dim = params.len() / 2;
    let mut sum_sq = 0.0;

    for i in 0..dim {
        let start = params[i];
        let end = params[dim + i];
        let diff = end - start;
        sum_sq += diff * diff;
    }

    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_2d() {
        let params = [0.0, 0.0, 10.0, 0.0];

        let p0 = evaluate(&params, 0.0);
        assert_eq!(p0, vec![0.0, 0.0]);

        let p1 = evaluate(&params, 1.0);
        assert_eq!(p1, vec![10.0, 0.0]);

        let p_mid = evaluate(&params, 0.5);
        assert_eq!(p_mid, vec![5.0, 0.0]);
    }

    #[test]
    fn test_evaluate_3d() {
        let params = [0.0, 0.0, 0.0, 10.0, 5.0, 3.0];

        let p0 = evaluate(&params, 0.0);
        assert_eq!(p0, vec![0.0, 0.0, 0.0]);

        let p1 = evaluate(&params, 1.0);
        assert_eq!(p1, vec![10.0, 5.0, 3.0]);

        let p_mid = evaluate(&params, 0.5);
        assert_eq!(p_mid, vec![5.0, 2.5, 1.5]);
    }

    #[test]
    fn test_tangent_2d() {
        let params = [1.0, 2.0, 4.0, 6.0];
        let tan = tangent(&params);
        assert_eq!(tan, vec![3.0, 4.0]);
    }

    #[test]
    fn test_tangent_3d() {
        let params = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tan = tangent(&params);
        assert_eq!(tan, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_length_2d() {
        let params = [0.0, 0.0, 3.0, 4.0];
        let len = length(&params);
        assert_eq!(len, 5.0);
    }

    #[test]
    fn test_length_3d() {
        let params = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let len = length(&params);
        assert!((len - 3.0_f64.sqrt()).abs() < 1e-10);
    }
}
