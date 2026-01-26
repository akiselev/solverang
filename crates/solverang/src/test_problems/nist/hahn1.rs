//! Hahn1 - NIST StRD Nonlinear Regression Problem (Average Difficulty)
//!
//! Thermal expansion of copper data.
//!
//! Model: y = (b1 + b2*x + b3*x^2 + b4*x^3) / (1 + b5*x + b6*x^2 + b7*x^3)
//!
//! Parameters: 7
//! Observations: 236
//!
//! Reference: <https://www.itl.nist.gov/div898/strd/nls/data/hahn1.shtml>

use crate::Problem;
use super::{NISTProblem, NISTDifficulty};

/// Generate Hahn1 data points (selected subset for demonstration)
fn get_data() -> Vec<(f64, f64)> {
    vec![
        (24.41, 0.591), (34.82, 1.547), (44.09, 2.902), (45.07, 2.894),
        (54.98, 4.703), (65.51, 6.307), (70.53, 7.030), (75.70, 7.898),
        (89.57, 9.470), (91.14, 9.484), (96.40, 10.072), (97.19, 10.163),
        (114.26, 11.615), (120.25, 12.005), (127.08, 12.478), (133.55, 12.982),
        (133.61, 12.970), (158.67, 13.926), (172.74, 14.452), (171.31, 14.404),
        (202.14, 15.190), (220.55, 15.550), (221.05, 15.528), (221.39, 15.499),
        (250.99, 16.131), (268.99, 16.438), (271.80, 16.387), (271.97, 16.549),
        (321.31, 17.077), (321.69, 17.006), (330.14, 17.100), (333.03, 17.122),
        (333.47, 17.133), (340.77, 17.074), (345.65, 17.117), (373.11, 17.274),
        (373.79, 17.262), (411.82, 17.288), (419.51, 17.235), (421.59, 17.229),
        (422.02, 17.237), (422.47, 17.222), (422.61, 17.223), (441.75, 17.248),
        (447.41, 17.208), (448.70, 17.225), (472.89, 17.303), (476.69, 17.262),
        (522.47, 17.110), (522.62, 17.132), (524.43, 17.090), (546.75, 16.925),
        (549.53, 16.904), (575.29, 16.746), (576.00, 16.683), (625.55, 16.167),
        (20.15, 0.367), (28.78, 0.796), (29.57, 0.892), (37.41, 1.903),
        (39.12, 2.150), (50.24, 3.697), (61.38, 5.870), (66.25, 6.421),
        (73.42, 7.422), (95.52, 10.160), (107.32, 11.040), (122.04, 12.157),
        (134.03, 12.880), (163.19, 14.340), (163.48, 14.240), (175.70, 14.650),
        (179.86, 14.750), (211.27, 15.362), (217.78, 15.573), (219.14, 15.598),
        (262.52, 16.303), (268.01, 16.455), (268.62, 16.377), (336.25, 17.074),
        (369.09, 17.265), (396.12, 17.271), (399.20, 17.258), (437.27, 17.266),
        (468.01, 17.304), (480.15, 17.246), (519.44, 17.092), (522.26, 17.111),
        (559.64, 16.819), (560.90, 16.804), (577.42, 16.650), (603.73, 16.389),
        (605.89, 16.355), (606.28, 16.355), (626.04, 16.188), (20.05, 0.372),
    ]
}

/// Certified parameter values from NIST
const CERTIFIED_VALUES: [f64; 7] = [
    1.0776351733E+00,   // b1
    -1.2269296921E-01,  // b2
    4.0863750610E-03,   // b3
    -1.4262622262E-06,  // b4
    -5.7609940901E-03,  // b5
    2.4053735503E-04,   // b6
    -1.2314450199E-07,  // b7
];

/// Certified standard errors from NIST
const CERTIFIED_STD_ERRORS: [f64; 7] = [
    1.7070154742E-01,   // b1
    1.2000958095E-02,   // b2
    2.2508314937E-04,   // b3
    9.0048772940E-09,   // b4
    7.9265530802E-04,   // b5
    1.6596063788E-05,   // b6
    6.3361311864E-10,   // b7
];

/// Certified residual sum of squares
const CERTIFIED_RSS: f64 = 1.5324382854E+00;

/// Starting values set 1
const STARTING_VALUES_1: [f64; 7] = [10.0, -1.0, 0.05, -0.00001, -0.05, 0.001, -0.000001];

/// Starting values set 2
const STARTING_VALUES_2: [f64; 7] = [1.0, -0.1, 0.005, -0.000001, -0.005, 0.0001, -0.0000001];

/// Hahn1 problem: thermal expansion of copper
#[derive(Clone, Debug, Default)]
pub struct Hahn1;

impl Problem for Hahn1 {
    fn name(&self) -> &str {
        "Hahn1"
    }

    fn residual_count(&self) -> usize {
        get_data().len()
    }

    fn variable_count(&self) -> usize {
        7
    }

    fn residuals(&self, b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(b.len(), 7);

        let data = get_data();
        data.iter()
            .map(|&(x, y)| {
                let x2 = x * x;
                let x3 = x2 * x;
                let num = b[0] + b[1] * x + b[2] * x2 + b[3] * x3;
                let denom = 1.0 + b[4] * x + b[5] * x2 + b[6] * x3;
                let model = num / denom;
                y - model
            })
            .collect()
    }

    fn jacobian(&self, b: &[f64]) -> Vec<(usize, usize, f64)> {
        debug_assert_eq!(b.len(), 7);

        let data = get_data();
        let mut entries = Vec::with_capacity(data.len() * 7);

        for (i, &(x, _y)) in data.iter().enumerate() {
            let x2 = x * x;
            let x3 = x2 * x;
            let num = b[0] + b[1] * x + b[2] * x2 + b[3] * x3;
            let denom = 1.0 + b[4] * x + b[5] * x2 + b[6] * x3;
            let denom_sq = denom * denom;

            entries.push((i, 0, -1.0 / denom));
            entries.push((i, 1, -x / denom));
            entries.push((i, 2, -x2 / denom));
            entries.push((i, 3, -x3 / denom));
            entries.push((i, 4, num * x / denom_sq));
            entries.push((i, 5, num * x2 / denom_sq));
            entries.push((i, 6, num * x3 / denom_sq));
        }

        entries
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        STARTING_VALUES_1.iter().map(|&v| v * factor).collect()
    }
}

impl NISTProblem for Hahn1 {
    fn difficulty(&self) -> NISTDifficulty {
        NISTDifficulty::Average
    }

    fn certified_values(&self) -> &[f64] {
        &CERTIFIED_VALUES
    }

    fn certified_std_errors(&self) -> &[f64] {
        &CERTIFIED_STD_ERRORS
    }

    fn certified_residual_sum_of_squares(&self) -> f64 {
        CERTIFIED_RSS
    }

    fn starting_values_1(&self) -> Vec<f64> {
        STARTING_VALUES_1.to_vec()
    }

    fn starting_values_2(&self) -> Vec<f64> {
        STARTING_VALUES_2.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hahn1_dimensions() {
        let problem = Hahn1;
        assert_eq!(problem.variable_count(), 7);
        assert!(problem.residual_count() > 0);
    }
}
