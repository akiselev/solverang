//! Meta-constraints for parameter relationships.

use crate::geometry::params::ConstraintId;
use crate::geometry::constraint::{Constraint, Nonlinearity};

/// Fixed parameter constraint: single parameter equals a constant.
///
/// # Equation
/// ```text
/// params[param_idx] - target = 0
/// ```
#[derive(Clone, Debug)]
pub struct FixedParamConstraint {
    id: ConstraintId,
    param_idx: usize,
    target: f64,
    deps: Vec<usize>,
}

impl FixedParamConstraint {
    /// Create a new fixed parameter constraint.
    ///
    /// # Arguments
    /// - `id`: Unique constraint identifier
    /// - `param_idx`: Index of the parameter to fix
    /// - `target`: Target value for the parameter
    pub fn new(id: ConstraintId, param_idx: usize, target: f64) -> Self {
        Self {
            id,
            param_idx,
            target,
            deps: vec![param_idx],
        }
    }
}

impl Constraint for FixedParamConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "FixedParam"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        vec![params[self.param_idx] - self.target]
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![(0, self.param_idx, 1.0)]
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Linear
    }
}

/// Equal parameter constraint: two parameters must be equal.
///
/// # Equation
/// ```text
/// params[param_a] - params[param_b] = 0
/// ```
#[derive(Clone, Debug)]
pub struct EqualParamConstraint {
    id: ConstraintId,
    param_a: usize,
    param_b: usize,
    deps: Vec<usize>,
}

impl EqualParamConstraint {
    /// Create a new equal parameter constraint.
    ///
    /// # Arguments
    /// - `id`: Unique constraint identifier
    /// - `param_a`: Index of first parameter
    /// - `param_b`: Index of second parameter
    pub fn new(id: ConstraintId, param_a: usize, param_b: usize) -> Self {
        Self {
            id,
            param_a,
            param_b,
            deps: vec![param_a, param_b],
        }
    }
}

impl Constraint for EqualParamConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "EqualParam"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        vec![params[self.param_a] - params[self.param_b]]
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, self.param_a, 1.0),
            (0, self.param_b, -1.0),
        ]
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Linear
    }
}

/// Parameter range constraint: enforce low ≤ param ≤ high using slack variables.
///
/// # Equations (2)
/// ```text
/// params[param_idx] - low - params[slack_low]² = 0
/// high - params[param_idx] - params[slack_high]² = 0
/// ```
///
/// The slack variables ensure the inequalities are satisfied:
/// - If slack_low ≥ 0, then param ≥ low
/// - If slack_high ≥ 0, then param ≤ high
#[derive(Clone, Debug)]
pub struct ParamRangeConstraint {
    id: ConstraintId,
    param_idx: usize,
    low: f64,
    high: f64,
    slack_low: usize,
    slack_high: usize,
    deps: Vec<usize>,
}

impl ParamRangeConstraint {
    /// Create a new parameter range constraint.
    ///
    /// # Arguments
    /// - `id`: Unique constraint identifier
    /// - `param_idx`: Index of the parameter to constrain
    /// - `low`: Lower bound (inclusive)
    /// - `high`: Upper bound (inclusive)
    /// - `slack_low`: Index of scalar entity used as slack variable for lower bound
    /// - `slack_high`: Index of scalar entity used as slack variable for upper bound
    pub fn new(
        id: ConstraintId,
        param_idx: usize,
        low: f64,
        high: f64,
        slack_low: usize,
        slack_high: usize,
    ) -> Self {
        assert!(
            low <= high,
            "Lower bound {} must be <= upper bound {}",
            low,
            high
        );

        Self {
            id,
            param_idx,
            low,
            high,
            slack_low,
            slack_high,
            deps: vec![param_idx, slack_low, slack_high],
        }
    }
}

impl Constraint for ParamRangeConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "ParamRange"
    }

    fn equation_count(&self) -> usize {
        2
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        let param = params[self.param_idx];
        let slack_low = params[self.slack_low];
        let slack_high = params[self.slack_high];

        vec![
            param - self.low - slack_low * slack_low,
            self.high - param - slack_high * slack_high,
        ]
    }

    fn jacobian(&self, params: &[f64]) -> Vec<(usize, usize, f64)> {
        let slack_low = params[self.slack_low];
        let slack_high = params[self.slack_high];

        vec![
            // Equation 0: param - low - slack_low² = 0
            (0, self.param_idx, 1.0),
            (0, self.slack_low, -2.0 * slack_low),
            // Equation 1: high - param - slack_high² = 0
            (1, self.param_idx, -1.0),
            (1, self.slack_high, -2.0 * slack_high),
        ]
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Moderate  // Quadratic in slack variables
    }
}

/// Ratio parameter constraint: param_a = ratio * param_b
///
/// # Equation
/// ```text
/// params[param_a] - ratio * params[param_b] = 0
/// ```
#[derive(Clone, Debug)]
pub struct RatioParamConstraint {
    id: ConstraintId,
    param_a: usize,
    param_b: usize,
    ratio: f64,
    deps: Vec<usize>,
}

impl RatioParamConstraint {
    /// Create a new ratio parameter constraint.
    ///
    /// # Arguments
    /// - `id`: Unique constraint identifier
    /// - `param_a`: Index of first parameter (dependent)
    /// - `param_b`: Index of second parameter (independent)
    /// - `ratio`: Ratio k such that param_a = k * param_b
    pub fn new(id: ConstraintId, param_a: usize, param_b: usize, ratio: f64) -> Self {
        Self {
            id,
            param_a,
            param_b,
            ratio,
            deps: vec![param_a, param_b],
        }
    }
}

impl Constraint for RatioParamConstraint {
    fn id(&self) -> ConstraintId {
        self.id
    }

    fn name(&self) -> &'static str {
        "RatioParam"
    }

    fn equation_count(&self) -> usize {
        1
    }

    fn dependencies(&self) -> &[usize] {
        &self.deps
    }

    fn residuals(&self, params: &[f64]) -> Vec<f64> {
        vec![params[self.param_a] - self.ratio * params[self.param_b]]
    }

    fn jacobian(&self, _params: &[f64]) -> Vec<(usize, usize, f64)> {
        vec![
            (0, self.param_a, 1.0),
            (0, self.param_b, -self.ratio),
        ]
    }

    fn nonlinearity_hint(&self) -> Nonlinearity {
        Nonlinearity::Linear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== FixedParamConstraint Tests =====

    #[test]
    fn test_fixed_param_satisfied() {
        let id = ConstraintId(0);
        let constraint = FixedParamConstraint::new(id, 5, 42.0);

        let mut params = vec![0.0; 10];
        params[5] = 42.0;

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_fixed_param_not_satisfied() {
        let id = ConstraintId(0);
        let constraint = FixedParamConstraint::new(id, 3, 10.0);

        let mut params = vec![0.0; 10];
        params[3] = 15.0;

        let residuals = constraint.residuals(&params);
        assert!((residuals[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_param_jacobian() {
        let id = ConstraintId(0);
        let constraint = FixedParamConstraint::new(id, 7, 100.0);

        let params = vec![0.0; 10];
        let jac = constraint.jacobian(&params);

        assert_eq!(jac.len(), 1);
        assert_eq!(jac[0], (0, 7, 1.0));
    }

    #[test]
    fn test_fixed_param_metadata() {
        let id = ConstraintId(42);
        let constraint = FixedParamConstraint::new(id, 5, 10.0);

        assert_eq!(constraint.id(), ConstraintId(42));
        assert_eq!(constraint.name(), "FixedParam");
        assert_eq!(constraint.equation_count(), 1);
        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::Linear);
        assert_eq!(constraint.dependencies(), &[5]);
    }

    // ===== EqualParamConstraint Tests =====

    #[test]
    fn test_equal_param_satisfied() {
        let id = ConstraintId(0);
        let constraint = EqualParamConstraint::new(id, 2, 5);

        let mut params = vec![0.0; 10];
        params[2] = 7.5;
        params[5] = 7.5;

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_equal_param_not_satisfied() {
        let id = ConstraintId(0);
        let constraint = EqualParamConstraint::new(id, 1, 3);

        let mut params = vec![0.0; 10];
        params[1] = 10.0;
        params[3] = 15.0;

        let residuals = constraint.residuals(&params);
        assert!((residuals[0] - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_equal_param_jacobian() {
        let id = ConstraintId(0);
        let constraint = EqualParamConstraint::new(id, 2, 8);

        let params = vec![0.0; 10];
        let jac = constraint.jacobian(&params);

        assert_eq!(jac.len(), 2);
        assert!(jac.contains(&(0, 2, 1.0)));
        assert!(jac.contains(&(0, 8, -1.0)));
    }

    #[test]
    fn test_equal_param_metadata() {
        let id = ConstraintId(42);
        let constraint = EqualParamConstraint::new(id, 3, 7);

        assert_eq!(constraint.id(), ConstraintId(42));
        assert_eq!(constraint.name(), "EqualParam");
        assert_eq!(constraint.equation_count(), 1);
        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::Linear);
        assert_eq!(constraint.dependencies(), &[3, 7]);
    }

    // ===== ParamRangeConstraint Tests =====

    #[test]
    fn test_param_range_within_bounds() {
        let id = ConstraintId(0);
        let constraint = ParamRangeConstraint::new(id, 5, 10.0, 20.0, 6, 7);

        let mut params = vec![0.0; 10];
        params[5] = 15.0;  // Within [10, 20]
        params[6] = 2.236068;  // sqrt(5)
        params[7] = 2.236068;  // sqrt(5)

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 2);

        // param - low - slack_low² = 15 - 10 - 5 = 0
        assert!(residuals[0].abs() < 1e-5);

        // high - param - slack_high² = 20 - 15 - 5 = 0
        assert!(residuals[1].abs() < 1e-5);
    }

    #[test]
    fn test_param_range_at_lower_bound() {
        let id = ConstraintId(0);
        let constraint = ParamRangeConstraint::new(id, 5, 10.0, 20.0, 6, 7);

        let mut params = vec![0.0; 10];
        params[5] = 10.0;  // At lower bound
        params[6] = 0.0;   // slack_low = 0
        params[7] = 3.162278;  // sqrt(10)

        let residuals = constraint.residuals(&params);

        assert!(residuals[0].abs() < 1e-10, "Lower bound equation");
        assert!(residuals[1].abs() < 1e-5, "Upper bound equation");
    }

    #[test]
    fn test_param_range_at_upper_bound() {
        let id = ConstraintId(0);
        let constraint = ParamRangeConstraint::new(id, 5, 10.0, 20.0, 6, 7);

        let mut params = vec![0.0; 10];
        params[5] = 20.0;  // At upper bound
        params[6] = 3.162278;  // sqrt(10)
        params[7] = 0.0;   // slack_high = 0

        let residuals = constraint.residuals(&params);

        assert!(residuals[0].abs() < 1e-5, "Lower bound equation");
        assert!(residuals[1].abs() < 1e-10, "Upper bound equation");
    }

    #[test]
    fn test_param_range_jacobian_numerical() {
        let id = ConstraintId(0);
        let constraint = ParamRangeConstraint::new(id, 5, 10.0, 20.0, 6, 7);

        let mut params = vec![0.0; 10];
        params[5] = 15.0;
        params[6] = 2.0;
        params[7] = 1.5;

        let jac = constraint.jacobian(&params);
        let h = 1e-7;

        for &(row, col, analytical) in &jac {
            let mut params_plus = params.clone();
            params_plus[col] += h;
            let res_plus = constraint.residuals(&params_plus);

            let mut params_minus = params.clone();
            params_minus[col] -= h;
            let res_minus = constraint.residuals(&params_minus);

            let numerical = (res_plus[row] - res_minus[row]) / (2.0 * h);
            let error = (analytical - numerical).abs();

            assert!(
                error < 1e-4,
                "Jacobian mismatch at ({},{}): analytical={}, numerical={}, error={}",
                row, col, analytical, numerical, error
            );
        }
    }

    #[test]
    #[should_panic(expected = "Lower bound 20 must be <= upper bound 10")]
    fn test_param_range_invalid_bounds() {
        let id = ConstraintId(0);
        ParamRangeConstraint::new(id, 5, 20.0, 10.0, 6, 7);
    }

    #[test]
    fn test_param_range_metadata() {
        let id = ConstraintId(42);
        let constraint = ParamRangeConstraint::new(id, 5, 0.0, 1.0, 6, 7);

        assert_eq!(constraint.id(), ConstraintId(42));
        assert_eq!(constraint.name(), "ParamRange");
        assert_eq!(constraint.equation_count(), 2);
        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::Moderate);
        assert_eq!(constraint.dependencies(), &[5, 6, 7]);
    }

    // ===== RatioParamConstraint Tests =====

    #[test]
    fn test_ratio_param_satisfied() {
        let id = ConstraintId(0);
        let constraint = RatioParamConstraint::new(id, 2, 5, 3.0);

        let mut params = vec![0.0; 10];
        params[2] = 15.0;
        params[5] = 5.0;  // 15 = 3 * 5

        let residuals = constraint.residuals(&params);
        assert_eq!(residuals.len(), 1);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_ratio_param_not_satisfied() {
        let id = ConstraintId(0);
        let constraint = RatioParamConstraint::new(id, 3, 7, 2.5);

        let mut params = vec![0.0; 10];
        params[3] = 10.0;
        params[7] = 3.0;  // Should be 7.5, not 3.0

        let residuals = constraint.residuals(&params);
        // 10 - 2.5 * 3 = 10 - 7.5 = 2.5
        assert!((residuals[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_ratio_param_negative_ratio() {
        let id = ConstraintId(0);
        let constraint = RatioParamConstraint::new(id, 1, 4, -2.0);

        let mut params = vec![0.0; 10];
        params[1] = -10.0;
        params[4] = 5.0;  // -10 = -2 * 5

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_ratio_param_zero_ratio() {
        let id = ConstraintId(0);
        let constraint = RatioParamConstraint::new(id, 2, 3, 0.0);

        let mut params = vec![0.0; 10];
        params[2] = 0.0;
        params[3] = 100.0;  // 0 = 0 * 100

        let residuals = constraint.residuals(&params);
        assert!(residuals[0].abs() < 1e-10);
    }

    #[test]
    fn test_ratio_param_jacobian() {
        let id = ConstraintId(0);
        let constraint = RatioParamConstraint::new(id, 2, 8, 4.5);

        let params = vec![0.0; 10];
        let jac = constraint.jacobian(&params);

        assert_eq!(jac.len(), 2);
        assert!(jac.contains(&(0, 2, 1.0)));
        assert!(jac.contains(&(0, 8, -4.5)));
    }

    #[test]
    fn test_ratio_param_metadata() {
        let id = ConstraintId(42);
        let constraint = RatioParamConstraint::new(id, 5, 10, 1.5);

        assert_eq!(constraint.id(), ConstraintId(42));
        assert_eq!(constraint.name(), "RatioParam");
        assert_eq!(constraint.equation_count(), 1);
        assert_eq!(constraint.nonlinearity_hint(), Nonlinearity::Linear);
        assert_eq!(constraint.dependencies(), &[5, 10]);
    }

    #[test]
    fn test_ratio_param_jacobian_numerical() {
        let id = ConstraintId(0);
        let constraint = RatioParamConstraint::new(id, 3, 7, 2.5);

        let mut params = vec![0.0; 10];
        params[3] = 10.0;
        params[7] = 4.0;

        let jac = constraint.jacobian(&params);
        let h = 1e-7;

        for &(row, col, analytical) in &jac {
            let mut params_plus = params.clone();
            params_plus[col] += h;
            let res_plus = constraint.residuals(&params_plus);

            let mut params_minus = params.clone();
            params_minus[col] -= h;
            let res_minus = constraint.residuals(&params_minus);

            let numerical = (res_plus[row] - res_minus[row]) / (2.0 * h);
            let error = (analytical - numerical).abs();

            assert!(
                error < 1e-4,
                "Jacobian mismatch at ({},{}): analytical={}, numerical={}, error={}",
                row, col, analytical, numerical, error
            );
        }
    }
}
