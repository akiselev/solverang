//! Inequality constraint support via slack variable transformation.
//!
//! This module provides tools for converting inequality constraints g(x) >= 0
//! into equality constraints that standard Newton-Raphson and LM solvers can handle.
//!
//! # Transformation Method
//!
//! For an inequality constraint g(x) >= 0, we introduce a slack variable s
//! and transform it to the equality: g(x) - s^2 = 0
//!
//! Using s^2 instead of s ensures the slack is always non-negative.
//!
//! # Applications
//!
//! - Design Rule Checking (DRC): minimum clearance constraints
//! - Bounds on variables: x >= lower, x <= upper
//! - Collision avoidance: distance(a, b) >= min_distance

use crate::problem::Problem;

/// An inequality constraint of the form g(x) >= 0.
///
/// The constraint value g(x) should be positive when the constraint is satisfied.
pub trait InequalityConstraint: Send + Sync {
    /// Evaluate the constraint: returns g(x).
    ///
    /// The constraint is satisfied when g(x) >= 0.
    fn evaluate(&self, x: &[f64]) -> f64;

    /// Compute the gradient of g with respect to x.
    ///
    /// Returns (variable_index, partial_derivative) pairs.
    fn gradient(&self, x: &[f64]) -> Vec<(usize, f64)>;

    /// Name of this constraint for debugging.
    fn name(&self) -> &str {
        "inequality"
    }
}

/// A problem that includes inequality constraints.
pub trait InequalityProblem: Problem {
    /// Number of inequality constraints.
    fn inequality_count(&self) -> usize;

    /// Evaluate inequality constraint i: returns g_i(x) where g_i(x) >= 0 is required.
    fn inequality_value(&self, index: usize, x: &[f64]) -> f64;

    /// Compute the gradient of inequality i with respect to x.
    ///
    /// Returns (variable_index, partial_derivative) pairs.
    fn inequality_gradient(&self, index: usize, x: &[f64]) -> Vec<(usize, f64)>;
}

/// Transforms a problem with inequalities into a pure equality problem.
///
/// For each inequality g_i(x) >= 0, introduces a slack variable s_i
/// and adds the equality constraint: g_i(x) - s_i^2 = 0
///
/// The extended variable vector is [x_0, ..., x_{n-1}, s_0, ..., s_{k-1}]
/// where k is the number of inequality constraints.
pub struct SlackVariableTransform<P> {
    inner: P,
    n_inequalities: usize,
    slack_start: usize,
}

impl<P: InequalityProblem> SlackVariableTransform<P> {
    /// Create a slack variable transformation for a problem with inequalities.
    pub fn new(inner: P) -> Self {
        let n_inequalities = inner.inequality_count();
        let slack_start = inner.variable_count();
        Self {
            inner,
            n_inequalities,
            slack_start,
        }
    }

    /// Get a reference to the inner problem.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Extract the original variables from an extended solution vector.
    pub fn extract_original_variables(&self, extended_x: &[f64]) -> Vec<f64> {
        extended_x[..self.slack_start].to_vec()
    }

    /// Extract the slack variables from an extended solution vector.
    pub fn extract_slack_variables(&self, extended_x: &[f64]) -> Vec<f64> {
        extended_x[self.slack_start..].to_vec()
    }

    /// Create an extended initial point with appropriate slack values.
    ///
    /// Slack variables are initialized to sqrt(g_i(x0)) if g_i(x0) > 0,
    /// or a small positive value if g_i(x0) <= 0.
    pub fn extended_initial_point(&self, x0: &[f64]) -> Vec<f64> {
        let mut extended = x0.to_vec();
        extended.reserve(self.n_inequalities);

        for i in 0..self.n_inequalities {
            let g_value = self.inner.inequality_value(i, x0);
            // s^2 = g(x), so s = sqrt(g(x)) if g(x) > 0
            let slack = if g_value > 0.0 {
                g_value.sqrt()
            } else {
                // Constraint violated or at boundary - use small positive value
                0.1
            };
            extended.push(slack);
        }

        extended
    }

    /// Check if all inequality constraints are satisfied in the original space.
    pub fn inequalities_satisfied(&self, x: &[f64]) -> bool {
        for i in 0..self.n_inequalities {
            if self.inner.inequality_value(i, x) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Get the violation amount for each inequality (negative if violated).
    pub fn inequality_violations(&self, x: &[f64]) -> Vec<f64> {
        (0..self.n_inequalities)
            .map(|i| self.inner.inequality_value(i, x))
            .collect()
    }
}

impl<P: InequalityProblem> Problem for SlackVariableTransform<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn variable_count(&self) -> usize {
        self.inner.variable_count() + self.n_inequalities
    }

    fn residual_count(&self) -> usize {
        self.inner.residual_count() + self.n_inequalities
    }

    fn residuals(&self, extended_x: &[f64]) -> Vec<f64> {
        // Split extended vector
        let original_x = &extended_x[..self.slack_start];
        let slacks = &extended_x[self.slack_start..];

        // Get original equality residuals
        let mut residuals = self.inner.residuals(original_x);

        // Add transformed inequality residuals: g_i(x) - s_i^2 = 0
        for i in 0..self.n_inequalities {
            let g_value = self.inner.inequality_value(i, original_x);
            let slack = slacks.get(i).copied().unwrap_or(0.0);
            residuals.push(g_value - slack * slack);
        }

        residuals
    }

    fn jacobian(&self, extended_x: &[f64]) -> Vec<(usize, usize, f64)> {
        let original_x = &extended_x[..self.slack_start];
        let slacks = &extended_x[self.slack_start..];

        // Start with original Jacobian entries
        let mut jacobian = self.inner.jacobian(original_x);

        // Add Jacobian entries for transformed inequality constraints
        let equality_count = self.inner.residual_count();

        for i in 0..self.n_inequalities {
            let row = equality_count + i;

            // Partial derivatives w.r.t. original variables: dg_i/dx_j
            for (var_idx, partial) in self.inner.inequality_gradient(i, original_x) {
                if var_idx < self.slack_start {
                    jacobian.push((row, var_idx, partial));
                }
            }

            // Partial derivative w.r.t. slack variable: d(g - s^2)/ds = -2s
            let slack = slacks.get(i).copied().unwrap_or(0.0);
            let slack_col = self.slack_start + i;
            jacobian.push((row, slack_col, -2.0 * slack));
        }

        jacobian
    }

    fn initial_point(&self, factor: f64) -> Vec<f64> {
        let x0 = self.inner.initial_point(factor);
        self.extended_initial_point(&x0)
    }
}

/// A clearance constraint for Design Rule Checking.
///
/// Ensures that the distance between two entities is at least `min_clearance`.
/// The constraint is: distance(entity_a, entity_b) - min_clearance >= 0
pub struct ClearanceConstraint {
    /// First entity (index into some entity array).
    pub entity_a: usize,
    /// Second entity (index into some entity array).
    pub entity_b: usize,
    /// Minimum required clearance.
    pub min_clearance: f64,
}

impl ClearanceConstraint {
    /// Create a new clearance constraint.
    pub fn new(entity_a: usize, entity_b: usize, min_clearance: f64) -> Self {
        Self {
            entity_a,
            entity_b,
            min_clearance,
        }
    }

    /// Compute the clearance given positions of the two entities.
    ///
    /// This is a helper for 2D point entities where each entity has (x, y) coordinates.
    /// The variable vector layout is [x0, y0, x1, y1, x2, y2, ...].
    pub fn compute_clearance_2d(&self, x: &[f64]) -> Option<f64> {
        let idx_a = self.entity_a * 2;
        let idx_b = self.entity_b * 2;

        let xa = x.get(idx_a)?;
        let ya = x.get(idx_a + 1)?;
        let xb = x.get(idx_b)?;
        let yb = x.get(idx_b + 1)?;

        let dx = xb - xa;
        let dy = yb - ya;
        let distance = (dx * dx + dy * dy).sqrt();

        Some(distance)
    }

    /// Compute the gradient of 2D distance with respect to positions.
    ///
    /// Returns (variable_index, partial_derivative) pairs.
    pub fn gradient_2d(&self, x: &[f64]) -> Vec<(usize, f64)> {
        let idx_a = self.entity_a * 2;
        let idx_b = self.entity_b * 2;

        let xa = x.get(idx_a).copied().unwrap_or(0.0);
        let ya = x.get(idx_a + 1).copied().unwrap_or(0.0);
        let xb = x.get(idx_b).copied().unwrap_or(0.0);
        let yb = x.get(idx_b + 1).copied().unwrap_or(0.0);

        let dx = xb - xa;
        let dy = yb - ya;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < 1e-10 {
            // Points coincident - gradient undefined, return zeros
            return vec![
                (idx_a, 0.0),
                (idx_a + 1, 0.0),
                (idx_b, 0.0),
                (idx_b + 1, 0.0),
            ];
        }

        // d(distance)/d(xa) = -dx/distance
        // d(distance)/d(ya) = -dy/distance
        // d(distance)/d(xb) = dx/distance
        // d(distance)/d(yb) = dy/distance
        vec![
            (idx_a, -dx / distance),
            (idx_a + 1, -dy / distance),
            (idx_b, dx / distance),
            (idx_b + 1, dy / distance),
        ]
    }
}

/// A simple bounds constraint: lower <= x_i <= upper.
///
/// This is represented as two inequalities:
/// - x_i - lower >= 0
/// - upper - x_i >= 0
pub struct BoundsConstraint {
    /// Variable index.
    pub variable: usize,
    /// Lower bound (or f64::NEG_INFINITY for no lower bound).
    pub lower: f64,
    /// Upper bound (or f64::INFINITY for no upper bound).
    pub upper: f64,
}

impl BoundsConstraint {
    /// Create a bounds constraint.
    pub fn new(variable: usize, lower: f64, upper: f64) -> Self {
        Self {
            variable,
            lower,
            upper,
        }
    }

    /// Create a lower bound only: x_i >= lower.
    pub fn lower_bound(variable: usize, lower: f64) -> Self {
        Self {
            variable,
            lower,
            upper: f64::INFINITY,
        }
    }

    /// Create an upper bound only: x_i <= upper.
    pub fn upper_bound(variable: usize, upper: f64) -> Self {
        Self {
            variable,
            lower: f64::NEG_INFINITY,
            upper,
        }
    }

    /// Check if this constraint has a lower bound.
    pub fn has_lower(&self) -> bool {
        self.lower.is_finite()
    }

    /// Check if this constraint has an upper bound.
    pub fn has_upper(&self) -> bool {
        self.upper.is_finite()
    }

    /// Evaluate the lower bound constraint: x_i - lower >= 0.
    pub fn lower_value(&self, x: &[f64]) -> f64 {
        x.get(self.variable).copied().unwrap_or(0.0) - self.lower
    }

    /// Evaluate the upper bound constraint: upper - x_i >= 0.
    pub fn upper_value(&self, x: &[f64]) -> f64 {
        self.upper - x.get(self.variable).copied().unwrap_or(0.0)
    }

    /// Gradient of lower bound constraint: 1 for the variable.
    pub fn lower_gradient(&self) -> (usize, f64) {
        (self.variable, 1.0)
    }

    /// Gradient of upper bound constraint: -1 for the variable.
    pub fn upper_gradient(&self) -> (usize, f64) {
        (self.variable, -1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test problem with one equality and one inequality
    struct SimpleInequalityProblem;

    impl Problem for SimpleInequalityProblem {
        fn name(&self) -> &str {
            "simple-inequality"
        }

        fn residual_count(&self) -> usize {
            1 // One equality constraint
        }

        fn variable_count(&self) -> usize {
            2
        }

        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            // x + y = 3
            vec![x[0] + x[1] - 3.0]
        }

        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 1.0), (0, 1, 1.0)]
        }

        fn initial_point(&self, _factor: f64) -> Vec<f64> {
            vec![1.0, 1.0]
        }
    }

    impl InequalityProblem for SimpleInequalityProblem {
        fn inequality_count(&self) -> usize {
            1
        }

        fn inequality_value(&self, index: usize, x: &[f64]) -> f64 {
            match index {
                0 => x[0] - 1.0, // x >= 1
                _ => 0.0,
            }
        }

        fn inequality_gradient(&self, index: usize, _x: &[f64]) -> Vec<(usize, f64)> {
            match index {
                0 => vec![(0, 1.0)],
                _ => vec![],
            }
        }
    }

    #[test]
    fn test_slack_transform_dimensions() {
        let problem = SimpleInequalityProblem;
        let transformed = SlackVariableTransform::new(problem);

        assert_eq!(transformed.variable_count(), 3); // 2 original + 1 slack
        assert_eq!(transformed.residual_count(), 2); // 1 equality + 1 transformed inequality
    }

    #[test]
    fn test_slack_transform_residuals() {
        let problem = SimpleInequalityProblem;
        let transformed = SlackVariableTransform::new(problem);

        // x=2, y=1 satisfies x+y=3
        // inequality x>=1 gives g=2-1=1
        // slack s=1, so s^2=1, residual = g - s^2 = 1 - 1 = 0
        let extended_x = vec![2.0, 1.0, 1.0]; // [x, y, s]
        let residuals = transformed.residuals(&extended_x);

        assert_eq!(residuals.len(), 2);
        assert!((residuals[0] - 0.0).abs() < 1e-10, "equality residual");
        assert!((residuals[1] - 0.0).abs() < 1e-10, "inequality residual");
    }

    #[test]
    fn test_slack_transform_jacobian() {
        let problem = SimpleInequalityProblem;
        let transformed = SlackVariableTransform::new(problem);

        let extended_x = vec![2.0, 1.0, 1.0];
        let jacobian = transformed.jacobian(&extended_x);

        // Should have entries for:
        // Row 0 (equality): d/dx = 1, d/dy = 1
        // Row 1 (inequality): d/dx = 1, d/ds = -2s = -2
        assert!(jacobian.contains(&(0, 0, 1.0)));
        assert!(jacobian.contains(&(0, 1, 1.0)));
        assert!(jacobian.contains(&(1, 0, 1.0)));
        assert!(jacobian.contains(&(1, 2, -2.0)));
    }

    #[test]
    fn test_extended_initial_point() {
        let problem = SimpleInequalityProblem;
        let transformed = SlackVariableTransform::new(problem);

        let x0 = vec![2.0, 1.0]; // g = 2 - 1 = 1, so s = sqrt(1) = 1
        let extended = transformed.extended_initial_point(&x0);

        assert_eq!(extended.len(), 3);
        assert!((extended[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_variables() {
        let problem = SimpleInequalityProblem;
        let transformed = SlackVariableTransform::new(problem);

        let extended_x = vec![2.0, 1.0, 1.5];
        let original = transformed.extract_original_variables(&extended_x);
        let slacks = transformed.extract_slack_variables(&extended_x);

        assert_eq!(original, vec![2.0, 1.0]);
        assert_eq!(slacks, vec![1.5]);
    }

    #[test]
    fn test_inequalities_satisfied() {
        let problem = SimpleInequalityProblem;
        let transformed = SlackVariableTransform::new(problem);

        assert!(transformed.inequalities_satisfied(&[2.0, 1.0])); // x=2 >= 1
        assert!(!transformed.inequalities_satisfied(&[0.5, 2.5])); // x=0.5 < 1
    }

    #[test]
    fn test_clearance_constraint_2d() {
        let constraint = ClearanceConstraint::new(0, 1, 1.0);

        // Entity 0 at (0, 0), Entity 1 at (3, 4) -> distance = 5
        let x = vec![0.0, 0.0, 3.0, 4.0];
        let clearance = constraint.compute_clearance_2d(&x).expect("should compute");

        assert!((clearance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_clearance_gradient_2d() {
        let constraint = ClearanceConstraint::new(0, 1, 1.0);

        // Entity 0 at (0, 0), Entity 1 at (3, 4) -> distance = 5
        let x = vec![0.0, 0.0, 3.0, 4.0];
        let gradient = constraint.gradient_2d(&x);

        // d(dist)/d(x0) = -dx/dist = -3/5 = -0.6
        // d(dist)/d(y0) = -dy/dist = -4/5 = -0.8
        // d(dist)/d(x1) = dx/dist = 3/5 = 0.6
        // d(dist)/d(y1) = dy/dist = 4/5 = 0.8
        assert_eq!(gradient.len(), 4);

        let find_grad = |idx: usize| gradient.iter().find(|(i, _)| *i == idx).map(|(_, v)| *v);

        assert!((find_grad(0).unwrap() - (-0.6)).abs() < 1e-10);
        assert!((find_grad(1).unwrap() - (-0.8)).abs() < 1e-10);
        assert!((find_grad(2).unwrap() - 0.6).abs() < 1e-10);
        assert!((find_grad(3).unwrap() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_bounds_constraint() {
        let bounds = BoundsConstraint::new(0, 1.0, 5.0);

        assert!(bounds.has_lower());
        assert!(bounds.has_upper());

        let x = vec![3.0, 0.0];

        // Lower: x - 1 = 2 >= 0
        assert!((bounds.lower_value(&x) - 2.0).abs() < 1e-10);

        // Upper: 5 - x = 2 >= 0
        assert!((bounds.upper_value(&x) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounds_constraint_partial() {
        let lower_only = BoundsConstraint::lower_bound(0, 1.0);
        assert!(lower_only.has_lower());
        assert!(!lower_only.has_upper());

        let upper_only = BoundsConstraint::upper_bound(0, 5.0);
        assert!(!upper_only.has_lower());
        assert!(upper_only.has_upper());
    }

    #[test]
    fn test_bounds_gradient() {
        let bounds = BoundsConstraint::new(0, 1.0, 5.0);

        let (idx, grad) = bounds.lower_gradient();
        assert_eq!(idx, 0);
        assert!((grad - 1.0).abs() < 1e-10);

        let (idx, grad) = bounds.upper_gradient();
        assert_eq!(idx, 0);
        assert!((grad - (-1.0)).abs() < 1e-10);
    }
}
