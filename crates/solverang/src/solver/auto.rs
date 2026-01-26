//! Automatic solver selection.
//!
//! This module provides an `AutoSolver` that automatically selects between
//! Newton-Raphson and Levenberg-Marquardt based on problem characteristics.

use crate::problem::Problem;
use crate::solver::config::SolverConfig;
use crate::solver::levenberg_marquardt::LMSolver;
use crate::solver::lm_config::LMConfig;
use crate::solver::newton_raphson::Solver as NRSolver;
use crate::solver::result::SolveResult;

/// Solver selection strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SolverChoice {
    /// Always use Newton-Raphson solver.
    NewtonRaphson,

    /// Always use Levenberg-Marquardt solver.
    LevenbergMarquardt,

    /// Automatically select based on problem characteristics.
    ///
    /// Selection criteria:
    /// - Square system (m == n): prefer Newton-Raphson for speed
    /// - Over-determined (m > n): use LM for least-squares solution
    /// - Under-determined (m < n): use LM as it handles this better
    #[default]
    Auto,
}

/// Automatic solver that selects between Newton-Raphson and Levenberg-Marquardt.
///
/// This solver analyzes problem characteristics and chooses the most appropriate
/// algorithm:
///
/// - **Square systems (m = n)**: Uses Newton-Raphson, which converges quadratically
///   near the solution and is generally faster for well-posed problems.
///
/// - **Over-determined systems (m > n)**: Uses Levenberg-Marquardt, which is designed
///   for least-squares problems and finds the minimum-norm residual solution.
///
/// - **Under-determined systems (m < n)**: Uses Levenberg-Marquardt, which handles
///   these cases more gracefully than pure Newton-Raphson.
///
/// # Example
///
/// ```rust
/// use solverang::{Problem, AutoSolver, SolverChoice};
///
/// struct MyProblem;
///
/// impl Problem for MyProblem {
///     fn name(&self) -> &str { "example" }
///     fn residual_count(&self) -> usize { 2 }
///     fn variable_count(&self) -> usize { 2 }
///     fn residuals(&self, x: &[f64]) -> Vec<f64> {
///         vec![x[0] * x[0] + x[1] - 1.0, x[0] + x[1] * x[1] - 1.0]
///     }
///     fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
///         vec![
///             (0, 0, 2.0 * x[0]), (0, 1, 1.0),
///             (1, 0, 1.0), (1, 1, 2.0 * x[1]),
///         ]
///     }
///     fn initial_point(&self, f: f64) -> Vec<f64> { vec![0.5 * f, 0.5 * f] }
/// }
///
/// let solver = AutoSolver::new();
/// let result = solver.solve(&MyProblem, &[0.5, 0.5]);
/// assert!(result.is_converged());
/// ```
pub struct AutoSolver {
    /// Configuration for Newton-Raphson solver
    nr_config: SolverConfig,
    /// Configuration for Levenberg-Marquardt solver
    lm_config: LMConfig,
    /// Solver selection strategy
    choice: SolverChoice,
}

impl AutoSolver {
    /// Create a new auto-solver with default configurations.
    pub fn new() -> Self {
        Self {
            nr_config: SolverConfig::default(),
            lm_config: LMConfig::default(),
            choice: SolverChoice::Auto,
        }
    }

    /// Create an auto-solver with custom configurations.
    pub fn with_configs(nr_config: SolverConfig, lm_config: LMConfig) -> Self {
        Self {
            nr_config,
            lm_config,
            choice: SolverChoice::Auto,
        }
    }

    /// Set the solver selection strategy.
    #[must_use]
    pub fn with_choice(mut self, choice: SolverChoice) -> Self {
        self.choice = choice;
        self
    }

    /// Set the Newton-Raphson configuration.
    #[must_use]
    pub fn with_nr_config(mut self, config: SolverConfig) -> Self {
        self.nr_config = config;
        self
    }

    /// Set the Levenberg-Marquardt configuration.
    #[must_use]
    pub fn with_lm_config(mut self, config: LMConfig) -> Self {
        self.lm_config = config;
        self
    }

    /// Get the Newton-Raphson configuration.
    pub fn nr_config(&self) -> &SolverConfig {
        &self.nr_config
    }

    /// Get the Levenberg-Marquardt configuration.
    pub fn lm_config(&self) -> &LMConfig {
        &self.lm_config
    }

    /// Get the solver selection strategy.
    pub fn choice(&self) -> SolverChoice {
        self.choice
    }

    /// Solve the problem, automatically selecting the best solver.
    ///
    /// # Arguments
    ///
    /// * `problem` - The nonlinear problem to solve
    /// * `x0` - Initial guess for the solution
    ///
    /// # Returns
    ///
    /// A `SolveResult` indicating success or failure of the optimization.
    pub fn solve<P: Problem + ?Sized>(&self, problem: &P, x0: &[f64]) -> SolveResult {
        match self.select_solver(problem) {
            SolverChoice::NewtonRaphson => {
                let solver = NRSolver::new(self.nr_config.clone());
                solver.solve(problem, x0)
            }
            SolverChoice::LevenbergMarquardt | SolverChoice::Auto => {
                let solver = LMSolver::new(self.lm_config.clone());
                solver.solve(problem, x0)
            }
        }
    }

    /// Solve using the problem's default initial point.
    pub fn solve_from_initial<P: Problem + ?Sized>(&self, problem: &P, factor: f64) -> SolveResult {
        let x0 = problem.initial_point(factor);
        self.solve(problem, &x0)
    }

    /// Determine which solver to use based on problem characteristics and configuration.
    fn select_solver<P: Problem + ?Sized>(&self, problem: &P) -> SolverChoice {
        match self.choice {
            SolverChoice::NewtonRaphson => SolverChoice::NewtonRaphson,
            SolverChoice::LevenbergMarquardt => SolverChoice::LevenbergMarquardt,
            SolverChoice::Auto => self.auto_select(problem),
        }
    }

    /// Automatically select a solver based on problem characteristics.
    ///
    /// Selection criteria:
    /// - Square systems (m == n): prefer Newton-Raphson
    /// - Non-square systems: use Levenberg-Marquardt
    fn auto_select<P: Problem + ?Sized>(&self, problem: &P) -> SolverChoice {
        let m = problem.residual_count();
        let n = problem.variable_count();

        if m == n {
            // Square system: Newton-Raphson converges faster when applicable
            SolverChoice::NewtonRaphson
        } else {
            // Non-square system: LM handles these better
            SolverChoice::LevenbergMarquardt
        }
    }

    /// Determine which solver was selected for a problem (for diagnostics).
    pub fn which_solver<P: Problem + ?Sized>(&self, problem: &P) -> SolverChoice {
        self.select_solver(problem)
    }
}

impl Default for AutoSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// A robust solver that tries multiple strategies.
///
/// This solver attempts Newton-Raphson first, and if it fails to converge,
/// falls back to Levenberg-Marquardt. This provides the speed of NR when
/// it works, with the robustness of LM as a fallback.
pub struct RobustSolver {
    /// Configuration for Newton-Raphson solver
    nr_config: SolverConfig,
    /// Configuration for Levenberg-Marquardt solver
    lm_config: LMConfig,
    /// Tolerance for considering NR "failed" (triggers fallback)
    fallback_threshold: f64,
}

impl RobustSolver {
    /// Create a new robust solver with default configurations.
    pub fn new() -> Self {
        Self {
            nr_config: SolverConfig::default(),
            lm_config: LMConfig::robust(),
            fallback_threshold: 1e-4,
        }
    }

    /// Set the Newton-Raphson configuration.
    #[must_use]
    pub fn with_nr_config(mut self, config: SolverConfig) -> Self {
        self.nr_config = config;
        self
    }

    /// Set the Levenberg-Marquardt configuration.
    #[must_use]
    pub fn with_lm_config(mut self, config: LMConfig) -> Self {
        self.lm_config = config;
        self
    }

    /// Set the residual threshold for triggering LM fallback.
    ///
    /// If NR completes but the residual norm exceeds this threshold,
    /// LM will be tried as a fallback.
    #[must_use]
    pub fn with_fallback_threshold(mut self, threshold: f64) -> Self {
        self.fallback_threshold = threshold;
        self
    }

    /// Solve the problem, using LM as fallback if NR struggles.
    ///
    /// The strategy is:
    /// 1. Try Newton-Raphson first
    /// 2. If NR converges with acceptable residual, return that result
    /// 3. If NR fails or has high residual, try Levenberg-Marquardt
    /// 4. Return the better of the two results
    pub fn solve<P: Problem + ?Sized>(&self, problem: &P, x0: &[f64]) -> SolveResult {
        let m = problem.residual_count();
        let n = problem.variable_count();

        // For overdetermined systems, go directly to LM
        if m > n {
            let solver = LMSolver::new(self.lm_config.clone());
            return solver.solve(problem, x0);
        }

        // Try Newton-Raphson first
        let nr_solver = NRSolver::new(self.nr_config.clone());
        let nr_result = nr_solver.solve(problem, x0);

        // Check if NR succeeded well enough
        let nr_good = match &nr_result {
            SolveResult::Converged { residual_norm, .. } => {
                *residual_norm < self.fallback_threshold
            }
            SolveResult::NotConverged { residual_norm, .. } => {
                *residual_norm < self.fallback_threshold
            }
            SolveResult::Failed { .. } => false,
        };

        if nr_good {
            return nr_result;
        }

        // NR didn't work well, try LM
        let lm_solver = LMSolver::new(self.lm_config.clone());
        let lm_result = lm_solver.solve(problem, x0);

        // Return the better result (lower residual norm)
        let nr_norm = nr_result.residual_norm();
        let lm_norm = lm_result.residual_norm();

        match (nr_norm, lm_norm) {
            (Some(nr), Some(lm)) if nr < lm => nr_result,
            (Some(_), Some(_)) => lm_result,
            (None, Some(_)) => lm_result,
            (Some(_), None) => nr_result,
            (None, None) => {
                // Both failed, prefer LM's error as it's usually more informative
                lm_result
            }
        }
    }

    /// Solve using the problem's default initial point.
    pub fn solve_from_initial<P: Problem + ?Sized>(&self, problem: &P, factor: f64) -> SolveResult {
        let x0 = problem.initial_point(factor);
        self.solve(problem, &x0)
    }
}

impl Default for RobustSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Square problem (m = n)
    struct SquareProblem;

    impl Problem for SquareProblem {
        fn name(&self) -> &str {
            "square"
        }
        fn residual_count(&self) -> usize {
            2
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] + x[1] - 1.0, x[0] + x[1] * x[1] - 1.0]
        }
        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![
                (0, 0, 2.0 * x[0]),
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 1, 2.0 * x[1]),
            ]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![0.5 * factor, 0.5 * factor]
        }
    }

    // Overdetermined problem (m > n)
    struct OverdeterminedProblem;

    impl Problem for OverdeterminedProblem {
        fn name(&self) -> &str {
            "overdetermined"
        }
        fn residual_count(&self) -> usize {
            4
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![
                x[0] - 1.0,
                x[1] - 2.0,
                x[0] + x[1] - 3.0,
                x[0] - x[1] + 1.0,
            ]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![
                (0, 0, 1.0),
                (0, 1, 0.0),
                (1, 0, 0.0),
                (1, 1, 1.0),
                (2, 0, 1.0),
                (2, 1, 1.0),
                (3, 0, 1.0),
                (3, 1, -1.0),
            ]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![0.0 * factor, 0.0 * factor]
        }
    }

    // Underdetermined problem (m < n)
    struct UnderdeterminedProblem;

    impl Problem for UnderdeterminedProblem {
        fn name(&self) -> &str {
            "underdetermined"
        }
        fn residual_count(&self) -> usize {
            1
        }
        fn variable_count(&self) -> usize {
            2
        }
        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] + x[1] - 1.0]
        }
        fn jacobian(&self, _x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 1.0), (0, 1, 1.0)]
        }
        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![0.0 * factor, 0.0 * factor]
        }
    }

    #[test]
    fn test_auto_select_square() {
        let solver = AutoSolver::new();
        let problem = SquareProblem;

        assert_eq!(solver.which_solver(&problem), SolverChoice::NewtonRaphson);
    }

    #[test]
    fn test_auto_select_overdetermined() {
        let solver = AutoSolver::new();
        let problem = OverdeterminedProblem;

        assert_eq!(
            solver.which_solver(&problem),
            SolverChoice::LevenbergMarquardt
        );
    }

    #[test]
    fn test_auto_select_underdetermined() {
        let solver = AutoSolver::new();
        let problem = UnderdeterminedProblem;

        assert_eq!(
            solver.which_solver(&problem),
            SolverChoice::LevenbergMarquardt
        );
    }

    #[test]
    fn test_forced_newton_raphson() {
        let solver = AutoSolver::new().with_choice(SolverChoice::NewtonRaphson);
        let problem = OverdeterminedProblem;

        assert_eq!(solver.which_solver(&problem), SolverChoice::NewtonRaphson);
    }

    #[test]
    fn test_forced_levenberg_marquardt() {
        let solver = AutoSolver::new().with_choice(SolverChoice::LevenbergMarquardt);
        let problem = SquareProblem;

        assert_eq!(
            solver.which_solver(&problem),
            SolverChoice::LevenbergMarquardt
        );
    }

    #[test]
    fn test_solve_square_problem() {
        let solver = AutoSolver::new();
        let problem = SquareProblem;
        let result = solver.solve(&problem, &[0.5, 0.5]);

        assert!(result.is_converged(), "Result: {:?}", result);
    }

    #[test]
    fn test_solve_overdetermined_problem() {
        let solver = AutoSolver::new();
        let problem = OverdeterminedProblem;
        let result = solver.solve(&problem, &[0.0, 0.0]);

        // Should converge to least-squares solution
        assert!(
            result.is_converged() || result.is_completed(),
            "Result: {:?}",
            result
        );
    }

    #[test]
    fn test_solve_underdetermined_problem() {
        let solver = AutoSolver::new();
        let problem = UnderdeterminedProblem;
        let result = solver.solve(&problem, &[0.0, 0.0]);

        // Should find a solution (infinitely many exist)
        assert!(
            result.is_converged() || result.is_completed(),
            "Result: {:?}",
            result
        );

        if let Some(solution) = result.solution() {
            // Verify the solution satisfies the constraint
            let sum = solution[0] + solution[1];
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Solution should satisfy x + y = 1, got sum = {}",
                sum
            );
        }
    }

    #[test]
    fn test_robust_solver_square() {
        let solver = RobustSolver::new();
        let problem = SquareProblem;
        let result = solver.solve(&problem, &[0.5, 0.5]);

        assert!(result.is_converged(), "Result: {:?}", result);
    }

    #[test]
    fn test_robust_solver_overdetermined() {
        let solver = RobustSolver::new();
        let problem = OverdeterminedProblem;
        let result = solver.solve(&problem, &[0.0, 0.0]);

        assert!(
            result.is_converged() || result.is_completed(),
            "Result: {:?}",
            result
        );
    }

    #[test]
    fn test_auto_solver_configs() {
        let nr_config = SolverConfig::fast();
        let lm_config = LMConfig::robust();

        let solver = AutoSolver::with_configs(nr_config.clone(), lm_config.clone());

        assert_eq!(solver.nr_config().max_iterations, nr_config.max_iterations);
        assert_eq!(solver.lm_config().patience, lm_config.patience);
    }

    #[test]
    fn test_solve_from_initial() {
        let solver = AutoSolver::new();
        let problem = SquareProblem;
        let result = solver.solve_from_initial(&problem, 1.0);

        assert!(result.is_converged(), "Result: {:?}", result);
    }

    #[test]
    fn test_robust_solver_from_initial() {
        let solver = RobustSolver::new();
        let problem = SquareProblem;
        let result = solver.solve_from_initial(&problem, 1.0);

        assert!(result.is_converged(), "Result: {:?}", result);
    }

    #[test]
    fn test_robust_solver_configs() {
        let solver = RobustSolver::new()
            .with_nr_config(SolverConfig::fast())
            .with_lm_config(LMConfig::robust())
            .with_fallback_threshold(1e-6);

        // Just verify it can be configured without panicking
        let problem = SquareProblem;
        let result = solver.solve(&problem, &[0.5, 0.5]);
        assert!(result.is_converged() || result.is_completed());
    }
}
