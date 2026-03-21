//! JIT-enabled solver implementation.
//!
//! This module provides a solver that can use JIT-compiled constraint evaluation
//! for improved performance on large constraint systems.

use crate::jit::{CompiledConstraints, JITCompiler, JITConfig, JITError, JITFunction};
use crate::problem::Problem;
use crate::solver::result::{SolveError, SolveResult};
use nalgebra::{DMatrix, DVector};

/// A solver that uses JIT compilation for large constraint systems.
///
/// For problems above a configurable threshold, this solver compiles the
/// constraint evaluation to native code, eliminating virtual function call
/// overhead during the iterative solve loop.
///
/// # Example
///
/// ```rust,ignore
/// use solverang::solver::JITSolver;
/// use solverang::jit::JITConfig;
///
/// let solver = JITSolver::new(JITConfig::default());
/// let result = solver.solve(&problem, &initial_point);
/// ```
pub struct JITSolver {
    /// Configuration.
    config: JITConfig,

    /// Cached JIT compiler.
    compiler: Option<JITCompiler>,
}

impl JITSolver {
    /// Create a new JIT-enabled solver.
    pub fn new(config: JITConfig) -> Self {
        Self {
            config,
            compiler: None,
        }
    }

    /// Create a solver with default configuration.
    pub fn default_solver() -> Self {
        Self::new(JITConfig::default())
    }

    /// Get the solver configuration.
    pub fn config(&self) -> &JITConfig {
        &self.config
    }

    /// Check if JIT will be used for the given problem.
    pub fn will_use_jit<P: Problem>(&self, problem: &P) -> bool {
        if self.config.force_interpreted {
            return false;
        }

        if self.config.force_jit {
            return crate::jit::jit_available();
        }

        let estimated_work = problem.residual_count() * self.config.estimated_iterations;
        estimated_work > self.config.jit_threshold && crate::jit::jit_available()
    }

    /// Solve a problem using either JIT or interpreted evaluation.
    pub fn solve<P: Problem>(&mut self, problem: &P, x0: &[f64]) -> SolveResult {
        let n = problem.variable_count();
        let m = problem.residual_count();

        // Validate problem dimensions
        if n == 0 {
            return SolveResult::Failed {
                error: SolveError::NoVariables,
            };
        }

        if m == 0 {
            return SolveResult::Failed {
                error: SolveError::NoEquations,
            };
        }

        if x0.len() != n {
            return SolveResult::Failed {
                error: SolveError::DimensionMismatch {
                    expected: n,
                    got: x0.len(),
                },
            };
        }

        // For now, we always use interpreted since we can't automatically lower
        // arbitrary Problem implementations. JIT requires explicit lowering of
        // constraint types.
        self.solve_interpreted(problem, x0)
    }

    /// Solve using JIT-compiled evaluation.
    ///
    /// This method is called when a compiled JIT function is provided directly.
    pub fn solve_with_jit(&self, jit_fn: &JITFunction, x0: &[f64]) -> SolveResult {
        let n = jit_fn.variable_count();
        let m = jit_fn.residual_count();

        if x0.len() != n {
            return SolveResult::Failed {
                error: SolveError::DimensionMismatch {
                    expected: n,
                    got: x0.len(),
                },
            };
        }

        let mut x = DVector::from_column_slice(x0);
        let mut residuals = vec![0.0; m];
        let mut jac_values = vec![0.0; jit_fn.jacobian_nnz()];

        for iteration in 0..self.config.max_iterations {
            // Evaluate residuals using JIT
            jit_fn.evaluate_residuals(x.as_slice(), &mut residuals);

            // Check for non-finite residuals
            if residuals.iter().any(|r| !r.is_finite()) {
                return SolveResult::Failed {
                    error: SolveError::NonFiniteResiduals,
                };
            }

            let r = DVector::from_column_slice(&residuals);
            let norm = r.norm();

            // Check convergence
            if norm < self.config.tolerance {
                return SolveResult::Converged {
                    solution: x.as_slice().to_vec(),
                    iterations: iteration,
                    residual_norm: norm,
                };
            }

            // Evaluate Jacobian using JIT
            jit_fn.evaluate_jacobian(x.as_slice(), &mut jac_values);

            // Check for non-finite Jacobian entries
            if jac_values.iter().any(|v| !v.is_finite()) {
                return SolveResult::Failed {
                    error: SolveError::NonFiniteJacobian,
                };
            }

            // Build Jacobian matrix from COO format
            let mut j = DMatrix::zeros(m, n);
            for (entry, &val) in jit_fn.jacobian_pattern().iter().zip(jac_values.iter()) {
                let row = entry.row as usize;
                let col = entry.col as usize;
                if row < m && col < n {
                    j[(row, col)] = val;
                }
            }

            // Solve J * delta = -r for the Newton step
            let delta = match self.solve_linear(&j, &(-&r)) {
                Some(d) => d,
                None => {
                    return SolveResult::Failed {
                        error: SolveError::SingularJacobian,
                    };
                }
            };

            // Update solution
            x += delta;
        }

        // Did not converge within max iterations
        let norm: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

        SolveResult::NotConverged {
            solution: x.as_slice().to_vec(),
            iterations: self.config.max_iterations,
            residual_norm: norm,
        }
    }

    /// Solve using interpreted (non-JIT) evaluation.
    fn solve_interpreted<P: Problem>(&self, problem: &P, x0: &[f64]) -> SolveResult {
        let n = problem.variable_count();
        let m = problem.residual_count();

        let mut x = DVector::from_column_slice(x0);

        for iteration in 0..self.config.max_iterations {
            // Compute residuals
            let residuals = problem.residuals(x.as_slice());

            // Check for non-finite residuals
            if residuals.iter().any(|r| !r.is_finite()) {
                return SolveResult::Failed {
                    error: SolveError::NonFiniteResiduals,
                };
            }

            let r = DVector::from_column_slice(&residuals);
            let norm = r.norm();

            // Check convergence
            if norm < self.config.tolerance {
                return SolveResult::Converged {
                    solution: x.as_slice().to_vec(),
                    iterations: iteration,
                    residual_norm: norm,
                };
            }

            // Compute Jacobian
            let jac_entries = problem.jacobian(x.as_slice());

            // Check for non-finite Jacobian entries
            if jac_entries.iter().any(|(_, _, v)| !v.is_finite()) {
                return SolveResult::Failed {
                    error: SolveError::NonFiniteJacobian,
                };
            }

            let mut j = DMatrix::zeros(m, n);
            for (row, col, val) in jac_entries {
                if row < m && col < n {
                    j[(row, col)] = val;
                }
            }

            // Solve J * delta = -r for the Newton step
            let delta = match self.solve_linear(&j, &(-&r)) {
                Some(d) => d,
                None => {
                    return SolveResult::Failed {
                        error: SolveError::SingularJacobian,
                    };
                }
            };

            // Update solution
            x += delta;
        }

        // Did not converge within max iterations
        let residuals = problem.residuals(x.as_slice());
        let norm: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

        SolveResult::NotConverged {
            solution: x.as_slice().to_vec(),
            iterations: self.config.max_iterations,
            residual_norm: norm,
        }
    }

    /// Solve the linear system J * delta = rhs.
    fn solve_linear(&self, j: &DMatrix<f64>, rhs: &DVector<f64>) -> Option<DVector<f64>> {
        let n_rows = j.nrows();
        let n_cols = j.ncols();

        if n_rows == n_cols {
            // Square system: try LU decomposition first
            if let Some(solution) = j.clone().lu().solve(rhs) {
                return Some(solution);
            }
        }

        // Rectangular or singular: use SVD-based pseudoinverse
        let svd = j.clone().svd(true, true);
        svd.solve(rhs, 1e-10).ok()
    }

    /// Compile constraints for JIT evaluation.
    ///
    /// Returns a JIT function that can be reused for multiple solves.
    pub fn compile(&mut self, constraints: &CompiledConstraints) -> Result<JITFunction, JITError> {
        // Ensure compiler is initialized
        if self.compiler.is_none() {
            self.compiler = Some(JITCompiler::new()?);
        }

        let compiler = self.compiler.as_mut().ok_or(JITError::NotAvailable)?;
        compiler.compile(constraints)
    }
}

impl Default for JITSolver {
    fn default() -> Self {
        Self::default_solver()
    }
}

/// Result of JIT compilation attempt.
#[derive(Debug)]
pub enum JITCompilationResult {
    /// Successfully compiled.
    Compiled(JITFunction),

    /// JIT is not available on this platform.
    NotAvailable,

    /// Compilation failed.
    Failed(JITError),

    /// Problem is too small for JIT to be beneficial.
    TooSmall,
}

/// Try to compile a problem for JIT evaluation.
pub fn try_compile(constraints: &CompiledConstraints, threshold: usize) -> JITCompilationResult {
    if !crate::jit::jit_available() {
        return JITCompilationResult::NotAvailable;
    }

    let estimated_work = constraints.n_residuals * 50;
    if estimated_work < threshold {
        return JITCompilationResult::TooSmall;
    }

    match JITCompiler::new() {
        Ok(mut compiler) => match compiler.compile(constraints) {
            Ok(jit_fn) => JITCompilationResult::Compiled(jit_fn),
            Err(e) => JITCompilationResult::Failed(e),
        },
        Err(e) => JITCompilationResult::Failed(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleProblem;

    impl Problem for SimpleProblem {
        fn name(&self) -> &str {
            "simple"
        }

        fn residual_count(&self) -> usize {
            1
        }

        fn variable_count(&self) -> usize {
            1
        }

        fn residuals(&self, x: &[f64]) -> Vec<f64> {
            vec![x[0] * x[0] - 2.0]
        }

        fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
            vec![(0, 0, 2.0 * x[0])]
        }

        fn initial_point(&self, factor: f64) -> Vec<f64> {
            vec![1.0 * factor]
        }
    }

    #[test]
    fn test_jit_solver_interpreted() {
        let mut solver = JITSolver::new(JITConfig::always_interpreted());
        let result = solver.solve(&SimpleProblem, &[1.5]);

        assert!(result.is_converged());
        let solution = result.solution().expect("should have solution");
        assert!(
            (solution[0] - std::f64::consts::SQRT_2).abs() < 1e-6,
            "solution should be sqrt(2), got {}",
            solution[0]
        );
    }

    #[test]
    fn test_jit_solver_config() {
        let config = JITConfig::default();
        assert_eq!(config.jit_threshold, 1000);
        assert_eq!(config.max_iterations, 200);
        assert!(!config.force_jit);
        assert!(!config.force_interpreted);
    }

    #[test]
    fn test_will_use_jit() {
        // Small problem should not use JIT
        let solver = JITSolver::new(JITConfig::default());
        assert!(!solver.will_use_jit(&SimpleProblem));

        // Force JIT should use JIT (if available)
        let solver_jit = JITSolver::new(JITConfig::always_jit());
        assert_eq!(
            solver_jit.will_use_jit(&SimpleProblem),
            crate::jit::jit_available()
        );

        // Force interpreted should not use JIT
        let solver_interp = JITSolver::new(JITConfig::always_interpreted());
        assert!(!solver_interp.will_use_jit(&SimpleProblem));
    }

    #[test]
    fn test_jit_solver_dimension_mismatch() {
        let mut solver = JITSolver::default();
        let result = solver.solve(&SimpleProblem, &[1.0, 2.0]);

        assert!(!result.is_converged());
        assert!(!result.is_completed());
        assert!(matches!(
            result.error(),
            Some(SolveError::DimensionMismatch { .. })
        ));
    }
}
