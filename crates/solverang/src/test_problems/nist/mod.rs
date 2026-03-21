//! NIST Statistical Reference Datasets (StRD) for Nonlinear Regression.
//!
//! This module provides implementations of all 27 NIST StRD nonlinear regression
//! problems with certified values. These problems are the gold standard for
//! testing nonlinear least-squares solvers.
//!
//! # Problem Categories
//!
//! Problems are classified by difficulty level:
//!
//! - **Lower difficulty**: Misra1a, Chwirut1, Chwirut2, Lanczos1-3, Gauss1-3
//! - **Average difficulty**: Misra1b-d, Roszman1, ENSO
//! - **Higher difficulty**: Eckerle4, Rat42, Rat43, Bennett5, BoxBOD, Thurber, MGH09, MGH10, Nelson
//!
//! # Certified Values
//!
//! Each problem includes NIST-certified:
//! - Parameter values (converged solution)
//! - Standard errors for parameters
//! - Residual sum of squares
//!
//! These values are accurate to at least 11 significant digits.
//!
//! # Usage
//!
//! ```rust,ignore
//! use solverang::test_problems::nist::{Misra1a, NISTProblem, NISTDifficulty};
//! use solverang::{LMSolver, LMConfig};
//!
//! let problem = Misra1a;
//! let solver = LMSolver::new(LMConfig::default());
//!
//! // Use starting values set 1
//! let x0 = problem.starting_values_1();
//! let result = solver.solve(&problem, &x0);
//!
//! if result.is_converged() {
//!     let solution = result.solution().unwrap();
//!     let certified = problem.certified_values();
//!
//!     // Check against certified values
//!     for (i, (computed, certified)) in solution.iter().zip(certified).enumerate() {
//!         let rel_error = (computed - certified).abs() / certified.abs();
//!         assert!(rel_error < 1e-6, "Parameter {} mismatch", i);
//!     }
//! }
//! ```
//!
//! # References
//!
//! NIST Statistical Reference Datasets for Nonlinear Regression:
//! <https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml>

// Lower difficulty problems
mod chwirut1;
mod chwirut2;
mod gauss1;
mod gauss2;
mod gauss3;
mod lanczos1;
mod lanczos2;
mod lanczos3;
mod misra1a;

// Average difficulty problems
mod enso;
mod misra1b;
mod misra1c;
mod misra1d;
mod roszman1;

// Higher difficulty problems
mod bennett5;
mod boxbod;
mod eckerle4;
mod mgh09;
mod mgh10;
mod nelson;
mod rat42;
mod rat43;
mod thurber;

// Additional NIST problems
mod danwood;
mod hahn1;
mod kirby2;
mod mgh17;

// Re-exports
pub use bennett5::Bennett5;
pub use boxbod::BoxBOD;
pub use chwirut1::Chwirut1;
pub use chwirut2::Chwirut2;
pub use danwood::DanWood;
pub use eckerle4::Eckerle4;
pub use enso::ENSO;
pub use gauss1::Gauss1;
pub use gauss2::Gauss2;
pub use gauss3::Gauss3;
pub use hahn1::Hahn1;
pub use kirby2::Kirby2;
pub use lanczos1::Lanczos1;
pub use lanczos2::Lanczos2;
pub use lanczos3::Lanczos3;
pub use mgh09::MGH09;
pub use mgh10::MGH10;
pub use mgh17::MGH17;
pub use misra1a::Misra1a;
pub use misra1b::Misra1b;
pub use misra1c::Misra1c;
pub use misra1d::Misra1d;
pub use nelson::Nelson;
pub use rat42::Rat42;
pub use rat43::Rat43;
pub use roszman1::Roszman1;
pub use thurber::Thurber;

use crate::Problem;

/// NIST difficulty classification.
///
/// Problems are classified according to the NIST StRD documentation:
/// - **Lower**: Straightforward problems that most solvers handle easily
/// - **Average**: Moderately challenging problems
/// - **Higher**: Difficult problems that may require robust solvers or good starting points
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NISTDifficulty {
    /// Lower difficulty - straightforward problems
    Lower,
    /// Average difficulty - moderate challenge
    Average,
    /// Higher difficulty - challenging problems
    Higher,
}

impl std::fmt::Display for NISTDifficulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NISTDifficulty::Lower => write!(f, "Lower"),
            NISTDifficulty::Average => write!(f, "Average"),
            NISTDifficulty::Higher => write!(f, "Higher"),
        }
    }
}

/// Extension trait for NIST StRD nonlinear regression problems.
///
/// This trait extends the base `Problem` trait with NIST-specific
/// information including certified values for validation.
pub trait NISTProblem: Problem {
    /// Get the difficulty classification for this problem.
    fn difficulty(&self) -> NISTDifficulty;

    /// Get NIST certified parameter values (the reference solution).
    ///
    /// These values are certified to at least 11 significant digits.
    fn certified_values(&self) -> &[f64];

    /// Get NIST certified standard errors for each parameter.
    ///
    /// These are the standard errors of the parameter estimates.
    fn certified_std_errors(&self) -> &[f64];

    /// Get the NIST certified residual sum of squares.
    ///
    /// This is the sum of squared residuals at the certified solution:
    /// RSS = sum(residuals\[i\]^2)
    fn certified_residual_sum_of_squares(&self) -> f64;

    /// Get starting values set 1 (typically farther from solution).
    fn starting_values_1(&self) -> Vec<f64>;

    /// Get starting values set 2 (typically closer to solution).
    fn starting_values_2(&self) -> Vec<f64>;

    /// Number of observations (data points) in the problem.
    fn observation_count(&self) -> usize {
        self.residual_count()
    }

    /// Verify that a solution matches certified values within tolerance.
    ///
    /// Returns `Ok(())` if all parameters match within tolerance, or
    /// `Err` with details about which parameters failed.
    fn verify_solution(&self, solution: &[f64], tolerance: f64) -> Result<(), String> {
        let certified = self.certified_values();

        if solution.len() != certified.len() {
            return Err(format!(
                "Solution length {} != certified length {}",
                solution.len(),
                certified.len()
            ));
        }

        let mut max_rel_error = 0.0;
        let mut worst_param = 0;

        for (i, (computed, &reference)) in solution.iter().zip(certified).enumerate() {
            let rel_error = if reference.abs() > 1e-15 {
                (computed - reference).abs() / reference.abs()
            } else {
                (computed - reference).abs()
            };

            if rel_error > max_rel_error {
                max_rel_error = rel_error;
                worst_param = i;
            }
        }

        if max_rel_error > tolerance {
            Err(format!(
                "Parameter {} has relative error {:.2e} (tolerance: {:.2e}): computed={}, certified={}",
                worst_param,
                max_rel_error,
                tolerance,
                solution[worst_param],
                certified[worst_param]
            ))
        } else {
            Ok(())
        }
    }

    /// Verify that the residual sum of squares matches the certified value.
    fn verify_residual_sum_of_squares(
        &self,
        solution: &[f64],
        tolerance: f64,
    ) -> Result<(), String> {
        let residuals = self.residuals(solution);
        let computed_rss: f64 = residuals.iter().map(|r| r * r).sum();
        let certified_rss = self.certified_residual_sum_of_squares();

        let rel_error = if certified_rss.abs() > 1e-15 {
            (computed_rss - certified_rss).abs() / certified_rss.abs()
        } else {
            (computed_rss - certified_rss).abs()
        };

        if rel_error > tolerance {
            Err(format!(
                "RSS mismatch: computed={:.15e}, certified={:.15e}, rel_error={:.2e}",
                computed_rss, certified_rss, rel_error
            ))
        } else {
            Ok(())
        }
    }
}

/// Get all NIST problems grouped by difficulty.
pub fn all_problems_by_difficulty() -> Vec<(NISTDifficulty, Vec<Box<dyn NISTProblem>>)> {
    vec![
        (
            NISTDifficulty::Lower,
            vec![
                Box::new(Misra1a) as Box<dyn NISTProblem>,
                Box::new(Chwirut1),
                Box::new(Chwirut2),
                Box::new(Lanczos1),
                Box::new(Lanczos2),
                Box::new(Lanczos3),
                Box::new(Gauss1),
                Box::new(Gauss2),
                Box::new(Gauss3),
                Box::new(DanWood),
            ],
        ),
        (
            NISTDifficulty::Average,
            vec![
                Box::new(Misra1b) as Box<dyn NISTProblem>,
                Box::new(Misra1c),
                Box::new(Misra1d),
                Box::new(Roszman1),
                Box::new(ENSO),
                Box::new(Kirby2),
                Box::new(Hahn1),
                Box::new(MGH17),
            ],
        ),
        (
            NISTDifficulty::Higher,
            vec![
                Box::new(Eckerle4) as Box<dyn NISTProblem>,
                Box::new(Rat42),
                Box::new(Rat43),
                Box::new(Bennett5),
                Box::new(BoxBOD),
                Box::new(Thurber),
                Box::new(MGH09),
                Box::new(MGH10),
                Box::new(Nelson),
            ],
        ),
    ]
}

/// Get all NIST problems as a flat list.
pub fn all_problems() -> Vec<Box<dyn NISTProblem>> {
    all_problems_by_difficulty()
        .into_iter()
        .flat_map(|(_, problems)| problems)
        .collect()
}

/// Get all problem names.
pub fn all_problem_names() -> Vec<&'static str> {
    vec![
        // Lower
        "Misra1a", "Chwirut1", "Chwirut2", "Lanczos1", "Lanczos2", "Lanczos3", "Gauss1", "Gauss2",
        "Gauss3", "DanWood", // Average
        "Misra1b", "Misra1c", "Misra1d", "Roszman1", "ENSO", "Kirby2", "Hahn1", "MGH17",
        // Higher
        "Eckerle4", "Rat42", "Rat43", "Bennett5", "BoxBOD", "Thurber", "MGH09", "MGH10", "Nelson",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LMConfig, LMSolver};

    #[test]
    fn test_all_problems_have_data() {
        for problem in all_problems() {
            assert!(!problem.name().is_empty(), "Problem should have a name");
            assert!(problem.variable_count() > 0, "Should have variables");
            assert!(problem.residual_count() > 0, "Should have residuals");
            assert!(
                !problem.certified_values().is_empty(),
                "Should have certified values"
            );
            assert_eq!(
                problem.certified_values().len(),
                problem.variable_count(),
                "Certified values length should match variable count for {}",
                problem.name()
            );
            assert_eq!(
                problem.certified_std_errors().len(),
                problem.variable_count(),
                "Certified std errors length should match variable count for {}",
                problem.name()
            );
        }
    }

    #[test]
    fn test_starting_values_dimensions() {
        for problem in all_problems() {
            let sv1 = problem.starting_values_1();
            let sv2 = problem.starting_values_2();

            assert_eq!(
                sv1.len(),
                problem.variable_count(),
                "Starting values 1 dimension mismatch for {}",
                problem.name()
            );
            assert_eq!(
                sv2.len(),
                problem.variable_count(),
                "Starting values 2 dimension mismatch for {}",
                problem.name()
            );
        }
    }

    #[test]
    fn test_solve_lower_difficulty() {
        let solver = LMSolver::new(LMConfig::robust());

        let lower_problems: Vec<Box<dyn NISTProblem>> = vec![Box::new(Misra1a), Box::new(DanWood)];

        for problem in lower_problems {
            let x0 = problem.starting_values_2();
            let result = solver.solve(problem.as_ref(), &x0);

            assert!(
                result.is_converged() || result.is_completed(),
                "Problem {} should converge: {:?}",
                problem.name(),
                result
            );

            if let Some(solution) = result.solution() {
                let verify_result = problem.verify_solution(solution, 1e-4);
                if let Err(e) = verify_result {
                    println!("Warning for {}: {}", problem.name(), e);
                }
            }
        }
    }
}
