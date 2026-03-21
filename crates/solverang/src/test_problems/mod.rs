//! MINPACK Test Problem Suite
//!
//! This module provides implementations of the More-Garbow-Hillstrom (MGH) test
//! problems and MINPACK nonlinear equation test problems for validating and
//! benchmarking the constraint solver.
//!
//! # Problem Categories
//!
//! ## Least-Squares Problems (MGH, from test_lmder.f90)
//! - 18 standard problems for testing least-squares solvers
//! - Overdetermined systems (m equations, n variables, m >= n)
//! - 53 test cases with varying dimensions and starting point factors
//!
//! ## Nonlinear Equation Problems (from test_hybrj.f90)
//! - 14 standard problems for testing root-finding solvers
//! - Square systems (n equations, n variables)
//! - 55 test cases with varying dimensions and starting point factors
//!
//! # Usage
//!
//! ```rust
//! use solverang::test_problems::{Rosenbrock, HelicalValley};
//! use solverang::{Problem, Solver, SolverConfig};
//!
//! // Get a specific problem
//! let problem = Rosenbrock;
//!
//! // Evaluate at a point
//! let x = problem.initial_point(1.0);
//! let residuals = problem.residuals(&x);
//! let jacobian = problem.jacobian(&x);
//!
//! // Solve the problem
//! let solver = Solver::new(SolverConfig::default());
//! let result = solver.solve(&problem, &x);
//! ```
//!
//! # References
//!
//! - More, J.J., Garbow, B.S., Hillstrom, K.E.: Testing Unconstrained
//!   Optimization Software. ACM Trans. Math. Softw. 7, 17-41 (1981)
//! - MINPACK library: <https://github.com/fortran-lang/minpack>

// Least-squares problems (from test_lmder.f90)
mod bard;
mod box3d;
mod brown_almost_linear;
mod brown_dennis;
mod chebyquad;
mod freudenstein_roth;
mod helical_valley;
mod jennrich_sampson;
mod kowalik_osborne;
mod linear_full_rank;
mod linear_rank1;
mod linear_rank1_zero_columns;
mod meyer;
mod osborne1;
mod osborne2;
mod powell_singular;
mod rosenbrock;
mod watson;

// Nonlinear equation problems (from test_hybrj.f90)
mod broyden_banded;
mod broyden_tridiagonal;
mod discrete_boundary_value;
mod discrete_integral_equation;
mod powell_badly_scaled;
mod trigonometric;
mod variably_dimensioned;
mod wood;

// Test data module
mod test_data;

// NIST StRD nonlinear regression problems (feature-gated)
#[cfg(feature = "nist")]
pub mod nist;

// Re-exports for least-squares problems
pub use bard::Bard;
pub use box3d::Box3D;
pub use brown_almost_linear::BrownAlmostLinear;
pub use brown_dennis::BrownDennis;
pub use chebyquad::Chebyquad;
pub use freudenstein_roth::FreudensteinRoth;
pub use helical_valley::HelicalValley;
pub use jennrich_sampson::JennrichSampson;
pub use kowalik_osborne::KowalikOsborne;
pub use linear_full_rank::LinearFullRank;
pub use linear_rank1::LinearRank1;
pub use linear_rank1_zero_columns::LinearRank1ZeroColumns;
pub use meyer::Meyer;
pub use osborne1::Osborne1;
pub use osborne2::Osborne2;
pub use powell_singular::PowellSingular;
pub use rosenbrock::Rosenbrock;
pub use watson::Watson;

// Re-exports for nonlinear equation problems
pub use broyden_banded::BroydenBanded;
pub use broyden_tridiagonal::BroydenTridiagonal;
pub use discrete_boundary_value::DiscreteBoundaryValue;
pub use discrete_integral_equation::DiscreteIntegralEquation;
pub use powell_badly_scaled::PowellBadlyScaled;
pub use trigonometric::Trigonometric;
pub use variably_dimensioned::VariablyDimensioned;
pub use wood::Wood;

// Re-export test data
pub use test_data::{
    all_least_squares_test_cases, all_nonlinear_test_cases, create_problem_for_case,
    LeastSquaresTestCase, NonlinearTestCase, TestCaseId, INFO_ORIGINAL,
};

use crate::Problem;

/// Get a least-squares problem by its MGH problem number (1-18).
///
/// Returns None for invalid problem numbers.
pub fn least_squares_problem(number: usize) -> Option<Box<dyn Problem>> {
    match number {
        1 => Some(Box::new(LinearFullRank::default())),
        2 => Some(Box::new(LinearRank1::default())),
        3 => Some(Box::new(LinearRank1ZeroColumns::default())),
        4 => Some(Box::new(Rosenbrock)),
        5 => Some(Box::new(HelicalValley)),
        6 => Some(Box::new(PowellSingular)),
        7 => Some(Box::new(FreudensteinRoth)),
        8 => Some(Box::new(Bard)),
        9 => Some(Box::new(KowalikOsborne)),
        10 => Some(Box::new(Meyer)),
        11 => Some(Box::new(Watson::new(6))),
        12 => Some(Box::new(Box3D::default())),
        13 => Some(Box::new(JennrichSampson::default())),
        14 => Some(Box::new(BrownDennis::default())),
        15 => Some(Box::new(Chebyquad::new(5))),
        16 => Some(Box::new(BrownAlmostLinear::new(5))),
        17 => Some(Box::new(Osborne1)),
        18 => Some(Box::new(Osborne2)),
        _ => None,
    }
}

/// Get a nonlinear equation problem by its HYBRJ problem number (1-14).
///
/// Returns None for invalid problem numbers.
pub fn nonlinear_problem(number: usize) -> Option<Box<dyn Problem>> {
    match number {
        1 => Some(Box::new(Rosenbrock)),     // Same as MGH #4
        2 => Some(Box::new(PowellSingular)), // Same as MGH #6
        3 => Some(Box::new(PowellBadlyScaled)),
        4 => Some(Box::new(Wood)),
        5 => Some(Box::new(HelicalValley)), // Same as MGH #5
        6 => Some(Box::new(Watson::new(6))),
        7 => Some(Box::new(Chebyquad::new(5))),
        8 => Some(Box::new(BrownAlmostLinear::new(5))),
        9 => Some(Box::new(DiscreteBoundaryValue::new(10))),
        10 => Some(Box::new(DiscreteIntegralEquation::new(10))),
        11 => Some(Box::new(Trigonometric::new(10))),
        12 => Some(Box::new(VariablyDimensioned::new(10))),
        13 => Some(Box::new(BroydenTridiagonal::new(10))),
        14 => Some(Box::new(BroydenBanded::new(10))),
        _ => None,
    }
}

/// Get all problem names for the least-squares suite.
pub fn least_squares_problem_names() -> Vec<&'static str> {
    vec![
        "Linear Full Rank",
        "Linear Rank 1",
        "Linear Rank 1 Zero Columns",
        "Rosenbrock",
        "Helical Valley",
        "Powell Singular",
        "Freudenstein-Roth",
        "Bard",
        "Kowalik-Osborne",
        "Meyer",
        "Watson",
        "Box 3D",
        "Jennrich-Sampson",
        "Brown-Dennis",
        "Chebyquad",
        "Brown Almost-Linear",
        "Osborne 1",
        "Osborne 2",
    ]
}

/// Get all problem names for the nonlinear equations suite.
pub fn nonlinear_problem_names() -> Vec<&'static str> {
    vec![
        "Rosenbrock",
        "Powell Singular",
        "Powell Badly Scaled",
        "Wood",
        "Helical Valley",
        "Watson",
        "Chebyquad",
        "Brown Almost-Linear",
        "Discrete Boundary Value",
        "Discrete Integral Equation",
        "Trigonometric",
        "Variably Dimensioned",
        "Broyden Tridiagonal",
        "Broyden Banded",
    ]
}
