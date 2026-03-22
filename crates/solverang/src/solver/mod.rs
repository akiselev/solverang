//! Solver implementations for nonlinear systems.
//!
//! This module provides multiple solver implementations for finding roots of
//! nonlinear systems F(x) = 0 and least-squares problems min ||F(x)||^2.
//!
//! # Available Solvers
//!
//! - [`Solver`] (Newton-Raphson): Fast convergence for well-posed square systems
//! - [`LMSolver`] (Levenberg-Marquardt): Robust for over-constrained systems and poor starting points
//! - [`AutoSolver`]: Automatically selects the best solver based on problem characteristics
//! - [`RobustSolver`]: Tries NR first, falls back to LM if needed
//! - [`ParallelSolver`]: Decomposes into independent components and solves in parallel
//! - [`SparseSolver`]: Optimized for large, sparse systems
//!
//! # Choosing a Solver
//!
//! | System Type | Recommended Solver |
//! |-------------|-------------------|
//! | Square (m = n), good starting point | `Solver` (Newton-Raphson) |
//! | Square (m = n), uncertain starting point | `AutoSolver` or `RobustSolver` |
//! | Over-determined (m > n) | `LMSolver` or `AutoSolver` |
//! | Under-determined (m < n) | `LMSolver` |
//! | Large, sparse systems | `SparseSolver` |
//! | Independent sub-problems | `ParallelSolver` |
//!
//! # Example
//!
//! ```rust
//! use solverang::{Problem, AutoSolver, SolveResult};
//!
//! struct MyProblem;
//!
//! impl Problem for MyProblem {
//!     fn name(&self) -> &str { "example" }
//!     fn residual_count(&self) -> usize { 2 }
//!     fn variable_count(&self) -> usize { 2 }
//!     fn residuals(&self, x: &[f64]) -> Vec<f64> {
//!         vec![x[0] * x[0] - 1.0, x[1] - x[0]]
//!     }
//!     fn jacobian(&self, x: &[f64]) -> Vec<(usize, usize, f64)> {
//!         vec![(0, 0, 2.0 * x[0]), (0, 1, 0.0), (1, 0, -1.0), (1, 1, 1.0)]
//!     }
//!     fn initial_point(&self, f: f64) -> Vec<f64> { vec![0.5 * f, 0.5 * f] }
//! }
//!
//! let solver = AutoSolver::new();
//! let result = solver.solve(&MyProblem, &[0.5, 0.5]);
//! assert!(result.is_converged());
//! ```

mod auto;
pub mod bfgs;
mod config;
mod levenberg_marquardt;
mod lm_adapter;
mod lm_config;
mod newton_raphson;
mod parallel;
mod result;
mod sparse_solver;

#[cfg(feature = "jit")]
mod jit_solver;

// Newton-Raphson solver
pub use config::SolverConfig;
pub use newton_raphson::Solver;

// Levenberg-Marquardt solver
pub use levenberg_marquardt::LMSolver;
pub use lm_config::LMConfig;

// Auto-selection solvers
pub use auto::{AutoSolver, RobustSolver, SolverChoice};

// Parallel solver
pub use parallel::{ParallelSolver, ParallelSolverConfig};

// Sparse solver
pub use sparse_solver::{should_use_sparse, SparseSolver, SparseSolverConfig};

// Optimization solvers
pub use bfgs::BfgsSolver;

// Result types
pub use result::{SolveError, SolveResult};

// JIT solver
#[cfg(feature = "jit")]
pub use jit_solver::{try_compile, JITCompilationResult, JITSolver};
