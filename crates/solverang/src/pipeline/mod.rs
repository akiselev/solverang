//! Solve pipeline orchestration module.
//!
//! This module provides the infrastructure for extracting sub-problems from
//! constraint system components, selecting appropriate solvers, and orchestrating
//! the solve process across all dirty components.

pub mod extract;
pub mod select;
pub mod warm_start;
pub mod solve;

pub use solve::SolvePipeline;
pub use select::SolverSelection;
pub use extract::ComponentProblem;
