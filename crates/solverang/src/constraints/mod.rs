//! Constraint types and transformations.
//!
//! This module provides abstractions for different types of constraints
//! and transformations to convert between them.
//!
//! # Constraint Types
//!
//! - **Equality constraints**: g(x) = 0
//! - **Inequality constraints**: g(x) >= 0 (converted to equality via slack variables)
//! - **Bounds constraints**: lower <= x_i <= upper

mod inequality;

pub use inequality::{
    BoundsConstraint, ClearanceConstraint, InequalityConstraint, InequalityProblem,
    SlackVariableTransform,
};
