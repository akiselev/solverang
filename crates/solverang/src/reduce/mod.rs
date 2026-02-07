//! Symbolic reduction passes for the constraint solver.
//!
//! These modules implement pre-solve reductions that simplify the constraint
//! system before handing it to the numerical solver. Each pass identifies
//! structural opportunities to shrink the problem:
//!
//! - [`substitute`] -- Fixed-parameter elimination: removes parameters whose
//!   values are already known, and identifies constraints that become trivially
//!   satisfied once those values are substituted.
//!
//! - [`merge`] -- Coincident parameter merging: when a constraint enforces
//!   `param_a == param_b`, one parameter can be replaced by the other
//!   everywhere, reducing the variable count by one per merge.
//!
//! - [`eliminate`] -- Trivial constraint detection: when a single-equation
//!   constraint has exactly one free parameter, its value can be computed
//!   analytically and removed from the solve.

pub mod substitute;
pub mod merge;
pub mod eliminate;

pub use substitute::{analyze_substitutions, is_trivially_satisfied, SubstitutionResult};
pub use merge::{build_substitution_map, detect_merges, MergeResult, ParamMerge};
pub use eliminate::{apply_eliminations, detect_trivial_eliminations, TrivialElimination};
