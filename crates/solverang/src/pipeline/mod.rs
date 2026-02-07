//! Pluggable solve pipeline.
//!
//! Each phase can be independently swapped with a custom implementation.
//!
//! ```text
//! Decompose → Analyze → Reduce → Solve → PostProcess
//! ```

pub mod analyze;
pub mod decompose;
pub mod traits;
pub mod types;
pub mod post_process;
