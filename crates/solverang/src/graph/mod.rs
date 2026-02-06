pub mod union_find;
pub mod incremental;
pub mod dm;
pub mod redundancy;
pub mod diagnostics;

pub use incremental::{IncrementalGraph, EntityMeta, ConstraintMeta};
pub use dm::{DMDecomposition, DMBlock};
pub use diagnostics::{DOFAnalysis, ConstraintStatus};
pub use union_find::ComponentId;
