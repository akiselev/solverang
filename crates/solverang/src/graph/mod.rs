//! Constraint graph representation, decomposition, and analysis.
//!
//! This module provides the bipartite entity-constraint graph, rigid cluster
//! types for grouped solving, decomposition utilities that bridge the
//! new ID-based constraint system with the existing union-find decomposer,
//! and analysis tools for redundancy detection and DOF computation.

pub mod bipartite;
pub mod cluster;
pub mod decompose;
pub mod dof;
pub mod pattern;
pub mod redundancy;

pub use bipartite::ConstraintGraph;
pub use cluster::{ClusterStatus, RigidCluster};
pub use decompose::decompose_clusters;
pub use dof::{analyze_dof, quick_dof, DofAnalysis, EntityDof};
pub use redundancy::{
    analyze_redundancy, ConflictGroup, RedundancyAnalysis, RedundantConstraint,
};
