//! Default implementation of the [`Analyze`] pipeline phase.
//!
//! [`DefaultAnalyze`] wraps the existing analysis modules
//! ([`graph::redundancy`], [`graph::dof`], [`graph::pattern`]) behind the
//! [`Analyze`] trait, allowing the user to toggle individual analyses and
//! configure tolerances.
//!
//! [`NoopAnalyze`] is a zero-cost alternative that skips all analysis and
//! returns an empty [`ClusterAnalysis`].

use crate::constraint::Constraint;
use crate::entity::Entity;
use crate::graph::dof::analyze_dof;
use crate::graph::pattern::detect_patterns;
use crate::graph::redundancy::analyze_redundancy;
use crate::param::ParamStore;
use crate::system::DiagnosticIssue;

use super::traits::Analyze;
use super::types::{ClusterAnalysis, ClusterData};

// ---------------------------------------------------------------------------
// DefaultAnalyze
// ---------------------------------------------------------------------------

/// Default analyzer that delegates to `graph::redundancy`, `graph::dof`, and
/// `graph::pattern`.
///
/// Each sub-analysis can be independently enabled or disabled via boolean
/// flags. The `tolerance` field controls the SVD singular-value cutoff used
/// by redundancy analysis.
pub struct DefaultAnalyze {
    /// Whether to run DOF analysis.
    pub run_dof: bool,
    /// Whether to run redundancy / conflict analysis.
    pub run_redundancy: bool,
    /// Whether to run solvable-pattern detection.
    pub run_patterns: bool,
    /// SVD tolerance for redundancy analysis.
    pub tolerance: f64,
}

impl Default for DefaultAnalyze {
    fn default() -> Self {
        Self {
            run_dof: true,
            run_redundancy: true,
            run_patterns: true,
            tolerance: 1e-10,
        }
    }
}

impl Analyze for DefaultAnalyze {
    fn analyze(
        &self,
        cluster: &ClusterData,
        constraints: &[Option<Box<dyn Constraint>>],
        entities: &[Option<Box<dyn Entity>>],
        store: &ParamStore,
    ) -> ClusterAnalysis {
        // --- Collect constraint references ---
        let constraint_refs: Vec<(usize, &dyn Constraint)> = cluster
            .constraint_indices
            .iter()
            .filter_map(|&idx| {
                constraints
                    .get(idx)
                    .and_then(|opt| opt.as_deref())
                    .map(|c| (idx, c))
            })
            .collect();

        // --- Collect entity references ---
        // Entity IDs have a raw_index() that corresponds to the index in the
        // system's entities vec.
        let entity_refs: Vec<&dyn Entity> = cluster
            .entity_ids
            .iter()
            .filter_map(|eid| {
                let idx = eid.raw_index() as usize;
                entities
                    .get(idx)
                    .and_then(|opt| opt.as_deref())
            })
            .collect();

        // --- Build solver mapping ---
        let mapping = store.build_solver_mapping_for(&cluster.param_ids);

        let mut diagnostics = Vec::new();

        // --- Redundancy analysis ---
        let redundancy = if self.run_redundancy {
            let result = analyze_redundancy(
                &constraint_refs,
                store,
                &mapping,
                self.tolerance,
            );
            // Convert redundant constraints to diagnostics.
            for rc in &result.redundant {
                diagnostics.push(DiagnosticIssue::RedundantConstraint {
                    constraint: rc.id,
                    implied_by: vec![],
                });
            }
            // Convert conflict groups to diagnostics.
            for cg in &result.conflicts {
                diagnostics.push(DiagnosticIssue::ConflictingConstraints {
                    constraints: cg.constraint_ids.clone(),
                });
            }
            Some(result)
        } else {
            None
        };

        // --- DOF analysis ---
        let dof = if self.run_dof {
            let result = analyze_dof(&entity_refs, &constraint_refs, store, &mapping);
            // Convert under-constrained entities to diagnostics.
            for ed in &result.entities {
                if ed.dof > 0 {
                    diagnostics.push(DiagnosticIssue::UnderConstrained {
                        entity: ed.entity_id,
                        free_directions: ed.dof,
                    });
                }
            }
            Some(result)
        } else {
            None
        };

        // --- Pattern detection ---
        let patterns = if self.run_patterns {
            detect_patterns(&entity_refs, &constraint_refs, store)
        } else {
            Vec::new()
        };

        ClusterAnalysis {
            cluster_id: cluster.id,
            dof,
            redundancy,
            patterns,
            diagnostics,
        }
    }
}

// ---------------------------------------------------------------------------
// NoopAnalyze
// ---------------------------------------------------------------------------

/// A no-op analyzer that skips all analysis.
///
/// Returns an empty [`ClusterAnalysis`] with the correct cluster ID.
/// Useful when speed is more important than diagnostics.
pub struct NoopAnalyze;

impl Analyze for NoopAnalyze {
    fn analyze(
        &self,
        cluster: &ClusterData,
        _constraints: &[Option<Box<dyn Constraint>>],
        _entities: &[Option<Box<dyn Entity>>],
        _store: &ParamStore,
    ) -> ClusterAnalysis {
        ClusterAnalysis {
            cluster_id: cluster.id,
            dof: None,
            redundancy: None,
            patterns: Vec::new(),
            diagnostics: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::Constraint;
    use crate::entity::Entity;
    use crate::id::{ClusterId, ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    // -- Stub entity ---------------------------------------------------------

    struct StubEntity {
        id: EntityId,
        params: Vec<ParamId>,
    }

    impl Entity for StubEntity {
        fn id(&self) -> EntityId {
            self.id
        }
        fn params(&self) -> &[ParamId] {
            &self.params
        }
        fn name(&self) -> &str {
            "stub"
        }
    }

    // -- Stub constraint -----------------------------------------------------

    struct StubConstraint {
        id: ConstraintId,
        entities: Vec<EntityId>,
        params: Vec<ParamId>,
        neq: usize,
        residual_fn: Box<dyn Fn(&ParamStore) -> Vec<f64> + Send + Sync>,
        jacobian_fn: Box<dyn Fn(&ParamStore) -> Vec<(usize, ParamId, f64)> + Send + Sync>,
    }

    impl Constraint for StubConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "stub"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entities
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            self.neq
        }
        fn residuals(&self, store: &ParamStore) -> Vec<f64> {
            (self.residual_fn)(store)
        }
        fn jacobian(&self, store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            (self.jacobian_fn)(store)
        }
    }

    // -- Tests ---------------------------------------------------------------

    #[test]
    fn noop_analyze_returns_empty() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);

        let entity: Box<dyn Entity> = Box::new(StubEntity {
            id: eid,
            params: vec![px],
        });

        let cid = ConstraintId::new(0, 0);
        let constraint: Box<dyn Constraint> = Box::new(StubConstraint {
            id: cid,
            entities: vec![eid],
            params: vec![px],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(px) - 1.0]),
            jacobian_fn: Box::new(move |_| vec![(0, px, 1.0)]),
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(constraint)];
        let entities: Vec<Option<Box<dyn Entity>>> = vec![Some(entity)];

        let cluster = ClusterData {
            id: ClusterId(0),
            constraint_indices: vec![0],
            param_ids: vec![px],
            entity_ids: vec![eid],
        };

        let analyzer = NoopAnalyze;
        let result = analyzer.analyze(&cluster, &constraints, &entities, &store);

        assert_eq!(result.cluster_id, ClusterId(0));
        assert!(result.dof.is_none());
        assert!(result.redundancy.is_none());
        assert!(result.patterns.is_empty());
        assert!(result.diagnostics.is_empty());
    }

    #[test]
    fn default_analyze_skips_patterns_when_disabled() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);

        let entity: Box<dyn Entity> = Box::new(StubEntity {
            id: eid,
            params: vec![px],
        });

        let cid = ConstraintId::new(0, 0);
        let constraint: Box<dyn Constraint> = Box::new(StubConstraint {
            id: cid,
            entities: vec![eid],
            params: vec![px],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(px) - 1.0]),
            jacobian_fn: Box::new(move |_| vec![(0, px, 1.0)]),
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(constraint)];
        let entities: Vec<Option<Box<dyn Entity>>> = vec![Some(entity)];

        let cluster = ClusterData {
            id: ClusterId(0),
            constraint_indices: vec![0],
            param_ids: vec![px],
            entity_ids: vec![eid],
        };

        let analyzer = DefaultAnalyze {
            run_dof: true,
            run_redundancy: true,
            run_patterns: false,
            tolerance: 1e-10,
        };
        let result = analyzer.analyze(&cluster, &constraints, &entities, &store);

        assert_eq!(result.cluster_id, ClusterId(0));
        // DOF and redundancy should be present.
        assert!(result.dof.is_some());
        assert!(result.redundancy.is_some());
        // Patterns should be empty because we disabled them.
        assert!(result.patterns.is_empty());
    }

    #[test]
    fn default_analyze_detects_under_constrained() {
        // Entity with 2 free params but only 1 constraint => 1 DOF remaining.
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);
        let py = store.alloc(2.0, eid);

        let entity: Box<dyn Entity> = Box::new(StubEntity {
            id: eid,
            params: vec![px, py],
        });

        let cid = ConstraintId::new(0, 0);
        let constraint: Box<dyn Constraint> = Box::new(StubConstraint {
            id: cid,
            entities: vec![eid],
            params: vec![px],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(px) - 1.0]),
            jacobian_fn: Box::new(move |_| vec![(0, px, 1.0)]),
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(constraint)];
        let entities: Vec<Option<Box<dyn Entity>>> = vec![Some(entity)];

        let cluster = ClusterData {
            id: ClusterId(0),
            constraint_indices: vec![0],
            param_ids: vec![px, py],
            entity_ids: vec![eid],
        };

        let analyzer = DefaultAnalyze {
            run_dof: true,
            run_redundancy: false,
            run_patterns: false,
            tolerance: 1e-10,
        };
        let result = analyzer.analyze(&cluster, &constraints, &entities, &store);

        // Should have an UnderConstrained diagnostic for the entity.
        let under_constrained: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| matches!(d, DiagnosticIssue::UnderConstrained { .. }))
            .collect();
        assert!(
            !under_constrained.is_empty(),
            "Expected at least one UnderConstrained diagnostic"
        );
    }

    #[test]
    fn default_analyze_all_disabled_returns_empty_analysis() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let px = store.alloc(1.0, eid);

        let entity: Box<dyn Entity> = Box::new(StubEntity {
            id: eid,
            params: vec![px],
        });

        let cid = ConstraintId::new(0, 0);
        let constraint: Box<dyn Constraint> = Box::new(StubConstraint {
            id: cid,
            entities: vec![eid],
            params: vec![px],
            neq: 1,
            residual_fn: Box::new(move |s| vec![s.get(px) - 1.0]),
            jacobian_fn: Box::new(move |_| vec![(0, px, 1.0)]),
        });

        let constraints: Vec<Option<Box<dyn Constraint>>> = vec![Some(constraint)];
        let entities: Vec<Option<Box<dyn Entity>>> = vec![Some(entity)];

        let cluster = ClusterData {
            id: ClusterId(0),
            constraint_indices: vec![0],
            param_ids: vec![px],
            entity_ids: vec![eid],
        };

        let analyzer = DefaultAnalyze {
            run_dof: false,
            run_redundancy: false,
            run_patterns: false,
            tolerance: 1e-10,
        };
        let result = analyzer.analyze(&cluster, &constraints, &entities, &store);

        assert!(result.dof.is_none());
        assert!(result.redundancy.is_none());
        assert!(result.patterns.is_empty());
        assert!(result.diagnostics.is_empty());
    }
}
