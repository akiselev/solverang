//! Decomposition of the constraint graph into independent rigid clusters.
//!
//! This module wraps the existing union-find decomposer
//! ([`decompose_from_edges`](crate::decomposition::decompose_from_edges)) so that
//! it works with the new ID-typed constraint graph. The flow is:
//!
//! 1. Build an edge list `(constraint_idx, solver_column)` from the
//!    [`ConstraintGraph`] using a [`SolverMapping`](crate::param::SolverMapping).
//! 2. Delegate to `decompose_from_edges` to find connected components.
//! 3. Convert each [`Component`](crate::decomposition::Component) back into a
//!    [`RigidCluster`] by gathering the proper `ParamId`s and `EntityId`s.

use std::collections::HashSet;

use crate::constraint::Constraint;
use crate::graph::bipartite::ConstraintGraph;
use crate::graph::cluster::RigidCluster;
use crate::id::ClusterId;
use crate::param::ParamStore;

/// Decompose the constraint graph into independent rigid clusters using
/// union-find.
///
/// Each returned [`RigidCluster`] contains a set of constraints that share
/// parameters (directly or transitively) and must be solved together.
/// Independent clusters can be solved in parallel.
///
/// # Arguments
///
/// * `graph`       - The bipartite entity-constraint graph.
/// * `constraints` - The system's constraint vec (indexed by constraint idx).
/// * `store`       - The parameter store (used to determine which params are free).
///
/// # Returns
///
/// A `Vec<RigidCluster>` sorted by the first constraint index in each cluster.
/// Each cluster is assigned a sequential [`ClusterId`].
pub fn decompose_clusters(
    graph: &ConstraintGraph,
    constraints: &[Box<dyn Constraint>],
    store: &ParamStore,
) -> Vec<RigidCluster> {
    if graph.constraint_count() == 0 {
        return Vec::new();
    }

    // Build the solver mapping once — we need it for the edge list and for
    // mapping columns back to ParamIds.
    let mapping = store.build_solver_mapping();

    // Build the edge list from the constraint graph, reusing the mapping
    // we already built rather than constructing a second one internally.
    let edges = graph.to_constraint_variable_edges_with_mapping(&mapping);

    // Use max_constraint_index() instead of constraint_count() so that
    // sparse (non-contiguous) constraint indices are handled correctly.
    let constraint_count = graph.max_constraint_index();

    // Delegate to the existing union-find decomposition.
    let components = crate::decomposition::decompose_from_edges(
        constraint_count,
        mapping.len(),
        &edges,
    );

    // Convert each Component into a RigidCluster.
    components
        .into_iter()
        .enumerate()
        .map(|(cluster_idx, component)| {
            // Gather param IDs: map column indices back to ParamIds.
            let param_ids: Vec<_> = component
                .variable_indices
                .iter()
                .filter_map(|&col| mapping.col_to_param.get(col).copied())
                .collect();

            // Gather entity IDs from all constraints in this component.
            let mut entity_set = HashSet::new();
            for &cidx in &component.constraint_indices {
                if let Some(c) = constraints.get(cidx) {
                    for &eid in c.entity_ids() {
                        entity_set.insert(eid);
                    }
                }
            }
            let entity_ids: Vec<_> = entity_set.into_iter().collect();

            RigidCluster::new(
                ClusterId(cluster_idx),
                component.constraint_indices,
                param_ids,
                entity_ids,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::Constraint;
    use crate::entity::Entity;
    use crate::graph::bipartite::ConstraintGraph;
    use crate::id::{ConstraintId, EntityId, ParamId};
    use crate::param::ParamStore;

    // ---------------------------------------------------------------
    // Minimal stub implementations for testing.
    // ---------------------------------------------------------------

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
            "stub_entity"
        }
    }

    struct StubConstraint {
        id: ConstraintId,
        entities: Vec<EntityId>,
        params: Vec<ParamId>,
    }

    impl Constraint for StubConstraint {
        fn id(&self) -> ConstraintId {
            self.id
        }
        fn name(&self) -> &str {
            "stub_constraint"
        }
        fn entity_ids(&self) -> &[EntityId] {
            &self.entities
        }
        fn param_ids(&self) -> &[ParamId] {
            &self.params
        }
        fn equation_count(&self) -> usize {
            1
        }
        fn residuals(&self, _store: &ParamStore) -> Vec<f64> {
            vec![0.0]
        }
        fn jacobian(&self, _store: &ParamStore) -> Vec<(usize, ParamId, f64)> {
            vec![]
        }
    }

    // ---------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------

    #[test]
    fn test_empty_graph_produces_no_clusters() {
        let g = ConstraintGraph::new();
        let constraints: Vec<Box<dyn Constraint>> = vec![];
        let store = ParamStore::new();

        let clusters = decompose_clusters(&g, &constraints, &store);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_single_constraint_single_cluster() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let p0 = store.alloc(1.0, eid);

        let entity = StubEntity {
            id: eid,
            params: vec![p0],
        };

        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![p0],
        };

        let mut g = ConstraintGraph::new();
        g.add_entity(&entity);
        g.add_constraint(0, &c0);

        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(c0)];
        let clusters = decompose_clusters(&g, &constraints, &store);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].constraint_indices, vec![0]);
        assert_eq!(clusters[0].param_ids.len(), 1);
        assert!(clusters[0].entity_ids.contains(&eid));
    }

    #[test]
    fn test_two_independent_clusters() {
        let e0 = EntityId::new(0, 0);
        let e1 = EntityId::new(1, 0);
        let mut store = ParamStore::new();
        let p0 = store.alloc(0.0, e0);
        let p1 = store.alloc(0.0, e1);

        let ent0 = StubEntity {
            id: e0,
            params: vec![p0],
        };
        let ent1 = StubEntity {
            id: e1,
            params: vec![p1],
        };

        // Constraint 0 depends on p0, constraint 1 depends on p1.
        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![e0],
            params: vec![p0],
        };
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![e1],
            params: vec![p1],
        };

        let mut g = ConstraintGraph::new();
        g.add_entity(&ent0);
        g.add_entity(&ent1);
        g.add_constraint(0, &c0);
        g.add_constraint(1, &c1);

        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(c0), Box::new(c1)];
        let clusters = decompose_clusters(&g, &constraints, &store);

        assert_eq!(clusters.len(), 2);
        // Each cluster should have exactly one constraint.
        for cluster in &clusters {
            assert_eq!(cluster.constraint_indices.len(), 1);
            assert_eq!(cluster.param_ids.len(), 1);
        }
    }

    #[test]
    fn test_shared_param_merges_into_one_cluster() {
        let e0 = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let shared_param = store.alloc(0.0, e0);

        let ent = StubEntity {
            id: e0,
            params: vec![shared_param],
        };

        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![e0],
            params: vec![shared_param],
        };
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![e0],
            params: vec![shared_param],
        };

        let mut g = ConstraintGraph::new();
        g.add_entity(&ent);
        g.add_constraint(0, &c0);
        g.add_constraint(1, &c1);

        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(c0), Box::new(c1)];
        let clusters = decompose_clusters(&g, &constraints, &store);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].constraint_indices.len(), 2);
    }

    #[test]
    fn test_fixed_params_do_not_couple() {
        let e0 = EntityId::new(0, 0);
        let e1 = EntityId::new(1, 0);
        let mut store = ParamStore::new();
        let p_free_0 = store.alloc(0.0, e0);
        let p_shared_fixed = store.alloc(0.0, e0);
        let p_free_1 = store.alloc(0.0, e1);

        // Fix the shared parameter so it no longer couples constraints.
        store.fix(p_shared_fixed);

        let ent0 = StubEntity {
            id: e0,
            params: vec![p_free_0, p_shared_fixed],
        };
        let ent1 = StubEntity {
            id: e1,
            params: vec![p_free_1, p_shared_fixed],
        };

        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![e0],
            params: vec![p_free_0, p_shared_fixed],
        };
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![e1],
            params: vec![p_free_1, p_shared_fixed],
        };

        let mut g = ConstraintGraph::new();
        g.add_entity(&ent0);
        g.add_entity(&ent1);
        g.add_constraint(0, &c0);
        g.add_constraint(1, &c1);

        let constraints: Vec<Box<dyn Constraint>> = vec![Box::new(c0), Box::new(c1)];
        let clusters = decompose_clusters(&g, &constraints, &store);

        // The fixed parameter should not couple the two constraints.
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_cluster_ids_are_sequential() {
        let e0 = EntityId::new(0, 0);
        let e1 = EntityId::new(1, 0);
        let e2 = EntityId::new(2, 0);
        let mut store = ParamStore::new();
        let p0 = store.alloc(0.0, e0);
        let p1 = store.alloc(0.0, e1);
        let p2 = store.alloc(0.0, e2);

        let ent0 = StubEntity { id: e0, params: vec![p0] };
        let ent1 = StubEntity { id: e1, params: vec![p1] };
        let ent2 = StubEntity { id: e2, params: vec![p2] };

        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![e0],
            params: vec![p0],
        };
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![e1],
            params: vec![p1],
        };
        let c2 = StubConstraint {
            id: ConstraintId::new(2, 0),
            entities: vec![e2],
            params: vec![p2],
        };

        let mut g = ConstraintGraph::new();
        g.add_entity(&ent0);
        g.add_entity(&ent1);
        g.add_entity(&ent2);
        g.add_constraint(0, &c0);
        g.add_constraint(1, &c1);
        g.add_constraint(2, &c2);

        let constraints: Vec<Box<dyn Constraint>> = vec![
            Box::new(c0),
            Box::new(c1),
            Box::new(c2),
        ];
        let clusters = decompose_clusters(&g, &constraints, &store);

        assert_eq!(clusters.len(), 3);
        for (i, cluster) in clusters.iter().enumerate() {
            assert_eq!(cluster.id, ClusterId(i));
        }
    }
}
