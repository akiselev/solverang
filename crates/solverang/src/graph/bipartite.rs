//! Bipartite entity-constraint graph.
//!
//! [`ConstraintGraph`] maintains a bipartite adjacency between entities and
//! constraints, plus a secondary index from parameters to the constraints
//! that depend on them. It is the source of truth for structural queries
//! such as "which constraints touch this entity?" and provides the edge
//! list needed by the union-find decomposer.

use std::collections::HashMap;

use crate::constraint::Constraint;
use crate::entity::Entity;
use crate::id::{EntityId, ParamId};
use crate::param::ParamStore;

/// Bipartite graph connecting entities to constraints, with a secondary
/// parameter-to-constraint index.
///
/// Constraints are identified by their index into the system's constraint
/// vector (`usize`), **not** by [`ConstraintId`](crate::id::ConstraintId).
/// This keeps the graph independent of generational bookkeeping and makes
/// integration with the flat constraint vec straightforward.
#[derive(Clone, Debug)]
pub struct ConstraintGraph {
    /// entity_id -> list of constraint indices that reference it.
    entity_to_constraints: HashMap<EntityId, Vec<usize>>,
    /// constraint index -> list of entity IDs it binds.
    constraint_to_entities: HashMap<usize, Vec<EntityId>>,
    /// param_id -> list of constraint indices that depend on it.
    param_to_constraints: HashMap<ParamId, Vec<usize>>,
}

impl ConstraintGraph {
    /// Create an empty constraint graph.
    pub fn new() -> Self {
        Self {
            entity_to_constraints: HashMap::new(),
            constraint_to_entities: HashMap::new(),
            param_to_constraints: HashMap::new(),
        }
    }

    /// Register an entity so it appears in the graph.
    ///
    /// This inserts an empty adjacency list for the entity if it was not
    /// already present. It does **not** add any constraint edges — those
    /// are created via [`add_constraint`](Self::add_constraint).
    pub fn add_entity(&mut self, entity: &dyn Entity) {
        self.entity_to_constraints
            .entry(entity.id())
            .or_default();
    }

    /// Add a constraint and wire up all its entity and parameter edges.
    ///
    /// `idx` is the constraint's position in the system's constraint vec.
    pub fn add_constraint(&mut self, idx: usize, constraint: &dyn Constraint) {
        // Wire entity <-> constraint edges.
        let entity_ids = constraint.entity_ids().to_vec();
        for &eid in &entity_ids {
            self.entity_to_constraints
                .entry(eid)
                .or_default()
                .push(idx);
        }
        self.constraint_to_entities.insert(idx, entity_ids);

        // Wire param -> constraint index.
        for &pid in constraint.param_ids() {
            self.param_to_constraints
                .entry(pid)
                .or_default()
                .push(idx);
        }
    }

    /// Remove an entity and all of its adjacency edges.
    ///
    /// This only removes the entity's own adjacency list.  It does **not**
    /// remove any constraints that reference the entity — call
    /// [`remove_constraint`](Self::remove_constraint) first for those.
    pub fn remove_entity(&mut self, id: EntityId) {
        self.entity_to_constraints.remove(&id);
    }

    /// Remove a constraint and clean up all adjacency lists.
    ///
    /// `idx` is the constraint's position in the system's constraint vec.
    pub fn remove_constraint(&mut self, idx: usize, constraint: &dyn Constraint) {
        // Remove from entity -> constraint adjacency.
        for &eid in constraint.entity_ids() {
            if let Some(list) = self.entity_to_constraints.get_mut(&eid) {
                list.retain(|&i| i != idx);
            }
        }

        // Remove from param -> constraint adjacency.
        for &pid in constraint.param_ids() {
            if let Some(list) = self.param_to_constraints.get_mut(&pid) {
                list.retain(|&i| i != idx);
            }
        }

        // Remove the constraint's own adjacency entry.
        self.constraint_to_entities.remove(&idx);
    }

    /// Constraint indices that reference the given entity.
    ///
    /// Returns an empty slice if the entity is unknown.
    pub fn constraints_for_entity(&self, id: EntityId) -> &[usize] {
        self.entity_to_constraints
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Entity IDs bound by the constraint at `idx`.
    ///
    /// Returns an empty slice if the index is unknown.
    pub fn entities_for_constraint(&self, idx: usize) -> &[EntityId] {
        self.constraint_to_entities
            .get(&idx)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Constraint indices that depend on the given parameter.
    ///
    /// Returns an empty slice if the parameter is unknown.
    pub fn constraints_for_param(&self, id: ParamId) -> &[usize] {
        self.param_to_constraints
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Number of registered entities.
    pub fn entity_count(&self) -> usize {
        self.entity_to_constraints.len()
    }

    /// Number of registered constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraint_to_entities.len()
    }

    /// Convert the graph to an edge list suitable for
    /// [`decompose_from_edges`](crate::decomposition::decompose_from_edges).
    ///
    /// Each returned pair is `(constraint_idx, param_col)` where `param_col`
    /// is the column index of a **free** parameter in the solver mapping.
    /// Fixed parameters are excluded because they do not couple constraints.
    pub fn to_constraint_variable_edges(&self, store: &ParamStore) -> Vec<(usize, usize)> {
        let mapping = store.build_solver_mapping();
        self.to_constraint_variable_edges_with_mapping(&mapping)
    }

    /// Like [`Self::to_constraint_variable_edges`] but uses a prebuilt
    /// [`crate::param::SolverMapping`] to avoid redundant mapping construction.
    pub fn to_constraint_variable_edges_with_mapping(
        &self,
        mapping: &crate::param::SolverMapping,
    ) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();

        // Iterate param -> constraint edges and emit (constraint_idx, col).
        for (&pid, constraint_indices) in &self.param_to_constraints {
            if let Some(&col) = mapping.param_to_col.get(&pid) {
                for &cidx in constraint_indices {
                    edges.push((cidx, col));
                }
            }
        }

        edges
    }

    /// The maximum constraint index registered in this graph, plus one.
    ///
    /// This is useful for sizing union-find structures that need to cover
    /// the full range of constraint indices, even if they are sparse.
    pub fn max_constraint_index(&self) -> usize {
        self.constraint_to_entities
            .keys()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    }
}

impl Default for ConstraintGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_empty_graph() {
        let g = ConstraintGraph::new();
        assert_eq!(g.entity_count(), 0);
        assert_eq!(g.constraint_count(), 0);
    }

    #[test]
    fn test_add_entity_and_constraint() {
        let eid = EntityId::new(0, 0);
        let pid = ParamId::new(0, 0);

        let entity = StubEntity {
            id: eid,
            params: vec![pid],
        };

        let constraint = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![pid],
        };

        let mut g = ConstraintGraph::new();
        g.add_entity(&entity);
        g.add_constraint(0, &constraint);

        assert_eq!(g.entity_count(), 1);
        assert_eq!(g.constraint_count(), 1);
        assert_eq!(g.constraints_for_entity(eid), &[0]);
        assert_eq!(g.entities_for_constraint(0), &[eid]);
        assert_eq!(g.constraints_for_param(pid), &[0]);
    }

    #[test]
    fn test_remove_constraint() {
        let eid = EntityId::new(0, 0);
        let pid = ParamId::new(0, 0);

        let entity = StubEntity {
            id: eid,
            params: vec![pid],
        };
        let constraint = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![pid],
        };

        let mut g = ConstraintGraph::new();
        g.add_entity(&entity);
        g.add_constraint(0, &constraint);
        g.remove_constraint(0, &constraint);

        assert_eq!(g.constraint_count(), 0);
        assert_eq!(g.constraints_for_entity(eid), &[] as &[usize]);
        assert_eq!(g.constraints_for_param(pid), &[] as &[usize]);
    }

    #[test]
    fn test_remove_entity() {
        let eid = EntityId::new(0, 0);

        let entity = StubEntity {
            id: eid,
            params: vec![],
        };

        let mut g = ConstraintGraph::new();
        g.add_entity(&entity);
        assert_eq!(g.entity_count(), 1);

        g.remove_entity(eid);
        assert_eq!(g.entity_count(), 0);
    }

    #[test]
    fn test_unknown_lookups_return_empty() {
        let g = ConstraintGraph::new();
        let eid = EntityId::new(99, 0);
        let pid = ParamId::new(99, 0);

        assert!(g.constraints_for_entity(eid).is_empty());
        assert!(g.entities_for_constraint(42).is_empty());
        assert!(g.constraints_for_param(pid).is_empty());
    }

    #[test]
    fn test_to_constraint_variable_edges() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let p0 = store.alloc(1.0, eid);
        let p1 = store.alloc(2.0, eid);

        let entity = StubEntity {
            id: eid,
            params: vec![p0, p1],
        };

        let c0 = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![p0],
        };
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![eid],
            params: vec![p1],
        };

        let mut g = ConstraintGraph::new();
        g.add_entity(&entity);
        g.add_constraint(0, &c0);
        g.add_constraint(1, &c1);

        let edges = g.to_constraint_variable_edges(&store);
        assert_eq!(edges.len(), 2);

        // Each constraint should map to a different column.
        let cols: Vec<usize> = edges.iter().map(|&(_, c)| c).collect();
        assert!(cols.contains(&0));
        assert!(cols.contains(&1));
    }

    #[test]
    fn test_fixed_params_excluded_from_edges() {
        let eid = EntityId::new(0, 0);
        let mut store = ParamStore::new();
        let p0 = store.alloc(1.0, eid);
        let p1 = store.alloc(2.0, eid);
        store.fix(p0);

        let entity = StubEntity {
            id: eid,
            params: vec![p0, p1],
        };

        let constraint = StubConstraint {
            id: ConstraintId::new(0, 0),
            entities: vec![eid],
            params: vec![p0, p1],
        };

        let mut g = ConstraintGraph::new();
        g.add_entity(&entity);
        g.add_constraint(0, &constraint);

        let edges = g.to_constraint_variable_edges(&store);
        // Only p1 is free, so only one edge.
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_multiple_constraints_sharing_param() {
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
        let c1 = StubConstraint {
            id: ConstraintId::new(1, 0),
            entities: vec![eid],
            params: vec![p0],
        };

        let mut g = ConstraintGraph::new();
        g.add_entity(&entity);
        g.add_constraint(0, &c0);
        g.add_constraint(1, &c1);

        // Both constraints depend on the same param -> same column.
        let edges = g.to_constraint_variable_edges(&store);
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].1, edges[1].1);
    }
}
