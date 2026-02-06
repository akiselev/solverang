use std::collections::{HashMap, HashSet};
use super::union_find::{IncrementalUnionFind, ComponentId};
use crate::geometry::params::{EntityId, ConstraintId};

/// Metadata about an entity in the graph.
#[derive(Clone, Debug)]
pub struct EntityMeta {
    pub id: EntityId,
    /// Total number of parameters for this entity.
    pub param_count: usize,
    /// Number of fixed (non-variable) parameters.
    pub fixed_param_count: usize,
}

/// Metadata about a constraint in the graph.
#[derive(Clone, Debug)]
pub struct ConstraintMeta {
    pub id: ConstraintId,
    /// Number of scalar equations this constraint produces.
    pub equation_count: usize,
    /// Entities this constraint references.
    pub entity_deps: Vec<EntityId>,
    /// Raw parameter indices this constraint depends on.
    pub param_deps: Vec<usize>,
}

/// Incremental constraint graph with dirty tracking.
///
/// This is the central data structure for the differential dataflow layer.
/// It maintains a bipartite graph between entities and constraints, tracks
/// connected components incrementally, and marks components as dirty when
/// they need re-solving.
pub struct IncrementalGraph {
    // === Input state ===
    /// All entities in the graph.
    entities: HashMap<EntityId, EntityMeta>,
    /// All constraints in the graph.
    constraints: HashMap<ConstraintId, ConstraintMeta>,
    /// Entity → set of constraints referencing it.
    entity_to_constraints: HashMap<EntityId, HashSet<ConstraintId>>,
    /// Constraint → set of entities it references.
    constraint_to_entities: HashMap<ConstraintId, HashSet<EntityId>>,

    // === Incremental component tracking ===
    /// Mapping from EntityId to union-find index.
    entity_index: HashMap<EntityId, usize>,
    /// Reverse mapping: union-find index → EntityId.
    index_to_entity: Vec<EntityId>,
    /// Incremental union-find for component detection.
    uf: IncrementalUnionFind,

    // === Dirty tracking ===
    /// Components that need re-solving.
    dirty: HashSet<ComponentId>,

    // === Version tracking ===
    /// Monotonic version counter for change tracking.
    version: u64,
}

impl IncrementalGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            constraints: HashMap::new(),
            entity_to_constraints: HashMap::new(),
            constraint_to_entities: HashMap::new(),
            entity_index: HashMap::new(),
            index_to_entity: Vec::new(),
            uf: IncrementalUnionFind::new(0),
            dirty: HashSet::new(),
            version: 0,
        }
    }

    /// Add an entity to the graph.
    pub fn add_entity(&mut self, meta: EntityMeta) {
        let id = meta.id;

        // Allocate a union-find index for this entity
        let idx = self.index_to_entity.len();
        self.entity_index.insert(id, idx);
        self.index_to_entity.push(id);

        // Expand union-find to accommodate the new entity
        self.uf.resize(idx + 1);

        // Store metadata
        self.entities.insert(id, meta);
        self.entity_to_constraints.insert(id, HashSet::new());

        // New entity forms its own component, mark it dirty
        self.dirty.insert(ComponentId(self.uf.find(idx)));

        self.version += 1;
    }

    /// Remove an entity and all constraints referencing it.
    pub fn remove_entity(&mut self, id: EntityId) {
        if let Some(_meta) = self.entities.remove(&id) {
            // Find all constraints that reference this entity
            let constraints_to_remove: Vec<ConstraintId> = self.entity_to_constraints
                .get(&id)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .collect();

            // Remove all those constraints
            for cid in constraints_to_remove {
                self.remove_constraint(cid);
            }

            // Clean up entity mappings
            self.entity_to_constraints.remove(&id);

            // Note: We don't actually remove from union-find or compact indices
            // to avoid invalidating existing mappings. The entity index remains
            // allocated but unused. For large-scale removal, a full rebuild
            // would be needed.

            self.version += 1;
        }
    }

    /// Add a constraint to the graph.
    pub fn add_constraint(&mut self, meta: ConstraintMeta) {
        let id = meta.id;
        let entities = meta.entity_deps.clone();

        // Update bidirectional mappings
        for &eid in &entities {
            self.entity_to_constraints.entry(eid).or_default().insert(id);
        }
        let entity_set: HashSet<EntityId> = entities.iter().copied().collect();
        self.constraint_to_entities.insert(id, entity_set);

        // Store constraint metadata
        self.constraints.insert(id, meta);

        // Union all entities referenced by this constraint
        if entities.len() >= 2 {
            for window in entities.windows(2) {
                let idx_a = self.entity_index[&window[0]];
                let idx_b = self.entity_index[&window[1]];
                self.uf.add_edge(idx_a, idx_b);
            }
        }

        // Mark affected component(s) dirty
        if let Some(&eid) = entities.first() {
            let idx = self.entity_index[&eid];
            let comp = ComponentId(self.uf.find(idx));
            self.dirty.insert(comp);
        }

        self.version += 1;
    }

    /// Remove a constraint from the graph.
    pub fn remove_constraint(&mut self, id: ConstraintId) {
        if let Some(meta) = self.constraints.remove(&id) {
            let entities = meta.entity_deps;

            // Record which component(s) are affected BEFORE removal
            let affected_indices: Vec<usize> = entities
                .iter()
                .filter_map(|&eid| self.entity_index.get(&eid).copied())
                .collect();

            // Remove from bidirectional mappings
            for &eid in &entities {
                if let Some(constraints) = self.entity_to_constraints.get_mut(&eid) {
                    constraints.remove(&id);
                }
            }
            self.constraint_to_entities.remove(&id);

            // Remove edges from union-find (may cause component split)
            if entities.len() >= 2 {
                for window in entities.windows(2) {
                    if let (Some(&idx_a), Some(&idx_b)) = (
                        self.entity_index.get(&window[0]),
                        self.entity_index.get(&window[1]),
                    ) {
                        self.uf.remove_edge(idx_a, idx_b);
                    }
                }
            }

            // Mark affected component(s) dirty
            // After removal, entities may be in different components
            for idx in affected_indices {
                let comp = ComponentId(self.uf.find(idx));
                self.dirty.insert(comp);
            }

            self.version += 1;
        }
    }

    /// Mark the component containing the given entity as dirty.
    pub fn mark_dirty_by_entity(&mut self, id: EntityId) {
        if let Some(&idx) = self.entity_index.get(&id) {
            let comp = ComponentId(self.uf.find(idx));
            self.dirty.insert(comp);
        }
    }

    /// Mark the component containing the entity that owns the given parameter as dirty.
    pub fn mark_dirty_by_param(&mut self, _entity_id: EntityId) {
        // For now, this is the same as mark_dirty_by_entity
        // In a more sophisticated implementation, we might track which
        // specific parameters changed and only mark constraints that
        // depend on those parameters
        self.mark_dirty_by_entity(_entity_id);
    }

    /// Drain and return the set of dirty components.
    pub fn take_dirty(&mut self) -> Vec<ComponentId> {
        self.dirty.drain().collect()
    }

    /// Get the component ID containing the given entity.
    pub fn component_of(&mut self, entity: EntityId) -> Option<ComponentId> {
        self.entity_index.get(&entity).map(|&idx| ComponentId(self.uf.find(idx)))
    }

    /// Get all entities in a given component.
    pub fn entities_in_component(&mut self, comp: ComponentId) -> Vec<EntityId> {
        let comp_root = comp.0;
        self.entities
            .keys()
            .filter(|&&eid| {
                if let Some(&idx) = self.entity_index.get(&eid) {
                    self.uf.find(idx) == comp_root
                } else {
                    false
                }
            })
            .copied()
            .collect()
    }

    /// Get all constraints in a given component.
    /// A constraint is in a component if any of its entities are in that component.
    pub fn constraints_in_component(&mut self, comp: ComponentId) -> Vec<ConstraintId> {
        let entities_in_comp: HashSet<EntityId> =
            self.entities_in_component(comp).into_iter().collect();

        self.constraints
            .values()
            .filter(|meta| {
                meta.entity_deps.iter().any(|eid| entities_in_comp.contains(eid))
            })
            .map(|meta| meta.id)
            .collect()
    }

    /// Get the number of connected components.
    pub fn component_count(&mut self) -> usize {
        self.uf.component_count()
    }

    /// Calculate the degrees of freedom for a component.
    /// DOF = free_params - equations
    pub fn component_dof(&mut self, comp: ComponentId) -> i32 {
        let comp_root = comp.0;

        // Find all entities in this component
        let entities_in_comp: Vec<EntityId> = self.entities
            .keys()
            .filter(|&&eid| {
                if let Some(&idx) = self.entity_index.get(&eid) {
                    self.uf.find(idx) == comp_root
                } else {
                    false
                }
            })
            .copied()
            .collect();

        // Calculate total parameters and fixed parameters
        let mut total_params = 0;
        let mut fixed_params = 0;
        for &eid in &entities_in_comp {
            if let Some(meta) = self.entities.get(&eid) {
                total_params += meta.param_count;
                fixed_params += meta.fixed_param_count;
            }
        }

        let free_params = total_params - fixed_params;

        // Find all constraints in this component
        let entities_set: HashSet<EntityId> = entities_in_comp.into_iter().collect();
        let mut total_equations = 0;
        for meta in self.constraints.values() {
            if meta.entity_deps.iter().any(|eid| entities_set.contains(eid)) {
                total_equations += meta.equation_count;
            }
        }

        free_params as i32 - total_equations as i32
    }

    /// Get the current version counter.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Get the total number of entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Get the total number of constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Get metadata for an entity.
    pub fn get_entity(&self, id: EntityId) -> Option<&EntityMeta> {
        self.entities.get(&id)
    }

    /// Get metadata for a constraint.
    pub fn get_constraint(&self, id: ConstraintId) -> Option<&ConstraintMeta> {
        self.constraints.get(&id)
    }

    /// Get all constraints that reference a given entity.
    pub fn constraints_on_entity(&self, id: EntityId) -> Vec<ConstraintId> {
        self.entity_to_constraints
            .get(&id)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Get all entities referenced by a given constraint.
    pub fn entities_in_constraint(&self, id: ConstraintId) -> Vec<EntityId> {
        self.constraints
            .get(&id)
            .map(|meta| meta.entity_deps.clone())
            .unwrap_or_default()
    }
}

impl Default for IncrementalGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: usize, param_count: usize) -> EntityMeta {
        EntityMeta {
            id: EntityId(id),
            param_count,
            fixed_param_count: 0,
        }
    }

    fn make_constraint(id: usize, entities: Vec<usize>, equations: usize) -> ConstraintMeta {
        ConstraintMeta {
            id: ConstraintId(id),
            equation_count: equations,
            entity_deps: entities.into_iter().map(EntityId).collect(),
            param_deps: vec![],
        }
    }

    #[test]
    fn test_new_graph() {
        let graph = IncrementalGraph::new();
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.constraint_count(), 0);
        assert_eq!(graph.version(), 0);
    }

    #[test]
    fn test_add_entity() {
        let mut graph = IncrementalGraph::new();
        graph.add_entity(make_entity(0, 2));

        assert_eq!(graph.entity_count(), 1);
        assert_eq!(graph.version(), 1);

        let meta = graph.get_entity(EntityId(0)).unwrap();
        assert_eq!(meta.param_count, 2);
    }

    #[test]
    fn test_add_multiple_entities() {
        let mut graph = IncrementalGraph::new();
        graph.add_entity(make_entity(0, 2));
        graph.add_entity(make_entity(1, 2));
        graph.add_entity(make_entity(2, 3));

        assert_eq!(graph.entity_count(), 3);
        assert_eq!(graph.component_count(), 3); // All disjoint initially
    }

    #[test]
    fn test_add_constraint_connects_entities() {
        let mut graph = IncrementalGraph::new();
        graph.add_entity(make_entity(0, 2));
        graph.add_entity(make_entity(1, 2));

        assert_eq!(graph.component_count(), 2);

        // Add constraint that connects entity 0 and 1
        graph.add_constraint(make_constraint(0, vec![0, 1], 1));

        assert_eq!(graph.component_count(), 1);
        assert_eq!(graph.constraint_count(), 1);

        // Both entities should be in the same component
        let comp0 = graph.component_of(EntityId(0)).unwrap();
        let comp1 = graph.component_of(EntityId(1)).unwrap();
        assert_eq!(comp0, comp1);
    }

    #[test]
    fn test_remove_constraint_splits_component() {
        let mut graph = IncrementalGraph::new();
        graph.add_entity(make_entity(0, 2));
        graph.add_entity(make_entity(1, 2));
        graph.add_constraint(make_constraint(0, vec![0, 1], 1));

        assert_eq!(graph.component_count(), 1);

        // Remove the constraint
        graph.remove_constraint(ConstraintId(0));

        assert_eq!(graph.component_count(), 2);
        assert_eq!(graph.constraint_count(), 0);
    }

    #[test]
    fn test_dirty_tracking() {
        let mut graph = IncrementalGraph::new();
        graph.add_entity(make_entity(0, 2));
        graph.add_entity(make_entity(1, 2));

        // Adding entities marks them dirty
        let dirty = graph.take_dirty();
        assert!(!dirty.is_empty());

        // After taking, dirty set should be empty
        let dirty2 = graph.take_dirty();
        assert!(dirty2.is_empty());

        // Add constraint - should mark component dirty
        graph.add_constraint(make_constraint(0, vec![0, 1], 1));
        let dirty3 = graph.take_dirty();
        assert_eq!(dirty3.len(), 1);
    }

    #[test]
    fn test_mark_dirty_by_entity() {
        let mut graph = IncrementalGraph::new();
        graph.add_entity(make_entity(0, 2));

        // Clear initial dirty marks
        graph.take_dirty();

        // Explicitly mark entity as dirty
        graph.mark_dirty_by_entity(EntityId(0));

        let dirty = graph.take_dirty();
        assert_eq!(dirty.len(), 1);
    }

    #[test]
    fn test_entities_in_component() {
        let mut graph = IncrementalGraph::new();
        graph.add_entity(make_entity(0, 2));
        graph.add_entity(make_entity(1, 2));
        graph.add_entity(make_entity(2, 2));

        // Connect 0 and 1
        graph.add_constraint(make_constraint(0, vec![0, 1], 1));

        let comp = graph.component_of(EntityId(0)).unwrap();
        let mut entities = graph.entities_in_component(comp);
        entities.sort_by_key(|e| e.0);

        assert_eq!(entities, vec![EntityId(0), EntityId(1)]);

        // Entity 2 should be in its own component
        let comp2 = graph.component_of(EntityId(2)).unwrap();
        assert_ne!(comp, comp2);
        let entities2 = graph.entities_in_component(comp2);
        assert_eq!(entities2, vec![EntityId(2)]);
    }

    #[test]
    fn test_constraints_in_component() {
        let mut graph = IncrementalGraph::new();
        graph.add_entity(make_entity(0, 2));
        graph.add_entity(make_entity(1, 2));
        graph.add_entity(make_entity(2, 2));

        graph.add_constraint(make_constraint(0, vec![0, 1], 1));
        graph.add_constraint(make_constraint(1, vec![1, 2], 1));

        // All three entities should now be in one component
        let comp = graph.component_of(EntityId(0)).unwrap();
        let constraints = graph.constraints_in_component(comp);

        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn test_component_dof() {
        let mut graph = IncrementalGraph::new();

        // Two points (2 params each = 4 total), one distance constraint (1 equation)
        graph.add_entity(make_entity(0, 2));
        graph.add_entity(make_entity(1, 2));
        graph.add_constraint(make_constraint(0, vec![0, 1], 1));

        let comp = graph.component_of(EntityId(0)).unwrap();
        let dof = graph.component_dof(comp);

        assert_eq!(dof, 3); // 4 free params - 1 equation = 3 DOF
    }

    #[test]
    fn test_component_dof_with_fixed_params() {
        let mut graph = IncrementalGraph::new();

        // Two points, first one is fixed
        let mut meta0 = make_entity(0, 2);
        meta0.fixed_param_count = 2;
        graph.add_entity(meta0);
        graph.add_entity(make_entity(1, 2));

        graph.add_constraint(make_constraint(0, vec![0, 1], 1));

        let comp = graph.component_of(EntityId(0)).unwrap();
        let dof = graph.component_dof(comp);

        assert_eq!(dof, 1); // (4 total - 2 fixed) - 1 equation = 1 DOF
    }

    #[test]
    fn test_remove_entity() {
        let mut graph = IncrementalGraph::new();
        graph.add_entity(make_entity(0, 2));
        graph.add_entity(make_entity(1, 2));
        graph.add_constraint(make_constraint(0, vec![0, 1], 1));

        assert_eq!(graph.entity_count(), 2);
        assert_eq!(graph.constraint_count(), 1);

        // Remove entity 0 - should also remove the constraint
        graph.remove_entity(EntityId(0));

        assert_eq!(graph.entity_count(), 1);
        assert_eq!(graph.constraint_count(), 0);
    }

    #[test]
    fn test_constraints_on_entity() {
        let mut graph = IncrementalGraph::new();
        graph.add_entity(make_entity(0, 2));
        graph.add_entity(make_entity(1, 2));
        graph.add_entity(make_entity(2, 2));

        graph.add_constraint(make_constraint(0, vec![0, 1], 1));
        graph.add_constraint(make_constraint(1, vec![0, 2], 1));
        graph.add_constraint(make_constraint(2, vec![1, 2], 1));

        let constraints = graph.constraints_on_entity(EntityId(0));
        assert_eq!(constraints.len(), 2);

        let constraints1 = graph.constraints_on_entity(EntityId(1));
        assert_eq!(constraints1.len(), 2);
    }

    #[test]
    fn test_entities_in_constraint() {
        let mut graph = IncrementalGraph::new();
        graph.add_entity(make_entity(0, 2));
        graph.add_entity(make_entity(1, 2));

        graph.add_constraint(make_constraint(0, vec![0, 1], 1));

        let entities = graph.entities_in_constraint(ConstraintId(0));
        assert_eq!(entities.len(), 2);
        assert!(entities.contains(&EntityId(0)));
        assert!(entities.contains(&EntityId(1)));
    }

    #[test]
    fn test_version_increments() {
        let mut graph = IncrementalGraph::new();
        assert_eq!(graph.version(), 0);

        graph.add_entity(make_entity(0, 2));
        assert_eq!(graph.version(), 1);

        graph.add_constraint(make_constraint(0, vec![0], 1));
        assert_eq!(graph.version(), 2);

        graph.remove_constraint(ConstraintId(0));
        assert_eq!(graph.version(), 3);

        graph.remove_entity(EntityId(0));
        assert_eq!(graph.version(), 4);
    }

    #[test]
    fn test_complex_graph() {
        let mut graph = IncrementalGraph::new();

        // Create a more complex graph with multiple components
        for i in 0..6 {
            graph.add_entity(make_entity(i, 2));
        }

        // Connect 0-1-2 as one component
        graph.add_constraint(make_constraint(0, vec![0, 1], 1));
        graph.add_constraint(make_constraint(1, vec![1, 2], 1));

        // Connect 3-4 as another component
        graph.add_constraint(make_constraint(2, vec![3, 4], 1));

        // 5 remains isolated

        assert_eq!(graph.component_count(), 3);

        // Now connect the two components via 2-3
        graph.add_constraint(make_constraint(3, vec![2, 3], 1));
        assert_eq!(graph.component_count(), 2); // {0,1,2,3,4} and {5}
    }
}
