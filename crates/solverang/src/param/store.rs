//! [`ParamStore`] — central storage for all solvable parameter values.

use std::collections::HashMap;

use crate::id::{EntityId, Generation, ParamId};

/// Entry in the parameter store.
#[derive(Clone, Debug)]
struct ParamEntry {
    value: f64,
    owner: EntityId,
    fixed: bool,
    generation: Generation,
    alive: bool,
    lower: f64,
    upper: f64,
}

/// Central storage for all solvable parameter values.
///
/// Every solvable quantity in the system is a [`ParamId`] pointing into this store.
/// Entities own parameters. Constraints read parameters. The solver writes
/// parameters. The `ParamStore` is the single source of truth.
#[derive(Clone, Debug)]
pub struct ParamStore {
    entries: Vec<ParamEntry>,
    /// Free-list of reusable slots.
    free_list: Vec<u32>,
}

impl ParamStore {
    /// Create an empty parameter store.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            free_list: Vec::new(),
        }
    }

    /// Allocate a new parameter with the given initial value, owned by `owner`.
    pub fn alloc(&mut self, value: f64, owner: EntityId) -> ParamId {
        if let Some(index) = self.free_list.pop() {
            let entry = &mut self.entries[index as usize];
            let generation = entry.generation + 1;
            *entry = ParamEntry {
                value,
                owner,
                fixed: false,
                generation,
                alive: true,
                lower: f64::NEG_INFINITY,
                upper: f64::INFINITY,
            };
            ParamId::new(index, generation)
        } else {
            let index = self.entries.len() as u32;
            self.entries.push(ParamEntry {
                value,
                owner,
                fixed: false,
                generation: 0,
                alive: true,
                lower: f64::NEG_INFINITY,
                upper: f64::INFINITY,
            });
            ParamId::new(index, 0)
        }
    }

    /// Free a parameter, returning its slot to the free list.
    ///
    /// # Panics
    ///
    /// Panics if the id is invalid (stale generation or out of bounds).
    pub fn free(&mut self, id: ParamId) {
        let entry = self.entry_mut(id).expect("free: invalid ParamId");
        entry.alive = false;
        self.free_list.push(id.raw_index());
    }

    /// Get the current value of a parameter.
    ///
    /// # Panics
    ///
    /// Panics if the id is invalid.
    pub fn get(&self, id: ParamId) -> f64 {
        self.entry(id).expect("get: invalid ParamId").value
    }

    /// Set the value of a parameter.
    ///
    /// # Panics
    ///
    /// Panics if the id is invalid.
    pub fn set(&mut self, id: ParamId, value: f64) {
        self.entry_mut(id).expect("set: invalid ParamId").value = value;
    }

    /// Check whether a parameter is fixed (excluded from solving).
    pub fn is_fixed(&self, id: ParamId) -> bool {
        self.entry(id).expect("is_fixed: invalid ParamId").fixed
    }

    /// Mark a parameter as fixed (excluded from solving).
    pub fn fix(&mut self, id: ParamId) {
        self.entry_mut(id).expect("fix: invalid ParamId").fixed = true;
    }

    /// Mark a parameter as free (included in solving).
    pub fn unfix(&mut self, id: ParamId) {
        self.entry_mut(id).expect("unfix: invalid ParamId").fixed = false;
    }

    /// Returns the owner entity of this parameter.
    pub fn owner(&self, id: ParamId) -> EntityId {
        self.entry(id).expect("owner: invalid ParamId").owner
    }

    /// Number of currently alive, free (non-fixed) parameters.
    pub fn free_param_count(&self) -> usize {
        self.entries.iter().filter(|e| e.alive && !e.fixed).count()
    }

    /// Number of currently alive parameters (including fixed).
    pub fn alive_count(&self) -> usize {
        self.entries.iter().filter(|e| e.alive).count()
    }

    /// Build a mapping between free [`ParamId`]s and solver column indices.
    ///
    /// This mapping is rebuilt each time the set of free parameters changes
    /// (params added/removed/fixed/unfixed).
    pub fn build_solver_mapping(&self) -> SolverMapping {
        let mut param_to_col = HashMap::new();
        let mut col_to_param = Vec::new();

        for (i, entry) in self.entries.iter().enumerate() {
            if entry.alive && !entry.fixed {
                let id = ParamId::new(i as u32, entry.generation);
                let col = col_to_param.len();
                param_to_col.insert(id, col);
                col_to_param.push(id);
            }
        }

        SolverMapping {
            param_to_col,
            col_to_param,
        }
    }

    /// Build a solver mapping restricted to only the given param IDs (that are free).
    ///
    /// Duplicate `ParamId`s in the input are deduplicated so each parameter
    /// appears at most once in the mapping.
    pub fn build_solver_mapping_for(&self, params: &[ParamId]) -> SolverMapping {
        let mut param_to_col = HashMap::new();
        let mut col_to_param = Vec::new();

        for &id in params {
            if param_to_col.contains_key(&id) {
                continue;
            }
            if let Some(entry) = self.entry(id) {
                if !entry.fixed {
                    let col = col_to_param.len();
                    param_to_col.insert(id, col);
                    col_to_param.push(id);
                }
            }
        }

        SolverMapping {
            param_to_col,
            col_to_param,
        }
    }

    /// Extract free parameter values in solver column order.
    pub fn extract_free_values(&self, mapping: &SolverMapping) -> Vec<f64> {
        mapping
            .col_to_param
            .iter()
            .map(|&id| self.get(id))
            .collect()
    }

    /// Write solver values back into the store using the given mapping.
    pub fn write_free_values(&mut self, values: &[f64], mapping: &SolverMapping) {
        for (col, &id) in mapping.col_to_param.iter().enumerate() {
            if col < values.len() {
                self.set(id, values[col]);
            }
        }
    }

    /// Create a snapshot (clone) of this store for temporary mutations during solving.
    pub fn snapshot(&self) -> ParamStore {
        self.clone()
    }

    /// Iterate over all alive parameter IDs.
    pub fn alive_param_ids(&self) -> impl Iterator<Item = ParamId> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.alive)
            .map(|(i, e)| ParamId::new(i as u32, e.generation))
    }

    /// Iterate over all alive, free parameter IDs.
    pub fn free_param_ids(&self) -> impl Iterator<Item = ParamId> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.alive && !e.fixed)
            .map(|(i, e)| ParamId::new(i as u32, e.generation))
    }

    /// Set lower and upper bounds for a parameter.
    ///
    /// # Panics
    ///
    /// Panics if the id is invalid or if `lower > upper`.
    pub fn set_bounds(&mut self, id: ParamId, lower: f64, upper: f64) {
        let entry = self.entry_mut(id).expect("set_bounds: invalid ParamId");
        assert!(lower <= upper, "lower bound must be <= upper bound");
        entry.lower = lower;
        entry.upper = upper;
    }

    /// Get the (lower, upper) bounds for a parameter.
    ///
    /// # Panics
    ///
    /// Panics if the id is invalid.
    pub fn bounds(&self, id: ParamId) -> (f64, f64) {
        let entry = self.entry(id).expect("bounds: invalid ParamId");
        (entry.lower, entry.upper)
    }

    /// Returns `true` if at least one bound is finite.
    ///
    /// # Panics
    ///
    /// Panics if the id is invalid.
    pub fn has_finite_bounds(&self, id: ParamId) -> bool {
        let (l, u) = self.bounds(id);
        l.is_finite() || u.is_finite()
    }

    // --- Internal helpers ---

    fn entry(&self, id: ParamId) -> Option<&ParamEntry> {
        let idx = id.raw_index() as usize;
        self.entries
            .get(idx)
            .filter(|e| e.alive && e.generation == id.generation)
    }

    fn entry_mut(&mut self, id: ParamId) -> Option<&mut ParamEntry> {
        let idx = id.raw_index() as usize;
        self.entries
            .get_mut(idx)
            .filter(|e| e.alive && e.generation == id.generation)
    }
}

impl Default for ParamStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Bidirectional mapping: [`ParamId`] <-> column index in the Jacobian.
///
/// Built once per solve (or once per decomposition change). The solver works
/// in terms of column indices; the constraint system works in terms of `ParamId`s.
/// This mapping bridges the two.
#[derive(Clone, Debug)]
pub struct SolverMapping {
    /// Map from `ParamId` to column index.
    pub param_to_col: HashMap<ParamId, usize>,
    /// Map from column index to `ParamId`.
    pub col_to_param: Vec<ParamId>,
}

impl SolverMapping {
    /// Number of free parameters (columns) in this mapping.
    pub fn len(&self) -> usize {
        self.col_to_param.len()
    }

    /// Whether this mapping is empty (no free parameters).
    pub fn is_empty(&self) -> bool {
        self.col_to_param.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_owner() -> EntityId {
        EntityId::new(0, 0)
    }

    #[test]
    fn test_alloc_and_get() {
        let mut store = ParamStore::new();
        let id = store.alloc(3.14, dummy_owner());
        assert!((store.get(id) - 3.14).abs() < 1e-15);
    }

    #[test]
    fn test_set() {
        let mut store = ParamStore::new();
        let id = store.alloc(0.0, dummy_owner());
        store.set(id, 2.718);
        assert!((store.get(id) - 2.718).abs() < 1e-15);
    }

    #[test]
    fn test_fix_unfix() {
        let mut store = ParamStore::new();
        let id = store.alloc(1.0, dummy_owner());
        assert!(!store.is_fixed(id));

        store.fix(id);
        assert!(store.is_fixed(id));

        store.unfix(id);
        assert!(!store.is_fixed(id));
    }

    #[test]
    fn test_free_param_count() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(1.0, owner);
        let _b = store.alloc(2.0, owner);
        let _c = store.alloc(3.0, owner);

        assert_eq!(store.free_param_count(), 3);

        store.fix(a);
        assert_eq!(store.free_param_count(), 2);
    }

    #[test]
    fn test_solver_mapping() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);
        let c = store.alloc(3.0, owner);

        store.fix(b);

        let mapping = store.build_solver_mapping();
        assert_eq!(mapping.len(), 2);
        assert!(mapping.param_to_col.contains_key(&a));
        assert!(!mapping.param_to_col.contains_key(&b)); // fixed
        assert!(mapping.param_to_col.contains_key(&c));
    }

    #[test]
    fn test_extract_and_write_values() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let a = store.alloc(1.0, owner);
        let b = store.alloc(2.0, owner);

        let mapping = store.build_solver_mapping();
        let values = store.extract_free_values(&mapping);
        assert_eq!(values.len(), 2);

        // Write new values back
        store.write_free_values(&[10.0, 20.0], &mapping);
        assert!((store.get(a) - 10.0).abs() < 1e-15);
        assert!((store.get(b) - 20.0).abs() < 1e-15);
    }

    #[test]
    fn test_free_and_reuse() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let id1 = store.alloc(1.0, owner);
        store.free(id1);

        // Alloc should reuse the slot with a new generation
        let id2 = store.alloc(2.0, owner);
        assert_eq!(id2.raw_index(), id1.raw_index());
        assert_ne!(id1, id2); // Different generations
        assert!((store.get(id2) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_snapshot() {
        let mut store = ParamStore::new();
        let owner = dummy_owner();
        let id = store.alloc(5.0, owner);

        let mut snap = store.snapshot();
        snap.set(id, 99.0);

        // Original unchanged
        assert!((store.get(id) - 5.0).abs() < 1e-15);
        assert!((snap.get(id) - 99.0).abs() < 1e-15);
    }
}
