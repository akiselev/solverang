//! Generational index types for the constraint system.
//!
//! These types provide type-safe, generation-checked identifiers for parameters,
//! entities, constraints, and clusters. The generational index pattern prevents
//! use-after-free bugs: if an item is removed and its slot reused, the old ID
//! will have a stale generation and be detected as invalid.

use std::fmt;

/// Generation counter type. Incremented each time a slot is reused.
pub type Generation = u32;

/// A generational index for a parameter in the [`ParamStore`](crate::param::ParamStore).
///
/// Parameters are the fundamental solvable quantities. Every entity owns some
/// parameters (e.g., a 2D point owns two: x and y). The solver reads and writes
/// parameter values through the `ParamStore`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId {
    pub(crate) index: u32,
    pub(crate) generation: Generation,
}

/// A generational index for an entity in the constraint system.
///
/// Entities are named groups of parameters (points, circles, rigid bodies, etc.).
/// The solver treats all entities uniformly — it only cares about their parameter IDs.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId {
    pub(crate) index: u32,
    pub(crate) generation: Generation,
}

/// A generational index for a constraint in the constraint system.
///
/// Constraints produce residuals (equations that should be zero) and Jacobians
/// (partial derivatives). The solver uses these to find parameter values that
/// satisfy all constraints simultaneously.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstraintId {
    pub(crate) index: u32,
    pub(crate) generation: Generation,
}

/// Identifier for a cluster of coupled constraints.
///
/// Clusters are groups of constraints that share parameters (directly or
/// transitively) and must be solved together. Independent clusters can be
/// solved in parallel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClusterId(pub usize);

// --- Debug implementations ---

impl fmt::Debug for ParamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Param({}g{})", self.index, self.generation)
    }
}

impl fmt::Debug for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}g{})", self.index, self.generation)
    }
}

impl fmt::Debug for ConstraintId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Constraint({}g{})", self.index, self.generation)
    }
}

// --- Construction helpers (crate-internal) ---

impl ParamId {
    /// Create a new ParamId. Only used internally by ParamStore.
    pub(crate) fn new(index: u32, generation: Generation) -> Self {
        Self { index, generation }
    }

    /// Raw index (for internal use in mapping).
    pub(crate) fn raw_index(self) -> u32 {
        self.index
    }
}

impl EntityId {
    /// Create a new EntityId.
    pub(crate) fn new(index: u32, generation: Generation) -> Self {
        Self { index, generation }
    }

    /// Raw index (for internal use).
    pub(crate) fn raw_index(self) -> u32 {
        self.index
    }
}

impl ConstraintId {
    /// Create a new ConstraintId.
    pub(crate) fn new(index: u32, generation: Generation) -> Self {
        Self { index, generation }
    }

    /// Raw index (for internal use).
    pub(crate) fn raw_index(self) -> u32 {
        self.index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_id_equality() {
        let a = ParamId::new(0, 0);
        let b = ParamId::new(0, 0);
        let c = ParamId::new(0, 1); // Different generation
        let d = ParamId::new(1, 0); // Different index

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn test_entity_id_debug() {
        let id = EntityId::new(5, 2);
        assert_eq!(format!("{:?}", id), "Entity(5g2)");
    }

    #[test]
    fn test_constraint_id_debug() {
        let id = ConstraintId::new(3, 1);
        assert_eq!(format!("{:?}", id), "Constraint(3g1)");
    }

    #[test]
    fn test_ids_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ParamId::new(0, 0));
        set.insert(ParamId::new(1, 0));
        set.insert(ParamId::new(0, 0)); // duplicate
        assert_eq!(set.len(), 2);
    }
}
